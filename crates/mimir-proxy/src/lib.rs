//! Mimir's optional local API proxy ("Headroom"-style, our own implementation).
//!
//! It sits between the agent and `api.anthropic.com`: forwards every request
//! verbatim (streaming SSE back untouched) EXCEPT it may rewrite the body of
//! `POST /v1/messages` to add prompt-cache breakpoints, dedup repeated blocks,
//! and (optionally) prune stale tool results. Cache savings are MEASURED from
//! the response's `usage.cache_read_input_tokens`; dedup/prune are measured at
//! the request.
//!
//! It is OFF by default (you start it with `mimir proxy`) and it is a
//! man-in-the-middle on your prompts/completions — see docs/proxy.md. Auth
//! headers pass through untouched; the proxy never reads your API key.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Method, StatusCode};
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use futures_util::StreamExt;
use serde_json::Value;

mod optimize;
pub use optimize::{optimize_request, Optimization, OptimizeOpts};

/// Upper bound on a buffered request body (the messages JSON).
const MAX_BODY: usize = 64 * 1024 * 1024;
/// How far into the response we scan for the usage block before giving up.
const USAGE_SCAN_CAP: usize = 256 * 1024;

type Db = Arc<Mutex<rusqlite::Connection>>;

#[derive(Clone)]
pub struct ProxyConfig {
    pub bind: SocketAddr,
    pub upstream: String,
    pub dry_run: bool,
    pub cache: bool,
    pub dedup: bool,
    pub prune: bool,
    pub db_path: Option<PathBuf>,
    /// `recall_event` retention window in days for the idle-timer prune
    /// below (`config::LearnConfig::effective_retention_days`); `None`
    /// disables it. Unrelated to `prune` above (that's lossy tool-result
    /// trimming in the request-optimization pass, not ledger retention).
    pub event_retention_days: Option<u32>,
}

struct AppState {
    cfg: ProxyConfig,
    client: reqwest::Client,
    db: Option<Db>,
}

/// Run the proxy until the process is killed.
pub async fn serve(cfg: ProxyConfig) -> Result<()> {
    let db = cfg
        .db_path
        .as_ref()
        .and_then(|p| match mimir_core::db::open(p) {
            Ok(c) => Some(Arc::new(Mutex::new(c))),
            Err(e) => {
                tracing::warn!("proxy: savings ledger disabled ({e})");
                None
            }
        });
    let client = reqwest::Client::builder().build()?;
    let bind = cfg.bind;
    let upstream = cfg.upstream.clone();
    let dry = cfg.dry_run;

    // Reclaim the WAL on an idle timer. The proxy is long-lived and its
    // per-request savings writes leave a -wal that PASSIVE autocheckpoints may
    // never fully reset under a steady reader. A periodic TRUNCATE checkpoint
    // keeps it bounded; best-effort and non-fatal. Unlike the sync daemons this
    // reuses the shared savings connection (autocommit between requests, so no
    // read mark blocks it), and runs the blocking checkpoint on spawn_blocking
    // so a large TRUNCATE never stalls a runtime worker.
    //
    // Same connection also prunes old `recall_event` rows on the same idle
    // cadence, once at startup and then every tick — see `db::
    // spawn_checkpoint_timer`'s doc comment in mimir-core for why this rides
    // along with the checkpoint instead of its own timer. `injection_log`/
    // `session_state` rows get the same treatment, unconditionally (no user
    // config, unlike recall_event's retention) on the fixed
    // `mimir_core::db::SESSION_STATE_RETENTION_DAYS` cutoff — mirrors
    // `db::spawn_checkpoint_timer`'s variant for the sync daemons; the proxy
    // can't share that thread since it locks a request-shared connection
    // instead of opening its own.
    if let Some(db) = db.clone() {
        let event_retention_days = cfg.event_retention_days;
        if let Some(days) = event_retention_days {
            let db = db.clone();
            let _ = tokio::task::spawn_blocking(move || prune_events_once(&db, days)).await;
        }
        {
            let db = db.clone();
            let _ = tokio::task::spawn_blocking(move || prune_session_state_once(&db)).await;
        }
        tokio::spawn(async move {
            let period = std::time::Duration::from_secs(120);
            let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
            loop {
                tick.tick().await;
                let db = db.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if let Ok(conn) = db.lock() {
                        if let Err(e) = mimir_core::db::checkpoint_truncate(&conn) {
                            tracing::warn!("proxy: wal checkpoint failed ({e})");
                        }
                    }
                    if let Some(days) = event_retention_days {
                        prune_events_once(&db, days);
                    }
                    prune_session_state_once(&db);
                })
                .await;
            }
        });
    }

    let state = Arc::new(AppState { cfg, client, db });
    let app = Router::new().fallback(any(handle)).with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("mimir proxy listening on http://{bind} → {upstream}");
    eprintln!("  point your client at it:  export ANTHROPIC_BASE_URL=http://{bind}");
    if dry {
        eprintln!("  dry-run: measuring only, request bodies forwarded unchanged");
    }
    axum::serve(listener, app).await?;
    Ok(())
}

/// One `recall_event` prune pass against the proxy's shared connection, on
/// the same idle cadence as the WAL checkpoint above. Best-effort and
/// non-fatal, mirroring `mimir_core::db::spawn_checkpoint_timer`'s variant
/// for the sync daemons — this one just locks the shared `Db` first since
/// the proxy can't give the prune its own dedicated connection.
fn prune_events_once(db: &Db, days: u32) {
    let cutoff = mimir_core::model::now_unix() - i64::from(days) * 86_400;
    let Ok(conn) = db.lock() else { return };
    match mimir_core::store::prune_old_events(&conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!("proxy: pruned {n} old recall_event rows"),
        Ok(_) => {}
        Err(e) => tracing::warn!("proxy: recall_event prune failed ({e})"),
    }
}

/// One `injection_log`/`session_state` prune pass, mirroring
/// `mimir_core::db::spawn_checkpoint_timer`'s variant for the sync daemons
/// on the same fixed `mimir_core::db::SESSION_STATE_RETENTION_DAYS` cutoff
/// (unconditional — no user config, unlike `prune_events_once`'s `days`).
/// Best-effort and non-fatal, same locking shape as `prune_events_once`.
fn prune_session_state_once(db: &Db) {
    let cutoff =
        mimir_core::model::now_unix() - mimir_core::db::SESSION_STATE_RETENTION_DAYS * 86_400;
    let Ok(conn) = db.lock() else { return };
    match mimir_core::store::prune_session_state(&conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!("proxy: pruned {n} old session_state/injection_log rows"),
        Ok(_) => {}
        Err(e) => tracing::warn!("proxy: session_state prune failed ({e})"),
    }
}

async fn handle(State(state): State<Arc<AppState>>, req: axum::extract::Request) -> Response {
    match proxy(&state, req).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!("proxy error: {e}");
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("mimir proxy error: {e}")))
                .unwrap()
        }
    }
}

async fn proxy(state: &AppState, req: axum::extract::Request) -> Result<Response> {
    let (parts, body) = req.into_parts();
    let path_q = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/")
        .to_string();
    let url = format!("{}{}", state.cfg.upstream.trim_end_matches('/'), path_q);
    let body_bytes = axum::body::to_bytes(body, MAX_BODY).await?;

    let is_messages = parts.method == Method::POST && parts.uri.path().ends_with("/v1/messages");
    // we_cached: true only on requests where WE introduced caching (so the
    // measured cache reads downstream are ours to claim).
    let (send_bytes, we_cached) = if is_messages {
        optimize_messages_body(state, &body_bytes)
    } else {
        (body_bytes.to_vec(), false)
    };

    // Forward upstream, copying headers (minus hop-by-hop, host, content-length).
    let mut rb = state.client.request(parts.method.clone(), &url);
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop(name.as_str()) || name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        rb = rb.header(name.clone(), value.clone());
    }
    let upstream = rb.body(send_bytes).send().await?;

    let status = upstream.status();
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers().iter() {
        if is_hop_by_hop(name.as_str())
            || name == header::CONTENT_LENGTH
            || name == header::TRANSFER_ENCODING
        {
            continue;
        }
        builder = builder.header(name.clone(), value.clone());
    }

    // Tap the stream to record real cache savings, but only when we added the
    // breakpoints (otherwise the cache hits aren't ours to claim).
    let body = if we_cached {
        let mut scanner = UsageScanner::new(state.db.clone());
        let tapped = upstream.bytes_stream().inspect(move |chunk| {
            if let Ok(bytes) = chunk {
                scanner.feed(bytes);
            }
        });
        Body::from_stream(tapped)
    } else {
        Body::from_stream(upstream.bytes_stream())
    };
    Ok(builder.body(body)?)
}

/// Optimize a `/v1/messages` body. Returns the bytes to forward and whether we
/// introduced caching (and so should measure the response's cache reads).
fn optimize_messages_body(state: &AppState, body_bytes: &[u8]) -> (Vec<u8>, bool) {
    let Ok(json) = serde_json::from_slice::<Value>(body_bytes) else {
        return (body_bytes.to_vec(), false); // not JSON we understand — pass through
    };
    let we_cached = state.cfg.cache && !optimize::has_cache_control(&json);
    let (new_json, opt) = optimize_request(
        json,
        OptimizeOpts {
            cache: state.cfg.cache,
            dedup: state.cfg.dedup,
            prune: state.cfg.prune,
        },
    );
    if state.cfg.dry_run {
        if opt.deduped > 0 || opt.pruned > 0 {
            tracing::info!(
                deduped = opt.deduped,
                pruned = opt.pruned,
                "proxy dry-run: would optimize"
            );
        }
        return (body_bytes.to_vec(), false);
    }
    record_request_savings(&state.db, &opt);
    let bytes = serde_json::to_vec(&new_json).unwrap_or_else(|_| body_bytes.to_vec());
    (bytes, we_cached)
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn record_request_savings(db: &Option<Db>, opt: &Optimization) {
    let Some(db) = db else { return };
    let Ok(conn) = db.lock() else { return };
    if opt.deduped > 0 {
        let _ = mimir_core::savings::record(
            &conn,
            None,
            mimir_core::savings::source::PROXY_DEDUP,
            opt.deduped,
            0,
            serde_json::json!({}),
        );
    }
    if opt.pruned > 0 {
        let _ = mimir_core::savings::record(
            &conn,
            None,
            mimir_core::savings::source::PROXY_PRUNE,
            opt.pruned,
            0,
            serde_json::json!({}),
        );
    }
}

/// Scans the start of a response for the `usage` block and records the REAL
/// cache savings once. Forwards nothing — it's a passive tap.
///
/// Assumes the Anthropic SSE/JSON shape (a `"cache_read_input_tokens": N` field
/// near the start). On any other shape it simply finds nothing and records 0.
struct UsageScanner {
    db: Option<Db>,
    buf: String,
    done: bool,
}

impl UsageScanner {
    fn new(db: Option<Db>) -> Self {
        UsageScanner {
            db,
            buf: String::new(),
            done: false,
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        if self.done {
            return;
        }
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        if let Some(cache_read) = field_u64(&self.buf, "\"cache_read_input_tokens\"") {
            self.record(cache_read);
            self.done = true;
            self.buf = String::new();
        } else if self.buf.len() > USAGE_SCAN_CAP {
            self.done = true;
            self.buf = String::new();
        }
    }

    fn record(&self, cache_read: u64) {
        if cache_read == 0 {
            return;
        }
        let Some(db) = &self.db else { return };
        let Ok(conn) = db.lock() else { return };
        // Cached input bills at ~10% → ~90% saved on these tokens.
        let before = cache_read as usize;
        let after = before / 10;
        let _ = mimir_core::savings::record(
            &conn,
            None,
            mimir_core::savings::source::PROXY_CACHE,
            before,
            after,
            serde_json::json!({ "measured": true }),
        );
    }
}

/// Parse `"key": <digits>` out of a JSON fragment.
fn field_u64(s: &str, key: &str) -> Option<u64> {
    let i = s.find(key)?;
    let rest = s[i + key.len()..].trim_start_matches([':', ' ']);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest.get(..end).and_then(|d| d.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::{field_u64, prune_events_once, prune_session_state_once, Db};
    use std::sync::{Arc, Mutex};

    #[test]
    fn parses_usage_field() {
        let sse = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":2048}}}"#;
        assert_eq!(field_u64(sse, "\"cache_read_input_tokens\""), Some(2048));
        assert_eq!(field_u64(sse, "\"input_tokens\""), Some(12));
        assert_eq!(field_u64(sse, "\"missing\""), None);
    }

    /// The proxy can't give the prune its own dedicated connection (unlike
    /// `mimir_core::db::spawn_checkpoint_timer`'s variant) — it locks the
    /// same `Arc<Mutex<Connection>>` every request handler shares. This
    /// covers that locking wrapper directly, since `prune_old_events`
    /// itself is already covered in `mimir-core`.
    #[test]
    fn prune_events_once_removes_only_rows_past_the_window() {
        let conn = mimir_core::db::open_in_memory().unwrap();
        let node_id = conn
            .execute(
                "INSERT INTO node (uid, kind, body, created_at, updated_at) \
                 VALUES ('n1', 'memory', 'x', 0, 0)",
                [],
            )
            .map(|_| conn.last_insert_rowid())
            .unwrap();
        let now = mimir_core::model::now_unix();
        let old_at = now - 200 * 86_400;
        let recent_at = now - 10 * 86_400;
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'shown', ?2)",
            rusqlite::params![node_id, old_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'shown', ?2)",
            rusqlite::params![node_id, recent_at],
        )
        .unwrap();

        let db: Db = Arc::new(Mutex::new(conn));
        prune_events_once(&db, 180);

        let remaining: i64 = db
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM recall_event", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1, "only the 200-day-old row clears the 180-day cutoff");
    }

    /// Same locking-wrapper coverage as `prune_events_once_removes_only_rows_past_the_window`,
    /// for the `session_state`/`injection_log` prune this proxy timer also
    /// runs — `prune_session_state` itself is already covered in
    /// `mimir-core`; this just proves the proxy's shared-connection lock
    /// path calls it with the right (fixed) cutoff.
    #[test]
    fn prune_session_state_once_removes_only_rows_past_the_seven_day_window() {
        let conn = mimir_core::db::open_in_memory().unwrap();
        let now = mimir_core::model::now_unix();
        let old_at = now - 10 * 86_400;
        let recent_at = now - 86_400;
        conn.execute(
            "INSERT INTO session_state (session_id, key, value, updated_at) VALUES ('s', 'old', 'v', ?1)",
            rusqlite::params![old_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_state (session_id, key, value, updated_at) VALUES ('s', 'recent', 'v', ?1)",
            rusqlite::params![recent_at],
        )
        .unwrap();

        let db: Db = Arc::new(Mutex::new(conn));
        prune_session_state_once(&db);

        let remaining: i64 = db
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM session_state", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1, "only the 10-day-old row clears the fixed 7-day cutoff");
    }
}
