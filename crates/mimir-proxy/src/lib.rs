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
    use super::field_u64;

    #[test]
    fn parses_usage_field() {
        let sse = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":2048}}}"#;
        assert_eq!(field_u64(sse, "\"cache_read_input_tokens\""), Some(2048));
        assert_eq!(field_u64(sse, "\"input_tokens\""), Some(12));
        assert_eq!(field_u64(sse, "\"missing\""), None);
    }
}
