//! Mimir's optional local API proxy ("Headroom"-style, our own implementation).
//!
//! It sits between the agent and `api.anthropic.com`: forwards every request
//! verbatim (streaming SSE back untouched) EXCEPT it may rewrite the body of
//! `POST /v1/messages` to add prompt-cache breakpoints, dedup repeated blocks,
//! and (optionally) prune stale tool results. A `/v1/messages` body no pass
//! changed goes out as the very bytes that came in, and one a pass did change
//! as the client's text with only the changed values replaced (`patch`).
//! Cache savings are MEASURED from the response's
//! `usage.cache_read_input_tokens`; dedup/prune are measured at the request
//! and booked once upstream has accepted it.
//!
//! It is OFF by default (you start it with `mimir proxy`) and it is a
//! man-in-the-middle on your prompts/completions — see docs/proxy.md. Auth
//! headers pass through untouched; the proxy never reads your API key.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use futures_util::StreamExt;
use serde_json::Value;

mod optimize;
mod patch;
pub use optimize::{count_request_tokens, optimize_request, CacheTtl, Optimization, OptimizeOpts};

/// Upper bound on a buffered request body (the messages JSON).
const MAX_BODY: usize = 64 * 1024 * 1024;

type Db = Arc<Mutex<rusqlite::Connection>>;

/// Lock the shared ledger connection, taking it back if it is poisoned.
///
/// A panic while the lock is held poisons the mutex for good. The connection
/// behind it is still sound: every write the proxy makes is an autocommit
/// statement of its own, so a panic cannot leave a transaction half done. So
/// the lock is recovered and the poison cleared. Giving up instead would
/// switch off the savings ledger and both retention prunes until the proxy
/// is restarted, without a word.
fn ledger(db: &Db) -> MutexGuard<'_, rusqlite::Connection> {
    db.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("proxy: a ledger write panicked earlier; reusing its connection");
        db.clear_poison();
        poisoned.into_inner()
    })
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub bind: SocketAddr,
    pub upstream: String,
    pub dry_run: bool,
    pub cache: bool,
    /// Lifetime of the breakpoints we add. Only meaningful when `cache` is on
    /// AND the client set none itself — we never restamp someone else's.
    pub cache_ttl: CacheTtl,
    pub dedup: bool,
    pub prune: bool,
    /// Runaway circuit breaker: reject a `/v1/messages` request whose estimated
    /// input exceeds this many tokens. `0` disables it (the default). We
    /// **reject**, never truncate — silently dropping content would change what
    /// the model sees, and the whole point is to make a runaway loud.
    pub max_request_tokens: usize,
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

impl AppState {
    /// The circuit breaker's cap, if the breaker is on. `max_request_tokens`
    /// spells "off" as 0.
    fn breaker_cap(&self) -> Option<usize> {
        let cap = self.cfg.max_request_tokens;
        (cap > 0).then_some(cap)
    }
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
    // `mimir_core::db::SESSION_STATE_RETENTION_DAYS` cutoff. The proxy can't
    // share `db::spawn_checkpoint_timer`'s thread since it locks a
    // request-shared connection instead of opening its own; the passes it
    // runs under that lock are the core's own.
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
                    {
                        let conn = ledger(&db);
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

/// One prune pass over the usage ledgers against the proxy's shared
/// connection, on the same idle cadence as the WAL checkpoint above. The
/// pass itself is `mimir_core::db::prune_events_once`, the one `mimir mcp`
/// and `mimir serve` run; the proxy only supplies its own lock, because it
/// cannot give the prune a connection of its own.
fn prune_events_once(db: &Db, days: u32) {
    mimir_core::db::prune_events_once(&ledger(db), days);
}

/// One `injection_log`/`session_state` prune pass, likewise
/// `mimir_core::db::prune_session_state_once` under the proxy's lock: a
/// session's state goes only once the whole session has been silent for
/// `mimir_core::db::SESSION_STATE_RETENTION_DAYS`.
fn prune_session_state_once(db: &Db) {
    mimir_core::db::prune_session_state_once(&ledger(db));
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

/// A request body read into memory, or the fact that it is too big to be.
enum Buffered {
    Whole(Bytes),
    OverLimit,
}

/// Read `body` to its end, stopping as soon as it passes `limit` bytes. A
/// body over the limit is an answer, not a failure: the client is told so in
/// the same words as by the circuit breaker (see [`TooLarge`]). An `Err` is
/// the transport breaking mid-body.
async fn buffer_body(body: Body, limit: usize) -> Result<Buffered> {
    let mut stream = body.into_data_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.len() > limit - buf.len() {
            return Ok(Buffered::OverLimit);
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(Buffered::Whole(Bytes::from(buf)))
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
    let body_bytes = match buffer_body(body, MAX_BODY).await? {
        Buffered::Whole(bytes) => bytes,
        Buffered::OverLimit => {
            tracing::warn!(
                limit = MAX_BODY,
                "proxy: rejected a request body over the buffer limit"
            );
            return Ok(too_large_response(TooLarge::Bytes { limit: MAX_BODY }));
        }
    };

    let is_messages = parts.method == Method::POST && parts.uri.path().ends_with("/v1/messages");
    let prepared = if is_messages {
        prepare_messages_body(state, body_bytes)
    } else {
        Prepared::untouched(body_bytes)
    };

    // Circuit breaker. Checked AFTER the optimization passes, so dedup/prune
    // get their chance to bring an over-budget request back under the line
    // rather than us rejecting work the proxy could have salvaged.
    if let (Some(cap), Some(tokens)) = (state.breaker_cap(), prepared.gate_tokens) {
        if tokens > cap {
            tracing::warn!(
                tokens,
                cap,
                "proxy: rejected oversized request (max_request_tokens)"
            );
            return Ok(too_large_response(TooLarge::Tokens { tokens, cap }));
        }
    }

    // Forward upstream, copying headers (minus hop-by-hop, host, content-length).
    let mut rb = state.client.request(parts.method.clone(), &url);
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop(name.as_str()) || name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        rb = rb.header(name.clone(), value.clone());
    }
    let upstream = rb.body(prepared.body).send().await?;

    let status = upstream.status();
    // Only now are the request-side savings real: the breaker let the request
    // through, it reached upstream, and upstream took it. A request refused
    // anywhere on the way saved nobody anything.
    if status.is_success() {
        record_request_savings(&state.db, &prepared.saved);
    }
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
    let body = match (prepared.we_cached, content_coding(upstream.headers())) {
        (true, None) => {
            let mut scanner = UsageScanner::new(state.db.clone());
            let tapped = upstream.bytes_stream().inspect(move |chunk| {
                if let Ok(bytes) = chunk {
                    scanner.feed(bytes);
                }
            });
            Body::from_stream(tapped)
        }
        (true, Some(coding)) => {
            // The client asked for a compressed reply and reqwest is built
            // without decompression, so these bytes are not JSON to read.
            tracing::debug!(
                %coding,
                "proxy: response is compressed; cache reads not measured for this request"
            );
            Body::from_stream(upstream.bytes_stream())
        }
        (false, _) => Body::from_stream(upstream.bytes_stream()),
    };
    Ok(builder.body(body)?)
}

/// The response's content coding, when it has one that changes the bytes.
/// `identity` is the header's way of saying there is none.
fn content_coding(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(header::CONTENT_ENCODING)?;
    let coding = String::from_utf8_lossy(value.as_bytes())
        .trim()
        .to_ascii_lowercase();
    (!coding.is_empty() && coding != "identity").then_some(coding)
}

/// What the proxy sends upstream for one request, and what it may claim for
/// it once upstream has answered.
struct Prepared {
    /// The bytes to forward: the client's own, untouched, unless a pass
    /// changed the request.
    body: Bytes,
    /// A cache breakpoint of ours is in `body`, so the cache reads in the
    /// response are ours to measure.
    we_cached: bool,
    /// Estimated input tokens of `body`, for the circuit breaker and nothing
    /// else. `None` for a request the breaker does not judge: the breaker is
    /// off, or this is not a messages call, not JSON we can read, or a dry
    /// run.
    gate_tokens: Option<usize>,
    /// Tokens dedup and prune took out of `body`. Booked only after upstream
    /// has accepted the request.
    saved: Optimization,
}

impl Prepared {
    /// Forward `body` exactly as it came: nothing to claim, nothing to gate.
    fn untouched(body: Bytes) -> Self {
        Prepared {
            body,
            we_cached: false,
            gate_tokens: None,
            saved: Optimization::default(),
        }
    }
}

/// Run the optimization passes over a `/v1/messages` body and decide what to
/// forward for it.
fn prepare_messages_body(state: &AppState, body: Bytes) -> Prepared {
    let Ok(json) = serde_json::from_slice::<Value>(&body) else {
        // Not JSON we understand — pass through, ungated, so the circuit
        // breaker cannot fire on a body we failed to parse.
        return Prepared::untouched(body);
    };
    let (new_json, applied) = optimize::apply_passes(
        json,
        OptimizeOpts {
            cache: state.cfg.cache,
            cache_ttl: state.cfg.cache_ttl,
            dedup: state.cfg.dedup,
            prune: state.cfg.prune,
        },
    );
    if state.cfg.dry_run {
        if applied.saved.deduped > 0 || applied.saved.pruned > 0 {
            tracing::info!(
                deduped = applied.saved.deduped,
                pruned = applied.saved.pruned,
                "proxy dry-run: would optimize"
            );
        }
        // Ungated, so --dry-run stays what it says on the tin: measure only,
        // never alter the outcome of a request (including by rejecting it).
        return Prepared::untouched(body);
    }
    let breaker_on = state.breaker_cap().is_some();
    finish(body, &new_json, applied, breaker_on, write_rewritten)
}

/// The bytes of a request a pass changed: the client's own text with only
/// the changed values replaced (see [`patch`]), so that what no pass touched
/// reaches the API exactly as on the turns before. Should that not be
/// possible, the optimized value serialized whole.
fn write_rewritten(original: &[u8], optimized: &Value) -> serde_json::Result<Vec<u8>> {
    match patch::rewrite(original, optimized) {
        Some(bytes) => Ok(bytes),
        None => {
            // Loud, because it is not free: a body serialized whole has its
            // keys in sorted order, which is not the order this conversation
            // had on the turns before.
            tracing::warn!(
                "proxy: request text could not be patched; serializing it whole, \
                 which changes its key order and may cost this conversation a cache miss"
            );
            serde_json::to_vec(optimized)
        }
    }
}

/// Settle what goes upstream once the passes have run: `original` is what the
/// client sent, `optimized` what the passes made of it, `write` turns the
/// two into the bytes of a changed request. The flags always describe the
/// bytes that are actually forwarded.
///
/// The request is tokenized for the breaker only when `breaker_on`. Nothing
/// else reads that count (savings are counted by the passes, block by
/// replaced block), and it is not cheap: about 40% of the time spent
/// preparing a request, measured on a 4.4 MB body.
fn finish(
    original: Bytes,
    optimized: &Value,
    applied: optimize::Applied,
    breaker_on: bool,
    write: impl FnOnce(&[u8], &Value) -> serde_json::Result<Vec<u8>>,
) -> Prepared {
    let gate = |forwarded: &Value| breaker_on.then(|| count_request_tokens(forwarded));
    if !applied.changed() {
        // No pass touched the request, so its own bytes go out. Parsing JSON
        // and writing it back is not the identity (key order, spacing, the
        // spelling of numbers and escapes), and to the API's prompt cache a
        // different key order is a different prompt.
        return Prepared {
            body: original,
            we_cached: false,
            gate_tokens: gate(optimized),
            saved: Optimization::default(),
        };
    }
    match write(&original, optimized) {
        Ok(bytes) => Prepared {
            body: Bytes::from(bytes),
            we_cached: applied.we_cached(),
            gate_tokens: gate(optimized),
            saved: applied.saved,
        },
        Err(e) => {
            tracing::warn!(
                "proxy: optimized request could not be written, forwarding the original ({e})"
            );
            // None of the rewrite is in the bytes that go out: no breakpoint
            // to claim, no savings to book, and the breaker judges the
            // request the client sent.
            let gate_tokens = serde_json::from_slice::<Value>(&original)
                .ok()
                .and_then(|json| gate(&json));
            Prepared {
                body: original,
                we_cached: false,
                gate_tokens,
                saved: Optimization::default(),
            }
        }
    }
}

/// Why the proxy refused a request for its size. Whichever limit it was, the
/// client gets the one answer [`too_large_response`] builds.
enum TooLarge {
    /// The circuit breaker: more estimated input tokens than
    /// `max_request_tokens` allows.
    Tokens { tokens: usize, cap: usize },
    /// More bytes than the proxy buffers ([`MAX_BODY`]).
    Bytes { limit: usize },
}

/// An Anthropic-shaped `request_too_large` error (HTTP 413, the API's own
/// status and type for it), so the client's SDK raises a readable exception
/// instead of a bare gateway failure it can't interpret.
fn too_large_response(why: TooLarge) -> Response {
    let message = match why {
        TooLarge::Tokens { tokens, cap } => format!(
            "mimir proxy: request is ~{tokens} input tokens, over the configured \
             cap of {cap} ([proxy] max_request_tokens / --max-request-tokens). \
             The request was NOT sent upstream and you were not billed for it. \
             Start a fresh session, or raise/disable the cap (0 = off)."
        ),
        TooLarge::Bytes { limit } => format!(
            "mimir proxy: request body is over the {} MiB the proxy buffers. \
             The request was NOT sent upstream and you were not billed for it. \
             Start a fresh session, or send less in one request.",
            limit / (1024 * 1024)
        ),
    };
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": "request_too_large",
            "message": message,
        }
    });
    Response::builder()
        .status(StatusCode::PAYLOAD_TOO_LARGE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("static oversize response is always well-formed")
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
    if opt.deduped == 0 && opt.pruned == 0 {
        return;
    }
    let conn = ledger(db);
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

/// The response field cache savings are read from, as it appears on the
/// wire. A streamed reply carries it in `message_start`, its first event; a
/// non-streamed one in the `usage` object that follows all of its content.
const CACHE_READ_KEY: &str = "\"cache_read_input_tokens\"";

/// A passive tap on the response: finds `cache_read_input_tokens` as the
/// bytes stream past and records the REAL cache savings, once. Forwards
/// nothing.
///
/// Assumes the Anthropic SSE/JSON shape. On any other shape it simply finds
/// nothing and records nothing.
struct UsageScanner {
    db: Option<Db>,
    scan: FieldScan<'static>,
}

impl UsageScanner {
    fn new(db: Option<Db>) -> Self {
        UsageScanner {
            db,
            scan: FieldScan::new(CACHE_READ_KEY),
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        if let Some(cache_read) = self.scan.feed(bytes) {
            self.record(cache_read);
        }
    }

    fn record(&self, cache_read: u64) {
        if cache_read == 0 {
            return;
        }
        let Some(db) = &self.db else { return };
        let conn = ledger(db);
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

/// Finds the first `"key": <digits>` in a byte stream that arrives in chunks
/// of any size. Memory is constant however long the stream: whatever a chunk
/// boundary can split — the key, the separator, the digits — is carried in
/// [`ScanState`], and nothing is buffered.
struct FieldScan<'k> {
    /// The key with its quotes. Those two are its only quote bytes, so while
    /// a match is partial nothing already read can be the start of another,
    /// and a failed match starts over at the byte that broke it without
    /// looking back.
    ///
    /// After a complete match that is off by one byte: the closing quote
    /// could open the key again (`"key"key": 5`), and that overlapping
    /// second occurrence is not looked for. JSON cannot spell it, since a
    /// string is never followed directly by more text, so only a stream
    /// that is not JSON reads differently for it.
    key: &'k [u8],
    state: ScanState,
}

#[derive(Clone, Copy)]
enum ScanState {
    /// Looking for the key. Its first `matched` bytes are the last ones seen.
    Key { matched: usize },
    /// Past the key, before its value: whitespace and one colon may come.
    Separator { colon: bool },
    /// In the value's digits. They are the number only once a byte that is
    /// not a digit has followed: until then, more may be on the way.
    Digits { value: u64 },
    /// Found. Nothing further is read.
    Done,
}

impl<'k> FieldScan<'k> {
    fn new(key: &'k str) -> Self {
        let key = key.as_bytes();
        debug_assert!(
            key.len() >= 2 && key.iter().filter(|&&b| b == b'"').count() == 2,
            "the key is a JSON string, quotes included"
        );
        FieldScan {
            key,
            state: ScanState::Key { matched: 0 },
        }
    }

    /// Read the next chunk. Returns the value on the call that completes it;
    /// `None` on every call before, and after.
    ///
    /// A value that is not a run of digits (`null`, a string) is passed over
    /// and the search goes on to the next occurrence of the key.
    fn feed(&mut self, bytes: &[u8]) -> Option<u64> {
        for &b in bytes {
            self.state = match self.state {
                ScanState::Done => return None,
                ScanState::Key { matched } if b == self.key[matched] => {
                    if matched + 1 == self.key.len() {
                        ScanState::Separator { colon: false }
                    } else {
                        ScanState::Key {
                            matched: matched + 1,
                        }
                    }
                }
                ScanState::Key { .. } => self.restart(b),
                ScanState::Separator { colon } => match b {
                    b' ' | b'\t' | b'\r' | b'\n' => ScanState::Separator { colon },
                    b':' if !colon => ScanState::Separator { colon: true },
                    b'0'..=b'9' if colon => ScanState::Digits {
                        value: u64::from(b - b'0'),
                    },
                    _ => self.restart(b),
                },
                ScanState::Digits { value } => match b {
                    b'0'..=b'9' => {
                        let grown = value
                            .checked_mul(10)
                            .and_then(|v| v.checked_add(u64::from(b - b'0')));
                        match grown {
                            Some(value) => ScanState::Digits { value },
                            // Past u64: whatever this is, it is not a token count.
                            None => self.restart(b),
                        }
                    }
                    _ => {
                        self.state = ScanState::Done;
                        return Some(value);
                    }
                },
            };
        }
        None
    }

    /// The state after a byte that did not continue the match. That byte may
    /// itself be the quote that opens the key.
    fn restart(&self, b: u8) -> ScanState {
        ScanState::Key {
            matched: usize::from(b == self.key[0]),
        }
    }
}

/// Parse `"key": <digits>` out of a complete JSON fragment.
#[cfg(test)]
fn field_u64(s: &str, key: &str) -> Option<u64> {
    FieldScan::new(key).feed(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{field_u64, prune_events_once, prune_session_state_once, Db};
    use std::sync::{Arc, Mutex};

    /// An `AppState` whose upstream is a closed port, so "did we forward?" is
    /// observable: a forwarded request fails to connect (502 via `handle`),
    /// while a rejected one never gets that far.
    fn state_with_cap(max_request_tokens: usize) -> super::AppState {
        super::AppState {
            cfg: super::ProxyConfig {
                bind: "127.0.0.1:0".parse().unwrap(),
                // Port 1 is reserved and never listening — instant refusal.
                upstream: "http://127.0.0.1:1".into(),
                dry_run: false,
                cache: false,
                cache_ttl: super::CacheTtl::FiveMin,
                dedup: false,
                prune: false,
                max_request_tokens,
                db_path: None,
                event_retention_days: None,
            },
            client: reqwest::Client::new(),
            db: None,
        }
    }

    fn messages_request(system: &str) -> axum::extract::Request {
        let body = serde_json::json!({
            "model": "claude-opus-5",
            "system": system,
            "messages": [{"role": "user", "content": "hi"}],
        });
        axum::http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/messages")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn oversized_request_is_rejected_and_never_forwarded() {
        let state = state_with_cap(50);
        // ~1000 tokens of system prompt, far over the 50-token cap.
        let resp = super::proxy(&state, messages_request(&"word ".repeat(1000)))
            .await
            .expect("rejection is a normal response, not a transport error");

        assert_eq!(resp.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Must be Anthropic-shaped so the client SDK raises a real exception.
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "request_too_large");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not billed"),
            "the error must tell the user they were not billed"
        );
    }

    #[tokio::test]
    async fn under_cap_request_is_forwarded() {
        let state = state_with_cap(50_000);
        // Reaching a closed upstream is an Err — which is itself the proof
        // that we attempted the forward instead of short-circuiting.
        let result = super::proxy(&state, messages_request("short prompt")).await;
        assert!(
            result.is_err(),
            "an under-cap request must be forwarded (and so fail on the dead upstream)"
        );
    }

    #[tokio::test]
    async fn cap_of_zero_disables_the_breaker() {
        let state = state_with_cap(0);
        let result = super::proxy(&state, messages_request(&"word ".repeat(10_000))).await;
        assert!(
            result.is_err(),
            "cap=0 must forward even a huge request rather than reject it"
        );
    }

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
        assert_eq!(
            remaining, 1,
            "only the 200-day-old row clears the 180-day cutoff"
        );
    }

    /// The proxy runs the core's session prune, so a session that is still
    /// writing keeps all of its state (its transcript marker and its brief
    /// fire count are older than its newest row by design), and one that
    /// has been silent past the window loses all of it.
    #[test]
    fn prune_session_state_once_removes_sessions_silent_past_the_seven_day_window() {
        let conn = mimir_core::db::open_in_memory().unwrap();
        let now = mimir_core::model::now_unix();
        let old_at = now - 10 * 86_400;
        let recent_at = now - 86_400;
        for (session, key, at) in [
            ("running", "old", old_at),
            ("running", "recent", recent_at),
            ("silent", "old", old_at),
        ] {
            conn.execute(
                "INSERT INTO session_state (session_id, key, value, updated_at) \
                 VALUES (?1, ?2, 'v', ?3)",
                rusqlite::params![session, key, at],
            )
            .unwrap();
        }

        let db: Db = Arc::new(Mutex::new(conn));
        prune_session_state_once(&db);

        let remaining: Vec<String> = {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT session_id || ':' || key FROM session_state ORDER BY 1")
                .unwrap();
            let rows = stmt.query_map([], |r| r.get(0)).unwrap();
            rows.collect::<rusqlite::Result<_>>().unwrap()
        };
        assert_eq!(remaining, ["running:old", "running:recent"]);
    }

    fn memory_db() -> Db {
        Arc::new(Mutex::new(mimir_core::db::open_in_memory().unwrap()))
    }

    /// `(tokens_before, tokens_after)` of every ledger row from `source`.
    fn ledger_rows(db: &Db, source: &str) -> Vec<(i64, i64)> {
        let conn = db.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn
            .prepare("SELECT tokens_before, tokens_after FROM savings_event WHERE source = ?1")
            .unwrap();
        let rows = stmt
            .query_map([source], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        rows.map(Result::unwrap).collect()
    }

    /// A stand-in for the API on a loopback port. It answers every request
    /// the same way and keeps the bodies it was sent.
    struct Upstream {
        url: String,
        received: Arc<Mutex<Vec<Vec<u8>>>>,
    }

    impl Upstream {
        async fn answering(
            status: u16,
            headers: &[(&'static str, &'static str)],
            body: &'static str,
        ) -> Upstream {
            let received = Arc::new(Mutex::new(Vec::new()));
            let seen = received.clone();
            let headers = headers.to_vec();
            let app = axum::Router::new().fallback(axum::routing::any(
                move |req: axum::extract::Request| {
                    let seen = seen.clone();
                    let headers = headers.clone();
                    async move {
                        let bytes = axum::body::to_bytes(req.into_body(), usize::MAX)
                            .await
                            .unwrap();
                        seen.lock().unwrap().push(bytes.to_vec());
                        let mut resp = axum::response::Response::builder().status(status);
                        for (name, value) in headers {
                            resp = resp.header(name, value);
                        }
                        resp.body(axum::body::Body::from(body)).unwrap()
                    }
                },
            ));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Upstream { url, received }
        }

        /// A streamed reply whose `message_start` reports 5,000 tokens read
        /// from cache.
        async fn reporting_cache_read(headers: &[(&'static str, &'static str)]) -> Upstream {
            Upstream::answering(200, headers, CACHE_READ_5000).await
        }

        fn received(&self) -> Vec<Vec<u8>> {
            self.received.lock().unwrap().clone()
        }
    }

    const CACHE_READ_5000: &str = "event: message_start\ndata: {\"type\":\"message_start\",\
        \"message\":{\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":0,\
        \"cache_read_input_tokens\":5000,\"output_tokens\":1}}}\n\n";

    /// A proxy in front of `upstream` with every pass off and a ledger.
    fn state_for(upstream: &str) -> (super::AppState, Db) {
        let db = memory_db();
        let mut state = state_with_cap(0);
        state.cfg.upstream = upstream.to_owned();
        state.db = Some(db.clone());
        (state, db)
    }

    fn post_messages(body: impl Into<axum::body::Body>) -> axum::extract::Request {
        axum::http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/messages")
            .body(body.into())
            .unwrap()
    }

    /// Send `req` through the proxy and read the whole reply, as a client
    /// would: the usage tap only sees what is actually streamed back.
    async fn round_trip(
        state: &super::AppState,
        req: axum::extract::Request,
    ) -> (axum::http::StatusCode, Vec<u8>) {
        let resp = super::proxy(state, req).await.expect("upstream answered");
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, body.to_vec())
    }

    /// A conversation whose third message repeats the first: dedup has
    /// something to save.
    fn body_with_a_repeat() -> String {
        let big = "x ".repeat(400);
        serde_json::json!({
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": big}]},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": [{"type": "text", "text": big}]},
            ],
        })
        .to_string()
    }

    #[tokio::test]
    async fn an_unchanged_request_is_forwarded_byte_for_byte() {
        let up = Upstream::answering(200, &[], "{}").await;
        let (mut state, _) = state_for(&up.url);
        state.cfg.cache = true;
        state.cfg.dedup = true;
        state.cfg.prune = true;
        // Odd spacing, keys out of order, an escape and a float that a
        // re-serialisation would all normalise. The client's own breakpoint
        // keeps the cache pass out; nothing repeats; nothing is old.
        let body = "{ \"model\":\"claude-opus-5\",  \"max_tokens\": 1.0,\n \"messages\":[{\"role\":\"user\",\
                    \"content\":[{\"type\":\"text\",\"text\":\"caf\\u00e9\",\
                    \"cache_control\":{\"type\":\"ephemeral\"}}]}] }";
        let (status, _) = round_trip(&state, post_messages(body)).await;
        assert_eq!(status, 200);
        assert_eq!(up.received(), vec![body.as_bytes().to_vec()]);
    }

    #[tokio::test]
    async fn a_rewritten_request_keeps_what_no_pass_changed() {
        let up = Upstream::answering(200, &[], "{}").await;
        let (mut state, _) = state_for(&up.url);
        state.cfg.cache = true;
        // Keys in the order a client writes them, which is not the alphabet's.
        let body = r#"{"model":"claude-opus-5","max_tokens":1024,"tools":[{"name":"run","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"string"}}}}],"system":"be careful","messages":[{"role":"user","content":[{"type":"text","text":"go"}]}]}"#;
        round_trip(&state, post_messages(body)).await;

        let expected = r#"{"model":"claude-opus-5","max_tokens":1024,"tools":[{"name":"run","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"string"}}}}],"system":[{"cache_control":{"type":"ephemeral"},"text":"be careful","type":"text"}],"messages":[{"role":"user","content":[{"type":"text","text":"go","cache_control":{"type":"ephemeral"}}]}]}"#;
        assert_eq!(
            String::from_utf8(up.received().remove(0)).unwrap(),
            expected
        );
    }

    #[tokio::test]
    async fn the_first_rewrite_in_a_conversation_leaves_its_prefix_alone() {
        // The client caches for itself, so the proxy touches nothing until a
        // block repeats. On that turn the request is rewritten for the first
        // time, and everything before the repeat must come out exactly as
        // the turn before sent it, or the client's cached prefix is lost.
        let up = Upstream::answering(200, &[], "{}").await;
        let (mut state, _) = state_for(&up.url);
        state.cfg.cache = true;
        state.cfg.dedup = true;
        let big = "x ".repeat(400);
        let head = format!(
            r#"{{"model":"claude-opus-5","tools":[{{"name":"run","input_schema":{{"type":"object","properties":{{"zeta":{{"type":"string"}},"alpha":{{"type":"string"}}}}}},"cache_control":{{"type":"ephemeral"}}}}],"messages":[{{"role":"user","content":[{{"type":"text","text":"{big}"}}]}},{{"role":"assistant","content":[{{"type":"tool_use","id":"t1","name":"run","input":{{"zeta":"1","alpha":"2"}}}}]}}"#
        );
        let first = format!("{head}]}}");
        let second = format!(
            r#"{head},{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"t1","content":"{big}"}}]}}]}}"#
        );
        round_trip(&state, post_messages(first.clone())).await;
        round_trip(&state, post_messages(second)).await;

        let sent: Vec<String> = up
            .received()
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).unwrap())
            .collect();
        assert_eq!(sent[0], first, "nothing to change on the first turn");
        assert!(
            sent[1].contains("elided by mimir proxy"),
            "the second turn is rewritten"
        );
        assert!(
            sent[1].starts_with(&head),
            "the rewritten turn must start with the bytes of the turn before:\n{}",
            sent[1]
        );
    }

    #[tokio::test]
    async fn a_request_we_added_no_breakpoint_to_claims_no_cache_reads() {
        let up = Upstream::reporting_cache_read(&[]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.cache = true;

        // Nothing to put a breakpoint on.
        let empty = r#"{"model":"claude-opus-5","messages":[]}"#;
        // The client caches for itself, on a tool definition.
        let own = r#"{"model":"claude-opus-5","tools":[{"name":"t","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],"system":"be careful","messages":[{"role":"user","content":"hi"}]}"#;
        for body in [empty, own] {
            round_trip(&state, post_messages(body)).await;
        }

        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE),
            vec![],
            "cache reads on these requests are not the proxy's doing"
        );
        assert_eq!(
            up.received(),
            vec![empty.as_bytes().to_vec(), own.as_bytes().to_vec()],
            "and neither request got a second layer of breakpoints"
        );
    }

    #[tokio::test]
    async fn a_request_we_added_a_breakpoint_to_claims_its_cache_reads() {
        let up = Upstream::reporting_cache_read(&[]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.cache = true;
        round_trip(&state, messages_request("be careful")).await;

        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE),
            vec![(5000, 500)]
        );
        let sent: serde_json::Value = serde_json::from_slice(&up.received()[0]).unwrap();
        assert_eq!(sent["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[tokio::test]
    async fn a_compressed_response_is_not_scanned() {
        // Whatever the bytes of a gzip stream happen to spell, they are not
        // the usage block.
        let up = Upstream::reporting_cache_read(&[("content-encoding", "gzip")]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.cache = true;
        let (_, body) = round_trip(&state, messages_request("be careful")).await;

        assert_eq!(body, CACHE_READ_5000.as_bytes(), "passed through untouched");
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE),
            vec![]
        );

        // `identity` is the spelled-out way of saying "not compressed".
        let up = Upstream::reporting_cache_read(&[("content-encoding", "identity")]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.cache = true;
        round_trip(&state, messages_request("be careful")).await;
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE).len(),
            1
        );
    }

    #[tokio::test]
    async fn savings_are_booked_when_upstream_accepts_the_request() {
        let up = Upstream::answering(200, &[], "{}").await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.dedup = true;
        round_trip(&state, post_messages(body_with_a_repeat())).await;
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP).len(),
            1
        );
    }

    #[tokio::test]
    async fn savings_are_not_booked_for_a_request_upstream_refused() {
        let up = Upstream::answering(400, &[], r#"{"type":"error"}"#).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.dedup = true;
        let (status, _) = round_trip(&state, post_messages(body_with_a_repeat())).await;
        assert_eq!(status, 400);
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP),
            vec![]
        );
    }

    #[tokio::test]
    async fn savings_are_not_booked_for_a_request_that_was_never_sent() {
        // Rejected by the circuit breaker …
        let (mut state, db) = state_for("http://127.0.0.1:1");
        state.cfg.dedup = true;
        state.cfg.max_request_tokens = 50;
        let resp = super::proxy(&state, post_messages(body_with_a_repeat()))
            .await
            .unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP),
            vec![]
        );

        // … or lost on the way to an upstream that is not there.
        let (mut state, db) = state_for("http://127.0.0.1:1");
        state.cfg.dedup = true;
        assert!(super::proxy(&state, post_messages(body_with_a_repeat()))
            .await
            .is_err());
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP),
            vec![]
        );
    }

    #[tokio::test]
    async fn a_body_over_the_buffer_limit_gets_the_same_413_as_the_breaker() {
        let state = std::sync::Arc::new(state_with_cap(0));
        let resp = super::handle(
            axum::extract::State(state),
            post_messages(vec![0u8; super::MAX_BODY + 1]),
        )
        .await;

        assert_eq!(resp.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            resp.headers()[axum::http::header::CONTENT_TYPE],
            "application/json"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "request_too_large");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not billed"));
    }

    #[test]
    fn a_poisoned_ledger_lock_is_taken_back() {
        let db = memory_db();
        let holder = db.clone();
        let panicked = std::thread::spawn(move || {
            let _guard = holder.lock().unwrap();
            panic!("a ledger write went wrong");
        })
        .join();
        assert!(panicked.is_err() && db.is_poisoned());

        super::record_request_savings(
            &Some(db.clone()),
            &super::Optimization {
                deduped: 700,
                pruned: 0,
            },
        );
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP),
            vec![(700, 0)],
            "one panic must not switch the ledger off for good"
        );
        assert!(!db.is_poisoned(), "and the next writer finds a clean lock");
    }

    #[tokio::test]
    async fn buffer_body_stops_at_the_limit_however_the_body_is_chunked() {
        let chunked = |parts: &[&'static str]| {
            let chunks: Vec<Result<super::Bytes, std::io::Error>> = parts
                .iter()
                .map(|p| Ok(super::Bytes::from_static(p.as_bytes())))
                .collect();
            axum::body::Body::from_stream(futures_util::stream::iter(chunks))
        };
        match super::buffer_body(chunked(&["abc", "", "def"]), 6)
            .await
            .unwrap()
        {
            super::Buffered::Whole(bytes) => assert_eq!(&bytes[..], b"abcdef"),
            super::Buffered::OverLimit => panic!("a body of exactly the limit fits"),
        }
        for parts in [&["abcdefg"][..], &["abc", "def", "g"][..]] {
            assert!(matches!(
                super::buffer_body(chunked(parts), 6).await.unwrap(),
                super::Buffered::OverLimit
            ));
        }
    }

    fn passes(cache: bool, dedup: bool) -> super::OptimizeOpts {
        super::OptimizeOpts {
            cache,
            cache_ttl: super::CacheTtl::FiveMin,
            dedup,
            prune: false,
        }
    }

    #[test]
    fn a_rewrite_that_cannot_be_serialized_claims_nothing() {
        let original = super::Bytes::from(body_with_a_repeat());
        let json: serde_json::Value = serde_json::from_slice(&original).unwrap();
        let tokens_as_sent = super::count_request_tokens(&json);
        let (optimized, applied) = super::optimize::apply_passes(json, passes(true, true));
        assert!(applied.we_cached() && applied.saved.deduped > 0);
        assert!(super::count_request_tokens(&optimized) < tokens_as_sent);

        let prepared = super::finish(original.clone(), &optimized, applied, true, |_, _| {
            Err(serde_json::from_str::<serde_json::Value>("{").unwrap_err())
        });

        assert_eq!(prepared.body, original, "the client's bytes go out");
        assert!(!prepared.we_cached, "no breakpoint of ours is in them");
        assert_eq!(prepared.saved, super::Optimization::default());
        assert_eq!(
            prepared.gate_tokens,
            Some(tokens_as_sent),
            "the breaker judges the request that is forwarded"
        );
    }

    #[test]
    fn a_rewritten_request_carries_the_flags_of_its_rewrite() {
        let original = super::Bytes::from(body_with_a_repeat());
        let json: serde_json::Value = serde_json::from_slice(&original).unwrap();
        let (optimized, applied) = super::optimize::apply_passes(json, passes(true, true));
        let saved = applied.saved.deduped;

        let prepared = super::finish(
            original.clone(),
            &optimized,
            applied,
            true,
            super::write_rewritten,
        );

        assert_ne!(prepared.body, original);
        let sent: serde_json::Value = serde_json::from_slice(&prepared.body).unwrap();
        assert_eq!(sent, optimized);
        assert!(prepared.we_cached);
        assert_eq!(prepared.saved.deduped, saved);
        assert_eq!(
            prepared.gate_tokens,
            Some(super::count_request_tokens(&optimized))
        );
    }

    fn tapped(chunks: &[&[u8]]) -> Vec<(i64, i64)> {
        let db = memory_db();
        let mut scanner = super::UsageScanner::new(Some(db.clone()));
        for chunk in chunks {
            scanner.feed(chunk);
        }
        ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE)
    }

    #[test]
    fn tap_reads_the_same_value_wherever_the_chunks_break() {
        let whole = CACHE_READ_5000.as_bytes();
        assert_eq!(tapped(&[whole]), vec![(5000, 500)]);
        for cut in 1..whole.len() {
            let (a, b) = whole.split_at(cut);
            assert_eq!(tapped(&[a, b]), vec![(5000, 500)], "cut at byte {cut}");
        }
        // One byte at a time.
        let bytes: Vec<&[u8]> = whole.chunks(1).collect();
        assert_eq!(tapped(&bytes), vec![(5000, 500)]);
    }

    #[test]
    fn tap_does_not_take_a_number_that_may_still_be_growing() {
        // The digits end the input: nothing says the number is complete.
        assert_eq!(tapped(&[b"\"cache_read_input_tokens\":20"]), vec![]);
        assert_eq!(
            tapped(&[b"\"cache_read_input_tokens\":20", b"48}"]),
            vec![(2048, 204)]
        );
    }

    #[test]
    fn tap_finds_usage_at_the_end_of_a_large_body() {
        // A non-streaming response: `usage` comes after all of the content.
        let content = format!(
            "{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}],",
            "word ".repeat(200_000)
        );
        let usage = "\"usage\":{\"input_tokens\":3,\"cache_read_input_tokens\":900}}";
        assert!(content.len() > 512 * 1024);
        assert_eq!(
            tapped(&[content.as_bytes(), usage.as_bytes()]),
            vec![(900, 90)]
        );
    }

    #[test]
    fn tap_reads_past_what_is_not_the_number() {
        // A null count, the key quoted inside model output, then the real one.
        let stream = "\"cache_read_input_tokens\":null,\
            \"text\":\"the \\\"cache_read_input_tokens\\\": 1 field\",\
            \"cache_read_\"cache_read_input_tokens\" : 77,";
        assert_eq!(tapped(&[stream.as_bytes()]), vec![(77, 7)]);
    }

    #[test]
    fn tap_records_once() {
        let twice = [CACHE_READ_5000.as_bytes(), CACHE_READ_5000.as_bytes()];
        assert_eq!(tapped(&twice), vec![(5000, 500)]);
    }

    #[test]
    fn tap_wants_a_colon_between_the_key_and_its_number() {
        assert_eq!(tapped(&[b"\"cache_read_input_tokens\" 20,"]), vec![]);
        assert_eq!(
            tapped(&[b"\"cache_read_input_tokens\" 20, \"cache_read_input_tokens\": 30,"]),
            vec![(30, 3)]
        );
    }

    #[test]
    fn tap_books_nothing_for_a_response_that_read_nothing_from_cache() {
        let miss = "data: {\"type\":\"message_start\",\"message\":{\"usage\":\
            {\"cache_creation_input_tokens\":900,\"cache_read_input_tokens\":0}}}\n\n";
        assert_eq!(tapped(&[miss.as_bytes()]), vec![]);
    }

    #[test]
    fn a_poisoned_ledger_lock_still_takes_cache_reads() {
        let db = memory_db();
        let holder = db.clone();
        let panicked = std::thread::spawn(move || {
            let _guard = holder.lock().unwrap();
            panic!("a ledger write went wrong");
        })
        .join();
        assert!(panicked.is_err() && db.is_poisoned());

        let mut scanner = super::UsageScanner::new(Some(db.clone()));
        scanner.feed(CACHE_READ_5000.as_bytes());
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE),
            vec![(5000, 500)]
        );
    }

    /// The Claude Code shape: the client places its own breakpoints, and a
    /// block repeats, so dedup is the only pass with anything to do.
    fn body_with_a_repeat_and_its_own_breakpoints() -> String {
        let big = "x ".repeat(400);
        serde_json::json!({
            "model": "claude-opus-5",
            "system": [{"type": "text", "text": "be careful", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": big}]},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": [
                    {"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}
                ]},
            ],
        })
        .to_string()
    }

    #[tokio::test]
    async fn a_request_only_dedup_changed_claims_no_cache_reads() {
        // The request is rewritten, so it is "changed" — but the breakpoints
        // in it are the client's, and so are the cache reads they earn.
        let up = Upstream::reporting_cache_read(&[]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.cache = true;
        state.cfg.dedup = true;
        let body = body_with_a_repeat_and_its_own_breakpoints();
        round_trip(&state, post_messages(body.clone())).await;

        let sent = String::from_utf8(up.received().remove(0)).unwrap();
        assert!(sent.contains("elided by mimir proxy"), "dedup rewrote it");
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_DEDUP).len(),
            1
        );
        assert_eq!(
            ledger_rows(&db, mimir_core::savings::source::PROXY_CACHE),
            vec![],
            "the response's cache reads are not the proxy's doing"
        );

        // The same at the seam the forwarder reads its flags from.
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        let (optimized, applied) = super::optimize::apply_passes(json, passes(true, true));
        assert!(applied.changed() && !applied.we_cached());
        let prepared = super::finish(
            super::Bytes::from(body),
            &optimized,
            applied,
            false,
            super::write_rewritten,
        );
        assert!(!prepared.we_cached);
    }

    #[tokio::test]
    async fn a_dry_run_changes_nothing_and_books_nothing() {
        let up = Upstream::reporting_cache_read(&[]).await;
        let (mut state, db) = state_for(&up.url);
        state.cfg.dry_run = true;
        state.cfg.cache = true;
        state.cfg.dedup = true;
        state.cfg.prune = true;
        // A cap this request is far over. A dry run measures; it does not
        // reject either.
        state.cfg.max_request_tokens = 50;

        let body = body_with_a_repeat();
        let (status, _) = round_trip(&state, post_messages(body.clone())).await;

        assert_eq!(status, 200);
        assert_eq!(
            up.received(),
            vec![body.into_bytes()],
            "forwarded as it came"
        );
        for source in [
            mimir_core::savings::source::PROXY_CACHE,
            mimir_core::savings::source::PROXY_DEDUP,
            mimir_core::savings::source::PROXY_PRUNE,
        ] {
            assert_eq!(ledger_rows(&db, source), vec![], "{source}");
        }
    }

    #[test]
    fn a_request_the_patcher_refuses_is_serialized_whole() {
        // No pass takes a field away, so the pair is made by hand: it is the
        // one change the patcher will not write.
        let original = br#"{"zeta":1,"alpha":2}"#;
        let optimized = serde_json::json!({"zeta": 1});
        assert_eq!(super::patch::rewrite(original, &optimized), None);

        let bytes = super::write_rewritten(original, &optimized).unwrap();
        assert_eq!(bytes, br#"{"zeta":1}"#);
    }

    #[test]
    fn a_request_is_counted_only_for_a_breaker_that_is_on() {
        let body = super::Bytes::from(body_with_a_repeat());
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let as_sent = super::count_request_tokens(&json);

        for dedup in [false, true] {
            let mut off = state_with_cap(0);
            off.cfg.dedup = dedup;
            assert_eq!(
                super::prepare_messages_body(&off, body.clone()).gate_tokens,
                None,
                "nobody reads the count, so nobody pays for it"
            );

            let mut on = state_with_cap(1_000_000);
            on.cfg.dedup = dedup;
            let counted = super::prepare_messages_body(&on, body.clone())
                .gate_tokens
                .expect("a breaker that is on judges every request");
            if dedup {
                assert!(counted < as_sent, "it judges what is forwarded");
            } else {
                assert_eq!(counted, as_sent);
            }
        }
    }
}
