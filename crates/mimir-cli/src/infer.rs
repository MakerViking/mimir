//! Blocking HTTP client for `mimir daemon` inference delegation (`ureq`,
//! no async runtime — the engine that calls this is sync).
//!
//! One `DaemonClient` is attached to a session's engine at startup (see
//! `mcp::run`). Failure semantics are "degrade, never block": a transport
//! error or unexpected status marks the daemon dead for a cooldown window
//! (`available()` goes false, so the engine works locally at zero probe
//! cost), a 503 backs off longer (the daemon has no model — that won't
//! change soon), and a 401 or model mismatch disables delegation for the
//! session outright (neither self-heals; retrying would spam a misconfig
//! warning forever).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mimir_core::error::{Error, Result};
use mimir_core::{Mimir, RemoteInference};
use serde::Deserialize;

/// Cooldown after a transport error / unexpected status.
const DEAD_SHORT: Duration = Duration::from_secs(60);
/// Cooldown after a 503 (daemon up but model-less).
const DEAD_LONG: Duration = Duration::from_secs(300);
/// Connect budget: the daemon is loopback; anything slower is "down".
const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);

pub struct DaemonClient {
    /// `hooks.inject_url` minus its `/inject` path.
    base: String,
    token: Option<String>,
    embed_model: String,
    rerank_model: String,
    agent: ureq::Agent,
    dead_until: Mutex<Option<Instant>>,
    disabled: AtomicBool,
    dead_short: Duration,
    dead_long: Duration,
}

impl DaemonClient {
    pub fn from_config(config: &mimir_core::config::Config) -> Self {
        let trimmed = config.hooks.inject_url.trim_end_matches('/');
        let base = trimmed
            .strip_suffix("/inject")
            .unwrap_or(trimmed)
            .to_string();
        Self::new(
            base,
            crate::mcp::http_token_from_env(),
            config.embedding.model.clone(),
            config.rerank.model.clone(),
        )
    }

    fn new(base: String, token: Option<String>, embed_model: String, rerank_model: String) -> Self {
        DaemonClient {
            base,
            token,
            embed_model,
            rerank_model,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(CONNECT_TIMEOUT)
                .build(),
            dead_until: Mutex::new(None),
            disabled: AtomicBool::new(false),
            dead_short: DEAD_SHORT,
            dead_long: DEAD_LONG,
        }
    }

    /// `GET /inference`: is a daemon up, and does it serve OUR models?
    /// A name mismatch disables delegation permanently — vectors from a
    /// different embedder would silently poison the store.
    pub fn probe(&self) -> bool {
        #[derive(Deserialize)]
        struct Info {
            embedding_model: String,
            rerank_model: String,
        }
        let req = self
            .with_auth(self.agent.get(&format!("{}/inference", self.base)))
            .timeout(Duration::from_secs(1));
        match req.call() {
            Ok(r) => match r.into_json::<Info>() {
                Ok(info) => {
                    if info.embedding_model != self.embed_model
                        || info.rerank_model != self.rerank_model
                    {
                        tracing::warn!(
                            daemon_embed = %info.embedding_model,
                            daemon_rerank = %info.rerank_model,
                            local_embed = %self.embed_model,
                            local_rerank = %self.rerank_model,
                            "daemon serves different models; inference delegation disabled"
                        );
                        self.disabled.store(true, Ordering::Relaxed);
                        return false;
                    }
                    true
                }
                Err(_) => {
                    self.mark_dead(self.dead_short);
                    false
                }
            },
            Err(e) => {
                self.on_error(&e, "probe");
                false
            }
        }
    }

    fn with_auth(&self, req: ureq::Request) -> ureq::Request {
        match &self.token {
            Some(t) => req.set("Authorization", &format!("Bearer {t}")),
            None => req,
        }
    }

    fn mark_dead(&self, cooldown: Duration) {
        let mut dead = self.dead_until.lock().unwrap_or_else(|p| p.into_inner());
        *dead = Some(Instant::now() + cooldown);
    }

    /// Shared failure bookkeeping; returns the message for the caller's Err.
    fn on_error(&self, e: &ureq::Error, what: &str) -> String {
        match e {
            ureq::Error::Status(401, _) => {
                tracing::warn!(
                    "daemon rejected the bearer token (401); inference delegation \
                     disabled for this session — check MIMIR_HTTP_TOKEN"
                );
                self.disabled.store(true, Ordering::Relaxed);
                format!("daemon {what}: 401")
            }
            ureq::Error::Status(503, _) => {
                self.mark_dead(self.dead_long);
                format!("daemon {what}: model unavailable (503)")
            }
            ureq::Error::Status(code, _) => {
                self.mark_dead(self.dead_short);
                format!("daemon {what}: HTTP {code}")
            }
            other => {
                self.mark_dead(self.dead_short);
                format!("daemon {what}: {other}")
            }
        }
    }
}

impl RemoteInference for DaemonClient {
    fn available(&self) -> bool {
        if self.disabled.load(Ordering::Relaxed) {
            return false;
        }
        let dead = self.dead_until.lock().unwrap_or_else(|p| p.into_inner());
        dead.is_none_or(|t| Instant::now() >= t)
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Deserialize)]
        struct EmbedResponse {
            model: String,
            vectors: Vec<Vec<f32>>,
        }
        // Chunked so the daemon's shared engine mutex is released between
        // chunks — a big bulk embed must not starve its /inject hook calls.
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(mimir_core::embed::EMBED_BATCH) {
            let req = self
                .with_auth(self.agent.post(&format!("{}/embed", self.base)))
                .timeout(Duration::from_secs(10));
            let resp = req
                .send_json(serde_json::json!({ "texts": chunk }))
                .map_err(|e| Error::Embed(self.on_error(&e, "embed")))?;
            let resp: EmbedResponse = resp
                .into_json()
                .map_err(|e| Error::Embed(format!("daemon embed: bad response: {e}")))?;
            if resp.model != self.embed_model {
                tracing::warn!(
                    daemon = %resp.model, local = %self.embed_model,
                    "daemon embeds with a different model; delegation disabled"
                );
                self.disabled.store(true, Ordering::Relaxed);
                return Err(Error::Embed("daemon embed: model mismatch".into()));
            }
            if resp.vectors.len() != chunk.len() {
                self.mark_dead(self.dead_short);
                return Err(Error::Embed(format!(
                    "daemon embed: {} vectors for {} texts",
                    resp.vectors.len(),
                    chunk.len()
                )));
            }
            out.extend(resp.vectors);
        }
        Ok(out)
    }

    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        #[derive(Deserialize)]
        struct RerankResponse {
            model: String,
            scores: Vec<f32>,
        }
        let req = self
            .with_auth(self.agent.post(&format!("{}/rerank", self.base)))
            .timeout(Duration::from_secs(5));
        let resp = req
            .send_json(serde_json::json!({ "query": query, "docs": docs }))
            .map_err(|e| Error::Embed(self.on_error(&e, "rerank")))?;
        let resp: RerankResponse = resp
            .into_json()
            .map_err(|e| Error::Embed(format!("daemon rerank: bad response: {e}")))?;
        if resp.model != self.rerank_model {
            tracing::warn!(
                daemon = %resp.model, local = %self.rerank_model,
                "daemon reranks with a different model; delegation disabled"
            );
            self.disabled.store(true, Ordering::Relaxed);
            return Err(Error::Embed("daemon rerank: model mismatch".into()));
        }
        if resp.scores.len() != docs.len() {
            self.mark_dead(self.dead_short);
            return Err(Error::Embed(format!(
                "daemon rerank: {} scores for {} docs",
                resp.scores.len(),
                docs.len()
            )));
        }
        Ok(resp.scores)
    }
}

/// Attach delegation to an engine per `[daemon] inference`. Returns whether
/// a live, model-compatible daemon answered the startup probe — callers use
/// that to decide whether to eager-load a local reranker as the fallback.
/// `"off"` (or an unknown value, warned) attaches nothing: pure-local.
pub fn attach(m: &mut Mimir) -> bool {
    match m.config.daemon.inference.as_str() {
        "auto" => {}
        "off" => return false,
        other => {
            tracing::warn!(
                inference = other,
                "unknown [daemon] inference value; treating as off"
            );
            return false;
        }
    }
    let client = DaemonClient::from_config(&m.config);
    let alive = client.probe();
    m.set_remote_inference(Box::new(client));
    alive
}

/// Like [`attach`], but without the startup probe: background engines
/// (auto-sync ticks, startup maintenance) only touch the daemon when real
/// work exists, so a per-engine probe would be pure overhead on the common
/// nothing-pending tick. First-use failures take the client's normal
/// cooldown path. Callers gate on `inference == "auto"` themselves (this
/// runs on threads where a repeated unknown-value warn would just spam).
pub fn attach_lazy(m: &mut Mimir) {
    let client = DaemonClient::from_config(&m.config);
    m.set_remote_inference(Box::new(client));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn client_for(base: String) -> DaemonClient {
        let mut c = DaemonClient::new(
            base,
            None,
            "bge-small-en-v1.5".into(),
            "jina-reranker-v1-turbo-en".into(),
        );
        // Compressed test cooldowns — but with margins wide enough for slow
        // CI runners: a "still in cooldown" assertion only turns flaky if
        // the runner stalls for the WHOLE remaining window (seconds, not
        // the tens-of-ms the first version of these tests assumed — that
        // flaked on macOS CI).
        c.dead_short = Duration::from_millis(100);
        c.dead_long = Duration::from_secs(2);
        c
    }

    /// One-shot HTTP stub: accepts a single connection, drains the FULL
    /// request (headers + Content-Length body), then answers with a fixed
    /// status + JSON body. Draining first matters: responding while the
    /// client is still writing makes Windows reset the connection
    /// (os error 10054), which turns these tests into transport-error
    /// tests — caught by CI, not locally on Linux.
    fn stub(status: u16, body: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                let mut header_end = None;
                let mut content_len = 0usize;
                loop {
                    match sock.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => req.extend_from_slice(&buf[..n]),
                    }
                    if header_end.is_none() {
                        if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                            header_end = Some(pos + 4);
                            content_len = String::from_utf8_lossy(&req[..pos])
                                .lines()
                                .filter_map(|l| {
                                    let (k, v) = l.split_once(':')?;
                                    if k.eq_ignore_ascii_case("content-length") {
                                        v.trim().parse().ok()
                                    } else {
                                        None
                                    }
                                })
                                .next()
                                .unwrap_or(0);
                        }
                    }
                    if let Some(h) = header_end {
                        if req.len() >= h + content_len {
                            break;
                        }
                    }
                }
                let _ = write!(
                    sock,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn unreachable_daemon_goes_dead_then_recovers() {
        // Bind-then-drop: the port refuses connections afterwards.
        let addr = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap()
        };
        let c = client_for(format!("http://{addr}"));
        assert!(c.available(), "fresh client starts available");
        assert!(c.embed(&["hi".into()]).is_err());
        assert!(!c.available(), "transport error must start the cooldown");
        assert!(
            !c.disabled.load(Ordering::Relaxed),
            "a cooldown, not a permanent disable"
        );
        std::thread::sleep(Duration::from_millis(150));
        assert!(c.available(), "cooldown expiry must re-enable delegation");
    }

    #[test]
    fn model_mismatch_disables_permanently() {
        let base = stub(
            200,
            r#"{"model":"some-other-model","dim":4,"vectors":[[0.1,0.2,0.3,0.4]]}"#,
        );
        let c = client_for(base);
        assert!(c.embed(&["hi".into()]).is_err());
        assert!(
            c.disabled.load(Ordering::Relaxed),
            "a model mismatch must disable, not cool down"
        );
        std::thread::sleep(Duration::from_millis(150));
        assert!(!c.available(), "disable must outlive the short cooldown");
    }

    #[test]
    fn unauthorized_disables_permanently() {
        let base = stub(401, r#"{"error":"unauthorized"}"#);
        let c = client_for(base);
        assert!(c.rerank("q", &["a".into()]).is_err());
        assert!(
            c.disabled.load(Ordering::Relaxed),
            "401 must disable, not cool down"
        );
        std::thread::sleep(Duration::from_millis(150));
        assert!(!c.available(), "disable must outlive the short cooldown");
    }

    #[test]
    fn model_less_daemon_backs_off_longer() {
        let base = stub(503, r#"{"error":"embedding model unavailable"}"#);
        let c = client_for(base);
        assert!(c.embed(&["hi".into()]).is_err());
        assert!(!c.available());
        assert!(
            !c.disabled.load(Ordering::Relaxed),
            "503 is a cooldown, not a permanent disable"
        );
        // Well past the 100ms short cooldown, well short of the 2s long one.
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            !c.available(),
            "503 uses the long cooldown, not the short one"
        );
        std::thread::sleep(Duration::from_secs(2));
        assert!(c.available());
    }

    #[test]
    fn good_response_round_trips() {
        let base = stub(
            200,
            r#"{"model":"bge-small-en-v1.5","dim":4,"vectors":[[0.1,0.2,0.3,0.4]]}"#,
        );
        let c = client_for(base);
        let v = c.embed(&["hi".into()]).unwrap();
        assert_eq!(v, vec![vec![0.1, 0.2, 0.3, 0.4]]);
        assert!(c.available());
    }

    #[test]
    fn probe_rejects_mismatched_daemon() {
        let base = stub(
            200,
            r#"{"embedding_model":"bge-small-en-v1.5","rerank_model":"bge-reranker-base"}"#,
        );
        let c = client_for(base);
        assert!(!c.probe());
        assert!(!c.available(), "mismatched daemon must disable delegation");
    }
}
