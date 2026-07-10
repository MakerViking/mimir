//! `mimir serve` — the sync hub (via `tiny_http`, single-threaded, blocking).
//!
//! The hub is just a normal Mimir store (at MIMIR_HOME) reachable over HTTP.
//! Clients `GET /sync/pull?since=<n>` and `POST /sync/push`, both authed with a
//! bearer token. Intended transport is a tailnet or a TLS reverse proxy — see
//! docs/sync.md. One static token, no rate limiting: do not expose raw to the
//! public internet.

use anyhow::{anyhow, Result};
use mimir_core::{replicate, Mimir};
use tiny_http::{Method, Response, Server};

pub fn serve(bind: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let token = resolve_token(&mimir)?;

    let server = Server::http(bind).map_err(|e| anyhow!("cannot bind {bind}: {e}"))?;
    println!(
        "mimir sync hub listening on http://{bind}  (db: {})",
        mimir.paths.db_file.display()
    );
    println!(
        "clients: set MIMIR_SYNC_TOKEN, then [sync] mode=\"server\", endpoint=\"http://{bind}\""
    );

    // Long-lived writing daemon: reclaim the WAL on an idle timer so its read
    // marks don't starve PASSIVE autocheckpoints (same profile as `mimir mcp`).
    // Same timer also prunes old `recall_event` rows ([learn] event_retention_days).
    mimir_core::db::spawn_checkpoint_timer(
        mimir.paths.db_file.clone(),
        mimir.config.learn.effective_retention_days(),
    );

    for req in server.incoming_requests() {
        handle(&mut mimir, req, &token);
    }
    Ok(())
}

fn handle(mimir: &mut Mimir, mut req: tiny_http::Request, token: &str) {
    let (code, body) = match process(mimir, &mut req, token) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("sync request error: {e}");
            (500, "error".to_string())
        }
    };
    let _ = req.respond(Response::from_string(body).with_status_code(code));
}

fn process(mimir: &mut Mimir, req: &mut tiny_http::Request, token: &str) -> Result<(u16, String)> {
    // ---- auth (constant-time bearer compare) ----
    let provided = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    if !ct_eq(provided.as_bytes(), format!("Bearer {token}").as_bytes()) {
        return Ok((401, "unauthorized".to_string()));
    }

    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("");
    match (req.method(), path) {
        (Method::Get, "/sync/pull") => {
            let since = url
                .split_once("since=")
                .and_then(|(_, s)| s.split('&').next())
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            let batch = replicate::changes_since(&mimir.conn, since)?;
            Ok((200, serde_json::to_string(&batch)?))
        }
        (Method::Post, "/sync/push") => {
            let mut body = String::new();
            req.as_reader().read_to_string(&mut body)?;
            let batch: replicate::SyncBatch = serde_json::from_str(&body)?;
            let applied = replicate::apply_changes(&mimir.conn, &batch)?;
            let _ = mimir.embed_pending(); // hub may carry its own model
            let watermark = replicate::current_high_watermark(&mimir.conn)?;
            Ok((
                200,
                serde_json::json!({ "applied": applied, "watermark": watermark }).to_string(),
            ))
        }
        _ => Ok((404, "not found".to_string())),
    }
}

/// Token resolution: env wins; else a stable token persisted in the hub's meta
/// (generated + printed on first run) so restarts keep the same token.
fn resolve_token(mimir: &Mimir) -> Result<String> {
    if let Ok(t) = std::env::var("MIMIR_SYNC_TOKEN") {
        if !t.is_empty() {
            return Ok(t);
        }
    }
    if let Some(t) = replicate::get_str_meta(&mimir.conn, "server_token")? {
        eprintln!("using stored hub token (set MIMIR_SYNC_TOKEN to override)");
        return Ok(t);
    }
    let t = replicate::new_token();
    replicate::set_str_meta(&mimir.conn, "server_token", &t)?;
    eprintln!("generated a hub token (persisted; restarts reuse it):\n  MIMIR_SYNC_TOKEN={t}");
    Ok(t)
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}
