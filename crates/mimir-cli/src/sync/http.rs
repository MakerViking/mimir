//! Blocking HTTP client for server-mode sync (via `ureq`, no async runtime).

use anyhow::{anyhow, bail, Result};
use mimir_core::replicate::{ApplyStats, SyncBatch};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct PushResult {
    pub applied: ApplyStats,
    // The hub also sends `watermark` (its post-apply global high watermark);
    // there is deliberately no field for it — serde drops it — because a
    // client must never advance its own last_push from the hub's clock
    // domain (see srv_push).
}

fn auth(token: &str) -> String {
    format!("Bearer {token}")
}

/// `GET /sync/pull?since=<since>` → the hub's changes since `since`.
pub fn pull(endpoint: &str, token: &str, since: i64) -> Result<SyncBatch> {
    let url = format!("{}/sync/pull?since={since}", endpoint.trim_end_matches('/'));
    match ureq::get(&url).set("Authorization", &auth(token)).call() {
        Ok(r) => Ok(r.into_json::<SyncBatch>()?),
        Err(ureq::Error::Status(401, _)) => {
            bail!("hub rejected the token (401) — check MIMIR_SYNC_TOKEN")
        }
        Err(ureq::Error::Status(code, _)) => bail!("hub pull failed: HTTP {code}"),
        Err(e) => Err(anyhow!("hub unreachable at {endpoint}: {e}")),
    }
}

/// `POST /sync/push` with a batch body → the hub's apply stats.
pub fn push(endpoint: &str, token: &str, batch: &SyncBatch) -> Result<PushResult> {
    let url = format!("{}/sync/push", endpoint.trim_end_matches('/'));
    match ureq::post(&url)
        .set("Authorization", &auth(token))
        .send_json(batch)
    {
        Ok(r) => Ok(r.into_json::<PushResult>()?),
        Err(ureq::Error::Status(401, _)) => {
            bail!("hub rejected the token (401) — check MIMIR_SYNC_TOKEN")
        }
        Err(ureq::Error::Status(code, _)) => bail!("hub push failed: HTTP {code}"),
        Err(e) => Err(anyhow!("hub unreachable at {endpoint}: {e}")),
    }
}
