//! `mimir sync` / `mimir serve` — the OPTIONAL centralized-sync layer.
//!
//! Two transports share one replication core (mimir_core::replicate):
//! `file` (a folder replicated by Syncthing/Dropbox/…) and `server` (a
//! `mimir serve` hub). Everything here is gated on `[sync] mode != "off"`.
//!
//! Transport functions return summaries instead of printing, so both the CLI
//! (which prints) and the MCP background loop (which logs via tracing, since
//! stdout is the MCP protocol channel) can reuse them.

mod file;
mod http;
mod server;

pub(crate) use server::ct_eq;

use anyhow::{bail, Context, Result};
use mimir_core::replicate::{self, ApplyStats};
use mimir_core::Mimir;

/// Watermarks are persisted minus this grace window so a record written during
/// a sync round (clock skew / same-second) re-delivers next time; idempotent
/// last-write-wins apply makes the re-delivery free.
const GRACE_SECS: i64 = 60;

/// Read an env var, treating empty-but-set as unset — an unset shell variable
/// interpolated into a flag or env assignment must not count as configuration.
pub(crate) fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|t| !t.is_empty())
}

/// Cursor advancement for a sync watermark: the batch's high watermark,
/// clamped to the local wall clock (then minus [`GRACE_SECS`], floored at 0).
/// A single future-dated row — our own past clock skew, or a skewed peer's
/// row relayed by the hub — inflates the batch max arbitrarily; unclamped,
/// the cursor would jump past every normally-timestamped later change and
/// this client would silently stop sending (push) or receiving (pull) them.
/// Clamped, the worst case is the skewed row re-transferring each round,
/// which the idempotent apply makes free.
fn cursor_advance(batch_watermark: i64) -> i64 {
    (batch_watermark.min(mimir_core::model::now_unix()) - GRACE_SECS).max(0)
}

fn require_enabled(m: &Mimir) -> Result<()> {
    if !m.config.sync.enabled() {
        bail!(
            "sync is off. Enable it in config.toml under [sync] (mode = \"file\" or \
             \"server\") — see docs/sync.md."
        );
    }
    Ok(())
}

/// Mode-appropriate full sync, returning a one-line summary (no printing).
/// Used by the CLI `sync` command and the MCP background loop.
pub(crate) fn perform(m: &mut Mimir) -> Result<String> {
    match m.config.sync.mode.as_str() {
        "file" => {
            let dir = file_dir(m)?;
            let s = file::sync(m, &dir)?;
            Ok(format!(
                "file: wrote {} memories, merged {} peer file(s) → {} applied, {} deletes, {} edges",
                s.wrote, s.peers_merged, s.applied.nodes_upserted, s.applied.tombstones, s.applied.edges_upserted
            ))
        }
        "server" => {
            let (sent, pushed) = srv_push(m)?;
            let pulled = srv_pull(m)?;
            Ok(format!(
                "server: pushed {sent} (hub applied {}), pulled {} applied, {} deletes, {} edges",
                pushed.applied.nodes_upserted,
                pulled.nodes_upserted,
                pulled.tombstones,
                pulled.edges_upserted
            ))
        }
        "off" => bail!("sync is off"),
        other => bail!("unknown sync mode '{other}' (use \"file\" or \"server\")"),
    }
}

/// Full sync (the default `mimir sync`).
pub fn sync() -> Result<()> {
    let mut m = Mimir::open()?;
    require_enabled(&m)?;
    println!("{}", perform(&mut m)?);
    Ok(())
}

pub fn push() -> Result<()> {
    let mut m = Mimir::open()?;
    require_enabled(&m)?;
    match m.config.sync.mode.as_str() {
        "file" => {
            let dir = file_dir(&m)?;
            let n = file::write_snapshot(&m, &dir)?;
            println!("wrote snapshot of {n} memories → {}", dir.display());
        }
        "server" => {
            let (sent, res) = srv_push(&mut m)?;
            println!(
                "pushed {sent} change(s) → hub applied {}",
                res.applied.nodes_upserted
            );
        }
        other => bail!("unknown sync mode '{other}'"),
    }
    Ok(())
}

pub fn pull() -> Result<()> {
    let mut m = Mimir::open()?;
    require_enabled(&m)?;
    match m.config.sync.mode.as_str() {
        "file" => {
            let dir = file_dir(&m)?;
            let (peers, s) = file::merge_peers(&mut m, &dir)?;
            let _ = m.embed_pending();
            println!(
                "merged {peers} peer file(s) → {} applied, {} deletes, {} edges",
                s.nodes_upserted, s.tombstones, s.edges_upserted
            );
        }
        "server" => {
            let s = srv_pull(&mut m)?;
            println!(
                "pulled → {} applied, {} deletes, {} edges",
                s.nodes_upserted, s.tombstones, s.edges_upserted
            );
        }
        other => bail!("unknown sync mode '{other}'"),
    }
    Ok(())
}

pub fn status() -> Result<()> {
    let m = Mimir::open()?;
    let sc = &m.config.sync;
    println!("mode      {}", sc.mode);
    if !sc.enabled() {
        println!("(sync is off — see docs/sync.md to enable)");
        return Ok(());
    }
    match sc.mode.as_str() {
        "file" => println!("dir       {}", sc.dir),
        "server" => {
            println!("endpoint  {}", sc.endpoint);
            println!(
                "token     {}",
                if env_nonempty("MIMIR_SYNC_TOKEN").is_some() {
                    "set (MIMIR_SYNC_TOKEN)"
                } else {
                    "MISSING — set MIMIR_SYNC_TOKEN"
                }
            );
        }
        _ => {}
    }
    let pending =
        replicate::changes_since(&m.conn, replicate::get_watermark(&m.conn, "last_push")?)?;
    println!("pending   {} memory change(s) to push", pending.nodes.len());
    println!("auto      {} (every {} min)", sc.auto, sc.interval_mins);
    Ok(())
}

/// Run the `mimir serve` hub.
pub fn serve(bind: String) -> Result<()> {
    server::serve(&bind)
}

/// Print the active sync token so it can be copied to a client: the
/// `MIMIR_SYNC_TOKEN` env var if set, else the hub's persisted token (server
/// mode). The token goes to stdout (scriptable, e.g. `mimir sync token`); the
/// provenance note goes to stderr.
pub fn token() -> Result<()> {
    if let Some(t) = env_nonempty("MIMIR_SYNC_TOKEN") {
        println!("{t}");
        eprintln!("(from MIMIR_SYNC_TOKEN in the environment)");
        return Ok(());
    }
    let m = Mimir::open()?;
    if let Some(t) = replicate::get_str_meta(&m.conn, "server_token")? {
        println!("{t}");
        eprintln!("(this hub's persisted token — set MIMIR_SYNC_TOKEN to override)");
        return Ok(());
    }
    bail!(
        "no sync token found — set MIMIR_SYNC_TOKEN, or start the hub once \
         (`mimir serve`) to generate and persist one"
    );
}

// ---- helpers ----

fn file_dir(m: &Mimir) -> Result<std::path::PathBuf> {
    if m.config.sync.dir.is_empty() {
        bail!("file sync needs `[sync] dir = \"...\"` in config.toml (a folder your file-sync tool replicates)");
    }
    Ok(std::path::PathBuf::from(expand_tilde(&m.config.sync.dir)))
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(base) = directories::BaseDirs::new() {
            return base.home_dir().join(rest).to_string_lossy().into_owned();
        }
    }
    p.to_string()
}

fn server_token() -> Result<String> {
    env_nonempty("MIMIR_SYNC_TOKEN")
        .context("server-mode sync needs MIMIR_SYNC_TOKEN in the environment")
}

fn server_endpoint(m: &Mimir) -> Result<String> {
    if m.config.sync.endpoint.is_empty() {
        bail!("server sync needs `[sync] endpoint = \"http://...\"` in config.toml");
    }
    Ok(m.config.sync.endpoint.clone())
}

/// Push local changes; returns (count sent, hub result). Advances last_push.
fn srv_push(m: &mut Mimir) -> Result<(usize, http::PushResult)> {
    let (endpoint, token) = (server_endpoint(m)?, server_token()?);
    let since = replicate::get_watermark(&m.conn, "last_push")?;
    let batch = replicate::changes_since(&m.conn, since)?;
    let sent = batch.nodes.len();
    let res = http::push(&endpoint, &token, &batch)?;
    // Advance last_push from OUR batch's high watermark (local clock domain,
    // clamped to the local clock — see `cursor_advance`), never the hub's
    // post-apply global watermark: that one is a cross-peer clock domain a
    // skewed peer can inflate (see e2e `server_sync_survives_peer_clock_skew`).
    // The hub still sends its watermark on the wire; the client-side
    // `PushResult` deliberately has no field for it.
    replicate::set_watermark(&m.conn, "last_push", cursor_advance(batch.watermark))?;
    Ok((sent, res))
}

/// Pull + apply remote changes; returns apply stats. Advances last_pull and
/// embeds new arrivals locally.
fn srv_pull(m: &mut Mimir) -> Result<ApplyStats> {
    let (endpoint, token) = (server_endpoint(m)?, server_token()?);
    let since = replicate::get_watermark(&m.conn, "last_pull")?;
    let batch = http::pull(&endpoint, &token, since)?;
    let applied = replicate::apply_changes(&m.conn, &batch)?;
    replicate::set_watermark(&m.conn, "last_pull", cursor_advance(batch.watermark))?;
    let _ = m.embed_pending();
    Ok(applied)
}
