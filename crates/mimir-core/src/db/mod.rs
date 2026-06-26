pub mod migrations;

use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::Connection;

use crate::error::{Error, Result};

/// Busy timeout for every connection. Set via the rusqlite API the instant a
/// connection opens — BEFORE any pragma — because `PRAGMA journal_mode = WAL`
/// itself takes a momentary exclusive lock (to rewrite the header and create
/// the -wal/-shm files). At the default 0ms timeout that lock fails instantly
/// with SQLITE_BUSY under concurrency, killing even read-only commands at open.
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// How often long-lived daemons reclaim the WAL (see `spawn_checkpoint_timer`).
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(120);

/// Open (creating if needed) the Mimir database with standard pragmas
/// and the schema migrated to the current version.
pub fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    apply_pragmas(&conn)?;
    migrations::migrate(&conn)?;
    Ok(conn)
}

/// In-memory database for tests.
pub fn open_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    apply_pragmas(&conn)?;
    migrations::migrate(&conn)?;
    Ok(conn)
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    // busy_timeout is set earlier via the API (see BUSY_TIMEOUT) so it covers
    // the WAL switch below; it is deliberately NOT repeated here.
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA wal_autocheckpoint = 1000;",
    )?;
    Ok(())
}

/// Force a full WAL checkpoint and truncate the -wal file back to zero.
///
/// `wal_autocheckpoint` only does PASSIVE checkpoints, which a long-lived
/// reader's mark can repeatedly block, letting the -wal grow without bound
/// (we've seen 100 MB+). Long-running daemons call this on an idle timer to
/// reclaim it. TRUNCATE is best-effort: if a reader is mid-transaction it
/// checkpoints what it can and leaves the rest for next time.
pub fn checkpoint_truncate(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok(())
}

/// Spawn a daemon thread that reclaims the WAL every `CHECKPOINT_INTERVAL` on
/// its own dedicated connection — idle and autocommit between ticks, so it
/// holds no read mark that could block its own checkpoint. For long-lived sync
/// daemons (`mimir mcp`, `mimir serve`) whose persistent read marks would
/// otherwise starve PASSIVE autocheckpoints. Best-effort and non-fatal; the
/// async proxy keeps its own variant since it must reuse a shared connection.
pub fn spawn_checkpoint_timer(db_path: PathBuf) {
    std::thread::spawn(move || {
        let Ok(conn) = open(&db_path) else { return };
        loop {
            std::thread::sleep(CHECKPOINT_INTERVAL);
            if let Err(err) = checkpoint_truncate(&conn) {
                tracing::warn!(%err, "wal checkpoint failed");
            }
        }
    });
}

/// SQLite's data_version increments when ANOTHER connection commits.
/// Used to invalidate in-process caches (vector matrix, graph).
pub fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}
