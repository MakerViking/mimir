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
///
/// Also prunes `recall_event` on the same idle cadence, once at startup and
/// then every tick (see `store::prune_old_events`,
/// `config::LearnConfig::event_retention_days`) — the ledger is unbounded
/// (every recall/inject logs impressions), and this is the one place all of
/// `mimir mcp`/`mimir serve`'s idle-timer daemons already run background
/// maintenance, so pruning rides along instead of its own thread.
/// `event_retention_days: None` (explicit `0`, or the caller opting out)
/// disables the prune entirely; the checkpoint still runs either way.
pub fn spawn_checkpoint_timer(db_path: PathBuf, event_retention_days: Option<u32>) {
    std::thread::spawn(move || {
        let Ok(conn) = open(&db_path) else { return };
        if let Some(days) = event_retention_days {
            prune_events_once(&conn, days);
        }
        loop {
            std::thread::sleep(CHECKPOINT_INTERVAL);
            if let Err(err) = checkpoint_truncate(&conn) {
                tracing::warn!(%err, "wal checkpoint failed");
            }
            if let Some(days) = event_retention_days {
                prune_events_once(&conn, days);
            }
        }
    });
}

/// One `recall_event` prune pass, `days` back from now. Best-effort and
/// non-fatal, like the checkpoint it rides alongside.
fn prune_events_once(conn: &Connection, days: u32) {
    let cutoff = crate::model::now_unix() - i64::from(days) * 86_400;
    match crate::store::prune_old_events(conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!(rows = n, days, "pruned old recall_event rows"),
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "recall_event prune failed"),
    }
}

/// SQLite's data_version increments when ANOTHER connection commits.
/// Used to invalidate in-process caches (vector matrix, graph).
pub fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode, now_unix};
    use crate::store;

    fn seed_old_event(conn: &Connection, age_days: i64) {
        let node = store::insert_node(conn, NewNode::new(Kind::Memory)).unwrap();
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'shown', ?2)",
            rusqlite::params![node.id, now_unix() - age_days * 86_400],
        )
        .unwrap();
    }

    fn event_count(conn: &Connection) -> i64 {
        conn.query_row("SELECT count(*) FROM recall_event", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn prune_events_once_removes_rows_past_the_window() {
        let conn = open_in_memory().unwrap();
        seed_old_event(&conn, 200);
        seed_old_event(&conn, 10);
        prune_events_once(&conn, 180);
        assert_eq!(event_count(&conn), 1, "only the 200-day-old row is past a 180-day window");
    }

    #[test]
    fn prune_events_once_is_a_noop_on_a_db_error_free_empty_table() {
        let conn = open_in_memory().unwrap();
        // Must not panic/error when there is nothing to prune.
        prune_events_once(&conn, 180);
        assert_eq!(event_count(&conn), 0);
    }

    /// Verifies the cadence hook actually fires: `spawn_checkpoint_timer`
    /// runs the prune once immediately at startup, not only after the first
    /// `CHECKPOINT_INTERVAL` sleep (which would make it wait 120s here).
    #[test]
    fn spawn_checkpoint_timer_prunes_once_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mimir.db");
        {
            let conn = open(&db_path).unwrap();
            seed_old_event(&conn, 200);
        }
        spawn_checkpoint_timer(db_path.clone(), Some(180));

        let conn = open(&db_path).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if event_count(&conn) == 0 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "startup prune did not run within 2s"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn spawn_checkpoint_timer_with_none_never_prunes() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mimir.db");
        {
            let conn = open(&db_path).unwrap();
            seed_old_event(&conn, 200);
        }
        spawn_checkpoint_timer(db_path.clone(), None);
        // No deadline to poll toward (absence isn't observable at a single
        // instant) — a short, generous sleep stands in for "never happens".
        std::thread::sleep(Duration::from_millis(300));
        let conn = open(&db_path).unwrap();
        assert_eq!(event_count(&conn), 1, "None must disable the prune entirely");
    }
}
