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

/// `injection_log`/`session_state` (v8) are ephemeral per-session working
/// state, not knowledge — see `store::prune_sessions`'s doc comment.
/// Fixed (not user-configurable, unlike `recall_event`'s retention): nothing
/// meaningful survives a session that has written nothing for 7 days. `pub` so
/// `mimir-proxy`'s own inline timer (`mimir-proxy/src/lib.rs`, which can't
/// share this module's thread since it locks a request-shared connection
/// instead of opening its own) uses the same cutoff rather than a
/// hand-copied one.
pub const SESSION_STATE_RETENTION_DAYS: i64 = 7;

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
    ensure_indexes_when_cheap(&conn);
    Ok(conn)
}

/// In-memory database for tests.
pub fn open_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    apply_pragmas(&conn)?;
    migrations::migrate(&conn)?;
    ensure_indexes_when_cheap(&conn);
    Ok(conn)
}

/// Indexes that change how fast the store is and nothing about what it
/// holds, as `(name, statement)`.
///
/// Both exist for one reason. `node` refers to itself through four foreign
/// keys, and with `foreign_keys = ON` SQLite answers every row delete by
/// looking for rows that still point at the deleted one, once per key.
/// `parent_id` and `collection_id` had an index for that lookup;
/// `project_id` and `superseded_by` did not, so each deleted row scanned the
/// whole table twice. That is unnoticeable on a small store and about 45 ms
/// per deleted row at half a million rows — and the code graph deletes a row
/// for every symbol that leaves a file.
///
/// They are deliberately not a schema version. An index is invisible to a
/// binary that does not know it and is kept up to date by whichever binary
/// writes, so adding one must not lock an older `mimir` out of the store the
/// way a `SCHEMA_VERSION` bump does.
const SUPPORT_INDEXES: &[(&str, &str)] = &[
    (
        "node_project",
        "CREATE INDEX IF NOT EXISTS node_project ON node(project_id)
         WHERE project_id IS NOT NULL",
    ),
    (
        "node_superseded_by",
        "CREATE INDEX IF NOT EXISTS node_superseded_by ON node(superseded_by)
         WHERE superseded_by IS NOT NULL",
    ),
];

/// Above this many `node` rows, building an index is left to an explicit
/// command (`mimir init`): it reads the whole table under the write lock,
/// and whichever process happens to open the store first may be a hook with
/// two seconds to live.
const INLINE_INDEX_MAX_ROWS: i64 = 200_000;

/// The [`SUPPORT_INDEXES`] this store does not have yet, by name.
pub fn missing_indexes(conn: &Connection) -> Result<Vec<&'static str>> {
    let mut missing = Vec::new();
    for (name, _) in SUPPORT_INDEXES {
        let present: bool = conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?1)",
            [name],
            |r| r.get(0),
        )?;
        if !present {
            missing.push(*name);
        }
    }
    Ok(missing)
}

/// Create every missing support index, however large the store. Returns how
/// many were created; calling it again creates none.
pub fn ensure_indexes(conn: &Connection) -> Result<usize> {
    let missing = missing_indexes(conn)?;
    for (name, statement) in SUPPORT_INDEXES {
        if missing.contains(name) {
            conn.execute_batch(statement)?;
        }
    }
    Ok(missing.len())
}

/// [`ensure_indexes`] on open, for a store small enough that it is quick.
/// Best-effort: a store that could not get its indexes now still works, only
/// slower, and `mimir doctor` says so.
///
/// It never waits for the write lock. Creating an index is a write, and an
/// open that queued behind another process's writer for the full
/// [`BUSY_TIMEOUT`] would turn "open the store to read one row" into a ten
/// second stall in whichever hook or command came next. If the store is
/// busy, the next open tries again.
fn ensure_indexes_when_cheap(conn: &Connection) {
    let small = || -> Result<bool> {
        if missing_indexes(conn)?.is_empty() {
            return Ok(false);
        }
        let rows: i64 =
            conn.query_row("SELECT COALESCE(max(id), 0) FROM node", [], |r| r.get(0))?;
        Ok(rows <= INLINE_INDEX_MAX_ROWS)
    };
    if !small().unwrap_or(false) {
        return;
    }
    if conn.busy_timeout(Duration::ZERO).is_ok() {
        if let Err(err) = ensure_indexes(conn) {
            tracing::debug!(%err, "support indexes not created on open");
        }
    }
    if let Err(err) = conn.busy_timeout(BUSY_TIMEOUT) {
        tracing::warn!(%err, "could not restore the busy timeout after the index check");
    }
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
///
/// Same tick also prunes `injection_log`/`session_state` on a fixed 7-day
/// cutoff (`store::prune_sessions`, `SESSION_STATE_RETENTION_DAYS`):
/// injection marks older than that, and the state of sessions that have
/// been silent that long — unconditionally, unlike the recall-event prune,
/// since there's no user config for it (both are ephemeral per-session
/// working state, not knowledge). `mimir
/// proxy`'s standalone process runs its own separate inline timer
/// (`mimir-proxy/src/lib.rs`) — it can't share this thread since it locks a
/// request-shared connection rather than opening its own — and that timer
/// calls the same two passes, [`prune_events_once`] and
/// [`prune_session_state_once`], so "a prune pass" has one definition.
pub fn spawn_checkpoint_timer(db_path: PathBuf, event_retention_days: Option<u32>) {
    std::thread::spawn(move || {
        let Ok(conn) = open(&db_path) else { return };
        if let Some(days) = event_retention_days {
            prune_events_once(&conn, days);
        }
        prune_session_state_once(&conn);
        loop {
            std::thread::sleep(CHECKPOINT_INTERVAL);
            if let Err(err) = checkpoint_truncate(&conn) {
                tracing::warn!(%err, "wal checkpoint failed");
            }
            if let Some(days) = event_retention_days {
                prune_events_once(&conn, days);
            }
            prune_session_state_once(&conn);
        }
    });
}

/// One prune pass over the usage ledgers (`recall_event`, `shadow_rank`,
/// `recall_miss`), `days` back from now. Best-effort and non-fatal, like the
/// checkpoint it rides alongside. Public so every long-lived process runs
/// this pass and not a copy of part of it.
pub fn prune_events_once(conn: &Connection, days: u32) {
    let cutoff = crate::model::now_unix() - i64::from(days) * 86_400;
    match crate::store::prune_old_events(conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!(rows = n, days, "pruned old recall_event rows"),
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "recall_event prune failed"),
    }
    match crate::shadow::prune(conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!(rows = n, days, "pruned old shadow_rank rows"),
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "shadow_rank prune failed"),
    }
    match crate::misses::prune(conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!(rows = n, days, "pruned old recall_miss rows"),
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "recall_miss prune failed"),
    }
}

/// One `injection_log`/`session_state` prune pass, `SESSION_STATE_RETENTION_DAYS`
/// back from now — measured per session for its state (see
/// `store::prune_sessions`). Best-effort and non-fatal, like the checkpoint
/// it rides alongside. Public for the same reason as [`prune_events_once`].
pub fn prune_session_state_once(conn: &Connection) {
    let cutoff = crate::model::now_unix() - SESSION_STATE_RETENTION_DAYS * 86_400;
    match crate::store::prune_sessions(conn, cutoff) {
        Ok(n) if n > 0 => tracing::info!(rows = n, "pruned old session_state/injection_log rows"),
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "session_state prune failed"),
    }
}

/// Name of the savepoint [`atomic`] nests with. SQLite resolves a savepoint
/// name to its most recent holder, so nested calls can all share it.
const ATOMIC_SAVEPOINT: &str = "mimir_atomic";

/// Rolls the savepoint back unless [`atomic`] released it, so an early
/// return or a panic inside the work cannot leave a half-applied step on the
/// caller's transaction.
struct SavepointGuard<'c> {
    conn: &'c Connection,
    released: bool,
}

impl Drop for SavepointGuard<'_> {
    fn drop(&mut self) {
        if !self.released {
            // `ROLLBACK TO` undoes the work but leaves the savepoint on the
            // stack; `RELEASE` takes it off.
            let _ = self.conn.execute_batch(&format!(
                "ROLLBACK TO {ATOMIC_SAVEPOINT}; RELEASE {ATOMIC_SAVEPOINT}"
            ));
        }
    }
}

/// Run `work` so that everything it writes lands together or not at all.
///
/// The store's mutations are several statements each — the column, the edge
/// and the ledger row of a supersession; the tombstone, the purge and the
/// ledger row of a forget. Run as separate autocommits, a failure between
/// two of them leaves a state no reader has a name for: a memory retired
/// with no date, a deletion that kept the deleted wording.
///
/// On a connection in autocommit mode this is one transaction, IMMEDIATE for
/// the reason `store::record_shown` gives: a DEFERRED transaction that reads
/// first and writes second upgrades its lock mid-flight, and `busy_timeout`
/// cannot wait on an upgrade. Inside a transaction the caller already holds
/// (consolidation's pass, a sync batch) it nests as a savepoint instead, so
/// a failed step undoes its own writes and leaves the caller's standing.
pub fn atomic<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if conn.is_autocommit() {
        let tx =
            rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
        let out = work()?;
        tx.commit()?;
        return Ok(out);
    }
    conn.execute_batch(&format!("SAVEPOINT {ATOMIC_SAVEPOINT}"))?;
    let mut guard = SavepointGuard {
        conn,
        released: false,
    };
    let out = work()?;
    conn.execute_batch(&format!("RELEASE {ATOMIC_SAVEPOINT}"))?;
    guard.released = true;
    Ok(out)
}

/// SQLite's data_version increments when ANOTHER connection commits.
/// Used to invalidate in-process caches (vector matrix, graph).
pub fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{now_unix, Kind, NewNode};
    use crate::store;

    /// Deleting a row makes SQLite look for rows that still point at it,
    /// once per foreign key. Each of those lookups has to be an index
    /// search: a scan costs time in proportion to the whole store, for every
    /// deleted symbol, chunk or memory.
    #[test]
    fn every_foreign_key_into_node_is_found_by_an_index() {
        let conn = open_in_memory().unwrap();
        assert!(missing_indexes(&conn).unwrap().is_empty());
        for child_key in ["project_id", "collection_id", "parent_id", "superseded_by"] {
            let plan: Vec<String> = conn
                .prepare(&format!(
                    "EXPLAIN QUERY PLAN SELECT 1 FROM node WHERE {child_key} = 7"
                ))
                .unwrap()
                .query_map([], |r| r.get::<_, String>(3))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert!(
                plan.iter().all(|step| step.contains("INDEX")),
                "{child_key}: {plan:?}"
            );
        }
    }

    /// A store written before these indexes existed gets them when it is
    /// opened, as long as it is small; `ensure_indexes` adds them to any
    /// store and is idempotent.
    #[test]
    fn a_store_without_the_support_indexes_gets_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mimir.db");
        {
            let conn = open(&path).unwrap();
            conn.execute_batch("DROP INDEX node_project; DROP INDEX node_superseded_by;")
                .unwrap();
            assert_eq!(
                missing_indexes(&conn).unwrap(),
                ["node_project", "node_superseded_by"]
            );
            assert_eq!(ensure_indexes(&conn).unwrap(), 2);
            assert_eq!(ensure_indexes(&conn).unwrap(), 0);
            conn.execute_batch("DROP INDEX node_project;").unwrap();
        }
        let reopened = open(&path).unwrap();
        assert!(missing_indexes(&reopened).unwrap().is_empty());
    }

    /// The index on open is opportunistic. With another process holding the
    /// write lock, an open returns at once without it, and the connection it
    /// hands back still waits for writers the usual length of time.
    #[test]
    fn an_open_does_not_queue_behind_a_writer_to_build_an_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mimir.db");
        let writer = open(&path).unwrap();
        writer.execute_batch("DROP INDEX node_project;").unwrap();
        writer.execute_batch("BEGIN IMMEDIATE;").unwrap();

        let started = std::time::Instant::now();
        let other = open(&path).unwrap();
        let waited = started.elapsed();

        assert!(
            waited < BUSY_TIMEOUT / 2,
            "the open waited {waited:?} for the write lock"
        );
        assert_eq!(missing_indexes(&other).unwrap(), ["node_project"]);
        let timeout_ms: i64 = other
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert_eq!(timeout_ms, BUSY_TIMEOUT.as_millis() as i64);

        writer.execute_batch("COMMIT;").unwrap();
        drop(other);
        assert!(missing_indexes(&open(&path).unwrap()).unwrap().is_empty());
    }

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

    fn seed_old_session_state(conn: &Connection, age_days: i64) {
        seed_session_state(conn, "s", age_days);
    }

    fn seed_session_state(conn: &Connection, session: &str, age_days: i64) {
        let key = format!("k{age_days}");
        conn.execute(
            "INSERT INTO session_state (session_id, key, value, updated_at) VALUES (?3, ?2, 'v', ?1)",
            rusqlite::params![now_unix() - age_days * 86_400, key, session],
        )
        .unwrap();
    }

    fn session_state_count(conn: &Connection) -> i64 {
        conn.query_row("SELECT count(*) FROM session_state", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn prune_events_once_removes_rows_past_the_window() {
        let conn = open_in_memory().unwrap();
        seed_old_event(&conn, 200);
        seed_old_event(&conn, 10);
        prune_events_once(&conn, 180);
        assert_eq!(
            event_count(&conn),
            1,
            "only the 200-day-old row is past a 180-day window"
        );
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
        assert_eq!(
            event_count(&conn),
            1,
            "None must disable the prune entirely"
        );
    }

    /// `session_state`/`injection_log` pruning is unconditional (no config
    /// gate) and runs at startup regardless of `event_retention_days`.
    #[test]
    fn spawn_checkpoint_timer_prunes_session_state_at_startup_even_with_events_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mimir.db");
        {
            let conn = open(&db_path).unwrap();
            seed_old_session_state(&conn, 30);
        }
        spawn_checkpoint_timer(db_path.clone(), None);

        let conn = open(&db_path).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if session_state_count(&conn) == 0 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "startup session_state prune did not run within 2s"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// The window is measured per session, from its newest row: a session
    /// still writing keeps even its ten-day-old rows, and one that has been
    /// silent for ten days loses everything.
    #[test]
    fn prune_session_state_once_removes_sessions_silent_past_the_seven_day_window() {
        let conn = open_in_memory().unwrap();
        seed_session_state(&conn, "running", 10);
        seed_session_state(&conn, "running", 1);
        seed_session_state(&conn, "ended", 10);
        seed_session_state(&conn, "ended", 8);
        prune_session_state_once(&conn);
        let left: Vec<String> = conn
            .prepare("SELECT DISTINCT session_id FROM session_state")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(left, vec!["running".to_string()]);
        assert_eq!(
            session_state_count(&conn),
            2,
            "a running session keeps its old rows too"
        );
    }

    fn rows(conn: &Connection) -> Vec<String> {
        conn.prepare("SELECT value FROM meta WHERE key LIKE 'atomic-%' ORDER BY key")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    fn put(conn: &Connection, key: &str) -> Result<()> {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?1)",
            [format!("atomic-{key}")],
        )?;
        Ok(())
    }

    #[test]
    fn atomic_commits_everything_or_nothing() {
        let conn = open_in_memory().unwrap();
        atomic(&conn, || {
            put(&conn, "a")?;
            put(&conn, "b")
        })
        .unwrap();
        assert_eq!(rows(&conn), vec!["atomic-a", "atomic-b"]);

        let failed: Result<()> = atomic(&conn, || {
            put(&conn, "c")?;
            Err(Error::Invalid("second step failed".into()))
        });
        assert!(failed.is_err());
        assert_eq!(rows(&conn), vec!["atomic-a", "atomic-b"], "c rolled back");
        assert!(conn.is_autocommit(), "no transaction may be left open");
    }

    /// Inside a caller's transaction a failed step undoes only itself — and
    /// that holds when steps are nested inside steps, which is what a review
    /// decision calling `store::supersede` is.
    #[test]
    fn atomic_nests_inside_a_callers_transaction() {
        let conn = open_in_memory().unwrap();
        let tx =
            rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)
                .unwrap();
        put(&tx, "outer").unwrap();
        let failed: Result<()> = atomic(&tx, || {
            put(&tx, "step")?;
            atomic(&tx, || put(&tx, "inner"))?;
            Err(Error::Invalid(
                "the step failed after its inner step".into(),
            ))
        });
        assert!(failed.is_err());
        atomic(&tx, || {
            put(&tx, "kept")?;
            let inner: Result<()> = atomic(&tx, || {
                put(&tx, "dropped")?;
                Err(Error::Invalid("only the inner step failed".into()))
            });
            assert!(inner.is_err());
            Ok(())
        })
        .unwrap();
        tx.commit().unwrap();
        assert_eq!(rows(&conn), vec!["atomic-kept", "atomic-outer"]);
    }
}
