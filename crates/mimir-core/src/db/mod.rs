pub mod migrations;

use std::path::Path;

use rusqlite::Connection;

use crate::error::{Error, Result};

/// Open (creating if needed) the Mimir database with standard pragmas
/// and the schema migrated to the current version.
pub fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let conn = Connection::open(path)?;
    apply_pragmas(&conn)?;
    migrations::migrate(&conn)?;
    Ok(conn)
}

/// In-memory database for tests.
pub fn open_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    apply_pragmas(&conn)?;
    migrations::migrate(&conn)?;
    Ok(conn)
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;
         PRAGMA wal_autocheckpoint = 1000;",
    )?;
    Ok(())
}

/// SQLite's data_version increments when ANOTHER connection commits.
/// Used to invalidate in-process caches (vector matrix, graph).
pub fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}
