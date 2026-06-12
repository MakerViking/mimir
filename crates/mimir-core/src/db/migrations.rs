use rusqlite::Connection;

use crate::error::{Error, Result};

/// Ordered migrations; index i migrates user_version i -> i+1.
const MIGRATIONS: &[&str] = &[
    // v1: initial schema
    r#"
CREATE TABLE node (
  id            INTEGER PRIMARY KEY,
  uid           TEXT NOT NULL UNIQUE,
  kind          TEXT NOT NULL,
  subkind       TEXT,
  project_id    INTEGER REFERENCES node(id),
  collection_id INTEGER REFERENCES node(id),
  parent_id     INTEGER REFERENCES node(id),
  title         TEXT,
  body          TEXT,
  path          TEXT,
  span_start    INTEGER,
  span_end      INTEGER,
  lang          TEXT,
  tags_text     TEXT NOT NULL DEFAULT '',
  content_hash  BLOB,
  meta          TEXT NOT NULL DEFAULT '{}',
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  access_count  INTEGER NOT NULL DEFAULT 0,
  last_accessed INTEGER,
  strength      REAL NOT NULL DEFAULT 1.0,
  pinned        INTEGER NOT NULL DEFAULT 0,
  superseded_by INTEGER REFERENCES node(id),
  deleted_at    INTEGER
);
CREATE INDEX node_kind_scope ON node(kind, project_id) WHERE deleted_at IS NULL;
CREATE INDEX node_coll_path  ON node(collection_id, path);
CREATE INDEX node_parent     ON node(parent_id);
CREATE INDEX node_hash       ON node(content_hash);
CREATE INDEX node_uid_tail   ON node(substr(uid, -6));

CREATE TABLE edge (
  id         INTEGER PRIMARY KEY,
  src        INTEGER NOT NULL REFERENCES node(id) ON DELETE CASCADE,
  dst        INTEGER NOT NULL REFERENCES node(id) ON DELETE CASCADE,
  rel        TEXT NOT NULL,
  weight     REAL NOT NULL DEFAULT 1.0,
  meta       TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  UNIQUE(src, dst, rel)
);
CREATE INDEX edge_src ON edge(src, rel);
CREATE INDEX edge_dst ON edge(dst, rel);

CREATE VIRTUAL TABLE node_fts USING fts5(
  title, body, path, tags_text,
  content='node', content_rowid='id',
  tokenize="unicode61 remove_diacritics 2 tokenchars '_-.'"
);
CREATE TRIGGER node_ai AFTER INSERT ON node BEGIN
  INSERT INTO node_fts(rowid, title, body, path, tags_text)
  VALUES (new.id, new.title, new.body, new.path, new.tags_text);
END;
CREATE TRIGGER node_ad AFTER DELETE ON node BEGIN
  INSERT INTO node_fts(node_fts, rowid, title, body, path, tags_text)
  VALUES ('delete', old.id, old.title, old.body, old.path, old.tags_text);
END;
CREATE TRIGGER node_au AFTER UPDATE OF title, body, path, tags_text ON node BEGIN
  INSERT INTO node_fts(node_fts, rowid, title, body, path, tags_text)
  VALUES ('delete', old.id, old.title, old.body, old.path, old.tags_text);
  INSERT INTO node_fts(rowid, title, body, path, tags_text)
  VALUES (new.id, new.title, new.body, new.path, new.tags_text);
END;

CREATE TABLE embedding (
  node_id      INTEGER PRIMARY KEY REFERENCES node(id) ON DELETE CASCADE,
  model        TEXT NOT NULL,
  dim          INTEGER NOT NULL,
  vec          BLOB NOT NULL,
  content_hash BLOB NOT NULL
);

CREATE TABLE recall_event (
  id         INTEGER PRIMARY KEY,
  node_id    INTEGER NOT NULL REFERENCES node(id) ON DELETE CASCADE,
  event      TEXT NOT NULL,
  query_hash BLOB,
  rank       INTEGER,
  score      REAL,
  at         INTEGER NOT NULL
);
CREATE INDEX recall_node ON recall_event(node_id, at);

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"#,
    // v2: porter stemming on the FTS index ("authentication" matches
    // "authenticate"). Tokenizer changes require dropping + rebuilding the
    // external-content table; 'rebuild' repopulates from node.
    r#"
DROP TRIGGER node_ai;
DROP TRIGGER node_ad;
DROP TRIGGER node_au;
DROP TABLE node_fts;
CREATE VIRTUAL TABLE node_fts USING fts5(
  title, body, path, tags_text,
  content='node', content_rowid='id',
  tokenize="porter unicode61 remove_diacritics 2 tokenchars '_-.'"
);
CREATE TRIGGER node_ai AFTER INSERT ON node BEGIN
  INSERT INTO node_fts(rowid, title, body, path, tags_text)
  VALUES (new.id, new.title, new.body, new.path, new.tags_text);
END;
CREATE TRIGGER node_ad AFTER DELETE ON node BEGIN
  INSERT INTO node_fts(node_fts, rowid, title, body, path, tags_text)
  VALUES ('delete', old.id, old.title, old.body, old.path, old.tags_text);
END;
CREATE TRIGGER node_au AFTER UPDATE OF title, body, path, tags_text ON node BEGIN
  INSERT INTO node_fts(node_fts, rowid, title, body, path, tags_text)
  VALUES ('delete', old.id, old.title, old.body, old.path, old.tags_text);
  INSERT INTO node_fts(rowid, title, body, path, tags_text)
  VALUES (new.id, new.title, new.body, new.path, new.tags_text);
END;
INSERT INTO node_fts(node_fts) VALUES ('rebuild');
"#,
    // v3: code-graph lookups — symbols are matched across re-extraction by
    // meta.stable_id, and call resolution buckets by bare name.
    r#"
CREATE INDEX node_symbol_stable ON node(json_extract(meta, '$.stable_id')) WHERE kind = 'symbol';
CREATE INDEX node_symbol_name   ON node(json_extract(meta, '$.name'))      WHERE kind = 'symbol';
"#,
];

pub const SCHEMA_VERSION: i64 = MIGRATIONS.len() as i64;

pub fn migrate(conn: &Connection) -> Result<()> {
    let found: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if found > SCHEMA_VERSION {
        return Err(Error::SchemaTooNew {
            found,
            supported: SCHEMA_VERSION,
        });
    }
    for (i, sql) in MIGRATIONS.iter().enumerate().skip(found as usize) {
        let target = i as i64 + 1;
        conn.execute_batch(&format!(
            "BEGIN;\n{sql}\nPRAGMA user_version = {target};\nCOMMIT;"
        ))?;
        tracing::info!(version = target, "applied schema migration");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_reaches_current_version_and_is_idempotent() {
        let conn = crate::db::open_in_memory().unwrap();
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // Re-running is a no-op, not an error.
        migrate(&conn).unwrap();
    }

    #[test]
    fn v2_porter_stemming_and_up_migration_from_v1() {
        // Fresh DB at current version: stemming works through the triggers.
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute(
            "INSERT INTO node (uid, kind, body, created_at, updated_at)
             VALUES ('X', 'memory', 'we authenticate against the broker', 1, 1)",
            [],
        )
        .unwrap();
        let hits = |q: &str| -> i64 {
            conn.query_row(
                "SELECT count(*) FROM node_fts WHERE node_fts MATCH ?1",
                [q],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(hits("authentication"), 1, "stemmed match");
        assert_eq!(hits("authenticated"), 1, "stemmed match");
    }

    #[test]
    fn migrating_v1_data_rebuilds_fts() {
        // Simulate a v1 database with existing rows, then migrate to v2:
        // the rebuilt index must cover pre-existing content.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "BEGIN;\n{}\nPRAGMA user_version = 1;\nCOMMIT;",
            MIGRATIONS[0]
        ))
        .unwrap();
        conn.execute(
            "INSERT INTO node (uid, kind, body, created_at, updated_at)
             VALUES ('Y', 'memory', 'configuring the indexer pipeline', 1, 1)",
            [],
        )
        .unwrap();
        migrate(&conn).unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM node_fts WHERE node_fts MATCH 'configure'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "old rows must be searchable (stemmed) after rebuild");
    }

    #[test]
    fn refuses_newer_database() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {}", SCHEMA_VERSION + 1))
            .unwrap();
        assert!(matches!(migrate(&conn), Err(Error::SchemaTooNew { .. })));
    }
}
