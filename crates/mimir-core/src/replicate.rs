//! Transport-agnostic replication for the OPTIONAL sync layer.
//!
//! Scope: GLOBAL memories, plus memories of **sync-enabled projects** (those
//! with a portable key — see `scope::portable_key`), and the edges among synced
//! memories. A project memory carries its project's portable key on the wire; on
//! apply it resolves to the local project, creating a path-less *shadow* project
//! if that project hasn't been opened on this machine yet (adopted on first
//! local open). Identity is the node `uid` (a globally unique ULID), so merges
//! are uid-keyed last-write-wins on `updated_at` — convergent and idempotent
//! across machines. Caveat: `updated_at` is second-resolution and an incoming
//! row wins only when STRICTLY newer, so two edits of the same memory within
//! one second resolve by arrival order (one is silently kept, the other
//! dropped). Tombstones ride along as `deleted_at`. Local-only signals
//! (strength, access_count, recall_event) and embeddings are never synced; a
//! peer recomputes embeddings locally.

use rusqlite::{params, Connection, OptionalExtension, Row, TransactionBehavior};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::Rel;
use crate::store;

/// One memory node on the wire. `type` is the memory subkind; `content_hash`
/// is hex-encoded; `deleted_at` is the tombstone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRecord {
    pub uid: String,
    #[serde(rename = "type")]
    pub mtype: Option<String>,
    pub title: Option<String>,
    pub body: Option<String>,
    pub tags: Vec<String>,
    pub pinned: bool,
    pub content_hash: Option<String>,
    pub meta: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
    pub deleted_at: Option<i64>,
    /// Portable key of the owning project (`None` = global memory). Absent on the
    /// wire for global records, so existing global-only peers are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
}

/// A memory↔memory edge on the wire, identified by endpoint uids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEdge {
    pub src_uid: String,
    pub dst_uid: String,
    pub rel: String,
    pub weight: f64,
    pub created_at: i64,
}

/// A unit of replication: nodes, their edges, and the high watermark the
/// sender observed (for the receiver to advance its cursor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBatch {
    pub nodes: Vec<SyncRecord>,
    pub edges: Vec<SyncEdge>,
    pub watermark: i64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ApplyStats {
    pub nodes_upserted: usize,
    pub nodes_skipped: usize,
    pub edges_upserted: usize,
    pub edges_skipped: usize,
    pub tombstones: usize,
}

/// SQL predicate (memory alias `m` LEFT JOINed to its project alias `p`)
/// selecting the memories that replicate: globals (`project_id IS NULL`), plus
/// memories of a sync-enabled project that carries a portable key.
const SYNCABLE: &str = "m.kind='memory' AND (m.project_id IS NULL OR \
    (json_extract(p.meta,'$.sync')=1 AND json_extract(p.meta,'$.portable_key') IS NOT NULL))";

/// Rust twin of `SYNCABLE`'s project condition: a project's memories sync only
/// when its meta has `sync = true` AND a `portable_key`. Keep in lockstep with
/// the SQL predicate above — callers (e.g. `reproject`'s warning) rely on the
/// two agreeing.
pub fn project_is_syncable(meta: &serde_json::Value) -> bool {
    meta.get("sync").and_then(|v| v.as_bool()) == Some(true) && meta.get("portable_key").is_some()
}

/// "Changed since `since`" = syncable memories whose latest activity
/// (max of updated_at and deleted_at) is newer than `since`, plus all edges
/// among synced memories (sparse; shipped whole and upserted idempotently).
pub fn changes_since(conn: &Connection, since: i64) -> Result<SyncBatch> {
    let mut stmt = conn.prepare(&format!(
        "SELECT m.uid, m.subkind, m.title, m.body, m.tags_text, m.pinned, m.content_hash, m.meta,
                m.created_at, m.updated_at, m.deleted_at,
                json_extract(p.meta,'$.portable_key') AS project_key
         FROM node m
         LEFT JOIN node p ON p.id = m.project_id AND p.kind='project'
         WHERE {SYNCABLE}
           AND max(m.updated_at, COALESCE(m.deleted_at, 0)) > ?1
         ORDER BY max(m.updated_at, COALESCE(m.deleted_at, 0)) ASC",
    ))?;
    let nodes = stmt
        .query_map([since], |r: &Row| {
            let hash: Option<Vec<u8>> = r.get("content_hash")?;
            let tags_text: String = r.get("tags_text")?;
            let meta_text: String = r.get("meta")?;
            Ok(SyncRecord {
                uid: r.get("uid")?,
                mtype: r.get("subkind")?,
                title: r.get("title")?,
                body: r.get("body")?,
                tags: tags_text.split_whitespace().map(String::from).collect(),
                pinned: r.get::<_, i64>("pinned")? != 0,
                content_hash: hash.as_deref().map(to_hex),
                meta: serde_json::from_str(&meta_text).unwrap_or(serde_json::Value::Null),
                created_at: r.get("created_at")?,
                updated_at: r.get("updated_at")?,
                deleted_at: r.get("deleted_at")?,
                project_key: r.get("project_key")?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    // Edges between two syncable memories (each endpoint global, or in a
    // sync-enabled keyed project).
    let mut estmt = conn.prepare(
        "SELECT s.uid, d.uid, e.rel, e.weight, e.created_at
         FROM edge e
         JOIN node s ON s.id = e.src
         JOIN node d ON d.id = e.dst
         LEFT JOIN node sp ON sp.id = s.project_id AND sp.kind='project'
         LEFT JOIN node dp ON dp.id = d.project_id AND dp.kind='project'
         WHERE s.kind='memory' AND d.kind='memory'
           AND (s.project_id IS NULL OR (json_extract(sp.meta,'$.sync')=1
                    AND json_extract(sp.meta,'$.portable_key') IS NOT NULL))
           AND (d.project_id IS NULL OR (json_extract(dp.meta,'$.sync')=1
                    AND json_extract(dp.meta,'$.portable_key') IS NOT NULL))",
    )?;
    let edges = estmt
        .query_map([], |r: &Row| {
            Ok(SyncEdge {
                src_uid: r.get(0)?,
                dst_uid: r.get(1)?,
                rel: r.get(2)?,
                weight: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(SyncBatch {
        nodes,
        edges,
        watermark: current_high_watermark(conn)?,
    })
}

/// The full synced set, tombstones included (for the file transport, which
/// ships a convergent snapshot rather than an incremental diff).
pub fn snapshot(conn: &Connection) -> Result<SyncBatch> {
    changes_since(conn, i64::MIN)
}

/// Newest activity timestamp across the synced set (the next watermark).
pub fn current_high_watermark(conn: &Connection) -> Result<i64> {
    let v: Option<i64> = conn
        .query_row(
            &format!(
                "SELECT max(max(m.updated_at, COALESCE(m.deleted_at, 0)))
                 FROM node m
                 LEFT JOIN node p ON p.id = m.project_id AND p.kind='project'
                 WHERE {SYNCABLE}"
            ),
            [],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(v.unwrap_or(0))
}

/// Merge a batch in: uid-keyed last-write-wins for nodes, endpoint-uid-resolved
/// upsert for edges (danglers skipped, never fatal). Idempotent.
pub fn apply_changes(conn: &Connection, batch: &SyncBatch) -> Result<ApplyStats> {
    let mut stats = ApplyStats::default();
    // IMMEDIATE + one transaction for the whole batch: a crash or error
    // mid-apply must not leave a partial batch (some nodes/edges upserted,
    // others not) on disk. Same lock-upgrade rationale as store::record_shown.
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    for n in &batch.nodes {
        let hash: Option<Vec<u8>> = n.content_hash.as_deref().and_then(from_hex);
        let tags_text = n.tags.join(" ");
        // Resolve the owning project from its portable key (None = global). A
        // key with no local project yet gets a path-less shadow project, adopted
        // when that project is later opened locally.
        let project_id: Option<i64> = match n.project_key.as_deref() {
            None => None,
            Some(key) => Some(store::resolve_or_shadow_project(&tx, key)?),
        };
        // Upsert; on uid conflict, overwrite only the synced fields and only when
        // the incoming record is strictly newer. Local signals (strength,
        // access_count, …) are never touched.
        let changed = tx.execute(
            "INSERT INTO node (uid, kind, subkind, project_id, title, body, tags_text, content_hash,
                               meta, pinned, created_at, updated_at, deleted_at)
             VALUES (?1, 'memory', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(uid) DO UPDATE SET
               subkind=excluded.subkind, project_id=excluded.project_id,
               title=excluded.title, body=excluded.body,
               tags_text=excluded.tags_text, content_hash=excluded.content_hash,
               meta=excluded.meta, pinned=excluded.pinned,
               updated_at=excluded.updated_at, deleted_at=excluded.deleted_at
             WHERE max(excluded.updated_at, COALESCE(excluded.deleted_at, 0))
                     > max(node.updated_at, COALESCE(node.deleted_at, 0))
                OR (max(excluded.updated_at, COALESCE(excluded.deleted_at, 0))
                      = max(node.updated_at, COALESCE(node.deleted_at, 0))
                    AND excluded.deleted_at IS NOT NULL AND node.deleted_at IS NULL)",
            params![
                n.uid,
                n.mtype,
                project_id,
                n.title,
                n.body,
                tags_text,
                hash,
                n.meta.to_string(),
                n.pinned as i64,
                n.created_at,
                n.updated_at,
                n.deleted_at,
            ],
        )?;
        if changed > 0 {
            stats.nodes_upserted += 1;
            if n.deleted_at.is_some() {
                stats.tombstones += 1;
            }
        } else {
            stats.nodes_skipped += 1;
        }
    }

    for e in &batch.edges {
        let src = store::get_node_by_uid(&tx, &e.src_uid).ok();
        let dst = store::get_node_by_uid(&tx, &e.dst_uid).ok();
        match (src, dst, e.rel.parse::<Rel>()) {
            (Some(s), Some(d), Ok(rel)) => {
                store::link(&tx, s.id, d.id, rel, e.weight)?;
                stats.edges_upserted += 1;
            }
            _ => stats.edges_skipped += 1,
        }
    }
    tx.commit()?;
    Ok(stats)
}

const META_PREFIX: &str = "sync.";

pub fn get_watermark(conn: &Connection, key: &str) -> Result<i64> {
    let v: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            [format!("{META_PREFIX}{key}")],
            |r| r.get(0),
        )
        .optional()?;
    Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
}

pub fn set_watermark(conn: &Connection, key: &str, value: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![format!("{META_PREFIX}{key}"), value.to_string()],
    )?;
    Ok(())
}

/// A fresh random bearer token (ULID; ~80 bits of entropy).
pub fn new_token() -> String {
    ulid::Ulid::new().to_string()
}

/// Read a `sync.<key>` string from the meta table.
pub fn get_str_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            [format!("{META_PREFIX}{key}")],
            |r| r.get(0),
        )
        .optional()?)
}

/// Write a `sync.<key>` string into the meta table.
pub fn set_str_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![format!("{META_PREFIX}{key}"), value],
    )?;
    Ok(())
}

/// Stable per-install id (for naming this machine's snapshot file). Generated
/// once and persisted in meta.
pub fn machine_id(conn: &Connection) -> Result<String> {
    if let Some(id) = conn
        .query_row(
            "SELECT value FROM meta WHERE key='sync.machine_id'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()?
    {
        return Ok(id);
    }
    let id = ulid::Ulid::new().to_string();
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('sync.machine_id', ?1)",
        [&id],
    )?;
    Ok(id)
}

fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::{remember, Remember};
    use crate::model::{Kind, MemoryType};

    fn mem(conn: &Connection, text: &str) -> crate::model::Node {
        match remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec!["t".into()],
                project_id: None,
                force: true,
            },
        )
        .unwrap()
        {
            crate::memory::RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        }
    }

    fn synced_project(conn: &Connection, key: &str) -> i64 {
        let p = store::ensure_project(conn, &format!("/local/{key}"), "proj").unwrap();
        store::set_project_identity(conn, p.id, Some(key), true).unwrap();
        p.id
    }

    fn proj_mem(conn: &Connection, pid: i64, text: &str) -> crate::model::Node {
        match remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec!["t".into()],
                project_id: Some(pid),
                force: true,
            },
        )
        .unwrap()
        {
            crate::memory::RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        }
    }

    #[test]
    fn project_memory_syncs_via_key_and_creates_shadow() {
        let a = db::open_in_memory().unwrap();
        let pid = synced_project(&a, "git:github.com/o/r");
        let g = mem(&a, "global fact");
        let pm = proj_mem(&a, pid, "project fact");

        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 2, "global + synced-project memory");
        let pr = batch.nodes.iter().find(|r| r.uid == pm.uid).unwrap();
        assert_eq!(pr.project_key.as_deref(), Some("git:github.com/o/r"));
        let gr = batch.nodes.iter().find(|r| r.uid == g.uid).unwrap();
        assert_eq!(gr.project_key, None);

        // Fresh peer: a shadow project is created and the memory attaches to it;
        // the global memory stays global.
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &batch).unwrap();
        let shadow = store::find_project_by_key(&b, "git:github.com/o/r")
            .unwrap()
            .expect("shadow project created");
        let attached: Option<i64> = b
            .query_row("SELECT project_id FROM node WHERE uid=?1", [&pm.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(attached, Some(shadow.id));
        let gpid: Option<i64> = b
            .query_row("SELECT project_id FROM node WHERE uid=?1", [&g.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(gpid, None, "global stays global on the peer");
    }

    #[test]
    fn unsynced_project_memory_stays_local() {
        let a = db::open_in_memory().unwrap();
        let p = store::ensure_project(&a, "/local/x", "x").unwrap();
        // Has a key but sync=false → excluded from replication.
        store::set_project_identity(&a, p.id, Some("git:github.com/o/r"), false).unwrap();
        proj_mem(&a, p.id, "private fact");
        mem(&a, "global fact");
        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 1, "only the global memory syncs");
        assert_eq!(batch.nodes[0].project_key, None);
    }

    #[test]
    fn record_without_project_key_deserializes_as_global() {
        // An old global-only peer omits the field entirely.
        let json = r#"{"uid":"01ABC","type":"note","title":null,"body":"x","tags":[],
            "pinned":false,"content_hash":null,"meta":{},"created_at":1,"updated_at":1,
            "deleted_at":null}"#;
        let rec: SyncRecord = serde_json::from_str(json).unwrap();
        assert_eq!(rec.project_key, None);
    }

    #[test]
    fn roundtrip_and_idempotent() {
        let a = db::open_in_memory().unwrap();
        mem(&a, "alpha fact");
        mem(&a, "beta fact");
        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 2);

        let b = db::open_in_memory().unwrap();
        let s1 = apply_changes(&b, &batch).unwrap();
        assert_eq!(s1.nodes_upserted, 2);
        // Second apply is a no-op (last-write-wins guard).
        let s2 = apply_changes(&b, &batch).unwrap();
        assert_eq!(s2.nodes_upserted, 0);
        assert_eq!(s2.nodes_skipped, 2);
        assert_eq!(changes_since(&b, i64::MIN).unwrap().nodes.len(), 2);
    }

    #[test]
    fn last_write_wins_both_directions() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "shared fact");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        // Newer edit on A wins on B.
        a.execute(
            "UPDATE node SET body='edited', updated_at=updated_at+100 WHERE uid=?1",
            [&n.uid],
        )
        .unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        let body: String = b
            .query_row("SELECT body FROM node WHERE uid=?1", [&n.uid], |r| r.get(0))
            .unwrap();
        assert_eq!(body, "edited");

        // A stale older edit must NOT overwrite.
        let stale = SyncBatch {
            nodes: vec![SyncRecord {
                uid: n.uid.clone(),
                mtype: Some("note".into()),
                title: Some("x".into()),
                body: Some("STALE".into()),
                tags: vec![],
                pinned: false,
                content_hash: None,
                meta: serde_json::json!({}),
                created_at: n.created_at,
                updated_at: 1,
                deleted_at: None,
                project_key: None,
            }],
            edges: vec![],
            watermark: 0,
        };
        let s = apply_changes(&b, &stale).unwrap();
        assert_eq!(s.nodes_skipped, 1);
        let body: String = b
            .query_row("SELECT body FROM node WHERE uid=?1", [&n.uid], |r| r.get(0))
            .unwrap();
        assert_eq!(body, "edited", "stale write rejected");
    }

    #[test]
    fn tombstone_propagates_and_sticks() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "doomed fact");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        store::soft_delete(&a, n.id).unwrap();
        let batch = snapshot(&a).unwrap();
        assert!(batch
            .nodes
            .iter()
            .any(|r| r.uid == n.uid && r.deleted_at.is_some()));
        let s = apply_changes(&b, &batch).unwrap();
        assert_eq!(s.tombstones, 1);
        let del: Option<i64> = b
            .query_row("SELECT deleted_at FROM node WHERE uid=?1", [&n.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(del.is_some(), "deleted on peer");

        // A stale pre-delete snapshot must not resurrect it.
        let resurrect = SyncBatch {
            nodes: vec![SyncRecord {
                uid: n.uid.clone(),
                mtype: Some("note".into()),
                title: None,
                body: Some("back".into()),
                tags: vec![],
                pinned: false,
                content_hash: None,
                meta: serde_json::json!({}),
                created_at: n.created_at,
                updated_at: n.updated_at,
                deleted_at: None,
                project_key: None,
            }],
            edges: vec![],
            watermark: 0,
        };
        apply_changes(&b, &resurrect).unwrap();
        let del: Option<i64> = b
            .query_row("SELECT deleted_at FROM node WHERE uid=?1", [&n.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(del.is_some(), "stale write did not resurrect");
    }

    #[test]
    fn edge_skipped_until_endpoints_present() {
        let a = db::open_in_memory().unwrap();
        let x = mem(&a, "node x");
        let y = mem(&a, "node y");
        store::link(&a, x.id, y.id, Rel::Relates, 1.0).unwrap();

        // Deliver only the edge + node x → edge can't resolve y, skipped.
        let b = db::open_in_memory().unwrap();
        let full = snapshot(&a).unwrap();
        let partial = SyncBatch {
            nodes: full
                .nodes
                .iter()
                .filter(|r| r.uid == x.uid)
                .cloned()
                .collect(),
            edges: full.edges.clone(),
            watermark: full.watermark,
        };
        let s = apply_changes(&b, &partial).unwrap();
        assert_eq!(s.edges_skipped, 1);

        // Now deliver everything → edge resolves.
        let s = apply_changes(&b, &full).unwrap();
        assert_eq!(s.edges_upserted, 1);
    }

    #[test]
    fn soft_delete_bumps_updated_at() {
        let conn = db::open_in_memory().unwrap();
        let n = {
            let mut new = crate::model::NewNode::new(Kind::Memory);
            new.title = Some("x".into());
            new.body = Some("x".into());
            store::insert_node(&conn, new).unwrap()
        };
        let before: i64 = conn
            .query_row("SELECT updated_at FROM node WHERE id=?1", [n.id], |r| {
                r.get(0)
            })
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        store::soft_delete(&conn, n.id).unwrap();
        let after: i64 = conn
            .query_row("SELECT updated_at FROM node WHERE id=?1", [n.id], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(after > before, "soft_delete bumped updated_at");
    }
}
