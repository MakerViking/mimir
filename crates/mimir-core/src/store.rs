use rusqlite::{params, Connection, OptionalExtension, Row};
use ulid::Ulid;

use crate::error::{Error, Result};
use crate::model::{now_unix, Edge, Kind, NewNode, Node, Rel};

pub fn row_to_node(row: &Row) -> rusqlite::Result<Node> {
    let kind_text: String = row.get("kind")?;
    let rel_err = |e: Error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    };
    let meta_text: String = row.get("meta")?;
    Ok(Node {
        id: row.get("id")?,
        uid: row.get("uid")?,
        kind: kind_text.parse().map_err(rel_err)?,
        subkind: row.get("subkind")?,
        project_id: row.get("project_id")?,
        collection_id: row.get("collection_id")?,
        parent_id: row.get("parent_id")?,
        title: row.get("title")?,
        body: row.get("body")?,
        path: row.get("path")?,
        span_start: row.get("span_start")?,
        span_end: row.get("span_end")?,
        lang: row.get("lang")?,
        tags_text: row.get("tags_text")?,
        content_hash: row.get("content_hash")?,
        meta: serde_json::from_str(&meta_text).unwrap_or(serde_json::Value::Null),
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        access_count: row.get("access_count")?,
        last_accessed: row.get("last_accessed")?,
        strength: row.get("strength")?,
        pinned: row.get::<_, i64>("pinned")? != 0,
        superseded_by: row.get("superseded_by")?,
        deleted_at: row.get("deleted_at")?,
    })
}

pub const NODE_COLS: &str = "id, uid, kind, subkind, project_id, collection_id, parent_id, \
     title, body, path, span_start, span_end, lang, tags_text, content_hash, meta, \
     created_at, updated_at, access_count, last_accessed, strength, pinned, superseded_by, deleted_at";

pub fn insert_node(conn: &Connection, new: NewNode) -> Result<Node> {
    let kind = new
        .kind
        .ok_or_else(|| Error::Invalid("NewNode.kind is required".into()))?;
    let uid = Ulid::new().to_string();
    let now = now_unix();
    let tags_text = new.tags.join(" ");
    let meta = new.meta.unwrap_or_else(|| serde_json::json!({}));
    conn.execute(
        "INSERT INTO node (uid, kind, subkind, project_id, collection_id, parent_id, title, body, \
                           path, span_start, span_end, lang, tags_text, content_hash, meta, \
                           created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?16)",
        params![
            uid,
            kind.as_str(),
            new.subkind,
            new.project_id,
            new.collection_id,
            new.parent_id,
            new.title,
            new.body,
            new.path,
            new.span_start,
            new.span_end,
            new.lang,
            tags_text,
            new.content_hash,
            meta.to_string(),
            now,
        ],
    )?;
    let id = conn.last_insert_rowid();
    get_node(conn, id)
}

pub fn get_node(conn: &Connection, id: i64) -> Result<Node> {
    conn.query_row(
        &format!("SELECT {NODE_COLS} FROM node WHERE id = ?1"),
        [id],
        row_to_node,
    )
    .optional()?
    .ok_or_else(|| Error::NotFound(format!("node #{id}")))
}

/// Batch fetch, one round-trip instead of one query per id — the fused
/// hybrid-search candidate list can be up to 200 ids (two 100-cap legs).
/// Order is unspecified (SQLite's `IN` doesn't preserve list order); callers
/// that need a particular order must sort themselves. Ids with no matching
/// live row (e.g. deleted between fusion and fetch) are simply absent, not
/// an error.
pub fn get_nodes(conn: &Connection, ids: &[i64]) -> Result<Vec<Node>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_list = ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node WHERE id IN ({id_list})"
    ))?;
    let nodes = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(nodes)
}

/// Batch-fetch shown/opened counts (`recall_event`) for `ids`, one round
/// trip via `GROUP BY` — the impression-damp negative prior (`learn::
/// impression_damp`) needs this per fused-search candidate, and the score
/// site is hot, so a per-node query is not an option (see `search_hybrid`'s
/// call site: this runs alongside [`get_nodes`] over the same id list, and
/// only when `scoring.impression_alpha > 0.0` — otherwise skipped entirely).
/// Ids with no `recall_event` rows are simply absent from the result map
/// (equivalent to `ImpressionStats::default()`), not an error.
pub fn impression_stats(
    conn: &Connection,
    ids: &[i64],
) -> Result<std::collections::HashMap<i64, crate::learn::ImpressionStats>> {
    if ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let id_list = ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT node_id,
                SUM(CASE WHEN event = 'shown' THEN 1 ELSE 0 END) AS shown,
                SUM(CASE WHEN event = 'opened' THEN 1 ELSE 0 END) AS opened,
                MAX(CASE WHEN event = 'shown' THEN at END) AS last_shown_at
         FROM recall_event
         WHERE node_id IN ({id_list}) AND event IN ('shown', 'opened')
         GROUP BY node_id"
    ))?;
    let map = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                crate::learn::ImpressionStats {
                    shown: r.get(1)?,
                    opened: r.get(2)?,
                    last_shown_at: r.get(3)?,
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(map)
}

pub fn get_node_by_uid(conn: &Connection, uid: &str) -> Result<Node> {
    conn.query_row(
        &format!("SELECT {NODE_COLS} FROM node WHERE uid = ?1"),
        [uid],
        row_to_node,
    )
    .optional()?
    .ok_or_else(|| Error::NotFound(format!("node {uid}")))
}

/// Resolve a user-facing reference: full ULID, `m:ABCDEF` short form,
/// or a bare uid tail (≥4 chars). Deleted nodes are excluded.
pub fn resolve_ref(conn: &Connection, reference: &str) -> Result<Node> {
    let raw = reference.trim();
    // Strip a single-letter sigil prefix like "m:" if present.
    let tail = match raw.split_once(':') {
        Some((sigil, rest)) if sigil.len() == 1 => rest,
        _ => raw,
    };
    if tail.len() == 26 {
        if let Ok(node) = get_node_by_uid(conn, &tail.to_uppercase()) {
            // Same contract as the short-tail path below: deleted nodes
            // resolve to NotFound, not to a ghost you can edit/mark/open.
            if node.deleted_at.is_none() {
                return Ok(node);
            }
            return Err(Error::NotFound(format!("no node matching '{raw}'")));
        }
    }
    if tail.len() < 4 {
        return Err(Error::Invalid(format!(
            "reference '{raw}' too short; use at least 4 trailing characters of the id"
        )));
    }
    // The short-id form (`m:ABCDEF`) is exactly 6 trailing chars, which the
    // `node_uid_tail` index on substr(uid,-6) serves via equality — an O(log
    // N) seek. A `LIKE '%tail'` leading wildcard can't use any index and
    // full-scans (≈66 ms at 500k nodes), so only fall back to it for the
    // uncommon non-6-char tail.
    let up = tail.to_uppercase();
    let (sql, param) = if tail.len() == 6 {
        (
            format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE substr(uid, -6) = ?1 AND deleted_at IS NULL LIMIT 3"
            ),
            up,
        )
    } else {
        (
            format!(
                "SELECT {NODE_COLS} FROM node WHERE uid LIKE ?1 AND deleted_at IS NULL LIMIT 3"
            ),
            format!("%{up}"),
        )
    };
    let mut stmt = conn.prepare(&sql)?;
    let nodes: Vec<Node> = stmt
        .query_map([&param], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    match nodes.len() {
        0 => Err(Error::NotFound(format!("no node matching '{raw}'"))),
        1 => Ok(nodes.into_iter().next().unwrap()),
        n => Err(Error::AmbiguousRef(raw.to_string(), n)),
    }
}

pub fn touch_updated(conn: &Connection, id: i64) -> Result<()> {
    conn.execute(
        "UPDATE node SET updated_at = ?2 WHERE id = ?1",
        params![id, now_unix()],
    )?;
    Ok(())
}

pub fn soft_delete(conn: &Connection, id: i64) -> Result<()> {
    // Bump updated_at too: replication finds tombstones via updated_at, so a
    // delete that didn't advance it would never propagate.
    conn.execute(
        "UPDATE node SET deleted_at = ?2, updated_at = ?2 WHERE id = ?1 AND deleted_at IS NULL",
        params![id, now_unix()],
    )?;
    Ok(())
}

pub fn hard_delete(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM node WHERE id = ?1", [id])?;
    Ok(())
}

/// Mark `id` as superseded by `by`: it stops surfacing in recall (kept as
/// history). Bumps updated_at so the change replicates to a sync hub.
/// Refuses self-supersession (there is no un-supersede path, so a
/// self-reference would hide the node unrecoverably) and pinned nodes
/// (pinned means never decayed, never superseded — same invariant
/// consolidation honors).
pub fn set_superseded(conn: &Connection, id: i64, by: i64) -> Result<()> {
    if id == by {
        return Err(Error::Invalid("a node cannot supersede itself".into()));
    }
    let pinned: i64 =
        conn.query_row("SELECT pinned FROM node WHERE id = ?1", [id], |r| r.get(0))?;
    if pinned != 0 {
        return Err(Error::Invalid(format!(
            "node #{id} is pinned; pinned nodes never get superseded (unpin it first)"
        )));
    }
    conn.execute(
        "UPDATE node SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
        params![id, by, now_unix()],
    )?;
    Ok(())
}

/// Idempotent edge insert (UNIQUE(src,dst,rel) upserts weight).
pub fn link(conn: &Connection, src: i64, dst: i64, rel: Rel, weight: f64) -> Result<()> {
    conn.execute(
        "INSERT INTO edge (src, dst, rel, weight, created_at) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(src, dst, rel) DO UPDATE SET weight = excluded.weight",
        params![src, dst, rel.as_str(), weight, now_unix()],
    )?;
    Ok(())
}

pub fn edges_of(conn: &Connection, node_id: i64) -> Result<Vec<Edge>> {
    let mut stmt = conn.prepare(
        "SELECT id, src, dst, rel, weight, meta, created_at FROM edge
         WHERE src = ?1 OR dst = ?1",
    )?;
    let edges = stmt
        .query_map([node_id], |row| {
            let rel_text: String = row.get("rel")?;
            let meta_text: String = row.get("meta")?;
            Ok(Edge {
                id: row.get("id")?,
                src: row.get("src")?,
                dst: row.get("dst")?,
                rel: rel_text.parse().map_err(|e: Error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?,
                weight: row.get("weight")?,
                meta: serde_json::from_str(&meta_text).unwrap_or(serde_json::Value::Null),
                created_at: row.get("created_at")?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(edges)
}

/// Find or create the project node for a filesystem root.
/// Project identity is its canonical root path (stored in `path`).
pub fn ensure_project(conn: &Connection, root: &str, name: &str) -> Result<Node> {
    let existing = conn
        .query_row(
            &format!("SELECT {NODE_COLS} FROM node WHERE kind = 'project' AND path = ?1"),
            [root],
            row_to_node,
        )
        .optional()?;
    if let Some(node) = existing {
        return Ok(node);
    }
    let mut new = NewNode::new(Kind::Project);
    new.title = Some(name.to_string());
    new.path = Some(root.to_string());
    insert_node(conn, new)
}

/// Find a project node by its portable sync key (`meta.portable_key`).
pub fn find_project_by_key(conn: &Connection, key: &str) -> Result<Option<Node>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'project' AND json_extract(meta, '$.portable_key') = ?1"
            ),
            [key],
            row_to_node,
        )
        .optional()?)
}

/// Record a project's portable sync identity (`meta.portable_key` + `meta.sync`).
/// Idempotent and a no-op when unchanged, so it's cheap to call on every project
/// resolution. `key = None` clears the stored key (project stays local-only).
pub fn set_project_identity(
    conn: &Connection,
    project_id: i64,
    key: Option<&str>,
    sync: bool,
) -> Result<()> {
    let (cur_key, cur_sync): (Option<String>, bool) = conn.query_row(
        "SELECT json_extract(meta, '$.portable_key'),
                COALESCE(json_extract(meta, '$.sync'), 0)
         FROM node WHERE id = ?1",
        [project_id],
        |r| Ok((r.get(0)?, r.get::<_, i64>(1)? != 0)),
    )?;
    if cur_key.as_deref() == key && cur_sync == sync {
        return Ok(());
    }
    conn.execute(
        "UPDATE node
         SET meta = json_set(json_set(meta, '$.portable_key', ?2), '$.sync', json(?3))
         WHERE id = ?1 AND kind = 'project'",
        params![project_id, key, if sync { "true" } else { "false" }],
    )?;
    Ok(())
}

/// Set a memory's declared trigger phrases (`meta.fires_when`) — the
/// author's own "fires when ..." conditions, checked by `inject::
/// clears_floor` alongside (and as a bypass for) the inferred-relevance
/// floor. Overwrites any previous list; `phrases` is expected pre-sanitized
/// (see `memory::sanitize_fires_when`) — this is a raw setter, same
/// division of labor as `sanitize_tags` vs. `insert_node`.
pub fn set_fires_when(conn: &Connection, id: i64, phrases: &[String]) -> Result<()> {
    let json = serde_json::to_string(phrases).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.fires_when', json(?2)) WHERE id = ?1",
        params![id, json],
    )?;
    Ok(())
}

/// Bind an existing keyed (possibly synced-shadow) project to this machine's
/// local path + display name, clearing the shadow marker. Used when a project
/// that first arrived via sync is opened locally.
pub fn adopt_project(conn: &Connection, project_id: i64, path: &str, name: &str) -> Result<()> {
    conn.execute(
        "UPDATE node
         SET path = ?2, title = ?3, meta = json_remove(meta, '$.shadow')
         WHERE id = ?1 AND kind = 'project'",
        params![project_id, path, name],
    )?;
    Ok(())
}

/// Resolve a project by portable key for sync apply, creating a path-less
/// **shadow** project if none exists yet (adopted when the project is later
/// opened locally). Ensures a pulled project memory always has a project to
/// attach to, even before that project exists on this machine.
pub fn resolve_or_shadow_project(conn: &Connection, key: &str) -> Result<i64> {
    if let Some(p) = find_project_by_key(conn, key)? {
        return Ok(p.id);
    }
    let name = key.rsplit('/').next().unwrap_or(key).to_string();
    let mut new = NewNode::new(Kind::Project);
    new.title = Some(name);
    new.meta = Some(serde_json::json!({ "portable_key": key, "sync": true, "shadow": true }));
    Ok(insert_node(conn, new)?.id)
}

/// Live file nodes whose relative path ends with `suffix` (max 3).
pub fn files_by_path_suffix(conn: &Connection, suffix: &str) -> Result<Vec<Node>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'file' AND deleted_at IS NULL AND path LIKE ?1
         LIMIT 3"
    ))?;
    let nodes = stmt
        .query_map([format!("%{suffix}")], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(nodes)
}

/// Map of project node id → display name, for rendering `pr:name` scopes.
pub fn project_titles(conn: &Connection) -> Result<std::collections::HashMap<i64, String>> {
    let mut stmt =
        conn.prepare("SELECT id, title FROM node WHERE kind = 'project' AND deleted_at IS NULL")?;
    let map = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(map)
}

/// Delete `recall_event` rows older than `cutoff` (unix seconds), one
/// statement (see `config::LearnConfig::event_retention_days`). Runs off a
/// background idle timer (`db::spawn_checkpoint_timer`), never on the hot
/// recall/score path. Returns the row count removed, for logging/tests.
pub fn prune_old_events(conn: &Connection, cutoff: i64) -> Result<usize> {
    Ok(conn.execute("DELETE FROM recall_event WHERE at < ?1", params![cutoff])?)
}

/// True if `node_id` was already injected in `session_id` (the v8
/// `injection_log` table — see `inject::select_unseen_injection`, which
/// treats a read error here as "unseen" so a store hiccup degrades to the
/// pre-ledger behavior rather than blocking injection).
pub fn injection_seen(conn: &Connection, session_id: &str, node_id: i64) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM injection_log WHERE session_id = ?1 AND node_id = ?2",
            params![session_id, node_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

/// Record that `node_id` was just injected in `session_id`, so the same
/// memory doesn't re-inject on the session's next prompt. Idempotent
/// (`PRIMARY KEY (session_id, node_id)`): a repeat call just refreshes `at`.
pub fn record_injection(conn: &Connection, session_id: &str, node_id: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO injection_log (session_id, node_id, at) VALUES (?1, ?2, ?3)
         ON CONFLICT(session_id, node_id) DO UPDATE SET at = excluded.at",
        params![session_id, node_id, now_unix()],
    )?;
    Ok(())
}

/// Delete `injection_log` and `session_state` rows older than `cutoff`
/// (unix seconds) — both v8 tables are ephemeral per-session working state,
/// not knowledge, so they're pruned on the same idle cadence as
/// `recall_event` (see `prune_old_events`). Returns the total row count
/// removed across both tables. Exported for the ops-owned prune cadence
/// (`db::spawn_checkpoint_timer` or its cron equivalent) to wire in; not
/// called from anywhere in this module.
pub fn prune_session_state(conn: &Connection, cutoff: i64) -> Result<usize> {
    let a = conn.execute("DELETE FROM injection_log WHERE at < ?1", params![cutoff])?;
    let b = conn.execute(
        "DELETE FROM session_state WHERE updated_at < ?1",
        params![cutoff],
    )?;
    Ok(a + b)
}

/// Append to the implicit-feedback ledger (shown/opened/useful/not_useful).
pub fn record_event(
    conn: &Connection,
    node_id: i64,
    event: &str,
    query_hash: Option<&[u8]>,
    rank: Option<i64>,
    score: Option<f64>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO recall_event (node_id, event, query_hash, rank, score, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![node_id, event, query_hash, rank, score, now_unix()],
    )?;
    Ok(())
}

/// Log `shown` for a whole result page in one transaction
/// (per-row autocommits double recall latency for nothing).
pub fn record_shown(
    conn: &Connection,
    query_hash: &[u8],
    hits: &[(i64, i64, f64)], // (node_id, rank, score)
) -> Result<()> {
    // IMMEDIATE: recall's own write must wait for a concurrent writer, not
    // abort on a DEFERRED read→write lock upgrade (busy_timeout can't wait
    // on an upgrade).
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO recall_event (node_id, event, query_hash, rank, score, at)
             VALUES (?1, 'shown', ?2, ?3, ?4, ?5)",
        )?;
        let now = now_unix();
        for (node_id, rank, score) in hits {
            stmt.execute(params![node_id, query_hash, rank, score, now])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Pin/unpin: pinned nodes never decay and never get superseded.
pub fn set_pinned(conn: &Connection, id: i64, pinned: bool) -> Result<()> {
    conn.execute(
        "UPDATE node SET pinned = ?2 WHERE id = ?1",
        params![id, pinned],
    )?;
    Ok(())
}

/// Bump access stats; called when a node's full body is opened.
pub fn touch_accessed(conn: &Connection, id: i64) -> Result<()> {
    conn.execute(
        "UPDATE node SET access_count = access_count + 1, last_accessed = ?2 WHERE id = ?1",
        params![id, now_unix()],
    )?;
    Ok(())
}

pub fn count_by_kind(conn: &Connection) -> Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT kind, COUNT(*) FROM node WHERE deleted_at IS NULL GROUP BY kind ORDER BY kind",
    )?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn conn() -> Connection {
        db::open_in_memory().unwrap()
    }

    #[test]
    fn project_identity_set_and_lookup_by_key() {
        let c = conn();
        let p = ensure_project(&c, "/abs/path/proj", "proj").unwrap();
        assert!(find_project_by_key(&c, "git:github.com/o/r")
            .unwrap()
            .is_none());

        set_project_identity(&c, p.id, Some("git:github.com/o/r"), true).unwrap();
        let found = find_project_by_key(&c, "git:github.com/o/r").unwrap();
        assert_eq!(found.map(|n| n.id), Some(p.id));

        // sync flag stored as JSON boolean.
        let sync: i64 = c
            .query_row(
                "SELECT json_extract(meta,'$.sync') FROM node WHERE id=?1",
                [p.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(sync, 1);

        // Clearing the key makes it unfindable again (local-only).
        set_project_identity(&c, p.id, None, false).unwrap();
        assert!(find_project_by_key(&c, "git:github.com/o/r")
            .unwrap()
            .is_none());
    }

    fn memory(conn: &Connection, title: &str, body: &str) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some(title.into());
        new.body = Some(body.into());
        new.tags = vec!["alpha".into(), "beta".into()];
        insert_node(conn, new).unwrap()
    }

    #[test]
    fn insert_and_get_round_trip() {
        let conn = conn();
        let node = memory(&conn, "hello", "world");
        assert_eq!(node.uid.len(), 26);
        assert_eq!(node.kind, Kind::Memory);
        assert_eq!(node.subkind.as_deref(), Some("note"));
        assert_eq!(node.tags(), vec!["alpha", "beta"]);
        assert_eq!(node.strength, 1.0);
        assert!(!node.pinned);

        let by_id = get_node(&conn, node.id).unwrap();
        assert_eq!(by_id.uid, node.uid);
        let by_uid = get_node_by_uid(&conn, &node.uid).unwrap();
        assert_eq!(by_uid.id, node.id);
    }

    #[test]
    fn get_nodes_batches_and_matches_individual_lookups() {
        let conn = conn();
        let a = memory(&conn, "alpha", "x");
        let b = memory(&conn, "beta", "y");
        let c = memory(&conn, "gamma", "z");
        soft_delete(&conn, c.id).unwrap(); // still fetchable by id, just flagged

        let batch = get_nodes(&conn, &[a.id, b.id, c.id, 999_999]).unwrap();
        // Same set as calling get_node per id (999_999 simply absent, no error).
        let mut want = [
            get_node(&conn, a.id).unwrap(),
            get_node(&conn, b.id).unwrap(),
            get_node(&conn, c.id).unwrap(),
        ];
        let mut got = batch;
        let by_id = |n: &Node| n.id;
        want.sort_by_key(by_id);
        got.sort_by_key(by_id);
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(w.id, g.id);
            assert_eq!(w.title, g.title);
            assert_eq!(w.deleted_at.is_some(), g.deleted_at.is_some());
        }
    }

    #[test]
    fn get_nodes_empty_ids_is_empty() {
        let conn = conn();
        assert!(get_nodes(&conn, &[]).unwrap().is_empty());
    }

    #[test]
    fn impression_stats_batches_and_groups_correctly() {
        let conn = conn();
        let a = memory(&conn, "alpha", "x");
        let b = memory(&conn, "beta", "y");
        // a: shown twice, opened once. b: shown once, never opened.
        record_shown(&conn, b"q1", &[(a.id, 0, 1.0), (a.id, 1, 0.5)]).unwrap();
        record_shown(&conn, b"q1", &[(b.id, 0, 1.0)]).unwrap();
        record_event(&conn, a.id, "opened", None, None, None).unwrap();

        let stats = impression_stats(&conn, &[a.id, b.id, 999_999]).unwrap();
        assert_eq!(
            stats.len(),
            2,
            "absent id must simply be missing, not an error"
        );

        let sa = stats[&a.id];
        assert_eq!(sa.shown, 2);
        assert_eq!(sa.opened, 1);
        assert!(sa.last_shown_at.is_some());

        let sb = stats[&b.id];
        assert_eq!(sb.shown, 1);
        assert_eq!(sb.opened, 0);
    }

    #[test]
    fn impression_stats_empty_ids_is_empty() {
        let conn = conn();
        assert!(impression_stats(&conn, &[]).unwrap().is_empty());
    }

    #[test]
    fn prune_old_events_removes_only_rows_older_than_cutoff() {
        let conn = conn();
        let n = memory(&conn, "gamma", "z");
        let now = now_unix();
        // One old row, one recent row, planted with explicit timestamps
        // (record_event/record_shown always stamp `now_unix()`, so this
        // needs raw inserts to control `at`).
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'shown', ?2)",
            params![n.id, now - 200 * 86_400],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'shown', ?2)",
            params![n.id, now - 10 * 86_400],
        )
        .unwrap();

        let removed = prune_old_events(&conn, now - 180 * 86_400).unwrap();
        assert_eq!(
            removed, 1,
            "only the 200-day-old row clears the 180-day cutoff"
        );

        let remaining: i64 = conn
            .query_row("SELECT count(*) FROM recall_event", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1, "the 10-day-old row must survive");
    }

    #[test]
    fn prune_old_events_is_a_noop_when_nothing_is_old_enough() {
        let conn = conn();
        let n = memory(&conn, "delta", "w");
        record_event(&conn, n.id, "shown", None, None, None).unwrap();
        let removed = prune_old_events(&conn, now_unix() - 180 * 86_400).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn set_fires_when_round_trips_through_meta() {
        let conn = conn();
        let n = memory(&conn, "release checklist", "bump version, tag, publish");
        assert!(get_node(&conn, n.id)
            .unwrap()
            .meta
            .get("fires_when")
            .is_none());

        set_fires_when(
            &conn,
            n.id,
            &["release checklist".into(), "cutting a release".into()],
        )
        .unwrap();
        let got = get_node(&conn, n.id).unwrap();
        let phrases: Vec<String> = got
            .meta
            .get("fires_when")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(phrases, vec!["release checklist", "cutting a release"]);

        // Overwrites, doesn't append.
        set_fires_when(&conn, n.id, &["only one now".into()]).unwrap();
        let got = get_node(&conn, n.id).unwrap();
        assert_eq!(
            got.meta
                .get("fires_when")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn injection_ledger_records_and_checks_per_session() {
        let conn = conn();
        let n = memory(&conn, "gotcha", "x");
        assert!(!injection_seen(&conn, "sess-a", n.id).unwrap());

        record_injection(&conn, "sess-a", n.id).unwrap();
        assert!(injection_seen(&conn, "sess-a", n.id).unwrap());
        // A different session never saw it.
        assert!(!injection_seen(&conn, "sess-b", n.id).unwrap());

        // Idempotent: recording again doesn't error or duplicate.
        record_injection(&conn, "sess-a", n.id).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM injection_log", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn prune_session_state_removes_only_stale_rows_from_both_tables() {
        let conn = conn();
        let n = memory(&conn, "gamma", "z");
        let now = now_unix();
        conn.execute(
            "INSERT INTO injection_log (session_id, node_id, at) VALUES ('old', ?1, ?2)",
            params![n.id, now - 10 * 86_400],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO injection_log (session_id, node_id, at) VALUES ('fresh', ?1, ?2)",
            params![n.id, now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_state (session_id, key, value, updated_at) \
             VALUES ('old', 'k', 'v', ?1)",
            params![now - 10 * 86_400],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_state (session_id, key, value, updated_at) \
             VALUES ('fresh', 'k', 'v', ?1)",
            params![now],
        )
        .unwrap();

        let removed = prune_session_state(&conn, now - 7 * 86_400).unwrap();
        assert_eq!(removed, 2, "one stale row from each table");
        assert!(injection_seen(&conn, "fresh", n.id).unwrap());
        assert!(!injection_seen(&conn, "old", n.id).unwrap());
    }

    #[test]
    fn insert_requires_kind() {
        let conn = conn();
        let err = insert_node(&conn, NewNode::default()).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }

    #[test]
    fn supersede_refuses_self_and_pinned() {
        let conn = conn();
        let old = memory(&conn, "old", "x");
        let new = memory(&conn, "new", "y");

        // Self-supersession would hide a node with no un-supersede path.
        let err = set_superseded(&conn, old.id, old.id).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));

        // Pinned nodes never get superseded (same invariant consolidation honors).
        set_pinned(&conn, old.id, true).unwrap();
        let err = set_superseded(&conn, old.id, new.id).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
        assert!(get_node(&conn, old.id).unwrap().superseded_by.is_none());

        // Unpinned, the normal path still works.
        set_pinned(&conn, old.id, false).unwrap();
        set_superseded(&conn, old.id, new.id).unwrap();
        assert_eq!(get_node(&conn, old.id).unwrap().superseded_by, Some(new.id));
    }

    #[test]
    fn resolve_ref_forms() {
        let conn = conn();
        let node = memory(&conn, "target", "x");

        // Full ULID.
        assert_eq!(resolve_ref(&conn, &node.uid).unwrap().id, node.id);
        // Sigil + 6-char tail (the agent-facing short form).
        let tail6 = &node.uid[node.uid.len() - 6..];
        assert_eq!(
            resolve_ref(&conn, &format!("m:{tail6}")).unwrap().id,
            node.id
        );
        // Bare tail, lowercase.
        assert_eq!(
            resolve_ref(&conn, &tail6.to_lowercase()).unwrap().id,
            node.id
        );
        // Too short.
        assert!(matches!(resolve_ref(&conn, "abc"), Err(Error::Invalid(_))));
        // No match.
        assert!(matches!(
            resolve_ref(&conn, "ZZZZZZ"),
            Err(Error::NotFound(_))
        ));
    }

    #[test]
    fn resolve_ref_ambiguity_and_deleted_exclusion() {
        let conn = conn();
        let now = now_unix();
        // Two crafted uids sharing a tail — ULID alphabet, same last 6 chars.
        for prefix in ["01AAAAAAAAAAAAAAAAAA", "01BBBBBBBBBBBBBBBBBB"] {
            conn.execute(
                "INSERT INTO node (uid, kind, created_at, updated_at) VALUES (?1, 'memory', ?2, ?2)",
                params![format!("{prefix}XYXYXY"), now],
            )
            .unwrap();
        }
        assert!(matches!(
            resolve_ref(&conn, "XYXYXY"),
            Err(Error::AmbiguousRef(_, 2))
        ));
        // Soft-deleting one resolves the ambiguity.
        let id: i64 = conn
            .query_row("SELECT id FROM node WHERE uid LIKE '01AAA%'", [], |r| {
                r.get(0)
            })
            .unwrap();
        soft_delete(&conn, id).unwrap();
        let survivor = resolve_ref(&conn, "XYXYXY").unwrap();
        assert!(survivor.uid.starts_with("01BBB"));
    }

    #[test]
    fn soft_delete_hides_hard_delete_removes() {
        let conn = conn();
        let node = memory(&conn, "doomed", "x");
        soft_delete(&conn, node.id).unwrap();
        assert!(get_node(&conn, node.id).unwrap().deleted_at.is_some());
        assert!(count_by_kind(&conn).unwrap().is_empty());

        hard_delete(&conn, node.id).unwrap();
        assert!(matches!(get_node(&conn, node.id), Err(Error::NotFound(_))));
    }

    #[test]
    fn link_is_idempotent_upsert() {
        let conn = conn();
        let a = memory(&conn, "a", "x");
        let b = memory(&conn, "b", "y");
        link(&conn, a.id, b.id, Rel::About, 1.0).unwrap();
        link(&conn, a.id, b.id, Rel::About, 2.5).unwrap();
        let edges = edges_of(&conn, a.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 2.5);
        assert_eq!(edges[0].rel, Rel::About);
    }

    #[test]
    fn ensure_project_is_idempotent() {
        let conn = conn();
        let p1 = ensure_project(&conn, "/tmp/proj", "proj").unwrap();
        let p2 = ensure_project(&conn, "/tmp/proj", "proj").unwrap();
        assert_eq!(p1.id, p2.id);
        let other = ensure_project(&conn, "/tmp/other", "other").unwrap();
        assert_ne!(p1.id, other.id);
    }

    #[test]
    fn fts_triggers_keep_index_in_sync() {
        let conn = conn();
        let node = memory(&conn, "searchable title", "unique_body_token here");

        let hits = |q: &str| -> i64 {
            conn.query_row(
                "SELECT count(*) FROM node_fts WHERE node_fts MATCH ?1",
                [q],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(hits("unique_body_token"), 1);

        // UPDATE of an FTS column re-indexes.
        conn.execute(
            "UPDATE node SET body = 'replacement_token now' WHERE id = ?1",
            [node.id],
        )
        .unwrap();
        assert_eq!(hits("unique_body_token"), 0);
        assert_eq!(hits("replacement_token"), 1);

        // UPDATE of a non-FTS column must not desync (trigger doesn't fire).
        conn.execute("UPDATE node SET access_count = 5 WHERE id = ?1", [node.id])
            .unwrap();
        assert_eq!(hits("replacement_token"), 1);

        // DELETE removes from the index.
        hard_delete(&conn, node.id).unwrap();
        assert_eq!(hits("replacement_token"), 0);

        // Identifier-friendly tokenizer: snake_case stays one token.
        memory(&conn, "code", "the resolve_ref function");
        assert_eq!(hits("resolve_ref"), 1);
    }
}
