use rusqlite::{params, Connection, OptionalExtension, Row};
use ulid::Ulid;

use crate::error::{Error, Result};
use crate::model::{now_unix, Actor, Edge, Kind, MutationOp, NewNode, Node, Rel};

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
    resolve_ref_inner(conn, reference, false)
}

/// As [`resolve_ref`], but tombstones resolve too.
///
/// For readers that are *about* the deletion rather than about the memory —
/// `mimir audit` is the case: the thing you most want to look up is the one
/// that was deleted, and refusing to resolve it makes the ledger unreachable
/// exactly where it matters. Deliberately separate from `resolve_ref` so no
/// path that edits, marks, or opens a node can pick it up by accident and
/// hand back a ghost.
pub fn resolve_ref_including_deleted(conn: &Connection, reference: &str) -> Result<Node> {
    resolve_ref_inner(conn, reference, true)
}

fn resolve_ref_inner(conn: &Connection, reference: &str, include_deleted: bool) -> Result<Node> {
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
            if include_deleted || node.deleted_at.is_none() {
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
    let live = if include_deleted {
        ""
    } else {
        " AND deleted_at IS NULL"
    };
    let (sql, param) = if tail.len() == 6 {
        (
            format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE substr(uid, -6) = ?1{live} LIMIT 3"
            ),
            up,
        )
    } else {
        (
            format!("SELECT {NODE_COLS} FROM node WHERE uid LIKE ?1{live} LIMIT 3"),
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
    crate::text_clock::keep(conn, id)?;
    conn.execute(
        "UPDATE node SET updated_at = ?2 WHERE id = ?1",
        params![id, now_unix()],
    )?;
    Ok(())
}

/// Read a node's kind and content hash before a mutation destroys or changes
/// it, so the ledger can name *which* value moved without storing it.
fn audit_subject(conn: &Connection, id: i64) -> (Kind, Option<Vec<u8>>) {
    conn.query_row(
        "SELECT kind, content_hash FROM node WHERE id = ?1",
        [id],
        |r| {
            let kind: String = r.get(0)?;
            let hash: Option<Vec<u8>> = r.get(1)?;
            Ok((kind.parse().unwrap_or(Kind::Memory), hash))
        },
    )
    .unwrap_or((Kind::Memory, None))
}

/// How a node is named in a message a person reads: the short id they can
/// paste back into a command, then its title.
fn describe(node: &Node) -> String {
    let id = crate::model::short_uid(node.kind, &node.uid);
    match node.title.as_deref().filter(|t| !t.is_empty()) {
        Some(title) => format!("{id} \"{}\"", crate::format::truncate_chars(title, 60)),
        None => id,
    }
}

/// Soft-delete `id`. `actor` and `reason` go to the mutation ledger — see
/// [`crate::audit`] for why they are arguments rather than ambient state.
///
/// On a memory that consolidation archived, this turns the archival into a
/// deliberate forget: `meta.archived` is cleared, so `remember` stops
/// letting the fact back in, and its prior wordings go. `deleted_at` stays
/// where the archival put it — that is when the memory left the store, and
/// it is the date `recall --as-of` reads.
///
/// The tombstone, the purge and the ledger row land together or not at all
/// ([`crate::db::atomic`]).
pub fn soft_delete(conn: &Connection, id: i64, actor: Actor, reason: Option<&str>) -> Result<()> {
    crate::db::atomic(conn, || {
        let (kind, before) = audit_subject(conn, id);
        let now = now_unix();
        // Bump updated_at too: replication finds tombstones via updated_at, so a
        // delete that didn't advance it would never propagate.
        let mut changed = conn.execute(
            "UPDATE node SET deleted_at = ?2, updated_at = ?2 WHERE id = ?1 AND deleted_at IS NULL",
            params![id, now],
        )?;
        if changed == 0 {
            changed = conn.execute(
                &format!(
                    "UPDATE node SET meta = json_remove(meta, '$.archived'), updated_at = ?2
                     WHERE id = ?1 AND kind = 'memory' AND deleted_at IS NOT NULL AND {}",
                    crate::lifecycle::ARCHIVED_SQL
                ),
                params![id, now],
            )?;
        }
        // Deleting an already-deleted node changes nothing, so it records
        // nothing: a ledger that grows on no-ops stops being a history of what
        // happened.
        if changed > 0 {
            // A deletion that leaves prior wordings in `node_revision` has not
            // deleted anything: recall never reads that table, but `export` and
            // every backup do. Decay archival deliberately does NOT come through
            // here (it updates the row directly with `meta.archived`), so history
            // survives the one case nobody decided.
            crate::revision::purge(conn, id)?;
            crate::audit::try_record_for_kind(
                conn,
                kind,
                id,
                MutationOp::Forget,
                actor,
                reason,
                before.as_deref(),
                None,
            )?;
        }
        Ok(())
    })
}

/// Remove `id` for good: the row, its prior wordings and its review findings
/// go together, and what is left is one ledger row saying it happened.
///
/// Refused while another node still names this one in `superseded_by`. The
/// column is a foreign key, so the DELETE could not succeed anyway; saying
/// which memories hold it is what lets the caller do something about it.
pub fn hard_delete(conn: &Connection, id: i64, actor: Actor, reason: Option<&str>) -> Result<()> {
    crate::db::atomic(conn, || {
        let replaced = superseded_by_this(conn, id)?;
        if !replaced.is_empty() {
            let this = get_node(conn, id)
                .map(|n| crate::model::short_uid(n.kind, &n.uid))
                .unwrap_or_else(|_| format!("node #{id}"));
            let names = replaced.iter().map(describe).collect::<Vec<_>>().join(", ");
            let (count, verb) = match replaced.len() {
                1 => ("1 memory was".to_string(), "points"),
                n => (format!("{n} memories were"), "point"),
            };
            return Err(Error::Invalid(format!(
                "{this} cannot be removed permanently: {count} superseded by it and still \
                 {verb} at it ({names}). Forget it without --hard instead, or remove \
                 {} permanently first.",
                if replaced.len() == 1 {
                    "that one"
                } else {
                    "those"
                },
            )));
        }
        // Read the subject first — after the DELETE there is nothing left to ask.
        let (kind, before) = audit_subject(conn, id);
        let changed = conn.execute("DELETE FROM node WHERE id = ?1", [id])?;
        if changed > 0 {
            crate::revision::purge(conn, id)?;
            // `review` has no foreign key on the node, so a finding about a
            // row that no longer exists would otherwise wait on a person
            // forever with nothing left to decide.
            conn.execute(
                "DELETE FROM review WHERE node_id = ?1 OR other_id = ?1",
                [id],
            )?;
            crate::audit::try_record_for_kind(
                conn,
                kind,
                id,
                MutationOp::HardDelete,
                actor,
                reason,
                before.as_deref(),
                None,
            )?;
        }
        Ok(())
    })
}

/// Forget what `reference` names — the one implementation behind
/// `mimir forget` and the MCP `forget` tool, so the two cannot disagree
/// about what can be forgotten. Returns the node as it stood, for the
/// caller to name in its reply.
///
/// A stored node resolves first, exactly as [`resolve_ref`] would. Only when
/// nothing stored matches does a tombstone resolve, because two states are
/// reachable no other way: an archived memory that should become a
/// deliberate forget, and any tombstone that `hard` should remove.
///
/// Refused: a project (it is not a memory — it is what memories are filed
/// under, and tombstoning it removes none of them), and a soft forget of
/// what is already forgotten, which would report a deletion that did not
/// happen.
pub fn forget(
    conn: &Connection,
    reference: &str,
    hard: bool,
    actor: Actor,
    reason: Option<&str>,
) -> Result<Node> {
    let node = match resolve_ref(conn, reference) {
        Err(Error::NotFound(_)) => resolve_ref_including_deleted(conn, reference)?,
        other => other?,
    };
    if node.kind == Kind::Project {
        return Err(Error::Invalid(format!(
            "{} is a project, not a memory; `forget` removes memories, and forgetting a \
             project would remove none of the memories filed under it",
            describe(&node)
        )));
    }
    if hard {
        hard_delete(conn, node.id, actor, reason)?;
        return Ok(node);
    }
    if node.lifecycle(now_unix()) == crate::lifecycle::Lifecycle::Forgotten {
        return Err(Error::Invalid(format!(
            "{} was already forgotten on {}; `mimir forget --hard` removes it permanently",
            describe(&node),
            crate::format::full_date(node.deleted_at.unwrap_or(node.updated_at))
        )));
    }
    soft_delete(conn, node.id, actor, reason)?;
    Ok(node)
}

/// Every node whose `superseded_by` is `id`, tombstones included: the
/// foreign key holds for those too.
fn superseded_by_this(conn: &Connection, id: i64) -> Result<Vec<Node>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node WHERE superseded_by = ?1 ORDER BY id"
    ))?;
    let nodes = stmt
        .query_map([id], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(nodes)
}

/// Retire the memory `old_id` in favour of `by_id` — the one supersede
/// operation, behind `mimir supersede`, the MCP tool, a review decision and
/// consolidation's merge.
///
/// A supersession is three records that only mean something together: the
/// `superseded_by` column (what hides the old memory from every surface that
/// means "current"), the `supersedes` edge (what `get` draws), and the
/// ledger row (the only date the retirement has — see
/// [`crate::audit::superseded_at`]). They are written in one
/// [`crate::db::atomic`] step, so none of the three can exist without the
/// others.
///
/// There is no way back from superseded, so what would strand a memory is
/// refused up front:
///
/// - either node not being a memory;
/// - a memory superseding itself;
/// - a pinned `old` (pinned means never decayed, never superseded — the
///   invariant consolidation honours);
/// - a `by` that is archived or forgotten: the old memory would be hidden
///   behind something that is itself gone;
/// - a `by` whose own `superseded_by` chain leads back to `old`: a cycle
///   has no live end, so both would vanish.
///
/// Superseding a memory by the replacement it already has changes nothing
/// and records nothing. Re-pointing it at a different replacement is
/// allowed and recorded; the retirement keeps its first date.
pub fn supersede(
    conn: &Connection,
    old_id: i64,
    by_id: i64,
    actor: Actor,
    reason: Option<&str>,
) -> Result<()> {
    crate::db::atomic(conn, || {
        if old_id == by_id {
            return Err(Error::Invalid("a node cannot supersede itself".into()));
        }
        let old = get_node(conn, old_id)?;
        let by = get_node(conn, by_id)?;
        for node in [&old, &by] {
            if node.kind != Kind::Memory {
                return Err(Error::Invalid(format!(
                    "{} is a {}, not a memory; only a memory can supersede or be superseded",
                    describe(node),
                    node.kind
                )));
            }
        }
        if old.superseded_by == Some(by.id) {
            // Already so. The edge is re-asserted because it is idempotent
            // and an older build could set the column without it; the ledger
            // keeps its one row.
            return link(conn, by.id, old.id, Rel::Supersedes, 1.0);
        }
        if old.pinned {
            return Err(Error::Invalid(format!(
                "{} is pinned; pinned memories never get superseded (unpin it first)",
                describe(&old)
            )));
        }
        let state = by.lifecycle(now_unix());
        if state.is_tombstoned() {
            return Err(Error::Invalid(format!(
                "{} is {state}; a deleted memory cannot replace another",
                describe(&by)
            )));
        }
        // An expired replacement hides both: the old one is superseded, the
        // new one is past its date, and recall shows neither.
        if state == crate::lifecycle::Lifecycle::Expired {
            return Err(Error::Invalid(format!(
                "{} is expired; a memory that recall no longer shows cannot replace another",
                describe(&by)
            )));
        }
        if supersession_reaches(conn, &by, old.id)? {
            return Err(Error::Invalid(format!(
                "{} has itself been superseded, and that chain leads to {}; superseding \
                 in both directions would close a loop with no current memory in it",
                describe(&by),
                describe(&old)
            )));
        }
        crate::text_clock::keep(conn, old.id)?;
        conn.execute(
            "UPDATE node SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
            params![old.id, by.id, now_unix()],
        )?;
        link(conn, by.id, old.id, Rel::Supersedes, 1.0)?;
        crate::audit::try_record_for_kind(
            conn,
            old.kind,
            old.id,
            MutationOp::Supersede,
            actor,
            reason,
            old.content_hash.as_deref(),
            by.content_hash.as_deref(),
        )
    })
}

/// Whether following `from`'s `superseded_by` links arrives at `target`.
/// Bounded by the set of nodes already visited: an older build could write
/// a cycle, and `set_superseded` still can, so one may be on disk.
fn supersession_reaches(conn: &Connection, from: &Node, target: i64) -> Result<bool> {
    let mut seen = std::collections::HashSet::from([from.id]);
    let mut next = from.superseded_by;
    while let Some(id) = next {
        if id == target {
            return Ok(true);
        }
        if !seen.insert(id) {
            break;
        }
        next = conn
            .query_row("SELECT superseded_by FROM node WHERE id = ?1", [id], |r| {
                r.get::<_, Option<i64>>(0)
            })
            .optional()?
            .flatten();
    }
    Ok(false)
}

/// The bare column write under [`supersede`]: sets `superseded_by` and the
/// ledger row that dates it, and nothing else — no `supersedes` edge, no
/// check on what `by` is. Callers that mean "retire this memory" want
/// [`supersede`].
///
/// Bumps `updated_at`, which re-sends the row over sync. That does not carry
/// the retirement to a peer: the sync record (`replicate::SyncRecord`) has
/// no `superseded_by` field, so what a peer receives of a supersession is
/// the `supersedes` edge.
///
/// Refuses self-supersession (there is no un-supersede path, so a
/// self-reference would hide the node unrecoverably) and pinned nodes
/// (pinned means never decayed, never superseded — same invariant
/// consolidation honors).
///
/// The ledger row this writes is what dates the supersession:
/// `superseded_by` is a column with no timestamp of its own, and the
/// `updated_at` bump below is overwritten by the next edit.
pub fn set_superseded(
    conn: &Connection,
    id: i64,
    by: i64,
    actor: Actor,
    reason: Option<&str>,
) -> Result<()> {
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
    let (kind, before) = audit_subject(conn, id);
    let (_, after) = audit_subject(conn, by);
    crate::text_clock::keep(conn, id)?;
    conn.execute(
        "UPDATE node SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
        params![id, by, now_unix()],
    )?;
    crate::audit::record_for_kind(
        conn,
        kind,
        id,
        MutationOp::Supersede,
        actor,
        reason,
        before.as_deref(),
        after.as_deref(),
    );
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

/// Every edge touching `node_id`, in either direction.
///
/// A row whose `rel` this binary does not recognise is SKIPPED, not fatal.
/// The previous behaviour turned an unknown rel into a
/// `FromSqlConversionFailure` that failed the whole query, which meant a
/// store written by a newer build could make `get` explode on an older one —
/// and the two do coexist, because the CLI and the MCP server are installed
/// separately and upgrade at different times. Dropping the row degrades one
/// annotation; failing the query loses the whole record.
pub fn edges_of(conn: &Connection, node_id: i64) -> Result<Vec<Edge>> {
    let mut stmt = conn.prepare(
        "SELECT id, src, dst, rel, weight, meta, created_at FROM edge
         WHERE src = ?1 OR dst = ?1",
    )?;
    let edges = stmt
        .query_map([node_id], |row| {
            let rel_text: String = row.get("rel")?;
            let meta_text: String = row.get("meta")?;
            let Ok(rel) = rel_text.parse::<Rel>() else {
                return Ok(None);
            };
            Ok(Some(Edge {
                id: row.get("id")?,
                src: row.get("src")?,
                dst: row.get("dst")?,
                rel,
                weight: row.get("weight")?,
                meta: serde_json::from_str(&meta_text).unwrap_or(serde_json::Value::Null),
                created_at: row.get("created_at")?,
            }))
        })?
        .collect::<rusqlite::Result<Vec<Option<Edge>>>>()?;
    Ok(edges.into_iter().flatten().collect())
}

/// Find or create the project node for a filesystem root.
/// Project identity is its canonical root path (stored in `path`).
///
/// One row per path, also when several processes make first contact with a
/// directory at once, which is the ordinary case: a session start runs three
/// hooks side by side. Looking and then inserting let each of them find
/// nothing and insert its own row; stores written that way hold two or
/// three rows for one directory, of which every later lookup used one and
/// the others stayed empty. So the insert looks again under the write lock,
/// and a lookup that finds several rows takes the oldest, every time.
pub fn ensure_project(conn: &Connection, root: &str, name: &str) -> Result<Node> {
    let find = || -> Result<Option<Node>> {
        Ok(conn
            .query_row(
                &format!(
                    "SELECT {NODE_COLS} FROM node WHERE kind = 'project' AND path = ?1
                     ORDER BY id LIMIT 1"
                ),
                [root],
                row_to_node,
            )
            .optional()?)
    };
    if let Some(node) = find()? {
        return Ok(node);
    }
    crate::db::atomic(conn, || {
        if let Some(node) = find()? {
            return Ok(node);
        }
        let mut new = NewNode::new(Kind::Project);
        new.title = Some(name.to_string());
        new.path = Some(root.to_string());
        insert_node(conn, new)
    })
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
///
/// Bumps `updated_at`, like every setter of a field that syncs: `meta` is on
/// the wire, and replication selects rows by `updated_at`.
pub fn set_fires_when(conn: &Connection, id: i64, phrases: &[String]) -> Result<()> {
    let json = serde_json::to_string(phrases).unwrap_or_else(|_| "[]".into());
    crate::text_clock::keep(conn, id)?;
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.fires_when', json(?2)), updated_at = ?3
         WHERE id = ?1",
        params![id, json, now_unix()],
    )?;
    Ok(())
}

/// Declare when a memory stops being true. Both fields are optional and
/// independent; passing `None` for one leaves any existing value untouched.
///
/// `expires_at` (unix seconds) is machine-checkable and GATES recall once
/// passed — see `Node::is_expired` and the filter in `search`. `resolves_when`
/// is a free-text end condition that deliberately does NOT gate: we can't
/// evaluate "when upstream ships the fix", so the memory keeps surfacing and
/// carries its own falsification condition instead.
///
/// Stored in `meta` (same mechanism as `set_fires_when`) so it replicates —
/// `replicate::SyncRecord` carries `meta` but not arbitrary new columns —
/// and bumps `updated_at` so that it actually does.
pub fn set_expiry(
    conn: &Connection,
    id: i64,
    expires_at: Option<i64>,
    resolves_when: Option<&str>,
) -> Result<()> {
    let now = now_unix();
    crate::text_clock::keep(conn, id)?;
    if let Some(ts) = expires_at {
        conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.expires_at', ?2), updated_at = ?3
             WHERE id = ?1",
            params![id, ts, now],
        )?;
    }
    if let Some(cond) = resolves_when {
        conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.resolves_when', ?2), updated_at = ?3
             WHERE id = ?1",
            params![id, cond, now],
        )?;
    }
    Ok(())
}

/// Declare the *valid time* of the fact — when it began being true in the
/// world and when it stopped — as distinct from when the store learned it.
/// See [`crate::model::Node::valid_from`].
///
/// Both ends optional and independent; `None` leaves an existing value alone,
/// so an author who only learns the end date later can add it without
/// inventing a start. Refuses an inverted interval rather than storing it:
/// "valid from 2024 to 2019" is a typo every time, and silently keeping it
/// would make `valid_at` answer false for all time with nothing to show why —
/// the same "reject, don't mangle" call `parse_expires_in` makes.
///
/// Stored in `meta`, so it replicates without a schema bump.
pub fn set_valid_interval(
    conn: &Connection,
    id: i64,
    valid_from: Option<i64>,
    valid_to: Option<i64>,
) -> Result<()> {
    let existing = get_node(conn, id)?;
    let from = valid_from.or(existing.valid_from());
    let to = valid_to.or(existing.valid_to());
    if let (Some(f), Some(t)) = (from, to) {
        if t <= f {
            return Err(Error::Invalid(format!(
                "valid-to ({t}) must be after valid-from ({f})"
            )));
        }
    }
    if let Some(f) = valid_from {
        conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.valid_from', ?2) WHERE id = ?1",
            params![id, f],
        )?;
    }
    if let Some(t) = valid_to {
        conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.valid_to', ?2) WHERE id = ?1",
            params![id, t],
        )?;
    }
    touch_updated(conn, id)?;
    Ok(())
}

/// Record the author's own certainty. Stored in `meta` for the same reason
/// as `set_expiry`: it replicates without a schema bump (and, like it, bumps
/// `updated_at` so the change is picked up).
/// No gate and no score — see [`crate::model::MemoryConfidence`]. Writing
/// this changes what a reader sees, never what recall returns.
pub fn set_confidence(
    conn: &Connection,
    id: i64,
    confidence: crate::model::MemoryConfidence,
) -> Result<()> {
    crate::text_clock::keep(conn, id)?;
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.confidence', ?2), updated_at = ?3
         WHERE id = ?1",
        params![id, confidence.as_str(), now_unix()],
    )?;
    Ok(())
}

/// Bind an existing keyed (possibly synced-shadow) project to this machine's
/// local path + display name, clearing the shadow marker. Used when a project
/// that first arrived via sync is opened locally.
///
/// Writes only when something would change. Every resolution of a keyed
/// project comes through here, every hook event among them, and an `UPDATE`
/// takes the store's write lock whether or not it changes a row: a session
/// in a keyed project was queueing for that lock on every prompt and every
/// tool call, behind whatever else was writing.
pub fn adopt_project(conn: &Connection, project_id: i64, path: &str, name: &str) -> Result<()> {
    let bound: bool = conn
        .query_row(
            "SELECT path IS ?2 AND title IS ?3 AND json_extract(meta, '$.shadow') IS NULL
             FROM node WHERE id = ?1 AND kind = 'project'",
            params![project_id, path, name],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(false);
    if bound {
        return Ok(());
    }
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

/// Resolve a project by its display title, preferring a locally-bound project
/// (one with a `path`) over a synced shadow (path-less, `meta.shadow`) of the
/// same name — so `reproject --project foo` targets the real local project when
/// a shadow pulled in over sync shares its title. Returns `None` if nothing
/// matches; an error explaining the title must be unique if two
/// equally-preferred projects share it.
pub fn find_project_by_title(conn: &Connection, title: &str) -> Result<Option<Node>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'project' AND deleted_at IS NULL AND title = ?1"
    ))?;
    let mut nodes: Vec<Node> = stmt
        .query_map([title], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    // Prefer path-bearing (locally-bound) projects; drop shadows if any real
    // local project matched.
    if nodes.iter().any(|n| n.path.is_some()) {
        nodes.retain(|n| n.path.is_some());
    }
    match nodes.len() {
        0 => Ok(None),
        1 => Ok(Some(nodes.into_iter().next().unwrap())),
        n => Err(Error::Invalid(format!(
            "{n} projects share the title '{title}', so --project can't pick one; \
             give the target project a unique title first"
        ))),
    }
}

/// Move a memory to another project (`project_id = Some`) or to global scope
/// (`project_id = None`). A memory's project is fixed at `remember` time from
/// the session's working directory; this re-files one that landed in the wrong
/// place. Bumps `updated_at` so the change rides the existing last-write-wins
/// sync: `changes_since` emits the (new) owning project as `portable_key` and
/// `apply_changes` reattaches via `resolve_or_shadow_project`, so it converges
/// across the hub and every machine. Refuses non-memory nodes — files, symbols,
/// and doc/code chunks derive their project from their source, so moving one
/// would only desync it from the graph it belongs to.
///
/// Moving a memory to the project it is already in changes nothing, so it
/// records nothing and leaves `updated_at` alone.
pub fn reproject(
    conn: &Connection,
    node: &Node,
    project_id: Option<i64>,
    actor: Actor,
    reason: Option<&str>,
) -> Result<()> {
    if node.kind != Kind::Memory {
        return Err(Error::Invalid(format!(
            "node #{} is a {}, not a memory; only memories can be reprojected",
            node.id,
            node.kind.as_str()
        )));
    }
    crate::text_clock::keep(conn, node.id)?;
    // `IS NOT` rather than `!=`: global scope is a NULL project.
    let changed = conn.execute(
        "UPDATE node SET project_id = ?2, updated_at = ?3
         WHERE id = ?1 AND project_id IS NOT ?2",
        params![node.id, project_id, now_unix()],
    )?;
    if changed > 0 {
        crate::audit::record_for_kind(
            conn,
            node.kind,
            node.id,
            MutationOp::Reproject,
            actor,
            reason,
            node.content_hash.as_deref(),
            node.content_hash.as_deref(),
        );
    }
    Ok(())
}

/// Move every memory filed under project `from` to project `to`, as one
/// unit: [`reproject_all_created_since`] with no lower bound. Returns how
/// many moved.
pub fn reproject_all(
    conn: &Connection,
    from: i64,
    to: i64,
    actor: Actor,
    reason: Option<&str>,
) -> Result<usize> {
    Ok(reproject_all_created_since(conn, from, to, i64::MIN, actor, reason)?.len())
}

/// Move every memory filed under project `from` and created at or after
/// `since` (unix seconds) to project `to`, as one unit. Each memory moves
/// through [`reproject`], so each gets its ledger row and its `updated_at`
/// bump, and either all of them move or none does. The bump is what
/// carries the move over sync when `to` is a project that syncs; into one
/// that does not, the memories leave sync with the move, as any
/// `reproject` into a local project does. Returns the memories that moved,
/// as they were before the move.
///
/// For a project whose memories turn out to belong to another one as a
/// whole: a git linked worktree's, which belong to the repository's main
/// checkout (`Mimir::projects_for_cwd_moving_memories_home`), from the time
/// that worktree was made. Docs collections, code files and symbols stay
/// where they are; they describe the directory, not the repository.
///
/// Whether there is anything to move is asked first, outside the
/// transaction, of the memories that are not tombstones, through one index
/// (`node_kind_scope`) and the rows it names; finding a project's
/// tombstoned memories means reading every row the project owns, its whole
/// code graph included. When there is something to move, everything in
/// the bound moves, tombstones too. So nothing left means nothing happens
/// and nothing is written; and a project holding *only* forgotten or
/// archived memories keeps them, which no surface shows either way.
pub fn reproject_all_created_since(
    conn: &Connection,
    from: i64,
    to: i64,
    since: i64,
    actor: Actor,
    reason: Option<&str>,
) -> Result<Vec<Node>> {
    if from == to {
        return Ok(Vec::new());
    }
    let any_stored: bool = conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM node
                        WHERE kind = 'memory' AND project_id = ?1 AND deleted_at IS NULL
                          AND created_at >= ?2)",
        params![from, since],
        |r| r.get(0),
    )?;
    if !any_stored {
        return Ok(Vec::new());
    }
    crate::db::atomic(conn, || {
        let mut stmt = conn.prepare(&format!(
            "SELECT {NODE_COLS} FROM node
             WHERE kind = 'memory' AND project_id = ?1 AND created_at >= ?2 ORDER BY id"
        ))?;
        let memories: Vec<Node> = stmt
            .query_map(params![from, since], row_to_node)?
            .collect::<rusqlite::Result<_>>()?;
        for memory in &memories {
            reproject(conn, memory, Some(to), actor, reason)?;
        }
        Ok(memories)
    })
}

/// How many memories that are not tombstones are filed under `project`,
/// split at `since` (unix seconds): `(created at or after it, created
/// before it)`. With no `since`, every one of them counts as before. The
/// question [`reproject_all_created_since`] asks first, with numbers, for
/// a surface that reports what is left (`mimir status` in a linked
/// worktree).
pub fn memories_filed_under(
    conn: &Connection,
    project: i64,
    since: Option<i64>,
) -> Result<(usize, usize)> {
    let (after, before): (i64, i64) = conn.query_row(
        "SELECT COALESCE(SUM(?2 IS NOT NULL AND created_at >= ?2), 0),
                COALESCE(SUM(?2 IS NULL OR created_at < ?2), 0)
         FROM node WHERE kind = 'memory' AND project_id = ?1 AND deleted_at IS NULL",
        params![project, since],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Ok((
        usize::try_from(after).unwrap_or(0),
        usize::try_from(before).unwrap_or(0),
    ))
}

/// `meta` key on a project row naming the project its memories are filed
/// under, when that is another one: a git linked worktree's row holds the
/// `uid` of its main checkout's row. Absent on every other project row.
pub const WORKTREE_OF: &str = "worktree_of";

/// Record (`Some`) or remove (`None`) the home of a linked worktree's
/// project row ([`WORKTREE_OF`]). Reads first and writes only when the
/// stored value differs, since it runs on every resolution, hooks
/// included. A project row's `meta` does not ride sync, so nothing else is
/// bumped.
pub fn set_worktree_home(
    conn: &Connection,
    worktree_project_id: i64,
    home: Option<&Node>,
) -> Result<()> {
    let stored: Option<String> = conn
        .query_row(
            "SELECT json_extract(meta, '$.worktree_of') FROM node
             WHERE id = ?1 AND kind = 'project'",
            [worktree_project_id],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    let wanted = home.map(|h| h.uid.as_str());
    if stored.as_deref() == wanted {
        return Ok(());
    }
    match wanted {
        Some(uid) => conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.worktree_of', ?2)
             WHERE id = ?1 AND kind = 'project'",
            params![worktree_project_id, uid],
        )?,
        None => conn.execute(
            "UPDATE node SET meta = json_remove(meta, '$.worktree_of')
             WHERE id = ?1 AND kind = 'project'",
            [worktree_project_id],
        )?,
    };
    Ok(())
}

/// The project a linked worktree's row says its memories are filed under
/// ([`WORKTREE_OF`]): the live project row with that `uid`. `None` when the
/// row records no home, or the home it records is gone or not a project.
pub fn worktree_home(conn: &Connection, project: &Node) -> Result<Option<Node>> {
    let Some(uid) = project.meta.get(WORKTREE_OF).and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    Ok(conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE uid = ?1 AND kind = 'project' AND deleted_at IS NULL"
            ),
            [uid],
            row_to_node,
        )
        .optional()?)
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
/// removed across both tables.
///
/// Row by row in both tables, whatever session a row belongs to. The
/// store's own timer (`db::spawn_checkpoint_timer`) uses [`prune_sessions`]
/// instead, which does not take state away from a session that is still
/// running.
pub fn prune_session_state(conn: &Connection, cutoff: i64) -> Result<usize> {
    let a = conn.execute("DELETE FROM injection_log WHERE at < ?1", params![cutoff])?;
    let b = conn.execute(
        "DELETE FROM session_state WHERE updated_at < ?1",
        params![cutoff],
    )?;
    Ok(a + b)
}

/// Sessions whose newest row, in either per-session table, is older than
/// the bound parameter `?1`: nothing has been written for them since.
const ENDED_SESSIONS_SQL: &str = "SELECT session_id FROM (
         SELECT session_id, updated_at AS at FROM session_state
         UNION ALL
         SELECT session_id, at FROM injection_log)
     GROUP BY session_id HAVING max(at) < ?1";

/// One prune pass over the per-session tables that does not take state away
/// from a session that is still running. Returns the row count removed.
///
/// `session_state` is pruned by session, not by row: a session's rows go
/// only once the session has ended — its newest row, in either table, is
/// older than `cutoff` (unix seconds). [`prune_session_state`] ages each row
/// on its own, which cuts a session that outlives the window out from under
/// itself: its transcript marker and its brief fire count are written once
/// near the start and read for as long as it runs, so after seven days the
/// context guard estimates from the whole transcript again and the brief
/// forgets it already fired. An injection counts as a sign of life too, so
/// a long session that is still being fed memories keeps its state.
///
/// `injection_log` is pruned by row: a mark older than the window is
/// dropped even for a running session, so a memory that session was shown
/// more than a week ago can be shown to it again.
pub fn prune_sessions(conn: &Connection, cutoff: i64) -> Result<usize> {
    crate::db::atomic(conn, || {
        // State first, while `injection_log` still holds every row that can
        // show a session alive. (The other order gives the same answer — a
        // row older than the cutoff proves nothing either way — but this one
        // does not need the argument.)
        let state = conn.execute(
            &format!("DELETE FROM session_state WHERE session_id IN ({ENDED_SESSIONS_SQL})"),
            params![cutoff],
        )?;
        let injections =
            conn.execute("DELETE FROM injection_log WHERE at < ?1", params![cutoff])?;
        Ok(state + injections)
    })
}

/// Per-session KV working state (v8 `session_state` table). Shared by the
/// context guard (transcript markers, nag counters) and the session brief
/// (fire counts, `brief_shown:<id>` exposure keys); pruned together with
/// `injection_log` above.
pub fn get_session_state(conn: &Connection, session_id: &str, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT value FROM session_state WHERE session_id = ?1 AND key = ?2",
            params![session_id, key],
            |r| r.get(0),
        )
        .optional()?)
}

pub fn set_session_state(
    conn: &Connection,
    session_id: &str,
    key: &str,
    value: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO session_state (session_id, key, value, updated_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        params![session_id, key, value, crate::model::now_unix()],
    )?;
    Ok(())
}

pub fn clear_session_state(conn: &Connection, session_id: &str, key: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM session_state WHERE session_id = ?1 AND key = ?2",
        params![session_id, key],
    )?;
    Ok(())
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
///
/// A pin is a promise about a memory that is current, so pinning one that is
/// superseded, archived or forgotten is refused: nothing would surface it,
/// and the pin would sit on the row contradicting its own state. Unpinning
/// is always allowed, so a pin that predates the retirement or arrived over
/// sync can be taken off.
///
/// Setting the value a node already has changes nothing and records nothing,
/// and an id that names no node is not an error — the same no-op rule as
/// `soft_delete`. A real change bumps `updated_at`: `pinned` is on the sync
/// wire.
pub fn set_pinned(
    conn: &Connection,
    id: i64,
    pinned: bool,
    actor: Actor,
    reason: Option<&str>,
) -> Result<()> {
    crate::db::atomic(conn, || {
        let node = match get_node(conn, id) {
            Ok(node) => node,
            Err(Error::NotFound(_)) => return Ok(()),
            Err(err) => return Err(err),
        };
        if node.pinned == pinned {
            return Ok(());
        }
        if pinned {
            let state = node.lifecycle(now_unix());
            if state == crate::lifecycle::Lifecycle::Superseded {
                let by = node
                    .superseded_by
                    .and_then(|by| get_node(conn, by).ok())
                    .map(|by| describe(&by))
                    .unwrap_or_else(|| "a memory that no longer exists".into());
                return Err(Error::Invalid(format!(
                    "{} is superseded by {by}; a superseded memory is never shown as \
                     current, so a pin on it would hold nothing — pin the replacement instead",
                    describe(&node)
                )));
            }
            if state.is_tombstoned() {
                return Err(Error::Invalid(format!(
                    "{} is {state}; a deleted memory cannot be pinned",
                    describe(&node)
                )));
            }
        }
        crate::text_clock::keep(conn, id)?;
        conn.execute(
            "UPDATE node SET pinned = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, pinned, now_unix()],
        )?;
        crate::audit::try_record_for_kind(
            conn,
            node.kind,
            id,
            if pinned {
                MutationOp::Pin
            } else {
                MutationOp::Unpin
            },
            actor,
            reason,
            node.content_hash.as_deref(),
            node.content_hash.as_deref(),
        )
    })
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
    fn reproject_moves_memory_and_bumps_updated_at() {
        let c = conn();
        let src = ensure_project(&c, "/abs/src", "src").unwrap();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.title = Some("m".into());
        new.project_id = Some(src.id);
        let m = insert_node(&c, new).unwrap();
        assert_eq!(m.project_id, Some(src.id));

        // Move to another project.
        reproject(&c, &m, Some(dst.id), crate::model::Actor::Human, None).unwrap();
        let moved = get_node(&c, m.id).unwrap();
        assert_eq!(moved.project_id, Some(dst.id));
        assert!(
            moved.updated_at >= m.updated_at,
            "updated_at must not regress"
        );

        // Move to global (no project).
        reproject(&c, &m, None, crate::model::Actor::Human, None).unwrap();
        assert_eq!(get_node(&c, m.id).unwrap().project_id, None);
    }

    #[test]
    fn reproject_refuses_non_memory() {
        let c = conn();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        // A project node is not a memory.
        let err = reproject(&c, &dst, None, crate::model::Actor::Human, None).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "got {err:?}");
    }

    fn memory_in(conn: &Connection, project: i64, title: &str) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.title = Some(title.into());
        new.project_id = Some(project);
        insert_node(conn, new).unwrap()
    }

    fn ledger_rows(conn: &Connection) -> i64 {
        conn.query_row("SELECT count(*) FROM mutation", [], |r| r.get(0))
            .unwrap()
    }

    /// Every memory row of the project moves, whatever its lifecycle, each
    /// with a ledger row; what is not a memory, and what belongs to another
    /// project, stays.
    #[test]
    fn reproject_all_moves_every_memory_of_one_project_and_nothing_else() {
        let c = conn();
        let src = ensure_project(&c, "/abs/src", "src").unwrap();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        let other = ensure_project(&c, "/abs/other", "other").unwrap();
        let live = memory_in(&c, src.id, "live");
        let replaced = memory_in(&c, src.id, "replaced");
        supersede(&c, replaced.id, live.id, Actor::Human, None).unwrap();
        let forgotten = memory_in(&c, src.id, "forgotten");
        soft_delete(&c, forgotten.id, Actor::Human, None).unwrap();
        let elsewhere = memory_in(&c, other.id, "elsewhere");
        let mut symbol = NewNode::new(Kind::Symbol);
        symbol.title = Some("a_symbol".into());
        symbol.project_id = Some(src.id);
        let symbol = insert_node(&c, symbol).unwrap();
        c.execute("UPDATE node SET updated_at = 1000", []).unwrap();
        let before = ledger_rows(&c);

        let moved = reproject_all(&c, src.id, dst.id, Actor::System, Some("because")).unwrap();
        assert_eq!(moved, 3);
        for m in [&live, &replaced, &forgotten] {
            let now = get_node(&c, m.id).unwrap();
            assert_eq!(now.project_id, Some(dst.id), "{:?}", m.title);
            assert!(
                now.updated_at > 1000,
                "the bump is what carries it over sync"
            );
            let last = crate::audit::history(&c, m.id).unwrap().pop().unwrap();
            assert_eq!(last.op, MutationOp::Reproject);
            assert_eq!(last.actor, Actor::System);
            assert_eq!(last.reason.as_deref(), Some("because"));
        }
        assert_eq!(ledger_rows(&c), before + 3, "one ledger row per memory");
        let symbol = get_node(&c, symbol.id).unwrap();
        assert_eq!(symbol.project_id, Some(src.id), "the graph does not move");
        assert_eq!(symbol.updated_at, 1000);
        let elsewhere = get_node(&c, elsewhere.id).unwrap();
        assert_eq!(elsewhere.project_id, Some(other.id));
        assert_eq!(elsewhere.updated_at, 1000);
    }

    /// With nothing to move, nothing is written: not a row, not a ledger
    /// line. A project left holding only tombstones counts as nothing to
    /// move (see the function's doc for why the question stops there).
    #[test]
    fn reproject_all_writes_nothing_when_nothing_is_left() {
        let c = conn();
        let src = ensure_project(&c, "/abs/src", "src").unwrap();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        let moved = memory_in(&c, src.id, "moved");
        assert_eq!(
            reproject_all(&c, src.id, dst.id, Actor::System, None).unwrap(),
            1
        );
        c.execute("UPDATE node SET updated_at = 1000", []).unwrap();
        let (rows, changes) = (ledger_rows(&c), c.total_changes());

        assert_eq!(
            reproject_all(&c, src.id, dst.id, Actor::System, None).unwrap(),
            0
        );
        assert_eq!(get_node(&c, moved.id).unwrap().updated_at, 1000);
        assert_eq!(ledger_rows(&c), rows);
        assert_eq!(c.total_changes(), changes, "not one row was written");

        // A project is not moved onto itself.
        assert_eq!(
            reproject_all(&c, dst.id, dst.id, Actor::System, None).unwrap(),
            0
        );
        assert_eq!(c.total_changes(), changes);

        // Only a tombstone left behind: left alone.
        let gone = memory_in(&c, src.id, "gone");
        soft_delete(&c, gone.id, Actor::Human, None).unwrap();
        let changes = c.total_changes();
        assert_eq!(
            reproject_all(&c, src.id, dst.id, Actor::System, None).unwrap(),
            0
        );
        assert_eq!(get_node(&c, gone.id).unwrap().project_id, Some(src.id));
        assert_eq!(c.total_changes(), changes);
    }

    /// The bound is on `created_at`, inclusive, and it holds for the probe
    /// as for the move: memories older than it stay, and a project holding
    /// only those has nothing to move and writes nothing.
    #[test]
    fn reproject_all_created_since_moves_only_what_is_new_enough() {
        let c = conn();
        let src = ensure_project(&c, "/abs/src", "src").unwrap();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        let at = |title: &str, created: i64| {
            let node = memory_in(&c, src.id, title);
            c.execute(
                "UPDATE node SET created_at = ?2 WHERE id = ?1",
                params![node.id, created],
            )
            .unwrap();
            node
        };
        let old = at("older", 99);
        let edge = at("at the bound", 100);
        let new = at("newer", 101);
        assert_eq!(memories_filed_under(&c, src.id, Some(100)).unwrap(), (2, 1));
        assert_eq!(memories_filed_under(&c, src.id, None).unwrap(), (0, 3));
        assert_eq!(memories_filed_under(&c, dst.id, Some(100)).unwrap(), (0, 0));

        let moved = reproject_all_created_since(&c, src.id, dst.id, 100, Actor::System, None)
            .unwrap()
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();
        assert_eq!(moved, [edge.id, new.id]);
        assert_eq!(get_node(&c, old.id).unwrap().project_id, Some(src.id));
        assert_eq!(get_node(&c, edge.id).unwrap().project_id, Some(dst.id));
        assert_eq!(memories_filed_under(&c, src.id, Some(100)).unwrap(), (0, 1));

        let changes = c.total_changes();
        assert!(
            reproject_all_created_since(&c, src.id, dst.id, 100, Actor::System, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(c.total_changes(), changes, "only an older one is left");
    }

    /// A worktree row's home is recorded by `uid`, removed on request, and
    /// read back only while it names a live project; a value that is already
    /// stored is not written again.
    #[test]
    fn a_worktree_rows_home_is_recorded_and_read_back() {
        let c = conn();
        let home = ensure_project(&c, "/abs/main", "main").unwrap();
        let wt = ensure_project(&c, "/abs/wt", "wt").unwrap();
        assert!(worktree_home(&c, &wt).unwrap().is_none());

        set_worktree_home(&c, wt.id, Some(&home)).unwrap();
        let row = get_node(&c, wt.id).unwrap();
        assert_eq!(
            row.meta.get(WORKTREE_OF).and_then(|v| v.as_str()),
            Some(home.uid.as_str())
        );
        assert_eq!(
            worktree_home(&c, &row).unwrap().map(|p| p.id),
            Some(home.id)
        );
        let changes = c.total_changes();
        set_worktree_home(&c, wt.id, Some(&home)).unwrap();
        assert_eq!(c.total_changes(), changes, "read first, nothing to write");

        // A home that is gone, or a uid that is not a project's.
        let memory = memory_in(&c, home.id, "a memory");
        c.execute(
            "UPDATE node SET meta = json_set(meta, '$.worktree_of', ?2) WHERE id = ?1",
            params![wt.id, memory.uid],
        )
        .unwrap();
        assert!(worktree_home(&c, &get_node(&c, wt.id).unwrap())
            .unwrap()
            .is_none());
        c.execute(
            "UPDATE node SET meta = json_set(meta, '$.worktree_of', 'NO-SUCH-UID') WHERE id = ?1",
            [wt.id],
        )
        .unwrap();
        assert!(worktree_home(&c, &get_node(&c, wt.id).unwrap())
            .unwrap()
            .is_none());

        set_worktree_home(&c, wt.id, None).unwrap();
        assert_eq!(get_node(&c, wt.id).unwrap().meta.get(WORKTREE_OF), None);
        let changes = c.total_changes();
        set_worktree_home(&c, wt.id, None).unwrap();
        assert_eq!(c.total_changes(), changes);
    }

    /// All of them or none: a failure on the second memory undoes the first.
    #[test]
    fn reproject_all_moves_all_or_nothing() {
        let c = conn();
        let src = ensure_project(&c, "/abs/src", "src").unwrap();
        let dst = ensure_project(&c, "/abs/dst", "dst").unwrap();
        let first = memory_in(&c, src.id, "first");
        let second = memory_in(&c, src.id, "second");
        c.execute_batch(&format!(
            "CREATE TRIGGER refuse_second BEFORE UPDATE OF project_id ON node
             WHEN old.id = {} BEGIN SELECT RAISE(ABORT, 'refused'); END;",
            second.id
        ))
        .unwrap();
        let before = ledger_rows(&c);
        assert!(reproject_all(&c, src.id, dst.id, Actor::System, None).is_err());
        assert_eq!(get_node(&c, first.id).unwrap().project_id, Some(src.id));
        assert_eq!(get_node(&c, second.id).unwrap().project_id, Some(src.id));
        assert_eq!(ledger_rows(&c), before);
    }

    #[test]
    fn find_project_by_title_prefers_local_over_shadow() {
        let c = conn();
        // A synced shadow (path-less) and a local project share the title.
        let shadow = resolve_or_shadow_project(&c, "git:github.com/o/dup").unwrap();
        let local = ensure_project(&c, "/abs/dup", "dup").unwrap();
        // The shadow's title is derived from the key tail ("dup"), same as local.
        let found = find_project_by_title(&c, "dup").unwrap().unwrap();
        assert_eq!(found.id, local.id, "should pick the path-bearing local");
        assert_ne!(found.id, shadow);
        // No match returns None, not an error.
        assert!(find_project_by_title(&c, "nope").unwrap().is_none());
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
        soft_delete(&conn, c.id, crate::model::Actor::System, None).unwrap(); // still fetchable by id, just flagged

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
        let err =
            set_superseded(&conn, old.id, old.id, crate::model::Actor::System, None).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));

        // Pinned nodes never get superseded (same invariant consolidation honors).
        set_pinned(&conn, old.id, true, crate::model::Actor::System, None).unwrap();
        let err =
            set_superseded(&conn, old.id, new.id, crate::model::Actor::System, None).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
        assert!(get_node(&conn, old.id).unwrap().superseded_by.is_none());

        // Unpinned, the normal path still works.
        set_pinned(&conn, old.id, false, crate::model::Actor::System, None).unwrap();
        set_superseded(&conn, old.id, new.id, crate::model::Actor::System, None).unwrap();
        assert_eq!(get_node(&conn, old.id).unwrap().superseded_by, Some(new.id));
    }

    fn ledger(conn: &Connection, id: i64) -> Vec<MutationOp> {
        crate::audit::history(conn, id)
            .unwrap()
            .into_iter()
            .map(|m| m.op)
            .collect()
    }

    /// A memory as `remember` stores it: with the content hash the ledger
    /// and the duplicate check read.
    fn remembered(conn: &Connection, body: &str) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some(body.into());
        new.body = Some(body.into());
        new.content_hash = Some(crate::memory::content_hash(body));
        insert_node(conn, new).unwrap()
    }

    fn backdate(conn: &Connection, id: i64, secs: i64) -> i64 {
        conn.execute(
            "UPDATE node SET created_at = created_at - ?2, updated_at = updated_at - ?2
             WHERE id = ?1",
            params![id, secs],
        )
        .unwrap();
        get_node(conn, id).unwrap().updated_at
    }

    /// The statement `consolidate`'s decay pass runs.
    fn archive(conn: &Connection, id: i64, at: i64) {
        conn.execute(
            "UPDATE node SET deleted_at = ?2, meta = json_set(meta, '$.archived', 1)
             WHERE id = ?1",
            params![id, at],
        )
        .unwrap();
    }

    /// Make every insert into `table` fail, to stand in for the ledger or
    /// the edge write breaking halfway through an operation.
    fn break_inserts(conn: &Connection, table: &str) {
        conn.execute_batch(&format!(
            "CREATE TEMP TRIGGER break_{table} BEFORE INSERT ON main.{table}
             BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END;"
        ))
        .unwrap();
    }

    #[test]
    fn a_pin_that_changes_nothing_records_nothing() {
        let conn = conn();
        let node = remembered(&conn, "pin me once");
        set_pinned(&conn, node.id, true, Actor::Human, None).unwrap();
        set_pinned(&conn, node.id, true, Actor::Human, None).unwrap();
        assert_eq!(ledger(&conn, node.id), vec![MutationOp::Pin]);
        set_pinned(&conn, node.id, false, Actor::Human, None).unwrap();
        set_pinned(&conn, node.id, false, Actor::Human, None).unwrap();
        assert_eq!(
            ledger(&conn, node.id),
            vec![MutationOp::Pin, MutationOp::Unpin]
        );

        // An id that names nothing has no history to add to.
        set_pinned(&conn, 999_999, true, Actor::Human, None).unwrap();
        assert!(ledger(&conn, 999_999).is_empty());
    }

    /// `pinned` and `meta` are on the sync wire, and replication selects on
    /// `updated_at`: a setter that leaves it alone changes a field no peer
    /// will ever hear about.
    #[test]
    fn setting_a_synced_field_marks_the_row_changed() {
        type Setter = fn(&Connection, i64);
        let setters: [(&str, Setter); 5] = [
            ("set_pinned", |c, id| {
                set_pinned(c, id, true, Actor::Human, None).unwrap()
            }),
            ("set_expiry", |c, id| {
                set_expiry(c, id, Some(now_unix() + 86_400), Some("when it ships")).unwrap()
            }),
            ("set_fires_when", |c, id| {
                set_fires_when(c, id, &["release checklist".into()]).unwrap()
            }),
            ("set_confidence", |c, id| {
                set_confidence(c, id, crate::model::MemoryConfidence::Unsure).unwrap()
            }),
            ("set_anchors", |c, id| {
                crate::anchors::set_anchors(c, id, &["store.rs".into()]).unwrap()
            }),
        ];
        for (name, set) in setters {
            let conn = conn();
            let node = remembered(&conn, "an old global memory");
            let before = backdate(&conn, node.id, 30 * 86_400);
            // Where a peer's cursor stands: everything up to now is synced.
            let cursor = now_unix() - 1;
            assert!(crate::replicate::changes_since(&conn, cursor)
                .unwrap()
                .nodes
                .is_empty());

            set(&conn, node.id);

            assert!(
                get_node(&conn, node.id).unwrap().updated_at > before,
                "{name} must bump updated_at"
            );
            let sent: Vec<String> = crate::replicate::changes_since(&conn, cursor)
                .unwrap()
                .nodes
                .into_iter()
                .map(|n| n.uid)
                .collect();
            assert_eq!(sent, vec![node.uid.clone()], "{name} must replicate");
        }
    }

    #[test]
    fn pinning_respects_the_lifecycle() {
        let conn = conn();
        let old = remembered(&conn, "deploys go to eu-west-1");
        let new = remembered(&conn, "deploys go to eu-north-1");
        set_superseded(&conn, old.id, new.id, Actor::Human, None).unwrap();
        let err = set_pinned(&conn, old.id, true, Actor::Human, None).unwrap_err();
        let text = err.to_string();
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
        assert!(text.contains("superseded"), "{text}");
        assert!(
            text.contains(&crate::model::short_uid(Kind::Memory, &new.uid)),
            "must name the replacement: {text}"
        );
        assert!(!get_node(&conn, old.id).unwrap().pinned);

        let gone = remembered(&conn, "a fact someone deleted");
        soft_delete(&conn, gone.id, Actor::Human, None).unwrap();
        let text = set_pinned(&conn, gone.id, true, Actor::Human, None)
            .unwrap_err()
            .to_string();
        assert!(text.contains("forgotten"), "{text}");

        let decayed = remembered(&conn, "a fact that decayed");
        archive(&conn, decayed.id, now_unix());
        let text = set_pinned(&conn, decayed.id, true, Actor::Human, None)
            .unwrap_err()
            .to_string();
        assert!(text.contains("archived"), "{text}");

        // Unpinning is always allowed: a pin that predates the retirement
        // (or arrived over sync) must be removable.
        conn.execute("UPDATE node SET pinned = 1 WHERE id = ?1", [old.id])
            .unwrap();
        set_pinned(&conn, old.id, false, Actor::Human, None).unwrap();
        assert!(!get_node(&conn, old.id).unwrap().pinned);

        // Expired is not retired by anyone's decision; pinning it is allowed.
        let lapsed = remembered(&conn, "the cert is valid until spring");
        set_expiry(&conn, lapsed.id, Some(now_unix() - 60), None).unwrap();
        set_pinned(&conn, lapsed.id, true, Actor::Human, None).unwrap();
        assert!(get_node(&conn, lapsed.id).unwrap().pinned);
    }

    /// Several processes meeting a directory for the first time at the same
    /// moment register it once. (Each of these threads has a connection of
    /// its own to one store file, as each hook process does.)
    #[test]
    fn first_contact_from_several_processes_makes_one_project_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mimir.db");
        drop(db::open(&path).unwrap());
        for round in 0..25 {
            let root = format!("/home/u/project-{round}");
            let start = std::sync::Arc::new(std::sync::Barrier::new(6));
            let ids: Vec<i64> = (0..6)
                .map(|_| {
                    let (path, root, start) = (path.clone(), root.clone(), start.clone());
                    std::thread::spawn(move || {
                        let conn = db::open(&path).unwrap();
                        start.wait();
                        ensure_project(&conn, &root, "project").unwrap().id
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect();
            let conn = db::open(&path).unwrap();
            let rows: i64 = conn
                .query_row(
                    "SELECT count(*) FROM node WHERE kind = 'project' AND path = ?1",
                    [&root],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(rows, 1, "round {round}: {rows} rows for one directory");
            assert!(
                ids.iter().all(|id| *id == ids[0]),
                "round {round}: callers were handed different rows: {ids:?}"
            );
        }
    }

    /// A store that already holds several rows for one directory (written
    /// before the insert was guarded) resolves to the oldest of them, always
    /// the same one.
    #[test]
    fn a_directory_with_several_rows_resolves_to_the_oldest() {
        let c = db::open_in_memory().unwrap();
        let first = ensure_project(&c, "/home/u/twice", "twice").unwrap();
        let mut duplicate = NewNode::new(Kind::Project);
        duplicate.title = Some("twice".into());
        duplicate.path = Some("/home/u/twice".into());
        let second = insert_node(&c, duplicate).unwrap();
        assert!(second.id > first.id);
        for _ in 0..3 {
            assert_eq!(
                ensure_project(&c, "/home/u/twice", "twice").unwrap().id,
                first.id
            );
        }
    }

    /// Resolving a keyed project adopts it, on every command and every hook
    /// event. When the row is already bound to this path under this name
    /// that is a read: an `UPDATE` would take the write lock for nothing,
    /// and a hook would queue for it behind whatever else was writing.
    #[test]
    fn adopting_a_project_already_bound_writes_nothing() {
        let c = db::open_in_memory().unwrap();
        let shadow = resolve_or_shadow_project(&c, "git:github.com/o/r").unwrap();

        adopt_project(&c, shadow, "/home/u/r", "r").unwrap();
        let bound = get_node(&c, shadow).unwrap();
        assert_eq!(bound.path.as_deref(), Some("/home/u/r"));
        assert!(bound.meta.get("shadow").is_none());

        let before = c.total_changes();
        adopt_project(&c, shadow, "/home/u/r", "r").unwrap();
        assert_eq!(
            c.total_changes(),
            before,
            "nothing to change, nothing written"
        );

        // A move or a rename is still written.
        adopt_project(&c, shadow, "/home/u/elsewhere", "r").unwrap();
        assert_eq!(
            get_node(&c, shadow).unwrap().path.as_deref(),
            Some("/home/u/elsewhere")
        );
        adopt_project(&c, shadow, "/home/u/elsewhere", "renamed").unwrap();
        assert_eq!(
            get_node(&c, shadow).unwrap().title.as_deref(),
            Some("renamed")
        );
        // And a row that is not a project is left alone, as before.
        adopt_project(&c, i64::MAX, "/nowhere", "nothing").unwrap();
    }

    #[test]
    fn reprojecting_to_where_it_already_is_records_nothing() {
        let c = conn();
        let proj = ensure_project(&c, "/abs/here", "here").unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.title = Some("m".into());
        new.project_id = Some(proj.id);
        let m = insert_node(&c, new).unwrap();
        let before = backdate(&c, m.id, 3600);
        let m = get_node(&c, m.id).unwrap();

        reproject(&c, &m, Some(proj.id), Actor::Human, None).unwrap();
        assert!(ledger(&c, m.id).is_empty());
        assert_eq!(get_node(&c, m.id).unwrap().updated_at, before);

        // Global to global is the same no-op.
        reproject(&c, &m, None, Actor::Human, None).unwrap();
        let global = get_node(&c, m.id).unwrap();
        reproject(&c, &global, None, Actor::Human, None).unwrap();
        assert_eq!(ledger(&c, m.id), vec![MutationOp::Reproject]);
    }

    /// Archival is the one tombstone nobody decided, so `remember` lets the
    /// fact back in. Forgetting it afterwards has to take that away again.
    #[test]
    fn forgetting_an_archived_memory_makes_it_forgotten() {
        let conn = conn();
        let node = remembered(&conn, "cargo nextest needs a separate install");
        crate::revision::snapshot(&conn, &node);
        let archived_at = now_unix() - 5 * 86_400;
        archive(&conn, node.id, archived_at);
        backdate(&conn, node.id, 5 * 86_400);
        let now = now_unix();
        assert_eq!(
            get_node(&conn, node.id).unwrap().lifecycle(now),
            crate::lifecycle::Lifecycle::Archived
        );

        soft_delete(&conn, node.id, Actor::Human, Some("was never true")).unwrap();

        let gone = get_node(&conn, node.id).unwrap();
        assert_eq!(gone.lifecycle(now), crate::lifecycle::Lifecycle::Forgotten);
        assert!(gone.meta.get("archived").is_none());
        assert_eq!(
            gone.deleted_at,
            Some(archived_at),
            "it left the store when it was archived; `--as-of` reads that date"
        );
        assert!(gone.updated_at >= now, "the change must replicate");
        assert!(crate::revision::history(&conn, node.id).unwrap().is_empty());
        let h = crate::audit::history(&conn, node.id).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].op, MutationOp::Forget);
        assert_eq!(h[0].reason.as_deref(), Some("was never true"));

        // Forgetting what is already forgotten is still a no-op.
        soft_delete(&conn, node.id, Actor::Human, None).unwrap();
        assert_eq!(crate::audit::history(&conn, node.id).unwrap().len(), 1);
    }

    /// A forget that keeps the prior wordings, or leaves no ledger row, has
    /// not done what it reports.
    #[test]
    fn a_forget_lands_whole_or_not_at_all() {
        let conn = conn();
        let node = remembered(&conn, "staging rotates credentials on Sundays");
        break_inserts(&conn, "mutation");

        assert!(soft_delete(&conn, node.id, Actor::Human, None).is_err());
        assert!(
            get_node(&conn, node.id).unwrap().deleted_at.is_none(),
            "the tombstone must roll back with the ledger row"
        );
    }

    #[test]
    fn hard_delete_refuses_a_node_others_were_superseded_by() {
        let conn = conn();
        let old = remembered(&conn, "the retry budget is 3");
        let new = remembered(&conn, "the retry budget is 5");
        set_superseded(&conn, old.id, new.id, Actor::Human, None).unwrap();

        let err = hard_delete(&conn, new.id, Actor::Human, None).unwrap_err();
        let text = err.to_string();
        assert!(matches!(err, Error::Invalid(_)), "{err:?}");
        assert!(
            text.contains(&crate::model::short_uid(Kind::Memory, &old.uid)),
            "must name what was superseded by it: {text}"
        );
        assert!(get_node(&conn, new.id).is_ok(), "nothing was deleted");
        assert!(ledger(&conn, new.id).is_empty());

        // Once nothing points at it, it goes.
        hard_delete(&conn, old.id, Actor::Human, None).unwrap();
        hard_delete(&conn, new.id, Actor::Human, None).unwrap();
        assert!(matches!(get_node(&conn, new.id), Err(Error::NotFound(_))));
    }

    #[test]
    fn hard_delete_takes_the_review_rows_with_it() {
        let conn = conn();
        let a = remembered(&conn, "the cron runs at 03:00");
        let b = remembered(&conn, "the cron runs at 04:00");
        let c = remembered(&conn, "an unrelated memory");
        use crate::review::{enqueue, Finding};
        enqueue(&conn, Finding::Contradiction, a.id, Some(b.id), None).unwrap();
        enqueue(&conn, Finding::Contradiction, c.id, Some(a.id), None).unwrap();
        enqueue(&conn, Finding::Unresolved, c.id, None, None).unwrap();

        hard_delete(&conn, a.id, Actor::Human, None).unwrap();

        let left: Vec<i64> = conn
            .prepare("SELECT node_id FROM review")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(left, vec![c.id], "only the finding that never named it");
    }

    #[test]
    fn supersede_writes_the_column_the_edge_and_the_dated_row_together() {
        let conn = conn();
        let old = remembered(&conn, "deploys go to eu-west-1");
        let new = remembered(&conn, "deploys go to eu-north-1");

        supersede(&conn, old.id, new.id, Actor::Human, Some("region move")).unwrap();

        assert_eq!(get_node(&conn, old.id).unwrap().superseded_by, Some(new.id));
        let edges = edges_of(&conn, old.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(
            (edges[0].src, edges[0].dst, edges[0].rel),
            (new.id, old.id, Rel::Supersedes)
        );
        let h = crate::audit::history(&conn, old.id).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].op, MutationOp::Supersede);
        assert_eq!(h[0].reason.as_deref(), Some("region move"));
        assert_eq!(h[0].before_hash, old.content_hash);
        assert_eq!(h[0].after_hash, new.content_hash);
        assert!(crate::audit::superseded_at(&conn, old.id)
            .unwrap()
            .is_some());

        // Saying it again changes nothing and records nothing.
        supersede(&conn, old.id, new.id, Actor::Agent, None).unwrap();
        assert_eq!(crate::audit::history(&conn, old.id).unwrap().len(), 1);
        assert_eq!(edges_of(&conn, old.id).unwrap().len(), 1);
    }

    /// The failure four call sites shared: column first, edge second, no
    /// transaction. An error between the two left a memory retired with no
    /// edge to say by what.
    #[test]
    fn a_supersede_that_fails_halfway_leaves_nothing_behind() {
        for broken in ["edge", "mutation"] {
            let conn = conn();
            let old = remembered(&conn, "the retry budget is 3");
            let new = remembered(&conn, "the retry budget is 5");
            break_inserts(&conn, broken);

            assert!(supersede(&conn, old.id, new.id, Actor::Human, None).is_err());
            assert!(
                get_node(&conn, old.id).unwrap().superseded_by.is_none(),
                "a failed {broken} write must take the column back with it"
            );
            assert!(edges_of(&conn, old.id).unwrap().is_empty());
        }
    }

    /// Inside a caller's transaction (consolidation, a review decision) a
    /// failed supersede undoes itself and leaves the caller's work standing.
    #[test]
    fn supersede_nests_inside_a_callers_transaction() {
        let conn = conn();
        let old = remembered(&conn, "the retry budget is 3");
        let new = remembered(&conn, "the retry budget is 5");
        let tx =
            rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)
                .unwrap();
        let earlier = remembered(&tx, "written earlier in the same transaction");
        set_pinned(&tx, old.id, true, Actor::Human, None).unwrap();
        assert!(supersede(&tx, old.id, new.id, Actor::Human, None).is_err());
        set_pinned(&tx, old.id, false, Actor::Human, None).unwrap();
        supersede(&tx, old.id, new.id, Actor::Human, None).unwrap();
        tx.commit().unwrap();

        assert!(get_node(&conn, earlier.id).is_ok());
        assert_eq!(get_node(&conn, old.id).unwrap().superseded_by, Some(new.id));
    }

    /// An expired replacement strands the pair as surely as a deleted one:
    /// the old memory is superseded, the new one is past its date, and
    /// recall shows neither.
    #[test]
    fn supersede_refuses_an_expired_replacement() {
        let conn = db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.body = Some("the freeze is on until friday".into());
        let old = insert_node(&conn, new).unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.body = Some("the freeze was extended one day".into());
        let by = insert_node(&conn, new).unwrap();
        set_expiry(&conn, by.id, Some(now_unix() - 60), None).unwrap();

        let err = supersede(&conn, old.id, by.id, Actor::Human, None)
            .expect_err("an expired replacement must be refused")
            .to_string();

        assert!(err.contains("is expired"), "{err}");
        assert_eq!(get_node(&conn, old.id).unwrap().superseded_by, None);
    }

    #[test]
    fn supersede_refuses_what_would_strand_a_memory() {
        let conn = conn();
        let a = remembered(&conn, "first wording of the rule");
        let b = remembered(&conn, "second wording of the rule");
        let c = remembered(&conn, "third wording of the rule");
        let refused = |old: i64, by: i64| -> String {
            let err = supersede(&conn, old, by, Actor::Human, None).unwrap_err();
            assert!(matches!(err, Error::Invalid(_)), "{err:?}");
            err.to_string()
        };

        assert!(refused(a.id, a.id).contains("itself"));

        // Only memories.
        let project = ensure_project(&conn, "/abs/p", "p").unwrap();
        assert!(refused(project.id, a.id).contains("not a memory"));
        assert!(refused(a.id, project.id).contains("not a memory"));

        // A pinned memory is never superseded.
        set_pinned(&conn, a.id, true, Actor::Human, None).unwrap();
        assert!(refused(a.id, b.id).contains("pinned"));
        set_pinned(&conn, a.id, false, Actor::Human, None).unwrap();

        // A replacement that is itself gone would hide `a` behind nothing.
        let gone = remembered(&conn, "a wording someone deleted");
        soft_delete(&conn, gone.id, Actor::Human, None).unwrap();
        assert!(refused(a.id, gone.id).contains("forgotten"));
        let decayed = remembered(&conn, "a wording that decayed");
        archive(&conn, decayed.id, now_unix());
        assert!(refused(a.id, decayed.id).contains("archived"));

        // a → b → c is a chain; closing it into a loop is refused at any
        // distance, and names both ends.
        supersede(&conn, a.id, b.id, Actor::Human, None).unwrap();
        supersede(&conn, b.id, c.id, Actor::Human, None).unwrap();
        let text = refused(b.id, a.id);
        assert!(text.contains("loop"), "{text}");
        let text = refused(c.id, a.id);
        assert!(
            text.contains(&crate::model::short_uid(Kind::Memory, &a.uid))
                && text.contains(&crate::model::short_uid(Kind::Memory, &c.uid)),
            "{text}"
        );
        assert!(get_node(&conn, c.id).unwrap().superseded_by.is_none());

        // A cycle written before the guard existed must end the walk, not
        // spin it.
        let x = remembered(&conn, "ping");
        let y = remembered(&conn, "pong");
        set_superseded(&conn, x.id, y.id, Actor::System, None).unwrap();
        set_superseded(&conn, y.id, x.id, Actor::System, None).unwrap();
        let z = remembered(&conn, "pang");
        supersede(&conn, z.id, x.id, Actor::Human, None).unwrap();
    }

    #[test]
    fn forget_reaches_every_state_it_should() {
        let conn = conn();
        let short = |n: &Node| crate::model::short_uid(n.kind, &n.uid);
        let now = now_unix();

        let node = remembered(&conn, "a fact to remove entirely");
        let decayed = remembered(&conn, "a fact that decayed while idle");

        // A tombstone can be removed for good.
        forget(&conn, &short(&node), false, Actor::Human, None).unwrap();
        assert!(resolve_ref(&conn, &short(&node)).is_err());
        forget(&conn, &short(&node), true, Actor::Human, None).unwrap();
        assert!(matches!(get_node(&conn, node.id), Err(Error::NotFound(_))));

        // An archived memory becomes a deliberate forget.
        archive(&conn, decayed.id, now - 86_400);
        let was = forget(&conn, &short(&decayed), false, Actor::Human, None).unwrap();
        assert_eq!(was.lifecycle(now), crate::lifecycle::Lifecycle::Archived);
        assert_eq!(
            get_node(&conn, decayed.id).unwrap().lifecycle(now),
            crate::lifecycle::Lifecycle::Forgotten
        );

        // Forgetting it again would report a deletion that did not happen.
        let err = forget(&conn, &short(&decayed), false, Actor::Human, None).unwrap_err();
        assert!(err.to_string().contains("already forgotten"), "{err}");
        assert_eq!(ledger(&conn, decayed.id), vec![MutationOp::Forget]);

        // A project is not a memory.
        let project = ensure_project(&conn, "/abs/p", "p").unwrap();
        for hard in [false, true] {
            let err = forget(&conn, &short(&project), hard, Actor::Human, None).unwrap_err();
            assert!(err.to_string().contains("not a memory"), "{err}");
        }
        assert!(get_node(&conn, project.id).unwrap().deleted_at.is_none());

        // Nothing at all still reads as not found.
        assert!(matches!(
            forget(&conn, "ZZZZZZ", false, Actor::Human, None),
            Err(Error::NotFound(_))
        ));
    }

    /// A stored node and a tombstone can share a short id. `forget` must
    /// keep resolving to the stored one, as it did before tombstones could
    /// resolve at all.
    #[test]
    fn forget_prefers_a_stored_node_over_a_tombstone_with_the_same_tail() {
        let conn = conn();
        let now = now_unix();
        for (prefix, deleted) in [
            ("01AAAAAAAAAAAAAAAAAA", Some(now)),
            ("01BBBBBBBBBBBBBBBBBB", None),
        ] {
            conn.execute(
                "INSERT INTO node (uid, kind, created_at, updated_at, deleted_at)
                 VALUES (?1, 'memory', ?2, ?2, ?3)",
                params![format!("{prefix}XYXYXY"), now, deleted],
            )
            .unwrap();
        }
        let was = forget(&conn, "m:XYXYXY", false, Actor::Human, None).unwrap();
        assert!(was.uid.starts_with("01BBB"));
    }

    #[test]
    fn a_session_keeps_its_state_for_as_long_as_it_is_running() {
        let conn = conn();
        let n = memory(&conn, "gamma", "z");
        let now = now_unix();
        let old = now - 10 * 86_400;
        let state = |session: &str, key: &str, at: i64| {
            conn.execute(
                "INSERT INTO session_state (session_id, key, value, updated_at)
                 VALUES (?1, ?2, 'v', ?3)",
                params![session, key, at],
            )
            .unwrap();
        };
        let injected = |session: &str, at: i64| {
            conn.execute(
                "INSERT INTO injection_log (session_id, node_id, at) VALUES (?1, ?2, ?3)",
                params![session, n.id, at],
            )
            .unwrap();
        };
        // A long session: marker written ten days ago, still firing briefs.
        state("long", "transcript_marker", old);
        state("long", "brief_fire_count", now);
        injected("long", old);
        // A long session whose only recent sign of life is an injection.
        state("injecting", "transcript_marker", old);
        injected("injecting", now);
        // A session that ended ten days ago.
        state("ended", "transcript_marker", old);
        state("ended", "brief_fire_count", old);
        injected("ended", old);

        let removed = prune_sessions(&conn, now - 7 * 86_400).unwrap();

        assert_eq!(
            removed, 4,
            "the ended session's three rows, and the long session's old injection"
        );
        for session in ["long", "injecting"] {
            assert!(
                get_session_state(&conn, session, "transcript_marker")
                    .unwrap()
                    .is_some(),
                "{session} lost its transcript marker while still running"
            );
        }
        assert!(get_session_state(&conn, "ended", "transcript_marker")
            .unwrap()
            .is_none());
        // An injection mark lapses with the window whatever the session is
        // doing, so the memory can be shown to a long session again.
        assert!(!injection_seen(&conn, "long", n.id).unwrap());
        assert!(injection_seen(&conn, "injecting", n.id).unwrap());
        assert!(!injection_seen(&conn, "ended", n.id).unwrap());
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
        soft_delete(&conn, id, crate::model::Actor::System, None).unwrap();
        let survivor = resolve_ref(&conn, "XYXYXY").unwrap();
        assert!(survivor.uid.starts_with("01BBB"));
    }

    #[test]
    fn soft_delete_hides_hard_delete_removes() {
        let conn = conn();
        let node = memory(&conn, "doomed", "x");
        soft_delete(&conn, node.id, crate::model::Actor::System, None).unwrap();
        assert!(get_node(&conn, node.id).unwrap().deleted_at.is_some());
        assert!(count_by_kind(&conn).unwrap().is_empty());

        hard_delete(&conn, node.id, crate::model::Actor::System, None).unwrap();
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
        hard_delete(&conn, node.id, crate::model::Actor::System, None).unwrap();
        assert_eq!(hits("replacement_token"), 0);

        // Identifier-friendly tokenizer: snake_case stays one token.
        memory(&conn, "code", "the resolve_ref function");
        assert_eq!(hits("resolve_ref"), 1);
    }
}
