//! LLM-free memory consolidation: four passes, none destructive.
//!
//! 1. near-duplicate merge (cosine ≥ 0.92, same type+scope) → the older
//!    node is superseded, never deleted
//! 2. contradiction flagging (high similarity + negation mismatch) —
//!    reported to the human, never auto-resolved
//! 3. cluster distillation (cosine ≥ 0.80, clusters ≥ 5) → extractive
//!    summary node with `summarizes` edges
//! 4. decay archival (effective strength < 0.05, untouched > 90 days) →
//!    soft-deleted with `meta.archived = 1`. No command brings an archived
//!    memory back; what archival keeps open is the fact itself — a plain
//!    `remember` of the same text is accepted again, where a deliberately
//!    forgotten one would be refused
//!
//! Only *current* memories take part in passes 1 to 3 (see
//! [`crate::lifecycle`]): a superseded or expired memory can neither win a
//! merge nor be summarised.

use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, OptionalExtension};

use crate::error::Result;
use crate::learn;
use crate::model::{now_unix, Kind, NewNode, Rel};
use crate::store::{self, row_to_node, NODE_COLS};

const DUP_COSINE: f32 = 0.92;
const CONTRADICTION_COSINE: f32 = 0.85;
const CLUSTER_COSINE: f32 = 0.80;
const CLUSTER_MIN: usize = 5;
const ARCHIVE_EFFECTIVE: f64 = 0.05;
const ARCHIVE_IDLE_DAYS: i64 = 90;

#[derive(Debug, Default)]
pub struct Report {
    pub superseded: usize,
    /// (older uid+title, newer uid+title) pairs needing a human decision.
    pub contradictions: Vec<(String, String)>,
    pub distilled: usize,
    pub archived: usize,
}

impl Report {
    pub fn is_empty(&self) -> bool {
        self.superseded == 0
            && self.contradictions.is_empty()
            && self.distilled == 0
            && self.archived == 0
    }
}

struct Mem {
    id: i64,
    uid: String,
    title: String,
    body: String,
    subkind: String,
    project_id: Option<i64>,
    created_at: i64,
    pinned: bool,
    vec: Vec<f32>,
}

/// Run all four passes. `dry_run` reports without writing.
///
/// The mutating run is wrapped in a single transaction so it is all-or-
/// nothing: a mid-run error must not leave, e.g., a "Distilled:" summary
/// node without its `summarizes` edges (which would surface as recall noise
/// and re-distill forever, since coverage is measured by those edges).
pub fn consolidate(conn: &Connection, model: &str, dry_run: bool) -> Result<Report> {
    if dry_run {
        return consolidate_inner(conn, model, true);
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    let report = consolidate_inner(&tx, model, false)?;
    tx.commit()?;
    Ok(report)
}

fn consolidate_inner(conn: &Connection, model: &str, dry_run: bool) -> Result<Report> {
    let mut report = Report::default();
    let mems = load_memories(conn, model)?;

    // ---- pass 1+2: pairwise within (subkind, project) buckets ----
    let mut buckets: HashMap<(String, Option<i64>), Vec<usize>> = HashMap::new();
    for (i, m) in mems.iter().enumerate() {
        buckets
            .entry((m.subkind.clone(), m.project_id))
            .or_default()
            .push(i);
    }
    let mut superseded: HashSet<i64> = HashSet::new();
    for indices in buckets.values() {
        for (a_pos, &a) in indices.iter().enumerate() {
            for &b in &indices[a_pos + 1..] {
                let (ma, mb) = (&mems[a], &mems[b]);
                if superseded.contains(&ma.id) || superseded.contains(&mb.id) {
                    continue;
                }
                let sim = dot(&ma.vec, &mb.vec);
                if sim >= DUP_COSINE {
                    // Newer wins; pinned losers are kept as-is.
                    let (older, newer) = if ma.created_at <= mb.created_at {
                        (ma, mb)
                    } else {
                        (mb, ma)
                    };
                    if older.pinned {
                        continue;
                    }
                    if !dry_run {
                        // The one supersede operation, not raw SQL: column,
                        // `supersedes` edge and the ledger row that dates it
                        // (`recall --as-of` reads that row) land together.
                        // Nothing it refuses can reach here — both are
                        // current memories and pinned losers were skipped
                        // above.
                        store::supersede(
                            conn,
                            older.id,
                            newer.id,
                            crate::model::Actor::System,
                            Some("consolidate: near-duplicate, newer wins"),
                        )?;
                    }
                    superseded.insert(older.id);
                    report.superseded += 1;
                } else if sim >= CONTRADICTION_COSINE && negation_mismatch(&ma.body, &mb.body) {
                    report.contradictions.push((
                        format!("m:{} {}", tail(&ma.uid), ma.title),
                        format!("m:{} {}", tail(&mb.uid), mb.title),
                    ));
                    // The report scrolls past; the queue does not. Which of
                    // two contradicting memories is right depends on things
                    // this pass cannot know, so it stays a finding for a
                    // person rather than becoming an automatic supersession.
                    if !dry_run {
                        let _ = crate::review::enqueue(
                            conn,
                            crate::review::Finding::Contradiction,
                            ma.id,
                            Some(mb.id),
                            Some(&format!("{} / {}", ma.title, mb.title)),
                        );
                    }
                }
            }
        }
    }

    // ---- pass 3: cluster distillation (single-link, per project) ----
    let live: Vec<usize> = (0..mems.len())
        .filter(|i| !superseded.contains(&mems[*i].id))
        .collect();
    let clusters = single_link(&mems, &live, CLUSTER_COSINE);
    for cluster in clusters {
        if cluster.len() < CLUSTER_MIN {
            continue;
        }
        let ids: Vec<i64> = cluster.iter().map(|&i| mems[i].id).collect();
        if has_summary(conn, &ids)? {
            continue;
        }
        report.distilled += 1;
        if dry_run {
            continue;
        }
        let members: Vec<&Mem> = cluster.iter().map(|&i| &mems[i]).collect();
        let title = format!("Distilled: {}", theme(&members));
        let body = members
            .iter()
            .map(|m| format!("- {}", m.title))
            .collect::<Vec<_>>()
            .join("\n");
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("summary".into());
        new.title = Some(title);
        new.body = Some(body.clone());
        new.project_id = members[0].project_id;
        new.content_hash = Some(crate::memory::content_hash(&body));
        new.meta = Some(serde_json::json!({"distilled": true}));
        let summary = store::insert_node(conn, new)?;
        for id in ids {
            store::link(conn, summary.id, id, Rel::Summarizes, 1.0)?;
        }
    }

    // ---- pass 4: decay archival ----
    let now = now_unix();
    let idle_cutoff = now - ARCHIVE_IDLE_DAYS * 86_400;
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'memory' AND deleted_at IS NULL AND superseded_by IS NULL
           AND pinned = 0 AND created_at < ?1 AND updated_at < ?1
           AND (last_accessed IS NULL OR last_accessed < ?1)"
    ))?;
    let candidates: Vec<crate::model::Node> = stmt
        .query_map([idle_cutoff], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for node in candidates {
        // What pass 1 retired is not archived as well. A real run has
        // already written `superseded_by`, so the query above left those
        // out; a dry run has written nothing and needs telling.
        if superseded.contains(&node.id) {
            continue;
        }
        if learn::effective_strength(&node, now) < ARCHIVE_EFFECTIVE {
            report.archived += 1;
            if !dry_run {
                conn.execute(
                    "UPDATE node SET deleted_at = ?2,
                            meta = json_set(meta, '$.archived', 1)
                     WHERE id = ?1",
                    rusqlite::params![node.id, now],
                )?;
                // Recorded as `archive`, never `forget`. Nobody decided this
                // — it fell below a strength threshold while idle — and the
                // distinction is load-bearing twice over: `remember` lets an
                // archived fact back in without `--force`, and "why did this
                // memory disappear?" is the question the ledger most needs to
                // answer for the deletions nobody can remember making.
                crate::audit::record_for_kind(
                    conn,
                    node.kind,
                    node.id,
                    crate::model::MutationOp::Archive,
                    crate::model::Actor::System,
                    Some("consolidate: decayed below the strength floor while idle"),
                    node.content_hash.as_deref(),
                    None,
                );
            }
        }
    }

    if !dry_run {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_consolidate', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [now.to_string()],
        )?;
    }
    Ok(report)
}

/// How often the automatic pass runs.
const AUTO_INTERVAL_SECS: i64 = 7 * 86_400;

/// When the last real pass finished, or 0 if none ever has.
///
/// "No row yet" means never consolidated (run now); a real query error
/// must NOT — it would trigger an archival pass off a corrupt read.
fn last_consolidate(conn: &Connection) -> Result<i64> {
    let row = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'last_consolidate'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()?;
    Ok(row.and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Run consolidation if the configured cadence says it's due.
/// Never errors out the caller's request path.
pub fn maybe_auto(conn: &Connection, model: &str, cadence: &str) -> Option<Report> {
    // `cadence` is the raw `[consolidate] auto` string; the config enum is
    // the one place that says which spellings mean weekly, and an unknown
    // one means off there too.
    if crate::config::ConsolidateAuto::resolve(cadence) != crate::config::ConsolidateAuto::Weekly {
        return None;
    }
    // A plain read first, so the common case — not due — takes no lock on
    // the recall path.
    match last_consolidate(conn) {
        Ok(last) if now_unix() - last < AUTO_INTERVAL_SECS => return None,
        Ok(_) => {}
        Err(err) => {
            tracing::warn!(%err, "cannot read last_consolidate; skipping auto-consolidation");
            return None;
        }
    }
    match run_if_due(conn, model) {
        Ok(report) => report,
        Err(err) => {
            tracing::warn!(%err, "auto-consolidation failed");
            None
        }
    }
}

/// Run the pass if it is still due once this connection holds the write
/// lock, and stamp `last_consolidate` in the same transaction.
///
/// The `last_consolidate` row is the claim. The CLI's recall path and the
/// MCP server's startup thread both get here having read a stale stamp; the
/// second one waits for the first one's transaction, reads the stamp that
/// transaction wrote, and stands down. Deciding under the lock rather than
/// stamping first means the row only ever records a pass that completed: a
/// pass that fails rolls its claim back with everything else, and is tried
/// again by the next caller.
fn run_if_due(conn: &Connection, model: &str) -> Result<Option<Report>> {
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if now_unix() - last_consolidate(&tx)? < AUTO_INTERVAL_SECS {
        return Ok(None);
    }
    let report = consolidate_inner(&tx, model, false)?;
    tx.commit()?;
    Ok(Some(report))
}

/// Every current memory that has an embedding for `model`, in id order.
///
/// Current is decided by [`crate::lifecycle::Lifecycle`], not by the
/// `WHERE` clause, which cannot see expiry. An expired memory left in here
/// would still win "newer wins" in pass 1 and retire a live older one —
/// after which recall shows neither.
///
/// The order is fixed because pass 2 queues a contradiction as the pair
/// (first, second) and the review queue recognises a finding it already
/// holds by that pair; the same two memories met in the other order would
/// be queued a second time.
fn load_memories(conn: &Connection, model: &str) -> Result<Vec<Mem>> {
    // `n.`-qualified because `embedding` has a `content_hash` of its own.
    let node_cols = NODE_COLS
        .split(',')
        .map(|col| format!("n.{}", col.trim()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut stmt = conn.prepare(&format!(
        "SELECT {node_cols}, e.vec AS vec
         FROM node n JOIN embedding e ON e.node_id = n.id AND e.model = ?1
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL AND n.superseded_by IS NULL
         ORDER BY n.id"
    ))?;
    let now = now_unix();
    let mut mems = Vec::new();
    let rows = stmt.query_map([model], |r| {
        Ok((row_to_node(r)?, r.get::<_, Vec<u8>>("vec")?))
    })?;
    for row in rows {
        let (node, vec) = row?;
        if !node.lifecycle(now).is_current() {
            continue;
        }
        mems.push(Mem {
            id: node.id,
            uid: node.uid,
            title: node.title.unwrap_or_default(),
            body: node.body.unwrap_or_default(),
            // A memory stored without a type is a note everywhere it is
            // read, so it consolidates with the notes.
            subkind: node.subkind.unwrap_or_else(|| "note".into()),
            project_id: node.project_id,
            created_at: node.created_at,
            pinned: node.pinned,
            vec: crate::embed::from_blob(&vec),
        });
    }
    Ok(mems)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const NEGATIONS: &[&str] = &[
    "not",
    "never",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "isn't",
    "isnt",
    "won't",
    "wont",
    "no longer",
    "avoid",
    "instead",
    "deprecated",
    "wrong",
    "refuting",
    "refuted",
];

pub(crate) fn negation_mismatch(a: &str, b: &str) -> bool {
    let has = |s: &str| {
        let lower = s.to_lowercase();
        NEGATIONS.iter().any(|n| lower.contains(n))
    };
    has(a) != has(b)
}

/// Single-link clustering over the given indices.
fn single_link(mems: &[Mem], indices: &[usize], threshold: f32) -> Vec<Vec<usize>> {
    let mut parent: HashMap<usize, usize> = indices.iter().map(|&i| (i, i)).collect();
    // Iterative find with path compression: a recursive version overflows
    // the stack on long merge chains (thousands of near-identical memories,
    // e.g. after a bulk import).
    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        let mut root = x;
        while parent[&root] != root {
            root = parent[&root];
        }
        let mut cur = x;
        while parent[&cur] != root {
            let next = parent[&cur];
            parent.insert(cur, root);
            cur = next;
        }
        root
    }
    for (pos, &a) in indices.iter().enumerate() {
        for &b in &indices[pos + 1..] {
            if mems[a].project_id != mems[b].project_id {
                continue;
            }
            if dot(&mems[a].vec, &mems[b].vec) >= threshold {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent.insert(ra, rb);
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in indices {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Does a summary node already cover most of these members?
///
/// A summary that was deleted covers nothing, and neither does one that was
/// superseded by something that is not a summary: it is not shown, so the
/// cluster is as unsummarised as if it had never been written.
///
/// A summary superseded by another *summary* still covers its cluster. That
/// is what the near-duplicate pass does to two summaries that say the same
/// thing, and the survivor is the summary of both clusters. Without this,
/// the pass that merged them left one cluster uncovered, the next pass
/// distilled it again, the new summary was a near-duplicate of the survivor
/// and was merged in turn: one new summary row per pass, forever.
fn has_summary(conn: &Connection, member_ids: &[i64]) -> Result<bool> {
    let placeholders = member_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT e.dst, e.src FROM edge e
         JOIN node s ON s.id = e.src AND s.deleted_at IS NULL
         WHERE e.rel = 'summarizes' AND e.dst IN ({placeholders})"
    ))?;
    let links: Vec<(i64, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    let now = now_unix();
    let mut shown: HashMap<i64, bool> = HashMap::new();
    let mut covered: HashSet<i64> = HashSet::new();
    for (member, summary) in links {
        let stands = match shown.get(&summary) {
            Some(&known) => known,
            None => {
                let current = store::get_node(conn, summary)
                    .ok()
                    .and_then(|node| crate::lifecycle::resolve_current(conn, node, now));
                let stands = current.is_some_and(|node| node.subkind.as_deref() == Some("summary"));
                shown.insert(summary, stands);
                stands
            }
        };
        if stands {
            covered.insert(member);
        }
    }
    Ok(covered.len() * 2 >= member_ids.len())
}

/// Most frequent meaningful tokens across member titles.
fn theme(members: &[&Mem]) -> String {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for m in members {
        for tok in crate::memory::tokens(&m.title) {
            if tok.len() > 3 && !crate::search::fts::is_stopword(&tok) {
                *freq.entry(tok).or_default() += 1;
            }
        }
    }
    let mut toks: Vec<(String, usize)> = freq.into_iter().collect();
    toks.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let picked: Vec<String> = toks.into_iter().take(3).map(|(t, _)| t).collect();
    if picked.is_empty() {
        "related memories".into()
    } else {
        picked.join(", ")
    }
}

fn tail(uid: &str) -> &str {
    &uid[uid.len().saturating_sub(6)..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::embed::to_blob;
    use crate::model::Node;

    const MODEL: &str = "test-model";

    fn add(conn: &Connection, subkind: &str, title: &str, vec: &[f32], created_at: i64) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some(subkind.into());
        new.title = Some(title.into());
        new.body = Some(title.into());
        let node = store::insert_node(conn, new).unwrap();
        // A memory written at `created_at` and not touched since: both
        // stamps are old, which is what the decay pass asks for.
        conn.execute(
            "UPDATE node SET created_at = ?2, updated_at = ?2 WHERE id = ?1",
            rusqlite::params![node.id, created_at],
        )
        .unwrap();
        let mut v = vec.to_vec();
        crate::embed::l2_normalize(&mut v);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
             VALUES (?1, ?2, ?3, ?4, x'00')",
            rusqlite::params![node.id, MODEL, v.len() as i64, to_blob(&v)],
        )
        .unwrap();
        store::get_node(conn, node.id).unwrap()
    }

    #[test]
    fn near_dups_superseded_newer_wins() {
        let conn = db::open_in_memory().unwrap();
        let old = add(
            &conn,
            "note",
            "sqlite wal is great",
            &[1.0, 0.0, 0.01],
            1000,
        );
        let new = add(
            &conn,
            "note",
            "sqlite wal is great!",
            &[1.0, 0.0, 0.0],
            2000,
        );
        let report = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(report.superseded, 1);
        let old = store::get_node(&conn, old.id).unwrap();
        assert_eq!(old.superseded_by, Some(new.id));
        // Idempotent: superseded nodes don't re-trigger.
        let again = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(again.superseded, 0);
    }

    #[test]
    fn different_scope_or_type_never_merges() {
        let conn = db::open_in_memory().unwrap();
        add(&conn, "note", "fact", &[1.0, 0.0], 1000);
        add(&conn, "gotcha", "fact", &[1.0, 0.0], 2000);
        let report = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(report.superseded, 0);
    }

    #[test]
    fn contradictions_flagged_not_resolved() {
        let conn = db::open_in_memory().unwrap();
        // cosine ≈ 0.88: similar but below the dup threshold; one negated.
        add(&conn, "note", "the api needs auth", &[1.0, 0.0, 0.0], 1000);
        add(
            &conn,
            "note",
            "the api does not need auth",
            &[1.0, 0.54, 0.0],
            2000,
        );
        let report = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(report.contradictions.len(), 1);
        assert_eq!(report.superseded, 0);
        // Nothing was changed.
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE superseded_by IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn clusters_distilled_once() {
        let conn = db::open_in_memory().unwrap();
        // Points on an arc, 25° apart: adjacent cosine ≈ 0.91 (cluster
        // range, below the 0.92 dup threshold); chain connectivity does
        // the rest.
        for i in 0..6 {
            let theta = 25f32.to_radians() * i as f32;
            add(
                &conn,
                "gotcha",
                &format!("docker network gotcha number {i}"),
                &[theta.cos(), theta.sin()],
                1000 + i,
            );
        }
        let report = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(report.distilled, 1, "{report:?}");
        let summary: String = conn
            .query_row(
                "SELECT title FROM node WHERE subkind = 'summary'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(summary.starts_with("Distilled:"), "{summary}");
        // Second run: cluster already covered.
        let again = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(again.distilled, 0);
    }

    /// The decay pass takes a memory nobody has opened or changed for the
    /// idle window. One that changed recently is left alone however old the
    /// date it carries: an import keeps the day a note was written, and an
    /// edit or a new anchor is someone still using it.
    #[test]
    fn a_memory_that_changed_recently_is_not_archived() {
        let conn = db::open_in_memory().unwrap();
        let old_time = now_unix() - 500 * 86_400;
        let untouched = add(&conn, "idea", "stale idea", &[0.0, 1.0], old_time);
        let touched = add(
            &conn,
            "idea",
            "old idea, newly edited",
            &[1.0, 0.0],
            old_time,
        );
        conn.execute(
            "UPDATE node SET updated_at = ?2 WHERE id = ?1",
            rusqlite::params![touched.id, now_unix()],
        )
        .unwrap();

        let report = consolidate(&conn, MODEL, false).unwrap();

        assert_eq!(report.archived, 1, "{report:?}");
        assert!(store::get_node(&conn, untouched.id)
            .unwrap()
            .deleted_at
            .is_some());
        assert!(store::get_node(&conn, touched.id)
            .unwrap()
            .deleted_at
            .is_none());
    }

    #[test]
    fn decayed_idle_memories_archived_reversibly() {
        let conn = db::open_in_memory().unwrap();
        let old_time = now_unix() - 500 * 86_400; // idea half-life 60d → ~0.003
        let node = add(&conn, "idea", "stale idea", &[0.0, 1.0], old_time);
        let fresh = add(&conn, "idea", "fresh idea", &[1.0, 0.0], now_unix());
        let report = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(report.archived, 1);
        let archived = store::get_node(&conn, node.id).unwrap();
        assert!(archived.deleted_at.is_some());
        assert_eq!(
            archived.meta.get("archived").and_then(|v| v.as_i64()),
            Some(1)
        );
        assert!(store::get_node(&conn, fresh.id)
            .unwrap()
            .deleted_at
            .is_none());
    }

    /// "Newer wins" never looked at what the winner was. An expired newer
    /// memory would retire a live older one, and recall then shows neither:
    /// one is superseded, the other past its expiry.
    #[test]
    fn an_expired_memory_takes_no_part_in_consolidation() {
        let conn = db::open_in_memory().unwrap();
        let recent = now_unix() - 3600;
        let old = add(&conn, "note", "the freeze is on", &[1.0, 0.0, 0.01], recent);
        let new = add(
            &conn,
            "note",
            "the freeze is on!",
            &[1.0, 0.0, 0.0],
            recent + 60,
        );
        store::set_expiry(&conn, new.id, Some(now_unix() - 60), None).unwrap();

        let report = consolidate(&conn, MODEL, false).unwrap();

        assert_eq!(report.superseded, 0, "{report:?}");
        assert!(store::get_node(&conn, old.id)
            .unwrap()
            .lifecycle(now_unix())
            .is_current());
    }

    /// A merge is the column, the `supersedes` edge and the dated ledger
    /// row, written by the same operation every other supersession uses.
    #[test]
    fn a_merge_is_a_complete_supersession() {
        let conn = db::open_in_memory().unwrap();
        let old = add(
            &conn,
            "note",
            "sqlite wal is great",
            &[1.0, 0.0, 0.01],
            1000,
        );
        let new = add(
            &conn,
            "note",
            "sqlite wal is great!",
            &[1.0, 0.0, 0.0],
            2000,
        );
        consolidate(&conn, MODEL, false).unwrap();

        let edges = store::edges_of(&conn, old.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(
            (edges[0].src, edges[0].dst, edges[0].rel),
            (new.id, old.id, Rel::Supersedes)
        );
        assert!(crate::audit::superseded_at(&conn, old.id)
            .unwrap()
            .is_some());
    }

    fn arc(conn: &Connection, count: usize) -> Vec<Node> {
        // Points on an arc, 25° apart — see `clusters_distilled_once`.
        (0..count)
            .map(|i| {
                let theta = 25f32.to_radians() * i as f32;
                add(
                    conn,
                    "gotcha",
                    &format!("docker network gotcha number {i}"),
                    &[theta.cos(), theta.sin()],
                    now_unix() - 60 + i as i64,
                )
            })
            .collect()
    }

    fn summaries(conn: &Connection) -> Vec<Node> {
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {NODE_COLS} FROM node WHERE subkind = 'summary' ORDER BY id"
            ))
            .unwrap();
        let rows = stmt.query_map([], row_to_node).unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    }

    /// A summary that has been superseded is not shown, so a cluster it
    /// covered is not covered: it gets distilled again.
    #[test]
    fn a_superseded_summary_does_not_cover_its_cluster() {
        let conn = db::open_in_memory().unwrap();
        arc(&conn, 6);
        assert_eq!(consolidate(&conn, MODEL, false).unwrap().distilled, 1);
        assert_eq!(consolidate(&conn, MODEL, false).unwrap().distilled, 0);

        let summary = summaries(&conn).remove(0);
        let replacement = add(&conn, "note", "unrelated", &[0.0, -1.0], now_unix());
        store::supersede(
            &conn,
            summary.id,
            replacement.id,
            crate::model::Actor::Human,
            None,
        )
        .unwrap();

        assert_eq!(consolidate(&conn, MODEL, false).unwrap().distilled, 1);
    }

    /// Two summaries that say the same thing are merged like any other
    /// near-duplicates, and the survivor covers both clusters. The run must
    /// settle: no new summary on the next pass, or the pass after.
    #[test]
    fn merged_summaries_do_not_make_consolidation_distil_forever() {
        let conn = db::open_in_memory().unwrap();
        arc(&conn, 6);
        assert_eq!(consolidate(&conn, MODEL, false).unwrap().distilled, 1);
        let first = summaries(&conn).remove(0);

        // The summary gets its vector (as `embed_pending` would give it),
        // and a second summary saying the same thing is written later.
        let mut v = vec![0.6f32, -0.8];
        crate::embed::l2_normalize(&mut v);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
             VALUES (?1, ?2, ?3, ?4, x'00')",
            rusqlite::params![first.id, MODEL, v.len() as i64, to_blob(&v)],
        )
        .unwrap();
        add(
            &conn,
            "summary",
            "the same summary, written again",
            &[0.6, -0.8],
            now_unix() + 5,
        );

        let merged = consolidate(&conn, MODEL, false).unwrap();
        assert_eq!(merged.superseded, 1, "the two summaries merge: {merged:?}");
        assert_eq!(
            merged.distilled, 0,
            "its cluster is still covered: {merged:?}"
        );
        let rows = summaries(&conn).len();
        for pass in 0..3 {
            let report = consolidate(&conn, MODEL, false).unwrap();
            assert_eq!(
                (report.superseded, report.distilled),
                (0, 0),
                "pass {pass} did not settle: {report:?}"
            );
        }
        assert_eq!(summaries(&conn).len(), rows, "summary rows kept growing");
    }

    /// A summary is a memory, and its hash is what `remember` compares a new
    /// capture against. Hashed any other way, the same text offered again
    /// would not be recognised as the summary.
    #[test]
    fn a_summary_is_hashed_like_any_memory() {
        let conn = db::open_in_memory().unwrap();
        arc(&conn, 6);
        consolidate(&conn, MODEL, false).unwrap();
        let summary = summaries(&conn).remove(0);
        assert_eq!(
            summary.content_hash,
            Some(crate::memory::content_hash(
                summary.body.as_deref().unwrap()
            ))
        );
    }

    /// A dry run reports what a real run would do. The archive pass reads
    /// `superseded_by` from the database, which a dry run has not written,
    /// so it has to be told what pass 1 would have retired.
    #[test]
    fn a_dry_run_counts_what_a_real_run_does() {
        let build = || {
            let conn = db::open_in_memory().unwrap();
            let long_ago = now_unix() - 500 * 86_400;
            add(&conn, "idea", "a stale idea", &[1.0, 0.0, 0.01], long_ago);
            add(
                &conn,
                "idea",
                "a stale idea!",
                &[1.0, 0.0, 0.0],
                long_ago + 60,
            );
            conn
        };
        let dry = consolidate(&build(), MODEL, true).unwrap();
        let real = consolidate(&build(), MODEL, false).unwrap();

        assert_eq!((real.superseded, real.archived), (1, 1));
        assert_eq!(
            (dry.superseded, dry.archived),
            (real.superseded, real.archived),
            "one memory must not be counted both as superseded and as archived"
        );
    }

    fn file_conn(dir: &tempfile::TempDir) -> Connection {
        db::open(&dir.path().join("mimir.db")).unwrap()
    }

    /// The CLI's recall and the MCP server's startup thread both ask whether
    /// the weekly pass is due. Both can read "due" before either has run it;
    /// only one of them may then run it.
    #[test]
    fn two_processes_do_not_both_run_the_weekly_pass() {
        let dir = tempfile::tempdir().unwrap();
        let (first, second) = (file_conn(&dir), file_conn(&dir));
        let week_ago = now_unix() - 8 * 86_400;
        first
            .execute(
                "INSERT INTO meta (key, value) VALUES ('last_consolidate', ?1)",
                [week_ago.to_string()],
            )
            .unwrap();

        // Both have seen the stale stamp; the first runs the pass.
        assert!(run_if_due(&first, MODEL).unwrap().is_some());
        assert!(
            run_if_due(&second, MODEL).unwrap().is_none(),
            "the second must find the pass already claimed"
        );

        // And from the top: not due for another week, on either connection.
        assert!(maybe_auto(&first, MODEL, "weekly").is_none());
        assert!(maybe_auto(&second, MODEL, "weekly").is_none());
    }

    #[test]
    fn the_weekly_pass_runs_on_a_store_that_never_consolidated() {
        let dir = tempfile::tempdir().unwrap();
        let conn = file_conn(&dir);
        assert!(maybe_auto(&conn, MODEL, "off").is_none());
        assert!(maybe_auto(&conn, MODEL, "weekly").is_some());
        assert!(maybe_auto(&conn, MODEL, "weekly").is_none());
    }

    #[test]
    fn dry_run_changes_nothing() {
        let conn = db::open_in_memory().unwrap();
        add(&conn, "note", "dup a", &[1.0, 0.0], 1000);
        add(&conn, "note", "dup a!", &[1.0, 0.001], 2000);
        let report = consolidate(&conn, MODEL, true).unwrap();
        assert_eq!(report.superseded, 1);
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE superseded_by IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "dry run must not write");
    }
}
