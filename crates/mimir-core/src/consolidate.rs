//! LLM-free memory consolidation: four passes, none destructive.
//!
//! 1. near-duplicate merge (cosine ≥ 0.92, same type+scope) → the older
//!    node is superseded, never deleted
//! 2. contradiction flagging (high similarity + negation mismatch) —
//!    reported to the human, never auto-resolved
//! 3. cluster distillation (cosine ≥ 0.80, clusters ≥ 5) → extractive
//!    summary node with `summarizes` edges
//! 4. decay archival (effective strength < 0.05, untouched > 90 days) →
//!    soft-deleted with meta.archived = true (reversible)

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
pub fn consolidate(conn: &Connection, model: &str, dry_run: bool) -> Result<Report> {
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
                        conn.execute(
                            "UPDATE node SET superseded_by = ?2 WHERE id = ?1",
                            [older.id, newer.id],
                        )?;
                        store::link(conn, newer.id, older.id, Rel::Supersedes, 1.0)?;
                    }
                    superseded.insert(older.id);
                    report.superseded += 1;
                } else if sim >= CONTRADICTION_COSINE && negation_mismatch(&ma.body, &mb.body) {
                    report.contradictions.push((
                        format!("m:{} {}", tail(&ma.uid), ma.title),
                        format!("m:{} {}", tail(&mb.uid), mb.title),
                    ));
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
        new.content_hash = Some(blake3::hash(body.as_bytes()).as_bytes().to_vec());
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
           AND pinned = 0 AND created_at < ?1
           AND (last_accessed IS NULL OR last_accessed < ?1)"
    ))?;
    let candidates: Vec<crate::model::Node> = stmt
        .query_map([idle_cutoff], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for node in candidates {
        if learn::effective_strength(&node, now) < ARCHIVE_EFFECTIVE {
            report.archived += 1;
            if !dry_run {
                conn.execute(
                    "UPDATE node SET deleted_at = ?2,
                            meta = json_set(meta, '$.archived', 1)
                     WHERE id = ?1",
                    rusqlite::params![node.id, now],
                )?;
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

/// Run consolidation if the configured cadence says it's due.
/// Never errors out the caller's request path.
pub fn maybe_auto(conn: &Connection, model: &str, cadence: &str) -> Option<Report> {
    if cadence != "weekly" {
        return None;
    }
    // "No row yet" means never consolidated (run now); a real query error
    // must NOT — it would trigger an archival pass off a corrupt read.
    let last: i64 = match conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'last_consolidate'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()
    {
        Ok(row) => row.and_then(|v| v.parse().ok()).unwrap_or(0),
        Err(err) => {
            tracing::warn!(%err, "cannot read last_consolidate; skipping auto-consolidation");
            return None;
        }
    };
    if now_unix() - last < 7 * 86_400 {
        return None;
    }
    match consolidate(conn, model, false) {
        Ok(r) => Some(r),
        Err(err) => {
            tracing::warn!(%err, "auto-consolidation failed");
            None
        }
    }
}

fn load_memories(conn: &Connection, model: &str) -> Result<Vec<Mem>> {
    let mut stmt = conn.prepare(
        "SELECT n.id, n.uid, n.title, n.body, n.subkind, n.project_id, n.created_at,
                n.pinned, e.vec
         FROM node n JOIN embedding e ON e.node_id = n.id AND e.model = ?1
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL AND n.superseded_by IS NULL",
    )?;
    let mems = stmt
        .query_map([model], |r| {
            Ok(Mem {
                id: r.get(0)?,
                uid: r.get(1)?,
                title: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                body: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                subkind: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                project_id: r.get(5)?,
                created_at: r.get(6)?,
                pinned: r.get::<_, i64>(7)? != 0,
                vec: crate::embed::from_blob(&r.get::<_, Vec<u8>>(8)?),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
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

fn negation_mismatch(a: &str, b: &str) -> bool {
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
fn has_summary(conn: &Connection, member_ids: &[i64]) -> Result<bool> {
    let placeholders = member_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let covered: i64 = conn.query_row(
        &format!(
            "SELECT count(DISTINCT e.dst) FROM edge e
             JOIN node s ON s.id = e.src AND s.deleted_at IS NULL
             WHERE e.rel = 'summarizes' AND e.dst IN ({placeholders})"
        ),
        [],
        |r| r.get(0),
    )?;
    Ok(covered as usize * 2 >= member_ids.len())
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
        conn.execute(
            "UPDATE node SET created_at = ?2 WHERE id = ?1",
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
