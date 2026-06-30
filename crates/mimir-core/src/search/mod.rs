//! Hybrid search: ranked legs fused with reciprocal-rank fusion.
//!
//! Legs: FTS5 BM25 (always) + vector dot product (when a query embedding
//! is supplied). RRF is scale-free, so adding legs never requires re-tuning.

pub mod fts;
pub mod vector;

use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::Result;
use crate::model::{Kind, Node, Scope};
use crate::store;

/// Reciprocal-rank-fusion constant (standard k=60).
const RRF_K: f64 = 60.0;

/// How many candidates each leg contributes before fusion.
const LEG_CAP: usize = 100;

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub scope: Scope,
    /// Empty = all kinds.
    pub kinds: Vec<Kind>,
    /// Unix floor on created_at.
    pub since: Option<i64>,
    pub limit: usize,
    /// Weight of learned strength in the final score (config scoring.strength_alpha).
    pub strength_alpha: f64,
    /// Weight of the recency boost (config scoring.recency_alpha); 0.0 = off.
    pub recency_alpha: f64,
    /// Include superseded nodes in results (default false hides them).
    pub include_superseded: bool,
}

#[derive(Debug)]
pub struct Hit {
    pub node: Node,
    pub score: f64,
}

/// SQL predicate (aliased table `n`) for a scope.
/// Project scope includes unscoped (global) nodes: cross-project knowledge
/// is meant to surface inside any project.
pub(crate) fn scope_sql(scope: Scope) -> String {
    match scope {
        Scope::All => "1=1".into(),
        Scope::Global => "n.project_id IS NULL".into(),
        Scope::Project(id) => format!("(n.project_id = {id} OR n.project_id IS NULL)"),
    }
}

/// BM25-only search (no model needed).
pub fn search(conn: &Connection, query: &SearchQuery) -> Result<Vec<Hit>> {
    search_hybrid(conn, query, None, &mut None)
}

/// Hybrid search. `vector_query` = (model name, L2-normalized query
/// embedding); pass None to skip the vector leg.
pub fn search_hybrid(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
) -> Result<Vec<Hit>> {
    let mut legs: Vec<Vec<i64>> = vec![fts::leg(conn, query, LEG_CAP)?];
    if let Some((model, qvec)) = vector_query {
        legs.push(vector::leg(conn, cache, model, qvec, query, LEG_CAP)?);
    }

    let mut rrf: HashMap<i64, f64> = HashMap::new();
    for leg in &legs {
        for (rank, id) in leg.iter().enumerate() {
            *rrf.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }

    let now = crate::model::now_unix();
    let mut hits: Vec<Hit> = rrf
        .into_iter()
        .filter_map(|(id, base)| match store::get_node(conn, id) {
            // A stale vector cache may still hold soft-deleted nodes; they (and,
            // unless explicitly asked for, superseded memories) must never surface.
            Ok(node)
                if node.deleted_at.is_some()
                    || (node.superseded_by.is_some() && !query.include_superseded) =>
            {
                None
            }
            Ok(node) => {
                // Decayed strength is a tiebreaker multiplier, never a burier.
                let effective = crate::learn::effective_strength(&node, now);
                // Optional recency boost (config scoring.recency_alpha, default 0):
                // capped multiplicative, so it nudges near-ties without burying.
                let recency = crate::learn::recency_factor(&node, now);
                let score = base
                    * (1.0 + query.strength_alpha * (1.0 + effective).ln())
                    * (1.0 + query.recency_alpha * recency);
                Some(Ok(Hit { node, score }))
            }
            Err(e) => Some(Err(e)),
        })
        .collect::<Result<_>>()?;
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Collapse exact-duplicate content before truncating: the same file indexed
    // under two collection paths surfaces as separate nodes with identical
    // title+body. Keep the highest-ranked occurrence so `limit` yields distinct
    // results instead of copies of one document.
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut seen = std::collections::HashSet::new();
    hits.retain(|hit| {
        let mut hasher = DefaultHasher::new();
        hit.node.title.hash(&mut hasher);
        hit.node.body.hash(&mut hasher);
        seen.insert(hasher.finish())
    });
    hits.truncate(query.limit);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::{Kind, NewNode};

    fn add(conn: &Connection, kind: Kind, project: Option<i64>, title: &str, body: &str) -> Node {
        let mut new = NewNode::new(kind);
        new.title = Some(title.into());
        new.body = Some(body.into());
        new.project_id = project;
        store::insert_node(conn, new).unwrap()
    }

    fn q(text: &str) -> SearchQuery {
        SearchQuery {
            text: text.into(),
            scope: Scope::All,
            kinds: vec![],
            since: None,
            limit: 10,
            strength_alpha: 0.15,
            recency_alpha: 0.0,
            include_superseded: false,
        }
    }

    #[test]
    fn superseded_is_hidden_unless_requested() {
        let conn = db::open_in_memory().unwrap();
        let old = add(
            &conn,
            Kind::Memory,
            None,
            "alpha runbook",
            "old draft blocked",
        )
        .id;
        let new = add(&conn, Kind::Memory, None, "alpha runbook", "new final done").id;
        store::set_superseded(&conn, old, new).unwrap();

        // Default recall hides the superseded node, keeps its replacement.
        let hits = search(&conn, &q("alpha runbook")).unwrap();
        assert!(
            hits.iter().all(|h| h.node.id != old),
            "superseded node leaked into recall"
        );
        assert!(hits.iter().any(|h| h.node.id == new), "replacement missing");

        // Opt-in surfaces the superseded node again.
        let mut with = q("alpha runbook");
        with.include_superseded = true;
        let hits = search(&conn, &with).unwrap();
        assert!(
            hits.iter().any(|h| h.node.id == old),
            "include_superseded did not surface the superseded node"
        );
    }

    #[test]
    fn finds_by_body_and_ranks_title_matches_higher() {
        let conn = db::open_in_memory().unwrap();
        add(
            &conn,
            Kind::Memory,
            None,
            "about databases",
            "sqlite pragma tuning notes",
        );
        add(
            &conn,
            Kind::Memory,
            None,
            "sqlite checklist",
            "general things to remember",
        );
        let hits = search(&conn, &q("sqlite")).unwrap();
        assert_eq!(hits.len(), 2);
        // Title hits outrank body hits (bm25 column weighting).
        assert_eq!(hits[0].node.title.as_deref(), Some("sqlite checklist"));
    }

    #[test]
    fn scope_filtering() {
        let conn = db::open_in_memory().unwrap();
        let project = store::ensure_project(&conn, "/tmp/p", "p").unwrap();
        add(
            &conn,
            Kind::Memory,
            Some(project.id),
            "scoped fact",
            "alpha beaver",
        );
        add(&conn, Kind::Memory, None, "global fact", "alpha beaver");

        let mut query = q("beaver");
        query.scope = Scope::Project(project.id);
        assert_eq!(search(&conn, &query).unwrap().len(), 2); // project + global

        query.scope = Scope::Global;
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.title.as_deref(), Some("global fact"));

        query.scope = Scope::Project(project.id + 999);
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1); // only the global one leaks across projects
    }

    #[test]
    fn kind_filtering_and_deleted_exclusion() {
        let conn = db::open_in_memory().unwrap();
        add(&conn, Kind::Memory, None, "memory hit", "zebra pattern");
        let chunk = add(&conn, Kind::Chunk, None, "doc hit", "zebra pattern");

        let mut query = q("zebra");
        query.kinds = vec![Kind::Memory];
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.kind, Kind::Memory);

        store::soft_delete(&conn, chunk.id).unwrap();
        let hits = search(&conn, &q("zebra")).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn collapses_exact_duplicate_content() {
        let conn = db::open_in_memory().unwrap();
        // The same content indexed twice (one file under two collection paths)
        // must surface once; a hit with the same words but a different title is
        // genuinely distinct and must remain.
        add(
            &conn,
            Kind::Chunk,
            None,
            "dup title",
            "identical alpha body",
        );
        add(
            &conn,
            Kind::Chunk,
            None,
            "dup title",
            "identical alpha body",
        );
        add(
            &conn,
            Kind::Chunk,
            None,
            "other title",
            "identical alpha body words",
        );
        let hits = search(&conn, &q("identical")).unwrap();
        let dups = hits
            .iter()
            .filter(|h| h.node.title.as_deref() == Some("dup title"))
            .count();
        assert_eq!(dups, 1, "exact duplicates must collapse to one");
        assert!(hits
            .iter()
            .any(|h| h.node.title.as_deref() == Some("other title")));
    }

    #[test]
    fn weird_input_does_not_break_fts_syntax() {
        let conn = db::open_in_memory().unwrap();
        add(
            &conn,
            Kind::Memory,
            None,
            "quoted",
            "the \"weird\" AND OR NOT (input)",
        );
        for text in ["\"unbalanced", "AND OR", "a* b: (c)", "   ", "résumé café"] {
            // Must not error, whatever it returns.
            search(&conn, &q(text)).unwrap();
        }
        assert_eq!(search(&conn, &q("weird input")).unwrap().len(), 1);
    }
}

#[cfg(test)]
mod perf_tests {
    use super::*;
    use crate::embed::to_blob;
    use crate::model::{Kind, NewNode};
    use crate::store;

    /// 100k-chunk scale check (plan target: warm hybrid < 100 ms).
    /// Heavy to seed — run explicitly: cargo test -p mimir-mem-core --release perf_100k -- --ignored
    #[test]
    #[ignore]
    fn perf_100k_chunks_warm_search_under_100ms() {
        let conn = crate::db::open_in_memory().unwrap();
        const N: usize = 100_000;
        const DIM: usize = 384;
        // Deterministic LCG (no rand dep) for synthetic vectors.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };

        let tx = conn.unchecked_transaction().unwrap();
        for i in 0..N {
            let mut new = NewNode::new(Kind::Chunk);
            new.title = Some(format!("doc{} › section {}", i / 40, i % 40));
            // Selective vocabulary: ~100 docs per topic term, ~1.3k per
            // fragment term — realistic posting-list sizes.
            new.body = Some(format!(
                "chunk{i} topic{} fragment{} item{}",
                i % 997,
                i % 79,
                i % 13
            ));
            let node = store::insert_node(&tx, new).unwrap();
            let mut v: Vec<f32> = (0..DIM).map(|_| next()).collect();
            crate::embed::l2_normalize(&mut v);
            tx.execute(
                "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
                 VALUES (?1, 'perf', ?2, ?3, x'00')",
                rusqlite::params![node.id, DIM as i64, to_blob(&v)],
            )
            .unwrap();
        }
        tx.commit().unwrap();

        let mut qv: Vec<f32> = (0..DIM).map(|_| next()).collect();
        crate::embed::l2_normalize(&mut qv);
        let query = SearchQuery {
            text: "topic42 fragment7 item3".into(),
            scope: Scope::All,
            kinds: vec![],
            since: None,
            limit: 10,
            strength_alpha: 0.15,
            recency_alpha: 0.0,
            include_superseded: false,
        };
        let mut cache = None;
        // Cold: builds the 100k×384 matrix.
        search_hybrid(&conn, &query, Some(("perf", &qv)), &mut cache).unwrap();
        // Warm: the contractual measurement.
        let started = std::time::Instant::now();
        let hits = search_hybrid(&conn, &query, Some(("perf", &qv)), &mut cache).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(hits.len(), 10);
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "warm hybrid over 100k took {elapsed:?} (budget 100 ms)"
        );
    }

    /// Scaling profile: prints recall/resolve/RAM at increasing store sizes
    /// to locate the real cliffs (the brute-force vector scan is O(N) time
    /// and RAM). Explicit + verbose:
    ///   cargo test -p mimir-mem-core --release scaling_profile -- --ignored --nocapture
    #[test]
    #[ignore]
    fn scaling_profile() {
        const DIM: usize = 384;
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let median = |mut v: Vec<u128>| {
            v.sort_unstable();
            v[v.len() / 2]
        };

        println!(
            "\n{:>9}  {:>10}  {:>11}  {:>10}  {:>11}  {:>9}",
            "nodes", "seed", "cold(build)", "warm hyb", "bm25-only", "resolve"
        );
        for &n in &[50_000usize, 200_000, 500_000] {
            let conn = crate::db::open_in_memory().unwrap();
            let t = std::time::Instant::now();
            let tx = conn.unchecked_transaction().unwrap();
            for i in 0..n {
                let mut node = NewNode::new(Kind::Chunk);
                node.title = Some(format!("doc{} section {}", i / 40, i % 40));
                node.body = Some(format!(
                    "chunk{i} topic{} fragment{} item{}",
                    i % 997,
                    i % 79,
                    i % 13
                ));
                let inserted = store::insert_node(&tx, node).unwrap();
                let mut v: Vec<f32> = (0..DIM).map(|_| next()).collect();
                crate::embed::l2_normalize(&mut v);
                tx.execute(
                    "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
                     VALUES (?1, 'perf', ?2, ?3, x'00')",
                    rusqlite::params![inserted.id, DIM as i64, to_blob(&v)],
                )
                .unwrap();
            }
            tx.commit().unwrap();
            let seed = t.elapsed();

            let mut qv: Vec<f32> = (0..DIM).map(|_| next()).collect();
            crate::embed::l2_normalize(&mut qv);
            let query = SearchQuery {
                text: "topic42 fragment7 item3".into(),
                scope: Scope::All,
                kinds: vec![],
                since: None,
                limit: 10,
                strength_alpha: 0.15,
                recency_alpha: 0.0,
                include_superseded: false,
            };
            let mut cache = None;
            let t = std::time::Instant::now();
            search_hybrid(&conn, &query, Some(("perf", &qv)), &mut cache).unwrap();
            let cold = t.elapsed();
            let warm = median(
                (0..7)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        search_hybrid(&conn, &query, Some(("perf", &qv)), &mut cache).unwrap();
                        t.elapsed().as_micros()
                    })
                    .collect(),
            );
            let bm25 = median(
                (0..7)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        search(&conn, &query).unwrap();
                        t.elapsed().as_micros()
                    })
                    .collect(),
            );
            // resolve_ref: short-tail LIKE '%xxxxxx' — leading wildcard, full scan.
            let some_uid: String = conn
                .query_row(
                    "SELECT uid FROM node LIMIT 1 OFFSET ?1",
                    [(n / 2) as i64],
                    |r| r.get(0),
                )
                .unwrap();
            let tail = &some_uid[some_uid.len() - 6..];
            let resolve = median(
                (0..7)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        store::resolve_ref(&conn, tail).unwrap();
                        t.elapsed().as_micros()
                    })
                    .collect(),
            );
            let ram_mb = (n * DIM * 4) as f64 / 1_048_576.0;
            println!(
                "{n:>9}  {:>9.1?}  {:>11.1?}  {:>8} µs  {:>9} µs  {:>7} µs   matrix ~{ram_mb:.0} MB",
                seed, cold, warm, bm25, resolve
            );
        }
        println!();
    }
}
