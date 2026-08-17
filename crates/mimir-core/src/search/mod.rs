//! Hybrid search: ranked legs fused with reciprocal-rank fusion.
//!
//! Legs: FTS5 BM25 (always) + vector dot product (when a query embedding
//! is supplied) + an identifier-exact phrase leg (when the query text
//! contains code-identifier-shaped fragments, e.g. `Foo::bar`). RRF is
//! scale-free, so adding legs never requires re-tuning.

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
    /// Weight of the gotcha/decision type-priority boost (config
    /// scoring.type_prior_alpha); 0.0 = off.
    pub type_prior_alpha: f64,
    /// Sub-1.0 multiplier for `Kind::CodeChunk` hits (config
    /// scoring.code_damp); 1.0 = off. See `ScoringConfig::code_damp`.
    pub code_damp: f64,
    /// Include *retired* nodes in results (default false hides them). Retired
    /// means either superseded by a replacement, or past a declared
    /// `meta.expires_at`. Both are "kept as history, not current truth", so
    /// they share one archaeology flag. Note `resolves_when` does NOT retire a
    /// node — an unevaluatable condition can't gate.
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
/// embedding); pass None to skip the vector leg. No impression damp
/// (`scoring.impression_alpha`) — see [`search_hybrid_scored`] for that.
pub fn search_hybrid(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
) -> Result<Vec<Hit>> {
    search_hybrid_scored(conn, query, vector_query, cache, 0.0)
}

/// Same as [`search_hybrid`], with the shown-but-never-opened negative
/// prior (`learn::impression_damp`) applied at `impression_alpha`. Split
/// out as its own function (rather than a `SearchQuery` field) so existing
/// callers — notably `eval::mod`'s scoring loop, which constructs
/// `SearchQuery` and calls `search_hybrid`/`search_hybrid_with_legs`
/// directly and must stay untouched — keep compiling and keep running at
/// the always-safe `alpha = 0.0` no-op without needing to know this exists.
pub fn search_hybrid_scored(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
    impression_alpha: f64,
) -> Result<Vec<Hit>> {
    Ok(search_hybrid_with_legs_scored(conn, query, vector_query, cache, impression_alpha)?.0)
}

/// Same as [`search_hybrid`], but also returns the raw per-leg ranked id
/// lists (`legs[0]` = FTS, `legs[1]` = vector iff `vector_query` was
/// supplied) before RRF fusion. Used by callers that need to know *why* a
/// hit ranked where it did — e.g. the auto-recall relevance floor, which
/// requires lexical+semantic agreement on the top hit before injecting it
/// unprompted. `search_hybrid` is a thin wrapper over this so the scoring
/// formula has one source of truth.
///
/// A third, identifier-exact leg (see `fts::identifier_leg`) also feeds
/// RRF fusion below when the query has an identifier-shaped fragment, but
/// is deliberately NOT added to the returned `legs`: callers index it by
/// fixed position (`legs[1]` == vector iff requested), and inserting a
/// third leg would shift that when no vector leg exists. Folding it
/// straight into `rrf` keeps `legs`'s shape exactly as documented above.
///
/// No impression damp — see [`search_hybrid_with_legs_scored`] for that;
/// this wrapper stays at the always-safe `alpha = 0.0` no-op so existing
/// direct callers (`eval::mod`) don't need to change.
pub fn search_hybrid_with_legs(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
    search_hybrid_with_legs_scored(conn, query, vector_query, cache, 0.0)
}

/// The real implementation behind [`search_hybrid_with_legs`] /
/// [`search_hybrid`] / [`search_hybrid_scored`]: identical fusion + score
/// formula, plus the shown-but-never-opened negative prior at
/// `impression_alpha` (config `scoring.impression_alpha`, default 0.0/off
/// — see `learn::impression_damp`). `Mimir::search_with`/`search_with_legs`
/// (the only production callers that know the configured alpha) call this
/// directly; every other caller goes through one of the `0.0`-forwarding
/// wrappers above.
pub fn search_hybrid_with_legs_scored(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
    impression_alpha: f64,
) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
    let t_fts = std::time::Instant::now();
    let mut legs: Vec<Vec<i64>> = vec![fts::leg(conn, query, LEG_CAP)?];
    let fts_elapsed = t_fts.elapsed();
    let t_vec = std::time::Instant::now();
    if let Some((model, qvec)) = vector_query {
        legs.push(vector::leg(conn, cache, model, qvec, query, LEG_CAP)?);
    }
    let vec_elapsed = t_vec.elapsed();
    let t_ident = std::time::Instant::now();
    let ident_leg = fts::identifier_leg(conn, query, LEG_CAP)?;
    let ident_elapsed = t_ident.elapsed();

    let mut rrf: HashMap<i64, f64> = HashMap::new();
    for leg in legs.iter().chain(std::iter::once(&ident_leg)) {
        for (rank, id) in leg.iter().enumerate() {
            *rrf.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }

    let now = crate::model::now_unix();
    let fused_count = rrf.len();
    let t_getnode = std::time::Instant::now();
    let ids: Vec<i64> = rrf.keys().copied().collect();
    // Batched alongside get_nodes below (one round trip for the whole
    // candidate set, never per-node — this is a hot path). Skipped
    // entirely at the default alpha=0.0: no point paying a query whose
    // result the damp formula would ignore anyway (impression_damp always
    // returns 1.0 for alpha<=0.0).
    let impressions = if impression_alpha > 0.0 {
        store::impression_stats(conn, &ids)?
    } else {
        HashMap::new()
    };
    let mut hits: Vec<Hit> = store::get_nodes(conn, &ids)?
        .into_iter()
        .filter_map(|node| {
            // A stale vector cache may still hold soft-deleted nodes; they (and,
            // unless explicitly asked for, retired memories — superseded or
            // past their declared expiry) must never surface.
            if node.deleted_at.is_some()
                || ((node.superseded_by.is_some() || node.is_expired(now))
                    && !query.include_superseded)
            {
                return None;
            }
            // Decayed strength is a tiebreaker multiplier, never a burier.
            let effective = crate::learn::effective_strength(&node, now);
            // Optional recency boost (config scoring.recency_alpha, default 0):
            // capped multiplicative, so it nudges near-ties without burying.
            // Gated to kinds/subkinds with a defined half-life (the same
            // decaying memory types `effective_strength` uses) — a
            // never-decaying node (doc/code chunk, person/summary memory)
            // gets NO recency term at all, not `recency_factor`'s 1.0. 1.0
            // is the *max* of that (0,1] range, so applying it unconditionally
            // would hand every non-decaying kind a permanent boost relative to
            // every aging memory instead of leaving them recency-neutral.
            let recency_term =
                if crate::learn::half_life_days(node.kind, node.subkind.as_deref()).is_some() {
                    1.0 + query.recency_alpha * crate::learn::recency_factor(&node, now)
                } else {
                    1.0
                };
            // Optional type-priority boost (config scoring.type_prior_alpha):
            // gotcha/decision nudged over an equally-matching note/idea.
            let prior = crate::learn::type_prior(node.subkind.as_deref());
            let base = rrf[&node.id];
            // Code content dwarfs memories on a real corpus, so it needs a
            // damping multiplier to keep it from drowning out memories in
            // the default `all` pool (see ScoringConfig::code_damp).
            let code_damp = if node.kind == Kind::CodeChunk {
                query.code_damp
            } else {
                1.0
            };
            // Optional shown-but-never-opened negative prior (config
            // scoring.impression_alpha, default 0.0/off): see
            // `learn::impression_damp`.
            let impression_stats = impressions.get(&node.id).copied().unwrap_or_default();
            let impression_damp =
                crate::learn::impression_damp(&impression_stats, impression_alpha, now);
            let score = base
                * (1.0 + query.strength_alpha * (1.0 + effective).ln())
                * recency_term
                * (1.0 + query.type_prior_alpha * (prior - 1.0))
                * code_damp
                * impression_damp;
            Some(Hit { node, score })
        })
        .collect();
    let getnode_elapsed = t_getnode.elapsed();
    let t_rest = std::time::Instant::now();
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
    tracing::debug!(
        ?fts_elapsed,
        ?vec_elapsed,
        ?ident_elapsed,
        ?getnode_elapsed,
        sort_dedup_elapsed = ?t_rest.elapsed(),
        fused_count,
        "search_hybrid: timing split"
    );
    Ok((hits, legs))
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

    fn set_meta(conn: &Connection, id: i64, meta: serde_json::Value) {
        conn.execute(
            "UPDATE node SET meta = ?1 WHERE id = ?2",
            rusqlite::params![meta.to_string(), id],
        )
        .unwrap();
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
            type_prior_alpha: 0.0,
            code_damp: 1.0,
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
        store::set_superseded(&conn, old, new, crate::model::Actor::System, None).unwrap();

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
    fn expired_is_hidden_but_resolves_when_alone_is_not() {
        use crate::model::now_unix;
        let conn = db::open_in_memory().unwrap();
        let now = now_unix();

        let fresh = add(&conn, Kind::Memory, None, "beta runbook", "still true").id;
        let expired = add(&conn, Kind::Memory, None, "beta runbook", "no longer true").id;
        let conditional = add(&conn, Kind::Memory, None, "beta runbook", "true for now").id;

        // Expired an hour ago.
        set_meta(
            &conn,
            expired,
            serde_json::json!({ "expires_at": now - 3600 }),
        );
        // Carries only an unevaluatable end condition — must NOT be gated.
        set_meta(
            &conn,
            conditional,
            serde_json::json!({ "resolves_when": "when upstream ships the fix" }),
        );

        let hits = search(&conn, &q("beta runbook")).unwrap();
        assert!(
            hits.iter().all(|h| h.node.id != expired),
            "expired node leaked into recall"
        );
        assert!(
            hits.iter().any(|h| h.node.id == fresh),
            "fresh node missing"
        );
        assert!(
            hits.iter().any(|h| h.node.id == conditional),
            "resolves_when must not gate — the condition cannot be evaluated, so \
             the memory has to keep surfacing with its caveat attached"
        );

        // Archaeology opt-in surfaces the expired node again.
        let mut with = q("beta runbook");
        with.include_superseded = true;
        let hits = search(&conn, &with).unwrap();
        assert!(
            hits.iter().any(|h| h.node.id == expired),
            "include_superseded did not surface the expired node"
        );
    }

    #[test]
    fn expiry_in_the_future_does_not_gate() {
        use crate::model::now_unix;
        let conn = db::open_in_memory().unwrap();
        let now = now_unix();
        let later = add(&conn, Kind::Memory, None, "gamma runbook", "valid a while").id;
        set_meta(
            &conn,
            later,
            serde_json::json!({ "expires_at": now + 86_400 }),
        );

        let hits = search(&conn, &q("gamma runbook")).unwrap();
        assert!(
            hits.iter().any(|h| h.node.id == later),
            "a memory whose expiry has not yet arrived must still surface"
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

        store::soft_delete(&conn, chunk.id, crate::model::Actor::System, None).unwrap();
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

    #[test]
    fn identifier_leg_prefers_the_exact_adjacent_identifier() {
        let conn = db::open_in_memory().unwrap();
        // Same bag of words in both bodies, only order differs: one has the
        // identifier's own tokens adjacent (as "MatrixCache::ensure" itself
        // tokenizes), the other has them scattered. Plain BM25 is a
        // bag-of-words model — same term frequencies, same document length —
        // so it scores these two identically. Only the identifier leg's
        // phrase match (fts::identifier_leg) can tell them apart.
        let exact = add(
            &conn,
            Kind::Memory,
            None,
            "notes",
            "MatrixCache::ensure loads the model",
        );
        add(
            &conn,
            Kind::Memory,
            None,
            "notes",
            "ensure the model matrixcache loads",
        );
        let hits = search(&conn, &q("MatrixCache::ensure")).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].node.id, exact.id,
            "the doc with the literal adjacent identifier must rank first"
        );
        assert!(
            hits[0].score > hits[1].score,
            "identifier leg must strictly break the BM25 tie, not just happen to sort first"
        );
    }

    #[test]
    fn impression_alpha_zero_is_byte_identical_noop() {
        let conn = db::open_in_memory().unwrap();
        let heavily_shown = add(&conn, Kind::Memory, None, "notes", "zebra pattern alpha");
        add(&conn, Kind::Memory, None, "notes", "zebra pattern beta");
        // Pile up well past the impression-damp threshold, never opened —
        // if alpha=0.0 weren't a true no-op, this would move the ranking.
        let shown: Vec<(i64, i64, f64)> =
            (0..20).map(|rank| (heavily_shown.id, rank, 1.0)).collect();
        store::record_shown(&conn, b"q", &shown).unwrap();

        let with_impressions = search_hybrid(&conn, &q("zebra pattern"), None, &mut None).unwrap();
        let scores_a: Vec<f64> = with_impressions.iter().map(|h| h.score).collect();

        // Same query, no impression history at all seeded on a fresh store.
        let clean_conn = db::open_in_memory().unwrap();
        add(
            &clean_conn,
            Kind::Memory,
            None,
            "notes",
            "zebra pattern alpha",
        );
        add(
            &clean_conn,
            Kind::Memory,
            None,
            "notes",
            "zebra pattern beta",
        );
        let without_impressions =
            search_hybrid(&clean_conn, &q("zebra pattern"), None, &mut None).unwrap();
        let scores_b: Vec<f64> = without_impressions.iter().map(|h| h.score).collect();

        assert_eq!(
            scores_a, scores_b,
            "alpha=0.0 (the default forwarded by search_hybrid) must be bit-identical \
             regardless of impression history"
        );
    }

    #[test]
    fn impression_damp_demotes_shown_never_opened_past_threshold() {
        let conn = db::open_in_memory().unwrap();
        // Same bag of words, so BM25 alone ties them — only the impression
        // damp (once alpha > 0) can break the tie.
        let noisy = add(&conn, Kind::Memory, None, "notes", "zebra pattern alpha");
        let quiet = add(&conn, Kind::Memory, None, "notes", "zebra pattern beta");
        let shown: Vec<(i64, i64, f64)> = (0..20).map(|rank| (noisy.id, rank, 1.0)).collect();
        store::record_shown(&conn, b"q", &shown).unwrap();

        let hits = search_hybrid_scored(&conn, &q("zebra pattern"), None, &mut None, 0.4).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].node.id, quiet.id,
            "the never-shown-heavily node must outrank the shown-but-never-opened one"
        );
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn impression_damp_exempts_an_opened_node() {
        // Compares the SAME node's score with the damp off vs on, rather
        // than comparing two different nodes against each other: SQLite
        // FTS5's tie-break for exactly-equal bm25() scores is rowid order,
        // not insertion-independent, so two "identical bag of words" nodes
        // don't actually get identical RRF base scores (a prior version of
        // this test asserted hits[0].score == hits[1].score and flaked on
        // exactly that artifact). Isolating one node's own score at
        // alpha=0.0 vs alpha=0.4 sidesteps it entirely.
        let conn = db::open_in_memory().unwrap();
        let opened = add(&conn, Kind::Memory, None, "notes", "zebra pattern alpha");
        let shown: Vec<(i64, i64, f64)> = (0..20).map(|rank| (opened.id, rank, 1.0)).collect();
        store::record_shown(&conn, b"q", &shown).unwrap();
        crate::learn::record_opened(&conn, opened.id).unwrap();

        let query = q("zebra pattern");
        let undamped = search_hybrid_scored(&conn, &query, None, &mut None, 0.0).unwrap();
        let damped = search_hybrid_scored(&conn, &query, None, &mut None, 0.4).unwrap();
        let score_of = |hits: &[Hit]| {
            hits.iter()
                .find(|h| h.node.id == opened.id)
                .expect("opened node missing from hits")
                .score
        };
        assert_eq!(
            score_of(&undamped),
            score_of(&damped),
            "an opened node's own score must be identical whether or not \
             the damp is enabled — it's exempt regardless of alpha"
        );
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
            type_prior_alpha: 0.0,
            code_damp: 1.0,
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
                type_prior_alpha: 0.0,
                code_damp: 1.0,
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
