//! Hybrid search: ranked legs fused with reciprocal-rank fusion.
//!
//! Legs: FTS5 BM25 (always) + vector dot product (when a query embedding
//! is supplied) + an identifier-exact phrase leg (when the query text
//! contains code-identifier-shaped fragments, e.g. `Foo::bar`). RRF is
//! scale-free, so adding legs never requires re-tuning.
//!
//! # The `legs` a search returns
//!
//! The `*_with_legs` functions also hand back the ranked id lists the hits
//! were fused from, for the callers that need to know why a hit ranked
//! where it did (`inject`'s relevance floor, `strength`'s match label):
//!
//! - `legs[0]` is the lexical (FTS) leg: always present, possibly empty.
//! - `legs[1]` is the vector leg, present only when one **ran** — a query
//!   embedding was supplied *and* the leg ranked at least one node. A model
//!   with nothing to compare the query against (nothing embedded yet,
//!   vectors left over from another model, a dimension mismatch, a filter no
//!   embedded node passes) yields no list at all, the same as no model.
//!   An empty list would read as "the semantic retriever found nothing that
//!   agrees", and a reader that requires agreement would then refuse every
//!   hit on a store that answers fine with no model loaded.
//! - The identifier leg feeds fusion and is never in `legs`: the two slots
//!   above are fixed positions, and a third list would shift them.
//!
//! Readers ask [`vector_leg`] rather than indexing. "There is a `legs[1]`"
//! and "a vector leg ran" are the same fact only because this module keeps
//! them so.

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
    /// Answer as the store stood at this instant (unix seconds) instead of
    /// now — *transaction*-time travel, the axis `created_at`/`deleted_at`
    /// and the mutation ledger record.
    ///
    /// A node qualifies when it existed then (`created_at <= as_of`) and had
    /// not yet been deleted (`deleted_at` absent or later), and it counts as
    /// current unless it was already superseded or expired *by then* — which
    /// is why this needs [`crate::audit::superseded_at`]: `superseded_by` is
    /// a column with no date of its own.
    ///
    /// Deliberately separate from valid time (`meta.valid_from`/`valid_to`,
    /// see [`crate::model::Node::valid_from`]). "What did I believe in March"
    /// and "what was true in March" are different questions, and a store that
    /// answers only one of them while sounding like it answers both is worse
    /// than one that answers neither.
    ///
    /// **Known limit, stated rather than hidden:** the vector leg's matrix
    /// cache is built once per model over live nodes only, so a memory
    /// deleted since `as_of` can be proposed by the lexical leg but not the
    /// semantic one. As-of recall is therefore complete but ranked more
    /// lexically than a present-tense recall of the same words. Relaxing the
    /// cache would mean carrying every tombstone in a hot in-memory matrix
    /// forever to serve a rare query; `mimir recall --as-of` says so in its
    /// output instead.
    ///
    /// A memory edited since `as_of` comes back with the title and body it
    /// had then (`reword_as_of`). Both legs still match against the
    /// current text — only the current text is indexed — so such a memory
    /// is found by today's wording and shown in the old one.
    pub as_of: Option<i64>,
}

impl SearchQuery {
    /// The instant this query is asked about: `as_of`, or now.
    pub fn at(&self) -> i64 {
        self.as_of.unwrap_or_else(crate::model::now_unix)
    }
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

/// What one search can reach: a scope for memories and a scope for
/// everything else.
///
/// A [`Scope`] answers "which projects" with one project for every kind of
/// node. That is right in every checkout but one. A git linked worktree
/// reads its memories from its repository's project and its code and docs
/// from its own (`Mimir::projects_for_cwd`), and a recall there is one
/// ranked list drawn from both. Two searches merged afterwards would rank
/// two fused lists against each other by scores that were never fused
/// together, so the pair goes into the legs and there is one fusion.
///
/// `memories` is [`Kind::Memory`]. Every other kind a search returns
/// answers to `code_and_docs`, because every one of them is derived from,
/// or attached to, the files of one checkout: doc chunks, files and their
/// collection, the annotation written on one of those (filed under its
/// target's project), symbols, code chunks, communities.
///
/// The pair is asked in SQL by the lexical legs ([`Reach::sql`]) and in
/// Rust by the vector leg, which filters cached rows ([`Reach::admits`]).
/// The test `reach_has_one_answer_in_sql_and_rust` holds the two to one
/// answer.
///
/// The functions that take a `Reach` (the `*_within` ones) read it instead
/// of [`SearchQuery::scope`]; a caller of those sets the query's scope to
/// `memories`, which is what the miss ledger keys a recall by. The
/// functions without one pass [`Reach::uniform`] of the query's scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reach {
    /// Where memories are read.
    pub memories: Scope,
    /// Where everything that is not a memory is read.
    pub code_and_docs: Scope,
}

impl Reach {
    /// One scope for every kind: what a bare [`Scope`] means.
    pub fn uniform(scope: Scope) -> Reach {
        Reach {
            memories: scope,
            code_and_docs: scope,
        }
    }

    /// Whether memories and everything else are read in the same scope.
    pub fn is_uniform(self) -> bool {
        self.memories == self.code_and_docs
    }

    /// The scope a node of `kind` answers to.
    pub fn scope_of(self, kind: Kind) -> Scope {
        self.scope_of_kind_named(kind.as_str())
    }

    /// [`Reach::scope_of`] for a kind as the `node.kind` column spells it,
    /// for a reader that has the column and no [`Kind`] (the vector leg's
    /// cached rows). A spelling this build does not know is not a memory.
    pub(crate) fn scope_of_kind_named(self, kind: &str) -> Scope {
        if kind == Kind::Memory.as_str() {
            self.memories
        } else {
            self.code_and_docs
        }
    }

    /// Whether a node of `kind` owned by `project_id` (`None` = global) is
    /// within reach. The Rust twin of [`Reach::sql`].
    pub fn admits(self, kind: Kind, project_id: Option<i64>) -> bool {
        self.scope_of(kind).admits(project_id)
    }

    /// SQL predicate (aliased table `n`) for this reach.
    ///
    /// A uniform reach is exactly [`scope_sql`] of its one scope, character
    /// for character: outside a linked worktree a search runs the statement
    /// it always ran, so no plan and no ranking can move there.
    pub(crate) fn sql(self) -> String {
        if self.is_uniform() {
            return scope_sql(self.memories);
        }
        let memory = Kind::Memory.as_str();
        format!(
            "((n.kind = '{memory}' AND {}) OR (n.kind <> '{memory}' AND {}))",
            scope_sql(self.memories),
            scope_sql(self.code_and_docs)
        )
    }
}

/// Whether `node` was present in the store at the query's instant.
///
/// Without `as_of` this is the old rule verbatim: a tombstone never surfaces.
/// With it, a node counts as present when it had been created and had not yet
/// been deleted — the ordinary meaning of "was it there in March".
fn existed_at(_conn: &Connection, node: &Node, as_of: Option<i64>) -> bool {
    match as_of {
        None => node.deleted_at.is_none(),
        Some(at) => node.created_at <= at && node.deleted_at.is_none_or(|deleted| deleted > at),
    }
}

/// Whether `node` counted as retired — superseded or expired — at the
/// query's instant.
///
/// The supersession date is read from the mutation ledger rather than the
/// node, because `superseded_by` is a bare column: `set_superseded` bumps
/// `updated_at`, which the next edit overwrites. A supersession with no
/// recorded date is treated as always-having-been (the pre-ledger behaviour),
/// which keeps stores written before the ledger existed answering the way
/// they always did rather than quietly resurrecting retired memories.
fn retired_at(conn: &Connection, node: &Node, now: i64, as_of: Option<i64>) -> bool {
    if node.is_expired(now) {
        return true;
    }
    if node.superseded_by.is_none() {
        return false;
    }
    match as_of {
        None => true,
        Some(at) => crate::audit::superseded_at(conn, node.id)
            .ok()
            .flatten()
            .is_none_or(|when| when <= at),
    }
}

/// SQL predicate (aliased table `n`) for "present in the store", at now or at
/// an as-of instant. The Rust-side twin is [`existed_at`]; both must agree, so
/// they are written next to each other deliberately.
pub(crate) fn liveness_sql(as_of: Option<i64>) -> String {
    match as_of {
        None => "n.deleted_at IS NULL".into(),
        Some(at) => {
            format!("(n.created_at <= {at} AND (n.deleted_at IS NULL OR n.deleted_at > {at}))")
        }
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
/// lists before RRF fusion, in the shape the module docs describe (`legs[0]`
/// lexical, `legs[1]` the vector leg when one ran — read it with
/// [`vector_leg`]). Used by callers that need to know *why* a hit ranked
/// where it did — e.g. the auto-recall relevance floor, which requires
/// lexical+semantic agreement on the top hit before injecting it
/// unprompted. `search_hybrid` is a thin wrapper over this so the scoring
/// formula has one source of truth.
///
/// A third, identifier-exact leg (see `fts::identifier_leg`) also feeds
/// RRF fusion below when the query has an identifier-shaped fragment, but
/// is deliberately NOT added to the returned `legs`: the two slots are
/// fixed positions, and inserting a third leg would shift them when no
/// vector leg exists. Folding it straight into `rrf` keeps `legs`'s shape
/// exactly as documented.
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
    search_hybrid_with_legs_scored_within(
        conn,
        query,
        Reach::uniform(query.scope),
        vector_query,
        cache,
        impression_alpha,
    )
}

/// [`search_hybrid_with_legs_scored`] over a [`Reach`]: memories from one
/// scope and everything else from another, in one fusion. Every leg filters
/// by `reach`; `query.scope` is not read.
pub fn search_hybrid_with_legs_scored_within(
    conn: &Connection,
    query: &SearchQuery,
    reach: Reach,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
    impression_alpha: f64,
) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
    let t_fts = std::time::Instant::now();
    let mut legs: Vec<Vec<i64>> = vec![fts::leg_within(conn, query, reach, LEG_CAP)?];
    let fts_elapsed = t_fts.elapsed();
    let t_vec = std::time::Instant::now();
    if let Some((model, qvec)) = vector_query {
        let leg = vector::leg_within(conn, cache, model, qvec, query, reach, LEG_CAP)?;
        // A leg that ranked nothing did not run (see the module docs). It
        // contributes nothing to fusion either way, so the hits are the
        // same; what changes is that no reader mistakes it for a verdict.
        if !leg.is_empty() {
            legs.push(leg);
        }
    }
    let vec_elapsed = t_vec.elapsed();
    let t_ident = std::time::Instant::now();
    let ident_leg = fts::identifier_leg_within(conn, query, reach, LEG_CAP)?;
    let ident_elapsed = t_ident.elapsed();

    let mut rrf: HashMap<i64, f64> = HashMap::new();
    for leg in legs.iter().chain(std::iter::once(&ident_leg)) {
        for (rank, id) in leg.iter().enumerate() {
            *rrf.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }

    let now = query.at();
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
            //
            // Under `--as-of`, "deleted" and "superseded" are judged at that
            // instant rather than now: a memory deleted last week was still
            // there in March, and one superseded in June was still current in
            // March. The supersession date comes from the mutation ledger,
            // since `superseded_by` carries none.
            if !existed_at(conn, &node, query.as_of) {
                return None;
            }
            if !query.include_superseded && retired_at(conn, &node, now, query.as_of) {
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
            let recency_term = if crate::learn::node_half_life_days(&node).is_some() {
                1.0 + query.recency_alpha * crate::learn::recency_factor(&node, now)
            } else {
                1.0
            };
            // Optional type-priority boost (config scoring.type_prior_alpha):
            // gotcha/decision nudged over an equally-matching note/idea.
            let prior = crate::learn::node_type_prior(&node);
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
    if let Some(at) = query.as_of {
        for hit in &mut hits {
            reword_as_of(conn, &mut hit.node, at)?;
        }
    }
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

/// Give a memory the title and body it carried at `at`, when it has been
/// edited since ([`crate::revision::text_at`]); otherwise leave it alone.
///
/// Applied to the page an as-of search returns, so every surface that
/// prints a hit — the compact line, `--full`, `--json`, the MCP tool, the
/// match-strength label — reads the wording of then without having to know
/// to ask. An error is returned rather than swallowed: falling back to
/// today's text under a past date is the confident wrong answer the
/// revision table exists to prevent.
///
/// Only the text travels back. Retrieval itself still runs over the current
/// index, so a memory is found by what it says now and shown as it read
/// then; tags, links and the rest of the row are as they stand today.
fn reword_as_of(conn: &Connection, node: &mut Node, at: i64) -> Result<()> {
    if node.kind != Kind::Memory {
        return Ok(());
    }
    if let Some(then) = crate::revision::text_at(conn, node.id, at)? {
        node.title = then.title;
        node.body = then.body;
    }
    Ok(())
}

/// The vector leg of a `legs` list, if one ran for that query — the one
/// reading of the protocol in the module docs.
///
/// `None` covers every way of having no semantic ranking: no model, a
/// BM25-only search, or a model with nothing usable to rank. An empty slot
/// counts as `None` too, so legs assembled by hand cannot smuggle the
/// "ran, found nothing" state back in.
pub fn vector_leg(legs: &[Vec<i64>]) -> Option<&[i64]> {
    legs.get(1).map(Vec::as_slice).filter(|leg| !leg.is_empty())
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
            as_of: None,
        }
    }

    /// The point of transaction-time travel: a memory deleted since is back,
    /// judged by when the deletion happened rather than whether it happened.
    #[test]
    fn as_of_brings_back_what_was_live_then() {
        let conn = db::open_in_memory().unwrap();
        let id = add(
            &conn,
            Kind::Memory,
            None,
            "kafka retention",
            "kafka retention is seven days",
        )
        .id;
        // Created in the past, deleted more recently.
        conn.execute("UPDATE node SET created_at = 1000 WHERE id = ?1", [id])
            .unwrap();
        conn.execute("UPDATE node SET deleted_at = 5000 WHERE id = ?1", [id])
            .unwrap();

        assert!(
            search(&conn, &q("kafka retention")).unwrap().is_empty(),
            "a deleted memory must stay gone in the present tense"
        );

        let mut then = q("kafka retention");
        then.as_of = Some(3000);
        let hits = search(&conn, &then).unwrap();
        assert_eq!(hits.len(), 1, "it was live at 3000: {hits:?}");
        assert_eq!(hits[0].node.id, id);

        let mut before = q("kafka retention");
        before.as_of = Some(500);
        assert!(
            search(&conn, &before).unwrap().is_empty(),
            "it did not exist yet at 500"
        );

        let mut after = q("kafka retention");
        after.as_of = Some(9000);
        assert!(
            search(&conn, &after).unwrap().is_empty(),
            "it was already deleted by 9000"
        );
    }

    /// A supersession has a date only because the mutation ledger records
    /// one; `superseded_by` is a bare column. Without that row this query
    /// could not be answered, which is why the two features landed together.
    #[test]
    fn as_of_judges_supersession_by_when_it_happened() {
        let conn = db::open_in_memory().unwrap();
        let old = add(
            &conn,
            Kind::Memory,
            None,
            "deploy region",
            "deploys go to eu-west-1",
        )
        .id;
        let new = add(
            &conn,
            Kind::Memory,
            None,
            "deploy region",
            "deploys go to eu-north-1",
        )
        .id;
        conn.execute("UPDATE node SET created_at = 1000 WHERE id = ?1", [old])
            .unwrap();
        conn.execute("UPDATE node SET created_at = 4000 WHERE id = ?1", [new])
            .unwrap();
        store::set_superseded(&conn, old, new, crate::model::Actor::Human, None).unwrap();
        // Date the supersession in the ledger the way a real one would be.
        conn.execute(
            "UPDATE mutation SET at = 4000 WHERE node_id = ?1 AND op = 'supersede'",
            [old],
        )
        .unwrap();

        // Present tense: the replacement only.
        let now_hits = search(&conn, &q("deploy region")).unwrap();
        assert_eq!(now_hits.len(), 1);
        assert_eq!(now_hits[0].node.id, new);

        // Before the supersession, the old one was still current — and the
        // replacement had not been written yet.
        let mut then = q("deploy region");
        then.as_of = Some(2000);
        let hits = search(&conn, &then).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "expected only the pre-supersession fact: {hits:?}"
        );
        assert_eq!(hits[0].node.id, old);
    }

    /// As-of recall says it shows "what the store held then". For a memory
    /// edited since, that is the wording it had then: `memory::edit` keeps
    /// it for exactly this query, and until something read it the answer
    /// was today's text under a past date.
    #[test]
    fn as_of_shows_the_wording_a_memory_had_then() {
        use crate::memory::{self, Edit};
        let conn = db::open_in_memory().unwrap();
        let id = add(
            &conn,
            Kind::Memory,
            None,
            "deploy region is eu-west-1",
            "deploy region is eu-west-1",
        )
        .id;
        conn.execute("UPDATE node SET created_at = 1000 WHERE id = ?1", [id])
            .unwrap();
        // Edited now, long after the instant asked about below.
        memory::edit(
            &conn,
            id,
            Edit {
                text: Some("deploy region is eu-north-1".into()),
                ..Default::default()
            },
            crate::model::Actor::Human,
            None,
        )
        .unwrap();
        let untouched = add(
            &conn,
            Kind::Memory,
            None,
            "deploy region for staging",
            "deploy region for staging is local",
        )
        .id;
        conn.execute(
            "UPDATE node SET created_at = 1000 WHERE id = ?1",
            [untouched],
        )
        .unwrap();
        let text_of = |hits: &[Hit], id: i64| {
            let hit = hits.iter().find(|h| h.node.id == id).expect("hit present");
            (
                hit.node.title.clone().unwrap(),
                hit.node.body.clone().unwrap(),
            )
        };

        let now_hits = search(&conn, &q("deploy region")).unwrap();
        assert_eq!(
            text_of(&now_hits, id).1,
            "deploy region is eu-north-1",
            "the present tense reads the current text"
        );

        let mut then = q("deploy region");
        then.as_of = Some(2000);
        let then_hits = search(&conn, &then).unwrap();
        assert_eq!(
            text_of(&then_hits, id),
            (
                "deploy region is eu-west-1".to_string(),
                "deploy region is eu-west-1".to_string()
            ),
            "as of 2000 it still said eu-west-1"
        );
        assert_eq!(
            text_of(&then_hits, untouched).1,
            "deploy region for staging is local",
            "a memory never edited since keeps its one wording"
        );

        // Asked about an instant after the edit, the edit is what was held.
        let mut later = q("deploy region");
        later.as_of = Some(crate::model::now_unix() + 60);
        let later_hits = search(&conn, &later).unwrap();
        assert_eq!(text_of(&later_hits, id).1, "deploy region is eu-north-1");
    }

    /// Valid time and transaction time are different axes. Declaring that a
    /// fact stopped being true must not hide it — that is what `expires_at`
    /// is for, and conflating them would make the label unusable.
    #[test]
    fn a_closed_validity_interval_does_not_hide_the_memory() {
        let conn = db::open_in_memory().unwrap();
        let id = add(&conn, Kind::Memory, None, "diet", "vegetarian since 2019").id;
        store::set_valid_interval(&conn, id, Some(1_550_000_000), Some(1_720_000_000)).unwrap();

        let hits = search(&conn, &q("diet vegetarian")).unwrap();
        assert_eq!(hits.len(), 1, "a no-longer-current fact still surfaces");
        assert!(
            hits[0].node.validity_closed(crate::model::now_unix()),
            "and is marked closed so the reader can see it"
        );
        assert_eq!(hits[0].node.valid_at(1_600_000_000), Some(true));
        assert_eq!(hits[0].node.valid_at(1_800_000_000), Some(false));
    }

    /// The `legs` protocol from the module docs, end to end: a vector slot
    /// exists exactly when the vector leg ranked something.
    #[test]
    fn a_vector_leg_is_reported_only_when_it_ranked_something() {
        let conn = db::open_in_memory().unwrap();
        let id = add(&conn, Kind::Memory, None, "gamma runbook", "restart it").id;
        let embedded_as = |model: &str, vec: &[f32]| {
            conn.execute(
                "INSERT OR REPLACE INTO embedding (node_id, model, dim, vec, content_hash)
                 VALUES (?1, ?2, ?3, ?4, x'00')",
                rusqlite::params![id, model, vec.len() as i64, crate::embed::to_blob(vec)],
            )
            .unwrap();
        };
        let legs_for = |query: &SearchQuery, vector_query: Option<(&str, &[f32])>| {
            search_hybrid_with_legs(&conn, query, vector_query, &mut None)
                .unwrap()
                .1
        };
        let query = q("gamma runbook");
        let live: (&str, &[f32]) = ("live", &[1.0, 0.0]);

        assert_eq!(legs_for(&query, None), vec![vec![id]], "no model");
        assert_eq!(
            legs_for(&query, Some(live)),
            vec![vec![id]],
            "a model, nothing embedded"
        );
        embedded_as("retired", &[1.0, 0.0]);
        assert_eq!(
            legs_for(&query, Some(live)),
            vec![vec![id]],
            "vectors from another model"
        );
        embedded_as("live", &[1.0, 0.0, 0.0]);
        assert_eq!(
            legs_for(&query, Some(live)),
            vec![vec![id]],
            "a dimension the query does not have"
        );
        embedded_as("live", &[1.0, 0.0]);
        let mut docs_only = q("gamma runbook");
        docs_only.kinds = vec![Kind::Chunk];
        assert_eq!(
            legs_for(&docs_only, Some(live)),
            vec![Vec::<i64>::new()],
            "a filter no embedded node passes; the lexical slot stays, empty"
        );

        let legs = legs_for(&query, Some(live));
        assert_eq!(legs, vec![vec![id], vec![id]], "the leg ran");
        assert_eq!(vector_leg(&legs), Some(&[id][..]));
        assert_eq!(vector_leg(&legs[..1]), None);
        assert_eq!(vector_leg(&[vec![id], vec![]]), None);
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

    /// Every scope shape a half of a reach can take, over two projects.
    fn scope_shapes(a: i64, b: i64) -> [Scope; 4] {
        [
            Scope::All,
            Scope::Global,
            Scope::Project(a),
            Scope::Project(b),
        ]
    }

    /// What a search can reach is asked in SQL by the lexical legs and in
    /// Rust by the vector leg. Hold both to one answer: every kind of node,
    /// owned by each of three projects and by none, under every pair of
    /// scopes. And a uniform pair is the old predicate, character for
    /// character, so a search outside a linked worktree runs the statement
    /// it always ran.
    #[test]
    fn reach_has_one_answer_in_sql_and_rust() {
        let conn = db::open_in_memory().unwrap();
        let project = |path: &str| store::ensure_project(&conn, path, "p").unwrap().id;
        let (a, b, c) = (project("/abs/a"), project("/abs/b"), project("/abs/c"));
        let mut nodes = Vec::new();
        for kind in Kind::ALL {
            for owner in [None, Some(a), Some(b), Some(c)] {
                nodes.push(add(&conn, *kind, owner, "a node", "of some kind"));
            }
        }

        let mut split_pairs = 0;
        for memories in scope_shapes(a, b) {
            for code_and_docs in scope_shapes(a, b) {
                let reach = Reach {
                    memories,
                    code_and_docs,
                };
                let mut stmt = conn
                    .prepare(&format!(
                        "SELECT n.id FROM node n WHERE {} ORDER BY n.id",
                        reach.sql()
                    ))
                    .unwrap();
                let in_sql: Vec<i64> = stmt
                    .query_map([], |r| r.get(0))
                    .unwrap()
                    .collect::<rusqlite::Result<_>>()
                    .unwrap();
                // The fixture nodes, and the three project rows themselves
                // (kind `project`, owned by nobody).
                let in_rust: Vec<i64> = [a, b, c]
                    .into_iter()
                    .filter(|_| reach.admits(Kind::Project, None))
                    .chain(
                        nodes
                            .iter()
                            .filter(|n| reach.admits(n.kind, n.project_id))
                            .map(|n| n.id),
                    )
                    .collect();
                assert_eq!(in_sql, in_rust, "{reach:?}");

                if reach.is_uniform() {
                    assert_eq!(reach, Reach::uniform(memories));
                    assert_eq!(reach.sql(), scope_sql(memories), "{reach:?}");
                } else {
                    split_pairs += 1;
                    assert_ne!(reach.sql(), scope_sql(memories), "{reach:?}");
                    assert_ne!(reach.sql(), scope_sql(code_and_docs), "{reach:?}");
                }
            }
        }
        assert_eq!(split_pairs, 12, "every pair of different scopes was asked");

        // The pair a linked worktree has, spelled out once.
        let worktree = Reach {
            memories: Scope::Project(a),
            code_and_docs: Scope::Project(b),
        };
        for kind in Kind::ALL {
            let is_memory = *kind == Kind::Memory;
            assert_eq!(worktree.admits(*kind, Some(a)), is_memory, "{kind}");
            assert_eq!(worktree.admits(*kind, Some(b)), !is_memory, "{kind}");
            assert!(!worktree.admits(*kind, Some(c)), "{kind}");
            assert!(worktree.admits(*kind, None), "{kind}: global is in reach");
            assert_eq!(
                worktree.scope_of(*kind),
                worktree.scope_of_kind_named(kind.as_str())
            );
        }
        assert_eq!(
            worktree.scope_of_kind_named("a-kind-from-a-newer-build"),
            worktree.code_and_docs
        );
    }

    /// A linked worktree's store: the repository's project (`home`) holds
    /// the memories and the main checkout's own code and docs; the
    /// worktree's project (`own`) holds the code and docs of its branch.
    /// Every node says "quorum", so one query finds whatever is in reach.
    struct TwoProjects {
        home: i64,
        own: i64,
        /// title → id, for every node of the fixture.
        ids: HashMap<&'static str, i64>,
    }

    impl TwoProjects {
        fn new(conn: &Connection) -> TwoProjects {
            let home = store::ensure_project(conn, "/repo/main", "main")
                .unwrap()
                .id;
            let own = store::ensure_project(conn, "/repo/wt", "wt").unwrap().id;
            let other = store::ensure_project(conn, "/elsewhere", "other")
                .unwrap()
                .id;
            let fixture: [(&'static str, Kind, Option<i64>); 12] = [
                ("home memory", Kind::Memory, Some(home)),
                ("global memory", Kind::Memory, None),
                ("stray memory", Kind::Memory, Some(own)),
                ("other memory", Kind::Memory, Some(other)),
                ("own symbol", Kind::Symbol, Some(own)),
                ("own doc", Kind::Chunk, Some(own)),
                ("own code", Kind::CodeChunk, Some(own)),
                ("global doc", Kind::Chunk, None),
                ("home symbol", Kind::Symbol, Some(home)),
                ("home doc", Kind::Chunk, Some(home)),
                ("home code", Kind::CodeChunk, Some(home)),
                ("other symbol", Kind::Symbol, Some(other)),
            ];
            let ids = fixture
                .into_iter()
                .map(|(title, kind, owner)| {
                    let body = format!("quorum loss, as the {title} tells it");
                    (title, add(conn, kind, owner, title, &body).id)
                })
                .collect();
            TwoProjects { home, own, ids }
        }

        fn reach(&self) -> Reach {
            Reach {
                memories: Scope::Project(self.home),
                code_and_docs: Scope::Project(self.own),
            }
        }

        /// The fixture titles behind `ids`, sorted.
        fn titles(&self, ids: impl IntoIterator<Item = i64>) -> Vec<&'static str> {
            let mut titles: Vec<&'static str> = ids
                .into_iter()
                .map(|id| {
                    *self
                        .ids
                        .iter()
                        .find(|(_, known)| **known == id)
                        .map(|(title, _)| title)
                        .expect("a fixture node")
                })
                .collect();
            titles.sort_unstable();
            titles
        }
    }

    /// The decision in one query: the repository's memories and the
    /// worktree's own symbols, docs and code come back together, fused once;
    /// the main checkout's symbols, docs and code do not, and neither does a
    /// memory filed under the worktree's own row or anything of a third
    /// project.
    #[test]
    fn one_search_reads_memories_from_one_project_and_code_and_docs_from_another() {
        let conn = db::open_in_memory().unwrap();
        let store = TwoProjects::new(&conn);
        let mut query = q("quorum");
        query.limit = 50;
        query.scope = Scope::Project(store.home);
        let within = |query: &SearchQuery| {
            let (hits, legs) = search_hybrid_with_legs_scored_within(
                &conn,
                query,
                store.reach(),
                None,
                &mut None,
                0.0,
            )
            .unwrap();
            (
                store.titles(hits.iter().map(|h| h.node.id)),
                store.titles(legs[0].iter().copied()),
            )
        };

        let (hits, lexical_leg) = within(&query);
        assert_eq!(
            hits,
            [
                "global doc",
                "global memory",
                "home memory",
                "own code",
                "own doc",
                "own symbol",
            ]
        );
        assert_eq!(
            lexical_leg, hits,
            "the leg itself is filtered, not the page"
        );

        // The kind filters `recall --kind` offers, each on its own half.
        query.kinds = vec![Kind::Symbol, Kind::CodeChunk];
        assert_eq!(within(&query).0, ["own code", "own symbol"], "--kind code");
        query.kinds = vec![Kind::File, Kind::Chunk, Kind::Annotation];
        assert_eq!(within(&query).0, ["global doc", "own doc"], "--kind doc");
        query.kinds = vec![Kind::Memory];
        assert_eq!(
            within(&query).0,
            ["global memory", "home memory"],
            "--kind memory"
        );

        // Through the functions that take no reach, the query's scope is
        // the scope of everything, as it always was.
        query.kinds = vec![];
        let uniform = |scope: Scope| {
            let mut query = query.clone();
            query.scope = scope;
            store.titles(search(&conn, &query).unwrap().iter().map(|h| h.node.id))
        };
        assert_eq!(
            uniform(Scope::Project(store.home)),
            [
                "global doc",
                "global memory",
                "home code",
                "home doc",
                "home memory",
                "home symbol",
            ]
        );
        assert_eq!(
            uniform(Scope::Project(store.own)),
            [
                "global doc",
                "global memory",
                "own code",
                "own doc",
                "own symbol",
                "stray memory",
            ]
        );
    }

    /// The identifier leg is not in `legs`, so it is asked directly: a
    /// symbol named in the query is found in the worktree's graph and not
    /// in the main checkout's.
    #[test]
    fn the_identifier_leg_reads_the_same_reach() {
        let conn = db::open_in_memory().unwrap();
        let store = TwoProjects::new(&conn);
        let named = |owner: i64, title: &str| {
            add(
                &conn,
                Kind::Symbol,
                Some(owner),
                title,
                "fn retry_with_backoff()",
            )
            .id
        };
        let in_worktree = named(store.own, "retry_with_backoff in the worktree");
        let in_main = named(store.home, "retry_with_backoff in main");
        let query = q("how does retry_with_backoff decide");

        let within = fts::identifier_leg_within(&conn, &query, store.reach(), LEG_CAP).unwrap();
        assert_eq!(within, [in_worktree]);
        let mut everywhere = fts::identifier_leg(&conn, &query, LEG_CAP).unwrap();
        everywhere.sort_unstable();
        assert_eq!(everywhere, [in_worktree, in_main], "uniform: `Scope::All`");

        // And it is that leg, with that reach, the fusion is fed: the main
        // checkout's symbol does not come back through it.
        let (hits, _) = search_hybrid_with_legs_scored_within(
            &conn,
            &query,
            store.reach(),
            None,
            &mut None,
            0.0,
        )
        .unwrap();
        let found: Vec<i64> = hits.iter().map(|h| h.node.id).collect();
        assert!(found.contains(&in_worktree), "{found:?}");
        assert!(!found.contains(&in_main), "{found:?}");
    }

    /// The vector leg filters cached rows in Rust; it has to admit the same
    /// nodes the lexical leg's SQL does. Every node carries the same
    /// vector, so the leg ranks whatever its filter lets through.
    #[test]
    fn the_vector_leg_reads_the_same_reach() {
        let conn = db::open_in_memory().unwrap();
        let store = TwoProjects::new(&conn);
        for id in store.ids.values() {
            conn.execute(
                "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
                 VALUES (?1, 'live', 2, ?2, x'00')",
                rusqlite::params![id, crate::embed::to_blob(&[1.0, 0.0])],
            )
            .unwrap();
        }
        let mut query = q("quorum");
        query.limit = 50;
        let live: (&str, &[f32]) = ("live", &[1.0, 0.0]);
        let mut cache = None;

        let (hits, legs) = search_hybrid_with_legs_scored_within(
            &conn,
            &query,
            store.reach(),
            Some(live),
            &mut cache,
            0.0,
        )
        .unwrap();
        let in_reach = [
            "global doc",
            "global memory",
            "home memory",
            "own code",
            "own doc",
            "own symbol",
        ];
        let semantic = vector_leg(&legs).expect("the vector leg ran");
        assert_eq!(store.titles(semantic.iter().copied()), in_reach);
        assert_eq!(store.titles(legs[0].iter().copied()), in_reach);
        assert_eq!(store.titles(hits.iter().map(|h| h.node.id)), in_reach);

        // A kind filter narrows each half and never crosses to the other.
        query.kinds = vec![Kind::Symbol, Kind::CodeChunk];
        let leg = vector::leg_within(&conn, &mut cache, "live", live.1, &query, store.reach(), 50)
            .unwrap();
        assert_eq!(store.titles(leg), ["own code", "own symbol"]);

        // The leg that takes no reach filters by the query's scope alone.
        query.kinds = vec![];
        query.scope = Scope::Project(store.home);
        let leg = vector::leg(&conn, &mut cache, "live", live.1, &query, 50).unwrap();
        assert_eq!(
            store.titles(leg),
            [
                "global doc",
                "global memory",
                "home code",
                "home doc",
                "home memory",
                "home symbol",
            ]
        );
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
            as_of: None,
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
                as_of: None,
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
