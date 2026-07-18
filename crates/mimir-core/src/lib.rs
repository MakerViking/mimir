//! Mimir core engine: unified local-first memory for AI coding agents.
//!
//! Everything is a node (memories, doc chunks, files, code symbols,
//! projects, tags) in one SQLite database, searched by hybrid
//! BM25 + vector retrieval with reciprocal-rank fusion.

pub mod anchors;
pub mod config;
pub mod consolidate;
pub mod context_guard;
pub mod db;
pub mod embed;
pub mod error;
#[cfg(any(test, feature = "eval"))]
pub mod eval;
pub mod format;
pub mod import;
pub mod index;
pub mod inject;
pub mod learn;
pub mod memory;
pub mod model;
pub mod replicate;
pub mod savings;
pub mod scope;
pub mod search;
pub mod secrets;
pub mod store;
pub mod tokens;

use std::path::Path;

use rusqlite::Connection;

use crate::config::{Config, Paths};
use crate::error::Result;
use crate::model::Node;
use crate::search::vector::MatrixCache;
use crate::search::{Hit, SearchQuery};

/// Bound for the query-embedding memo: agents repeat queries (retries,
/// follow-ups) and the ~5–10 ms ONNX call is pure waste the second time.
const QUERY_CACHE_CAP: usize = 256;

/// Store-size gate for automatic reranking (config `[rerank] auto`): below
/// this many embeddable nodes, RRF fusion alone is already precise and a
/// cross-encoder pass only adds latency (eval-verified — the fixture
/// corpora used by `eval::tests` are tens of nodes, well clear of this, so
/// they never trip it either way). Kept as a doc-commented constant rather
/// than a config knob — nobody has needed to tune it yet.
const RERANK_MIN_STORE_NODES: i64 = 1000;

/// Engine facade. Owns one writer connection; cheap to open.
/// The embedding model loads lazily on first semantic use and is cached
/// for the process lifetime (long-lived in the MCP server).
pub struct Mimir {
    pub conn: Connection,
    pub config: Config,
    pub paths: Paths,
    embedder: Option<embed::Embedder>,
    embedder_tried: bool,
    reranker: Option<embed::Reranker>,
    reranker_tried: bool,
    matrix: Option<MatrixCache>,
    query_cache: std::collections::HashMap<String, Vec<f32>>,
}

impl Mimir {
    /// Open the standard per-user instance (honors `MIMIR_HOME`).
    pub fn open() -> Result<Self> {
        Self::open_at(Paths::resolve()?)
    }

    pub fn open_at(paths: Paths) -> Result<Self> {
        let config = Config::load(&paths.config_file)?;
        let conn = db::open(&paths.db_file)?;
        Ok(Mimir {
            conn,
            config,
            paths,
            embedder: None,
            embedder_tried: false,
            reranker: None,
            reranker_tried: false,
            matrix: None,
            query_cache: std::collections::HashMap::new(),
        })
    }

    /// In-memory instance for tests.
    pub fn open_in_memory() -> Result<Self> {
        Ok(Mimir {
            conn: db::open_in_memory()?,
            config: Config::default(),
            paths: Paths::under_root(Path::new(".")),
            embedder: None,
            embedder_tried: false,
            reranker: None,
            reranker_tried: false,
            matrix: None,
            query_cache: std::collections::HashMap::new(),
        })
    }

    /// Lazily load the embedding model. Never downloads unless asked;
    /// failure degrades to BM25-only (logged, not fatal).
    pub fn ensure_embedder(&mut self, allow_download: bool) -> Option<&mut embed::Embedder> {
        let downloadable =
            allow_download || embed::model_ready(&self.paths, &self.config.embedding.model);
        if self.embedder.is_none() && !self.embedder_tried && downloadable {
            self.embedder_tried = true;
            match embed::Embedder::load(
                &self.paths,
                &self.config.embedding.model,
                &self.config.embedding.device,
                allow_download,
            ) {
                Ok(e) => self.embedder = Some(e),
                Err(err) => tracing::warn!(%err, "embeddings unavailable; BM25-only"),
            }
        }
        self.embedder.as_mut()
    }

    /// Lazily load the cross-encoder reranker (marker-gated, like the
    /// embedder). Missing model degrades to un-reranked results.
    pub fn ensure_reranker(&mut self, allow_download: bool) -> Option<&mut embed::Reranker> {
        let downloadable =
            allow_download || embed::model_ready(&self.paths, &self.config.rerank.model);
        if self.reranker.is_none() && !self.reranker_tried && downloadable {
            self.reranker_tried = true;
            match embed::Reranker::load(
                &self.paths,
                &self.config.rerank.model,
                &self.config.embedding.device,
                allow_download,
            ) {
                Ok(r) => self.reranker = Some(r),
                Err(err) => tracing::warn!(%err, "reranker unavailable; skipping rerank"),
            }
        }
        self.reranker.as_mut()
    }

    /// True if the cross-encoder reranker is already resident in this
    /// process. Unlike `ensure_reranker`, checking this never triggers a
    /// load — it's what the `[rerank] auto = "warm"` gate uses to decide
    /// whether reranking is "free" right now.
    pub fn reranker_loaded(&self) -> bool {
        self.reranker.is_some()
    }

    /// Whether `search_with`'s automatic reranking (config `[rerank] auto`)
    /// should fire for `query`, given the current process/store state. Only
    /// governs the *default* — an explicit `rerank: true` argument to
    /// `search_with` always reranks regardless of this.
    fn should_auto_rerank(&mut self, query: &SearchQuery) -> bool {
        let resident = match self.config.rerank.auto.as_str() {
            "off" => return false,
            "warm" => self.reranker_loaded(),
            "always" => self.ensure_reranker(false).is_some(),
            other => {
                tracing::warn!(auto = other, "unknown [rerank] auto value; treating as off");
                return false;
            }
        };
        resident && rerank_query_gates_pass(&self.conn, query)
    }

    /// Hybrid search when the model is available locally, BM25-only otherwise.
    pub fn search(&mut self, query: &SearchQuery) -> Result<Vec<Hit>> {
        self.search_with(query, false)
    }

    /// Like [`Mimir::search`], but also returns the raw per-leg ranked id
    /// lists (`legs[0]` = FTS, `legs[1]` = vector iff a model is loaded) —
    /// used by callers that need to check lexical/semantic agreement on a
    /// hit before trusting it (e.g. the auto-recall relevance floor). No
    /// reranking: the floor check needs the legs' own top-K as computed
    /// before any cross-encoder rescoring.
    pub fn search_with_legs(&mut self, query: &SearchQuery) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
        let query_vec = self.query_embedding(&query.text);
        let alpha = self.config.scoring.impression_alpha;
        match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid_with_legs_scored(
                    &self.conn,
                    query,
                    Some((&model, &v)),
                    &mut self.matrix,
                    alpha,
                )
            }
            None => search::search_hybrid_with_legs_scored(
                &self.conn,
                query,
                None,
                &mut self.matrix,
                alpha,
            ),
        }
    }

    /// Like [`Mimir::search_with_legs`], but never touches the embedder —
    /// no `ensure_embedder` call, so no ONNX-load cost, regardless of
    /// whether a model is cached locally. BM25 + identifier legs only.
    /// Used by the cold `mimir recall-inject` CLI path when config
    /// `[hooks] cold_mode = "fast"` (default): measured cold (release
    /// build, bge-small-en cached, 3-memory store) at ~5-6ms end to end,
    /// vs ~230-240ms when the embedder + matrix load first — see
    /// `HooksConfig::cold_mode`'s doc comment for the full sandbox numbers.
    /// `legs[1]` (vector) is always absent from the result —
    /// `inject::clears_floor`'s rule 3 already degrades cleanly when it's
    /// missing (same shape a model-less machine takes today).
    pub fn search_with_legs_bm25_only(
        &mut self,
        query: &SearchQuery,
    ) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
        search::search_hybrid_with_legs_scored(
            &self.conn,
            query,
            None,
            &mut self.matrix,
            self.config.scoring.impression_alpha,
        )
    }

    /// Embed a query, memoized per process. None = no model / embed failed
    /// (search degrades to BM25-only).
    fn query_embedding(&mut self, text: &str) -> Option<Vec<f32>> {
        let started = std::time::Instant::now();
        if let Some(v) = self.query_cache.get(text) {
            tracing::debug!(elapsed = ?started.elapsed(), "query_embedding: cache hit");
            return Some(v.clone());
        }
        let vec = self.ensure_embedder(false).and_then(|e| {
            e.embed(vec![text.to_string()])
                .map_err(|err| tracing::warn!(%err, "query embedding failed; BM25-only"))
                .ok()
                .and_then(|mut v| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.remove(0))
                    }
                })
        })?;
        tracing::debug!(elapsed = ?started.elapsed(), chars = text.len(), "query_embedding: cache miss (model call)");
        if self.query_cache.len() >= QUERY_CACHE_CAP {
            self.query_cache.clear();
        }
        self.query_cache.insert(text.to_string(), vec.clone());
        Some(vec)
    }

    /// Search with optional cross-encoder reranking: over-fetch the fused
    /// candidates, rescore them against the query, keep the top `limit`.
    /// `rerank = true` always reranks (cold-loading the model if needed,
    /// same as today); `rerank = false` still reranks if config
    /// `[rerank] auto` and the per-query gates say it should — see
    /// `should_auto_rerank`.
    pub fn search_with(&mut self, query: &SearchQuery, rerank: bool) -> Result<Vec<Hit>> {
        let rerank = rerank || self.should_auto_rerank(query);
        let mut fetch_query = query.clone();
        if rerank {
            fetch_query.limit = query.limit.max(self.config.rerank.candidates);
        }

        let query_vec = self.query_embedding(&query.text);
        let alpha = self.config.scoring.impression_alpha;
        let mut hits = match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid_scored(
                    &self.conn,
                    &fetch_query,
                    Some((&model, &v)),
                    &mut self.matrix,
                    alpha,
                )?
            }
            // No embedder available: still route through the scored path
            // (vector_query = None) so the impression damp still applies —
            // otherwise a cold/model-less process would silently bypass a
            // configured impression_alpha instead of degrading to BM25-only
            // *and* keeping the damp, same as `search_with_legs` does.
            None => search::search_hybrid_scored(
                &self.conn,
                &fetch_query,
                None,
                &mut self.matrix,
                alpha,
            )?,
        };

        if rerank && hits.len() > 1 {
            if let Some(rr) = self.ensure_reranker(false) {
                let docs: Vec<String> = hits
                    .iter()
                    .map(|h| {
                        let title = h.node.title.as_deref().unwrap_or("");
                        let body = h.node.body.as_deref().unwrap_or("");
                        // Cross-encoders cap at ~512 tokens; keep pairs cheap.
                        format!("{title}\n{body}").chars().take(1500).collect()
                    })
                    .collect();
                match rr.rerank(&query.text, docs) {
                    Ok(ranked) => {
                        let mut slots: Vec<Option<Hit>> = hits.into_iter().map(Some).collect();
                        let mut reordered = Vec::with_capacity(slots.len());
                        for (idx, score) in ranked {
                            if let Some(mut hit) = slots.get_mut(idx).and_then(Option::take) {
                                hit.score = score as f64;
                                reordered.push(hit);
                            }
                        }
                        // Anything the reranker didn't score keeps fused order.
                        reordered.extend(slots.into_iter().flatten());
                        hits = reordered;
                    }
                    Err(err) => {
                        tracing::warn!(%err, "rerank failed; keeping fused order");
                    }
                }
            }
        }
        hits.truncate(query.limit);
        Ok(hits)
    }

    /// Embed all new/changed content. No-op (0) when no model is present.
    /// Patches the vector matrix cache in place for a small batch (the
    /// common `remember`-one-node case) instead of dropping it — same-
    /// connection writes are invisible to `data_version`, so a dropped
    /// cache would otherwise pay a full corpus reload on the next recall.
    /// A large batch (bulk import/reindex) still drops it: patching would
    /// cost about as much as a rebuild, for no benefit.
    pub fn embed_pending(&mut self) -> Result<usize> {
        self.ensure_embedder(false);
        let Mimir {
            conn,
            embedder,
            matrix,
            config,
            ..
        } = self;
        let ids = match embedder.as_mut() {
            Some(e) => embed::embed_pending(conn, e)?,
            None => Vec::new(),
        };
        // Above this, a linear-scan patch (see MatrixCache::upsert) approaches
        // the cost of just rebuilding; bulk operations aren't latency-sensitive
        // the way interactive recall is, so fall back to the old behavior.
        const PATCH_THRESHOLD: usize = 200;
        if !ids.is_empty() {
            if ids.len() <= PATCH_THRESHOLD {
                MatrixCache::upsert(conn, &config.embedding.model, &ids, matrix)?;
            } else {
                *matrix = None;
            }
        }
        Ok(ids.len())
    }

    /// Resolve (and lazily register) the project for a working directory, plus
    /// *how* it was detected. `Found` ⇒ the project node is ensured (created on
    /// first contact); `NotFound` ⇒ global scope, which the caller explains
    /// rather than degrading silently.
    pub fn detect_project(&self, cwd: &Path) -> Result<(Option<Node>, scope::Detection)> {
        let detection = scope::detect(cwd);
        let node = match &detection {
            scope::Detection::Found { root, .. } => {
                let canonical = scope::canonical_root(root);
                let name = scope::project_name(root);
                let key = scope::portable_key(root);
                let sync = scope::sync_opt_in(root);
                // If a project with this portable key already exists (e.g. a
                // shadow created by sync apply before this checkout was opened),
                // adopt it onto this machine's path instead of making a duplicate.
                let node = match key
                    .as_deref()
                    .map(|k| store::find_project_by_key(&self.conn, k))
                    .transpose()?
                    .flatten()
                {
                    Some(existing) => {
                        store::adopt_project(&self.conn, existing.id, &canonical, &name)?;
                        store::get_node(&self.conn, existing.id)?
                    }
                    None => store::ensure_project(&self.conn, &canonical, &name)?,
                };
                // Keep the project's portable sync identity fresh in meta (cheap,
                // no-op when unchanged) so project-scoped sync can resolve it by a
                // machine-stable key without touching the filesystem.
                store::set_project_identity(&self.conn, node.id, key.as_deref(), sync)?;
                Some(node)
            }
            scope::Detection::NotFound { .. } => None,
        };
        Ok((node, detection))
    }

    /// Project node for a working directory, or None outside any project.
    /// Thin wrapper over [`Mimir::detect_project`] for callers that don't need
    /// the detection reason.
    pub fn project_for_cwd(&self, cwd: &Path) -> Result<Option<Node>> {
        Ok(self.detect_project(cwd)?.0)
    }
}

/// Per-query gates for [`Mimir::should_auto_rerank`], independent of
/// whether the reranker model is resident: a single-token/exact-identifier
/// query or a tiny store are both cases where a cross-encoder only adds
/// cost. Split out into a free function (rather than inlined in
/// `should_auto_rerank`) so it's unit-testable without a downloaded model.
fn rerank_query_gates_pass(conn: &Connection, query: &SearchQuery) -> bool {
    // A single-token / exact-identifier query (`add_saturating`) is
    // already an exact-ish lexical match — the FTS+identifier legs nail it
    // directly, and a cross-encoder's fuzzy judgment can only reshuffle a
    // correct top hit downward, never sharpen it further.
    if query.text.split_whitespace().count() <= 1 {
        return false;
    }
    // A tiny store is already precise under RRF alone; see
    // RERANK_MIN_STORE_NODES's doc comment.
    !store_too_small_for_rerank(conn)
}

/// Store-size gate for [`Mimir::should_auto_rerank`]: counts embeddable
/// nodes (the same kinds `embed_pending` embeds) via the existing
/// `node(kind, project_id)` partial index, so this stays cheap even on a
/// large store — it's on the hot recall path.
fn store_too_small_for_rerank(conn: &Connection) -> bool {
    conn.query_row(
        &format!(
            "SELECT COUNT(*) FROM node WHERE deleted_at IS NULL AND kind IN ({})",
            embed::EMBEDDABLE_KINDS
        ),
        [],
        |r| r.get::<_, i64>(0),
    )
    // DB error here is unexpected (the same connection just ran the
    // search) — fail closed (skip rerank) rather than risk it on a query
    // that already hit trouble.
    .map(|n| n < RERANK_MIN_STORE_NODES)
    .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode};

    fn open() -> Mimir {
        Mimir::open_in_memory().unwrap()
    }

    fn q(text: &str) -> SearchQuery {
        SearchQuery {
            text: text.into(),
            scope: crate::model::Scope::All,
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
    fn auto_rerank_off_never_fires() {
        let mut m = open();
        m.config.rerank.auto = "off".into();
        // Even a huge store / multi-word query must not trip it — "off" is
        // an unconditional short-circuit before any of the other gates run.
        assert!(!m.should_auto_rerank(&q("a real multi word query")));
    }

    #[test]
    fn auto_rerank_warm_requires_already_loaded() {
        let mut m = open();
        m.config.rerank.auto = "warm".into();
        // No reranker model on disk in this test environment, and "warm"
        // must never itself trigger a load — `reranker_loaded` stays false,
        // so this returns false without ever calling `ensure_reranker`.
        assert!(!m.reranker_loaded());
        assert!(!m.should_auto_rerank(&q("a real multi word query")));
        assert!(
            !m.reranker_loaded(),
            "the warm gate must never load the model itself"
        );
    }

    #[test]
    fn auto_rerank_gates_single_token_queries() {
        // Independent of model residency or store size — this is exactly
        // what `rerank_query_gates_pass` isolates.
        let conn = db::open_in_memory().unwrap();
        assert!(!rerank_query_gates_pass(&conn, &q("resolve_ref")));
        assert!(!rerank_query_gates_pass(&conn, &q("")));
        assert!(!rerank_query_gates_pass(&conn, &q("   ")));
    }

    #[test]
    fn auto_rerank_gates_tiny_store() {
        let conn = db::open_in_memory().unwrap();
        // Multi-word query clears the token gate, so an empty store is what
        // trips this one.
        assert!(store_too_small_for_rerank(&conn));
        assert!(!rerank_query_gates_pass(
            &conn,
            &q("a real multi word query")
        ));
        // Seed past the threshold and confirm the gate flips.
        for i in 0..(RERANK_MIN_STORE_NODES as usize + 1) {
            let mut n = NewNode::new(Kind::Memory);
            n.title = Some(format!("seed {i}"));
            n.body = Some("filler".into());
            store::insert_node(&conn, n).unwrap();
        }
        assert!(!store_too_small_for_rerank(&conn));
        assert!(rerank_query_gates_pass(
            &conn,
            &q("a real multi word query")
        ));
    }
}
