//! Mimir core engine: unified local-first memory for AI coding agents.
//!
//! Everything is a node (memories, doc chunks, files, code symbols,
//! projects, tags) in one SQLite database, searched by hybrid
//! BM25 + vector retrieval with reciprocal-rank fusion.

pub mod config;
pub mod consolidate;
pub mod db;
pub mod embed;
#[cfg(any(test, feature = "eval"))]
pub mod eval;
pub mod error;
pub mod format;
pub mod import;
pub mod index;
pub mod learn;
pub mod memory;
pub mod model;
pub mod replicate;
pub mod savings;
pub mod scope;
pub mod search;
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
        match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid_with_legs(
                    &self.conn,
                    query,
                    Some((&model, &v)),
                    &mut self.matrix,
                )
            }
            None => search::search_hybrid_with_legs(&self.conn, query, None, &mut self.matrix),
        }
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
    pub fn search_with(&mut self, query: &SearchQuery, rerank: bool) -> Result<Vec<Hit>> {
        let mut fetch_query = query.clone();
        if rerank {
            fetch_query.limit = query.limit.max(self.config.rerank.candidates);
        }

        let query_vec = self.query_embedding(&query.text);
        let mut hits = match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid(
                    &self.conn,
                    &fetch_query,
                    Some((&model, &v)),
                    &mut self.matrix,
                )?
            }
            None => search::search(&self.conn, &fetch_query)?,
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
