//! Mimir core engine: unified local-first memory for AI coding agents.
//!
//! Everything is a node (memories, doc chunks, files, code symbols,
//! projects, tags) in one SQLite database, searched by hybrid
//! BM25 + vector retrieval with reciprocal-rank fusion.

pub mod config;
pub mod db;
pub mod embed;
pub mod error;
pub mod format;
pub mod index;
pub mod memory;
pub mod model;
pub mod scope;
pub mod search;
pub mod store;

use std::path::Path;

use rusqlite::Connection;

use crate::config::{Config, Paths};
use crate::error::Result;
use crate::model::Node;
use crate::search::vector::MatrixCache;
use crate::search::{Hit, SearchQuery};

/// Engine facade. Owns one writer connection; cheap to open.
/// The embedding model loads lazily on first semantic use and is cached
/// for the process lifetime (long-lived in the MCP server).
pub struct Mimir {
    pub conn: Connection,
    pub config: Config,
    pub paths: Paths,
    embedder: Option<embed::Embedder>,
    embedder_tried: bool,
    matrix: Option<MatrixCache>,
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
            matrix: None,
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
            matrix: None,
        })
    }

    /// Lazily load the embedding model. Never downloads unless asked;
    /// failure degrades to BM25-only (logged, not fatal).
    pub fn ensure_embedder(&mut self, allow_download: bool) -> Option<&mut embed::Embedder> {
        let downloadable =
            allow_download || embed::model_ready(&self.paths, &self.config.embedding.model);
        if self.embedder.is_none() && !self.embedder_tried && downloadable {
            self.embedder_tried = true;
            match embed::Embedder::load(&self.paths, &self.config.embedding.model, allow_download) {
                Ok(e) => self.embedder = Some(e),
                Err(err) => tracing::warn!(%err, "embeddings unavailable; BM25-only"),
            }
        }
        self.embedder.as_mut()
    }

    /// Hybrid search when the model is available locally, BM25-only otherwise.
    pub fn search(&mut self, query: &SearchQuery) -> Result<Vec<Hit>> {
        let query_vec = self.ensure_embedder(false).and_then(|e| {
            e.embed(vec![query.text.clone()])
                .map_err(|err| tracing::warn!(%err, "query embedding failed; BM25-only"))
                .ok()
                .and_then(|mut v| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.remove(0))
                    }
                })
        });
        match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid(&self.conn, query, Some((&model, &v)), &mut self.matrix)
            }
            None => search::search(&self.conn, query),
        }
    }

    /// Embed all new/changed content. No-op (0) when no model is present.
    pub fn embed_pending(&mut self) -> Result<usize> {
        self.ensure_embedder(false);
        let Mimir { conn, embedder, .. } = self;
        match embedder.as_mut() {
            Some(e) => embed::embed_pending(conn, e),
            None => Ok(0),
        }
    }

    /// Resolve (and lazily register) the project for a working directory.
    /// Returns None outside any git repository.
    pub fn project_for_cwd(&self, cwd: &Path) -> Result<Option<Node>> {
        let Some(root) = scope::find_project_root(cwd) else {
            return Ok(None);
        };
        let canonical = scope::canonical_root(&root);
        let name = scope::project_name(&root);
        store::ensure_project(&self.conn, &canonical, &name).map(Some)
    }
}
