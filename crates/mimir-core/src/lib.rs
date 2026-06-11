//! Mimir core engine: unified local-first memory for AI coding agents.
//!
//! Everything is a node (memories, doc chunks, files, code symbols,
//! projects, tags) in one SQLite database, searched by hybrid
//! BM25 + vector retrieval with reciprocal-rank fusion.

pub mod config;
pub mod db;
pub mod error;
pub mod format;
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

/// Engine facade. Owns one writer connection; cheap to open.
pub struct Mimir {
    pub conn: Connection,
    pub config: Config,
    pub paths: Paths,
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
        })
    }

    /// In-memory instance for tests.
    pub fn open_in_memory() -> Result<Self> {
        Ok(Mimir {
            conn: db::open_in_memory()?,
            config: Config::default(),
            paths: Paths::under_root(Path::new(".")),
        })
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
