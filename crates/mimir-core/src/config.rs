use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Resolved filesystem locations for the Mimir instance.
#[derive(Debug, Clone)]
pub struct Paths {
    pub config_file: PathBuf,
    pub db_file: PathBuf,
    pub models_dir: PathBuf,
}

impl Paths {
    /// Standard per-user locations via XDG / AppData / Library conventions.
    pub fn standard() -> Result<Self> {
        let dirs = directories::ProjectDirs::from("", "", "mimir")
            .ok_or_else(|| Error::Config("cannot resolve home directory".into()))?;
        Ok(Paths {
            config_file: dirs.config_dir().join("config.toml"),
            db_file: dirs.data_dir().join("mimir.db"),
            models_dir: dirs.cache_dir().join("models"),
        })
    }

    /// All state under one root — used by tests and `MIMIR_HOME`.
    pub fn under_root(root: &Path) -> Self {
        Paths {
            config_file: root.join("config.toml"),
            db_file: root.join("mimir.db"),
            models_dir: root.join("models"),
        }
    }

    /// Honors `MIMIR_HOME` (everything under one dir) else standard locations.
    pub fn resolve() -> Result<Self> {
        match std::env::var_os("MIMIR_HOME") {
            Some(root) => Ok(Self::under_root(Path::new(&root))),
            None => Self::standard(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub embedding: EmbeddingConfig,
    pub rerank: RerankConfig,
    pub scoring: ScoringConfig,
    pub output: OutputConfig,
    pub consolidate: ConsolidateConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// fastembed model name; switching models requires `mimir embed --all`.
    pub model: String,
    /// "auto" uses a GPU execution provider when one was compiled in
    /// (--features gpu-webgpu / gpu-cuda), falling back to CPU if it
    /// fails to initialize; "cpu" forces CPU even in a GPU build.
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsolidateConfig {
    /// "weekly" runs the four consolidation passes automatically
    /// (debounced, off the request path); "off" disables.
    pub auto: String,
}

impl Default for ConsolidateConfig {
    fn default() -> Self {
        ConsolidateConfig {
            auto: "weekly".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// fastembed cross-encoder used by `recall --rerank`.
    pub model: String,
    /// How many fused candidates the reranker scores.
    pub candidates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringConfig {
    /// Multiplier weight for learned strength in final ranking.
    pub strength_alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub default_limit: usize,
    pub snippet_chars: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            model: "bge-small-en-v1.5".into(),
            device: "auto".into(),
        }
    }
}

impl Default for RerankConfig {
    fn default() -> Self {
        RerankConfig {
            model: "jina-reranker-v1-turbo-en".into(),
            // Cost is linear in candidates (~65 ms each on CPU); 15 keeps
            // --rerank around a second while still covering the realistic
            // winners. Raise for deeper reshuffles.
            candidates: 15,
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            strength_alpha: 0.15,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            default_limit: 8,
            snippet_chars: 120,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                toml::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(Error::io(path, e)),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("serialize config: {e}")))?;
        std::fs::write(path, text).map_err(|e| Error::io(path, e))
    }
}
