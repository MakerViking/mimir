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
    pub auto: AutoConfig,
    pub sync: SyncConfig,
    pub proxy: ProxyConfig,
    pub savings: SavingsConfig,
    pub hooks: HooksConfig,
}

/// Settings for the opt-in Claude Code hooks (`mimir init --hooks`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HooksConfig {
    /// Base URL for the warm `/inject` endpoint the auto-recall
    /// (`--auto-recall`) UserPromptSubmit hook tries first, falling back to
    /// the cold `mimir recall-inject` CLI path when unreachable — only live
    /// while `mimir mcp --http` is running (see `mcp.rs::inject_router`).
    /// Baked into the generated hook script at install/re-init time; the
    /// `MIMIR_INJECT_URL` environment variable (read at hook-invocation
    /// time, not install time) still overrides it as the highest-precedence
    /// escape hatch — e.g. a per-machine bind without editing config.toml.
    /// Default matches the port documented in README.md's remote-MCP
    /// example (`127.0.0.1:8077`).
    pub inject_url: String,
}

impl Default for HooksConfig {
    fn default() -> Self {
        HooksConfig {
            inject_url: "http://127.0.0.1:8077/inject".into(),
        }
    }
}

/// Pricing for the token-savings analytics (`mimir savings`, dashboard).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SavingsConfig {
    /// Input-token price in USD per million tokens, used to turn tokens saved
    /// into dollars. Default is Sonnet-tier ($3/MTok); set to your model's
    /// input price (e.g. 15.0 for Opus) for an accurate figure.
    pub input_price_per_mtok: f64,
}

impl Default for SavingsConfig {
    fn default() -> Self {
        SavingsConfig {
            input_price_per_mtok: 3.0,
        }
    }
}

/// OPTIONAL local API proxy (`mimir proxy`). Off until you run it; these are
/// just its defaults. See docs/proxy.md — it is a man-in-the-middle on your
/// Anthropic traffic, so it ships conservative (cache on, pruning off).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProxyConfig {
    /// Address to listen on; point ANTHROPIC_BASE_URL here.
    pub bind: String,
    /// Upstream API base URL.
    pub upstream: String,
    /// Add prompt-cache breakpoints when the client set none (safe, ~10% billing).
    pub cache: bool,
    /// Replace later identical large blocks with a placeholder (safe, lossless).
    pub dedup: bool,
    /// Prune stale tool_result blocks from older turns (lossy — opt in).
    pub prune: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        ProxyConfig {
            bind: "127.0.0.1:8788".into(),
            upstream: "https://api.anthropic.com".into(),
            cache: true,
            dedup: true,
            prune: false,
        }
    }
}

/// OPTIONAL centralized sync of global memories. Off by default — when
/// `mode = "off"` nothing runs and there is zero added cost. See docs/sync.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    /// "off" (default) | "file" (a replicated folder) | "server" (a `mimir
    /// serve` hub).
    pub mode: String,
    /// file mode: a directory your file-sync tool (Syncthing/Dropbox/…)
    /// replicates between machines.
    pub dir: String,
    /// server mode: the hub URL, e.g. "http://unraid.tailnet:7777". The bearer
    /// token comes from the MIMIR_SYNC_TOKEN env var, never config.
    pub endpoint: String,
    /// Background-sync cadence in minutes (only when `auto = true`).
    pub interval_mins: u64,
    /// Run a background sync loop in the MCP server (opt-in).
    pub auto: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        SyncConfig {
            mode: "off".into(),
            dir: String::new(),
            endpoint: String::new(),
            interval_mins: 30,
            auto: false,
        }
    }
}

impl SyncConfig {
    pub fn enabled(&self) -> bool {
        self.mode != "off"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoConfig {
    /// Build/refresh the project's code graph when the MCP server starts
    /// in it (incremental: first contact extracts, afterwards no-ops in ms).
    pub graph: bool,
    /// Register + index the project's own markdown as a docs collection
    /// when the MCP server starts in it (gitignore-aware, incremental).
    pub docs: bool,
}

impl Default for AutoConfig {
    fn default() -> Self {
        AutoConfig {
            graph: true,
            docs: true,
        }
    }
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
    /// Recency boost weight; 0.0 = off. Fresher memories get a small
    /// multiplicative nudge, capped so it never buries a stronger match.
    /// Applied only to kinds/subkinds with a defined half-life (decision,
    /// gotcha, insight, note, idea memories — see `learn::half_life_days`);
    /// non-decaying kinds (docs, code, person/summary memories) get NO
    /// recency term at all, so they stay recency-neutral relative to aging
    /// memories instead of enjoying a permanent boost. Default 0.012: at
    /// this gated formula, a *fresh* decaying-kind memory gets up to a full
    /// `1+alpha` multiplier while every non-decaying kind sits flat at 1.0 —
    /// a real, uncancelled edge (unlike the pre-gating bug, where both got
    /// the same flat boost, so freshness never had to compete against
    /// symbols/docs/code). 0.4 was tuned against that ungated formula and no
    /// longer holds once gating is correct: on real embeddings, where
    /// score gaps are already tight, anything above ~0.015 lets a fresh,
    /// off-topic memory outrank a symbol/doc/code hit that's the actual
    /// answer (0.02 alone regressed real-model `core` MRR to 0.94; 0.4
    /// regressed hermetic `core` to 0.73). 0.012 is eval-tuned to be the
    /// largest weight that clears `core` at 1.0 on *both* hermetic and
    /// real-model modes, while still deciding a stale-vs-current Memory
    /// pair (e.g. a superseded-in-practice decision nobody marked
    /// superseded) and letting a still-relevant aged gotcha hold its rank
    /// against fresher, topically-adjacent doc/code noise.
    pub recency_alpha: f64,
    /// Type-priority weight; 0.0 = off. Nudges gotcha/decision memories —
    /// drift-preventers and policy calls — ahead of an equally-matching but
    /// non-critical note/idea/insight. The per-subkind prior is fixed (see
    /// `learn::type_prior`); this only scales how strongly it counts.
    /// Default 0.12: eval-tuned against real embeddings, not synthetic ones —
    /// the same weight that's safe on hermetic (clustered, well-separated)
    /// vectors already reorders real, fuzzier score gaps hard enough to bury
    /// a correct code-structure answer under a topically-adjacent decoy at
    /// 0.15+. 0.12 sits just under that observed cliff.
    pub type_prior_alpha: f64,
    /// Sub-1.0 multiplier applied to `Kind::CodeChunk` hits at the fused
    /// score. Code content dwarfs memories on a real project (~99% of
    /// nodes vs ~1%), so undamped it drowns out memories in the default
    /// `all` search pool even when a memory is the better answer. Default
    /// 0.85: the `eval::hermetic`/`eval::real_model` fixture corpus has no
    /// `CodeChunk` cases (yet), so this isn't eval-swept — it's verified by
    /// a real end-to-end smoke test (index this repo's own `crates/`, query
    /// a term shared by a memory and a code chunk) where 0.85 flips a tied
    /// ranking to favor the memory, while `code_damp = 1.0` does not. Both
    /// eval modes confirm this addition doesn't regress the existing
    /// (non-code) `core` scenario set.
    pub code_damp: f64,
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
            recency_alpha: 0.012,
            type_prior_alpha: 0.12,
            code_damp: 0.85,
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
