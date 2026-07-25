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
    pub learn: LearnConfig,
    pub daemon: DaemonConfig,
    pub brief: BriefConfig,
}

/// Settings for the session brief (`mimir brief show`, SessionStart hook):
/// a hard-capped digest of the project's most drift-preventing memories
/// (gotchas + decisions), fired once at session start and again after a
/// context clear/compaction. Design doc: docs/design/session-brief.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BriefConfig {
    /// Master switch, checked FIRST by `mimir brief show` (the hook entry
    /// is installed regardless, so flipping this off is the whole
    /// kill-switch — same rollback pattern as `hooks.context_guard`).
    /// Default true since 2026-07-25: the §9 eval gate passed with citable
    /// numbers — rules-pack baseline beaten (96.0 vs 4.0 pts catch@150),
    /// zero forbidden renders, marginal-catch curve calibrating this cap,
    /// and repeated-exposure compliance measured flat (0.889 at exposures
    /// 1/5/10, degradation within noise; preregistered Norn arc) — see
    /// docs/design/session-brief.md §9. Configs written by `mimir init`
    /// before the flip carry an explicit `enabled = false` and keep it.
    pub enabled: bool,
    /// Hard token cap for the startup fire, enforced by construction in
    /// the budget-fill loop (`tokens::count` over the cumulative rendered
    /// text). Whichever of this and `max_items` binds first wins.
    pub max_tokens: usize,
    /// Smaller cap for re-fires on clear/compact — the reset needs a
    /// reminder, not the full brief. Worst-case session cost is the
    /// stated constant `max_tokens + (max_fires_per_session-1) *
    /// recap_tokens` (450 tokens at defaults).
    pub recap_tokens: usize,
    /// Line-count cap, independent of tokens: many tiny gotchas must not
    /// balloon the brief into a list the agent skims rather than absorbs.
    pub max_items: usize,
    /// Only fires that actually render at least one line consume this
    /// budget — a zero-gotcha project must not exhaust it on empty
    /// attempts before its first real capture.
    pub max_fires_per_session: usize,
    /// Per-item character bound applied BEFORE the budget-fill loop, so
    /// the fill decision is only ever "does this whole bounded line fit"
    /// — never a mid-item cut that could read as a different claim.
    pub line_chars: usize,
    /// Relevance gate for GLOBAL memories when briefing a project:
    /// silence beats irrelevant filler (dogfood-labeled: a project with no
    /// gotchas of its own was getting other projects' gotchas as filler).
    /// A global candidate must share a significant tag/title word with the
    /// project's signature (project title + its own memories' tags and
    /// titles) to be admitted; a project with no signal of its own admits
    /// NO globals — un-judgeable relevance is treated as irrelevant.
    /// A PINNED global bypasses the gate (an explicit pin outranks the
    /// heuristic) but still competes under ranking and the token budget.
    /// Deliberately lexical, not vector-based: cosine against a stored-
    /// embedding project centroid was measured on the real store first and
    /// REJECTED — bge cosines between unrelated technical memories compress
    /// into ~0.62-0.79 and rank labeled-irrelevant items above relevant
    /// ones, so no threshold separates (measurement 2026-07-23, see
    /// docs/design/session-brief.md §3). Project-scoped candidates are
    /// never gated; briefs in global scope are never gated; `false`
    /// restores ungated pre-gate behavior.
    pub global_gate: bool,
    /// Relative quality floor: a candidate must score at least this
    /// fraction of the best ELIGIBLE candidate's score (measured before
    /// session suppression removes anything) to be rendered. Without it,
    /// rank is purely relative — once suppression rotates past the strong
    /// guards, "best available" quietly becomes "weak tail", and the brief
    /// serves scraped-bottom items rather than staying silent (observed in
    /// first-day dogfooding). Interaction to know: the floor is measured
    /// on the full score INCLUDING the 1.5x same-project affinity term, so
    /// when a project has guards of its own, unpinned cross-project
    /// memories sit below the default floor by construction — the brief
    /// becomes project-first, and a global needs a PIN to compete. That is
    /// deliberate (silence beats borrowed filler); pin the global if it
    /// genuinely guards this project too. Default 0.75, calibrated on the
    /// labeled dogfood cases; the drift eval refines it. `0.0` disables.
    pub score_floor: f64,
}

impl Default for BriefConfig {
    fn default() -> Self {
        BriefConfig {
            enabled: true,
            max_tokens: 150,
            recap_tokens: 100,
            max_items: 6,
            max_fires_per_session: 4,
            line_chars: 100,
            global_gate: true,
            score_floor: 0.75,
        }
    }
}

/// Settings for `mimir daemon` inference delegation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonConfig {
    /// Whether per-session MCP servers delegate GPU inference (bulk
    /// embedding + reranking) to a running `mimir daemon`, reached at the
    /// same address as `hooks.inject_url` (one setting decides where the
    /// daemon lives; bearer token from MIMIR_HTTP_TOKEN, same as the
    /// daemon itself reads).
    /// - "auto" (default): delegate when the daemon answers. Sessions then
    ///   load their own embedder CPU-only — measured *faster* than GPU for
    ///   batch-1 query embeds — and never load a local reranker while the
    ///   daemon is reachable, so a session holds zero GPU memory. Daemon
    ///   down at session start: exactly the old behavior ([rerank] auto =
    ///   "warm" eager-loads a local reranker on the configured device);
    ///   daemon dies mid-session: embeds fall back to the session's CPU
    ///   embedder in the same call, reranks degrade to fused order, and
    ///   delegation resumes on its own when the daemon returns. The only
    ///   deliberate cost: with no daemon at all, background bulk indexing
    ///   (first contact with a big repo) runs on CPU instead of GPU.
    /// - "off": pre-daemon behavior — every process loads its own models
    ///   on the [embedding] device and nothing is delegated.
    pub inference: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig {
            inference: "auto".into(),
        }
    }
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
    /// Governs ONLY the cold `mimir recall-inject` CLI fallback (no warm
    /// daemon reachable) — never the warm `/inject` HTTP endpoint, and
    /// never a normal `mimir recall`, both of which always do full hybrid
    /// search regardless of this setting.
    /// - `"fast"` (default): skip the embedder entirely — BM25 +
    ///   identifier legs only. Measured cold (release build, bge-small-en
    ///   already cached locally, 3-memory store): ~5-6ms end to end, vs
    ///   ~230-240ms for `"full"` on the identical prompt/store — about 40x,
    ///   not just "faster". The cost: it never fires on a purely-semantic
    ///   match (no shared lexical/identifier token with the prompt) while
    ///   cold. `inject::clears_floor` already degrades cleanly with no
    ///   vector leg (same path a model-less machine takes today), so this
    ///   doesn't weaken the silence-beats-wrong-injection contract — it
    ///   just narrows which real matches clear the floor cold.
    /// - `"full"`: restores hybrid search (embeds the query when a model
    ///   is available locally) on the cold path too, same as before this
    ///   setting existed.
    pub cold_mode: String,
    /// Context-window guard mode: `"off"` (default) | `"pause"` | `"handoff"`.
    /// See `mimir_core::context_guard` for the estimation/messaging logic and
    /// `mimir context-guard` (crates/mimir-cli) for the hook subcommands that
    /// consume it. `"off"` installs no new hook entries and changes nothing.
    /// `"pause"` nudges the user to deliberately `/clear`/`/compact` once the
    /// transcript crosses `context_guard_threshold_pct`. `"handoff"`
    /// instructs the agent to write a structured handoff memory via `mimir
    /// remember` before the user clears, then auto-restores it on the next
    /// `SessionStart`.
    pub context_guard: String,
    /// Percent of the estimated context window (see `context_window_tokens`)
    /// at which the guard first fires. Default 45 — well before Claude
    /// Code's own auto-compact, so there is time to act deliberately instead
    /// of losing control of *when* a compact happens.
    pub context_guard_threshold_pct: u8,
    /// The model's context window, in tokens, used as the denominator for
    /// the guard's percent estimate. Default 200_000 (Claude's standard
    /// window); raise it for a 1M-context model.
    pub context_window_tokens: u32,
    /// Approximate bytes-per-token used to convert the transcript file's
    /// byte size into an estimated token count — cheap (no real
    /// tokenization of a potentially large transcript on every prompt) but
    /// deliberately approximate. Default 8.0 is a rough transcript-JSONL
    /// average; tune it if your own transcripts run leaner/richer than that.
    pub transcript_bytes_per_token: f32,
}

impl Default for HooksConfig {
    fn default() -> Self {
        HooksConfig {
            inject_url: "http://127.0.0.1:8077/inject".into(),
            cold_mode: "fast".into(),
            context_guard: "off".into(),
            context_guard_threshold_pct: 45,
            context_window_tokens: 200_000,
            transcript_bytes_per_token: 8.0,
        }
    }
}

/// The lowest `event_retention_days` allowed to take effect as configured
/// (see [`LearnConfig::effective_retention_days`]) — below this, a shorter
/// window would start truncating the very history `impression_damp` (30-day
/// decay half-life) needs to compute against.
const MIN_SAFE_RETENTION_DAYS: u32 = 60;

/// Settings for the implicit-feedback ledger (`recall_event`: shown/opened,
/// logged by every recall/inject — see `store::record_shown`,
/// `learn::record_opened`, `scoring.impression_alpha`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LearnConfig {
    /// Days of `recall_event` history to keep before a background prune
    /// deletes older rows (see `db::spawn_checkpoint_timer`, which runs the
    /// prune on the same idle cadence as the WAL checkpoint, plus once at
    /// daemon startup). `0` disables pruning — keep forever.
    ///
    /// Default 180: `impression_damp`'s decay half-life is 30 days, so 180
    /// keeps 6 half-lives of signal, comfortably more than the decay ever
    /// weights meaningfully. See [`effective_retention_days`] for the safety
    /// floor on configuring this too low.
    ///
    /// [`effective_retention_days`]: LearnConfig::effective_retention_days
    pub event_retention_days: u32,
}

impl Default for LearnConfig {
    fn default() -> Self {
        LearnConfig {
            event_retention_days: 180,
        }
    }
}

impl LearnConfig {
    /// The retention window a prune should actually use: `None` means never
    /// prune (explicit `0`); otherwise the configured value, clamped up to
    /// [`MIN_SAFE_RETENTION_DAYS`] with a `tracing::warn!` if it was
    /// configured lower — a too-short window would silently truncate the
    /// history `impression_damp` needs, degrading a ranking signal without
    /// any visible symptom. `0` is exempt from the floor: it's an explicit
    /// "never prune" opt-out, not an accidentally-tiny window.
    pub fn effective_retention_days(&self) -> Option<u32> {
        match self.event_retention_days {
            0 => None,
            n if n < MIN_SAFE_RETENTION_DAYS => {
                tracing::warn!(
                    configured_days = n,
                    clamped_to = MIN_SAFE_RETENTION_DAYS,
                    "learn.event_retention_days is below the safe floor for \
                     impression_damp's 30-day decay half-life; clamping"
                );
                Some(MIN_SAFE_RETENTION_DAYS)
            }
            n => Some(n),
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
    /// When a caller doesn't explicitly ask for reranking, should
    /// `Mimir::search_with` default it on anyway? An explicit `rerank: true`
    /// argument always reranks (cold-loading the model if needed) in every
    /// mode — this only controls the default.
    ///
    /// Both sides of the trade are measured now (graded retrieval eval,
    /// 2026-07): reranking the top 15 fused candidates raises nDCG@10 by
    /// ~0.03 and puts the first relevant hit at rank 1 almost always, at
    /// ~84 ms/candidate on CPU (~1.3 s per recall at 15) vs ~12 ms/candidate
    /// on a GPU build (~0.18 s). Deeper pools measured WORSE (25 candidates
    /// regressed nDCG): keep `candidates` at 15.
    /// - `"off"` (default): automatic reranking never fires; only an
    ///   explicit request reranks. Still the right default because the cost
    ///   is hardware-bound and per-machine: ~1.3 s per recall is a bad
    ///   surprise at an interactive CPU CLI, so turning it on should be an
    ///   informed per-box choice, not a shipped one.
    /// - `"warm"`: rerank only when it's free *right now* — the model is
    ///   already resident in this process, or a live inference daemon
    ///   answers for it (`remote_alive`); checked, never loaded, so a
    ///   one-shot CLI call never eats a cold ONNX load just because this is
    ///   on. Long-lived processes (the MCP server) eager-load a local copy
    ///   at startup only when no daemon is reachable — see `mcp.rs`'s
    ///   startup wiring. A clear win on GPU builds; on CPU it's still
    ///   reasonable for agent-driven recall (the ~1.3 s hides inside an LLM
    ///   turn) — see the README's "[rerank] auto" guidance for the full
    ///   decision table.
    /// - `"always"`: also loads the model on demand if it isn't resident
    ///   yet (same cost as an explicit `--rerank`, just automatic).
    ///
    /// Either way it's still gated per-query (see
    /// `Mimir::should_auto_rerank`): a tiny store or a single-token/
    /// identifier-shaped query skips it, since a cross-encoder only adds
    /// cost (and can reshuffle a correct exact match) there.
    pub auto: String,
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
    /// Weight for the shown-but-never-opened negative prior (`learn::
    /// impression_damp`): a node repeatedly surfaced (`store::record_shown`,
    /// logged by every recall/inject) without ever being opened
    /// (`learn::record_opened`) gets a mild multiplicative damp, decaying
    /// back toward 1.0 as the impressions age so it isn't punished forever.
    /// An opened node is exempt outright, and a node needs a minimum
    /// impression count before this applies at all — see
    /// `learn::impression_damp` for the exact thresholds/floor.
    ///
    /// Default **0.0 (off)**: this is a real ranking-signal change, and the
    /// `eval::hermetic`/`eval::real_model` fixture corpora have no
    /// impression history to score it against — shipping it on by default
    /// would be an unmeasured change to production ranking, which is
    /// exactly how the `recency_alpha` regression (see that field's doc
    /// comment) happened. Opt in once you have real recall_event history to
    /// judge it against; a future eval fixture with seeded impressions
    /// should gate raising this default.
    pub impression_alpha: f64,
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
            auto: "off".into(),
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
            impression_alpha: 0.0,
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

#[cfg(test)]
mod learn_config_tests {
    use super::LearnConfig;

    #[test]
    fn zero_means_never_prune() {
        let cfg = LearnConfig {
            event_retention_days: 0,
        };
        assert_eq!(cfg.effective_retention_days(), None);
    }

    #[test]
    fn default_180_passes_through_unclamped() {
        assert_eq!(LearnConfig::default().effective_retention_days(), Some(180));
    }

    #[test]
    fn below_floor_is_clamped_up_to_the_floor() {
        let cfg = LearnConfig {
            event_retention_days: 7,
        };
        assert_eq!(cfg.effective_retention_days(), Some(60));
    }

    #[test]
    fn exactly_at_the_floor_is_not_clamped_further() {
        let cfg = LearnConfig {
            event_retention_days: 60,
        };
        assert_eq!(cfg.effective_retention_days(), Some(60));
    }
}
