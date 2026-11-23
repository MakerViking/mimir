//! Mimir core engine: unified local-first memory for AI coding agents.
//!
//! Everything is a node (memories, doc chunks, files, code symbols,
//! projects, tags) in one SQLite database, searched by hybrid
//! BM25 + vector retrieval with reciprocal-rank fusion.

pub mod anchors;
pub mod audit;
pub mod brief;
pub mod config;
pub mod consolidate;
pub mod context_guard;
pub mod db;
pub mod embed;
pub mod error;
#[cfg(any(test, feature = "eval"))]
pub mod eval;
pub mod format;
pub mod grounding;
pub mod import;
pub mod index;
pub mod inject;
pub mod learn;
pub mod lifecycle;
pub mod memory;
pub mod misses;
pub mod model;
pub mod replicate;
pub mod review;
pub mod revision;
pub mod savings;
pub mod scope;
pub mod search;
pub mod secrets;
pub mod shadow;
pub mod store;
pub mod strength;
pub mod text_clock;
pub mod tokens;

use std::path::Path;

use rusqlite::Connection;

use crate::config::{Config, Paths};
use crate::embed::ModelSlot;
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

/// Out-of-process inference — a running `mimir daemon` that already holds
/// the models. Implemented in mimir-cli (core stays HTTP/async-free); every
/// method is fallible and every caller falls back to the local path, so a
/// remote can degrade latency but never break recall.
pub trait RemoteInference: Send {
    /// Cheap liveness gate (last-known state + dead-timer, no I/O).
    fn available(&self) -> bool;
    /// L2-normalized vectors, one per text, same model/dim as local config.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    /// Relevance scores aligned with `docs` order.
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

/// What became of the rerank step of one search
/// ([`Mimir::search_with_outcome`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankOutcome {
    /// Nobody asked: the caller did not, and `[rerank] auto` did not fire.
    NotAsked,
    /// Fewer than two hits came back, so there was no order to change.
    NothingToRerank,
    /// The cross-encoder rescored the page.
    Applied,
    /// A rerank was wanted and could not be done: no reranker resident or
    /// loadable in this process, no live daemon to delegate to, or the call
    /// itself failed (logged). The hits are in fused order.
    Unavailable,
}

impl RerankOutcome {
    /// What to tell a caller who asked for a rerank by name and did not get
    /// one; `None` for every other outcome. One sentence for every surface,
    /// so the CLI's stderr line and the MCP tool's note say the same thing.
    pub fn note(self) -> Option<&'static str> {
        match self {
            RerankOutcome::Unavailable => Some(
                "no reranker available, so this is the fused order (the model is not \
                 downloaded: `mimir embed --fetch --rerank`; or it failed to load: see the log)",
            ),
            RerankOutcome::NotAsked | RerankOutcome::NothingToRerank | RerankOutcome::Applied => {
                None
            }
        }
    }
}

/// One search and what a caller needs to explain it: the hits, the legs
/// they were fused from, and whether the order is the reranker's.
#[derive(Debug)]
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    pub legs: Vec<Vec<i64>>,
    pub rerank: RerankOutcome,
}

/// The reason a memory's ledger row gives when
/// [`Mimir::projects_for_cwd_moving_memories_home`] moved it out of a linked
/// worktree's own project.
pub const WORKTREE_MEMORIES_MOVED: &str = "captured in a git linked worktree; memories are filed \
     under the project of the repository's main checkout";

/// Where the memories of a checkout are filed, as
/// [`Mimir::projects_for_cwd`] decided it: the state [`scope::Checkout`]
/// alone does not name, because it also depends on the two checkouts'
/// portable keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFiling {
    /// Under the checkout's own project: a checkout that is its own home,
    /// and the case of no project at all.
    Own,
    /// Under the project of the repository's main checkout: a linked
    /// worktree with no portable key of its own, or the main checkout's.
    Home,
    /// Under the checkout's own project although it is a linked worktree:
    /// it carries a portable key (a `.mimir` id on its branch) that the main
    /// checkout does not, so it is resolved exactly as before worktrees
    /// were told apart, by that key. The main checkout's project is never
    /// written to from here, and nothing is moved.
    OwnKey,
}

impl MemoryFiling {
    /// The state's stable name, for JSON and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryFiling::Own => "own",
            MemoryFiling::Home => "home",
            MemoryFiling::OwnKey => "own_key",
        }
    }
}

/// The two projects a working directory resolves to
/// ([`Mimir::projects_for_cwd`]).
///
/// A directory has one project row, and that one id used to answer two
/// questions: whose memories, and whose code graph. For a git linked
/// worktree the answers differ ([`scope::Checkout`]): the graph describes
/// the files in this directory, the memories describe the repository. So
/// the two are named, and every surface picks the one it means:
///
/// - [`ProjectContext::checkout`] owns what is derived from the files here:
///   the code graph, code-file rows, docs collections, graph freshness.
/// - [`ProjectContext::memory`] owns what was learned about the repository:
///   memories, the rules pack, the brief, guard anchors, injection scope,
///   the review queue, the miss ledger, the sync identity.
///
/// They are the same row unless [`ProjectContext::filing`] is
/// [`MemoryFiling::Home`]; then `memory` is the project of the repository's
/// main checkout. `recall` is the one surface that reads both at once,
/// memories and code and docs in one ranked list:
/// [`ProjectContext::query_reach`].
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// How the checkout's root was found, or where the search started when
    /// there is no project.
    pub detection: scope::Detection,
    /// Which checkout of its repository the root is;
    /// [`scope::Checkout::Standalone`] outside any project.
    pub checkout_kind: scope::Checkout,
    /// Where its memories are filed; [`MemoryFiling::Own`] outside any
    /// project.
    pub filing: MemoryFiling,
    /// The project whose path is this checkout's root. `None` outside any
    /// project.
    pub checkout: Option<Node>,
    /// The project this checkout's memories are filed under. `None` outside
    /// any project.
    pub memory: Option<Node>,
}

impl ProjectContext {
    /// Row id of the checkout project.
    pub fn checkout_id(&self) -> Option<i64> {
        self.checkout.as_ref().map(|p| p.id)
    }

    /// Row id of the memory project.
    pub fn memory_id(&self) -> Option<i64> {
        self.memory.as_ref().map(|p| p.id)
    }

    /// What a question asked from this working directory can reach:
    /// [`scope::query_reach`] of the two projects.
    pub fn query_reach(&self) -> search::Reach {
        scope::query_reach(self.memory_id(), self.checkout_id())
    }

    /// The canonical root of this checkout, which is its project row's
    /// path: the identity a session handoff records
    /// (`context_guard::handoff_is_for`). `None` outside any project.
    pub fn checkout_path(&self) -> Option<&str> {
        self.checkout.as_ref().and_then(|p| p.path.as_deref())
    }

    /// For a linked worktree, the line that says which project is which:
    /// [`linked_worktree_note`] when its memories go to the main checkout's
    /// project, [`own_key_worktree_note`] when it keeps its own. `None` for
    /// any other checkout, where there is one project and nothing to tell
    /// apart.
    pub fn worktree_note(&self) -> Option<String> {
        match self.filing {
            MemoryFiling::Own => None,
            MemoryFiling::Home => self.memory.as_ref().map(linked_worktree_note),
            MemoryFiling::OwnKey => self.checkout_kind.home().map(own_key_worktree_note),
        }
    }
}

/// What a status surface adds under the project of a linked worktree, given
/// its memory project `home` (the main checkout's row): that the checkout
/// is a linked worktree, of which main checkout, and where its memories
/// go. One sentence for `mimir status` and the MCP `status` tool.
pub fn linked_worktree_note(home: &Node) -> String {
    format!(
        "linked worktree of {}; memories are filed under project {}",
        home.path.as_deref().unwrap_or("?"),
        home.title.as_deref().unwrap_or("?")
    )
}

/// [`linked_worktree_note`] for a linked worktree that keeps a project of
/// its own ([`MemoryFiling::OwnKey`]), given the main checkout's root.
pub fn own_key_worktree_note(home: &Path) -> String {
    format!(
        "linked worktree of {}; keeps a project of its own, because its `.mimir` id is not \
         the main checkout's",
        home.display()
    )
}

/// What [`Mimir::projects_for_cwd_moving_memories_home`] moved: memories
/// filed under a linked worktree's own row, now under its home project.
#[derive(Debug, Clone)]
pub struct MovedHome {
    /// How many memories moved.
    pub count: usize,
    /// The worktree's own project row, where they were.
    pub worktree: Node,
    /// The project of the main checkout, where they are now.
    pub home: Node,
    /// Whether moving them made them part of what `mimir sync` sends: the
    /// home project syncs and the worktree's row did not.
    pub published: bool,
}

impl MovedHome {
    /// The one line a command that moved them says, on stderr.
    pub fn line(&self) -> String {
        let (noun, it, they) = if self.count == 1 {
            ("memory", "it", "it")
        } else {
            ("memories", "them", "they")
        };
        let synced = if self.published {
            format!("; that project syncs, so {they} will be sent at the next `mimir sync`")
        } else {
            String::new()
        };
        format!(
            "moved {} {noun} from linked worktree {} to project {} (`mimir audit` lists {it}){synced}",
            self.count,
            self.worktree.path.as_deref().unwrap_or("?"),
            self.home.title.as_deref().unwrap_or("?"),
        )
    }
}

/// The memories still filed under a linked worktree's own row, which
/// `mimir status` reports ([`Mimir::worktree_leftovers`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorktreeLeftovers {
    /// Written since the worktree was made, and not moved yet: the next
    /// command that moves (`Mimir::projects_for_cwd_moving_memories_home`)
    /// files them under the home project.
    pub to_move: usize,
    /// Older than the worktree (another checkout used this path before
    /// it), or of a worktree whose creation time cannot be read: they stay.
    pub kept: usize,
    /// When the worktree was made (`scope::worktree_created_at`).
    pub worktree_created_at: Option<i64>,
}

impl WorktreeLeftovers {
    /// Whether there is anything to report.
    pub fn is_empty(&self) -> bool {
        self.to_move == 0 && self.kept == 0
    }

    /// The lines `mimir status` prints, one per kind of leftover.
    pub fn lines(&self, home: &Node) -> Vec<String> {
        let home = home.title.as_deref().unwrap_or("?");
        let mut lines = Vec::new();
        if self.to_move > 0 {
            let (noun, pronoun) = if self.to_move == 1 {
                ("memory is", "it")
            } else {
                ("memories are", "them")
            };
            lines.push(format!(
                "{} {noun} still filed under this worktree's own project: not moved yet; the \
                 next memory command here (remember, recall, list, …) moves {pronoun} to \
                 project {home}",
                self.to_move
            ));
        }
        if self.kept > 0 {
            lines.push(format!(
                "{} {} filed under this worktree's own project and {}; {} not moved",
                self.kept,
                if self.kept == 1 {
                    "memory is"
                } else {
                    "memories are"
                },
                if self.worktree_created_at.is_some() {
                    "older than the worktree (another checkout used this path before)"
                } else {
                    "the worktree's creation time cannot be read"
                },
                if self.kept == 1 { "it is" } else { "they are" }
            ));
        }
        lines
    }
}

/// Engine facade. Owns one writer connection; cheap to open.
/// The embedding model loads lazily on first semantic use and is cached
/// for the process lifetime (long-lived in the MCP server).
pub struct Mimir {
    pub conn: Connection,
    pub config: Config,
    pub paths: Paths,
    embedder: ModelSlot<embed::Embedder>,
    reranker: ModelSlot<embed::Reranker>,
    matrix: Option<MatrixCache>,
    /// Set by a `&self` method whose write on this connection changed what
    /// the matrix caches per row, and which could not drop the cache itself
    /// (see [`Mimir::drop_stale_matrix`]).
    matrix_stale: std::sync::atomic::AtomicBool,
    query_cache: std::collections::HashMap<String, Vec<f32>>,
    remote: Option<Box<dyn RemoteInference>>,
    embedder_device: Option<String>,
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
            embedder: ModelSlot::Unloaded,
            reranker: ModelSlot::Unloaded,
            matrix: None,
            matrix_stale: std::sync::atomic::AtomicBool::new(false),
            query_cache: std::collections::HashMap::new(),
            remote: None,
            embedder_device: None,
        })
    }

    /// In-memory instance for tests.
    pub fn open_in_memory() -> Result<Self> {
        Ok(Mimir {
            conn: db::open_in_memory()?,
            config: Config::default(),
            paths: Paths::under_root(Path::new(".")),
            embedder: ModelSlot::Unloaded,
            reranker: ModelSlot::Unloaded,
            matrix: None,
            matrix_stale: std::sync::atomic::AtomicBool::new(false),
            query_cache: std::collections::HashMap::new(),
            remote: None,
            embedder_device: None,
        })
    }

    /// Attach an out-of-process inference backend (a running daemon).
    /// Callers that never attach one get the pure-local behavior.
    pub fn set_remote_inference(&mut self, remote: Box<dyn RemoteInference>) {
        self.remote = Some(remote);
    }

    /// Override the device the *embedder* loads on, without touching config.
    /// MCP sessions pass "cpu": batch-1 query embeds measured faster on CPU
    /// than GPU, and a session-local GPU embedder is exactly the per-session
    /// VRAM cost the daemon exists to remove. Deliberately does not affect
    /// the reranker — a local-fallback reranker keeps the configured device.
    ///
    /// Takes effect at the next load, so it has to be set before the model
    /// is resident: a loaded model stays on the device it loaded on, and
    /// asking for a different one then is logged rather than left to look
    /// as if it had worked.
    pub fn set_embedder_device(&mut self, device: &str) {
        if self.embedder.is_loaded() {
            if self.embedder_device() != device {
                tracing::warn!(
                    requested = device,
                    loaded_on = self.embedder_device(),
                    "embedder is already loaded; the device override has no effect"
                );
            }
            return;
        }
        self.embedder_device = Some(device.to_string());
    }

    /// The device the embedder loads on: the override, else config.
    fn embedder_device(&self) -> &str {
        self.embedder_device
            .as_deref()
            .unwrap_or(&self.config.embedding.device)
    }

    fn remote_alive(&self) -> bool {
        self.remote.as_ref().is_some_and(|r| r.available())
    }

    /// Lazily load the embedding model. Never downloads unless asked;
    /// failure degrades to BM25-only (logged, not fatal) and is retried on
    /// the terms [`ModelSlot`] sets out, not held for the life of the
    /// process.
    pub fn ensure_embedder(&mut self, allow_download: bool) -> Option<&mut embed::Embedder> {
        self.ensure_embedder_at(allow_download, std::time::Instant::now())
    }

    /// [`Mimir::ensure_embedder`] at a given instant, so the retry cooldown
    /// can be tested without waiting for it.
    fn ensure_embedder_at(
        &mut self,
        allow_download: bool,
        now: std::time::Instant,
    ) -> Option<&mut embed::Embedder> {
        let downloadable =
            allow_download || embed::model_ready(&self.paths, &self.config.embedding.model);
        if downloadable && self.embedder.wants_load(allow_download, now) {
            let loaded = embed::Embedder::load(
                &self.paths,
                &self.config.embedding.model,
                self.embedder_device(),
                allow_download,
            );
            if let Some(err) = self.embedder.settle(loaded, allow_download, now) {
                tracing::warn!(%err, "embeddings unavailable; BM25-only");
            }
        }
        self.embedder.get_mut()
    }

    /// Lazily load the cross-encoder reranker (marker-gated, like the
    /// embedder, and retried after a failure on the same terms). Missing
    /// model degrades to un-reranked results.
    pub fn ensure_reranker(&mut self, allow_download: bool) -> Option<&mut embed::Reranker> {
        self.ensure_reranker_at(allow_download, std::time::Instant::now())
    }

    fn ensure_reranker_at(
        &mut self,
        allow_download: bool,
        now: std::time::Instant,
    ) -> Option<&mut embed::Reranker> {
        let downloadable =
            allow_download || embed::model_ready(&self.paths, &self.config.rerank.model);
        if downloadable && self.reranker.wants_load(allow_download, now) {
            let loaded = embed::Reranker::load(
                &self.paths,
                &self.config.rerank.model,
                &self.config.embedding.device,
                allow_download,
            );
            if let Some(err) = self.reranker.settle(loaded, allow_download, now) {
                tracing::warn!(%err, "reranker unavailable; skipping rerank");
            }
        }
        self.reranker.get_mut()
    }

    /// True if the cross-encoder reranker is already resident in this
    /// process. Unlike `ensure_reranker`, checking this never triggers a
    /// load — it's what the `[rerank] auto = "warm"` gate uses to decide
    /// whether reranking is "free" right now.
    pub fn reranker_loaded(&self) -> bool {
        self.reranker.is_loaded()
    }

    /// Whether `search_with`'s automatic reranking (config `[rerank] auto`)
    /// should fire for `query`, given the current process/store state. Only
    /// governs the *default* — an explicit `rerank: true` argument to
    /// `search_with` always reranks regardless of this.
    fn should_auto_rerank(&mut self, query: &SearchQuery) -> bool {
        // The typed accessor is where an unrecognized value becomes "off";
        // `Config::issues` reports it once (doctor, server start-up) rather
        // than this path warning on every query.
        let available = match self.config.rerank.auto() {
            crate::config::RerankAuto::Off => return false,
            // "warm" = rerank only when it's free right now: a resident
            // local model, or a daemon that already holds one.
            crate::config::RerankAuto::Warm => self.reranker_loaded() || self.remote_alive(),
            crate::config::RerankAuto::Always => {
                self.remote_alive() || self.ensure_reranker(false).is_some()
            }
        };
        available && rerank_query_gates_pass(&self.conn, query)
    }

    /// Hybrid search when the model is available locally, BM25-only otherwise.
    pub fn search(&mut self, query: &SearchQuery) -> Result<Vec<Hit>> {
        self.search_with(query, false)
    }

    /// Like [`Mimir::search`], but also returns the raw per-leg ranked id
    /// lists (`legs[0]` = FTS, then the vector leg when one ran — see
    /// [`search`]'s module docs and [`search::vector_leg`]) —
    /// used by callers that need to check lexical/semantic agreement on a
    /// hit before trusting it (e.g. the auto-recall relevance floor). No
    /// reranking: the floor check needs the legs' own top-K as computed
    /// before any cross-encoder rescoring.
    pub fn search_with_legs(&mut self, query: &SearchQuery) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
        self.drop_stale_matrix();
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
    /// No vector leg runs, so [`search::vector_leg`] is `None` for the
    /// result — `inject::clears_floor`'s rule 3 already degrades cleanly
    /// then (the same state a model-less machine is in).
    pub fn search_with_legs_bm25_only(
        &mut self,
        query: &SearchQuery,
    ) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
        self.drop_stale_matrix();
        search::search_hybrid_with_legs_scored(
            &self.conn,
            query,
            None,
            &mut self.matrix,
            self.config.scoring.impression_alpha,
        )
    }

    /// Drop the vector matrix if a write on this connection outdated it.
    ///
    /// The matrix caches each row's project next to its vector, and is
    /// refreshed when *another* connection commits (`MatrixCache::ensure`).
    /// A write on this one is invisible to that check, so whoever makes one
    /// that changes a cached column says so; every search asks here first.
    /// The one such writer today is
    /// [`Mimir::projects_for_cwd_moving_memories_home`], which moves a
    /// linked worktree's memories to their home project.
    fn drop_stale_matrix(&mut self) {
        if *self.matrix_stale.get_mut() {
            self.matrix = None;
            *self.matrix_stale.get_mut() = false;
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
    /// `rerank = true` always reranks (cold-loading the model if needed,
    /// same as today); `rerank = false` still reranks if config
    /// `[rerank] auto` and the per-query gates say it should — see
    /// `should_auto_rerank`.
    pub fn search_with(&mut self, query: &SearchQuery, rerank: bool) -> Result<Vec<Hit>> {
        Ok(self.search_with_evidence(query, rerank)?.0)
    }

    /// [`Mimir::search_with`], plus the per-leg ranked id lists the hits were
    /// fused from (`legs[0]` = FTS, then the vector leg when one ran) —
    /// what `strength::classify` needs to say how much evidence stands behind
    /// each hit. The legs are the pre-rerank retrieval: reranking reorders
    /// the page, it does not change what either retriever found.
    pub fn search_with_evidence(
        &mut self,
        query: &SearchQuery,
        rerank: bool,
    ) -> Result<(Vec<Hit>, Vec<Vec<i64>>)> {
        let outcome = self.search_with_outcome(query, rerank)?;
        Ok((outcome.hits, outcome.legs))
    }

    /// [`Mimir::search_with_evidence`], plus what became of the rerank
    /// step. Every rerank failure degrades to the fused order, which is
    /// right for an automatic rerank and wrong to do silently for one the
    /// caller asked for by name: [`SearchOutcome::rerank`] is how a surface
    /// tells its user that `--rerank` changed nothing because there was no
    /// reranker to run.
    pub fn search_with_outcome(
        &mut self,
        query: &SearchQuery,
        rerank: bool,
    ) -> Result<SearchOutcome> {
        self.search_with_outcome_within(query, search::Reach::uniform(query.scope), rerank)
    }

    /// [`Mimir::search_with_outcome`] over a [`search::Reach`]: memories
    /// from one scope and code and docs from another, fused and reranked as
    /// one page. What `recall` runs, so that in a git linked worktree it
    /// answers with the repository's memories and the worktree's own code
    /// and docs (`scope::query_reach`). `query.scope` is not read; see
    /// [`search::Reach`].
    pub fn search_with_outcome_within(
        &mut self,
        query: &SearchQuery,
        reach: search::Reach,
        rerank: bool,
    ) -> Result<SearchOutcome> {
        self.drop_stale_matrix();
        let explicit = rerank;
        let rerank = rerank || self.should_auto_rerank(query);
        let mut fetch_query = query.clone();
        if rerank {
            fetch_query.limit = query.limit.max(self.config.rerank.candidates);
        }

        let query_vec = self.query_embedding(&query.text);
        let alpha = self.config.scoring.impression_alpha;
        let (mut hits, legs) = match query_vec {
            Some(v) => {
                let model = self.config.embedding.model.clone();
                search::search_hybrid_with_legs_scored_within(
                    &self.conn,
                    &fetch_query,
                    reach,
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
            None => search::search_hybrid_with_legs_scored_within(
                &self.conn,
                &fetch_query,
                reach,
                None,
                &mut self.matrix,
                alpha,
            )?,
        };

        let mut rerank_outcome = match (rerank, hits.len() > 1) {
            (false, _) => RerankOutcome::NotAsked,
            (true, false) => RerankOutcome::NothingToRerank,
            (true, true) => RerankOutcome::Unavailable,
        };
        if rerank && hits.len() > 1 {
            let docs: Vec<String> = hits
                .iter()
                .map(|h| {
                    let title = h.node.title.as_deref().unwrap_or("");
                    let body = h.node.body.as_deref().unwrap_or("");
                    // Cross-encoders cap at ~512 tokens; keep pairs cheap.
                    format!("{title}\n{body}").chars().take(1500).collect()
                })
                .collect();
            if let Some(ranked) = self.rerank_docs(&query.text, docs, explicit) {
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
                rerank_outcome = RerankOutcome::Applied;
            }
        }
        hits.truncate(query.limit);
        Ok(SearchOutcome {
            hits,
            legs,
            rerank: rerank_outcome,
        })
    }

    /// Rank `docs` against `text`, best-first `(doc_index, score)`. Priority:
    /// locally-resident reranker (same GPU, zero hop) → live daemon → cold
    /// local load — the last only when the caller explicitly asked to rerank
    /// or config says "always"; auto-"warm" never forces a model into this
    /// process. `None` = keep fused order (every failure degrades, none fail).
    fn rerank_docs(
        &mut self,
        text: &str,
        docs: Vec<String>,
        explicit: bool,
    ) -> Option<Vec<(usize, f32)>> {
        if self.reranker_loaded() {
            return self.local_rerank(text, docs);
        }
        if let Some(remote) = self.remote.as_ref().filter(|r| r.available()) {
            match remote.rerank(text, &docs) {
                Ok(scores) if scores.len() == docs.len() => {
                    let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    return Some(ranked);
                }
                Ok(scores) => tracing::warn!(
                    got = scores.len(),
                    want = docs.len(),
                    "daemon rerank score count mismatch; keeping fused order"
                ),
                Err(err) => tracing::warn!(%err, "daemon rerank failed; keeping fused order"),
            }
            // A live-but-failing daemon doesn't justify pulling a local
            // model into memory unless the caller demanded a rerank.
            if !explicit {
                return None;
            }
        }
        if explicit || self.config.rerank.auto() == crate::config::RerankAuto::Always {
            return self.local_rerank(text, docs);
        }
        None
    }

    fn local_rerank(&mut self, text: &str, docs: Vec<String>) -> Option<Vec<(usize, f32)>> {
        let rr = self.ensure_reranker(false)?;
        match rr.rerank(text, docs) {
            Ok(ranked) => Some(ranked),
            Err(err) => {
                tracing::warn!(%err, "rerank failed; keeping fused order");
                None
            }
        }
    }

    /// Embed all new/changed content. No-op (0) when no model is present.
    /// Patches the vector matrix cache in place for a small batch (the
    /// common `remember`-one-node case) instead of dropping it — same-
    /// connection writes are invisible to `data_version`, so a dropped
    /// cache would otherwise pay a full corpus reload on the next recall.
    /// A large batch (bulk import/reindex) still drops it: patching would
    /// cost about as much as a rebuild, for no benefit.
    pub fn embed_pending(&mut self) -> Result<usize> {
        // Probe before touching the model: background callers (auto-sync,
        // startup maintenance) call this speculatively every few minutes,
        // and on GPU builds a no-op load would still create (and tear down)
        // a GPU device each time — measured to leak driver memory.
        if !embed::has_pending(&self.conn, &self.config.embedding.model)? {
            return Ok(0);
        }
        let model = self.config.embedding.model.clone();
        // Daemon first (its GPU is faster for bulk batches); local on any
        // failure. Safe to rerun after a partial remote run: work commits
        // per batch and the pending selection is idempotent, so the local
        // pass just picks up whatever the remote didn't finish.
        let mut ids = None;
        if let Some(remote) = self.remote.as_ref().filter(|r| r.available()) {
            match embed::embed_pending(&self.conn, &model, &mut |texts| remote.embed(&texts)) {
                Ok(v) => ids = Some(v),
                Err(err) => {
                    tracing::warn!(%err, "daemon embed failed; falling back to local embedder")
                }
            }
        }
        let ids = match ids {
            Some(v) => v,
            None => {
                self.ensure_embedder(false);
                let Mimir { conn, embedder, .. } = self;
                match embedder.get_mut() {
                    Some(e) => embed::embed_pending(conn, &model, &mut |texts| e.embed(texts))?,
                    None => Vec::new(),
                }
            }
        };
        // Above this, a linear-scan patch (see MatrixCache::upsert) approaches
        // the cost of just rebuilding; bulk operations aren't latency-sensitive
        // the way interactive recall is, so fall back to the old behavior.
        const PATCH_THRESHOLD: usize = 200;
        if !ids.is_empty() {
            if ids.len() <= PATCH_THRESHOLD {
                MatrixCache::upsert(&self.conn, &model, &ids, &mut self.matrix)?;
            } else {
                self.matrix = None;
            }
        }
        Ok(ids.len())
    }

    /// Resolve (and lazily register) the projects for a working directory:
    /// the checkout's own and the one its memories are filed under. This is
    /// the one place the pair is derived; [`Mimir::detect_project`],
    /// [`Mimir::project_for_cwd`], [`Mimir::memory_project_for_cwd`] and
    /// [`Mimir::projects_for_cwd_moving_memories_home`] all read it.
    /// `NotFound` means no project at all, which the caller explains rather
    /// than degrading silently (new memories are global; what a read sees is
    /// `scope::query_scope` or `scope::push_scope`).
    ///
    /// It never moves a memory, and takes no write lock to find out whether
    /// there is one to move: hooks, `status`, `doctor` and the daemon resolve
    /// through here, and a hook must not wait behind another process's
    /// writer for a move it was not asked to make. Writes it can make are
    /// the ones every resolution made before linked worktrees were told
    /// apart (a project row on first contact, its sync identity when the
    /// marker changes), plus `meta.worktree_of` on the checkout's row when
    /// it changes ([`store::set_worktree_home`]); that one is attempted
    /// without waiting for the lock, and skipped when the store is busy.
    ///
    /// Outside a git linked worktree the two are one row, resolved as
    /// [`Mimir::register_root`] describes ([`MemoryFiling::Own`]).
    ///
    /// A linked worktree ([`scope::Checkout::LinkedWorktree`]) shares its
    /// main checkout's memory project when it has no portable key of its
    /// own (`scope::portable_key`: a `.mimir` id; a worktree has no `origin`
    /// of its own to read) or the same one as the main checkout
    /// ([`MemoryFiling::Home`]). Then the order matters:
    /// 1. The home project first, by that same resolution on the main
    ///    checkout's root. It creates the home row if Mimir never ran
    ///    there; and for a repository with a `.mimir` id, whose one shared
    ///    row may have been bound to a worktree's path last, it binds that
    ///    row back to the main checkout.
    /// 2. Then the worktree's own row, by path alone. It never adopts by
    ///    portable key: the sync identity is the home row's.
    ///
    /// A linked worktree that carries a key the main checkout does not (the
    /// branch has a `.mimir` the main checkout lags behind, or whose id
    /// differs, or the main checkout's marker does not parse) is resolved
    /// exactly as it was before the two were told apart: by its own key,
    /// as its own memory project ([`MemoryFiling::OwnKey`]). Its memories
    /// stay where that key's sync puts them, and the main checkout's
    /// project is not touched.
    pub fn projects_for_cwd(&self, cwd: &Path) -> Result<ProjectContext> {
        Ok(self.resolve_projects(cwd, MayMove::No)?.0)
    }

    /// [`Mimir::projects_for_cwd`], and then, in a linked worktree that
    /// shares its main checkout's memory project, move the memories still
    /// filed under the worktree's own row to that project: they were
    /// captured there while the worktree counted as a project of its own
    /// (an older Mimir, or before its main checkout could be found), and
    /// would otherwise be out of every reader's scope.
    ///
    /// For commands that read or write memories and are run on purpose (the
    /// CLI's memory commands, the MCP server's start-up), never for a hook.
    /// Only memories created since the worktree was made move
    /// (`scope::worktree_created_at`): an older one under the same path was
    /// written in an earlier checkout there, possibly of another
    /// repository, and stays; when that time cannot be read, nothing moves.
    /// A handoff among them that records no checkout is given this one's,
    /// in the same transaction, so it is still restored only here
    /// (`context_guard::handoff_is_for`).
    ///
    /// Best-effort: a move that fails is logged at warn level, nothing is
    /// half-moved, the resolution stands, and the next call tries again.
    /// Returns what moved, for the caller to say so.
    pub fn projects_for_cwd_moving_memories_home(
        &self,
        cwd: &Path,
    ) -> Result<(ProjectContext, Option<MovedHome>)> {
        self.resolve_projects(cwd, MayMove::Yes)
    }

    /// The one resolution behind [`Mimir::projects_for_cwd`] and
    /// [`Mimir::projects_for_cwd_moving_memories_home`]; `may_move` is the
    /// only thing that tells them apart.
    fn resolve_projects(
        &self,
        cwd: &Path,
        may_move: MayMove,
    ) -> Result<(ProjectContext, Option<MovedHome>)> {
        let detection = scope::detect(cwd);
        let scope::Detection::Found { root, .. } = &detection else {
            let nowhere = ProjectContext {
                detection,
                checkout_kind: scope::Checkout::Standalone,
                filing: MemoryFiling::Own,
                checkout: None,
                memory: None,
            };
            return Ok((nowhere, None));
        };
        let root = root.clone();
        let checkout_kind = scope::Checkout::of(&root);
        let wait = may_move == MayMove::Yes;
        let mut moved = None;
        let (filing, checkout, memory) = match checkout_kind.home() {
            Some(home) if shares_home(&root, home) => {
                let memory = self.register_root(home)?;
                let checkout = self.register_linked_worktree(&root)?;
                self.record_worktree_home(&checkout, Some(&memory), wait);
                if may_move == MayMove::Yes {
                    moved = self.move_memories_home(&root, &checkout, &memory);
                }
                (MemoryFiling::Home, checkout, memory)
            }
            home => {
                let node = self.register_root(&root)?;
                self.record_worktree_home(&node, None, wait);
                let filing = if home.is_some() {
                    MemoryFiling::OwnKey
                } else {
                    MemoryFiling::Own
                };
                (filing, node.clone(), node)
            }
        };
        let here = ProjectContext {
            detection,
            checkout_kind,
            filing,
            checkout: Some(checkout),
            memory: Some(memory),
        };
        Ok((here, moved))
    }

    /// Find or create the project of a root that is its own memory project,
    /// and keep its sync identity current. A root with a portable key
    /// (`scope::portable_key`) that a project already carries adopts that
    /// project onto this path; any other root is its own project, by path.
    fn register_root(&self, root: &Path) -> Result<Node> {
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
        Ok(node)
    }

    /// Find or create the row of a linked worktree's own checkout, by path.
    /// What the row already holds stays: a portable key it was given while
    /// the worktree counted as a project of its own is what
    /// `store::find_project_by_key` binds back to the main checkout once
    /// the main checkout carries that key too (step 1 of
    /// [`Mimir::projects_for_cwd`]).
    fn register_linked_worktree(&self, root: &Path) -> Result<Node> {
        store::ensure_project(
            &self.conn,
            &scope::canonical_root(root),
            &scope::project_name(root),
        )
    }

    /// Keep `meta.worktree_of` on a checkout's row saying which project its
    /// memories are filed under: `home` for a linked worktree that shares
    /// it, nothing for every other checkout. Read from the row in hand
    /// first, so the steady state costs no query. Best-effort, and with
    /// `wait` false it does not wait for the write lock either: a hook skips
    /// it when the store is busy, and a later resolution records it.
    fn record_worktree_home(&self, checkout: &Node, home: Option<&Node>, wait: bool) {
        let stored = checkout
            .meta
            .get(store::WORKTREE_OF)
            .and_then(|v| v.as_str());
        if stored == home.map(|h| h.uid.as_str()) {
            return;
        }
        let write = || store::set_worktree_home(&self.conn, checkout.id, home);
        let written = if wait {
            write()
        } else {
            without_busy_wait(&self.conn, write)
        };
        if let Err(err) = written {
            tracing::debug!(%err, project = checkout.id, "could not record the worktree's home");
        }
    }

    /// Move what is still filed under a linked worktree's own row, and was
    /// created since the worktree was made, to its home project; see
    /// [`Mimir::projects_for_cwd_moving_memories_home`].
    fn move_memories_home(&self, root: &Path, worktree: &Node, home: &Node) -> Option<MovedHome> {
        let since = scope::worktree_created_at(root)?;
        let moving = || -> Result<usize> {
            // Asked before the transaction, which takes the write lock.
            if store::memories_filed_under(&self.conn, worktree.id, Some(since))?.0 == 0 {
                return Ok(0);
            }
            crate::db::atomic(&self.conn, || {
                let moved = store::reproject_all_created_since(
                    &self.conn,
                    worktree.id,
                    home.id,
                    since,
                    crate::model::Actor::System,
                    Some(WORKTREE_MEMORIES_MOVED),
                )?;
                // A handoff from before checkouts were recorded is still
                // this worktree's state: tie it here, or a session in any
                // other checkout of the repository could restore it.
                if let Some(path) = worktree.path.as_deref() {
                    for memory in &moved {
                        let tags = memory.tags();
                        if tags.contains(&context_guard::HANDOFF_TAG)
                            && context_guard::handoff_checkout(memory).is_none()
                        {
                            context_guard::set_handoff_checkout(&self.conn, memory.id, path)?;
                        }
                    }
                }
                Ok(moved.len())
            })
        };
        match moving() {
            Ok(0) => None,
            Ok(count) => {
                // The matrix caches each memory's project; see
                // `drop_stale_matrix`.
                self.matrix_stale
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                let moved = MovedHome {
                    count,
                    worktree: worktree.clone(),
                    home: home.clone(),
                    published: replicate::project_is_syncable(&home.meta)
                        && !replicate::project_is_syncable(&worktree.meta),
                };
                tracing::info!("{}", moved.line());
                Some(moved)
            }
            Err(err) => {
                tracing::warn!(
                    %err,
                    worktree = worktree.id,
                    home = home.id,
                    "could not move a linked worktree's memories to its main checkout's project"
                );
                None
            }
        }
    }

    /// What is still filed under the own row of a linked worktree that
    /// shares its main checkout's memory project, for `mimir status`: how
    /// many memories the next moving resolution files home, and how many
    /// stay because they are older than the worktree. `None` for any other
    /// checkout. Reads only.
    pub fn worktree_leftovers(&self, here: &ProjectContext) -> Result<Option<WorktreeLeftovers>> {
        if here.filing != MemoryFiling::Home {
            return Ok(None);
        }
        let (Some(checkout), scope::Detection::Found { root, .. }) =
            (&here.checkout, &here.detection)
        else {
            return Ok(None);
        };
        let since = scope::worktree_created_at(root);
        let (to_move, kept) = store::memories_filed_under(&self.conn, checkout.id, since)?;
        Ok(Some(WorktreeLeftovers {
            to_move,
            kept,
            worktree_created_at: since,
        }))
    }

    /// The **checkout** project of a working directory (the row whose path
    /// is the checkout's root: what the code graph, code files and docs
    /// collections hang under), plus *how* its root was detected. `Found` ⇒
    /// the project node is ensured (created on first contact); `NotFound` ⇒
    /// no project. A surface that reads or writes memories wants
    /// [`Mimir::memory_project_for_cwd`], which differs in a linked
    /// worktree; see [`Mimir::projects_for_cwd`].
    pub fn detect_project(&self, cwd: &Path) -> Result<(Option<Node>, scope::Detection)> {
        let found = self.projects_for_cwd(cwd)?;
        Ok((found.checkout, found.detection))
    }

    /// The checkout project of a working directory, or None outside any
    /// project. Thin wrapper over [`Mimir::detect_project`] for callers that
    /// don't need the detection reason.
    pub fn project_for_cwd(&self, cwd: &Path) -> Result<Option<Node>> {
        Ok(self.detect_project(cwd)?.0)
    }

    /// The **memory** project of a working directory, or None outside any
    /// project: the project its memories, rules pack, brief, guard anchors
    /// and injection scope belong to. The checkout's own project, except in
    /// a git linked worktree, where it is the project of the repository's
    /// main checkout; see [`Mimir::projects_for_cwd`].
    pub fn memory_project_for_cwd(&self, cwd: &Path) -> Result<Option<Node>> {
        Ok(self.projects_for_cwd(cwd)?.memory)
    }

    /// The memory project of a checkout project that is already in hand
    /// (a row out of a listing, a server's start-up project), for a caller
    /// with no working directory to resolve from. The same answer
    /// [`Mimir::projects_for_cwd`] gives inside that checkout; the row
    /// itself when its directory is gone or no longer resolves to it.
    pub fn memory_project_of(&self, checkout: &Node) -> Result<Node> {
        let Some(path) = checkout.path.as_deref().map(Path::new) else {
            return Ok(checkout.clone());
        };
        if !path.is_dir() {
            return Ok(checkout.clone());
        }
        let found = self.projects_for_cwd(path)?;
        match (found.checkout, found.memory) {
            (Some(own), Some(memory)) if own.id == checkout.id => Ok(memory),
            _ => Ok(checkout.clone()),
        }
    }
}

/// Whether a resolution may move a linked worktree's memories home: the
/// switch between [`Mimir::projects_for_cwd`] and
/// [`Mimir::projects_for_cwd_moving_memories_home`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MayMove {
    No,
    Yes,
}

/// Whether the linked worktree at `root` files its memories under the
/// project of its main checkout at `home`: when it carries no portable key
/// of its own, or the same one. See [`Mimir::projects_for_cwd`].
fn shares_home(root: &Path, home: &Path) -> bool {
    match scope::portable_key(root) {
        None => true,
        Some(own) => scope::portable_key(home).as_deref() == Some(own.as_str()),
    }
}

/// Run `work` with the connection's busy timeout at zero, so a write that
/// would wait for another process's write lock fails at once instead, and
/// put the timeout back afterwards (as `db::open` set it).
fn without_busy_wait<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let before: i64 = conn.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?;
    conn.busy_timeout(std::time::Duration::ZERO)?;
    let out = work();
    conn.busy_timeout(std::time::Duration::from_millis(
        u64::try_from(before).unwrap_or(0),
    ))?;
    out
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

/// Store-size gate for [`Mimir::should_auto_rerank`]: are there fewer than
/// [`RERANK_MIN_STORE_NODES`] embeddable nodes — the same predicate
/// `embed_pending` selects from (`embed::embeddable_where`), so "embeddable"
/// means one thing. The count stops at the threshold, because the answer
/// only depends on reaching it: this is on the hot recall path, and a large
/// store should pay for about that many rows, not for counting all of them.
fn store_too_small_for_rerank(conn: &Connection) -> bool {
    conn.query_row(
        &format!(
            "SELECT COUNT(*) FROM (SELECT 1 FROM node n WHERE {} LIMIT ?1)",
            embed::embeddable_where()
        ),
        [RERANK_MIN_STORE_NODES],
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
            as_of: None,
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

    /// The size gate counts what could actually be embedded. A node with no
    /// body never is (`embed::embeddable_where`), however many there are.
    #[test]
    fn auto_rerank_store_size_ignores_nodes_that_cannot_be_embedded() {
        let conn = db::open_in_memory().unwrap();
        for i in 0..(RERANK_MIN_STORE_NODES as usize + 1) {
            let mut n = NewNode::new(Kind::Symbol);
            n.title = Some(format!("bodiless {i}"));
            store::insert_node(&conn, n).unwrap();
        }
        assert!(store_too_small_for_rerank(&conn));
        assert_eq!(
            embed::coverage(&conn, "any-model").unwrap(),
            (0, 0),
            "and the coverage count agrees there is nothing to embed"
        );
    }

    /// An engine whose configured models pass the "already downloaded" gate
    /// and then fail to load, with no network involved: the ready markers
    /// exist, and the names are not models.
    fn with_unloadable_models() -> (tempfile::TempDir, Mimir) {
        let dir = tempfile::tempdir().unwrap();
        let mut m = open();
        m.paths = Paths::under_root(dir.path());
        m.config.embedding.model = "no-such-embedder".into();
        m.config.rerank.model = "no-such-reranker".into();
        std::fs::create_dir_all(&m.paths.models_dir).unwrap();
        for name in ["no-such-embedder", "no-such-reranker"] {
            std::fs::write(m.paths.models_dir.join(format!(".{name}.ready")), b"ok").unwrap();
            assert!(embed::model_ready(&m.paths, name));
        }
        (dir, m)
    }

    /// When and how the slot last failed; panics if it is in another state.
    fn failure<T>(slot: &ModelSlot<T>) -> (std::time::Instant, bool) {
        match slot {
            ModelSlot::Failed {
                at,
                allowed_download,
                ..
            } => (*at, *allowed_download),
            ModelSlot::Unloaded => panic!("no load was attempted"),
            ModelSlot::Loaded(_) => panic!("the load succeeded"),
        }
    }

    /// A failed load used to be final for the life of the process, so a
    /// long-lived server stayed BM25-only until restarted and an explicit
    /// fetch in the same process could not try again.
    #[test]
    fn a_failed_embedder_load_is_retried_on_the_slots_terms() {
        use std::time::{Duration, Instant};
        let (_dir, mut m) = with_unloadable_models();
        let t0 = Instant::now();
        assert!(m.ensure_embedder_at(false, t0).is_none());
        assert_eq!(failure(&m.embedder), (t0, false));

        let t1 = t0 + Duration::from_secs(1);
        assert!(m.ensure_embedder_at(false, t1).is_none());
        assert_eq!(
            failure(&m.embedder),
            (t0, false),
            "a hot path must not repeat the load inside the cooldown"
        );

        assert!(m.ensure_embedder_at(true, t1).is_none());
        assert_eq!(
            failure(&m.embedder),
            (t1, true),
            "a caller allowing the download tries at once"
        );

        // Two failures in a row by now, so the wait is the second one's.
        let t2 = t1 + embed::retry_after(2);
        assert!(m.ensure_embedder_at(false, t2).is_none());
        assert_eq!(failure(&m.embedder), (t2, false), "and anyone does later");
    }

    #[test]
    fn a_failed_reranker_load_is_retried_on_the_slots_terms() {
        use std::time::{Duration, Instant};
        let (_dir, mut m) = with_unloadable_models();
        let t0 = Instant::now();
        assert!(m.ensure_reranker_at(false, t0).is_none());
        assert_eq!(failure(&m.reranker), (t0, false));
        assert!(!m.reranker_loaded());

        let t1 = t0 + Duration::from_secs(1);
        m.ensure_reranker_at(false, t1);
        assert_eq!(failure(&m.reranker), (t0, false));
        m.ensure_reranker_at(true, t1);
        assert_eq!(failure(&m.reranker), (t1, true));
        m.ensure_reranker_at(false, t1 + embed::retry_after(2));
        assert_eq!(failure(&m.reranker), (t1 + embed::retry_after(2), false));
    }

    /// A model that was never downloaded is not a failed load: nothing was
    /// attempted, so nothing is cooling down and the first call after the
    /// download loads it.
    #[test]
    fn a_model_that_is_not_downloaded_records_no_failure() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = open();
        m.paths = Paths::under_root(dir.path());
        assert!(m.ensure_embedder(false).is_none());
        assert!(m.ensure_reranker(false).is_none());
        assert!(matches!(m.embedder, ModelSlot::Unloaded));
        assert!(matches!(m.reranker, ModelSlot::Unloaded));
    }

    /// Asking for a rerank by name and silently getting the fused order back
    /// is the one degradation a caller cannot see for themselves.
    #[test]
    fn an_explicit_rerank_reports_whether_it_happened() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = open();
        m.paths = Paths::under_root(dir.path());
        m.config.rerank.auto = "off".into();
        for marker in ["gamma", "beta", "plain"] {
            let mut n = NewNode::new(Kind::Memory);
            n.title = Some(format!("target {marker}"));
            n.body = Some(format!("alpha topic {marker}"));
            store::insert_node(&m.conn, n).unwrap();
        }
        let query = q("alpha topic");
        let ids = |out: &SearchOutcome| out.hits.iter().map(|h| h.node.id).collect::<Vec<_>>();

        let fused = m.search_with_outcome(&query, false).unwrap();
        assert_eq!(fused.rerank, RerankOutcome::NotAsked);
        assert_eq!(fused.hits.len(), 3);

        let asked = m.search_with_outcome(&query, true).unwrap();
        assert_eq!(
            asked.rerank,
            RerankOutcome::Unavailable,
            "no reranker downloaded and no daemon"
        );
        assert_eq!(ids(&asked), ids(&fused), "the fused order is what is left");

        // Only one memory says "gamma": a page of one has no order to change.
        let single = m.search_with_outcome(&q("gamma"), true).unwrap();
        assert_eq!(single.hits.len(), 1);
        assert_eq!(single.rerank, RerankOutcome::NothingToRerank);

        m.set_remote_inference(Box::new(FakeRemote {
            up: true,
            fail: false,
        }));
        let reranked = m.search_with_outcome(&query, true).unwrap();
        assert_eq!(reranked.rerank, RerankOutcome::Applied);
        assert!(reranked.hits[0]
            .node
            .body
            .as_deref()
            .unwrap()
            .contains("gamma"));

        // The tuple-returning wrapper is the same search without the report.
        let (hits, legs) = m.search_with_evidence(&query, true).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.node.id).collect::<Vec<_>>(),
            ids(&reranked)
        );
        assert_eq!(legs, reranked.legs);
    }

    /// Canned out-of-process backend: embeds everything to a dim-4 vector,
    /// scores rerank docs by marker words (gamma > beta > rest), and can be
    /// switched into down/failing modes to exercise every fallback edge.
    struct FakeRemote {
        up: bool,
        fail: bool,
    }

    impl RemoteInference for FakeRemote {
        fn available(&self) -> bool {
            self.up
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.fail {
                return Err(crate::error::Error::Embed("fake remote failure".into()));
            }
            Ok(texts.iter().map(|_| vec![0.5, 0.5, 0.5, 0.5]).collect())
        }
        fn rerank(&self, _query: &str, docs: &[String]) -> Result<Vec<f32>> {
            if self.fail {
                return Err(crate::error::Error::Embed("fake remote failure".into()));
            }
            Ok(docs
                .iter()
                .map(|d| {
                    if d.contains("gamma") {
                        0.9
                    } else if d.contains("beta") {
                        0.5
                    } else {
                        0.1
                    }
                })
                .collect())
        }
    }

    fn remember_one(m: &Mimir, text: &str) -> i64 {
        match memory::remember(
            &m.conn,
            memory::Remember {
                text: text.into(),
                mtype: crate::model::MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            memory::RememberOutcome::Created(n) => n.id,
            other => panic!("expected Created, got {other:?}"),
        }
    }

    #[test]
    fn embed_pending_persists_remote_vectors() {
        let mut m = open();
        let id = remember_one(&m, "SCRAM auth rejects non-ASCII passwords");
        m.set_remote_inference(Box::new(FakeRemote {
            up: true,
            fail: false,
        }));
        assert_eq!(m.embed_pending().unwrap(), 1);
        let (model, dim): (String, i64) = m
            .conn
            .query_row(
                "SELECT model, dim FROM embedding WHERE node_id = ?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(model, m.config.embedding.model);
        assert_eq!(dim, 4);
        assert!(!embed::has_pending(&m.conn, &m.config.embedding.model).unwrap());
    }

    #[test]
    fn embed_pending_remote_failure_leaves_rows_pending() {
        let mut m = open();
        remember_one(&m, "SCRAM auth rejects non-ASCII passwords");
        m.set_remote_inference(Box::new(FakeRemote {
            up: true,
            fail: true,
        }));
        // Remote errs; the local fallback has no model in this test env, so
        // the run is a clean no-op — nothing persisted, nothing poisoned.
        assert_eq!(m.embed_pending().unwrap(), 0);
        assert!(embed::has_pending(&m.conn, &m.config.embedding.model).unwrap());
    }

    /// Seed enough embeddable nodes to clear RERANK_MIN_STORE_NODES, plus
    /// three recall targets distinguished by marker words.
    fn seed_rerankable_store(m: &Mimir) {
        for i in 0..(RERANK_MIN_STORE_NODES as usize + 1) {
            let mut n = NewNode::new(Kind::Memory);
            n.title = Some(format!("seed {i}"));
            n.body = Some("filler".into());
            store::insert_node(&m.conn, n).unwrap();
        }
        for marker in ["gamma", "beta", "plain"] {
            let mut n = NewNode::new(Kind::Memory);
            n.title = Some(format!("target {marker}"));
            n.body = Some(format!("alpha topic {marker}"));
            store::insert_node(&m.conn, n).unwrap();
        }
    }

    #[test]
    fn warm_auto_rerank_fires_via_remote_and_reorders() {
        let mut m = open();
        m.config.rerank.auto = "warm".into();
        seed_rerankable_store(&m);

        let query = q("alpha topic");
        // Warm + nothing resident + no remote: never fires.
        assert!(!m.should_auto_rerank(&query));
        // A down remote changes nothing.
        m.set_remote_inference(Box::new(FakeRemote {
            up: false,
            fail: false,
        }));
        assert!(!m.should_auto_rerank(&query));
        let fused: Vec<i64> = m
            .search_with(&query, false)
            .unwrap()
            .iter()
            .map(|h| h.node.id)
            .collect();
        assert!(fused.len() >= 3);

        // Live remote: warm fires and the fake's scores dictate the order.
        m.set_remote_inference(Box::new(FakeRemote {
            up: true,
            fail: false,
        }));
        assert!(m.should_auto_rerank(&query));
        let hits = m.search_with(&query, false).unwrap();
        let bodies: Vec<&str> = hits
            .iter()
            .take(2)
            .map(|h| h.node.body.as_deref().unwrap())
            .collect();
        assert!(bodies[0].contains("gamma"), "got {bodies:?}");
        assert!(bodies[1].contains("beta"), "got {bodies:?}");

        // Failing remote: warm still gates true, but the rerank degrades to
        // the fused order instead of cold-loading a local model.
        m.set_remote_inference(Box::new(FakeRemote {
            up: true,
            fail: true,
        }));
        let degraded: Vec<i64> = m
            .search_with(&query, false)
            .unwrap()
            .iter()
            .map(|h| h.node.id)
            .collect();
        assert_eq!(degraded, fused, "remote failure must keep fused order");
        assert!(
            !m.reranker_loaded(),
            "auto-warm must never pull a local model into the process"
        );
    }

    use std::path::PathBuf;

    use crate::model::{Actor, MutationOp};
    use crate::scope::{canonical_root, Checkout};

    /// When the fixtures' worktrees were made (`scope::worktree_created_at`):
    /// long before any memory a test writes, so every one of those is the
    /// worktree's to move unless a test says otherwise.
    const MADE_AT: i64 = 1_000_000;

    /// Set a file's modification time to `at`, in unix seconds.
    fn set_mtime(path: &Path, at: i64) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(at as u64))
            .unwrap();
    }

    /// A repository's main checkout and one linked worktree of it, as the
    /// files `git worktree add` leaves (`scope::Checkout::of` reads nothing
    /// else), made at [`MADE_AT`]. `marker` is the committed `.mimir`,
    /// which a worktree checks out like any tracked file; `origin` gives the
    /// main checkout an `origin` remote.
    fn repo_with_worktree(tmp: &Path, marker: Option<&str>, origin: bool) -> (PathBuf, PathBuf) {
        let main = tmp.join("main");
        let wt = tmp.join("wt");
        let gitdir = main.join(".git/worktrees/wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        set_mtime(&gitdir.join("commondir"), MADE_AT);
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        if origin {
            std::fs::write(
                main.join(".git/config"),
                "[remote \"origin\"]\n\turl = git@example.invalid:o/r.git\n",
            )
            .unwrap();
        }
        if let Some(marker) = marker {
            for root in [&main, &wt] {
                std::fs::write(root.join(".mimir"), marker).unwrap();
            }
        }
        (main, wt)
    }

    fn memory_under(m: &Mimir, project: Option<i64>, title: &str) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.title = Some(title.into());
        new.body = Some(format!("{title} body"));
        new.project_id = project;
        store::insert_node(&m.conn, new).unwrap()
    }

    fn project_rows(m: &Mimir) -> i64 {
        m.conn
            .query_row(
                "SELECT count(*) FROM node WHERE kind = 'project'",
                [],
                |r| r.get(0),
            )
            .unwrap()
    }

    fn ledger_rows(m: &Mimir) -> i64 {
        m.conn
            .query_row("SELECT count(*) FROM mutation", [], |r| r.get(0))
            .unwrap()
    }

    fn holds_no_sync_identity(project: &Node) -> bool {
        project.meta.get("portable_key").is_none_or(|v| v.is_null())
            && !replicate::project_is_syncable(&project.meta)
    }

    /// The decision, for a repository with no portable key, one keyed by its
    /// `origin` remote and one with a committed `.mimir` id, whichever
    /// checkout Mimir runs in first: the worktree has a checkout project of
    /// its own, its memory project is the main checkout's, neither row's
    /// path ever moves, and only the home row carries the sync identity.
    #[test]
    fn a_linked_worktree_has_its_own_checkout_project_and_its_repositorys_memory_project() {
        let repos: [(Option<&str>, bool, Option<&str>); 3] = [
            (None, false, None),
            (None, true, Some("git:example.invalid/o/r")),
            (Some("id = \"REPO1\"\nsync = true\n"), true, Some("REPO1")),
        ];
        for (marker, origin, key) in repos {
            for worktree_first in [false, true] {
                let case = format!("key {key:?}, worktree first: {worktree_first}");
                let tmp = tempfile::tempdir().unwrap();
                let (main, wt) = repo_with_worktree(tmp.path(), marker, origin);
                let m = open();
                let order: [&Path; 2] = if worktree_first {
                    [&wt, &main]
                } else {
                    [&main, &wt]
                };
                // Twice round, so a path that flips on the way back shows.
                for _ in 0..2 {
                    for dir in order {
                        m.projects_for_cwd(dir).unwrap();
                    }
                }
                let at_main = m.projects_for_cwd(&main).unwrap();
                let at_wt = m.projects_for_cwd(&wt).unwrap();

                let home = at_main.checkout.clone().expect("the main checkout");
                assert_eq!(at_main.checkout_kind, Checkout::Standalone, "{case}");
                assert_eq!(at_main.memory_id(), Some(home.id), "{case}");

                let own = at_wt.checkout.clone().expect("the worktree");
                assert_ne!(own.id, home.id, "{case}: one checkout project each");
                assert_eq!(at_wt.memory_id(), Some(home.id), "{case}");
                assert_eq!(
                    at_wt.checkout_kind.home(),
                    Some(Path::new(&canonical_root(&main))),
                    "{case}"
                );
                assert_eq!(project_rows(&m), 2, "{case}");

                // The older entry points mean the checkout; the new one, the
                // memory project.
                assert_eq!(m.project_for_cwd(&wt).unwrap().unwrap().id, own.id);
                assert_eq!(m.detect_project(&wt).unwrap().0.unwrap().id, own.id);
                assert_eq!(m.memory_project_for_cwd(&wt).unwrap().unwrap().id, home.id);
                assert_eq!(
                    m.memory_project_for_cwd(&main).unwrap().unwrap().id,
                    home.id
                );

                let row = |id: i64| store::get_node(&m.conn, id).unwrap();
                assert_eq!(
                    row(home.id).path,
                    Some(canonical_root(&main)),
                    "{case}: the home row stays on the main checkout"
                );
                assert_eq!(row(own.id).path, Some(canonical_root(&wt)), "{case}");
                assert!(holds_no_sync_identity(&row(own.id)), "{case}");
                assert!(holds_no_sync_identity(&own), "{case}: as returned, too");
                if let Some(key) = key {
                    assert_eq!(
                        store::find_project_by_key(&m.conn, key)
                            .unwrap()
                            .map(|p| p.id),
                        Some(home.id),
                        "{case}: the portable key finds the home row and no other"
                    );
                }
                assert_eq!(
                    replicate::project_is_syncable(&row(home.id).meta),
                    marker.is_some(),
                    "{case}: sync is the home row's, as its `.mimir` says"
                );
            }
        }
    }

    #[test]
    fn the_home_project_is_created_on_first_contact_from_a_worktree() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let nested = wt.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        let m = open();
        assert_eq!(project_rows(&m), 0);

        let at_wt = m.projects_for_cwd(&nested).unwrap();
        let home = at_wt.memory.expect("a memory project");
        assert_eq!(home.path, Some(canonical_root(&main)));
        assert_eq!(home.title.as_deref(), Some("main"));
        assert_eq!(
            at_wt.checkout.unwrap().path,
            Some(canonical_root(&wt)),
            "from a nested directory, the checkout is still the worktree's root"
        );
        // Running in the main checkout afterwards finds that same row.
        assert_eq!(m.project_for_cwd(&main).unwrap().unwrap().id, home.id);
        assert_eq!(project_rows(&m), 2);
    }

    /// A repository with a `.mimir` id had one row for every checkout, bound
    /// to whichever ran last. Resolving the home first binds it back to the
    /// main checkout; the worktree gets a row of its own; and the memories
    /// under the shared row are where they belong already.
    #[test]
    fn a_keyed_row_last_bound_to_a_worktree_goes_back_to_the_main_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) =
            repo_with_worktree(tmp.path(), Some("id = \"REPO1\"\nsync = true\n"), false);
        let m = open();
        let shared = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        store::set_project_identity(&m.conn, shared.id, Some("REPO1"), true).unwrap();
        let kept = memory_under(&m, Some(shared.id), "the repository's memory");
        m.conn
            .execute("UPDATE node SET updated_at = 1000", [])
            .unwrap();
        let ledger = ledger_rows(&m);

        let at_wt = m.projects_for_cwd(&wt).unwrap();
        assert_eq!(
            at_wt.memory_id(),
            Some(shared.id),
            "the keyed row is the home"
        );
        let home = store::get_node(&m.conn, shared.id).unwrap();
        assert_eq!(home.path, Some(canonical_root(&main)));
        assert_eq!(home.title.as_deref(), Some("main"));
        let own = at_wt.checkout.unwrap();
        assert_ne!(own.id, shared.id);
        assert_eq!(own.path, Some(canonical_root(&wt)));
        assert!(holds_no_sync_identity(&own));

        let kept = store::get_node(&m.conn, kept.id).unwrap();
        assert_eq!(kept.project_id, Some(shared.id));
        assert_eq!(kept.updated_at, 1000, "it never left its project");
        assert_eq!(ledger_rows(&m), ledger);
    }

    /// A second linked worktree of the fixture's main checkout, at
    /// `<tmp>/<name>`, made at [`MADE_AT`].
    fn second_worktree(tmp: &Path, main: &Path, name: &str) -> PathBuf {
        let wt = tmp.join(name);
        let gitdir = main.join(".git/worktrees").join(name);
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        set_mtime(&gitdir.join("commondir"), MADE_AT);
        wt
    }

    /// One project row as the store holds it now.
    fn project_row(m: &Mimir, id: i64) -> Node {
        store::get_node(&m.conn, id).unwrap()
    }

    fn sync_identity(project: &Node) -> (Option<String>, bool) {
        (
            project
                .meta
                .get("portable_key")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            replicate::project_is_syncable(&project.meta),
        )
    }

    /// The main checkout lags: the worktree's branch carries a `.mimir` id
    /// the main checkout does not have. The worktree keeps resolving the
    /// way it did before linked worktrees were told apart, by its own key:
    /// its row keeps the key and the sync flag, its memories stay under it,
    /// and the main checkout's project is not created or touched, whether
    /// or not the resolution may move.
    #[test]
    fn a_worktree_with_a_mimir_id_its_main_checkout_lacks_keeps_its_own_project() {
        for origin in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let (_main, wt) = repo_with_worktree(tmp.path(), None, origin);
            std::fs::write(wt.join(".mimir"), "id = \"BRANCH1\"\nsync = true\n").unwrap();
            let m = open();
            // What an older Mimir left: the worktree as a keyed, synced
            // project of its own, with a memory filed under it.
            let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
            store::set_project_identity(&m.conn, own.id, Some("BRANCH1"), true).unwrap();
            let kept = memory_under(&m, Some(own.id), "learned on the branch");
            let ledger = ledger_rows(&m);

            let (at_wt, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
            assert!(moved.is_none(), "origin {origin}");
            assert_eq!(at_wt.filing, MemoryFiling::OwnKey);
            assert!(at_wt.checkout_kind.is_linked_worktree(), "still a worktree");
            assert_eq!(at_wt.checkout_id(), Some(own.id));
            assert_eq!(at_wt.memory_id(), Some(own.id), "its own memory project");
            assert!(at_wt.query_reach().is_uniform());
            assert_eq!(
                sync_identity(&project_row(&m, own.id)),
                (Some("BRANCH1".into()), true),
                "origin {origin}: nothing is stripped"
            );
            assert_eq!(
                store::find_project_by_key(&m.conn, "BRANCH1")
                    .unwrap()
                    .map(|p| p.id),
                Some(own.id)
            );
            assert_eq!(
                store::get_node(&m.conn, kept.id).unwrap().project_id,
                Some(own.id)
            );
            assert_eq!(ledger_rows(&m), ledger, "nothing moved");
            assert_eq!(project_rows(&m), 1, "the main checkout's row is not made");
            let note = at_wt.worktree_note().unwrap();
            assert!(note.contains("keeps a project of its own"), "{note}");
            assert!(m.worktree_leftovers(&at_wt).unwrap().is_none());

            // And on a store that never saw it: the row is made with the
            // key, as it was before.
            let fresh = open();
            let at_wt = fresh.projects_for_cwd(&wt).unwrap();
            assert_eq!(at_wt.filing, MemoryFiling::OwnKey);
            assert_eq!(
                sync_identity(&project_row(&fresh, at_wt.memory_id().unwrap())),
                (Some("BRANCH1".into()), true)
            );
            assert_eq!(project_rows(&fresh), 1);
        }
    }

    /// A main checkout whose `.mimir` cannot be parsed has no key from it,
    /// so a worktree whose marker reads is not the same repository as far
    /// as the keys can tell, and keeps its own project and memories.
    #[test]
    fn an_unparseable_marker_in_the_main_checkout_leaves_the_worktree_its_own_project() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        std::fs::write(main.join(".mimir"), "id = = not toml\n").unwrap();
        std::fs::write(wt.join(".mimir"), "id = \"REPO1\"\nsync = true\n").unwrap();
        assert!(matches!(
            crate::scope::read_marker_state(&main),
            crate::scope::MarkerState::Unparseable(_)
        ));
        let m = open();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        store::set_project_identity(&m.conn, own.id, Some("REPO1"), true).unwrap();
        let kept = memory_under(&m, Some(own.id), "learned on the branch");

        let (at_wt, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert!(moved.is_none());
        assert_eq!(at_wt.filing, MemoryFiling::OwnKey);
        assert_eq!(at_wt.memory_id(), Some(own.id));
        assert_eq!(
            sync_identity(&project_row(&m, own.id)),
            (Some("REPO1".into()), true)
        );
        assert_eq!(
            store::get_node(&m.conn, kept.id).unwrap().project_id,
            Some(own.id)
        );

        // Both unparseable: no key on either side, so the worktree shares.
        std::fs::write(wt.join(".mimir"), "also = = broken\n").unwrap();
        assert_eq!(m.projects_for_cwd(&wt).unwrap().filing, MemoryFiling::Home);
    }

    /// Two worktrees carry the id the main checkout lacks, and the one keyed
    /// row was last bound to the other worktree. Before linked worktrees
    /// were told apart a resolution adopted that row onto its own path, and
    /// so it still does: key, sync flag and memories all stay on it.
    #[test]
    fn a_keyed_row_last_bound_to_another_worktree_is_adopted_as_before() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, true);
        let other = second_worktree(tmp.path(), &main, "wt2");
        for root in [&wt, &other] {
            std::fs::write(root.join(".mimir"), "id = \"REPOX\"\nsync = true\n").unwrap();
        }
        let m = open();
        let shared = store::ensure_project(&m.conn, &canonical_root(&other), "wt2").unwrap();
        store::set_project_identity(&m.conn, shared.id, Some("REPOX"), true).unwrap();
        let memories = [
            memory_under(&m, Some(shared.id), "importer retries with backoff"),
            memory_under(&m, Some(shared.id), "tenant table is sharded by region"),
        ];
        let ledger = ledger_rows(&m);

        let (at_wt, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert!(moved.is_none());
        assert_eq!(at_wt.filing, MemoryFiling::OwnKey);
        assert_eq!(at_wt.memory_id(), Some(shared.id));
        let row = project_row(&m, shared.id);
        assert_eq!(
            row.path,
            Some(canonical_root(&wt)),
            "adopted onto this path"
        );
        assert_eq!(sync_identity(&row), (Some("REPOX".into()), true));
        for memory in &memories {
            assert_eq!(
                store::get_node(&m.conn, memory.id).unwrap().project_id,
                Some(shared.id)
            );
        }
        assert_eq!(ledger_rows(&m), ledger);
        assert_eq!(project_rows(&m), 1);
    }

    /// Memories captured in a worktree while it was a project of its own
    /// move to the home project the first time a resolution that may move
    /// runs there: each through the ledger, as the system, with the reason.
    /// A resolution that may not move leaves them. What is derived from the
    /// worktree's files stays. A second resolution finds nothing left and
    /// writes nothing at all.
    #[test]
    fn memories_filed_under_a_worktrees_own_row_move_home_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let m = open();
        let home = store::ensure_project(&m.conn, &canonical_root(&main), "main").unwrap();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        let stranded = [
            memory_under(&m, Some(own.id), "learned in the worktree"),
            memory_under(&m, Some(own.id), "an older wording"),
            memory_under(&m, Some(own.id), "forgotten in the worktree"),
        ];
        store::supersede(&m.conn, stranded[1].id, stranded[0].id, Actor::Human, None).unwrap();
        store::soft_delete(&m.conn, stranded[2].id, Actor::Human, None).unwrap();
        let native = memory_under(&m, Some(home.id), "learned in the main checkout");
        let global = memory_under(&m, None, "true everywhere");
        let mut symbol = NewNode::new(Kind::Symbol);
        symbol.title = Some("in_the_worktrees_graph".into());
        symbol.project_id = Some(own.id);
        let symbol = store::insert_node(&m.conn, symbol).unwrap();
        m.conn
            .execute("UPDATE node SET updated_at = 1000", [])
            .unwrap();
        let ledger = ledger_rows(&m);
        let row = |id: i64| store::get_node(&m.conn, id).unwrap();

        // Resolving reads: everything is where it was.
        let at_wt = m.projects_for_cwd(&wt).unwrap();
        assert_eq!(at_wt.memory_id(), Some(home.id));
        for memory in &stranded {
            assert_eq!(row(memory.id).project_id, Some(own.id));
        }
        assert_eq!(ledger_rows(&m), ledger);

        let (at_wt, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert_eq!(at_wt.memory_id(), Some(home.id));
        assert_eq!(at_wt.checkout_id(), Some(own.id));
        let moved = moved.expect("something moved");
        assert_eq!(moved.count, stranded.len());
        assert_eq!((moved.worktree.id, moved.home.id), (own.id, home.id));
        assert!(!moved.published, "the home project does not sync");
        assert_eq!(
            moved.line(),
            format!(
                "moved 3 memories from linked worktree {} to project main (`mimir audit` lists \
                 them)",
                canonical_root(&wt)
            )
        );
        for memory in &stranded {
            let moved = row(memory.id);
            assert_eq!(moved.project_id, Some(home.id), "{:?}", memory.title);
            assert!(moved.updated_at > 1000, "bumped, so the move syncs");
            let last = audit::history(&m.conn, memory.id).unwrap().pop().unwrap();
            assert_eq!(last.op, MutationOp::Reproject);
            assert_eq!(last.actor, Actor::System);
            assert_eq!(last.reason.as_deref(), Some(WORKTREE_MEMORIES_MOVED));
        }
        assert_eq!(ledger_rows(&m), ledger + stranded.len() as i64);
        assert_eq!(
            (row(native.id).project_id, row(native.id).updated_at),
            (Some(home.id), 1000)
        );
        assert_eq!(
            (row(global.id).project_id, row(global.id).updated_at),
            (None, 1000)
        );
        assert_eq!(
            (row(symbol.id).project_id, row(symbol.id).updated_at),
            (Some(own.id), 1000),
            "the code graph stays with the checkout"
        );

        // Nothing left: nothing happens, in either checkout.
        m.conn
            .execute("UPDATE node SET updated_at = 2000", [])
            .unwrap();
        let (ledger, changes) = (ledger_rows(&m), m.conn.total_changes());
        for dir in [&wt, &main, &wt] {
            let (_, moved) = m.projects_for_cwd_moving_memories_home(dir).unwrap();
            assert!(moved.is_none());
        }
        for memory in &stranded {
            assert_eq!(row(memory.id).updated_at, 2000, "{:?}", memory.title);
        }
        assert_eq!(ledger_rows(&m), ledger);
        assert_eq!(
            m.conn.total_changes(),
            changes,
            "a resolution with nothing to move writes no row"
        );
    }

    /// The resolution hooks, `status` and the daemon use never moves and
    /// never writes for a pending move, however many times it runs; it
    /// reports what is waiting instead.
    #[test]
    fn a_resolution_that_may_not_move_leaves_the_memories_and_reports_them() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let m = open();
        let home = store::ensure_project(&m.conn, &canonical_root(&main), "main").unwrap();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        let pending = memory_under(&m, Some(own.id), "learned in the worktree");
        let at_wt = m.projects_for_cwd(&wt).unwrap();
        let (ledger, changes) = (ledger_rows(&m), m.conn.total_changes());
        for _ in 0..3 {
            m.projects_for_cwd(&wt).unwrap();
        }
        assert_eq!((ledger_rows(&m), m.conn.total_changes()), (ledger, changes));
        assert_eq!(
            store::get_node(&m.conn, pending.id).unwrap().project_id,
            Some(own.id)
        );
        let left = m.worktree_leftovers(&at_wt).unwrap().unwrap();
        assert_eq!(
            left,
            WorktreeLeftovers {
                to_move: 1,
                kept: 0,
                worktree_created_at: Some(MADE_AT),
            }
        );
        let lines = left.lines(&home);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].starts_with("1 memory is still filed under this worktree's own project")
                && lines[0].contains("moves it to project main"),
            "{lines:?}"
        );
        assert_eq!(m.conn.total_changes(), changes, "the report only reads");

        // The main checkout has nothing of the kind to report.
        let at_main = m.projects_for_cwd(&main).unwrap();
        assert!(m.worktree_leftovers(&at_main).unwrap().is_none());
    }

    /// A worktree's own row only ever held what was written after the
    /// worktree was made; an older memory under the same path belongs to a
    /// checkout that was there before, possibly of another repository, and
    /// stays where it is. The bound is the second the worktree was made.
    #[test]
    fn only_memories_written_since_the_worktree_was_made_move_home() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let made = 1_700_000_000;
        set_mtime(&main.join(".git/worktrees/wt/commondir"), made);
        let m = open();
        let home = store::ensure_project(&m.conn, &canonical_root(&main), "main").unwrap();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        let created = |title: &str, at: i64| {
            let node = memory_under(&m, Some(own.id), title);
            m.conn
                .execute(
                    "UPDATE node SET created_at = ?2 WHERE id = ?1",
                    rusqlite::params![node.id, at],
                )
                .unwrap();
            node
        };
        let before = created("another repository's, from this path", made - 1);
        let at = created("written as the worktree was made", made);
        let after = created("written in the worktree", made + 60);

        let at_wt = m.projects_for_cwd(&wt).unwrap();
        assert_eq!(
            m.worktree_leftovers(&at_wt).unwrap().unwrap(),
            WorktreeLeftovers {
                to_move: 2,
                kept: 1,
                worktree_created_at: Some(made),
            }
        );
        let (_, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert_eq!(moved.unwrap().count, 2);
        let project_of = |node: &Node| store::get_node(&m.conn, node.id).unwrap().project_id;
        assert_eq!(project_of(&before), Some(own.id), "older than the worktree");
        assert_eq!(project_of(&at), Some(home.id));
        assert_eq!(project_of(&after), Some(home.id));

        // What stays is reported as staying, and the probe asks the same
        // bound: nothing more moves, and nothing is written for it.
        let left = m.worktree_leftovers(&at_wt).unwrap().unwrap();
        assert_eq!((left.to_move, left.kept), (0, 1));
        assert!(left.lines(&home)[0].contains("older than the worktree"));
        let changes = m.conn.total_changes();
        let (_, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert!(moved.is_none());
        assert_eq!(m.conn.total_changes(), changes);
    }

    /// A handoff an older Mimir wrote in a worktree records no checkout.
    /// Moved home without one, it would be restored by a session in any
    /// checkout of the repository; the move ties it to the worktree it was
    /// written in, in the same transaction. One that records a checkout
    /// keeps it.
    #[test]
    fn a_handoff_moved_home_keeps_the_checkout_it_was_written_in() {
        use crate::context_guard::{self, HANDOFF_TAG};
        use crate::model::Scope;
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let other = second_worktree(tmp.path(), &main, "wt-b");
        let m = open();
        store::ensure_project(&m.conn, &canonical_root(&main), "main").unwrap();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        let handoff = |text: &str| {
            let mut new = NewNode::new(Kind::Memory);
            new.title = Some(text.into());
            new.body = Some(text.into());
            new.project_id = Some(own.id);
            new.tags = vec![HANDOFF_TAG.into()];
            store::insert_node(&m.conn, new).unwrap()
        };
        let legacy = handoff("handoff: A is mid-rebase onto main");
        let recorded = handoff("handoff: a snapshot that knows where it was written");
        context_guard::set_handoff_checkout(&m.conn, recorded.id, "/somewhere/else").unwrap();
        let plain = memory_under(&m, Some(own.id), "a plain note");

        let (at_wt, moved) = m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert_eq!(moved.unwrap().count, 3);
        let get = |node: &Node| store::get_node(&m.conn, node.id).unwrap();
        let wt_path = canonical_root(&wt);
        assert_eq!(
            context_guard::handoff_checkout(&get(&legacy)),
            Some(wt_path.as_str())
        );
        assert_eq!(
            context_guard::handoff_checkout(&get(&recorded)),
            Some("/somewhere/else")
        );
        assert_eq!(context_guard::handoff_checkout(&get(&plain)), None);

        let scope = Scope::Project(at_wt.memory_id().unwrap());
        let pending_in = |root: &Path| {
            context_guard::current_handoff_in(&m.conn, scope, Some(&canonical_root(root)))
                .unwrap()
                .map(|n| n.id)
        };
        assert_eq!(pending_in(&other), None, "not worktree B's");
        assert_eq!(pending_in(&main), None, "nor the main checkout's");
        assert_eq!(pending_in(&wt), Some(legacy.id));
    }

    /// The row of a worktree that shares its home's memory project says
    /// which project that is (`store::WORKTREE_OF`); no other row does, and
    /// a row that stops sharing loses it. Recorded once: the steady state
    /// writes nothing.
    #[test]
    fn a_worktree_row_records_its_home_while_it_shares_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let m = open();
        let at_wt = m.projects_for_cwd(&wt).unwrap();
        let (own, home) = (
            at_wt.checkout.clone().unwrap(),
            at_wt.memory.clone().unwrap(),
        );
        let own_now = project_row(&m, own.id);
        assert_eq!(
            own_now
                .meta
                .get(store::WORKTREE_OF)
                .and_then(|v| v.as_str()),
            Some(home.uid.as_str())
        );
        assert_eq!(
            store::worktree_home(&m.conn, &own_now)
                .unwrap()
                .map(|p| p.id),
            Some(home.id)
        );
        assert!(store::worktree_home(&m.conn, &project_row(&m, home.id))
            .unwrap()
            .is_none());
        let changes = m.conn.total_changes();
        m.projects_for_cwd(&wt).unwrap();
        m.projects_for_cwd(&main).unwrap();
        assert_eq!(m.conn.total_changes(), changes, "nothing to record twice");

        // The branch gets an id the main checkout lacks: the worktree keeps
        // its own project, and its row no longer names a home.
        std::fs::write(wt.join(".mimir"), "id = \"BRANCH1\"\n").unwrap();
        let own_key = m.projects_for_cwd(&wt).unwrap();
        assert_eq!(own_key.filing, MemoryFiling::OwnKey);
        let row = project_row(&m, own_key.checkout_id().unwrap());
        assert_eq!(row.meta.get(store::WORKTREE_OF), None);
        assert!(store::worktree_home(&m.conn, &row).unwrap().is_none());
    }

    /// The `worktree_of` write is allowed on a hook's resolution, but it
    /// must not make the hook wait: with another process holding the write
    /// lock it is skipped at once, and recorded by a later resolution.
    #[test]
    fn recording_a_worktrees_home_does_not_wait_for_a_busy_store() {
        let tmp = tempfile::tempdir().unwrap();
        let (_main, wt) = repo_with_worktree(tmp.path(), None, false);
        let store_dir = tempfile::tempdir().unwrap();
        let m = Mimir::open_at(Paths::under_root(store_dir.path())).unwrap();
        let own = m.projects_for_cwd(&wt).unwrap().checkout.unwrap();
        store::set_worktree_home(&m.conn, own.id, None).unwrap();

        let writer = rusqlite::Connection::open(&m.paths.db_file).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let started = std::time::Instant::now();
        let at_wt = m.projects_for_cwd(&wt).expect("the resolution stands");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "waited {:?}",
            started.elapsed()
        );
        assert_eq!(at_wt.filing, MemoryFiling::Home);
        assert_eq!(project_row(&m, own.id).meta.get(store::WORKTREE_OF), None);
        let timeout: i64 = m
            .conn
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert_eq!(timeout, 10_000, "the connection's own timeout is back");
        writer.execute_batch("COMMIT").unwrap();

        m.projects_for_cwd(&wt).unwrap();
        assert!(project_row(&m, own.id)
            .meta
            .get(store::WORKTREE_OF)
            .is_some());
    }

    /// The move is best-effort. When it cannot be made the worktree still
    /// resolves, nothing is half-moved, and the next resolution moves it.
    #[test]
    fn a_move_that_fails_does_not_fail_the_resolution() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let m = open();
        let home = store::ensure_project(&m.conn, &canonical_root(&main), "main").unwrap();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        let first = memory_under(&m, Some(own.id), "first");
        let second = memory_under(&m, Some(own.id), "second");
        m.conn
            .execute_batch(&format!(
                "CREATE TRIGGER refuse_second BEFORE UPDATE OF project_id ON node
                 WHEN old.id = {} BEGIN SELECT RAISE(ABORT, 'refused'); END;",
                second.id
            ))
            .unwrap();
        let row = |id: i64| store::get_node(&m.conn, id).unwrap();

        let (at_wt, moved) = m
            .projects_for_cwd_moving_memories_home(&wt)
            .expect("the resolution stands");
        assert!(moved.is_none());
        assert_eq!(at_wt.memory_id(), Some(home.id));
        assert_eq!(at_wt.checkout_id(), Some(own.id));
        assert_eq!(row(first.id).project_id, Some(own.id), "all or nothing");
        assert_eq!(row(second.id).project_id, Some(own.id));

        m.conn.execute_batch("DROP TRIGGER refuse_second").unwrap();
        m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        assert_eq!(row(first.id).project_id, Some(home.id));
        assert_eq!(row(second.id).project_id, Some(home.id));
    }

    /// The vector matrix caches each row's project and cannot see a write
    /// made on its own connection, so a move has to drop it: otherwise the
    /// vector leg of a long-lived engine keeps filtering the moved memories
    /// by the project they left.
    #[test]
    fn moving_memories_home_drops_the_cached_vector_matrix() {
        let tmp = tempfile::tempdir().unwrap();
        let (_main, wt) = repo_with_worktree(tmp.path(), None, false);
        let mut m = open();
        let own = store::ensure_project(&m.conn, &canonical_root(&wt), "wt").unwrap();
        MatrixCache::ensure(&m.conn, "some-model", &mut m.matrix).unwrap();
        assert!(m.matrix.is_some());

        // Nothing moved: the cache stands.
        m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        m.search_with_legs_bm25_only(&q("anything at all")).unwrap();
        assert!(m.matrix.is_some());

        memory_under(&m, Some(own.id), "learned in the worktree");
        m.projects_for_cwd_moving_memories_home(&wt).unwrap();
        m.search_with_legs_bm25_only(&q("anything at all")).unwrap();
        assert!(m.matrix.is_none(), "the next search starts from the store");
    }

    #[test]
    fn the_memory_project_of_a_checkout_row_is_the_one_its_directory_resolves_to() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let m = open();
        let at_wt = m.projects_for_cwd(&wt).unwrap();
        let (own, home) = (at_wt.checkout.unwrap(), at_wt.memory.unwrap());
        assert_eq!(m.memory_project_of(&own).unwrap().id, home.id);
        assert_eq!(m.memory_project_of(&home).unwrap().id, home.id);
        assert_eq!(home.path, Some(canonical_root(&main)));

        // A row with no directory to ask (a synced shadow), and one whose
        // directory is gone, are their own memory project.
        let shadow = store::get_node(
            &m.conn,
            store::resolve_or_shadow_project(&m.conn, "git:example.invalid/o/else").unwrap(),
        )
        .unwrap();
        assert_eq!(m.memory_project_of(&shadow).unwrap().id, shadow.id);
        std::fs::remove_dir_all(&wt).unwrap();
        assert_eq!(m.memory_project_of(&own).unwrap().id, own.id);
    }

    /// `recall` in a linked worktree, from the resolver to the page: the
    /// reach the two projects give is the repository's memories with the
    /// worktree's own code, in one ranked list. In the main checkout the
    /// two projects are one and the reach is the one scope it always was.
    #[test]
    fn a_question_from_a_linked_worktree_reaches_its_memories_and_its_own_code() {
        use crate::model::Scope;
        use crate::search::Reach;
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = repo_with_worktree(tmp.path(), None, false);
        let mut m = open();
        let at_wt = m.projects_for_cwd(&wt).unwrap();
        let (home, own) = (at_wt.memory_id().unwrap(), at_wt.checkout_id().unwrap());
        assert_eq!(
            at_wt.query_reach(),
            Reach {
                memories: Scope::Project(home),
                code_and_docs: Scope::Project(own),
            }
        );
        let at_main = m.projects_for_cwd(&main).unwrap();
        assert_eq!(
            at_main.query_reach(),
            Reach::uniform(Scope::Project(home)),
            "one project, one scope"
        );
        assert_eq!(
            m.projects_for_cwd(tmp.path()).unwrap().query_reach(),
            Reach::uniform(Scope::All),
            "outside any project a question searches everything"
        );

        memory_under(&m, Some(home), "the quorum gotcha");
        let symbol = |project: i64, title: &str| {
            let mut new = NewNode::new(Kind::Symbol);
            new.title = Some(title.into());
            new.body = Some(format!("fn {title}() checks the quorum"));
            new.project_id = Some(project);
            store::insert_node(&m.conn, new).unwrap();
        };
        symbol(own, "quorum_on_the_branch");
        symbol(home, "quorum_in_main");

        let asked_from = |m: &mut Mimir, reach: Reach| {
            let mut query = q("quorum");
            query.scope = reach.memories;
            let mut titles: Vec<String> = m
                .search_with_outcome_within(&query, reach, false)
                .unwrap()
                .hits
                .into_iter()
                .filter_map(|hit| hit.node.title)
                .collect();
            titles.sort();
            titles
        };
        assert_eq!(
            asked_from(&mut m, at_wt.query_reach()),
            ["quorum_on_the_branch", "the quorum gotcha"]
        );
        assert_eq!(
            asked_from(&mut m, at_main.query_reach()),
            ["quorum_in_main", "the quorum gotcha"]
        );
        // The method without a reach is the uniform one.
        let mut query = q("quorum");
        query.scope = Scope::Project(home);
        let uniform: Vec<i64> = m
            .search_with_outcome(&query, false)
            .unwrap()
            .hits
            .iter()
            .map(|hit| hit.node.id)
            .collect();
        let same: Vec<i64> = m
            .search_with_outcome_within(&query, Reach::uniform(query.scope), false)
            .unwrap()
            .hits
            .iter()
            .map(|hit| hit.node.id)
            .collect();
        assert_eq!(uniform, same);
        assert_eq!(uniform.len(), 2);
    }
}
