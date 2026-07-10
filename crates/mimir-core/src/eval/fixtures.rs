//! Hand-authored corpus + labeled question set for the eval harness.
//!
//! The corpus spans six question categories: code-structure (symbols),
//! code-behavior (what code/docs say a piece of the system does),
//! gotcha-decision (a memory documenting a fact or a choice),
//! drift-preventer (a memory documenting a past mistake and its fix, so an
//! agent doesn't repeat it), code-vs-memory (scenario family 5), and
//! identifier-lookup (scenario family 5b). A few fixtures are pure
//! distractors, targeted by no question, to keep precision meaningful.
//!
//! Fixtures are grouped into `topic` clusters. Real-model mode ignores
//! `topic` entirely (it embeds the actual text); hermetic mode uses it to
//! generate deterministic synthetic vectors so items in the same topic sit
//! near each other and unrelated topics land far apart.
//!
//! ## Headroom scenarios
//!
//! The original 13-fixture/9-question corpus (still present below, marked
//! `Core`) scored MRR 1.0 / recall 1.0 everywhere — every answer was too
//! cleanly separable from its distractors to show a future ranking change
//! (recency boost, gotcha/decision type priority, a lower RRF k) moving the
//! needle. Scenario families below are deliberately harder, added for
//! that purpose:
//!
//! 1. **Recency** (`mem_decision_*_old` / `*_new` pairs) — a superseded
//!    Decision vs. its replacement. Ground truth is always "the current
//!    one," decided independently of how any leg happens to score it.
//! 2. **Type priority** (`mem_gotcha_*` / `mem_note_*` pairs) — a genuine
//!    drift-preventer competing with a plausible but non-critical Note.
//!    Ground truth is always the gotcha.
//! 3. **Rank position** — a relevant doc surrounded by same-domain
//!    distractors so it isn't guaranteed fused rank 1.
//! 4. **Generic first-prompt drift** — short, non-keyword-exact "anything I
//!    should know" queries against a pool of past-mistake memories plus
//!    unrelated background notes.
//! 5. **Code vs. memory tie** (`mem_decision_retry_backoff` +
//!    `code_retry_backoff_*`) — a curated memory competing against raw
//!    `Kind::CodeChunk` hits (WS1) on the same topic, so `scoring.code_damp`
//!    has an automated regression guard instead of only a manual smoke
//!    test. Ground truth is always the memory: a human asking a behavior
//!    question wants the curated "why," not a raw function body, even when
//!    the code matches just as well lexically. The question is phrased
//!    conceptually (no literal identifier) on purpose — see 5b.
//!    5b. **Identifier lookup** (same fixtures as 5, different question) — a
//!    query naming the literal identifier `retry_with_backoff`, regression
//!    guard for the identifier-exact RRF leg (`search/mod.rs`). Ground
//!    truth flips to the function body here: an identifier-explicit
//!    mechanics question ("what does `retry_with_backoff` do with the
//!    attempt number") has a genuinely different human-correct answer than
//!    5's conceptual phrasing of the same topic — conflating the two in one
//!    fixture pair previously made the identifier leg's landing look like a
//!    5-regression when it was actually working as intended.
//! 6. **Cross-kind recency** (`mem_gotcha_wal_autocheckpoint` +
//!    `chunk_wal_docs_overview` + `code_wal_open_fn`) — a year-old gotcha
//!    that is still the correct answer, competing against a fresh doc chunk
//!    and a fresh code chunk on the same topic. Non-decaying kinds
//!    (Chunk/CodeChunk) must stay recency-neutral against an aging memory,
//!    not enjoy `recency_alpha` just for being newer — this is the
//!    regression guard for that gating (see `learn::half_life_days` callers
//!    in `search/mod.rs`). Ground truth is always the gotcha.
//!
//! A separate corpus-reusing question set, [`InjectQuestion`], drives the
//! preventer end-to-end eval (`eval::mod`'s `run_inject_eval_*`): it scores
//! the actual `inject::select_injection` decision (inject the right thing,
//! or correctly stay silent) rather than a raw ranked list. See its docs
//! for scenario family 7 (client-side git enrichment).
//!
//! Labels in every scenario are "what a human would say is the correct
//! answer," authored before looking at how the current ranking scores it.
//! Where a scenario still ceilings at MRR/recall 1.0 despite being a
//! reasonable case, it stays in the corpus anyway (see the eval report) —
//! the point is honest headroom, not a rigged baseline.
//!
//! [`Question::set`] tags each new-scenario question `Tuning` or `Holdout`
//! (~2/3 / ~1/3) so a future ranking-tuning workstream can report train vs.
//! holdout separately and catch overfitting to the tuning subset. The
//! original 9 questions are tagged `Core` and sit outside that split.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use rusqlite::{params, Connection};

use crate::embed::{l2_normalize, to_blob};
use crate::error::Result;
use crate::model::{now_unix, Kind, NewNode};
use crate::store;

use super::{Category, QuestionSet};

/// Model name for the hermetic vector leg — never a real embedding model.
pub const SYNTHETIC_MODEL: &str = "eval-synthetic";
const DIM: usize = 32;

struct Fixture {
    key: &'static str,
    kind: Kind,
    subkind: Option<&'static str>,
    title: &'static str,
    body: &'static str,
    path: Option<&'static str>,
    lang: Option<&'static str>,
    topic: &'static str,
    /// Days before "now" this node was created (`None` = created at insert
    /// time). Only the recency-scenario family backdates; `learn.rs`'s
    /// strength/recency decay both read from `created_at` (or
    /// `last_accessed`, which these fixtures never set).
    age_days: Option<i64>,
}

const FIXTURES: &[Fixture] = &[
    // ---- Core (original 13 + 4 hard distractors; unchanged from WS0) ----
    Fixture {
        key: "mem_gotcha_busy",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "SQLite busy errors under concurrent writers",
        body: "Concurrent writers on the same SQLite file can hit SQLITE_BUSY if a \
               transaction is not IMMEDIATE. Set a busy_timeout pragma and open write \
               transactions as IMMEDIATE so a waiting writer blocks instead of erroring.",
        path: None,
        lang: None,
        topic: "busy_timeout",
        age_days: None,
    },
    Fixture {
        key: "mem_decision_rrf",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Chose reciprocal rank fusion over a learned reranker for the default path",
        body: "RRF combines the BM25 and vector legs with a single constant (k=60) and \
               needs no training data or per-leg weight tuning. A learned cross-encoder \
               reranker is still available as an opt-in second pass over the fused \
               candidates.",
        path: None,
        lang: None,
        topic: "rrf_decision",
        age_days: None,
    },
    Fixture {
        key: "mem_insight_recency",
        kind: Kind::Memory,
        subkind: Some("insight"),
        title: "Recency boost defaults to off",
        body: "Turning on the recency multiplier by default made stale but strong \
               memories lose to fresh but weak ones. The recency_alpha config now \
               defaults to 0.0; strength decay already carries most of the useful time \
               signal.",
        path: None,
        lang: None,
        topic: "recency_config",
        age_days: None,
    },
    Fixture {
        key: "mem_note_model_size",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Default embedding model download size",
        body: "The quantized bge-small-en-v1.5 ONNX model is about 34 megabytes and \
               downloads once into the models cache directory on first `mimir init`.",
        path: None,
        lang: None,
        topic: "model_size",
        age_days: None,
    },
    Fixture {
        key: "mem_drift_no_verify",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Never skip git hooks with --no-verify",
        body: "A past commit used --no-verify to get past a failing pre-commit hook, \
               which let a broken build merge to main. Hooks must not be skipped unless \
               the user explicitly asks; fix the underlying failure instead.",
        path: None,
        lang: None,
        topic: "no_verify_hooks",
        age_days: None,
    },
    Fixture {
        key: "mem_drift_matrix_cache",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Vector matrix cache goes stale after same-connection writes",
        body: "We once assumed embedding a batch would auto-refresh the in-memory vector \
               matrix, so newly embedded nodes stayed invisible to search until the next \
               external commit. The fix: embed_pending explicitly drops the cache after \
               writing, forcing a rebuild on the next search.",
        path: None,
        lang: None,
        topic: "matrix_cache",
        age_days: None,
    },
    Fixture {
        key: "doc_hybrid_search",
        kind: Kind::Chunk,
        subkind: None,
        title: "Hybrid search",
        body: "Mimir fuses two ranked legs: an FTS5 BM25 text leg and a vector leg over \
               L2-normalized embeddings, combined with reciprocal rank fusion so no leg \
               needs to be re-weighted when the other changes.",
        path: Some("docs/search.md"),
        lang: None,
        topic: "vector_scoring",
        age_days: None,
    },
    Fixture {
        key: "doc_config_scoring",
        kind: Kind::Chunk,
        subkind: None,
        title: "Scoring configuration",
        body: "scoring.strength_alpha weights learned strength into the final rank. \
               scoring.recency_alpha adds an optional recency multiplier and defaults to \
               0.0, meaning no recency boost unless you opt in.",
        path: Some("docs/config.md"),
        lang: None,
        topic: "recency_config",
        age_days: None,
    },
    Fixture {
        key: "doc_quickstart",
        kind: Kind::Chunk,
        subkind: None,
        title: "Quickstart",
        body: "Install the mimir binary, run `mimir init` to set up the database and \
               download the default embedding model, then start the MCP server for your \
               editor or agent.",
        path: Some("docs/quickstart.md"),
        lang: None,
        topic: "quickstart",
        age_days: None,
    },
    Fixture {
        key: "symbol_search_hybrid",
        kind: Kind::Symbol,
        subkind: None,
        title: "search_hybrid",
        body: "pub fn search_hybrid(conn: &Connection, query: &SearchQuery, vector_query: \
               Option<(&str, &[f32])>, cache: &mut Option<MatrixCache>) -> \
               Result<Vec<Hit>> — runs the BM25 leg and, if a query vector is given, the \
               vector leg, then fuses both with reciprocal rank fusion.",
        path: Some("crates/mimir-core/src/search/mod.rs"),
        lang: Some("rust"),
        topic: "fusion_fn",
        age_days: None,
    },
    Fixture {
        key: "symbol_l2_normalize",
        kind: Kind::Symbol,
        subkind: None,
        title: "l2_normalize",
        body: "pub fn l2_normalize(v: &mut [f32]) — scales a vector to unit length in \
               place so a later dot product between two normalized vectors equals their \
               cosine similarity.",
        path: Some("crates/mimir-core/src/embed.rs"),
        lang: Some("rust"),
        topic: "l2norm_fn",
        age_days: None,
    },
    Fixture {
        key: "chunk_vector_leg_code",
        kind: Kind::Chunk,
        subkind: None,
        title: "vector leg scoring loop",
        body: "let dot: f32 = row.iter().zip(query_vec).map(|(a, b)| a * b).sum(); — each \
               candidate row is scored by a plain dot product against the query vector, \
               then the results are sorted best first and truncated to the leg cap.",
        path: Some("crates/mimir-core/src/search/vector.rs"),
        lang: Some("rust"),
        topic: "vector_scoring",
        age_days: None,
    },
    Fixture {
        key: "chunk_matrix_invalidation_code",
        kind: Kind::Chunk,
        subkind: None,
        title: "matrix cache invalidation contract",
        body: "Other connections' commits are caught via PRAGMA data_version. \
               Same-connection embedding mutations do not auto-invalidate the cache; the \
               only same-connection mutator is embed_pending, and it drops the cache \
               after calling it.",
        path: Some("crates/mimir-core/src/search/vector.rs"),
        lang: Some("rust"),
        topic: "matrix_cache",
        age_days: None,
    },
    // Hard distractors: none is any question's relevant id. Each
    // deliberately reuses vocabulary from a real answer (so the BM25 leg
    // has to work for its ranking).
    //
    // decoy_gotcha_wal is deliberately placed in the SAME topic cluster as
    // its near-answer (mem_gotcha_busy), so it also competes on the
    // hermetic vector leg.
    Fixture {
        key: "decoy_gotcha_wal",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "SQLite WAL checkpoint starves concurrent readers",
        body: "A long-running writer on a SQLite database in WAL mode can delay the \
               checkpoint, growing the -wal file until concurrent readers slow down. \
               Run PRAGMA wal_checkpoint(TRUNCATE) periodically outside of busy write \
               windows.",
        path: None,
        lang: None,
        topic: "busy_timeout",
        age_days: None,
    },
    Fixture {
        key: "decoy_symbol_dot_product",
        kind: Kind::Symbol,
        subkind: None,
        title: "dot_product",
        body: "fn dot_product(a: &[f32], b: &[f32]) -> f32 — a generic helper that sums \
               the elementwise product of two equal-length vectors; used outside the \
               search hot path where an unrolled loop is not worth the complexity.",
        path: Some("crates/mimir-core/src/tokens.rs"),
        lang: Some("rust"),
        topic: "dot_product_decoy",
        age_days: None,
    },
    Fixture {
        key: "decoy_decision_fts_choice",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Chose SQLite FTS5 over a separate search index for the text leg",
        body: "FTS5 ships inside SQLite, needs no second process or index file to keep in \
               sync, and is fast enough at our scale. A dedicated search engine was \
               ruled out as unnecessary operational weight for the default path.",
        path: None,
        lang: None,
        topic: "fts_choice_decoy",
        age_days: None,
    },
    Fixture {
        key: "decoy_note_reranker_size",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Reranker model download size",
        body: "The default cross-encoder reranker, jina-reranker-v1-turbo-en, is about \
               150 megabytes and downloads once into the same models cache directory as \
               the embedding model.",
        path: None,
        lang: None,
        topic: "reranker_size_decoy",
        age_days: None,
    },
    // ---- Scenario family 1: recency / supersede ----
    // Each pair shares one topic (so hermetic mode gives them near-identical
    // synthetic vectors — the vector leg genuinely can't separate them) and
    // the old node's wording deliberately echoes the query's literal
    // phrasing more closely than the new node's, which uses updated
    // terminology. That's a realistic failure mode (institutional phrasing
    // outlives the decision it described), not a rigged one: the label is
    // "the current decision," full stop, decided before any leg was scored.
    //
    // These pairs are NOT wired via `store::set_superseded` — search_hybrid
    // hides superseded nodes by default (`include_superseded: false`, same
    // default this harness uses), so a properly superseded old node would
    // never compete for rank at all, making the scenario trivial rather
    // than a test of recency-aware ranking. Modeling "nobody got around to
    // marking the old one superseded" is the realistic and harder case a
    // recency boost is meant to help with.
    Fixture {
        key: "mem_decision_pool_old",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Current policy: single shared SQLite connection, no pool",
        body: "A single shared connection avoids SQLite's file-locking contention \
               entirely; a connection pool just adds queueing overhead for the same \
               underlying writer lock.",
        path: None,
        lang: None,
        topic: "pool_decision",
        age_days: Some(500),
    },
    Fixture {
        key: "mem_decision_pool_new",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Revisit: WAL mode plus busy_timeout beats the single-connection rule",
        body: "With WAL mode and a sane busy_timeout, multiple connections coexist fine; \
               we dropped the single-shared-connection rule and now open a connection per \
               async task.",
        path: None,
        lang: None,
        topic: "pool_decision",
        age_days: Some(3),
    },
    Fixture {
        key: "mem_decision_logging_old",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "CLI default log format: plain text lines to stdout",
        body: "Plain lines are grep-friendly and need no schema, so the CLI logs plain \
               text to stdout by default.",
        path: None,
        lang: None,
        topic: "logging_decision",
        age_days: Some(600),
    },
    Fixture {
        key: "mem_decision_logging_new",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Default logs switched to structured JSON",
        body: "tracing-subscriber now emits structured JSON by default so the dashboard \
               can parse fields without regex; pass --log-format=text to get the old \
               plain lines back.",
        path: None,
        lang: None,
        topic: "logging_decision",
        age_days: Some(12),
    },
    Fixture {
        key: "mem_decision_release_old",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Releases are cut manually by tagging main",
        body: "A maintainer runs the version bump and git tag by hand, then cargo \
               publish for each crate in dependency order.",
        path: None,
        lang: None,
        topic: "release_process",
        age_days: Some(400),
    },
    Fixture {
        key: "mem_decision_release_new",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Releases now run through a scripted release checklist",
        body: "The manual tag-and-publish steps moved into scripts/release.sh, which \
               bumps versions, tags, and publishes crates in order so a maintainer can't \
               skip a step.",
        path: None,
        lang: None,
        topic: "release_process",
        age_days: Some(5),
    },
    // ---- Scenario family 2: type priority (gotcha vs. note) ----
    // Each pair shares one topic. The gotcha encodes a real past-mistake +
    // fix (the human-correct answer, always); the note is a plausible,
    // adjacent, non-critical fact whose wording happens to echo the query
    // at least as closely. There is currently no subkind-aware boost in
    // search_hybrid — this is exactly the gap a future gotcha/decision
    // type-priority multiplier would close.
    Fixture {
        key: "mem_gotcha_log_redaction",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Never log full request bodies — they contain API keys",
        body: "A past incident logged raw request bodies at debug level, which briefly \
               wrote a customer's API key into the log aggregator. Redact Authorization \
               and any *_key/*_token fields before logging, always.",
        path: None,
        lang: None,
        topic: "log_redaction",
        age_days: None,
    },
    Fixture {
        key: "mem_note_log_verbosity",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Log verbosity levels used in this repo",
        body: "We use error/warn/info/debug/trace. debug is noisy in request handlers and \
               includes headers and bodies for troubleshooting; keep it off in \
               production.",
        path: None,
        lang: None,
        topic: "log_redaction",
        age_days: None,
    },
    Fixture {
        key: "mem_gotcha_migration_order",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Never run schema migrations out of order across replicas",
        body: "Running migration 0007 before 0006 completed on a replica once corrupted \
               the derived index; migrations must apply strictly in numeric order, never \
               parallelized across shards.",
        path: None,
        lang: None,
        topic: "migration_order",
        age_days: None,
    },
    Fixture {
        key: "mem_note_migration_naming",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Migration file naming convention",
        body: "Migration files are named NNNN_description.sql with a zero-padded 4-digit \
               sequence number; the number is the only thing that determines apply \
               order.",
        path: None,
        lang: None,
        topic: "migration_order",
        age_days: None,
    },
    Fixture {
        key: "mem_gotcha_test_flake",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Don't silence flaky tests with #[ignore] — fix the race",
        body: "A flaky integration test was marked #[ignore] to unblock a release once; \
               six months later the underlying race condition caused a real production \
               bug that the ignored test would have caught. Root-cause flaky tests, never \
               just silence them.",
        path: None,
        lang: None,
        topic: "flaky_tests",
        age_days: None,
    },
    Fixture {
        key: "mem_note_test_runtime",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Test suite runtime",
        body: "The full test suite takes about 90 seconds locally and 4 minutes in CI \
               including the ignored perf tests; run cargo test without --ignored for the \
               fast path.",
        path: None,
        lang: None,
        topic: "flaky_tests",
        age_days: None,
    },
    // ---- Scenario family 3: rank position (for future RRF k tuning) ----
    // Each relevant doc sits among same-domain distractors that share
    // enough vocabulary/topic to compete for fused rank. RRF_K is a private
    // constant in search/mod.rs (not exposed as a parameter here, and this
    // harness must not touch that file), so this family can only establish
    // the baseline fact — relevant doc buried below rank 1 — not empirically
    // confirm that a lower k promotes it. See eval::tests for a diagnostic
    // that prints the actual ranked list per question.
    Fixture {
        key: "mem_note_cli_json_flag",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "--json output flag",
        body: "Most subcommands accept --json to print machine-readable output instead \
               of the default table; useful for scripting.",
        path: None,
        lang: None,
        topic: "cli_json_flag",
        age_days: None,
    },
    Fixture {
        key: "doc_config_output_format",
        kind: Kind::Chunk,
        subkind: None,
        title: "Output configuration",
        body: "output.default_limit and output.snippet_chars control how many results \
               print and how long each snippet is by default.",
        path: Some("docs/config.md"),
        lang: None,
        topic: "cli_output_config",
        age_days: None,
    },
    Fixture {
        key: "mem_note_cli_table_width",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Table column width",
        body: "The default table output truncates long titles at 60 characters; pass a \
               wider terminal to see more.",
        path: None,
        lang: None,
        topic: "cli_output_config",
        age_days: None,
    },
    Fixture {
        key: "mem_note_mcp_transport",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "MCP server listens on stdio by default, not a port",
        body: "The MCP server talks over stdio to the editor/agent; there is no network \
               port unless you run mimir with --http, in which case it binds 127.0.0.1 \
               on a configurable port.",
        path: None,
        lang: None,
        topic: "mcp_transport",
        age_days: None,
    },
    Fixture {
        key: "mem_note_proxy_port",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Proxy default bind address",
        body: "The optional API proxy binds 127.0.0.1:8788 by default; point \
               ANTHROPIC_BASE_URL at it.",
        path: None,
        lang: None,
        topic: "proxy_bind",
        age_days: None,
    },
    Fixture {
        key: "doc_config_sync_endpoint",
        kind: Kind::Chunk,
        subkind: None,
        title: "Sync server mode",
        body: "server mode points sync.endpoint at a mimir serve hub URL like \
               http://unraid.tailnet:7777; the bearer token comes from MIMIR_SYNC_TOKEN, \
               never config.",
        path: Some("docs/config.md"),
        lang: None,
        topic: "sync_endpoint",
        age_days: None,
    },
    Fixture {
        key: "mem_note_embed_device",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "embedding.device = auto picks a GPU EP if compiled in",
        body: "auto uses a GPU execution provider only when the binary was built with \
               --features gpu-webgpu or gpu-cuda; otherwise it silently runs on CPU. Set \
               device to \"cpu\" to force CPU even in a GPU build.",
        path: None,
        lang: None,
        topic: "device_config",
        age_days: None,
    },
    Fixture {
        key: "mem_note_gpu_feature",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "gpu-webgpu and gpu-cuda are mutually exclusive",
        body: "The two GPU features select different prebuilt onnxruntime binaries; \
               enabling both together is a compile error, so pick one at build time.",
        path: None,
        lang: None,
        topic: "gpu_features",
        age_days: None,
    },
    // ---- Scenario family 4: generic first-prompt drift query ----
    // Short, non-keyword-exact "anything I should know" queries scored
    // against a pool of real past-mistake memories (some reused from Core)
    // plus unrelated background notes.
    Fixture {
        key: "mem_drift_force_push",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Never force-push to main",
        body: "A rebase once got force-pushed straight to main and clobbered two \
               teammates' commits; force-push only your own feature branches, never a \
               shared one, and never main.",
        path: None,
        lang: None,
        topic: "force_push_gotcha",
        age_days: None,
    },
    Fixture {
        key: "mem_note_repo_layout",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Repo layout",
        body: "crates/mimir-core is the engine, mimir-cli is the binary, mimir-graph is \
               code-graph extraction, mimir-proxy is the optional API proxy.",
        path: None,
        lang: None,
        topic: "repo_layout",
        age_days: None,
    },
    Fixture {
        key: "mem_note_build_time",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Build time",
        body: "A clean workspace build takes about 3 minutes; incremental builds are a \
               few seconds.",
        path: None,
        lang: None,
        topic: "build_time",
        age_days: None,
    },
    Fixture {
        key: "mem_note_license",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "License",
        body: "Dual-licensed MIT OR Apache-2.0, same as most of the Rust ecosystem.",
        path: None,
        lang: None,
        topic: "license",
        age_days: None,
    },
    Fixture {
        key: "mem_note_editor_support",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Editor support",
        body: "Any MCP-capable client works: Claude Code, Claude Desktop, Cursor, \
               Windsurf, and others that speak the MCP stdio or HTTP transport.",
        path: None,
        lang: None,
        topic: "editor_support",
        age_days: None,
    },
    Fixture {
        key: "mem_note_ci_setup",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "CI setup",
        body: "GitHub Actions runs cargo test and cargo clippy on every push; releases \
               are gated on a separate manual workflow.",
        path: None,
        lang: None,
        topic: "ci_setup",
        age_days: None,
    },
    Fixture {
        key: "mem_note_versioning",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Versioning",
        body: "All workspace crates share one version number bumped together in the root \
               Cargo.toml, even when only one crate changed.",
        path: None,
        lang: None,
        topic: "versioning",
        age_days: None,
    },
    // ---- Scenario family 5: code-vs-memory tie (code_damp regression guard) ----
    // WS1 indexes source-code content as Kind::CodeChunk, which floods the
    // default pool unless scoring.code_damp (0.85) sub-1.0's it. Before this
    // scenario, no fixture exercised that: the guard existed only as a
    // manual smoke test. All three fixtures share one topic (so hermetic
    // mode gives them near-identical synthetic vectors — ranking has to be
    // decided by the BM25 leg + code_damp, not vector distance) and real
    // vocabulary overlap ("transient", "retry", "backoff", "sync"). Also
    // backs family 5b's identifier-lookup question below, where ground
    // truth flips to `code_retry_backoff_fn` (see module docs, families
    // 5 & 5b).
    Fixture {
        key: "mem_decision_retry_backoff",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Retry policy for sync failures",
        body: "A flaky network blip during sync used to retry immediately and hammer the \
               hub; the retry path now waits with exponential backoff (200ms base, doubling \
               per try, capped at 5s) plus jitter, and gives up after 5 tries.",
        path: None,
        lang: None,
        topic: "retry_backoff_tie",
        age_days: None,
    },
    Fixture {
        key: "code_retry_backoff_fn",
        kind: Kind::CodeChunk,
        subkind: None,
        title: "crates/mimir-core/src/replicate.rs › retry_with_backoff (function) — \
                fn retry_with_backoff(attempt: u32) -> Duration",
        body: "fn retry_with_backoff(attempt: u32) -> Duration {\n\
               \x20   // Computes how long retry_with_backoff should wait before the next\n\
               \x20   // attempt: exponential backoff plus jitter, capped at 5 seconds.\n\
               \x20   let base = Duration::from_millis(200);\n\
               \x20   let capped = base * 2u32.pow(attempt.min(5));\n\
               \x20   let jitter = Duration::from_millis(rand_jitter_ms());\n\
               \x20   capped.min(Duration::from_secs(5)) + jitter\n\
               }",
        path: Some("crates/mimir-core/src/replicate.rs"),
        lang: Some("rust"),
        topic: "retry_backoff_tie",
        age_days: None,
    },
    Fixture {
        key: "code_retry_backoff_caller",
        kind: Kind::CodeChunk,
        subkind: None,
        title: "crates/mimir-core/src/replicate.rs › sync_push (function) — \
                async fn sync_push(hub: &Hub) -> Result<()>",
        body: "async fn sync_push(hub: &Hub) -> Result<()> {\n\
               \x20   for attempt in 0..MAX_RETRY_ATTEMPTS {\n\
               \x20       match hub.push(&batch).await {\n\
               \x20           Ok(()) => return Ok(()),\n\
               \x20           Err(e) if e.is_transient() => {\n\
               \x20               tokio::time::sleep(retry_with_backoff(attempt)).await;\n\
               \x20               continue;\n\
               \x20           }\n\
               \x20           Err(e) => return Err(e),\n\
               \x20       }\n\
               \x20   }\n\
               \x20   Err(Error::RetriesExhausted)\n\
               }",
        path: Some("crates/mimir-core/src/replicate.rs"),
        lang: Some("rust"),
        topic: "retry_backoff_tie",
        age_days: None,
    },
    // ---- Scenario family 6: cross-kind recency (recency-gating regression
    // guard) ----
    // A year-old gotcha is still the human-correct answer, competing
    // against a fresh doc chunk and a fresh code chunk on the same topic —
    // both non-decaying kinds (`learn::half_life_days` returns `None` for
    // Chunk/CodeChunk) that must NOT get a `recency_alpha` boost just for
    // being newer than the gotcha. Before the recency-gating fix, this is
    // exactly the shape that regressed: a permanent boost for every
    // non-decaying node relative to any aging memory, unrelated to whether
    // either one actually answers the question. Ground truth is always the
    // gotcha; the doc/code chunks are topically adjacent but don't explain
    // the actual cause.
    Fixture {
        key: "mem_gotcha_wal_autocheckpoint",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Why the WAL file keeps growing and never shrinks back down",
        body: "A read connection that never closes holds SQLite's WAL file open, so the \
               default automatic checkpoint (every 1000 pages) never fires: the WAL file \
               keeps growing and never shrinks back down. Long-lived read connections must \
               run 'PRAGMA wal_checkpoint(TRUNCATE)' periodically or the -wal sidecar file \
               balloons.",
        path: None,
        lang: None,
        topic: "wal_growth",
        age_days: Some(365),
    },
    Fixture {
        key: "chunk_wal_docs_overview",
        kind: Kind::Chunk,
        subkind: None,
        title: "docs/storage.md \u{203a} Write-Ahead Logging",
        body: "Mimir opens SQLite in WAL mode for concurrent readers and a single writer. \
               The main db and its -wal/-shm sidecar files must live on the same \
               filesystem; network filesystems are not supported.",
        path: Some("docs/storage.md"),
        lang: None,
        topic: "wal_growth",
        age_days: None,
    },
    Fixture {
        key: "code_wal_open_fn",
        kind: Kind::CodeChunk,
        subkind: None,
        title: "crates/mimir-core/src/db.rs \u{203a} open (function) — \
                pub fn open(path: &Path) -> Result<Connection>",
        body: "pub fn open(path: &Path) -> Result<Connection> {\n\
               \x20   let conn = Connection::open(path)?;\n\
               \x20   conn.pragma_update(None, \"journal_mode\", \"WAL\")?;\n\
               \x20   conn.pragma_update(None, \"busy_timeout\", 5000)?;\n\
               \x20   Ok(conn)\n\
               }",
        path: Some("crates/mimir-core/src/db.rs"),
        lang: Some("rust"),
        topic: "wal_growth",
        age_days: None,
    },
    // ---- Background noise: never targeted by any question ----
    // Bulk and realistic messiness for the BM25 leg to compete against;
    // each sits in its own topic so hermetic mode never clusters one near a
    // real answer by accident.
    Fixture {
        key: "filler_editor_plugin_config",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Editor plugin config",
        body: "The VS Code extension reads .mimir/settings.json for per-project \
               overrides.",
        path: None,
        lang: None,
        topic: "filler_01",
        age_days: None,
    },
    Fixture {
        key: "filler_stats_json_idea",
        kind: Kind::Memory,
        subkind: Some("idea"),
        title: "Consider a --json flag for mimir stats",
        body: "Would let dashboards parse stats output without scraping the table.",
        path: None,
        lang: None,
        topic: "filler_02",
        age_days: None,
    },
    Fixture {
        key: "filler_batch_size_insight",
        kind: Kind::Memory,
        subkind: Some("insight"),
        title: "Batch size 64 balanced latency and throughput for embedding",
        body: "Larger batches saturated memory on 8GB machines without a proportional \
               speed gain.",
        path: None,
        lang: None,
        topic: "filler_03",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_download_stall",
        kind: Kind::Chunk,
        subkind: None,
        title: "Troubleshooting: model download stalls",
        body: "If mimir init hangs on download, check that huggingface.co isn't blocked \
               by a corporate proxy.",
        path: Some("docs/troubleshooting.md"),
        lang: None,
        topic: "filler_04",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_uninstall",
        kind: Kind::Chunk,
        subkind: None,
        title: "Uninstalling Mimir",
        body: "Delete the config, data, and cache directories reported by mimir paths; \
               there's no installer to run in reverse.",
        path: Some("docs/uninstall.md"),
        lang: None,
        topic: "filler_05",
        age_days: None,
    },
    Fixture {
        key: "filler_symbol_resolve_ref",
        kind: Kind::Symbol,
        subkind: None,
        title: "resolve_ref",
        body: "pub fn resolve_ref(conn: &Connection, reference: &str) -> Result<Node> — \
               resolves a full ULID, short m:ABCDEF form, or a bare uid tail.",
        path: Some("crates/mimir-core/src/store.rs"),
        lang: Some("rust"),
        topic: "filler_06",
        age_days: None,
    },
    Fixture {
        key: "filler_symbol_short_uid",
        kind: Kind::Symbol,
        subkind: None,
        title: "short_uid",
        body: "pub fn short_uid(kind: Kind, uid: &str) -> String — builds the sigil:tail \
               display form of a node id.",
        path: Some("crates/mimir-core/src/model.rs"),
        lang: Some("rust"),
        topic: "filler_07",
        age_days: None,
    },
    Fixture {
        key: "filler_tag_syntax",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Tag syntax",
        body: "Tags are space-separated lowercase words stored in tags_text; there is no \
               tag hierarchy.",
        path: None,
        lang: None,
        topic: "filler_08",
        age_days: None,
    },
    Fixture {
        key: "filler_consolidation_cadence",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Consolidation cadence",
        body: "consolidate.auto defaults to weekly and runs four passes off the request \
               path.",
        path: None,
        lang: None,
        topic: "filler_09",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_replication",
        kind: Kind::Chunk,
        subkind: None,
        title: "Replication overview",
        body: "replicate.rs mirrors soft-deletes and supersessions to keep a sync hub \
               consistent without hard deletes.",
        path: Some("docs/sync.md"),
        lang: None,
        topic: "filler_10",
        age_days: None,
    },
    Fixture {
        key: "filler_decision_blake3",
        kind: Kind::Memory,
        subkind: Some("decision"),
        title: "Chose blake3 over sha256 for content hashing",
        body: "blake3 is faster and the hash is only used for change detection, not \
               cryptographic security.",
        path: None,
        lang: None,
        topic: "filler_11",
        age_days: None,
    },
    Fixture {
        key: "filler_gotcha_models_path",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "Don't hardcode the models cache path",
        body: "A script once assumed ~/.cache/mimir/models, which breaks on Windows and \
               when MIMIR_HOME is set; always resolve via Paths.",
        path: None,
        lang: None,
        topic: "filler_12",
        age_days: None,
    },
    Fixture {
        key: "filler_batching_insight",
        kind: Kind::Memory,
        subkind: Some("insight"),
        title: "Batching embed_pending writes avoided lock contention",
        body: "Committing every 256 rows instead of one giant transaction let concurrent \
               readers keep working during a big import.",
        path: None,
        lang: None,
        topic: "filler_13",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_savings",
        kind: Kind::Chunk,
        subkind: None,
        title: "Savings dashboard",
        body: "mimir savings estimates dollars saved from avoided re-reads, priced via \
               savings.input_price_per_mtok.",
        path: Some("docs/savings.md"),
        lang: None,
        topic: "filler_14",
        age_days: None,
    },
    Fixture {
        key: "filler_symbol_record_opened",
        kind: Kind::Symbol,
        subkind: None,
        title: "record_opened",
        body: "pub fn record_opened(conn: &Connection, node_id: i64) -> Result<()> — logs \
               an access and applies the daily-capped strength bonus.",
        path: Some("crates/mimir-core/src/learn.rs"),
        lang: Some("rust"),
        topic: "filler_15",
        age_days: None,
    },
    Fixture {
        key: "filler_graph_languages",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Supported languages for code graph extraction",
        body: "Go, Python, Rust, TypeScript, Java, Ruby, C, C#, and T-SQL via tree-sitter \
               grammars.",
        path: None,
        lang: None,
        topic: "filler_16",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_import",
        kind: Kind::Chunk,
        subkind: None,
        title: "Import formats",
        body: "mimir import reads markdown, plain text, and JSON exports from a handful \
               of other note apps.",
        path: Some("docs/import.md"),
        lang: None,
        topic: "filler_17",
        age_days: None,
    },
    Fixture {
        key: "filler_token_counting",
        kind: Kind::Memory,
        subkind: Some("note"),
        title: "Token counting",
        body: "tokens.rs estimates token counts per kind so savings and context-budget \
               numbers don't need a live tokenizer call.",
        path: None,
        lang: None,
        topic: "filler_18",
        age_days: None,
    },
    Fixture {
        key: "filler_per_project_config_idea",
        kind: Kind::Memory,
        subkind: Some("idea"),
        title: "Per-project config overrides",
        body: "Nothing implemented yet; today config.toml is one file for the whole user, \
               not per project.",
        path: None,
        lang: None,
        topic: "filler_19",
        age_days: None,
    },
    Fixture {
        key: "filler_doc_prune",
        kind: Kind::Chunk,
        subkind: None,
        title: "Uninstall vs prune",
        body: "mimir prune removes soft-deleted nodes past a retention window; it does \
               not touch the config or models cache.",
        path: Some("docs/prune.md"),
        lang: None,
        topic: "filler_20",
        age_days: None,
    },
    // ---- Fires-when: a gotcha whose title/body deliberately share only one
    // lexical token ("confetti") with the paraphrased question below — see
    // FIRES_WHEN and INJECT_QUESTIONS' fires-when pair. One shared token is
    // required, not zero: RRF fusion (search/mod.rs) sums a rank-based
    // score across every leg a node appears in, so a node absent from the
    // FTS leg entirely (zero shared tokens) loses to nodes that share even
    // one *generic* word (e.g. "need") with the query and so appear in
    // both legs — no vector-leg rank, however strong, recovers from being
    // single-leg. One *rare* shared token (unique to this fixture in the
    // whole corpus) is enough to also win the FTS leg outright, while
    // staying below MIN_OVERLAP=2 so the normal floor's rule 2 still
    // blocks it without the trigger. A release/checklist themed fixture
    // was tried first and got crowded out of the fused top-5 by the
    // *already-existing* `mem_decision_release_new` fixture, which
    // legitimately shares that vocabulary — the fires-when candidate never
    // even reached `select_injection` to be checked. This topic and its
    // "confetti" anchor word have no other fixture anywhere in the corpus.
    Fixture {
        key: "mem_gotcha_confetti_deploy_trigger",
        kind: Kind::Memory,
        subkind: Some("gotcha"),
        title: "The confetti animation fired during the wrong build last quarter",
        body: "An internal-only visual effect played in front of a client because \
               nobody disabled it first. Switch it off before anything external happens.",
        path: None,
        lang: None,
        topic: "confetti_deploy_trigger",
        age_days: None,
    },
];

/// Author-declared `meta.fires_when` trigger phrases for specific fixtures
/// (mirrors `CODE_SPANS`'s per-key override pattern) — kept out of the
/// shared `Fixture` struct so the other fixtures don't all need an empty
/// `fires_when: &[]` field.
const FIRES_WHEN: &[(&str, &[&str])] = &[(
    "mem_gotcha_confetti_deploy_trigger",
    &["confetti toggle before demo", "disable easter egg for customers"],
)];

pub type FixtureIds = HashMap<&'static str, i64>;

/// 1-based inclusive line-span overrides for `Kind::CodeChunk` fixtures
/// only — kept out of the shared `Fixture` struct so the other 65+
/// non-code fixtures don't all need `span_start`/`span_end: None`
/// boilerplate on every literal.
const CODE_SPANS: &[(&str, i64, i64)] = &[
    ("code_retry_backoff_fn", 142, 149),
    ("code_retry_backoff_caller", 160, 172),
];

/// Insert every fixture node. No embeddings — callers add those (hermetic
/// synthetic vectors or real-model vectors) separately.
pub fn insert_fixture_nodes(conn: &Connection) -> Result<FixtureIds> {
    let mut ids = HashMap::new();
    let now = now_unix();
    for f in FIXTURES {
        let mut new = NewNode::new(f.kind);
        new.subkind = f.subkind.map(str::to_string);
        new.title = Some(f.title.to_string());
        new.body = Some(f.body.to_string());
        new.path = f.path.map(str::to_string);
        new.lang = f.lang.map(str::to_string);
        if let Some(&(_, start, end)) = CODE_SPANS.iter().find(|(key, _, _)| *key == f.key) {
            new.span_start = Some(start);
            new.span_end = Some(end);
        }
        let node = store::insert_node(conn, new)?;
        if let Some(&(_, phrases)) = FIRES_WHEN.iter().find(|(key, _)| *key == f.key) {
            let owned: Vec<String> = phrases.iter().map(|s| s.to_string()).collect();
            store::set_fires_when(conn, node.id, &owned)?;
        }
        if let Some(days) = f.age_days {
            // NewNode/insert_node always stamps now_unix(); backdate via a
            // direct UPDATE for the recency scenario instead of adding a
            // created_at override to store.rs (out of scope for this pass).
            let ts = now - days * 86_400;
            conn.execute(
                "UPDATE node SET created_at = ?2, updated_at = ?2 WHERE id = ?1",
                params![node.id, ts],
            )?;
        }
        ids.insert(f.key, node.id);
    }
    Ok(ids)
}

/// (node id, embeddable text) pairs, mirroring how `embed::embed_pending`
/// builds its text (title + body) — for real-model embedding.
pub fn fixture_texts(ids: &FixtureIds) -> Vec<(i64, String)> {
    FIXTURES
        .iter()
        .map(|f| (ids[f.key], format!("{}\n{}", f.title, f.body)))
        .collect()
}

/// Reverse lookup for diagnostics: which fixture key a node id belongs to.
pub fn key_for_id(ids: &FixtureIds, id: i64) -> &'static str {
    ids.iter()
        .find(|(_, v)| **v == id)
        .map(|(k, _)| *k)
        .unwrap_or("?")
}

fn seed_for(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Deterministic xorshift, same construction as the perf tests in
/// `search/mod.rs` and `search/vector.rs` — no rand dependency needed.
fn lcg_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed | 1; // xorshift requires a nonzero state
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    (0..dim).map(|_| next()).collect()
}

fn topic_vector(topic: &str) -> Vec<f32> {
    let mut v = lcg_vector(seed_for(&format!("topic:{topic}")), DIM);
    l2_normalize(&mut v);
    v
}

/// A topic's centroid nudged by an item-specific seed: members of one topic
/// sit near each other but aren't identical; independent random unit
/// vectors in 32 dimensions are near-orthogonal, so unrelated topics
/// separate cleanly.
fn clustered_vector(topic: &str, item_seed: &str) -> Vec<f32> {
    let base = topic_vector(topic);
    let noise = lcg_vector(seed_for(item_seed), DIM);
    let mut v: Vec<f32> = base
        .iter()
        .zip(&noise)
        .map(|(b, n)| 0.85 * b + 0.15 * n)
        .collect();
    l2_normalize(&mut v);
    v
}

/// Seed every fixture with a deterministic synthetic embedding under
/// [`SYNTHETIC_MODEL`].
pub fn seed_synthetic_vectors(conn: &Connection, ids: &FixtureIds) -> Result<()> {
    for f in FIXTURES {
        let vec = clustered_vector(f.topic, f.key);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash) \
             VALUES (?1, ?2, ?3, ?4, x'00')",
            params![ids[f.key], SYNTHETIC_MODEL, DIM as i64, to_blob(&vec)],
        )?;
    }
    Ok(())
}

/// A query's synthetic embedding: same topic cluster as its relevant docs.
pub fn synthetic_query_vector(topic: &str, query: &str) -> Vec<f32> {
    clustered_vector(topic, &format!("query:{query}"))
}

pub struct Question {
    pub query: &'static str,
    pub category: Category,
    pub relevant: &'static [&'static str],
    /// Topic used only by hermetic mode to place the synthetic query
    /// vector near its relevant docs' cluster. When a question's relevant
    /// set spans multiple topics (only the generic-drift family does this),
    /// this anchors the query vector to one representative member; the
    /// rest of the relevant set relies on the BM25 leg alone in hermetic
    /// mode. That's an honest constraint, not a shortcut — a generic query
    /// doesn't have one obvious semantic cluster either.
    pub topic: &'static str,
    pub set: QuestionSet,
}

const QUESTIONS: &[Question] = &[
    // ---- Core (original 9, unaffected by the tuning/holdout split) ----
    Question {
        query: "which function fuses the BM25 and vector search results together",
        category: Category::CodeStructure,
        relevant: &["symbol_search_hybrid"],
        topic: "fusion_fn",
        set: QuestionSet::Core,
    },
    Question {
        query: "what does the l2_normalize function do to a vector",
        category: Category::CodeStructure,
        relevant: &["symbol_l2_normalize"],
        topic: "l2norm_fn",
        set: QuestionSet::Core,
    },
    Question {
        query: "how is each candidate scored against the query embedding in the vector leg",
        category: Category::CodeBehavior,
        relevant: &["chunk_vector_leg_code", "doc_hybrid_search"],
        topic: "vector_scoring",
        set: QuestionSet::Core,
    },
    Question {
        query: "why does the matrix cache stay stale right after embedding new nodes on the same connection",
        category: Category::CodeBehavior,
        relevant: &["chunk_matrix_invalidation_code", "mem_drift_matrix_cache"],
        topic: "matrix_cache",
        set: QuestionSet::Core,
    },
    Question {
        query: "how do I avoid SQLITE_BUSY errors from concurrent writers",
        category: Category::GotchaDecision,
        relevant: &["mem_gotcha_busy"],
        topic: "busy_timeout",
        set: QuestionSet::Core,
    },
    Question {
        query: "why was reciprocal rank fusion chosen over a learned reranker",
        category: Category::GotchaDecision,
        relevant: &["mem_decision_rrf"],
        topic: "rrf_decision",
        set: QuestionSet::Core,
    },
    Question {
        query: "is the recency boost turned on by default",
        category: Category::GotchaDecision,
        relevant: &["mem_insight_recency", "doc_config_scoring"],
        topic: "recency_config",
        set: QuestionSet::Core,
    },
    Question {
        query: "why should git hooks never be skipped with no-verify",
        category: Category::DriftPreventer,
        relevant: &["mem_drift_no_verify"],
        topic: "no_verify_hooks",
        set: QuestionSet::Core,
    },
    Question {
        query: "what earlier mistake explains why embed_pending has to drop the vector cache",
        category: Category::DriftPreventer,
        relevant: &["mem_drift_matrix_cache", "chunk_matrix_invalidation_code"],
        topic: "matrix_cache",
        set: QuestionSet::Core,
    },
    // ---- Scenario family 1: recency / supersede ----
    Question {
        query: "what's our current policy on SQLite connection pooling",
        category: Category::GotchaDecision,
        relevant: &["mem_decision_pool_new"],
        topic: "pool_decision",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "does the CLI log plain text or JSON by default",
        category: Category::GotchaDecision,
        relevant: &["mem_decision_logging_new"],
        topic: "logging_decision",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "how do we cut a release",
        category: Category::GotchaDecision,
        relevant: &["mem_decision_release_new"],
        topic: "release_process",
        set: QuestionSet::Holdout,
    },
    // ---- Scenario family 2: type priority ----
    Question {
        query: "what should I be careful about when logging request data",
        category: Category::DriftPreventer,
        relevant: &["mem_gotcha_log_redaction"],
        topic: "log_redaction",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "how does migration order get decided",
        category: Category::DriftPreventer,
        relevant: &["mem_gotcha_migration_order"],
        topic: "migration_order",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "what's our policy on flaky or ignored tests",
        category: Category::DriftPreventer,
        relevant: &["mem_gotcha_test_flake"],
        topic: "flaky_tests",
        set: QuestionSet::Holdout,
    },
    // ---- Scenario family 3: rank position ----
    Question {
        query: "how do I make the CLI print JSON instead of a table",
        category: Category::CodeBehavior,
        relevant: &["mem_note_cli_json_flag"],
        topic: "cli_json_flag",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "does the MCP server open a network port",
        category: Category::CodeBehavior,
        relevant: &["mem_note_mcp_transport"],
        topic: "mcp_transport",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "how do I control whether Mimir uses the GPU for embeddings",
        category: Category::CodeBehavior,
        relevant: &["mem_note_embed_device"],
        topic: "device_config",
        set: QuestionSet::Holdout,
    },
    // ---- Scenario family 4: generic first-prompt drift query ----
    Question {
        query: "anything I should watch out for before I start working in this repo",
        category: Category::DriftPreventer,
        relevant: &["mem_drift_no_verify", "mem_drift_matrix_cache", "mem_drift_force_push"],
        topic: "force_push_gotcha",
        set: QuestionSet::Tuning,
    },
    Question {
        query: "what should I know about past mistakes in the search or storage code before I touch it",
        category: Category::DriftPreventer,
        relevant: &["mem_drift_matrix_cache"],
        topic: "matrix_cache",
        set: QuestionSet::Holdout,
    },
    // ---- Scenario family 5: code-vs-memory tie (code_damp regression guard) ----
    // Conceptual phrasing, deliberately no literal identifier — see module
    // docs, family 5 vs. 5b. Naming `retry_with_backoff` here would let the
    // identifier-exact RRF leg promote the code chunks and defeat the
    // code_damp guard this question exists to exercise.
    Question {
        query: "what's our policy for retrying a sync failure and how long does it wait \
                between attempts",
        category: Category::CodeVsMemory,
        relevant: &["mem_decision_retry_backoff"],
        topic: "retry_backoff_tie",
        set: QuestionSet::Tuning,
    },
    // ---- Scenario family 5b: identifier lookup (identifier RRF leg guard) ----
    // Same fixtures as family 5, but the query DOES name the literal
    // identifier — ground truth flips to the function body (see module
    // docs). Guards the identifier leg the same way family 5 guards
    // code_damp: an identifier-explicit mechanics question is a genuinely
    // different human-correct answer, not a fixture-tuning trick.
    Question {
        query: "what does retry_with_backoff do with the attempt number to compute how long \
                it waits",
        category: Category::IdentifierLookup,
        relevant: &["code_retry_backoff_fn"],
        topic: "retry_backoff_tie",
        set: QuestionSet::Tuning,
    },
    // ---- Scenario family 6: cross-kind recency (recency-gating guard) ----
    Question {
        query: "why does the WAL file keep growing and never shrink back down",
        category: Category::DriftPreventer,
        relevant: &["mem_gotcha_wal_autocheckpoint"],
        topic: "wal_growth",
        set: QuestionSet::Tuning,
    },
];

pub fn questions() -> &'static [Question] {
    QUESTIONS
}

pub fn relevant_ids(ids: &FixtureIds, q: &Question) -> HashSet<i64> {
    q.relevant.iter().map(|k| ids[k]).collect()
}

/// One question for the preventer end-to-end eval (`eval::mod`'s
/// `run_inject_eval_*`): unlike [`Question`], which scores a raw ranked
/// list, this scores the actual product decision — does
/// `inject::select_injection` fire, and on the right node? Reuses the same
/// corpus (`FIXTURES` above), no new fixture nodes.
pub struct InjectQuestion {
    pub query: &'static str,
    /// Simulated `git diff --name-only HEAD` path-stems for this prompt
    /// (scenario family 7 only; empty everywhere else). See `inject.rs`'s
    /// enrichment guard docs — these strengthen the search query and can
    /// count toward the overlap floor, but must never *alone* clear it.
    pub enrich: &'static [&'static str],
    /// `true` = a labeled preventer exists and injecting it is correct;
    /// `false` = staying silent is correct (either the only relevant hit is
    /// a non-preventer kind, or nothing in the corpus is relevant at all).
    pub expects_injection: bool,
    /// Fixture keys acceptable as the injected node when
    /// `expects_injection`. Empty when `expects_injection` is false.
    pub acceptable: &'static [&'static str],
    /// Same role as [`Question::topic`]: anchors the hermetic synthetic
    /// query vector. For negative questions this is chosen so the vector
    /// leg does NOT cluster near any preventer memory, so a false positive
    /// can only come from a real floor bug, not an accidental hermetic
    /// vector coincidence.
    pub topic: &'static str,
}

const INJECT_QUESTIONS: &[InjectQuestion] = &[
    // ---- Positive: a labeled gotcha/decision exists; injecting it is
    // correct. Reuses Core + scenario-family query phrasing so this eval
    // and the ranking eval stay honest about what "correct" means for the
    // same prompts.
    InjectQuestion {
        query: "why should git hooks never be skipped with no-verify",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_drift_no_verify"],
        topic: "no_verify_hooks",
    },
    InjectQuestion {
        query: "what should I be careful about when logging request data",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_gotcha_log_redaction"],
        topic: "log_redaction",
    },
    InjectQuestion {
        query: "how does migration order get decided",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_gotcha_migration_order"],
        topic: "migration_order",
    },
    InjectQuestion {
        query: "what's our policy on flaky or ignored tests",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_gotcha_test_flake"],
        topic: "flaky_tests",
    },
    InjectQuestion {
        query: "why does the WAL file keep growing and never shrink back down",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_gotcha_wal_autocheckpoint"],
        topic: "wal_growth",
    },
    InjectQuestion {
        query: "why was reciprocal rank fusion chosen over a learned reranker",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_decision_rrf"],
        topic: "rrf_decision",
    },
    InjectQuestion {
        query: "what's our current policy on SQLite connection pooling",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_decision_pool_new"],
        topic: "pool_decision",
    },
    InjectQuestion {
        query: "anything I should watch out for before I start working in this repo",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_drift_no_verify", "mem_drift_matrix_cache", "mem_drift_force_push"],
        topic: "force_push_gotcha",
    },
    InjectQuestion {
        query: "what should I know about past mistakes in the search or storage code before I touch it",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_drift_matrix_cache"],
        topic: "matrix_cache",
    },
    // Short prompt (2 non-stopword tokens: "force-push", "risk") — below
    // MIN_OVERLAP=2 needs BOTH its non-stopword tokens to land in the hay,
    // which "risk" alone won't (the fixture never uses that word); this is
    // the honest case scenario-family-7's short-prompt relaxation targets.
    InjectQuestion {
        query: "force-push risk?",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_drift_force_push"],
        topic: "force_push_gotcha",
    },
    // ---- Negative: the only relevant fixture is a non-preventer kind
    // (rule 1 correctly blocks it) — staying silent is the correct call.
    InjectQuestion {
        query: "how do I make the CLI print JSON instead of a table",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "cli_json_flag",
    },
    InjectQuestion {
        query: "does the MCP server open a network port",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "mcp_transport",
    },
    InjectQuestion {
        query: "how do I control whether Mimir uses the GPU for embeddings",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "device_config",
    },
    InjectQuestion {
        query: "what license is this project released under",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "license",
    },
    // ---- Negative: nothing in the corpus is relevant at all.
    InjectQuestion {
        query: "please refactor this function to be more readable",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "filler_01",
    },
    // ---- Scenario family 7: client-side git enrichment ----
    // "Anchor prompt" is deliberately generic (low/no lexical overlap with
    // any memory on its own); enrichment stems come from a simulated
    // `git diff --name-only HEAD` in a repo that's mid-edit on the matrix
    // cache code. Ground truth is the gotcha that code touches.
    InjectQuestion {
        query: "anything I should know before touching this?",
        enrich: &["vector", "cache"],
        expects_injection: true,
        acceptable: &["mem_drift_matrix_cache"],
        topic: "matrix_cache",
    },
    // Guard case: enrichment terms alone reach the overlap floor against a
    // DIFFERENT, unrelated memory (migration_order) — chosen so the query's
    // topic anchor keeps the vector leg from ever agreeing with it. Zero
    // raw-prompt overlap + no vector agreement must mean the enrichment
    // can't self-license an injection; this must stay silent.
    InjectQuestion {
        query: "let's get started",
        enrich: &["migration", "order"],
        expects_injection: false,
        acceptable: &[],
        topic: "editor_support",
    },
    // ---- Fires-when: an author-declared trigger on `mem_gotcha_confetti_
    // deploy_trigger` (see FIRES_WHEN in fixtures above). Its title/body
    // share exactly one lexical token ("confetti") with the question below
    // — enough to survive RRF fusion via the FTS leg (see the fixture's
    // comment), but still below MIN_OVERLAP=2, so rule 2 (and, for this
    // fixture's node type, rule 1 already passes since it's a gotcha)
    // would leave this silent without the trigger. Positive: the prompt
    // matches "confetti toggle before demo" (order-free); the trigger
    // bypass must convert this from silent-wrong to injected-correct.
    // Negative: same fixture, a prompt that matches neither declared
    // trigger phrase nor the normal floor must still stay silent — a
    // fires_when list is not a blanket license to inject on any prompt.
    InjectQuestion {
        query: "before we demo this to a customer, is there a confetti toggle I need to flip?",
        enrich: &[],
        expects_injection: true,
        acceptable: &["mem_gotcha_confetti_deploy_trigger"],
        topic: "confetti_deploy_trigger",
    },
    InjectQuestion {
        query: "anything unusual about how the sqlite fts tokenizer handles unicode?",
        enrich: &[],
        expects_injection: false,
        acceptable: &[],
        topic: "fts_tokenizer_unicode",
    },
];

pub fn inject_questions() -> &'static [InjectQuestion] {
    INJECT_QUESTIONS
}
