# Changelog

All notable changes are documented here. Versions follow semver; the CLI,
the `mimir-mem` crate, and the on-disk schema move together.

## [Unreleased]
### Changed
- **Ranking now favours fresh and preventer-type memories.** Recency is on
  by default (`scoring.recency_alpha` 0.0 → 0.4), so a current fact outranks
  a stale one on an equal match, and gotcha/decision memories get a modest
  priority nudge (`scoring.type_prior_alpha` 0.12) so a hard-won preventer
  isn't buried under an incidental note. Both are tunable in `config.toml`;
  RRF fusion `k` is unchanged. On the (reproducible) retrieval eval this
  lifts real-model drift-preventer recall 0.71 → 0.86 with no regression to
  the previously-correct set.

### Fixed
- **Warm recall latency ~254 ms → single-digit ms.** The cost was a full
  in-memory vector-matrix rebuild on every recall that followed any
  `remember`/embed (the cache was dropped wholesale, and same-connection
  writes don't bump SQLite's `data_version`). The cache now patches only the
  changed rows in place (~258 ms → ~6 ms on a 97k-node store); the fused
  `get_node` fan-out is also batched into one query. Result ordering is
  unchanged (parity-tested).
- **`mimir code add <dir>` was a silent no-op on a path already registered
  for docs.** Collections are now keyed by (path, kind), not path alone, so
  a code collection and a docs collection can coexist at the same root — the
  common case for `auto.docs` roots. Also: non-source, non-markdown files in
  either a docs or a code collection (`Cargo.toml`, `Dockerfile`,
  `.env.example`, a workflow `.yml`, ...) are now indexed too, as
  plain-text/config chunks (`Kind::Chunk`, so `code_damp` doesn't suppress
  them) — whitelisted by extension/name, lockfiles and real `.env` files
  excluded outright, size-capped. `collection_stats` and `mimir docs list`
  now count these alongside tree-sitter chunks instead of undercounting;
  the dashboard's embeddable-kind coverage query draws from the same
  `EMBEDDABLE_KINDS` list `embed_pending` uses instead of a second,
  hand-copied one; and `CodeChunk` search-result breadcrumbs now carry the
  file's relative path instead of just its stem, so same-named files in
  different directories (`mod.rs`, `index.ts`, ...) no longer produce
  identical, ambiguous crumbs. Plain-text/config chunk crumbs also switched
  from stem to full file name — `.env.example`'s title was displaying as
  `.env` (Rust's `file_stem` strips the extension *after* a leading dot,
  not the leading dot itself), which read as if a real, blacklisted `.env`
  had been indexed.

### Added
- **Opt-in per-prompt auto-recall: `mimir init --hooks --auto-recall`**
  installs a UserPromptSubmit hook that injects at most one relevant memory
  into each prompt's context — the counterpart to the existing static
  SessionStart rules pack. Off by default; `--hooks` alone is unchanged.
  Errs toward silence: only gotcha/decision memories (or pinned ones)
  qualify, and a hit must clear a documented relevance floor (minimum
  prompt/memory term overlap against title+body+tags, plus
  lexical+semantic leg agreement when the embedding model is loaded)
  before it's ever shown — a wrong injected memory is worse than none.
  Compact single-line format, capped at ~200 tokens. THOR has this; Mimir
  didn't — closing the gap. New CLI plumbing: `mimir recall-inject
  <prompt>` (not exposed on the MCP tool surface). The hook tries a warm
  `GET /inject` endpoint on `mimir mcp --http` first (one process-wide
  engine, ~30ms once loaded vs ~285ms for a fresh CLI process on the same
  store) and falls back to the cold CLI path when no daemon answers —
  correctness is identical either way; only latency differs. The warm
  endpoint's address is now a real config key, `[hooks] inject_url`
  (default `http://127.0.0.1:8077/inject`, matching the port already
  documented for `mimir mcp --http`), baked into the generated hook script
  at install time; `MIMIR_INJECT_URL` still overrides it at invocation
  time as the highest-precedence escape hatch.
- **Code content in recall: `mimir code add <dir>`** indexes source files
  chunked on tree-sitter symbol boundaries — function/method *bodies* (not
  just signatures) are now searchable, as `Kind::CodeChunk` nodes alongside
  the existing signature-only `Symbol` nodes. Chunks land in the default
  `recall` pool (and under `--kind code`); a tunable `scoring.code_damp`
  (default 0.85) keeps code from drowning out memories given its much
  larger corpus share. Idea credit: [@nworks3d](https://github.com/nworks3d)'s
  THOR fork of Mimir — thanks!

### Internal
- New `mimir_core::inject` module: the auto-recall relevance floor,
  formatting, and token budget, single-sourced between the cold CLI path
  (`mimir recall-inject`) and the warm HTTP endpoint (`mcp.rs`'s
  `/inject`), so the two can never disagree on what counts as relevant.
  Token-budget truncation switched from exact BPE counting
  (`mimir_core::tokens::count`) to a chars/4 heuristic on this hot path —
  it runs on every prompt, and the ~100ms one-time tokenizer-table-parse
  cost wasn't worth paying for a 200-token budget that doesn't need BPE
  precision; `tokens::count` is unchanged everywhere precision matters
  (the savings ledger).
- `search_hybrid` gains a sibling `search_hybrid_with_legs` (and
  `Mimir::search_with_legs`) that also returns the raw per-leg ranked id
  lists before RRF fusion — needed by auto-recall's relevance floor, kept
  as a thin wrapper so the scoring formula stays single-sourced.
- New workspace crate **`mimir-syntax`**: the tree-sitter parsing layer
  (symbol/call/import extraction) split out of `mimir-graph` into its own
  dependency-free crate, so both `mimir-graph` and the new `mimir-core`
  code-chunker can consume it without a `mimir-core` ↔ `mimir-graph` cycle.
  `mimir-graph` re-exports the same items, so this is a no-op for existing
  callers. Workspace is now five crates: `mimir-core`, `mimir-graph`,
  `mimir-syntax`, `mimir-proxy`, `mimir-cli`.
- **Reproducible retrieval-eval harness** (`crates/mimir-core/src/eval`,
  dev-only `eval` cargo feature): a seeded corpus + labelled question set
  with a tuning/holdout split scoring precision@k / recall / MRR in both a
  hermetic synthetic-vector mode and a real-model mode, so ranking changes
  are measured, not asserted. Not shipped on any user surface.


## [0.13.0] — 2026-07-03
### Added
- **Remote MCP: `mimir mcp --http <addr>`** serves the same tool router over
  Streamable-HTTP for networked clients (claude.ai web/mobile, agents on
  other machines) — stdio stays the default and is untouched. The transport
  carries no auth of its own, so non-loopback binds are refused unless
  `--http-allow-remote` states the exposure is deliberate; front it with TLS
  + an auth gate (see [docs/central-memory-hub.md](docs/central-memory-hub.md)).
  Contributed by [@nworks3d](https://github.com/nworks3d) (#8, #9) — thanks!
- **Memory hygiene over MCP: `forget`, `consolidate`, `supersede`** — agents
  can now prune (soft delete only; permanent deletion stays a human CLI
  action), dry-run/apply consolidation, and retire a stale memory in favor of
  its replacement from any MCP surface. Plus a CLI `mimir supersede <old>
  --by <new>` verb, `recall --include-superseded`, a sync source-identity
  line in `status`, and an opt-in `scoring.recency_alpha` boost (default off).
  Contributed by [@nworks3d](https://github.com/nworks3d) (#10) — thanks!

## [0.12.0] — 2026-06-26
### Fixed
- **No more `SQLITE_BUSY` under concurrent sessions** — multiple Claude Code
  sessions sharing the global store hit intermittent "database is locked"
  failures, including read-only commands dying at open. Three fixes, all within
  SQLite's own WAL + `BEGIN IMMEDIATE` + `busy_timeout` (no write-buffer daemon,
  which would sacrifice read-after-write consistency and durability):
  - `busy_timeout` was the 4th statement in the pragma batch, so
    `PRAGMA journal_mode = WAL` — which takes a momentary exclusive lock at open
    — ran first at the default 0ms timeout and failed instantly under
    contention. It's now set via the rusqlite API immediately after `open`,
    before any pragma, and raised to 10s.
  - `embed_pending` held the single writer lock for its entire run, including
    the CPU-bound model inference. It now commits in batches and runs inference
    *outside* the transaction, so a large embed never starves other writers.
  - Long-lived daemons' read marks blocked passive WAL autocheckpoints, letting
    the `-wal` file grow without bound (100 MB+). `mimir mcp`, `mimir serve`, and
    `mimir proxy` now run a periodic `wal_checkpoint(TRUNCATE)` on an idle timer.

## [0.11.0] — 2026-06-22
### Added
- **C# and SQL in the code graph** — tree-sitter symbol extraction now covers
  C# (`tree-sitter-c-sharp`) and SQL (`tree-sitter-sequel-tsql`, T-SQL). C#
  contributes namespaces, classes, interfaces, structs, records, enums,
  methods, properties and constructors, with call edges from method
  invocations and import edges from `using` directives. SQL contributes
  tables, views, functions and stored procedures; since SQL has no call graph,
  the call edge is reused as a dependency edge (dependent → table) for foreign
  keys, view `FROM`/`JOIN` references, and procedure/function body references.
  Wiring a language is a single-gate change (file extension → `Lang`), so
  `graph build`, `outline` and `peek` all pick the new languages up at once.
  Supported code-graph languages are now Rust, TypeScript/JS, Python, Go,
  Java, Ruby, C, C# and SQL.

## [0.10.0] — 2026-06-18
### Added
- **Project-scoped sync** — a project's memories can now replicate across
  machines, not just global memories. Opt in with `mimir project init --sync`,
  which writes a committed `.mimir` marker (a stable `id` + `sync = true`). The
  project's **portable key** — the `.mimir` `id`, else the normalized git
  `origin` remote (read from `.git/config`, no subprocess) — identifies it the
  same way on every checkout, so project memories converge by key instead of by
  absolute path. Pulled memories attach to a path-less *shadow* project that's
  adopted onto the local path when the project is first opened. Only memories
  sync — the code graph, indexed docs, and embeddings stay per-checkout. Opt-in
  per project; unmarked projects stay local. Schema migration **v6**.
- **`mimir project init [--sync]`** — write/refresh the `.mimir` identity marker
  (generates a stable ULID id; `--sync` opts the project's memories into sync).

## [0.9.1] — 2026-06-18
### Added
- **`mimir sync token`** — prints the active sync token (the `MIMIR_SYNC_TOKEN`
  env var, else the hub's persisted token) to stdout so it can be copied to a
  client, e.g. `docker exec mimir-hub mimir sync token`. No more digging through
  `docker logs` or SQLite to recover an auto-generated token.
### Docs
- README gained a **Token savings** section (outline/peek, `mimir run` output
  filters + the non-lossy content-command cap, the optional API proxy).
- `docs/sync.md`: documents where the auto-generated token is printed and how to
  retrieve it, a concrete client MCP-server env snippet, and hosting/platform
  notes (Intel Mac / older Linux / WSL2 → `cargo install`; don't host the hub in
  WSL2; attach a persistent volume for cloud-container hubs).

## [0.9.0] — 2026-06-18
### Added
- **Zero-init project detection** — non-git project directories are detected via
  build-file markers (`Cargo.toml`/`package.json`/`pyproject.toml`/`go.mod`/
  `go.work`/`deno.json[c]`/`pnpm-workspace.yaml`) instead of silently falling
  back to global scope. Detection is never silent: `mimir status` shows
  `[via: Cargo.toml]` / `[via: git]` or an explained global line plus a
  `touch .mimir` hint (text + JSON `scope`/`detected_via`); the MCP `status`
  tool and startup log carry the reason. CLI graph queries lazily build the
  graph on first use.
- **Content-command volume cap** — high-volume coreutils/search/`kubectl`
  (`cat`/`head`/`tail`/`ls`/`find`/`grep`/`rg`/`ps`/`df`/`du`/`tree`/`kubectl`)
  are now wrapped by `mimir run` and bounded with a **non-lossy** volume cap
  (head + tail + every signal line; the bulky middle elided behind a visible
  marker — never per-line dropped). `tail -f`/`journalctl -f` pass through so
  the wrapper can't hang. Savings record under a distinct `cap` ledger source.
### Fixed
- **Sync hub Docker build** — build on Debian trixie (glibc 2.41); the prebuilt
  ONNX Runtime pulled in by `fastembed`/`ort` links against glibc ≥ 2.38
  (`__isoc23_strtoll`) and failed to link on bookworm.

## [0.8.0] — 2026-06-16
### Added
- **Token-savings system** — Mimir now measures and reduces token spend, with a
  `savings_event` ledger (schema v5), a bundled tokenizer, and `mimir savings`
  (with dollar figures via `[savings]`, `--oneline` for statuslines) + a
  `/m-savings` slash command and dashboard panel.
- **`outline` / `peek`** — dense signature maps (code via tree-sitter, plus
  markdown/JSON/YAML) and single-symbol bodies, as MCP tools and CLI. Reading a
  file's outline costs ~12% of reading the whole file.
- **`run` / `rewrite`** — command-output filters (cargo, git, npm/pnpm/yarn/bun,
  pytest, go, make, docker/podman, pip, jest/vitest/eslint/tsc/next,
  terraform/tofu) via a declarative registry with an always-keep-errors safety
  net and a generic head+tail+signal volume cap. Opt-in PreToolUse hook
  installer (`mimir init --hooks`) replaces standalone output-filtering tools.
- **`rules`** — a per-project pinned "rules pack" auto-injected at session start
  via a SessionStart hook, so the agent stops re-deriving conventions.
- **Optional API proxy** (new `mimir-mem-proxy` crate, off by default):
  `mimir proxy` adds prompt-cache breakpoints (savings measured from
  `usage.cache_read_input_tokens`), lossless repeated-block dedup, and opt-in
  pruning. See [docs/proxy.md](docs/proxy.md) and [docs/benchmarks.md](docs/benchmarks.md).

## [0.7.0] — 2026-06-15
### Added
- **Optional centralized sync** (off by default): share global memories across
  installs via a replicated **file** folder (Syncthing/Dropbox/…) or a
  **`mimir serve`** hub (Docker image + compose included). `mimir sync`,
  `mimir serve`, a `[sync]` config section, opt-in background sync, and a
  config-gated `/m-sync` slash command. See [docs/sync.md](docs/sync.md).
- `recall --min-score` filters out hits below a fused relevance threshold, and
  `--json` records now include the hit's fused score. (Thanks to @nworks3d.)
### Fixed
- `store::soft_delete` now bumps `updated_at` (so deletes propagate in sync and
  are visible to change-tracking).
- **Windows:** projects on UNC network shares can now be indexed —
  `canonical_root` rewrites the verbatim `\\?\UNC\server\share` form to a valid
  `\\server\share` path instead of an unopenable `UNC\…`. (Thanks to @nworks3d.)
- `recall` collapses exact-duplicate content before truncating, so duplicates no
  longer crowd distinct results out of a `--limit`. (Thanks to @nworks3d.)

## [0.6.0] — 2026-06-13
### Added
- **Code-graph languages: Java, Ruby, and C** (joining Rust, TypeScript/JS,
  Python, Go) — symbol extraction, call resolution, and import/include edges.
- `CHANGELOG.md`, `SECURITY.md` (private disclosure + threat model), and
  `CONTRIBUTING.md`.
- Direct integration tests for the MCP tool handlers.

## [0.5.6] — 2026-06-13
### Fixed
- **Scale:** `resolve_ref` (used by `get`/`mark`/`edit`/`link`) no longer
  full-scans the node table. The 6-char short-id form now uses the
  `node_uid_tail` index via `substr(uid, -6)` equality — ~66 ms → ~24 µs at
  500k nodes, flat at any size.
### Added
- `scaling_profile` benchmark (ignored test) and a **Scaling** section in the
  README with measured recall/RAM at 50k/200k/500k embedded nodes.

## [0.5.5] — 2026-06-13
### Fixed
- **Perf:** added an `embedding(model, content_hash)` index (v4 migration);
  bulk (re-)embedding was O(N²) without it.
- **Robustness:** all multi-statement write transactions now begin
  `IMMEDIATE`, so overlapping writers wait (busy_timeout) instead of aborting
  with "database is locked".
- **Robustness:** `consolidate()` runs in one transaction — a mid-run failure
  can no longer leave an orphan "Distilled:" summary that pollutes recall.
- `parse_since` no longer panics on a multibyte `--since` unit.

## [0.5.4] — 2026-06-13
### Security
- Fixed a stored XSS in the graph visualization: an attacker-influenced tag
  could inject markup into the sidebar of the `file://` HTML page. Sinks are
  now escaped, the JS escaper covers `>`, and tags are sanitized at ingest.
- Dashboard/graph HTML is written `0600` (was world-readable in the shared
  temp dir).

## [0.5.3] — 2026-06-13
### Fixed
- `graph viz` builds the code graph automatically on first run, so `/m-graph`
  is one step instead of silently rendering an empty graph.

## [0.5.2] — 2026-06-13
### Added
- Project roots are recognized by `.git`/`.hg`/`.svn`/`.jj`, and `touch
  .mimir` marks a project root for code outside version control.

## [0.5.1] — 2026-06-13
### Fixed
- Audit release: 13 correctness fixes (soft-deleted-node resolution, a
  one-transaction graph build that survives crashes, mtime handling on
  exotic filesystems, dashboard error propagation, and more).

## [0.5.0] — 2026-06-13
### Added
- `mimir graph viz` — interactive, self-contained HTML graph visualization.
- `mimir report` — day/week/month/year/all-time activity table.
- `mimir link --scan` — auto-link memories to the code symbols they mention;
  the MCP `remember` tool gained a `link` parameter for capture-time linking.
- The MCP server auto-builds the code graph and indexes the repo's markdown
  on session start.
- `mimir init` installs `/m-*` slash commands for Claude Code, Codex,
  OpenCode, Gemini CLI, and Cursor.

## [0.4.0] — 2026-06-12
### Added
- Initial public release: typed memories, doc indexing, code graph, hybrid
  BM25 + local-ONNX-vector search with optional cross-encoder reranking,
  self-learning strength/decay, LLM-free weekly consolidation, importers,
  the MCP server, the dashboard, and prebuilt binaries.

[Unreleased]: https://github.com/MakerViking/mimir/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/MakerViking/mimir/releases/tag/v0.7.0
[0.6.0]: https://github.com/MakerViking/mimir/releases/tag/v0.6.0
[0.5.6]: https://github.com/MakerViking/mimir/releases/tag/v0.5.6
[0.5.5]: https://github.com/MakerViking/mimir/releases/tag/v0.5.5
[0.5.4]: https://github.com/MakerViking/mimir/releases/tag/v0.5.4
[0.5.3]: https://github.com/MakerViking/mimir/releases/tag/v0.5.3
[0.5.2]: https://github.com/MakerViking/mimir/releases/tag/v0.5.2
[0.5.1]: https://github.com/MakerViking/mimir/releases/tag/v0.5.1
[0.5.0]: https://github.com/MakerViking/mimir/releases/tag/v0.5.0
[0.4.0]: https://github.com/MakerViking/mimir/releases/tag/v0.4.0
