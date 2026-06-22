# Changelog

All notable changes are documented here. Versions follow semver; the CLI,
the `mimir-mem` crate, and the on-disk schema move together.

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
