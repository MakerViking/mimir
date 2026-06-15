# Changelog

All notable changes are documented here. Versions follow semver; the CLI,
the `mimir-mem` crate, and the on-disk schema move together.

## [Unreleased]
### Added
- **Optional centralized sync** (off by default): share global memories across
  installs via a replicated **file** folder (Syncthing/Dropbox/…) or a
  **`mimir serve`** hub (Docker image + compose included). `mimir sync`,
  `mimir serve`, a `[sync]` config section, opt-in background sync, and a
  config-gated `/m-sync` slash command. See [docs/sync.md](docs/sync.md).
### Fixed
- `store::soft_delete` now bumps `updated_at` (so deletes propagate in sync and
  are visible to change-tracking).

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

[Unreleased]: https://github.com/MakerViking/mimir/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/MakerViking/mimir/releases/tag/v0.6.0
[0.5.6]: https://github.com/MakerViking/mimir/releases/tag/v0.5.6
[0.5.5]: https://github.com/MakerViking/mimir/releases/tag/v0.5.5
[0.5.4]: https://github.com/MakerViking/mimir/releases/tag/v0.5.4
[0.5.3]: https://github.com/MakerViking/mimir/releases/tag/v0.5.3
[0.5.2]: https://github.com/MakerViking/mimir/releases/tag/v0.5.2
[0.5.1]: https://github.com/MakerViking/mimir/releases/tag/v0.5.1
[0.5.0]: https://github.com/MakerViking/mimir/releases/tag/v0.5.0
[0.4.0]: https://github.com/MakerViking/mimir/releases/tag/v0.4.0
