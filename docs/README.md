# Docs index

Guides that go deeper than the main [README](../README.md). Each is
self-contained; skim the table, then open the one you need.

| doc | covers |
|---|---|
| [central-memory-hub.md](central-memory-hub.md) | Running one shared store reachable from every machine and from web/mobile AI clients: `mimir serve` (sync hub), `mimir mcp --http` (`/mcp` + the warm `/inject` auto-recall endpoint), reverse-proxy/auth, backups. |
| [sync.md](sync.md) | The opt-in sync layer in depth: file mode vs. server mode, project-scoped vs. global memories, conflict handling, security model. |
| [proxy.md](proxy.md) | `mimir proxy`, the optional Anthropic-API man-in-the-middle that adds prompt-cache breakpoints and lossless dedup. |
| [benchmarks.md](benchmarks.md) | How the token-savings numbers are measured and how to reproduce them. |
| [design/project-sync.md](design/project-sync.md) | Design notes for project-scoped sync (internal, historical — describes the plan behind what `sync.md` documents as shipped). |

## FAQ

**Where's my data?**
One SQLite file, on your machine: `~/.local/share/mimir/mimir.db` (Linux;
platform-standard equivalents elsewhere), or everything under `$MIMIR_HOME`
if you set it. Nothing leaves your machine unless you turn on sync. `mimir
status` shows the exact path.

**How do I sync across machines?**
It's opt-in and off by default — see [sync.md](sync.md) for the two modes
(a synced folder, or a small `mimir serve` hub) and
[central-memory-hub.md](central-memory-hub.md) if you also want a remote MCP
endpoint for web/mobile clients.

**How do I save tokens?**
`outline`/`peek` instead of full file reads, `mimir run -- <cmd>` to strip
build/test noise, and the optional `mimir proxy` for cache breakpoints and
dedup on raw API traffic — see the README's Token savings section and
[benchmarks.md](benchmarks.md) for the measured numbers.

**How does auto-recall decide to stay silent?**
`mimir init --hooks --auto-recall` only injects a memory when it clears a
relevance floor (term overlap, plus lexical+semantic agreement when the
embedding model is loaded); below that floor it injects nothing on purpose —
a wrong memory in context is worse than none. See the README's Token savings
section for the full mechanics (warm `/inject` endpoint vs. cold CLI
fallback, git-diff enrichment).

**What languages are supported?**
14 via tree-sitter, for both the code graph (`mimir graph build`) and code
content search (`mimir code add`): Rust, TypeScript/JS/TSX, Python, Go, Java,
Ruby, C, C++, C#, Kotlin, Swift, PHP, SQL. Plus config/plain-text files
(`.toml`, `.yaml`, `.json`, `.sh`, `Dockerfile`, `Makefile`, `.env.example`,
`.txt`, `.rst`, …) that `mimir code add` chunks without parsing.
