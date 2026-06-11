# Mimir

**Unified, local-first memory for AI coding agents.** One SQLite store where
typed memories, indexed docs, and (soon) code symbols are all nodes in one
graph — searched together by hybrid BM25 + local-ONNX semantic retrieval, and
exposed to agents as a single, globally-registered MCP server.

- **One store, every project.** A single database with project scoping —
  cross-project knowledge surfaces wherever it's relevant, project noise
  doesn't.
- **Typed memories.** `gotcha`, `decision`, `insight`, `idea`, `note`,
  `person` — with tags, links, and automatic near-duplicate refusal.
- **Docs search.** Point it at any folder of markdown; an incremental,
  hash-driven indexer chunks it heading-aware (code fences never split) and
  keeps it fresh in milliseconds.
- **Hybrid search.** FTS5 BM25 (porter-stemmed, stopword-aware) + bge-small
  embeddings (quantized ONNX, in-process, no daemon, no API key), fused with
  reciprocal-rank fusion. Optional cross-encoder reranking (`--rerank`) when
  you want maximum precision over speed. No model downloaded? Everything
  still works, BM25-only.
- **Made for agents.** Default output is one ~25-token line per hit.
  The MCP server registers once (`--scope user`) and serves every repo,
  detecting the current project from its working directory.
- **Local and private.** A memory tool holding your decisions and notes must
  be beyond suspicion: everything stays on disk, **zero telemetry**, ever.

## Install

```sh
cargo install --path crates/mimir-cli   # from a checkout (crates.io soon)
```

## Quickstart

```sh
mimir init                 # creates config + db, downloads the embedding model (~34 MB)
mimir init --no-model      # …or stay BM25-only / offline

# memories
mimir remember "SCRAM auth rejects non-ASCII passwords" -t gotcha --tags auth,postgres
mimir recall postgres password trouble
mimir get m:ABCDEF         # full body (also: mimir get notes.md:10-40)

# docs
mimir docs add ~/notes --name notes
mimir index                # incremental; re-run any time

# precision dial (all optional)
mimir embed --fetch --rerank                  # one-time reranker download (~150 MB)
mimir recall tricky semantic question --rerank  # cross-encoder rescoring (~2 s)
# config.toml: embedding.model = "bge-base-en-v1.5" — stronger semantic
# matching at the same query latency (index-time embedding is ~4x slower)

# agents (Claude Code etc.) — register once, works in every repo
claude mcp add --scope user mimir -- mimir mcp
```

MCP tools: `recall`, `remember`, `get`, `link`, `status`.

## How it works

Everything is a node — memories, files, chunks, projects, collections, tags,
annotations — in one SQLite database (WAL, FTS5, no extensions). Embeddings
are plain f32 blobs keyed by content hash + model, brute-force scanned
in-process (exact, single-digit ms at ≤200k items). Search legs are fused
with RRF (k=60); learned strength only ever acts as a tiebreaker. Concurrent
CLI + MCP-server access is the normal, supported case.

State lives in the platform-standard directories
(`~/.local/share/mimir`, `~/.config/mimir`, `~/.cache/mimir` on Linux);
set `MIMIR_HOME=<dir>` to put everything under one directory instead.

## Roadmap

- **v0.2** — code graph: tree-sitter symbol extraction (Rust/TS/Python),
  call/import edges, impact queries, memory↔code links.
- **v0.3** — self-learning: recall feedback ledger, strength/decay ranking,
  LLM-free consolidation (dedup, contradiction flagging, distillation).
- **v1.0** — importers (OpenBrain, Claude auto-memory, QMD), prebuilt
  binaries, more languages.

## License

MIT or Apache-2.0, at your option.
