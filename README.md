<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="Mimir — unified, local-first memory for AI coding agents" width="360">
  </picture>
</div>

<div align="center">

[![crates.io](https://img.shields.io/crates/v/mimir-mem.svg)](https://crates.io/crates/mimir-mem)
[![CI](https://github.com/MakerViking/mimir/actions/workflows/ci.yml/badge.svg)](https://github.com/MakerViking/mimir/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

**Unified, local-first memory for AI coding agents.** One SQLite store where
typed memories, indexed docs, and code symbols are all nodes in one graph —
searched together by hybrid BM25 + local-ONNX semantic retrieval, and exposed
to agents as a single, globally-registered MCP server.

<div align="center">
  <img src="assets/demo.gif" alt="Storing two memories, then recalling the right one from a vague query" width="720">
  <br>
  <em>Store it once. Ask in plain language. The right memory comes back.</em>
  &nbsp;·&nbsp;
  <a href="https://youtu.be/VOU68g1I1-I">▶ watch the 90-second tour</a>
</div>

![Mimir benchmark: 7–360× faster than the tools it replaces](assets/benchmark.svg)

> **How this was measured.** Not a formal benchmark — a switch-over test on
> my own real data, run on the same machine against the three tools Mimir
> replaces (OpenBrain, QMD, Graphify): same queries, same corpus, wall-clock
> timed. Corpus: **104 memories, 642 doc chunks, and a 2,495-file TypeScript
> repo (11,735 symbols)**. CPU numbers; the GPU build is faster still (recall
> 22 ms → 7 ms). Each bar names the operation and the tool it beats — the
> 360× is one task (code-graph refresh: Graphify's 3m 18s vs Mimir's 0.55s),
> not a blended average. Numbers move with corpus size and hardware; treat
> them as "what happened when I switched," not a universal promise.

## Why

Mimir replaced three tools that each did their job fine: OpenBrain
(semantic memory service), QMD (markdown search), and Graphify (code
knowledge graph). The problem was never that they didn't work — it was
that they were three daemons, three stores, three query surfaces, and
none of them knew about each other. A memory couldn't point at the
function it was about; doc search couldn't surface the decision that
explained the doc. Running three systems where one could do the job —
and do it better, *because* everything lives in one graph — was too
enticing not to build. The speedups in the chart above are real, but
they're a side effect; the point is the links.

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
- **Code graph.** Tree-sitter symbol extraction (Rust, TypeScript/JS,
  Python, Go, Java, Ruby, C, C++, C#, Kotlin, Swift, PHP, SQL) with
  call/import edges: `graph callers`, `impact` (blast radius of a diff),
  `path`, `hubs` — and code symbols participate in semantic recall. Link
  memories to functions and they surface together.
- **Code content search.** `mimir code add <dir>` indexes source files
  chunked on tree-sitter symbol boundaries, so recall matches function/method
  *bodies* — inline comments, string literals, implementation details — not
  just signatures. Same incremental, hash-driven indexer as docs; a tunable
  `scoring.code_damp` (default 0.85) keeps code's much larger corpus share
  from drowning out memories. Config/plain-text files a source tree needs
  but tree-sitter can't parse (`.toml`, `.yaml`/`.yml`, `.json`, `.sh`,
  `Dockerfile`, `Makefile`, `.env.example`, `.txt`, `.rst`, …) are chunked
  too; lockfiles (`Cargo.lock`, `package-lock.json`, …) and a real `.env`
  are never indexed. Idea credit:
  [@nworks3d](https://github.com/nworks3d)'s THOR fork of Mimir.
- **Self-learning.** Recall usage strengthens what helps (`mark` for
  explicit feedback); typed half-life decay quiets what doesn't; weekly
  LLM-free consolidation dedups, flags contradictions, distills clusters,
  and archives the dead — never destructively.
- **Made for agents.** Default output is one ~25-token line per hit.
  The MCP server registers once (`--scope user`) and serves every repo,
  detecting the current project from its working directory. On session
  start it auto-builds the project's code graph and indexes its markdown
  (background thread, incremental, milliseconds after first contact) —
  zero setup per project. Opt out in `config.toml`: `[auto] graph/docs = false`.
- **Local and private.** A memory tool holding your decisions and notes must
  be beyond suspicion: everything stays on disk, **zero telemetry**, ever.

## Install

Prebuilt binary (Linux x86_64/aarch64, macOS Apple Silicon):

```sh
curl -fsSL https://raw.githubusercontent.com/MakerViking/mimir/main/install.sh | sh
```

Windows: grab `mimir-windows-x86_64.zip` from the
[latest release](https://github.com/MakerViking/mimir/releases) and put
`mimir.exe` on your PATH.

From source (any platform with [Rust](https://rustup.rs)):

```sh
cargo install mimir-mem                 # the binary is named `mimir`
cargo install --path crates/mimir-cli   # …or from a checkout
```

Use the from-source path on **Intel Macs** (no prebuilt) and on **older Linux /
WSL2** distros (the prebuilt Linux binary targets a recent glibc). WSL2 works
fine as a sync *client*; see [docs/sync.md](docs/sync.md) for where to run the
optional *hub*.

That's the whole install. CPU-only by default, and it's plenty fast —
the GPU build is an optional power-user step, tucked away below.

<details>
<summary><b>Optional GPU acceleration</b> (Vulkan / CUDA — opt-in build)</summary>

GPU is an opt-in build feature (pick **one**):

```sh
# Cross-vendor: Vulkan (Linux), D3D12 (Windows), Metal (macOS) via Dawn.
# The right choice for AMD/Intel GPUs.
RUST_MIN_STACK=33554432 cargo install mimir-mem --features gpu-webgpu

# NVIDIA CUDA 12/13:
RUST_MIN_STACK=33554432 cargo install mimir-mem --features gpu-cuda
```

Notes:
- `RUST_MIN_STACK` works around a rustc/LLVM ThinLTO crash when linking the
  large onnxruntime GPU binary.
- The webgpu build dynamically links `libwebgpu_dawn.so` — copy it from the
  build cache next to the binary (the binary's `$ORIGIN` rpath finds it
  there), or set `LD_LIBRARY_PATH`:
  `cp $(find ~/.cache/ort.pyke.io -name 'libwebgpu_dawn.so' | head -1) ~/.cargo/bin/`
- `config.toml: embedding.device = "cpu"` forces CPU in a GPU build;
  the default `"auto"` falls back to CPU if GPU init fails.

**Windows, honestly:** the prebuilt `mimir.exe` is CPU-only, so any GPU
flavor means compiling from source — and on Windows that means installing
Rust ([rustup](https://rustup.rs)) *plus* the Visual Studio Build Tools
C++ workload (several GB of toolchain). That's standing up a native dev
environment, not flipping a feature — weigh it against what GPU actually
buys you (bulk indexing and fast `--rerank`; single-query embeds are
*faster on CPU*, which is why sessions use CPU by design). If you do build:

- The CUDA Toolkit is **not** needed to compile — onnxruntime is
  downloaded prebuilt during the build. The `gpu-cuda` binary *does* need
  NVIDIA's CUDA 12/13 + cuDNN 9 runtime DLLs findable at run time (the
  toolkit installer is the easiest way to get them).
- NVIDIA users have a lighter path: `gpu-webgpu` uses DirectX 12 on
  Windows — no CUDA install at all (upstream marks the WebGPU provider
  experimental).
- **Smart App Control breaks `cargo` builds entirely** (field-reported):
  build scripts are freshly compiled, unsigned executables, which SAC
  blocks — for *any* Rust build, either GPU flavor. Disabling SAC is
  permanent (Windows won't re-enable it without a reset), so if SAC is
  on, our honest advice is to stay on the prebuilt CPU binary rather
  than trade SAC for a GPU build.
- Set the stack workaround PowerShell-style:
  `$env:RUST_MIN_STACK=33554432` before `cargo install`.

Measured on an RX 6900 XT (Vulkan): bulk embedding 2.3× faster, recall
22 ms → 7 ms, `--rerank` 1.9 s → 0.14 s.

</details>

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
mimir recall tricky semantic question --rerank  # cross-encoder rescoring (~1 s CPU, ~0.15 s GPU)
# config.toml: embedding.model = "bge-base-en-v1.5" — stronger semantic
# matching at the same query latency (index-time embedding is ~4x slower)
# config.toml: embedding.model = "granite-embedding-small-r2" — same
# precision as the default, ~3x faster per-embed (re-embeds the store)
# config.toml: rerank.model = "jina-reranker-v1-turbo-en-int8" — ~1.2x
# faster reranking, mild risk of reshuffling a top-3 result

# code graph (the MCP server runs build + docs indexing automatically
# on session start — these are for manual/CLI use)
mimir graph build                 # tree-sitter extraction, incremental
mimir graph callers resolve_ref   # who calls this?
mimir graph impact $(git diff --name-only)   # blast radius of a change
mimir graph viz --open            # interactive graph map (self-contained HTML)
mimir link m:ABC123 my_function --rel about  # decisions ↔ code
mimir link --scan --all-projects  # auto-link memories to symbols they name

# code content (function/method bodies, not just signatures, in recall)
mimir code add ~/src/myproject --name myproject
mimir index                       # same incremental indexer as docs

# feedback & hygiene
mimir mark m:ABC123 --useful      # strengthen future ranking
mimir consolidate --dry-run       # dedup/contradictions/distill/archive
mimir dashboard --open            # self-contained HTML telemetry panel
mimir report                      # activity table: day/week/month/year/all-time

# escape hatches
mimir import openbrain export.txt | claude-memory <dir> | qmd
mimir export > backup.jsonl       # everything, always yours

# agents (Claude Code etc.) — register once, works in every repo
claude mcp add --scope user mimir -- mimir mcp
```

### Grounding: which memories point at real code

A memory linked to an indexed symbol makes a claim Mimir can re-check. When
the symbol goes away, the link is *falsified* — not the memory, which is
often still right, but the connection nobody has revisited since the code
moved. That shows up wherever you meet the memory:

```sh
mimir grounding                   # 151 grounded, 0 stale, 577 ungrounded
mimir grounding --stale           # the ones whose target is gone
```

```
m:QJ18BM [gotcha pr:mimir 08-11 stale-link] retry_with_backoff doubles the delay …
```

It is reported, never scored: grounding changes what you can see about a
memory, not which memories come back. Most memories are legitimately
ungrounded — a note about a DNS quirk or a person names no code — so treat
it as a signal on the code-referencing subset, not a number to maximise.

**Grounding only exists if something creates the links.** `--all-projects`
matters more than it looks: most memories are *global*, and a scan compares
them against the graph of one project at a time, so scanning only where you
happen to be standing leaves the rest unlinked. On a real 728-memory store,
scanning one project moved grounding from 1.4% to 2.6%; scanning all 24 took
it to 20.7%.

```sh
mimir graph build                        # in each project you care about
mimir link --scan --all-projects --dry-run   # see what it would link
mimir link --scan --all-projects             # ~600 links on a mature store
```

`mimir doctor` reminds you if you have code graphs and have never scanned,
and goes quiet once you have.

### The fast path (recommended setup)

Three commands get you the lowest-latency configuration Mimir has — measured
on the same machine as the benchmark above:

```bash
mimir daemon &                        # warm engine: /inject answers in ~7-40 ms
mimir init --hooks --auto-recall      # per-prompt recall via the warm path
mimir doctor                          # confirms "daemon: warm (...)"
```

- **With the daemon running**, every auto-recall injection is served warm
  with full hybrid BM25+vector search — ~7 ms on a small store, ~40 ms
  median on a 102k-node store (measured live; the store grows, the number
  moves).
- **Without it**, the hook falls back to the cold CLI path, which defaults
  to `cold_mode = "fast"` — still ~5–6 ms, lexical + identifier matching
  only (semantic-only matches wait for the daemon).
- Either way the engine's own warm recall is single-digit ms; there is no
  configuration in which the hook blocks your prompt for half a second.
- **On GPU builds the daemon is also the only GPU process.** It serves
  `/embed` and `/rerank` alongside `/inject`, and every MCP session
  delegates bulk embedding and reranking to it (`[daemon] inference =
  "auto"`, the default) while running its own query embedder CPU-only —
  which is measured *faster* than GPU for single-query embeds. Net: N
  concurrent agent sessions cost one model's worth of GPU memory instead of
  N. No daemon, no problem: sessions fall back to fully-local inference
  automatically; `[daemon] inference = "off"` restores the old
  every-process-loads-its-own behavior.

![GPU memory before and after the shared inference daemon: five sessions at ~11.4 GiB total and growing, versus zero per session plus one 0.3–2.4 GiB daemon recycled daily](assets/daemon-vram.svg)

To keep the daemon across reboots, install the systemd user unit:
`cp contrib/mimir-daemon.service ~/.config/systemd/user/ && systemctl --user enable --now mimir-daemon`.

### Default vs quality — one config away

Mimir's defaults are deliberately **speed- and token-biased**: a fresh
install answers fast, injects little, and never surprises a CPU-only box.
Everything that trades a little latency (or a bounded token budget) for
measurably better retrieval is **opt-in** — each knob was measured before
it earned a place here, and the chart names what every one buys and costs:

![Default vs quality profile: reranked recall buys +0.03 nDCG@10 for ~180 ms GPU / ~1.3 s CPU; per-prompt auto-recall costs 40 ms and ~71 tokens when it fires; cold semantic injection costs ~235 ms; the session brief costs at most 450 tokens per session](assets/quality-profile.svg)

The whole quality tier, as config:

```toml
[rerank]
auto = "warm"        # +0.03 nDCG@10, first hit at rank 1 · ~0.18 s/recall
                     # on GPU, ~1.3 s on CPU — see the decision table below

[hooks]
cold_mode = "full"   # semantic matching even with no daemon · ~235 ms cold
context_guard = "handoff"  # deliberate /clear + auto-restored handoff memory

[brief]
enabled = true       # gotcha/decision digest at session start · ≤450 tok/session
```

plus the auto-recall hook at install time (`mimir init --hooks
--auto-recall`) and the daemon itself (above) — which is not a trade-off
but free quality: warm ~40 ms injections and zero GPU memory per session.
One more knob, `[scoring] impression_alpha`, damps memories that keep
surfacing without ever being opened; it's off by default because it needs
months of usage history to mean anything and has no measured delta yet —
turn it on only once `mimir report` shows real recall traffic.

> **Honesty box.** Every number above — and in both charts — was measured
> on **one machine and one real store: mine** (102k nodes grown from daily
> use; retrieval-quality deltas from an 837-label graded eval over Mimir's
> own repo). None of it is a synthetic benchmark, and none of it is a
> promise: a different corpus, GPU, or working style will shift the
> numbers, which is exactly why these are knobs and not defaults. If you
> benchmark Mimir against something else, please say which profile you
> ran — "defaults" and "quality" are different tools on purpose.

### Session brief

**On by default** (since its eval gate passed — see the caveats below for
the numbers and how to switch it off). A fresh session (or one that just
compacted) has forgotten every gotcha it ever learned — that window is
where an agent drifts into violating a constraint nobody repeated.
`mimir brief show` (a SessionStart hook entry `mimir init --hooks`
installs) injects a hard-capped digest of the current project's most
drift-preventing memories — gotchas and decisions, ranked by the signals
the store already has (pin, decayed strength, recency, same-project
affinity), one imperative line each:

```
Mimir guards (do not violate):
- GOTCHA [m:ABC123]: never test against prod without MIMIR_HOME isolation
- DECISION [m:XY9QRT] (10mo): we chose sqlite over postgres — don't relitigate
```

Zero authoring required, zero per-prompt cost — it fires once at startup
(default cap 150 tokens / 6 lines) and again after a `/clear` or compact
with a smaller 100-token recap, at most 4 rendered fires per session:
worst case **450 tokens per session, ever**, each fire recorded in the
savings ledger as spend. A recap re-anchors the top guards the wipe made
the agent forget; a startup fire skips what was already shown this session
(and, across sessions, anything briefed in the last 6 hours). Memories
covered by your rules pack, superseded, or below the relevance/quality
bar for this project are never shown; `mimir brief` previews the exact
output with per-candidate scores so "why did this appear?" always has an
answer.

```toml
[brief]
enabled = false       # the kill-switch (default true)
max_tokens = 150      # startup fire cap (recap_tokens = 100 for re-fires)
max_items = 6
```

Honest caveats: it needs a hook-running client (Claude Code — MCP-only
clients like Claude Desktop don't execute SessionStart hooks and get no
brief). It shipped disabled until its drift-eval gate passed, and the
default flipped only on those numbers (scoped to hook-running clients):
selection catches 96.0 pts of labeled must-know guards at the 150-token
cap vs 4.0 for a hand-written rules pack alone, the 150 cap is where the
marginal-catch curve flattens (85.3/96.0/96.0/96.0 at 100/150/250/400),
zero forbidden guards ever rendered across the fixture families, and a
preregistered repeated-exposure experiment found no boilerplate
blindness — subject compliance held at 0.889 whether the brief had been
seen 1, 5, or 10 times. Configs written by `mimir init` before the flip
carry an explicit `enabled = false` and keep it — flip the key in place
(don't append a second `[brief]` table). The session-boundary briefing
idea comes from [@nworks3d](https://github.com/nworks3d)'s THOR fork of
Mimir (built clean-room from the concept) — thanks!

### Context guard

Opt-in, off by default. `mimir init --hooks --context-guard pause` (or
`handoff`) adds three more hook entries that estimate how full the
context window is from the transcript file's **size** (no JSONL parsing,
so it's cheap on every prompt) and act once the estimate crosses
`context_guard_threshold_pct` (default 45%, of `context_window_tokens`,
default 200,000):

- **`pause`** nudges you, at most once per +10 percentage-point band, to
  deliberately `/clear` or `/compact` instead of leaving it to Claude
  Code's own auto-compact. If an *automatic* compact is attempted anyway
  while still over threshold, `PreCompact` blocks it — your own
  `/compact` is never blocked.
- **`handoff`** does the same nudge, plus instructs the agent to save a
  `session-handoff`-tagged memory before you clear. The next
  `SessionStart` after a clear or compact restores the latest handoff
  memory automatically, so the new session picks up where the old one
  left off instead of starting cold.

Tune the estimate in `config.toml`:

```toml
[hooks]
context_guard = "pause"              # off (default) | pause | handoff
context_guard_threshold_pct = 45
context_window_tokens = 200000       # raise for a 1M-context model
transcript_bytes_per_token = 8.0     # rough transcript-JSONL average
```

**Guard anchors** are a separate, independent opt-in — `mimir init --hooks`
always installs the hook, but it stays dormant until a memory declares one:

```bash
mimir remember "This table needs a migration, not an ALTER" --anchor "schema.sql"
```

The pattern matches a path suffix, so it fires the moment a matching file
is edited/written *or* a matching command runs (e.g. `psql -f schema.sql`)
— surfaced via `PreToolUse`, before the tool call happens, no prompt
needed. Up to 8 patterns per memory; `--anchor` is repeatable.

Related: `mimir remember --fires-when "phrase"` (also repeatable) declares
trigger phrase(s) that bypass auto-recall's inferred-relevance floor on a
close match — for facts that are easy for BM25/vector search to under-rank
but you always want recalled when that phrase comes up.

### Already have a memory system?

Don't start from zero — migrate, verify, then retire the old one.
Claude Code auto-memory directories import directly
(`mimir import claude-memory ~/.claude/projects/<project>/memory`). For
anything else, your agent is the importer — tell it:

> Read every entry in [my old memory system] and store each one in Mimir
> with the `remember` tool — pick a fitting type (gotcha/decision/insight/
> idea/note/person), keep the original wording, add tags. When done,
> compare counts with `mimir status` and spot-check a few searches with
> `recall`.

Re-running is safe: `remember` refuses near-duplicates, so a second pass
only fills gaps. **Verify before you delete** — compare entry counts,
recall a handful of things you actually remember storing — and only then
unplug the old system's MCP server. (And `mimir export` keeps the exit
door open in the other direction: everything, always yours.)

MCP tools: `recall`, `remember`, `get`, `link`, `graph`, `mark`, `status`,
`outline`, `peek`, and the hygiene set `forget` / `consolidate` / `supersede`
(soft-delete and dry-run by default — permanent deletion stays a human CLI
action).

### Works with any MCP client

Nothing here is Claude-specific: `mimir mcp` is a standard stdio MCP
server, so any MCP-capable agent can use it — Cursor, Windsurf, Cline,
Zed, VS Code (Copilot agent mode), Gemini CLI, Codex CLI, … For clients
configured via JSON, the entry is simply:

```json
{ "mcpServers": { "mimir": { "command": "mimir", "args": ["mcp"] } } }
```

Clients that can't launch a local process (claude.ai web/mobile, agents on
another machine) can reach a store over the network instead:
`mimir mcp --http 127.0.0.1:8077` serves the same tools via Streamable-HTTP —
bind to localhost and front it with TLS + an auth gate, and optionally require
a bearer token (`MIMIR_HTTP_TOKEN` / `--http-token`) as defense-in-depth; see
[docs/central-memory-hub.md](docs/central-memory-hub.md).

The project is detected from the directory the client launches the server
in (override with the `MIMIR_PROJECT` env var), walking up from there in
order: (1) a VCS / explicit root — `.git`, `.hg`, `.svn`, `.jj`, or a
`touch .mimir` marker; else (2) the nearest build file — `Cargo.toml`,
`package.json`, `pyproject.toml`, `go.mod`, `go.work`, `deno.json`, or
`pnpm-workspace.yaml`; else (3) global scope. Identity is the resolved root
path, so the same folder always maps to the same project, git or not — no
per-project init. Mimir never degrades silently: `mimir status` always shows
the detected project and how it was found (`[via: Cargo.toml]`), or, when
nothing matches, says so and points at `touch .mimir`. And agents without MCP
can just shell out — the CLI's default output is the same token-lean format
the server returns.

> **A memory's project is decided by the directory your session runs in** —
> not by what you happen to be editing. Start each project's session inside
> that project's folder so captures tag correctly (every capture confirms
> its project as `pr:<name>` in the output, so a wrong one is visible
> immediately). Pin a stable identity with `mimir project init --sync` and
> commit the `.mimir` file. If something still lands in the wrong place,
> move it with `mimir reproject <ref> --project <name>` (or `--global`).

`mimir init` also installs a set of `/m-*` slash commands for the agent
CLIs it finds on the machine — Claude Code, Codex, OpenCode, Gemini CLI
and Cursor (the `m-` prefix keeps them clear of your own commands):

| command | does |
|---|---|
| `/m-recall <query>` | search memories, docs and code |
| `/m-remember <fact>` | capture a memory (typed, tagged, linked) |
| `/m-graph` | open the interactive graph visualization |
| `/m-impact` | blast radius of your uncommitted changes |
| `/m-scan` | auto-link memories to the code they mention |
| `/m-report` | activity table: day/week/month/year/all-time |
| `/m-stats` | open the stats dashboard |
| `/m-doctor` | health check |

Only apps already present get them, and existing command files are never
overwritten, so your edits survive upgrades (re-run `mimir init` any
time; it's idempotent).

## Token savings

Mimir doubles as a token-saving layer for your agent — and it's **measured, not
vibes**: `mimir savings` reports the tokens (and dollars) avoided, with a
dashboard panel and a `/m-savings` command.

- **`outline` / `peek`** — read a file's *shape* (signatures via tree-sitter,
  plus markdown/JSON/YAML) or a single symbol's body instead of the whole file.
  Typically **~88–95% fewer tokens** than a full read — the biggest single lever.
- **`mimir run -- <cmd>`** — run a command and strip the noise: build/test/
  package/infra progress chatter is dropped (errors and warnings are *always*
  kept), and high-volume `cat`/`grep`/`find`/`ls`/`kubectl`/… output is
  **volume-capped non-lossily** (head + tail + every signal line; the bulky
  middle elided behind a visible marker). `mimir init --hooks` wires this in as
  a PreToolUse hook so it happens automatically.
- **`mimir init --hooks --auto-recall`** (opt-in, off by default) — injects at
  most one relevant gotcha/decision memory into each prompt's context via a
  UserPromptSubmit hook, on top of the static SessionStart rules pack. Errs
  toward silence: a hit must clear a relevance floor (term overlap, plus
  lexical+semantic agreement when the embedding model is loaded) before it's
  ever shown, capped at ~200 tokens. The hook also enriches a thin prompt with
  up to 8 changed-file stems from `git diff` — it can extend a real overlap,
  never single-handedly clear the floor on its own. It tries a warm `GET
  /inject` endpoint first (one process-wide engine kept alive by `mimir mcp
  --http 127.0.0.1:8077`, ~ms once loaded) and falls back to the cold
  `mimir recall-inject` CLI path when no daemon answers — same relevance
  logic either way, only latency differs. That cold fallback defaults to
  `[hooks] cold_mode = "fast"` (BM25 + identifier legs only, no ONNX load:
  ~5-6ms) rather than `"full"` hybrid search (~230-240ms) — it narrows which
  matches clear the floor while cold (no purely-semantic hits) without
  weakening the silence-beats-wrong-injection contract itself; set `"full"`
  to trade that latency back for semantic coverage on the cold path. The warm
  endpoint's address is `[hooks] inject_url` in `config.toml` (default
  `http://127.0.0.1:8077/inject`), overridable per-invocation with
  `MIMIR_INJECT_URL`. `mimir daemon` is a
  short alias for `mimir mcp --http <addr>` that reads that same address, so
  there's one command and one config key for "start the warm path." Run it
  as a service with the sample unit at
  [contrib/mimir-daemon.service](contrib/mimir-daemon.service) (`cp` it to
  `~/.config/systemd/user/`, then `systemctl --user enable --now
  mimir-daemon`); `mimir doctor` reports whether it's actually reachable.
- **Optional API proxy** (`mimir proxy`, off by default) — adds prompt-cache
  breakpoints and lossless repeated-block dedup to Anthropic API traffic. The
  breakpoint TTL is configurable (`--cache-ttl 5m|1h`); 5m is the default
  because a 1h write costs 2x input vs 1.25x, so it only pays off when turns
  idle more than 5 minutes apart. See
  [docs/proxy.md](docs/proxy.md) and [docs/benchmarks.md](docs/benchmarks.md).

## Optional: centralized sync

Want your memories on more than one machine? Mimir has an **opt-in** sync layer
— off by default, with zero added cost and the zero-telemetry promise intact
unless you turn it on. It shares your **global memories** plus any projects
you opt in (`mimir project init --sync` gives a project a portable identity;
local-SQLite stays authoritative; merges are last-write-wins), via whichever
path suits you:

- **File** — point it at a folder Syncthing/Dropbox/iCloud/git already
  replicates. No server, no token: `[sync] mode = "file", dir = "~/Synced/mimir"`.
- **Server** — run a hub with the same binary (`mimir serve`), deployable via
  the included `Dockerfile`/`docker-compose.yml` on a NAS, Pi, VPS, or any
  Docker host, reached over your tailnet or behind TLS.

Then `mimir sync` (or enable background sync). Full setup, recipes, and the
security model are in **[docs/sync.md](docs/sync.md)**.

## How it works

Everything is a node — memories, files, chunks, projects, collections, tags,
annotations — in one SQLite database (WAL, FTS5, no extensions). Embeddings
are plain f32 blobs keyed by content hash + model, brute-force scanned
in-process (exact, single-digit ms at ≤200k items). Search legs are fused
with RRF (k=60); learned strength only ever acts as a tiebreaker. Concurrent
CLI + MCP-server access is the normal, supported case.

Ranking has a few tunable knobs in `config.toml`, all sane by default:
`[scoring] recency_alpha` (default `0.012`) nudges fresher decaying-kind
memories (gotcha/decision/insight/note/idea); `type_prior_alpha` (default
`0.12`) nudges gotcha/decision ahead of an equally-matching note/idea;
`code_damp` (default `0.85`) keeps `mimir code add` content from drowning out
memories; `impression_alpha` (default `0.0`, off) mildly damps a node that's
been shown repeatedly but never opened — opt in once you have enough
`recall_event` history for it to mean something. That history is itself
bounded by `[learn] event_retention_days` (default `180`, `0` keeps forever):
a background prune drops older `recall_event` rows on daemon startup and on
the same idle cadence as the WAL checkpoint, sized so 180 days holds 6 of
`impression_alpha`'s 30-day decay half-lives — comfortably more than the
decay ever weights meaningfully; values below a 60-day safety floor are
clamped up (with a log warning) rather than silently starving the signal.
`[rerank] auto = "off" |
"warm" | "always"` (default `"off"`) controls whether a plain `recall`
auto-reranks without an explicit `--rerank`. The trade-off, measured on a
graded retrieval eval over this repo's own docs + source (837 dual-judged
labels, 60 queries): reranking the top 15 fused candidates raises nDCG@10
by ~0.03 and puts the first relevant hit at rank 1 almost always. The cost
is where your hardware decides:

| build | cross-encoder cost | per recall (15 candidates) |
|---|---|---|
| CPU | ~84 ms/candidate | ~1.3 s |
| GPU | ~12 ms/candidate | ~0.18 s |

`"warm"` fires only when reranking is free *right now*: the model is
already resident in this process, or a live `mimir daemon` answers for it
(`[daemon] inference = "auto"`). Long-lived processes (the MCP server,
`mimir daemon`) eager-load a local copy at startup only when no daemon is
reachable; one-shot CLI calls never pay a cold model load. Pick by how you
consume recall:
- **GPU build**: `"warm"` is a straightforward win — the quality gain at
  ~0.18 s is below anything you'll notice, and with `mimir daemon` running
  every session shares its one GPU copy.
- **CPU build, agent-driven recall** (MCP): `"warm"` is still a reasonable
  choice — ~1.3 s disappears inside an LLM turn that takes seconds anyway;
  you're trading invisible latency for better-ordered results.
- **CPU build, interactive CLI use**: leave it `"off"` — 1.3 s per recall
  is very noticeable at a prompt. Explicit `--rerank` is always there for
  the queries worth the wait.
- The config is per-machine and the cost is hardware-bound: re-decide per
  box rather than copying a config across machines.
- Keep `candidates = 15`: 25+ measured *worse* (the reranker overrides
  correct fusion ranks more often than it rescues deep candidates), and
  cost scales linearly with the count.
`"always"` additionally cold-loads the model on demand (same cost as an
explicit `--rerank`, just automatic). Per-prompt auto-recall injection
never reranks regardless — only explicit `recall` calls are affected.

State lives in the platform-standard directories
(`~/.local/share/mimir`, `~/.config/mimir`, `~/.cache/mimir` on Linux);
set `MIMIR_HOME=<dir>` to put everything under one directory instead.

### Scaling

The exact brute-force vector scan is O(n) in both time and RAM, so recall
latency and memory grow linearly with the embedded-node count. Measured
(CPU, synthetic corpus; `cargo test --release scaling_profile -- --ignored
--nocapture`):

| embedded nodes | warm hybrid recall | matrix RAM |
|---|---|---|
| 50k | ~16 ms | ~75 MB |
| 200k | ~55 ms | ~290 MB |
| 500k | ~130 ms | ~730 MB |

Everyday operations (`get`/`mark`/`edit`, by id) stay flat — microseconds —
at every size. The sweet spot is up to a couple hundred thousand embedded
nodes, where recall is comfortably interactive; beyond that it degrades
gracefully rather than falling over. If a store ever genuinely outgrows
this, the planned path is training-free vector quantization (8–16× less RAM,
exact-ish scan preserved) rather than an approximate index — keeping recall
exact is the point.

## Roadmap

v0.4 shipped the complete original blueprint: memories, docs, code graph,
hybrid + reranked search, self-learning, importers, prebuilt binaries,
and the crates.io release ([mimir-mem](https://crates.io/crates/mimir-mem)).
Daily use has driven everything since: the token-savings layer (outline/peek,
command filters, the optional proxy), opt-in sync with project scoping,
six more languages in the code graph (C#, SQL, C++, Kotlin, Swift, PHP),
code content in recall, opt-in per-prompt auto-recall, remote MCP over HTTP,
agent-side memory hygiene, and concurrency hardening for many simultaneous
sessions. Next: whatever using it daily teaches us — see
[CHANGELOG.md](CHANGELOG.md) for the full history.

## Contributing & security

Bug reports, language adapters, and docs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Release history is in
[CHANGELOG.md](CHANGELOG.md). Thanks to
[@nworks3d](https://github.com/nworks3d) for the remote MCP transport and
the memory-hygiene tools. For security issues, please follow
[SECURITY.md](SECURITY.md) (private disclosure) rather than a public issue.

## Support

Mimir is free and stays free. If it earns a place in your daily loop, you
can support development on [Patreon (MuninWorks)](https://www.patreon.com/MuninWorks)
— patronage covers the servers, domains, and AI tooling behind this and my
other projects.

## License

MIT or Apache-2.0, at your option.
