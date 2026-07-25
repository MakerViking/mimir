# Changelog

All notable changes are documented here. Versions follow semver; the CLI,
the `mimir-mem` crate, and the on-disk schema move together.

## [Unreleased]
### Added
- **`mimir savings` shows a "Spent" rollup.** Cost sources (the session
  brief) have always been excluded from every saved-aggregate so injected
  tokens can never net against real savings — but that made the cost
  invisible. The report now prints a `spent` line (all-time + today +
  fire count, in tokens and dollars, explicitly "not netted above"), the
  JSON output gains a windowed `spent` object, and the dashboard's
  savings panel shows spent-on-injected-context. Silent while nothing
  has ever fired. This was a stated precondition of the brief's
  default-flip (design §5).
- **Prebuilt Windows GPU binary (`windows-x86_64-gpu-webgpu`).** Releases
  now attach a `gpu-webgpu` build of `mimir.exe` with its runtime
  libraries (Dawn + the DirectX shader compiler DLLs) in the zip —
  GPU acceleration on Windows without standing up a Rust/MSVC toolchain,
  and without any CUDA install (WebGPU rides DirectX 12, any GPU vendor).
  Labeled experimental: the upstream WebGPU execution provider is
  experimental and CI runners can't hardware-test it; `device = "auto"`
  falls back to CPU if GPU init fails, so the failure mode is CPU-speed
  operation, not breakage. Prompted by a field report of a from-source
  GPU build colliding with Windows Smart App Control (which blocks
  freshly compiled, unsigned build scripts — and is permanent to disable).
  Hardware-validated on Windows 11 + NVIDIA before first release — thanks
  RacOutlaw for testing (GPU active, and ~9.4 GB of no-longer-needed
  build toolchain reclaimed).
- **Session-brief drift eval (design §9) is implemented** —
  `crates/mimir-core/src/eval/brief.rs`: store-shape fixtures with
  labeled expected/forbidden sets drive brief selection+composition as
  pure functions, asserting the §9 gates in plain `cargo test`
  (zero forbidden guards ever rendered, token-budget adherence at every
  fixture, rules-pack baseline beaten, and the 100/150/250/400
  marginal-catch curve calibrating the default 150-token cap). The
  repeated-exposure replay prints per-session briefs for the remaining
  agent-compliance measurement. Dev-only; the `[brief] enabled` default
  is unchanged until that last measurement lands.
- **Guard anchors are now settable from the MCP `remember` tool**, not just
  the CLI `--anchor` flag. An agent that captures a memory mid-task can
  attach the file/command patterns that should surface it at act-time
  (`anchors: ["deploy.sh"]`), turning the guard from human-curated-only into
  something the agent populates as it works. Same sanitize/cap rules as the
  CLI (up to 8 patterns, 200 chars each, optional `file:` prefix); the
  0.14.0 CLI-only limitation is lifted.

### Changed
- **The session brief is now ON by default** (`[brief] enabled = true`).
  The 0.16.0 design gated this flip on an eval, and the gate has been
  passed with citable numbers (scoped to hook-running clients): selection
  catches 96.0 pts of labeled must-know guards at the default 150-token
  cap vs 4.0 pts for a hand-written rules pack alone; the marginal-catch
  curve (85.3/96.0/96.0/96.0 at 100/150/250/400 tokens) calibrates
  exactly the shipped cap; zero forbidden guards ever rendered across the
  fixture families; and a preregistered repeated-exposure experiment
  (sealed protocol, mechanical scoring) found no boilerplate blindness —
  subject compliance held at 0.889 whether the same brief had been seen
  1, 5, or 10 times. `enabled = false` remains the one-line kill-switch.
  **Upgrade note:** configs written by `mimir init` before this release
  contain an explicit `enabled = false` and keep it — flip the existing
  key in place (appending a second `[brief]` table is a TOML parse
  error).
- **Session brief: recaps re-anchor instead of rotating.** A
  `clear`/`compact` fire no longer excludes what this session already
  showed — the context was just wiped, so the top guards shown before the
  reset are exactly what the agent forgot; re-showing them (under the
  smaller `recap_tokens` cap) is the point of the fire. First-day
  dogfooding caught the old behavior serving the unseen weak tail at the
  precise moment the #1 guard mattered most. Cross-session dedup (the 6 h
  wall-clock window) still applies, and startup fires keep full
  suppression.
- **Session brief: relative score floor — silence over scraped bottom.**
  Candidates must score ≥ `[brief] score_floor` (default 0.75) of the
  best eligible candidate, measured before suppression — so once the
  strong guards are suppressed or a store is thin, the brief goes silent
  rather than serving weak-tail items as "best available". The reference
  includes the project-affinity term, making briefs project-first by
  construction: an unpinned cross-project memory yields to a project's
  own guards, and a pinned one competes normally. `0.0` disables.
- **Session brief: a pinned global bypasses the relevance gate** — the
  gate silences automatic noise; an explicit pin is the user saying
  "never miss this" and a lexical heuristic must not overrule it.

## [0.16.0] - 2026-07-23
### Added
- **Session brief: global-relevance gate — silence beats irrelevant
  filler.** When briefing a project, a cross-project (global) memory now
  has to be relevant to *that* project to spend brief budget: it must
  share a significant tag/title word with the project's signature (its
  title plus its own memories' tags and titles; `[brief] global_gate`,
  default on, `false` disables). A project with no memories of its own
  briefs nothing rather than other projects' gotchas. Project-scoped
  memories are never gated. Deliberately lexical: a stored-embedding
  centroid-cosine gate was implemented and measured on a real 102k-node
  store first and rejected — embedding cosines between unrelated
  technical memories compress into a ~0.62–0.79 band and mis-rank
  labeled-irrelevant items, so no threshold separates. Born from
  first-day dogfooding, where an unrelated project's entire brief was
  three other projects' gotchas.
- **Session brief — capped digest of drift-preventing memories at session
  start.** `mimir brief show` (a SessionStart hook entry `mimir init
  --hooks` now installs, inert by default) injects the current scope's
  top gotchas and decisions — ranked by existing signals only (pin,
  decayed strength, recency, same-project affinity; no query, no model,
  cheap SQL suited to a cold one-shot process) — as one imperative
  `- GOTCHA [m:ref]: …` line each, hard-capped at `[brief] max_tokens`
  (150) and `max_items` (6). Fires at `startup` and again after
  `clear`/`compact` with a smaller `recap_tokens` (100) cap of
  not-yet-shown items, never on `resume` (fail-closed on unknown
  sources), at most `max_fires_per_session` (4) *rendered* fires: worst
  case 450 tokens per session by construction, each fire recorded in the
  savings ledger as spend (`source = "brief"`, honestly a cost — it never
  counts as savings). Excluded from selection: superseded and deleted
  memories, anything content-duplicated in the project's rules pack,
  items already briefed this session (plus a 6-hour any-session
  wall-clock fallback for clients that mint fresh session ids), and — in
  handoff mode — the memory the context guard is restoring on the same
  event. `mimir brief` previews the exact output with per-candidate
  scores. **Default `enabled = false`**: the per-prompt silence-first
  `/inject` path is untouched, and the brief stays opt-in until its drift
  eval gate is passed with citable numbers (design + gate:
  docs/design/session-brief.md). Idea credit:
  [@nworks3d](https://github.com/nworks3d)'s THOR fork of Mimir — its
  session-boundary briefing channel is what this answers; written
  clean-room from that one-sentence concept (THOR is GPLv3; its source was
  never read). Thanks!

### Changed
- **`contrib/mimir-daemon.service` recycles the daemon daily**
  (`RuntimeMaxSec=86400`, `Restart=always`). On GPU builds the ONNX
  WebGPU/Dawn runtime allocates VRAM at inference time into shape-keyed
  pools that are never freed, and the daemon — as the one process doing
  GPU inference — accumulates them without bound (~1.2 GiB/day measured
  under normal agent load). A daily restart bounds it; sessions are
  unaffected by design (a mid-call daemon loss falls back to CPU-local
  inference in the same call and delegation auto-resumes). Existing
  installs: re-copy the unit and `systemctl --user daemon-reload`.

## [0.15.0] - 2026-07-20
### Added
- **Shared inference daemon — zero-VRAM sessions on GPU builds.** `mimir
  daemon` (and `mimir mcp --http`) now serves three inference-delegation
  endpoints on its warm engine: `GET /inference` (which models it holds),
  `POST /embed` (L2-normalized vectors), `POST /rerank` (cross-encoder
  scores). With the new `[daemon] inference = "auto"` (default), every stdio
  MCP session and background engine loads its embedder **CPU-only** —
  measured *faster* than GPU for the batch-1 query embeds sessions actually
  do — and routes bulk embedding and reranking to the one daemon process, so
  N concurrent agent sessions cost one model's worth of GPU memory instead
  of N (previously ~0.7 GiB VRAM per session and growing). Fully graceful:
  daemon absent at session start ⇒ exactly the old local behavior (including
  `[rerank] auto = "warm"` eager-loading a local reranker); daemon dies
  mid-session ⇒ embeds fall back to the session's CPU embedder in the same
  call, reranks degrade to fused order, and delegation resumes on its own; a
  bearer-token rejection or a daemon serving different model names disables
  delegation for that session (mismatched vectors would poison the store).
  `[daemon] inference = "off"` restores the fully-local pre-daemon behavior.
- **Secrets guard on `remember`** — capture now refuses text, tags, or
  `fires_when` phrases that look like they contain an actual credential: private-key blocks, AWS access
  key ids, GitHub tokens/PATs, Slack tokens, JWTs. High-precision structural
  patterns only (no entropy heuristics), so ordinary prose *about* keys,
  tokens and passwords is never blocked; the refusal message names what was
  matched so an agent can redact and retry. Applies to the CLI `remember`
  command and the MCP `remember` tool — and `mimir edit`, which rewrites the
  same fields — while `mimir import` and sync replication are deliberately
  unguarded (restoring your own backup is not a capture). Matching runs once
  per write via lazily-compiled regexes; recall and inject hot paths are
  untouched.
- **MCP schema diet** — the tool and parameter descriptions the MCP server
  ships to every client session were compressed ~34% (4.5k → 3.0k chars)
  without renaming or behaviorally changing any tool or parameter; norms that
  were repeated per-tool now live only in the server instructions.
- **`mimir reproject <ref> --project <name> | --global`** — move a memory to
  another project, or to global scope, after the fact. A memory's project is
  fixed at `remember` time from the session's working directory, so a session
  started in the wrong directory files it under the wrong project with no way
  to correct it (`edit` changes text/title/type/tags, not the project). The
  target is resolved by name, preferring a locally-bound project over a synced
  shadow of the same name; the change bumps `updated_at` so it rides the
  existing last-write-wins sync when the target is sync-enabled — moving into
  a plain local project prints a note that the memory will stop syncing.
  Only memories can be reprojected
  (files/symbols/doc-chunks derive their project from their source). Thanks to
  @nworks3d for the report and reference implementation (#12).
- **Optional bearer-token auth for `mimir mcp --http`** — `--http-token
  <token>` or the `MIMIR_HTTP_TOKEN` env var (preferred, so the token stays
  out of process lists; the flag wins if both are set) requires an
  `Authorization: Bearer <token>` header on the HTTP transport, validated with
  a constant-time compare. Defense-in-depth on top of the loopback-bind gate:
  a misconfigured fronting proxy or an exposed port no longer means instant
  full read/write access to the store. No token configured = unchanged
  behavior. The non-loopback bind refusal stays regardless. `mimir daemon`
  (the alias for `mimir mcp --http`) honors the env var too, and an empty
  token — flag or env — is treated as unset rather than enabling an
  unsatisfiable gate (#11).
- **`mimir doctor --check` watchdog mode** — silent and exit 0 when healthy;
  prints only the failing checks (stderr) and exits non-zero otherwise, so a
  timer/cron can alert on store breakage with zero cost while things are fine.
  Runs the hard checks only (db open, `PRAGMA integrity_check`, FTS5 query),
  skipping the informational gpu/model/daemon lines and the daemon probe's
  network wait.
- **FTS5 index consistency check in `doctor`** (both modes). `node_fts` is an
  external-content table, so `PRAGMA integrity_check` can pass while the
  search index has silently drifted from the `node` table — the state where
  recall returns nothing. Doctor now runs FTS5's two-argument
  `('integrity-check', 1)` command, which compares the index against the
  content table (the one-argument form passes on drift), and on failure
  prints the one-line rebuild remedy.
- **`contrib/mimir-watchdog.{service,timer}`** — hourly systemd user units
  wiring `doctor --check` to a desktop notification on failure.

### Fixed
- **GPU memory leak in long-lived sessions** — `embed_pending` loaded the
  embedding model *before* checking whether anything needed embedding, so
  every speculative background call (the auto-sync loop, every
  `interval_mins`) created and tore down a whole GPU device just to discover
  there was no work — and repeated Vulkan/Dawn device create/teardown leaks
  driver memory (observed: `mimir mcp` sessions growing from ~0.7 to ~6 GiB
  VRAM over a day). A cheap `EXISTS` probe now gates the model load, and
  background engines run their embedder on CPU, so GPU device churn is gone
  entirely.
- **Sync: client push watermark could be poisoned by another peer's clock
  skew.** `sync push` advanced the client's `last_push` cursor from the hub's
  post-apply *global* high-watermark instead of the local batch it actually
  sent. If any peer's clock ran ahead (or a backdated write inflated the hub
  watermark), a client's own subsequent edits could silently fail the
  changes-since filter and never be pushed again — no error, permanent skip.
  `last_push` now advances from the local batch watermark (same clock domain
  as the query it gates); the wire format is unchanged. Additionally, both
  sync cursors (`last_push`, `last_pull`) are clamped to the local wall
  clock, so a single future-dated row — your own past clock skew, or a skewed
  peer's row relayed by the hub — can no longer pin a cursor into the future
  and silently stop that client from sending or receiving later changes; the
  worst case is now the skewed row re-transferring each round, which the
  idempotent apply makes free. Regression tests cover the skew scenario,
  push-retry idempotency, and both cursor-clamp failure modes.
- **Sync: batch apply is now transactional.** `apply_changes` wraps the
  node+edge apply loop in a single immediate transaction, so a crash
  mid-batch can no longer leave a partially applied sync (previously harmless
  only because retries are idempotent; now it can't happen at all).

## [0.14.0] - 2026-07-10
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
- **Auto-rerank, gated by warmth: `[rerank] auto = "warm" | "always" | "off"`**
  (default `"off"`). Previously the cross-encoder only ever ran on an
  explicit `recall --rerank`, cold-loading the model every time from a
  one-shot CLI process. With `auto = "warm"`, `mimir mcp` and `mimir mcp
  --http`'s shared `/inject` engine eager-load the reranker at startup, so
  once warm, ordinary recalls through the daemon get reranked — a one-shot
  CLI process still never eats a cold ONNX load just because this is on,
  since `"warm"` only fires when the model is already resident. Per-session
  HTTP `/mcp` engines deliberately don't eager-load (memory would multiply
  per concurrent session). Also gated per-query, independent of warmth: a
  store under ~1k embeddable nodes, or a single-token/exact-identifier-shaped
  query, skips reranking — a cross-encoder only adds cost (and can reshuffle
  a correct exact match) in either case. Ships **off by default** on the
  branch's own evidence standard: the cost is measured (~84 ms per candidate
  on realistic code chunks → ~1.3 s per warm recall at the default 15
  candidates) while the benefit is not — the retrieval eval doesn't exercise
  reranking yet, and sandbox checks showed it can worsen some queries.
  Revisit the default once rerank is wired into the eval harness and shows a
  measured win.
- **Third RRF leg: exact identifier/entity match.** Queries shaped like
  code lookups (`MatrixCache::ensure`, `add_saturating`) were getting
  fuzzed apart by the embedding leg and diluted by the FTS leg's
  OR-of-every-token match (which lets an unrelated doc rank via a
  throwaway word like "work" in "how does MatrixCache::ensure work").
  `search_hybrid` now extracts identifier-shaped fragments (`::` paths,
  `snake_case`, `camelCase`/`PascalCase`, dotted paths) from the query and
  runs a cheap, capped exact-phrase FTS match on just those, fused in as a
  third RRF leg. Empty (zero cost) on ordinary prose queries; acronym-led
  proper nouns like `SQLite` or `GitHub` are deliberately excluded (an
  upper-to-lower letter run, not camelCase's defining lower-to-upper
  transition) so they don't misfire as identifiers.
- **Auto-recall enrichment: the hook now looks at your working tree.**
  `mimir-recall.sh` runs a capped `git diff --name-only HEAD` and passes
  the changed files' stems (up to 8) alongside the prompt — on both the
  warm `/inject?enrich=` param and the cold `recall-inject --enrich` path.
  It can extend a real prompt/memory overlap (e.g. a short prompt that
  wouldn't otherwise clear the floor on its own) but can never
  single-handedly clear it: at least one raw-prompt-token overlap, or
  confirmed embedding-leg agreement in a store large enough for that
  agreement to mean something, must independently hold. Without that
  guard, any prompt on a branch touching file `X` would auto-inject every
  gotcha whose title happens to mention `X` — sandbox-verified against
  exactly that failure mode (a 2-memory store made "agreement" trivially
  true for every candidate) before landing.
- **Four more languages for the code graph and `mimir code add`: C++,
  Kotlin, Swift, PHP.** Same tree-sitter-backed symbol/call/import
  extraction as the existing languages — classes, methods (nested and
  out-of-line, e.g. C++'s `Type::method`), calls, and imports all resolve
  and qualify correctly. Two notable extension-mapping/behavior decisions:
  `.h` now parses as C++, not C — the C++ grammar reads C-style headers
  acceptably, but the reverse isn't true (plain C can't parse `class`,
  `namespace`, or templates at all), and `.c` still always resolves to C.
  A PHP file without an opening `<?php` tag parses as plain HTML/text and
  yields zero symbols by design, matching real PHP semantics. Also fixes a
  gap found while building this: PHP static calls (`Type::method()`, the
  `scoped_call_expression` node) were previously unrecognized as call
  edges.
- **`granite-embedding-small-r2` is now a selectable `embedding.model`**
  (`ibm-granite/granite-embedding-small-english-r2`, via the
  `onnx-community` ONNX export) — evaluated as an A/B candidate against
  the default `bge-small-en-v1.5` (same 384 dims, so the schema doesn't
  change, but see the caveat below). It isn't in fastembed's built-in
  registry, so selecting it downloads the ONNX graph + tokenizer files
  directly from its HF repo (via `hf-hub`, same cache dir/marker-file gate
  as registry models) and loads via fastembed's user-defined-model path.
  **The default is unchanged** — this is opt-in only, by setting
  `embedding.model = "granite-embedding-small-r2"` in `mimir.toml` then
  `mimir init` (or `mimir embed --fetch`). Verdict from running both
  through Mimir's own retrieval eval (not IBM's CoIR/BEIR numbers, which
  don't transfer 1:1 to our corpus/query shapes): precision and recall are
  identical to bge-small across every category and question set; MRR is
  marginally higher (0.858 → 0.866 overall) and per-embed latency on short
  text is roughly 3x lower in a release build (~20ms → ~6ms), but the
  preventer confusion-matrix eval (the metric that actually matters for
  drift prevention) is byte-for-byte identical between the two models. Not
  a case for churning the default: the eval gain is real but small, and
  switching a running store's default model forces re-embedding
  everything in it (same dimensionality does not imply a compatible
  embedding space — nearest-neighbor distances between the two models'
  vectors aren't meaningfully comparable). Worth having selectable for
  anyone who wants the latency win or wants to re-run the comparison
  themselves; not worth flipping unilaterally.
- **`[hooks] cold_mode = "fast" | "full"`** (default `"fast"`) governs only
  the cold `mimir recall-inject` CLI fallback (never the warm `/inject`
  endpoint or a normal `mimir recall`, both of which always do full hybrid
  search). `"fast"` skips the embedder entirely — BM25 + identifier legs
  only, no ONNX/matrix load. Measured cold, release build, bge-small-en
  already cached, 3-memory store: ~5-6ms end to end vs ~230-240ms for
  `"full"` on the identical prompt/store, both on a firing and a silent
  prompt. The relevance floor (`inject::clears_floor`) already degrades
  cleanly with no vector leg — same path a model-less machine takes
  today — so `"fast"` narrows which real matches clear the floor cold
  (semantic-only matches won't fire) without weakening the
  silence-beats-wrong-injection contract itself. Set `"full"` to restore
  the pre-existing cold-path behavior.
- **Shown-but-never-opened negative prior: `scoring.impression_alpha`**
  (default `0.0`, off). A node repeatedly surfaced in results but never
  opened gets a bounded multiplicative demotion (floor ~0.7x, decaying
  back toward 1.0 over ~30 days) once it's been shown at least 10 times —
  opened nodes are always exempt, and impression counts are fetched in one
  batched query alongside the existing candidate fetch, so the score site
  stays a single round trip regardless of alpha. Shipped off by default:
  the eval fixtures have no impression history to measure this against, so
  turning it on by default would be an unmeasured ranking change — exactly
  how a prior recency regression made it through review undetected. Opt in
  via `config.toml` once you have real usage history to tune against.
- **`mimir daemon`**: a thin, discoverable alias for `mimir mcp --http <addr>`
  that resolves `<addr>` from `[hooks] inject_url` (scheme and `/inject`
  path stripped), so the same config key drives both the auto-recall hook's
  target and the daemon's bind address — no separate "which port did I
  configure" step. Prints the resolved warm URL on startup. `mimir doctor`
  gained a matching informational check: a ~1s `GET` against the configured
  `inject_url` reports warm/cold, never failing `doctor`'s exit code (same
  precedent as the existing "model" check) — run cold, it just tells you to
  `mimir daemon`. A sample systemd user unit ships at
  [contrib/mimir-daemon.service](contrib/mimir-daemon.service). Deliberately
  scoped to an alias plus a doctor check: no auto-spawn, no process
  supervision — systemd (or your init of choice) already does that job.
- **`[learn] event_retention_days`** (default `180`, `0` = keep forever)
  bounds the previously-unbounded `recall_event` ledger (every recall logs
  an impression). Pruning is a single batched `DELETE ... WHERE at <
  cutoff` — backed by a new `recall_event(at)` index (migration v7), since
  the existing `(node_id, at)` composite index doesn't serve a bare `at`
  predicate — invoked once at daemon startup and on the same idle
  WAL-checkpoint cadence the `mcp`/`serve`/`proxy` daemons already run
  (~every 2 min). Measured on a seeded 5,500-row `recall_event` table
  (5,000 rows aged past the retention window, 500 within it): one daemon
  startup prunes it to exactly 500 rows, no more, no less. Configuring
  below 60 days logs a warning and clamps up to 60 rather than silently
  truncating `scoring.impression_alpha`'s 30-day decay stats to fewer than
  ~2 half-lives of history.
- **`jina-reranker-v1-turbo-en-int8` is now a selectable `rerank.model`**
  (same `jinaai/jina-reranker-v1-turbo-en` repo, ~38 MB int8 ONNX export vs
  the default's ~151 MB fp32) — evaluated as a latency candidate against the
  fp32 default. Not in fastembed's registry (it hardcodes the fp32 file for
  this repo), so selecting it downloads `onnx/model_int8.onnx` directly via
  `hf-hub` and loads through fastembed's user-defined-reranker path, same
  marker-gate convention as the embedding side. **The default is unchanged**
  — opt-in via `rerank.model = "jina-reranker-v1-turbo-en-int8"` in
  `mimir.toml`. Measured on this crate's own source (50 candidate code
  chunks, release build, warm session, 20-run median): int8 is only
  ~1.2x faster per rerank call (~4.2-4.7s → ~3.6-3.7s at this candidate
  count), well short of the ~2-3x CPU headroom int8 quantization typically
  buys — worth knowing before assuming a bigger win. Ordering agreement
  across 10 diverse queries: the #1 (most-shown) result was identical
  between fp32 and int8 in all 10; the full top-3 set matched in 8/10, with
  one case where int8 demoted a genuinely on-topic chunk to the weakest
  (#3) slot in favor of a less relevant one. Given the modest speedup and
  that one real (if minor) quality cost, this isn't a case for flipping the
  default either — shipped selectable for anyone who wants the CPU headroom
  and can tolerate mild top-3 reshuffling.
- **Opt-in context-window guard: `mimir init --hooks --context-guard pause`**
  (or `handoff`), default `"off"`. Three new hook entries
  (`UserPromptSubmit`, `PreCompact`, `SessionStart`) estimate how full the
  context window is from the transcript file's *size* — cheap, no JSONL
  parsing on every prompt — and act once the estimate crosses
  `[hooks] context_guard_threshold_pct` (default 45% of
  `context_window_tokens`, default 200,000; both tunable, e.g. for a
  1M-context model). `"pause"` nudges toward a deliberate `/clear`/
  `/compact` at most once per +10 percentage-point band, and blocks an
  *automatic* compact attempted over threshold (`PreCompact`
  `decision:block`) — a user's own `/compact` is never blocked. `"handoff"`
  additionally instructs the agent to save a `session-handoff`-tagged
  memory before the clear, then auto-restores the latest one on the next
  `SessionStart` after a clear/compact, so the new session isn't starting
  cold. Clean-room design, byte-size estimation and message wording are
  Mimir-native. New CLI surface: `mimir context-guard prompt|precompact|
  session-start|pretool` — hook-event entry points, not meant to be run by
  hand. `--context-guard` implies `--hooks`; `off` (the default) installs
  nothing new and leaves the previous hook set byte-identical.
- **Guard anchors: `mimir remember --anchor "<pattern>"`** (repeatable, up
  to 8 per memory). Surfaces a memory on `PreToolUse` the moment a matching
  file is edited/written, or a matching command mentions it — before the
  tool call happens, no prompt needed. The pattern matches on path suffix
  (`"store.rs"` matches regardless of how deep it lives), tried against
  every `/`-delimited suffix and, for Bash, every whitespace/quote-
  delimited token of the command. Deduped per session via the same
  `injection_log` ledger auto-recall's `--session` dedup uses, so a match
  fires once per session, not once per matching tool call. `mimir init
  --hooks` installs the `PreToolUse` matcher unconditionally — it's inert
  until a memory actually declares an anchor. Clean-room design: the
  functional idea ("a memory can declare which files/commands should
  surface it") is THOR-inspired, but this module was written from that
  one-sentence spec only — THOR's (GPLv3) implementation was never read.
- **`mimir remember --fires-when "<phrase>"`** (repeatable) declares
  trigger phrase(s) that bypass auto-recall's inferred-relevance floor on a
  close match — for facts that are easy for BM25/vector search to
  under-rank but should always fire when that phrase comes up. Not exposed
  on the MCP `remember` tool (CLI-only, same reasoning as `--anchor`).
  `mimir recall-inject --session <id>` threads a session id through so
  auto-recall and guard anchors share one per-session "already shown" list
  instead of two independent ones.

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
- **The eval harness now scores the actual `inject::select_injection`
  decision**, not just retrieval rank: a confusion table
  (injected-correct/wrong, silent-correct/wrong) per fixture, including
  negative fixtures (a hit exists but must stay silent) and enriched-query
  variants. `cargo test -p mimir-mem-core eval::tests::eval_inject_baseline_report
  -- --ignored --nocapture` prints it — the number to compare across floor
  changes; the product rule enforced on every change is that
  injected-wrong must never rise, even if the change also fixes an
  intended case.
- **Real-model eval is now parameterized by model**: `MIMIR_EVAL_MODEL`
  overrides the configured default for `run_real_model` /
  `run_inject_eval_real_model` (dev-only, unset → identical behavior to
  before). This is what made the granite-vs-bge-small A/B above
  measurable without touching `mimir.toml` per run. New direct dependency
  `hf-hub` (already present transitively via fastembed's own default
  features, so this adds no new transitive surface) does the
  download+cache for models outside fastembed's registry.

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
