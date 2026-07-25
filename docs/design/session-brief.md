# Design: session brief — `mimir brief show`

Status: **spec, not implemented** (2026-07-23). Product of a 15-agent design
workflow: 4 code-ground readers + prior-art research → 3 opposed designs →
3-lens judge panel (winner: "Lean Brief", 24/19/10.5) → adversarial review
(16 findings, 5 blockers — all resolved below) → completeness critic.
Competitor context: THOR's 2026-07-22 public benchmark; its session channel
(the work of [@nworks3d](https://github.com/nworks3d), whose THOR fork of
Mimir is the idea source here) is idea-level inspiration only (GPLv3 — no
source read for this design).

## 1. Problem

Mimir's drift-prevention coverage by moment:

| moment | today | gap |
|---|---|---|
| session start / post-compaction | hand-curated single-blob rules pack | no auto-composed coverage; only what the user thought to pin |
| every prompt | `/inject`, silence-first + `fires_when` | none — deliberate design, untouched by this spec |
| moment of action | anchors (file/cmd guards) | none — untouched |

The fresh-session window is the measured weak axis (THOR 2026-07-22 round:
drift 79.7% vs our best 64.4%; our as-deployed hook 7.6% on a corpus whose
prompts never name the constraint). The fix must NOT touch the silence-first
per-prompt floor and must treat token cost as a first-class budget.

## 2. The feature in one paragraph

`mimir brief show` — a SessionStart hook command that prints a hard-capped
(default **150 tokens / 6 lines**) digest of the project's most
drift-preventing memories (gotchas + decisions, ranked by existing signals
only), fires on `startup` and again on `clear`/`compact` (never `resume`),
suppresses per-session repeats, records its cost to the savings ledger, and
shipped **`enabled = false`** until the eval gate (§9) was passed with
citable numbers — **gate passed and default flipped to true 2026-07-25**
(see §9's implementation note). Zero authoring required; zero new
user-facing concepts; no new schema.

## 3. Selection

Two stages, pure SQL + Rust, **no query text and no semantic scan** —
session start is a cold one-shot process; a matrix build (~300 ms at 100k
nodes) is architecturally excluded.

**Stage 1 — candidate fetch** (hits `node_kind_scope`; scans only the
memory subset, ~1k rows at 100k-node scale, sub-ms):

```sql
SELECT id, subkind, tags_text, meta, strength, pinned, access_count,
       last_accessed, created_at, updated_at, project_id
FROM node
WHERE kind = 'memory'
  AND deleted_at IS NULL
  AND superseded_by IS NULL                      -- blocker fix: match recall's default
  AND (project_id = ?1 OR project_id IS NULL)
  AND subkind IN ('gotcha','decision')           -- pin is a BOOST, not an admission path
```

No `LIMIT` (blocker fix: the drafted `ORDER BY updated_at LIMIT 200` would
silently drop old-but-vital candidates; the full memory-subset scan is
already sub-ms, a cap buys nothing).

Rust-side exclusions after the fetch (each set is tiny):
- **Rules-pack overlap** — only when a project was detected: drop the
  `mimir-rules`-tagged pinned node (rules_cmd already prints it), and drop
  any candidate whose normalized body-prefix (~80 chars, case/space-folded)
  appears in the rules-pack body (blocker fix: node-identity dedup alone
  permits content-level double-injection). In **global scope** the
  rules-pack channel never fires, so no rules exclusion applies there
  (serious fix: otherwise such nodes are shown by neither channel).
- **Session suppression** — drop ids in this session's `brief_shown` set
  (§6).
- **Handoff overlap** — when firing on `clear`/`compact` with
  `[hooks] context_guard = "handoff"`, drop the node the handoff restore is
  about to print (same lookup context_guard runs; serious fix for the
  compact-boundary triple-stack).

**Stage 2 — score in Rust, reusing `learn.rs` verbatim** (no new formula
module; `impression_stats` is one batched round-trip, not per-node):

```
score = (pinned ? 2.0 : 1.0)
      * type_prior(subkind)
      * (1 + strength_alpha * ln(1 + effective_strength(node, now)))
      * recency_term(node, now)
      * impression_damp(stats, impression_alpha, now)
      * (node.project_id == current_project ? 1.5 : 1.0)   -- project affinity
```

The project-affinity term is a serious-severity fix: without it a large
personal global pool crowds out a small project's own gotchas.

**Pinned staleness (serious fix):** pin exempts decay everywhere else and
must keep doing so; the brief instead renders the age of any item not
updated in >90 days (`(gotcha, 8mo)`) so a stale-but-pinned rule is at
least visibly old. No decay-math change anywhere.

**`fires_when` is deliberately unused here** (critic-verified as no
invariant violation): those phrases match prompt text, and session start
has no prompt. They stay fully honored on the next per-prompt injection.

**Global-relevance gate (added 2026-07-23 from dogfood ground truth).**
First real-store dogfooding showed a project with no gotchas of its own
receiving three other projects' gotchas as its entire brief; the user
labeled all three irrelevant — *silence beats filler*. So when briefing a
project (`[brief] global_gate = true`, default), each GLOBAL candidate
must share a significant (len > 3) tag/title word with the **project
signature** — the project's title plus its own memories' tags and titles.
A project with no signal of its own admits NO globals: un-judgeable
relevance is treated as irrelevant, per the label. Exception (added
2026-07-23): a PINNED global bypasses the gate — the gate silences
*automatic* noise, and an explicit pin is the user saying "never miss
this"; the heuristic must not overrule that decision. Gate-only bypass:
ranking, budget cap and the subkind admission gate still apply.
Project-scoped candidates are never gated; global-scope briefs are never
gated; `false` disables. Pinned by hermetic fixtures including the exact
labeled silence case.

*Measured and rejected alternative:* cosine against a project centroid of
already-stored embeddings (read, not computed — it would have satisfied
the cold-CLI constraint). Implemented first and measured on the real
102k-node store: bge cosines between unrelated technical memories
compress into ~0.62–0.79 (anisotropy), and the ordering itself mis-ranks
— the labeled-irrelevant colored-3MF global scored near the TOP of its
project's distribution — so no threshold separates. The implementation
lives in git history should a better-separating embedder arrive; do not
re-add it without a new separation measurement.

## 4. Rendering

```
Mimir guards (do not violate):
- GOTCHA [m:ABCDEF]: <body, truncated word-safe to line_chars=100>
- DECISION [m:XX9QRT] (10mo): <body…>
```

- Literal `GOTCHA:`/`DECISION:` prefixes — imperative framing measurably
  outlives symbols/emoji (omission-constraint decay; prior-art finding).
- Compact node ref included (judge graft): ~4 tokens/line buys `mimir get`
  traceability — the user can always answer "why did this appear?".
- Per-item truncation happens **before** the budget-fill loop, so the fill
  decision is only ever "does this whole bounded line fit" — a mid-item cut
  could read as a different (wrong) claim, which the eval penalizes
  asymmetrically.
- Zero candidates ⇒ print nothing at all (matches rules_cmd's silent-empty
  precedent).
- No LLM rewriting of bodies — would need a model call at cold session
  start; rejected (§10).

## 5. Budget

- `[brief] max_tokens` (default **150**; recap fires on `clear`/`compact`
  use `recap_tokens`, default **100** — judge graft: the reset needs a
  reminder, not the full brief).
- `[brief] max_items` (default **6**) binds independently (many tiny
  gotchas must not balloon line count).
- Fill is ranked-order with **best-fit fallback** (judge graft): if the
  next-ranked line doesn't fit, keep scanning lower-ranked shorter lines
  rather than stopping — strictly more value under the same hard cap.
- Worst-case session cost is a stated constant:
  `max_tokens + (max_fires_per_session − 1) × recap_tokens` = 150 + 3×100 =
  **450 tokens** per session, ever. No adaptive growth.
- Every rendered fire records `savings_event` with new `source::BRIEF`,
  `before=0, after=tokens(rendered)` — honestly a **cost**, not a saving.
  Deferred with trigger: `mimir savings`/dashboard grows a "Spent" rollup
  before `enabled` may default to true (otherwise the cost nets invisibly
  against unrelated savings).

## 6. Moments and suppression

`mimir brief show` reads the SessionStart stdin JSON exactly as
`context-guard session-start` does (`session_id`, `source`, `cwd`).
First line of the command: `if !config.brief.enabled { return Ok(()) }` —
this is the kill-switch (critic: there is no hook-uninstall mechanism in
the CLI; config-flag-checked-first is the established rollback pattern).

| `source` | behavior |
|---|---|
| `startup` | fire |
| `clear` / `compact` | fire (recap budget) — the context reset is precisely the drift boundary |
| `resume` | never fire — the transcript still contains any earlier brief |
| `fork` (v2.1.214+; older clients report it as `resume`) | never fire — a forked transcript carries the prior context, same reasoning as `resume` |
| anything else | **fail closed** — never fire (minor fix; matches context_guard's inert-default arm) |

Suppression state — reuses existing tables, no schema change:
- **Fire cap:** `session_state(session_id, 'brief_fire_count')`, checked
  before any query. Default `max_fires_per_session = 4`. **Only fires that
  render ≥1 line consume budget** (blocker fix: a zero-gotcha project must
  not exhaust its budget on empty attempts before its first real capture).
- **Per-item dedup — fire-kind-dependent (revised 2026-07-23 from
  dogfooding):** the original design excluded same-session shown items on
  EVERY fire, which inverted the recap's purpose — after a clear/compact
  the context is wiped, so the top guards shown pre-reset are exactly what
  the agent forgot, yet the rotation served the unseen weak tail instead.
  Now: **startup fires** suppress this session's shown set plus the 6 h
  any-session wall-clock window (unchanged); **recap fires**
  (`clear`/`compact`) suppress only OTHER sessions' recent shows — the
  same session's guards are deliberately re-anchored under the smaller
  `recap_tokens` cap. A client that mints a fresh session_id on compact
  degrades to startup semantics via the wall-clock — safe, just less
  re-anchoring. Keys stay `session_state` `brief_shown:<node_id>` —
  **deliberately NOT the shared `injection_log`** (blocker fix: writing the
  shared ledger would permanently mute the higher-fidelity per-prompt floor
  for exactly the nodes the brief showed). Both stores share the existing
  7-day prune.
- **Score floor (added 2026-07-23 from dogfooding):** a candidate must
  score ≥ `[brief] score_floor` (default 0.75) × the best ELIGIBLE
  candidate's score, measured BEFORE suppression removes anything — so
  rotation can never redefine "best" downward and the brief prefers
  silence over scraping the bottom of the ranked list. Interaction stated
  plainly: the reference score includes the 1.5× project-affinity term,
  so when a project has its own guards, unpinned globals sit below the
  default floor by construction — project-first on purpose; pin a global
  to give it standing. `0.0` disables.

Ground-truth note that shaped this: the per-prompt `injection_log` ledger
is currently **plumbed but dead in the installed hook path** (the hook
never passes `session_id`) — nothing here builds on it being populated.

## 7. Config

```toml
[brief]
enabled = true           # default since 2026-07-25, per the gate in §9; false = kill-switch
max_tokens = 150
recap_tokens = 100
max_items = 6
max_fires_per_session = 4
line_chars = 100
```

All defaults follow house style (`#[serde(default)]` struct; an old
config.toml with no `[brief]` section deserializes to exactly this —
migration is clean by construction, no user action on upgrade).

## 8. Surfaces and degradation

- New CLI: `mimir brief show` (hook entry point) and `mimir brief`
  (human-facing dry-run: prints the same output plus scores/why — the
  explainability surface).
- `mimir init --hooks` installs the SessionStart entry (marker substring
  `mimir brief show`, own array entry — never edits the rules entry; the
  hooks reader confirmed stdouts concatenate).
- **No MCP tool** — same posture as rules pack and context guard. Stated
  consequence (critic): hookless MCP clients (Claude Desktop, Cursor, …)
  do not get the brief, silently. Documented in README, not worked around.
- Rules pack **coexists** in v1 (subsumption was Brief Zero's bet; judges
  scored the migration risk as not worth v1 complexity). Revisit-trigger:
  if post-ship telemetry shows most briefs duplicate rules-pack content.

| state | behavior |
|---|---|
| empty store / new project | silent, zero cost, fire budget preserved |
| global scope (no project) | fires over global memories; rules-exclusion disabled |
| 100k nodes / 400 gotchas | full memory-subset scan, sub-ms; caps bound output |
| daemon down / CPU box | irrelevant — no daemon or model involved anywhere |
| store where everything is pinned | pin is only a boost; caps still bind |

## 9. Eval gate (ship condition — the feature does not default on without this)

The existing harness has no query-less fixture shape (critic), so add
`BriefQuestion { store_fixture, expected_ids, forbidden_ids }` driving the
selector as a pure function (same pattern as `select_injection`).

Fixture families (graft from Broadnet, plus the boilerplate family from
the attack pass): pinned-must-include · project-vs-global crowd-out ·
must-not-surface (superseded / rules-dup / already-shown) · over-budget
truncation order · fresh-capture inclusion · **repeated-exposure**: replay
the corpus across N simulated sessions and measure (real-model run)
whether compliance with a repeated gotcha degrades by session N vs
session 1 — selection staying correct is not enough if the agent goes
boilerplate-blind.

Gates, literally asserted:
1. `brief_recall@budget` beats the rules-pack baseline catch-rate on the
   same fixtures (baseline = what `mimir rules show` alone surfaces).
2. Zero `forbidden_ids` ever rendered (analog of the injected-wrong=0 pin).
3. Budget adherence asserted at every fixture (`tokens(rendered) ≤ cap`).
4. `max_tokens` default is calibrated by the marginal-catch curve (graft):
   rerun at 100/150/250/400; each +100 tokens must buy ≥3 pts of catch or
   the smaller cap wins.
5. Any published number is labeled as scoped to hook-running clients
   (minor fix — this is not a general session-drift improvement claim).

**Implemented 2026-07-25** in `crates/mimir-core/src/eval/brief.rs`
(deterministic, model-free, plain `cargo test`; report via
`cargo test -p mimir-mem-core eval::brief -- --ignored --nocapture`).
All four assertable gates pass: catch@150 = 96.0 pts vs rules-baseline
4.0 pts (gate 1, non-strawman — one hand-promoted guard IS rules-caught);
zero forbidden ever selected or rendered at any cap (gate 2); budget
adherence at every fixture × cap (gate 3); marginal-catch curve
85.3/96.0/96.0/96.0 at 100/150/250/400 calibrates to the shipped 150
default under the pro-rata ≥3 pts/100-tok rule (gate 4). Dogfood labels
are encoded as fixtures (BookForge=silence, ARIA=canonical-gotcha-first,
rotation-to-weak-tail=wrong). The repeated-exposure family's *selection*
half is asserted (replay across simulated sessions: top guard keeps its
seat, recap re-anchors, within-window sibling stays silent); the
*compliance* half (agent boilerplate-blindness by session N) cannot be
asserted in Rust — the report prints each replay brief verbatim as the
corpus for that labeling. **That measurement ran 2026-07-25** as a
preregistered, sealed-protocol experiment (Norn arc; oracle + 12
mechanical guard fixtures in the repo-local eval/compliance/): a Sonnet
subject answering violation-tempting tasks against simulated transcripts
carrying the same brief block 1, 5, and 10 times (recency-controlled —
everything after the last brief block is byte-identical across
conditions). Result, median of 5 preregistered runs on the 9-fixture
train split: compliance 0.889 flat at every exposure level, degradation
−0.111 against a measured 0.111 noise floor — **no boilerplate
blindness detected**. On those numbers plus the assertable gates above,
`enabled` defaulted to true (2026-07-25). Scope caveats carried with the
claim: hook-running clients, Sonnet subject, mechanical fixtures,
flat-prompt transcript simulation.
Numbers above are scoped to hook-running clients (gate 5).

## 10. Explicitly not doing (with triggers where deferred)

- **THOR-style always-on courier / never-empty injection** — violates the
  token axiom; the per-prompt floor stays silence-first.
- **LLM rewriting/summarizing of bodies** — model call at cold start.
- **Semantic/query-driven selection** — no query exists; cold matrix cost.
- **True sliding suppression window with decay** — ranked-exclusion over
  `brief_shown` converges to the same behavior with zero new math.
- **Rules-pack subsumption** — deferred; trigger above (§8).
- **Anchor-match ranking input** (glob changed files into the score) —
  promising judge graft, but new signal plumbing; deferred to v2,
  trigger: v1 eval shows selection recall (not rendering) is the binding
  constraint.
- **Pinned-only candidate pool** (one attacker's redaction concern) —
  **overruled**: the v0.15 secrets guard already gates capture, and the
  brief exposes the same content class the per-prompt `/inject` surface
  already auto-prints; pinned-only would gut zero-authoring value.

## 11. Open questions — RESOLVED 2026-07-23 (verified against the official hooks docs)

1. **Is `session_id` stable across `clear`/`compact`?** **Undocumented
   upstream** — the hooks reference specifies the `session_id` field and
   the `source` values but nowhere states whether clear/compact/resume
   mint or reuse the id, nor the post-compaction session lifecycle. The
   design deliberately does not depend on the answer: stable id ⇒ the
   `brief_shown` keys suppress; minted id ⇒ the 6 h wall-clock fallback
   suppresses. Consequence worth knowing: `max_fires_per_session` is
   per-*session_id*, so a client that mints ids on clear effectively
   resets the fire budget — bounded anyway by the wall-clock exclusion
   (a re-fire can only show not-recently-shown items). Tighten only if
   upstream ever documents the lifecycle.
2. **The `source` enum.** Five documented values: `startup`, `resume`,
   `clear`, `compact`, and `fork` (v2.1.214+; forked sessions reported
   `resume` before that). `fork` lands in the fail-closed arm, which is
   the correct behavior for it (forked transcript carries prior context),
   not an accident — recorded in §6. Whether `resume` can follow a
   compaction is also undocumented; the fail-closed default absorbs any
   surprise sequence.

## 12. Docs / release plan

CHANGELOG `[Unreleased]` Added-entry in house shape (bold title, mechanism
+ default + degradation prose); README section beside context_guard's,
stating opt-in default, the 450-token worst case, and the hookless-client
posture; `docs/benchmarks.md` gains the brief eval reproduction section
when the gate runs. Rollback at every layer: config flag (off by default),
hook entry removable by hand, no schema migration to unwind.
