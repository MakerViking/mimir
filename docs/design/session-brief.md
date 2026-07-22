# Design: session brief — `mimir brief show`

Status: **spec, not implemented** (2026-07-23). Product of a 15-agent design
workflow: 4 code-ground readers + prior-art research → 3 opposed designs →
3-lens judge panel (winner: "Lean Brief", 24/19/10.5) → adversarial review
(16 findings, 5 blockers — all resolved below) → completeness critic.
Competitor context: THOR's 2026-07-22 public benchmark; its session channel
is idea-level inspiration only (GPLv3 — no source read for this design).

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
ships **`enabled = false`** until the eval gate (§9) is passed with citable
numbers. Zero authoring required; zero new user-facing concepts; no new
schema.

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
| anything else | **fail closed** — never fire (minor fix; matches context_guard's inert-default arm) |

Suppression state — reuses existing tables, no schema change:
- **Fire cap:** `session_state(session_id, 'brief_fire_count')`, checked
  before any query. Default `max_fires_per_session = 4`. **Only fires that
  render ≥1 line consume budget** (blocker fix: a zero-gotcha project must
  not exhaust its budget on empty attempts before its first real capture).
- **Per-item dedup:** `session_state` keys `brief_shown:<node_id>` —
  **deliberately NOT the shared `injection_log`** (blocker fix: writing the
  shared ledger would permanently mute the higher-fidelity per-prompt floor
  for exactly the nodes the brief showed; the floor's own relevance rules
  must keep gating re-display independently). Both stores share the
  existing 7-day prune.
- **Wall-clock fallback** (serious fix): also drop any candidate whose
  `brief_shown` key exists for *any* session in the last 6 h — covers the
  unverified case where `resume` (or a compact) mints a fresh `session_id`.

Ground-truth note that shaped this: the per-prompt `injection_log` ledger
is currently **plumbed but dead in the installed hook path** (the hook
never passes `session_id`) — nothing here builds on it being populated.

## 7. Config

```toml
[brief]
enabled = false          # flips to true only per the gate in §9
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

## 11. Open questions (with recommended answers)

1. **Is `session_id` stable across `clear`/`compact`?** The whole
   suppression scheme prefers it; the 6 h wall-clock fallback (§6) makes
   the answer non-fatal either way. Verify against live hooks during
   implementation, before trusting `max_fires_per_session` semantics.
2. **`resume` after compaction on some clients?** Fail-closed table (§6)
   already covers unknown values; verify the enum against current hook
   docs at implementation time.

## 12. Docs / release plan

CHANGELOG `[Unreleased]` Added-entry in house shape (bold title, mechanism
+ default + degradation prose); README section beside context_guard's,
stating opt-in default, the 450-token worst case, and the hookless-client
posture; `docs/benchmarks.md` gains the brief eval reproduction section
when the gate runs. Rollback at every layer: config flag (off by default),
hook entry removable by hand, no schema migration to unwind.
