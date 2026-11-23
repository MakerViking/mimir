# State machines in Mimir

**Status:** describes the code as of the "explicit state machines" change.
**Audience:** anyone changing how something in the store is created, retired,
synced or shown.

A state machine is *implicit* when no single type names its states: a
lifecycle spread over nullable columns, a mode kept as a string and compared
in several files, a flag in `meta` JSON, an ordering that only works because
every caller happens to follow it. Each reader then re-derives the state for
itself, and two readers eventually disagree. Every bug this document's change
fixed had that shape.

The rule the code now follows:

> **Name the state once, derive it in one place, and have every surface print
> that name.**

`crates/mimir-cli/tests/one_definition.rs` holds the code to this: it reads
the production source and fails when a state is spelled outside the files
that own it. When you move a definition, move its rule there too.

This document is the map: for each state machine, where its one definition
lives. If you are about to compare a column to `NULL`, a config string to a
literal, or a `meta` key to a value, the thing you want is probably in one of
the tables below. The last section lists what is still implicit, and why.

## 1. Memories

| State machine | States | The one definition | Who reads it |
|---|---|---|---|
| Memory lifecycle | live, superseded, expired, archived, forgotten | `lifecycle::Lifecycle::of`; SQL twins `lifecycle::ARCHIVED_SQL`, `lifecycle::expired_sql`; counts in `lifecycle::counts` | compact line and `get` (`format.rs`), `--json`, brief, guard anchors, handoff restore, duplicate check, consolidation, review, sync |
| "The current version of this" | itself, its live replacement, or nothing | `lifecycle::resolve_current` (follows `superseded_by`, bounded) | guard anchors, handoff restore, summary coverage |
| Supersession | not superseded / superseded since a date | `store::supersede` (column + `supersedes` edge + ledger row, atomic, guarded); date from `audit::superseded_at` | CLI, MCP, review, consolidation; `replicate::merge_supersession` for the arriving case |
| Forget | soft, hard, archived-then-forgotten | `store::forget` | CLI, MCP |
| Capture | stored / refused (duplicate, forgotten) / rejected | `capture::capture` in the CLI crate: validate, secret guard, link target, one transaction | `mimir remember`, MCP `remember` |
| Secret guard | passed / refused and recorded | `capture::refuse_secrets` | capture, `edit`, `anchor`, `rules set`, `docs note` |
| Memory type | the six types; none means note | `Node::memory_type`, `Node::type_label`, `MemoryType::is_preventer` | decay, brief, injection, labels, `list -t` |
| Validity interval | undeclared, open, not yet, closed | `model::Validity`, `Node::validity` | `format.rs` |
| Review finding | open, decided, resolved | `review` module doc; `review::resolved_outside` decides whether a finding's subject still stands | `mimir review`, `doctor` |
| Edge relation | authored, retirement, extracted, observed | `Rel::origin` | `link` guards and help, `mimir report`, graph pruning and tombstoning |
| Recall event | shown, opened, useful, not useful | `model::RecallEvent` | learning, shadow ranking, dashboard, report |
| Session handoff | pending, restored; written in one checkout or in none | `context_guard::current_handoff_in`, `pending_handoff_in`, `handoff_being_restored_in`, `mark_handoff_restored`, `rearm_handoff`; the mark is `meta.restored_at`, the checkout `meta.handoff_checkout`, the rule `handoff_is_for` | SessionStart hook, brief exclusion, capture's re-arm, `get` |

Transitions between lifecycle states are in the module doc of
`lifecycle.rs`. Two rules there are easy to break:

- A WHERE clause can say "not deleted" and "not superseded". It cannot say
  "not expired" without the clock; use `Lifecycle::of` on the row, or
  `expired_sql(now)`.
- `store::set_superseded` is the low-level column write and is kept only as
  public API. Everything in the repository calls `store::supersede`.

## 2. Sync

| State machine | States | The one definition |
|---|---|---|
| Does this memory sync | in the set / not | `replicate::syncable_sql` (SQL) and `replicate::project_is_syncable` (Rust), held together by a test |
| Which version of a memory wins | the greater `Standing` | `replicate::Standing`: when the text was decided, then forgotten over live, then the last write, then a digest. A total order, and it has to stay one; `random_histories_on_three_stores_always_converge` holds it to that. A local decay archive is not a write (`Held::record`) |
| When a memory's text was decided | recorded and about this text / not recorded (as old as the last write) | `text_clock::read`; kept by `text_clock::keep`, which every write that leaves the text alone calls before it moves `updated_at` |
| What a record carries | live row, deliberate tombstone | `replicate::keep_archival_local`: an archived row travels as the live row it was, and an archived tombstone from an older peer deletes nothing |
| What a hub relays | written since / arrived since | `replicate::relay_since` (hub pulls) next to `replicate::changes_since` (client pushes). Arrival is the `mutation` row with actor `sync` that `apply_changes` writes for every row it applies |
| Cursor | last push, last pull | `sync::cursor_advance` (clamped to the local clock, minus the grace window); `--full` ignores it |
| Sync mode | off, file, server | `config::SyncMode` for the config, `sync::Mode::of` for the commands |
| Token source | env, stored, missing | `sync::TokenSource::find` |

What does *not* replicate, by design: decay archival, strength and access
counts, recall events, embeddings, `co_recalled` edges, and the uid of a
replacement that is not itself in the synced set.

## 3. Retrieval, embeddings, models

| State machine | States | The one definition |
|---|---|---|
| Did a vector leg run | not run / ran with these ids | `search::vector_leg(&legs)`; an empty or unusable leg is "not run" |
| Embedding row | missing, stale, current | `embed::pending_where` (the selection), `embed::coverage` (the count) |
| Model slot | unloaded, loaded, failed (with a retry time) | `embed::ModelSlot`, `embed::retry_after` |
| Rerank outcome | not asked, nothing to rerank, applied, unavailable | `RerankOutcome`; its `note()` is the sentence both surfaces print |
| Injection floor | clears / does not | `inject::clears_floor`; a session scan stops at the first hit that fails it |
| Miss ledger feeder | observe / do not | `misses::should_observe` |
| Collection kind | docs, code | `index::CollectionKind` |
| What a search can reach | one scope for every kind / memories in one scope, code and docs in another | `search::Reach` (`sql` for the lexical legs, `admits` for the vector leg, held together by `reach_has_one_answer_in_sql_and_rust`); the `*_within` functions take it, the others pass `Reach::uniform` |

## 4. Projects, scope, config, hooks

| State machine | States | The one definition |
|---|---|---|
| Scope when there is no project | every project (a question) / global only (something volunteered) | `scope::query_scope`, `scope::push_scope` |
| `.mimir` marker | absent, parsed, unparseable | `scope::MarkerState` |
| Which checkout a root is | standalone / linked worktree of a home | `scope::Checkout::of` (files only: the `.git` file, its `gitdir`, that directory's `commondir`) |
| Whose code graph, whose memories | own (one row) / home (a linked worktree files under its main checkout) / own key (a linked worktree whose `.mimir` id the main checkout does not carry keeps its own) | `Mimir::projects_for_cwd` → `ProjectContext` (`filing`); `projects_for_cwd_moving_memories_home` is the one resolution that may move memories, called by memory commands and MCP start-up only. `detect_project` and `project_for_cwd` read the checkout, `memory_project_for_cwd` the memory project; the row records its home in `store::WORKTREE_OF` |
| Reach of a question that can be answered by a memory, a doc or code | uniform / split | `scope::query_reach` (`query_scope` of the memory project and of the checkout project); `ProjectContext::query_reach`. Used by CLI and MCP `recall` |
| Memories under a worktree's own row | none / to move (written since the worktree was made: `scope::worktree_created_at`) / kept (older, another checkout's) | `store::reproject_all`, from `projects_for_cwd_moving_memories_home`; reported by `mimir status` |
| Rules pack in force | none / the one pack / the newest text of several | `rules_cmd::current_rules` (ordered by `text_clock::decided_at`); `find_rules` is its first |
| Project presence | on disk / directory gone / could not be checked (treated as on disk) | `mimir_graph::missing_projects`; acted on by `mimir_graph::prune_graph` (in batches, resumable). "Something left to prune" is `MissingProject::has_graph_rows` |
| Config modes | per setting; an unknown value falls back and is reported | `config.rs`: one enum per setting, `Config::issues()` |
| Hook event | start-up, resume, clear, compact, other | `context_guard::SessionSource`, `HookEvent` in `context_guard_cmd.rs` |
| Hook failure | never reaches the session | `context_guard_cmd::fail_open` |
| Installed hook entry | absent, current, written by an older version | `commands::merge_hooks`; script versions via `# mimir-hook-version` |
| Daemon address | config value or `MIMIR_INJECT_URL` | `infer::DaemonAddr` |
| Server versus store | same schema / store is newer | `mcp::stale_server_error`, checked on every tool call and on the daemon's `/inject` (which answers 503, sending the hook to its cold path) |
| MCP tool outcome | answer, `refused: …`, `error: …` | `mcp::mark_error` marks every failure |

## 5. Code graph

| State machine | States | The one definition |
|---|---|---|
| Has a graph | never built / built | `mimir_graph::has_graph` (marker `meta.graph_built_at`, or live symbols for older stores) |
| Which rules resolved it | this build's / an older build's | `mimir_graph::rules_note` (stamp `meta.graph_rules` against `GRAPH_RULES`); `wants_full_rebuild` decides whether an ordinary build upgrades it (small projects only: `AUTO_FULL_MAX_FILES`, `AUTO_FULL_MAX_SYMBOLS`) |
| Freshness | current, behind by n, not checked | `mimir_graph::Freshness`, `freshness`, `file_is_current` |
| Live code file | the SQL predicate | `store_graph::LIVE_CODE_FILE` |
| Impact answer | not in graph, contained, reached | `mimir_graph::Impact` |
| Call edge confidence | receiver-typed, same-file, imported-name, unique-global, ambiguous | `mimir_graph::CallTier` (weight, resolved flag, label); stored in edge meta |
| Removed symbol | deleted / kept as a tombstone because someone linked to it | `store_graph::persist_file` with `Rel::origin` |
| Incremental update | ends where a full build ends | `store_graph::run_update` and `dependents`; the property test in `tests/incremental.rs` |
| Import resolution | per language | `imports::resolve_import` |
| What an answer looks like | one renderer for CLI and MCP | `mimir_graph::render` |

## 6. Proxy

| State machine | States | The one definition |
|---|---|---|
| Who owns caching | the client / the proxy / nobody | `optimize::has_cache_control`, and `Applied::we_cached()` (true only when a breakpoint was added) |
| Block kind | text, tool result, image, thinking, other | `optimize::BlockKind` |
| Request outcome | unchanged, rewritten, too large | `lib.rs`: `finish`, `TooLarge` |
| Usage tap | scanning, recorded | `lib.rs`: `UsageScanner` over `FieldScan` |
| Program class (`run`, `rewrite`) | how output is filtered; whether the hook wraps it (not a command that never exits: `filters::rules::follows`) | `filters::classify` |

## 7. Still implicit, on purpose

These were found and left as they are. Each needs a decision that is larger
than naming a state.

1. **Two full clones of one repository are one project.** They share a
   portable key and therefore one project row, and `store::adopt_project`
   rebinds its path, and with it the code graph, to whichever clone ran
   last. A linked worktree no longer has this problem or its opposite (see
   `scope::Checkout` and `Mimir::projects_for_cwd` in section 4). Telling
   clones apart needs an identity per clone that is neither the path alone
   nor the portable key alone.
2. **A handoff belongs to a scope and a checkout, not a session.** Two
   sessions in one checkout still share it. See the doc on
   `context_guard::current_handoff_in`.
3. **A main checkout made with `git init --separate-git-dir`** is not found
   from its worktrees: the repository directory records no path back to its
   working tree, and a directory holding only `.git` looks the same as a
   freshly initialised checkout. Its worktrees share a project under the
   repository directory instead.
4. **Unknown values of persisted enums** are handled per reader: an unknown
   `Rel` is skipped, an unknown ledger op reads as `edit`, an unknown `Kind`
   fails the row. One policy would need new enum variants in public types.
5. **Row ids are reused after a hard delete** when the deleted row was the
   newest. `audit::superseded_at` is fenced against it; `audit::history`, the
   injection ledger and the brief's shown-set are not.
6. **Stale vectors are searched.** A row whose text changed is pending for
   the embed pass and still valid to the vector leg until it is re-embedded.
7. **Two definitions of one name and kind in one file share a symbol row**
   (two `impl From<_> for Error`). Giving each a row needs to know which one
   a call site belongs to, and the extractor does not record a call's line.
   Until it does, one row with edges beats two rows of which one has none.
8. **The symbol kind of a free function inside a `mod`** is `method`. The
   kind is part of a symbol's identity, so correcting it would re-key
   symbols and drop the links people made to them.
9. **Sync still trusts each machine's clock.** A version is placed by the
   seconds its writer stamped, so a machine whose clock runs behind loses to
   one that wrote earlier in real time. Ordering by what a writer had *seen*
   needs a version vector per row; the text clock (`text_clock.rs`) removes
   the cases that bit in practice (an annotation undoing a forget or a
   rewrite) without one.
10. **Two changes to one memory in the same second, one synced away in
   between,** settle on one of the two by digest, not on the later. Whole
   seconds cannot say which was later.
11. **A build without the text clock** annotates a row without keeping the
    clock, and orders by the last write alone. Mixed fleets agree except
    where an annotation followed an unseen forget or rewrite.
