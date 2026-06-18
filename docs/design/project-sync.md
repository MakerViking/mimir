# Design proposal: project-scoped sync (Phase 2)

**Status:** draft for review (no code yet). **Target:** v0.10.0.
**Context:** today's sync (v0.7.0+) replicates only *global* memories. This adds
*project-scoped* memories for users who work on the same project across machines.

## 1. Problem

A project's identity is `scope::canonical_root()` — the **canonicalized absolute
path** of its root. That's correct locally but **not portable**: the same repo is

| machine | path |
|---|---|
| workstation | `/home/thomash/Koding/projects/Mimir` |
| WSL2 | `/home/thomas/dev/Mimir` |
| Mac | `/Users/thomas/code/Mimir` |

A project memory keyed to one path can't find its project on another machine —
it would attach to nothing or spawn a phantom project. Global memories sync today
precisely because they carry *no* project key. So "global-only" wasn't a
shortcut; it was the only thing with a portable identity.

## 2. Goals / non-goals

**Goals**
- The same project's memories converge across a user's machines.
- **Opt-in** and privacy-preserving — never replicate every local repo (client /
  private work) by default.
- Zero-config for the common case where it's safe.
- Reuse the existing convergent last-write-wins (LWW) machinery.

**Non-goals (stay local / per-checkout, unchanged)**
- Code graph, indexed docs, embeddings, usage signals (strength/recall history).
  These are tied to a specific checkout and recompute locally.
- Cross-*user* sharing or any notion of access control beyond the single user's
  own machines (out of scope).

## 3. Portable project key (PPK)

A string identical across machines for "the same project." Resolution order:

1. **Committed `.mimir` id (source of truth).** `.mimir` is already a recognized
   root marker; extend it to hold a stable id:
   ```toml
   # .mimir  (committed to the repo)
   id   = "01JZ8X…"     # ULID, generated once
   sync = true          # opt this project's memories into sync
   ```
   PPK = `id`. Portable, works for non-git, survives remote renames and path
   changes. Written by `mimir project init --sync` (or `touch .mimir` + edit).

2. **Normalized git remote (zero-config convenience).** Parse `.git/config`
   **as a file** (no git subprocess — preserves the deliberate stance) for
   `[remote "origin"] url`, normalize (drop scheme/user/`.git`, lowercase
   host+path): `git@github.com:MakerViking/mimir.git` →
   `git:github.com/makerviking/mimir`. Used only when a project is opted in
   without a committed id (see §4), or under the broad mode (§4b).

3. **Fallback: local-only.** No id and not opted in → today's behavior. Safe.

## 4. Opt-in model (the key privacy decision)

**Default — explicit per project.** A project's memories sync only if it carries
a committed `.mimir` with `sync = true`. Everything else stays local. This fits
Mimir's off-by-default ethos and protects private/contractor repos you happen to
have checked out. The committed id is the PPK.

**4b. Optional broad mode (power users).** A global switch
`[sync] projects = "all-git"` (default `"marked"`) syncs *every* git project
using the normalized remote as the PPK — frictionless, but the user explicitly
accepts that all git repos replicate. Default stays `"marked"`.

> Decision needed: ship only "marked" in v0.10.0 and add "all-git" later, or
> both at once. Recommendation: **"marked" only first** — smaller blast radius.

## 5. Storage & schema

- Add `projects.portable_key TEXT` (nullable; set when a project is sync-enabled).
- Sync payloads for a project memory carry the **PPK**, not the local
  `project_id`.
- **Apply (pull):** resolve PPK → local project by `portable_key`. If none
  exists yet (the project hasn't been opened on this machine), create a **shadow
  project** row (`portable_key` set, `root` empty/unknown). When the machine
  later detects a project whose `.mimir` id / remote matches the key, it adopts
  the shadow (fills in the local root). Pulled memories are immediately
  searchable in global recall regardless; they just bind to a concrete local
  path lazily.
- **Migration:** existing project memories have `portable_key = NULL` → remain
  local-only until their project is opted in. No destructive change.

## 6. Merge semantics

Unchanged in spirit: LWW, uid-keyed. Project memories key on
`(portable_key, uid)`; global on `(uid)`. The existing convergent apply path
extends naturally — `changes_since` includes `portable_key`; the hub stores it;
peers resolve it locally.

## 7. Edge cases

- **Forks** — different remote URL → different PPK → kept separate (usually
  correct). A committed shared `.mimir` id would deliberately merge them.
- **Multiple remotes** — use `origin`; a committed id wins if present
  (deterministic).
- **Monorepo** — the git root is one project (current behavior); subdir projects
  already collapse to the repo root, consistently across machines.
- **Remote renamed/moved** — breaks a remote-derived key; the committed `.mimir`
  id is immune. Argues for id-as-source-of-truth (§3.1 over §3.2).
- **Same key, different local roots on two machines** — fine; that's the whole
  point (LWW converges).

## 8. Rollout

1. PPK infra: `projects.portable_key`, `.mimir` id/sync format, `mimir project
   init --sync`, `.git/config` remote parser.
2. Extend sync payload + apply to carry/resolve PPK for marked projects.
3. Shadow-project creation + lazy adoption on first local detection.
4. (Later) `[sync] projects = "all-git"` broad mode.

## 9. Open questions

- Opt-in granularity for v0.10.0: **"marked" only** (recommended) vs also
  "all-git".
- Shadow-project UX — how `mimir status` / the dashboard present a project that
  has synced memories but no local checkout yet.
- Do project **links/annotations** sync alongside memories (probably yes, same
  as global)?
- Should `.mimir` `sync = true` imply generating an `id` if absent (yes —
  `mimir project init --sync` does both)?

## 10. Docs follow-up

Once shipped, broaden the README/`sync.md` framing ("Want your memories on more
than one machine?") to cover project memories, and document the `.mimir`
id/`sync` format and the opt-in model. Until then, keep the current honest
"global memories" wording.
