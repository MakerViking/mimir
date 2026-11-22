use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use mimir_core::config::{ColdMode, Config, ContextGuardMode, Paths};
use mimir_core::index::CollectionKind;
use mimir_core::lifecycle::Lifecycle;
use mimir_core::memory;
use mimir_core::model::{now_unix, short_uid, Kind, MemoryType, Node, Rel, RelOrigin, Scope};
use mimir_core::search::{Reach, SearchQuery};
use mimir_core::{db, store, Mimir};

use crate::capture::{self, Capture, Captured, MissingLink};

pub fn init(
    no_model: bool,
    hooks: bool,
    auto_recall: bool,
    context_guard: Option<&str>,
) -> Result<()> {
    let paths = Paths::resolve()?;
    let mut config = Config::load(&paths.config_file)?;
    let mut hooks = hooks;
    if let Some(mode) = context_guard {
        let Some(mode) = ContextGuardMode::parse(mode) else {
            bail!(
                "--context-guard must be one of: {} (got {mode:?})",
                ContextGuardMode::ACCEPTED.join(", ")
            );
        };
        config.hooks.context_guard = mode.as_str().to_string();
        // A guard that is on needs its hook entries, so asking for one asks
        // for hooks. `off` needs none: on its own it records the mode and
        // installs nothing.
        hooks |= mode != ContextGuardMode::Off;
    }
    config.save(&paths.config_file)?;
    // Opening creates + migrates the database.
    let conn = db::open(&paths.db_file)?;
    println!("config  {}", paths.config_file.display());
    println!("db      {}", paths.db_file.display());
    // A store too large to index on open gets its lookup indexes here, where
    // someone is watching and nothing has a deadline.
    let missing = db::missing_indexes(&conn)?;
    if !missing.is_empty() {
        println!(
            "index   adding {} lookup index(es); this reads the whole store once and can \
             take a minute on a large one…",
            missing.len()
        );
        db::ensure_indexes(&conn)?;
        println!("index   added — removing a row no longer scans every other row");
    }
    drop(conn);
    if no_model {
        println!(
            "model   skipped (BM25-only; run `mimir embed --fetch` to enable semantic search)"
        );
    } else {
        match mimir_core::embed::Embedder::load(
            &paths,
            &config.embedding.model,
            config.embedding.device().as_str(),
            true,
        ) {
            Ok(e) => println!("model   {} ready ({}-dim)", e.name, e.dim),
            Err(err) => {
                eprintln!("model   download failed ({err}); search is BM25-only until `mimir embed --fetch` succeeds")
            }
        }
    }
    install_agent_commands(&config);
    // A failed model download above leaves a supported state (BM25-only).
    // A failed hook install leaves the thing this run was asked for undone,
    // so it is remembered and becomes the exit code once the rest is said.
    let mut hooks_failed = false;
    if hooks {
        if let Err(err) = install_hooks(&config, auto_recall) {
            eprintln!("hooks   install failed: {err}");
            hooks_failed = true;
        }
    }
    println!();
    println!("Register the MCP server once, globally:");
    println!("  claude mcp add --scope user mimir -- mimir mcp");
    if !hooks {
        println!();
        println!("Token-saving hooks (filter command output + inject project rules) are opt-in:");
        println!("  mimir init --hooks");
    } else if !auto_recall {
        println!();
        println!("Per-prompt auto-recall (inject a relevant memory into every prompt) is opt-in:");
        println!("  mimir init --hooks --auto-recall");
    }
    if hooks && config.hooks.context_guard() == ContextGuardMode::Off {
        println!();
        println!("Context-window guard (nudge before an auto-compact takes control) is opt-in:");
        println!("  mimir init --hooks --context-guard pause   # or: handoff");
    }
    if hooks_failed {
        bail!(
            "the hooks were not installed (see `hooks   install failed` above); \
             config and database are in place"
        );
    }
    Ok(())
}

/// The PreToolUse hook script: delegates all rewrite logic to `mimir rewrite`,
/// so rules live in the Rust binary (single source of truth), not this file.
const MIMIR_REWRITE_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir PreToolUse hook — rewrites noisy commands through `mimir run` to save
# tokens. All logic lives in `mimir rewrite`. Requires: mimir, jq.
command -v jq >/dev/null 2>&1 || exit 0
command -v mimir >/dev/null 2>&1 || exit 0
INPUT=$(cat)
CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')
[ -z "$CMD" ] && exit 0
REWRITTEN=$(mimir rewrite "$CMD" 2>/dev/null) || exit 0
[ "$CMD" = "$REWRITTEN" ] && exit 0
UPDATED=$(printf '%s' "$INPUT" | jq -c --arg cmd "$REWRITTEN" '.tool_input | .command = $cmd')
jq -n --argjson updated "$UPDATED" '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "Mimir token-saving rewrite",
    "updatedInput": $updated
  }
}'
"#;

/// The opt-in UserPromptSubmit hook script: tries the warm `/inject` HTTP
/// endpoint first (only live while `mimir mcp --http` is running — see
/// `mcp.rs::inject_router`), falling back to the cold `mimir recall-inject`
/// CLI path when that's unreachable. Both paths run the exact same
/// relevance-floor/formatting/budget logic (`mimir_core::inject::compute`)
/// over the same project: the script sends the event's `cwd` to the warm
/// endpoint, which resolves the project from it, and runs the cold command
/// from that same directory. Which one answers therefore decides latency
/// and what the search can see — the warm endpoint always has the vector
/// leg, the cold path only under `[hooks] cold_mode = "full"` — never which
/// project is searched or how a hit is judged.
/// Unlike `MIMIR_REWRITE_SH`, plain stdout (not a `hookSpecificOutput` JSON
/// envelope) is how Claude Code injects UserPromptSubmit context — same
/// mechanism as the SessionStart rules hook.
///
/// The warm endpoint's address is `config.hooks.inject_url`
/// (`HooksConfig::inject_url`, default `127.0.0.1:8077`, matching the port
/// documented in README.md's remote-MCP example) — baked into the script at
/// install time by `render_recall_script`. `MIMIR_INJECT_URL` still
/// overrides it at hook-invocation time, for a one-off run against a
/// different bind without touching config.toml (the same override
/// `infer::DaemonAddr` applies for the daemon, the delegation client and
/// `doctor`). If no daemon answers there, curl fails fast (2s timeout) and
/// behavior is identical to before this endpoint existed — just slower.
///
/// A daemon started with `MIMIR_HTTP_TOKEN` requires the bearer on every
/// route, `/inject` included, so the script sends it whenever the variable
/// is set in the hook's environment — on curl's stdin (`-H @-`), which
/// keeps it out of the process list. A curl older than 7.55 does not read
/// headers that way; there the request is rejected and the hook goes cold,
/// as it does without a token.
///
/// Also extracts a cheap enrichment signal from the caller's working tree:
/// the stems (basename, extension stripped) of up to 8 files changed since
/// HEAD, via `git diff --name-only`. Passed as `enrich=` on the warm URL and
/// `--enrich` on the cold CLI path — never mixed into `prompt` itself.
/// `inject::compute`/`clears_floor` treat it as strictly weaker than the raw
/// prompt: it can extend a real overlap but can never single-handedly clear
/// the relevance floor (see `inject.rs`'s self-licensing guard doc comment).
/// Silent if `git` isn't on PATH or the cwd isn't a repo — enrichment is a
/// nice-to-have, not a requirement.
const MIMIR_RECALL_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 5
# Mimir UserPromptSubmit hook — prints at most one relevant memory (or
# nothing) as extra context for this turn. Tries the warm HTTP endpoint
# (fast; requires `mimir mcp --http` to be running) first, falls back to
# the cold CLI path (slower, always available). All relevance-floor logic
# lives in mimir_core::inject::compute, shared by both. Requires: mimir, jq;
# curl and git are optional (git enrichment and the warm path are both
# skipped gracefully when unavailable).
command -v jq >/dev/null 2>&1 || exit 0
command -v mimir >/dev/null 2>&1 || exit 0
INPUT=$(cat)
PROMPT=$(printf '%s' "$INPUT" | jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
# Claude Code supplies session_id on every hook payload. Passing it through
# is what makes per-session dedup work: without it the ledger key is empty,
# nothing is ever recorded as already-injected, and the SAME memory can be
# re-injected on every prompt of a long session — the exact "flooded by
# stale crap" failure the relevance floor exists to prevent.
SESSION=$(printf '%s' "$INPUT" | jq -r '.session_id // empty')
INJECT_URL="${MIMIR_INJECT_URL:-__MIMIR_INJECT_URL_DEFAULT__}"
PROJECT_DIR=$(printf '%s' "$INPUT" | jq -r '.cwd // empty')
[ -z "$PROJECT_DIR" ] && PROJECT_DIR="$PWD"
ENRICH=""
if command -v git >/dev/null 2>&1; then
    ENRICH=$(git -C "$PROJECT_DIR" diff --name-only HEAD 2>/dev/null | head -8 \
        | sed -E 's#.*/##; s#\.[^./]+$##' | tr '\n' ' ' | sed -E 's/^ +| +$//g')
fi
if command -v curl >/dev/null 2>&1; then
    ENC_PROMPT=$(printf '%s' "$PROMPT" | jq -sRr @uri)
    URL="${INJECT_URL}?prompt=${ENC_PROMPT}"
    if [ -n "$ENRICH" ]; then
        ENC_ENRICH=$(printf '%s' "$ENRICH" | jq -sRr @uri)
        URL="${URL}&enrich=${ENC_ENRICH}"
    fi
    if [ -n "$SESSION" ]; then
        ENC_SESSION=$(printf '%s' "$SESSION" | jq -sRr @uri)
        URL="${URL}&session=${ENC_SESSION}"
    fi
    # One daemon serves every project, so it is told which directory this
    # prompt came from and scopes the search to that directory's project —
    # the same directory the cold path below runs from.
    ENC_CWD=$(printf '%s' "$PROJECT_DIR" | jq -sRr @uri)
    URL="${URL}&cwd=${ENC_CWD}"
    # A daemon started with MIMIR_HTTP_TOKEN rejects a request without the
    # bearer. Sent on stdin (curl 7.55+) so it never shows in a process list.
    WARM_OK=""
    if [ -n "${MIMIR_HTTP_TOKEN:-}" ]; then
        WARM=$(printf 'Authorization: Bearer %s\n' "$MIMIR_HTTP_TOKEN" \
            | curl -sf --max-time 2 -H @- "$URL" 2>/dev/null) && WARM_OK=1
    else
        WARM=$(curl -sf --max-time 2 "$URL" 2>/dev/null) && WARM_OK=1
    fi
    if [ -n "$WARM_OK" ]; then
        [ -n "$WARM" ] && printf '%s\n' "$WARM"
        exit 0
    fi
fi
# The cold path scopes by its working directory: make that the event's.
cd "$PROJECT_DIR" 2>/dev/null
COLD=(mimir recall-inject)
[ -n "$ENRICH" ] && COLD+=(--enrich "$ENRICH")
[ -n "$SESSION" ] && COLD+=(--session "$SESSION")
"${COLD[@]}" -- "$PROMPT" 2>/dev/null
exit 0
"#;

/// The PreToolUse guard-anchors hook script (installed for the tools in
/// `context_guard_cmd::ANCHOR_TOOLS`): unconditional
/// under `--hooks` (no separate opt-in flag) — a memory with no
/// `meta.anchors` makes every invocation a silent no-op, so there is
/// nothing to gate. All matching logic lives in `mimir_core::anchors` /
/// `mimir context-guard pretool`; this script only pipes the hook's stdin
/// JSON straight through (no jq needed — the JSON is parsed in Rust).
const MIMIR_ANCHORS_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir PreToolUse guard-anchors hook — surfaces at most one anchored
# memory (see `mimir remember --anchor`) as extra context when a matching
# file is edited/written, or mentioned in a Bash command. Requires: mimir.
command -v mimir >/dev/null 2>&1 || exit 0
mimir context-guard pretool 2>/dev/null
exit 0
"#;

/// The opt-in `[hooks] context_guard != "off"` hook scripts — one each for
/// UserPromptSubmit, PreCompact, SessionStart. All logic lives in
/// `mimir_core::context_guard` / `mimir context-guard <subcommand>`; each
/// script only pipes stdin through (no jq — parsed in Rust).
const MIMIR_CONTEXT_GUARD_PROMPT_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir UserPromptSubmit context-guard hook — see `mimir_core::context_guard`
# and `[hooks] context_guard` in config.toml. Requires: mimir.
command -v mimir >/dev/null 2>&1 || exit 0
mimir context-guard prompt 2>/dev/null
exit 0
"#;
const MIMIR_CONTEXT_GUARD_PRECOMPACT_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir PreCompact context-guard hook — see `mimir_core::context_guard`
# and `[hooks] context_guard` in config.toml. Requires: mimir.
command -v mimir >/dev/null 2>&1 || exit 0
mimir context-guard precompact 2>/dev/null
exit 0
"#;
const MIMIR_CONTEXT_GUARD_SESSION_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir SessionStart context-guard hook — see `mimir_core::context_guard`
# and `[hooks] context_guard` in config.toml. Requires: mimir.
command -v mimir >/dev/null 2>&1 || exit 0
mimir context-guard session-start 2>/dev/null
exit 0
"#;

/// Bakes `config.hooks.inject_url` into `MIMIR_RECALL_SH`'s fallback default,
/// split out as a pure function so it's unit-testable without touching disk.
fn render_recall_script(inject_url: &str) -> String {
    MIMIR_RECALL_SH.replace("__MIMIR_INJECT_URL_DEFAULT__", inject_url)
}

/// Which opt-in a hook script belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HookGroup {
    /// Installed by `--hooks` itself.
    Base,
    /// Installed by `--auto-recall`.
    AutoRecall,
    /// Installed while `[hooks] context_guard` is not off.
    ContextGuard,
}

const REWRITE_SCRIPT: &str = "mimir-rewrite.sh";
const RECALL_SCRIPT: &str = "mimir-recall.sh";
const ANCHORS_SCRIPT: &str = "mimir-anchors.sh";
const GUARD_PROMPT_SCRIPT: &str = "mimir-context-guard-prompt.sh";
const GUARD_PRECOMPACT_SCRIPT: &str = "mimir-context-guard-precompact.sh";
const GUARD_SESSION_SCRIPT: &str = "mimir-context-guard-session.sh";

/// Every hook script `mimir init --hooks` can install: file name, the text
/// this binary ships, and the opt-in it belongs to. Each text carries a
/// `# mimir-hook-version: N` line, which `doctor` compares against the
/// installed copy.
const HOOK_SCRIPTS: &[(&str, &str, HookGroup)] = &[
    (REWRITE_SCRIPT, MIMIR_REWRITE_SH, HookGroup::Base),
    (RECALL_SCRIPT, MIMIR_RECALL_SH, HookGroup::AutoRecall),
    (ANCHORS_SCRIPT, MIMIR_ANCHORS_SH, HookGroup::Base),
    (
        GUARD_PROMPT_SCRIPT,
        MIMIR_CONTEXT_GUARD_PROMPT_SH,
        HookGroup::ContextGuard,
    ),
    (
        GUARD_PRECOMPACT_SCRIPT,
        MIMIR_CONTEXT_GUARD_PRECOMPACT_SH,
        HookGroup::ContextGuard,
    ),
    (
        GUARD_SESSION_SCRIPT,
        MIMIR_CONTEXT_GUARD_SESSION_SH,
        HookGroup::ContextGuard,
    ),
];

/// The `# mimir-hook-version: N` a hook script carries, if it has one.
fn hook_version(script: &str) -> Option<u32> {
    script.lines().find_map(|line| {
        line.strip_prefix("# mimir-hook-version:")?
            .trim()
            .parse()
            .ok()
    })
}

/// Installed hook scripts that are older than the ones this binary ships,
/// as `(name, installed version, shipped version)`. A script with no
/// version line predates versioning and counts as 0. Scripts that are not
/// installed are not listed: every hook is opt-in.
fn outdated_hook_scripts(hooks_dir: &std::path::Path) -> Vec<(&'static str, u32, u32)> {
    HOOK_SCRIPTS
        .iter()
        .filter_map(|(name, shipped, _)| {
            let installed = std::fs::read_to_string(hooks_dir.join(name)).ok()?;
            let installed = hook_version(&installed).unwrap_or(0);
            let shipped = hook_version(shipped).unwrap_or(0);
            (installed < shipped).then_some((*name, installed, shipped))
        })
        .collect()
}

/// The command a `SessionStart` entry runs for a `mimir` subcommand that
/// needs no wrapper script: the subcommand, guarded so a machine where
/// `mimir` is gone from PATH (uninstalled, or a session with another
/// environment) starts its sessions without a hook error. The subcommands
/// fail open by themselves; this covers the one failure they cannot.
///
/// The guard is POSIX shell, which is what Claude Code runs a hook command
/// in everywhere except Windows without Git Bash, where it is PowerShell.
/// The installer cannot know which of the two a Windows machine will use,
/// so there the command stays bare — valid in both, unguarded as before.
fn guarded_hook_command(subcommand: &str) -> String {
    if cfg!(windows) {
        format!("mimir {subcommand}")
    } else {
        format!("command -v mimir >/dev/null 2>&1 || exit 0; mimir {subcommand}")
    }
}

/// The anchors matcher earlier versions installed. An entry still carrying
/// it is brought up to [`crate::context_guard_cmd::anchor_matcher`] on the
/// next `mimir init --hooks`; any other matcher is somebody's own edit and
/// is left alone.
const SUPERSEDED_ANCHOR_MATCHERS: &[&str] = &["Bash|Edit|Write"];

/// Install the opt-in Claude Code hooks: a PreToolUse(Bash) rewrite hook, a
/// PreToolUse guard-anchors hook, and the two SessionStart hooks that
/// inject the project rules pack and the session brief — all unconditional
/// under `--hooks` — plus (when `auto_recall`) a UserPromptSubmit hook that
/// injects at most one relevant memory per prompt, and (when the context
/// guard is on) the UserPromptSubmit/PreCompact/SessionStart context-guard
/// hooks. Idempotent, and never clobbers existing hooks: an entry Mimir
/// installed in an older shape is brought up to date in place, everything
/// else is left as found. `auto_recall=false` leaves
/// `hooks.UserPromptSubmit`'s auto-recall entry untouched, and a guard that
/// is off (the default) adds none of the context-guard entries at all
/// (see `merge_hook_settings`'s unit tests).
///
/// Order matters, and is: read and merge `settings.json`, then write the
/// scripts, then write `settings.json`. A settings file that cannot be
/// parsed or merged stops the install before anything is on disk. The
/// scripts go first so no entry ever points at a script that is not there;
/// the price is that a settings file that parses but cannot be *written*
/// leaves the scripts behind, unreferenced, and the error names the file. The
/// backup is taken only when there is a file to back up and it is about to
/// change — a re-run that changes nothing rewrites nothing.
///
/// A script is written when its hook is being installed now, and also when
/// a copy is already installed: `mimir-recall.sh` carries
/// `config.hooks.inject_url`, and every script carries a version, so a
/// plain `mimir init --hooks` after a config change or an upgrade always
/// brings what is installed up to date.
fn install_hooks(config: &Config, auto_recall: bool) -> Result<()> {
    if std::env::var_os("MIMIR_HOME").is_some() {
        // Isolated instances never touch the user's agent config.
        println!(
            "hooks   skipped — MIMIR_HOME is set, and an isolated instance never touches \
             ~/.claude (unset MIMIR_HOME to install hooks)"
        );
        return Ok(());
    }
    let base = directories::BaseDirs::new().context("cannot resolve home directory")?;
    let claude = base.home_dir().join(".claude");
    if !claude.is_dir() {
        println!("hooks   ~/.claude not found — skipping (is Claude Code installed?)");
        return Ok(());
    }

    let hooks_dir = claude.join("hooks");
    let script_path = |name: &str| hooks_dir.join(name).to_string_lossy().into_owned();
    let installed = |name: &str| hooks_dir.join(name).is_file();
    let guard_on = config.hooks.context_guard() != ContextGuardMode::Off;

    // 1. Read settings.json and merge into it, in memory.
    let settings_path = claude.join("settings.json");
    let existing = match std::fs::read_to_string(&settings_path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    // An empty file is a settings file nobody has put anything in yet (a
    // `touch`, or a tool that created it and has not written). It says the
    // same as no file, and is not a reason to refuse.
    let before: serde_json::Value = match &existing {
        Some(text) if text.trim().is_empty() => serde_json::json!({}),
        Some(text) => serde_json::from_str(text)
            .with_context(|| format!("{} is not valid JSON", settings_path.display()))?,
        None => serde_json::json!({}),
    };
    let rewrite_script = script_path(REWRITE_SCRIPT);
    let anchors_script = script_path(ANCHORS_SCRIPT);
    let recall_script = auto_recall.then(|| script_path(RECALL_SCRIPT));
    let context_guard_scripts = guard_on.then(|| {
        (
            script_path(GUARD_PROMPT_SCRIPT),
            script_path(GUARD_PRECOMPACT_SCRIPT),
            script_path(GUARD_SESSION_SCRIPT),
        )
    });
    let (after, messages) = merge_hooks(
        before.clone(),
        &HookPlan {
            rewrite_script: &rewrite_script,
            recall_script: recall_script.as_deref(),
            anchors_script: &anchors_script,
            context_guard_scripts: context_guard_scripts
                .as_ref()
                .map(|(p, c, s)| (p.as_str(), c.as_str(), s.as_str())),
            brief_enabled: config.brief.enabled,
        },
    )?;

    // 2. Write the delegate scripts (executable).
    std::fs::create_dir_all(&hooks_dir)?;
    let write_script = |name: &str, content: &str| -> Result<()> {
        let path = hooks_dir.join(name);
        std::fs::write(&path, content).with_context(|| format!("write {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .with_context(|| format!("make {} executable", path.display()))?;
        }
        Ok(())
    };
    let recall = render_recall_script(crate::infer::DaemonAddr::configured(config));
    for (name, shipped, group) in HOOK_SCRIPTS {
        let wanted = match group {
            HookGroup::Base => true,
            HookGroup::AutoRecall => auto_recall,
            HookGroup::ContextGuard => guard_on,
        };
        if wanted || installed(name) {
            let content = if *name == RECALL_SCRIPT {
                recall.as_str()
            } else {
                shipped
            };
            write_script(name, content)?;
        }
    }

    // 3. Write settings.json, backing up what was there, if it changes.
    let changed = after != before;
    let backup = claude.join("settings.json.mimir-bak");
    if changed {
        if let Some(text) = &existing {
            std::fs::write(&backup, text)
                .with_context(|| format!("write the backup {}", backup.display()))?;
        }
        std::fs::write(&settings_path, serde_json::to_string_pretty(&after)?)
            .with_context(|| format!("write {}", settings_path.display()))?;
    }
    println!("hooks   {}", messages.join("; "));
    if changed && existing.is_some() {
        println!("hooks   backup at {}", backup.display());
    }
    Ok(())
}

/// What one `mimir init --hooks` run wants in `settings.json`.
struct HookPlan<'a> {
    rewrite_script: &'a str,
    /// `None` leaves the auto-recall entry untouched.
    recall_script: Option<&'a str>,
    anchors_script: &'a str,
    /// Prompt, precompact and session-start script paths, in that order;
    /// `None` adds none of the context-guard entries.
    context_guard_scripts: Option<(&'a str, &'a str, &'a str)>,
    /// `[brief] enabled`, so the line about the brief entry says whether
    /// the entry does anything.
    brief_enabled: bool,
}

/// Pure settings.json merge, split out of `install_hooks` so the merge
/// logic (idempotency, which keys get touched) is unit-testable without a
/// real `~/.claude` — `install_hooks` early-returns under `MIMIR_HOME`, so
/// this is the only way to test it at all. `recall_script = None` must
/// leave `hooks.UserPromptSubmit`'s auto-recall entry completely untouched
/// — that's what makes `auto_recall=false` byte-identical to
/// pre-auto-recall behavior. Likewise `context_guard_scripts = None`
/// (prompt, precompact, session_start script paths, in that order) must
/// add none of the UserPromptSubmit/PreCompact/SessionStart context-guard
/// entries — what makes `context_guard == "off"` byte-identical to
/// pre-context-guard behavior.
///
/// [`merge_hooks`] with the brief in its default state.
#[cfg(test)]
fn merge_hook_settings(
    root: serde_json::Value,
    rewrite_script: &str,
    recall_script: Option<&str>,
    anchors_script: &str,
    context_guard_scripts: Option<(&str, &str, &str)>,
) -> Result<(serde_json::Value, Vec<String>)> {
    merge_hooks(
        root,
        &HookPlan {
            rewrite_script,
            recall_script,
            anchors_script,
            context_guard_scripts,
            brief_enabled: mimir_core::config::BriefConfig::default().enabled,
        },
    )
}

/// Merge `plan` into a parsed `settings.json`, returning the result and one
/// message per hook saying what happened to it.
///
/// Each of Mimir's entries is in one of three states — absent, installed
/// in the current shape, or installed in a shape an older version wrote —
/// and the merge moves it to "current" from either of the others without
/// ever adding a second copy. Entries Mimir did not write, and Mimir
/// entries somebody edited by hand, are left exactly as found.
fn merge_hooks(
    mut root: serde_json::Value,
    plan: &HookPlan,
) -> Result<(serde_json::Value, Vec<String>)> {
    if !root.is_object() {
        bail!("settings.json is not a JSON object");
    }
    let hooks = root
        .as_object_mut()
        .unwrap()
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("settings.json `hooks` is not an object")?;

    let mut messages: Vec<String> = Vec::new();

    // SessionStart: inject the rules pack (stdout becomes session context).
    let session = hooks
        .entry("SessionStart")
        .or_insert_with(|| serde_json::json!([]));
    let session_arr = session
        .as_array_mut()
        .context("hooks.SessionStart is not an array")?;
    // `--hook` is what tells the command to read the event Claude Code
    // puts on its stdin; an entry written before the flag existed is the
    // bare command, which reads nothing and scopes by the process cwd.
    let rules_command = guarded_hook_command("rules show --hook");
    if upgrade_command(session_arr, "mimir rules show", &rules_command) {
        messages.push(
            "SessionStart (project rules) upgraded — now follows the session's directory and \
             skips when `mimir` is not on PATH"
                .into(),
        );
    } else if entries_mention(session_arr, "mimir rules show") {
        messages.push("SessionStart already installed".into());
    } else {
        session_arr.push(serde_json::json!({
            "hooks": [{ "type": "command", "command": rules_command }]
        }));
        messages.push("SessionStart (project rules) added".into());
    }

    // SessionStart: the session brief (own entry — stdouts concatenate).
    // Installed unconditionally; `[brief] enabled` is the command's own
    // first check, so an entry for a disabled brief costs nothing.
    let brief_state = if plan.brief_enabled {
        "on: `[brief] enabled = true`"
    } else {
        "inert while `[brief] enabled = false`"
    };
    let brief_command = guarded_hook_command("brief show");
    if upgrade_command(session_arr, "mimir brief show", &brief_command) {
        messages.push(format!(
            "SessionStart (brief) upgraded — now skips when `mimir` is not on PATH ({brief_state})"
        ));
    } else if entries_mention(session_arr, "mimir brief show") {
        messages.push(format!(
            "SessionStart (brief) already installed ({brief_state})"
        ));
    } else {
        session_arr.push(serde_json::json!({
            "hooks": [{ "type": "command", "command": brief_command }]
        }));
        messages.push(format!(
            "SessionStart (session brief) added ({brief_state})"
        ));
    }

    // PreToolUse(Bash): the rewrite hook. Skip if another rewrite hook (e.g.
    // RTK) is present — running both would double-wrap commands.
    let pre = hooks
        .entry("PreToolUse")
        .or_insert_with(|| serde_json::json!([]));
    let pre_arr = pre
        .as_array_mut()
        .context("hooks.PreToolUse is not an array")?;
    if entries_mention(pre_arr, "mimir-rewrite") {
        messages.push("PreToolUse already installed".into());
    } else if entries_mention(pre_arr, "rtk") || entries_mention(pre_arr, "rewrite") {
        messages.push(
            "PreToolUse SKIPPED — another rewrite hook (e.g. RTK) is present. Remove it \
             from ~/.claude/settings.json, then re-run `mimir init --hooks`."
                .into(),
        );
    } else {
        pre_arr.push(serde_json::json!({
            "matcher": "Bash",
            "hooks": [{ "type": "command", "command": plan.rewrite_script }]
        }));
        messages.push("PreToolUse (command filter) added".into());
    }

    // PreToolUse: guard anchors, for every tool the handler reads a target
    // from. Unconditional under `--hooks` — see `MIMIR_ANCHORS_SH`'s doc
    // comment for why there's no separate opt-in.
    let anchor_matcher = crate::context_guard_cmd::anchor_matcher();
    // An entry is Mimir's to rewrite only when it is exactly what Mimir
    // wrote: one hook, running this install's anchors script, under a
    // matcher an earlier version installed. A matcher is per entry, so an
    // entry that also carries somebody's own hook, or whose command merely
    // contains the script's name, is theirs: widening it would start
    // running their hook on tools they never asked for.
    let wrote_it = |entry: &serde_json::Value| {
        let one_hook_ours = entry
            .get("hooks")
            .and_then(|h| h.as_array())
            .is_some_and(|hooks| {
                hooks.len() == 1
                    && hooks[0].get("command").and_then(|c| c.as_str()) == Some(plan.anchors_script)
            });
        let old_matcher = entry
            .get("matcher")
            .and_then(|m| m.as_str())
            .is_some_and(|m| SUPERSEDED_ANCHOR_MATCHERS.contains(&m));
        one_hook_ours && old_matcher
    };
    let mut upgraded = false;
    for entry in pre_arr.iter_mut().filter(|entry| wrote_it(entry)) {
        entry["matcher"] = serde_json::json!(anchor_matcher);
        upgraded = true;
    }
    if upgraded {
        messages.push(format!(
            "PreToolUse anchors upgraded — matcher is now {anchor_matcher}"
        ));
    } else if entries_mention(pre_arr, "mimir-anchors") {
        messages.push("PreToolUse anchors already installed".into());
    } else {
        pre_arr.push(serde_json::json!({
            "matcher": anchor_matcher,
            "hooks": [{ "type": "command", "command": plan.anchors_script }]
        }));
        messages.push("PreToolUse (guard anchors) added".into());
    }

    // UserPromptSubmit: opt-in auto-recall. Only touched when asked for.
    if let Some(recall_script) = plan.recall_script {
        let prompt = hooks
            .entry("UserPromptSubmit")
            .or_insert_with(|| serde_json::json!([]));
        let prompt_arr = prompt
            .as_array_mut()
            .context("hooks.UserPromptSubmit is not an array")?;
        if entries_mention(prompt_arr, "mimir-recall") {
            messages.push("UserPromptSubmit already installed".into());
        } else {
            prompt_arr.push(serde_json::json!({
                "hooks": [{ "type": "command", "command": recall_script }]
            }));
            messages.push("UserPromptSubmit (auto-recall) added".into());
        }
    }

    // Context guard: UserPromptSubmit + PreCompact + SessionStart entries,
    // only when the guard is on — this is what keeps a default (`"off"`)
    // install byte-identical to pre-context-guard settings.json output (see
    // `merge_hook_settings`'s unit tests).
    if let Some((prompt_script, precompact_script, session_script)) = plan.context_guard_scripts {
        let cg_prompt = hooks
            .entry("UserPromptSubmit")
            .or_insert_with(|| serde_json::json!([]));
        let cg_prompt_arr = cg_prompt
            .as_array_mut()
            .context("hooks.UserPromptSubmit is not an array")?;
        if entries_mention(cg_prompt_arr, "mimir-context-guard-prompt") {
            messages.push("UserPromptSubmit context-guard already installed".into());
        } else {
            cg_prompt_arr.push(serde_json::json!({
                "hooks": [{ "type": "command", "command": prompt_script }]
            }));
            messages.push("UserPromptSubmit (context guard) added".into());
        }

        let precompact = hooks
            .entry("PreCompact")
            .or_insert_with(|| serde_json::json!([]));
        let precompact_arr = precompact
            .as_array_mut()
            .context("hooks.PreCompact is not an array")?;
        if entries_mention(precompact_arr, "mimir-context-guard-precompact") {
            messages.push("PreCompact already installed".into());
        } else {
            precompact_arr.push(serde_json::json!({
                "hooks": [{ "type": "command", "command": precompact_script }]
            }));
            messages.push("PreCompact (context guard) added".into());
        }

        let session_cg = hooks
            .entry("SessionStart")
            .or_insert_with(|| serde_json::json!([]));
        let session_cg_arr = session_cg
            .as_array_mut()
            .context("hooks.SessionStart is not an array")?;
        if entries_mention(session_cg_arr, "mimir-context-guard-session") {
            messages.push("SessionStart context-guard already installed".into());
        } else {
            session_cg_arr.push(serde_json::json!({
                "hooks": [{ "type": "command", "command": session_script }]
            }));
            messages.push("SessionStart (context guard) added".into());
        }
    }

    Ok((root, messages))
}

/// Rewrite every hook whose command is *exactly* `from` to run `to`; true
/// when at least one was. Exact on purpose: a command somebody extended by
/// hand is theirs. Nothing to do, and `false`, when the two are the same.
fn upgrade_command(entries: &mut [serde_json::Value], from: &str, to: &str) -> bool {
    if from == to {
        return false;
    }
    let mut replaced = false;
    for entry in entries {
        let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) else {
            continue;
        };
        for hook in hooks {
            if hook.get("command").and_then(|c| c.as_str()) == Some(from) {
                hook["command"] = serde_json::json!(to);
                replaced = true;
            }
        }
    }
    replaced
}

/// True if any hook entry (or its nested hooks) has a command containing `needle`.
fn entries_mention(entries: &[serde_json::Value], needle: &str) -> bool {
    entries.iter().any(|e| {
        e.get("hooks")
            .and_then(|h| h.as_array())
            .map(|arr| {
                arr.iter().any(|h| {
                    h.get("command")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.contains(needle))
                })
            })
            .unwrap_or(false)
    })
}

/// One Mimir slash command: name carries the `m-` prefix (collision safety
/// with users' own commands), `allowed` is Claude-Code-only tool pre-approval.
struct SlashCmd {
    name: &'static str,
    desc: &'static str,
    body: &'static str,
    allowed: Option<&'static str>,
}

/// Every `/m-*` slash command Mimir ships. `{args}` becomes the app's own
/// argument placeholder at render time.
const SLASH_COMMANDS: &[SlashCmd] = &[
    SlashCmd {
        name: "m-graph",
        desc: "Open the interactive Mimir graph visualization (current project)",
        body: "Run `mimir graph viz --open {args}` with your shell tool from the current \
            project root, then report the output path it prints. If it fails with \
            \"not inside a project\", relay the suggestion in the error: it needs a \
            project root (.git/.hg/.svn/.jj), or `touch .mimir` to mark one.",
        allowed: Some("Bash(mimir graph viz:*)"),
    },
    SlashCmd {
        name: "m-stats",
        desc: "Open the Mimir stats dashboard (memories, docs, code, learning)",
        body: "Run `mimir dashboard --open {args}` with your shell tool, then report the \
            output path it prints.",
        allowed: Some("Bash(mimir dashboard:*)"),
    },
    SlashCmd {
        name: "m-report",
        desc: "Mimir activity report: day / week / month / year / all-time",
        body: "Run `mimir report` with your shell tool and show its complete output \
            verbatim in a code block. Do not summarize or reformat the table.",
        allowed: Some("Bash(mimir report:*)"),
    },
    SlashCmd {
        name: "m-savings",
        desc: "Mimir token-savings report (outline/peek/command-filter/proxy)",
        body: "Run `mimir savings` with your shell tool and show its complete output \
            verbatim. It reports tokens saved today/week/month/all-time and by source.",
        allowed: Some("Bash(mimir savings:*)"),
    },
    SlashCmd {
        name: "m-scan",
        desc: "Auto-link Mimir memories to the code symbols they mention",
        body: "Run `mimir link --scan` with your shell tool from the current project \
            root and show its output. If links were created, suggest /m-graph to see \
            the new memory-to-code connections.",
        allowed: Some("Bash(mimir link:*)"),
    },
    SlashCmd {
        name: "m-recall",
        desc: "Search Mimir memory (memories, docs, code)",
        body: "Run `mimir recall {args}` with your shell tool and show the results. \
            If a hit looks like exactly what the user needs, also run \
            `mimir get <id>` on it and show the full body.",
        allowed: Some("Bash(mimir recall:*), Bash(mimir get:*)"),
    },
    SlashCmd {
        name: "m-remember",
        desc: "Save a memory to Mimir",
        body: "Store this in Mimir: {args}\n\nUse the mimir remember MCP tool (or \
            `mimir remember` via shell). Pick the fitting type (gotcha / decision / \
            insight / idea / note / person) and concise tags. If it is about specific \
            code, pass `link` with the symbol name. Confirm what was stored.",
        allowed: Some("mcp__mimir__remember, mcp__mimir__recall, Bash(mimir remember:*)"),
    },
    SlashCmd {
        name: "m-impact",
        desc: "Blast radius of the current uncommitted changes (Mimir code graph)",
        body: "Run `mimir graph impact $(git diff --name-only)` with your shell tool \
            from the current project root and show the affected symbols. If the diff \
            is empty, say there are no uncommitted changes to analyze.",
        allowed: Some("Bash(mimir graph impact:*)"),
    },
    SlashCmd {
        name: "m-doctor",
        desc: "Mimir health check (database, search index, models)",
        body: "Run `mimir doctor` and `mimir status` with your shell tool and show \
            both outputs verbatim. If any check is not ok, explain what it means and \
            how to fix it.",
        allowed: Some("Bash(mimir doctor:*), Bash(mimir status:*)"),
    },
];

/// Installed only when `[sync]` is enabled (re-run `mimir init` after enabling).
const SYNC_SLASH_COMMANDS: &[SlashCmd] = &[SlashCmd {
    name: "m-sync",
    desc: "Sync Mimir memories with the central store",
    body: "Run `mimir sync` with your shell tool and show the push/pull summary. \
        If it reports an auth or connection error, check MIMIR_SYNC_TOKEN and the \
        [sync] endpoint/dir in the Mimir config.",
    allowed: Some("Bash(mimir sync:*)"),
}];

/// Install the `/m-*` slash commands for the agent CLIs that support
/// user-level custom commands. Installed only for apps already present on
/// the machine; existing files are never overwritten (user edits win).
/// Re-running `mimir init` after an upgrade refreshes missing files.
fn install_agent_commands(config: &Config) {
    // An isolated instance (tests, scratch homes) must not touch the user's
    // agent configs — MIMIR_HOME means "everything under one directory".
    if std::env::var_os("MIMIR_HOME").is_some() {
        return;
    }
    // /m-sync is only useful (and only installed) when sync is enabled.
    let extra: &[SlashCmd] = if config.sync.enabled() {
        SYNC_SLASH_COMMANDS
    } else {
        &[]
    };

    let md = |cmd: &SlashCmd, with_allowed: bool| {
        let allowed = match (with_allowed, cmd.allowed) {
            (true, Some(a)) => format!("allowed-tools: {a}\n"),
            _ => String::new(),
        };
        format!(
            "---\ndescription: {}\n{allowed}---\n\n{}\n",
            cmd.desc,
            cmd.body.replace("{args}", "$ARGUMENTS")
        )
    };
    let toml = |cmd: &SlashCmd| {
        format!(
            "description = \"{}\"\nprompt = \"\"\"\n{}\n\"\"\"\n",
            cmd.desc,
            cmd.body.replace("{args}", "{{args}}")
        )
    };

    let Some(base) = directories::BaseDirs::new() else {
        return;
    };
    let home = base.home_dir();

    // (app, detect dir, target dir, file ext, claude-style allowed-tools?)
    const APPS: &[(&str, &str, &str, &str, bool)] = &[
        ("claude", ".claude", ".claude/commands", "md", true),
        ("codex", ".codex", ".codex/prompts", "md", false),
        (
            "opencode",
            ".config/opencode",
            ".config/opencode/command",
            "md",
            false,
        ),
        ("gemini", ".gemini", ".gemini/commands", "toml", false),
        ("cursor", ".cursor", ".cursor/commands", "md", false),
    ];

    let mut installed: Vec<String> = Vec::new();
    let mut detected = 0usize;
    for (app, detect, target, ext, with_allowed) in APPS {
        if !home.join(detect).is_dir() {
            continue;
        }
        detected += 1;
        let dir = home.join(target);
        if std::fs::create_dir_all(&dir).is_err() {
            continue;
        }
        let mut wrote = Vec::new();
        for cmd in SLASH_COMMANDS.iter().chain(extra) {
            let content = if *ext == "toml" {
                toml(cmd)
            } else {
                md(cmd, *with_allowed)
            };
            let path = dir.join(format!("{}.{ext}", cmd.name));
            if !path.exists() && std::fs::write(&path, content).is_ok() {
                wrote.push(format!("/{}", cmd.name));
            }
        }
        if !wrote.is_empty() {
            installed.push(format!("{app} ({})", wrote.join(" ")));
        }
    }
    // Always say what happened — a silent installer is undiagnosable
    // (a stale release binary once looked identical to "no agents found").
    if !installed.is_empty() {
        println!("agents  slash commands installed: {}", installed.join(", "));
    } else if detected == 0 {
        println!(
            "agents  no agent CLI config dirs found (~/.claude, ~/.codex, \
             ~/.config/opencode, ~/.gemini, ~/.cursor) — slash commands not installed"
        );
    } else {
        println!("agents  slash commands already present (nothing new to install)");
    }
}

/// Embed pending content; --fetch additionally allows the model download,
/// --rerank (with --fetch) also downloads the reranker.
pub fn embed(fetch: bool, rerank: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    if mimir.ensure_embedder(fetch).is_none() {
        bail!("embedding model unavailable; run `mimir embed --fetch` (or `mimir init`) to download it");
    }
    if rerank {
        if mimir.ensure_reranker(fetch).is_none() {
            bail!("reranker model unavailable; run `mimir embed --fetch --rerank` to download it");
        }
        println!("reranker {} ready", mimir.config.rerank.model);
    }
    let n = mimir.embed_pending()?;
    println!("embedded {n} node(s)");
    Ok(())
}

/// Count tokens in stdin (or the given text) with Mimir's bundled tokenizer —
/// the same counter the savings ledger uses, handy for measuring/benchmarking.
pub fn tokens(text: Vec<String>) -> Result<()> {
    let input = if text.is_empty() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        text.join(" ")
    };
    println!("{}", mimir_core::tokens::count(&input));
    Ok(())
}

pub fn status(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let counts = store::count_by_kind(&mimir.conn)?;
    let db_size = std::fs::metadata(&mimir.paths.db_file)
        .with_context(|| format!("stat {}", mimir.paths.db_file.display()))?
        .len();
    // `project` is the checkout's own; where its memories go is a second
    // answer in a linked worktree, and status says both. Status reads: it
    // reports memories a linked worktree's own row still holds, and leaves
    // moving them to the next memory command.
    let here = mimir.projects_for_cwd(&std::env::current_dir()?)?;
    let (project, detection) = (&here.checkout, &here.detection);
    let leftovers = mimir
        .worktree_leftovers(&here)?
        .filter(|left| !left.is_empty());
    let via = match detection {
        mimir_core::scope::Detection::Found { via, .. } => Some(mimir_core::scope::via_label(via)),
        mimir_core::scope::Detection::NotFound { .. } => None,
    };

    if json {
        let counts_json: serde_json::Map<String, serde_json::Value> = counts
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v)))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "db": mimir.paths.db_file,
                "db_bytes": db_size,
                "project": project.as_ref().and_then(|p| p.title.clone()),
                "project_path": project.as_ref().and_then(|p| p.path.clone()),
                "scope": if project.is_some() { "project" } else { "global" },
                // `scope` is where a new memory goes. Where a read looks is
                // a different answer outside a project, so it has its own key.
                "recall_scope": match mimir_core::scope::query_scope(here.memory_id()) {
                    Scope::Project(_) => "project",
                    Scope::Global => "global",
                    Scope::All => "all",
                },
                "detected_via": via,
                // `project` above is the checkout's own (its code graph).
                // Its memories are filed under `memory_project`: the same
                // project, except in a git linked worktree.
                "checkout": project.as_ref().map(|_| here.checkout_kind.as_str()),
                "worktree_of": here.checkout_kind.home(),
                "memory_project": here.memory.as_ref().and_then(|p| p.title.clone()),
                "memory_project_path": here.memory.as_ref().and_then(|p| p.path.clone()),
                // `home`: a linked worktree's memories go to the main
                // checkout's project; `own_key`: it keeps its own, by the
                // `.mimir` id only it carries; `own`: everything else.
                "memory_filing": project.as_ref().map(|_| here.filing.as_str()),
                // What a linked worktree's own row still holds: `to_move`
                // goes home with the next memory command, `kept` is older
                // than the worktree and stays. `null` when there is none.
                "worktree_memories": leftovers.map(|left| serde_json::json!({
                    "to_move": left.to_move,
                    "kept": left.kept,
                    "worktree_created_at": left.worktree_created_at,
                })),
                "counts": counts_json,
            })
        );
        return Ok(());
    }

    match (project, detection) {
        (Some(p), _) => {
            println!(
                "project {} ({})  [via: {}]",
                p.title.as_deref().unwrap_or("?"),
                p.path.as_deref().unwrap_or("?"),
                via.unwrap_or("?")
            );
            if let Some(note) = here.worktree_note() {
                println!("        {note}");
            }
            if let (Some(left), Some(home)) = (leftovers, &here.memory) {
                for line in left.lines(home) {
                    println!("        {line}");
                }
            }
        }
        // Outside a project the two directions differ, so both are said:
        // what is written is global, what is asked searches everything
        // (`scope::query_scope`).
        (None, mimir_core::scope::Detection::NotFound { from }) => println!(
            "project (none — global scope) — no git root or project marker found above {}; \
             new memories are global, recall searches every project \
             (touch .mimir here to make it a project)",
            from.display()
        ),
        (None, _) => println!(
            "project (none — global scope; new memories are global, recall searches every project)"
        ),
    }
    if counts.is_empty() {
        println!("store   empty");
    } else {
        let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
        println!("store   {}", summary.join(", "));
    }
    println!(
        "db      {} ({} KB)",
        mimir.paths.db_file.display(),
        db_size / 1024
    );
    println!("sync    {}", crate::sync::status_line(&mimir));
    Ok(())
}

pub fn doctor(check_only: bool) -> Result<()> {
    let paths = Paths::resolve()?;
    let mut failures = 0;

    let check = |name: &str, ok: bool, detail: String, failures: &mut i32| {
        if !ok {
            *failures += 1;
        }
        if check_only {
            // Watchdog mode: silent when healthy, failures only, on stderr.
            if !ok {
                eprintln!("FAIL {name}: {detail}");
            }
        } else {
            let mark = if ok { "ok " } else { "FAIL" };
            println!("{mark}  {name}: {detail}");
        }
    };

    match db::open(&paths.db_file) {
        Ok(conn) => {
            check(
                "db",
                true,
                paths.db_file.display().to_string(),
                &mut failures,
            );
            let integrity: String = conn
                .query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .unwrap_or_else(|e| format!("error: {e}"));
            check("integrity", integrity == "ok", integrity, &mut failures);
            let fts = conn
                .prepare("SELECT count(*) FROM node_fts")
                .and_then(|mut s| s.query_row([], |r| r.get::<_, i64>(0)));
            check(
                "fts5",
                fts.is_ok(),
                fts.map(|n| format!("{n} rows indexed"))
                    .unwrap_or_else(|e| e.to_string()),
                &mut failures,
            );
            // node_fts is an external-content table, so PRAGMA integrity_check
            // can pass while the index has drifted from `node` — the state
            // where recall silently returns nothing. Only the two-argument
            // form of FTS5's own command compares the index against the
            // content table (the one-argument form checks internal structure
            // only and passes on drift — verified). Needs SQLite ≥3.42;
            // guaranteed by the bundled rusqlite build.
            let fts_index = conn.execute(
                "INSERT INTO node_fts(node_fts, rank) VALUES('integrity-check', 1)",
                [],
            );
            check(
                "fts5 index",
                fts_index.is_ok(),
                fts_index
                    .map(|_| "consistent with node table".into())
                    .unwrap_or_else(|e| {
                        format!(
                            "out of sync with node table ({e}) — rebuild with: sqlite3 {} \
                             \"INSERT INTO node_fts(node_fts) VALUES('rebuild')\"",
                            paths.db_file.display()
                        )
                    }),
                &mut failures,
            );
            // Informational, and never a failure: a refusal is the guard
            // working. What's worth seeing is a *retry* — one value offered
            // repeatedly means something upstream keeps trying to store it
            // and doesn't know it's being turned away. The two numbers cover
            // different spans, and the line says so: the values are the ones
            // refused in the last 30 days, the attempts are every offer of
            // those values since each was first seen (the ledger keeps one
            // running count per value, not one row per offer).
            let month_ago = mimir_core::model::now_unix() - 30 * 86_400;
            if let Ok((distinct, offers)) = mimir_core::secrets::refusal_counts(&conn, month_ago) {
                if distinct > 0 {
                    check(
                        "secret guard",
                        true,
                        format!(
                            "{distinct} value(s) refused in the last 30d, offered {offers} \
                             time(s) in all since first seen{} (fingerprints only — the \
                             values were never stored; `mimir refusals` for detail)",
                            if offers > distinct {
                                " — something is retrying"
                            } else {
                                ""
                            }
                        ),
                        &mut failures,
                    );
                }
            }
            // Also informational. Stale grounding is the one signal here
            // that Mimir derived by checking rather than by policy: the
            // memory said it was about a symbol, and the symbol is gone.
            if let Ok((grounded, stale, ungrounded)) = mimir_core::grounding::tally(&conn) {
                if grounded + stale + ungrounded > 0 {
                    check(
                        "grounding",
                        true,
                        format!(
                            "{grounded} grounded, {stale} stale, {ungrounded} ungrounded{}",
                            if stale > 0 {
                                " — `mimir grounding --stale` to review"
                            } else {
                                ""
                            }
                        ),
                        &mut failures,
                    );
                }
            }
            // The nudge that grounding actually depends on. Anchors sat at
            // zero adoption for months because the only way to set one was
            // a command nobody knew to run; this is the same failure, and
            // the fix is to say so at the moment someone is already looking
            // at the store's health rather than in a README they read once.
            if let Some(detail) = link_scan_hint(&conn) {
                check("link scan", true, detail, &mut failures);
            }
            // Two more things a store can be in that nothing else reports.
            // Both are hints, not failures: the store is sound, only slower
            // and larger than it needs to be. Watchdog mode prints no hint,
            // so it does not pay for them either: the second one stats every
            // project directory and counts the rows of the missing ones.
            if !check_only {
                if let Ok(missing) = db::missing_indexes(&conn) {
                    if !missing.is_empty() {
                        check(
                            "indexes",
                            true,
                            format!(
                                "{} lookup index(es) not built yet ({}), so removing any \
                                 row scans the whole store — run `mimir init` once to build \
                                 them",
                                missing.len(),
                                missing.join(", ")
                            ),
                            &mut failures,
                        );
                    }
                }
                if let Some(detail) = missing_projects_hint(&conn) {
                    check("projects", true, detail, &mut failures);
                }
            }
            // Same argument for the review queue: findings nobody is told
            // about are findings nobody acts on, which is the state
            // `consolidate`'s contradiction report was already in. `mimir
            // review` runs its detectors before it lists, so the count here
            // does too — otherwise this line says "nothing waiting" (by
            // saying nothing) about a queue the command it points at would
            // fill. Watchdog mode prints no such line and skips the scan.
            if !check_only {
                if let Ok(m) = Mimir::open() {
                    let _ = mimir_core::review::scan(&m.conn);
                    if let Some(detail) = crate::review_cmd::summary(&m) {
                        check("review", true, detail, &mut failures);
                    }
                }
            }
        }
        Err(e) => check("db", false, e.to_string(), &mut failures),
    }

    // The config every command loads. A file that does not parse stops all
    // of them; a mode nobody recognizes is quieter — the setting falls back
    // (see `Config::issues`) and this is the surface that says so, one line
    // per value, with what it is behaving as meanwhile.
    let config = match Config::load(&paths.config_file) {
        Ok(config) => {
            let issues = config.issues();
            if issues.is_empty() {
                check(
                    "config",
                    true,
                    paths.config_file.display().to_string(),
                    &mut failures,
                );
            }
            for issue in issues {
                // A value with a fallback is a misspelling of a setting that
                // still behaves as something documented: worth a FAIL line
                // to whoever runs `doctor`, not worth the watchdog raising
                // an alarm every time it runs on a store that is sound.
                // Only a value with no fallback (the proxy will not start)
                // fails the unattended check.
                if check_only && issue.behaves_as.is_some() {
                    continue;
                }
                check("config", false, issue.to_string(), &mut failures);
            }
            config
        }
        Err(e) => {
            check("config", false, e.to_string(), &mut failures);
            Config::default()
        }
    };

    // Everything below is informational (can never fail the exit code), so
    // watchdog mode skips it — including the daemon probe's network wait.
    if !check_only {
        println!(
            "ok   gpu: {}",
            mimir_core::embed::gpu_backend()
                .unwrap_or("not compiled in (CPU; rebuild with --features gpu-webgpu or gpu-cuda)")
        );

        // The engine's own predicate: the ready marker of the *configured*
        // model. A models directory that merely has something in it — a
        // reranker, another embedder, half a download — is not a model
        // search can load.
        let model = &config.embedding.model;
        check(
            "model",
            true, // informational until embeddings land; BM25-only is a valid state
            if mimir_core::embed::model_ready(&paths, model) {
                format!("{model} present at {}", paths.models_dir.display())
            } else {
                format!(
                    "{model} not downloaded (search is BM25-only until `mimir init` fetches it)"
                )
            },
            &mut failures,
        );

        // Informational only (ok=true regardless): whether `mimir daemon` /
        // `mimir mcp --http` is actually up. Absence is a normal, supported
        // state — the hooks fall back to the cold `mimir recall-inject` path —
        // so this must never fail `doctor`'s exit code, same precedent as the
        // "model" check above. Probed the way the hook script probes it: the
        // same address (`DaemonAddr`), and the same bearer when
        // MIMIR_HTTP_TOKEN is set, so "warm" here means warm for the hook.
        let addr = crate::infer::DaemonAddr::resolve(&config);
        let inject_url = addr.inject_url();
        let delegation = config.daemon.inference();
        let token = crate::mcp::http_token_from_env();
        let mut probe = ureq::get(inject_url).timeout(std::time::Duration::from_secs(1));
        if let Some(token) = &token {
            probe = probe.set("Authorization", &format!("Bearer {token}"));
        }
        check(
            "daemon",
            true,
            match probe.call() {
                Ok(_) => format!(
                    "warm ({inject_url} reachable — hooks use the fast HTTP path; \
                     inference delegation: {delegation})"
                ),
                Err(ureq::Error::Status(401, _)) => format!(
                    "cold ({inject_url} answers 401 — the daemon requires a bearer token and \
                     MIMIR_HTTP_TOKEN {}; hooks fall back to `mimir recall-inject`; \
                     inference delegation: {delegation})",
                    if token.is_some() {
                        "in this environment is not the one it was started with"
                    } else {
                        "is not set in this environment"
                    }
                ),
                Err(ureq::Error::Status(code, _)) => format!(
                    "cold ({inject_url} answers HTTP {code}, which a Mimir daemon does not — \
                     hooks fall back to `mimir recall-inject`; inference delegation: \
                     {delegation})"
                ),
                Err(_) => format!(
                    "cold ({inject_url} not reachable — hooks fall back to `mimir recall-inject`; \
                     run `mimir daemon` for the warm path; inference delegation: {delegation})"
                ),
            },
            &mut failures,
        );

        // Installed hook scripts against the ones this binary ships. A
        // script is a copy made at `init` time, so an upgraded binary keeps
        // running the old copy until someone re-installs — and nothing else
        // would ever say so.
        if let Some(base) = directories::BaseDirs::new() {
            let hooks_dir = base.home_dir().join(".claude").join("hooks");
            let installed = HOOK_SCRIPTS
                .iter()
                .filter(|(name, ..)| hooks_dir.join(name).is_file())
                .count();
            let outdated = outdated_hook_scripts(&hooks_dir);
            if !outdated.is_empty() {
                let list: Vec<String> = outdated
                    .iter()
                    .map(|(name, found, shipped)| format!("{name} v{found}, shipped v{shipped}"))
                    .collect();
                check(
                    "hooks",
                    true,
                    format!(
                        "{} of {installed} installed script(s) older than this binary ({}) — \
                         re-run `mimir init --hooks`",
                        outdated.len(),
                        list.join("; ")
                    ),
                    &mut failures,
                );
            } else if installed > 0 {
                check(
                    "hooks",
                    true,
                    format!("{installed} installed script(s) match this binary"),
                    &mut failures,
                );
            }
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    Ok(())
}

// ---------- memory verbs ----------

#[allow(clippy::too_many_arguments)]
pub fn remember(
    json: bool,
    text: String,
    mtype: &str,
    tags: Vec<String>,
    global: bool,
    force: bool,
    link_ref: Option<String>,
    fires_when: Vec<String>,
    anchors: Vec<String>,
    expires_in: Option<String>,
    resolves_when: Option<String>,
    confidence: Option<String>,
    valid_from: Option<String>,
    valid_to: Option<String>,
) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let mtype: MemoryType = mtype.parse()?;
    // The checkout this was run from resolves a symbol name given to
    // `--link`, global memory or not; `--global` only decides where the
    // memory itself lives. A global capture therefore must not fail because
    // the directory it was typed in could not be resolved to a project.
    //
    // Two projects, then: the checkout's own for the symbol lookup, and the
    // memory project for the memory, which in a linked worktree is the
    // main checkout's.
    let detected = mimir.projects_for_cwd(&std::env::current_dir()?);
    let (working_project, memory_project) = if global {
        detected
            .ok()
            .map_or((None, None), |p| (p.checkout_id(), p.memory_id()))
    } else {
        let here = detected?;
        (here.checkout_id(), here.memory_id())
    };
    let outcome = capture::capture(
        &mut mimir,
        Capture {
            text,
            mtype,
            tags,
            project_id: if global { None } else { memory_project },
            working_project,
            force,
            actor: mimir_core::model::Actor::Human,
            surface: mimir_core::secrets::Surface::CliRemember,
            link: link_ref,
            missing_link: MissingLink::Reject,
            fires_when,
            anchors,
            expires_in,
            resolves_when,
            confidence,
            valid_from,
            valid_to,
        },
    )
    .map_err(capture::Rejected::into_anyhow)?;
    let projects = store::project_titles(&mimir.conn)?;
    let snippet = mimir.config.output.snippet_chars;
    match outcome {
        Captured::Stored(stored) => {
            if json {
                let mut value = node_json(&stored.node, &projects);
                stored.extend_json(&mut value);
                println!("{value}");
            } else {
                println!("{}", line(&stored.node, &projects, snippet));
                for detail in stored.detail_lines(|n| line(n, &projects, 0)) {
                    println!("{detail}");
                }
            }
            Ok(())
        }
        Captured::Duplicate {
            node: existing,
            rearmed_handoff,
        } => bail!(
            "refused: near-duplicate of\n  {}\n{}",
            line(&existing, &projects, snippet),
            if rearmed_handoff {
                capture::REARMED_HANDOFF
            } else {
                "use --force to store anyway"
            }
        ),
        Captured::Forgotten { node: gone, on, .. } => bail!(
            "refused: this was forgotten on {}\n  {}\nuse --force to bring it back deliberately",
            mimir_core::format::full_date(on),
            line(&gone, &projects, snippet)
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn recall(
    json: bool,
    text: String,
    kind: &str,
    global: bool,
    all: bool,
    since: Option<String>,
    limit: Option<usize>,
    full: bool,
    rerank: bool,
    linked: bool,
    min_score: Option<f64>,
    include_superseded: bool,
    as_of: Option<i64>,
) -> Result<()> {
    let mut mimir = Mimir::open()?;
    // Memories and code and docs are read in two scopes that differ in a
    // linked worktree. The query's own scope is the memories', which is
    // what the miss ledger files this question under.
    let place = read_reach(&mimir, global, all)?;
    let reach = place.reach;
    let query = SearchQuery {
        scope: reach.memories,
        kinds: parse_kind_filter(kind)?,
        since: since.map(|s| parse_since(&s)).transpose()?,
        limit: limit.unwrap_or(mimir.config.output.default_limit),
        strength_alpha: mimir.config.scoring.strength_alpha,
        recency_alpha: mimir.config.scoring.recency_alpha,
        type_prior_alpha: mimir.config.scoring.type_prior_alpha,
        code_damp: mimir.config.scoring.code_damp,
        include_superseded,
        as_of,
        text,
    };
    let mimir_core::SearchOutcome {
        mut hits,
        legs,
        rerank: reranked,
    } = mimir.search_with_outcome_within(&query, reach, rerank)?;
    // Another checkout's session handoff is not this session's to find; it
    // does not count as an answer either.
    hits.retain(|hit| place.shows(&hit.node));
    // `--rerank` was asked for by name, so falling back to the fused order
    // is said out loud — on stderr, where it cannot end up in a pipe that
    // parses the hits.
    if let Some(note) = reranked.note().filter(|_| rerank) {
        eprintln!("--rerank: {note}");
    }
    let mut strengths = mimir_core::strength::classify_page(&hits, &query.text, &legs);
    // Judged on the page as retrieved, before `--min-score` trims it: a
    // threshold the caller chose is not evidence the store lacks an answer.
    // Which recalls count at all is `misses::should_observe`'s call, shared
    // with the MCP tool.
    if mimir_core::misses::should_observe(mimir.config.learn.log_misses, &query) {
        mimir_core::misses::observe(
            &mimir.conn,
            &query.text,
            mimir_core::misses::scope_project(&query.scope),
            &strengths,
            mimir_core::model::now_unix(),
        );
    }
    if let Some(min) = min_score {
        (hits, strengths) = hits
            .into_iter()
            .zip(strengths)
            .filter(|(hit, _)| hit.score >= min)
            .unzip();
    }

    let query_hash = blake3::hash(query.text.as_bytes());
    let shown: Vec<(i64, i64, f64)> = hits
        .iter()
        .enumerate()
        .map(|(rank, hit)| (hit.node.id, rank as i64, hit.score))
        .collect();
    store::record_shown(&mimir.conn, query_hash.as_bytes(), &shown)?;
    if mimir.config.learn.shadow_ranking {
        mimir_core::shadow::observe(
            &mimir.conn,
            query_hash.as_bytes(),
            &hits,
            mimir_core::model::now_unix(),
        );
    }
    if let Some(report) = mimir_core::consolidate::maybe_auto(
        &mimir.conn,
        &mimir.config.embedding.model,
        &mimir.config.consolidate.auto,
    ) {
        if !report.is_empty() {
            eprintln!(
                "(consolidated: {} superseded, {} distilled, {} archived)",
                report.superseded, report.distilled, report.archived
            );
        }
    }

    let projects = store::project_titles(&mimir.conn)?;
    if let Some(at) = as_of {
        if !json {
            println!(
                "as of {} — what the store held then, not what was true then. \
                 A hit is shown in the wording it had then, but it is found by \
                 the wording it has now. Ranking is more lexical than usual: \
                 the semantic index only covers memories that are still live.",
                mimir_core::format::full_date(at)
            );
        }
    }
    if hits.is_empty() && !json {
        println!("no results");
        return Ok(());
    }
    // One lookup for the whole page; see `grounding::stale_ids`.
    let stale = mimir_core::grounding::stale_ids(
        &mimir.conn,
        &hits.iter().map(|h| h.node.id).collect::<Vec<_>>(),
    )
    .unwrap_or_default();
    for (hit, strength) in hits.iter().zip(&strengths) {
        if json {
            let mut value = node_json(&hit.node, &projects);
            value["score"] = serde_json::json!(hit.score);
            // On a recall hit `strength` is the match label ("strong" /
            // "weak"), where every other JSON surface puts the node's
            // learned strength under that key. Scripts read the label
            // there, so it stays; the number keeps a key of its own
            // instead of being overwritten out of the output.
            value["node_strength"] = serde_json::json!(hit.node.strength);
            value["strength"] = serde_json::json!(strength.as_str());
            println!("{value}");
        } else if full {
            print_full(&hit.node, &mimir, &projects)?;
            println!();
        } else {
            println!(
                "{}",
                line_with_grounding(
                    &hit.node,
                    &projects,
                    mimir.config.output.snippet_chars,
                    Some(&query.text),
                    mimir_core::format::HitMarks {
                        stale_grounding: stale.contains(&hit.node.id),
                        weak_match: *strength == mimir_core::strength::MatchStrength::Weak,
                    }
                )
            );
        }
        if linked && !json {
            // `edges_of` is an unordered `src=?1 OR dst=?1`, so the four slots
            // go to whatever the table hands back first. A node can carry 32
            // derived co_recall edges and only one or two authored ones, which
            // is exactly the wrong way round for a flag whose whole job is to
            // show the links somebody drew. The "often recalled with" line in
            // the full record is where the derived ones belong.
            for edge in store::edges_of(&mimir.conn, hit.node.id)?
                .iter()
                .filter(|e| e.rel != Rel::CoRecalled)
                .take(4)
            {
                let other_id = if edge.src == hit.node.id {
                    edge.dst
                } else {
                    edge.src
                };
                let Ok(other) = store::get_node(&mimir.conn, other_id) else {
                    continue;
                };
                let l = match other.kind {
                    Kind::Symbol => mimir_graph::symbol_line(&other),
                    _ => line(&other, &projects, 60),
                };
                println!("  ~{} {}", edge.rel, l);
            }
        }
    }
    Ok(())
}

/// Print at most one relevant memory for `prompt`, or nothing if none
/// clears the relevance floor. This is the COLD fallback path: it opens a
/// fresh `Mimir` (ONNX load + full matrix rebuild) every invocation, so the
/// hook script (`MIMIR_RECALL_SH`) tries the warm `/inject` HTTP endpoint
/// first (see `mcp.rs::inject_router`, only live while `mimir mcp --http`
/// is running) and only falls back to this command when that's
/// unreachable. All relevance-floor/formatting/budget logic lives in
/// `mimir_core::inject::compute`/`compute_with_mode` — single-sourced so the
/// warm and cold paths can never disagree on *how* a hit is judged, only on
/// whether a vector leg is available to judge it with.
///
/// Config `[hooks] cold_mode` governs whether this cold path pays the
/// embedder's load cost:
///   - `"fast"` (default): BM25-only, via `compute_with_mode(.., bm25_only
///     = true)` — never calls `ensure_embedder`, so no ONNX load and no
///     matrix build. This is the mode actually measured for hook latency
///     (see CHANGELOG); an unrecognized value behaves as `"full"`
///     (`ColdMode::UNKNOWN`) and is reported by `mimir doctor`, not by
///     every prompt's hook.
///   - `"full"`: restores the pre-`cold_mode` behavior — hybrid search with
///     the embedder loaded cold, same as the warm endpoint.
///
/// Only this cold CLI path reads `cold_mode`; the warm `/inject` endpoint
/// always uses the best available signal via the unchanged `compute`.
///
/// `enrich`: optional working-tree signal (changed-file stems from
/// `MIMIR_RECALL_SH`'s `git diff`), passed straight through to
/// `inject::compute_with_mode` — see that function's doc comment for how
/// it's used.
///
/// `session`: the per-session dedup key. An empty one is no key, as on the
/// warm path — an empty key would be one ledger shared by every session
/// that sent none.
///
/// A hook entry point, so it never fails the prompt it runs on: an error
/// is a line on stderr and exit 0 (`context_guard_cmd::fail_open`).
pub fn recall_inject(
    prompt: String,
    enrich: Option<String>,
    session: Option<String>,
) -> Result<()> {
    crate::context_guard_cmd::fail_open("recall-inject", || {
        let mut mimir = Mimir::open()?;
        let place = read_reach(&mimir, false, false)?;
        let checkout = place.handoffs_of.flatten();
        let enrich = enrich.unwrap_or_default();
        let bm25_only = mimir.config.hooks.cold_mode() == ColdMode::Fast;
        if let Some(text) = mimir_core::inject::compute_in_checkout(
            &mut mimir,
            &prompt,
            &enrich,
            place.reach.memories,
            bm25_only,
            session.as_deref().filter(|s| !s.is_empty()),
            checkout.as_deref(),
        )? {
            println!("{text}");
        }
        Ok(())
    })
}

/// `mimir daemon` — a thin, discoverable alias for `mimir mcp --http <addr>`.
/// The bind address is `infer::DaemonAddr`: `config.hooks.inject_url` (the
/// exact same key `MIMIR_RECALL_SH` already reads — see
/// `render_recall_script`) unless `MIMIR_INJECT_URL` overrides it, as it
/// does for the hook script, the delegation client and `doctor`. So there
/// is one answer to "where does the warm path live" instead of several
/// that could drift apart. No auto-spawn, no process supervision: this is
/// purely a memorable name for a command the hooks already know how to
/// fall back from — see `contrib/mimir-daemon.service` for running it
/// unattended.
///
/// Loopback only. `daemon` takes no flags, so an address that is not
/// loopback is refused here, with the command that can serve it, rather
/// than by the transport's own refusal, which recommends a flag this
/// command does not have.
pub fn daemon() -> Result<()> {
    let mimir = Mimir::open()?;
    let daemon_addr = crate::infer::DaemonAddr::resolve(&mimir.config);
    let addr = daemon_addr.bind()?;
    if !crate::mcp::bind_is_loopback(&addr) {
        bail!(
            "refusing to run the daemon on non-loopback address `{addr}` (from {}): its HTTP \
             surface gives anyone who can reach the port full read/write access to the store. \
             Point it at 127.0.0.1 — or, if the exposure is deliberate (e.g. a container port \
             published to 127.0.0.1 only), serve the same endpoints with \
             `mimir mcp --http {addr} --http-allow-remote`.",
            daemon_addr.origin()
        );
    }
    println!(
        "mimir daemon: warm path at http://{addr}/inject — the auto-recall hook will use \
         this instead of the cold `mimir recall-inject` fallback; /embed and /rerank serve \
         inference delegation for MCP sessions ([daemon] inference = \"auto\")"
    );
    // Same HTTP surface as `mimir mcp --http`, so honor the same env-var
    // bearer gate (there is no --http-token flag on daemon by design).
    crate::mcp::run(Some(addr), false, crate::mcp::http_token_from_env())
}

/// Strip `config.hooks.inject_url` (e.g. `"http://127.0.0.1:8077/inject"`)
/// down to the bare `host:port` that `mimir mcp --http` binds to: the
/// configured layer of `infer::DaemonAddr::bind`, where the parsing lives.
#[cfg(test)]
fn inject_addr(inject_url: &str) -> Result<String> {
    crate::infer::DaemonAddr::from_parts(inject_url, None).bind()
}

pub fn get(json: bool, refs: Vec<String>) -> Result<()> {
    let mimir = Mimir::open()?;
    let projects = store::project_titles(&mimir.conn)?;
    for (i, r) in refs.iter().enumerate() {
        if i > 0 && !json {
            println!();
        }
        if let Some(slice) = mimir_core::index::file_slice(&mimir.conn, r)? {
            println!("{slice}");
            continue;
        }
        let node = store::resolve_ref(&mimir.conn, r)?;
        mimir_core::learn::record_opened(&mimir.conn, node.id)?;
        if json {
            println!("{}", node_json(&node, &projects));
        } else {
            print_full(&node, &mimir, &projects)?;
        }
    }
    Ok(())
}

pub fn list(
    json: bool,
    mtype: Option<String>,
    tag: Option<String>,
    global: bool,
    all: bool,
    limit: usize,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let place = read_reach(&mimir, global, all)?;
    let scope = place.reach.memories;
    let mtype = mtype.map(|t| t.parse::<MemoryType>()).transpose()?;
    let nodes = match &place.handoffs_of {
        None => memory::list(&mimir.conn, scope, mtype, tag.as_deref(), limit)?,
        Some(checkout) => mimir_core::context_guard::list_in_checkout(
            &mimir.conn,
            scope,
            mtype,
            tag.as_deref(),
            limit,
            checkout.as_deref(),
        )?,
    };
    let projects = store::project_titles(&mimir.conn)?;
    if nodes.is_empty() && !json {
        println!("no memories");
        return Ok(());
    }
    for node in &nodes {
        if json {
            println!("{}", node_json(node, &projects));
        } else {
            println!(
                "{}",
                line(node, &projects, mimir.config.output.snippet_chars)
            );
        }
    }
    Ok(())
}

pub fn mark(reference: &str, useful: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = resolve_any(&mimir, reference)?;
    let strength = mimir_core::learn::apply_mark(&mimir.conn, node.id, useful)?;
    println!(
        "{} {} → strength {strength:.2}",
        short_uid(node.kind, &node.uid),
        if useful { "useful" } else { "noise" },
    );
    Ok(())
}

/// Resolve a `--link` / `link:` target by node id *or* by symbol name.
///
/// Both surfaces advertise "a code symbol or node", but only ever called
/// `store::resolve_ref`, which resolves ids and nothing else — so linking a
/// memory to `retry_with_backoff` failed with "no node matching", and the
/// only links anyone could make by hand were between things they already
/// had ids for. Since a link to an indexed symbol is exactly what makes a
/// memory *grounded* (see `mimir_core::grounding`), leaving this broken
/// would have shipped a grounding feature that nearly nothing could reach —
/// the same shape as the anchors that sat at zero adoption because the only
/// way to set one was at capture time.
pub(crate) fn resolve_link_target(
    conn: &rusqlite::Connection,
    reference: &str,
    project_id: Option<i64>,
) -> Result<Node> {
    match store::resolve_ref(conn, reference) {
        Ok(node) => Ok(node),
        Err(err) => {
            let Some(pid) = project_id else {
                return Err(err.into());
            };
            // Symbol lookup is project-scoped; outside a project there is
            // nothing to search, so report the original id-shaped error.
            mimir_graph::resolve_symbol(conn, pid, reference).map_err(|_| err.into())
        }
    }
}

pub fn grounding(stale_only: bool, limit: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    let (grounded, stale, ungrounded) = mimir_core::grounding::tally(&mimir.conn)?;
    println!("{grounded} grounded, {stale} stale, {ungrounded} ungrounded");
    if !stale_only {
        println!(
            "\nGrounded means the memory links to something Mimir indexes and can \
             re-check.\nUngrounded is normal — most notes aren't about a specific \
             symbol or file."
        );
        return Ok(());
    }
    let rows = mimir_core::grounding::stale_memories(&mimir.conn, limit)?;
    if rows.is_empty() {
        println!("\nnothing stale");
        return Ok(());
    }
    let projects = store::project_titles(&mimir.conn)?;
    println!();
    for (node, target) in &rows {
        println!(
            "{}",
            line(node, &projects, mimir.config.output.snippet_chars)
        );
        println!("    was about: {target} (no longer indexed)");
    }
    println!(
        "\nStale means the thing it pointed at is gone, not that the memory is \
         wrong.\nRe-link with `mimir link`, retire with `mimir supersede`, or leave it."
    );
    Ok(())
}

/// The miss ledger: recall queries that came back with no strong match.
pub fn misses(
    dismiss: Option<i64>,
    all: bool,
    resolved: bool,
    limit: usize,
    json: bool,
) -> Result<()> {
    let mimir = Mimir::open()?;
    if let Some(id) = dismiss {
        if !mimir_core::misses::dismiss(&mimir.conn, id)? {
            return Err(anyhow!("no miss #{id} (see `mimir misses`)"));
        }
        println!("dismissed miss #{id}");
        return Ok(());
    }
    let project = if all {
        None
    } else {
        mimir_core::misses::scope_project(&read_scope(&mimir, false, false)?)
    };
    let rows = mimir_core::misses::list(&mimir.conn, project, resolved, limit)?;
    let projects = store::project_titles(&mimir.conn)?;
    if json {
        for m in &rows {
            println!(
                "{}",
                serde_json::json!({
                    "id": m.id,
                    "query": m.query,
                    "project": m.project_id.and_then(|id| projects.get(&id)),
                    "hits": m.hits,
                    "count": m.count,
                    "first_at": m.first_at,
                    "last_at": m.last_at,
                    "resolved_at": m.resolved_at,
                })
            );
        }
        return Ok(());
    }
    if rows.is_empty() {
        println!("no misses — every recall logged here found a strong match");
        if !mimir.config.learn.log_misses {
            println!("(logging is off: `[learn] log_misses = false`)");
        }
        return Ok(());
    }
    println!(
        "{:>5} {:>5}  {:<6} {:<12} query",
        "id", "asked", "last", "project"
    );
    for m in &rows {
        let project = m
            .project_id
            .and_then(|id| projects.get(&id))
            .map_or("-", String::as_str);
        let state = match (m.resolved_at, m.hits) {
            (Some(_), _) => "  (answered since)",
            (None, 0) => "  (nothing came back)",
            (None, _) => "",
        };
        println!(
            "{:>5} {:>5}  {:<6} {:<12} {}{state}",
            m.id,
            m.count,
            mimir_core::format::month_day(m.last_at),
            project,
            m.query
        );
    }
    println!(
        "\nEach of these got no strong match: nothing that both shares the query's \
         words and ranks near the top of both search legs.\nWrite the memory that \
         answers it (`mimir remember`), or `mimir misses --dismiss <id>` if it was noise."
    );
    Ok(())
}

/// Shadow-ranking scoreboard: each candidate against the live ranking.
pub fn shadow(days: Option<u32>, json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let since = days.map_or(0, |d| mimir_core::model::now_unix() - i64::from(d) * 86_400);
    let e = mimir_core::shadow::evaluate(&mimir.conn, mimir_core::shadow::COACTIVATION, since)?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "candidate": e.candidate,
                "pages": e.pages,
                "decided": e.decided,
                "live_mrr": e.live_mrr,
                "shadow_mrr": e.shadow_mrr,
                "wins": e.wins,
                "losses": e.losses,
                "ties": e.ties,
                "sign_test_p": e.sign_test_p(),
                "verdict": e.verdict(),
            })
        );
        return Ok(());
    }
    println!(
        "candidate: {} — boost a memory co-recalled with one opened in the last 30 min",
        e.candidate
    );
    println!(
        "pages it would have reordered: {}   decided by an open: {}",
        e.pages, e.decided
    );
    if e.decided > 0 {
        println!(
            "MRR of the opened hit   live {:.3}   shadow {:.3}",
            e.live_mrr, e.shadow_mrr
        );
        let p = e
            .sign_test_p()
            .map_or("-".to_string(), |p| format!("{p:.3}"));
        println!(
            "shadow ranked it higher {} · lower {} · same {}   (sign test p = {p})",
            e.wins, e.losses, e.ties
        );
    }
    println!("verdict: {}", e.verdict());
    if !mimir.config.learn.shadow_ranking {
        println!("(shadow ranking is off: `[learn] shadow_ranking = false`)");
    }
    println!(
        "\nAgents open what they were shown near the top, and they were shown the live \
         order — so this comparison favours live.\nA shadow lead is real evidence; a \
         shadow deficit is not evidence of harm. Nothing goes live from here on its own."
    );
    Ok(())
}

pub fn refusals(limit: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    let rows = mimir_core::secrets::refusals(&mimir.conn, limit)?;
    if rows.is_empty() {
        println!("nothing refused");
        return Ok(());
    }
    println!(
        "{:<24} {:<14} {:>6}  {:<12} {:<12}",
        "kind", "surface", "offers", "first", "last"
    );
    for r in &rows {
        println!(
            "{:<24} {:<14} {:>6}  {:<12} {:<12}",
            r.kind,
            r.surface,
            r.count,
            mimir_core::format::full_date(r.first_seen),
            mimir_core::format::full_date(r.last_seen)
        );
    }
    println!(
        "\nfingerprints only — the refused values were never written to disk, \
         so there is nothing here to recover them from."
    );
    Ok(())
}

/// What was done to memories, and by whom. Reads the mutation ledger.
///
/// Prints hashes as short prefixes rather than in full: the point of the
/// column is to tell two values apart and to match one you already hold, and
/// 64 hex characters per row would make the table unreadable while adding
/// nothing a `mimir audit --json` consumer can't get.
pub fn audit(reference: Option<&str>, limit: usize, json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let rows = match reference {
        Some(r) => {
            let node = store::resolve_ref_including_deleted(&mimir.conn, r)?;
            mimir_core::audit::history(&mimir.conn, node.id)?
        }
        None => mimir_core::audit::recent(&mimir.conn, limit)?,
    };
    // `--json` prints one object per row and nothing else, so an empty
    // ledger is empty output, as it is for `list` and `recall`; the sentence
    // is for a person reading a terminal.
    if rows.is_empty() && !json {
        println!("nothing recorded");
        return Ok(());
    }
    if json {
        for m in &rows {
            println!(
                "{}",
                serde_json::json!({
                    "at": m.at,
                    "node_id": m.node_id,
                    "op": m.op.as_str(),
                    "actor": m.actor.as_str(),
                    "reason": m.reason,
                    "before_hash": m.before_hash.as_deref().map(mimir_core::format::hex),
                    "after_hash": m.after_hash.as_deref().map(mimir_core::format::hex),
                })
            );
        }
        return Ok(());
    }
    println!(
        "{:<12} {:<10} {:<7} {:<10} why",
        "when", "op", "actor", "value"
    );
    for m in &rows {
        println!(
            "{:<12} {:<10} {:<7} {:<10} {}",
            mimir_core::format::full_date(m.at),
            m.op.as_str(),
            m.actor.as_str(),
            m.before_hash
                .as_ref()
                .map(|h| mimir_core::format::hex(&h[..4]))
                .unwrap_or_else(|| "-".into()),
            m.reason.as_deref().unwrap_or("")
        );
    }
    println!(
        "\nthe `value` column is a hash prefix, not the text — a record of a \
         deletion that quotes what was deleted is a second copy of it."
    );
    Ok(())
}

/// A memory's life: every recorded change, and every wording it has had.
///
/// The atlas's criterion D — "correction is not amnesia", so
/// "where did I *use* to live?" has to keep working after a correction.
pub fn history(reference: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = store::resolve_ref_including_deleted(&mimir.conn, reference)?;
    let changes = mimir_core::audit::history(&mimir.conn, node.id)?;
    let versions = mimir_core::revision::history(&mimir.conn, node.id)?;

    println!(
        "{} {}",
        short_uid(node.kind, &node.uid),
        node.title.as_deref().unwrap_or("")
    );
    println!(
        "  {} created",
        mimir_core::format::full_date(node.created_at)
    );
    for v in &versions {
        println!(
            "  {} said: {}",
            mimir_core::format::full_date(v.at),
            mimir_core::format::truncate_chars(
                &mimir_core::format::collapse_ws(v.body.as_deref().unwrap_or("")),
                100
            )
        );
    }
    for m in &changes {
        println!(
            "  {} {} by {}{}",
            mimir_core::format::full_date(m.at),
            m.op.as_str(),
            m.actor.as_str(),
            m.reason
                .as_deref()
                .map(|r| format!(" — {r}"))
                .unwrap_or_default()
        );
    }
    if versions.is_empty() && changes.is_empty() {
        println!("  (nothing has happened to it since)");
    }
    // Say what this does NOT cover, rather than letting an empty list read as
    // "nothing happened": a deliberate forget purges prior wordings on
    // purpose, so their absence here is the deletion working.
    //
    // Only a deliberate forget, though. Decay archival also tombstones, and
    // it keeps prior wordings on purpose — telling that reader they were
    // purged was simply false.
    match node.lifecycle(now_unix()) {
        Lifecycle::Forgotten => {
            println!("\nforgotten — prior wordings were purged with it, by design.")
        }
        Lifecycle::Archived => println!(
            "\narchived — it decayed while idle and nobody deleted it, so prior wordings are kept."
        ),
        Lifecycle::Live | Lifecycle::Expired | Lifecycle::Superseded => {}
    }
    Ok(())
}

pub fn consolidate(dry_run: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let report =
        mimir_core::consolidate::consolidate(&mimir.conn, &mimir.config.embedding.model, dry_run)?;
    print_consolidate_report(&report, dry_run);
    Ok(())
}

fn print_consolidate_report(report: &mimir_core::consolidate::Report, dry_run: bool) {
    let prefix = if dry_run { "would " } else { "" };
    if report.is_empty() {
        println!("nothing to consolidate");
        return;
    }
    if report.superseded > 0 {
        println!("{prefix}supersede {} near-duplicate(s)", report.superseded);
    }
    if report.distilled > 0 {
        println!(
            "{prefix}distill {} cluster(s) into summaries",
            report.distilled
        );
    }
    if report.archived > 0 {
        println!("{prefix}archive {} decayed memorie(s)", report.archived);
    }
    for (a, b) in &report.contradictions {
        println!("possible contradiction (review by hand):\n  {a}\n  {b}");
    }
}

/// Forget a memory. Which references resolve, and what may be forgotten,
/// is decided in `store::forget` — the same function the MCP tool calls —
/// so `--hard` reaches an existing tombstone and a plain forget turns an
/// archived memory into a forgotten one.
pub fn forget(reference: &str, hard: bool, reason: Option<&str>) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = store::forget(
        &mimir.conn,
        reference,
        hard,
        mimir_core::model::Actor::Human,
        reason,
    )?;
    println!(
        "forgot {} {}{}",
        short_uid(node.kind, &node.uid),
        node.title.as_deref().unwrap_or(""),
        if hard { " (permanently)" } else { "" }
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn edit(
    json: bool,
    reference: &str,
    text: String,
    title: Option<String>,
    mtype: Option<String>,
    tags: Option<Vec<String>>,
    pin: Option<bool>,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = store::resolve_ref(&mimir.conn, reference)?;
    let mtype = mtype.map(|t| t.parse::<MemoryType>()).transpose()?;
    // Same guard as `remember`: an edited body, title or tag set is a
    // capture too.
    let declared: Vec<String> = title.iter().cloned().collect();
    capture::refuse_secrets(
        &mimir.conn,
        &text,
        tags.as_deref().unwrap_or(&[]),
        &declared,
        mimir_core::secrets::Surface::CliEdit,
    )?;
    let edit = memory::Edit {
        text: if text.is_empty() { None } else { Some(text) },
        title,
        mtype,
        tags,
    };
    if edit.text.is_none()
        && edit.title.is_none()
        && edit.mtype.is_none()
        && edit.tags.is_none()
        && pin.is_none()
    {
        bail!("nothing to change: pass TEXT, --title, --type, --tags, or --pin/--unpin");
    }
    if let Some(pin) = pin {
        store::set_pinned(
            &mimir.conn,
            node.id,
            pin,
            mimir_core::model::Actor::Human,
            None,
        )?;
    }
    let updated = memory::edit(
        &mimir.conn,
        node.id,
        edit,
        mimir_core::model::Actor::Human,
        None,
    )?;
    let projects = store::project_titles(&mimir.conn)?;
    if json {
        println!("{}", node_json(&updated, &projects));
    } else {
        println!(
            "{}",
            line(&updated, &projects, mimir.config.output.snippet_chars)
        );
    }
    Ok(())
}

pub fn reproject(json: bool, reference: &str, project: Option<String>, global: bool) -> Result<()> {
    if !global && project.is_none() {
        bail!("pass --project <name> to move it to a project, or --global for global scope");
    }
    let mimir = Mimir::open()?;
    let node = store::resolve_ref(&mimir.conn, reference)?;
    let target: Option<i64> = match project {
        Some(name) => {
            let named = store::find_project_by_title(&mimir.conn, &name)?.ok_or_else(|| {
                anyhow::anyhow!("no project named '{name}' (titles match exactly)")
            })?;
            // A linked worktree's name stands for the project its memories
            // are filed under. Left on the worktree's own row, the memory
            // would be out of every reader's scope until the next time
            // something ran there and moved it.
            let proj = mimir.memory_project_of(&named)?;
            if proj.id != named.id {
                eprintln!(
                    "note: '{name}' is a {}",
                    mimir_core::linked_worktree_note(&proj)
                );
            }
            // A memory only rides sync while global or in a sync-enabled
            // project; moving it into a plain local project silently takes it
            // out of that set, so say so instead of letting peers diverge
            // quietly.
            if !mimir_core::replicate::project_is_syncable(&proj.meta) {
                eprintln!(
                    "note: project '{name}' is not sync-enabled — this memory will stop \
                     syncing; peers that already pulled it keep their current copy"
                );
            }
            Some(proj.id)
        }
        None => None, // --global
    };
    store::reproject(
        &mimir.conn,
        &node,
        target,
        mimir_core::model::Actor::Human,
        None,
    )?;
    let updated = store::get_node(&mimir.conn, node.id)?;
    let projects = store::project_titles(&mimir.conn)?;
    if json {
        println!("{}", node_json(&updated, &projects));
    } else {
        println!(
            "{}",
            line(&updated, &projects, mimir.config.output.snippet_chars)
        );
    }
    Ok(())
}

/// Help text for `mimir link --rel`, built from the relation list so the
/// vocabulary a user reads is the one the guard below enforces.
pub fn rel_help() -> String {
    let accepted: Vec<&str> = Rel::authorable().map(Rel::as_str).collect();
    format!(
        "{}. Not accepted: `supersedes` (use `mimir supersede`), `co_recalled` (the \
         store writes those for itself from what you actually recall), and the \
         relations Mimir derives from the code graph",
        accepted.join("|")
    )
}

/// The relations `mimir link` will not author, and why. Lives apart from
/// `link` because `link` opens the store before it ever looks at the
/// relation, and a refusal that needs a database on disk to exercise is a
/// refusal nobody tests.
fn reject_unauthorable_rel(rel: Rel) -> Result<()> {
    // See the same guard in the MCP `link` tool: the `supersedes` edge and the
    // `superseded_by` column are independent, and only the column hides a node
    // from recall. Allowing it here would silently record a retirement that
    // retires nothing.
    if rel.origin() == RelOrigin::Retirement {
        bail!(
            "`--rel supersedes` records an edge but does NOT retire the old node — \
             it would look like a supersede and change nothing. Use `mimir supersede \
             <old> <new>` instead (it sets both)."
        );
    }
    // What Mimir derives from the things it indexes is rebuilt from them: the
    // extractor deletes a file's `calls` and `imports` edges before it writes
    // the fresh ones, so a hand-made edge of those relations disappears on
    // the next `graph build` — and until then it is indistinguishable from
    // one the extractor resolved.
    if rel.origin() == RelOrigin::Extracted {
        bail!(
            "`--rel {rel}` is not a link you can make by hand: Mimir derives `{rel}` \
             edges from what it indexes (the code graph, communities, tags) and \
             replaces them when it re-reads the source. Use `link --rel relates` or \
             `--rel about` for a link you actually mean."
        );
    }
    // `co_recalled` is not a claim anybody makes: the store writes it for
    // itself when it observes two memories opened together. Hand-authoring one
    // forges a usage record for a co-recall that never happened — and because
    // the observer skips any pair that already has an authored edge, it would
    // also suppress the real edge if that pair ever does get recalled together.
    if rel.origin() == RelOrigin::Observed {
        bail!(
            "`--rel co_recalled` is the store's own observation of memories recalled \
             together — writing one by hand forges usage that never happened, and it \
             would stop the store recording the real co-recall for that pair. Use \
             `link --rel relates` for a link you actually mean."
        );
    }
    Ok(())
}

pub fn link(a: &str, b: &str, rel: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let rel: Rel = rel.parse()?;
    reject_unauthorable_rel(rel)?;
    let src = resolve_any(&mimir, a)?;
    let dst = resolve_any(&mimir, b)?;
    store::link(&mimir.conn, src.id, dst.id, rel, 1.0)?;
    println!(
        "{} —{rel}→ {}",
        short_uid(src.kind, &src.uid),
        short_uid(dst.kind, &dst.uid)
    );
    Ok(())
}

/// Set guard anchors on an existing memory. `remember --anchor` only covers
/// capture time, which is why anchor adoption tends to sit at zero: by the
/// time you know which file a memory guards, the memory already exists.
/// Patterns REPLACE any existing set (same semantics as `set_anchors`).
pub fn anchor(reference: &str, patterns: Vec<String>) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = resolve_any(&mimir, reference)?;
    // The PreToolUse reader only looks at memories, so an anchor on anything
    // else would be accepted here and then never fire.
    if node.kind != Kind::Memory {
        bail!(
            "{} is a {}, not a memory; guard anchors only fire for memories",
            short_uid(node.kind, &node.uid),
            node.kind
        );
    }
    let clean = mimir_core::anchors::sanitize_anchors(patterns);
    if clean.is_empty() {
        bail!("no usable anchor patterns after sanitizing (empty or oversized are dropped)");
    }
    // An anchor is stored, shown and synced like the memory it sits on, so
    // it passes the same guard as `remember --anchor`.
    capture::refuse_secrets(
        &mimir.conn,
        "",
        &[],
        &clean,
        mimir_core::secrets::Surface::CliEdit,
    )?;
    mimir_core::anchors::set_anchors(&mimir.conn, node.id, &clean)?;
    println!(
        "{} anchored to {}",
        short_uid(node.kind, &node.uid),
        clean.join(", ")
    );
    Ok(())
}

/// Mark OLD as superseded by NEW: OLD stops surfacing in recall (kept as
/// history) and a `supersedes` edge is recorded. One operation
/// (`store::supersede`), which also says what it refuses and why.
pub fn supersede(old: &str, by: &str, reason: Option<&str>) -> Result<()> {
    let mimir = Mimir::open()?;
    let old = resolve_any(&mimir, old)?;
    let new = resolve_any(&mimir, by)?;
    store::supersede(
        &mimir.conn,
        old.id,
        new.id,
        mimir_core::model::Actor::Human,
        reason,
    )?;
    println!(
        "{} superseded by {}",
        short_uid(old.kind, &old.uid),
        short_uid(new.kind, &new.uid)
    );
    Ok(())
}

/// `doctor`'s line about projects whose directory is gone, or `None` when
/// there is nothing left behind for any. A deleted worktree or a scratch
/// clone keeps its whole code graph until someone prunes it, and nothing
/// else ever says so.
fn missing_projects_hint(conn: &rusqlite::Connection) -> Option<String> {
    let missing: Vec<_> = mimir_graph::missing_projects(conn)
        .ok()?
        .into_iter()
        .filter(|m| m.has_graph_rows())
        .collect();
    if missing.is_empty() {
        return None;
    }
    let symbols: i64 = missing.iter().map(|m| m.symbols).sum();
    Some(format!(
        "{} project(s) point at a directory that no longer exists and still hold {symbols} \
         symbols — `mimir project prune` lists them",
        missing.len()
    ))
}

/// `doctor`'s line about auto-linking, or `None` when there is nothing
/// useful to say (no code graphs, or a recent scan).
///
/// Deliberately quiet once you have scanned: a health check that repeats
/// advice you already took is one people learn to skim, and then it cannot
/// tell them anything.
fn link_scan_hint(conn: &rusqlite::Connection) -> Option<String> {
    // The same projects `link --scan --all-projects` would range over.
    let graphed = mimir_graph::graphed_projects(conn).ok()?.len();
    if graphed == 0 {
        return None;
    }
    let last: Option<i64> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'last_link_scan'",
            [],
            |r| r.get::<_, String>(0),
        )
        .ok()
        .and_then(|v| v.parse().ok());
    let days = last.map(|t| (mimir_core::model::now_unix() - t) / 86_400);
    match days {
        // Symbols exist and nothing has ever been linked from them.
        None => Some(format!(
            "{graphed} project(s) have a code graph, never scanned — \
             `mimir link --scan --all-projects` links memories to symbols they name"
        )),
        Some(d) if d >= 30 => Some(format!(
            "last scanned {d}d ago; graphs move — \
             `mimir link --scan --all-projects` to catch up"
        )),
        Some(_) => None,
    }
}

/// Auto-link memories to the code symbols their text literally mentions.
///
/// Precision-first: only code-shaped names (snake_case, ::path, CamelCase)
/// on word boundaries; names resolving to more than three symbols are
/// skipped as ambiguous. Idempotent — existing edges are kept.
///
/// `all_projects` exists because of a mismatch measured on a real store:
/// 93% of memories are *global*, but a scan only ever compares them
/// against the graph of the project you happen to be standing in. The same
/// global Rust gotcha can match a symbol in any project; running only in
/// the current one leaves the rest unlinked. Measured: scanning one project
/// took grounding from 1.4% to 2.6%, scanning four took it to 15.4%.
pub fn link_scan(dry_run: bool, all_projects: bool) -> Result<()> {
    let mimir = Mimir::open()?;

    // Each target is a checkout project: the scan reads its symbols. The
    // memories compared against them are its memory project's (see
    // `scan_one_project`).
    let targets: Vec<Node> = if all_projects {
        // Only projects that actually have a graph; the rest cost a query
        // and can match nothing.
        mimir_graph::graphed_projects(&mimir.conn)?
    } else {
        vec![mimir.project_for_cwd(&std::env::current_dir()?)?.context(
            "not inside a project (the scan links memories to this project's symbols) \
                 — or pass --all-projects to scan every project that has a code graph",
        )?]
    };
    if targets.is_empty() {
        println!("no project has a code graph yet — run `mimir graph build` in one first");
        return Ok(());
    }

    // Every edge, once. The inner loop asks "is this pair already linked?"
    // per candidate, and re-reading the table per project turned that into
    // the dominant cost on a store with a hundred thousand symbols.
    //
    // The set is rel-agnostic, so a derived co_recall edge in it would read as
    // "already linked" and cost the pair its real `about` edge, silently. Today
    // co_recalled only joins memory to memory and this scan links memory to
    // symbol, so the two cannot actually collide — the exclusion is here so
    // that stays true if the observer ever learns to relate other kinds.
    let mut existing: std::collections::HashSet<(i64, i64)> = Default::default();
    {
        let mut stmt = mimir
            .conn
            .prepare("SELECT src, dst FROM edge WHERE rel != ?1")?;
        // Bound rather than inlined so the stored spelling comes from the enum
        // and cannot drift away from it in a string literal here.
        let mut rows = stmt.query([Rel::CoRecalled.as_str()])?;
        while let Some(r) = rows.next()? {
            let (s, d): (i64, i64) = (r.get(0)?, r.get(1)?);
            existing.insert((s, d));
            existing.insert((d, s));
        }
    }

    let mut total = 0usize;
    for proj in &targets {
        let memories_of = mimir.memory_project_of(proj)?;
        let created = scan_one_project(&mimir, proj, memories_of.id, dry_run, &mut existing)?;
        total += created;
        if all_projects {
            println!(
                "{} {created} link(s) in {}",
                if dry_run { "would create" } else { "created" },
                proj.title.as_deref().unwrap_or("(unnamed)")
            );
        }
    }
    println!(
        "{} {total} link(s) across {} project(s)",
        if dry_run { "would create" } else { "created" },
        targets.len()
    );
    if !dry_run {
        // Mirrors `last_consolidate`, so `doctor` can say how stale this is
        // instead of nagging about a scan you ran an hour ago.
        mimir.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_link_scan', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [mimir_core::model::now_unix().to_string()],
        )?;
    }
    Ok(())
}

/// One project's pass: the symbols of the checkout project `proj` against
/// the memories of `memory_project` and the global ones. The two are one
/// project except for a linked worktree, whose memories are its main
/// checkout's while the symbols they name are in its own graph. Returns
/// links created (or, in a dry run, the number that would be).
fn scan_one_project(
    mimir: &Mimir,
    proj: &Node,
    memory_project: i64,
    dry_run: bool,
    existing: &mut std::collections::HashSet<(i64, i64)>,
) -> Result<usize> {
    // name → symbol nodes carrying it (bare name from meta, fallback title)
    let mut by_name: HashMap<String, Vec<(i64, String)>> = HashMap::new();
    {
        let mut stmt = mimir.conn.prepare(
            "SELECT id, uid, COALESCE(json_extract(meta,'$.name'), title) FROM node
             WHERE kind='symbol' AND project_id=?1 AND deleted_at IS NULL",
        )?;
        let mut rows = stmt.query([proj.id])?;
        while let Some(r) = rows.next()? {
            let (id, uid): (i64, String) = (r.get(0)?, r.get(1)?);
            let name: Option<String> = r.get(2)?;
            if let Some(name) = name {
                if name.len() >= 4 {
                    by_name.entry(name).or_default().push((id, uid));
                }
            }
        }
    }

    let memories: Vec<(i64, String, String)> = {
        let mut stmt = mimir.conn.prepare(
            "SELECT id, uid, COALESCE(title,'') || ' ' || COALESCE(body,'') FROM node
             WHERE kind='memory' AND deleted_at IS NULL AND superseded_by IS NULL
               AND (project_id=?1 OR project_id IS NULL)",
        )?;
        let rows = stmt.query_map([memory_project], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };

    let mut created = 0usize;
    for (mid, muid, text) in &memories {
        for (name, syms) in &by_name {
            if syms.len() > 3 {
                continue; // same name everywhere = ambiguous, skip
            }
            if !mentions_symbol(text, name) {
                continue;
            }
            for (sid, suid) in syms {
                if existing.contains(&(*mid, *sid)) {
                    continue;
                }
                if !dry_run {
                    store::link(&mimir.conn, *mid, *sid, Rel::Mentions, 1.0)?;
                }
                println!(
                    "{}m:{} —mentions→ {name} (c:{})",
                    if dry_run { "would link " } else { "" },
                    tail(muid),
                    tail(suid)
                );
                existing.insert((*mid, *sid));
                created += 1;
            }
        }
    }
    Ok(created)
}

fn tail(uid: &str) -> &str {
    &uid[uid.len().saturating_sub(6)..]
}

/// True when `text` mentions `name` as code: word-boundary matched AND
/// code-shaped (snake_case, ::path, or mixed-case with an uppercase letter
/// *after* the first — `MimirServer` yes, sentence-case `Pending` no).
/// Plain English words never match, even in backticks: `main` is usually
/// a git branch, not fn main. Precision beats recall here — a wrong link
/// pollutes recall, a missing one just waits for the next scan.
fn mentions_symbol(text: &str, name: &str) -> bool {
    let mixed_case = name.len() >= 6
        && name.chars().skip(1).any(|c| c.is_uppercase())
        && name.chars().any(|c| c.is_lowercase());
    let code_shaped = name.contains('_') || name.contains("::") || mixed_case;
    if !code_shaped {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = text[start..].find(name) {
        let i = start + pos;
        let j = i + name.len();
        let is_word = |c: Option<char>| c.map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
        let pre = text[..i].chars().next_back();
        let post = text[j..].chars().next();
        if !is_word(pre) && !is_word(post) {
            return true;
        }
        start = j;
    }
    false
}

// ---------- docs & index ----------

pub fn docs_add(path: &str, name: Option<String>, global: bool) -> Result<()> {
    add_collection_cmd(path, name, global, CollectionKind::Docs)
}

/// Register + index a source-code collection: chunks function/method bodies
/// (not just signatures) on tree-sitter symbol boundaries for recall —
/// mirrors `docs add`, indexing source instead of markdown. Idea credit:
/// nworks3d's THOR fork of Mimir (see CHANGELOG.md).
pub fn code_add(path: &str, name: Option<String>, global: bool) -> Result<()> {
    add_collection_cmd(path, name, global, CollectionKind::Code)
}

fn add_collection_cmd(
    path: &str,
    name: Option<String>,
    global: bool,
    kind: CollectionKind,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let root = std::path::Path::new(path);
    let canonical = std::fs::canonicalize(root).with_context(|| format!("no such dir: {path}"))?;
    let name = name.unwrap_or_else(|| {
        canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    });
    let project = if global {
        None
    } else {
        mimir.project_for_cwd(&canonical)?
    };
    let coll = mimir_core::index::add_collection_of_kind(
        &mimir.conn,
        &canonical,
        &name,
        project.as_ref().map(|p| p.id),
        kind,
    )?;
    println!(
        "{} {} {}",
        short_uid(coll.kind, &coll.uid),
        name,
        coll.path.as_deref().unwrap_or("?")
    );
    println!("run `mimir index` to scan it");
    Ok(())
}

/// The error `index::find_collection` gives for a reference nothing answers
/// to, for the commands that look collections up by kind instead.
fn no_such_collection(reference: &str) -> anyhow::Error {
    mimir_core::error::Error::NotFound(format!("collection '{reference}'")).into()
}

/// What a `docs` subcommand found for a reference: the collection it acts
/// on, and every other collection that answers to the same name or path.
struct DocsCmdTarget {
    acted_on: Node,
    others: Vec<Node>,
}

/// The collection a `docs` subcommand means by `reference`, and the ones it
/// does not.
///
/// A docs and a code collection can share a root and the name derived from
/// it (`mimir code add .` beside the auto-registered docs collection), and
/// then `reference` names both. A `docs` command means the docs one. Only
/// when no docs collection answers does it reach a code collection, because
/// `docs remove` and `docs note` are also the only commands there are for
/// those — so a second `docs remove <name>` on such a root removes the code
/// collection. A collection named `reference` always comes before one that
/// only lives at that path, whatever their kinds.
///
/// Within one kind the oldest match is acted on (two projects can each
/// register a `docs` folder under that name). Whatever else matched comes
/// back as `others`, and the command says so: acting on one of several
/// without naming it, or without saying the rest are still there, is the
/// silent "first row wins" this replaces.
fn collection_for_docs_cmd(
    conn: &rusqlite::Connection,
    reference: &str,
) -> Result<Option<DocsCmdTarget>> {
    use mimir_core::index::find_collections;
    // One lookup across both kinds, so a name is tried before a path for
    // the whole set: `find_collections` returns the name matches when there
    // are any and only then falls back to the path. Asking per kind let a
    // docs collection that merely *sits at* `reference` win over a code
    // collection that is *named* `reference`, and `docs remove proj`
    // removed a collection called something else. Within what matched, docs
    // come first (the sort is stable, so each kind stays oldest first).
    let mut matches = find_collections(conn, reference, None)?;
    matches.sort_by_key(|node| CollectionKind::of(node) != CollectionKind::Docs);
    let mut matches = matches.into_iter();
    Ok(matches.next().map(|acted_on| DocsCmdTarget {
        acted_on,
        others: matches.collect(),
    }))
}

/// `docs collection l:ABC123`: a collection named by what tells it apart
/// from another with the same name.
fn kind_and_id(collection: &Node) -> String {
    format!(
        "{} collection {}",
        CollectionKind::of(collection),
        short_uid(collection.kind, &collection.uid)
    )
}

/// The clause naming the collections a `docs` command left alone, for a
/// sentence the caller finishes: `a code collection l:ABC123 also answers to
/// 'proj'`, or the count and the list when there are several.
fn also_answering(others: &[Node], reference: &str) -> String {
    match others {
        [one] => format!("a {} also answers to '{reference}'", kind_and_id(one)),
        many => format!(
            "{} other collections also answer to '{reference}' ({})",
            many.len(),
            many.iter().map(kind_and_id).collect::<Vec<_>>().join(", ")
        ),
    }
}

pub fn docs_list(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let collections = mimir_core::index::list_collections(&mimir.conn)?;
    if collections.is_empty() && !json {
        println!("no collections (add one with `mimir docs add <path>`)");
        return Ok(());
    }
    let projects = store::project_titles(&mimir.conn)?;
    for coll in collections {
        let (files, chunks) = mimir_core::index::collection_stats(&mimir.conn, coll.id)?;
        if json {
            let mut v = node_json(&coll, &projects);
            v["files"] = serde_json::json!(files);
            v["chunks"] = serde_json::json!(chunks);
            println!("{v}");
        } else {
            let kind = CollectionKind::of(&coll);
            println!(
                "{} [{kind}] {} {} ({files} files, {chunks} chunks)",
                short_uid(coll.kind, &coll.uid),
                coll.title.as_deref().unwrap_or("?"),
                coll.path.as_deref().unwrap_or("?"),
            );
        }
    }
    Ok(())
}

pub fn docs_remove(name: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let Some(DocsCmdTarget {
        acted_on: coll,
        others,
    }) = collection_for_docs_cmd(&mimir.conn, name)?
    else {
        return Err(no_such_collection(name));
    };
    mimir_core::index::remove_collection(&mimir.conn, coll.id)?;
    let title = coll.title.as_deref().unwrap_or(name);
    if others.is_empty() {
        println!("removed {title}");
    } else {
        // More than one collection answered to `name`: say which one went,
        // and that the rest are still registered and how to reach them.
        let (verb, next) = if others.len() == 1 {
            ("is", "it")
        } else {
            ("are", "the next one")
        };
        println!(
            "removed {} collection {title} ({}); {} and {verb} still registered, \
             run the command again to remove {next}",
            CollectionKind::of(&coll),
            short_uid(coll.kind, &coll.uid),
            also_answering(&others, name),
        );
    }
    Ok(())
}

pub fn docs_note(target: &str, text: String) -> Result<()> {
    let mimir = Mimir::open()?;
    let (target_node, others) = match collection_for_docs_cmd(&mimir.conn, target)? {
        Some(found) => (found.acted_on, found.others),
        None => (store::resolve_ref(&mimir.conn, target)?, Vec::new()),
    };
    // A note is free text stored and embedded like any memory.
    capture::refuse_secrets(
        &mimir.conn,
        &text,
        &[],
        &[],
        mimir_core::secrets::Surface::CliRemember,
    )?;
    let note = mimir_core::index::annotate(&mimir.conn, &target_node, &text)?;
    println!(
        "{} describes {} {}",
        short_uid(note.kind, &note.uid),
        short_uid(target_node.kind, &target_node.uid),
        target_node.title.as_deref().unwrap_or("")
    );
    if !others.is_empty() {
        // The line above carries an id, not a kind; with several
        // collections under one name that is not enough to tell which got
        // the note. An id as the target reaches any of the others.
        let (verb, how) = if others.len() == 1 {
            ("was", "pass its id as the target to annotate it")
        } else {
            ("were", "pass an id as the target to annotate one of them")
        };
        println!(
            "the note is on the {} collection; {} and {verb} not annotated, {how}",
            CollectionKind::of(&target_node),
            also_answering(&others, target),
        );
    }
    Ok(())
}

pub fn index(name: Option<String>) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let results = match name {
        // Every collection the name or path answers to, not the first: a
        // root registered for both docs and code is one name for two
        // collections, and "index this root" means both of them.
        Some(n) => {
            let collections = mimir_core::index::find_collections(&mimir.conn, &n, None)?;
            if collections.is_empty() {
                return Err(no_such_collection(&n));
            }
            let mut results = Vec::with_capacity(collections.len());
            for coll in collections {
                let stats = mimir_core::index::index_collection(&mut mimir.conn, &coll)?;
                results.push((coll.title.unwrap_or_else(|| n.clone()), stats));
            }
            results
        }
        None => mimir_core::index::index_all(&mut mimir.conn)?,
    };
    if results.is_empty() {
        println!("no collections (add one with `mimir docs add <path>`)");
        return Ok(());
    }
    for (name, s) in results {
        println!(
            "{name}: {} files seen, {} indexed ({} chunks), {} unchanged, {} removed",
            s.seen, s.indexed, s.chunks, s.unchanged, s.removed
        );
    }
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

// ---------- import / export ----------

pub fn import_openbrain(file: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let text = if file == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read_to_string(file).with_context(|| format!("read {file}"))?
    };
    let stats = mimir_core::import::openbrain(&mimir.conn, &text)?;
    finish_import(&mut mimir, stats)
}

pub fn import_claude_memory(dir: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let stats = mimir_core::import::claude_memory(&mimir.conn, std::path::Path::new(dir))?;
    finish_import(&mut mimir, stats)
}

pub fn import_qmd(file: Option<String>) -> Result<()> {
    let mimir = Mimir::open()?;
    let path = match file {
        Some(f) => std::path::PathBuf::from(f),
        None => directories::BaseDirs::new()
            .context("cannot resolve home")?
            .home_dir()
            .join(".config/qmd/index.yml"),
    };
    let yml = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let collections = mimir_core::import::qmd_collections(&yml);
    if collections.is_empty() {
        bail!("no collections found in {}", path.display());
    }
    for (name, root) in &collections {
        let root_path = std::path::Path::new(root);
        if !root_path.is_dir() {
            eprintln!("skipping {name}: {root} is not a directory");
            continue;
        }
        let coll = mimir_core::index::add_collection(&mimir.conn, root_path, name, None, "docs")?;
        println!(
            "registered {} {} {}",
            short_uid(coll.kind, &coll.uid),
            name,
            root
        );
    }
    println!("run `mimir index` to scan them");
    Ok(())
}

fn finish_import(mimir: &mut Mimir, stats: mimir_core::import::ImportStats) -> Result<()> {
    println!(
        "imported {} memorie(s), skipped {} duplicate(s)",
        stats.imported, stats.skipped_duplicates
    );
    if stats.skipped_forgotten > 0 {
        println!(
            "skipped {} previously forgotten memorie(s) — re-add with \
             `mimir remember --force` if that was wrong",
            stats.skipped_forgotten
        );
    }
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

pub fn export() -> Result<()> {
    let mimir = Mimir::open()?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let n = mimir_core::import::export_jsonl(&mimir.conn, &mut lock)?;
    eprintln!("exported {n} line(s)");
    Ok(())
}

// ---------- helpers ----------

/// Resolve ids first, then symbol names within the current project — so
/// `mimir link m:ABC123 resolve_ref --rel about` just works.
fn resolve_any(mimir: &Mimir, reference: &str) -> Result<Node> {
    match store::resolve_ref(&mimir.conn, reference) {
        Ok(node) => Ok(node),
        Err(id_err) => {
            if let Some(proj) = mimir.project_for_cwd(&std::env::current_dir()?)? {
                if let Ok(sym) = mimir_graph::resolve_symbol(&mimir.conn, proj.id, reference) {
                    return Ok(sym);
                }
            }
            Err(id_err.into())
        }
    }
}

/// Before a CLI command that reads or writes memories: move what a linked
/// worktree's own project row still holds to the project its memories are
/// filed under (`Mimir::projects_for_cwd_moving_memories_home`), and say so
/// in one line on stderr when anything moved. The commands that do this
/// are listed in one place, `main.rs`'s `moves_worktree_memories`; hooks,
/// `status` and `doctor` never do.
///
/// Best-effort: the command runs either way. A store that cannot be opened
/// is left for the command to report; a move that fails is a warning line
/// from the engine, and the next memory command tries again.
pub fn move_worktree_memories_here() {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let Ok(mimir) = Mimir::open() else {
        return;
    };
    match mimir.projects_for_cwd_moving_memories_home(&cwd) {
        Ok((_, Some(moved))) => eprintln!("{}", moved.line()),
        Ok((_, None)) => {}
        Err(err) => tracing::debug!(%err, "could not resolve the projects to move memories home"),
    }
}

/// What a read from the working directory reaches: `--all` and `-g` say it
/// outright, for every kind; otherwise it is
/// [`mimir_core::scope::query_reach`] of the directory's two projects — the
/// project + global inside one, everything outside (reads want breadth).
/// The two are one except in a linked worktree, which reads the memories of
/// its repository's main checkout and its own code and docs.
///
/// It also says whose session handoffs the read may show: a scoped read
/// leaves out the handoffs written in other checkouts
/// ([`mimir_core::context_guard::shown_in_checkout`]), `--all` shows them.
fn read_reach(mimir: &Mimir, global: bool, all: bool) -> Result<ReadPlace> {
    if all {
        return Ok(ReadPlace {
            reach: Reach::uniform(Scope::All),
            handoffs_of: None,
        });
    }
    let cwd = std::env::current_dir()?;
    if global {
        // The checkout a handoff records is its project row's path, which
        // is the detected root's canonical form: no project row is needed
        // to know it, and `-g` registers none.
        let checkout = match mimir_core::scope::detect(&cwd) {
            mimir_core::scope::Detection::Found { root, .. } => {
                Some(mimir_core::scope::canonical_root(&root))
            }
            mimir_core::scope::Detection::NotFound { .. } => None,
        };
        return Ok(ReadPlace {
            reach: Reach::uniform(Scope::Global),
            handoffs_of: Some(checkout),
        });
    }
    let here = mimir.projects_for_cwd(&cwd)?;
    Ok(ReadPlace {
        reach: here.query_reach(),
        handoffs_of: Some(here.checkout_path().map(str::to_owned)),
    })
}

/// What [`read_reach`] found: the scopes a read reaches, and whose session
/// handoffs it may show.
struct ReadPlace {
    reach: Reach,
    /// `Some(checkout)` for a scoped read: only the handoffs that
    /// [`mimir_core::context_guard::handoff_is_for`] that checkout (`None`
    /// outside any project). `None` for `--all`: every handoff.
    handoffs_of: Option<Option<String>>,
}

impl ReadPlace {
    /// Whether this read may show `node`.
    fn shows(&self, node: &Node) -> bool {
        match &self.handoffs_of {
            None => true,
            Some(checkout) => {
                mimir_core::context_guard::shown_in_checkout(node, checkout.as_deref())
            }
        }
    }
}

/// Scope for reading memories and nothing else (`misses`): the memories
/// half of [`read_reach`].
fn read_scope(mimir: &Mimir, global: bool, all: bool) -> Result<Scope> {
    Ok(read_reach(mimir, global, all)?.reach.memories)
}

fn parse_kind_filter(kind: &str) -> Result<Vec<Kind>> {
    Ok(match kind {
        // No kind filter — deliberately includes CodeChunk. The point of
        // indexing function bodies (not just signatures) is that a plain
        // `recall` finds them without the caller knowing to ask for
        // `--kind code`; ScoringConfig::code_damp keeps code from drowning
        // out memories given its much larger corpus share.
        "all" => vec![],
        "memory" => vec![Kind::Memory],
        "doc" => vec![Kind::File, Kind::Chunk, Kind::Annotation],
        // Symbol = signature/doc only; CodeChunk = actual body/content text
        // (see chunker::chunk_source). Both belong under `code`.
        "code" => vec![Kind::Symbol, Kind::CodeChunk],
        other => bail!("unknown --kind '{other}' (use all|memory|doc|code)"),
    })
}

/// "12h" | "7d" | "2w" | "3m" | "1y" → unix cutoff.
fn parse_since(s: &str) -> Result<i64> {
    // Split on the last CHARACTER, not the last byte — a multibyte unit
    // (e.g. "5µ") would otherwise slice mid-codepoint and panic.
    let split = s.char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
    let (num, unit) = s.split_at(split);
    let n: i64 = num
        .parse()
        .with_context(|| format!("bad --since '{s}' (use e.g. 12h, 7d, 2w, 3m, 1y)"))?;
    let secs = match unit {
        "h" => 3_600,
        "d" => 86_400,
        "w" => 604_800,
        "m" => 2_592_000,
        "y" => 31_536_000,
        _ => bail!("bad --since unit '{unit}' (use h, d, w, m, y)"),
    };
    Ok(now_unix() - n * secs)
}

fn line(node: &Node, projects: &HashMap<i64, String>, snippet_chars: usize) -> String {
    line_q(node, projects, snippet_chars, None)
}

/// [`line`] for ranked recall results, where the query is available and the
/// snippet can be centred on what matched.
fn line_q(
    node: &Node,
    projects: &HashMap<i64, String>,
    snippet_chars: usize,
    query: Option<&str>,
) -> String {
    line_with_grounding(
        node,
        projects,
        snippet_chars,
        query,
        mimir_core::format::HitMarks::default(),
    )
}

/// [`line_for_query`] plus the stale-grounding marker. Separate so the many
/// callers that render a single echoed node (remember, mark, list …) don't
/// have to run a grounding query they'd almost always get `false` from;
/// only the recall paths, which already have the whole hit set in hand,
/// pay for the one batch lookup.
fn line_with_grounding(
    node: &Node,
    projects: &HashMap<i64, String>,
    snippet_chars: usize,
    query: Option<&str>,
    marks: mimir_core::format::HitMarks,
) -> String {
    let project = node
        .project_id
        .and_then(|id| projects.get(&id))
        .map(String::as_str);
    mimir_core::format::agent_line_for_query(node, project, snippet_chars, query, marks)
}

fn print_full(node: &Node, mimir: &Mimir, projects: &HashMap<i64, String>) -> Result<()> {
    println!(
        "{}",
        mimir_core::format::full_record(&mimir.conn, node, projects)?
    );
    Ok(())
}

fn node_json(node: &Node, projects: &HashMap<i64, String>) -> serde_json::Value {
    let mut value = serde_json::json!({
        "id": short_uid(node.kind, &node.uid),
        "uid": node.uid,
        "kind": node.kind.as_str(),
        // A memory's type as every other surface reads it: one stored
        // without a type is a note, not null.
        "type": if node.kind == Kind::Memory {
            Some(node.type_label())
        } else {
            node.subkind.as_deref()
        },
        "project": node.project_id.and_then(|id| projects.get(&id)),
        "title": node.title,
        "body": node.body,
        "path": node.path,
        "tags": node.tags(),
        "created_at": node.created_at,
        "updated_at": node.updated_at,
        "access_count": node.access_count,
        "strength": node.strength,
    });
    // The state a text line spells as one word, for a caller parsing JSON.
    // Always present on a memory — "live" included, so a script can match on
    // the field instead of on its absence. Memories only, like the label.
    if node.kind == Kind::Memory {
        value["lifecycle"] = serde_json::json!(node.lifecycle(now_unix()).as_str());
    }
    value
}

#[cfg(test)]
mod since_tests {
    use super::parse_since;

    #[test]
    fn multibyte_unit_errors_not_panics() {
        // Regression: split_at on a byte offset panicked mid-codepoint.
        assert!(parse_since("5µ").is_err());
        assert!(parse_since("7€").is_err());
        assert!(parse_since("").is_err());
        assert!(parse_since("3d").is_ok());
    }
}

#[cfg(test)]
mod scan_tests {
    use super::mentions_symbol;

    #[test]
    fn snake_case_matches_on_word_boundaries() {
        assert!(mentions_symbol(
            "the record_opened path is the entry",
            "record_opened"
        ));
        assert!(!mentions_symbol("we prerecord_opened it", "record_opened"));
        assert!(mentions_symbol(
            "learn::record_opened is the single entry",
            "record_opened"
        ));
    }

    #[test]
    fn plain_words_never_match_even_backticked() {
        assert!(!mentions_symbol("we should update the docs", "update"));
        assert!(!mentions_symbol("we changed `update` semantics", "update"));
        assert!(!mentions_symbol("force push `main` to origin", "main"));
    }

    #[test]
    fn camel_case_matches_but_sentence_case_does_not() {
        assert!(mentions_symbol(
            "the MimirServer struct owns the router",
            "MimirServer"
        ));
        assert!(!mentions_symbol("nothing here", "MimirServer"));
        assert!(!mentions_symbol("Pending tasks for tomorrow", "Pending"));
    }
}

#[cfg(test)]
mod hooks_tests {
    use super::{merge_hook_settings, render_recall_script};

    /// A custom `config.hooks.inject_url` must land verbatim as the
    /// script's `MIMIR_INJECT_URL` fallback default — the actual install
    /// path (`install_hooks`) is only exercised end-to-end by the
    /// fake-$HOME e2e test (crates/mimir-cli/tests/e2e.rs), since it needs
    /// a real `~/.claude` dir and MIMIR_HOME must be *unset* for it to run
    /// at all; this covers the pure rendering logic in isolation.
    #[test]
    fn custom_inject_url_lands_in_generated_script() {
        let script = render_recall_script("http://10.0.0.5:9999/inject");
        assert!(
            script.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://10.0.0.5:9999/inject}""#),
            "custom URL missing from script:\n{script}"
        );
        // The env-var override placeholder syntax itself must survive
        // untouched — only the default inside it gets substituted.
        assert!(script.contains("${MIMIR_INJECT_URL:-"));
        assert!(!script.contains("__MIMIR_INJECT_URL_DEFAULT__"));
    }

    #[test]
    fn default_inject_url_matches_documented_port() {
        let script = render_recall_script("http://127.0.0.1:8077/inject");
        assert!(
            script.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://127.0.0.1:8077/inject}""#)
        );
    }

    /// `auto_recall=false` (recall_script = None) must be byte-identical to
    /// pre-auto-recall behavior: no `UserPromptSubmit` key appears at all.
    /// `context_guard_scripts = None` must likewise add none of the
    /// context-guard entries — this is the full default-off byte-identical
    /// case: only SessionStart(rules) + PreToolUse(rewrite, anchors) exist.
    #[test]
    fn no_recall_script_leaves_user_prompt_submit_untouched() {
        let (root, messages) = merge_hook_settings(
            serde_json::json!({}),
            "/path/mimir-rewrite.sh",
            None,
            "/path/mimir-anchors.sh",
            None,
        )
        .unwrap();
        assert!(
            root["hooks"].get("UserPromptSubmit").is_none(),
            "auto_recall=false and context_guard=off must not add hooks.UserPromptSubmit, got: {root}"
        );
        assert!(
            root["hooks"].get("PreCompact").is_none(),
            "context_guard=off must not add hooks.PreCompact, got: {root}"
        );
        // The pre-existing hooks still get installed as before, plus the
        // unconditional guard-anchors entry.
        assert!(root["hooks"]["SessionStart"].is_array());
        assert_eq!(
            root["hooks"]["SessionStart"].as_array().unwrap().len(),
            2,
            "context_guard=off must add exactly two SessionStart entries (rules + brief), not three"
        );
        let pre_arr = root["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(
            pre_arr.len(),
            2,
            "rewrite + anchors, unconditionally under --hooks"
        );
        assert!(messages.iter().any(|m| m.contains("SessionStart")));
        assert!(messages
            .iter()
            .any(|m| m.contains("PreToolUse (command filter)")));
        assert!(messages
            .iter()
            .any(|m| m.contains("PreToolUse (guard anchors)")));
        assert!(!messages.iter().any(|m| m.contains("UserPromptSubmit")));
        assert!(!messages.iter().any(|m| m.contains("PreCompact")));
    }

    /// `auto_recall=true` adds exactly one `UserPromptSubmit` entry
    /// pointing at the recall script, and re-running is a no-op (idempotent).
    #[test]
    fn recall_script_adds_one_entry_idempotently() {
        let (root, messages) = merge_hook_settings(
            serde_json::json!({}),
            "/path/mimir-rewrite.sh",
            Some("/path/mimir-recall.sh"),
            "/path/mimir-anchors.sh",
            None,
        )
        .unwrap();
        let entries = root["hooks"]["UserPromptSubmit"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(messages.iter().any(|m| m.contains("UserPromptSubmit")));

        // Re-run against the already-merged settings: still exactly one entry,
        // and the message says "already installed" instead of "added".
        let (root2, messages2) = merge_hook_settings(
            root,
            "/path/mimir-rewrite.sh",
            Some("/path/mimir-recall.sh"),
            "/path/mimir-anchors.sh",
            None,
        )
        .unwrap();
        let entries2 = root2["hooks"]["UserPromptSubmit"].as_array().unwrap();
        assert_eq!(entries2.len(), 1, "re-running must not duplicate the entry");
        assert!(messages2
            .iter()
            .any(|m| m.contains("UserPromptSubmit already installed")));
    }

    /// Guard anchors install unconditionally under `--hooks` regardless of
    /// `auto_recall`/`context_guard`, and re-running is idempotent.
    #[test]
    fn anchors_entry_is_unconditional_and_idempotent() {
        let (root, messages) = merge_hook_settings(
            serde_json::json!({}),
            "/path/mimir-rewrite.sh",
            None,
            "/path/mimir-anchors.sh",
            None,
        )
        .unwrap();
        let pre_arr = root["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre_arr.len(), 2);
        assert!(messages
            .iter()
            .any(|m| m.contains("PreToolUse (guard anchors) added")));

        let (root2, messages2) = merge_hook_settings(
            root,
            "/path/mimir-rewrite.sh",
            None,
            "/path/mimir-anchors.sh",
            None,
        )
        .unwrap();
        let pre_arr2 = root2["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(
            pre_arr2.len(),
            2,
            "re-running must not duplicate the anchors entry"
        );
        assert!(messages2
            .iter()
            .any(|m| m.contains("PreToolUse anchors already installed")));
    }

    /// `context_guard_scripts = Some(...)` adds exactly one UserPromptSubmit,
    /// one PreCompact, and a second SessionStart entry, all idempotently.
    #[test]
    fn context_guard_scripts_add_three_entries_idempotently() {
        let scripts = (
            "/path/mimir-context-guard-prompt.sh",
            "/path/mimir-context-guard-precompact.sh",
            "/path/mimir-context-guard-session.sh",
        );
        let (root, messages) = merge_hook_settings(
            serde_json::json!({}),
            "/path/mimir-rewrite.sh",
            None,
            "/path/mimir-anchors.sh",
            Some(scripts),
        )
        .unwrap();
        assert_eq!(
            root["hooks"]["UserPromptSubmit"].as_array().unwrap().len(),
            1
        );
        assert_eq!(root["hooks"]["PreCompact"].as_array().unwrap().len(), 1);
        assert_eq!(
            root["hooks"]["SessionStart"].as_array().unwrap().len(),
            3,
            "rules entry + brief entry + context-guard entry"
        );
        assert!(messages
            .iter()
            .any(|m| m.contains("UserPromptSubmit (context guard) added")));
        assert!(messages
            .iter()
            .any(|m| m.contains("PreCompact (context guard) added")));
        assert!(messages
            .iter()
            .any(|m| m.contains("SessionStart (context guard) added")));

        let (root2, messages2) = merge_hook_settings(
            root,
            "/path/mimir-rewrite.sh",
            None,
            "/path/mimir-anchors.sh",
            Some(scripts),
        )
        .unwrap();
        assert_eq!(
            root2["hooks"]["UserPromptSubmit"].as_array().unwrap().len(),
            1
        );
        assert_eq!(root2["hooks"]["PreCompact"].as_array().unwrap().len(), 1);
        assert_eq!(root2["hooks"]["SessionStart"].as_array().unwrap().len(), 3);
        assert!(messages2
            .iter()
            .any(|m| m.contains("UserPromptSubmit context-guard already installed")));
        assert!(messages2
            .iter()
            .any(|m| m.contains("PreCompact already installed")));
        assert!(messages2
            .iter()
            .any(|m| m.contains("SessionStart context-guard already installed")));
    }
}

#[cfg(test)]
mod inject_addr_tests {
    use super::inject_addr;

    #[test]
    fn strips_scheme_and_inject_path() {
        assert_eq!(
            inject_addr("http://127.0.0.1:8077/inject").unwrap(),
            "127.0.0.1:8077"
        );
        assert_eq!(
            inject_addr("https://10.0.0.5:9999/inject").unwrap(),
            "10.0.0.5:9999"
        );
    }

    #[test]
    fn tolerates_a_bare_host_port_with_no_scheme_or_path() {
        assert_eq!(inject_addr("127.0.0.1:8077").unwrap(), "127.0.0.1:8077");
    }

    #[test]
    fn empty_url_is_an_error_not_a_panic() {
        assert!(inject_addr("").is_err());
        assert!(inject_addr("http://").is_err());
    }
}

#[cfg(test)]
mod link_target_tests {
    use super::resolve_link_target;
    use mimir_core::model::{Kind, NewNode};
    use mimir_core::store;

    /// `--link` advertises "a code symbol or node" but only ever resolved
    /// ids, so linking a memory to a symbol by name failed outright — and a
    /// link to an indexed symbol is precisely what makes a memory grounded.
    #[test]
    fn resolves_a_symbol_by_bare_name_not_just_by_id() {
        let conn = mimir_core::db::open_in_memory().unwrap();
        let mut proj = NewNode::new(Kind::Project);
        proj.title = Some("probe".into());
        let project = store::insert_node(&conn, proj).unwrap();

        let mut sym = NewNode::new(Kind::Symbol);
        sym.title = Some("retry_with_backoff".into());
        sym.project_id = Some(project.id);
        let symbol = store::insert_node(&conn, sym).unwrap();

        let found =
            resolve_link_target(&conn, "retry_with_backoff", Some(project.id)).expect("by name");
        assert_eq!(found.id, symbol.id);

        // The id form must keep working, and must not need a project.
        let by_id = resolve_link_target(&conn, &symbol.uid, None).expect("by id");
        assert_eq!(by_id.id, symbol.id);
    }

    /// Outside a project there is nothing to search, so the caller should
    /// see the id-shaped error rather than a confusing symbol-lookup one.
    #[test]
    fn unknown_name_still_errors() {
        let conn = mimir_core::db::open_in_memory().unwrap();
        assert!(resolve_link_target(&conn, "nope_not_here", None).is_err());
        assert!(resolve_link_target(&conn, "nope_not_here", Some(1)).is_err());
    }
}

#[cfg(test)]
mod link_scan_hint_tests {
    use super::link_scan_hint;
    use mimir_core::model::{Kind, NewNode};
    use mimir_core::store;

    fn store_with_graph(symbols: bool) -> rusqlite::Connection {
        let conn = mimir_core::db::open_in_memory().unwrap();
        if symbols {
            let mut p = NewNode::new(Kind::Project);
            p.title = Some("probe".into());
            let proj = store::insert_node(&conn, p).unwrap();
            let mut s = NewNode::new(Kind::Symbol);
            s.title = Some("retry_with_backoff".into());
            s.project_id = Some(proj.id);
            store::insert_node(&conn, s).unwrap();
        }
        conn
    }

    fn set_last_scan(conn: &rusqlite::Connection, days_ago: i64) {
        let at = mimir_core::model::now_unix() - days_ago * 86_400;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_link_scan', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [at.to_string()],
        )
        .unwrap();
    }

    /// Nothing to link against means nothing to say. Advising a scan on a
    /// store with no code graph is advice that cannot be acted on.
    #[test]
    fn silent_without_any_code_graph() {
        assert!(link_scan_hint(&store_with_graph(false)).is_none());
    }

    #[test]
    fn speaks_up_when_graphs_exist_and_nothing_was_ever_scanned() {
        let hint = link_scan_hint(&store_with_graph(true)).expect("should nudge");
        assert!(hint.contains("never scanned"), "{hint}");
        assert!(hint.contains("--all-projects"), "{hint}");
    }

    /// The property that keeps the nudge worth reading: it stops once you
    /// have acted on it. A health check that repeats advice you already
    /// took is one people learn to skim past.
    #[test]
    fn goes_quiet_after_a_recent_scan() {
        let conn = store_with_graph(true);
        set_last_scan(&conn, 1);
        assert!(link_scan_hint(&conn).is_none());
    }

    /// ...but graphs move, so it comes back rather than staying silent for
    /// good after one scan a year ago.
    #[test]
    fn returns_when_the_last_scan_has_gone_stale() {
        let conn = store_with_graph(true);
        set_last_scan(&conn, 45);
        let hint = link_scan_hint(&conn).expect("should nudge again");
        assert!(hint.contains("45d ago"), "{hint}");
    }
}

#[cfg(test)]
mod link_rel_guard_tests {
    use super::reject_unauthorable_rel;
    use mimir_core::model::Rel;

    /// A hand-written `co_recalled` edge is a forged usage record, and it
    /// would also cost the pair its real derived edge later (the observer
    /// skips pairs that already carry an authored one). The message has to
    /// leave the user somewhere to go, so it names the relation they meant.
    #[test]
    fn co_recalled_is_refused_and_points_at_relates() {
        let err = reject_unauthorable_rel(Rel::CoRecalled)
            .expect_err("co_recalled must not be authorable")
            .to_string();
        assert!(err.contains("co_recalled"), "{err}");
        assert!(err.contains("relates"), "{err}");
    }

    /// The pre-existing refusal, kept honest now that it shares a function.
    #[test]
    fn supersedes_is_still_refused() {
        assert!(reject_unauthorable_rel(Rel::Supersedes).is_err());
    }

    /// Everything a person can legitimately draw still passes — a guard
    /// that over-refuses is worse than the forgery it prevents.
    #[test]
    fn authored_relations_pass() {
        for rel in [Rel::Relates, Rel::About, Rel::Links, Rel::Mentions] {
            assert!(reject_unauthorable_rel(rel).is_ok(), "{rel} was refused");
        }
    }

    /// A `calls` or `imports` edge drawn by hand is deleted the next time
    /// the extractor re-reads the file, and until then nothing tells it
    /// apart from one the extractor resolved. The refusal names the
    /// relation that was asked for and the ones to use instead.
    #[test]
    fn relations_mimir_derives_are_refused() {
        for rel in [
            Rel::Calls,
            Rel::Imports,
            Rel::Defines,
            Rel::MemberOf,
            Rel::Tagged,
        ] {
            let err = reject_unauthorable_rel(rel)
                .expect_err("a derived relation must not be authorable")
                .to_string();
            assert!(err.contains(&format!("--rel {rel}")), "{err}");
            assert!(err.contains("relates") && err.contains("about"), "{err}");
        }
    }

    /// The guard and the list agree on every relation, so a new relation
    /// cannot be accepted by one and refused by the other.
    #[test]
    fn the_guard_accepts_exactly_the_authorable_relations() {
        let authorable: Vec<Rel> = Rel::authorable().collect();
        for rel in Rel::ALL.iter().copied() {
            assert_eq!(
                reject_unauthorable_rel(rel).is_ok(),
                authorable.contains(&rel),
                "{rel}"
            );
        }
    }

    /// `mimir link --help` used to offer `supersedes` (always refused) and
    /// leave out nothing else the guard refuses. The help is now built from
    /// the same list the guard reads.
    #[test]
    fn the_help_offers_what_the_guard_accepts() {
        let help = super::rel_help();
        let offered = help.split('.').next().unwrap();
        for rel in Rel::ALL.iter().copied() {
            let listed = offered.split('|').any(|word| word == rel.as_str());
            assert_eq!(
                listed,
                reject_unauthorable_rel(rel).is_ok(),
                "{rel} in {offered:?}"
            );
        }
    }
}

#[cfg(test)]
mod hook_upgrade_tests {
    use super::*;

    fn plan<'a>(brief_enabled: bool) -> HookPlan<'a> {
        HookPlan {
            rewrite_script: "/h/mimir-rewrite.sh",
            recall_script: None,
            anchors_script: "/h/mimir-anchors.sh",
            context_guard_scripts: None,
            brief_enabled,
        }
    }

    /// What `mimir init --hooks` wrote before the matcher covered every
    /// tool the handler reads and before the SessionStart commands were
    /// guarded.
    fn older_install() -> serde_json::Value {
        serde_json::json!({
            "hooks": {
                "SessionStart": [
                    { "hooks": [{ "type": "command", "command": "mimir rules show" }] },
                    { "hooks": [{ "type": "command", "command": "mimir brief show" }] },
                ],
                "PreToolUse": [
                    { "matcher": "Bash", "hooks": [
                        { "type": "command", "command": "/h/mimir-rewrite.sh" }
                    ] },
                    { "matcher": "Bash|Edit|Write", "hooks": [
                        { "type": "command", "command": "/h/mimir-anchors.sh" }
                    ] },
                ],
            }
        })
    }

    #[test]
    fn every_shipped_script_carries_a_version() {
        for (name, text, _) in HOOK_SCRIPTS {
            assert!(hook_version(text).is_some(), "{name} has no version line");
        }
        assert_eq!(hook_version(MIMIR_RECALL_SH), Some(5));
        assert_eq!(hook_version("#!/usr/bin/env bash\necho hi\n"), None);
        assert_eq!(hook_version("# mimir-hook-version: seven\n"), None);
    }

    /// The script is the one reader of the daemon's address outside this
    /// process: it must apply the same override, by the same name, and tell
    /// the daemon which directory the prompt came from.
    #[test]
    fn the_recall_script_spells_the_daemon_address_rule_and_sends_the_cwd() {
        let script = render_recall_script("http://127.0.0.1:8077/inject");
        assert!(script.contains(&format!(
            "INJECT_URL=\"${{{}:-http://127.0.0.1:8077/inject}}\"",
            crate::infer::INJECT_URL_ENV
        )));
        assert!(script.contains("URL=\"${URL}&cwd=${ENC_CWD}\""), "{script}");
        assert!(script.contains("cd \"$PROJECT_DIR\""), "{script}");
        assert!(script.contains("Authorization: Bearer %s"), "{script}");
    }

    #[test]
    fn outdated_scripts_are_the_installed_ones_older_than_shipped() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            outdated_hook_scripts(dir.path()).is_empty(),
            "none installed"
        );

        std::fs::write(dir.path().join(ANCHORS_SCRIPT), MIMIR_ANCHORS_SH).unwrap();
        assert!(outdated_hook_scripts(dir.path()).is_empty(), "current");

        std::fs::write(
            dir.path().join(RECALL_SCRIPT),
            "#!/usr/bin/env bash\n# mimir-hook-version: 4\n",
        )
        .unwrap();
        // A copy from before scripts were versioned at all.
        std::fs::write(dir.path().join(REWRITE_SCRIPT), "#!/usr/bin/env bash\n").unwrap();
        let mut outdated = outdated_hook_scripts(dir.path());
        outdated.sort();
        assert_eq!(
            outdated,
            vec![(RECALL_SCRIPT, 4, 5), (REWRITE_SCRIPT, 0, 1)]
        );
    }

    /// An older install is brought to the current shape in place: same
    /// number of entries, the anchors matcher widened to every tool the
    /// handler reads, and nothing left to do on the next run.
    #[test]
    fn an_older_install_is_upgraded_in_place_and_then_stable() {
        let (root, messages) = merge_hooks(older_install(), &plan(true)).unwrap();
        let pre = root["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 2);
        assert_eq!(pre[1]["matcher"], "Bash|Edit|Write|MultiEdit|NotebookEdit");
        assert_eq!(
            pre[0]["matcher"], "Bash",
            "the rewrite hook stays Bash-only"
        );
        assert!(
            messages
                .iter()
                .any(|m| m.starts_with("PreToolUse anchors upgraded")),
            "{messages:?}"
        );
        let session = root["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(session.len(), 2);
        assert_eq!(
            session[0]["hooks"][0]["command"],
            guarded_hook_command("rules show --hook")
        );
        assert_eq!(
            session[1]["hooks"][0]["command"],
            guarded_hook_command("brief show")
        );

        let (again, messages) = merge_hooks(root.clone(), &plan(true)).unwrap();
        assert_eq!(again, root, "a second run has nothing to change");
        assert!(!messages.iter().any(|m| m.contains("upgraded")));
        assert!(!messages.iter().any(|m| m.contains("added")));
    }

    /// Only what Mimir itself wrote is upgraded: a matcher or a command
    /// somebody changed by hand is kept exactly.
    #[test]
    fn hand_edited_entries_are_left_alone() {
        let mut edited = older_install();
        edited["hooks"]["PreToolUse"][1]["matcher"] = "Edit".into();
        edited["hooks"]["SessionStart"][0]["hooks"][0]["command"] =
            "mimir rules show | head -50".into();
        let (root, messages) = merge_hooks(edited, &plan(true)).unwrap();
        assert_eq!(root["hooks"]["PreToolUse"][1]["matcher"], "Edit");
        assert_eq!(
            root["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            "mimir rules show | head -50"
        );
        assert!(messages.contains(&"SessionStart already installed".to_string()));
        assert!(messages.contains(&"PreToolUse anchors already installed".to_string()));
        assert_eq!(root["hooks"]["SessionStart"].as_array().unwrap().len(), 2);
    }

    /// A matcher belongs to the whole entry. Mimir widens it only on an
    /// entry that is exactly its own: one hook, running the anchors script
    /// of this install. An entry that shares the matcher with somebody's
    /// hook, or whose command only contains the script's name, is left
    /// alone, and a stock entry behind it is still upgraded.
    #[test]
    fn an_entry_that_is_not_only_mimirs_keeps_its_matcher() {
        let mut shared = older_install();
        shared["hooks"]["PreToolUse"][1]["hooks"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "type": "command", "command": "~/bin/my-own-lint.sh" }));
        let (root, messages) = merge_hooks(shared, &plan(true)).unwrap();
        assert_eq!(
            root["hooks"]["PreToolUse"][1]["matcher"], "Bash|Edit|Write",
            "the user's own hook would start running on tools they never asked for"
        );
        assert!(messages.contains(&"PreToolUse anchors already installed".to_string()));

        let mut lookalike = older_install();
        lookalike["hooks"]["PreToolUse"]
            .as_array_mut()
            .unwrap()
            .insert(
                1,
                serde_json::json!({ "matcher": "Bash|Edit|Write", "hooks": [
                    { "type": "command", "command": "~/bin/not-mimir-anchors-but-mine.sh" }
                ] }),
            );
        let (root, messages) = merge_hooks(lookalike, &plan(true)).unwrap();
        let pre = root["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre[1]["matcher"], "Bash|Edit|Write", "not Mimir's entry");
        assert_eq!(
            pre[2]["matcher"],
            crate::context_guard_cmd::anchor_matcher(),
            "the stock entry behind it is still upgraded"
        );
        assert!(messages
            .iter()
            .any(|m| m.starts_with("PreToolUse anchors upgraded")));
    }

    #[cfg(not(windows))]
    #[test]
    fn session_start_commands_are_guarded_against_a_missing_binary() {
        assert_eq!(
            guarded_hook_command("rules show"),
            "command -v mimir >/dev/null 2>&1 || exit 0; mimir rules show"
        );
        // The guarded form still carries the text the installer recognizes
        // its own entry by, so it is found — not added again — next run.
        let entry = serde_json::json!({
            "hooks": [{ "type": "command", "command": guarded_hook_command("brief show") }]
        });
        assert!(entries_mention(&[entry], "mimir brief show"));
    }

    /// The line about the brief entry describes the entry: on when the
    /// brief is enabled (the default), inert when it is not.
    #[test]
    fn the_brief_message_says_whether_the_brief_is_enabled() {
        let brief_line = |enabled: bool, root: serde_json::Value| {
            merge_hooks(root, &plan(enabled))
                .unwrap()
                .1
                .into_iter()
                .find(|m| m.contains("brief"))
                .unwrap()
        };
        let fresh = || serde_json::json!({});
        assert_eq!(
            brief_line(true, fresh()),
            "SessionStart (session brief) added (on: `[brief] enabled = true`)"
        );
        assert_eq!(
            brief_line(false, fresh()),
            "SessionStart (session brief) added (inert while `[brief] enabled = false`)"
        );
        let installed = merge_hooks(fresh(), &plan(true)).unwrap().0;
        assert_eq!(
            brief_line(false, installed),
            "SessionStart (brief) already installed (inert while `[brief] enabled = false`)"
        );
    }
}
