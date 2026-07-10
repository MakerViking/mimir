//! `mimir context-guard <subcommand>` — the hook-event entry points behind
//! [`mimir_core::context_guard`] (context-window pause/handoff nudges) and
//! [`mimir_core::anchors`] (guard anchors). Each subcommand reads one hook
//! event's JSON from stdin.
//!
//! Field names and the "how does this reach the model" mechanics below were
//! verified against the live Claude Code hooks docs while implementing
//! this (not assumed):
//! - `UserPromptSubmit` and `SessionStart` are the two events where plain
//!   stdout (no JSON envelope) is injected as context.
//! - `PreToolUse` supports `hookSpecificOutput.additionalContext` to add
//!   context WITHOUT touching the allow/deny decision — it is not limited
//!   to `permissionDecision`/`permissionDecisionReason`.
//! - Blocking (`PreCompact`'s `{"decision":"block","reason":...}`) must be
//!   printed to stdout on exit 0. Exit 2 is a separate, mutually exclusive
//!   mechanism that reads stderr instead and ignores any JSON — mixing the
//!   two does nothing useful, so these subcommands always exit 0.
//!
//! Fail-open throughout: a parse error, missing field, or DB error is
//! swallowed (treated as "say nothing"), never a nonzero exit — a hook that
//! can break a prompt/tool-call/compact by erroring is worse than one that
//! occasionally stays quiet.

use std::io::Read as _;

use anyhow::Result;
use mimir_core::context_guard;
use mimir_core::model::Scope;
use mimir_core::Mimir;

fn read_stdin_json() -> serde_json::Value {
    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_err() {
        return serde_json::Value::Null;
    }
    serde_json::from_str(&buf).unwrap_or(serde_json::Value::Null)
}

fn transcript_bytes(input: &serde_json::Value) -> u64 {
    input
        .get("transcript_path")
        .and_then(|v| v.as_str())
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Scope inferred from a hook event's `cwd` field: the project it maps to,
/// or `Scope::Global` if there isn't one (no project marker at/above cwd,
/// or `cwd` missing). Mirrors `commands::read_scope`'s project-detection
/// fallback, but hooks never pass `--global`/`--all`, so there is nothing
/// to thread through beyond the one lookup.
fn scope_for_cwd(mimir: &Mimir, input: &serde_json::Value) -> Scope {
    input
        .get("cwd")
        .and_then(|v| v.as_str())
        .and_then(|c| mimir.project_for_cwd(std::path::Path::new(c)).ok().flatten())
        .map(|p| Scope::Project(p.id))
        .unwrap_or(Scope::Global)
}

/// `UserPromptSubmit`: nudge toward a deliberate `/clear`/`/compact` (or a
/// handoff-then-clear) once the transcript crosses the configured
/// threshold. Plain stdout, no JSON — see the module doc comment. Silent
/// whenever `[hooks] context_guard = "off"` (the default), below
/// threshold, or already nagged at this +10% band this session
/// (`context_guard::should_emit`).
pub fn prompt() -> Result<()> {
    let input = read_stdin_json();
    let mimir = Mimir::open()?;
    let mode = &mimir.config.hooks.context_guard;
    if mode == "off" {
        return Ok(());
    }
    let Some(session_id) = input.get("session_id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let bytes = transcript_bytes(&input);
    let marker = context_guard::transcript_marker(&mimir.conn, session_id).unwrap_or(0);
    let pct = context_guard::estimate_pct(bytes, marker, &mimir.config.hooks);
    let threshold = mimir.config.hooks.context_guard_threshold_pct;
    if context_guard::should_emit(&mimir.conn, session_id, pct, threshold).unwrap_or(false) {
        if let Some(msg) = context_guard::guard_message(pct, threshold, mode) {
            println!("{msg}");
        }
    }
    Ok(())
}

/// `PreCompact`: on an AUTO-triggered compaction only — never a user's
/// explicit `/compact`, which always stays silent — blocks it with the
/// guard message as the block reason once over threshold. Unlike
/// [`prompt`], there is no +10% nag cadence here: every auto-compact
/// attempt while over threshold is blocked, since letting even one through
/// silently would hand control of *when* a compact happens back to Claude
/// Code, defeating the point of `"pause"`/`"handoff"` mode.
pub fn precompact() -> Result<()> {
    let input = read_stdin_json();
    let mimir = Mimir::open()?;
    let mode = &mimir.config.hooks.context_guard;
    if mode == "off" {
        return Ok(());
    }
    if input.get("trigger").and_then(|v| v.as_str()) != Some("auto") {
        return Ok(()); // never block a user's explicit /compact
    }
    let Some(session_id) = input.get("session_id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let bytes = transcript_bytes(&input);
    let marker = context_guard::transcript_marker(&mimir.conn, session_id).unwrap_or(0);
    let pct = context_guard::estimate_pct(bytes, marker, &mimir.config.hooks);
    let threshold = mimir.config.hooks.context_guard_threshold_pct;
    if let Some(reason) = context_guard::guard_message(pct, threshold, mode) {
        println!("{}", serde_json::json!({ "decision": "block", "reason": reason }));
    }
    Ok(())
}

/// `SessionStart`: updates the transcript-marker/nag bookkeeping
/// (`context_guard::handle_session_start`) for every `source`, and in
/// `"handoff"` mode, restores the most recent `session-handoff`-tagged
/// memory as plain-stdout context right after a `/clear` or a `compact`.
/// Entirely skipped — no DB writes either — when `context_guard = "off"`,
/// so an off install leaves zero new `session_state` rows behind.
pub fn session_start() -> Result<()> {
    let input = read_stdin_json();
    let mimir = Mimir::open()?;
    let mode = &mimir.config.hooks.context_guard;
    if mode == "off" {
        return Ok(());
    }
    let Some(session_id) = input.get("session_id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let source = input.get("source").and_then(|v| v.as_str()).unwrap_or("startup");
    let bytes = transcript_bytes(&input);
    let _ = context_guard::handle_session_start(&mimir.conn, session_id, source, bytes);

    if mode == "handoff" && matches!(source, "clear" | "compact") {
        let scope = scope_for_cwd(&mimir, &input);
        if let Ok(Some(text)) = context_guard::restore_handoff_text(&mimir.conn, scope) {
            println!("{text}");
        }
    }
    Ok(())
}

/// `PreToolUse` (Bash/Edit/Write matcher): guard anchors — see
/// `mimir_core::anchors`. Independent of `[hooks] context_guard`: dormant
/// until a memory declares `meta.anchors` via `mimir remember --anchor`.
pub fn pretool() -> Result<()> {
    let input = read_stdin_json();
    let mimir = Mimir::open()?;
    let Some(session_id) = input.get("session_id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let tool_name = input.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
    let (target, is_command) = match tool_name {
        "Bash" => (input.pointer("/tool_input/command").and_then(|v| v.as_str()), true),
        "Edit" | "Write" | "MultiEdit" => {
            (input.pointer("/tool_input/file_path").and_then(|v| v.as_str()), false)
        }
        "NotebookEdit" => (
            input.pointer("/tool_input/notebook_path").and_then(|v| v.as_str()),
            false,
        ),
        _ => (None, false),
    };
    let Some(target) = target else { return Ok(()) };
    let scope = scope_for_cwd(&mimir, &input);
    if let Ok(Some(text)) =
        mimir_core::anchors::find_and_record(&mimir.conn, session_id, scope, target, is_command)
    {
        println!(
            "{}",
            serde_json::json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": text
                }
            })
        );
    }
    Ok(())
}
