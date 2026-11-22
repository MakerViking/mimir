//! `mimir context-guard <subcommand>` — the hook-event entry points behind
//! [`mimir_core::context_guard`] (context-window pause/handoff nudges) and
//! [`mimir_core::anchors`] (guard anchors). Each subcommand reads one hook
//! event's JSON from stdin.
//!
//! This module also owns [`HookEvent`], the one reader of that JSON, and
//! [`fail_open`], the one wrapper every hook entry point returns through —
//! the session brief (`brief_cmd`) and the rules hook (`rules_cmd`) use
//! both, so no hook reads the event or handles an error its own way.
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
//! treated as "say nothing", never a nonzero exit — a hook that can break a
//! prompt/tool-call/compact by erroring is worse than one that occasionally
//! stays quiet. An error that stopped a hook is written to stderr, which
//! the wrapper scripts discard and which, on exit 0, is never fed to the
//! model — so it is there for whoever goes looking, and nowhere else.

use std::io::Read as _;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use mimir_core::config::ContextGuardMode;
use mimir_core::context_guard::{self, SessionSource};
use mimir_core::model::{Node, Scope};
use mimir_core::{Mimir, ProjectContext};

/// One Claude Code hook event, as far as any Mimir hook reads it.
///
/// Parsing never fails: input that is not a JSON object is an event with
/// nothing in it, and every accessor answers "absent". What a hook does
/// with an absent field is its own rule; how the field is read is not.
pub(crate) struct HookEvent {
    raw: serde_json::Value,
}

impl HookEvent {
    /// Read an event from its JSON text.
    pub(crate) fn parse(text: &str) -> HookEvent {
        HookEvent {
            raw: serde_json::from_str(text).unwrap_or(serde_json::Value::Null),
        }
    }

    /// Read the event Claude Code pipes to a hook: all of stdin, to EOF.
    /// For the commands that are only ever run as hooks.
    pub(crate) fn from_stdin() -> HookEvent {
        let mut buf = String::new();
        if std::io::stdin().read_to_string(&mut buf).is_err() {
            buf.clear();
        }
        HookEvent::parse(&buf)
    }

    /// The event on stdin, if there is one — for a command that is both a
    /// hook and something a person types (`mimir rules show`).
    ///
    /// `None` when stdin is a terminal, when it carries no JSON object, or
    /// when nothing closes it within `wait`. The bound is what makes this
    /// safe to call from a hand-typed command: a hook's event arrives the
    /// moment the process starts, but a stdin inherited from a wrapper that
    /// never closes it would otherwise hold the command forever.
    pub(crate) fn from_piped_stdin(wait: Duration) -> Option<HookEvent> {
        use std::io::IsTerminal as _;
        if std::io::stdin().is_terminal() {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = std::io::stdin().read_to_string(&mut buf);
            let _ = tx.send(buf);
        });
        let event = HookEvent::parse(&rx.recv_timeout(wait).ok()?);
        event.raw.is_object().then_some(event)
    }

    fn str_field(&self, pointer: &str) -> Option<&str> {
        self.raw.pointer(pointer).and_then(|v| v.as_str())
    }

    /// The session the event belongs to. An empty id is no id: keyed state
    /// under `""` would be shared by every session that sent none.
    pub(crate) fn session_id(&self) -> Option<&str> {
        self.str_field("/session_id").filter(|s| !s.is_empty())
    }

    /// What a `SessionStart` event says started the session; see
    /// [`SessionSource::parse`] for the missing and the unknown case.
    pub(crate) fn source(&self) -> SessionSource {
        SessionSource::parse(self.str_field("/source"))
    }

    /// The directory the session is working in.
    pub(crate) fn cwd(&self) -> Option<&Path> {
        self.str_field("/cwd")
            .filter(|c| !c.is_empty())
            .map(Path::new)
    }

    /// `PreCompact`'s trigger: `"auto"` or `"manual"`.
    pub(crate) fn trigger(&self) -> Option<&str> {
        self.str_field("/trigger")
    }

    /// Current size of the session's transcript file; 0 when the event
    /// names none or it cannot be read.
    pub(crate) fn transcript_bytes(&self) -> u64 {
        self.str_field("/transcript_path")
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// The projects the event's `cwd` resolves to
    /// (`Mimir::projects_for_cwd`), or `None` when the event carries no
    /// `cwd`. A detection error is logged and reads as no project: every
    /// caller is a hook, and a hook narrows its scope rather than failing.
    fn projects(&self, mimir: &Mimir) -> Option<ProjectContext> {
        let cwd = self.cwd()?;
        match mimir.projects_for_cwd(cwd) {
            Ok(projects) => Some(projects),
            Err(err) => {
                tracing::warn!(%err, cwd = %cwd.display(), "hook: project detection failed");
                None
            }
        }
    }

    /// The project whose memories the event's session works with: the
    /// memory project of its `cwd`, which in a git linked worktree is the
    /// main checkout's. `None` outside any project and when the event
    /// carries no `cwd`. Everything a hook volunteers is a memory, so this
    /// is the project every hook scopes by.
    pub(crate) fn project(&self, mimir: &Mimir) -> Option<Node> {
        self.projects(mimir)?.memory
    }

    /// The scope for what a hook volunteers into this session:
    /// [`mimir_core::scope::push_scope`] of the event's project.
    pub(crate) fn push_scope(&self, mimir: &Mimir) -> Scope {
        mimir_core::scope::push_scope(self.project(mimir).map(|p| p.id))
    }

    /// Where the event's session is, from one resolution: the scope a hook
    /// volunteers into ([`HookEvent::push_scope`]), and the checkout a
    /// session handoff is tied to.
    pub(crate) fn place(&self, mimir: &Mimir) -> Place {
        let projects = self.projects(mimir);
        Place {
            scope: mimir_core::scope::push_scope(projects.as_ref().and_then(|p| p.memory_id())),
            checkout: projects
                .and_then(|p| p.checkout)
                .and_then(|checkout| checkout.path),
        }
    }

    /// What a `PreToolUse` event is about to act on, for the guard anchors:
    /// the target text and whether it is a shell command rather than a
    /// path. `None` for a tool [`ANCHOR_TOOLS`] does not list.
    pub(crate) fn anchor_target(&self) -> Option<(&str, bool)> {
        let tool = self.str_field("/tool_name")?;
        let (_, pointer, is_command) = ANCHOR_TOOLS.iter().find(|(name, ..)| *name == tool)?;
        Some((self.str_field(pointer)?, *is_command))
    }
}

/// Where a hook event's session is working; see [`HookEvent::place`].
pub(crate) struct Place {
    /// [`mimir_core::scope::push_scope`] of the session's memory project.
    pub(crate) scope: Scope,
    /// The canonical root of the session's checkout, the identity a handoff
    /// records (`context_guard::handoff_is_for`). `None` outside any
    /// project.
    pub(crate) checkout: Option<String>,
}

/// Every tool the guard-anchors hook reads a target from: its name, where
/// in the event the target is, and whether that target is a shell command.
/// The `PreToolUse` matcher `mimir init --hooks` installs is built from
/// this table ([`anchor_matcher`]), so the hook is never registered for a
/// tool it ignores, nor blind to one it handles.
pub(crate) const ANCHOR_TOOLS: &[(&str, &str, bool)] = &[
    ("Bash", "/tool_input/command", true),
    ("Edit", "/tool_input/file_path", false),
    ("Write", "/tool_input/file_path", false),
    ("MultiEdit", "/tool_input/file_path", false),
    ("NotebookEdit", "/tool_input/notebook_path", false),
];

/// The `PreToolUse` matcher covering exactly [`ANCHOR_TOOLS`].
pub(crate) fn anchor_matcher() -> String {
    ANCHOR_TOOLS
        .iter()
        .map(|(name, ..)| *name)
        .collect::<Vec<_>>()
        .join("|")
}

/// Run a hook entry point so that it cannot fail the event it is attached
/// to: an error becomes one line on stderr and the command still exits 0
/// with whatever it had already printed.
///
/// Every hook-facing command returns through here, including the ones
/// installed behind a wrapper script that masks the exit code anyway — the
/// guarantee belongs to the command, not to how it happened to be
/// installed.
pub(crate) fn fail_open(what: &str, run: impl FnOnce() -> Result<()>) -> Result<()> {
    if let Err(err) = run() {
        eprintln!("mimir {what}: {err:#} (hook failed open; nothing more was injected)");
    }
    Ok(())
}

/// A nudge that is due, and the percent it should be recorded at once it
/// has been printed.
struct Nudge {
    text: String,
    pct: u8,
}

/// The session's estimated context use, in percent of the window.
fn context_pct(mimir: &Mimir, session_id: &str, event: &HookEvent) -> u8 {
    let marker = context_guard::transcript_marker(&mimir.conn, session_id).unwrap_or(0);
    context_guard::estimate_pct(event.transcript_bytes(), marker, &mimir.config.hooks)
}

/// The `UserPromptSubmit` nudge for this event, if one is due. Reads state
/// and writes none: [`prompt`] records the nag after printing.
fn prompt_nudge(mimir: &Mimir, event: &HookEvent) -> Option<Nudge> {
    let mode = mimir.config.hooks.context_guard();
    if mode == ContextGuardMode::Off {
        return None;
    }
    let session_id = event.session_id()?;
    let pct = context_pct(mimir, session_id, event);
    let threshold = mimir.config.hooks.context_guard_threshold_pct;
    if !context_guard::nag_due(&mimir.conn, session_id, pct, threshold).unwrap_or(false) {
        return None;
    }
    let text = context_guard::guard_message_for(pct, threshold, mode)?;
    Some(Nudge { text, pct })
}

/// `UserPromptSubmit`: nudge toward a deliberate `/clear`/`/compact` (or a
/// handoff-then-clear) once the transcript crosses the configured
/// threshold. Plain stdout, no JSON — see the module doc comment. Silent
/// whenever the guard is off (the default, and what a misspelled mode is),
/// below threshold, or already nagged at this +10% band this session
/// (`context_guard::nag_due`). The nag is recorded after the nudge is
/// printed, so a nudge that was never emitted cannot silence the next one.
pub fn prompt() -> Result<()> {
    fail_open("context-guard prompt", || {
        let event = HookEvent::from_stdin();
        let mimir = Mimir::open()?;
        if let Some(nudge) = prompt_nudge(&mimir, &event) {
            println!("{}", nudge.text);
            if let Some(session_id) = event.session_id() {
                let _ = context_guard::record_nag(&mimir.conn, session_id, nudge.pct);
            }
        }
        Ok(())
    })
}

/// The `PreCompact` block reason for this event, if the compaction should
/// be blocked.
fn precompact_block(mimir: &Mimir, event: &HookEvent) -> Option<String> {
    let mode = mimir.config.hooks.context_guard();
    if mode == ContextGuardMode::Off {
        return None;
    }
    if event.trigger() != Some("auto") {
        return None; // never block a user's explicit /compact
    }
    let session_id = event.session_id()?;
    let pct = context_pct(mimir, session_id, event);
    context_guard::guard_message_for(pct, mimir.config.hooks.context_guard_threshold_pct, mode)
}

/// `PreCompact`: on an AUTO-triggered compaction only — never a user's
/// explicit `/compact`, which always stays silent — blocks it with the
/// guard message as the block reason once over threshold. Unlike
/// [`prompt`], there is no +10% nag cadence here: every auto-compact
/// attempt while over threshold is blocked, since letting even one through
/// silently would hand control of *when* a compact happens back to Claude
/// Code, defeating the point of `"pause"`/`"handoff"` mode.
pub fn precompact() -> Result<()> {
    fail_open("context-guard precompact", || {
        let event = HookEvent::from_stdin();
        let mimir = Mimir::open()?;
        if let Some(reason) = precompact_block(&mimir, &event) {
            println!(
                "{}",
                serde_json::json!({ "decision": "block", "reason": reason })
            );
        }
        Ok(())
    })
}

/// Apply a `SessionStart` event: move the marker/nag state, and return the
/// handoff text to restore, if this event restores one.
fn session_start_restore(mimir: &Mimir, event: &HookEvent) -> Option<(i64, String)> {
    let mode = mimir.config.hooks.context_guard();
    if mode == ContextGuardMode::Off {
        return None;
    }
    let session_id = event.session_id()?;
    let source = event.source();
    let _ =
        context_guard::on_session_start(&mimir.conn, session_id, source, event.transcript_bytes());
    if mode != ContextGuardMode::Handoff || !source.is_context_reset() {
        return None;
    }
    // The handoff written in this checkout: the scope is the memory
    // project, which a session in another worktree of the repository
    // shares, and its handoff is not this session's to resume from.
    let place = event.place(mimir);
    context_guard::pending_handoff_in(&mimir.conn, place.scope, place.checkout.as_deref())
        .ok()
        .flatten()
        .map(|(node, text)| (node.id, text))
}

/// `SessionStart`: updates the transcript-marker/nag bookkeeping
/// (`context_guard::on_session_start`) for every `source`, and in
/// `"handoff"` mode restores the pending `session-handoff` memory as
/// plain-stdout context right after a `/clear` or a compaction — once: the
/// restore consumes it (`context_guard::restore_handoff_text`).
/// Entirely skipped — no DB writes either — when the guard is off, so an
/// off install leaves zero new `session_state` rows behind.
pub fn session_start() -> Result<()> {
    fail_open("context-guard session-start", || {
        let event = HookEvent::from_stdin();
        let mimir = Mimir::open()?;
        deliver_handoff(&mimir, &event, &mut std::io::stdout().lock())
    })
}

/// Write the handoff this event restores, if any, to `out`, and only then
/// mark it restored.
///
/// Delivered first, consumed second: if the reader is gone the write fails
/// (or kills the process) before the handoff is marked, and the next clear
/// still has it. The other order loses a handoff for good whenever the hook
/// is cut off between the two.
fn deliver_handoff(mimir: &Mimir, event: &HookEvent, out: &mut impl std::io::Write) -> Result<()> {
    if let Some((handoff, text)) = session_start_restore(mimir, event) {
        writeln!(out, "{text}")?;
        out.flush()?;
        context_guard::mark_handoff_restored(&mimir.conn, handoff);
    }
    Ok(())
}

/// `PreToolUse` (the [`anchor_matcher`] tools): guard anchors — see
/// `mimir_core::anchors`. Independent of `[hooks] context_guard`: dormant
/// until a memory declares `meta.anchors` via `mimir remember --anchor`.
pub fn pretool() -> Result<()> {
    fail_open("context-guard pretool", || {
        let event = HookEvent::from_stdin();
        let mimir = Mimir::open()?;
        let Some(session_id) = event.session_id() else {
            return Ok(());
        };
        let Some((target, is_command)) = event.anchor_target() else {
            return Ok(());
        };
        let scope = event.push_scope(&mimir);
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::config::Paths;
    use mimir_core::store;

    /// A store in a temp dir with the guard in `mode`, and a transcript
    /// file big enough to read as 50% of a 1000-token window.
    fn guarded(mode: &str) -> (tempfile::TempDir, Mimir, String) {
        let dir = tempfile::tempdir().unwrap();
        let mut mimir = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        mimir.config.hooks.context_guard = mode.into();
        mimir.config.hooks.context_guard_threshold_pct = 45;
        mimir.config.hooks.context_window_tokens = 1000;
        mimir.config.hooks.transcript_bytes_per_token = 8.0;
        let transcript = dir.path().join("transcript.jsonl");
        std::fs::write(&transcript, vec![b'x'; 4000]).unwrap();
        let transcript = transcript.to_string_lossy().into_owned();
        (dir, mimir, transcript)
    }

    fn event(fields: serde_json::Value) -> HookEvent {
        HookEvent::parse(&fields.to_string())
    }

    fn session_rows(mimir: &Mimir) -> i64 {
        mimir
            .conn
            .query_row("SELECT count(*) FROM session_state", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn an_event_that_is_not_json_has_nothing_in_it() {
        for text in ["", "not json", "[1, 2]", "null"] {
            let e = HookEvent::parse(text);
            assert_eq!(e.session_id(), None, "{text:?}");
            assert_eq!(e.cwd(), None);
            assert_eq!(e.trigger(), None);
            assert_eq!(e.transcript_bytes(), 0);
            assert_eq!(e.anchor_target(), None);
            // No `source` at all is a start-up, by the one rule.
            assert_eq!(e.source(), SessionSource::Startup);
        }
    }

    #[test]
    fn an_empty_session_id_or_cwd_is_absent() {
        let e = event(serde_json::json!({ "session_id": "", "cwd": "" }));
        assert_eq!(e.session_id(), None);
        assert_eq!(e.cwd(), None);
        let e =
            event(serde_json::json!({ "session_id": "s1", "cwd": "/tmp/x", "source": "compact" }));
        assert_eq!(e.session_id(), Some("s1"));
        assert_eq!(e.cwd(), Some(Path::new("/tmp/x")));
        assert_eq!(e.source(), SessionSource::Compact);
    }

    /// Outside a project a hook pushes global memories only; inside one,
    /// that project plus global. Neither ever widens to every project.
    #[test]
    fn hooks_push_into_the_events_project_or_global_only() {
        let (dir, mimir, _) = guarded("off");
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let nested = repo.join("src");
        std::fs::create_dir_all(&nested).unwrap();
        let bare = dir.path().join("scratch");
        std::fs::create_dir_all(&bare).unwrap();

        let inside = event(serde_json::json!({ "cwd": nested }));
        let project = inside.project(&mimir).expect("the repo is a project");
        assert_eq!(inside.push_scope(&mimir), Scope::Project(project.id));

        let outside = event(serde_json::json!({ "cwd": bare }));
        assert_eq!(outside.push_scope(&mimir), Scope::Global);
        assert_eq!(
            event(serde_json::json!({})).push_scope(&mimir),
            Scope::Global
        );
    }

    #[test]
    fn the_matcher_names_every_tool_the_handler_reads_and_no_other() {
        assert_eq!(anchor_matcher(), "Bash|Edit|Write|MultiEdit|NotebookEdit");
        for (tool, pointer, is_command) in ANCHOR_TOOLS {
            // Build the event the table says this tool sends.
            let key = pointer.rsplit('/').next().unwrap();
            let e = event(serde_json::json!({
                "tool_name": tool,
                "tool_input": { key: "the-target" }
            }));
            assert_eq!(
                e.anchor_target(),
                Some(("the-target", *is_command)),
                "{tool}"
            );
        }
        let other = event(serde_json::json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/lib.rs" }
        }));
        assert_eq!(other.anchor_target(), None);
    }

    /// The nudge is decided without writing anything; only `prompt`, after
    /// printing, records the nag.
    #[test]
    fn deciding_a_nudge_is_due_records_nothing() {
        let (_dir, mimir, transcript) = guarded("pause");
        let e = event(serde_json::json!({ "session_id": "s1", "transcript_path": transcript }));
        let nudge = prompt_nudge(&mimir, &e).expect("50% is over the 45% threshold");
        assert_eq!(nudge.pct, 50);
        assert!(nudge.text.contains("~50% full"), "{}", nudge.text);
        assert_eq!(session_rows(&mimir), 0, "asking must not record");
        assert!(prompt_nudge(&mimir, &e).is_some(), "still due");

        context_guard::record_nag(&mimir.conn, "s1", nudge.pct).unwrap();
        assert!(prompt_nudge(&mimir, &e).is_none(), "nagged at this band");
    }

    /// A mode config.toml misspells used to pass every `== "off"` gate,
    /// write the nag and marker rows, and then print nothing. It is off.
    #[test]
    fn a_misspelled_mode_is_off_at_every_gate() {
        let (_dir, mimir, transcript) = guarded("handof");
        let e = event(serde_json::json!({
            "session_id": "s1",
            "transcript_path": transcript,
            "trigger": "auto",
            "source": "clear",
        }));
        assert!(prompt_nudge(&mimir, &e).is_none());
        assert!(precompact_block(&mimir, &e).is_none());
        assert!(session_start_restore(&mimir, &e).is_none());
        assert_eq!(
            session_rows(&mimir),
            0,
            "an off guard writes no session state"
        );
    }

    #[test]
    fn precompact_blocks_only_an_automatic_compaction_over_threshold() {
        let (_dir, mimir, transcript) = guarded("pause");
        let auto = event(serde_json::json!({
            "session_id": "s1", "transcript_path": transcript, "trigger": "auto"
        }));
        assert!(precompact_block(&mimir, &auto)
            .unwrap()
            .contains("~50% full"));
        // No nag cadence here: the second attempt is blocked like the first.
        assert!(precompact_block(&mimir, &auto).is_some());
        let manual = event(serde_json::json!({
            "session_id": "s1", "transcript_path": transcript, "trigger": "manual"
        }));
        assert!(precompact_block(&mimir, &manual).is_none());
    }

    /// Handoff mode restores on a reset, once, and a start-up or a resume
    /// restores nothing. A missing `source` is a start-up for the marker.
    #[test]
    fn session_start_restores_a_handoff_once_and_only_after_a_reset() {
        let (_dir, mimir, transcript) = guarded("handoff");
        mimir_core::memory::remember(
            &mimir.conn,
            mimir_core::memory::Remember {
                text: "handoff: the migration is half applied; run step 4 next".into(),
                mtype: mimir_core::model::MemoryType::Note,
                tags: vec![context_guard::HANDOFF_TAG.into()],
                project_id: None,
                force: false,
                actor: mimir_core::model::Actor::Agent,
            },
        )
        .unwrap();
        let at = |source: Option<&str>| {
            let mut fields = serde_json::json!({
                "session_id": "s1", "transcript_path": transcript
            });
            if let Some(source) = source {
                fields["source"] = source.into();
            }
            session_start_restore(&mimir, &event(fields))
        };

        assert_eq!(at(Some("startup")), None);
        assert_eq!(at(Some("resume")), None);
        store::set_session_state(&mimir.conn, "s1", "guard_last_pct", "70").unwrap();
        assert_eq!(
            at(None),
            None,
            "no source: a start-up, which restores nothing"
        );
        assert_eq!(
            store::get_session_state(&mimir.conn, "s1", "guard_last_pct").unwrap(),
            None,
            "and, as a start-up, it reset the nag state"
        );

        let (handoff, restored) = at(Some("clear")).expect("a reset restores the handoff");
        assert!(
            restored.starts_with("Restored session handoff:"),
            "{restored}"
        );
        assert!(restored.contains("run step 4 next"));
        assert_eq!(
            context_guard::transcript_marker(&mimir.conn, "s1").unwrap(),
            4000
        );
        // Producing the text consumes nothing: a hook that dies before it
        // has printed leaves the handoff for the next reset.
        assert_eq!(
            at(Some("compact")).map(|(id, _)| id),
            Some(handoff),
            "not delivered yet, so still pending"
        );

        // Delivery to a reader that is gone fails, and marks nothing.
        struct Gone;
        impl std::io::Write for Gone {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
        }
        let reset = event(serde_json::json!({
            "session_id": "s1", "transcript_path": transcript, "source": "clear"
        }));
        assert!(deliver_handoff(&mimir, &reset, &mut Gone).is_err());
        assert_eq!(
            at(Some("compact")).map(|(id, _)| id),
            Some(handoff),
            "the write failed, so the handoff is still pending"
        );
        // A reader that takes the bytes but cannot be flushed got nothing
        // it can be relied on to have seen, either.
        struct Unflushed(Vec<u8>);
        impl std::io::Write for Unflushed {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
        }
        assert!(deliver_handoff(&mimir, &reset, &mut Unflushed(Vec::new())).is_err());
        assert!(at(Some("compact")).is_some(), "not flushed: still pending");

        // Delivered: the text is out, and only then is the handoff spent.
        let mut out = Vec::new();
        deliver_handoff(&mimir, &reset, &mut out).unwrap();
        let printed = String::from_utf8(out).unwrap();
        assert!(
            printed.starts_with("Restored session handoff:"),
            "{printed}"
        );
        assert!(printed.contains("run step 4 next") && printed.ends_with('\n'));
        assert_eq!(at(Some("compact")), None, "already restored: not replayed");
        let mut again = Vec::new();
        deliver_handoff(&mimir, &reset, &mut again).unwrap();
        assert!(again.is_empty(), "nothing is delivered twice");
    }

    /// A repository's main checkout and two linked worktrees of it, as the
    /// files `git worktree add` leaves.
    fn repo_with_two_worktrees(dir: &Path) -> [std::path::PathBuf; 3] {
        let main = dir.join("main");
        let worktree = |name: &str| {
            let root = dir.join(name);
            let gitdir = main.join(".git/worktrees").join(name);
            std::fs::create_dir_all(&gitdir).unwrap();
            std::fs::create_dir_all(root.join("src")).unwrap();
            std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
            std::fs::write(root.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
            std::fs::write(
                gitdir.join("gitdir"),
                format!("{}\n", root.join(".git").display()),
            )
            .unwrap();
            root
        };
        let (wt_a, wt_b) = (worktree("wt-a"), worktree("wt-b"));
        [main, wt_a, wt_b]
    }

    /// Worktrees of one repository share a memory project, and sessions run
    /// in several at once. A handoff saved by the session in one of them is
    /// restored by a reset there, and by a reset in no other checkout.
    #[test]
    fn a_handoff_saved_in_one_worktree_is_restored_there_and_nowhere_else() {
        use crate::capture::{capture, Capture, Captured, MissingLink};
        let (dir, mut mimir, transcript) = guarded("handoff");
        let [main, wt_a, wt_b] = repo_with_two_worktrees(dir.path());

        // Saved from worktree A, the way `mimir remember` run there does.
        let here = mimir.projects_for_cwd(&wt_a).unwrap();
        assert_ne!(here.checkout_id(), here.memory_id());
        let saved = capture(
            &mut mimir,
            Capture {
                text: "handoff: A is mid-rebase; resolve src/lib.rs next".into(),
                mtype: mimir_core::model::MemoryType::Note,
                tags: vec![context_guard::HANDOFF_TAG.into()],
                project_id: here.memory_id(),
                working_project: here.checkout_id(),
                force: false,
                actor: mimir_core::model::Actor::Agent,
                surface: mimir_core::secrets::Surface::CliRemember,
                link: None,
                missing_link: MissingLink::Reject,
                fires_when: Vec::new(),
                anchors: Vec::new(),
                expires_in: None,
                resolves_when: None,
                confidence: None,
                valid_from: None,
                valid_to: None,
            },
        );
        assert!(matches!(saved, Ok(Captured::Stored(_))), "{saved:?}");

        let reset_in = |cwd: &Path, session: &str| {
            event(serde_json::json!({
                "session_id": session,
                "transcript_path": transcript,
                "source": "clear",
                "cwd": cwd,
            }))
        };
        // All three checkouts push into one scope, the repository's: the
        // scope alone cannot say whose handoff this is.
        let scope = reset_in(&wt_a, "a").push_scope(&mimir);
        assert_eq!(scope, Scope::Project(here.memory_id().unwrap()));
        assert_eq!(reset_in(&wt_b, "b").push_scope(&mimir), scope);
        assert_eq!(reset_in(&main, "m").push_scope(&mimir), scope);

        assert_eq!(
            session_start_restore(&mimir, &reset_in(&wt_b, "b")),
            None,
            "a session in worktree B must not resume from A's state"
        );
        assert_eq!(
            session_start_restore(&mimir, &reset_in(&main, "m")),
            None,
            "nor one in the main checkout"
        );
        // A session anywhere inside worktree A is in that checkout.
        let mut out = Vec::new();
        deliver_handoff(&mimir, &reset_in(&wt_a.join("src"), "a"), &mut out).unwrap();
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("A is mid-rebase"), "{printed}");
        assert_eq!(
            session_start_restore(&mimir, &reset_in(&wt_a, "a")),
            None,
            "restored once"
        );
    }

    #[test]
    fn a_failing_hook_reports_on_stderr_and_still_succeeds() {
        let ran = fail_open("test hook", || anyhow::bail!("the store is on fire"));
        assert!(ran.is_ok(), "a hook error must never become an exit code");
        assert!(fail_open("test hook", || Ok(())).is_ok());
    }
}
