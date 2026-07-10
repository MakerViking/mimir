//! Context-window guard: pure estimation + messaging logic behind `mimir
//! context-guard` (see `crates/mimir-cli/src/context_guard_cmd.rs` for the
//! hook-event subcommands that call into this module).
//!
//! Clean-room design, off by default (`[hooks] context_guard = "off"`, see
//! `config::HooksConfig`). Two modes:
//! - `"pause"` — nudges the *user* to deliberately `/clear`/`/compact` once
//!   the transcript crosses a configurable percent of the model's context
//!   window, instead of losing control of *when* an auto-compact happens.
//! - `"handoff"` — instructs the *agent* to write one structured handoff
//!   memory (tagged `session-handoff`) before the user clears, then
//!   auto-restores it on the next `SessionStart`.
//!
//! Sizing is a byte-count approximation of the transcript file, not real
//! tokenization — cheap enough to run on every prompt, deliberately tunable
//! (`[hooks] transcript_bytes_per_token`) rather than exact. All DB access
//! here fails open: a missing session, missing table row, or query error
//! degrades to "estimate from the full transcript" / "don't nag" rather
//! than blocking or erroring the hook.

use rusqlite::{params, Connection, OptionalExtension};

use crate::config::HooksConfig;
use crate::error::Result;
use crate::memory;
use crate::model::{now_unix, Scope};

/// Estimate the percent of the context window the transcript has used since
/// `marker_bytes` (the size recorded at the last `/clear`/`/compact`, or 0
/// for a fresh session — see [`handle_session_start`]). Pure and clamped to
/// 0-100; never panics on a marker ahead of `transcript_bytes_now` (clock
/// skew, a marker written after this read raced it) — that just clamps to 0.
pub fn estimate_pct(transcript_bytes_now: u64, marker_bytes: u64, cfg: &HooksConfig) -> u8 {
    let new_bytes = transcript_bytes_now.saturating_sub(marker_bytes) as f64;
    let bytes_per_token = f64::from(cfg.transcript_bytes_per_token).max(0.01);
    let window_tokens = f64::from(cfg.context_window_tokens.max(1));
    let pct = (new_bytes / bytes_per_token / window_tokens) * 100.0;
    pct.clamp(0.0, 100.0).round() as u8
}

/// The guard's message for a firing at `pct`, or `None` below `threshold`.
/// Content differs by mode; an unrecognized mode (including `"off"`, which
/// callers should already have short-circuited on) is silent.
pub fn guard_message(pct: u8, threshold: u8, mode: &str) -> Option<String> {
    if pct < threshold {
        return None;
    }
    match mode {
        "pause" => Some(format!(
            "[mimir context guard] This session's context is ~{pct}% full. Consider a \
             deliberate `/clear` or `/compact` now, on your own terms, rather than letting \
             it happen mid-task. To keep continuity across the clear, save a handoff first: \
             `mimir remember \"<what's in flight, next step>\" -t note --tags session-handoff`."
        )),
        "handoff" => Some(format!(
            "[mimir context guard] This session's context is ~{pct}% full. Before anything \
             else, write ONE structured handoff memory summarizing exactly what's in flight \
             and the next concrete step:\n  mimir remember \"<handoff>\" -t note --tags \
             session-handoff\nThen tell the user to run `/clear` — the next session \
             auto-restores this handoff on SessionStart."
        )),
        _ => None,
    }
}

/// Decide whether this `pct` should actually emit a message: the first time
/// it crosses `threshold` in this session, then again only every +10pct
/// after that (via the `guard_last_pct` session_state key) — never every
/// prompt. Updates `guard_last_pct` when it returns `true`. Fails open: a
/// read/write error is logged by the caller's `Result`, not swallowed here,
/// but callers on the hook path should treat any `Err` as "don't emit"
/// rather than surfacing it to the user.
pub fn should_emit(conn: &Connection, session_id: &str, pct: u8, threshold: u8) -> Result<bool> {
    if pct < threshold {
        return Ok(false);
    }
    let last: Option<u8> =
        get_session_state(conn, session_id, "guard_last_pct")?.and_then(|s| s.parse().ok());
    let emit = match last {
        None => true,
        Some(last) => pct >= last.saturating_add(10),
    };
    if emit {
        set_session_state(conn, session_id, "guard_last_pct", &pct.to_string())?;
    }
    Ok(emit)
}

/// Handle a `SessionStart` event's marker semantics:
/// - `"startup"` (a genuinely new session): clear the marker and the nag
///   counter, so estimation starts from an empty transcript.
/// - `"clear"` | `"compact"`: record the transcript's current byte size as
///   the new marker, so later estimates only count content written *after*
///   this point — and clear the nag counter, since the percent is about to
///   drop back near zero.
/// - anything else (`"resume"`, future sources): leave both alone — the
///   transcript continues from where it was, so the existing marker/nag
///   state is still correct.
pub fn handle_session_start(
    conn: &Connection,
    session_id: &str,
    source: &str,
    transcript_bytes: u64,
) -> Result<()> {
    match source {
        "startup" => {
            clear_session_state(conn, session_id, "transcript_marker")?;
            clear_session_state(conn, session_id, "guard_last_pct")?;
        }
        "clear" | "compact" => {
            set_session_state(
                conn,
                session_id,
                "transcript_marker",
                &transcript_bytes.to_string(),
            )?;
            clear_session_state(conn, session_id, "guard_last_pct")?;
        }
        _ => {}
    }
    Ok(())
}

/// Current transcript marker for a session (0 if absent — fail-open: a
/// missing marker means "estimate from the whole transcript").
pub fn transcript_marker(conn: &Connection, session_id: &str) -> Result<u64> {
    Ok(get_session_state(conn, session_id, "transcript_marker")?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0))
}

/// The most recent `session-handoff`-tagged memory's body for `scope`,
/// prefixed for direct SessionStart stdout injection, or `None` if there
/// isn't one. Reuses `memory::list`'s existing tag filter rather than a new
/// query — it already returns newest-first.
pub fn restore_handoff_text(conn: &Connection, scope: Scope) -> Result<Option<String>> {
    let hits = memory::list(conn, scope, None, Some("session-handoff"), 1)?;
    Ok(hits
        .into_iter()
        .next()
        .and_then(|n| n.body)
        .map(|body| format!("Restored session handoff:\n{body}")))
}

fn get_session_state(conn: &Connection, session_id: &str, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT value FROM session_state WHERE session_id = ?1 AND key = ?2",
            params![session_id, key],
            |r| r.get(0),
        )
        .optional()?)
}

fn set_session_state(conn: &Connection, session_id: &str, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO session_state (session_id, key, value, updated_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        params![session_id, key, value, now_unix()],
    )?;
    Ok(())
}

fn clear_session_state(conn: &Connection, session_id: &str, key: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM session_state WHERE session_id = ?1 AND key = ?2",
        params![session_id, key],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn cfg() -> HooksConfig {
        HooksConfig {
            context_guard: "pause".into(),
            context_guard_threshold_pct: 45,
            context_window_tokens: 1000,
            transcript_bytes_per_token: 8.0,
            ..Default::default()
        }
    }

    #[test]
    fn estimate_pct_math_and_clamping() {
        let c = cfg();
        // 1000 tokens * 8 bytes/token = 8000 bytes fills the whole window.
        assert_eq!(estimate_pct(0, 0, &c), 0);
        assert_eq!(estimate_pct(3600, 0, &c), 45); // 3600/8/1000*100 = 45
        assert_eq!(estimate_pct(8000, 0, &c), 100);
        assert_eq!(estimate_pct(16000, 0, &c), 100, "clamped, never over 100");
        // Marker subtracts prior content.
        assert_eq!(estimate_pct(4000, 4000, &c), 0);
        assert_eq!(estimate_pct(7600, 4000, &c), 45);
        // A marker ahead of "now" (race/clock skew) clamps to 0, not panic.
        assert_eq!(estimate_pct(100, 4000, &c), 0);
    }

    #[test]
    fn guard_message_respects_threshold_and_mode() {
        assert!(guard_message(44, 45, "pause").is_none(), "below threshold");
        let pause = guard_message(45, 45, "pause").unwrap();
        assert!(pause.contains("45%"));
        assert!(pause.contains("/clear"));
        let handoff = guard_message(50, 45, "handoff").unwrap();
        assert!(handoff.contains("mimir remember"));
        assert!(handoff.contains("session-handoff"));
        assert!(
            guard_message(90, 45, "off").is_none(),
            "unrecognized mode is silent"
        );
    }

    #[test]
    fn should_emit_fires_once_at_crossing_then_every_ten_points() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        // Below threshold: never emits, never records state.
        assert!(!should_emit(&conn, sid, 40, 45).unwrap());
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            None
        );
        // First crossing: emits.
        assert!(should_emit(&conn, sid, 45, 45).unwrap());
        // Same band again: silent (no new +10 crossed).
        assert!(!should_emit(&conn, sid, 48, 45).unwrap());
        assert!(!should_emit(&conn, sid, 54, 45).unwrap());
        // +10 over the last emitted pct: fires again.
        assert!(should_emit(&conn, sid, 55, 45).unwrap());
        assert!(!should_emit(&conn, sid, 60, 45).unwrap());
        assert!(should_emit(&conn, sid, 65, 45).unwrap());
    }

    #[test]
    fn should_emit_is_independent_per_session() {
        let conn = db::open_in_memory().unwrap();
        assert!(should_emit(&conn, "a", 50, 45).unwrap());
        assert!(
            should_emit(&conn, "b", 50, 45).unwrap(),
            "a fresh session isn't nagged-out by another"
        );
    }

    #[test]
    fn session_start_startup_clears_marker_and_nag() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        set_session_state(&conn, sid, "transcript_marker", "9000").unwrap();
        set_session_state(&conn, sid, "guard_last_pct", "70").unwrap();
        handle_session_start(&conn, sid, "startup", 0).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 0);
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            None
        );
    }

    #[test]
    fn session_start_clear_and_compact_record_marker_and_reset_nag() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        set_session_state(&conn, sid, "guard_last_pct", "70").unwrap();
        handle_session_start(&conn, sid, "clear", 12345).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 12345);
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            None
        );

        handle_session_start(&conn, sid, "compact", 999).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 999);
    }

    #[test]
    fn session_start_resume_leaves_marker_and_nag_untouched() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        set_session_state(&conn, sid, "transcript_marker", "4242").unwrap();
        set_session_state(&conn, sid, "guard_last_pct", "55").unwrap();
        handle_session_start(&conn, sid, "resume", 999_999).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 4242);
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            Some("55".into())
        );
    }

    #[test]
    fn transcript_marker_is_zero_when_absent_fail_open() {
        let conn = db::open_in_memory().unwrap();
        assert_eq!(transcript_marker(&conn, "never-seen").unwrap(), 0);
    }

    #[test]
    fn restore_handoff_text_finds_the_newest_tagged_memory() {
        let conn = db::open_in_memory().unwrap();
        let old = memory::remember(
            &conn,
            memory::Remember {
                text: "old handoff: refactor foo".into(),
                mtype: crate::model::MemoryType::Note,
                tags: vec!["session-handoff".into()],
                project_id: None,
                force: false,
            },
        )
        .unwrap();
        // Backdate instead of sleeping across a real second boundary — `list`
        // orders by `created_at`, which is second-resolution.
        if let memory::RememberOutcome::Created(node) = old {
            conn.execute(
                "UPDATE node SET created_at = created_at - 10 WHERE id = ?1",
                params![node.id],
            )
            .unwrap();
        }
        memory::remember(
            &conn,
            memory::Remember {
                text: "new handoff: fix bar next".into(),
                mtype: crate::model::MemoryType::Note,
                tags: vec!["session-handoff".into()],
                project_id: None,
                force: false,
            },
        )
        .unwrap();
        let text = restore_handoff_text(&conn, Scope::Global).unwrap().unwrap();
        assert!(text.starts_with("Restored session handoff:"));
        assert!(text.contains("fix bar next"), "picks the newest: {text}");
    }

    #[test]
    fn restore_handoff_text_is_none_without_a_tagged_memory() {
        let conn = db::open_in_memory().unwrap();
        assert_eq!(restore_handoff_text(&conn, Scope::Global).unwrap(), None);
    }
}
