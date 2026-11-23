//! Context-window guard: pure estimation + messaging logic behind `mimir
//! context-guard` (see `crates/mimir-cli/src/context_guard_cmd.rs` for the
//! hook-event subcommands that call into this module).
//!
//! Clean-room design, off by default (`[hooks] context_guard = "off"`, see
//! [`ContextGuardMode`]). Two modes:
//! - `"pause"` — nudges the *user* to deliberately `/clear`/`/compact` once
//!   the transcript crosses a configurable percent of the model's context
//!   window, instead of losing control of *when* an auto-compact happens.
//! - `"handoff"` — instructs the *agent* to write one structured handoff
//!   memory (tagged `session-handoff`) before the user clears, then
//!   restores it, once, into the session that follows.
//!
//! Sizing is a byte-count approximation of the transcript file, not real
//! tokenization — cheap enough to run on every prompt, deliberately tunable
//! (`[hooks] transcript_bytes_per_token`) rather than exact. All DB access
//! here fails open: a missing session, missing table row, or query error
//! degrades to "estimate from the full transcript" / "don't nag" rather
//! than blocking or erroring the hook.
//!
//! # The states this module names
//!
//! ```text
//!   per session   nag:      never ── nudge printed at pct ──▶ nagged(pct)
//!                           nagged(p) ── nudge printed at ≥ p+10 ──▶ nagged(pct)
//!                           any ── SessionStart startup | clear | compact ──▶ never
//!
//!   per handoff   pending ── restored into a session ──▶ restored (meta.restored_at)
//! ```
//!
//! The nag state moves only when a nudge was actually printed
//! ([`nag_due`] decides, [`record_nag`] is called after the print). A
//! handoff is a snapshot for *the next* session, so it is consumed by the
//! restore: [`restore_handoff_text`] marks it, and [`current_handoff`] no
//! longer returns it.
//!
//! A handoff also stays with the checkout it was written in
//! (`meta.handoff_checkout`, see [`handoff_is_for`]). Its scope is the
//! memory project, which every linked worktree of a repository shares with
//! the main checkout, and sessions run in several of them at once: without
//! the checkout, the first of them to clear would be handed whichever
//! snapshot was written last, by any of them. The lookups that take a
//! checkout end in `_in` ([`current_handoff_in`], [`pending_handoff_in`],
//! [`restore_handoff_text_in`], [`handoff_being_restored_in`]) and are the
//! ones the product calls; the older names without it return the scope's
//! newest handoff whatever checkout it records, as they always did. A
//! handoff that records no checkout (an older Mimir wrote it, or a session
//! outside any project) is any checkout's to take.

use rusqlite::Connection;

use crate::config::{ContextGuardMode, HooksConfig};
use crate::error::Result;
use crate::memory;
use crate::model::{now_unix, Node, Scope};

/// The tag that makes a memory a session handoff.
pub const HANDOFF_TAG: &str = "session-handoff";

/// What a `SessionStart` hook event says started the session, parsed once
/// so the context guard, the session brief and the rules hook cannot each
/// read the `source` string their own way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionSource {
    /// A new session with an empty context.
    Startup,
    /// An earlier session continued: its transcript, and whatever was
    /// injected into it, is still in context.
    Resume,
    /// The user ran `/clear`.
    Clear,
    /// The context was compacted, by hand or automatically.
    Compact,
    /// A value this build does not know (a newer client's `fork`, say).
    Other,
}

impl SessionSource {
    /// Read the event's `source` field.
    ///
    /// A **missing** field is [`SessionSource::Startup`]: a client too old
    /// to say what kind of start this is can only be announcing a start.
    /// A field that is present but unrecognized is [`SessionSource::Other`],
    /// which every surface treats as "leave everything alone" — a value
    /// nobody here understands is not evidence the context is empty.
    pub fn parse(source: Option<&str>) -> SessionSource {
        match source {
            None | Some("startup") => SessionSource::Startup,
            Some("resume") => SessionSource::Resume,
            Some("clear") => SessionSource::Clear,
            Some("compact") => SessionSource::Compact,
            Some(_) => SessionSource::Other,
        }
    }

    /// The name hook events use for this source (`"other"` for the rest).
    pub fn as_str(self) -> &'static str {
        match self {
            SessionSource::Startup => "startup",
            SessionSource::Resume => "resume",
            SessionSource::Clear => "clear",
            SessionSource::Compact => "compact",
            SessionSource::Other => "other",
        }
    }

    /// The context was just wiped mid-session: `/clear` or a compaction.
    /// What a handoff restore and a brief recap fire on.
    pub fn is_context_reset(self) -> bool {
        matches!(self, SessionSource::Clear | SessionSource::Compact)
    }

    /// The session begins with nothing in context: a start-up or a reset.
    /// The two cases where injected context is new to the model; a resume
    /// already carries it, and an unknown source is not assumed to lack it.
    pub fn starts_with_empty_context(self) -> bool {
        self == SessionSource::Startup || self.is_context_reset()
    }
}

impl std::fmt::Display for SessionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Estimate the percent of the context window the transcript has used since
/// `marker_bytes` (the size recorded at the last `/clear`/`/compact`, or 0
/// for a fresh session — see [`on_session_start`]). Pure and clamped to
/// 0-100; never panics on a marker ahead of `transcript_bytes_now` (clock
/// skew, a marker written after this read raced it) — that just clamps to 0.
pub fn estimate_pct(transcript_bytes_now: u64, marker_bytes: u64, cfg: &HooksConfig) -> u8 {
    let new_bytes = transcript_bytes_now.saturating_sub(marker_bytes) as f64;
    let bytes_per_token = f64::from(cfg.transcript_bytes_per_token).max(0.01);
    let window_tokens = f64::from(cfg.context_window_tokens.max(1));
    let pct = (new_bytes / bytes_per_token / window_tokens) * 100.0;
    pct.clamp(0.0, 100.0).round() as u8
}

/// The guard's message for a firing at `pct` in `mode`, or `None` below
/// `threshold` and in [`ContextGuardMode::Off`].
///
/// The pause nudge asks for a `session-handoff` memory but pause mode never
/// restores one, so it also says how to get the memory back and which mode
/// would have done that unprompted.
pub fn guard_message_for(pct: u8, threshold: u8, mode: ContextGuardMode) -> Option<String> {
    if pct < threshold {
        return None;
    }
    match mode {
        ContextGuardMode::Off => None,
        ContextGuardMode::Pause => Some(format!(
            "[mimir context guard] This session's context is ~{pct}% full. Consider a \
             deliberate `/clear` or `/compact` now, on your own terms, rather than letting \
             it happen mid-task. To keep continuity across the clear, save a handoff first: \
             `mimir remember \"<what's in flight, next step>\" -t note --tags {HANDOFF_TAG}`. \
             Pause mode does not bring it back by itself: after the clear, run `mimir list \
             --tag {HANDOFF_TAG} -n 1` (`[hooks] context_guard = \"handoff\"` restores it \
             automatically)."
        )),
        ContextGuardMode::Handoff => Some(format!(
            "[mimir context guard] This session's context is ~{pct}% full. Before anything \
             else, write ONE structured handoff memory summarizing exactly what's in flight \
             and the next concrete step:\n  mimir remember \"<handoff>\" -t note --tags \
             {HANDOFF_TAG}\nThen tell the user to run `/clear` — the next session \
             auto-restores this handoff on SessionStart."
        )),
    }
}

/// [`guard_message_for`], taking the mode as config.toml spells it. A
/// spelling that names no mode is off, like everywhere else the setting is
/// read (see [`ContextGuardMode`]).
pub fn guard_message(pct: u8, threshold: u8, mode: &str) -> Option<String> {
    guard_message_for(pct, threshold, ContextGuardMode::resolve(mode))
}

/// Whether a nudge is due at `pct`: the first time the session crosses
/// `threshold`, then again only every +10 points after the last nudge that
/// was printed — never every prompt. Reads the `guard_last_pct` session
/// state and changes nothing; the caller prints, then calls [`record_nag`].
pub fn nag_due(conn: &Connection, session_id: &str, pct: u8, threshold: u8) -> Result<bool> {
    if pct < threshold {
        return Ok(false);
    }
    let last: Option<u8> =
        get_session_state(conn, session_id, "guard_last_pct")?.and_then(|s| s.parse().ok());
    Ok(match last {
        None => true,
        Some(last) => pct >= last.saturating_add(10),
    })
}

/// Record that a nudge was printed at `pct`, which is what [`nag_due`]
/// measures the next +10 points from. Call it only after the print: a nag
/// recorded for a message nobody saw would silence the real one.
pub fn record_nag(conn: &Connection, session_id: &str, pct: u8) -> Result<()> {
    set_session_state(conn, session_id, "guard_last_pct", &pct.to_string())
}

/// [`nag_due`] and [`record_nag`] in one step, for a caller that emits
/// unconditionally once this returns `true`. Fails open: a read/write error
/// is the caller's `Result`, and callers on the hook path should treat any
/// `Err` as "don't emit" rather than surfacing it to the user.
pub fn should_emit(conn: &Connection, session_id: &str, pct: u8, threshold: u8) -> Result<bool> {
    let emit = nag_due(conn, session_id, pct, threshold)?;
    if emit {
        record_nag(conn, session_id, pct)?;
    }
    Ok(emit)
}

/// Apply a `SessionStart` event to the session's marker and nag state:
/// - [`SessionSource::Startup`]: clear both, so estimation starts from an
///   empty transcript.
/// - [`SessionSource::Clear`] | [`SessionSource::Compact`]: record the
///   transcript's current byte size as the new marker, so later estimates
///   only count content written *after* this point — and clear the nag
///   state, since the percent is about to drop back near zero.
/// - [`SessionSource::Resume`] | [`SessionSource::Other`]: leave both
///   alone — the transcript continues from where it was, so the existing
///   marker and nag state are still correct.
pub fn on_session_start(
    conn: &Connection,
    session_id: &str,
    source: SessionSource,
    transcript_bytes: u64,
) -> Result<()> {
    match source {
        SessionSource::Startup => {
            clear_session_state(conn, session_id, "transcript_marker")?;
            clear_session_state(conn, session_id, "guard_last_pct")?;
        }
        SessionSource::Clear | SessionSource::Compact => {
            set_session_state(
                conn,
                session_id,
                "transcript_marker",
                &transcript_bytes.to_string(),
            )?;
            clear_session_state(conn, session_id, "guard_last_pct")?;
        }
        SessionSource::Resume | SessionSource::Other => {}
    }
    Ok(())
}

/// [`on_session_start`], taking the event's `source` string as it arrived.
pub fn handle_session_start(
    conn: &Connection,
    session_id: &str,
    source: &str,
    transcript_bytes: u64,
) -> Result<()> {
    on_session_start(
        conn,
        session_id,
        SessionSource::parse(Some(source)),
        transcript_bytes,
    )
}

/// Current transcript marker for a session (0 if absent — fail-open: a
/// missing marker means "estimate from the whole transcript").
pub fn transcript_marker(conn: &Connection, session_id: &str) -> Result<u64> {
    Ok(get_session_state(conn, session_id, "transcript_marker")?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0))
}

/// `node.meta` key holding when (unix seconds) a handoff was restored into
/// a session. In `meta` so it replicates like `expires_at` does: a handoff
/// restored on one machine is not restored again on another.
const RESTORED_AT: &str = "restored_at";

/// How long after its restore a handoff still counts as the one the
/// current event is restoring — see [`handoff_being_restored`].
const SAME_EVENT_SECS: i64 = 60;

/// When `node` was restored into a session, if it was.
pub fn restored_at(node: &Node) -> Option<i64> {
    node.meta.get(RESTORED_AT).and_then(|v| v.as_i64())
}

/// `node.meta` key holding the checkout a handoff was written in: the
/// canonical root of that checkout (`scope::canonical_root`, which is also
/// its project row's `path`). A path, so it means something on the machine
/// that wrote it and nowhere else; a handoff that syncs to another machine
/// waits there for a session in a checkout at the same path.
const HANDOFF_CHECKOUT: &str = "handoff_checkout";

/// How many of a scope's newest handoffs [`newest_handoff_for`] looks through
/// for one this checkout may take. The others in between were written by
/// sessions in other checkouts of the same repository since this one last
/// saved its own; a session clears minutes after saving, so this is far
/// more than ever stand in the way.
const HANDOFF_SCAN: usize = 256;

/// The checkout `node` was written in as a handoff, if it recorded one. A
/// handoff written before checkouts were recorded, or outside any project,
/// has none.
pub fn handoff_checkout(node: &Node) -> Option<&str> {
    node.meta.get(HANDOFF_CHECKOUT).and_then(|v| v.as_str())
}

/// Record the checkout a handoff was written in. Capture calls this for
/// every memory it stores with [`HANDOFF_TAG`]; in `meta`, and with
/// `updated_at` bumped, like every field that syncs.
pub fn set_handoff_checkout(conn: &Connection, node_id: i64, checkout: &str) -> Result<()> {
    crate::text_clock::keep(conn, node_id)?;
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.handoff_checkout', ?2), updated_at = ?3
         WHERE id = ?1",
        rusqlite::params![node_id, checkout, now_unix()],
    )?;
    Ok(())
}

/// Whether a session working in `checkout` (the canonical root of its
/// checkout; `None` outside any project) may take the handoff `tagged`:
/// when the handoff was written in that same checkout, or records none.
///
/// "The same checkout" is the same path *and* the same linked worktree: a
/// worktree removed and added again at that path (for another task, or
/// another repository's branch) is a new checkout, and a handoff written
/// before it was made is not its own. That one question reads the
/// worktree's `commondir` (`scope::worktree_made_after`), and only for a
/// handoff whose recorded path is this checkout's.
///
/// The one rule for every direction a handoff moves in: restoring it into a
/// session, re-arming a consumed one when it is saved again, and showing it
/// on a surface scoped to a session ([`shown_in_checkout`]).
pub fn handoff_is_for(tagged: &Node, checkout: Option<&str>) -> bool {
    match handoff_checkout(tagged) {
        None => true,
        Some(written_in) => {
            Some(written_in) == checkout
                && !crate::scope::worktree_made_after(
                    std::path::Path::new(written_in),
                    tagged.created_at,
                )
        }
    }
}

/// Whether a surface scoped to a session in `checkout` may show `node`:
/// any memory but a handoff [`handoff_is_for`] another checkout. A handoff
/// is one session's state, and `recall`, `list` or a per-prompt injection
/// that hands it to a session in a sibling worktree tells that session to
/// resume someone else's task. A surface asked for everything (`--all`,
/// `all_projects`) or for one memory by id does not ask.
pub fn shown_in_checkout(node: &Node, checkout: Option<&str>) -> bool {
    !node.tags().contains(&HANDOFF_TAG) || handoff_is_for(node, checkout)
}

/// [`memory::list`] as a session in `checkout` may see it
/// ([`shown_in_checkout`]): up to `limit` memories, newest first, after the
/// handoffs of other checkouts are left out, so `-n 1` is still the newest
/// one that is this session's.
pub fn list_in_checkout(
    conn: &Connection,
    scope: Scope,
    mtype: Option<crate::model::MemoryType>,
    tag: Option<&str>,
    limit: usize,
    checkout: Option<&str>,
) -> Result<Vec<Node>> {
    let mut fetch = limit;
    loop {
        let rows = memory::list(conn, scope, mtype, tag, fetch)?;
        let exhausted = rows.len() < fetch;
        let shown: Vec<Node> = rows
            .into_iter()
            .filter(|node| shown_in_checkout(node, checkout))
            .take(limit)
            .collect();
        if shown.len() == limit || exhausted {
            return Ok(shown);
        }
        fetch = fetch.saturating_mul(2).max(fetch + HANDOFF_SCAN);
    }
}

/// Which handoffs a lookup may return: those [`handoff_is_for`] one
/// checkout, or the scope's newest whatever checkout it records (what the
/// functions without `_in` have always returned).
#[derive(Debug, Clone, Copy)]
enum Taker<'a> {
    AnyCheckout,
    Checkout(Option<&'a str>),
}

/// The newest handoff in `scope` that `taker` may take, and when it was
/// consumed, if it was.
///
/// It is the newest [`HANDOFF_TAG`]-tagged memory that [`handoff_is_for`]
/// the taker's checkout (any, for [`Taker::AnyCheckout`]), resolved through
/// [`crate::lifecycle::resolve_current`]: itself when live, its replacement
/// when it was superseded, nothing when it expired. Never an *older*
/// handoff of the same checkout: a handoff is a snapshot for the next
/// session, so when the newest one was retired or already restored, the
/// ones before it are stale by construction. The restore mark counts on
/// either node: on the tagged memory (restored, then merged into another)
/// or on its replacement (merged, then restored).
fn newest_handoff_for(
    conn: &Connection,
    scope: Scope,
    taker: Taker<'_>,
    now: i64,
) -> Result<Option<(Node, Option<i64>)>> {
    let (scan, checkout) = match taker {
        Taker::AnyCheckout => (1, None),
        Taker::Checkout(checkout) => (HANDOFF_SCAN, Some(checkout)),
    };
    let Some(tagged) = memory::list(conn, scope, None, Some(HANDOFF_TAG), scan)?
        .into_iter()
        .find(|tagged| checkout.is_none_or(|checkout| handoff_is_for(tagged, checkout)))
    else {
        return Ok(None);
    };
    let on_tagged = restored_at(&tagged);
    Ok(crate::lifecycle::resolve_current(conn, tagged, now)
        .filter(|n| scope.admits(n.project_id))
        .map(|current| {
            let consumed = on_tagged.max(restored_at(&current));
            (current, consumed)
        }))
}

/// [`current_handoff_in`], for either [`Taker`].
fn current_handoff_for(conn: &Connection, scope: Scope, taker: Taker<'_>) -> Result<Option<Node>> {
    Ok(newest_handoff_for(conn, scope, taker, now_unix())?
        .filter(|(_, consumed)| consumed.is_none())
        .map(|(node, _)| node))
}

/// The handoff a session start in `checkout` would restore, or `None`: the
/// newest one in `scope` for that checkout (see [`newest_handoff_for`])
/// unless it was already restored.
///
/// One function on purpose. The restore and the brief's "don't brief what
/// the guard is about to restore" exclusion both need *this* node, and they
/// used to look it up separately — and `memory::list` returns everything
/// stored, so both took a handoff its author had already retired.
///
/// A handoff belongs to a scope and a checkout, not to a session: nothing
/// on the memory says which session wrote it, and a cleared session comes
/// back under a new id. So with two sessions in one checkout, the first to
/// clear takes the newest handoff, and once that one is consumed an older,
/// still unrestored one is *not* handed to the other session. Falling back
/// to it would also hand a single session the snapshot it wrote two resets
/// ago, and a wrong "this is where you left off" is worse than none: the
/// handoff a session did not get is still there (`mimir list --tag
/// session-handoff`). Sessions in *different* checkouts of one repository
/// meet here only over a handoff that records no checkout (written by an
/// older Mimir, or outside any project): that one goes to whichever of them
/// clears first, once, as every handoff did before checkouts were
/// recorded. Every other handoff is handed to its own checkout alone.
pub fn current_handoff_in(
    conn: &Connection,
    scope: Scope,
    checkout: Option<&str>,
) -> Result<Option<Node>> {
    current_handoff_for(conn, scope, Taker::Checkout(checkout))
}

/// The scope's pending handoff whatever checkout it was written in: the
/// newest handoff in `scope`, unless it was already restored. What this
/// function returned before handoffs recorded their checkout, and still
/// does. Every surface in the product restores into a session in a
/// checkout and calls [`current_handoff_in`] instead; a session outside any
/// project is `current_handoff_in(…, None)`.
pub fn current_handoff(conn: &Connection, scope: Scope) -> Result<Option<Node>> {
    current_handoff_for(conn, scope, Taker::AnyCheckout)
}

/// [`handoff_being_restored_in`], for either [`Taker`].
fn handoff_being_restored_for(
    conn: &Connection,
    scope: Scope,
    taker: Taker<'_>,
    now: i64,
) -> Result<Option<Node>> {
    Ok(newest_handoff_for(conn, scope, taker, now)?
        .filter(|(_, consumed)| consumed.is_none_or(|at| now - at <= SAME_EVENT_SECS))
        .map(|(node, _)| node))
}

/// The handoff the event being handled at `now` restores into a session in
/// `checkout`: [`current_handoff_in`], or the one whose restore was
/// recorded in the last minute.
///
/// For the session brief's exclusion. The guard's hook and the brief's hook
/// run side by side on the same `SessionStart` event, in no fixed order, so
/// by the time the brief asks, the guard may already have consumed the
/// handoff — which is exactly the case where the brief must not repeat it.
pub fn handoff_being_restored_in(
    conn: &Connection,
    scope: Scope,
    checkout: Option<&str>,
    now: i64,
) -> Result<Option<Node>> {
    handoff_being_restored_for(conn, scope, Taker::Checkout(checkout), now)
}

/// [`handoff_being_restored_in`] whatever checkout the handoff records, as
/// before checkouts were recorded; see [`current_handoff`]. In-product
/// callers use [`handoff_being_restored_in`].
pub fn handoff_being_restored(conn: &Connection, scope: Scope, now: i64) -> Result<Option<Node>> {
    handoff_being_restored_for(conn, scope, Taker::AnyCheckout, now)
}

/// Restore the handoff pending for a session in `checkout`: its body,
/// prefixed for direct SessionStart stdout injection, or `None` if there is
/// none to restore.
///
/// Producing the text consumes the handoff. The memory is marked with
/// `meta.restored_at` and its `updated_at` is bumped so the mark replicates;
/// from then on [`current_handoff_in`] is `None` until a newer handoff is
/// written, and a second clear does not replay a snapshot of a session two
/// resets back. The memory itself stays, readable with `mimir get`.
///
/// A mark that cannot be written is logged and the text still returned: a
/// handoff shown twice costs a few lines, one withheld costs the session
/// its continuity.
pub fn restore_handoff_text_in(
    conn: &Connection,
    scope: Scope,
    checkout: Option<&str>,
) -> Result<Option<String>> {
    restore_handoff_text_for(conn, scope, Taker::Checkout(checkout))
}

/// [`restore_handoff_text_in`], for either [`Taker`].
fn restore_handoff_text_for(
    conn: &Connection,
    scope: Scope,
    taker: Taker<'_>,
) -> Result<Option<String>> {
    let Some((node, text)) = pending_handoff_for(conn, scope, taker)? else {
        return Ok(None);
    };
    mark_handoff_restored(conn, node.id);
    Ok(Some(text))
}

/// [`restore_handoff_text_in`] whatever checkout the handoff records, as
/// before checkouts were recorded; see [`current_handoff`]. In-product
/// callers use [`restore_handoff_text_in`].
pub fn restore_handoff_text(conn: &Connection, scope: Scope) -> Result<Option<String>> {
    restore_handoff_text_for(conn, scope, Taker::AnyCheckout)
}

/// The handoff a restore into a session in `checkout` would deliver, and
/// the text it would deliver, with nothing consumed yet. The two halves of
/// [`restore_handoff_text_in`] are separate so a caller can hand the text
/// over first and consume the handoff only once that has happened: a hook
/// whose reader is gone dies writing to it, and a handoff marked before the
/// write would be spent on a session that never saw it.
pub fn pending_handoff_in(
    conn: &Connection,
    scope: Scope,
    checkout: Option<&str>,
) -> Result<Option<(Node, String)>> {
    pending_handoff_for(conn, scope, Taker::Checkout(checkout))
}

/// [`pending_handoff_in`], for either [`Taker`].
fn pending_handoff_for(
    conn: &Connection,
    scope: Scope,
    taker: Taker<'_>,
) -> Result<Option<(Node, String)>> {
    let Some(node) = current_handoff_for(conn, scope, taker)? else {
        return Ok(None);
    };
    let Some(body) = node.body.as_deref() else {
        return Ok(None);
    };
    let text = format!("Restored session handoff:\n{body}");
    Ok(Some((node, text)))
}

/// [`pending_handoff_in`] whatever checkout the handoff records, as before
/// checkouts were recorded; see [`current_handoff`]. In-product callers use
/// [`pending_handoff_in`].
pub fn pending_handoff(conn: &Connection, scope: Scope) -> Result<Option<(Node, String)>> {
    pending_handoff_for(conn, scope, Taker::AnyCheckout)
}

/// Consume a handoff: mark it restored now, and bump `updated_at` so the
/// mark replicates. A mark that cannot be written is logged and nothing
/// else: a handoff shown twice costs a few lines, an error here would cost
/// the session start.
pub fn mark_handoff_restored(conn: &Connection, node_id: i64) {
    let now = now_unix();
    // The mark annotates the handoff; its text is as old as it was. Best
    // effort like the mark itself.
    let _ = crate::text_clock::keep(conn, node_id);
    if let Err(err) = conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.restored_at', ?2), updated_at = ?2
         WHERE id = ?1",
        rusqlite::params![node_id, now],
    ) {
        tracing::warn!(%err, node = node_id, "could not mark the handoff as restored");
    }
}

/// Make a consumed handoff restorable again. For the one case where the
/// same handoff is saved a second time: the store refuses the text as a
/// duplicate of the memory it already holds, so that memory is what the
/// next clear has to deliver.
pub fn rearm_handoff(conn: &Connection, node_id: i64) -> Result<()> {
    crate::text_clock::keep(conn, node_id)?;
    conn.execute(
        "UPDATE node SET meta = json_remove(meta, '$.restored_at'), updated_at = ?2
         WHERE id = ?1 AND json_extract(meta, '$.restored_at') IS NOT NULL",
        rusqlite::params![node_id, now_unix()],
    )?;
    Ok(())
}

use crate::store::{clear_session_state, get_session_state, set_session_state};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use rusqlite::params;

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
                actor: crate::model::Actor::System,
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
                actor: crate::model::Actor::System,
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

    fn handoff(conn: &Connection, text: &str) -> Node {
        match memory::remember(
            conn,
            memory::Remember {
                text: text.into(),
                mtype: crate::model::MemoryType::Note,
                tags: vec![HANDOFF_TAG.into()],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            memory::RememberOutcome::Created(node) => node,
            other => panic!("expected Created, got {other:?}"),
        }
    }

    #[test]
    fn session_source_is_parsed_in_one_place() {
        for (raw, source) in [
            (Some("startup"), SessionSource::Startup),
            (Some("resume"), SessionSource::Resume),
            (Some("clear"), SessionSource::Clear),
            (Some("compact"), SessionSource::Compact),
            (Some("fork"), SessionSource::Other),
            (Some(""), SessionSource::Other),
            // A client that does not say is announcing a start.
            (None, SessionSource::Startup),
        ] {
            assert_eq!(SessionSource::parse(raw), source, "{raw:?}");
        }
        assert!(SessionSource::Clear.is_context_reset());
        assert!(SessionSource::Compact.is_context_reset());
        assert!(!SessionSource::Startup.is_context_reset());
        for source in [
            SessionSource::Startup,
            SessionSource::Clear,
            SessionSource::Compact,
        ] {
            assert!(source.starts_with_empty_context(), "{source}");
        }
        for source in [SessionSource::Resume, SessionSource::Other] {
            assert!(!source.starts_with_empty_context(), "{source}");
        }
    }

    /// An unknown `source` value moves no state — the same arm a resume
    /// takes — while a missing one is a start-up.
    #[test]
    fn session_start_with_an_unknown_source_leaves_state_alone() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        set_session_state(&conn, sid, "transcript_marker", "4242").unwrap();
        set_session_state(&conn, sid, "guard_last_pct", "55").unwrap();
        on_session_start(&conn, sid, SessionSource::parse(Some("fork")), 999).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 4242);
        on_session_start(&conn, sid, SessionSource::parse(None), 999).unwrap();
        assert_eq!(transcript_marker(&conn, sid).unwrap(), 0);
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            None
        );
    }

    /// A mode config.toml misspells is off for the message too, not just
    /// for the gates in front of it.
    #[test]
    fn a_misspelled_mode_says_nothing() {
        assert!(guard_message(90, 45, "handof").is_none());
        assert!(guard_message(90, 45, "Pause").is_none());
        assert!(guard_message_for(90, 45, ContextGuardMode::Off).is_none());
    }

    /// Pause mode asks for a handoff memory it will not restore, so the
    /// nudge has to say how to get it back.
    #[test]
    fn the_pause_nudge_says_how_to_get_the_handoff_back() {
        let pause = guard_message_for(50, 45, ContextGuardMode::Pause).unwrap();
        assert!(pause.contains("--tags session-handoff"), "{pause}");
        assert!(
            pause.contains("`mimir list --tag session-handoff -n 1`"),
            "{pause}"
        );
        assert!(
            pause.contains("context_guard = \"handoff\"") && pause.contains("automatically"),
            "{pause}"
        );
        // Handoff mode restores by itself and must not send anyone looking.
        let handoff = guard_message_for(50, 45, ContextGuardMode::Handoff).unwrap();
        assert!(!handoff.contains("mimir recall"), "{handoff}");
    }

    /// Deciding a nudge is due writes nothing: the nag state moves only
    /// once the caller has printed and says so.
    #[test]
    fn a_nag_is_recorded_only_when_the_caller_says_it_was_emitted() {
        let conn = db::open_in_memory().unwrap();
        let sid = "s1";
        assert!(nag_due(&conn, sid, 50, 45).unwrap());
        assert_eq!(
            get_session_state(&conn, sid, "guard_last_pct").unwrap(),
            None,
            "asking must not record"
        );
        assert!(nag_due(&conn, sid, 50, 45).unwrap(), "still due");
        record_nag(&conn, sid, 50).unwrap();
        assert!(!nag_due(&conn, sid, 55, 45).unwrap());
        assert!(nag_due(&conn, sid, 60, 45).unwrap());
        assert!(!nag_due(&conn, sid, 30, 45).unwrap(), "below threshold");
    }

    #[test]
    fn a_handoff_is_restored_once() {
        let conn = db::open_in_memory().unwrap();
        let written = handoff(&conn, "handoff: finish the importer, then run the eval");
        let before = crate::store::get_node(&conn, written.id).unwrap();
        assert_eq!(restored_at(&before), None);
        conn.execute(
            "UPDATE node SET updated_at = updated_at - 500 WHERE id = ?1",
            params![written.id],
        )
        .unwrap();

        let text = restore_handoff_text(&conn, Scope::Global).unwrap().unwrap();
        assert!(text.contains("finish the importer"), "{text}");
        assert_eq!(
            restore_handoff_text(&conn, Scope::Global).unwrap(),
            None,
            "the second clear must not replay a handoff already restored"
        );
        assert_eq!(
            current_handoff(&conn, Scope::Global).unwrap().map(|n| n.id),
            None
        );

        // The mark is in meta and the row was touched, so it replicates.
        let after = crate::store::get_node(&conn, written.id).unwrap();
        let at = restored_at(&after).expect("meta.restored_at");
        assert!((now_unix() - at).abs() <= 5, "{at}");
        assert!(after.updated_at >= at, "updated_at must carry the mark");
        assert_eq!(after.body, before.body, "the memory itself is untouched");
        assert!(after.deleted_at.is_none());
    }

    /// The next handoff is a new memory and is restored in its turn; the
    /// consumed one never comes back in its place.
    #[test]
    fn a_newer_handoff_is_restored_after_an_older_one_was_consumed() {
        let conn = db::open_in_memory().unwrap();
        let first = handoff(&conn, "handoff: wire the parser into the CLI");
        conn.execute(
            "UPDATE node SET created_at = created_at - 100 WHERE id = ?1",
            params![first.id],
        )
        .unwrap();
        assert!(restore_handoff_text(&conn, Scope::Global)
            .unwrap()
            .is_some());

        let second = handoff(&conn, "handoff: the parser is in; write its fuzz test");
        assert_eq!(
            current_handoff(&conn, Scope::Global).unwrap().map(|n| n.id),
            Some(second.id)
        );
        let text = restore_handoff_text(&conn, Scope::Global).unwrap().unwrap();
        assert!(text.contains("write its fuzz test"), "{text}");
        assert_eq!(
            restore_handoff_text(&conn, Scope::Global).unwrap(),
            None,
            "never falls back to the first, which was consumed long ago"
        );
    }

    /// The brief runs beside the guard on the same event. Whichever goes
    /// first, the brief must see the handoff that event restores.
    #[test]
    fn the_handoff_of_this_event_is_visible_to_the_brief_either_side_of_the_restore() {
        let conn = db::open_in_memory().unwrap();
        let written = handoff(&conn, "handoff: rebase onto main before the release");
        let now = now_unix();
        let seen = |at: i64| {
            handoff_being_restored(&conn, Scope::Global, at)
                .unwrap()
                .map(|n| n.id)
        };
        assert_eq!(seen(now), Some(written.id), "brief first");
        restore_handoff_text(&conn, Scope::Global).unwrap().unwrap();
        assert_eq!(seen(now_unix()), Some(written.id), "guard first");
        assert_eq!(
            seen(now_unix() + SAME_EVENT_SECS + 5),
            None,
            "a later event"
        );
    }

    /// Consolidation can merge a restored handoff into a near-duplicate.
    /// The replacement inherits the role (see `lifecycle::resolve_current`)
    /// and with it the fact that the role was already played.
    #[test]
    fn a_restored_handoff_stays_consumed_when_it_is_later_superseded() {
        let conn = db::open_in_memory().unwrap();
        let written = handoff(&conn, "handoff: profile the slow recall path");
        restore_handoff_text(&conn, Scope::Global).unwrap().unwrap();
        let merged = match memory::remember(
            &conn,
            memory::Remember {
                text: "handoff: profile the slow recall path first".into(),
                mtype: crate::model::MemoryType::Note,
                tags: Vec::new(),
                project_id: None,
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            memory::RememberOutcome::Created(node) => node,
            other => panic!("expected Created, got {other:?}"),
        };
        crate::store::set_superseded(
            &conn,
            written.id,
            merged.id,
            crate::model::Actor::System,
            None,
        )
        .unwrap();
        assert_eq!(restore_handoff_text(&conn, Scope::Global).unwrap(), None);
    }

    /// A handoff in the shared memory project, written in `checkout` (or
    /// before checkouts were recorded: `None`), `age` seconds ago.
    fn handoff_from(
        conn: &Connection,
        project: i64,
        checkout: Option<&str>,
        text: &str,
        age: i64,
    ) -> Node {
        let node = handoff(conn, text);
        conn.execute(
            "UPDATE node SET project_id = ?2, created_at = created_at - ?3 WHERE id = ?1",
            params![node.id, project, age],
        )
        .unwrap();
        if let Some(checkout) = checkout {
            set_handoff_checkout(conn, node.id, checkout).unwrap();
        }
        crate::store::get_node(conn, node.id).unwrap()
    }

    /// The shared memory project of a repository with a main checkout and
    /// two linked worktrees, and the scope all three read with.
    fn shared_project(conn: &Connection) -> Scope {
        Scope::Project(crate::store::ensure_project(conn, MAIN, "main").unwrap().id)
    }
    const MAIN: &str = "/repo/main";
    const WT_A: &str = "/repo/wt-a";
    const WT_B: &str = "/repo/wt-b";

    /// Worktrees of one repository share a memory project, and sessions run
    /// in several at once. Each is handed the handoff written in its own
    /// checkout: not a sibling's, although that one is newer and in scope,
    /// and the main checkout is handed neither.
    #[test]
    fn a_handoff_is_restored_only_in_the_checkout_it_was_written_in() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        let from_a = handoff_from(&conn, project, Some(WT_A), "handoff: A is mid-rebase", 200);
        let from_b = handoff_from(
            &conn,
            project,
            Some(WT_B),
            "handoff: B is writing tests",
            100,
        );
        assert_eq!(handoff_checkout(&from_a), Some(WT_A));
        let pending = |checkout: Option<&str>| {
            current_handoff_in(&conn, scope, checkout)
                .unwrap()
                .map(|n| n.id)
        };
        assert_eq!(pending(Some(WT_A)), Some(from_a.id));
        assert_eq!(pending(Some(WT_B)), Some(from_b.id));
        assert_eq!(pending(Some(MAIN)), None, "the main checkout wrote neither");
        assert_eq!(pending(None), None, "nor did a session in no checkout");
        assert_eq!(
            restore_handoff_text_in(&conn, scope, Some(MAIN)).unwrap(),
            None
        );

        // B clears: it gets its own, and consumes only its own.
        let text = restore_handoff_text_in(&conn, scope, Some(WT_B))
            .unwrap()
            .unwrap();
        assert!(text.contains("B is writing tests"), "{text}");
        assert_eq!(pending(Some(WT_B)), None);
        assert_eq!(pending(Some(WT_A)), Some(from_a.id), "A's is still waiting");

        // The brief's exclusion follows the same checkout.
        let now = now_unix();
        let excluded = |checkout: Option<&str>| {
            handoff_being_restored_in(&conn, scope, checkout, now)
                .unwrap()
                .map(|n| n.id)
        };
        assert_eq!(excluded(Some(WT_B)), Some(from_b.id), "just restored there");
        assert_eq!(excluded(Some(WT_A)), Some(from_a.id), "about to be, there");
        assert_eq!(excluded(Some(MAIN)), None);

        let (_, text) = pending_handoff_in(&conn, scope, Some(WT_A))
            .unwrap()
            .unwrap();
        assert!(text.contains("A is mid-rebase"), "{text}");
    }

    /// The four lookups that existed before handoffs recorded a checkout
    /// keep their meaning: the scope's newest handoff, whatever checkout it
    /// names. Only the `_in` forms the product calls tell checkouts apart.
    #[test]
    fn the_lookups_without_a_checkout_return_the_scopes_newest_handoff_as_before() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        handoff_from(&conn, project, Some(WT_A), "handoff: A is mid-rebase", 200);
        let from_b = handoff_from(
            &conn,
            project,
            Some(WT_B),
            "handoff: B is writing tests",
            100,
        );
        assert_eq!(
            current_handoff(&conn, scope).unwrap().map(|n| n.id),
            Some(from_b.id),
            "the newest, written in another checkout than any caller's"
        );
        assert_eq!(
            pending_handoff(&conn, scope).unwrap().map(|(n, _)| n.id),
            Some(from_b.id)
        );
        assert_eq!(
            handoff_being_restored(&conn, scope, now_unix())
                .unwrap()
                .map(|n| n.id),
            Some(from_b.id)
        );
        assert!(current_handoff_in(&conn, scope, None).unwrap().is_none());
        assert!(current_handoff_in(&conn, scope, Some(MAIN))
            .unwrap()
            .is_none());

        let text = restore_handoff_text(&conn, scope).unwrap().unwrap();
        assert!(text.contains("B is writing tests"), "{text}");
        // Consumed, and "never an older handoff" holds across checkouts for
        // these, as it did when there was one per scope.
        assert!(current_handoff(&conn, scope).unwrap().is_none());
        assert!(restore_handoff_text(&conn, scope).unwrap().is_none());
        assert!(current_handoff_in(&conn, scope, Some(WT_A))
            .unwrap()
            .is_some());
    }

    /// A handoff written before checkouts were recorded names none, and is
    /// restored as it always was: into whichever session in its scope
    /// clears first, once.
    #[test]
    fn a_handoff_that_records_no_checkout_is_restored_as_before() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        let legacy = handoff_from(&conn, project, None, "handoff: from an older Mimir", 100);
        assert_eq!(handoff_checkout(&legacy), None);
        assert!(handoff_is_for(&legacy, Some(WT_A)) && handoff_is_for(&legacy, None));
        for checkout in [Some(WT_A), Some(WT_B), Some(MAIN), None] {
            assert_eq!(
                current_handoff_in(&conn, scope, checkout)
                    .unwrap()
                    .map(|n| n.id),
                Some(legacy.id),
                "{checkout:?}"
            );
        }
        assert_eq!(
            current_handoff(&conn, scope).unwrap().map(|n| n.id),
            Some(legacy.id)
        );
        // Restored into one of them, it is spent for all of them.
        assert!(restore_handoff_text_in(&conn, scope, Some(WT_B))
            .unwrap()
            .unwrap()
            .contains("from an older Mimir"));
        for checkout in [Some(WT_A), Some(WT_B), Some(MAIN), None] {
            assert!(current_handoff_in(&conn, scope, checkout)
                .unwrap()
                .is_none());
        }
    }

    /// "Never an older handoff" is counted per checkout: a sibling's newer
    /// handoffs neither hide this checkout's own nor bring back the one it
    /// already consumed.
    #[test]
    fn a_checkouts_older_handoff_is_not_restored_past_its_own_consumed_one() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        handoff_from(
            &conn,
            project,
            Some(WT_A),
            "handoff: A, two resets ago",
            400,
        );
        let newest_a = handoff_from(&conn, project, Some(WT_A), "handoff: A, last reset", 300);
        handoff_from(&conn, project, Some(WT_B), "handoff: B, since then", 200);
        handoff_from(&conn, project, Some(WT_B), "handoff: B, just now", 100);
        let pending_in_a = || {
            current_handoff_in(&conn, scope, Some(WT_A))
                .unwrap()
                .map(|n| n.id)
        };
        assert_eq!(
            pending_in_a(),
            Some(newest_a.id),
            "B's newer ones do not hide it"
        );
        mark_handoff_restored(&conn, newest_a.id);
        assert_eq!(pending_in_a(), None, "and A's older one stays in the past");
    }

    /// A surface scoped to a session shows every memory but the handoffs
    /// written in other checkouts; a handoff that records none is shown
    /// everywhere, as is any memory that is not a handoff. `list` counts
    /// `-n` after leaving those out, so `-n 1` is still this session's
    /// newest.
    #[test]
    fn a_scoped_list_leaves_out_the_handoffs_of_other_checkouts() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        let older_a = handoff_from(&conn, project, Some(WT_A), "handoff: A, earlier", 300);
        let legacy = handoff_from(&conn, project, None, "handoff: from an older Mimir", 400);
        let newest_b = handoff_from(&conn, project, Some(WT_B), "handoff: B, just now", 100);
        let note = match memory::remember(
            &conn,
            memory::Remember {
                text: "a plain note in the repository".into(),
                mtype: crate::model::MemoryType::Note,
                tags: Vec::new(),
                project_id: Some(project),
                force: false,
                actor: crate::model::Actor::Human,
            },
        )
        .unwrap()
        {
            memory::RememberOutcome::Created(node) => node,
            other => panic!("expected Created, got {other:?}"),
        };
        assert!(shown_in_checkout(&note, Some(WT_A)) && shown_in_checkout(&note, None));
        assert!(shown_in_checkout(&legacy, Some(WT_A)) && shown_in_checkout(&legacy, None));
        assert!(shown_in_checkout(&older_a, Some(WT_A)));
        assert!(!shown_in_checkout(&older_a, Some(WT_B)));
        assert!(!shown_in_checkout(&older_a, None));

        let ids = |checkout: Option<&str>, limit: usize| -> Vec<i64> {
            list_in_checkout(&conn, scope, None, Some(HANDOFF_TAG), limit, checkout)
                .unwrap()
                .iter()
                .map(|n| n.id)
                .collect()
        };
        assert_eq!(ids(Some(WT_A), 1), [older_a.id], "past B's newer one");
        assert_eq!(ids(Some(WT_A), 10), [older_a.id, legacy.id]);
        assert_eq!(ids(Some(WT_B), 1), [newest_b.id]);
        assert_eq!(ids(Some(MAIN), 10), [legacy.id]);
        assert_eq!(ids(None, 10), [legacy.id]);
        assert_eq!(ids(Some(WT_A), 0), Vec::<i64>::new());
        // Untagged, the note is in every checkout's list.
        let all: Vec<i64> = list_in_checkout(&conn, scope, None, None, 10, Some(MAIN))
            .unwrap()
            .iter()
            .map(|n| n.id)
            .collect();
        assert!(
            all.contains(&note.id) && all.contains(&legacy.id),
            "{all:?}"
        );
        assert!(!all.contains(&older_a.id) && !all.contains(&newest_b.id));
    }

    /// More handoffs of other checkouts than one fetch holds stand between
    /// a checkout and its own: the list keeps reading until it has `-n` or
    /// the store runs out.
    #[test]
    fn a_scoped_list_reads_past_many_handoffs_of_other_checkouts() {
        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        let own = handoff_from(&conn, project, Some(WT_A), "handoff: A's own", 10_000);
        for i in 0..(HANDOFF_SCAN + 5) {
            handoff_from(
                &conn,
                project,
                Some(WT_B),
                &format!("handoff: B number {i} of many"),
                5_000 - i as i64,
            );
        }
        let found = list_in_checkout(&conn, scope, None, Some(HANDOFF_TAG), 1, Some(WT_A)).unwrap();
        assert_eq!(found.iter().map(|n| n.id).collect::<Vec<_>>(), [own.id]);
    }

    /// A worktree removed and added again at the same path is a new
    /// checkout: the handoff a session there wrote before is not handed to
    /// a session in the new one, and is not shown to it; one written since
    /// is.
    #[test]
    fn a_handoff_from_an_earlier_worktree_at_the_same_path_is_not_this_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("main");
        let wt = tmp.path().join("wt");
        let gitdir = main.join(".git/worktrees/wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        let made = now_unix() - 1_000;
        std::fs::File::options()
            .write(true)
            .open(gitdir.join("commondir"))
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(made as u64))
            .unwrap();
        let checkout = crate::scope::canonical_root(&wt);

        let conn = db::open_in_memory().unwrap();
        let scope = shared_project(&conn);
        let Scope::Project(project) = scope else {
            unreachable!()
        };
        let before = handoff_from(
            &conn,
            project,
            Some(&checkout),
            "handoff: the last task here",
            2_000,
        );
        assert!(!handoff_is_for(&before, Some(&checkout)));
        assert!(!shown_in_checkout(&before, Some(&checkout)));
        assert!(current_handoff_in(&conn, scope, Some(&checkout))
            .unwrap()
            .is_none());
        let since = handoff_from(&conn, project, Some(&checkout), "handoff: this task", 10);
        assert!(handoff_is_for(&since, Some(&checkout)));
        assert_eq!(
            current_handoff_in(&conn, scope, Some(&checkout))
                .unwrap()
                .map(|n| n.id),
            Some(since.id)
        );
    }
}
