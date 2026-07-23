//! `mimir brief` — the session brief: a hard-capped digest of the
//! project's most drift-preventing memories (gotchas + decisions),
//! injected once at session start and again after a context
//! clear/compaction. Selection/rendering live in `mimir_core::brief`;
//! this module owns the hook-event plumbing, suppression bookkeeping and
//! the savings-ledger cost record. Design: docs/design/session-brief.md.
//!
//! Two entry points:
//! - `show()` — the SessionStart hook (stdin JSON: session_id, source,
//!   cwd). Silent unless `[brief] enabled = true` (the kill-switch; the
//!   hook entry itself is always installed, same rollback pattern as the
//!   context guard).
//! - `dry_run()` — human-facing `mimir brief`: prints exactly what a
//!   startup fire would inject, then the ranked candidates with scores,
//!   so "why did this appear?" always has an answer. Records nothing.

use std::collections::HashSet;
use std::io::Read;

use anyhow::Result;
use mimir_core::model::{now_unix, short_uid, Scope};
use mimir_core::savings::{self, source};
use mimir_core::{brief, memory, store, tokens, Mimir};

const FIRE_COUNT_KEY: &str = "brief_fire_count";
const SHOWN_PREFIX: &str = "brief_shown:";
/// Wall-clock fallback: any node briefed in ANY session inside this
/// window is excluded, covering the unverified case where a resume (or a
/// client's compact) mints a fresh session_id — without it, a fresh id
/// would re-show the identical brief minutes later.
const RECENT_SHOWN_SECS: i64 = 6 * 3600;

fn read_stdin_json() -> serde_json::Value {
    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_err() {
        return serde_json::Value::Null;
    }
    serde_json::from_str(&buf).unwrap_or(serde_json::Value::Null)
}

/// Same cwd→scope resolution as the context guard's hook entry.
fn scope_for_cwd(mimir: &Mimir, input: &serde_json::Value) -> Scope {
    input
        .get("cwd")
        .and_then(|v| v.as_str())
        .and_then(|c| {
            mimir
                .project_for_cwd(std::path::Path::new(c))
                .ok()
                .flatten()
        })
        .map(|p| Scope::Project(p.id))
        .unwrap_or(Scope::Global)
}

/// Node ids to suppress for this fire. Semantics differ by fire kind:
///
/// - `recap = false` (startup): this session's own shown set PLUS the
///   recent wall-clock window across ALL sessions — a fresh session inside
///   the window shouldn't repeat what a sibling just showed.
/// - `recap = true` (clear/compact): the context was just WIPED — the
///   top-ranked guards that were shown before the reset are exactly what
///   the agent forgot, so re-anchoring them is the point of the fire, not
///   a duplicate. Only OTHER sessions' recent shows are suppressed (a
///   client that mints a fresh session_id on compact degrades to startup
///   semantics via the wall-clock — safe, just less re-anchoring).
fn suppressed(mimir: &Mimir, session_id: &str, now: i64, recap: bool) -> Result<HashSet<i64>> {
    let sql = if recap {
        "SELECT DISTINCT key FROM session_state
         WHERE key LIKE ?1 AND session_id != ?2 AND updated_at > ?3"
    } else {
        "SELECT DISTINCT key FROM session_state
         WHERE key LIKE ?1 AND (session_id = ?2 OR updated_at > ?3)"
    };
    let mut stmt = mimir.conn.prepare(sql)?;
    let rows = stmt.query_map(
        rusqlite::params![
            format!("{SHOWN_PREFIX}%"),
            session_id,
            now - RECENT_SHOWN_SECS
        ],
        |r| r.get::<_, String>(0),
    )?;
    let mut out = HashSet::new();
    for key in rows {
        if let Ok(id) = key?[SHOWN_PREFIX.len()..].parse() {
            out.insert(id);
        }
    }
    Ok(out)
}

/// Rules-pack body for content dedup — only when a project was detected;
/// in global scope the rules channel never fires, so nothing is excluded
/// there (otherwise such memories would be shown by neither channel).
fn rules_body(mimir: &Mimir, scope: Scope) -> Option<String> {
    match scope {
        Scope::Project(pid) => crate::rules_cmd::find_rules(&mimir.conn, pid)
            .ok()
            .flatten()
            .and_then(|n| n.body),
        _ => None,
    }
}

/// SessionStart hook entry.
pub fn show() -> Result<()> {
    let input = read_stdin_json();
    let mimir = Mimir::open()?;
    if !mimir.config.brief.enabled {
        return Ok(());
    }
    let Some(session_id) = input.get("session_id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let event_source = input.get("source").and_then(|v| v.as_str()).unwrap_or("");
    // resume and fork (v2.1.214+; older clients report forks as "resume"):
    // the transcript already contains prior context including any earlier
    // brief. Unknown values: fail closed (never fire), matching the
    // context guard's inert-default arm. session_id stability across
    // clear/compact is undocumented upstream — the wall-clock fallback in
    // suppressed() covers a minted id, so nothing here depends on it.
    let budget = match event_source {
        "startup" => mimir.config.brief.max_tokens,
        "clear" | "compact" => mimir.config.brief.recap_tokens,
        _ => return Ok(()),
    };
    let recap = matches!(event_source, "clear" | "compact");
    let now = now_unix();
    // Fire-cap check before any selection work — a capped session pays
    // zero query cost. Only RENDERED fires increment it (below): a
    // zero-gotcha project must not burn its budget on empty attempts.
    let fires: usize = store::get_session_state(&mimir.conn, session_id, FIRE_COUNT_KEY)?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if fires >= mimir.config.brief.max_fires_per_session {
        return Ok(());
    }

    let scope = scope_for_cwd(&mimir, &input);
    let mut exclude = suppressed(&mimir, session_id, now, recap)?;
    // Compact-boundary overlap: in handoff mode the context guard is about
    // to restore the handoff memory on this same event — don't brief it too.
    if matches!(event_source, "clear" | "compact") && mimir.config.hooks.context_guard == "handoff"
    {
        if let Some(n) = memory::list(&mimir.conn, scope, None, Some("session-handoff"), 1)?
            .into_iter()
            .next()
        {
            exclude.insert(n.id);
        }
    }

    let rules = rules_body(&mimir, scope);
    let ranked = brief::select(
        &mimir.conn,
        scope,
        &exclude,
        rules.as_deref(),
        &mimir.config,
        now,
    )?;
    let composed = brief::compose(&ranked, &mimir.config.brief, budget, now);
    if composed.shown.is_empty() {
        return Ok(());
    }

    println!("{}", composed.text);

    // Bookkeeping is best-effort: the injection already happened, so a
    // bookkeeping error must not fail the hook (worst case: a repeat
    // brief next fire, bounded by the caps).
    let _ = store::set_session_state(
        &mimir.conn,
        session_id,
        FIRE_COUNT_KEY,
        &(fires + 1).to_string(),
    );
    for id in &composed.shown {
        let _ =
            store::set_session_state(&mimir.conn, session_id, &format!("{SHOWN_PREFIX}{id}"), "1");
    }
    let project_id = match scope {
        Scope::Project(id) => Some(id),
        _ => None,
    };
    // before=0: this is honestly a COST record, not a saving — see
    // savings::source::BRIEF.
    let _ = savings::record(
        &mimir.conn,
        project_id,
        source::BRIEF,
        0,
        tokens::count(&composed.text),
        serde_json::json!({ "items": composed.shown.len(), "source": event_source }),
    );
    Ok(())
}

/// Human-facing `mimir brief`: what a startup fire would inject, plus the
/// full ranked candidate list with scores. No suppression state is read
/// or written, nothing is recorded — pure preview/explainability.
pub fn dry_run() -> Result<()> {
    let mimir = Mimir::open()?;
    let scope = mimir
        .project_for_cwd(&std::env::current_dir()?)?
        .map(|p| Scope::Project(p.id))
        .unwrap_or(Scope::Global);
    let now = now_unix();
    let rules = rules_body(&mimir, scope);
    let ranked = brief::select(
        &mimir.conn,
        scope,
        &HashSet::new(),
        rules.as_deref(),
        &mimir.config,
        now,
    )?;
    let composed = brief::compose(
        &ranked,
        &mimir.config.brief,
        mimir.config.brief.max_tokens,
        now,
    );
    if composed.text.is_empty() {
        println!("(empty brief — no gotcha/decision memories qualify in this scope)");
    } else {
        println!("{}", composed.text);
    }
    if !mimir.config.brief.enabled {
        println!(
            "\nnote: [brief] enabled = false — the SessionStart hook currently injects nothing"
        );
    }
    if !ranked.is_empty() {
        println!(
            "\ncandidates (score desc; * = would be shown at {} tokens):",
            mimir.config.brief.max_tokens
        );
        let shown: HashSet<i64> = composed.shown.iter().copied().collect();
        for r in &ranked {
            let mark = if shown.contains(&r.node.id) { "*" } else { " " };
            let flags = [
                r.node.pinned.then_some("pinned"),
                match (scope, r.node.project_id) {
                    (Scope::Project(p), Some(np)) if p == np => Some("project"),
                    _ => Some("global"),
                },
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(", ");
            println!(
                "{mark} {:>6.3}  {}  {}  [{flags}]",
                r.score,
                short_uid(r.node.kind, &r.node.uid),
                r.node.subkind.as_deref().unwrap_or("memory"),
            );
        }
    }
    Ok(())
}
