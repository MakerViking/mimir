//! Self-learning signals: usage-driven strength with type-aware decay.
//!
//! Strength is always a *tiebreaker* in ranking, never a gate — a decayed
//! memory still surfaces when it's the best match. Signals: `opened`
//! (auto, +0.3), explicit `mark` (±1.0); both capped at one effect per
//! node per day so loops can't pump a memory.

use rusqlite::Connection;

use crate::error::Result;
use crate::model::{now_unix, Kind, Node};
use crate::store;

pub const STRENGTH_MIN: f64 = 0.05;
pub const STRENGTH_MAX: f64 = 10.0;
const OPENED_BONUS: f64 = 0.3;
const MARK_DELTA: f64 = 1.0;

/// Half-life in days for a memory subkind. None = never decays
/// (person notes, docs, code, summaries).
pub fn half_life_days(kind: Kind, subkind: Option<&str>) -> Option<f64> {
    if kind != Kind::Memory {
        return None;
    }
    match subkind.unwrap_or("note") {
        "gotcha" => Some(180.0),
        "decision" => Some(365.0),
        "insight" => Some(120.0),
        "note" => Some(90.0),
        "idea" => Some(60.0),
        _ => None, // person, summary
    }
}

/// Strength after time decay: `strength · 2^(−age/half_life)`, where age
/// counts from the last meaningful touch (access beats creation).
/// Pinned nodes never decay.
pub fn effective_strength(node: &Node, now: i64) -> f64 {
    if node.pinned {
        return node.strength;
    }
    let Some(half_life) = half_life_days(node.kind, node.subkind.as_deref()) else {
        return node.strength;
    };
    let reference = node.last_accessed.unwrap_or(node.created_at);
    let age_days = (now - reference).max(0) as f64 / 86_400.0;
    node.strength * 2f64.powf(-age_days / half_life)
}

/// Time-recency factor in (0, 1]: `2^(-age/half_life)` from CREATION (newer
/// information wins), using the same type-aware half-lives as decay. Returns
/// 1.0 for pinned or never-decaying kinds (docs, code, person, summary) —
/// but 1.0 is the *max* of this range, not a neutral "no boost" value, so
/// callers MUST gate its use on `half_life_days(kind, subkind).is_some()`
/// (see the score site in `search/mod.rs`) rather than applying it
/// unconditionally; otherwise every non-decaying node gets a permanent
/// boost relative to every aging memory instead of staying recency-neutral.
/// Independent of learned strength; used as an opt-in ranking boost
/// (scoring.recency_alpha) so a fresh "it's done" memory can outrank a stale
/// "in progress" one even on a slightly weaker text match.
pub fn recency_factor(node: &Node, now: i64) -> f64 {
    if node.pinned {
        return 1.0;
    }
    let Some(half_life) = half_life_days(node.kind, node.subkind.as_deref()) else {
        return 1.0;
    };
    let age_days = (now - node.created_at).max(0) as f64 / 86_400.0;
    2f64.powf(-age_days / half_life)
}

/// Fixed per-subkind priority multiplier: gotcha (drift-preventer) and
/// decision (policy call) memories are worth surfacing over an
/// equally-matching but non-critical note/idea/insight/summary/person, or a
/// non-Memory node. Scaled by `scoring.type_prior_alpha` (0.0 = off) at the
/// fused score site — this table only says which subkinds get a prior, not
/// how strongly it counts.
pub fn type_prior(subkind: Option<&str>) -> f64 {
    match subkind {
        Some("gotcha") | Some("decision") => 1.25,
        _ => 1.0,
    }
}

/// Per-node impression counts from the `recall_event` ledger
/// (`store::record_shown` / `learn::record_opened`), batch-fetched by
/// `store::impression_stats` for [`impression_damp`] below.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImpressionStats {
    pub shown: i64,
    pub opened: i64,
    /// Unix time of the most recent `shown` event, if any.
    pub last_shown_at: Option<i64>,
}

/// Minimum `shown` count before the negative prior applies at all — a node
/// unlucky enough to surface a handful of times without being opened yet
/// isn't evidence of anything; this floor requires a real pattern.
const IMPRESSION_MIN_SHOWN: i64 = 10;

/// Half-life (days, from the most recent `shown` event) the "shown without
/// opened" signal decays on — so a node that goes quiet stops being
/// punished for a stale pattern rather than carrying the damp forever.
const IMPRESSION_DECAY_HALF_LIFE_DAYS: f64 = 30.0;

/// Floor on the damp multiplier: however large `impression_alpha` is
/// configured, this signal can never cut a score by more than 30%. It's a
/// tiebreaker demotion, not a ban — the node may still be the best match.
const IMPRESSION_DAMP_FLOOR: f64 = 0.7;

/// Negative-prior multiplier (config `scoring.impression_alpha`, default
/// 0.0/off) for a node that's been shown many times but never opened: a
/// standing signal that it keeps surfacing without being useful. Returns
/// exactly `1.0` (a true no-op, not just numerically close) whenever:
/// `alpha <= 0.0`, the node has ANY `opened` event (an opened node is never
/// damped by this — being opened at least once outweighs any number of
/// unopened impressions), or `shown` hasn't reached [`IMPRESSION_MIN_SHOWN`].
/// Otherwise the damp decays with the age of the most recent impression
/// (see [`IMPRESSION_DECAY_HALF_LIFE_DAYS`]) and is bounded below by
/// [`IMPRESSION_DAMP_FLOOR`].
pub fn impression_damp(stats: &ImpressionStats, alpha: f64, now: i64) -> f64 {
    if alpha <= 0.0 || stats.opened > 0 || stats.shown < IMPRESSION_MIN_SHOWN {
        return 1.0;
    }
    let Some(last_shown_at) = stats.last_shown_at else {
        return 1.0;
    };
    let age_days = (now - last_shown_at).max(0) as f64 / 86_400.0;
    let decay = 2f64.powf(-age_days / IMPRESSION_DECAY_HALF_LIFE_DAYS);
    (1.0 - alpha * decay).max(IMPRESSION_DAMP_FLOOR)
}

/// Has this node already received an event of this type today?
fn capped_today(conn: &Connection, node_id: i64, event: &str) -> Result<bool> {
    let now = now_unix();
    let midnight = now - (now % 86_400);
    let n: i64 = conn.query_row(
        "SELECT count(*) FROM recall_event
         WHERE node_id = ?1 AND event = ?2 AND at >= ?3",
        rusqlite::params![node_id, event, midnight],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

fn adjust_strength(conn: &Connection, node_id: i64, delta: f64) -> Result<()> {
    conn.execute(
        "UPDATE node SET strength = max(?2, min(?3, strength + ?4)) WHERE id = ?1",
        rusqlite::params![node_id, STRENGTH_MIN, STRENGTH_MAX, delta],
    )?;
    Ok(())
}

/// The single entry point for "an agent opened this node's full body":
/// bumps access stats, logs the event, and applies the (daily-capped)
/// strength bonus.
pub fn record_opened(conn: &Connection, node_id: i64) -> Result<()> {
    let first_today = !capped_today(conn, node_id, "opened")?;
    store::touch_accessed(conn, node_id)?;
    store::record_event(conn, node_id, "opened", None, None, None)?;
    if first_today {
        adjust_strength(conn, node_id, OPENED_BONUS)?;
    }
    Ok(())
}

/// Explicit feedback: ±1.0 strength, one effect per node per day.
/// Returns the node's new strength.
pub fn apply_mark(conn: &Connection, node_id: i64, useful: bool) -> Result<f64> {
    let event = if useful { "useful" } else { "not_useful" };
    let first_today = !capped_today(conn, node_id, event)?;
    store::record_event(conn, node_id, event, None, None, None)?;
    if first_today {
        adjust_strength(conn, node_id, if useful { MARK_DELTA } else { -MARK_DELTA })?;
    }
    Ok(
        conn.query_row("SELECT strength FROM node WHERE id = ?1", [node_id], |r| {
            r.get(0)
        })?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::NewNode;

    fn memory(conn: &Connection, subkind: &str) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some(subkind.into());
        new.title = Some("t".into());
        new.body = Some("b".into());
        store::insert_node(conn, new).unwrap()
    }

    #[test]
    fn decay_math() {
        let conn = db::open_in_memory().unwrap();
        let mut node = memory(&conn, "note"); // 90d half-life
        let now = node.created_at;
        assert!((effective_strength(&node, now) - 1.0).abs() < 1e-9);
        let half = effective_strength(&node, now + 90 * 86_400);
        assert!((half - 0.5).abs() < 1e-9, "one half-life → 0.5, got {half}");

        node.pinned = true;
        assert_eq!(effective_strength(&node, now + 900 * 86_400), 1.0);

        node.pinned = false;
        node.subkind = Some("person".into());
        assert_eq!(effective_strength(&node, now + 900 * 86_400), 1.0);

        // Access resets the decay clock.
        node.subkind = Some("note".into());
        node.last_accessed = Some(now + 90 * 86_400);
        let fresh = effective_strength(&node, now + 90 * 86_400);
        assert!((fresh - 1.0).abs() < 1e-9);
    }

    #[test]
    fn opened_bonus_capped_per_day() {
        let conn = db::open_in_memory().unwrap();
        let node = memory(&conn, "note");
        record_opened(&conn, node.id).unwrap();
        record_opened(&conn, node.id).unwrap();
        record_opened(&conn, node.id).unwrap();
        let n = store::get_node(&conn, node.id).unwrap();
        assert!(
            (n.strength - 1.3).abs() < 1e-9,
            "one bonus only: {}",
            n.strength
        );
        assert_eq!(n.access_count, 3, "access count still counts every open");
    }

    #[test]
    fn mark_caps_and_bounds() {
        let conn = db::open_in_memory().unwrap();
        let node = memory(&conn, "gotcha");
        assert!((apply_mark(&conn, node.id, true).unwrap() - 2.0).abs() < 1e-9);
        // Second mark same day: no further effect.
        assert!((apply_mark(&conn, node.id, true).unwrap() - 2.0).abs() < 1e-9);
        // Noise marks floor at STRENGTH_MIN eventually.
        conn.execute("UPDATE node SET strength = 0.5 WHERE id = ?1", [node.id])
            .unwrap();
        let s = apply_mark(&conn, node.id, false).unwrap();
        assert!((s - STRENGTH_MIN).abs() < 1e-9, "floored: {s}");
    }

    fn stats(shown: i64, opened: i64, last_shown_at: Option<i64>) -> ImpressionStats {
        ImpressionStats {
            shown,
            opened,
            last_shown_at,
        }
    }

    #[test]
    fn impression_damp_alpha_zero_is_exact_noop() {
        // Way past the threshold, never opened, shown just now: if alpha
        // were doing anything at all it would show here.
        let s = stats(1000, 0, Some(1_000_000));
        assert_eq!(impression_damp(&s, 0.0, 1_000_000), 1.0);
    }

    #[test]
    fn impression_damp_requires_minimum_shown_count() {
        let now = 1_000_000;
        let s = stats(IMPRESSION_MIN_SHOWN - 1, 0, Some(now));
        assert_eq!(
            impression_damp(&s, 0.5, now),
            1.0,
            "below the threshold, no damp regardless of alpha"
        );
        let s = stats(IMPRESSION_MIN_SHOWN, 0, Some(now));
        assert!(
            impression_damp(&s, 0.5, now) < 1.0,
            "right at the threshold, freshly shown, damp must apply"
        );
    }

    #[test]
    fn impression_damp_exempts_opened_nodes() {
        let now = 1_000_000;
        // Huge shown count, only ONE open ever — still fully exempt.
        let s = stats(500, 1, Some(now));
        assert_eq!(
            impression_damp(&s, 0.9, now),
            1.0,
            "any opened event exempts a node from this damp entirely"
        );
    }

    #[test]
    fn impression_damp_is_bounded_by_the_floor() {
        let now = 1_000_000;
        let s = stats(500, 0, Some(now)); // shown "just now": decay ~= 1.0
        let damp = impression_damp(&s, 5.0, now); // absurdly large alpha
        assert!(
            damp >= IMPRESSION_DAMP_FLOOR - 1e-9,
            "damp {damp} must never drop below the floor"
        );
    }

    #[test]
    fn impression_damp_decays_back_toward_one_over_time() {
        let now = 1_000_000;
        let s_recent = stats(500, 0, Some(now));
        let one_half_life_ago = now - (IMPRESSION_DECAY_HALF_LIFE_DAYS as i64 * 86_400);
        let s_stale = stats(500, 0, Some(one_half_life_ago));
        let recent = impression_damp(&s_recent, 0.4, now);
        let stale = impression_damp(&s_stale, 0.4, now);
        assert!(
            stale > recent,
            "a stale unopened-impression pattern ({stale}) must be punished less than a fresh one ({recent})"
        );
        // One half-life out, the alpha's effect should be roughly halved:
        // damp ~= 1 - alpha/2, not clamped by the floor at these numbers.
        assert!((stale - (1.0 - 0.4 / 2.0)).abs() < 0.01, "stale={stale}");
    }
}
