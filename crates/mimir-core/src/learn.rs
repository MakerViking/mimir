//! Self-learning signals: usage-driven strength with type-aware decay.
//!
//! Strength is always a *tiebreaker* in ranking, never a gate — a decayed
//! memory still surfaces when it's the best match. Signals: `opened`
//! (auto, +0.3), explicit `mark` (±1.0); both capped at one effect per
//! node per day so loops can't pump a memory.

use rusqlite::{Connection, OptionalExtension};

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
    // Co-activation is an observation ABOUT this open, not a part of it: if
    // the edge write fails, the open still happened and the caller must not
    // be told it didn't. Warned rather than propagated — two of the callers
    // (`peek`, `index::file_slice`) discard this Result, so an error returned
    // from here would go nowhere quietly, which is worse than a log line.
    if let Err(e) = record_coactivation(conn, node_id, now_unix()) {
        tracing::warn!(node_id, error = %e, "co-activation edges not recorded");
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

// ---------------------------------------------------------------------------
// Co-activation: the only edges the store draws for itself.
// ---------------------------------------------------------------------------
//
// Two memories opened in the same stretch of work are probably about the same
// stretch of work. Nothing else in the store notices that: `relates` edges are
// authored, `about` edges point at code, and recall treats every hit as
// independent. This records the observation and nothing more — phase 1 does
// not touch ranking, because an unmeasured scoring change is how the
// per-file-cap experiment lost 0.070 nDCG on an idea that sounded just as
// reasonable as this one. The edges are written now so there is something real
// to measure later.
//
// The signal is co-*opened*, not co-shown. `shown` fires for all eight hits of
// every recall — twenty-eight pairs a query, almost all of them noise. An open
// means an agent pulled the body, which is evidence rather than a guess.

/// Hebbian learning rate: a fire moves the weight `rate` of the way to 1.0.
/// The target is inside the legal range, so the update cannot overshoot and
/// needs no clamp — `w + r(1-w)` is a convex combination of two in-range
/// values. Reaching the visible threshold takes five co-opens.
pub const COACTIVATION_RATE: f64 = 0.15;

/// Half-life of an un-refreshed co-activation, in days. Applied lazily on
/// read (like [`effective_strength`]) rather than by a sweep, so there is no
/// background job and no write amplification on a table with one row per pair.
pub const COACTIVATION_HALF_LIFE_DAYS: f64 = 30.0;

/// Effective weight a pair must reach before anything displays it. Below this
/// the edge exists but is still a rumour: one shared session is a coincidence.
pub const COACTIVATION_VISIBLE: f64 = 0.5;

/// How close together two opens must fall to count as the same stretch of
/// work. Long enough to span a detour, short enough that yesterday's session
/// is not "the same stretch".
pub const COACTIVATION_WINDOW_SECS: i64 = 30 * 60;

/// Most co-activation edges kept per node, weakest evicted first. Without a
/// cap, a memory opened in every session pairs with the entire store and its
/// edges stop carrying information.
const COACTIVATION_MAX_PER_NODE: usize = 32;

/// Co-activation weight after time decay: `weight · 2^(−age/half_life)`,
/// age counted from the last fire. Same shape as [`effective_strength`]; a
/// pair that stops co-occurring fades instead of standing forever.
pub fn coactivation_weight(stored: f64, last_fired_at: i64, now: i64) -> f64 {
    let age_days = (now - last_fired_at).max(0) as f64 / 86_400.0;
    stored * 2f64.powf(-age_days / COACTIVATION_HALF_LIFE_DAYS)
}

/// Start of `now`'s UTC day. Mirrors [`capped_today`]'s boundary, using
/// `rem_euclid` so a pre-epoch timestamp floors instead of rounding toward
/// zero. The cap it feeds is per PAIR, which is why `capped_today` itself
/// cannot be reused: that one asks whether a node was opened today, and both
/// nodes of a pair can be opened repeatedly without the pair ever firing.
fn midnight(now: i64) -> i64 {
    now - now.rem_euclid(86_400)
}

/// Is this node a live memory? Co-activation is memory↔memory only.
/// `record_opened` also fires for symbols (`graph node`, `peek`) and for file
/// nodes (`index::file_slice`), and "this function was read near that note"
/// is a different claim with a different meaning — the code graph already
/// records what code relates to what code.
fn is_live_memory(conn: &Connection, node_id: i64) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT count(*) FROM node
         WHERE id = ?1 AND kind = 'memory'
           AND deleted_at IS NULL AND superseded_by IS NULL",
        [node_id],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

/// Does a pair already have an edge somebody meant? Authored edges are
/// written in either orientation (`supersede` writes new→old, `link` writes
/// as given), so both directions are checked, and every rel counts: if a
/// person already said how these two relate, the store has nothing to add.
fn has_authored_edge(conn: &Connection, a: i64, b: i64) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT count(*) FROM edge
         WHERE rel <> 'co_recalled'
           AND ((src = ?1 AND dst = ?2) OR (src = ?2 AND dst = ?1))",
        [a, b],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

/// Strengthen one pair, or create it at `COACTIVATION_RATE`. Returns false if
/// the pair already fired today.
///
/// Endpoints are stored canonically (`src` = the lower rowid) because
/// `UNIQUE(src, dst, rel)` is directional: inserting the same pair both ways
/// round would make two rows that the cap, the eviction and the authored-edge
/// check would each see as unrelated.
///
/// This cannot go through `store::link`, which overwrites `weight` on
/// conflict and never writes `meta` — it would reset the accumulated weight
/// and drop the timestamp the decay is measured from. `link_call` in
/// mimir-graph is the precedent for an edge writer that carries its own meta.
fn fire_pair(conn: &Connection, a: i64, b: i64, now: i64) -> Result<bool> {
    let (src, dst) = (a.min(b), a.max(b));
    let existing: Option<(i64, f64, Option<i64>)> = conn
        .query_row(
            "SELECT id, weight, json_extract(meta, '$.last_fired_at') FROM edge
             WHERE src = ?1 AND dst = ?2 AND rel = 'co_recalled'",
            [src, dst],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;

    match existing {
        Some((id, weight, last_fired_at)) => {
            let last_fired_at = last_fired_at.unwrap_or(0);
            if last_fired_at >= midnight(now) {
                return Ok(false);
            }
            let decayed = coactivation_weight(weight, last_fired_at, now);
            let next = decayed + COACTIVATION_RATE * (1.0 - decayed);
            conn.execute(
                "UPDATE edge SET weight = ?2,
                    meta = json_set(meta, '$.last_fired_at', ?3)
                 WHERE id = ?1",
                rusqlite::params![id, next, now],
            )?;
        }
        None => {
            conn.execute(
                "INSERT INTO edge (src, dst, rel, weight, meta, created_at)
                 VALUES (?1, ?2, 'co_recalled', ?3, json_object('last_fired_at', ?4), ?4)",
                rusqlite::params![src, dst, COACTIVATION_RATE, now],
            )?;
        }
    }
    Ok(true)
}

/// Trim `node_id` back to [`COACTIVATION_MAX_PER_NODE`] co-activation edges,
/// weakest first.
///
/// Ranking happens in Rust over the (at most ~33) candidate rows rather than
/// in SQL, because the comparison is on the DECAYED weight: ordering by the
/// stored column would keep a strong-but-stale pair over a fresh one, which
/// is the opposite of what the decay is for. `id` breaks ties, since the
/// same-day cap makes exact float ties common.
fn evict_beyond_cap(conn: &Connection, node_id: i64, now: i64) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT id, weight, COALESCE(json_extract(meta, '$.last_fired_at'), created_at)
         FROM edge WHERE rel = 'co_recalled' AND (src = ?1 OR dst = ?1)",
    )?;
    let mut rows: Vec<(i64, f64)> = stmt
        .query_map([node_id], |r| {
            let id: i64 = r.get(0)?;
            let weight: f64 = r.get(1)?;
            let last_fired_at: i64 = r.get(2)?;
            Ok((id, coactivation_weight(weight, last_fired_at, now)))
        })?
        .collect::<rusqlite::Result<_>>()?;
    if rows.len() <= COACTIVATION_MAX_PER_NODE {
        return Ok(());
    }
    rows.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    for (id, _) in &rows[..rows.len() - COACTIVATION_MAX_PER_NODE] {
        conn.execute("DELETE FROM edge WHERE id = ?1", [id])?;
    }
    Ok(())
}

/// Pair `node_id` with every other live memory opened in the last
/// [`COACTIVATION_WINDOW_SECS`], strengthening one edge per pair per day.
/// Returns how many pairs fired.
///
/// Call this AFTER the node's own `opened` row is written: the window is read
/// from `recall_event`, so ordering decides whether two opens in the same
/// instant see each other.
pub fn record_coactivation(conn: &Connection, node_id: i64, now: i64) -> Result<usize> {
    if !is_live_memory(conn, node_id)? {
        return Ok(0);
    }
    let mut stmt = conn.prepare(
        "SELECT DISTINCT e.node_id FROM recall_event e
         JOIN node n ON n.id = e.node_id
         WHERE e.event = 'opened' AND e.at >= ?2 AND e.node_id <> ?1
           AND n.kind = 'memory'
           AND n.deleted_at IS NULL AND n.superseded_by IS NULL",
    )?;
    let partners: Vec<i64> = stmt
        .query_map(
            rusqlite::params![node_id, now - COACTIVATION_WINDOW_SECS],
            |r| r.get(0),
        )?
        .collect::<rusqlite::Result<_>>()?;

    let mut fired = Vec::new();
    for partner in partners {
        if has_authored_edge(conn, node_id, partner)? {
            continue;
        }
        if fire_pair(conn, node_id, partner, now)? {
            fired.push(partner);
        }
    }
    if !fired.is_empty() {
        evict_beyond_cap(conn, node_id, now)?;
    }
    Ok(fired.len())
}

/// The co-activation partners of `node_id` that have earned display, strongest
/// first: effective weight at or above [`COACTIVATION_VISIBLE`], dead and
/// retired endpoints dropped. Returns `(node_id, effective_weight)`.
///
/// Endpoints are filtered here rather than at write time because `forget` and
/// `supersede` are soft — neither deletes the row, so `ON DELETE CASCADE`
/// never fires and edges to retired memories survive indefinitely.
pub fn coactivation_partners(conn: &Connection, node_id: i64, now: i64) -> Result<Vec<(i64, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT CASE WHEN e.src = ?1 THEN e.dst ELSE e.src END,
                e.weight,
                COALESCE(json_extract(e.meta, '$.last_fired_at'), e.created_at)
         FROM edge e
         JOIN node n ON n.id = CASE WHEN e.src = ?1 THEN e.dst ELSE e.src END
         WHERE e.rel = 'co_recalled' AND (e.src = ?1 OR e.dst = ?1)
           AND n.deleted_at IS NULL AND n.superseded_by IS NULL",
    )?;
    let mut partners: Vec<(i64, f64)> = stmt
        .query_map([node_id], |r| {
            let other: i64 = r.get(0)?;
            let weight: f64 = r.get(1)?;
            let last_fired_at: i64 = r.get(2)?;
            Ok((other, coactivation_weight(weight, last_fired_at, now)))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .filter(|(_, w)| *w >= COACTIVATION_VISIBLE)
        .collect();
    partners.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(partners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::{Actor, NewNode, Rel};

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

    // -----------------------------------------------------------------
    // Co-activation
    // -----------------------------------------------------------------

    /// A fixed synthetic clock, never `now_unix()`. Both day boundaries in
    /// this feature (`midnight`, and `capped_today` behind `record_opened`)
    /// are derived from the timestamp itself, so a test that planted an
    /// event at "sixty seconds ago" would land the two ends of a pair on
    /// opposite sides of UTC midnight whenever CI happened to run at 00:00.
    /// This instant sits 80_000s into its own day — far enough in that a
    /// full window can be subtracted without crossing back into yesterday.
    const T: i64 = 1_700_000_000;

    /// Plant an `opened` row directly. `record_opened` stamps `now_unix()`
    /// and would put the event outside any synthetic window.
    fn opened_at(conn: &Connection, node_id: i64, at: i64) {
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'opened', ?2)",
            rusqlite::params![node_id, at],
        )
        .unwrap();
    }

    /// Write a co-activation edge with a chosen weight and age, for the
    /// cases that would otherwise need dozens of simulated days to reach.
    fn seed_co_edge(conn: &Connection, a: i64, b: i64, weight: f64, last_fired_at: i64) {
        conn.execute(
            "INSERT INTO edge (src, dst, rel, weight, meta, created_at)
             VALUES (?1, ?2, 'co_recalled', ?3, json_object('last_fired_at', ?4), ?4)",
            rusqlite::params![a.min(b), a.max(b), weight, last_fired_at],
        )
        .unwrap();
    }

    /// Every co-activation edge in the store, insertion order, as
    /// `(src, dst, weight, last_fired_at)`. Read raw rather than through
    /// `store::edges_of` so the stored orientation is visible: `edges_of`
    /// returns both directions of the same row identically.
    fn co_edges(conn: &Connection) -> Vec<(i64, i64, f64, i64)> {
        let mut stmt = conn
            .prepare(
                "SELECT src, dst, weight, json_extract(meta, '$.last_fired_at')
                 FROM edge WHERE rel = 'co_recalled' ORDER BY id",
            )
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    /// The other end of every co-activation edge touching `node_id`,
    /// unfiltered — unlike `coactivation_partners`, which hides the weak
    /// and the retired and so cannot show what the cap actually kept.
    fn co_partner_ids(conn: &Connection, node_id: i64) -> Vec<i64> {
        let mut stmt = conn
            .prepare(
                "SELECT CASE WHEN src = ?1 THEN dst ELSE src END FROM edge
                 WHERE rel = 'co_recalled' AND (src = ?1 OR dst = ?1) ORDER BY id",
            )
            .unwrap();
        stmt.query_map([node_id], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    fn retire(conn: &Connection, id: i64, by: i64) {
        conn.execute(
            "UPDATE node SET superseded_by = ?2 WHERE id = ?1",
            rusqlite::params![id, by],
        )
        .unwrap();
    }

    #[test]
    fn a_pair_in_the_window_gets_one_edge_stored_the_same_way_round_either_way() {
        // "Opened second" is whichever node `record_coactivation` is called
        // for. The row must not encode that: `UNIQUE(src, dst, rel)` is
        // directional, so an orientation that followed call order would let
        // one pair exist as two rows the cap and the eviction see as
        // unrelated.
        for lower_opened_first in [true, false] {
            let conn = db::open_in_memory().unwrap();
            let lower = memory(&conn, "note");
            let higher = memory(&conn, "note");
            assert!(lower.id < higher.id, "insertion order fixes the rowids");
            let (earlier, later) = if lower_opened_first {
                (lower.id, higher.id)
            } else {
                (higher.id, lower.id)
            };
            opened_at(&conn, earlier, T - 60);
            let fired = record_coactivation(&conn, later, T).unwrap();
            assert_eq!(fired, 1, "one partner in the window");

            let edges = co_edges(&conn);
            assert_eq!(edges.len(), 1, "exactly one edge: {edges:?}");
            let (src, dst, weight, last_fired_at) = edges[0];
            assert_eq!(
                (src, dst),
                (lower.id, higher.id),
                "src must be the lower rowid whichever end was opened second: {edges:?}"
            );
            assert!(
                (weight - COACTIVATION_RATE).abs() < 1e-9,
                "a new pair starts at the learning rate, got {weight}"
            );
            assert_eq!(last_fired_at, T, "decay is measured from this stamp");
        }
    }

    #[test]
    fn opens_further_apart_than_the_window_never_pair() {
        let conn = db::open_in_memory().unwrap();
        let a = memory(&conn, "note");
        let b = memory(&conn, "note");
        opened_at(&conn, a.id, T - COACTIVATION_WINDOW_SECS - 1);
        let fired = record_coactivation(&conn, b.id, T).unwrap();
        assert_eq!(
            fired, 0,
            "one second past the window is a different session"
        );
        assert!(co_edges(&conn).is_empty(), "{:?}", co_edges(&conn));

        // The other side of the same second. Without this the assertion
        // above would also pass for a query that never matches anything.
        let conn = db::open_in_memory().unwrap();
        let a = memory(&conn, "note");
        let b = memory(&conn, "note");
        opened_at(&conn, a.id, T - COACTIVATION_WINDOW_SECS);
        let fired = record_coactivation(&conn, b.id, T).unwrap();
        assert_eq!(fired, 1, "the window edge itself still counts");
    }

    #[test]
    fn a_pair_fires_once_a_day_and_compounds_on_the_decayed_weight() {
        let conn = db::open_in_memory().unwrap();
        let a = memory(&conn, "note");
        let b = memory(&conn, "note");
        opened_at(&conn, a.id, T - 60);
        assert_eq!(record_coactivation(&conn, b.id, T).unwrap(), 1);

        // Same day, same partner still inside the window: the pair is
        // capped, so nothing about the row may move — not the weight and
        // not the stamp the decay is measured from.
        let again = record_coactivation(&conn, b.id, T + 60).unwrap();
        assert_eq!(again, 0, "second fire of the day is a no-op");
        let (_, _, weight, last_fired_at) = co_edges(&conn)[0];
        assert!(
            (weight - COACTIVATION_RATE).abs() < 1e-9,
            "weight must not move on a capped fire, got {weight}"
        );
        assert_eq!(last_fired_at, T, "nor may the stamp be refreshed");

        // Exactly one day later. Decay first, then the Hebbian step on the
        // decayed value:
        //   decayed = 0.15 · 2^(-1/30)          = 0.14657399526513687
        //   next    = decayed + 0.15(1-decayed) = 0.27458789597536637
        let tomorrow = T + 86_400;
        opened_at(&conn, a.id, tomorrow - 60);
        assert_eq!(record_coactivation(&conn, b.id, tomorrow).unwrap(), 1);
        let (_, _, weight, last_fired_at) = co_edges(&conn)[0];
        assert!(
            (weight - 0.274_587_895_975_366_37).abs() < 1e-12,
            "a day-old 0.15 pair must land on 0.2745878959753664, got {weight}"
        );
        assert_eq!(last_fired_at, tomorrow);
        assert_eq!(co_edges(&conn).len(), 1, "still one row, not a second one");
    }

    #[test]
    fn an_authored_edge_suppresses_the_pair_in_either_orientation() {
        // `link` writes as given and `supersede` writes new→old, so the
        // authored edge can point either way; both must silence the pair.
        for authored_low_to_high in [true, false] {
            let conn = db::open_in_memory().unwrap();
            let a = memory(&conn, "note");
            let b = memory(&conn, "note");
            let (src, dst) = if authored_low_to_high {
                (a.id, b.id)
            } else {
                (b.id, a.id)
            };
            store::link(&conn, src, dst, Rel::Relates, 1.0).unwrap();
            opened_at(&conn, a.id, T - 60);

            let fired = record_coactivation(&conn, b.id, T).unwrap();
            assert_eq!(fired, 0, "authored {src}->{dst} must suppress the pair");
            assert!(
                co_edges(&conn).is_empty(),
                "no co_recalled row may be added beside an authored one: {:?}",
                co_edges(&conn)
            );
        }
    }

    #[test]
    fn only_live_memories_pair() {
        let conn = db::open_in_memory().unwrap();
        let live = memory(&conn, "note");
        let symbol = store::insert_node(&conn, NewNode::new(Kind::Symbol)).unwrap();
        let deleted = memory(&conn, "note");
        store::soft_delete(&conn, deleted.id, Actor::Human, None).unwrap();
        let superseded = memory(&conn, "note");
        retire(&conn, superseded.id, live.id);

        for id in [symbol.id, deleted.id, superseded.id] {
            opened_at(&conn, id, T - 60);
        }
        let fired = record_coactivation(&conn, live.id, T).unwrap();
        assert_eq!(
            fired, 0,
            "a symbol, a forgotten memory and a retired one are not partners"
        );

        // And the same rule read from the other end: the node being opened
        // is the one that isn't a live memory.
        opened_at(&conn, live.id, T - 60);
        let fired = record_coactivation(&conn, symbol.id, T).unwrap();
        assert_eq!(fired, 0, "a symbol never co-activates with a memory");
        assert!(co_edges(&conn).is_empty(), "{:?}", co_edges(&conn));
    }

    #[test]
    fn partners_are_strongest_first_above_the_threshold_and_drop_dead_ends() {
        let conn = db::open_in_memory().unwrap();
        let hub = memory(&conn, "note");
        let strong = memory(&conn, "note");
        let mid = memory(&conn, "note");
        let exact = memory(&conn, "note");
        let weak = memory(&conn, "note");
        let faded = memory(&conn, "note");
        let deleted = memory(&conn, "note");
        let superseded = memory(&conn, "note");

        seed_co_edge(&conn, hub.id, mid.id, 0.6, T);
        seed_co_edge(&conn, hub.id, strong.id, 0.9, T);
        seed_co_edge(&conn, hub.id, exact.id, COACTIVATION_VISIBLE, T);
        seed_co_edge(&conn, hub.id, weak.id, COACTIVATION_VISIBLE - 0.01, T);
        // Two half-lives of silence takes a once-strong pair to 0.225.
        seed_co_edge(&conn, hub.id, faded.id, 0.9, T - 60 * 86_400);
        seed_co_edge(&conn, hub.id, deleted.id, 1.0, T);
        seed_co_edge(&conn, hub.id, superseded.id, 1.0, T);
        store::soft_delete(&conn, deleted.id, Actor::Human, None).unwrap();
        retire(&conn, superseded.id, hub.id);

        let got = coactivation_partners(&conn, hub.id, T).unwrap();
        let ids: Vec<i64> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![strong.id, mid.id, exact.id],
            "strongest first, threshold inclusive, soft-deleted and retired ends gone: {got:?}"
        );
        assert!(
            (got[0].1 - 0.9).abs() < 1e-9,
            "the effective weight is returned, not the stored one: {got:?}"
        );
    }

    #[test]
    fn the_per_node_cap_evicts_the_weakest_by_decayed_weight() {
        let conn = db::open_in_memory().unwrap();
        let hub = memory(&conn, "note");

        // One short of the cap, all fired in the same session.
        let fresh: Vec<i64> = (0..COACTIVATION_MAX_PER_NODE - 1)
            .map(|_| memory(&conn, "note").id)
            .collect();
        for id in &fresh {
            opened_at(&conn, *id, T - 60);
        }
        let fired = record_coactivation(&conn, hub.id, T).unwrap();
        assert_eq!(fired, COACTIVATION_MAX_PER_NODE - 1, "all pair at once");

        // A pair with a big stored weight that has not fired in eight
        // years: 0.9 · 2^(-100) ≈ 7e-31, far below a fresh 0.15. Created
        // last on purpose, so it holds the HIGHEST edge id — if eviction
        // ranked by the stored column, or fell through to the id
        // tiebreaker, this row would be the one that survived.
        let stale = memory(&conn, "note");
        seed_co_edge(&conn, hub.id, stale.id, 0.9, T - 3000 * 86_400);
        assert_eq!(
            co_partner_ids(&conn, hub.id).len(),
            COACTIVATION_MAX_PER_NODE
        );

        // One more fresh pair takes the node one over the cap.
        let newcomer = memory(&conn, "note");
        opened_at(&conn, newcomer.id, T - 60);
        let fired = record_coactivation(&conn, hub.id, T).unwrap();
        assert_eq!(fired, 1, "the ones already fired today must not fire again");

        let kept = co_partner_ids(&conn, hub.id);
        assert_eq!(kept.len(), COACTIVATION_MAX_PER_NODE, "cap holds: {kept:?}");
        assert!(
            !kept.contains(&stale.id),
            "the strong-but-stale pair is the weakest once decayed: {kept:?}"
        );
        assert!(
            kept.contains(&newcomer.id),
            "the pair that just fired must not be the one evicted: {kept:?}"
        );
        for id in &fresh {
            assert!(
                kept.contains(id),
                "fresh pair {id} evicted instead: {kept:?}"
            );
        }
    }
}
