//! Token-savings ledger: append-only record of every context-shrinking
//! operation, plus the aggregations the dashboard/report render.
//!
//! A "source" is one of: `outline`, `peek`, `filter`, `cap`, `proxy_cache`,
//! `proxy_prune`. `saved = before - after` (cache savings are billed-token
//! reductions; the rest are tokens the agent never had to read).

use rusqlite::{params, Connection};
use serde_json::Value;

use crate::error::Result;
use crate::model::now_unix;

/// Canonical `source` identifiers — one truth, so the producers (outline, peek,
/// filter, proxy) and the dashboard never drift on a renamed string.
pub mod source {
    pub const OUTLINE: &str = "outline";
    pub const PEEK: &str = "peek";
    pub const FILTER: &str = "filter";
    /// Non-lossy volume cap on high-volume content commands (coreutils/kubectl),
    /// kept distinct from `filter` (per-line noise dropping) in the ledger.
    pub const CAP: &str = "cap";
    pub const PROXY_CACHE: &str = "proxy_cache";
    pub const PROXY_DEDUP: &str = "proxy_dedup";
    pub const PROXY_PRUNE: &str = "proxy_prune";
    /// Session-brief injection (`mimir brief show`). Recorded with
    /// `before = 0` — honestly a COST, not a saving; cost sources are
    /// excluded from every "saved" aggregate (see [`COST_SOURCES_SQL`])
    /// so injected tokens can never silently net against real savings.
    /// A "Spent" rollup in `mimir savings` is required before `[brief]
    /// enabled` may default on (see docs/design/session-brief.md §5).
    pub const BRIEF: &str = "brief";
}

/// Aggregate counters for a set of savings events.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Totals {
    pub events: i64,
    pub before: i64,
    pub after: i64,
}

impl Totals {
    pub fn saved(&self) -> i64 {
        (self.before - self.after).max(0)
    }
}

/// Tokens saved → dollars, the one home for the pricing math. `price_per_mtok`
/// is the input-token price in USD per million tokens.
pub fn to_dollars(saved_tokens: i64, price_per_mtok: f64) -> f64 {
    saved_tokens as f64 * price_per_mtok / 1_000_000.0
}

/// Saved-token sums for the standard reporting windows, in a single table scan.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub day: i64,
    pub week: i64,
    pub month: i64,
    pub all: Totals,
}

/// SQL `IN`-list of pure-cost sources (tokens injected, nothing shrunk).
/// Every "saved" aggregate (`summary`, `totals`, `daily_saved`) excludes
/// them — a cost event must never net against real savings, which the raw
/// `before - after` sums would silently do. They remain visible in
/// `by_source` (its per-source `Totals::saved()` clamps to 0).
const COST_SOURCES_SQL: &str = "('brief')";

/// One-scan equivalent of calling `totals` for day/week/month/all.
pub fn summary(conn: &Connection, now: i64) -> Result<Summary> {
    let row = conn.query_row(
        &format!(
            "SELECT
               COALESCE(SUM(CASE WHEN at >= ?1 THEN tokens_before - tokens_after END), 0),
               COALESCE(SUM(CASE WHEN at >= ?2 THEN tokens_before - tokens_after END), 0),
               COALESCE(SUM(CASE WHEN at >= ?3 THEN tokens_before - tokens_after END), 0),
               COUNT(*), COALESCE(SUM(tokens_before), 0), COALESCE(SUM(tokens_after), 0)
             FROM savings_event WHERE source NOT IN {COST_SOURCES_SQL}"
        ),
        params![now - 86_400, now - 7 * 86_400, now - 30 * 86_400],
        |r| {
            Ok(Summary {
                day: r.get(0)?,
                week: r.get(1)?,
                month: r.get(2)?,
                all: Totals {
                    events: r.get(3)?,
                    before: r.get(4)?,
                    after: r.get(5)?,
                },
            })
        },
    )?;
    Ok(row)
}

/// Injected-context spend for the standard reporting windows — the mirror
/// of [`Summary`] over the COST sources only (today: the session brief).
/// This is the "Spent" rollup the brief's default-flip gate requires:
/// cost events are excluded from every saved-aggregate, and without this
/// they'd be excluded into invisibility.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spent {
    pub day: i64,
    pub week: i64,
    pub month: i64,
    pub all: i64,
    /// Rendered fires, all time.
    pub events: i64,
}

/// One-scan windowed spend over the cost sources. Spend is
/// `tokens_after - tokens_before` (cost sources record `before = 0`
/// today; the subtraction stays honest if a future cost source ever
/// records a nonzero before).
pub fn spent(conn: &Connection, now: i64) -> Result<Spent> {
    let row = conn.query_row(
        &format!(
            "SELECT
               COALESCE(SUM(CASE WHEN at >= ?1 THEN tokens_after - tokens_before END), 0),
               COALESCE(SUM(CASE WHEN at >= ?2 THEN tokens_after - tokens_before END), 0),
               COALESCE(SUM(CASE WHEN at >= ?3 THEN tokens_after - tokens_before END), 0),
               COALESCE(SUM(tokens_after - tokens_before), 0), COUNT(*)
             FROM savings_event WHERE source IN {COST_SOURCES_SQL}"
        ),
        params![now - 86_400, now - 7 * 86_400, now - 30 * 86_400],
        |r| {
            Ok(Spent {
                day: r.get(0)?,
                week: r.get(1)?,
                month: r.get(2)?,
                all: r.get(3)?,
                events: r.get(4)?,
            })
        },
    )?;
    Ok(row)
}

/// Append one savings event. `project_id` is optional (the proxy may run
/// outside any project). Never fails the caller's main work — callers that
/// only want best-effort logging can ignore the error.
pub fn record(
    conn: &Connection,
    project_id: Option<i64>,
    source: &str,
    before: usize,
    after: usize,
    meta: Value,
) -> Result<()> {
    conn.execute(
        "INSERT INTO savings_event (at, project_id, source, tokens_before, tokens_after, meta)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            now_unix(),
            project_id,
            source,
            before as i64,
            after as i64,
            meta.to_string(),
        ],
    )?;
    Ok(())
}

/// Grand totals since `since` (unix seconds; None = all time).
pub fn totals(conn: &Connection, since: Option<i64>) -> Result<Totals> {
    let cutoff = since.unwrap_or(0);
    let row = conn.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(tokens_before),0), COALESCE(SUM(tokens_after),0)
             FROM savings_event WHERE at >= ?1 AND source NOT IN {COST_SOURCES_SQL}"
        ),
        [cutoff],
        |r| {
            Ok(Totals {
                events: r.get(0)?,
                before: r.get(1)?,
                after: r.get(2)?,
            })
        },
    )?;
    Ok(row)
}

/// Per-source breakdown since `since`, best savers first.
pub fn by_source(conn: &Connection, since: Option<i64>) -> Result<Vec<(String, Totals)>> {
    let cutoff = since.unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT source, COUNT(*), COALESCE(SUM(tokens_before),0), COALESCE(SUM(tokens_after),0)
         FROM savings_event WHERE at >= ?1
         GROUP BY source
         ORDER BY (SUM(tokens_before) - SUM(tokens_after)) DESC",
    )?;
    let rows = stmt
        .query_map([cutoff], |r| {
            Ok((
                r.get::<_, String>(0)?,
                Totals {
                    events: r.get(1)?,
                    before: r.get(2)?,
                    after: r.get(3)?,
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// Tokens saved per UTC day for the last `days` days, as (day_index, saved)
/// where day_index = at / 86400. Sparse: days with no events are omitted.
pub fn daily_saved(conn: &Connection, days: i64, now: i64) -> Result<Vec<(i64, i64)>> {
    let cutoff = now - days * 86_400;
    let mut stmt = conn.prepare(&format!(
        "SELECT at/86400 AS day, COALESCE(SUM(tokens_before - tokens_after),0)
         FROM savings_event WHERE at >= ?1 AND source NOT IN {COST_SOURCES_SQL}
         GROUP BY day ORDER BY day",
    ))?;
    let rows = stmt
        .query_map([cutoff], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[test]
    fn record_and_aggregate() {
        let conn = db::open_in_memory().unwrap();
        record(&conn, None, "outline", 1000, 80, serde_json::json!({})).unwrap();
        record(
            &conn,
            Some(1),
            "peek",
            500,
            40,
            serde_json::json!({"sym": "x"}),
        )
        .unwrap();
        record(&conn, Some(1), "outline", 200, 20, serde_json::json!({})).unwrap();

        let t = totals(&conn, None).unwrap();
        assert_eq!(t.events, 3);
        assert_eq!(t.before, 1700);
        assert_eq!(t.after, 140);
        assert_eq!(t.saved(), 1560);

        let by = by_source(&conn, None).unwrap();
        // outline saved 1000-80 + 200-20 = 1100; peek saved 460 → outline first.
        assert_eq!(by[0].0, "outline");
        assert_eq!(by[0].1.saved(), 1100);
        assert_eq!(by[1].0, "peek");
        assert_eq!(by[1].1.saved(), 460);
    }

    #[test]
    fn summary_matches_per_window_totals() {
        let conn = db::open_in_memory().unwrap();
        record(&conn, None, "outline", 1000, 100, serde_json::json!({})).unwrap();
        record(&conn, Some(1), "filter", 500, 50, serde_json::json!({})).unwrap();
        let now = crate::model::now_unix();
        let s = summary(&conn, now).unwrap();
        // both events are recent → counted in every window
        assert_eq!(s.day, 1350);
        assert_eq!(s.week, 1350);
        assert_eq!(s.month, 1350);
        assert_eq!(s.all.saved(), 1350);
        assert_eq!(s.all.events, 2);
    }

    #[test]
    fn to_dollars_scales() {
        assert!((to_dollars(1_000_000, 3.0) - 3.0).abs() < 1e-9);
        assert_eq!(to_dollars(0, 15.0), 0.0);
    }

    #[test]
    fn saved_never_negative() {
        let t = Totals {
            events: 1,
            before: 10,
            after: 25,
        };
        assert_eq!(t.saved(), 0);
    }

    /// A pure-cost source (brief) must never net against real savings in
    /// any "saved" aggregate — but stays visible in the per-source view.
    #[test]
    fn cost_sources_never_reduce_saved_aggregates() {
        let conn = db::open_in_memory().unwrap();
        let now = crate::model::now_unix();
        record(
            &conn,
            None,
            source::FILTER,
            1000,
            200,
            serde_json::json!({}),
        )
        .unwrap();
        record(&conn, None, source::BRIEF, 0, 86, serde_json::json!({})).unwrap();
        let s = summary(&conn, now).unwrap();
        assert_eq!(
            s.day, 800,
            "brief cost must not net against today's savings"
        );
        assert_eq!(s.all.saved(), 800);
        let t = totals(&conn, None).unwrap();
        assert_eq!(t.saved(), 800);
        let daily = daily_saved(&conn, 7, now).unwrap();
        assert_eq!(daily.iter().map(|(_, s)| s).sum::<i64>(), 800);
        let per_source = by_source(&conn, None).unwrap();
        let brief_row = per_source.iter().find(|(s, _)| s == source::BRIEF);
        assert!(
            brief_row.is_some_and(|(_, t)| t.after == 86 && t.saved() == 0),
            "brief stays visible per-source with its cost, clamped to 0 saved"
        );
    }

    /// The Spent rollup mirrors summary's windows over cost sources only:
    /// savings events never leak in, cost events all count.
    #[test]
    fn spent_rollup_counts_only_cost_sources() {
        let conn = db::open_in_memory().unwrap();
        let now = crate::model::now_unix();
        record(
            &conn,
            None,
            source::FILTER,
            1000,
            200,
            serde_json::json!({}),
        )
        .unwrap();
        record(&conn, None, source::BRIEF, 0, 86, serde_json::json!({})).unwrap();
        record(&conn, None, source::BRIEF, 0, 100, serde_json::json!({})).unwrap();
        let sp = spent(&conn, now).unwrap();
        assert_eq!(sp.day, 186, "both brief fires count, filter never does");
        assert_eq!(sp.week, 186);
        assert_eq!(sp.month, 186);
        assert_eq!(sp.all, 186);
        assert_eq!(sp.events, 2);

        // An empty ledger spends nothing (and errors nowhere).
        let empty = db::open_in_memory().unwrap();
        assert_eq!(spent(&empty, now).unwrap(), Spent::default());
    }
}
