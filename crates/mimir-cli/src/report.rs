//! `mimir report` — activity across day/week/month/year/all-time, as a
//! compact text table agents can show inline (or `--json` for machines).

use anyhow::Result;
use mimir_core::model::{now_unix, short_uid, Kind};
use mimir_core::Mimir;

const PERIODS: [(&str, i64); 5] = [
    ("day", 86_400),
    ("week", 7 * 86_400),
    ("month", 30 * 86_400),
    ("year", 365 * 86_400),
    ("all", 0), // sentinel: no cutoff
];

pub fn report(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let conn = &mimir.conn;
    let now = now_unix();
    let cutoff = |secs: i64| if secs == 0 { 0 } else { now - secs };

    // One row of five period-counts for a timestamped table/filter.
    let counts = |sql: &str| -> Result<Vec<i64>> {
        let mut stmt = conn.prepare(sql)?;
        let row = stmt.query_row(
            rusqlite::params![
                cutoff(PERIODS[0].1),
                cutoff(PERIODS[1].1),
                cutoff(PERIODS[2].1),
                cutoff(PERIODS[3].1),
            ],
            |r| {
                Ok(vec![
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                ])
            },
        )?;
        Ok(row)
    };
    let node_counts = |kind: &str| -> Result<Vec<i64>> {
        counts(&format!(
            "SELECT COALESCE(sum(created_at>=?1),0), COALESCE(sum(created_at>=?2),0),
                    COALESCE(sum(created_at>=?3),0), COALESCE(sum(created_at>=?4),0), count(*)
             FROM node WHERE kind='{kind}' AND deleted_at IS NULL"
        ))
    };
    let event_counts = |event: &str| -> Result<Vec<i64>> {
        counts(&format!(
            "SELECT COALESCE(sum(at>=?1),0), COALESCE(sum(at>=?2),0), COALESCE(sum(at>=?3),0),
                    COALESCE(sum(at>=?4),0), count(*)
             FROM recall_event WHERE event='{event}'"
        ))
    };

    let rows: Vec<(&str, Vec<i64>)> = vec![
        ("memories captured", node_counts("memory")?),
        ("doc chunks indexed", node_counts("chunk")?),
        ("symbols extracted", node_counts("symbol")?),
        (
            "links created",
            counts(
                "SELECT COALESCE(sum(created_at>=?1),0), COALESCE(sum(created_at>=?2),0),
                        COALESCE(sum(created_at>=?3),0), COALESCE(sum(created_at>=?4),0),
                        count(*) FROM edge",
            )?,
        ),
        (
            "searches run",
            counts(
                "SELECT count(DISTINCT CASE WHEN at>=?1 THEN query_hash END),
                        count(DISTINCT CASE WHEN at>=?2 THEN query_hash END),
                        count(DISTINCT CASE WHEN at>=?3 THEN query_hash END),
                        count(DISTINCT CASE WHEN at>=?4 THEN query_hash END),
                        count(DISTINCT query_hash)
                 FROM recall_event WHERE query_hash IS NOT NULL",
            )?,
        ),
        ("results shown", event_counts("shown")?),
        ("memories opened", event_counts("opened")?),
        ("marked useful", event_counts("useful")?),
    ];

    // Most-recalled memories this month, for a human hook at the bottom.
    let top: Vec<(String, String, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT n.uid, COALESCE(n.title,'(untitled)'), count(*) c
             FROM recall_event e JOIN node n ON n.id = e.node_id
             WHERE e.at >= ?1 AND n.kind='memory' AND n.deleted_at IS NULL
             GROUP BY n.id ORDER BY c DESC LIMIT 3",
        )?;
        let mut rows = stmt.query([cutoff(30 * 86_400)])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            out.push((r.get(0)?, r.get(1)?, r.get(2)?));
        }
        out
    };
    let last_consolidate: Option<i64> = conn
        .query_row("SELECT value FROM meta WHERE key='last_consolidate'", [], |r| {
            r.get::<_, String>(0)
        })
        .ok()
        .and_then(|v| v.parse().ok());

    if json {
        let periods: Vec<&str> = PERIODS.iter().map(|(n, _)| *n).collect();
        let metrics: serde_json::Map<String, serde_json::Value> = rows
            .iter()
            .map(|(label, vals)| (label.to_string(), serde_json::json!(vals)))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "periods": periods,
                "metrics": metrics,
                "top_recalled_month": top.iter().map(|(uid, title, c)|
                    serde_json::json!({"uid": uid, "title": title, "count": c})).collect::<Vec<_>>(),
                "last_consolidate": last_consolidate,
            })
        );
        return Ok(());
    }

    let db_size = std::fs::metadata(&mimir.paths.db_file)
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "mimir report · {} · db {:.1} MB",
        date(now),
        db_size as f64 / 1_048_576.0
    );
    print!("{:<20}", "");
    for (name, _) in PERIODS {
        print!("{name:>7}");
    }
    println!();
    for (label, vals) in &rows {
        print!("{label:<20}");
        for v in vals {
            print!("{:>7}", human(*v));
        }
        println!();
    }
    if !top.is_empty() {
        let line = top
            .iter()
            .map(|(uid, title, c)| {
                format!("{} \"{}\" {c}×", short_uid(Kind::Memory, uid), clip(title, 40))
            })
            .collect::<Vec<_>>()
            .join(" · ");
        println!("top recalled (month)  {line}");
    }
    if let Some(ts) = last_consolidate {
        println!("last consolidation    {}", date(ts));
    }
    Ok(())
}

fn human(n: i64) -> String {
    if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Unix timestamp → YYYY-MM-DD (civil-from-days; no chrono dependency).
fn date(ts: i64) -> String {
    let days = ts.div_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_formats_known_timestamps() {
        assert_eq!(super::date(0), "1970-01-01");
        assert_eq!(super::date(1_781_246_133), "2026-06-12");
    }

    #[test]
    fn human_compacts_large_numbers() {
        assert_eq!(super::human(263), "263");
        assert_eq!(super::human(29_040), "29.0k");
    }
}
