//! `mimir savings` — token-savings analytics (Mimir's `rtk gain` equivalent).
//!
//! Totals the savings ledger across today / week / month / all-time, breaks it
//! down by source, and converts tokens into dollars via the configured input
//! price. `--oneline` prints a compact segment for a statusline.

use anyhow::Result;
use mimir_core::model::now_unix;
use mimir_core::savings::{self, to_dollars};
use mimir_core::Mimir;

pub fn savings(json: bool, oneline: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let price = mimir.config.savings.input_price_per_mtok;
    let s = savings::summary(&mimir.conn, now_unix())?;

    if oneline {
        // Compact statusline segment, e.g. "◇ 1.2M tok / $3.60 saved (today 45k)".
        println!(
            "◇ {} tok / {} saved (today {})",
            human(s.all.saved()),
            usd(s.all.saved(), price),
            human(s.day)
        );
        return Ok(());
    }

    let spent = savings::spent(&mimir.conn, now_unix())?;
    let by_source = savings::by_source(&mimir.conn, None)?;

    if json {
        let by: serde_json::Map<String, serde_json::Value> = by_source
            .into_iter()
            .map(|(src, t)| {
                (
                    src,
                    serde_json::json!({
                        "saved": t.saved(),
                        "events": t.events,
                        "saved_usd": to_dollars(t.saved(), price),
                    }),
                )
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "saved": { "day": s.day, "week": s.week, "month": s.month, "all": s.all.saved() },
                "saved_usd_all": to_dollars(s.all.saved(), price),
                "spent": {
                    "day": spent.day, "week": spent.week, "month": spent.month,
                    "all": spent.all, "events": spent.events,
                    "all_usd": to_dollars(spent.all, price),
                },
                "input_price_per_mtok": price,
                "events_all": s.all.events,
                "tokens_before_all": s.all.before,
                "tokens_after_all": s.all.after,
                "by_source": by,
            })
        );
        return Ok(());
    }

    println!("mimir token savings");
    for (label, saved) in [("today", s.day), ("week", s.week), ("month", s.month)] {
        println!(
            "  {label:<6} {:>8} tok  {}",
            human(saved),
            usd(saved, price)
        );
    }
    println!(
        "  {:<6} {:>8} tok  {}  ({} events)",
        "all",
        human(s.all.saved()),
        usd(s.all.saved(), price),
        s.all.events
    );
    if s.all.before > 0 {
        let pct = 100.0 * s.all.saved() as f64 / s.all.before as f64;
        println!("  ratio  {pct:.0}% of source tokens avoided  (input @ ${price}/MTok)");
    }
    // The other side of the ledger: context Mimir injected (the session
    // brief). Never netted against the savings above — shown so the cost
    // is a number, not a footnote. Silent while nothing has ever fired.
    if spent.events > 0 {
        println!(
            "  spent  {:>8} tok  {}  on injected context (today {}, {} fires) — not netted above",
            human(spent.all),
            usd(spent.all, price),
            human(spent.day),
            spent.events
        );
    }
    if by_source.is_empty() {
        println!(
            "\nno savings recorded yet — try `mimir outline <file>` or `mimir run -- cargo build`"
        );
    } else {
        println!("\nby source:");
        for (src, t) in &by_source {
            println!(
                "  {src:<12} {:>8} saved  {}  ({} events)",
                human(t.saved()),
                usd(t.saved(), price),
                t.events
            );
        }
    }
    Ok(())
}

fn usd(tokens: i64, price_per_mtok: f64) -> String {
    format!("${:.2}", to_dollars(tokens, price_per_mtok))
}

fn human(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}
