//! `mimir review` — the findings the store will not decide for you.
//!
//! Lists what `consolidate`, `grounding` and the tombstone guard noticed, and
//! records what a person decided about each. The decision is the point: it is
//! what puts a human's own words into the mutation ledger, which otherwise
//! knows only that something was superseded and never why.

use anyhow::{bail, Result};
use mimir_core::model::short_uid;
use mimir_core::review::{self, Disposition};
use mimir_core::{format, store, Mimir};

/// List open findings, refreshing the detectors that own no other write path.
pub fn list(json: bool, limit: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    // Scan before listing: a queue that only fills when someone remembers to
    // run consolidate is one that reads as empty when it isn't.
    let _ = review::scan(&mimir.conn);
    let items = review::open(&mimir.conn, limit)?;
    if items.is_empty() {
        if !json {
            println!("nothing waiting on you");
        }
        return Ok(());
    }
    for item in &items {
        let node = store::resolve_ref_including_deleted(&mimir.conn, &item.node_id.to_string())
            .or_else(|_| store::get_node(&mimir.conn, item.node_id));
        let title = node
            .as_ref()
            .ok()
            .and_then(|n| n.title.clone())
            .unwrap_or_default();
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "id": item.id,
                    "at": item.at,
                    "finding": item.finding.as_str(),
                    "node_id": item.node_id,
                    "other_id": item.other_id,
                    "detail": item.detail,
                    "title": title,
                })
            );
            continue;
        }
        println!(
            "#{} {} [{}] {}",
            item.id,
            format::full_date(item.at),
            item.finding.as_str(),
            title
        );
        if let Some(other) = item.other_id {
            if let Ok(o) = store::get_node(&mimir.conn, other) {
                println!(
                    "      vs {} {}",
                    short_uid(o.kind, &o.uid),
                    o.title.as_deref().unwrap_or("")
                );
            }
        }
        if let Some(detail) = &item.detail {
            println!("      {detail}");
        }
    }
    if !json {
        println!(
            "\ndecide with: mimir review <id> --keep | --supersede | --dismiss [--reason ...]\n\
             nothing here resolves on its own — that is the point."
        );
    }
    Ok(())
}

/// Record a decision. Exactly one of the three flags.
pub fn decide(
    id: i64,
    keep: bool,
    supersede: bool,
    dismiss: bool,
    reason: Option<String>,
) -> Result<()> {
    let chosen: Vec<Disposition> = [
        (keep, Disposition::Keep),
        (supersede, Disposition::Supersede),
        (dismiss, Disposition::Dismiss),
    ]
    .into_iter()
    .filter_map(|(on, d)| on.then_some(d))
    .collect();
    let disposition = match chosen.as_slice() {
        [one] => *one,
        [] => bail!("pass one of --keep, --supersede or --dismiss"),
        _ => bail!("pass only one of --keep, --supersede or --dismiss"),
    };

    let mimir = Mimir::open()?;
    let item = review::decide(&mimir.conn, id, disposition, reason.as_deref())?;
    println!(
        "review #{id} {} ({})",
        disposition.as_str(),
        item.finding.as_str()
    );
    if disposition == Disposition::Supersede {
        println!("the older memory is retired; `mimir audit` records who and why.");
    }
    Ok(())
}

/// One line for `doctor`: how many findings are waiting. Silent at zero — a
/// health check that speaks when there is nothing to do is one people learn
/// to skim.
pub fn summary(mimir: &Mimir) -> Option<String> {
    let n = review::open_count(&mimir.conn).ok()?;
    (n > 0).then(|| {
        format!(
            "{n} finding{} waiting on you (`mimir review`)",
            if n == 1 { "" } else { "s" }
        )
    })
}
