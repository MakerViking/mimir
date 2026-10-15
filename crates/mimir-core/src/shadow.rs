//! Shadow ranking: a candidate ranking signal run beside the live one,
//! logged, and scored against what agents went on to open — never shown.
//!
//! Every ranking change in this store has to clear the retrieval eval, and
//! the eval is a fixed, hand-labelled corpus: it can say a change does not
//! break the questions someone wrote down, not that it helps the questions
//! agents actually ask. Shadow mode is the other half. A candidate computes
//! its own order for each real recall page; when that order differs from the
//! live one, both are kept; when an agent then opens a hit from the page,
//! the two orders are scored on where they had put it.
//!
//! **The comparison is biased toward the live ranking, on purpose of
//! nobody's.** Agents open what they were shown near the top, and what they
//! were shown is the live order — so a candidate starts every page behind.
//! A shadow lead is therefore conservative evidence (it won despite the
//! bias); a shadow deficit is not evidence of harm. The report says so.
//!
//! **Nothing promotes itself.** A candidate that wins here goes live by a
//! human changing the ranking code, which then has to pass the retrieval
//! eval gate like any other ranking change. The shadow only produces the
//! evidence for that decision.
//!
//! The one candidate today is [`COACTIVATION`]: phase 2 of `learn`'s
//! co-activation edges. Phase 1 records which memories are opened together
//! and deliberately ranks nothing; this asks whether ranking on it would
//! have helped.

use std::collections::HashMap;

use rusqlite::{params, Connection};

use crate::error::Result;
use crate::learn::{self, COACTIVATION_WINDOW_SECS};
use crate::model::Kind;
use crate::search::Hit;

/// The co-activation candidate: a memory co-activated with something opened
/// in the current stretch of work is probably part of that work.
pub const COACTIVATION: &str = "coactivation";

/// How strongly a co-activation edge lifts a hit: `score · (1 + BOOST · w)`
/// with `w` the edge's effective weight, so a fully-formed edge (w = 1) is
/// worth half again the hit's score. A guess, and labelled as one — the
/// point of running it in shadow is to find out.
pub const BOOST: f64 = 0.5;

/// How long after a page an `opened` event still counts as a verdict on that
/// page. An agent reads a recall and opens from it within seconds; ten
/// minutes covers a slow read without crediting the page with opens that
/// came from a later, different question.
pub const VERDICT_WINDOW_SECS: i64 = 10 * 60;

/// Fewer decided pages than this and the report declines to call it.
pub const MIN_DECIDED: usize = 20;

/// Memories "in play" right now — partners of anything opened in the last
/// [`COACTIVATION_WINDOW_SECS`] — mapped to the strongest visible edge
/// weight that put them there.
pub fn primed(conn: &Connection, now: i64) -> Result<HashMap<i64, f64>> {
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT node_id FROM recall_event
         WHERE event = 'opened' AND at >= ?1",
    )?;
    let opened: Vec<i64> = stmt
        .query_map([now - COACTIVATION_WINDOW_SECS], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut primed: HashMap<i64, f64> = HashMap::new();
    for id in opened {
        for (partner, weight) in learn::coactivation_partners(conn, id, now)? {
            let slot = primed.entry(partner).or_default();
            *slot = slot.max(weight);
        }
    }
    Ok(primed)
}

/// The candidate's order for a live page: indices into `hits`, best first.
/// Only memories can be primed (the edges are memory↔memory); everything
/// else keeps its live score. Ties keep live order, so an unprimed page
/// comes back exactly as it went in.
pub fn coactivation_order(hits: &[Hit], primed: &HashMap<i64, f64>) -> Vec<usize> {
    let boosted: Vec<f64> = hits
        .iter()
        .map(|hit| {
            let w = if hit.node.kind == Kind::Memory {
                primed.get(&hit.node.id).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            hit.score * (1.0 + BOOST * w)
        })
        .collect();
    let mut order: Vec<usize> = (0..hits.len()).collect();
    order.sort_by(|&a, &b| boosted[b].total_cmp(&boosted[a]).then(a.cmp(&b)));
    order
}

/// Run the candidates on one live page and keep the result if it differs.
/// The single entry point both recall surfaces call, right where they log
/// `shown`. Best-effort like the other ledgers: a failure here is a warning,
/// never a failed recall.
pub fn observe(conn: &Connection, query_hash: &[u8], hits: &[Hit], now: i64) {
    if let Err(err) = observe_inner(conn, query_hash, hits, now) {
        tracing::warn!(%err, "shadow ranking not recorded");
    }
}

fn observe_inner(conn: &Connection, query_hash: &[u8], hits: &[Hit], now: i64) -> Result<()> {
    if hits.len() < 2 {
        return Ok(());
    }
    let primed = primed(conn, now)?;
    if primed.is_empty() {
        return Ok(());
    }
    let order = coactivation_order(hits, &primed);
    if order.iter().enumerate().all(|(rank, &idx)| rank == idx) {
        return Ok(());
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO shadow_rank (candidate, query_hash, at, node_id, live_rank, shadow_rank)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for (shadow_rank, &live_rank) in order.iter().enumerate() {
            stmt.execute(params![
                COACTIVATION,
                query_hash,
                now,
                hits[live_rank].node.id,
                live_rank as i64,
                shadow_rank as i64
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// How one candidate has done against the live ranking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Evaluation {
    pub candidate: String,
    /// Pages where the candidate's order differed from the live one.
    pub pages: usize,
    /// …of which an agent then opened at least one hit.
    pub decided: usize,
    /// Mean reciprocal rank of the first opened hit, live vs candidate.
    pub live_mrr: f64,
    pub shadow_mrr: f64,
    /// Decided pages where the candidate ranked the opened hit higher /
    /// lower / the same.
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
}

impl Evaluation {
    /// Two-sided exact sign-test p over wins vs losses (ties dropped). `None`
    /// with nothing decided either way.
    pub fn sign_test_p(&self) -> Option<f64> {
        sign_test(self.wins, self.losses)
    }

    /// One line a person can act on.
    pub fn verdict(&self) -> &'static str {
        if self.decided < MIN_DECIDED {
            return "not enough evidence yet";
        }
        match self.sign_test_p() {
            Some(p) if p < 0.05 && self.wins > self.losses => {
                "candidate AHEAD — worth a ranking change, then the eval gate"
            }
            Some(p) if p < 0.05 => {
                "candidate behind — not evidence of harm (the comparison favours live)"
            }
            _ => "no significant difference",
        }
    }
}

/// One recall page: its query hash and when it was shown.
type PageKey<'a> = (&'a [u8], i64);

/// Score `candidate` over pages since `since`.
///
/// An `opened` event is a verdict on exactly one page: the most recent one
/// that showed that node before the open (from the `shown` rows every recall
/// writes), and only if it came within [`VERDICT_WINDOW_SECS`]. A flat
/// "opened within ten minutes of the page" rule would credit one open to
/// every page that happened to list the node in that time — including the
/// ones the agent never read.
pub fn evaluate(conn: &Connection, candidate: &str, since: i64) -> Result<Evaluation> {
    struct Row {
        hash: Vec<u8>,
        at: i64,
        node: i64,
        live: i64,
        shadow: i64,
    }
    let mut stmt = conn.prepare(
        "SELECT query_hash, at, node_id, live_rank, shadow_rank FROM shadow_rank
         WHERE candidate = ?1 AND at >= ?2",
    )?;
    let rows: Vec<Row> = stmt
        .query_map(params![candidate, since], |r| {
            Ok(Row {
                hash: r.get(0)?,
                at: r.get(1)?,
                node: r.get(2)?,
                live: r.get(3)?,
                shadow: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;

    // Which page each open was a verdict on: (node, query_hash, shown_at).
    let mut verdicts: Vec<(i64, Vec<u8>, i64)> = Vec::new();
    let mut nodes: Vec<i64> = rows.iter().map(|r| r.node).collect();
    nodes.sort_unstable();
    nodes.dedup();
    let mut events = conn.prepare_cached(
        "SELECT event, query_hash, at FROM recall_event
         WHERE node_id = ?1 AND event IN ('shown', 'opened') AND at >= ?2
         ORDER BY at, id",
    )?;
    for node in nodes {
        let mut last_shown: Option<(Vec<u8>, i64)> = None;
        let mut events = events.query(params![node, since])?;
        while let Some(r) = events.next()? {
            let event: String = r.get(0)?;
            let at: i64 = r.get(2)?;
            if event == "shown" {
                last_shown = r.get::<_, Option<Vec<u8>>>(1)?.map(|hash| (hash, at));
            } else if let Some((hash, shown_at)) = &last_shown {
                if at - shown_at <= VERDICT_WINDOW_SECS {
                    verdicts.push((node, hash.clone(), *shown_at));
                }
            }
        }
    }

    // `shown` and the shadow row are stamped by separate clock reads in the
    // same recall, so they can straddle a second boundary.
    let opened_from = |row: &Row| {
        verdicts.iter().any(|(node, hash, at)| {
            *node == row.node && *hash == row.hash && (at - row.at).abs() <= 1
        })
    };
    // Per page: the best (live, shadow) rank among its opened hits, `None`
    // while nothing on it was opened.
    let mut pages: HashMap<PageKey, Option<(i64, i64)>> = HashMap::new();
    for row in &rows {
        let page = pages.entry((row.hash.as_slice(), row.at)).or_default();
        if opened_from(row) {
            *page = Some(match *page {
                Some((live, shadow)) => (live.min(row.live), shadow.min(row.shadow)),
                None => (row.live, row.shadow),
            });
        }
    }

    let mut eval = Evaluation {
        candidate: candidate.to_string(),
        pages: pages.len(),
        ..Evaluation::default()
    };
    let (mut live_sum, mut shadow_sum) = (0.0, 0.0);
    for (live, shadow) in pages.values().flatten() {
        eval.decided += 1;
        live_sum += 1.0 / (*live as f64 + 1.0);
        shadow_sum += 1.0 / (*shadow as f64 + 1.0);
        match shadow.cmp(live) {
            std::cmp::Ordering::Less => eval.wins += 1,
            std::cmp::Ordering::Greater => eval.losses += 1,
            std::cmp::Ordering::Equal => eval.ties += 1,
        }
    }
    if eval.decided > 0 {
        eval.live_mrr = live_sum / eval.decided as f64;
        eval.shadow_mrr = shadow_sum / eval.decided as f64;
    }
    Ok(eval)
}

/// Drop rows older than `cutoff`. Runs with the `recall_event` prune: a
/// shadow row is only ever judged against `opened` events, so it is useless
/// once those are gone.
pub fn prune(conn: &Connection, cutoff: i64) -> Result<usize> {
    Ok(conn.execute("DELETE FROM shadow_rank WHERE at < ?1", [cutoff])?)
}

/// Exact two-sided binomial sign test at p = 0.5, in log space so a few
/// thousand pages cannot underflow.
fn sign_test(wins: usize, losses: usize) -> Option<f64> {
    let n = wins + losses;
    if n == 0 {
        return None;
    }
    let k = wins.min(losses);
    let ln_half_n = n as f64 * 0.5f64.ln();
    let mut ln_choose = 0.0; // ln C(n, 0)
    let mut tail = 0.0;
    for i in 0..=k {
        if i > 0 {
            ln_choose += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        tail += (ln_choose + ln_half_n).exp();
    }
    Some((2.0 * tail).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::NewNode;
    use crate::store;

    fn memory(conn: &Connection, text: &str) -> crate::model::Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some(text.into());
        new.body = Some(text.into());
        store::insert_node(conn, new).unwrap()
    }

    fn page(nodes: &[&crate::model::Node]) -> Vec<Hit> {
        nodes
            .iter()
            .enumerate()
            .map(|(i, n)| Hit {
                node: (*n).clone(),
                // Close scores, so a full-strength edge can reorder them.
                score: 1.0 - 0.1 * i as f64,
            })
            .collect()
    }

    fn opened_at(conn: &Connection, node_id: i64, at: i64) {
        conn.execute(
            "INSERT INTO recall_event (node_id, event, at) VALUES (?1, 'opened', ?2)",
            params![node_id, at],
        )
        .unwrap();
    }

    fn co_edge(conn: &Connection, a: i64, b: i64, weight: f64, at: i64) {
        conn.execute(
            "INSERT INTO edge (src, dst, rel, weight, meta, created_at)
             VALUES (?1, ?2, 'co_recalled', ?3, json_object('last_fired_at', ?4), ?4)",
            params![a.min(b), a.max(b), weight, at],
        )
        .unwrap();
    }

    /// What a real recall does: log `shown` for the page, then run the
    /// shadow — with the synthetic clock, which `store::record_shown` (it
    /// stamps `now_unix()`) cannot take.
    fn show(conn: &Connection, hash: &[u8], hits: &[Hit], at: i64) {
        for (rank, hit) in hits.iter().enumerate() {
            conn.execute(
                "INSERT INTO recall_event (node_id, event, query_hash, rank, at)
                 VALUES (?1, 'shown', ?2, ?3, ?4)",
                params![hit.node.id, hash, rank as i64, at],
            )
            .unwrap();
        }
        observe(conn, hash, hits, at);
    }

    fn shadow_rows(conn: &Connection) -> i64 {
        conn.query_row("SELECT count(*) FROM shadow_rank", [], |r| r.get(0))
            .unwrap()
    }

    const T: i64 = 1_700_000_000;

    #[test]
    fn an_unprimed_page_keeps_live_order_and_writes_nothing() {
        let conn = db::open_in_memory().unwrap();
        let (a, b, c) = (memory(&conn, "a"), memory(&conn, "b"), memory(&conn, "c"));
        let hits = page(&[&a, &b, &c]);
        assert_eq!(coactivation_order(&hits, &HashMap::new()), vec![0, 1, 2]);
        observe(&conn, b"q", &hits, T);
        assert_eq!(shadow_rows(&conn), 0);
    }

    #[test]
    fn a_partner_of_something_just_opened_moves_up_in_shadow_only() {
        let conn = db::open_in_memory().unwrap();
        let (a, b, c, seed) = (
            memory(&conn, "a"),
            memory(&conn, "b"),
            memory(&conn, "c"),
            memory(&conn, "seed"),
        );
        co_edge(&conn, seed.id, c.id, 1.0, T);
        opened_at(&conn, seed.id, T - 60);
        let hits = page(&[&a, &b, &c]);

        let primed = primed(&conn, T).unwrap();
        assert_eq!(primed.get(&c.id).copied(), Some(1.0), "{primed:?}");
        // c: 0.8 · 1.5 = 1.2, ahead of a's 1.0.
        assert_eq!(coactivation_order(&hits, &primed), vec![2, 0, 1]);

        observe(&conn, b"q", &hits, T);
        let ranks: Vec<(i64, i64, i64)> = conn
            .prepare("SELECT node_id, live_rank, shadow_rank FROM shadow_rank ORDER BY shadow_rank")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(ranks, vec![(c.id, 2, 0), (a.id, 0, 1), (b.id, 1, 2)]);
    }

    #[test]
    fn a_stale_or_weak_edge_primes_nothing() {
        let conn = db::open_in_memory().unwrap();
        let (seed, weak, stale) = (memory(&conn, "s"), memory(&conn, "w"), memory(&conn, "o"));
        co_edge(&conn, seed.id, weak.id, 0.3, T);
        // Full weight, but five half-lives old: 1.0 · 2^-5 ≈ 0.03.
        co_edge(&conn, seed.id, stale.id, 1.0, T - 150 * 86_400);
        opened_at(&conn, seed.id, T - 60);
        assert!(primed(&conn, T).unwrap().is_empty());
        // And an open outside the working window primes nothing either.
        let conn = db::open_in_memory().unwrap();
        let (seed, partner) = (memory(&conn, "s"), memory(&conn, "p"));
        co_edge(&conn, seed.id, partner.id, 1.0, T);
        opened_at(&conn, seed.id, T - COACTIVATION_WINDOW_SECS - 1);
        assert!(primed(&conn, T).unwrap().is_empty());
    }

    #[test]
    fn each_open_is_a_verdict_on_the_page_it_came_from_only() {
        let conn = db::open_in_memory().unwrap();
        let (a, b, c, seed) = (
            memory(&conn, "a"),
            memory(&conn, "b"),
            memory(&conn, "c"),
            memory(&conn, "seed"),
        );
        co_edge(&conn, seed.id, c.id, 1.0, T);
        opened_at(&conn, seed.id, T - 60);
        let hits = page(&[&a, &b, &c]);

        // Page 1: the agent opens c — the hit shadow promoted (live 2,
        // shadow 0). A win.
        show(&conn, b"q1", &hits, T);
        opened_at(&conn, c.id, T + 30);
        // Page 2, a minute later: the agent opens a (live 0, shadow 1). A
        // loss — and it must not also count against page 1, which was still
        // inside its window but was not where this open came from.
        show(&conn, b"q2", &hits, T + 100);
        opened_at(&conn, a.id, T + 140);
        // Page 3: nothing opened. Undecided.
        show(&conn, b"q3", &hits, T + 200);

        let e = evaluate(&conn, COACTIVATION, 0).unwrap();
        assert_eq!((e.pages, e.decided), (3, 2), "{e:?}");
        assert_eq!((e.wins, e.losses, e.ties), (1, 1, 0), "{e:?}");
        assert!((e.live_mrr - (1.0 / 3.0 + 1.0) / 2.0).abs() < 1e-9, "{e:?}");
        assert!((e.shadow_mrr - (1.0 + 0.5) / 2.0).abs() < 1e-9, "{e:?}");
        assert_eq!(e.verdict(), "not enough evidence yet");
    }

    #[test]
    fn an_open_long_after_the_page_is_no_verdict() {
        let conn = db::open_in_memory().unwrap();
        let (a, b, c, seed) = (
            memory(&conn, "a"),
            memory(&conn, "b"),
            memory(&conn, "c"),
            memory(&conn, "seed"),
        );
        co_edge(&conn, seed.id, c.id, 1.0, T);
        opened_at(&conn, seed.id, T - 60);
        show(&conn, b"q", &page(&[&a, &b, &c]), T);
        opened_at(&conn, c.id, T + VERDICT_WINDOW_SECS + 1);
        let e = evaluate(&conn, COACTIVATION, 0).unwrap();
        assert_eq!((e.pages, e.decided), (1, 0));
        assert_eq!(prune(&conn, T + 1).unwrap(), 3);
    }

    #[test]
    fn sign_test_matches_known_values() {
        assert_eq!(sign_test(0, 0), None);
        // 10 vs 0: 2 · 0.5^10.
        assert!((sign_test(10, 0).unwrap() - 2.0 / 1024.0).abs() < 1e-12);
        // 5 vs 5 is as balanced as it gets.
        assert!((sign_test(5, 5).unwrap() - 1.0).abs() < 1e-12);
        // 15 vs 5, two-sided: 0.04139…
        assert!((sign_test(15, 5).unwrap() - 0.041_389_465).abs() < 1e-6);
        // Large n must not underflow to 0 or NaN.
        let p = sign_test(2_600, 2_400).unwrap();
        assert!(p > 0.0 && p < 0.01, "{p}");
    }

    #[test]
    fn verdict_needs_enough_pages_and_significance() {
        let mut e = Evaluation {
            decided: MIN_DECIDED,
            wins: 18,
            losses: 2,
            ..Evaluation::default()
        };
        assert!(e.verdict().starts_with("candidate AHEAD"));
        e.wins = 11;
        e.losses = 9;
        assert_eq!(e.verdict(), "no significant difference");
        e.decided = MIN_DECIDED - 1;
        assert_eq!(e.verdict(), "not enough evidence yet");
    }
}
