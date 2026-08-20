//! The review queue: what the store noticed and deliberately will not decide.
//!
//! Mimir already detected several things it then did nothing with.
//! `consolidate` found contradicting memories and printed them into a report
//! that scrolled past. `grounding` knew which links had gone stale. A memory
//! could sit past its declared expiry with an unmet `resolves_when` forever.
//! Each of those is a judgement call — which of two contradicting facts is
//! right depends on things the store does not know — so resolving them
//! automatically would be guessing with a confident face, and the previous
//! behaviour of mentioning them once was the same as not detecting them.
//!
//! So: the machine says what it noticed, a person says what it means, and the
//! saying is recorded. A disposition writes the human's own words into
//! [`crate::audit`], which is the difference between "superseded on the 14th"
//! and "superseded on the 14th because the cron moved after the incident".
//!
//! Deliberately read-only over MCP. An agent can see the queue — it is useful
//! context — and must not dispose of it, because the whole point of a review
//! queue is that a person looked.

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{Error, Result};
use crate::model::{now_unix, Actor, MutationOp};

/// What kind of thing was noticed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finding {
    /// Two memories that appear to contradict each other.
    Contradiction,
    /// A memory whose grounding link points at something no longer indexed.
    StaleGrounding,
    /// Past its declared expiry, with a `resolves_when` nobody has confirmed.
    Unresolved,
    /// A deliberately forgotten fact that keeps being offered again.
    Resurfacing,
}

impl Finding {
    pub fn as_str(self) -> &'static str {
        match self {
            Finding::Contradiction => "contradiction",
            Finding::StaleGrounding => "stale-grounding",
            Finding::Unresolved => "unresolved",
            Finding::Resurfacing => "resurfacing",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "contradiction" => Finding::Contradiction,
            "stale-grounding" => Finding::StaleGrounding,
            "unresolved" => Finding::Unresolved,
            "resurfacing" => Finding::Resurfacing,
            _ => return None,
        })
    }
}

/// What a person decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Both are right, or the finding is not a problem. Nothing changes.
    Keep,
    /// The older one is retired in favour of the newer.
    Supersede,
    /// Not worth acting on, and not worth showing again.
    Dismiss,
}

impl Disposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Disposition::Keep => "keep",
            Disposition::Supersede => "supersede",
            Disposition::Dismiss => "dismiss",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    pub id: i64,
    pub at: i64,
    pub finding: Finding,
    pub node_id: i64,
    pub other_id: Option<i64>,
    pub detail: Option<String>,
}

/// Queue a finding, or refresh one already queued.
///
/// Idempotent on (kind, node_id, other_id): a nightly detector re-running must
/// not grow a second copy of yesterday's finding. A row already *decided* stays
/// decided — re-raising something a person dismissed would make dismissal
/// meaningless, which is how a queue becomes something people stop opening.
pub fn enqueue(
    conn: &Connection,
    finding: Finding,
    node_id: i64,
    other_id: Option<i64>,
    detail: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO review (at, kind, node_id, other_id, detail)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(kind, node_id, COALESCE(other_id, -1)) DO UPDATE SET
           at = excluded.at, detail = excluded.detail
         WHERE review.disposition IS NULL",
        params![now_unix(), finding.as_str(), node_id, other_id, detail],
    )?;
    Ok(())
}

/// Everything still waiting on a person, oldest first.
pub fn open(conn: &Connection, limit: usize) -> Result<Vec<Item>> {
    let mut stmt = conn.prepare(
        "SELECT id, at, kind, node_id, other_id, detail FROM review
         WHERE disposition IS NULL ORDER BY at ASC, id ASC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit as i64], |r| {
        Ok(Item {
            id: r.get("id")?,
            at: r.get("at")?,
            finding: Finding::parse(&r.get::<_, String>("kind")?).unwrap_or(Finding::Contradiction),
            node_id: r.get("node_id")?,
            other_id: r.get("other_id")?,
            detail: r.get("detail")?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// How many findings are waiting — for `doctor` and the session brief, which
/// want the number without the list.
pub fn open_count(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT count(*) FROM review WHERE disposition IS NULL",
        [],
        |r| r.get(0),
    )?)
}

/// Cap on findings raised per detector per scan. A queue that lands two
/// thousand items in one night is one nobody opens, and the same detector
/// will raise the rest next time.
const SCAN_LIMIT: usize = 25;

/// Run the detectors that own no other write path, and queue what they find.
///
/// Contradictions are not here: `consolidate` finds those mid-pass, with the
/// embedding matrix already in hand, and enqueues them itself. This covers
/// the two conditions that were previously visible only if someone happened
/// to run the right command and read the output.
///
/// Returns how many findings are open afterwards.
pub fn scan(conn: &Connection) -> Result<i64> {
    // Grounding that has been falsified: the memory pointed at a symbol or
    // file, and that artifact is gone. Not "wrong" — a note about a renamed
    // function is usually still good — but nobody has revisited it since the
    // ground moved, and that is precisely a thing for a person to look at.
    for (node, what) in crate::grounding::stale_memories(conn, SCAN_LIMIT)? {
        enqueue(conn, Finding::StaleGrounding, node.id, None, Some(&what))?;
    }

    // Past its declared expiry while carrying a `resolves_when` nobody can
    // evaluate. `expires_at` already hides it from recall, so without this it
    // would sit retired with its end-condition unexamined forever — the
    // author wrote down what would falsify it precisely so someone would
    // check.
    let now = now_unix();
    let mut stmt = conn.prepare(&format!(
        "SELECT {} FROM node
         WHERE kind = 'memory' AND deleted_at IS NULL
           AND json_extract(meta, '$.resolves_when') IS NOT NULL
           AND COALESCE(json_extract(meta, '$.expires_at'), 1e18) <= ?1
         ORDER BY created_at DESC LIMIT {SCAN_LIMIT}",
        crate::store::NODE_COLS
    ))?;
    let nodes: Vec<crate::model::Node> = stmt
        .query_map([now], crate::store::row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    for node in nodes {
        let detail = node.resolves_when().map(str::to_string);
        enqueue(conn, Finding::Unresolved, node.id, None, detail.as_deref())?;
    }

    open_count(conn)
}

/// Record what a person decided, and carry it out.
///
/// `Supersede` retires `node_id` in favour of `other_id`, which is the only
/// disposition that changes the store — and it goes through
/// `store::set_superseded`, so it is dated in the ledger like any other
/// supersession and carries the reviewer's reason.
pub fn decide(
    conn: &Connection,
    id: i64,
    disposition: Disposition,
    reason: Option<&str>,
) -> Result<Item> {
    let item = conn
        .query_row(
            "SELECT id, at, kind, node_id, other_id, detail FROM review
             WHERE id = ?1 AND disposition IS NULL",
            [id],
            |r| {
                Ok(Item {
                    id: r.get("id")?,
                    at: r.get("at")?,
                    finding: Finding::parse(&r.get::<_, String>("kind")?)
                        .unwrap_or(Finding::Contradiction),
                    node_id: r.get("node_id")?,
                    other_id: r.get("other_id")?,
                    detail: r.get("detail")?,
                })
            },
        )
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("no open review item #{id}")))?;

    if disposition == Disposition::Supersede {
        let other = item.other_id.ok_or_else(|| {
            Error::Invalid(format!(
                "review #{id} is a {} finding with nothing to supersede it with",
                item.finding.as_str()
            ))
        })?;
        // Oldest loses, matching `consolidate`'s dedup rule, so a reviewer
        // never has to remember which id the detector put first.
        let (older, newer) = order_by_age(conn, item.node_id, other)?;
        crate::store::set_superseded(conn, older, newer, Actor::Human, reason)?;
        crate::store::link(conn, newer, older, crate::model::Rel::Supersedes, 1.0)?;
    } else {
        // Keep and Dismiss change nothing about the memory, but the *decision*
        // is the thing worth recording: "a person looked at this on the 14th
        // and said both are right" is not the same as nobody having looked.
        crate::audit::record(
            conn,
            item.node_id,
            MutationOp::Review,
            Actor::Human,
            Some(&decision_note(disposition, item.finding, reason)),
            None,
            None,
        );
    }

    conn.execute(
        "UPDATE review SET disposition = ?2, decided_at = ?3, decided_reason = ?4
         WHERE id = ?1",
        params![id, disposition.as_str(), now_unix(), reason],
    )?;
    Ok(item)
}

fn decision_note(d: Disposition, f: Finding, reason: Option<&str>) -> String {
    let base = format!("review: {} — {}", f.as_str(), d.as_str());
    match reason {
        Some(r) => format!("{base} — {r}"),
        None => base,
    }
}

fn order_by_age(conn: &Connection, a: i64, b: i64) -> Result<(i64, i64)> {
    let created = |id: i64| -> Result<i64> {
        Ok(
            conn.query_row("SELECT created_at FROM node WHERE id = ?1", [id], |r| {
                r.get(0)
            })?,
        )
    };
    Ok(if created(a)? <= created(b)? {
        (a, b)
    } else {
        (b, a)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode};
    use crate::{audit, db, store};

    fn conn() -> Connection {
        db::open_in_memory().unwrap()
    }

    fn memory(conn: &Connection, body: &str, created_at: i64) -> i64 {
        let mut n = NewNode::new(Kind::Memory);
        n.title = Some(body.into());
        n.body = Some(body.into());
        n.content_hash = Some(crate::memory::content_hash(body));
        let id = store::insert_node(conn, n).unwrap().id;
        conn.execute(
            "UPDATE node SET created_at = ?2 WHERE id = ?1",
            params![id, created_at],
        )
        .unwrap();
        id
    }

    #[test]
    fn a_finding_waits_for_a_person_and_records_who_decided() {
        let c = conn();
        let a = memory(&c, "the cron runs at 03:00", 1000);
        let b = memory(&c, "the cron runs at 04:00", 2000);
        enqueue(
            &c,
            Finding::Contradiction,
            a,
            Some(b),
            Some("03:00 vs 04:00"),
        )
        .unwrap();

        assert_eq!(open_count(&c).unwrap(), 1);
        let items = open(&c, 10).unwrap();
        assert_eq!(items[0].finding, Finding::Contradiction);

        decide(
            &c,
            items[0].id,
            Disposition::Supersede,
            Some("schedule changed after the incident"),
        )
        .unwrap();

        assert_eq!(
            open_count(&c).unwrap(),
            0,
            "a decided item leaves the queue"
        );
        assert_eq!(
            store::get_node(&c, a).unwrap().superseded_by,
            Some(b),
            "the older memory must be the one retired"
        );
        let h = audit::history(&c, a).unwrap();
        assert_eq!(h.last().unwrap().op, MutationOp::Supersede);
        assert_eq!(h.last().unwrap().actor, Actor::Human);
        assert_eq!(
            h.last().unwrap().reason.as_deref(),
            Some("schedule changed after the incident"),
            "the reviewer's own words are what the ledger was missing"
        );
    }

    /// A nightly detector re-running must not grow a second copy of
    /// yesterday's finding, and must not re-raise what someone dismissed —
    /// either one turns the queue into something people stop opening.
    #[test]
    fn re_detecting_neither_duplicates_nor_reopens() {
        let c = conn();
        let a = memory(&c, "left", 1000);
        let b = memory(&c, "right", 2000);

        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        assert_eq!(open_count(&c).unwrap(), 1, "same finding, one row");

        let id = open(&c, 10).unwrap()[0].id;
        decide(&c, id, Disposition::Dismiss, Some("both are fine")).unwrap();
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        assert_eq!(
            open_count(&c).unwrap(),
            0,
            "a dismissed finding must stay dismissed"
        );
    }

    /// `keep` and `dismiss` change nothing about the memory — but that a
    /// person looked is itself the record worth having.
    #[test]
    fn keeping_records_the_decision_without_touching_the_memory() {
        let c = conn();
        let a = memory(&c, "both of these are true", 1000);
        let b = memory(&c, "and so is this", 2000);
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;

        decide(&c, id, Disposition::Keep, Some("different environments")).unwrap();

        assert!(
            store::get_node(&c, a).unwrap().superseded_by.is_none(),
            "keep must not retire anything"
        );
        let h = audit::history(&c, a).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].op, MutationOp::Review);
        assert!(h[0]
            .reason
            .as_deref()
            .unwrap()
            .contains("different environments"));
    }

    #[test]
    fn superseding_needs_something_to_supersede_with() {
        let c = conn();
        let a = memory(&c, "a memory whose link went stale", 1000);
        enqueue(&c, Finding::StaleGrounding, a, None, Some("symbol gone")).unwrap();
        let id = open(&c, 10).unwrap()[0].id;

        assert!(
            decide(&c, id, Disposition::Supersede, None).is_err(),
            "there is no replacement, so this must error rather than guess"
        );
        assert_eq!(
            open_count(&c).unwrap(),
            1,
            "a failed decision must leave the item open"
        );
    }

    #[test]
    fn deciding_twice_is_refused() {
        let c = conn();
        let a = memory(&c, "something", 1000);
        enqueue(&c, Finding::Unresolved, a, None, None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;
        decide(&c, id, Disposition::Dismiss, None).unwrap();
        assert!(decide(&c, id, Disposition::Keep, None).is_err());
    }

    /// An expired memory whose author wrote down what would falsify it, with
    /// nobody having checked. `expires_at` already hid it from recall, so
    /// without the queue the end-condition sits unexamined forever — which
    /// defeats the point of writing one down.
    #[test]
    fn scan_queues_an_expired_memory_with_an_unmet_condition() {
        let c = conn();
        let id = memory(&c, "deluge needs the libtorrent 2.0 workaround", 1000);
        crate::store::set_expiry(&c, id, Some(2000), Some("when upstream ships 2.1")).unwrap();

        scan(&c).unwrap();

        let items = open(&c, 10).unwrap();
        assert_eq!(items.len(), 1, "expected one finding: {items:?}");
        assert_eq!(items[0].finding, Finding::Unresolved);
        assert_eq!(items[0].detail.as_deref(), Some("when upstream ships 2.1"));
    }

    /// A live memory with an unmet condition is not a finding — it is an
    /// ordinary memory doing exactly what it should.
    #[test]
    fn scan_leaves_an_unexpired_condition_alone() {
        let c = conn();
        let id = memory(&c, "still true for now", 1000);
        crate::store::set_expiry(
            &c,
            id,
            Some(now_unix() + 86_400),
            Some("when the migration lands"),
        )
        .unwrap();

        scan(&c).unwrap();
        assert_eq!(open_count(&c).unwrap(), 0);
    }

    /// Scanning repeatedly is what a nightly `consolidate` does. The queue
    /// must not grow a row per run.
    #[test]
    fn scanning_repeatedly_is_idempotent() {
        let c = conn();
        let id = memory(&c, "expired with a condition", 1000);
        crate::store::set_expiry(&c, id, Some(2000), Some("someday")).unwrap();

        for _ in 0..3 {
            scan(&c).unwrap();
        }
        assert_eq!(open_count(&c).unwrap(), 1);
    }
}
