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
//! Deliberately not on the MCP surface: no tool lists the queue and none
//! disposes of it, because the whole point of a review queue is that a
//! person looked. `mimir review` reads and decides it; `mimir doctor` says
//! how many findings are waiting.
//!
//! # The life of a finding
//!
//! ```text
//!   (detector) ── enqueue ──▶ Open       disposition IS NULL
//!   Open ── decide (a person) ─▶ Decided   keep | supersede | dismiss; final
//!   Open ── scan: the subject no longer stands ─▶ Resolved   "resolved"
//!   Resolved ── enqueue (detected again) ─▶ Open
//! ```
//!
//! *Open* is not just the column. A finding is about something — two
//! memories that contradict, a link that went stale — and the store moves on
//! without consulting the queue: one of the two is superseded, the symbol is
//! indexed again. [`resolved_outside`] is the one place that says whether a
//! finding's subject still stands, and everything that reads the queue goes
//! through it, so an item nobody can act on is neither listed nor counted
//! even before a scan has closed it.
//!
//! *Resolved* is the scan's closure, kept apart from a person's `dismiss`
//! because it promises less. A dismissal is a decision and is never
//! re-raised. A resolution is an observation, and if the condition comes
//! back, so does the finding.

use rusqlite::{params, Connection};

use crate::error::{Error, Result};
use crate::lifecycle::Lifecycle;
use crate::model::{now_unix, short_uid, Actor, MutationOp};

/// `review.disposition` for a finding that [`scan`] closed because its
/// subject no longer stands. Not a [`Disposition`]: nobody decided it.
const RESOLVED: &str = "resolved";

/// Prefix of the `decided_reason` a scan closure carries.
const RESOLVED_REASON: &str = "resolved outside review";

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
/// A row the scan closed as resolved is different: nobody decided that, so
/// detecting the condition again opens it again.
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
           at = excluded.at, detail = excluded.detail,
           disposition = NULL, decided_at = NULL, decided_reason = NULL
         WHERE review.disposition IS NULL OR review.disposition = ?6",
        params![
            now_unix(),
            finding.as_str(),
            node_id,
            other_id,
            detail,
            RESOLVED
        ],
    )?;
    Ok(())
}

/// A row with `disposition IS NULL`, before anyone has asked whether this
/// binary knows its kind or whether its subject still stands.
struct Undecided {
    id: i64,
    at: i64,
    kind: String,
    node_id: i64,
    other_id: Option<i64>,
    detail: Option<String>,
}

impl Undecided {
    /// The typed item, or `None` for a kind this binary does not know. The
    /// CLI and the MCP server are installed separately and upgrade at
    /// different times, so a newer build's finding can be sitting in the
    /// table; it is skipped rather than shown as something it is not.
    fn item(self) -> Option<Item> {
        Some(Item {
            id: self.id,
            at: self.at,
            finding: Finding::parse(&self.kind)?,
            node_id: self.node_id,
            other_id: self.other_id,
            detail: self.detail,
        })
    }
}

fn undecided(conn: &Connection, id: Option<i64>) -> Result<Vec<Undecided>> {
    let mut stmt = conn.prepare(
        "SELECT id, at, kind, node_id, other_id, detail FROM review
         WHERE disposition IS NULL AND (?1 IS NULL OR id = ?1)
         ORDER BY at ASC, id ASC",
    )?;
    let rows = stmt.query_map([id], |r| {
        Ok(Undecided {
            id: r.get("id")?,
            at: r.get("at")?,
            kind: r.get("kind")?,
            node_id: r.get("node_id")?,
            other_id: r.get("other_id")?,
            detail: r.get("detail")?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Why the memory `node_id` can no longer be the subject of a finding about
/// a current memory, or `None` if it still can: it is there, not a
/// tombstone, and not superseded. `also_expired` adds expiry, for the
/// findings that are about what is claimed *now*.
fn not_current(
    conn: &Connection,
    node_id: i64,
    also_expired: bool,
    now: i64,
) -> Result<Option<String>> {
    let node = match crate::store::get_node(conn, node_id) {
        Ok(node) => node,
        Err(Error::NotFound(_)) => return Ok(Some("the memory no longer exists".into())),
        Err(err) => return Err(err),
    };
    let state = node.lifecycle(now);
    let gone = match state {
        Lifecycle::Live => false,
        Lifecycle::Expired => also_expired,
        Lifecycle::Superseded | Lifecycle::Archived | Lifecycle::Forgotten => true,
    };
    Ok(gone.then(|| format!("{} is {state}", short_uid(node.kind, &node.uid))))
}

/// Whether what `item` is about has stopped being the case without anyone
/// deciding the item — and if so, what happened, for the record the scan
/// leaves. `None` means the finding still stands and is a person's to
/// decide.
///
/// Each finding has its own subject, and for two of them the subject is
/// *in* a non-current state by definition: an unresolved end-condition
/// belongs to an expired memory, a resurfacing fact to a forgotten one.
/// So this is the negation of what each detector looks for, not one blanket
/// "is the node live".
fn resolved_outside(conn: &Connection, item: &Item, now: i64) -> Result<Option<String>> {
    match item.finding {
        // Two memories both claimed as current truth. Once either is not —
        // gone, retired or expired — there is no contradiction left.
        Finding::Contradiction => {
            for id in std::iter::once(item.node_id).chain(item.other_id) {
                if let Some(why) = not_current(conn, id, true, now)? {
                    return Ok(Some(why));
                }
            }
            Ok(None)
        }
        Finding::StaleGrounding => {
            if let Some(why) = not_current(conn, item.node_id, false, now)? {
                return Ok(Some(why));
            }
            // Not stale any more can mean two different things, and the
            // closing note is read as the reason: the artifact came back, or
            // its row was removed outright and the link went with it.
            Ok(match crate::grounding::grounding(conn, item.node_id)? {
                crate::grounding::Grounding::Stale { .. } => None,
                crate::grounding::Grounding::Grounded { .. } => {
                    Some("what it links to is indexed again".to_string())
                }
                crate::grounding::Grounding::Ungrounded => Some(
                    "what it linked to was removed from the index, and the link with it"
                        .to_string(),
                ),
            })
        }
        Finding::Unresolved => not_current(conn, item.node_id, false, now),
        Finding::Resurfacing => match crate::store::get_node(conn, item.node_id) {
            Ok(node) if node.lifecycle(now) == Lifecycle::Forgotten => Ok(None),
            Ok(node) => Ok(Some(format!(
                "{} is no longer a forgotten memory",
                short_uid(node.kind, &node.uid)
            ))),
            Err(Error::NotFound(_)) => Ok(Some("the memory no longer exists".into())),
            Err(err) => Err(err),
        },
    }
}

/// Every finding a person can act on, oldest first: undecided, of a kind
/// this binary knows, and about something that still stands.
fn standing(conn: &Connection, limit: usize) -> Result<Vec<Item>> {
    let now = now_unix();
    let mut out = Vec::new();
    for row in undecided(conn, None)? {
        if out.len() >= limit {
            break;
        }
        let Some(item) = row.item() else { continue };
        if resolved_outside(conn, &item, now)?.is_none() {
            out.push(item);
        }
    }
    Ok(out)
}

/// Everything still waiting on a person, oldest first.
///
/// A finding whose subject no longer stands (see [`resolved_outside`]) is
/// not waiting on anyone and is left out, whether or not a [`scan`] has
/// closed it yet. Reading the queue writes nothing.
pub fn open(conn: &Connection, limit: usize) -> Result<Vec<Item>> {
    standing(conn, limit)
}

/// How many findings are waiting — for `doctor`, which wants the number
/// without the list. Always the length of what [`open`] would return.
pub fn open_count(conn: &Connection) -> Result<i64> {
    Ok(standing(conn, usize::MAX)?.len() as i64)
}

/// Cap on findings raised per detector per scan. A queue that lands two
/// thousand items in one night is one nobody opens, and the same detector
/// will raise the rest next time.
const SCAN_LIMIT: usize = 25;

/// Bring the queue up to date with the store: close what was resolved
/// outside review, then run the detectors that own no other write path and
/// queue what they find.
///
/// Contradictions are not detected here: `consolidate` finds those mid-pass,
/// with the embedding matrix already in hand, and enqueues them itself. This
/// covers the two conditions that were previously visible only if someone
/// happened to run the right command and read the output.
///
/// Returns how many findings are open afterwards.
pub fn scan(conn: &Connection) -> Result<i64> {
    // First, the findings the store has moved past. Closed with the reason
    // rather than deleted, so "why did this leave my queue?" has an answer.
    // No ledger row: nothing was done to the memory here — whatever resolved
    // it is already in the ledger under its own name.
    let now = now_unix();
    for row in undecided(conn, None)? {
        let Some(item) = row.item() else { continue };
        if let Some(why) = resolved_outside(conn, &item, now)? {
            conn.execute(
                "UPDATE review SET disposition = ?2, decided_at = ?3, decided_reason = ?4
                 WHERE id = ?1 AND disposition IS NULL",
                params![item.id, RESOLVED, now, format!("{RESOLVED_REASON}: {why}")],
            )?;
        }
    }

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
    // check. Not a superseded one: its replacement is what is claimed now,
    // and the condition went with the claim.
    let mut stmt = conn.prepare(&format!(
        "SELECT {} FROM node
         WHERE kind = 'memory' AND deleted_at IS NULL AND superseded_by IS NULL
           AND json_extract(meta, '$.resolves_when') IS NOT NULL
           AND {}
         ORDER BY created_at DESC LIMIT {SCAN_LIMIT}",
        crate::store::NODE_COLS,
        crate::lifecycle::expired_sql(now)
    ))?;
    let nodes: Vec<crate::model::Node> = stmt
        .query_map([], crate::store::row_to_node)?
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
/// `store::supersede`, so it is dated in the ledger like any other
/// supersession and carries the reviewer's reason.
///
/// The effect and the closing of the item are one [`crate::db::atomic`]
/// step: a memory is never retired by an item that then stays open, and an
/// item is never closed over an effect that did not happen.
///
/// A supersede decision is refused when either memory is no longer current.
/// The queue can be older than the store, and by then the choice the item
/// offered is gone: retiring the survivor in favour of a memory that was
/// itself deleted or replaced would hide both. `keep` and `dismiss` stay
/// available — they change nothing, and dismissing is how such an item is
/// cleared by hand before a scan does it.
pub fn decide(
    conn: &Connection,
    id: i64,
    disposition: Disposition,
    reason: Option<&str>,
) -> Result<Item> {
    crate::db::atomic(conn, || {
        let row = undecided(conn, Some(id))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::NotFound(format!("no open review item #{id}")))?;
        let kind = row.kind.clone();
        let item = row.item().ok_or_else(|| {
            Error::Invalid(format!(
                "review #{id} is a '{kind}' finding, which this version of mimir does not \
                 know; decide it with the version that raised it"
            ))
        })?;

        if disposition == Disposition::Supersede {
            let other = item.other_id.ok_or_else(|| {
                Error::Invalid(format!(
                    "review #{id} is a {} finding with nothing to supersede it with",
                    item.finding.as_str()
                ))
            })?;
            let now = now_unix();
            for node_id in [item.node_id, other] {
                if let Some(why) = not_current(conn, node_id, true, now)? {
                    return Err(Error::Invalid(format!(
                        "review #{id} is out of date: {why}, so there is no longer a choice \
                         between two current memories. Clear it with \
                         `mimir review {id} --dismiss`."
                    )));
                }
            }
            // Oldest loses, matching `consolidate`'s dedup rule, so a reviewer
            // never has to remember which id the detector put first.
            let (older, newer) = order_by_age(conn, item.node_id, other)?;
            crate::store::supersede(conn, older, newer, Actor::Human, reason)?;
        } else {
            // Keep and Dismiss change nothing about the memory, but the *decision*
            // is the thing worth recording: "a person looked at this on the 14th
            // and said both are right" is not the same as nobody having looked.
            crate::audit::try_record(
                conn,
                item.node_id,
                MutationOp::Review,
                Actor::Human,
                Some(&decision_note(disposition, item.finding, reason)),
                None,
                None,
            )?;
        }

        // Still open, checked again in the statement that closes it: two
        // people deciding the same item must not both succeed.
        let closed = conn.execute(
            "UPDATE review SET disposition = ?2, decided_at = ?3, decided_reason = ?4
             WHERE id = ?1 AND disposition IS NULL",
            params![id, disposition.as_str(), now_unix(), reason],
        )?;
        if closed == 0 {
            return Err(Error::NotFound(format!("no open review item #{id}")));
        }
        Ok(item)
    })
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
        // The link really is stale: a finding about a grounding that is fine
        // is not open at all (see `resolved_outside`).
        let sym = symbol(&c, "a_removed_function");
        store::link(&c, a, sym, crate::model::Rel::About, 1.0).unwrap();
        store::soft_delete(&c, sym, Actor::System, None).unwrap();
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

    fn break_inserts(conn: &Connection, table: &str) {
        conn.execute_batch(&format!(
            "CREATE TEMP TRIGGER break_{table} BEFORE INSERT ON main.{table}
             BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END;"
        ))
        .unwrap();
    }

    fn row(conn: &Connection, id: i64) -> (Option<String>, Option<String>) {
        conn.query_row(
            "SELECT disposition, decided_reason FROM review WHERE id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    }

    /// A supersede decision is the memory retired *and* the item closed. A
    /// failure partway used to leave the first without the second: the
    /// memory gone from recall, the finding still asking what to do.
    #[test]
    fn a_decision_lands_whole_or_not_at_all() {
        let c = conn();
        let a = memory(&c, "the cron runs at 03:00", 1000);
        let b = memory(&c, "the cron runs at 04:00", 2000);
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;
        break_inserts(&c, "edge");

        assert!(decide(&c, id, Disposition::Supersede, None).is_err());

        assert!(store::get_node(&c, a).unwrap().superseded_by.is_none());
        assert!(audit::history(&c, a).unwrap().is_empty());
        assert_eq!(open_count(&c).unwrap(), 1, "the item is still open");
    }

    /// Two people deciding one item must not both succeed. The closing
    /// UPDATE checks again that the item is open, and a decision that loses
    /// takes its ledger note back with it.
    #[test]
    fn a_decision_that_lost_the_race_is_refused_and_leaves_no_note() {
        let c = conn();
        let a = memory(&c, "both of these are true", 1000);
        let b = memory(&c, "and so is this", 2000);
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;
        // Stand-in for the other decision landing between this one's read
        // and its close.
        c.execute_batch(&format!(
            "CREATE TEMP TRIGGER other_decider AFTER INSERT ON main.mutation
             BEGIN UPDATE review SET disposition = 'keep' WHERE id = {id}; END;"
        ))
        .unwrap();

        let err = decide(&c, id, Disposition::Dismiss, Some("mine")).unwrap_err();

        assert!(matches!(err, Error::NotFound(_)), "{err:?}");
        assert!(
            audit::history(&c, a).unwrap().is_empty(),
            "the losing decision must not leave its note in the ledger"
        );
    }

    /// The queue can be older than the store. By the time a person decides,
    /// one side of a contradiction may already be gone, and retiring the
    /// survivor in its favour would hide both.
    #[test]
    fn superseding_an_item_the_store_has_moved_past_is_refused() {
        let c = conn();
        let a = memory(&c, "the cron runs at 03:00", 1000);
        let b = memory(&c, "the cron runs at 04:00", 2000);
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;
        // The newer one is forgotten outside review.
        store::soft_delete(&c, b, Actor::Human, None).unwrap();

        let err = decide(&c, id, Disposition::Supersede, None).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("out of date"), "{text}");
        assert!(text.contains("--dismiss"), "{text}");
        assert!(
            store::get_node(&c, a).unwrap().superseded_by.is_none(),
            "the surviving memory must not be retired in favour of a tombstone"
        );
        assert_eq!(
            row(&c, id).0,
            None,
            "a refused decision leaves it undecided"
        );

        // Dismissing it is what the error asks for, and it works.
        decide(&c, id, Disposition::Dismiss, Some("b was deleted")).unwrap();
        assert_eq!(row(&c, id).0.as_deref(), Some("dismiss"));
    }

    /// What a finding is about can stop being the case without anyone
    /// touching the queue. Such an item is not waiting on a person any more:
    /// it is neither listed nor counted, and the next scan closes it saying
    /// why.
    #[test]
    fn a_finding_resolved_outside_review_leaves_the_queue() {
        let c = conn();
        let a = memory(&c, "the cron runs at 03:00", 1000);
        let b = memory(&c, "the cron runs at 04:00", 2000);
        let other = memory(&c, "an unrelated contradiction, left", 1000);
        let side = memory(&c, "an unrelated contradiction, right", 2000);
        enqueue(&c, Finding::Contradiction, a, Some(b), None).unwrap();
        enqueue(&c, Finding::Contradiction, other, Some(side), None).unwrap();
        let id = open(&c, 10).unwrap()[0].id;
        assert_eq!(open_count(&c).unwrap(), 2);

        store::supersede(&c, a, b, Actor::Human, None).unwrap();

        let listed: Vec<i64> = open(&c, 10).unwrap().iter().map(|i| i.node_id).collect();
        assert_eq!(listed, vec![other], "before any scan");
        assert_eq!(open_count(&c).unwrap(), 1);
        assert_eq!(row(&c, id).0, None, "reading the queue writes nothing");

        assert_eq!(scan(&c).unwrap(), 1);
        let (disposition, reason) = row(&c, id);
        assert_eq!(disposition.as_deref(), Some("resolved"));
        let reason = reason.unwrap();
        assert!(reason.starts_with("resolved outside review"), "{reason}");
        assert!(reason.contains("superseded"), "{reason}");
    }

    fn symbol(conn: &Connection, name: &str) -> i64 {
        let mut n = NewNode::new(Kind::Symbol);
        n.title = Some(name.into());
        store::insert_node(conn, n).unwrap().id
    }

    /// A stale-grounding finding follows the grounding: indexed again, it
    /// closes; gone again, it comes back. Nobody decided the first closure,
    /// so it must not silence the second finding the way a dismissal does.
    #[test]
    fn a_stale_grounding_finding_follows_the_grounding() {
        let c = conn();
        let m = memory(&c, "prune_on_write must run before the flush", 1000);
        let sym = symbol(&c, "prune_on_write");
        store::link(&c, m, sym, crate::model::Rel::About, 1.0).unwrap();
        let set_deleted = |deleted: Option<i64>| {
            c.execute(
                "UPDATE node SET deleted_at = ?2 WHERE id = ?1",
                params![sym, deleted],
            )
            .unwrap();
        };

        set_deleted(Some(now_unix()));
        assert_eq!(scan(&c).unwrap(), 1);
        let id = open(&c, 10).unwrap()[0].id;

        set_deleted(None);
        assert_eq!(open_count(&c).unwrap(), 0, "grounded again, before a scan");
        assert_eq!(scan(&c).unwrap(), 0);
        assert_eq!(row(&c, id).0.as_deref(), Some("resolved"));

        set_deleted(Some(now_unix()));
        assert_eq!(scan(&c).unwrap(), 1, "stale again is a finding again");
        assert_eq!(row(&c, id), (None, None));

        // A person's dismissal, unlike the scan's closure, is final.
        decide(&c, id, Disposition::Dismiss, None).unwrap();
        assert_eq!(scan(&c).unwrap(), 0);
    }

    /// The subject of each finding is in the state that *makes* it a
    /// finding — an unresolved one is expired, a resurfacing one is a
    /// tombstone. Neither state may be mistaken for "resolved".
    #[test]
    fn a_finding_whose_subject_still_stands_stays_open() {
        let c = conn();
        let lapsed = memory(&c, "deluge needs the libtorrent 2.0 workaround", 1000);
        store::set_expiry(&c, lapsed, Some(2000), Some("when upstream ships 2.1")).unwrap();
        let gone = memory(&c, "the ops cron runs at 03:00 UTC", 1000);
        store::soft_delete(&c, gone, Actor::Human, None).unwrap();
        enqueue(&c, Finding::Resurfacing, gone, None, None).unwrap();

        assert_eq!(scan(&c).unwrap(), 2);
        assert_eq!(scan(&c).unwrap(), 2, "and a second scan closes neither");

        // Each closes when its own subject goes away.
        store::soft_delete(&c, lapsed, Actor::Human, None).unwrap();
        assert_eq!(open_count(&c).unwrap(), 1);
        store::hard_delete(&c, gone, Actor::Human, None).unwrap();
        assert_eq!(scan(&c).unwrap(), 0);
    }

    /// A superseded memory's unmet end-condition is nobody's problem: the
    /// replacement is what is claimed now.
    #[test]
    fn scan_does_not_raise_an_expired_condition_on_a_superseded_memory() {
        let c = conn();
        let old = memory(&c, "use the 2.0 workaround", 1000);
        let new = memory(&c, "2.1 shipped; no workaround needed", 2000);
        store::set_expiry(&c, old, Some(2000), Some("when upstream ships 2.1")).unwrap();
        store::supersede(&c, old, new, Actor::Human, None).unwrap();

        assert_eq!(scan(&c).unwrap(), 0);
        let rows: i64 = c
            .query_row("SELECT count(*) FROM review", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0, "nothing raised, so nothing to close either");
    }

    /// The CLI and the MCP server upgrade separately, so this binary can
    /// meet a finding written by a newer one. It is not this binary's to
    /// show, decide or close — and it is certainly not a contradiction.
    #[test]
    fn a_finding_of_an_unknown_kind_is_left_alone() {
        let c = conn();
        let a = memory(&c, "something a newer build noticed", 1000);
        c.execute(
            "INSERT INTO review (at, kind, node_id) VALUES (1, 'from-a-newer-build', ?1)",
            [a],
        )
        .unwrap();
        let id = c.last_insert_rowid();

        assert!(open(&c, 10).unwrap().is_empty());
        assert_eq!(open_count(&c).unwrap(), 0);
        let err = decide(&c, id, Disposition::Dismiss, None).unwrap_err();
        assert!(err.to_string().contains("from-a-newer-build"), "{err}");
        assert_eq!(scan(&c).unwrap(), 0);
        assert_eq!(
            row(&c, id),
            (None, None),
            "still open for a build that knows it"
        );
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
