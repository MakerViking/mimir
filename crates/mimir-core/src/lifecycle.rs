//! The lifecycle of a node, named once.
//!
//! A memory is always in exactly one of five states, but no column says
//! which. The state is spread over `deleted_at`, `superseded_by`,
//! `meta.expires_at` and `meta.archived`, and every reader used to work it
//! out for itself. Recall got it right. The session brief forgot expiry,
//! guard anchors forgot both expiry and supersession, the handoff restore
//! forgot both too — and no surface ever *said* which state a node was in,
//! so a superseded memory brought back by `--include-superseded` printed
//! exactly like a live one.
//!
//! [`Lifecycle`] is the one derivation. A reader that needs "is this current
//! truth" asks [`Node::lifecycle`]; a surface that prints a node prints
//! [`Lifecycle::label`], which is `None` for the ordinary case and one word
//! for the four that change what a reader should do with the line.
//!
//! ```text
//!   Live | Expired
//!        ── store::supersede ──────────────▶ Superseded   (no way back)
//!   Live ── clock passes meta.expires_at ──▶ Expired
//!   Live | Superseded | Expired | Archived
//!        ── store::soft_delete (`forget`) ─▶ Forgotten    (prior wordings purged)
//!   Live | Expired
//!        ── consolidate's decay pass ──────▶ Archived     (prior wordings kept)
//!   Archived ── plain `remember` ──────────▶ a NEW Live node; the tombstone stays
//!   Forgotten ── `remember --force` ───────▶ a NEW Live node; the tombstone stays
//!   any ── store::hard_delete ─────────────▶ gone         (one ledger row remains)
//! ```
//!
//! Nothing restores a tombstone in place: coming back always means a new
//! node. Forgetting an archived memory is the one move between the two
//! tombstones, and it only goes that way.
//!
//! `pinned` is not a state. It is a guard on two of the arrows: a pinned
//! node is never superseded and never archived — and, the other way round,
//! only a memory that is live or expired can be pinned
//! (`store::set_pinned`). The one path that clears a tombstone in place is
//! a strictly newer write arriving over sync (`replicate::apply_changes`); a
//! tombstone arriving that way owes the same side effects as a local one,
//! and gets them from this same derivation.
//!
//! # What a `WHERE` clause can and cannot say
//!
//! Two SQL spellings are in use across the crate. They are two different
//! questions, not two attempts at one — and neither is "current":
//!
//! - `deleted_at IS NULL` — *stored*: not a tombstone. Used by export, sync
//!   and the counts of what is kept.
//! - `… AND superseded_by IS NULL` — stored and not replaced. Used by
//!   learning, consolidation and the tallies.
//!
//! What neither can see is **expiry**, which lives in `meta` on purpose (see
//! [`Node::expires_at`]: it replicates for free and needs no schema bump).
//! So anything that decides what to put in front of an agent *as current
//! truth* fetches its rows and asks [`Node::lifecycle`]; it does not trust a
//! `WHERE` clause to have said it. The test at the bottom of this file holds
//! every such surface to the same answer.
//!
//! Counting is the exception that has to be SQL — nobody loads a store to
//! say how big it is. [`counts`] is that one statement, and a test holds it
//! to [`Lifecycle::of`] node by node.

use rusqlite::Connection;

use crate::model::Node;
use crate::store;

/// Every state, for the code that has to treat each one.
const STATES: [Lifecycle; 5] = [
    Lifecycle::Live,
    Lifecycle::Expired,
    Lifecycle::Superseded,
    Lifecycle::Archived,
    Lifecycle::Forgotten,
];

/// Where a node is in its life. Exactly one holds at any instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lifecycle {
    /// Current truth. The only state recall, the session brief, guard
    /// anchors and the handoff restore may surface unasked.
    Live,
    /// Past its author-declared `meta.expires_at`. Kept as history; shown
    /// only when history is asked for (`--include-superseded`, `--as-of`).
    Expired,
    /// Replaced by a newer node (`superseded_by`). Kept as history, on the
    /// same terms as [`Lifecycle::Expired`].
    Superseded,
    /// Tombstoned by consolidation's decay pass (`deleted_at` plus
    /// `meta.archived`). Nobody decided this, which is why `remember` lets
    /// the same fact back in without `--force`.
    Archived,
    /// Tombstoned on purpose (`deleted_at` alone). `remember` refuses to
    /// re-learn it unless a human passes `--force`.
    Forgotten,
}

impl Lifecycle {
    /// Derive the state of `node` at `now` (unix seconds).
    ///
    /// When more than one condition holds, the most final one wins: a
    /// tombstone outranks retirement (a superseded memory that was then
    /// forgotten is forgotten), and supersession outranks expiry (it names
    /// the replacement, which is the more useful thing to tell a reader).
    pub fn of(node: &Node, now: i64) -> Lifecycle {
        if node.deleted_at.is_some() {
            return if is_archived(node) {
                Lifecycle::Archived
            } else {
                Lifecycle::Forgotten
            };
        }
        if node.superseded_by.is_some() {
            return Lifecycle::Superseded;
        }
        if node.is_expired(now) {
            return Lifecycle::Expired;
        }
        Lifecycle::Live
    }

    /// The state's stable name, for JSON and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::Live => "live",
            Lifecycle::Expired => "expired",
            Lifecycle::Superseded => "superseded",
            Lifecycle::Archived => "archived",
            Lifecycle::Forgotten => "forgotten",
        }
    }

    /// The word a compact line carries for this state, or `None` for
    /// [`Lifecycle::Live`].
    ///
    /// Same asymmetry as `unsure` and `weak` on that line: the ordinary case
    /// goes unsaid and costs nothing, and a word is spent only where acting
    /// on the line without noticing would be a mistake.
    pub fn label(self) -> Option<&'static str> {
        match self {
            Lifecycle::Live => None,
            other => Some(other.as_str()),
        }
    }

    /// Current truth: safe to surface unasked.
    pub fn is_current(self) -> bool {
        self == Lifecycle::Live
    }

    /// Kept as history, not current truth: superseded or expired. The same
    /// meaning `SearchQuery::include_superseded` documents.
    pub fn is_retired(self) -> bool {
        matches!(self, Lifecycle::Expired | Lifecycle::Superseded)
    }

    /// Soft-deleted: archived or forgotten.
    pub fn is_tombstoned(self) -> bool {
        matches!(self, Lifecycle::Archived | Lifecycle::Forgotten)
    }
}

impl std::fmt::Display for Lifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// SQL for "this tombstone is decay archival, not a deliberate forget", over
/// an unaliased `node` row. `consolidate` writes the integer 1; a JSON
/// `true` reads as 1 in SQL too.
pub(crate) const ARCHIVED_SQL: &str = "COALESCE(json_extract(meta, '$.archived'), 0) = 1";

/// Rust twin of [`ARCHIVED_SQL`]: whatever compares equal to 1 there counts
/// here — the integer, a JSON `true`, and a real 1.0.
fn is_archived(node: &Node) -> bool {
    meta_says_archived(&node.meta)
}

/// The archived flag as it reads in a `meta` object on its own. Replication
/// asks this of a record on the wire, before there is a node to derive a
/// [`Lifecycle`] from.
pub(crate) fn meta_says_archived(meta: &serde_json::Value) -> bool {
    meta.get("archived")
        .is_some_and(|v| v.as_f64() == Some(1.0) || v.as_bool() == Some(true))
}

impl Node {
    /// This node's [`Lifecycle`] at `now` (unix seconds).
    pub fn lifecycle(&self, now: i64) -> Lifecycle {
        Lifecycle::of(self, now)
    }
}

/// SQL twin of [`Node::is_expired`] at `now`, over an unaliased `node` row.
/// `Node::expires_at` takes any JSON number and rounds a fractional one up,
/// so both an integer and a real are an expiry here too; `x <= now` on the
/// raw value is the same test as `ceil(x) <= now` for a whole-second `now`.
/// Text, null and a missing key are not an expiry on either side.
pub(crate) fn expired_sql(now: i64) -> String {
    format!(
        "(json_type(meta, '$.expires_at') IN ('integer', 'real') \
         AND json_extract(meta, '$.expires_at') <= {now})"
    )
}

/// How many memories are in each [`Lifecycle`] state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LifecycleCounts {
    pub live: i64,
    pub superseded: i64,
    pub expired: i64,
    pub archived: i64,
    pub forgotten: i64,
}

impl LifecycleCounts {
    /// The count for one state.
    pub fn get(&self, state: Lifecycle) -> i64 {
        match state {
            Lifecycle::Live => self.live,
            Lifecycle::Superseded => self.superseded,
            Lifecycle::Expired => self.expired,
            Lifecycle::Archived => self.archived,
            Lifecycle::Forgotten => self.forgotten,
        }
    }

    fn slot(&mut self, state: Lifecycle) -> &mut i64 {
        match state {
            Lifecycle::Live => &mut self.live,
            Lifecycle::Superseded => &mut self.superseded,
            Lifecycle::Expired => &mut self.expired,
            Lifecycle::Archived => &mut self.archived,
            Lifecycle::Forgotten => &mut self.forgotten,
        }
    }

    /// Memories that are not tombstones: live, superseded and expired. What
    /// the store still holds, current or not.
    pub fn stored(&self) -> i64 {
        self.live + self.superseded + self.expired
    }

    /// Every memory row there is, tombstones included.
    pub fn total(&self) -> i64 {
        self.stored() + self.archived + self.forgotten
    }
}

/// Count every memory by [`Lifecycle`] state, at `now`, in one statement.
///
/// The one count of memories. "How many memories" used to have three
/// answers — not deleted, not deleted and not superseded, and what recall
/// would return — depending on which surface was asked, and none of them
/// knew about expiry. A surface that prints a number of memories reads the
/// field it means from here.
///
/// The `CASE` is [`Lifecycle::of`] line for line, with the same precedence:
/// a tombstone outranks supersession, which outranks expiry.
pub fn counts_at(conn: &Connection, now: i64) -> crate::error::Result<LifecycleCounts> {
    let mut stmt = conn.prepare(&format!(
        "SELECT state, count(*) FROM (
             SELECT CASE
                 WHEN deleted_at IS NOT NULL THEN
                     CASE WHEN {ARCHIVED_SQL} THEN '{archived}' ELSE '{forgotten}' END
                 WHEN superseded_by IS NOT NULL THEN '{superseded}'
                 WHEN {expired_sql} THEN '{expired}'
                 ELSE '{live}'
             END AS state
             FROM node WHERE kind = 'memory')
         GROUP BY state",
        archived = Lifecycle::Archived,
        forgotten = Lifecycle::Forgotten,
        superseded = Lifecycle::Superseded,
        expired = Lifecycle::Expired,
        live = Lifecycle::Live,
        expired_sql = expired_sql(now),
    ))?;
    let mut counts = LifecycleCounts::default();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    for row in rows {
        let (name, n) = row?;
        for state in STATES {
            if state.as_str() == name {
                *counts.slot(state) = n;
            }
        }
    }
    Ok(counts)
}

/// [`counts_at`] the present moment.
pub fn counts(conn: &Connection) -> crate::error::Result<LifecycleCounts> {
    counts_at(conn, crate::model::now_unix())
}

/// How far [`resolve_current`] follows a `superseded_by` chain. Chains are
/// one or two links in practice; the bound exists because nothing in the
/// schema forbids a cycle.
const MAX_SUPERSEDE_HOPS: usize = 8;

/// The live node that now stands where `node` stood, if anything does.
///
/// Some memories carry a *role* as well as a text: a guard anchored to a
/// file, a handoff waiting for the next session. Retiring the text must not
/// silently retire the role, and must not keep serving the retired text
/// either. So:
///
/// - live → the node itself;
/// - superseded → whatever its `superseded_by` chain ends in, if that is
///   live. The replacement inherits the role, which is what "supersede"
///   means, and is the only reading under which consolidation merging two
///   near-duplicates cannot quietly switch a guard off;
/// - expired, archived or forgotten → `None`. Its author, the clock or a
///   deletion said it should stop, and nothing was put in its place.
///
/// The caller still owns scope: a replacement can live in another project.
pub fn resolve_current(conn: &Connection, node: Node, now: i64) -> Option<Node> {
    let mut node = node;
    for _ in 0..=MAX_SUPERSEDE_HOPS {
        match node.lifecycle(now) {
            Lifecycle::Live => return Some(node),
            Lifecycle::Superseded => {
                node = store::get_node(conn, node.superseded_by?).ok()?;
            }
            Lifecycle::Expired | Lifecycle::Archived | Lifecycle::Forgotten => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{self, Remember, RememberOutcome};
    use crate::model::{now_unix, Actor, Kind, MemoryType, Scope};
    use crate::search::{self, SearchQuery};
    use crate::{anchors, brief, context_guard, db, store};
    use rusqlite::Connection;
    use std::collections::{HashMap, HashSet};

    fn remember(conn: &Connection, text: &str, tags: &[&str]) -> Node {
        match memory::remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Gotcha,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                project_id: None,
                force: true,
                actor: Actor::System,
            },
        )
        .unwrap()
        {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        }
    }

    /// Move a freshly stored memory into `state` through the same writes
    /// production uses, so the fixture cannot drift from the real columns.
    fn put_in_state(conn: &Connection, node: &Node, state: Lifecycle) {
        match state {
            Lifecycle::Live => {}
            Lifecycle::Expired => {
                store::set_expiry(conn, node.id, Some(now_unix() - 60), None).unwrap();
            }
            Lifecycle::Superseded => {
                let newer = remember(conn, &format!("replacement for node {}", node.id), &[]);
                store::set_superseded(conn, node.id, newer.id, Actor::System, None).unwrap();
            }
            Lifecycle::Archived => {
                // The statement `consolidate`'s decay pass runs.
                conn.execute(
                    "UPDATE node SET deleted_at = ?2,
                            meta = json_set(meta, '$.archived', 1)
                     WHERE id = ?1",
                    rusqlite::params![node.id, now_unix()],
                )
                .unwrap();
            }
            Lifecycle::Forgotten => {
                store::soft_delete(conn, node.id, Actor::Human, None).unwrap();
            }
        }
    }

    /// One gotcha per state, each anchored to its own file, each sharing the
    /// word "quorum" so one query finds all.
    fn one_of_each(conn: &Connection) -> HashMap<Lifecycle, Node> {
        let mut out = HashMap::new();
        for state in STATES {
            let node = remember(
                conn,
                &format!("quorum loss on the {state} replica needs a manual failover"),
                &[],
            );
            anchors::set_anchors(conn, node.id, &[format!("{state}.rs")]).unwrap();
            put_in_state(conn, &node, state);
            out.insert(state, store::get_node(conn, node.id).unwrap());
        }
        out
    }

    /// A `session-handoff` memory written `age` seconds ago, so "newest" is
    /// decided by the fixture and not by rows sharing one second.
    fn handoff(conn: &Connection, text: &str, age: i64) -> Node {
        let node = remember(conn, text, &["session-handoff"]);
        conn.execute(
            "UPDATE node SET created_at = created_at - ?2 WHERE id = ?1",
            [node.id, age],
        )
        .unwrap();
        node
    }

    #[test]
    fn each_state_is_derived_from_its_columns() {
        let conn = db::open_in_memory().unwrap();
        let nodes = one_of_each(&conn);
        let now = now_unix();
        for state in STATES {
            assert_eq!(nodes[&state].lifecycle(now), state);
        }
        // An expiry in the future is not expiry.
        let later = remember(&conn, "the cert rotates next quarter", &[]);
        store::set_expiry(&conn, later.id, Some(now + 86_400), None).unwrap();
        let later = store::get_node(&conn, later.id).unwrap();
        assert_eq!(later.lifecycle(now), Lifecycle::Live);
        assert_eq!(later.lifecycle(now + 86_400), Lifecycle::Expired);
    }

    #[test]
    fn the_most_final_state_wins() {
        let conn = db::open_in_memory().unwrap();
        let now = now_unix();

        // Superseded and expired: say superseded, it names the replacement.
        let both = remember(&conn, "the old pager rota", &[]);
        put_in_state(&conn, &both, Lifecycle::Expired);
        put_in_state(&conn, &both, Lifecycle::Superseded);
        let both = store::get_node(&conn, both.id).unwrap();
        assert_eq!(both.lifecycle(now), Lifecycle::Superseded);

        // Then forgotten: a tombstone outranks retirement.
        put_in_state(&conn, &both, Lifecycle::Forgotten);
        let gone = store::get_node(&conn, both.id).unwrap();
        assert_eq!(gone.lifecycle(now), Lifecycle::Forgotten);
    }

    #[test]
    fn predicates_partition_the_states() {
        for state in STATES {
            let flags = [
                state.is_current(),
                state.is_retired(),
                state.is_tombstoned(),
            ];
            assert_eq!(
                flags.iter().filter(|f| **f).count(),
                1,
                "{state} must be exactly one of current / retired / tombstoned"
            );
            assert_eq!(state.label().is_none(), state.is_current());
        }
    }

    /// `counts` is SQL and `Lifecycle::of` is Rust; this is what keeps the
    /// one from drifting off the other. Every memory row is classified both
    /// ways, including the combinations where more than one condition holds
    /// and the meta values a peer or an older build could have written.
    #[test]
    fn counts_agree_with_the_derivation_node_by_node() {
        let conn = db::open_in_memory().unwrap();
        let now = now_unix();
        one_of_each(&conn);
        // A second live memory, so no two states share a count by accident.
        remember(&conn, "an ordinary current memory", &[]);

        // Superseded and expired; then one of those forgotten as well.
        for forget in [false, true] {
            let node = remember(&conn, &format!("retired twice over {forget}"), &[]);
            put_in_state(&conn, &node, Lifecycle::Expired);
            put_in_state(&conn, &node, Lifecycle::Superseded);
            if forget {
                put_in_state(&conn, &node, Lifecycle::Forgotten);
            }
        }
        // Expired, then archived by decay.
        let node = remember(&conn, "expired and then decayed", &[]);
        put_in_state(&conn, &node, Lifecycle::Expired);
        put_in_state(&conn, &node, Lifecycle::Archived);

        // Meta as other writers may have left it.
        let raw_meta = |text: &str, meta: &str, deleted: bool| {
            let node = remember(&conn, text, &[]);
            conn.execute(
                "UPDATE node SET meta = ?2, deleted_at = ?3 WHERE id = ?1",
                rusqlite::params![node.id, meta, deleted.then_some(now)],
            )
            .unwrap();
        };
        raw_meta("archived as a json true", r#"{"archived":true}"#, true);
        raw_meta("archived as a real", r#"{"archived":1.0}"#, true);
        raw_meta(
            "archived as a string is not archived",
            r#"{"archived":"1"}"#,
            true,
        );
        raw_meta("archived zero is forgotten", r#"{"archived":0}"#, true);
        raw_meta(
            "archived flag without a tombstone",
            r#"{"archived":1}"#,
            false,
        );
        raw_meta(
            "a real expiry is not an expiry",
            r#"{"expires_at":1.5}"#,
            false,
        );
        raw_meta(
            "a text expiry is not an expiry",
            r#"{"expires_at":"1"}"#,
            false,
        );
        raw_meta(
            "a null expiry is not an expiry",
            r#"{"expires_at":null}"#,
            false,
        );
        raw_meta(
            "an expiry at this very second",
            &format!(r#"{{"expires_at":{now}}}"#),
            false,
        );
        raw_meta(
            "an expiry one second out",
            &format!(r#"{{"expires_at":{}}}"#, now + 1),
            false,
        );

        // Not memories: never counted, whatever their columns say.
        store::ensure_project(&conn, "/abs/p", "p").unwrap();
        let mut chunk = crate::model::NewNode::new(Kind::Chunk);
        chunk.title = Some("a doc chunk".into());
        let chunk = store::insert_node(&conn, chunk).unwrap();
        store::soft_delete(&conn, chunk.id, Actor::System, None).unwrap();

        let mut want = LifecycleCounts::default();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {} FROM node WHERE kind = 'memory'",
                store::NODE_COLS
            ))
            .unwrap();
        let nodes: Vec<Node> = stmt
            .query_map([], store::row_to_node)
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        for node in &nodes {
            *want.slot(node.lifecycle(now)) += 1;
        }

        let got = counts_at(&conn, now).unwrap();
        assert_eq!(got, want);
        for state in STATES {
            assert!(got.get(state) > 0, "the fixture must cover {state}");
        }
        assert_eq!(got.total(), nodes.len() as i64);
        assert_eq!(
            got.stored(),
            nodes.iter().filter(|n| n.deleted_at.is_none()).count() as i64
        );
        // The clock moving is enough to change the answer.
        assert_eq!(counts_at(&conn, now + 1).unwrap().expired, got.expired + 1);
    }

    fn recall_ids(conn: &Connection, include_superseded: bool) -> HashSet<i64> {
        let query = SearchQuery {
            text: "quorum".into(),
            scope: Scope::All,
            kinds: vec![Kind::Memory],
            since: None,
            limit: 20,
            strength_alpha: 0.0,
            recency_alpha: 0.0,
            type_prior_alpha: 0.0,
            code_damp: 1.0,
            include_superseded,
            as_of: None,
        };
        search::search(conn, &query)
            .unwrap()
            .into_iter()
            .map(|h| h.node.id)
            .collect()
    }

    /// The point of the type. Every surface that hands an agent something
    /// *as current truth* must agree on which states that is — and before
    /// this module, three of the four below did not agree with recall.
    #[test]
    fn every_surface_that_means_current_agrees() {
        let conn = db::open_in_memory().unwrap();
        let nodes = one_of_each(&conn);
        let now = now_unix();
        let id_of = |state: Lifecycle| nodes[&state].id;
        let only_live = HashSet::from([id_of(Lifecycle::Live)]);

        // Recall: live by default; history on request; a tombstone never.
        let default: HashSet<i64> = recall_ids(&conn, false)
            .intersection(&nodes.values().map(|n| n.id).collect())
            .copied()
            .collect();
        assert_eq!(default, only_live, "recall");
        let with_history = recall_ids(&conn, true);
        for state in STATES {
            assert_eq!(
                with_history.contains(&id_of(state)),
                !state.is_tombstoned(),
                "recall --include-superseded, {state}"
            );
        }

        // Session brief.
        let briefed: HashSet<i64> = brief::select(
            &conn,
            Scope::Global,
            &HashSet::new(),
            None,
            &crate::config::Config::default(),
            now,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.node.id)
        .filter(|id| nodes.values().any(|n| n.id == *id))
        .collect();
        assert_eq!(briefed, only_live, "session brief");

        // Guard anchors: each state is anchored to its own file. A live
        // guard speaks for itself, a superseded one through its replacement
        // (see `resolve_current`), and the rest stay silent — what must never
        // reach the agent is the retired text.
        for state in STATES {
            let fired = anchors::find_and_record(
                &conn,
                "session-1",
                Scope::Global,
                &format!("/repo/src/{state}.rs"),
                false,
            )
            .unwrap();
            let own_text = format!("quorum loss on the {state} replica");
            match state {
                Lifecycle::Live => {
                    assert!(fired.unwrap().contains(&own_text), "guard anchor, live")
                }
                Lifecycle::Superseded => {
                    let fired = fired.expect("a superseded guard keeps guarding");
                    assert!(fired.contains("replacement for node"), "{fired}");
                    assert!(!fired.contains(&own_text), "retired text leaked: {fired}");
                }
                _ => assert!(fired.is_none(), "guard anchor, {state}: {fired:?}"),
            }
        }
    }

    #[test]
    fn resolve_current_follows_replacements_and_stops_at_an_ending() {
        let conn = db::open_in_memory().unwrap();
        let now = now_unix();
        let get = |id: i64| store::get_node(&conn, id).unwrap();
        let resolve = |id: i64| resolve_current(&conn, get(id), now).map(|n| n.id);

        // a → b → c, each superseded by the next; c is live.
        let a = remember(&conn, "first wording of the rule", &[]);
        let b = remember(&conn, "second wording of the rule", &[]);
        let c = remember(&conn, "third wording of the rule", &[]);
        store::set_superseded(&conn, a.id, b.id, Actor::System, None).unwrap();
        store::set_superseded(&conn, b.id, c.id, Actor::System, None).unwrap();
        assert_eq!(resolve(c.id), Some(c.id), "live resolves to itself");
        assert_eq!(
            resolve(a.id),
            Some(c.id),
            "the chain is followed to its end"
        );

        // The replacement expiring ends the role for the whole chain.
        store::set_expiry(&conn, c.id, Some(now - 60), None).unwrap();
        assert_eq!(resolve(a.id), None);
        assert_eq!(resolve(c.id), None);

        // Nothing in the schema forbids a cycle; it must end, not spin.
        let x = remember(&conn, "ping", &[]);
        let y = remember(&conn, "pong", &[]);
        store::set_superseded(&conn, x.id, y.id, Actor::System, None).unwrap();
        store::set_superseded(&conn, y.id, x.id, Actor::System, None).unwrap();
        assert_eq!(resolve(x.id), None);
    }

    /// A handoff is a snapshot for the *next* session. If the newest one was
    /// retired, an older one is stale by construction — restoring it would be
    /// worse than restoring nothing.
    #[test]
    fn handoff_restore_never_falls_back_to_an_older_snapshot() {
        let conn = db::open_in_memory().unwrap();
        let restored = || {
            context_guard::current_handoff(&conn, Scope::Global)
                .unwrap()
                .map(|n| n.id)
        };
        let older = handoff(&conn, "handoff: finish the migration", 300);
        assert_eq!(restored(), Some(older.id));

        // The newest expires: nothing, not the older snapshot.
        let newest = handoff(&conn, "handoff: ship the release notes", 100);
        assert_eq!(restored(), Some(newest.id));
        put_in_state(&conn, &newest, Lifecycle::Expired);
        assert_eq!(restored(), None, "must not fall back to a stale handoff");
    }

    /// A handoff that was superseded hands its role to the replacement —
    /// consolidation merging two near-identical handoffs must not lose it.
    #[test]
    fn handoff_restore_follows_a_superseded_handoff_to_its_replacement() {
        let conn = db::open_in_memory().unwrap();
        let written = handoff(&conn, "handoff: fix the flaky sync test next", 50);
        let merged = remember(&conn, "handoff: fix the flaky sync test first", &[]);
        store::set_superseded(&conn, written.id, merged.id, Actor::System, None).unwrap();
        let restored = context_guard::current_handoff(&conn, Scope::Global)
            .unwrap()
            .expect("the replacement");
        assert_eq!(restored.id, merged.id);
        assert!(context_guard::restore_handoff_text(&conn, Scope::Global)
            .unwrap()
            .unwrap()
            .contains("fix the flaky sync test first"));
    }
}
