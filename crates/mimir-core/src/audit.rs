//! The mutation ledger: who changed a memory, when, and why.
//!
//! Everything else in the store describes what a memory *says*. This records
//! what was *done* to it. Before it existed, `soft_delete` and
//! `set_superseded` updated a column in place and left nothing behind, so a
//! correction was structural but anonymous: you could see that a memory had
//! been retired and never who retired it, when, or on whose say-so.
//!
//! Two rules hold the design together.
//!
//! **No values.** Only `before_hash`/`after_hash`, blake3 over normalized text
//! via [`crate::memory::content_hash`]. An audit that quotes what it audits is
//! a second copy of a deleted fact sitting in a table that recall never reads
//! but `export` and every backup do. The hashes still prove *which* value was
//! removed to anyone who holds a candidate, and recover nothing for anyone who
//! doesn't. The refusal ledger made this call first, for the same reason.
//!
//! **Memories only.** Indexers soft-delete thousands of chunk and symbol rows
//! on every reindex. Recording those would be honest and useless: the handful
//! of rows a person would ever want to read would be buried under machine
//! churn, and a log nobody can read is not an audit.

use rusqlite::{params, Connection};

use crate::error::Result;
use crate::model::{now_unix, Actor, Kind, MutationOp};

/// One recorded change.
#[derive(Debug, Clone)]
pub struct Mutation {
    pub at: i64,
    pub node_id: i64,
    pub op: MutationOp,
    pub actor: Actor,
    pub reason: Option<String>,
    pub before_hash: Option<Vec<u8>>,
    pub after_hash: Option<Vec<u8>>,
}

/// Append a mutation row — but only for memories (see the module note).
///
/// Deliberately infallible from the caller's point of view: a mutation that
/// succeeded must not be reported as failed because the ledger write hiccuped,
/// and the store's other per-operation ledgers (`recall_event`,
/// `injection_log`) already fail open for the same reason. A dropped row is
/// visible as a gap; a spurious error would be a lie about what happened to
/// the memory.
pub fn record(
    conn: &Connection,
    node_id: i64,
    op: MutationOp,
    actor: Actor,
    reason: Option<&str>,
    before_hash: Option<&[u8]>,
    after_hash: Option<&[u8]>,
) {
    let is_memory: rusqlite::Result<String> =
        conn.query_row("SELECT kind FROM node WHERE id = ?1", [node_id], |r| {
            r.get(0)
        });
    match is_memory {
        Ok(kind) if kind == Kind::Memory.as_str() => {}
        // A hard delete removes the row before we can read its kind. The
        // caller passes the kind it saw; see `record_for_kind`.
        _ => return,
    }
    record_for_kind(
        conn,
        Kind::Memory,
        node_id,
        op,
        actor,
        reason,
        before_hash,
        after_hash,
    );
}

/// As [`record`], for callers that already know the kind — notably
/// `hard_delete`, where the row is gone by the time anyone could look it up.
#[allow(clippy::too_many_arguments)]
pub fn record_for_kind(
    conn: &Connection,
    kind: Kind,
    node_id: i64,
    op: MutationOp,
    actor: Actor,
    reason: Option<&str>,
    before_hash: Option<&[u8]>,
    after_hash: Option<&[u8]>,
) {
    if kind != Kind::Memory {
        return;
    }
    let _ = conn.execute(
        "INSERT INTO mutation (at, node_id, op, actor, reason, before_hash, after_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            now_unix(),
            node_id,
            op.as_str(),
            actor.as_str(),
            reason,
            before_hash,
            after_hash
        ],
    );
}

fn row_to_mutation(row: &rusqlite::Row) -> rusqlite::Result<Mutation> {
    Ok(Mutation {
        at: row.get("at")?,
        node_id: row.get("node_id")?,
        op: row
            .get::<_, String>("op")?
            .parse()
            .unwrap_or(MutationOp::Edit),
        actor: row
            .get::<_, String>("actor")?
            .parse()
            .unwrap_or(Actor::System),
        reason: row.get("reason")?,
        before_hash: row.get("before_hash")?,
        after_hash: row.get("after_hash")?,
    })
}

/// Every recorded change to one memory, oldest first — its life story.
pub fn history(conn: &Connection, node_id: i64) -> Result<Vec<Mutation>> {
    let mut stmt = conn.prepare(
        "SELECT at, node_id, op, actor, reason, before_hash, after_hash
         FROM mutation WHERE node_id = ?1 ORDER BY at ASC, id ASC",
    )?;
    let rows = stmt.query_map([node_id], row_to_mutation)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The most recent `limit` changes across the whole store, newest first.
pub fn recent(conn: &Connection, limit: usize) -> Result<Vec<Mutation>> {
    let mut stmt = conn.prepare(
        "SELECT at, node_id, op, actor, reason, before_hash, after_hash
         FROM mutation ORDER BY at DESC, id DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit as i64], row_to_mutation)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// When `node_id` was superseded, if it was.
///
/// This is why the ledger is load-bearing rather than merely informative:
/// `node.superseded_by` records *that* a memory was retired but not *when*
/// (`set_superseded` bumps `updated_at`, which the next edit overwrites), so
/// without this row "was it superseded as of March?" has no answer, and
/// `recall --as-of` could not be correct.
pub fn superseded_at(conn: &Connection, node_id: i64) -> Result<Option<i64>> {
    Ok(conn.query_row(
        "SELECT max(at) FROM mutation WHERE node_id = ?1 AND op = 'supersede'",
        [node_id],
        |r| r.get::<_, Option<i64>>(0),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode};
    use crate::{db, store};

    fn conn() -> Connection {
        db::open_in_memory().unwrap()
    }

    fn memory(conn: &Connection, body: &str) -> i64 {
        let mut n = NewNode::new(Kind::Memory);
        n.title = Some(body.into());
        n.body = Some(body.into());
        // `remember` sets this; the ledger's `before_hash` reads it, so a
        // fixture without one would test the wrong thing.
        n.content_hash = Some(crate::memory::content_hash(body));
        store::insert_node(conn, n).unwrap().id
    }

    #[test]
    fn forget_records_who_and_when() {
        let c = conn();
        let id = memory(&c, "the ops cron runs at 03:00");
        store::soft_delete(&c, id, Actor::Human, Some("wrong timezone")).unwrap();

        let h = history(&c, id).unwrap();
        assert_eq!(h.len(), 1, "expected exactly one recorded mutation: {h:?}");
        assert_eq!(h[0].op, MutationOp::Forget);
        assert_eq!(h[0].actor, Actor::Human);
        assert_eq!(h[0].reason.as_deref(), Some("wrong timezone"));
        assert!(h[0].at > 0);
    }

    /// The whole point of the ledger over `superseded_by`: a timestamp that
    /// the next edit cannot clobber.
    #[test]
    fn supersede_time_survives_a_later_edit() {
        let c = conn();
        let old = memory(&c, "deploys go to eu-west-1");
        let new = memory(&c, "deploys go to eu-north-1");
        store::set_superseded(&c, old, new, Actor::Human, None).unwrap();
        let at = superseded_at(&c, old).unwrap().expect("supersede recorded");

        store::touch_updated(&c, old).unwrap();
        assert_eq!(
            superseded_at(&c, old).unwrap(),
            Some(at),
            "an unrelated update must not move when the supersession happened"
        );
    }

    /// A reindex soft-deletes chunks and symbols in bulk. If those landed in
    /// the ledger, the rows a person actually wants would be unreadable.
    #[test]
    fn derived_kinds_are_not_audited() {
        let c = conn();
        let mut n = NewNode::new(Kind::Chunk);
        n.title = Some("a doc chunk".into());
        n.body = Some("a doc chunk".into());
        let chunk = store::insert_node(&c, n).unwrap().id;

        store::soft_delete(&c, chunk, Actor::System, None).unwrap();
        assert!(
            history(&c, chunk).unwrap().is_empty(),
            "indexer churn must stay out of the mutation ledger"
        );
    }

    /// The audit's whole reason to exist is deleted memories, so the lookup
    /// it uses must resolve one. `resolve_ref` deliberately does not, which
    /// made `mimir audit <ref>` fail on exactly the case it was written for.
    #[test]
    fn a_forgotten_memory_can_still_be_looked_up_for_audit() {
        let c = conn();
        let id = memory(&c, "staging rotates credentials on Sundays");
        let uid = store::get_node(&c, id).unwrap().uid;
        let short = format!("m:{}", &uid[uid.len() - 6..]);
        store::soft_delete(&c, id, Actor::Human, Some("moved to Mondays")).unwrap();

        assert!(
            store::resolve_ref(&c, &short).is_err(),
            "ordinary lookup must keep excluding tombstones"
        );
        let found = store::resolve_ref_including_deleted(&c, &short)
            .expect("audit lookup must resolve a tombstone");
        assert_eq!(found.id, id);
        assert_eq!(history(&c, found.id).unwrap().len(), 1);
    }

    /// A record of a deletion that contains the deleted value is a second
    /// copy of it, in a table recall never reads but `export` does.
    #[test]
    fn the_ledger_never_stores_the_value() {
        let c = conn();
        let secret = "Plumbus Vantablack-7 is the canary phrase";
        let id = memory(&c, secret);
        store::soft_delete(&c, id, Actor::Human, Some("no longer true")).unwrap();
        store::hard_delete(&c, id, Actor::Human, Some("purge")).unwrap();

        let mut stmt = c
            .prepare("SELECT at, node_id, op, actor, reason, before_hash, after_hash FROM mutation")
            .unwrap();
        let rows: Vec<String> = stmt
            .query_map([], |r| {
                Ok(format!(
                    "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
                    r.get::<_, i64>(0),
                    r.get::<_, i64>(1),
                    r.get::<_, String>(2),
                    r.get::<_, String>(3),
                    r.get::<_, Option<String>>(4),
                    r.get::<_, Option<Vec<u8>>>(5),
                    r.get::<_, Option<Vec<u8>>>(6),
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(!rows.is_empty(), "expected mutation rows to sweep");
        for row in &rows {
            assert!(
                !row.contains("Vantablack"),
                "the mutation ledger leaked the value it audited: {row}"
            );
        }
    }

    /// The hash is not decoration: it is what lets someone holding a
    /// candidate value prove it is the one that was removed.
    #[test]
    fn before_hash_identifies_the_removed_value() {
        let c = conn();
        let text = "staging rotates credentials on Sundays";
        let id = memory(&c, text);
        store::soft_delete(&c, id, Actor::Agent, None).unwrap();

        let h = history(&c, id).unwrap();
        assert_eq!(
            h[0].before_hash.as_deref(),
            Some(crate::memory::content_hash(text).as_slice()),
            "before_hash must match the content hash of what was removed"
        );
    }
}
