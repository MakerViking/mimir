//! Content history: what a memory used to say.
//!
//! [`crate::memory::edit`] overwrites `body` and `title` in place. Before this
//! table existed, the previous wording was simply gone — which is fine for
//! recall (nobody wants last week's typo ranked) and fatal for
//! `recall --as-of`, because an as-of query with no prior text can only tell
//! you a row *existed* then and would hand back today's wording under a past
//! date. A confident wrong answer is worse than no answer.
//!
//! The invariant that makes this safe rather than a liability lives in
//! [`purge`]: a deliberate `forget` deletes a node's revisions. Otherwise a
//! deleted value survives in a table that recall never reads but `export` and
//! every backup do, which is the leak the deletion existed to prevent.

use rusqlite::{params, Connection};

use crate::error::Result;
use crate::model::{now_unix, Kind, Node};

/// A memory's text as it stood, and when it stopped standing that way.
#[derive(Debug, Clone)]
pub struct Revision {
    pub at: i64,
    pub title: Option<String>,
    pub body: Option<String>,
}

/// Snapshot a memory's current text before an edit replaces it.
///
/// Fails open, like the mutation ledger: an edit that landed must not be
/// reported as failed because the history write hiccuped. Memories only —
/// doc chunks and symbols are re-derived from a source that is itself the
/// history, so versioning them would duplicate the repository.
pub fn snapshot(conn: &Connection, node: &Node) {
    if node.kind != Kind::Memory {
        return;
    }
    let _ = conn.execute(
        "INSERT INTO node_revision (node_id, at, title, body, meta)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            node.id,
            now_unix(),
            node.title,
            node.body,
            node.meta.to_string()
        ],
    );
}

/// Every superseded version of `node_id`, oldest first.
pub fn history(conn: &Connection, node_id: i64) -> Result<Vec<Revision>> {
    let mut stmt = conn.prepare(
        "SELECT at, title, body FROM node_revision
         WHERE node_id = ?1 ORDER BY at ASC, id ASC",
    )?;
    let rows = stmt.query_map([node_id], |r| {
        Ok(Revision {
            at: r.get("at")?,
            title: r.get("title")?,
            body: r.get("body")?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The text `node_id` carried at `at`, or `None` if the current text already
/// applies (no revision was taken after that instant).
///
/// The earliest revision *after* `at` holds the text that was live at `at`,
/// because a revision records what the memory said until the edit that
/// replaced it.
pub fn text_at(conn: &Connection, node_id: i64, at: i64) -> Result<Option<Revision>> {
    let mut stmt = conn.prepare(
        "SELECT at, title, body FROM node_revision
         WHERE node_id = ?1 AND at > ?2 ORDER BY at ASC, id ASC LIMIT 1",
    )?;
    let mut rows = stmt.query_map(params![node_id, at], |r| {
        Ok(Revision {
            at: r.get("at")?,
            title: r.get("title")?,
            body: r.get("body")?,
        })
    })?;
    Ok(rows.next().transpose()?)
}

/// Drop every stored version of `node_id`.
///
/// Called on deliberate deletion. Not on decay archival: nobody decided that,
/// `remember` already lets an archived fact back in without `--force`, and
/// destroying history for a background pass's bookkeeping would lose the one
/// thing that could explain a memory's wording later.
pub fn purge(conn: &Connection, node_id: i64) -> Result<usize> {
    Ok(conn.execute("DELETE FROM node_revision WHERE node_id = ?1", [node_id])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Actor, MemoryType, NewNode};
    use crate::{db, memory, store};

    fn conn() -> Connection {
        db::open_in_memory().unwrap()
    }

    fn memory_node(conn: &Connection, body: &str) -> Node {
        let mut n = NewNode::new(Kind::Memory);
        n.title = Some(body.into());
        n.body = Some(body.into());
        n.content_hash = Some(memory::content_hash(body));
        store::insert_node(conn, n).unwrap()
    }

    #[test]
    fn an_edit_keeps_what_the_memory_used_to_say() {
        let c = conn();
        let node = memory_node(&c, "deploys go to eu-west-1");
        memory::edit(
            &c,
            node.id,
            memory::Edit {
                text: Some("deploys go to eu-north-1".into()),
                ..Default::default()
            },
            Actor::Human,
            Some("region migration"),
        )
        .unwrap();

        let h = history(&c, node.id).unwrap();
        assert_eq!(h.len(), 1, "the pre-edit text must be kept: {h:?}");
        assert_eq!(h[0].body.as_deref(), Some("deploys go to eu-west-1"));
        assert_eq!(
            store::get_node(&c, node.id).unwrap().body.as_deref(),
            Some("deploys go to eu-north-1"),
            "the live row must still carry the new text"
        );
    }

    /// The invariant that keeps this table from becoming a leak: a deletion
    /// that leaves prior wordings behind has not deleted anything.
    #[test]
    fn forgetting_purges_the_history_too() {
        let c = conn();
        let node = memory_node(&c, "Plumbus Vantablack-7 is the staging root password hint");
        memory::edit(
            &c,
            node.id,
            memory::Edit {
                text: Some("Plumbus Vantablack-7 was rotated out".into()),
                ..Default::default()
            },
            Actor::Human,
            None,
        )
        .unwrap();
        assert_eq!(history(&c, node.id).unwrap().len(), 1);

        store::soft_delete(
            &c,
            node.id,
            Actor::Human,
            Some("should never have been written"),
        )
        .unwrap();

        assert!(
            history(&c, node.id).unwrap().is_empty(),
            "a deliberate forget must leave no prior version behind"
        );
        let leaked: i64 = c
            .query_row(
                "SELECT count(*) FROM node_revision WHERE body LIKE '%Vantablack%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leaked, 0, "the deleted value survived in node_revision");
    }

    /// Decay archival is not a decision anyone made, and `remember` already
    /// treats it differently. Destroying history for it would lose the only
    /// record of how a memory came to be worded the way it is.
    #[test]
    fn decay_archival_keeps_the_history() {
        let c = conn();
        let node = memory_node(&c, "cargo nextest needs a separate install");
        memory::edit(
            &c,
            node.id,
            memory::Edit {
                text: Some("cargo nextest needs a separate install on CI".into()),
                ..Default::default()
            },
            Actor::System,
            None,
        )
        .unwrap();
        // Exactly what consolidate's decay pass does.
        c.execute(
            "UPDATE node SET deleted_at = 1, meta = json_set(meta, '$.archived', 1) WHERE id = ?1",
            [node.id],
        )
        .unwrap();

        assert_eq!(
            history(&c, node.id).unwrap().len(),
            1,
            "archival is not a deletion and must not destroy history"
        );
    }

    #[test]
    fn text_at_returns_the_wording_that_was_live_then() {
        let c = conn();
        let node = memory_node(&c, "the retry budget is 3");
        c.execute(
            "INSERT INTO node_revision (node_id, at, title, body)
             VALUES (?1, 1000, 'the retry budget is 3', 'the retry budget is 3')",
            [node.id],
        )
        .unwrap();

        let then = text_at(&c, node.id, 500).unwrap().expect("a prior version");
        assert_eq!(then.body.as_deref(), Some("the retry budget is 3"));
        assert!(
            text_at(&c, node.id, 2000).unwrap().is_none(),
            "after the last revision the current text applies"
        );
    }

    #[test]
    fn ordinary_capture_writes_no_revision() {
        let c = conn();
        let node = memory_node(&c, "nothing has happened to this yet");
        assert!(
            history(&c, node.id).unwrap().is_empty(),
            "history is for changes, not for existing"
        );
        let _ = MemoryType::Note;
    }
}
