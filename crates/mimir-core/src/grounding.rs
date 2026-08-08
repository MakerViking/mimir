//! Whether a memory is attached to something Mimir can go and check.
//!
//! Every other signal on a memory is a *policy*: strength rises because it
//! got used, a type prior favours gotchas, a human marks it useful. None of
//! those can ever be proven wrong by the code that assigns them — they are
//! opinions about a claim, computed from how people treated the claim.
//!
//! Grounding is different, and that is the whole point of it. A memory is
//! grounded when it links to an artifact Mimir indexes from disk — a code
//! symbol, a source chunk, a doc chunk, a file. That link is a falsifiable
//! assertion: the indexer and `graph build` soft-delete artifacts that stop
//! existing, so "this memory is about `retry_with_backoff`" becomes checkable
//! by looking, and stops being true the moment the symbol is deleted.
//!
//! [`Grounding::Stale`] is that check failing — the memory still claims to be
//! about something the codebase no longer has. It does **not** mean the
//! memory is wrong; a note about a function that was renamed is usually still
//! valuable. It means nobody has revisited it since the ground moved.
//!
//! Deliberately inert in ranking. Grounding is reported, never scored: it
//! changes what you can see about a memory, not which memories you get back.
//! Making it a ranking input is a real option, but it is a different decision
//! with a different blast radius, and it would need the drift-eval baseline
//! re-cut rather than smuggled in behind a display field.

use rusqlite::Connection;

use crate::error::Result;
use crate::model::Kind;

/// Artifact kinds whose existence Mimir verifies against disk. A link to
/// any of these is checkable; a link to another memory or a tag is not,
/// which is why those don't ground anything.
const GROUNDING_KINDS: [Kind; 4] = [Kind::Symbol, Kind::CodeChunk, Kind::Chunk, Kind::File];

/// What a memory is attached to, and whether that thing still exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grounding {
    /// Linked to an artifact that is still indexed.
    Grounded { kind: Kind, label: String },
    /// Linked to an artifact that has since been soft-deleted — the file
    /// left the collection, or the symbol went away on the last
    /// `graph build`. The claim was checked and it failed.
    Stale { kind: Kind, label: String },
    /// No link to anything checkable: a free-standing assertion, trusted
    /// entirely because someone wrote it down.
    Ungrounded,
}

impl Grounding {
    /// Short display token. Kept blunt on purpose — `stale` should read as
    /// something to look at, `ungrounded` as a plain statement of fact and
    /// not an accusation, since most memories are legitimately ungrounded.
    pub fn label(&self) -> &'static str {
        match self {
            Grounding::Grounded { .. } => "grounded",
            Grounding::Stale { .. } => "stale",
            Grounding::Ungrounded => "ungrounded",
        }
    }

    pub fn is_stale(&self) -> bool {
        matches!(self, Grounding::Stale { .. })
    }

    /// What it points at, for display.
    pub fn target(&self) -> Option<(Kind, &str)> {
        match self {
            Grounding::Grounded { kind, label } | Grounding::Stale { kind, label } => {
                Some((*kind, label.as_str()))
            }
            Grounding::Ungrounded => None,
        }
    }
}

/// Resolve one memory's grounding.
///
/// A live artifact wins over a dead one: a memory linked to both a deleted
/// symbol and a live file is grounded, not stale. Only when every artifact
/// it points at is gone does it count as stale — otherwise ordinary
/// re-chunking, which soft-deletes and re-inserts, would flap the status of
/// memories that are perfectly well attached.
pub fn grounding(conn: &Connection, node_id: i64) -> Result<Grounding> {
    let kinds = GROUNDING_KINDS
        .iter()
        .map(|k| format!("'{}'", k.as_str()))
        .collect::<Vec<_>>()
        .join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT n.kind, COALESCE(n.title, n.path, ''), n.deleted_at IS NULL AS alive
         FROM edge e
         JOIN node n ON n.id = CASE WHEN e.src = ?1 THEN e.dst ELSE e.src END
         WHERE (e.src = ?1 OR e.dst = ?1) AND n.kind IN ({kinds})
         ORDER BY alive DESC"
    ))?;
    let mut rows = stmt.query([node_id])?;
    let mut stale: Option<Grounding> = None;
    while let Some(row) = rows.next()? {
        let kind: String = row.get(0)?;
        let label: String = row.get(1)?;
        let alive: bool = row.get(2)?;
        let Ok(kind) = kind.parse::<Kind>() else {
            continue;
        };
        if alive {
            return Ok(Grounding::Grounded { kind, label });
        }
        stale.get_or_insert(Grounding::Stale { kind, label });
    }
    Ok(stale.unwrap_or(Grounding::Ungrounded))
}

/// (grounded, stale, ungrounded) across every live memory. Used by `doctor`
/// to surface the one number worth acting on: how many memories describe
/// something the codebase no longer contains.
pub fn tally(conn: &Connection) -> Result<(usize, usize, usize)> {
    let mut stmt = conn.prepare(
        "SELECT id FROM node WHERE kind = 'memory' AND deleted_at IS NULL
         AND superseded_by IS NULL",
    )?;
    let ids: Vec<i64> = stmt
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let (mut grounded, mut stale, mut ungrounded) = (0, 0, 0);
    for id in ids {
        match grounding(conn, id)? {
            Grounding::Grounded { .. } => grounded += 1,
            Grounding::Stale { .. } => stale += 1,
            Grounding::Ungrounded => ungrounded += 1,
        }
    }
    Ok((grounded, stale, ungrounded))
}

/// Every live memory whose grounding has been falsified, newest first.
pub fn stale_memories(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<(crate::model::Node, String)>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {} FROM node WHERE kind = 'memory' AND deleted_at IS NULL
         AND superseded_by IS NULL ORDER BY created_at DESC",
        crate::store::NODE_COLS
    ))?;
    let nodes: Vec<crate::model::Node> = stmt
        .query_map([], crate::store::row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for node in nodes {
        if out.len() >= limit {
            break;
        }
        if let Grounding::Stale { kind, label } = grounding(conn, node.id)? {
            out.push((node, format!("{} {label}", kind.as_str())));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::{NewNode, Rel};
    use crate::store;

    fn memory(conn: &Connection, body: &str) -> i64 {
        let mut n = NewNode::new(Kind::Memory);
        n.subkind = Some("note".into());
        n.title = Some(body.into());
        n.body = Some(body.into());
        store::insert_node(conn, n).unwrap().id
    }

    fn artifact(conn: &Connection, kind: Kind, title: &str) -> i64 {
        let mut n = NewNode::new(kind);
        n.title = Some(title.into());
        store::insert_node(conn, n).unwrap().id
    }

    #[test]
    fn unlinked_memory_is_ungrounded() {
        let conn = db::open_in_memory().unwrap();
        let m = memory(&conn, "the retry backoff doubles each attempt");
        assert_eq!(grounding(&conn, m).unwrap(), Grounding::Ungrounded);
    }

    #[test]
    fn memory_linked_to_a_live_symbol_is_grounded() {
        let conn = db::open_in_memory().unwrap();
        let m = memory(&conn, "the retry backoff doubles each attempt");
        let s = artifact(&conn, Kind::Symbol, "retry_with_backoff");
        store::link(&conn, m, s, Rel::About, 1.0).unwrap();
        assert!(matches!(
            grounding(&conn, m).unwrap(),
            Grounding::Grounded { kind: Kind::Symbol, label } if label == "retry_with_backoff"
        ));
    }

    /// The falsification: nothing about the memory changed, the codebase
    /// did. This is the one status that a policy-assigned trust label could
    /// never produce about itself.
    #[test]
    fn grounding_goes_stale_when_the_symbol_disappears() {
        let conn = db::open_in_memory().unwrap();
        let m = memory(&conn, "the retry backoff doubles each attempt");
        let s = artifact(&conn, Kind::Symbol, "retry_with_backoff");
        store::link(&conn, m, s, Rel::About, 1.0).unwrap();
        assert!(!grounding(&conn, m).unwrap().is_stale());

        store::soft_delete(&conn, s).unwrap();
        let g = grounding(&conn, m).unwrap();
        assert!(g.is_stale(), "symbol is gone; the claim is falsified");
        assert_eq!(g.target(), Some((Kind::Symbol, "retry_with_backoff")));
    }

    /// Re-chunking soft-deletes and re-inserts constantly. A memory that
    /// still has one live artifact must not flap to stale just because
    /// another one it references was rewritten.
    #[test]
    fn one_live_artifact_outweighs_a_dead_one() {
        let conn = db::open_in_memory().unwrap();
        let m = memory(&conn, "the indexer walks gitignore files");
        let dead = artifact(&conn, Kind::CodeChunk, "old_chunk");
        let live = artifact(&conn, Kind::File, "src/index/mod.rs");
        store::link(&conn, m, dead, Rel::About, 1.0).unwrap();
        store::link(&conn, m, live, Rel::About, 1.0).unwrap();
        store::soft_delete(&conn, dead).unwrap();
        assert!(matches!(
            grounding(&conn, m).unwrap(),
            Grounding::Grounded {
                kind: Kind::File,
                ..
            }
        ));
    }

    /// A link to another memory is not grounding. Two assertions agreeing
    /// with each other is not evidence, and letting memory→memory links
    /// count would make grounding trivially self-satisfiable.
    #[test]
    fn linking_to_another_memory_grounds_nothing() {
        let conn = db::open_in_memory().unwrap();
        let a = memory(&conn, "the retry backoff doubles each attempt");
        let b = memory(&conn, "backoff was tuned in the 0.9 release");
        store::link(&conn, a, b, Rel::Relates, 1.0).unwrap();
        assert_eq!(grounding(&conn, a).unwrap(), Grounding::Ungrounded);
    }

    #[test]
    fn tally_and_stale_listing_agree() {
        let conn = db::open_in_memory().unwrap();
        let grounded = memory(&conn, "chunker splits on symbol boundaries");
        let s = artifact(&conn, Kind::Symbol, "chunk_source");
        store::link(&conn, grounded, s, Rel::About, 1.0).unwrap();

        let broken = memory(&conn, "the old pruner ran on every write");
        let gone = artifact(&conn, Kind::Symbol, "prune_on_write");
        store::link(&conn, broken, gone, Rel::About, 1.0).unwrap();
        store::soft_delete(&conn, gone).unwrap();

        memory(&conn, "prefer ripgrep over find on this box");

        assert_eq!(tally(&conn).unwrap(), (1, 1, 1));
        let stale = stale_memories(&conn, 10).unwrap();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0.id, broken);
        assert_eq!(stale[0].1, "symbol prune_on_write");
    }
}
