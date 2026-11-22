//! Projects whose directory is gone, and removing what was derived for them.
//!
//! A project is registered the first time anything runs inside a directory,
//! and its code graph is built on first use. Nothing ever unregistered one:
//! a deleted worktree, a scratch clone under `/tmp`, a renamed checkout each
//! left a project row and its whole graph behind, and the store only grew.
//! "The directory is gone" was a state the store could be in and had no name
//! for, so no surface could report it and no command could act on it.
//!
//! [`missing_projects`] names it. [`prune_graph`] acts on it, and only on
//! what Mimir derived from the files that are gone: symbols, code-file rows
//! and communities, which `mimir graph build` would recreate if the
//! directory came back. Memories are never touched, and neither is anything
//! a person linked to: a symbol or a code file that carries an authored link
//! is kept as a tombstone, so the memory that pointed at it reads "stale"
//! instead of silently losing its link.
//!
//! "Gone" means the filesystem answered that nothing is at that path. A
//! path that could not be checked (no permission to look, a network mount
//! that errors) is not gone, and neither is one this process cannot place
//! (a relative path, which would be resolved against wherever the command
//! happens to run).

use std::path::Path;

use mimir_core::error::Result;
use mimir_core::model::{now_unix, Node, Rel, RelOrigin};
use mimir_core::store::{row_to_node, NODE_COLS};
use rusqlite::{Connection, OptionalExtension};

/// A registered project whose directory is not on disk, with what the store
/// still holds for it.
#[derive(Debug, Clone)]
pub struct MissingProject {
    pub project: Node,
    /// Symbol rows, tombstoned ones included.
    pub symbols: i64,
    /// Code-file rows (not docs-collection files).
    pub files: i64,
    /// Memories filed under the project. Never removed; reported so the
    /// listing can say why a project row stays.
    pub memories: i64,
    /// Symbol and code-file rows that are not tombstones: what a prune has
    /// not dealt with yet. After a prune this is zero, whatever was kept.
    pub unpruned: i64,
}

impl MissingProject {
    /// Whether [`prune_graph`] has anything left to do for this project.
    /// Not "holds any row": what a prune keeps (a tombstone someone linked
    /// to) stays for good, and a listing that went on reporting it would be
    /// advice the reader had already taken.
    pub fn has_graph_rows(&self) -> bool {
        self.unpruned > 0
    }
}

/// What one [`prune_graph`] call removed and kept.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pruned {
    pub symbols_removed: usize,
    pub files_removed: usize,
    /// Symbols kept as tombstones because someone linked to them.
    pub symbols_kept: usize,
    /// Code files kept as tombstones: someone linked to the file itself, or
    /// it holds a kept symbol.
    pub files_kept: usize,
    /// The project row itself went: nothing refers to it any more.
    pub project_removed: bool,
    /// Links that were moved to the same symbol or file in the home project
    /// (see [`prune_graph_toward`]), instead of keeping a tombstone for them.
    pub links_moved: usize,
}

/// SQL list of the relations Mimir derives for itself. An edge of any other
/// relation on a symbol is a link somebody made.
pub(crate) fn derived_rels_sql() -> String {
    Rel::ALL
        .iter()
        .filter(|rel| rel.origin() == RelOrigin::Extracted)
        .map(|rel| format!("'{}'", rel.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

/// Whether the directory a project was registered with is known to be gone.
/// Only a definite "nothing is there" counts; see the module doc.
fn directory_is_gone(path: &str) -> bool {
    let path = Path::new(path);
    path.is_absolute() && matches!(path.try_exists(), Ok(false))
}

/// Live projects registered with a path that does not exist, largest graph
/// first. A project with no path (a shadow created by sync, waiting to be
/// opened on this machine) is not missing: it never had a directory here.
///
/// "Does not exist" is a plain filesystem check, so a project on a drive
/// that is not mounted right now is listed too. That is why pruning is a
/// separate, explicit step, and why it removes only what can be rebuilt.
pub fn missing_projects(conn: &Connection) -> Result<Vec<MissingProject>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'project' AND deleted_at IS NULL AND path IS NOT NULL
         ORDER BY id"
    ))?;
    let projects: Vec<Node> = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    let mut missing: Vec<MissingProject> = projects
        .into_iter()
        .filter(|project| project.path.as_deref().is_some_and(directory_is_gone))
        .map(|project| MissingProject {
            project,
            symbols: 0,
            files: 0,
            memories: 0,
            unpruned: 0,
        })
        .collect();

    // What the store holds for them, counted in one pass per batch. One
    // query per project and kind was three reads of the whole node table
    // for every missing project on a store that does not have the
    // `node_project` index yet, which is exactly the store `doctor` is asked
    // about. The ids are this store's own row ids, so they go in as text.
    let mut at: std::collections::HashMap<i64, usize> = missing
        .iter()
        .enumerate()
        .map(|(i, m)| (m.project.id, i))
        .collect();
    let ids: Vec<i64> = missing.iter().map(|m| m.project.id).collect();
    for batch in ids.chunks(COUNT_BATCH) {
        let list = batch
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut stmt = conn.prepare(&format!(
            "SELECT project_id,
                    COALESCE(sum(kind = 'symbol'), 0),
                    COALESCE(sum(kind = 'file' AND collection_id IS NULL), 0),
                    COALESCE(sum(kind = 'memory'), 0),
                    COALESCE(sum(deleted_at IS NULL
                                 AND (kind = 'symbol'
                                      OR (kind = 'file' AND collection_id IS NULL))), 0)
             FROM node WHERE project_id IN ({list})
             GROUP BY project_id"
        ))?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;
        for row in rows {
            let (project_id, symbols, files, memories, unpruned) = row?;
            if let Some(i) = at.remove(&project_id) {
                missing[i].symbols = symbols;
                missing[i].files = files;
                missing[i].memories = memories;
                missing[i].unpruned = unpruned;
            }
        }
    }
    missing.sort_by_key(|m| (std::cmp::Reverse(m.symbols), m.project.id));
    Ok(missing)
}

/// How many projects one counting query covers.
const COUNT_BATCH: usize = 500;

/// How long one transaction of a prune holds the write lock before it
/// commits and lets go.
///
/// A removed symbol takes its edges, its vector and its search-index entry
/// with it, so the write lock is held for the whole of a transaction, and
/// every other writer to the store (a `remember`, a hook) waits that long.
/// One transaction per project was a minute and more for a large project.
/// A fixed number of rows per transaction is no bound either: on a store of
/// 3.7 million rows, 2,000 rows usually took a third of a second and, with
/// the disk busy, once took 6.6 seconds. So the bound is the time itself,
/// checked after every row.
const PRUNE_HOLD: std::time::Duration = std::time::Duration::from_millis(250);

/// How long a prune stays off the write lock between two transactions.
///
/// Committing does not hand the lock to whoever is waiting for it: a
/// waiting SQLite connection polls, at intervals that grow to 100 ms, and a
/// pruner that takes the lock again at once wins that race almost every
/// time. The pause is longer than the longest poll interval, so a waiting
/// writer always gets a turn.
const PRUNE_YIELD: std::time::Duration = std::time::Duration::from_millis(120);

/// How many rows one search for removable rows returns. Not a bound on
/// anything: the rows found are removed over as many transactions as
/// [`PRUNE_HOLD`] makes of them.
const PRUNE_SEARCH: i64 = 2_000;

/// Remove the code graph of `project`: its symbols, code-file rows and
/// communities, with their edges and vectors.
///
/// A symbol or a code file with an authored link is tombstoned and kept, and
/// so is the file a kept symbol hangs under. The project row is removed when
/// nothing refers to it afterwards (no memory, no kept row, no docs
/// collection); otherwise it stays, still naming the scope its memories were
/// written in.
///
/// The work is done in short transactions with a pause after each
/// ([`PRUNE_HOLD`], [`PRUNE_YIELD`]), so no other writer waits long. Every batch leaves the store consistent, and a prune that is
/// interrupted is finished by running it again: what is left is still a
/// project whose directory is gone. Called inside a transaction the caller
/// holds, the batches nest in it and the whole prune is the caller's unit.
///
/// The caller decides *that* a project is to be pruned; this does not check
/// the directory again. It expects the store's support indexes to exist
/// (`mimir_core::db::ensure_indexes`): without them each deleted row costs a
/// scan of the whole node table.
pub fn prune_graph(conn: &Connection, project: &Node) -> Result<Pruned> {
    prune_graph_toward(conn, project, None)
}

/// [`prune_graph`] for a project that was a git linked worktree of `home`.
///
/// A memory linked to a symbol of the worktree is about that symbol in the
/// repository, not in one checkout of it. When the worktree goes, the link
/// moves to the same symbol of the home project (same file path, qualified
/// name and kind), and to the same file for a link to a file, so the memory
/// stays grounded instead of reading as stale while the code it describes
/// is unchanged in the main checkout. Only a link with no counterpart in
/// the home keeps its tombstone.
pub fn prune_graph_toward(
    conn: &Connection,
    project: &Node,
    home: Option<&Node>,
) -> Result<Pruned> {
    let links_moved = match home {
        Some(home) if home.id != project.id => {
            mimir_core::db::atomic(conn, || relink_toward(conn, project.id, home.id))?
        }
        _ => 0,
    };
    let mut out = prune_graph_in_batches(conn, project, PRUNE_HOLD, PRUNE_YIELD)?;
    out.links_moved = links_moved;
    Ok(out)
}

/// Move every authored link on a symbol or code file of `from` to the row of
/// `to` that stands for the same thing, where there is one. Returns how many
/// links moved. A link the counterpart already has is dropped, not doubled.
fn relink_toward(conn: &Connection, from: i64, to: i64) -> Result<usize> {
    let derived = derived_rels_sql();
    // Each linked row of `from` with its counterpart in `to`.
    let mut stmt = conn.prepare(&format!(
        "SELECT n.id, n.kind, n.path, n.title, n.subkind FROM node n
         WHERE n.project_id = ?1
           AND (n.kind = 'symbol' OR (n.kind = 'file' AND n.collection_id IS NULL))
           AND (EXISTS (SELECT 1 FROM edge e WHERE e.src = n.id AND e.rel NOT IN ({derived}))
             OR EXISTS (SELECT 1 FROM edge e WHERE e.dst = n.id AND e.rel NOT IN ({derived})))"
    ))?;
    // (id, kind, path, qualified name, symbol kind)
    type Linked = (i64, String, Option<String>, Option<String>, Option<String>);
    let linked: Vec<Linked> = stmt
        .query_map([from], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    let mut moved = 0;
    for (id, kind, path, title, subkind) in linked {
        let counterpart: Option<i64> = match (kind.as_str(), path, title, subkind) {
            ("symbol", Some(path), Some(qualified), Some(subkind)) => conn
                .query_row(
                    "SELECT id FROM node
                     WHERE kind = 'symbol' AND deleted_at IS NULL
                       AND json_extract(meta, '$.stable_id') = ?1",
                    [crate::store_graph::stable_id(
                        to, &path, &qualified, &subkind,
                    )],
                    |r| r.get(0),
                )
                .optional()?,
            ("file", Some(path), _, _) => conn
                .query_row(
                    "SELECT id FROM node
                     WHERE kind = 'file' AND collection_id IS NULL AND deleted_at IS NULL
                       AND project_id = ?1 AND path = ?2",
                    rusqlite::params![to, path],
                    |r| r.get(0),
                )
                .optional()?,
            _ => None,
        };
        let Some(counterpart) = counterpart else {
            continue;
        };
        for end in ["src", "dst"] {
            moved += conn.execute(
                &format!(
                    "UPDATE OR IGNORE edge SET {end} = ?2
                     WHERE {end} = ?1 AND rel NOT IN ({derived})"
                ),
                [id, counterpart],
            )?;
            // What is left on this end is a link the counterpart already had.
            conn.execute(
                &format!("DELETE FROM edge WHERE {end} = ?1 AND rel NOT IN ({derived})"),
                [id],
            )?;
        }
    }
    Ok(moved)
}

/// [`prune_graph`] with the pacing passed in: `hold` is how long one
/// transaction may run before it commits (zero commits after every row),
/// `pause` how long the lock is left alone afterwards.
fn prune_graph_in_batches(
    conn: &Connection,
    project: &Node,
    hold: std::time::Duration,
    pause: std::time::Duration,
) -> Result<Pruned> {
    use mimir_core::db::atomic;
    let derived = derived_rels_sql();
    let pid = project.id;
    let mut out = Pruned::default();
    // "Somebody linked to this row": an edge on either end that is not one
    // Mimir derived. Asked per row, through the row's own id, so it never
    // walks the edge table; and asked inside each batch, so a link made
    // while a prune is running is honoured.
    let unlinked = |row: &str| {
        format!(
            "NOT EXISTS (SELECT 1 FROM edge e
                         WHERE e.src = {row}.id AND e.rel NOT IN ({derived}))
             AND NOT EXISTS (SELECT 1 FROM edge e
                             WHERE e.dst = {row}.id AND e.rel NOT IN ({derived}))"
        )
    };
    const CODE_FILE: &str = "kind = 'file' AND collection_id IS NULL";

    // Find rows, then remove them in transactions of their own, pausing
    // after each to let a waiting writer in.
    //
    // The search runs before the write lock is taken. It is most of the
    // reading there is to do (a cold project's rows, an edge lookup per
    // row), and done under the lock it was measured holding every other
    // writer for four seconds to find that a project had nothing left to
    // remove. The removal repeats the search's condition for each row it
    // was handed, so a link somebody makes between the two is honoured.
    //
    // Inside a transaction the caller holds there is nobody to let in, and a
    // pause would only stretch the caller's hold on the lock.
    let yields = conn.is_autocommit() && !pause.is_zero();
    let remove = |what: &str, removable: &str| -> Result<usize> {
        let find =
            format!("SELECT n.id FROM node n WHERE n.project_id = ?1 AND {removable} LIMIT ?2");
        let delete =
            format!("DELETE FROM node AS n WHERE n.id = ?1 AND n.project_id = ?2 AND {removable}");
        let mut total = 0;
        loop {
            let found: Vec<i64> = conn
                .prepare(&find)?
                .query_map([pid, PRUNE_SEARCH], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            let mut rest = found.as_slice();
            let mut removed_of_these = 0;
            while !rest.is_empty() {
                let started = std::time::Instant::now();
                let (tried, removed) = atomic(conn, || {
                    let mut one = conn.prepare_cached(&delete)?;
                    let (mut tried, mut removed) = (0, 0);
                    for id in rest {
                        removed += one.execute([*id, pid])?;
                        tried += 1;
                        if started.elapsed() >= hold {
                            break;
                        }
                    }
                    Ok((tried, removed))
                })?;
                tracing::debug!(
                    project = pid,
                    what,
                    removed,
                    took_ms = started.elapsed().as_millis() as u64,
                    "prune batch"
                );
                rest = &rest[tried..];
                removed_of_these += removed;
                if yields {
                    std::thread::sleep(pause);
                }
            }
            total += removed_of_these;
            // Nothing found, or found and no longer removable (someone else
            // changed those rows in between): this prune is done with them.
            if removed_of_these == 0 {
                return Ok(total);
            }
        }
    };
    const CHILDLESS: &str = "NOT EXISTS (SELECT 1 FROM node c WHERE c.parent_id = n.id)";

    // Communities first: their `member_of` edges go with them.
    remove("communities", "n.kind = 'community'")?;

    // Then the symbols nobody linked to.
    out.symbols_removed = remove(
        "symbols",
        &format!("n.kind = 'symbol' AND {} AND {CHILDLESS}", unlinked("n")),
    )?;

    // Then the code files nobody linked to and nothing hangs under. A file
    // that still holds a symbol holds a kept one: it is what the stale link
    // is "in".
    out.files_removed = remove(
        "files",
        &format!(
            "n.kind = 'file' AND n.collection_id IS NULL AND {} AND {CHILDLESS}",
            unlinked("n")
        ),
    )?;

    // What is left was kept on purpose: tombstone it, count it, and remove
    // the project row when it is the last thing standing.
    atomic(conn, || {
        conn.execute(
            &format!(
                "UPDATE node SET deleted_at = ?2
                 WHERE project_id = ?1 AND deleted_at IS NULL
                   AND (kind = 'symbol' OR ({CODE_FILE}))"
            ),
            rusqlite::params![pid, now_unix()],
        )?;
        let left = |what: &str| -> Result<usize> {
            let n: i64 = conn.query_row(
                &format!("SELECT count(*) FROM node WHERE project_id = ?1 AND {what}"),
                [pid],
                |r| r.get(0),
            )?;
            Ok(n as usize)
        };
        out.symbols_kept = left("kind = 'symbol'")?;
        out.files_kept = left(CODE_FILE)?;

        // A row a live linked worktree names as its home is referenced too:
        // that worktree's memories are filed under it, and its running
        // sessions hold its id. Removing it while the main checkout's
        // directory is only away (unmounted, being re-cloned) turned every
        // later capture there into a foreign-key error.
        let referenced: bool = conn.query_row(
            &format!(
                "SELECT EXISTS (SELECT 1 FROM node WHERE project_id = ?1)
                     OR EXISTS (SELECT 1 FROM edge WHERE src = ?1 OR dst = ?1)
                     OR EXISTS (SELECT 1 FROM node w
                                WHERE w.kind = 'project' AND w.deleted_at IS NULL
                                  AND json_extract(w.meta, '$.{}') = ?2)",
                mimir_core::store::WORKTREE_OF
            ),
            rusqlite::params![pid, project.uid],
            |r| r.get(0),
        )?;
        if !referenced {
            out.project_removed =
                conn.execute("DELETE FROM node WHERE id = ?1 AND kind = 'project'", [pid])? > 0;
        }
        Ok(())
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::memory::{self, Remember, RememberOutcome};
    use mimir_core::model::{Actor, MemoryType};
    use mimir_core::{store, Mimir};

    /// A project of six files with five functions each, one of which a
    /// memory is linked to, and one file a memory is linked to as a whole.
    fn linked_project(mimir: &mut Mimir) -> (Node, i64, i64) {
        let dir = tempfile::tempdir().unwrap();
        for file in 0..6 {
            let body: String = (0..5)
                .map(|f| format!("pub fn f{file}_{f}() {{}}\n"))
                .collect();
            std::fs::write(dir.path().join(format!("m{file}.rs")), body).unwrap();
        }
        let project =
            store::ensure_project(&mimir.conn, &dir.path().to_string_lossy(), "gone").unwrap();
        crate::update(&mut mimir.conn, &project, dir.path()).unwrap();
        let note = |text: &str| match memory::remember(
            &mimir.conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Gotcha,
                tags: Vec::new(),
                project_id: None,
                force: false,
                actor: Actor::Human,
            },
        )
        .unwrap()
        {
            RememberOutcome::Created(node) => node,
            other => panic!("expected a new memory, got {other:?}"),
        };
        let symbol = crate::resolve_symbol(&mimir.conn, project.id, "f2_3").unwrap();
        let about_symbol = note("f2_3 must stay idempotent, the retry path calls it twice");
        store::link(&mimir.conn, about_symbol.id, symbol.id, Rel::About, 1.0).unwrap();
        let file: i64 = mimir
            .conn
            .query_row(
                "SELECT id FROM node WHERE project_id = ?1 AND kind = 'file' AND path = 'm4.rs'",
                [project.id],
                |r| r.get(0),
            )
            .unwrap();
        let about_file = note("m4.rs is generated by the scaffold script");
        store::link(&mimir.conn, about_file.id, file, Rel::About, 1.0).unwrap();
        (project, symbol.id, file)
    }

    /// Every row of the store that could tell two prunes apart.
    fn remaining(conn: &Connection) -> Vec<(String, String, bool)> {
        conn.prepare(
            "SELECT kind, COALESCE(title, path, ''), deleted_at IS NOT NULL
             FROM node WHERE kind IN ('project', 'file', 'symbol', 'community', 'memory')
             ORDER BY 1, 2",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
    }

    /// How long a transaction may run decides how long the write lock is
    /// held and nothing else: committing after every row and committing once
    /// remove and keep the same rows and report the same counts.
    #[test]
    fn a_prune_in_small_batches_ends_where_one_big_batch_ends() {
        use std::time::Duration;
        let outcome = |hold: Duration| {
            let mut mimir = Mimir::open_in_memory().unwrap();
            let (project, symbol, file) = linked_project(&mut mimir);
            let pruned =
                prune_graph_in_batches(&mimir.conn, &project, hold, Duration::ZERO).unwrap();
            for kept in [symbol, file] {
                assert!(
                    store::get_node(&mimir.conn, kept)
                        .unwrap()
                        .deleted_at
                        .is_some(),
                    "hold {hold:?}: a linked row is kept as a tombstone"
                );
            }
            let edges: i64 = mimir
                .conn
                .query_row("SELECT count(*) FROM edge", [], |r| r.get(0))
                .unwrap();
            (pruned, remaining(&mimir.conn), edges)
        };
        let whole = outcome(Duration::MAX);
        assert_eq!(
            whole.0,
            Pruned {
                symbols_removed: 29,
                files_removed: 4,
                symbols_kept: 1,
                files_kept: 2,
                project_removed: false,
                links_moved: 0,
            }
        );
        assert_eq!(whole.2, 2, "only the two authored links are left");
        assert_eq!(outcome(Duration::ZERO), whole, "one row per transaction");
    }

    /// A prune cut off part-way leaves a store the next prune finishes: the
    /// project is still listed, and nothing kept was lost in between.
    #[test]
    fn a_prune_that_was_interrupted_is_finished_by_the_next_one() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let (project, symbol, file) = linked_project(&mut mimir);
        // One batch of symbols and nothing more, as if the process died.
        let derived = derived_rels_sql();
        let gone = mimir
            .conn
            .execute(
                &format!(
                    "DELETE FROM node WHERE id IN (
                         SELECT n.id FROM node n
                         WHERE n.project_id = ?1 AND n.kind = 'symbol'
                           AND NOT EXISTS (SELECT 1 FROM edge e WHERE e.dst = n.id
                                           AND e.rel NOT IN ({derived}))
                         LIMIT 10)"
                ),
                [project.id],
            )
            .unwrap();
        assert_eq!(gone, 10);

        let pruned = prune_graph(&mimir.conn, &project).unwrap();

        assert_eq!(pruned.symbols_removed, 19);
        assert_eq!((pruned.symbols_kept, pruned.files_kept), (1, 2));
        for kept in [symbol, file] {
            assert!(store::get_node(&mimir.conn, kept)
                .unwrap()
                .deleted_at
                .is_some());
        }
    }

    /// Inside a transaction the caller holds, the batches are savepoints of
    /// it: rolling the caller's transaction back undoes the whole prune.
    #[test]
    fn a_prune_inside_a_callers_transaction_is_the_callers_unit() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let (project, _, _) = linked_project(&mut mimir);
        let before = remaining(&mimir.conn);

        mimir.conn.execute_batch("BEGIN IMMEDIATE;").unwrap();
        prune_graph_in_batches(
            &mimir.conn,
            &project,
            std::time::Duration::ZERO,
            PRUNE_YIELD,
        )
        .unwrap();
        assert_ne!(remaining(&mimir.conn), before);
        mimir.conn.execute_batch("ROLLBACK;").unwrap();

        assert_eq!(remaining(&mimir.conn), before);
    }
}
