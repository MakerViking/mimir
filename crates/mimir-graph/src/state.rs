//! Whether a project has a code graph, decided once.
//!
//! Four places used to answer this by counting the project's symbols. That
//! cannot tell a project whose graph was never built from one that was built
//! and found nothing to index — so in a project with no code, every query
//! announced "building the code graph (first run for this project)…" and
//! built it again.
//!
//! The answer is a marker: an update that commits writes the time into the
//! project node's `meta.graph_built_at`. A store written before the marker
//! existed has none, so live symbols still count as proof of a build.
//!
//! # Which rules built it
//!
//! A graph is also the product of the rules that resolved its edges, and
//! those change between releases: which file an import names, when a call is
//! a match and when it is a guess. An incremental update applies the current
//! rules only to the files it re-reads, so after an upgrade a graph is a
//! mixture, and nothing in it says so — `graph check` compares files, and
//! the files have not changed.
//!
//! So the project node also carries `meta.graph_rules`: the rules version
//! ([`GRAPH_RULES`]) of the last build that re-extracted every file. When it
//! is missing or older than this binary's, [`rules_note`] is the line a
//! graph answer adds, naming the command that rebuilds it. A small project
//! does not wait for that command: its next ordinary build is a full one
//! ([`wants_full_rebuild`]), and says so. A large one is never rebuilt
//! unasked.

use mimir_core::error::Result;
use mimir_core::model::{now_unix, Node};
use mimir_core::store::{row_to_node, NODE_COLS};
use rusqlite::Connection;

/// The version of the call and import resolution rules this build applies.
/// Raised whenever a change to them would give an unchanged source tree
/// different edges, so that graphs resolved by the previous ones can be told
/// apart. 1 is the first version that is recorded at all.
pub const GRAPH_RULES: i64 = 1;

/// Record that an update of `project_id` committed now. Called inside the
/// update's transaction, so the marker and the graph it vouches for land
/// together or not at all.
///
/// `every_file` says the update extracted every file the project has (its
/// first build, or a `--full` one): only then is the whole graph the work of
/// this build's rules, and only then is their version stamped.
pub(crate) fn mark_built(conn: &Connection, project_id: i64, every_file: bool) -> Result<()> {
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.graph_built_at', ?2) WHERE id = ?1",
        rusqlite::params![project_id, now_unix()],
    )?;
    if every_file {
        conn.execute(
            "UPDATE node SET meta = json_set(meta, '$.graph_rules', ?2) WHERE id = ?1",
            rusqlite::params![project_id, GRAPH_RULES],
        )?;
    }
    Ok(())
}

/// Withdraw the rules stamp: an update found part of the graph to be the
/// work of an older build (which can share the store, and rebuilds files by
/// its own rules without knowing a stamp exists).
pub(crate) fn unstamp_rules(conn: &Connection, project_id: i64) -> Result<()> {
    conn.execute(
        "UPDATE node SET meta = json_remove(meta, '$.graph_rules') WHERE id = ?1",
        [project_id],
    )?;
    Ok(())
}

/// The rules version the project's whole graph was last built with, or
/// `None` when no build has recorded one (a graph from before versions were
/// recorded, or no graph).
pub fn graph_rules(conn: &Connection, project_id: i64) -> Result<Option<i64>> {
    Ok(conn.query_row(
        "SELECT json_extract(meta, '$.graph_rules') FROM node WHERE id = ?1",
        [project_id],
        |r| Ok(r.get::<_, Option<i64>>(0).unwrap_or(None)),
    )?)
}

/// The line a graph answer adds when the project's graph was resolved, in
/// whole or in part, by an older build's rules. `None` when the graph is
/// this build's work, when a newer build made it (this one cannot improve
/// on that), and when there is no graph to speak of.
pub fn rules_note(conn: &Connection, project_id: i64) -> Result<Option<String>> {
    if graph_rules(conn, project_id)?.is_some_and(|built_with| built_with >= GRAPH_RULES) {
        return Ok(None);
    }
    if !has_graph(conn, project_id)? {
        return Ok(None);
    }
    Ok(Some(
        "note: this graph was built with an older Mimir's call and import rules — \
         `mimir graph build --full` rebuilds it with the current ones"
            .to_string(),
    ))
}

/// The most code files a project may have for an ordinary build to rebuild
/// its whole graph unasked (see [`wants_full_rebuild`]). Above it the cost
/// is the user's to choose, and the note from [`rules_note`] asks them to.
pub const AUTO_FULL_MAX_FILES: i64 = 1_500;

/// The same limit counted in symbols, because files are a poor measure of
/// work: a project of a thousand generated files can hold a hundred
/// thousand symbols. A full build is one transaction, so its length is how
/// long every other writer to the store waits; both limits together keep an
/// unasked rebuild to a few seconds on ordinary hardware, well inside the
/// time another process will wait for the lock.
pub const AUTO_FULL_MAX_SYMBOLS: i64 = 15_000;

/// Whether an ordinary build of this project should be a full one: its
/// graph was resolved by an older build's rules ([`rules_note`]) and it is
/// small enough that rebuilding is cheaper than leaving a note on every
/// answer until somebody acts on it. After an upgrade this is how most
/// projects move to the current rules without anyone doing anything: the
/// MCP server's start-up build and `mimir graph build` both ask.
pub fn wants_full_rebuild(conn: &Connection, project_id: i64) -> Result<bool> {
    if rules_note(conn, project_id)?.is_none() {
        return Ok(false);
    }
    let files: i64 = conn.query_row(
        &format!(
            "SELECT count(*) FROM (SELECT 1 FROM node WHERE {} LIMIT ?2)",
            crate::store_graph::LIVE_CODE_FILE
        ),
        rusqlite::params![project_id, AUTO_FULL_MAX_FILES + 1],
        |r| r.get(0),
    )?;
    if files > AUTO_FULL_MAX_FILES {
        return Ok(false);
    }
    let symbols: i64 = conn.query_row(
        "SELECT count(*) FROM (SELECT 1 FROM node
                               WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL
                               LIMIT ?2)",
        rusqlite::params![project_id, AUTO_FULL_MAX_SYMBOLS + 1],
        |r| r.get(0),
    )?;
    Ok(symbols <= AUTO_FULL_MAX_SYMBOLS)
}

/// When the project's graph was last updated (unix seconds), or `None` for a
/// project that has never been built — or was last built before the marker
/// was recorded.
pub fn graph_built_at(conn: &Connection, project_id: i64) -> Result<Option<i64>> {
    Ok(conn.query_row(
        "SELECT json_extract(meta, '$.graph_built_at') FROM node WHERE id = ?1",
        [project_id],
        |r| r.get::<_, Option<i64>>(0),
    )?)
}

/// Does this project have a code graph? True once an update has committed
/// for it, however little that update found.
pub fn has_graph(conn: &Connection, project_id: i64) -> Result<bool> {
    if graph_built_at(conn, project_id)?.is_some() {
        return Ok(true);
    }
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM node
                        WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL)",
        [project_id],
        |r| r.get(0),
    )?)
}

/// Every live project that has a graph, by title: what `--all-projects`
/// ranges over.
pub fn graphed_projects(conn: &Connection) -> Result<Vec<Node>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'project' AND deleted_at IS NULL ORDER BY title, id"
    ))?;
    let projects: Vec<Node> = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for project in projects {
        if has_graph(conn, project.id)? {
            out.push(project);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::model::{Kind, NewNode};
    use mimir_core::{store, Mimir};

    /// The three states, and the one the count could not tell apart: built,
    /// with nothing in it.
    #[test]
    fn built_and_empty_is_not_never_built() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let empty = tempfile::tempdir().unwrap();
        std::fs::write(empty.path().join("README.txt"), "no code here\n").unwrap();
        let proj =
            store::ensure_project(&mimir.conn, &empty.path().to_string_lossy(), "empty").unwrap();

        assert!(!has_graph(&mimir.conn, proj.id).unwrap(), "never built");
        assert_eq!(graph_built_at(&mimir.conn, proj.id).unwrap(), None);
        assert!(graphed_projects(&mimir.conn).unwrap().is_empty());

        let stats = crate::update(&mut mimir.conn, &proj, empty.path()).unwrap();
        assert_eq!(stats.symbols, 0);
        assert!(has_graph(&mimir.conn, proj.id).unwrap(), "built, and empty");
        assert!(graph_built_at(&mimir.conn, proj.id).unwrap().is_some());
        let listed = graphed_projects(&mimir.conn).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, proj.id);
    }

    /// A store built before the marker existed: symbols, no marker. It has a
    /// graph, and must not be rebuilt as a "first run".
    #[test]
    fn symbols_without_a_marker_still_count() {
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = store::ensure_project(&mimir.conn, "/somewhere/old", "old").unwrap();
        let mut sym = NewNode::new(Kind::Symbol);
        sym.title = Some("legacy".into());
        sym.project_id = Some(proj.id);
        store::insert_node(&mimir.conn, sym).unwrap();

        assert_eq!(graph_built_at(&mimir.conn, proj.id).unwrap(), None);
        assert!(has_graph(&mimir.conn, proj.id).unwrap());
    }

    /// The rules stamp through a graph's life: written by the first build,
    /// absent on a graph from before it existed (which an ordinary build
    /// does not fix, since it re-reads only what changed), written by a full
    /// rebuild, and stale once this build's rules are newer.
    #[test]
    fn the_rules_note_follows_the_stamp() {
        use crate::{update_with, UpdateOptions};
        const NOTE: &str = "note: this graph was built with an older Mimir's call and import \
                            rules — `mimir graph build --full` rebuilds it with the current ones";
        let mut mimir = Mimir::open_in_memory().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let proj =
            store::ensure_project(&mimir.conn, &dir.path().to_string_lossy(), "probe").unwrap();
        let set_stamp = |conn: &Connection, sql: &str| {
            conn.execute(
                &format!("UPDATE node SET meta = {sql} WHERE id = ?1"),
                [proj.id],
            )
            .unwrap();
        };

        // No graph: nothing to say about its rules.
        assert_eq!(rules_note(&mimir.conn, proj.id).unwrap(), None);

        // Stamped by its first build.
        crate::update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(
            graph_rules(&mimir.conn, proj.id).unwrap(),
            Some(GRAPH_RULES)
        );
        assert_eq!(rules_note(&mimir.conn, proj.id).unwrap(), None);

        // Unstamped: a graph from a build that recorded no version.
        set_stamp(&mimir.conn, "json_remove(meta, '$.graph_rules')");
        assert_eq!(
            rules_note(&mimir.conn, proj.id).unwrap().as_deref(),
            Some(NOTE)
        );
        // An ordinary build re-reads only what changed; the rest of the
        // graph is still the older build's.
        std::fs::write(dir.path().join("b.rs"), "pub fn b() {}\n").unwrap();
        crate::update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(graph_rules(&mimir.conn, proj.id).unwrap(), None);
        assert_eq!(
            rules_note(&mimir.conn, proj.id).unwrap().as_deref(),
            Some(NOTE)
        );
        // A full one is this build's work throughout.
        update_with(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions { full: true },
        )
        .unwrap();
        assert_eq!(
            graph_rules(&mimir.conn, proj.id).unwrap(),
            Some(GRAPH_RULES)
        );
        assert_eq!(rules_note(&mimir.conn, proj.id).unwrap(), None);

        // Stale: stamped by rules older than this build's.
        set_stamp(
            &mimir.conn,
            &format!("json_set(meta, '$.graph_rules', {})", GRAPH_RULES - 1),
        );
        assert_eq!(
            rules_note(&mimir.conn, proj.id).unwrap().as_deref(),
            Some(NOTE)
        );
        // Stamped by a newer build: not something this one can rebuild better.
        set_stamp(
            &mimir.conn,
            &format!("json_set(meta, '$.graph_rules', {})", GRAPH_RULES + 1),
        );
        assert_eq!(rules_note(&mimir.conn, proj.id).unwrap(), None);
    }

    /// A graph that has symbols and no marker at all — one a release before
    /// the marker built — has no stamp either, and says so.
    #[test]
    fn a_graph_older_than_the_marker_is_built_with_older_rules() {
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = store::ensure_project(&mimir.conn, "/somewhere/old", "old").unwrap();
        let mut sym = NewNode::new(Kind::Symbol);
        sym.title = Some("legacy".into());
        sym.project_id = Some(proj.id);
        store::insert_node(&mimir.conn, sym).unwrap();
        assert!(rules_note(&mimir.conn, proj.id).unwrap().is_some());
    }

    /// After an upgrade most projects should reach the current rules with
    /// nobody doing anything: a small project whose graph an older build
    /// resolved is rebuilt in full by an ordinary build. A large one is not
    /// (that cost is the user's to choose), and a graph already on this
    /// build's rules never is.
    #[test]
    fn a_small_project_on_older_rules_wants_a_full_rebuild() {
        use crate::{update_with, UpdateOptions};
        let mut mimir = Mimir::open_in_memory().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let proj =
            store::ensure_project(&mimir.conn, &dir.path().to_string_lossy(), "probe").unwrap();
        assert!(
            !wants_full_rebuild(&mimir.conn, proj.id).unwrap(),
            "no graph"
        );

        crate::update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(
            !wants_full_rebuild(&mimir.conn, proj.id).unwrap(),
            "built by this build's rules"
        );

        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.graph_rules') WHERE id = ?1",
                [proj.id],
            )
            .unwrap();
        assert!(wants_full_rebuild(&mimir.conn, proj.id).unwrap());

        // Past the size where a rebuild is quick, it is left to `--full`.
        let mut insert = mimir
            .conn
            .prepare(
                "INSERT INTO node (uid, kind, project_id, path, created_at, updated_at)
                 VALUES (?1, 'file', ?2, ?3, 0, 0)",
            )
            .unwrap();
        for i in 0..AUTO_FULL_MAX_FILES {
            insert
                .execute(rusqlite::params![
                    format!("F{i:025}"),
                    proj.id,
                    format!("gen/f{i}.rs")
                ])
                .unwrap();
        }
        drop(insert);
        assert!(!wants_full_rebuild(&mimir.conn, proj.id).unwrap());
        mimir
            .conn
            .execute("DELETE FROM node WHERE path LIKE 'gen/%'", [])
            .unwrap();
        assert!(wants_full_rebuild(&mimir.conn, proj.id).unwrap());

        // Few files can still be a great many symbols; that is left too.
        let mut insert = mimir
            .conn
            .prepare(
                "INSERT INTO node (uid, kind, project_id, path, title, created_at, updated_at)
                 VALUES (?1, 'symbol', ?2, 'gen/big.rs', ?3, 0, 0)",
            )
            .unwrap();
        for i in 0..AUTO_FULL_MAX_SYMBOLS {
            insert
                .execute(rusqlite::params![
                    format!("S{i:025}"),
                    proj.id,
                    format!("generated_{i}")
                ])
                .unwrap();
        }
        drop(insert);
        assert!(!wants_full_rebuild(&mimir.conn, proj.id).unwrap());
        mimir
            .conn
            .execute("DELETE FROM node WHERE path LIKE 'gen/%'", [])
            .unwrap();
        assert!(wants_full_rebuild(&mimir.conn, proj.id).unwrap());

        // The full build it asked for settles it.
        update_with(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions { full: true },
        )
        .unwrap();
        assert!(!wants_full_rebuild(&mimir.conn, proj.id).unwrap());
        assert_eq!(rules_note(&mimir.conn, proj.id).unwrap(), None);
    }

    /// The marker is added to the project's meta, beside whatever is there:
    /// the sync identity lives in the same object.
    #[test]
    fn the_marker_leaves_the_rest_of_the_project_meta_alone() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let proj =
            store::ensure_project(&mimir.conn, &dir.path().to_string_lossy(), "keyed").unwrap();
        store::set_project_identity(&mimir.conn, proj.id, Some("github.com/x/y"), true).unwrap();

        crate::update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let after = store::get_node(&mimir.conn, proj.id).unwrap();
        assert_eq!(
            after.meta.get("portable_key").and_then(|v| v.as_str()),
            Some("github.com/x/y")
        );
        assert_eq!(after.meta.get("sync").and_then(|v| v.as_bool()), Some(true));
        assert!(after.meta.get("graph_built_at").is_some());
    }
}
