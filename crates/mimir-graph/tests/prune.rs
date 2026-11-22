//! A project whose directory is gone: listed as missing, and pruned of what
//! was derived for it without touching anything a person wrote or linked.

use std::path::Path;

use mimir_core::grounding::{grounding, Grounding};
use mimir_core::memory::{self, Remember, RememberOutcome};
use mimir_core::model::{Actor, Kind, MemoryType, Node, Rel};
use mimir_core::{store, Mimir};
use mimir_graph::{missing_projects, prune_graph, prune_graph_toward, resolve_symbol, update};
use rusqlite::Connection;

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// A built project with two files: `main → run → slugify`.
fn built(conn: &mut Connection, name: &str) -> (tempfile::TempDir, Node) {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "src/util.rs", "pub fn slugify() {}\n");
    write(
        dir.path(),
        "src/main.rs",
        "fn main() { run(); }\n\nfn run() { slugify(); }\n",
    );
    let project = store::ensure_project(conn, &dir.path().to_string_lossy(), name).unwrap();
    update(conn, &project, dir.path()).unwrap();
    (dir, project)
}

fn remember(conn: &Connection, text: &str, project_id: Option<i64>) -> Node {
    match memory::remember(
        conn,
        Remember {
            text: text.into(),
            mtype: MemoryType::Gotcha,
            tags: Vec::new(),
            project_id,
            force: false,
            actor: Actor::Human,
        },
    )
    .unwrap()
    {
        RememberOutcome::Created(node) => node,
        other => panic!("expected a new memory, got {other:?}"),
    }
}

fn count(conn: &Connection, sql: &str, id: i64) -> i64 {
    conn.query_row(sql, [id], |r| r.get(0)).unwrap()
}

/// Only a project registered with a directory that is no longer there is
/// missing. One that is still on disk is not, however long it has been
/// since anything ran in it.
#[test]
fn a_project_is_missing_once_its_directory_is_gone() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (kept_dir, kept) = built(&mut mimir.conn, "kept");
    let (gone_dir, gone) = built(&mut mimir.conn, "gone");
    assert!(missing_projects(&mimir.conn).unwrap().is_empty());

    drop(gone_dir);
    let missing = missing_projects(&mimir.conn).unwrap();
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].project.id, gone.id);
    assert_eq!(missing[0].symbols, 3);
    assert_eq!(missing[0].files, 2);
    assert_eq!(missing[0].memories, 0);
    assert_eq!(missing[0].unpruned, 5);
    assert!(missing[0].has_graph_rows());
    assert!(kept_dir.path().exists() && kept.id != gone.id);
}

/// With nothing written about it, a missing project leaves nothing behind:
/// its symbols, files, edges and its own row all go, and a project that is
/// still on disk is not touched.
#[test]
fn pruning_a_project_nobody_wrote_about_removes_it_whole() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (_kept_dir, kept) = built(&mut mimir.conn, "kept");
    let (gone_dir, gone) = built(&mut mimir.conn, "gone");
    drop(gone_dir);
    let kept_rows = count(
        &mimir.conn,
        "SELECT count(*) FROM node WHERE project_id = ?1",
        kept.id,
    );
    let edges_before: i64 = mimir
        .conn
        .query_row("SELECT count(*) FROM edge", [], |r| r.get(0))
        .unwrap();

    let pruned = prune_graph(&mimir.conn, &gone).unwrap();

    assert_eq!(pruned.symbols_removed, 3);
    assert_eq!(pruned.files_removed, 2);
    assert_eq!(pruned.symbols_kept, 0);
    assert!(pruned.project_removed);
    assert_eq!(
        count(
            &mimir.conn,
            "SELECT count(*) FROM node WHERE project_id = ?1 OR id = ?1",
            gone.id
        ),
        0
    );
    assert_eq!(
        count(
            &mimir.conn,
            "SELECT count(*) FROM node WHERE project_id = ?1",
            kept.id
        ),
        kept_rows,
        "the project that is still on disk lost rows"
    );
    let edges_after: i64 = mimir
        .conn
        .query_row("SELECT count(*) FROM edge", [], |r| r.get(0))
        .unwrap();
    assert!(
        edges_after < edges_before && edges_after > 0,
        "the gone project's edges went and the kept one's stayed: \
         {edges_before} → {edges_after}"
    );
    assert!(resolve_symbol(&mimir.conn, kept.id, "slugify").is_ok());
    assert!(missing_projects(&mimir.conn).unwrap().is_empty());
}

/// What a person wrote survives a prune. A memory filed under the project
/// keeps the project row, and a symbol someone linked to stays as a
/// tombstone, so the linked memory reads "stale" (the thing it is about is
/// gone) instead of quietly becoming ungrounded.
#[test]
fn pruning_keeps_memories_and_what_they_are_linked_to() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (gone_dir, gone) = built(&mut mimir.conn, "gone");
    let filed = remember(
        &mimir.conn,
        "the release script must run from the repo root",
        Some(gone.id),
    );
    let about = remember(
        &mimir.conn,
        "slugify drops non-ASCII letters instead of transliterating them",
        None,
    );
    let slugify = resolve_symbol(&mimir.conn, gone.id, "slugify").unwrap();
    store::link(&mimir.conn, about.id, slugify.id, Rel::About, 1.0).unwrap();
    assert!(matches!(
        grounding(&mimir.conn, about.id).unwrap(),
        Grounding::Grounded { .. }
    ));
    drop(gone_dir);
    assert_eq!(missing_projects(&mimir.conn).unwrap()[0].memories, 1);

    let pruned = prune_graph(&mimir.conn, &gone).unwrap();

    assert_eq!(pruned.symbols_kept, 1);
    assert_eq!(pruned.symbols_removed, 2);
    assert_eq!(pruned.files_removed, 1, "util.rs still holds slugify");
    assert_eq!(pruned.files_kept, 1);
    assert!(!pruned.project_removed);

    let kept_symbol = store::get_node(&mimir.conn, slugify.id).unwrap();
    assert!(kept_symbol.deleted_at.is_some(), "kept as a tombstone");
    assert!(matches!(
        grounding(&mimir.conn, about.id).unwrap(),
        Grounding::Stale {
            kind: Kind::Symbol,
            ..
        }
    ));
    assert!(store::get_node(&mimir.conn, filed.id)
        .unwrap()
        .deleted_at
        .is_none());
    assert_eq!(
        store::get_node(&mimir.conn, gone.id).unwrap().kind,
        Kind::Project
    );

    // A second prune finds nothing more to remove and changes nothing.
    let again = prune_graph(&mimir.conn, &gone).unwrap();
    assert_eq!(again.symbols_removed, 0);
    assert_eq!(again.files_removed, 0);
    // The directory is still gone and the kept rows are still there, but
    // there is nothing left for a prune to do, and the listing knows it.
    let still = &missing_projects(&mimir.conn).unwrap()[0];
    assert_eq!((still.symbols, still.files, still.memories), (1, 1, 1));
    assert_eq!(still.unpruned, 0);
    assert!(!still.has_graph_rows());
}

/// A memory can be about a file as a whole. That link is as much somebody's
/// work as a link to a symbol, so the file row stays as a tombstone and the
/// memory reads as stale; it used to be deleted, and the link with it.
#[test]
fn pruning_keeps_a_file_someone_linked_to() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (gone_dir, gone) = built(&mut mimir.conn, "gone");
    let about = remember(
        &mimir.conn,
        "main.rs is generated by the scaffold script, do not edit it by hand",
        None,
    );
    let main_rs: i64 = mimir
        .conn
        .query_row(
            "SELECT id FROM node WHERE project_id = ?1 AND kind = 'file' AND path = 'src/main.rs'",
            [gone.id],
            |r| r.get(0),
        )
        .unwrap();
    store::link(&mimir.conn, about.id, main_rs, Rel::About, 1.0).unwrap();
    drop(gone_dir);

    let pruned = prune_graph(&mimir.conn, &gone).unwrap();

    assert_eq!(pruned.files_kept, 1);
    assert_eq!(pruned.files_removed, 1);
    assert_eq!(pruned.symbols_kept, 0);
    assert_eq!(pruned.symbols_removed, 3);
    assert!(!pruned.project_removed);
    assert!(store::get_node(&mimir.conn, main_rs)
        .unwrap()
        .deleted_at
        .is_some());
    assert!(matches!(
        grounding(&mimir.conn, about.id).unwrap(),
        Grounding::Stale {
            kind: Kind::File,
            ..
        }
    ));
}

/// "Gone" is a definite answer from the filesystem. A path this process
/// cannot place (a relative one would be resolved against wherever the
/// command happens to run) is not gone, and a prune must never be offered
/// for it.
#[test]
fn a_relative_project_path_is_not_missing() {
    let mimir = Mimir::open_in_memory().unwrap();
    store::ensure_project(&mimir.conn, "checkouts/relative", "relative").unwrap();
    assert!(missing_projects(&mimir.conn).unwrap().is_empty());
}

/// Nor is a directory this process is not allowed to look at: "permission
/// denied" is not "nothing there".
#[cfg(unix)]
#[test]
fn a_directory_that_cannot_be_checked_is_not_missing() {
    use std::os::unix::fs::PermissionsExt;
    let mimir = Mimir::open_in_memory().unwrap();
    let outer = tempfile::tempdir().unwrap();
    let locked = outer.path().join("locked");
    std::fs::create_dir_all(locked.join("repo")).unwrap();
    store::ensure_project(
        &mimir.conn,
        &locked.join("repo").to_string_lossy(),
        "unreadable",
    )
    .unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    // Root reads through any mode; there is nothing to test then.
    let enforced = locked.join("repo").try_exists().is_err();

    let missing = missing_projects(&mimir.conn);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

    if enforced {
        assert!(missing.unwrap().is_empty());
    }
}

/// The counts come from one grouped read, and a project the store holds
/// nothing for still gets its zeros.
#[test]
fn missing_projects_are_counted_together() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (first_dir, first) = built(&mut mimir.conn, "first");
    let (second_dir, second) = built(&mut mimir.conn, "second");
    remember(
        &mimir.conn,
        "second deploys from the release branch",
        Some(second.id),
    );
    let empty_dir = tempfile::tempdir().unwrap();
    let empty =
        store::ensure_project(&mimir.conn, &empty_dir.path().to_string_lossy(), "empty").unwrap();
    drop((first_dir, second_dir, empty_dir));

    let missing = missing_projects(&mimir.conn).unwrap();
    let of = |id: i64| {
        let m = missing.iter().find(|m| m.project.id == id).unwrap();
        (m.symbols, m.files, m.memories)
    };
    assert_eq!(of(first.id), (3, 2, 0));
    assert_eq!(of(second.id), (3, 2, 1));
    assert_eq!(of(empty.id), (0, 0, 0));
}

/// A memory linked to a symbol of a worktree is about that symbol in the
/// repository. When the worktree is pruned, the link moves to the same
/// symbol of the main checkout's project and the memory stays grounded; a
/// link to something only the worktree had keeps its tombstone and reads as
/// stale, as before.
#[test]
fn pruning_a_worktree_moves_its_links_to_the_same_code_at_home() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (_home_dir, home) = built(&mut mimir.conn, "home");
    let (wt_dir, worktree) = built(&mut mimir.conn, "home-wt");
    write(
        wt_dir.path(),
        "src/branch.rs",
        "pub fn only_on_the_branch() {}\n",
    );
    update(&mut mimir.conn, &worktree, wt_dir.path()).unwrap();

    let shared = remember(
        &mimir.conn,
        "slugify drops non-ASCII letters instead of transliterating them",
        Some(home.id),
    );
    let branch_only = remember(
        &mimir.conn,
        "only_on_the_branch is a stub until the parser lands",
        Some(home.id),
    );
    let on_wt = |name: &str| resolve_symbol(&mimir.conn, worktree.id, name).unwrap();
    store::link(&mimir.conn, shared.id, on_wt("slugify").id, Rel::About, 1.0).unwrap();
    store::link(
        &mimir.conn,
        branch_only.id,
        on_wt("only_on_the_branch").id,
        Rel::About,
        1.0,
    )
    .unwrap();
    drop(wt_dir);

    let pruned = prune_graph_toward(&mimir.conn, &worktree, Some(&home)).unwrap();

    assert_eq!(pruned.links_moved, 1);
    assert_eq!(
        pruned.symbols_kept, 1,
        "only the branch-only symbol is kept"
    );
    let home_slugify = resolve_symbol(&mimir.conn, home.id, "slugify").unwrap();
    let linked_to: i64 = mimir
        .conn
        .query_row(
            "SELECT dst FROM edge WHERE src = ?1 AND rel = 'about'",
            [shared.id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(linked_to, home_slugify.id);
    assert!(matches!(
        grounding(&mimir.conn, shared.id).unwrap(),
        Grounding::Grounded { .. }
    ));
    assert!(matches!(
        grounding(&mimir.conn, branch_only.id).unwrap(),
        Grounding::Stale { .. }
    ));

    // Without a home it is the plain prune: the link keeps its tombstone.
    let mut plain = Mimir::open_in_memory().unwrap();
    let (dir, project) = built(&mut plain.conn, "alone");
    let note = remember(&plain.conn, "slugify is called from run", None);
    let slugify = resolve_symbol(&plain.conn, project.id, "slugify").unwrap();
    store::link(&plain.conn, note.id, slugify.id, Rel::About, 1.0).unwrap();
    drop(dir);
    let alone = prune_graph(&plain.conn, &project).unwrap();
    assert_eq!((alone.links_moved, alone.symbols_kept), (0, 1));
}

/// A project row that a live linked worktree names as its home is not
/// removed by a prune, even when nothing else refers to it: the worktree's
/// memories are filed under it.
#[test]
fn a_home_a_worktree_still_points_at_is_kept() {
    let mut mimir = Mimir::open_in_memory().unwrap();
    let (home_dir, home) = built(&mut mimir.conn, "home");
    let worktree = store::ensure_project(&mimir.conn, "/tmp/home-wt-kept", "home-wt").unwrap();
    store::set_worktree_home(&mimir.conn, worktree.id, Some(&home)).unwrap();
    drop(home_dir);

    let pruned = prune_graph(&mimir.conn, &home).unwrap();
    assert!(!pruned.project_removed);
    assert_eq!(
        store::get_node(&mimir.conn, home.id).unwrap().kind,
        Kind::Project
    );

    // Once no worktree points at it, it goes like any other.
    store::set_worktree_home(&mimir.conn, worktree.id, None).unwrap();
    assert!(prune_graph(&mimir.conn, &home).unwrap().project_removed);
}
