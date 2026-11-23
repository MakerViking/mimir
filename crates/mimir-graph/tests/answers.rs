//! What a graph query says: the one rendering the CLI and the MCP tool share,
//! and the two things an answer must never leave out — that a result is a
//! guess, and that a symbol belongs to some other project.

use std::path::Path;

use mimir_core::model::Node;
use mimir_core::{store, Mimir};
use mimir_graph::render::{self, Direction};
use mimir_graph::{resolve_symbol, update, CodeGraph};
use rusqlite::Connection;

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn project(conn: &Connection, root: &Path, name: &str) -> Node {
    store::ensure_project(conn, &root.to_string_lossy(), name).unwrap()
}

/// `Database::save` and `Cache::save`; `typed` reaches the first through its
/// receiver's type, `guess` reaches both through a receiver nobody can type,
/// and `outer` calls `guess`.
fn two_saves() -> (tempfile::TempDir, Mimir, Node) {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/db.rs",
        "pub struct Database;\nimpl Database { pub fn save(&self) -> u8 { 1 } }\n",
    );
    write(
        dir.path(),
        "src/cache.rs",
        "pub struct Cache;\nimpl Cache { pub fn save(&self) -> u8 { 2 } }\n",
    );
    write(
        dir.path(),
        "src/app.rs",
        "fn typed() {\n    let db = Database::new();\n    db.save();\n}\n\
         fn guess(m: Mystery) { m.save(); }\n\
         fn outer(m: Mystery) { guess(m); }\n",
    );
    let mut mimir = Mimir::open_in_memory().unwrap();
    let proj = project(&mimir.conn, dir.path(), "probe");
    update(&mut mimir.conn, &proj, dir.path()).unwrap();
    (dir, mimir, proj)
}

/// Strip the id column, which differs per run: `c:XXXXXX [fn rust] …`.
fn without_ids(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| {
            let indent = line.len() - line.trim_start().len();
            let rest = line.trim_start();
            let rest = match rest.split_once(' ') {
                Some((id, tail)) if id.starts_with("c:") || id.starts_with("d:") => tail,
                _ => rest,
            };
            format!("{}{rest}", " ".repeat(indent))
        })
        .collect()
}

/// The lines of an answer in a fixed order. Results at the same distance
/// come out in the order the store happens to list their edges, which no
/// reader relies on; the header stays first and a count stays last because
/// neither is indented and both sort outside the `[`-led result lines.
fn any_order(text: &str) -> Vec<String> {
    let mut lines = without_ids(text);
    lines.sort();
    lines
}

fn sorted<const N: usize>(mut lines: [&str; N]) -> Vec<String> {
    lines.sort();
    lines.iter().map(|l| l.to_string()).collect()
}

/// A caller reached only through a one-in-two guess is listed, and labelled.
/// So is everything reached through it. A caller with a resolved edge is not.
#[test]
fn a_result_reached_only_through_a_guess_says_so() {
    let (_dir, mimir, proj) = two_saves();
    let graph = CodeGraph::load(&mimir.conn, proj.id).unwrap();
    let db_save = resolve_symbol(&mimir.conn, proj.id, "Database::save").unwrap();
    let cache_save = resolve_symbol(&mimir.conn, proj.id, "Cache::save").unwrap();

    let callers = render::relation(&mimir.conn, &graph, &db_save, 3, Direction::Callers).unwrap();
    assert_eq!(
        without_ids(&callers)[0],
        "[method rust] src/db.rs:2-2 Database::save",
        "the symbol asked about comes first"
    );
    assert_eq!(
        any_order(&callers),
        sorted([
            "[method rust] src/db.rs:2-2 Database::save",
            "  [function rust] src/app.rs:1-4 typed",
            "  [function rust] src/app.rs:5-5 guess (unconfirmed call)",
            "    [function rust] src/app.rs:6-6 outer (unconfirmed call)",
        ])
    );
    // `Cache::save` has no confirmed caller at all.
    let callers =
        render::relation(&mimir.conn, &graph, &cache_save, 1, Direction::Callers).unwrap();
    assert_eq!(
        without_ids(&callers),
        [
            "[method rust] src/cache.rs:2-2 Cache::save",
            "  [function rust] src/app.rs:5-5 guess (unconfirmed call)",
        ]
    );

    // The other direction: `outer → guess` is certain, what `guess` calls is not.
    let outer = resolve_symbol(&mimir.conn, proj.id, "outer").unwrap();
    let calls = render::relation(&mimir.conn, &graph, &outer, 3, Direction::Calls).unwrap();
    assert_eq!(
        any_order(&calls),
        sorted([
            "[function rust] src/app.rs:6-6 outer",
            "  [function rust] src/app.rs:5-5 guess",
            "    [method rust] src/cache.rs:2-2 Cache::save (unconfirmed call)",
            "    [method rust] src/db.rs:2-2 Database::save (unconfirmed call)",
        ])
    );

    // And the flag itself, for a caller that reads the graph directly.
    let traced = graph.callers_traced(db_save.id, 3);
    let typed = resolve_symbol(&mimir.conn, proj.id, "typed").unwrap();
    let guess = resolve_symbol(&mimir.conn, proj.id, "guess").unwrap();
    assert!(traced.iter().any(|t| t.id == typed.id && t.confirmed));
    assert!(traced.iter().any(|t| t.id == guess.id && !t.confirmed));
    assert!(graph.call_is_confirmed(typed.id, db_save.id));
    assert!(!graph.call_is_confirmed(guess.id, db_save.id));
}

/// One confirmed route is enough: a symbol reached both through a guess and
/// through resolved edges is not a guess.
#[test]
fn a_result_with_one_resolved_route_is_confirmed() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/db.rs",
        "pub struct Database;\nimpl Database { pub fn save(&self) -> u8 { 1 } }\n",
    );
    write(
        dir.path(),
        "src/cache.rs",
        "pub struct Cache;\nimpl Cache { pub fn save(&self) -> u8 { 2 } }\n",
    );
    // `both` calls `save` twice: once typed, once not.
    write(
        dir.path(),
        "src/app.rs",
        "fn both(m: Mystery) {\n    m.save();\n    let db = Database::new();\n    db.save();\n}\n",
    );
    let mut mimir = Mimir::open_in_memory().unwrap();
    let proj = project(&mimir.conn, dir.path(), "probe");
    update(&mut mimir.conn, &proj, dir.path()).unwrap();
    let graph = CodeGraph::load(&mimir.conn, proj.id).unwrap();
    let db_save = resolve_symbol(&mimir.conn, proj.id, "Database::save").unwrap();
    let cache_save = resolve_symbol(&mimir.conn, proj.id, "Cache::save").unwrap();
    assert!(graph.callers_traced(db_save.id, 2)[0].confirmed);
    assert!(!graph.callers_traced(cache_save.id, 2)[0].confirmed);
}

/// `impact` carries the same label, and counts what it lists as what it is.
#[test]
fn impact_labels_guesses_and_counts_symbols_and_files_apart() {
    let (dir, mut mimir, proj) = two_saves();
    // A file that only imports cache.rs: listed as a file, not as a symbol.
    write(
        dir.path(),
        "src/uses.rs",
        "use crate::cache::Cache;\n\npub const N: u8 = 1;\n",
    );
    update(&mut mimir.conn, &proj, dir.path()).unwrap();
    let graph = CodeGraph::load(&mimir.conn, proj.id).unwrap();

    let trace = graph
        .impact_of_paths_traced(&mimir.conn, &["src/cache.rs".to_string()], 3)
        .unwrap();
    let text = render::impact(&trace, 3).unwrap();
    assert_eq!(
        text.lines().last(),
        Some("(2 affected symbols and 1 importing file(s) within 3 hops)")
    );
    assert_eq!(
        any_order(&text),
        sorted([
            "[function rust] src/app.rs:5-5 guess (unconfirmed call)",
            "[file rust] src/uses.rs (reached by import only)",
            "  [function rust] src/app.rs:6-6 outer (unconfirmed call)",
            "(2 affected symbols and 1 importing file(s) within 3 hops)",
        ])
    );

    let trace = graph
        .impact_of_paths_traced(&mimir.conn, &["src/db.rs".to_string()], 1)
        .unwrap();
    let text = render::impact(&trace, 1).unwrap();
    assert_eq!(
        any_order(&text),
        sorted([
            "[function rust] src/app.rs:1-4 typed",
            "[function rust] src/app.rs:5-5 guess (unconfirmed call)",
            "(2 affected symbols within 1 hops)",
        ])
    );

    // The two answers that are not a list.
    let contained = graph
        .impact_of_paths_traced(&mimir.conn, &["src/app.rs".to_string()], 3)
        .unwrap();
    assert_eq!(
        render::impact(&contained, 3).as_deref(),
        Some("no impact outside the changed files")
    );
    let missing = graph
        .impact_of_paths_traced(&mimir.conn, &["src/nope.rs".to_string()], 3)
        .unwrap();
    assert_eq!(render::impact(&missing, 3), None);
    assert_eq!(
        render::NOT_IN_GRAPH,
        "none of those paths are in the code graph: they are not source files Mimir indexes, \
         or they were added since the last build"
    );
}

/// The words the CLI and the MCP tool share for an empty answer. They
/// used to differ: "no callees found" against "no calls found", and a
/// missing path that named its ends on one surface only.
#[test]
fn the_empty_answers_have_one_wording() {
    let (_dir, mimir, proj) = two_saves();
    let graph = CodeGraph::load(&mimir.conn, proj.id).unwrap();
    let outer = resolve_symbol(&mimir.conn, proj.id, "outer").unwrap();
    let db_save = resolve_symbol(&mimir.conn, proj.id, "Database::save").unwrap();
    let typed = resolve_symbol(&mimir.conn, proj.id, "typed").unwrap();

    let callers = render::relation(&mimir.conn, &graph, &outer, 3, Direction::Callers).unwrap();
    assert_eq!(
        without_ids(&callers),
        ["[function rust] src/app.rs:6-6 outer", "no callers found"]
    );
    let calls = render::relation(&mimir.conn, &graph, &db_save, 3, Direction::Calls).unwrap();
    assert_eq!(
        without_ids(&calls),
        [
            "[method rust] src/db.rs:2-2 Database::save",
            "no callees found"
        ]
    );
    assert_eq!(
        render::path(&mimir.conn, &graph, &db_save, &outer).unwrap(),
        "no call path Database::save → outer"
    );

    // A path, one hop per line; a hop over a guessed edge is labelled.
    let path = render::path(&mimir.conn, &graph, &typed, &db_save).unwrap();
    assert_eq!(
        without_ids(&path),
        [
            "[function rust] src/app.rs:1-4 typed",
            "  [method rust] src/db.rs:2-2 Database::save",
        ]
    );
    let path = render::path(&mimir.conn, &graph, &outer, &db_save).unwrap();
    assert_eq!(
        without_ids(&path),
        [
            "[function rust] src/app.rs:6-6 outer",
            "  [function rust] src/app.rs:5-5 guess",
            "    [method rust] src/db.rs:2-2 Database::save (unconfirmed call)",
        ]
    );

    let hubs = render::hubs(&mimir.conn, &graph, 10).unwrap();
    assert!(hubs.lines().all(|l| l.starts_with('↑')), "{hubs}");
    let empty = Mimir::open_in_memory().unwrap();
    let nothing = project(&empty.conn, Path::new("/nowhere"), "empty");
    let graph = CodeGraph::load(&empty.conn, nothing.id).unwrap();
    assert_eq!(
        render::hubs(&empty.conn, &graph, 10).unwrap(),
        "no call edges yet (run `mimir graph build`)"
    );
}

/// `node`: signature, then edges. A guessed call edge is labelled; a memory
/// linked to the symbol is listed by id and title.
#[test]
fn node_lists_edges_and_labels_a_guessed_call() {
    let (_dir, mimir, proj) = two_saves();
    let guess = resolve_symbol(&mimir.conn, proj.id, "guess").unwrap();
    let memory = match mimir_core::memory::remember(
        &mimir.conn,
        mimir_core::memory::Remember {
            text: "guess is called on the hot path".into(),
            mtype: mimir_core::model::MemoryType::Note,
            tags: vec![],
            project_id: Some(proj.id),
            force: false,
            actor: mimir_core::model::Actor::System,
        },
    )
    .unwrap()
    {
        mimir_core::memory::RememberOutcome::Created(n) => n,
        other => panic!("{other:?}"),
    };
    store::link(
        &mimir.conn,
        memory.id,
        guess.id,
        mimir_core::model::Rel::About,
        1.0,
    )
    .unwrap();

    let text = render::node(&mimir.conn, &guess).unwrap();
    let lines = without_ids(&text);
    assert_eq!(lines[0], "[function rust] src/app.rs:5-5 guess");
    assert_eq!(lines[1], "fn guess(m: Mystery)");
    let edges: std::collections::BTreeSet<&str> = text.lines().skip(2).collect();
    let has = |needle: &str| edges.iter().any(|l| l.contains(needle));
    assert!(
        has("src/db.rs:2-2 Database::save (unconfirmed call)") && has("→ calls"),
        "{text}"
    );
    assert!(
        has("src/cache.rs:2-2 Cache::save (unconfirmed call)"),
        "{text}"
    );
    assert!(has("← calls") && has("outer"), "{text}");
    assert!(
        edges
            .iter()
            .any(|l| l.starts_with("← about m:") && l.contains("guess is called")),
        "{text}"
    );
    assert!(
        !edges
            .iter()
            .any(|l| l.contains("outer") && l.contains("unconfirmed")),
        "{text}"
    );
}

/// A symbol id is global. Asked about from another project it used to
/// resolve, miss that project's graph, and come back "no callers found".
#[test]
fn a_symbol_id_from_another_project_is_named_as_such() {
    let here = tempfile::tempdir().unwrap();
    let there = tempfile::tempdir().unwrap();
    write(here.path(), "src/a.rs", "pub fn local() {}\n");
    write(
        there.path(),
        "src/b.rs",
        "pub fn elsewhere() {}\nfn user() { elsewhere(); }\n",
    );
    let mut mimir = Mimir::open_in_memory().unwrap();
    let ours = project(&mimir.conn, here.path(), "ours");
    let theirs = project(&mimir.conn, there.path(), "theirs");
    update(&mut mimir.conn, &ours, here.path()).unwrap();
    update(&mut mimir.conn, &theirs, there.path()).unwrap();

    let foreign = resolve_symbol(&mimir.conn, theirs.id, "elsewhere").unwrap();
    let id = mimir_core::model::short_uid(foreign.kind, &foreign.uid);

    let err = resolve_symbol(&mimir.conn, ours.id, &id).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("{id} belongs to project theirs — run this from that checkout")
    );
    // From its own project the id resolves as before.
    assert_eq!(
        resolve_symbol(&mimir.conn, theirs.id, &id).unwrap().id,
        foreign.id
    );
    // A local id, and a name, are unaffected.
    let local = resolve_symbol(&mimir.conn, ours.id, "local").unwrap();
    let local_id = mimir_core::model::short_uid(local.kind, &local.uid);
    assert_eq!(
        resolve_symbol(&mimir.conn, ours.id, &local_id).unwrap().id,
        local.id
    );
    assert!(resolve_symbol(&mimir.conn, ours.id, "elsewhere").is_err());
}

/// Communities are groups of symbols. File nodes sit in the same graph (for
/// `imports`) and used to come out as communities of one; and the order of
/// groups and members came from hash iteration.
#[test]
fn communities_hold_symbols_only_in_a_stable_order() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/a.rs",
        "pub fn a1() { a2(); a3(); }\npub fn a2() { a3(); }\npub fn a3() {}\n",
    );
    write(
        dir.path(),
        "src/b.rs",
        "pub fn b1() { b2(); }\npub fn b2() {}\npub fn alone() {}\n",
    );
    let mut mimir = Mimir::open_in_memory().unwrap();
    let proj = project(&mimir.conn, dir.path(), "probe");
    update(&mut mimir.conn, &proj, dir.path()).unwrap();

    let symbol_ids: std::collections::HashSet<i64> = {
        let mut stmt = mimir
            .conn
            .prepare("SELECT id FROM node WHERE kind = 'symbol'")
            .unwrap();
        let rows = stmt.query_map([], |r| r.get(0)).unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    };
    let groups = CodeGraph::load(&mimir.conn, proj.id)
        .unwrap()
        .communities(1);
    assert_eq!(
        groups.iter().map(Vec::len).collect::<Vec<_>>(),
        [3, 2, 1],
        "largest first; the two files are not among them: {groups:?}"
    );
    for group in &groups {
        assert!(group.iter().all(|id| symbol_ids.contains(id)), "{group:?}");
        assert!(group.windows(2).all(|w| w[0] < w[1]), "members by id");
    }
    for _ in 0..5 {
        let again = CodeGraph::load(&mimir.conn, proj.id)
            .unwrap()
            .communities(1);
        assert_eq!(again, groups);
    }
}
