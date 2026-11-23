//! The one promise an incremental graph has to keep: after any sequence of
//! edits, `update` leaves exactly the symbols and edges that building the
//! final tree from nothing would. An agent asks `callers` to decide what a
//! change touches; an edge that is only right "if you had rebuilt" is a wrong
//! answer that looks like a right one.
//!
//! Each scenario applies its edits one step at a time to a single store, and
//! after every step builds the same tree into a second, empty store. The two
//! are compared by what a reader can see — path, qualified name, kind, lines
//! and signature of every symbol; both ends, relation, weight and tier of
//! every edge — never by row id, which legitimately differs.

use std::path::Path;

use mimir_core::model::{Kind, NewNode, Node, Rel};
use mimir_core::{store, Mimir};
use mimir_graph::{update, update_with, UpdateOptions, UpdateReport};
use rusqlite::Connection;

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn project(conn: &Connection, root: &Path) -> Node {
    store::ensure_project(conn, &root.to_string_lossy(), "probe").unwrap()
}

/// Live symbols, one line each, sorted: where it is, what it is, and what
/// the row says about it (lines, bare name, signature and doc).
fn symbols(conn: &Connection, project_id: i64) -> Vec<String> {
    let mut stmt = conn
        .prepare(
            "SELECT path || ' ' || title || ' [' || subkind || '] '
                    || span_start || '-' || span_end
                    || ' name=' || COALESCE(json_extract(meta, '$.name'), '')
                    || ' body=' || replace(COALESCE(body, ''), char(10), ' / ')
             FROM node
             WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL
             ORDER BY 1",
        )
        .unwrap();
    let rows = stmt.query_map([project_id], |r| r.get(0)).unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

/// Live call and import edges, one line each, sorted. Each end carries its
/// kind and first line, so an edge that lands on the wrong one of two
/// same-named symbols is a difference; so do weight and the stored meta, so
/// an edge that is present but mislabelled (a guess stored as certain) is
/// one too.
fn edges(conn: &Connection, project_id: i64) -> Vec<String> {
    let mut stmt = conn
        .prepare(
            "SELECT s.path || ' ' || COALESCE(s.title, '')
                    || '[' || COALESCE(s.subkind, 'file') || ':' || COALESCE(s.span_start, 0) || ']'
                    || ' -' || e.rel || '-> '
                    || d.path || ' ' || COALESCE(d.title, '')
                    || '[' || COALESCE(d.subkind, 'file') || ':' || COALESCE(d.span_start, 0) || ']'
                    || ' w=' || printf('%.3f', e.weight) || ' ' || e.meta
             FROM edge e
             JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
             WHERE e.rel IN ('calls', 'imports') AND s.project_id = ?1
               AND s.deleted_at IS NULL AND d.deleted_at IS NULL
             ORDER BY 1",
        )
        .unwrap();
    let rows = stmt.query_map([project_id], |r| r.get(0)).unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

/// One edit: write a file, or (with `None`) delete it.
type Edit = (&'static str, Option<&'static str>);

/// A project directory and the store that is updated in place as it changes.
struct World {
    dir: tempfile::TempDir,
    incremental: Mimir,
    proj: Node,
    steps: usize,
}

impl World {
    fn new() -> World {
        let dir = tempfile::tempdir().unwrap();
        let incremental = Mimir::open_in_memory().unwrap();
        let proj = project(&incremental.conn, dir.path());
        World {
            dir,
            incremental,
            proj,
            steps: 0,
        }
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }

    fn apply(&self, edits: &[Edit]) {
        for (rel, content) in edits {
            match content {
                Some(c) => write(self.root(), rel, c),
                None => std::fs::remove_file(self.root().join(rel)).unwrap(),
            }
        }
    }

    fn update(&mut self, options: UpdateOptions) -> UpdateReport {
        update_with(
            &mut self.incremental.conn,
            &self.proj,
            self.dir.path(),
            options,
        )
        .unwrap()
    }

    /// Hold the store to a from-scratch build of the tree as it is now.
    fn matches_fresh(&mut self, what: &str) {
        let mut fresh = Mimir::open_in_memory().unwrap();
        let fresh_proj = project(&fresh.conn, self.root());
        update(&mut fresh.conn, &fresh_proj, self.root()).unwrap();
        let step = self.steps;
        self.steps += 1;
        assert_eq!(
            symbols(&self.incremental.conn, self.proj.id),
            symbols(&fresh.conn, fresh_proj.id),
            "symbols differ from a fresh build after step {step} ({what})"
        );
        assert_eq!(
            edges(&self.incremental.conn, self.proj.id),
            edges(&fresh.conn, fresh_proj.id),
            "edges differ from a fresh build after step {step} ({what})"
        );
    }

    /// Edit, update incrementally, compare.
    fn step(&mut self, what: &str, edits: &[Edit]) {
        self.apply(edits);
        self.update(UpdateOptions::default());
        self.matches_fresh(what);
    }

    /// Give a file its own mtime, so a test does not depend on how finely
    /// this filesystem stamps files.
    fn stamp(&self, rel: &str, secs: u64) {
        let at = std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs);
        std::fs::File::options()
            .write(true)
            .open(self.root().join(rel))
            .unwrap()
            .set_modified(at)
            .unwrap();
    }

    /// (row id, live, first line) of every symbol row titled `title`,
    /// tombstones included.
    fn rows(&self, title: &str) -> Vec<(i64, bool, i64)> {
        let mut stmt = self
            .incremental
            .conn
            .prepare(
                "SELECT id, deleted_at IS NULL, COALESCE(span_start, 0) FROM node
                 WHERE kind = 'symbol' AND title = ?1 ORDER BY id",
            )
            .unwrap();
        let rows = stmt
            .query_map([title], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    }

    /// A memory linked to a symbol: what turns a removed symbol into a
    /// tombstone instead of a deleted row.
    fn link_memory(&self, text: &str, symbol: i64) -> i64 {
        let mut memory = NewNode::new(Kind::Memory);
        memory.title = Some(text.to_string());
        memory.body = Some(text.to_string());
        memory.project_id = Some(self.proj.id);
        let memory = store::insert_node(&self.incremental.conn, memory).unwrap();
        store::link(&self.incremental.conn, memory.id, symbol, Rel::About, 1.0).unwrap();
        memory.id
    }
}

/// Apply `steps` incrementally and hold every intermediate state to a
/// from-scratch build of the same tree.
fn incremental_matches_fresh(steps: &[&[Edit]]) {
    let mut world = World::new();
    for step in steps {
        world.step("edit", step);
    }
}

/// The caller never changes; everything it calls does.
#[test]
fn a_callee_added_moved_or_removed_elsewhere_updates_unchanged_callers() {
    incremental_matches_fresh(&[
        // Nothing to call yet: `helper` does not exist.
        &[("src/a.rs", Some("fn caller() { helper(); }\n"))],
        // It appears in another file: a.rs now has an edge it did not have.
        &[("src/b.rs", Some("pub fn helper() {}\n"))],
        // A second one makes the same call ambiguous.
        &[("src/c.rs", Some("pub fn helper() -> u8 { 1 }\n"))],
        // Back to one: the surviving edge is a unique match again.
        &[("src/c.rs", None)],
        // Moved: gone from b.rs (which still exists), present in d.rs.
        &[
            ("src/b.rs", Some("pub fn unrelated() {}\n")),
            ("src/d.rs", Some("pub fn helper() {}\n")),
        ],
        // Removed outright from a file that stays.
        &[("src/d.rs", Some("pub fn also_unrelated() {}\n"))],
    ]);
}

/// Same, through a receiver type: the unchanged caller's edge depends on
/// which types own a method of that name.
#[test]
fn a_method_appearing_on_another_type_updates_unchanged_callers() {
    incremental_matches_fresh(&[
        &[
            (
                "src/db.rs",
                Some("pub struct Database;\nimpl Database { pub fn save(&self) -> u8 { 1 } }\n"),
            ),
            (
                "src/app.rs",
                Some("fn persist(x: Mystery) {\n    x.save();\n}\n"),
            ),
        ],
        &[(
            "src/cache.rs",
            Some("pub struct Cache;\nimpl Cache { pub fn save(&self) -> u8 { 2 } }\n"),
        )],
        &[(
            "src/cache.rs",
            Some("pub struct Cache;\nimpl Cache { pub fn store(&self) -> u8 { 2 } }\n"),
        )],
    ]);
}

/// An import that pointed at nothing starts to resolve when its target
/// file is created, and stops when the file goes away — the importer is
/// untouched both times.
#[test]
fn an_import_target_appearing_or_vanishing_updates_unchanged_importers() {
    incremental_matches_fresh(&[
        &[(
            "src/main.rs",
            Some("use crate::limits::MAX;\n\nfn main() { let _ = MAX; }\n"),
        )],
        &[(
            "src/limits.rs",
            Some("pub const MAX: u8 = 9;\n\npub fn unused() {}\n"),
        )],
        &[("src/limits.rs", None)],
        // And in TypeScript, where the import also decides the call edge.
        &[
            (
                "web/app.ts",
                Some("import { fetchUser } from \"./api\";\nexport function load() { return fetchUser(1); }\n"),
            ),
            (
                "web/other.ts",
                Some("export function fetchUser(id: string) { return id; }\n"),
            ),
            (
                "web/third.ts",
                Some("export function fetchUser(id: boolean) { return id; }\n"),
            ),
        ],
        &[(
            "web/api.ts",
            Some("export function fetchUser(id: number) { return id; }\n"),
        )],
        &[("web/api.ts", None)],
    ]);
}

/// A file deleted and later restored with the same content: its symbols
/// come back as the same rows, and so must the edges of files that call it.
#[test]
fn a_file_deleted_and_restored_gets_its_callers_back() {
    incremental_matches_fresh(&[
        &[
            ("src/util.rs", Some("pub fn slugify() {}\n")),
            ("src/main.rs", Some("fn main() { slugify(); }\n")),
        ],
        &[("src/util.rs", None)],
        &[("src/util.rs", Some("pub fn slugify() {}\n"))],
    ]);
}

/// Two symbols of one file sharing a qualified name, reordered. No symbol
/// appears or disappears, so nothing brings the caller in the other file
/// back to be re-resolved: whichever of the two stands for the name must be
/// the same one before and after.
#[test]
fn a_reorder_of_two_kinds_sharing_a_name_moves_no_edge() {
    incremental_matches_fresh(&[
        &[
            (
                "lib/shapes.py",
                Some("class Shape:\n    pass\n\ndef make():\n    pass\n\nmake = 3\n"),
            ),
            (
                "lib/both.ts",
                Some("export interface Options { a: number }\nexport function Options() { return 1; }\n"),
            ),
            (
                "lib/use.ts",
                Some("export function run() { return Options(); }\n"),
            ),
        ],
        &[(
            "lib/both.ts",
            Some("export function Options() { return 1; }\nexport interface Options { a: number }\n"),
        )],
    ]);
}

/// One method given to a type by two trait impls: one row for the pair,
/// through a swap, a removal and a return.
#[test]
fn two_definitions_of_one_method_swapped_removed_and_restored() {
    const DISPLAY: &str = "impl std::fmt::Display for T {\n    fn fmt(&self) -> u8 { 1 }\n}\n";
    const BOTH: &str = "pub struct T;\nimpl std::fmt::Display for T {\n    fn fmt(&self) -> u8 { 1 }\n}\nimpl std::fmt::Debug for T {\n    fn fmt(&self) -> u16 { 2 }\n}\n";
    const SWAPPED: &str = "pub struct T;\nimpl std::fmt::Debug for T {\n    fn fmt(&self) -> u16 { 2 }\n}\nimpl std::fmt::Display for T {\n    fn fmt(&self) -> u8 { 1 }\n}\n";
    const ONLY_DEBUG: &str =
        "pub struct T;\nimpl std::fmt::Debug for T {\n    fn fmt(&self) -> u16 { 2 }\n}\n";
    assert!(BOTH.contains(DISPLAY) && SWAPPED.ends_with(DISPLAY));
    incremental_matches_fresh(&[
        &[
            ("src/t.rs", Some(BOTH)),
            (
                "src/main.rs",
                Some("fn main(t: T) { t.fmt(); }\nfn other(x: Mystery) { x.fmt(); }\n"),
            ),
        ],
        &[("src/t.rs", Some(SWAPPED))],
        &[("src/t.rs", Some(ONLY_DEBUG))],
        &[("src/t.rs", Some(BOTH))],
        &[("src/t.rs", Some("pub struct T;\n"))],
        &[("src/t.rs", Some(SWAPPED))],
    ]);
}

#[test]
fn a_file_renamed_under_its_importer_and_caller() {
    incremental_matches_fresh(&[
        &[
            ("src/a.rs", Some("pub fn helper() {}\n")),
            (
                "src/main.rs",
                Some("use crate::a::helper;\nfn main() { helper(); }\n"),
            ),
        ],
        &[
            ("src/a.rs", None),
            ("src/b.rs", Some("pub fn helper() {}\n")),
        ],
        &[
            ("src/b.rs", None),
            ("src/a.rs", Some("pub fn helper() {}\n")),
        ],
        // A rename that only changes case is still another path.
        &[
            ("src/a.rs", None),
            ("src/A.rs", Some("pub fn helper() {}\n")),
        ],
    ]);
}

#[test]
fn a_symbol_moved_between_files_in_one_step() {
    incremental_matches_fresh(&[
        &[
            ("src/a.rs", Some("pub fn helper() {}\npub fn stay() {}\n")),
            ("src/b.rs", Some("pub fn other() {}\n")),
            ("src/main.rs", Some("fn main() { helper(); stay(); }\n")),
        ],
        &[
            ("src/a.rs", Some("pub fn stay() {}\n")),
            ("src/b.rs", Some("pub fn other() {}\npub fn helper() {}\n")),
        ],
        // The two files trade their whole contents.
        &[
            ("src/b.rs", Some("pub fn stay() {}\n")),
            ("src/a.rs", Some("pub fn other() {}\npub fn helper() {}\n")),
        ],
    ]);
}

/// An imported file goes and comes back, in three languages; then a
/// directory index appears beside the TypeScript module and takes over when
/// the module file is removed.
#[test]
fn an_imported_file_deleted_restored_and_shadowed() {
    const API: &str = "export function fetchUser(id: number) { return id; }\n";
    const UTIL: &str = "def slug():\n    pass\n";
    const LIMITS: &str = "pub fn max() {}\n";
    incremental_matches_fresh(&[
        &[
            (
                "web/app.ts",
                Some("import { fetchUser } from \"./api\";\nexport function load() { return fetchUser(1); }\n"),
            ),
            ("web/api.ts", Some(API)),
            (
                "web/other.ts",
                Some("export function fetchUser(id: string) { return id; }\n"),
            ),
            ("pkg/__init__.py", Some("")),
            (
                "pkg/main.py",
                Some("from .util import slug\n\ndef run():\n    slug()\n"),
            ),
            ("pkg/util.py", Some(UTIL)),
            ("other/util.py", Some(UTIL)),
            (
                "src/main.rs",
                Some("use crate::limits::max;\nfn main() { max(); }\n"),
            ),
            ("src/limits.rs", Some(LIMITS)),
        ],
        &[("web/api.ts", None)],
        &[("pkg/util.py", None)],
        &[("src/limits.rs", None)],
        &[
            ("web/api.ts", Some(API)),
            ("pkg/util.py", Some(UTIL)),
            ("src/limits.rs", Some(LIMITS)),
        ],
        &[(
            "web/api/index.ts",
            Some("export function fetchUser() {}\n"),
        )],
        &[("web/api.ts", None)],
    ]);
}

/// The importer is the file that changes: same call, another source for it.
#[test]
fn a_file_that_changes_only_its_imports() {
    incremental_matches_fresh(&[
        &[
            (
                "web/app.ts",
                Some("import { fetchUser } from \"./api\";\nexport function load() { return fetchUser(1); }\n"),
            ),
            (
                "web/api.ts",
                Some("export function fetchUser(id: number) { return id; }\n"),
            ),
            (
                "web/other.ts",
                Some("export function fetchUser(id: string) { return id; }\n"),
            ),
        ],
        &[(
            "web/app.ts",
            Some("import { fetchUser } from \"./other\";\nexport function load() { return fetchUser(1); }\n"),
        )],
        &[(
            "web/app.ts",
            Some("export function load() { return fetchUser(1); }\n"),
        )],
    ]);
}

/// One name, defined in one file, then two, three, four and five: unique,
/// a guess between two, between three, then too many to keep — and back.
#[test]
fn one_name_defined_in_more_and_more_files() {
    const HELPER: Option<&str> = Some("pub fn helper() {}\n");
    incremental_matches_fresh(&[
        &[
            ("src/caller.rs", Some("fn caller() { helper(); }\n")),
            ("src/h1.rs", HELPER),
        ],
        &[("src/h2.rs", HELPER)],
        &[("src/h3.rs", HELPER)],
        &[("src/h4.rs", HELPER)],
        &[("src/h5.rs", HELPER)],
        &[("src/h5.rs", None), ("src/h4.rs", None)],
        &[("src/h3.rs", None), ("src/h2.rs", None)],
        &[("src/h1.rs", None)],
    ]);
}

/// Source that stops parsing cleanly, turns to garbage, and recovers.
#[test]
fn source_that_breaks_and_is_mended() {
    const VALID: &str = "pub fn fine() {}\npub fn helper() { fine(); }\n";
    incremental_matches_fresh(&[
        &[
            ("src/a.rs", Some(VALID)),
            ("src/main.rs", Some("fn main() { helper(); fine(); }\n")),
        ],
        &[(
            "src/a.rs",
            Some("pub fn fine() {}\npub fn helper( {{{ fine(); \n"),
        )],
        &[("src/a.rs", Some("}}}} ((( \u{0} \u{fffd} fn\n"))],
        &[("src/a.rs", Some(VALID))],
    ]);
}

/// One name through a change of kind, then of owner.
#[test]
fn a_symbol_that_changes_kind_or_owner() {
    incremental_matches_fresh(&[
        &[
            (
                "src/t.rs",
                Some("pub struct Thing;\nimpl Thing { pub fn new() -> Thing { Thing } }\n"),
            ),
            (
                "src/main.rs",
                Some("fn main() { let t = Thing::new(); }\nfn mk() { Thing(); }\n"),
            ),
        ],
        &[(
            "src/t.rs",
            Some("pub enum Thing { A }\nimpl Thing { pub fn new() -> Thing { Thing::A } }\n"),
        )],
        &[("src/t.rs", Some("pub fn Thing() {}\n"))],
        &[(
            "src/t.rs",
            Some("pub struct Other;\nimpl Other { pub fn new() -> Other { Other } }\n"),
        )],
    ]);
}

/// Ten names appear at once — more than the dependents search looks for as
/// text, so every file's stored calls are parsed instead — and two that are
/// not ASCII.
#[test]
fn many_names_at_once_and_names_outside_ascii() {
    incremental_matches_fresh(&[
        &[(
            "app/caller.py",
            Some("def run():\n    a1()\n    a2()\n    a3()\n    a4()\n    a5()\n    a6()\n    a7()\n    a8()\n    a9()\n    a10()\n    caf\u{e9}()\n    na\u{ef}ve()\n"),
        )],
        &[(
            "app/defs.py",
            Some("def a1(): pass\ndef a2(): pass\ndef a3(): pass\ndef a4(): pass\ndef a5(): pass\ndef a6(): pass\ndef a7(): pass\ndef a8(): pass\ndef a9(): pass\ndef a10(): pass\n"),
        )],
        &[("app/u.py", Some("def caf\u{e9}(): pass\n"))],
        &[("app/v.py", Some("def na\u{ef}ve(): pass\n"))],
        &[("app/u.py", None)],
        &[("app/defs.py", Some("def a1(): pass\n"))],
    ]);
}

/// A directory is renamed: every import that went into it moves with it.
#[test]
fn a_directory_renamed_under_its_importers() {
    incremental_matches_fresh(&[
        &[
            (
                "src/main.rs",
                Some("use crate::net::client::get;\nfn main() { get(); }\n"),
            ),
            ("src/net/mod.rs", Some("pub mod client;\n")),
            ("src/net/client.rs", Some("pub fn get() {}\n")),
            (
                "web/app.ts",
                Some("import { h } from \"./lib/h\";\nexport function go() { return h(); }\n"),
            ),
            ("web/lib/h.ts", Some("export function h() {}\n")),
        ],
        &[
            ("src/net/mod.rs", None),
            ("src/net/client.rs", None),
            ("src/http/mod.rs", Some("pub mod client;\n")),
            ("src/http/client.rs", Some("pub fn get() {}\n")),
            ("web/lib/h.ts", None),
            ("web/util/h.ts", Some("export function h() {}\n")),
        ],
    ]);
}

/// A symbol a memory is linked to leaves and returns, alone and with its
/// file. It is one row throughout — a tombstone while it is gone — and the
/// live graph is what a fresh build gives at every step, also after a full
/// rebuild made while it is gone.
#[test]
fn a_linked_symbol_leaves_and_returns_as_the_same_row() {
    const WITHOUT: Option<&str> = Some("pub fn keep() {}\n");
    let mut w = World::new();
    w.step(
        "two symbols, both called",
        &[
            (
                "src/util.rs",
                Some("pub fn slugify() {}\npub fn keep() {}\n"),
            ),
            ("src/main.rs", Some("fn main() { slugify(); keep(); }\n")),
        ],
    );
    let id = w.rows("slugify")[0].0;
    let memory = w.link_memory("slugify lowercases", id);

    w.step("the linked symbol is removed", &[("src/util.rs", WITHOUT)]);
    assert_eq!(w.rows("slugify"), [(id, false, 1)], "a tombstone, same row");
    w.step(
        "an unrelated file appears",
        &[("src/x.rs", Some("pub fn x() {}\n"))],
    );
    w.step(
        "the symbol returns, lower down",
        &[(
            "src/util.rs",
            Some("pub fn keep() {}\npub fn slugify() {}\n"),
        )],
    );
    assert_eq!(w.rows("slugify"), [(id, true, 2)], "revived, same row");

    w.step("removed again", &[("src/util.rs", WITHOUT)]);
    w.step("its file is deleted", &[("src/util.rs", None)]);
    assert_eq!(w.rows("slugify"), [(id, false, 2)]);
    w.step("the file returns without it", &[("src/util.rs", WITHOUT)]);
    assert_eq!(w.rows("slugify"), [(id, false, 2)]);
    w.step(
        "and then with it",
        &[(
            "src/util.rs",
            Some("pub fn slugify() {}\npub fn keep() {}\n"),
        )],
    );
    assert_eq!(w.rows("slugify"), [(id, true, 1)]);
    let links: i64 = w
        .incremental
        .conn
        .query_row(
            "SELECT count(*) FROM edge WHERE src = ?1 AND dst = ?2",
            [memory, id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(links, 1, "the link was never lost");
    // The file came back as the row it had: one row per path, not a second
    // one beside a tombstone.
    let file_rows: i64 = w
        .incremental
        .conn
        .query_row(
            "SELECT count(*) FROM node WHERE kind = 'file' AND path = 'src/util.rs'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(file_rows, 1);

    w.step("removed a third time", &[("src/util.rs", WITHOUT)]);
    w.update(UpdateOptions { full: true });
    w.matches_fresh("a full rebuild with a tombstone in the store");
    assert_eq!(
        w.rows("slugify"),
        [(id, false, 1)],
        "the tombstone survives it"
    );
}

/// A file whose stat was refreshed by a touch has had its meta rewritten by
/// SQLite. Its stored calls must still be found when a callee appears, and
/// a touch alone must re-extract nothing.
#[test]
fn a_touched_file_is_left_alone_and_still_found_as_a_dependent() {
    let mut w = World::new();
    w.step(
        "a caller with nothing to call",
        &[("src/a.rs", Some("fn caller() { helper(); }\n"))],
    );
    w.stamp("src/a.rs", 1_900_000_000);
    let report = w.update(UpdateOptions::default());
    assert_eq!(
        (
            report.stats.files_indexed,
            report.stats.unchanged,
            report.reresolved
        ),
        (0, 1, 0),
        "{report:?}"
    );
    w.matches_fresh("touched");
    w.step(
        "the callee appears",
        &[("src/b.rs", Some("pub fn helper() {}\n"))],
    );
}

/// A dependent that cannot be read when its edges need rebuilding keeps
/// them for now. Once it can be read, one more update ends where a fresh
/// build does.
#[cfg(unix)]
#[test]
fn an_unreadable_dependent_catches_up_once_it_can_be_read() {
    use std::os::unix::fs::PermissionsExt;
    let mut w = World::new();
    w.step(
        "a caller with nothing to call",
        &[("src/a.rs", Some("fn caller() { helper(); }\n"))],
    );
    let caller = w.root().join("src/a.rs");
    std::fs::set_permissions(&caller, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::File::open(&caller).is_ok() {
        // Permissions do not bind this user (root): nothing to test with.
        return;
    }
    w.apply(&[("src/b.rs", Some("pub fn helper() {}\n"))]);
    let report = w.update(UpdateOptions::default());
    assert_eq!((report.unreadable, report.reresolved), (1, 0), "{report:?}");

    std::fs::set_permissions(&caller, std::fs::Permissions::from_mode(0o644)).unwrap();
    w.update(UpdateOptions::default());
    w.matches_fresh("after the dependent became readable");
}

/// Small deterministic generator: the scenarios below are random in shape
/// and identical on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, of: &[&'a str]) -> &'a str {
        of[self.below(of.len())]
    }
}

const NAMES: [&str; 4] = ["alpha", "beta", "gamma", "delta"];
const TYPES: [&str; 3] = ["Store", "Cache", "Index"];
const SLOTS: usize = 6;

/// A random Rust file: free functions and methods drawn from a small pool
/// of names (so the same name keeps turning up in several files and on
/// several types), each calling a few names bare, through a type, through
/// `self` and through an untyped receiver, under imports of other slots.
fn random_rust(rng: &mut Rng) -> String {
    let mut out = String::new();
    for _ in 0..rng.below(3) {
        out.push_str(&format!(
            "use crate::f{}::{};\n",
            rng.below(SLOTS),
            rng.pick(&NAMES)
        ));
    }
    let calls = |rng: &mut Rng| -> String {
        let mut body = String::new();
        for _ in 0..rng.below(4) {
            let name = rng.pick(&NAMES);
            match rng.below(4) {
                0 => body.push_str(&format!(" {name}();")),
                1 => body.push_str(&format!(" {}::{name}();", rng.pick(&TYPES))),
                2 => body.push_str(&format!(" self.{name}();")),
                _ => body.push_str(&format!(" thing.{name}();")),
            }
        }
        body
    };
    for _ in 0..rng.below(3) {
        let body = calls(rng);
        out.push_str(&format!("pub fn {}() {{{body} }}\n", rng.pick(&NAMES)));
    }
    for _ in 0..rng.below(3) {
        let ty = rng.pick(&TYPES);
        out.push_str(&format!("pub struct {ty};\nimpl {ty} {{\n"));
        for _ in 0..1 + rng.below(2) {
            let body = calls(rng);
            out.push_str(&format!(
                "    pub fn {}(&self) {{{body} }}\n",
                rng.pick(&NAMES)
            ));
        }
        out.push_str("}\n");
    }
    out
}

/// Many random edit sequences, each step checked against a fresh build.
/// The hand-written scenarios above each pin one way an unchanged file goes
/// stale; this is for the ways nobody thought to write down.
#[test]
fn random_edit_sequences_match_a_fresh_build_at_every_step() {
    for seed in 1..=12u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let dir = tempfile::tempdir().unwrap();
        let mut incremental = Mimir::open_in_memory().unwrap();
        let proj = project(&incremental.conn, dir.path());
        let mut clock = 1_700_000_000u64;

        for step in 0..30 {
            for _ in 0..1 + rng.below(3) {
                let rel = format!("src/f{}.rs", rng.below(SLOTS));
                let path = dir.path().join(&rel);
                if rng.below(5) == 0 {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                write(dir.path(), &rel, &random_rust(&mut rng));
                // Every write gets its own mtime, so the test does not
                // depend on how finely this filesystem stamps files.
                clock += 1;
                let stamp = std::time::UNIX_EPOCH + std::time::Duration::from_secs(clock);
                std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_modified(stamp)
                    .unwrap();
            }
            update(&mut incremental.conn, &proj, dir.path()).unwrap();

            let mut fresh = Mimir::open_in_memory().unwrap();
            let fresh_proj = project(&fresh.conn, dir.path());
            update(&mut fresh.conn, &fresh_proj, dir.path()).unwrap();

            assert_eq!(
                symbols(&incremental.conn, proj.id),
                symbols(&fresh.conn, fresh_proj.id),
                "seed {seed}: symbols differ from a fresh build after step {step}"
            );
            assert_eq!(
                edges(&incremental.conn, proj.id),
                edges(&fresh.conn, fresh_proj.id),
                "seed {seed}: edges differ from a fresh build after step {step}"
            );
        }
    }
}

/// `--full` is the way out of a graph whose edges are wrong for a reason an
/// incremental update cannot see (here: someone deleted and forged edges
/// behind its back). It re-extracts every file, ends where a fresh build
/// would, and keeps every symbol's row — so nothing linked to one is lost.
#[test]
fn a_full_rebuild_repairs_edges_an_incremental_update_cannot_see() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "src/util.rs",
        "pub fn slugify() {}\npub fn other() {}\n",
    );
    write(dir.path(), "src/main.rs", "fn main() { slugify(); }\n");
    let mut mimir = Mimir::open_in_memory().unwrap();
    let proj = project(&mimir.conn, dir.path());
    update(&mut mimir.conn, &proj, dir.path()).unwrap();
    let expected = edges(&mimir.conn, proj.id);
    assert_eq!(expected.len(), 1, "{expected:?}");

    let ids = |conn: &Connection| -> Vec<(String, i64)> {
        let mut stmt = conn
            .prepare("SELECT title, id FROM node WHERE kind = 'symbol' ORDER BY title")
            .unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    };
    let before = ids(&mimir.conn);

    // main → slugify becomes main → other.
    mimir
        .conn
        .execute(
            "UPDATE edge SET dst = (SELECT id FROM node WHERE title = 'other')
             WHERE rel = 'calls'",
            [],
        )
        .unwrap();
    assert_ne!(edges(&mimir.conn, proj.id), expected);

    // Nothing on disk changed, so an ordinary update has nothing to do.
    let report = update_with(&mut mimir.conn, &proj, dir.path(), UpdateOptions::default()).unwrap();
    assert_eq!(report.stats.files_indexed, 0);
    assert_ne!(edges(&mimir.conn, proj.id), expected);

    let report = update_with(
        &mut mimir.conn,
        &proj,
        dir.path(),
        UpdateOptions { full: true },
    )
    .unwrap();
    assert_eq!(report.stats.files_indexed, 2, "every file re-extracted");
    assert_eq!(report.stats.unchanged, 0);
    assert_eq!(edges(&mimir.conn, proj.id), expected);
    assert_eq!(
        ids(&mimir.conn),
        before,
        "symbol rows survive a full rebuild"
    );
}

const LANGS: [&str; 3] = ["rs", "ts", "py"];
const DIRS: usize = 3;
const FILES: usize = 3;
const WORDS: [&str; 5] = ["alpha", "beta", "gamma", "delta", "new"];

fn random_ts(rng: &mut Rng) -> String {
    let mut out = String::new();
    for _ in 0..rng.below(3) {
        if rng.below(2) == 0 {
            out.push_str(&format!(
                "import {{ {} }} from \"./f{}\";\n",
                rng.pick(&WORDS),
                rng.below(FILES)
            ));
        } else {
            out.push_str(&format!(
                "import {{ {} }} from \"../d{}/f{}\";\n",
                rng.pick(&WORDS),
                rng.below(DIRS),
                rng.below(FILES)
            ));
        }
    }
    for _ in 0..rng.below(4) {
        let mut body = String::new();
        for _ in 0..rng.below(4) {
            let name = rng.pick(&WORDS);
            match rng.below(3) {
                0 => body.push_str(&format!(" {name}();")),
                1 => body.push_str(&format!(" this.{name}();")),
                _ => body.push_str(&format!(" thing.{name}();")),
            }
        }
        out.push_str(&format!(
            "export function {}() {{{body} }}\n",
            rng.pick(&WORDS)
        ));
    }
    if rng.below(2) == 0 {
        out.push_str(&format!("export class {} {{\n", rng.pick(&TYPES)));
        for _ in 0..1 + rng.below(2) {
            out.push_str(&format!(
                "  {}() {{ this.{}(); {}(); }}\n",
                rng.pick(&WORDS),
                rng.pick(&WORDS),
                rng.pick(&WORDS)
            ));
        }
        out.push_str("}\n");
    }
    out
}

fn random_python(rng: &mut Rng) -> String {
    let mut out = String::new();
    for _ in 0..rng.below(3) {
        match rng.below(3) {
            0 => out.push_str(&format!(
                "from .f{} import {}\n",
                rng.below(FILES),
                rng.pick(&WORDS)
            )),
            1 => out.push_str(&format!(
                "from src.d{}.f{} import {}\n",
                rng.below(DIRS),
                rng.below(FILES),
                rng.pick(&WORDS)
            )),
            _ => out.push_str(&format!("import f{}\n", rng.below(FILES))),
        }
    }
    for _ in 0..rng.below(4) {
        out.push_str(&format!("def {}():\n", rng.pick(&WORDS)));
        for _ in 0..rng.below(3) {
            out.push_str(&format!("    {}()\n", rng.pick(&WORDS)));
        }
        out.push_str("    pass\n\n");
    }
    if rng.below(2) == 0 {
        out.push_str(&format!("class {}:\n", rng.pick(&TYPES)));
        for _ in 0..1 + rng.below(2) {
            out.push_str(&format!(
                "    def {}(self):\n        self.{}()\n        {}()\n\n",
                rng.pick(&WORDS),
                rng.pick(&WORDS),
                rng.pick(&WORDS)
            ));
        }
    }
    out
}

/// Rust with every import anchor, and now and then one function defined
/// twice (the `#[cfg]` pair that shares a row).
fn random_rust_in_a_tree(rng: &mut Rng) -> String {
    let mut out = String::new();
    for _ in 0..rng.below(3) {
        let (dir, file, name) = (rng.below(DIRS), rng.below(FILES), rng.pick(&WORDS));
        match rng.below(3) {
            0 => out.push_str(&format!("use crate::d{dir}::f{file}::{name};\n")),
            1 => out.push_str(&format!("use super::f{file}::{name};\n")),
            _ => out.push_str(&format!("use f{file}::{name};\n")),
        }
    }
    let calls = |rng: &mut Rng| -> String {
        let mut body = String::new();
        for _ in 0..rng.below(4) {
            let name = rng.pick(&WORDS);
            match rng.below(4) {
                0 => body.push_str(&format!(" {name}();")),
                1 => body.push_str(&format!(" {}::{name}();", rng.pick(&TYPES))),
                2 => body.push_str(&format!(" self.{name}();")),
                _ => body.push_str(&format!(" thing.{name}();")),
            }
        }
        body
    };
    for _ in 0..rng.below(4) {
        let body = calls(rng);
        let name = rng.pick(&WORDS);
        out.push_str(&format!("pub fn {name}() {{{body} }}\n"));
        if rng.below(5) == 0 {
            let body = calls(rng);
            out.push_str(&format!("#[cfg(test)]\npub fn {name}() {{{body} }}\n"));
        }
    }
    for _ in 0..rng.below(3) {
        let ty = rng.pick(&TYPES);
        out.push_str(&format!("pub struct {ty};\nimpl {ty} {{\n"));
        for _ in 0..1 + rng.below(3) {
            let body = calls(rng);
            out.push_str(&format!(
                "    pub fn {}(&self) {{{body} }}\n",
                rng.pick(&WORDS)
            ));
        }
        out.push_str("}\n");
    }
    out
}

/// Random sequences across three languages in a small directory tree: files
/// written, deleted, renamed and touched, with memories linked to symbols
/// along the way so that tombstones take part. Every step is held to a fresh
/// build, and at the end a full rebuild of the store must change nothing a
/// reader can see.
#[test]
fn random_sequences_in_three_languages_with_linked_memories() {
    for seed in 1..=8u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut w = World::new();
        let mut clock = 1_700_000_000u64;
        for step in 0..30 {
            for _ in 0..1 + rng.below(3) {
                let ext = rng.pick(&LANGS);
                let rel = format!("src/d{}/f{}.{ext}", rng.below(DIRS), rng.below(FILES));
                let path = w.root().join(&rel);
                match rng.below(20) {
                    0..=2 => {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    3..=4 => {
                        let to = format!("src/d{}/f{}.{ext}", rng.below(DIRS), rng.below(FILES));
                        if path.exists() {
                            let to = w.root().join(to);
                            std::fs::create_dir_all(to.parent().unwrap()).unwrap();
                            let _ = std::fs::rename(&path, &to);
                        }
                        continue;
                    }
                    // A touch: new mtime below, same bytes.
                    5 if path.exists() => {}
                    5 => continue,
                    _ => {
                        let content = match ext {
                            "rs" => random_rust_in_a_tree(&mut rng),
                            "ts" => random_ts(&mut rng),
                            _ => random_python(&mut rng),
                        };
                        write(w.root(), &rel, &content);
                    }
                }
                clock += 1;
                w.stamp(&rel, clock);
            }
            w.update(UpdateOptions::default());
            w.matches_fresh(&format!("seed {seed}, step {step}"));

            if rng.below(3) == 0 {
                let live: Vec<i64> = {
                    let mut stmt = w
                        .incremental
                        .conn
                        .prepare(
                            "SELECT id FROM node
                             WHERE kind = 'symbol' AND deleted_at IS NULL ORDER BY id",
                        )
                        .unwrap();
                    let rows = stmt.query_map([], |r| r.get(0)).unwrap();
                    rows.collect::<rusqlite::Result<_>>().unwrap()
                };
                if !live.is_empty() {
                    w.link_memory(&format!("note {seed}/{step}"), live[rng.below(live.len())]);
                }
            }
        }
        let before = (
            symbols(&w.incremental.conn, w.proj.id),
            edges(&w.incremental.conn, w.proj.id),
        );
        w.update(UpdateOptions { full: true });
        let after = (
            symbols(&w.incremental.conn, w.proj.id),
            edges(&w.incremental.conn, w.proj.id),
        );
        assert_eq!(
            before, after,
            "seed {seed}: a full rebuild changed the graph"
        );
    }
}
