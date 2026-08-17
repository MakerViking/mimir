//! Code graph for Mimir: tree-sitter symbol extraction (no LLM, no type
//! checker) persisted into the unified node/edge store, plus petgraph
//! queries (callers/calls/impact/path/hubs/communities).
//!
//! Symbols are ordinary nodes — they embed into semantic recall and link
//! to memories like everything else. Stable identity across re-extraction
//! lives in meta.stable_id; see store_graph.

// extract/languages live in mimir-syntax (dependency-free from mimir-core,
// so mimir-core can also use them without a cycle back through this crate);
// re-exported here so existing `mimir_graph::extract`/`mimir_graph::languages`
// call sites are unaffected.
pub use mimir_syntax::{extract, languages};

pub mod queries;
pub mod store_graph;

use mimir_core::error::{Error, Result};
use mimir_core::model::Node;
use mimir_core::store::{self, row_to_node, NODE_COLS};
use rusqlite::Connection;

pub use queries::CodeGraph;
pub use store_graph::{update, GraphStats};

/// Resolve a symbol by short id, exact qualified name, or bare name.
/// Bare-name matches must be unique; ambiguity lists the candidates.
pub fn resolve_symbol(conn: &Connection, project_id: i64, reference: &str) -> Result<Node> {
    if let Ok(node) = store::resolve_ref(conn, reference) {
        if matches!(node.kind, mimir_core::model::Kind::Symbol) {
            return Ok(node);
        }
    }
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL
           AND (title = ?2 OR json_extract(meta, '$.name') = ?2)
         LIMIT 6"
    ))?;
    let matches: Vec<Node> = stmt
        .query_map(rusqlite::params![project_id, reference], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    match matches.len() {
        0 => Err(Error::NotFound(format!("symbol '{reference}'"))),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => Err(Error::AmbiguousRef(
            format!(
                "{reference} (matches {})",
                matches
                    .iter()
                    .filter_map(|m| m.title.as_deref())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            n,
        )),
    }
}

/// One-line agent format for a symbol: `c:TAIL [method rust] path:12-40 qualified`.
pub fn symbol_line(node: &Node) -> String {
    let id = mimir_core::model::short_uid(node.kind, &node.uid);
    let kind = node.subkind.as_deref().unwrap_or("symbol");
    let lang = node.lang.as_deref().unwrap_or("?");
    let span = match (node.span_start, node.span_end) {
        (Some(a), Some(b)) => format!(":{a}-{b}"),
        _ => String::new(),
    };
    format!(
        "{id} [{kind} {lang}] {}{span} {}",
        node.path.as_deref().unwrap_or("?"),
        node.title.as_deref().unwrap_or("?")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::Mimir;
    use std::path::Path;

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn project(conn: &Connection, root: &Path) -> Node {
        store::ensure_project(conn, &root.to_string_lossy(), "test-proj").unwrap()
    }

    fn setup() -> (tempfile::TempDir, Mimir, Node) {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/util.rs",
            "/// Slugs.\npub fn slugify(s: &str) -> String { s.to_lowercase() }\n",
        );
        write(
            dir.path(),
            "src/main.rs",
            "use crate::util::slugify;\n\nfn main() { run(); }\n\nfn run() -> String { slugify(\"Hi\") }\n",
        );
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        (dir, mimir, proj)
    }

    #[test]
    fn build_persists_symbols_and_resolves_calls() {
        let (dir, mut mimir, proj) = setup();
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.files_indexed, 2);
        assert_eq!(stats.symbols, 3, "slugify, main, run");
        assert!(stats.calls_resolved >= 2, "{stats:?}"); // main→run, run→slugify

        let run = resolve_symbol(&mimir.conn, proj.id, "run").unwrap();
        assert_eq!(run.subkind.as_deref(), Some("function"));
        assert!(run.body.as_deref().unwrap().contains("fn run()"));

        let slug = resolve_symbol(&mimir.conn, proj.id, "slugify").unwrap();
        assert!(
            slug.body.as_deref().unwrap().contains("Slugs."),
            "doc embedded"
        );

        let graph = CodeGraph::load(&mimir.conn, proj.id).unwrap();
        let callers = graph.callers(slug.id, 5);
        let ids: Vec<i64> = callers.iter().map(|r| r.id).collect();
        assert!(ids.contains(&run.id), "run calls slugify");
        let main = resolve_symbol(&mimir.conn, proj.id, "main").unwrap();
        assert!(ids.contains(&main.id), "main → run → slugify transitively");

        // path main → slugify
        let p = graph.path(main.id, slug.id).unwrap();
        assert_eq!(p.first(), Some(&main.id));
        assert_eq!(p.last(), Some(&slug.id));
    }

    #[test]
    fn incremental_update_preserves_ids_and_links() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let slug_before = resolve_symbol(&mimir.conn, proj.id, "slugify").unwrap();

        // Link a memory to the symbol (the survival contract under test).
        let memory = mimir_core::memory::remember(
            &mimir.conn,
            mimir_core::memory::Remember {
                text: "slugify must stay ASCII-only".into(),
                mtype: mimir_core::model::MemoryType::Decision,
                tags: vec![],
                project_id: Some(proj.id),
                force: false,
                actor: mimir_core::model::Actor::System,
            },
        )
        .unwrap();
        let mem = match memory {
            mimir_core::memory::RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        };
        store::link(
            &mimir.conn,
            mem.id,
            slug_before.id,
            mimir_core::model::Rel::About,
            1.0,
        )
        .unwrap();

        // Shift the symbol down and change its body.
        write(
            dir.path(),
            "src/util.rs",
            "// new header\n\n/// Slugs v2.\npub fn slugify(s: &str) -> String { s.trim().to_lowercase() }\n",
        );
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.files_indexed, 1);
        assert_eq!(stats.unchanged, 1);

        let slug_after = resolve_symbol(&mimir.conn, proj.id, "slugify").unwrap();
        assert_eq!(slug_after.id, slug_before.id, "stable id preserved");
        assert_eq!(slug_after.uid, slug_before.uid);
        assert_eq!(slug_after.span_start, Some(4), "span updated");
        assert!(slug_after.body.unwrap().contains("Slugs v2."));

        let edges = store::edges_of(&mimir.conn, mem.id).unwrap();
        assert_eq!(edges.len(), 1, "memory→symbol link survived re-extraction");
        assert_eq!(edges[0].dst, slug_after.id);
    }

    #[test]
    fn removed_symbols_and_files_disappear() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        // Remove `run` from main.rs.
        write(dir.path(), "src/main.rs", "fn main() {}\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(resolve_symbol(&mimir.conn, proj.id, "run").is_err());

        // Remove the whole util file.
        std::fs::remove_file(dir.path().join("src/util.rs")).unwrap();
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.removed, 1);
        assert!(resolve_symbol(&mimir.conn, proj.id, "slugify").is_err());
    }

    #[test]
    fn cross_language_project() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "web/api.ts",
            "export function fetchUser(id: number) { return id; }\n",
        );
        write(
            dir.path(),
            "web/app.ts",
            "import { fetchUser } from \"./api\";\nexport function load() { return fetchUser(1); }\n",
        );
        write(
            dir.path(),
            "tools/gen.py",
            "def generate():\n    return emit()\n\ndef emit():\n    return 1\n",
        );
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let mut conn = mimir.conn;
        let stats = update(&mut conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.files_indexed, 3);
        assert!(stats.calls_resolved >= 2, "{stats:?}"); // load→fetchUser (import tier), generate→emit
        assert!(stats.imports >= 1, "app.ts imports api.ts: {stats:?}");

        let graph = CodeGraph::load(&conn, proj.id).unwrap();
        let fetch = resolve_symbol(&conn, proj.id, "fetchUser").unwrap();
        let load = resolve_symbol(&conn, proj.id, "load").unwrap();
        assert!(graph.callers(fetch.id, 2).iter().any(|r| r.id == load.id));
    }

    /// Each newer language: a file extracts its symbols and resolves a
    /// same-file call (tier 1), proving the grammar wiring is correct.
    fn extracts(file: &str, src: &str, expect_symbols: &[&str], caller: &str, callee: &str) {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), file, src);
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let mut conn = mimir.conn;
        update(&mut conn, &proj, dir.path()).unwrap();
        for sym in expect_symbols {
            assert!(
                resolve_symbol(&conn, proj.id, sym).is_ok(),
                "{file}: expected symbol `{sym}`"
            );
        }
        let graph = CodeGraph::load(&conn, proj.id).unwrap();
        let callee_sym = resolve_symbol(&conn, proj.id, callee).unwrap();
        let caller_sym = resolve_symbol(&conn, proj.id, caller).unwrap();
        assert!(
            graph
                .callers(callee_sym.id, 2)
                .iter()
                .any(|r| r.id == caller_sym.id),
            "{file}: expected call {caller} -> {callee}"
        );
    }

    #[test]
    fn java_extraction() {
        extracts(
            "Main.java",
            "class Main {\n  void run() { helper(); }\n  int helper() { return 42; }\n}\n",
            &["Main", "run", "helper"],
            "run",
            "helper",
        );
    }

    #[test]
    fn ruby_extraction() {
        // `helper()` with parens is an unambiguous call; a bare paren-less
        // `helper` is indistinguishable from a local variable in Ruby and is
        // deliberately not captured (it would flood the graph with false
        // edges). Symbols (class/module/method) always extract.
        extracts(
            "app.rb",
            "class App\n  def run\n    helper()\n  end\n  def helper\n    42\n  end\nend\n",
            &["App", "run", "helper"],
            "run",
            "helper",
        );
    }

    #[test]
    fn c_extraction() {
        extracts(
            "main.c",
            "int helper(void) { return 42; }\nint run(void) { return helper(); }\n",
            &["helper", "run"],
            "run",
            "helper",
        );
    }

    #[test]
    fn csharp_extraction() {
        extracts(
            "App.cs",
            "namespace App {\n  class Worker {\n    void Run() { Help(); }\n    void Help() { }\n  }\n}\n",
            &["Worker", "Run", "Help"],
            "Run",
            "Help",
        );
    }

    #[test]
    fn sql_extraction() {
        // SQL has no calls; the dependency edge (view → table) rides the same
        // resolver, so `active_orders` resolves as a "caller" of `orders`.
        extracts(
            "schema.sql",
            "CREATE TABLE orders (id INT PRIMARY KEY);\nCREATE VIEW active_orders AS SELECT id FROM orders;\n",
            &["orders", "active_orders"],
            "active_orders",
            "orders",
        );
    }
}
