//! End-to-end test of the phase-1 surface, driving the real binary with
//! an isolated MIMIR_HOME. BM25-only (init --no-model) so it runs
//! offline and fast; the vector path has unit coverage in mimir-core.

use std::path::Path;
use std::process::{Command, Output};

struct Harness {
    home: tempfile::TempDir,
    cwd: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        Harness {
            home: tempfile::tempdir().unwrap(),
            cwd: tempfile::tempdir().unwrap(),
        }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_mimir"))
            .args(args)
            .env("MIMIR_HOME", self.home.path())
            .current_dir(self.cwd.path())
            .output()
            .expect("binary runs")
    }

    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "mimir {args:?} failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn write(&self, rel: &str, content: &str) {
        let path = self.cwd.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
}

#[test]
fn full_phase1_flow() {
    let h = Harness::new();

    // init (offline)
    let out = h.ok(&["init", "--no-model"]);
    assert!(out.contains("claude mcp add"), "init prints MCP hint");
    assert!(h.home.path().join("mimir.db").exists());

    // remember ×3 (one global)
    let out = h.ok(&[
        "remember",
        "SQLite WAL mode lets a reader and writer coexist",
        "-t",
        "gotcha",
        "--tags",
        "sqlite",
    ]);
    assert!(out.starts_with("m:"), "agent-format line, got: {out}");
    h.ok(&[
        "remember",
        "We chose RRF over score normalization",
        "-t",
        "decision",
    ]);
    h.ok(&[
        "remember",
        "-g",
        "Always pin CI toolchains",
        "-t",
        "insight",
    ]);

    // duplicate refused with nonzero exit
    let dup = h.run(&[
        "remember",
        "SQLite WAL mode lets a reader and writer coexist",
    ]);
    assert!(!dup.status.success(), "duplicate must be refused");
    assert!(String::from_utf8_lossy(&dup.stderr).contains("near-duplicate"));

    // docs add + index
    h.write(
        "docs/guide.md",
        "# Setup\n\nInstall the widget frobnicator from the official site.\n\n# Teardown\n\nRemove all frobnicated widgets carefully.\n",
    );
    h.ok(&["docs", "add", "docs", "--name", "guide-docs"]);
    let out = h.ok(&["index"]);
    assert!(out.contains("1 indexed"), "index output: {out}");

    // recall across kinds
    let out = h.ok(&["recall", "frobnicator", "--kind", "doc"]);
    assert!(out.contains("guide"), "doc recall: {out}");
    let out = h.ok(&["recall", "WAL", "--kind", "memory"]);
    assert!(out.contains("gotcha"), "memory recall: {out}");

    // recall respects --json (one JSON object per line)
    let out = h.ok(&["recall", "WAL", "--json"]);
    let first = out.lines().next().unwrap();
    let v: serde_json::Value = serde_json::from_str(first).expect("valid JSON");
    assert_eq!(v["kind"], "memory");

    // get by short id (from list) shows the full body
    let listing = h.ok(&["list", "-t", "decision"]);
    let id = listing.split_whitespace().next().unwrap().to_string();
    let out = h.ok(&["get", &id]);
    assert!(out.contains("We chose RRF"), "get output: {out}");

    // get path:lines slices the file from disk
    let out = h.ok(&["get", "guide.md:3-3"]);
    assert!(out.contains("3  Install the widget"), "slice: {out}");

    // link two memories, edge shows in get
    let other = h.ok(&["list", "-t", "gotcha"]);
    let gotcha_id = other.split_whitespace().next().unwrap().to_string();
    h.ok(&["link", &id, &gotcha_id, "--rel", "relates"]);
    let out = h.ok(&["get", &id]);
    assert!(out.contains("relates"), "edge missing: {out}");

    // incremental: reindex with no changes
    let out = h.ok(&["index"]);
    assert!(out.contains("1 unchanged"), "incremental: {out}");

    // file removal soft-deletes
    std::fs::remove_file(h.cwd.path().join("docs/guide.md")).unwrap();
    let out = h.ok(&["index"]);
    assert!(out.contains("1 removed"), "removal: {out}");
    let out = h.ok(&["recall", "frobnicator", "--kind", "doc"]);
    assert!(out.contains("no results"), "deleted doc still found: {out}");

    // forget hides from list
    h.ok(&["forget", &gotcha_id]);
    let out = h.ok(&["list", "-t", "gotcha"]);
    assert!(out.contains("no memories"), "forget failed: {out}");

    // status + doctor stay healthy at the end
    let out = h.ok(&["status"]);
    assert!(out.contains("memory"), "status: {out}");
    h.ok(&["doctor"]);
}

#[test]
fn status_works_outside_any_project() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let out = h.ok(&["status"]);
    assert!(
        out.contains("none — global scope") || out.contains("project"),
        "status: {out}"
    );
}

#[test]
fn isolated_home_never_touches_user_dirs() {
    // Guard: MIMIR_HOME must fully isolate (config, db, models in one dir).
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let entries: Vec<String> = std::fs::read_dir(h.home.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(entries.iter().any(|e| e == "config.toml"), "{entries:?}");
    assert!(entries.iter().any(|e| e == "mimir.db"), "{entries:?}");
    assert!(Path::new(env!("CARGO_BIN_EXE_mimir")).exists());
}
