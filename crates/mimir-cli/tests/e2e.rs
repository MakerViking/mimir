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
            // Never let a test reach the real user's agent configs.
            .env("HOME", self.home.path())
            .env("USERPROFILE", self.home.path())
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

#[test]
fn hooks_install_bakes_custom_inject_url_into_recall_script() {
    // `install_hooks` deliberately early-returns whenever MIMIR_HOME is set
    // (isolated instances must never touch a real ~/.claude) — so unlike
    // every other test in this file, this one sets only HOME/USERPROFILE
    // (the fake-$HOME trick), leaving MIMIR_HOME unset, to actually exercise
    // the install path while staying fully sandboxed: Paths::standard()
    // resolves config/db/models under $HOME too (directories::ProjectDirs
    // honors $HOME on Linux/macOS), so nothing here can reach the real
    // user's files as long as HOME points at the temp dir.
    let fake_home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();

    // A real Claude Code install looks like ~/.claude existing already —
    // install_hooks skips silently otherwise.
    std::fs::create_dir_all(fake_home.path().join(".claude")).unwrap();

    // Seed config.toml with a custom inject_url *before* `init` runs, at
    // the exact path Paths::standard() resolves to under XDG defaults.
    let config_dir = fake_home.path().join(".config/mimir");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[hooks]\ninject_url = \"http://10.0.0.5:9999/inject\"\n",
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_mimir"))
        .args(["init", "--no-model", "--hooks", "--auto-recall"])
        .env("HOME", fake_home.path())
        .env("USERPROFILE", fake_home.path())
        .env_remove("MIMIR_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("XDG_DATA_HOME")
        .env_remove("XDG_CACHE_HOME")
        .current_dir(cwd.path())
        .output()
        .expect("binary runs");
    assert!(
        out.status.success(),
        "init failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let script = std::fs::read_to_string(fake_home.path().join(".claude/hooks/mimir-recall.sh"))
        .expect("mimir-recall.sh must have been written");
    assert!(
        script.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://10.0.0.5:9999/inject}""#),
        "custom inject_url not baked into installed script:\n{script}"
    );

    // Re-running after changing the config must rewrite the script with
    // the new URL — install_hooks writes unconditionally, no staleness.
    std::fs::write(
        config_dir.join("config.toml"),
        "[hooks]\ninject_url = \"http://192.168.1.1:7000/inject\"\n",
    )
    .unwrap();
    let out2 = Command::new(env!("CARGO_BIN_EXE_mimir"))
        .args(["init", "--no-model", "--hooks", "--auto-recall"])
        .env("HOME", fake_home.path())
        .env("USERPROFILE", fake_home.path())
        .env_remove("MIMIR_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("XDG_DATA_HOME")
        .env_remove("XDG_CACHE_HOME")
        .current_dir(cwd.path())
        .output()
        .expect("binary runs");
    assert!(out2.status.success());
    let script2 =
        std::fs::read_to_string(fake_home.path().join(".claude/hooks/mimir-recall.sh")).unwrap();
    assert!(
        script2.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://192.168.1.1:7000/inject}""#),
        "re-init did not rewrite the script with the new inject_url:\n{script2}"
    );
}

#[test]
fn context_guard_hooks_install_and_are_idempotent() {
    // Same fake-$HOME trick as the recall-hook test above: install_hooks
    // early-returns under MIMIR_HOME, so this exercises the real install
    // path while staying fully sandboxed via $HOME.
    let fake_home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(fake_home.path().join(".claude")).unwrap();

    let run_init = || {
        Command::new(env!("CARGO_BIN_EXE_mimir"))
            .args(["init", "--no-model", "--hooks", "--context-guard", "pause"])
            .env("HOME", fake_home.path())
            .env("USERPROFILE", fake_home.path())
            .env_remove("MIMIR_HOME")
            .env_remove("XDG_CONFIG_HOME")
            .env_remove("XDG_DATA_HOME")
            .env_remove("XDG_CACHE_HOME")
            .current_dir(cwd.path())
            .output()
            .expect("binary runs")
    };

    let out = run_init();
    assert!(
        out.status.success(),
        "init failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let hooks_dir = fake_home.path().join(".claude/hooks");
    for script in [
        "mimir-anchors.sh",
        "mimir-context-guard-prompt.sh",
        "mimir-context-guard-precompact.sh",
        "mimir-context-guard-session.sh",
    ] {
        assert!(
            hooks_dir.join(script).is_file(),
            "{script} must have been written"
        );
    }

    let settings: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fake_home.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    let hooks = &settings["hooks"];
    assert_eq!(
        hooks["PreToolUse"].as_array().unwrap().len(),
        2,
        "rewrite + anchors"
    );
    assert_eq!(hooks["UserPromptSubmit"].as_array().unwrap().len(), 1);
    assert_eq!(hooks["PreCompact"].as_array().unwrap().len(), 1);
    assert_eq!(
        hooks["SessionStart"].as_array().unwrap().len(),
        2,
        "rules pack + context guard"
    );

    // Re-running must not duplicate any entry.
    let out2 = run_init();
    assert!(out2.status.success());
    let settings2: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fake_home.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    let hooks2 = &settings2["hooks"];
    assert_eq!(hooks2["PreToolUse"].as_array().unwrap().len(), 2);
    assert_eq!(hooks2["UserPromptSubmit"].as_array().unwrap().len(), 1);
    assert_eq!(hooks2["PreCompact"].as_array().unwrap().len(), 1);
    assert_eq!(hooks2["SessionStart"].as_array().unwrap().len(), 2);
}

#[test]
fn context_guard_off_by_default_adds_no_new_hook_entries() {
    // The default (`mimir init --hooks`, no --context-guard) must be
    // byte-identical, for hooks purposes, to pre-context-guard behavior:
    // no UserPromptSubmit/PreCompact entries, and no second SessionStart
    // entry, and none of the three context-guard scripts get written —
    // only the unconditional anchors script does.
    let fake_home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(fake_home.path().join(".claude")).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_mimir"))
        .args(["init", "--no-model", "--hooks"])
        .env("HOME", fake_home.path())
        .env("USERPROFILE", fake_home.path())
        .env_remove("MIMIR_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("XDG_DATA_HOME")
        .env_remove("XDG_CACHE_HOME")
        .current_dir(cwd.path())
        .output()
        .expect("binary runs");
    assert!(out.status.success());

    let hooks_dir = fake_home.path().join(".claude/hooks");
    assert!(hooks_dir.join("mimir-anchors.sh").is_file());
    for script in [
        "mimir-context-guard-prompt.sh",
        "mimir-context-guard-precompact.sh",
        "mimir-context-guard-session.sh",
    ] {
        assert!(
            !hooks_dir.join(script).exists(),
            "{script} must not be written when context_guard is off"
        );
    }

    let settings: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fake_home.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    let hooks = &settings["hooks"];
    assert_eq!(
        hooks["PreToolUse"].as_array().unwrap().len(),
        2,
        "rewrite + anchors"
    );
    assert!(hooks.get("UserPromptSubmit").is_none());
    assert!(hooks.get("PreCompact").is_none());
    assert_eq!(
        hooks["SessionStart"].as_array().unwrap().len(),
        1,
        "rules pack only"
    );
}

#[test]
fn concurrent_writers_and_readers_no_sqlite_busy() {
    // CLI + MCP running at once is the normal, supported case. Simulate:
    // one thread writes memories while another searches, both as real
    // separate processes (separate connections, WAL).
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&[
        "remember",
        "seed memory for concurrent search",
        "-t",
        "note",
    ]);

    let home = h.home.path().to_path_buf();
    let cwd = h.cwd.path().to_path_buf();
    let run = move |args: Vec<String>, home: std::path::PathBuf, cwd: std::path::PathBuf| {
        Command::new(env!("CARGO_BIN_EXE_mimir"))
            .args(&args)
            .env("MIMIR_HOME", home)
            .current_dir(cwd)
            .output()
            .expect("spawn")
    };

    let writer = {
        let (home, cwd) = (home.clone(), cwd.clone());
        std::thread::spawn(move || {
            for i in 0..12 {
                let out = run(
                    vec![
                        "remember".into(),
                        format!("concurrent fact number {i} about turbines"),
                        "--force".into(),
                    ],
                    home.clone(),
                    cwd.clone(),
                );
                assert!(
                    out.status.success(),
                    "writer {i}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        })
    };
    let reader = std::thread::spawn(move || {
        for i in 0..12 {
            let out = run(
                vec!["recall".into(), "concurrent turbines".into()],
                home.clone(),
                cwd.clone(),
            );
            assert!(
                out.status.success(),
                "reader {i}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(!err.contains("locked"), "SQLITE_BUSY leaked: {err}");
        }
    });
    writer.join().unwrap();
    reader.join().unwrap();
}

#[test]
fn multiple_concurrent_writers_never_lock() {
    // The previous test had ONE writer; WAL handles that trivially. The real
    // hazard is two+ writers overlapping: with DEFERRED transactions one
    // aborts with "database is locked" on the read→write lock upgrade. With
    // IMMEDIATE transactions + busy_timeout they serialize and wait. Each
    // recall also writes (record_shown), so a recall+remember loop is a
    // genuine writer.
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let home = h.home.path().to_path_buf();
    let cwd = h.cwd.path().to_path_buf();

    let worker = |home: std::path::PathBuf, cwd: std::path::PathBuf, tag: usize| {
        std::thread::spawn(move || {
            for i in 0..10 {
                for args in [
                    vec![
                        "remember".to_string(),
                        format!("writer {tag} fact {i} about pumps and valves"),
                        "--force".to_string(),
                    ],
                    vec!["recall".to_string(), "pumps valves".to_string()],
                ] {
                    let out = Command::new(env!("CARGO_BIN_EXE_mimir"))
                        .args(&args)
                        .env("MIMIR_HOME", &home)
                        .env("HOME", &home)
                        .current_dir(&cwd)
                        .output()
                        .expect("spawn");
                    let err = String::from_utf8_lossy(&out.stderr);
                    assert!(out.status.success(), "w{tag} {args:?}: {err}");
                    assert!(!err.contains("locked"), "SQLITE_BUSY leaked: {err}");
                }
            }
        })
    };

    let handles: Vec<_> = (0..3)
        .map(|t| worker(home.clone(), cwd.clone(), t))
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn server_mode_sync_roundtrip_and_auth() {
    use std::process::Stdio;
    let bin = env!("CARGO_BIN_EXE_mimir");
    // Per-test-binary port to avoid collisions with parallel test binaries.
    let port: u16 = 40000 + (std::process::id() % 2000) as u16;
    let endpoint = format!("http://127.0.0.1:{port}");
    let token = "test-token-xyz";

    let hub = tempfile::tempdir().unwrap();
    let c1 = tempfile::tempdir().unwrap();
    let c2 = tempfile::tempdir().unwrap();

    let init = |home: &Path| {
        let out = Command::new(bin)
            .args(["init", "--no-model"])
            .env("MIMIR_HOME", home)
            .env("HOME", home)
            .output()
            .unwrap();
        assert!(out.status.success());
    };
    init(hub.path());
    init(c1.path());
    init(c2.path());
    // Minimal config puts the clients in server mode (other sections default).
    for c in [c1.path(), c2.path()] {
        std::fs::write(
            c.join("config.toml"),
            format!("[sync]\nmode = \"server\"\nendpoint = \"{endpoint}\"\n"),
        )
        .unwrap();
    }

    let mut hubproc = Command::new(bin)
        .args(["serve", "--bind", &format!("127.0.0.1:{port}")])
        .env("MIMIR_HOME", hub.path())
        .env("HOME", hub.path())
        .env("MIMIR_SYNC_TOKEN", token)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    let client = |home: &Path, tok: &str, args: &[&str]| -> Output {
        Command::new(bin)
            .args(args)
            .env("MIMIR_HOME", home)
            .env("HOME", home)
            .env("MIMIR_SYNC_TOKEN", tok)
            .output()
            .unwrap()
    };

    // Wait for the hub to accept connections.
    let mut ready = false;
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if client(c1.path(), token, &["sync", "pull"]).status.success() {
            ready = true;
            break;
        }
    }
    assert!(ready, "hub did not become ready");

    client(
        c1.path(),
        token,
        &[
            "remember",
            "gearbox oil change interval is 2000 hours",
            "-t",
            "note",
            "-g",
        ],
    );
    assert!(client(c1.path(), token, &["sync"]).status.success());
    assert!(client(c2.path(), token, &["sync"]).status.success());
    let recall = client(c2.path(), token, &["recall", "gearbox oil"]);
    let stdout = String::from_utf8_lossy(&recall.stdout);
    assert!(
        stdout.contains("gearbox"),
        "c2 recalled the synced memory: {stdout}"
    );

    // Wrong token is rejected.
    let bad = client(c2.path(), "wrong-token", &["sync"]);
    assert!(!bad.status.success(), "wrong token must fail");

    let _ = hubproc.kill();
    let _ = hubproc.wait();
}
