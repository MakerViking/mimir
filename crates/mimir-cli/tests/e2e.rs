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

    // watchdog mode: healthy = completely silent, exit 0
    let out = h.run(&["doctor", "--check"]);
    assert!(
        out.status.success(),
        "doctor --check failed on healthy store"
    );
    assert!(
        out.stdout.is_empty(),
        "doctor --check must be silent when healthy"
    );
}

#[test]
fn doctor_check_fails_on_corrupt_store() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&["doctor", "--check"]);

    std::fs::write(h.home.path().join("mimir.db"), "not a sqlite database").unwrap();
    let out = h.run(&["doctor", "--check"]);
    assert!(!out.status.success(), "doctor --check passed on garbage db");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("FAIL db"), "expected FAIL db line, got: {err}");
}

#[test]
fn doctor_check_detects_fts_index_drift() {
    // The silent-recall failure mode: node_fts (external content) drifts
    // from node, PRAGMA integrity_check stays green, searches return
    // nothing. Reproduce by deleting a node with the sync trigger dropped.
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&["remember", "fts drift canary", "-t", "note"]);
    h.ok(&["doctor", "--check"]);

    let conn = rusqlite::Connection::open(h.home.path().join("mimir.db")).unwrap();
    conn.execute_batch("DROP TRIGGER node_ad; DELETE FROM node WHERE kind = 'memory';")
        .unwrap();
    drop(conn);

    let out = h.run(&["doctor", "--check"]);
    assert!(
        !out.status.success(),
        "doctor --check passed on drifted FTS index"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("FAIL fts5 index"),
        "expected FAIL fts5 index, got: {err}"
    );
    assert!(err.contains("rebuild"), "remedy hint missing: {err}");
}

#[test]
fn remember_refuses_secret_smuggled_in_tags() {
    // The secrets guard scans `text`, but a real secret can be smuggled
    // through `--tags` just as easily — `tags_text` is indexed and
    // searched the same as the body, so an unscanned tag would leak the
    // secret into recall in plain text.
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);

    let out = h.run(&[
        "remember",
        "harmless note about a deploy",
        "-t",
        "note",
        "--tags",
        "AKIAABCDEFGHIJKLMNOP,deploy",
        "-g",
    ]);
    assert!(!out.status.success(), "secret-in-tags must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("AWS access key"),
        "expected secret refusal, got: {err}"
    );

    let hits = h.ok(&["recall", "AKIAABCDEFGHIJKLMNOP", "-g"]);
    assert!(
        !hits.contains("AKIAABCDEFGHIJKLMNOP"),
        "secret must not be stored/recallable: {hits}"
    );
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

// The fake-$HOME sandbox below can't work on Windows: `directories` resolves
// paths through the Known Folder API, which ignores HOME/USERPROFILE env
// overrides — the subprocess would either fail to resolve a home or escape
// the sandbox into the real profile. Hook installation logic is identical
// across platforms, so Linux/macOS coverage is sufficient.
#[cfg(not(windows))]
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
    // the exact path Paths::standard() resolves to under the fake $HOME —
    // which differs per platform (ProjectDirs: XDG on Linux, Application
    // Support on macOS).
    let config_dir = if cfg!(target_os = "macos") {
        fake_home.path().join("Library/Application Support/mimir")
    } else {
        fake_home.path().join(".config/mimir")
    };
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

// Windows: see the cfg note on hooks_install_bakes_custom_inject_url_….
#[cfg(not(windows))]
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

// Windows: see the cfg note on hooks_install_bakes_custom_inject_url_….
#[cfg(not(windows))]
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
    // Per-test-binary port to avoid collisions with parallel test binaries.
    let port: u16 = 40000 + (std::process::id() % 2000) as u16;
    let endpoint = format!("http://127.0.0.1:{port}");
    let token = "test-token-xyz";

    let hub = tempfile::tempdir().unwrap();
    init_server_home(hub.path());
    let _hub = spawn_hub(hub.path(), port, token);
    wait_for_hub(port);

    let c1 = tempfile::tempdir().unwrap();
    let c2 = tempfile::tempdir().unwrap();
    for c in [c1.path(), c2.path()] {
        init_server_home(c);
        write_server_config(c, &endpoint);
    }

    sync_client_ok(
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
    sync_client_ok(c1.path(), token, &["sync"]);
    sync_client_ok(c2.path(), token, &["sync"]);
    let stdout = sync_client_ok(c2.path(), token, &["recall", "gearbox oil"]);
    assert!(
        stdout.contains("gearbox"),
        "c2 recalled the synced memory: {stdout}"
    );

    // Wrong token is rejected.
    let bad = sync_client(c2.path(), "wrong-token", &["sync"]);
    assert!(!bad.status.success(), "wrong token must fail");
}

// ---- shared plumbing for the server-mode sync tests ----

fn init_server_home(home: &Path) {
    let bin = env!("CARGO_BIN_EXE_mimir");
    let out = Command::new(bin)
        .args(["init", "--no-model"])
        .env("MIMIR_HOME", home)
        .env("HOME", home)
        .output()
        .unwrap();
    assert!(out.status.success());
}

fn write_server_config(home: &Path, endpoint: &str) {
    std::fs::write(
        home.join("config.toml"),
        format!("[sync]\nmode = \"server\"\nendpoint = \"{endpoint}\"\n"),
    )
    .unwrap();
}

/// Hub process wrapper that kills the child on drop, so a panicking test
/// can't leak a live `mimir serve` holding its port.
struct HubGuard(std::process::Child);

impl Drop for HubGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_hub(home: &Path, port: u16, token: &str) -> HubGuard {
    let bin = env!("CARGO_BIN_EXE_mimir");
    HubGuard(
        Command::new(bin)
            .args(["serve", "--bind", &format!("127.0.0.1:{port}")])
            .env("MIMIR_HOME", home)
            .env("HOME", home)
            .env("MIMIR_SYNC_TOKEN", token)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    )
}

/// Poll the hub's listening port instead of round-tripping a full client
/// process, since we only need to know the socket is accepting connections.
fn wait_for_hub(port: u16) {
    for _ in 0..50 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("hub did not become ready on port {port}");
}

fn sync_client(home: &Path, token: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mimir"))
        .args(args)
        .env("MIMIR_HOME", home)
        .env("HOME", home)
        .env("MIMIR_SYNC_TOKEN", token)
        .output()
        .unwrap()
}

fn sync_client_ok(home: &Path, token: &str, args: &[&str]) -> String {
    let out = sync_client(home, token, args);
    assert!(
        out.status.success(),
        "mimir {args:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Rewind a client's own `sync.last_push` watermark by writing straight into
/// its sqlite db — simulating a retried push after the client crashed (or the
/// response was lost) between the hub applying a batch and the client
/// persisting its advanced watermark.
fn rewind_last_push(db_file: &Path) {
    let conn = rusqlite::Connection::open(db_file).unwrap();
    mimir_core::replicate::set_watermark(&conn, "last_push", 0).unwrap();
}

/// Directly doctor a record's `updated_at`, bypassing the real system
/// clock — used to simulate a peer with a badly skewed clock, and to step
/// deterministically past the 1-second timestamp resolution instead of
/// sleeping.
fn skew_updated_at(db_file: &Path, body_needle: &str, future_updated_at: i64) {
    let conn = rusqlite::Connection::open(db_file).unwrap();
    let changed = conn
        .execute(
            "UPDATE node SET updated_at = ?1 WHERE body LIKE ?2",
            rusqlite::params![future_updated_at, format!("%{body_needle}%")],
        )
        .unwrap();
    assert_eq!(changed, 1, "expected exactly one node to skew");
}

/// Regression for the watermark clock-domain bug: retrying an already-applied
/// push (e.g. after the client crashed before persisting its advanced
/// watermark) must be a hub-side no-op, not a duplicate.
#[test]
fn server_sync_push_retry_is_idempotent() {
    // Own port range so this doesn't collide with the other server-mode tests
    // running concurrently in the same test binary.
    let port: u16 = 45000 + (std::process::id() % 2000) as u16;
    let endpoint = format!("http://127.0.0.1:{port}");
    let token = "test-token-repush";

    let hub = tempfile::tempdir().unwrap();
    init_server_home(hub.path());
    let _hub = spawn_hub(hub.path(), port, token);
    wait_for_hub(port);

    let x = tempfile::tempdir().unwrap();
    let y = tempfile::tempdir().unwrap();
    for home in [x.path(), y.path()] {
        init_server_home(home);
        write_server_config(home, &endpoint);
    }

    sync_client_ok(
        x.path(),
        token,
        &[
            "remember",
            "gasket torque spec is 45 Nm",
            "-t",
            "note",
            "-g",
        ],
    );

    let push1 = sync_client_ok(x.path(), token, &["sync", "push"]);
    assert!(
        push1.contains("hub applied 1"),
        "first push must apply the new memory: {push1}"
    );

    rewind_last_push(&x.path().join("mimir.db"));
    let push2 = sync_client_ok(x.path(), token, &["sync", "push"]);
    assert!(
        push2.contains("hub applied 0"),
        "retried push of the same batch must be a no-op on the hub, not a duplicate: {push2}"
    );

    // A fresh peer sees exactly one copy — no duplicate landed on the hub.
    sync_client_ok(y.path(), token, &["sync"]);
    let listing = sync_client_ok(y.path(), token, &["list", "-g", "--json"]);
    let n = listing
        .lines()
        .filter(|l| l.contains("gasket torque"))
        .count();
    assert_eq!(n, 1, "exactly one copy after a retried push: {listing}");
}

/// Regression for cursor clamping (`cursor_advance` in sync/mod.rs): one
/// future-dated row must not let a client's own push cursor (self-poisoning)
/// or a reader's pull cursor (receive-blindness) jump past normally-
/// timestamped later changes. Both asserts fail on unclamped cursors.
#[test]
fn server_sync_cursor_clamped_to_local_clock() {
    let port: u16 = 55000 + (std::process::id() % 2000) as u16;
    let endpoint = format!("http://127.0.0.1:{port}");
    let token = "test-token-clamp";

    let hub = tempfile::tempdir().unwrap();
    init_server_home(hub.path());
    let _hub = spawn_hub(hub.path(), port, token);
    wait_for_hub(port);

    let x = tempfile::tempdir().unwrap();
    let y = tempfile::tempdir().unwrap();
    for home in [x.path(), y.path()] {
        init_server_home(home);
        write_server_config(home, &endpoint);
    }

    // X's own store ends up holding a future-dated row (its clock was briefly
    // wrong), and the row reaches the hub.
    sync_client_ok(
        x.path(),
        token,
        &["remember", "clamp skew marker", "-t", "note", "-g"],
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    skew_updated_at(
        &x.path().join("mimir.db"),
        "clamp skew marker",
        now + 10 * 365 * 24 * 3600,
    );
    sync_client_ok(x.path(), token, &["sync", "push"]);

    // Y pulls the poisoned row; its pull cursor must stay clamped to Y's own
    // clock instead of jumping ten years ahead.
    sync_client_ok(y.path(), token, &["sync", "pull"]);

    // Self-poisoning check: X's OWN next memory must still reach the hub even
    // though X's table contains the future-dated row.
    sync_client_ok(
        x.path(),
        token,
        &[
            "remember",
            "x normal memory about camshafts",
            "-t",
            "note",
            "-g",
        ],
    );
    let push = sync_client_ok(x.path(), token, &["sync", "push"]);
    assert!(
        push.contains("hub applied 1"),
        "X's own later memory must survive X's own skewed row: {push}"
    );

    // Receive-blindness check: Y's next pull must deliver X's new memory.
    sync_client_ok(y.path(), token, &["sync", "pull"]);
    let list_y = sync_client_ok(y.path(), token, &["list", "-g", "--json"]);
    assert!(
        list_y.contains("camshafts"),
        "Y must keep receiving new changes after pulling a poisoned row: {list_y}"
    );
}

/// Regression for the srv_push watermark bug: `last_push` must come from the
/// pushing client's OWN local clock domain (`batch.watermark`), not from the
/// hub's post-apply GLOBAL watermark. A peer with a badly skewed clock must
/// not be able to silently blind another, well-behaved peer to its own future
/// pushes.
#[test]
fn server_sync_survives_peer_clock_skew() {
    let port: u16 = 50000 + (std::process::id() % 2000) as u16;
    let endpoint = format!("http://127.0.0.1:{port}");
    let token = "test-token-skew";

    let hub = tempfile::tempdir().unwrap();
    init_server_home(hub.path());
    let _hub = spawn_hub(hub.path(), port, token);
    wait_for_hub(port);

    let x = tempfile::tempdir().unwrap();
    let y = tempfile::tempdir().unwrap();
    for home in [x.path(), y.path()] {
        init_server_home(home);
        write_server_config(home, &endpoint);
    }
    // A fresh reader peer to check the hub's state: a first-ever pull
    // (since = 0) returns everything, keeping the observer independent of
    // any cursor behavior. Pull-side cursor poisoning is real too — it is
    // covered by `server_sync_cursor_clamped_to_local_clock` below — so each
    // check uses a NEW temp home rather than leaning on the clamp under test.
    let hub_state = || -> String {
        let reader = tempfile::tempdir().unwrap();
        init_server_home(reader.path());
        write_server_config(reader.path(), &endpoint);
        sync_client_ok(reader.path(), token, &["sync"]);
        sync_client_ok(reader.path(), token, &["list", "-g", "--json"])
    };

    // X writes a memory, then we doctor its updated_at ~10 years into the
    // future (simulating a peer with a badly wrong clock) before pushing.
    // This pushes the hub's GLOBAL high watermark years ahead.
    sync_client_ok(
        x.path(),
        token,
        &["remember", "skew marker record", "-t", "note", "-g"],
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    skew_updated_at(
        &x.path().join("mimir.db"),
        "skew marker record",
        now + 10 * 365 * 24 * 3600,
    );
    sync_client_ok(x.path(), token, &["sync", "push"]);

    // Y observes the hub's now-inflated global watermark via an otherwise
    // empty push (Y has nothing local to send yet). Under the old code this
    // alone would poison Y's last_push, since it was taken from the hub's
    // global watermark instead of Y's own local batch.
    sync_client_ok(y.path(), token, &["sync", "push"]);

    // Y does real work and pushes it — it must reach the hub despite the
    // hub's watermark being years in the future.
    sync_client_ok(
        y.path(),
        token,
        &[
            "remember",
            "y's real memory about bearings",
            "-t",
            "note",
            "-g",
        ],
    );
    let push = sync_client_ok(y.path(), token, &["sync", "push"]);
    assert!(
        push.contains("hub applied 1"),
        "Y's memory must reach the hub despite peer clock skew: {push}"
    );

    let listing = hub_state();
    assert!(
        listing.contains("bearings"),
        "Y's memory landed on the hub: {listing}"
    );

    // Y edits the memory and pushes again — the edit must land too.
    let list_y = sync_client_ok(y.path(), token, &["list", "-g"]);
    let id = list_y
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    sync_client_ok(
        y.path(),
        token,
        &["edit", id, "y's real memory about bearings, revised torque"],
    );
    // `updated_at` is second-resolution and last-write-wins overwrites only
    // on a STRICTLY newer timestamp, so a same-second edit would spuriously
    // fail for reasons unrelated to the watermark fix under test. Step the
    // edited row 2s ahead deterministically instead of sleeping past the
    // second boundary.
    skew_updated_at(&y.path().join("mimir.db"), "revised torque", now + 2);
    let push2 = sync_client_ok(y.path(), token, &["sync", "push"]);
    assert!(
        push2.contains("hub applied 1"),
        "Y's edit must also reach the hub: {push2}"
    );

    let listing2 = hub_state();
    assert!(
        listing2.contains("revised torque"),
        "Y's edit landed on the hub: {listing2}"
    );
}

/// A stdio MCP session with the default `[daemon] inference = "auto"` and NO
/// daemon running: the startup probe fails fast, the session comes up, and
/// recall answers locally — delegation must never block or break a session.
#[test]
fn mcp_stdio_recall_works_without_daemon() {
    use std::io::{BufRead, BufReader, Write};

    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&[
        "remember",
        "SCRAM auth rejects non-ASCII passwords",
        "-t",
        "gotcha",
    ]);
    // Point delegation at a port nothing listens on (fast refusal) instead
    // of the default :8077, where the developer's real daemon might live.
    std::fs::write(
        h.home.path().join("config.toml"),
        "[hooks]\ninject_url = \"http://127.0.0.1:9/inject\"\n",
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_mimir"))
        .arg("mcp")
        .env("MIMIR_HOME", h.home.path())
        .env("HOME", h.home.path())
        .env("USERPROFILE", h.home.path())
        .current_dir(h.cwd.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    // Take the pipes before the kill-on-drop guard owns the child; `stdin`
    // must stay alive (open) until the response is read — EOF would race
    // the session shutdown against the reply.
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let _guard = HubGuard(child);

    let requests = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall","arguments":{"query":"SCRAM auth passwords"}}}"#,
        "\n"
    );
    stdin.write_all(requests.as_bytes()).unwrap();
    stdin.flush().unwrap();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.contains("\"id\":2") {
                let _ = tx.send(line);
                return;
            }
        }
    });
    let response = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("recall response within 30s despite the dead daemon URL");
    assert!(
        response.contains("SCRAM"),
        "recall must answer locally when the daemon is unreachable: {response}"
    );
    drop(stdin);
}

/// `mimir daemon` exposes the inference-delegation endpoints: the probe
/// answers the configured model names, and a model-less store says 503 on
/// /embed (the client's back-off signal) rather than pretending to help.
#[test]
fn daemon_serves_inference_endpoints() {
    let home = tempfile::tempdir().unwrap();
    init_server_home(home.path());
    // Different base than the sync-hub tests so parallel binaries can't collide.
    let port: u16 = 43000 + (std::process::id() % 2000) as u16;
    std::fs::write(
        home.path().join("config.toml"),
        format!("[hooks]\ninject_url = \"http://127.0.0.1:{port}/inject\"\n"),
    )
    .unwrap();
    let bin = env!("CARGO_BIN_EXE_mimir");
    let _daemon = HubGuard(
        Command::new(bin)
            .arg("daemon")
            .env("MIMIR_HOME", home.path())
            .env("HOME", home.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_for_hub(port);

    let info: serde_json::Value = ureq::get(&format!("http://127.0.0.1:{port}/inference"))
        .call()
        .expect("/inference answers")
        .into_json()
        .unwrap();
    assert_eq!(info["embedding_model"], "bge-small-en-v1.5");
    assert_eq!(info["rerank_model"], "jina-reranker-v1-turbo-en");

    let embed = ureq::post(&format!("http://127.0.0.1:{port}/embed"))
        .set("Content-Type", "application/json")
        .send_string(r#"{"texts":["hi"]}"#);
    match embed {
        Err(ureq::Error::Status(code, _)) => assert_eq!(code, 503, "model-less store => 503"),
        other => panic!("expected 503 from /embed on a model-less store, got {other:?}"),
    }
}
