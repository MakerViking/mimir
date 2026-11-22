//! End-to-end test of the phase-1 surface, driving the real binary with
//! an isolated MIMIR_HOME. BM25-only (init --no-model) so it runs
//! offline and fast; the vector path has unit coverage in mimir-core.

use std::path::Path;
use std::process::{Command, Output};

/// The binary under test, started without the developer's own Mimir
/// environment. A machine that runs Mimir for real exports a sync token, and
/// may export an HTTP token, a daemon address or a project directory; a test
/// that inherited them would carry a real credential into every process it
/// starts, or bind and probe the address the real daemon is listening on.
/// A test that needs one of these sets it explicitly on the returned command.
fn mimir_bin() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mimir"));
    for var in [
        "MIMIR_SYNC_TOKEN",
        "MIMIR_HTTP_TOKEN",
        "MIMIR_INJECT_URL",
        "MIMIR_PROJECT",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

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
        mimir_bin()
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

    let out = mimir_bin()
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
    let out2 = mimir_bin()
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
        mimir_bin()
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
        3,
        "rules pack + brief + context guard"
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
    assert_eq!(hooks2["SessionStart"].as_array().unwrap().len(), 3);
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

    let out = mimir_bin()
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
        2,
        "rules pack + brief only"
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
        mimir_bin()
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
                    let out = mimir_bin()
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
    let port: u16 = free_port();
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
    let out = mimir_bin()
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
    HubGuard(
        mimir_bin()
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
    mimir_bin()
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
    let port: u16 = free_port();
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
/// timestamped later changes. The first assert fails on an unclamped push
/// cursor. The second did on an unclamped pull cursor for as long as the hub
/// stamped a pull with its newest row timestamp; this hub stamps it with its
/// own clock (`replicate::relay_since`), so on the pull side the clamp
/// guards against a hub on an older build.
#[test]
fn server_sync_cursor_clamped_to_local_clock() {
    let port: u16 = free_port();
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
    let port: u16 = free_port();
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
    // second boundary — measured from a FRESH timestamp, not the `now`
    // captured at test start: a dozen process spawns happen in between,
    // and on a slow CI runner (observed on windows-latest) more than 2s
    // of drift made `now + 2` OLDER than the row's real updated_at, so
    // the hub's LWW correctly refused the edit and the test flaked.
    let edit_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 2;
    skew_updated_at(&y.path().join("mimir.db"), "revised torque", edit_ts);
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

    let mut child = mimir_bin()
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
    let port: u16 = free_port();
    std::fs::write(
        home.path().join("config.toml"),
        format!("[hooks]\ninject_url = \"http://127.0.0.1:{port}/inject\"\n"),
    )
    .unwrap();
    let _daemon = HubGuard(
        mimir_bin()
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

/// Session brief: startup fire renders, same-session repeat is suppressed,
/// resume and unknown sources are silent, and the kill-switch works.
#[test]
fn session_brief_fires_suppresses_and_respects_kill_switch() {
    use std::io::Write;
    use std::process::Stdio;

    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&[
        "remember",
        "never test against prod without MIMIR_HOME isolation",
        "-t",
        "gotcha",
    ]);
    h.ok(&[
        "remember",
        "we chose sqlite over postgres",
        "-t",
        "decision",
    ]);

    let run_show = |source: &str, session: &str| -> String {
        let mut child = mimir_bin()
            .args(["brief", "show"])
            .env("MIMIR_HOME", h.home.path())
            .env("HOME", h.home.path())
            .env("USERPROFILE", h.home.path())
            .current_dir(h.cwd.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let event = format!(
            r#"{{"session_id":"{session}","source":"{source}","cwd":{:?}}}"#,
            h.cwd.path().to_string_lossy()
        );
        child
            .stdin
            .take()
            .unwrap()
            .write_all(event.as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "brief show must exit 0");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    // Kill-switch: an explicit `enabled = false` keeps the hook silent
    // (the default has been ON since the §9 gate passed, so the switch is
    // exercised by writing false, not by relying on the default).
    std::fs::write(
        h.home.path().join("config.toml"),
        "[brief]\nenabled = false\n",
    )
    .unwrap();
    assert_eq!(
        run_show("startup", "s1"),
        "",
        "disabled brief must be silent"
    );

    std::fs::write(
        h.home.path().join("config.toml"),
        "[brief]\nenabled = true\n",
    )
    .unwrap();

    // Startup fire renders both memories with imperative labels + header.
    let first = run_show("startup", "s1");
    assert!(first.contains("Mimir guards (do not violate):"), "{first}");
    assert!(first.contains("- GOTCHA [m:") && first.contains("- DECISION [m:"));

    // Compact recap RE-ANCHORS: the context was wiped, so the same-session
    // shown set does NOT suppress — the top guards come back (under the
    // smaller recap budget). Only OTHER sessions' recent shows suppress.
    let recap1 = run_show("compact", "s1");
    assert!(
        recap1.contains("never test against prod"),
        "recap must re-show the top guard after a context reset: {recap1:?}"
    );

    // A newly captured gotcha joins the next recap alongside the re-anchored
    // guards.
    h.ok(&[
        "remember",
        "new gotcha captured mid-session",
        "-t",
        "gotcha",
    ]);
    let recap2 = run_show("compact", "s1");
    assert!(
        recap2.contains("new gotcha captured mid-session"),
        "fresh capture must surface on recap: {recap2:?}"
    );

    // A second session inside the wall-clock window IS suppressed by s1's
    // shows (startup semantics keep the cross-session dedup).
    assert_eq!(
        run_show("startup", "s2"),
        "",
        "sibling session within 6h must not repeat s1's brief"
    );

    // Resume and unknown sources: fail closed even with unseen content.
    h.ok(&["remember", "another gotcha for resume test", "-t", "gotcha"]);
    assert_eq!(run_show("resume", "s1"), "", "resume must never fire");
    assert_eq!(
        run_show("mystery", "s1"),
        "",
        "unknown source must fail closed"
    );

    // Dry-run preview works and lists candidates with scores.
    let preview = h.ok(&["brief"]);
    assert!(preview.contains("candidates (score desc"), "{preview}");
}

/// Shutdown of the stdio server (`mimir mcp`).
///
/// These exist because fourteen orphaned servers accumulated on a real
/// machine, the oldest fourteen days old, each holding the SQLite database
/// open. The EOF path was never the bug — it works, and the first test here
/// pins that — so a fix that only hardened EOF handling would have changed
/// nothing. The other two cover the ways the client can vanish *without*
/// producing an EOF.
#[cfg(unix)]
mod shutdown {
    use super::*;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    /// Poll for exit rather than `wait()`, so a hang fails the test instead
    /// of hanging the suite.
    fn exits_within(child: &mut std::process::Child, secs: u64) -> bool {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        false
    }

    fn spawn_server(h: &Harness) -> std::process::Child {
        mimir_bin()
            .arg("mcp")
            .env("MIMIR_HOME", h.home.path())
            .env("HOME", h.home.path())
            .current_dir(h.cwd.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawns")
    }

    #[test]
    fn exits_when_stdin_closes() {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let mut child = spawn_server(&h);
        drop(child.stdin.take()); // client closed the pipe
        assert!(
            exits_within(&mut child, 10),
            "server outlived its stdin closing"
        );
    }

    /// The client is gone but something else still holds the descriptor, so
    /// no EOF is ever delivered. Before the parent-death watchdog this ran
    /// forever.
    ///
    /// `sh` spawns the server and exits immediately, which reparents it
    /// exactly as a departing client does — after staying alive for a few
    /// seconds first, which is the real shape of the bug: a client serves a
    /// session and *then* exits. (A parent that dies before the server's
    /// `main` has recorded its pid is a genuine residual gap — see
    /// `PARENT_AT_START` in `mcp.rs`; it is a millisecond-wide window and
    /// not what produced the orphans.)
    /// The server's stdin stays the pipe
    /// whose write end this test holds for the whole run, so EOF cannot be
    /// what ends it; the pid goes through a file rather than stdout, because
    /// the server inherits `sh`'s stdout and reading that to EOF would block
    /// until the server had already exited — which silently made an earlier
    /// version of this test pass against the unfixed binary.
    #[test]
    fn exits_when_the_parent_dies_without_eof() {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let pidfile = h.cwd.path().join("server.pid");
        let mut launcher = Command::new("sh")
            .arg("-c")
            // `exec 3<&0` then `<&3`: a non-interactive shell redirects a
            // background job's stdin from /dev/null, which would hand the
            // server an instant EOF and make this test pass against an
            // unfixed binary. Duplicating the descriptor defeats that.
            .arg(format!(
                "exec 3<&0; {} mcp <&3 >/dev/null 2>&1 & echo $! > {}; sleep 4",
                env!("CARGO_BIN_EXE_mimir"),
                pidfile.display()
            ))
            .env("MIMIR_HOME", h.home.path())
            .env("HOME", h.home.path())
            .current_dir(h.cwd.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawns");
        let stdin_holder = launcher.stdin.take().expect("piped"); // held open
        launcher.wait().unwrap(); // sh exits; the server is reparented

        let pid: i32 = std::fs::read_to_string(&pidfile)
            .expect("pidfile")
            .trim()
            .parse()
            .expect("pid");

        let deadline = Instant::now() + Duration::from_secs(15);
        let mut gone = false;
        while Instant::now() < deadline {
            // signal 0 probes liveness without delivering anything
            if unsafe { libc::kill(pid, 0) } != 0 {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        drop(stdin_holder);
        if !gone {
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        assert!(gone, "server survived its parent with stdin still open");
    }

    #[test]
    fn exits_on_sigterm() {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let mut child = spawn_server(&h);
        let _stdin = child.stdin.take(); // keep it open: no EOF
        std::thread::sleep(Duration::from_millis(500));
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
        assert!(
            exits_within(&mut child, 10),
            "server ignored SIGTERM — installing a handler removed the \
             default action, so this must work"
        );
    }
}

/// `remember --link <unknown>` must fail before anything is written. The
/// link target used to be resolved after the insert, so the memory was
/// stored, the command exited 1, and the expiry, confidence and anchors the
/// caller asked for were skipped — while a corrected retry was refused as a
/// duplicate.
#[test]
fn remember_with_an_unknown_link_target_stores_nothing() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);

    let out = h.run(&[
        "remember",
        "-g",
        "The staging database rotates credentials every 30 days",
        "--link",
        "nope_not_here",
        "--expires-in",
        "30d",
    ]);
    assert!(!out.status.success(), "an unknown link target is an error");

    let listed = h.ok(&["list", "-g"]);
    assert!(
        listed.contains("no memories"),
        "the failed capture must not have been stored: {listed}"
    );

    // The corrected retry is a first capture, not a duplicate.
    let out = h.ok(&[
        "remember",
        "-g",
        "The staging database rotates credentials every 30 days",
        "--expires-in",
        "30d",
    ]);
    assert!(
        out.contains("expires in 30d"),
        "retry kept its expiry: {out}"
    );
}

/// `remember --json` answers with one JSON object. It used to print the
/// memory as JSON and then the expiry, the confidence and the link as prose
/// lines under it, so a caller parsing the output got a parse error on line
/// two for exactly the captures that asked for the most.
#[test]
fn remember_json_is_one_object_carrying_what_was_attached() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);

    let out = h.ok(&[
        "--json",
        "remember",
        "-g",
        "The staging database rotates credentials every 30 days",
        "--expires-in",
        "30d",
        "--confidence",
        "likely",
        "--resolves-when",
        "when the vault migration lands",
    ]);
    let lines: Vec<&str> = out.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 1, "expected a single JSON line: {out}");
    let value: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSON");
    assert_eq!(value["kind"], "memory");
    assert_eq!(value["lifecycle"], "live");
    assert!(value["expires_at"].is_i64(), "expiry as a field: {value}");
    assert_eq!(value["confidence"], "likely");
    assert_eq!(value["resolves_when"], "when the vault migration lands");
}

// ---------- memory lifecycle: forget, edit, supersede, anchor, rules ----------

/// First whitespace-delimited token of a command's stdout: the short id on
/// an agent-format line.
fn first_token(out: &str) -> String {
    out.split_whitespace().next().unwrap().to_string()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn db_conn(h: &Harness) -> rusqlite::Connection {
    rusqlite::Connection::open(h.home.path().join("mimir.db")).unwrap()
}

/// `forget` reaches every state it should: `--hard` an existing tombstone,
/// an archived memory, and it says no — in a sentence — to what it cannot
/// mean.
#[test]
fn forget_reaches_tombstones_and_archived_memories() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);

    // A tombstone can be removed for good; before, the reference no longer
    // resolved and the row could never be purged.
    let id = first_token(&h.ok(&["remember", "the canary lives in the staging vault"]));
    h.ok(&["forget", &id, "--reason", "wrong vault"]);
    let again = h.run(&["forget", &id]);
    assert!(!again.status.success());
    assert!(
        stderr_of(&again).contains("already forgotten"),
        "{}",
        stderr_of(&again)
    );
    let out = h.ok(&["forget", &id, "--hard"]);
    assert!(out.contains("(permanently)"), "{out}");
    let gone: i64 = db_conn(&h)
        .query_row(
            "SELECT count(*) FROM node WHERE body LIKE '%canary lives%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(gone, 0, "the row must be gone");

    // An archived memory: decay tombstoned it, so `remember` would take the
    // fact back. Forgetting it makes that a refusal.
    let text = "nextest needs its own install step on the build image";
    let id = first_token(&h.ok(&["remember", text]));
    db_conn(&h)
        .execute(
            "UPDATE node SET deleted_at = updated_at,
                    meta = json_set(meta, '$.archived', 1)
             WHERE body = ?1",
            [text],
        )
        .unwrap();
    let out = h.ok(&["forget", &id]);
    assert!(out.starts_with("forgot "), "{out}");
    let out = h.ok(&["history", &id]);
    assert!(out.contains("forgotten"), "history must say so: {out}");
    let readd = h.run(&["remember", text]);
    assert!(!readd.status.success());
    assert!(
        stderr_of(&readd).contains("forgotten"),
        "{}",
        stderr_of(&readd)
    );

    // What something was superseded by cannot be hard-deleted from under it.
    let old = first_token(&h.ok(&["remember", "alerts page the platform rota"]));
    let new = first_token(&h.ok(&["remember", "paging moved to the sre team in june"]));
    h.ok(&["supersede", &old, "--by", &new]);
    let refused = h.run(&["forget", &new, "--hard"]);
    assert!(!refused.status.success());
    let err = stderr_of(&refused);
    assert!(err.contains(&old), "must name what points at it: {err}");
    assert!(!err.contains("FOREIGN KEY"), "{err}");
    h.ok(&["get", &new]);
}

#[test]
fn forget_refuses_a_project() {
    let h = Harness::new();
    h.write(".mimir", "");
    h.ok(&["init", "--no-model"]);
    h.ok(&["remember", "a memory filed under the project"]);
    let uid: String = db_conn(&h)
        .query_row("SELECT uid FROM node WHERE kind = 'project'", [], |r| {
            r.get(0)
        })
        .unwrap();

    for args in [
        vec!["forget", uid.as_str()],
        vec!["forget", uid.as_str(), "--hard"],
    ] {
        let out = h.run(&args);
        assert!(!out.status.success(), "{args:?}");
        let err = stderr_of(&out);
        assert!(err.contains("is a project, not a memory"), "{err}");
    }
    let deleted: Option<i64> = db_conn(&h)
        .query_row("SELECT deleted_at FROM node WHERE uid = ?1", [&uid], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(deleted.is_none());
}

/// `mimir edit <ref> --pin` pins and does nothing else: no revision, no
/// `edit` row in the ledger.
#[test]
fn edit_with_only_a_pin_only_pins() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let id = first_token(&h.ok(&["remember", "always rebase before merging to main"]));

    let out = h.ok(&["edit", &id, "--pin"]);
    assert!(out.starts_with(&id), "prints the memory: {out}");
    h.ok(&["edit", &id, "--pin"]);

    let audit = h.ok(&["audit", &id, "--json"]);
    let ops: Vec<String> = audit
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["op"].to_string())
        .collect();
    assert_eq!(ops, vec!["\"pin\""], "one pin, no edit: {audit}");
    let history = h.ok(&["history", &id]);
    assert!(
        !history.contains("said:"),
        "no revision was taken: {history}"
    );

    // A real edit still records itself, and keeps the old wording.
    h.ok(&["edit", &id, "always rebase before merging to trunk"]);
    let history = h.ok(&["history", &id]);
    assert!(
        history.contains("said: always rebase before merging to main"),
        "{history}"
    );
}

/// Supersede is one operation with one set of guards, whichever way it is
/// asked for.
#[test]
fn supersede_guards_and_records_once() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let old_text = "deploys go out from the bastion host by hand";
    let old = first_token(&h.ok(&["remember", old_text]));
    let new = first_token(&h.ok(&["remember", "the pipeline ships every merge on its own"]));

    h.ok(&[
        "supersede",
        &old,
        "--by",
        &new,
        "--reason",
        "pipeline landed",
    ]);
    // Saying it twice changes nothing and records nothing.
    h.ok(&["supersede", &old, "--by", &new]);
    let audit = h.ok(&["audit", &old, "--json"]);
    assert_eq!(audit.lines().count(), 1, "one supersede row: {audit}");
    let shown = h.ok(&["get", &old]);
    assert!(shown.contains("supersedes"), "the edge is there: {shown}");

    // No loops: the replacement cannot be retired in favour of what it replaced.
    let looped = h.run(&["supersede", &new, "--by", &old]);
    assert!(!looped.status.success());
    assert!(
        stderr_of(&looped).contains("loop"),
        "{}",
        stderr_of(&looped)
    );
    assert!(h
        .ok(&["recall", "pipeline ships", "--kind", "memory"])
        .contains(&new));

    // A retired memory cannot be pinned, and the refusal names its replacement.
    let pinned = h.run(&["edit", &old, "--pin"]);
    assert!(!pinned.status.success());
    let err = stderr_of(&pinned);
    assert!(err.contains("superseded") && err.contains(&new), "{err}");

    // The retired text offered again is a new memory, not a duplicate of a
    // row recall will never show.
    let again = first_token(&h.ok(&["remember", old_text]));
    assert_ne!(again, old);
}

/// The PreToolUse reader only looks at memories, so an anchor anywhere else
/// would be accepted and never fire.
#[test]
fn anchor_refuses_a_node_that_is_not_a_memory() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write(
        "docs/guide.md",
        "# Setup\n\nInstall the widget frobnicator from the official site.\n",
    );
    h.ok(&["docs", "add", "docs", "--name", "guide-docs"]);
    h.ok(&["index"]);
    let hit = h.ok(&["recall", "frobnicator", "--kind", "doc", "--json"]);
    let doc: serde_json::Value = serde_json::from_str(hit.lines().next().unwrap()).unwrap();
    let uid = doc["uid"].as_str().expect("a doc hit carries its uid");

    let out = h.run(&["anchor", uid, "--pattern", "guide.md"]);
    assert!(!out.status.success());
    assert!(
        stderr_of(&out).contains("not a memory"),
        "{}",
        stderr_of(&out)
    );

    let id = first_token(&h.ok(&["remember", "the guide is generated; edit the template"]));
    let out = h.ok(&["anchor", &id, "--pattern", "guide.md"]);
    assert!(out.contains("anchored to guide.md"), "{out}");
}

/// Replacing the rules pack is an edit like any other: the previous rules
/// stay in its history and the change is in the ledger.
#[test]
fn rules_set_keeps_the_previous_rules_in_history() {
    let h = Harness::new();
    h.write(".mimir", "");
    h.ok(&["init", "--no-model"]);

    h.ok(&["rules", "set", "Always use tabs."]);
    let out = h.ok(&["rules", "set", "Always use spaces."]);
    assert!(out.contains("updated rules pack"), "{out}");
    assert!(h.ok(&["rules", "show"]).contains("Always use spaces."));

    let id = first_token(&h.ok(&["list", "--tag", "mimir-rules"]));
    let history = h.ok(&["history", &id]);
    assert!(history.contains("said: Always use tabs."), "{history}");
    let audit = h.ok(&["audit", &id]);
    assert!(audit.contains("edit"), "{audit}");
    assert!(audit.contains("pin"), "{audit}");
}

/// `mimir review` names the memory a finding is about, looked up by the row
/// id the queue stores.
#[test]
fn review_lists_a_finding_with_its_memory() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let text = "the ops cron runs at three in the morning utc";
    let id = first_token(&h.ok(&["remember", text]));
    h.ok(&["forget", &id]);
    // Offered again after being forgotten: refused, and queued as a finding.
    assert!(!h.run(&["remember", text]).status.success());

    let out = h.ok(&["review"]);
    assert!(out.contains("[resurfacing]"), "{out}");
    assert!(out.contains("the ops cron runs"), "titled: {out}");

    // Removing the tombstone removes the finding with it.
    h.ok(&["forget", &id, "--hard"]);
    assert!(h.ok(&["review"]).contains("nothing waiting on you"));
}

/// The MCP `forget` and `supersede` tools are the same operations as the
/// CLI's, with the same refusals — and their descriptions promise only what
/// the store does.
#[test]
fn mcp_forget_and_supersede_share_the_cli_operations() {
    use std::io::{BufRead, BufReader, Write};

    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let old = first_token(&h.ok(&["remember", "the bastion host is how deploys go out"]));
    let new = first_token(&h.ok(&["remember", "every merge is shipped by the pipeline"]));
    std::fs::write(
        h.home.path().join("config.toml"),
        "[hooks]\ninject_url = \"http://127.0.0.1:9/inject\"\n",
    )
    .unwrap();

    let mut child = mimir_bin()
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
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let _guard = HubGuard(child);
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    // One request at a time: the calls below depend on each other's effects.
    let mut next_id = 0;
    let mut request = |method: &str, params: serde_json::Value| -> String {
        next_id += 1;
        let message = serde_json::json!({
            "jsonrpc": "2.0", "id": next_id, "method": method, "params": params,
        });
        writeln!(stdin, "{message}").unwrap();
        if next_id == 1 {
            writeln!(
                stdin,
                r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
            )
            .unwrap();
        }
        stdin.flush().unwrap();
        let wanted = format!("\"id\":{next_id}");
        loop {
            let line = rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .unwrap_or_else(|_| panic!("no response to {method} within 30s"));
            if line.contains(&wanted) {
                return line;
            }
        }
    };
    let tool = |name: &str, arguments: serde_json::Value| serde_json::json!({ "name": name, "arguments": arguments });

    request(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "e2e", "version": "0" },
        }),
    );

    let supersede =
        |old: &str, by: &str| tool("supersede", serde_json::json!({ "old": old, "by": by }));
    let out = request("tools/call", supersede(&old, &new));
    assert!(out.contains("superseded by"), "{out}");
    let out = request("tools/call", supersede(&new, &old));
    assert!(out.contains("loop"), "a cycle must be refused: {out}");

    let forget = || tool("forget", serde_json::json!({ "ref": old }));
    let out = request("tools/call", forget());
    assert!(out.contains("forgot "), "{out}");
    let out = request("tools/call", forget());
    assert!(out.contains("already forgotten"), "{out}");

    let tools = request("tools/list", serde_json::json!({}));
    assert!(!tools.contains("reversible"), "{tools}");
    assert!(
        !tools.contains("via a `supersedes` edge"),
        "the column, not the edge, is what retires a memory: {tools}"
    );
    drop(stdin);

    // What the agent did is in the ledger under its own name.
    let audit = h.ok(&["audit", &old, "--json"]);
    let rows: Vec<serde_json::Value> = audit
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let ops: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r["op"].as_str().unwrap(), r["actor"].as_str().unwrap()))
        .collect();
    assert_eq!(ops, vec![("supersede", "agent"), ("forget", "agent")]);
}

// ---- sync: arrival order, supersession, archival, status ----

/// A port nothing is listening on, from the OS. Every test that starts a hub
/// or a daemon takes its port here: a window keyed on the pid sits inside
/// the range the OS hands out ephemeral ports from, and under load a test
/// would find its port taken ("hub did not become ready").
/// A port nothing is listening on, and that no other test in this run was
/// given. The listener is closed before the server under test binds the
/// port, and in that gap the kernel may hand the same number to the next
/// asker: two tests running in parallel then start two servers on one port
/// and one of them fails. Remembering what was handed out closes that gap
/// between the tests of this binary (another process can still take it).
fn free_port() -> u16 {
    static TAKEN: std::sync::Mutex<Vec<u16>> = std::sync::Mutex::new(Vec::new());
    let mut taken = TAKEN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    loop {
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        if !taken.contains(&port) {
            taken.push(port);
            return port;
        }
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// A hub and `n` clients configured against it.
fn hub_with_clients(n: usize) -> (HubGuard, tempfile::TempDir, Vec<tempfile::TempDir>) {
    let port = free_port();
    let endpoint = format!("http://127.0.0.1:{port}");
    let hub = tempfile::tempdir().unwrap();
    init_server_home(hub.path());
    let guard = spawn_hub(hub.path(), port, SYNC_TEST_TOKEN);
    wait_for_hub(port);
    let clients: Vec<tempfile::TempDir> = (0..n)
        .map(|_| {
            let home = tempfile::tempdir().unwrap();
            init_server_home(home.path());
            write_server_config(home.path(), &endpoint);
            home
        })
        .collect();
    (guard, hub, clients)
}

const SYNC_TEST_TOKEN: &str = "e2e-sync-token";

/// A memory written on one machine while it was offline reaches the hub late,
/// carrying the time it was written. A peer that pulled in the meantime has a
/// cursor past that time, and must still receive it: what a pull has not seen
/// is decided by when a row reached the hub, not by when it was written.
#[test]
fn server_sync_delivers_a_row_pushed_long_after_it_was_written() {
    let (_guard, _hub, clients) = hub_with_clients(2);
    let (a, b) = (clients[0].path(), clients[1].path());
    let t = SYNC_TEST_TOKEN;

    // B is in regular contact: its own note is on the hub and its pull cursor
    // has moved up to the present.
    sync_client_ok(
        b,
        t,
        &["remember", "b's note about couplings", "-t", "note", "-g"],
    );
    sync_client_ok(b, t, &["sync"]);

    // A wrote a note an hour ago and has not been online since.
    sync_client_ok(
        a,
        t,
        &["remember", "a's note about flywheels", "-t", "note", "-g"],
    );
    skew_updated_at(&a.join("mimir.db"), "flywheels", unix_now() - 3600);
    let push = sync_client_ok(a, t, &["sync", "push"]);
    assert!(push.contains("hub applied 1"), "{push}");

    sync_client_ok(b, t, &["sync", "pull"]);
    let listing = sync_client_ok(b, t, &["list", "-g", "--json"]);
    assert!(
        listing.contains("flywheels"),
        "B must receive a row that reached the hub after B's last pull, \
         however old its own timestamp is: {listing}"
    );
}

/// `list -g --json` as parsed lines.
fn listed(home: &Path) -> Vec<serde_json::Value> {
    sync_client_ok(home, SYNC_TEST_TOKEN, &["list", "-g", "--json"])
        .lines()
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect()
}

/// The listed memory whose body contains `needle`, if it is listed at all.
fn listed_one(home: &Path, needle: &str) -> Option<serde_json::Value> {
    listed(home)
        .into_iter()
        .find(|m| m["body"].as_str().is_some_and(|b| b.contains(needle)))
}

/// Supersession and a deliberate forget replicate through a hub; decay
/// archival does not. Before, the first never arrived (the wire carried the
/// `supersedes` edge without the column that hides a memory) and the last
/// did (an idle note on one machine vanished from the one using it).
#[test]
fn server_sync_carries_supersession_and_forget_but_not_archival() {
    let (_guard, _hub, clients) = hub_with_clients(2);
    let (a, b) = (clients[0].path(), clients[1].path());
    let t = SYNC_TEST_TOKEN;

    for text in [
        "deploys go out on fridays",
        "release trains leave on tuesdays only",
        "the idle note about gaskets",
        "the staging password hint",
    ] {
        sync_client_ok(a, t, &["remember", text, "-t", "note", "-g", "--force"]);
    }
    sync_client_ok(a, t, &["sync"]);
    sync_client_ok(b, t, &["sync"]);
    assert_eq!(listed(b).len(), 4, "B holds A's four memories");

    let id = |home: &Path, needle: &str| -> String {
        listed_one(home, needle).expect(needle)["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    // On A: one memory superseded, one forgotten, one archived by decay
    // (written the way `consolidate` writes it).
    sync_client_ok(
        a,
        t,
        &["supersede", &id(a, "fridays"), "--by", &id(a, "tuesdays")],
    );
    sync_client_ok(a, t, &["forget", &id(a, "password hint")]);
    {
        let conn = rusqlite::Connection::open(a.join("mimir.db")).unwrap();
        let changed = conn
            .execute(
                "UPDATE node SET deleted_at = ?1, meta = json_set(meta, '$.archived', 1)
                 WHERE body LIKE '%gaskets%'",
                [unix_now()],
            )
            .unwrap();
        assert_eq!(changed, 1);
    }
    assert!(listed_one(a, "gaskets").is_none(), "archived on A");

    sync_client_ok(a, t, &["sync"]);
    sync_client_ok(b, t, &["sync"]);

    assert_eq!(
        listed_one(b, "fridays").expect("kept as history")["lifecycle"],
        "superseded",
        "the retirement reached B"
    );
    assert_eq!(listed_one(b, "tuesdays").unwrap()["lifecycle"], "live");
    assert!(
        listed_one(b, "password hint").is_none(),
        "a deliberate forget still replicates"
    );
    assert_eq!(
        listed_one(b, "gaskets").expect("still on B")["lifecycle"],
        "live",
        "A's idleness is not B's"
    );

    // B's untouched copy says nothing new to A: it stays archived there.
    sync_client_ok(a, t, &["sync"]);
    assert!(listed_one(a, "gaskets").is_none(), "still archived on A");
}

/// `sync status` in server mode: the pending count is what the hub has not
/// been sent, and the token line is the resolution `sync token` uses.
#[test]
fn sync_status_counts_pending_from_what_was_pushed_and_names_the_token_source() {
    let (_guard, _hub, clients) = hub_with_clients(1);
    let c = clients[0].path();
    let t = SYNC_TEST_TOKEN;
    let status = |token: Option<&str>| -> String {
        let mut cmd = mimir_bin();
        cmd.args(["sync", "status"])
            .env("MIMIR_HOME", c)
            .env("HOME", c)
            .env_remove("MIMIR_SYNC_TOKEN");
        if let Some(token) = token {
            cmd.env("MIMIR_SYNC_TOKEN", token);
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "{out:?}");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    sync_client_ok(
        c,
        t,
        &["remember", "bearing preload is 12 Nm", "-t", "note", "-g"],
    );
    assert!(
        status(Some(t)).contains("pending   1 memory change(s) to push"),
        "{}",
        status(Some(t))
    );
    sync_client_ok(c, t, &["sync", "push"]);
    // The push cursor resumes a minute back, so counting from it said 1 here
    // forever.
    assert!(
        status(Some(t)).contains("pending   0 memory change(s) to push"),
        "nothing is pending right after a push: {}",
        status(Some(t))
    );

    assert!(status(Some(t)).contains("token     env (MIMIR_SYNC_TOKEN)"));
    let missing = status(None);
    assert!(missing.contains("token     missing"), "{missing}");
    let db = rusqlite::Connection::open(c.join("mimir.db")).unwrap();
    assert_eq!(
        mimir_core::replicate::get_str_meta(&db, "server_token").unwrap(),
        None,
        "looking at the token must not create one"
    );
    // A store that has served as a hub holds a persisted token, which
    // `sync token` prints — so `sync status` must not call it missing.
    mimir_core::replicate::set_str_meta(&db, "server_token", "persisted-by-serve").unwrap();
    let stored = status(None);
    assert!(stored.contains("token     stored"), "{stored}");
    assert!(
        !stored.contains("persisted-by-serve"),
        "status names the source, never the token: {stored}"
    );

    let overview = sync_client_ok(c, t, &["status"]);
    assert!(
        overview.contains("sync    server http://127.0.0.1:")
            && overview.contains("local watermarks push="),
        "{overview}"
    );
}

/// File mode has a directory and no cursors: the status surfaces say where
/// the snapshots go, and do not invent a pending count or watermarks that
/// nothing in this mode ever writes.
#[test]
fn status_in_file_mode_shows_the_directory_and_no_cursors() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let dir = h.home.path().join("shared");
    std::fs::write(
        h.home.path().join("config.toml"),
        format!("[sync]\nmode = \"file\"\ndir = \"{}\"\n", dir.display()),
    )
    .unwrap();
    h.ok(&["remember", "a fact worth sharing", "-t", "note", "-g"]);
    h.ok(&["sync"]);

    let status = h.ok(&["sync", "status"]);
    assert!(
        status.contains(&format!("dir       {}", dir.display())),
        "{status}"
    );
    assert!(!status.contains("pending"), "{status}");
    assert!(!status.contains("token"), "{status}");

    let overview = h.ok(&["status"]);
    assert!(
        overview.contains(&format!("sync    file {} (manual)", dir.display())),
        "{overview}"
    );
    assert!(!overview.contains("watermarks"), "{overview}");
}

/// A `[sync] mode` that is not a mode is the same error from every sync
/// subcommand, `status` included, and names the values that are. `mimir
/// status` reports it and carries on: it is how one finds out.
#[test]
fn an_unknown_sync_mode_is_one_error_from_every_sync_subcommand() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    std::fs::write(
        h.home.path().join("config.toml"),
        "[sync]\nmode = \"sever\"\nendpoint = \"http://127.0.0.1:9\"\n",
    )
    .unwrap();
    let expected =
        "unknown [sync] mode \"sever\" in config.toml — use \"off\", \"file\" or \"server\"";

    for args in [
        &["sync"][..],
        &["sync", "push"],
        &["sync", "pull"],
        &["sync", "status"],
    ] {
        let out = h.run(args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "mimir {args:?} must fail");
        assert!(stderr.contains(expected), "mimir {args:?}: {stderr}");
    }
    let overview = h.ok(&["status"]);
    assert!(overview.contains(expected), "{overview}");
}

/// A pull cursor the hub cannot read is a 400, not a request for everything.
#[test]
fn hub_answers_400_to_an_unparseable_pull_cursor() {
    let (_guard, _hub, clients) = hub_with_clients(1);
    let c = clients[0].path();
    sync_client_ok(
        c,
        SYNC_TEST_TOKEN,
        &["remember", "a memory on the hub", "-t", "note", "-g"],
    );
    sync_client_ok(c, SYNC_TEST_TOKEN, &["sync", "push"]);

    let endpoint = std::fs::read_to_string(c.join("config.toml")).unwrap();
    let endpoint = endpoint
        .split('"')
        .find(|part| part.starts_with("http://"))
        .unwrap()
        .to_string();
    // The status, and for a 200 how many nodes the batch carried.
    let pull = |query: &str| -> (u16, Option<usize>) {
        match ureq::get(&format!("{endpoint}/sync/pull{query}"))
            .set("Authorization", &format!("Bearer {SYNC_TEST_TOKEN}"))
            .call()
        {
            Ok(response) => {
                let batch: serde_json::Value = response.into_json().unwrap();
                (200, Some(batch["nodes"].as_array().unwrap().len()))
            }
            Err(ureq::Error::Status(code, _)) => (code, None),
            Err(e) => panic!("hub unreachable: {e}"),
        }
    };

    assert_eq!(pull("?since=yesterday"), (400, None));
    // A first pull, with or without an explicit zero, is still everything.
    assert_eq!(pull("?since=0"), (200, Some(1)));
    assert_eq!(pull(""), (200, Some(1)));
}

/// `recall --rerank` with no reranker on the machine used to print the fused
/// order exactly as if it had reranked. The hits still come back (stdout is
/// untouched, exit 0); stderr says the flag did nothing and how to fix it.
#[test]
fn recall_rerank_without_a_reranker_says_so_on_stderr() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&["remember", "-g", "Rotate the staging API token monthly"]);
    h.ok(&["remember", "-g", "The staging API gateway times out at 30s"]);

    let plain = h.run(&["recall", "-g", "staging API"]);
    assert!(plain.status.success());
    assert!(
        !String::from_utf8_lossy(&plain.stderr).contains("rerank"),
        "no --rerank, nothing to report: {}",
        String::from_utf8_lossy(&plain.stderr)
    );

    let asked = h.run(&["recall", "-g", "staging API", "--rerank"]);
    assert!(asked.status.success(), "a missing reranker is not an error");
    let stderr = String::from_utf8_lossy(&asked.stderr);
    assert!(
        stderr.contains("--rerank: no reranker available, so this is the fused order")
            && stderr.contains("`mimir embed --fetch --rerank`"),
        "stderr: {stderr}"
    );
    assert_eq!(
        asked.stdout, plain.stdout,
        "the hits are the fused order, unchanged"
    );
}

/// The miss ledger answers "what does the store not know?". A recall that
/// reaches into history (`--include-superseded`) and finds a strong match on
/// a retired memory has not shown that the store knows: the answer it found
/// is one recall hides. It used to close the question.
#[test]
fn a_strong_hit_on_a_retired_memory_does_not_resolve_a_miss() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let open_misses = |h: &Harness| h.ok(&["misses", "--all"]);

    // Asked before anything was written: a miss.
    h.ok(&["recall", "-g", "kafka retention policy"]);
    assert!(
        open_misses(&h).contains("kafka retention policy"),
        "the unanswered recall is in the ledger"
    );

    // An answer that has since been retired, by a memory worded differently
    // enough that it does not answer the same question.
    let old = h.ok(&["remember", "-g", "Kafka retention policy is seven days"]);
    let old = old.split_whitespace().next().unwrap().to_string();
    let new = h.ok(&[
        "remember",
        "-g",
        "Stream log lifetime is owned by the platform team",
    ]);
    let new = new.split_whitespace().next().unwrap().to_string();
    h.ok(&["supersede", &old, "--by", &new]);

    // History can find it, and labels it; the question stays open.
    let history = h.ok(&[
        "recall",
        "-g",
        "kafka retention policy",
        "--include-superseded",
    ]);
    assert!(history.contains("superseded"), "history recall: {history}");
    assert!(!history.contains(" weak]"), "a strong hit: {history}");
    assert!(
        open_misses(&h).contains("kafka retention policy"),
        "a retired memory must not close the question: {}",
        open_misses(&h)
    );

    // A live answer does close it.
    h.ok(&[
        "remember",
        "-g",
        "Kafka retention policy is fourteen days since the audit",
    ]);
    h.ok(&["recall", "-g", "kafka retention policy"]);
    assert!(
        !open_misses(&h).contains("kafka retention policy"),
        "a live strong hit resolves it: {}",
        open_misses(&h)
    );
}

/// One root registered for both docs and code is one name and one path for
/// two collections. Looking it up used to act on whichever row came first:
/// `mimir index <name>` indexed one of them, and `docs remove <name>`
/// removed one of them, with nothing to say which.
#[test]
fn a_root_registered_for_docs_and_code_is_never_resolved_by_row_order() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write(
        "proj/README.md",
        "# Widgets\n\nFrobnicate the widget twice.\n",
    );
    h.write(
        "proj/src/lib.rs",
        "pub fn frobnicate_widget() -> u32 {\n    42\n}\n",
    );
    // Code first, so it is the older row and the one "first match" finds.
    h.ok(&["code", "add", "proj", "--name", "proj"]);
    h.ok(&["docs", "add", "proj", "--name", "proj"]);

    let listed = h.ok(&["docs", "list"]);
    assert!(listed.contains("[code] proj"), "{listed}");
    assert!(listed.contains("[docs] proj"), "{listed}");

    // Indexing by name indexes both: two report lines, files in each.
    let out = h.ok(&["index", "proj"]);
    let lines: Vec<&str> = out.lines().filter(|l| l.starts_with("proj: ")).collect();
    assert_eq!(lines.len(), 2, "one line per collection: {out}");
    assert!(
        lines.iter().all(|l| l.contains("1 indexed")),
        "each found its own file: {out}"
    );
    let indexed = h.ok(&["docs", "list"]);
    for kind in ["[code] proj", "[docs] proj"] {
        let row = indexed.lines().find(|l| l.contains(kind)).unwrap();
        assert!(row.contains("(1 files"), "{kind} was indexed: {indexed}");
    }
    let doc = h.ok(&["recall", "frobnicate widget", "--kind", "doc"]);
    assert!(doc.contains("Widgets"), "{doc}");
    let code = h.ok(&["recall", "frobnicate_widget", "--kind", "code"]);
    assert!(code.contains("frobnicate_widget"), "{code}");

    let id_of = |listing: &str, kind: &str| {
        listing
            .lines()
            .find(|l| l.contains(kind))
            .and_then(|l| l.split_whitespace().next())
            .unwrap()
            .to_string()
    };
    let docs_id = id_of(&listed, "[docs] proj");
    let code_id = id_of(&listed, "[code] proj");

    // A docs command means the docs collection, though the code one is
    // older — and because two collections answered, it says which one it
    // acted on and that the other was left alone.
    let note = h.ok(&[
        "docs", "note", "proj", "covers", "the", "v2", "widget", "api",
    ]);
    let mut note_lines = note.lines();
    let first = note_lines.next().unwrap();
    assert!(
        first.contains(&format!("describes {docs_id} proj")),
        "the note is attached to the docs collection: {note}"
    );
    assert_eq!(
        note_lines.next(),
        Some(
            format!(
                "the note is on the docs collection; a code collection {code_id} also \
                 answers to 'proj' and was not annotated, pass its id as the target to \
                 annotate it"
            )
            .as_str()
        ),
        "{note}"
    );
    // The id it was told about does reach the code collection.
    let note = h.ok(&["docs", "note", &code_id, "the", "widget", "crate"]);
    assert!(
        note.contains(&format!("describes {code_id} proj")),
        "{note}"
    );
    assert_eq!(
        note.lines().count(),
        1,
        "one target, nothing to add: {note}"
    );

    let removed = h.ok(&["docs", "remove", "proj"]);
    assert_eq!(
        removed.trim_end(),
        format!(
            "removed docs collection proj ({docs_id}); a code collection {code_id} also \
             answers to 'proj' and is still registered, run the command again to remove it"
        )
    );
    let listed = h.ok(&["docs", "list"]);
    assert!(!listed.contains("[docs] proj"), "{listed}");
    assert!(
        listed.contains("[code] proj"),
        "the code collection is untouched: {listed}"
    );

    // With no docs collection left, the same command reaches the code one,
    // as it always has: `docs remove` is the only remove there is. One
    // match, so the plain line it has always printed.
    let removed = h.ok(&["docs", "remove", "proj"]);
    assert_eq!(removed, "removed proj\n");
    let listed = h.ok(&["docs", "list"]);
    assert!(listed.contains("no collections"), "{listed}");

    let gone = h.run(&["docs", "remove", "proj"]);
    assert!(!gone.status.success());
    assert!(
        String::from_utf8_lossy(&gone.stderr).contains("not found: collection 'proj'"),
        "{}",
        String::from_utf8_lossy(&gone.stderr)
    );
    let gone = h.run(&["index", "proj"]);
    assert!(!gone.status.success());
    assert!(String::from_utf8_lossy(&gone.stderr).contains("not found: collection 'proj'"));
}

/// In `recall --json` the key `strength` is the match label, which is what
/// scripts read there; the node's learned strength is a number under
/// `node_strength` rather than being overwritten out of the output.
#[test]
fn recall_json_keeps_the_match_label_and_adds_the_learned_strength() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&["remember", "-g", "Rotate the staging API token monthly"]);

    let out = h.ok(&["recall", "-g", "staging API token", "--json"]);
    let hit: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(hit["strength"], "strong", "{hit}");
    assert_eq!(hit["node_strength"], 1.0, "{hit}");

    // `list --json` and `get --json` never had the label: there the number
    // is still `strength`.
    let out = h.ok(&["list", "-g", "--json"]);
    let listed: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(listed["strength"], 1.0, "{listed}");
    assert!(listed.get("node_strength").is_none(), "{listed}");
}

/// Several collections can answer to one name without any of them being
/// code: two folders registered under the same name are enough. A `docs`
/// command acts on the oldest docs collection, and every time it leaves
/// others behind it names them, so no removal or note lands on "one of
/// them" unannounced.
#[test]
fn a_docs_command_names_every_other_collection_that_answered() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write("one/README.md", "# One\n\nfirst folder\n");
    h.write("two/README.md", "# Two\n\nsecond folder\n");
    h.write("two/src/lib.rs", "pub fn two() {}\n");
    h.ok(&["docs", "add", "one", "--name", "shared"]);
    h.ok(&["docs", "add", "two", "--name", "shared"]);
    h.ok(&["code", "add", "two", "--name", "shared"]);

    // Listed by title, so read the ids back by root instead.
    let listed = h.ok(&["docs", "list"]);
    let id_at = |kind: &str, root: &str| {
        listed
            .lines()
            .find(|l| l.contains(kind) && l.contains(&format!("/{root} (")))
            .and_then(|l| l.split_whitespace().next())
            .unwrap_or_else(|| panic!("no {kind} collection at {root}: {listed}"))
            .to_string()
    };
    let (docs_one, docs_two, code_two) = (
        id_at("[docs]", "one"),
        id_at("[docs]", "two"),
        id_at("[code]", "two"),
    );

    let removed = h.ok(&["docs", "remove", "shared"]);
    assert_eq!(
        removed.trim_end(),
        format!(
            "removed docs collection shared ({docs_one}); 2 other collections also answer \
             to 'shared' (docs collection {docs_two}, code collection {code_two}) and are \
             still registered, run the command again to remove the next one"
        )
    );

    let removed = h.ok(&["docs", "remove", "shared"]);
    assert_eq!(
        removed.trim_end(),
        format!(
            "removed docs collection shared ({docs_two}); a code collection {code_two} also \
             answers to 'shared' and is still registered, run the command again to remove it"
        )
    );
    assert_eq!(h.ok(&["docs", "remove", "shared"]), "removed shared\n");
    assert!(h.ok(&["docs", "list"]).contains("no collections"));
}

// ---------------------------------------------------------------------------
// Runtime surfaces: config modes, the hook entry points, hook installation,
// the daemon's address, `doctor` and `status`.
// ---------------------------------------------------------------------------

/// A URL nothing listens on. `doctor` and the stdio server probe the daemon
/// address; the tests below keep that probe off whatever really runs on the
/// default port of the machine they execute on.
const NO_DAEMON: &str = "http://127.0.0.1:9/inject";

fn out_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Run `cmd` with `input` on its stdin and collect what it wrote.
fn run_with_stdin(cmd: &mut Command, input: &str) -> Output {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawns");
    // A command that never reads its stdin (a guarded hook whose binary is
    // missing exits at once) can be gone before this write happens. That is
    // a closed pipe, not a failed test: what the command printed and how it
    // exited are still the answer.
    match child.stdin.take().unwrap().write_all(input.as_bytes()) {
        Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => {}
        other => other.expect("write the event to stdin"),
    }
    child.wait_with_output().unwrap()
}

impl Harness {
    /// The binary inside this harness's sandbox, for a caller that needs to
    /// add stdin, environment or a different working directory.
    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = mimir_bin();
        cmd.args(args)
            .env("MIMIR_HOME", self.home.path())
            .env("HOME", self.home.path())
            .env("USERPROFILE", self.home.path())
            .env("MIMIR_INJECT_URL", NO_DAEMON)
            .env_remove("MIMIR_HTTP_TOKEN")
            .current_dir(self.cwd.path());
        cmd
    }

    /// Run a hook entry point the way Claude Code does: event JSON on stdin.
    fn hook(&self, args: &[&str], event: &serde_json::Value) -> Output {
        run_with_stdin(&mut self.command(args), &event.to_string())
    }

    fn session_state_rows(&self) -> i64 {
        rusqlite::Connection::open(self.home.path().join("mimir.db"))
            .unwrap()
            .query_row("SELECT count(*) FROM session_state", [], |r| r.get(0))
            .unwrap()
    }

    /// A transcript file that reads as 50% of a 1000-token context window
    /// at the default 8 bytes per token.
    fn half_full_transcript(&self) -> String {
        let path = self.cwd.path().join("transcript.jsonl");
        std::fs::write(&path, vec![b'x'; 4000]).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn write_config(&self, toml: &str) {
        std::fs::write(self.home.path().join("config.toml"), toml).unwrap();
    }
}

/// Every hook-facing command, on a store that cannot be opened: exit 0,
/// nothing injected, the reason on stderr. A hook that exits non-zero shows
/// an error in every session; these used `?` on the open.
#[test]
fn hook_commands_fail_open_when_the_store_cannot_be_opened() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    // A config that does not parse stops `Mimir::open` for every command.
    h.write_config("[hooks\ncontext_guard = ");
    let event = serde_json::json!({
        "session_id": "s1",
        "source": "startup",
        "cwd": h.cwd.path(),
        "trigger": "auto",
        "tool_name": "Bash",
        "tool_input": { "command": "ls" },
    });
    for args in [
        &["brief", "show"][..],
        &["rules", "show", "--hook"],
        &["context-guard", "prompt"],
        &["context-guard", "precompact"],
        &["context-guard", "session-start"],
        &["context-guard", "pretool"],
        &["recall-inject", "--", "anything at all"],
    ] {
        let out = h.hook(args, &event);
        let err = out_text(&out.stderr);
        assert!(
            out.status.success(),
            "mimir {args:?} must exit 0 as a hook, stderr: {err}"
        );
        assert_eq!(
            out_text(&out.stdout),
            "",
            "mimir {args:?} injected something"
        );
        assert!(
            err.contains("hook failed open"),
            "mimir {args:?} must say why on stderr: {err}"
        );
    }

    // `rules show` typed by hand is not a hook: the error is an error.
    let by_hand = h
        .command(&["rules", "show"])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(
        !by_hand.status.success(),
        "a hand-typed command still fails"
    );
    assert!(!h.run(&["status"]).status.success());
}

/// A `SessionStart` event with no `source` is a start-up for the context
/// guard; it used to be "never fire" for the brief. One parser, one answer.
#[test]
fn a_session_start_without_a_source_is_a_startup_for_the_brief_too() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&[
        "remember",
        "never run the migration against the live store",
        "-t",
        "gotcha",
    ]);
    let no_source = serde_json::json!({ "session_id": "s1", "cwd": h.cwd.path() });
    let out = h.hook(&["brief", "show"], &no_source);
    assert!(out.status.success());
    let brief = out_text(&out.stdout);
    assert!(
        brief.contains("Mimir guards (do not violate):"),
        "{brief:?}"
    );
    assert!(brief.contains("never run the migration"), "{brief:?}");

    // An unknown value is still not a start-up.
    let unknown = serde_json::json!({
        "session_id": "s2", "source": "fork", "cwd": h.cwd.path()
    });
    assert_eq!(out_text(&h.hook(&["brief", "show"], &unknown).stdout), "");
}

/// `[hooks] context_guard` set to a value that names no mode is off — at the
/// gates, for the session state, and in what `doctor` says about it. It
/// used to pass every `== "off"` gate, write the nag row and print nothing.
#[test]
fn a_misspelled_context_guard_mode_is_off_and_doctor_says_so() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write_config("[hooks]\ncontext_guard = \"handof\"\ncontext_window_tokens = 1000\n");
    let event = serde_json::json!({
        "session_id": "s1",
        "source": "clear",
        "trigger": "auto",
        "cwd": h.cwd.path(),
        "transcript_path": h.half_full_transcript(),
    });
    for sub in ["prompt", "precompact", "session-start"] {
        let out = h.hook(&["context-guard", sub], &event);
        assert!(out.status.success());
        assert_eq!(out_text(&out.stdout), "", "context-guard {sub}");
    }
    assert_eq!(
        h.session_state_rows(),
        0,
        "a guard that is off writes no session state"
    );

    let line = "config: [hooks] context_guard = \"handof\" is not a known value \
                (accepted: off, pause, handoff); behaving as \"off\"";
    let doctor = h.command(&["doctor"]).output().unwrap();
    assert!(!doctor.status.success(), "an invalid mode fails doctor");
    let report = out_text(&doctor.stdout);
    assert!(report.contains(&format!("FAIL  {line}")), "{report}");

    // The unattended check stays quiet: the guard is off, which is a
    // documented state, and the store is sound. A timer that turns every
    // failure into a desktop alarm is not where a misspelling is reported.
    let check = h.command(&["doctor", "--check"]).output().unwrap();
    assert!(check.status.success(), "{}", out_text(&check.stderr));
    assert_eq!(out_text(&check.stdout), "");
}

/// One FAIL line per invalid mode, each naming section, key, value found,
/// the accepted values and what the setting does meanwhile.
#[test]
fn doctor_lists_every_invalid_config_mode() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let healthy = out_text(&h.command(&["doctor"]).output().unwrap().stdout);
    assert!(healthy.contains("ok   config: "), "{healthy}");

    h.write_config(
        "[sync]\nmode = \"sever\"\n[rerank]\nauto = \"alwyas\"\n\
         [daemon]\ninference = \"on\"\n[proxy]\ncache_ttl = \"10m\"\n",
    );
    let doctor = h.command(&["doctor"]).output().unwrap();
    assert!(!doctor.status.success());
    let report = out_text(&doctor.stdout);
    for line in [
        "FAIL  config: [sync] mode = \"sever\" is not a known value (accepted: off, file, \
         server); behaving as \"off\"",
        "FAIL  config: [rerank] auto = \"alwyas\" is not a known value (accepted: off, warm, \
         always); behaving as \"off\"",
        "FAIL  config: [daemon] inference = \"on\" is not a known value (accepted: auto, \
         off); behaving as \"off\"",
        "FAIL  config: [proxy] cache_ttl = \"10m\" is not a known value (accepted: 5m, 1h); \
         `mimir proxy` refuses to start",
    ] {
        assert!(report.contains(line), "missing {line:?} in:\n{report}");
    }
    assert!(
        report.contains("inference delegation: off)"),
        "doctor reports the mode in effect, not the typo: {report}"
    );
    assert!(out_text(&doctor.stderr).contains("4 check(s) failed"));

    // The unattended check (a timer that raises a desktop alarm on failure)
    // is about the store. A misspelled value that falls back to a documented
    // one is not an alarm; the one with no fallback still is.
    let watchdog = h.command(&["doctor", "--check"]).output().unwrap();
    let alarm = out_text(&watchdog.stderr);
    assert!(!watchdog.status.success());
    assert!(alarm.contains("[proxy] cache_ttl"), "{alarm}");
    assert!(
        !alarm.contains("[sync] mode") && !alarm.contains("[rerank] auto"),
        "a value with a fallback raised the alarm: {alarm}"
    );
    h.write_config("[rerank]\nauto = \"Warm\"\n[embedding]\ndevice = \"gpu\"\n");
    let quiet = h.command(&["doctor", "--check"]).output().unwrap();
    assert!(
        quiet.status.success(),
        "values that only fall back must not fail the watchdog: {}",
        out_text(&quiet.stderr)
    );
}

/// Pause mode: the nudge fires once per band, says how to get the handoff
/// back, and blocks an automatic compaction with the same text.
#[test]
fn the_pause_nudge_fires_once_and_says_how_to_get_the_handoff_back() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write_config("[hooks]\ncontext_guard = \"pause\"\ncontext_window_tokens = 1000\n");
    let event = serde_json::json!({
        "session_id": "s1",
        "trigger": "auto",
        "cwd": h.cwd.path(),
        "transcript_path": h.half_full_transcript(),
    });
    let first = out_text(&h.hook(&["context-guard", "prompt"], &event).stdout);
    assert!(first.contains("~50% full"), "{first}");
    assert!(first.contains("--tags session-handoff"), "{first}");
    assert!(
        first.contains("`mimir list --tag session-handoff -n 1`"),
        "{first}"
    );
    assert!(first.contains("context_guard = \"handoff\""), "{first}");
    assert_eq!(
        out_text(&h.hook(&["context-guard", "prompt"], &event).stdout),
        "",
        "already nudged at this band"
    );

    let block = out_text(&h.hook(&["context-guard", "precompact"], &event).stdout);
    let block: serde_json::Value = serde_json::from_str(block.trim()).expect("a JSON decision");
    assert_eq!(block["decision"], "block");
    assert!(block["reason"].as_str().unwrap().contains("~50% full"));

    // The recall the nudge recommends does find a memory saved its way.
    h.ok(&[
        "remember",
        "in flight: the importer handles CSV; next is the XLSX path",
        "-t",
        "note",
        "--tags",
        "session-handoff",
    ]);
    let found = h.ok(&["recall", "session-handoff"]);
    assert!(found.contains("in flight: the importer"), "{found}");
}

/// Handoff mode restores the handoff into the session after a clear — once.
/// The next clear, with no new handoff written, restores nothing, and the
/// memory itself is still there to read.
#[test]
fn a_handoff_is_restored_into_one_session_only() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write_config("[hooks]\ncontext_guard = \"handoff\"\n");
    let stored = h.ok(&[
        "remember",
        "handoff: schema v3 is applied on staging only; backfill prod next",
        "-t",
        "note",
        "--tags",
        "session-handoff",
    ]);
    let id = stored.split_whitespace().next().unwrap().to_string();
    let reset = |source: &str| {
        let event = serde_json::json!({
            "session_id": "s1", "source": source, "cwd": h.cwd.path()
        });
        out_text(&h.hook(&["context-guard", "session-start"], &event).stdout)
    };

    assert_eq!(reset("startup"), "", "a start-up restores nothing");
    let restored = reset("clear");
    assert!(
        restored.starts_with("Restored session handoff:\n"),
        "{restored}"
    );
    assert!(restored.contains("backfill prod next"), "{restored}");
    assert_eq!(reset("compact"), "", "restored once, not on every reset");
    assert_eq!(reset("clear"), "");

    let kept = h.ok(&["get", &id]);
    assert!(kept.contains("backfill prod next"), "{kept}");

    // A new handoff is a new snapshot and is restored in its turn. (The
    // first is backdated: "newest" is decided by `created_at`, which has
    // second resolution, and a real handoff is never written in the same
    // second as the one before it.)
    rusqlite::Connection::open(h.home.path().join("mimir.db"))
        .unwrap()
        .execute("UPDATE node SET created_at = created_at - 600", [])
        .unwrap();
    h.ok(&[
        "remember",
        "handoff: prod is backfilled; drop the legacy column next",
        "-t",
        "note",
        "--tags",
        "session-handoff",
    ]);
    assert!(reset("clear").contains("drop the legacy column"));
}

/// The cold auto-recall path with an empty `--session`: no session, so no
/// dedup ledger — the same reading the warm endpoint gives an empty id.
/// It used to dedup every id-less caller against one shared `""` session.
#[test]
fn recall_inject_treats_an_empty_session_id_as_none() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.ok(&[
        "remember",
        "terraform plan fails unless VAULT_ADDR is exported first",
        "-t",
        "gotcha",
    ]);
    let inject = |session: &str| {
        h.ok(&[
            "recall-inject",
            "--session",
            session,
            "--",
            "why does terraform plan fail on staging",
        ])
    };
    assert!(inject("").contains("VAULT_ADDR"));
    assert!(
        inject("").contains("VAULT_ADDR"),
        "an empty id must not be a session of its own"
    );
    // A real id still dedups.
    assert!(inject("s1").contains("VAULT_ADDR"));
    assert_eq!(inject("s1"), "");
}

/// `rules show` as a SessionStart hook scopes by the event's `cwd`, like
/// the brief and the context guard beside it — not by the directory the
/// hook process happens to run in.
#[test]
fn rules_show_as_a_hook_uses_the_events_cwd() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let repo = h.cwd.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let set = h
        .command(&["rules", "set", "Tabs, never spaces."])
        .current_dir(&repo)
        .output()
        .unwrap();
    assert!(set.status.success(), "{}", out_text(&set.stderr));

    // The hook process runs outside the repo; the event says where the
    // session is.
    let event = serde_json::json!({ "session_id": "s1", "source": "startup", "cwd": repo });
    let shown = out_text(&h.hook(&["rules", "show", "--hook"], &event).stdout);
    assert!(shown.contains("Tabs, never spaces."), "{shown:?}");

    // Typed by hand, without `--hook`: the working directory, and stdin is
    // not read at all, even with an event sitting on it. A command a person
    // types must not swallow the input of the loop or pipe it runs inside.
    let by_hand = h.hook(&["rules", "show"], &event);
    assert!(by_hand.status.success());
    assert_eq!(out_text(&by_hand.stdout), "", "the cwd is not a project");

    // As a hook with nothing on stdin: the working directory, exit 0.
    let no_event = h
        .command(&["rules", "show", "--hook"])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(no_event.status.success());
    assert_eq!(out_text(&no_event.stdout), "");
}

/// Outside a project a write is global and a read searches everything.
/// `status` used to call both "global scope".
#[test]
fn status_outside_a_project_says_what_reads_and_writes_do() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let status = h.ok(&["status"]);
    let first = status.lines().next().unwrap();
    assert!(first.contains("none — global scope"), "{first}");
    assert!(first.contains("new memories are global"), "{first}");
    assert!(first.contains("recall searches every project"), "{first}");

    let json: serde_json::Value = serde_json::from_str(h.ok(&["--json", "status"]).trim()).unwrap();
    assert_eq!(json["scope"], "global", "the capture scope keeps its name");
    assert_eq!(json["recall_scope"], "all");
}

/// "model: present" means the configured model's ready marker — the
/// predicate search itself uses — not "the models directory has something
/// in it".
#[test]
fn doctor_model_line_follows_the_configured_models_ready_marker() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let models = h.home.path().join("models");
    std::fs::create_dir_all(models.join("some-other-model")).unwrap();
    std::fs::write(models.join(".jina-reranker-v1-turbo-en.ready"), "").unwrap();
    let model_line = || {
        out_text(&h.command(&["doctor"]).output().unwrap().stdout)
            .lines()
            .find(|l| l.contains(" model: "))
            .unwrap()
            .to_string()
    };
    assert!(
        model_line().contains("bge-small-en-v1.5 not downloaded"),
        "a reranker and another model are not the embedder: {}",
        model_line()
    );
    std::fs::write(models.join(".bge-small-en-v1.5.ready"), "").unwrap();
    assert!(
        model_line().contains("bge-small-en-v1.5 present at"),
        "{}",
        model_line()
    );
}

/// Under `MIMIR_HOME`, `init --hooks` installs nothing — and says so,
/// instead of returning success with no output about hooks at all.
#[test]
fn init_hooks_under_mimir_home_says_it_skipped_and_why() {
    let h = Harness::new();
    let out = h.ok(&["init", "--no-model", "--hooks"]);
    assert!(out.contains("hooks   skipped — MIMIR_HOME is set"), "{out}");
    assert!(!h.home.path().join(".claude").exists());
}

/// `mimir daemon` takes no flags, so its refusal of a non-loopback address
/// must not recommend one. It names the command that can serve the address.
#[test]
fn daemon_refuses_a_non_loopback_address_with_a_command_that_exists() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let port = free_port();
    let out = h
        .command(&["daemon"])
        .env("MIMIR_INJECT_URL", format!("http://0.0.0.0:{port}/inject"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = out_text(&out.stderr);
    assert!(
        err.contains(&format!(
            "mimir mcp --http 0.0.0.0:{port} --http-allow-remote"
        )),
        "{err}"
    );
    assert!(err.contains("from MIMIR_INJECT_URL"), "{err}");
    assert!(
        !err.contains("or pass --http-allow-remote"),
        "that is `mimir mcp`'s refusal, about a flag `daemon` lacks: {err}"
    );
    assert_eq!(
        out_text(&out.stdout),
        "",
        "nothing was announced as serving"
    );
}

/// One address for the daemon, whoever asks: `MIMIR_INJECT_URL` moves the
/// daemon's bind and `doctor`'s probe together (it used to move only the
/// hook script). And the probe carries the bearer, so a token-protected
/// daemon reads as warm to the environment that has the token and says
/// "401" — not "not reachable" — to the one that does not.
#[test]
fn daemon_and_doctor_agree_on_the_address_and_the_bearer() {
    let home = tempfile::tempdir().unwrap();
    init_server_home(home.path());
    // The config points at a dead port: only the override can make this work.
    std::fs::write(
        home.path().join("config.toml"),
        format!("[hooks]\ninject_url = \"{NO_DAEMON}\"\n"),
    )
    .unwrap();
    let port = free_port();
    let url = format!("http://127.0.0.1:{port}/inject");
    let mimir = |args: &[&str]| {
        let mut cmd = mimir_bin();
        cmd.args(args)
            .env("MIMIR_HOME", home.path())
            .env("HOME", home.path())
            .env("MIMIR_INJECT_URL", &url)
            .env_remove("MIMIR_HTTP_TOKEN");
        cmd
    };
    let _daemon = HubGuard(
        mimir(&["daemon"])
            .env("MIMIR_HTTP_TOKEN", "s3cret")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_for_hub(port);

    let daemon_line = |token: Option<&str>| {
        let mut cmd = mimir(&["doctor"]);
        if let Some(token) = token {
            cmd.env("MIMIR_HTTP_TOKEN", token);
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "the daemon line never fails doctor");
        out_text(&out.stdout)
            .lines()
            .find(|l| l.contains(" daemon: "))
            .unwrap()
            .to_string()
    };
    let with_token = daemon_line(Some("s3cret"));
    assert!(
        with_token.contains(&format!("daemon: warm ({url} reachable")),
        "{with_token}"
    );
    let without = daemon_line(None);
    assert!(without.contains("daemon: cold ("), "{without}");
    assert!(without.contains("answers 401"), "{without}");
    assert!(
        without.contains("is not set in this environment"),
        "{without}"
    );
    let wrong = daemon_line(Some("nope"));
    assert!(
        wrong.contains("is not the one it was started with"),
        "{wrong}"
    );
}

// Windows: see the cfg note on hooks_install_bakes_custom_inject_url_….
#[cfg(not(windows))]
mod hook_install {
    use super::*;

    /// The fake-`$HOME` sandbox the hook-installation tests above use:
    /// `MIMIR_HOME` unset so `install_hooks` runs, everything else resolved
    /// under a temp `$HOME` with a `~/.claude` in it.
    struct FakeHome {
        home: tempfile::TempDir,
        cwd: tempfile::TempDir,
    }

    impl FakeHome {
        fn new() -> Self {
            let home = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(home.path().join(".claude")).unwrap();
            FakeHome {
                home,
                cwd: tempfile::tempdir().unwrap(),
            }
        }

        fn command(&self, args: &[&str]) -> Command {
            let mut cmd = mimir_bin();
            cmd.args(args)
                .env("HOME", self.home.path())
                .env("USERPROFILE", self.home.path())
                .env("MIMIR_INJECT_URL", NO_DAEMON)
                .env_remove("MIMIR_HOME")
                .env_remove("MIMIR_HTTP_TOKEN")
                .env_remove("XDG_CONFIG_HOME")
                .env_remove("XDG_DATA_HOME")
                .env_remove("XDG_CACHE_HOME")
                .current_dir(self.cwd.path());
            cmd
        }

        fn run(&self, args: &[&str]) -> Output {
            self.command(args).output().expect("binary runs")
        }

        fn ok(&self, args: &[&str]) -> String {
            let out = self.run(args);
            assert!(
                out.status.success(),
                "mimir {args:?} failed:\nstdout: {}\nstderr: {}",
                out_text(&out.stdout),
                out_text(&out.stderr)
            );
            out_text(&out.stdout)
        }

        fn claude(&self, rel: &str) -> std::path::PathBuf {
            self.home.path().join(".claude").join(rel)
        }

        /// Where `Paths::standard()` puts config.toml under this `$HOME`.
        fn write_config(&self, toml: &str) {
            let dir = if cfg!(target_os = "macos") {
                self.home.path().join("Library/Application Support/mimir")
            } else {
                self.home.path().join(".config/mimir")
            };
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.toml"), toml).unwrap();
        }

        fn settings(&self) -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(self.claude("settings.json")).unwrap())
                .unwrap()
        }
    }

    /// The command string of the one hook in a settings entry.
    fn command_of(entry: &serde_json::Value) -> &str {
        entry["hooks"][0]["command"].as_str().unwrap()
    }

    /// `--context-guard off` records the mode; it is not a request for
    /// hooks. It used to install the base hooks all the same.
    #[test]
    fn context_guard_off_on_its_own_installs_no_hooks() {
        let f = FakeHome::new();
        let out = f.ok(&["init", "--no-model", "--context-guard", "off"]);
        assert!(!out.contains("hooks   "), "no hook line at all: {out}");
        assert!(!f.claude("settings.json").exists());
        assert!(!f.claude("hooks").exists());

        // With --hooks the base hooks go in, and still no guard entries.
        f.ok(&["init", "--no-model", "--hooks", "--context-guard", "off"]);
        let hooks = &f.settings()["hooks"];
        assert_eq!(hooks["SessionStart"].as_array().unwrap().len(), 2);
        assert!(hooks.get("PreCompact").is_none());

        // And a mode that is not one is refused before anything is written.
        let bad = f.run(&["init", "--no-model", "--context-guard", "paws"]);
        assert!(!bad.status.success());
        assert!(
            out_text(&bad.stderr)
                .contains("--context-guard must be one of: off, pause, handoff (got \"paws\")"),
            "{}",
            out_text(&bad.stderr)
        );
    }

    /// A settings.json that cannot be parsed stops the install before any
    /// script is written, leaves the file alone, and makes `init --hooks`
    /// exit non-zero: the thing it was asked to do did not happen.
    #[test]
    fn an_unparseable_settings_file_fails_the_install_and_writes_nothing() {
        let f = FakeHome::new();
        let broken = "{ \"hooks\": { \"SessionStart\": [ ";
        std::fs::write(f.claude("settings.json"), broken).unwrap();

        let out = f.run(&["init", "--no-model", "--hooks"]);
        assert!(!out.status.success(), "a failed hook install is a failure");
        let err = out_text(&out.stderr);
        assert!(
            err.contains("hooks   install failed: ")
                && err.contains("settings.json is not valid JSON"),
            "{err}"
        );
        assert!(err.contains("the hooks were not installed"), "{err}");
        assert!(
            !f.claude("hooks").exists(),
            "no script may be written before settings.json is known to be mergeable"
        );
        assert_eq!(
            std::fs::read_to_string(f.claude("settings.json")).unwrap(),
            broken
        );
        assert!(!f.claude("settings.json.mimir-bak").exists());
        // Config and database were still created: only the hooks failed.
        assert!(out_text(&out.stdout).contains("claude mcp add"));
    }

    /// An empty settings.json is a file nobody has written anything to yet
    /// (a `touch`). It is installed into like a missing one, not refused as
    /// invalid JSON.
    #[test]
    fn an_empty_settings_file_is_treated_as_empty_settings() {
        let f = FakeHome::new();
        std::fs::write(f.claude("settings.json"), "  \n").unwrap();

        let out = f.run(&["init", "--no-model", "--hooks"]);

        assert!(out.status.success(), "{}", out_text(&out.stderr));
        let settings = f.settings();
        assert_eq!(
            settings["hooks"]["SessionStart"].as_array().unwrap().len(),
            2
        );
        assert_eq!(settings["hooks"]["PreToolUse"].as_array().unwrap().len(), 2);
    }

    /// The backup exists to undo a change. No file, or no change, means no
    /// backup and no line claiming one.
    #[test]
    fn the_settings_backup_is_taken_only_when_an_existing_file_changes() {
        let f = FakeHome::new();
        let backup = f.claude("settings.json.mimir-bak");

        let first = f.ok(&["init", "--no-model", "--hooks"]);
        assert!(!first.contains("backup at"), "nothing to back up: {first}");
        assert!(!backup.exists());

        let installed = std::fs::read_to_string(f.claude("settings.json")).unwrap();
        let second = f.ok(&["init", "--no-model", "--hooks"]);
        assert!(!second.contains("backup at"), "nothing changed: {second}");
        assert!(!backup.exists());
        assert_eq!(
            std::fs::read_to_string(f.claude("settings.json")).unwrap(),
            installed,
            "a re-run that changes nothing rewrites nothing"
        );

        let third = f.ok(&["init", "--no-model", "--hooks", "--auto-recall"]);
        assert!(third.contains("hooks   backup at "), "{third}");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), installed);
    }

    /// Entries an older version installed are brought up to date in place:
    /// the anchors matcher gains the tools the handler already reads, and
    /// the two bare SessionStart commands gain their guard. Nothing is
    /// added twice and nothing else in the file moves.
    #[test]
    fn a_reinstall_upgrades_older_entries_in_place() {
        let f = FakeHome::new();
        let hooks_dir = f.claude("hooks");
        let old = serde_json::json!({
            "model": "opus",
            "hooks": {
                "SessionStart": [
                    { "hooks": [{ "type": "command", "command": "mimir rules show" }] },
                    { "hooks": [{ "type": "command", "command": "mimir brief show" }] },
                ],
                "PreToolUse": [
                    { "matcher": "Bash", "hooks": [{
                        "type": "command",
                        "command": hooks_dir.join("mimir-rewrite.sh"),
                    }] },
                    { "matcher": "Bash|Edit|Write", "hooks": [{
                        "type": "command",
                        "command": hooks_dir.join("mimir-anchors.sh"),
                    }] },
                ],
            },
        });
        std::fs::write(f.claude("settings.json"), old.to_string()).unwrap();

        let out = f.ok(&["init", "--no-model", "--hooks"]);
        assert!(
            out.contains("SessionStart (project rules) upgraded"),
            "{out}"
        );
        assert!(out.contains("SessionStart (brief) upgraded"), "{out}");
        assert!(
            out.contains(
                "PreToolUse anchors upgraded — matcher is now \
                 Bash|Edit|Write|MultiEdit|NotebookEdit"
            ),
            "{out}"
        );

        let settings = f.settings();
        assert_eq!(settings["model"], "opus", "the rest of the file is kept");
        let session = settings["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(session.len(), 2, "upgraded in place, not added again");
        assert_eq!(
            command_of(&session[0]),
            "command -v mimir >/dev/null 2>&1 || exit 0; mimir rules show --hook"
        );
        assert_eq!(
            command_of(&session[1]),
            "command -v mimir >/dev/null 2>&1 || exit 0; mimir brief show"
        );
        let pre = settings["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 2);
        assert_eq!(pre[1]["matcher"], "Bash|Edit|Write|MultiEdit|NotebookEdit");

        // Up to date now: a further run changes nothing.
        let again = f.ok(&["init", "--no-model", "--hooks"]);
        assert!(!again.contains("upgraded"), "{again}");
        assert_eq!(f.settings(), settings);

        // A matcher somebody narrowed by hand is theirs to keep.
        let mut edited = settings.clone();
        edited["hooks"]["PreToolUse"][1]["matcher"] = "Edit".into();
        std::fs::write(f.claude("settings.json"), edited.to_string()).unwrap();
        f.ok(&["init", "--no-model", "--hooks"]);
        assert_eq!(f.settings()["hooks"]["PreToolUse"][1]["matcher"], "Edit");
    }

    /// The two SessionStart commands have no wrapper script, so the guard
    /// is in the command: on a machine where `mimir` is not on PATH the
    /// hook exits 0 and says nothing, instead of failing every session.
    #[test]
    fn session_start_commands_are_silent_when_mimir_is_not_on_path() {
        let f = FakeHome::new();
        f.ok(&["init", "--no-model", "--hooks"]);
        let settings = f.settings();
        let event = serde_json::json!({
            "session_id": "s1", "source": "startup", "cwd": f.cwd.path()
        })
        .to_string();
        let bin_dir = Path::new(env!("CARGO_BIN_EXE_mimir")).parent().unwrap();
        for entry in settings["hooks"]["SessionStart"].as_array().unwrap() {
            let command = command_of(entry);

            let missing = run_with_stdin(
                Command::new("/bin/sh")
                    .args(["-c", command])
                    .env("PATH", "/nonexistent-dir-for-this-test"),
                &event,
            );
            assert!(missing.status.success(), "{command}");
            assert_eq!(out_text(&missing.stdout), "");
            assert_eq!(out_text(&missing.stderr), "", "{command}");

            // With the binary on PATH the same command runs it.
            let mut sh = Command::new("/bin/sh");
            sh.args(["-c", command])
                .env("PATH", bin_dir)
                .env("HOME", f.home.path())
                .env_remove("MIMIR_HOME")
                .env_remove("XDG_CONFIG_HOME")
                .env_remove("XDG_DATA_HOME")
                .env_remove("XDG_CACHE_HOME")
                .current_dir(f.cwd.path());
            let ran = run_with_stdin(&mut sh, &event);
            assert!(ran.status.success(), "{command}: {}", out_text(&ran.stderr));
        }
    }

    /// The line about the brief entry says what the entry does. It used to
    /// say "inert until `[brief] enabled = true`" on an install where the
    /// brief was already enabled, which is the default.
    #[test]
    fn the_brief_line_reflects_whether_the_brief_is_enabled() {
        let f = FakeHome::new();
        let on = f.ok(&["init", "--no-model", "--hooks"]);
        assert!(
            on.contains("SessionStart (session brief) added (on: `[brief] enabled = true`)"),
            "{on}"
        );
        assert!(!on.contains("inert"), "{on}");

        let g = FakeHome::new();
        g.write_config("[brief]\nenabled = false\n");
        let off = g.ok(&["init", "--no-model", "--hooks"]);
        assert!(
            off.contains(
                "SessionStart (session brief) added (inert while `[brief] enabled = false`)"
            ),
            "{off}"
        );
    }

    /// An installed hook script is a copy made at `init` time. `doctor`
    /// says when a copy is older than the binary, and the remedy it names —
    /// a plain `mimir init --hooks` — is enough to refresh it.
    #[test]
    fn doctor_reports_outdated_hook_scripts_and_init_hooks_refreshes_them() {
        let f = FakeHome::new();
        f.ok(&["init", "--no-model", "--hooks", "--auto-recall"]);
        let hooks_line = || {
            f.ok(&["doctor"])
                .lines()
                .find(|l| l.contains(" hooks: "))
                .map(str::to_string)
        };
        assert_eq!(
            hooks_line().as_deref(),
            Some("ok   hooks: 3 installed script(s) match this binary")
        );

        // What an install made by an earlier version left behind.
        let recall = f.claude("hooks/mimir-recall.sh");
        let current = std::fs::read_to_string(&recall).unwrap();
        assert!(current.contains("# mimir-hook-version: 5\n"), "{current}");
        std::fs::write(
            &recall,
            current.replace("# mimir-hook-version: 5\n", "# mimir-hook-version: 4\n"),
        )
        .unwrap();
        let stale = hooks_line().unwrap();
        assert!(
            stale.contains("1 of 3 installed script(s) older than this binary"),
            "{stale}"
        );
        assert!(stale.contains("mimir-recall.sh v4, shipped v5"), "{stale}");
        assert!(stale.contains("re-run `mimir init --hooks`"), "{stale}");

        // Exactly the command doctor named — no --auto-recall.
        f.ok(&["init", "--no-model", "--hooks"]);
        assert_eq!(std::fs::read_to_string(&recall).unwrap(), current);
        assert!(hooks_line().unwrap().contains("match this binary"));
    }

    fn have(tool: &str) -> bool {
        Command::new("/bin/sh")
            .args(["-c", &format!("command -v {tool} >/dev/null 2>&1")])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Undo `jq`'s `@uri` encoding.
    fn percent_decode(text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 3 <= bytes.len() {
                if let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
                    out.push(byte);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Answer one HTTP request on `listener` with `body` and hand back the
    /// request head (request line and headers) as it arrived.
    fn serve_once(
        listener: std::net::TcpListener,
        body: &'static str,
    ) -> std::sync::mpsc::Receiver<String> {
        use std::io::{Read, Write};
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Ok((mut sock, _)) = listener.accept() else {
                return;
            };
            let mut head = Vec::new();
            let mut buf = [0u8; 2048];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match sock.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                }
            }
            let _ = write!(
                sock,
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tx.send(String::from_utf8_lossy(&head).into_owned());
        });
        rx
    }

    /// The installed auto-recall script, run for real. Warm: it tells the
    /// daemon which directory the prompt came from, and presents the bearer
    /// when the environment has one. Cold: it runs the fallback from that
    /// same directory, so both paths scope to the event's project rather
    /// than to wherever the hook process was started.
    #[test]
    fn the_auto_recall_script_scopes_both_paths_by_the_events_cwd() {
        if !(have("bash") && have("jq") && have("curl")) {
            eprintln!("skipped: the hook script needs bash, jq and curl");
            return;
        }
        let f = FakeHome::new();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        f.write_config(&format!(
            "[hooks]\ninject_url = \"http://127.0.0.1:{port}/inject\"\n"
        ));
        f.ok(&["init", "--no-model", "--hooks", "--auto-recall"]);

        // Two projects; the memory belongs to one of them.
        let project = |name: &str| {
            let root = f.cwd.path().join(name);
            std::fs::create_dir_all(root.join(".git")).unwrap();
            root
        };
        let (infra, website) = (project("infra"), project("website"));
        let stored = f
            .command(&[
                "remember",
                "terraform plan fails unless VAULT_ADDR is exported first",
                "-t",
                "gotcha",
            ])
            .current_dir(&infra)
            .output()
            .unwrap();
        assert!(stored.status.success(), "{}", out_text(&stored.stderr));

        let script = f.claude("hooks/mimir-recall.sh");
        let path = format!(
            "{}:{}",
            Path::new(env!("CARGO_BIN_EXE_mimir"))
                .parent()
                .unwrap()
                .display(),
            std::env::var("PATH").unwrap_or_default()
        );
        // The hook process always starts in `website`; only the event says
        // where the session is.
        let runs = std::cell::Cell::new(0u32);
        let run_hook = |event_cwd: &Path, token: Option<&str>| {
            let mut cmd = Command::new("bash");
            cmd.arg(&script)
                .env("PATH", &path)
                .env("HOME", f.home.path())
                .env_remove("MIMIR_HOME")
                .env_remove("MIMIR_INJECT_URL")
                .env_remove("MIMIR_HTTP_TOKEN")
                .env_remove("XDG_CONFIG_HOME")
                .env_remove("XDG_DATA_HOME")
                .env_remove("XDG_CACHE_HOME")
                .current_dir(&website);
            if let Some(token) = token {
                cmd.env("MIMIR_HTTP_TOKEN", token);
            }
            // A session of its own for every run: within one session the
            // injection ledger shows a memory once, and a second run would
            // come back empty for that reason and not because of its scope.
            let session = format!("s{}", runs.get());
            runs.set(runs.get() + 1);
            let event = serde_json::json!({
                "prompt": "why does terraform plan fail on staging",
                "session_id": session,
                "cwd": event_cwd,
            });
            let out = run_with_stdin(&mut cmd, &event.to_string());
            assert!(out.status.success(), "the hook script always exits 0");
            out_text(&out.stdout)
        };

        // Warm, with a token in the environment.
        let request = serve_once(
            listener.try_clone().unwrap(),
            "Relevant memory: from the daemon",
        );
        assert_eq!(
            run_hook(&infra, Some("s3cret")),
            "Relevant memory: from the daemon\n"
        );
        let head = request
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the script called the warm endpoint");
        let request_line = head.lines().next().unwrap();
        let cwd_param = request_line
            .split(['?', '&', ' '])
            .find_map(|part| part.strip_prefix("cwd="))
            .unwrap_or_else(|| panic!("no cwd in {request_line}"));
        assert_eq!(percent_decode(cwd_param), infra.to_string_lossy());
        assert!(
            head.lines()
                .any(|l| l.eq_ignore_ascii_case("authorization: bearer s3cret")),
            "the bearer must be sent when MIMIR_HTTP_TOKEN is set:\n{head}"
        );

        // Warm, without one: no Authorization header is invented.
        let request = serve_once(listener.try_clone().unwrap(), "");
        assert_eq!(run_hook(&infra, None), "", "an empty answer prints nothing");
        let head = request
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert!(
            !head.to_ascii_lowercase().contains("authorization:"),
            "{head}"
        );

        // Cold: nothing listens any more, so the script falls back to
        // `mimir recall-inject` — from the event's directory.
        drop(listener);
        let from_infra = run_hook(&infra, None);
        assert!(
            from_infra.contains("VAULT_ADDR"),
            "the cold path must scope to the event's project: {from_infra:?}"
        );
        assert_eq!(
            run_hook(&website, None),
            "",
            "and a prompt typed in the other project must not get that memory"
        );
    }
}

/// The code graph through the real binary: a query on a graph that is
/// behind the working tree still answers, and says so on stderr (stdout
/// stays the answer); nothing but `graph build` brings it up to date; and
/// `--full` re-extracts every file whatever the stored hashes say.
#[test]
fn graph_queries_note_a_stale_graph_and_build_full_re_extracts() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write(".mimir", "");
    h.write("src/util.rs", "pub fn slugify() {}\n");
    h.write(
        "src/main.rs",
        "fn main() { run(); }\n\nfn run() { slugify(); }\n",
    );
    const NOTE: &str =
        "note: the code graph is 1 file(s) behind the working tree — run `mimir graph build`";
    let stderr = |out: &Output| String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = |out: &Output| String::from_utf8_lossy(&out.stdout).into_owned();

    let built = h.ok(&["graph", "build"]);
    assert!(
        built.contains("2 indexed") && built.contains("2 call sites resolved"),
        "{built}"
    );

    // Current: the answer, and nothing about freshness.
    let out = h.run(&["graph", "callers", "slugify"]);
    assert!(out.status.success());
    let lines = stdout(&out);
    assert!(
        lines.contains(" run\n") && lines.contains(" main\n"),
        "{lines}"
    );
    assert!(!stderr(&out).contains("note:"), "{}", stderr(&out));
    // The empty answers, in the words the MCP tool uses too.
    let calls = h.ok(&["graph", "calls", "slugify"]);
    assert!(calls.ends_with("slugify\nno callees found\n"), "{calls}");
    assert_eq!(
        h.ok(&["graph", "path", "slugify", "main"]),
        "no call path slugify → main\n"
    );

    // Edited since the build.
    h.write("src/util.rs", "// header\n\npub fn slugify() {}\n");
    let out = h.run(&["graph", "callers", "slugify"]);
    assert!(out.status.success(), "a stale graph still answers");
    assert!(stderr(&out).contains(NOTE), "{}", stderr(&out));
    assert!(!stdout(&out).contains("note:"), "stdout is the answer only");
    let out = h.run(&["graph", "node", "slugify"]);
    assert!(stderr(&out).contains(NOTE), "{}", stderr(&out));
    let out = h.run(&["graph", "impact", "src/util.rs"]);
    assert!(stderr(&out).contains(NOTE), "{}", stderr(&out));
    let out = h.run(&["peek", "slugify"]);
    assert!(
        stderr(&out).contains("note: src/util.rs changed since the code graph was built"),
        "{}",
        stderr(&out)
    );
    // Asking did not fix it: `check` still fails, as it did before.
    assert!(!h.run(&["graph", "check"]).status.success());

    // `--full` ignores the stored hashes; an ordinary build right after has
    // nothing left to do.
    let full = h.ok(&["graph", "build", "--full"]);
    assert!(full.contains("2 indexed, 0 unchanged"), "{full}");
    let again = h.ok(&["graph", "build"]);
    assert!(again.contains("0 indexed, 2 unchanged"), "{again}");
    h.ok(&["graph", "check"]);
    let out = h.run(&["graph", "callers", "slugify"]);
    assert!(!stderr(&out).contains("note:"), "{}", stderr(&out));
}

/// A project with nothing to index has a graph once it has been built. It
/// used to count as never built, so every query announced a first run and
/// built it again.
#[test]
fn graph_first_run_build_happens_once_in_a_project_without_code() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write(".mimir", "");
    h.write("notes.txt", "nothing to parse here\n");
    const FIRST_RUN: &str = "building the code graph (first run for this project)";

    let first = h.run(&["graph", "hubs"]);
    assert!(first.status.success());
    assert!(
        String::from_utf8_lossy(&first.stderr).contains(FIRST_RUN),
        "the first query builds"
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("no call edges yet"));

    let second = h.run(&["graph", "hubs"]);
    assert!(second.status.success());
    assert!(
        !String::from_utf8_lossy(&second.stderr).contains(FIRST_RUN),
        "the second does not: {}",
        String::from_utf8_lossy(&second.stderr)
    );
}

/// A `.mimir` that does not parse is a file the user has to look at, not one
/// to replace: the id in it may be the one every other clone and every synced
/// memory of the project goes by.
#[test]
fn project_init_refuses_to_overwrite_a_marker_it_cannot_parse() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    // A merge left its conflict markers in the file.
    let broken = "<<<<<<< HEAD\nid = \"PROJ123\"\n=======\nid = \"PROJ456\"\n>>>>>>> theirs\n";
    h.write(".mimir", broken);

    let out = h.run(&["project", "init", "--sync"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "init must refuse: {stdout}\n{stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(h.cwd.path().join(".mimir")).unwrap(),
        broken,
        "the file is left exactly as it was"
    );
    assert!(
        !stdout.contains("(created)"),
        "nothing was created: {stdout}"
    );
    assert!(
        stderr.contains(".mimir") && stderr.contains("line 1"),
        "names the file and where it stops parsing: {stderr}"
    );
    assert!(
        stderr.contains("move it aside"),
        "says what to do next: {stderr}"
    );
}

/// The store is opened before the marker is written, so a store that cannot
/// be opened does not leave a project marked whose identity was never
/// recorded.
#[test]
fn project_init_leaves_no_marker_when_the_store_cannot_be_opened() {
    let h = Harness::new();
    std::fs::write(h.home.path().join("mimir.db"), "not a sqlite database").unwrap();

    let out = h.run(&["project", "init"]);
    assert!(!out.status.success(), "the store is broken");
    assert!(
        !h.cwd.path().join(".mimir").exists(),
        "no marker may be left behind by a failed init"
    );
}

#[test]
fn project_init_no_sync_turns_sync_off_again() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let marker = h.cwd.path().join(".mimir");
    let id_line = |text: &str| {
        text.lines()
            .find(|l| l.starts_with("id = "))
            .map(str::to_owned)
    };

    let out = h.ok(&["project", "init", "--sync"]);
    assert!(out.contains("(created)"), "{out}");
    assert!(out.contains("sync     true"), "{out}");
    let on = std::fs::read_to_string(&marker).unwrap();
    assert!(on.contains("sync = true"), "{on}");

    // A plain re-run keeps what is there …
    let out = h.ok(&["project", "init"]);
    assert!(!out.contains("(created)"), "{out}");
    assert!(out.contains("sync     true"), "{out}");

    // … and --no-sync is the way back.
    let out = h.ok(&["project", "init", "--no-sync"]);
    assert!(out.contains("sync     false"), "{out}");
    let off = std::fs::read_to_string(&marker).unwrap();
    assert!(off.contains("sync = false"), "{off}");
    assert_eq!(id_line(&on), id_line(&off), "the id never changes");
    assert!(id_line(&off).is_some());

    let both = h.run(&["project", "init", "--sync", "--no-sync"]);
    assert_eq!(both.status.code(), Some(2), "the two flags contradict");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), off);
}

/// A child killed by a signal is reported the way a shell reports it, not as
/// an ordinary failure.
#[cfg(unix)]
#[test]
fn run_reports_a_signal_killed_child_as_128_plus_the_signal() {
    let h = Harness::new();
    let out = h.run(&["run", "--", "sh", "-c", "kill -TERM $$"]);
    assert_eq!(out.status.code(), Some(128 + 15));

    let out = h.run(&["run", "--", "sh", "-c", "exit 7"]);
    assert_eq!(out.status.code(), Some(7), "an exit code passes through");
}

/// The hook and `mimir run` read one classification: what the hook wraps as
/// a content command is what `run` then only caps.
#[cfg(unix)]
#[test]
fn rewrite_and_run_agree_on_what_a_command_is() {
    let h = Harness::new();
    let lines: String = (0..1000)
        .map(|i| format!("Downloading part {i}\n"))
        .collect();
    h.write("big.txt", &lines);

    let out = h.ok(&["rewrite", "cat big.txt"]);
    assert_eq!(out.trim(), "mimir run -- cat big.txt");
    let out = h.ok(&["run", "--", "cat", "big.txt"]);
    assert!(
        out.contains("Downloading part 0\n"),
        "content is never line-filtered"
    );
    assert!(out.contains("lines elided by mimir"), "only capped");

    // Left alone: a follow that never exits, a quiet git subcommand, a
    // program nothing is known about.
    for cmd in ["tail -f big.txt", "git log", "journalctl -f", "echo hi"] {
        let out = h.run(&["rewrite", cmd]);
        assert_eq!(out.status.code(), Some(1), "{cmd}");
        assert!(out.stdout.is_empty(), "{cmd}");
    }
}

/// The secret guard covers every place free text is stored, not only the
/// body and the tags of a new memory. An anchor pattern (given with the
/// memory or added later), a falsifying condition, a title, the rules pack
/// and a docs note all end up in the
/// store, in recall, and on other machines over sync.
#[test]
fn the_secret_guard_covers_every_surface_that_stores_text() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write(".mimir", "");
    h.write("docs/guide.md", "# Guide\n\nInstall the widget.\n");
    h.ok(&["docs", "add", "docs", "--name", "guide-docs"]);
    h.ok(&["index"]);
    let secret = format!("AKIA{}", "IOSFODNN7EXAMPLE");
    let kept = h.ok(&["remember", "-g", "the uploader retries three times"]);
    let id = kept.split_whitespace().next().unwrap().to_string();

    let condition = format!("when {secret} is revoked");
    let attempts: Vec<Vec<&str>> = vec![
        vec![
            "remember",
            "-g",
            "rotate the uploader key",
            "--anchor",
            &secret,
        ],
        vec![
            "remember",
            "-g",
            "rotate the uploader key",
            "--resolves-when",
            &condition,
        ],
        vec!["edit", &id, "--title", &secret],
        vec!["anchor", &id, "--pattern", &secret],
        vec!["rules", "set", &secret],
        vec!["docs", "note", "guide-docs", &secret],
    ];
    for args in &attempts {
        let out = h.run(args);
        assert!(!out.status.success(), "{args:?} must be refused");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("AWS access key"), "{args:?}: {err}");
    }

    // Nothing was written by any of them.
    let export = h.ok(&["export"]);
    assert!(!export.contains(&secret), "a refused value was stored");
    let refusals = h.ok(&["refusals"]);
    assert!(!refusals.contains(&secret), "the ledger must not hold it");
}

/// `sync pull --full` collects what an ordinary pull can no longer see. A
/// client's cursor only moves forward, so a row that sits below it (one an
/// older hub received late and never relayed) stays missing until something
/// asks from the beginning. The full pull is that, and changes nothing else.
#[test]
fn sync_pull_full_collects_what_sits_below_the_cursor() {
    let (_guard, _hub, clients) = hub_with_clients(2);
    let (a, b) = (clients[0].path(), clients[1].path());
    let t = SYNC_TEST_TOKEN;

    sync_client_ok(
        a,
        t,
        &["remember", "a's note about camshafts", "-t", "note", "-g"],
    );
    sync_client_ok(a, t, &["sync", "push"]);

    // B's cursor is ahead of everything on the hub, as it would be after an
    // older hub let this row slip below it.
    {
        let conn = rusqlite::Connection::open(b.join("mimir.db")).unwrap();
        mimir_core::replicate::set_watermark(&conn, "last_pull", unix_now() + 3600).unwrap();
    }
    sync_client_ok(b, t, &["sync", "pull"]);
    assert!(
        listed_one(b, "camshafts").is_none(),
        "an ordinary pull asks only for what is above the cursor"
    );

    let out = sync_client_ok(b, t, &["sync", "pull", "--full"]);
    assert!(out.contains("pulled everything → 1 applied"), "{out}");
    assert!(listed_one(b, "camshafts").is_some());

    // Repeating it applies nothing: the merge is last-write-wins.
    let again = sync_client_ok(b, t, &["sync", "pull", "--full"]);
    assert!(again.contains("pulled everything → 0 applied"), "{again}");
}

/// A project whose directory is deleted used to keep its whole code graph
/// for good: nothing reported it and nothing removed it. `project prune`
/// lists such projects and changes nothing until it is told to; `doctor`
/// points at it.
#[test]
fn project_prune_lists_then_removes_the_graph_of_a_directory_that_is_gone() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let scratch = tempfile::tempdir().unwrap();
    std::fs::write(scratch.path().join(".mimir"), "").unwrap();
    std::fs::create_dir_all(scratch.path().join("src")).unwrap();
    std::fs::write(
        scratch.path().join("src/lib.rs"),
        "pub fn slugify() {}\npub fn run() { slugify(); }\n",
    )
    .unwrap();
    let built = mimir_bin()
        .args(["graph", "build"])
        .env("MIMIR_HOME", h.home.path())
        .env("HOME", h.home.path())
        .current_dir(scratch.path())
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let present = h.ok(&["project", "prune"]);
    assert!(
        present.contains("no project has a code graph left behind"),
        "a project that is still on disk is not listed: {present}"
    );

    drop(scratch);
    let listed = h.ok(&["project", "prune"]);
    assert!(
        listed.contains("1 project(s) point at a directory that no longer exists")
            && listed.contains("2 symbols and 1 code files")
            && listed.contains("nothing was changed"),
        "{listed}"
    );
    let doctor = String::from_utf8_lossy(&h.run(&["doctor"]).stdout).into_owned();
    assert!(
        doctor.contains("projects: 1 project(s) point at a directory that no longer exists"),
        "{doctor}"
    );
    // Listing twice is still only listing.
    assert!(h.ok(&["project", "prune"]).contains("nothing was changed"));

    let applied = h.ok(&["project", "prune", "--apply"]);
    assert!(
        applied.contains(
            "removed 2 symbols and 1 code files from 1 project(s); 1 project row(s) removed"
        ),
        "{applied}"
    );
    let after = h.ok(&["project", "prune"]);
    assert!(
        after.contains("no project has a code graph left behind"),
        "{after}"
    );
}

/// What a prune keeps (a symbol a memory is linked to, and its file) stays in
/// the store for good. Once pruned, the project is not reported again: not
/// by `project prune`, not by `doctor`. A listing that went on naming it
/// would be advice already taken.
#[test]
fn project_prune_does_not_report_a_project_it_has_already_pruned() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let scratch = tempfile::tempdir().unwrap();
    std::fs::write(scratch.path().join(".mimir"), "").unwrap();
    std::fs::create_dir_all(scratch.path().join("src")).unwrap();
    std::fs::write(
        scratch.path().join("src/lib.rs"),
        "pub fn slugify() {}\npub fn run() { slugify(); }\n",
    )
    .unwrap();
    let in_scratch = |args: &[&str]| {
        let out = mimir_bin()
            .args(args)
            .env("MIMIR_HOME", h.home.path())
            .env("HOME", h.home.path())
            .current_dir(scratch.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    in_scratch(&["graph", "build"]);
    let stored = in_scratch(&[
        "remember",
        "-g",
        "slugify drops non-ASCII letters instead of transliterating them",
        "--link",
        "slugify",
    ]);
    let id = stored.split_whitespace().next().unwrap().to_string();
    drop(scratch);

    let applied = h.ok(&["project", "prune", "--apply"]);
    assert!(
        applied.contains("removed 1 symbols and 0 code files from 1 project(s)")
            && applied.contains("1 symbol(s) and 1 code file(s) were kept as tombstones"),
        "{applied}"
    );

    let after = h.ok(&["project", "prune"]);
    assert!(
        after.contains("no project has a code graph left behind"),
        "{after}"
    );
    let doctor = String::from_utf8_lossy(&h.run(&["doctor"]).stdout).into_owned();
    assert!(!doctor.contains("projects:"), "{doctor}");
    // The memory is still there, and so is what it was about.
    let shown = h.ok(&["get", &id]);
    assert!(shown.contains("slugify drops non-ASCII letters"), "{shown}");
}

/// `sync push --full` sends what the cursor has already passed. A hub on an
/// older build dropped part of each record (which memory replaced it), and
/// nothing sends those rows again on its own: the push cursor only moves
/// forward. After the hub is upgraded, one full push repairs it.
#[test]
fn sync_push_full_resends_what_the_cursor_has_passed() {
    let (_guard, hub, clients) = hub_with_clients(1);
    let a = clients[0].path();
    let t = SYNC_TEST_TOKEN;

    let old = sync_client_ok(a, t, &["remember", "deploys go out on fridays", "-g"]);
    let new = sync_client_ok(
        a,
        t,
        &["remember", "release trains leave on tuesdays only", "-g"],
    );
    let id = |line: &str| line.split_whitespace().next().unwrap().to_string();
    sync_client_ok(a, t, &["supersede", &id(&old), "--by", &id(&new)]);
    sync_client_ok(a, t, &["sync", "push"]);

    // What an older hub leaves behind: the rows, without the retirement.
    let hub_db = hub.path().join("mimir.db");
    let retired_on_hub = || -> i64 {
        rusqlite::Connection::open(&hub_db)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM node WHERE superseded_by IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap()
    };
    assert_eq!(retired_on_hub(), 1);
    rusqlite::Connection::open(&hub_db)
        .unwrap()
        .execute("UPDATE node SET superseded_by = NULL", [])
        .unwrap();

    // The cursor is past both rows, so an ordinary push has nothing to say.
    {
        let conn = rusqlite::Connection::open(a.join("mimir.db")).unwrap();
        mimir_core::replicate::set_watermark(&conn, "last_push", unix_now() + 60).unwrap();
    }
    let plain = sync_client_ok(a, t, &["sync", "push"]);
    assert!(plain.contains("pushed 0 change(s)"), "{plain}");
    assert_eq!(retired_on_hub(), 0);

    let full = sync_client_ok(a, t, &["sync", "push", "--full"]);
    assert!(full.contains("pushed 2 memories (everything)"), "{full}");
    assert_eq!(retired_on_hub(), 1, "the retirement reached the hub");
}

/// A name is tried before a path, across both kinds of collection. A docs
/// collection that merely sits in a folder called `proj` must not be acted
/// on by `docs remove proj` when another collection is *named* `proj`.
#[test]
fn a_docs_command_prefers_the_collection_named_so_over_one_located_there() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    h.write("other/main.rs", "fn main() {}\n");
    h.write("proj/guide.md", "# Guide\n\nInstall the widget.\n");
    h.ok(&["code", "add", "other", "--name", "proj"]);
    h.ok(&["docs", "add", "proj", "--name", "handbook"]);

    let out = h.ok(&["docs", "remove", "proj"]);

    assert!(out.starts_with("removed proj"), "{out}");
    let listed = h.ok(&["docs", "list"]);
    assert!(
        listed.contains("handbook") && !listed.contains("proj  "),
        "the collection named `proj` is the one that went: {listed}"
    );
}

/// A real repository with one commit and one linked worktree, made by git
/// itself, so the tests below read exactly what `git worktree add` writes.
struct WorktreeRepo {
    _dir: tempfile::TempDir,
    main: std::path::PathBuf,
    wt: std::path::PathBuf,
}

/// Run git in `dir` with nothing of the machine's own configuration: no
/// global or system config, a throwaway identity, no signing, no prompts.
/// `None` when git cannot be started at all.
fn git(dir: &Path, args: &[&str]) -> Option<Output> {
    let no_config = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let mut cmd = Command::new("git");
    cmd.args([
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=",
    ])
    .args(args)
    .current_dir(dir)
    .env("GIT_CONFIG_GLOBAL", no_config)
    .env("GIT_CONFIG_SYSTEM", no_config)
    .env("GIT_CONFIG_NOSYSTEM", "1")
    .env("GIT_TERMINAL_PROMPT", "0");
    // A test run started from inside a git hook inherits that repository.
    for var in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
    ] {
        cmd.env_remove(var);
    }
    cmd.output().ok()
}

impl WorktreeRepo {
    /// `main/` with one committed source file (and a committed `.mimir` id
    /// when `marker` is set) and `wt/`, a linked worktree of it on a branch
    /// of its own. `None`, with a printed note, when git is not on PATH.
    fn new(h: &Harness, marker: bool) -> Option<Self> {
        if git(Path::new("."), &["--version"]).is_none_or(|out| !out.status.success()) {
            eprintln!("skipped: `git` is not on PATH, and this test needs a real linked worktree");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        // Canonical, so the paths compared below are the ones Mimir stores.
        let base = dir.path().canonicalize().unwrap();
        let (main, wt) = (base.join("main"), base.join("wt"));
        std::fs::create_dir_all(main.join("src")).unwrap();
        std::fs::write(
            main.join("src/lib.rs"),
            "pub fn slugify() {}\npub fn run() { slugify(); }\n",
        )
        .unwrap();
        let must = |dir: &Path, args: &[&str]| {
            let out = git(dir, args).expect("git runs");
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        must(&main, &["init", "-q"]);
        if marker {
            // Before the commit, so the worktree checks the marker out too.
            in_dir(h, &main, &["project", "init"]);
        }
        must(&main, &["add", "-A"]);
        must(&main, &["commit", "-q", "-m", "first"]);
        must(&main, &["worktree", "add", "-q", "-b", "feature", "../wt"]);
        assert!(
            wt.join(".git").is_file(),
            "a linked worktree's .git is a file"
        );
        Some(WorktreeRepo {
            _dir: dir,
            main,
            wt,
        })
    }
}

/// Run `mimir` in `dir` inside the harness's sandbox and return its stdout;
/// the command has to succeed.
fn in_dir(h: &Harness, dir: &Path, args: &[&str]) -> String {
    let out = h.command(args).current_dir(dir).output().unwrap();
    assert!(
        out.status.success(),
        "mimir {args:?} in {}:\nstdout: {}\nstderr: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Every project row as `(id, path, portable key)`, straight from the store.
fn project_rows(h: &Harness) -> Vec<(i64, Option<String>, Option<String>)> {
    let conn = rusqlite::Connection::open(h.home.path().join("mimir.db")).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT id, path, json_extract(meta, '$.portable_key') FROM node
             WHERE kind = 'project' ORDER BY id",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

/// How many live symbols the store holds for the project registered at
/// `root`, by name.
fn symbols_of(h: &Harness, root: &Path) -> Vec<String> {
    let conn = rusqlite::Connection::open(h.home.path().join("mimir.db")).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT COALESCE(json_extract(s.meta, '$.name'), s.title) FROM node s
             JOIN node p ON p.id = s.project_id
             WHERE s.kind = 'symbol' AND s.deleted_at IS NULL AND p.path = ?1
             ORDER BY 1",
        )
        .unwrap();
    let rows = stmt
        .query_map([root.to_string_lossy()], |r| r.get::<_, String>(0))
        .unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

/// Memories belong to the repository: what is remembered in the main
/// checkout is listed and recalled from a linked worktree of it, and what
/// is remembered in the worktree is listed from the main checkout. Both
/// for a repository identified by its path alone and for one with a
/// committed `.mimir` id.
#[test]
fn a_linked_worktree_shares_the_memories_of_its_main_checkout() {
    for marker in [false, true] {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let Some(repo) = WorktreeRepo::new(&h, marker) else {
            return;
        };
        in_dir(
            &h,
            &repo.main,
            &[
                "remember",
                "the importer batches inserts by five hundred rows",
                "-t",
                "gotcha",
            ],
        );
        let listed = in_dir(&h, &repo.wt, &["list"]);
        assert!(
            listed.contains("importer batches inserts"),
            "marker {marker}: remembered in main, listed from the worktree: {listed}"
        );
        assert!(
            !listed.contains("pr:wt"),
            "and it is the main checkout's project it is filed under: {listed}"
        );
        let recalled = in_dir(&h, &repo.wt, &["recall", "importer batches inserts"]);
        assert!(
            recalled.contains("importer batches inserts"),
            "marker {marker}: recalled from the worktree: {recalled}"
        );

        let stored = in_dir(
            &h,
            &repo.wt.join("src"),
            &[
                "remember",
                "the feature branch renames the retry budget to attempts",
                "-t",
                "decision",
            ],
        );
        assert!(
            stored.contains("pr:main"),
            "marker {marker}: filed under the main checkout's project as it is stored, \
             not moved there afterwards: {stored}"
        );
        let listed = in_dir(&h, &repo.main, &["list"]);
        assert!(
            listed.contains("renames the retry budget") && listed.contains("importer batches"),
            "marker {marker}: remembered in the worktree, listed from main: {listed}"
        );
        // One project holds both, and it is the main checkout's.
        let conn = rusqlite::Connection::open(h.home.path().join("mimir.db")).unwrap();
        let owners: Vec<Option<String>> = conn
            .prepare(
                "SELECT p.path FROM node m LEFT JOIN node p ON p.id = m.project_id
                 WHERE m.kind = 'memory'",
            )
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let main_path = repo.main.to_string_lossy().into_owned();
        assert_eq!(
            owners,
            vec![Some(main_path.clone()), Some(main_path)],
            "marker {marker}"
        );
    }
}

/// The code graph belongs to the checkout: building it in a linked worktree
/// builds the worktree's own, under its own project row, and the main
/// checkout's graph and row stay as they were. For a repository with a
/// `.mimir` id the two used to share one row, whose path (and so whose
/// graph) followed whichever checkout ran last.
#[test]
fn a_linked_worktree_builds_its_own_code_graph_and_leaves_the_main_checkouts_alone() {
    for marker in [false, true] {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let Some(repo) = WorktreeRepo::new(&h, marker) else {
            return;
        };
        in_dir(&h, &repo.main, &["graph", "build"]);
        assert_eq!(symbols_of(&h, &repo.main), ["run", "slugify"]);

        // The branch in the worktree has a function main does not.
        std::fs::write(
            repo.wt.join("src/lib.rs"),
            "pub fn slugify() {}\npub fn run() { slugify(); }\npub fn only_on_the_branch() {}\n",
        )
        .unwrap();
        in_dir(&h, &repo.wt, &["graph", "build"]);
        assert_eq!(
            symbols_of(&h, &repo.wt),
            ["only_on_the_branch", "run", "slugify"],
            "marker {marker}: the worktree's graph, under its own project"
        );
        assert_eq!(
            symbols_of(&h, &repo.main),
            ["run", "slugify"],
            "marker {marker}: the main checkout's graph is untouched"
        );

        let found = in_dir(&h, &repo.wt, &["graph", "node", "only_on_the_branch"]);
        assert!(found.contains("only_on_the_branch"), "{found}");
        let missing = h
            .command(&["graph", "node", "only_on_the_branch"])
            .current_dir(&repo.main)
            .output()
            .unwrap();
        assert!(
            !String::from_utf8_lossy(&missing.stdout).contains("src/lib.rs"),
            "marker {marker}: main's graph does not know the branch's function: {}",
            String::from_utf8_lossy(&missing.stdout)
        );

        // Two rows, each on its own directory, whichever ran last; the
        // portable key, when there is one, is the main checkout's alone.
        in_dir(&h, &repo.main, &["status"]);
        in_dir(&h, &repo.wt, &["status"]);
        let rows = project_rows(&h);
        let row_at = |root: &Path| {
            rows.iter()
                .find(|(_, path, _)| path.as_deref() == Some(&*root.to_string_lossy()))
                .unwrap_or_else(|| panic!("no project row at {}: {rows:?}", root.display()))
        };
        assert_eq!(rows.len(), 2, "marker {marker}: {rows:?}");
        assert_eq!(row_at(&repo.main).2.is_some(), marker, "{rows:?}");
        assert_eq!(row_at(&repo.wt).2, None, "{rows:?}");
    }
}

/// `mimir status` says which is which in a linked worktree: the project is
/// the checkout's own, and one line names the main checkout and the project
/// the memories are filed under. In the main checkout there is nothing to
/// tell apart and the line is absent.
#[test]
fn status_in_a_linked_worktree_names_the_main_checkout_and_the_memory_project() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    let (main, wt) = (
        repo.main.to_string_lossy().into_owned(),
        repo.wt.to_string_lossy().into_owned(),
    );
    let text = in_dir(&h, &repo.wt, &["status"]);
    assert!(
        text.contains(&format!("project wt ({wt})  [via: git]")),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "linked worktree of {main}; memories are filed under project main"
        )),
        "{text}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.wt, &["status", "--json"])).unwrap();
    assert_eq!(json["project"], "wt");
    assert_eq!(json["project_path"], wt.as_str());
    assert_eq!(json["checkout"], "linked_worktree");
    assert_eq!(json["worktree_of"], main.as_str());
    assert_eq!(json["memory_project"], "main");
    assert_eq!(json["memory_project_path"], main.as_str());

    let text = in_dir(&h, &repo.main, &["status"]);
    assert!(text.contains(&format!("project main ({main})")), "{text}");
    assert!(!text.contains("linked worktree"), "{text}");
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.main, &["status", "--json"])).unwrap();
    assert_eq!(json["checkout"], "standalone");
    assert_eq!(json["worktree_of"], serde_json::Value::Null);
    assert_eq!(json["memory_project"], "main");
    assert_eq!(json["memory_project_path"], main.as_str());

    // Outside any project there is no checkout to classify.
    let json: serde_json::Value = serde_json::from_str(&h.ok(&["status", "--json"])).unwrap();
    assert_eq!(json["checkout"], serde_json::Value::Null);
    assert_eq!(json["memory_project"], serde_json::Value::Null);
}

/// A worktree is deleted when its branch merges. What was remembered in it
/// stays with the repository, and nothing holds its project row back:
/// `project prune --apply` removes the row along with its code graph.
#[test]
fn a_deleted_worktrees_project_is_pruned_and_its_memories_stay_with_the_repository() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    in_dir(&h, &repo.main, &["status"]);
    in_dir(&h, &repo.wt, &["graph", "build"]);
    in_dir(
        &h,
        &repo.wt,
        &[
            "remember",
            "slugify must keep digits; the branch learned this the hard way",
            "-t",
            "gotcha",
        ],
    );
    assert_eq!(project_rows(&h).len(), 2);

    std::fs::remove_dir_all(&repo.wt).unwrap();
    let applied = in_dir(&h, &repo.main, &["project", "prune", "--apply"]);
    assert!(
        applied.contains("removed 2 symbols and 1 code files from 1 project(s); 1 project row(s) removed, 0 kept"),
        "{applied}"
    );
    let rows = project_rows(&h);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(
        rows[0].1.as_deref(),
        Some(&*repo.main.to_string_lossy()),
        "the row that is left is the main checkout's"
    );
    let listed = in_dir(&h, &repo.main, &["list"]);
    assert!(
        listed.contains("slugify must keep digits"),
        "the memory outlived the worktree it was written in: {listed}"
    );
}

/// An MCP server lives as long as its session and resolves its project once,
/// at start-up. Started in a linked worktree it keeps two: memories go to,
/// and come from, the repository's project, while the code graph it builds
/// and reads is the worktree's own.
#[test]
fn an_mcp_server_in_a_linked_worktree_shares_memories_and_keeps_its_own_graph() {
    use std::io::{BufRead, BufReader, Write};
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    in_dir(
        &h,
        &repo.main,
        &[
            "remember",
            "the importer batches inserts by five hundred rows",
            "-t",
            "gotcha",
        ],
    );
    std::fs::write(
        repo.wt.join("src/lib.rs"),
        "pub fn slugify() {}\npub fn run() { slugify(); }\npub fn only_on_the_branch() {}\n",
    )
    .unwrap();

    let mut child = h
        .command(&["mcp"])
        .current_dir(&repo.wt)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let _guard = HubGuard(child);
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut next_id = 0;
    let mut request = |method: &str, params: serde_json::Value| -> String {
        next_id += 1;
        let message = serde_json::json!({
            "jsonrpc": "2.0", "id": next_id, "method": method, "params": params,
        });
        writeln!(stdin, "{message}").unwrap();
        if next_id == 1 {
            writeln!(
                stdin,
                r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
            )
            .unwrap();
        }
        stdin.flush().unwrap();
        let wanted = format!("\"id\":{next_id}");
        loop {
            let line = rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .unwrap_or_else(|_| panic!("no response to {method} within 30s"));
            if line.contains(&wanted) {
                return line;
            }
        }
    };
    request(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "e2e", "version": "0" },
        }),
    );
    let mut call = |name: &str, arguments: serde_json::Value| {
        request(
            "tools/call",
            serde_json::json!({ "name": name, "arguments": arguments }),
        )
    };

    let status = call("status", serde_json::json!({}));
    assert!(
        status.contains("project wt  [via: git]")
            && status.contains("memories are filed under project main"),
        "{status}"
    );

    // Remembered in the main checkout, recalled through the worktree's server.
    let recalled = call(
        "recall",
        serde_json::json!({ "query": "importer batches inserts", "kind": "memory" }),
    );
    assert!(recalled.contains("importer batches inserts"), "{recalled}");

    // Remembered through the worktree's server, filed under the repository.
    let stored = call(
        "remember",
        serde_json::json!({
            "text": "the feature branch renames the retry budget to attempts",
            "type": "decision",
        }),
    );
    assert!(
        stored.contains("stored m:") && stored.contains("pr:main"),
        "{stored}"
    );
    let listed = in_dir(&h, &repo.main, &["list"]);
    assert!(listed.contains("renames the retry budget"), "{listed}");

    // The graph the server builds at start-up is the worktree's own, under
    // the worktree's own project; nothing is built for the main checkout.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !symbols_of(&h, &repo.wt).contains(&"only_on_the_branch".to_string()) {
        assert!(
            std::time::Instant::now() < deadline,
            "the start-up graph build never reached the worktree's project: {:?}",
            project_rows(&h)
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(symbols_of(&h, &repo.main).is_empty());
    let peeked = call(
        "peek",
        serde_json::json!({ "symbol": "only_on_the_branch" }),
    );
    assert!(peeked.contains("pub fn only_on_the_branch()"), "{peeked}");

    // `recall` reads both projects at once: the worktree's own symbol with
    // the default kinds and with `kind: code`, next to the repository's
    // memory it found above.
    for arguments in [
        serde_json::json!({ "query": "only_on_the_branch" }),
        serde_json::json!({ "query": "only_on_the_branch", "kind": "code" }),
    ] {
        let found = call("recall", arguments.clone());
        assert!(
            found.contains("only_on_the_branch") && found.contains("pr:wt"),
            "{arguments}: {found}"
        );
    }
}

/// A handoff saved by a session in a linked worktree is that session's
/// state. The main checkout shares the worktree's memory project, and a
/// reset there must not restore it; a reset in the worktree does. The
/// brief's "don't repeat what the guard restores" exclusion follows the
/// same checkout.
#[test]
fn a_handoff_saved_in_a_linked_worktree_is_not_restored_in_the_main_checkout() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    h.write_config("[hooks]\ncontext_guard = \"handoff\"\n\n[brief]\nenabled = true\n");
    // A decision, so that it is also something a brief would show.
    in_dir(
        &h,
        &repo.wt,
        &[
            "remember",
            "handoff: the feature branch is mid-rebase; resolve src/lib.rs next",
            "-t",
            "decision",
            "--tags",
            "session-handoff",
        ],
    );
    let reset = |hook: &[&str], cwd: &Path, session: &str| {
        let event = serde_json::json!({
            "session_id": session, "source": "clear", "cwd": cwd
        });
        out_text(&h.hook(hook, &event).stdout)
    };

    // The brief of the session about to get the handoff restored leaves it
    // out; the main checkout's brief has no such reason to.
    let brief = reset(&["brief", "show"], &repo.wt, "in-worktree");
    assert!(!brief.contains("mid-rebase"), "{brief}");
    let brief = reset(&["brief", "show"], &repo.main, "in-main");
    assert!(brief.contains("mid-rebase"), "{brief}");

    let guard = ["context-guard", "session-start"];
    assert_eq!(
        reset(&guard, &repo.main, "in-main"),
        "",
        "the main checkout must not resume from the worktree's state"
    );
    let restored = reset(&guard, &repo.wt.join("src"), "in-worktree");
    assert!(
        restored.starts_with("Restored session handoff:\n") && restored.contains("mid-rebase"),
        "{restored}"
    );
    assert_eq!(reset(&guard, &repo.wt, "in-worktree"), "", "restored once");
}

/// Every surface that reads or writes memories goes through the
/// repository's project when it runs in a linked worktree: the rules pack,
/// the brief, per-prompt injection, guard anchors, the miss ledger and
/// `reproject`. `link --scan` uses both projects: the memories are the
/// repository's, the symbols they name are in the worktree's own graph.
#[test]
fn memory_surfaces_in_a_linked_worktree_use_the_repositorys_project() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    let (main, wt) = (repo.main.as_path(), repo.wt.as_path());
    in_dir(
        &h,
        main,
        &[
            "remember",
            "terraform plan fails unless VAULT_ADDR is exported first",
            "-t",
            "gotcha",
            "--anchor",
            "deploy.sh",
        ],
    );

    // One rules pack for the repository, read and replaced from either.
    in_dir(&h, main, &["rules", "set", "Tabs, never spaces."]);
    let shown = in_dir(&h, wt, &["rules", "show"]);
    assert!(shown.contains("Tabs, never spaces."), "{shown}");
    let replaced = in_dir(&h, wt, &["rules", "set", "Spaces, never tabs."]);
    assert!(
        replaced.starts_with("updated rules pack for main"),
        "the worktree replaces the repository's pack, it does not start its own: {replaced}"
    );
    let shown = in_dir(&h, main, &["rules", "show"]);
    assert!(
        shown.contains("Spaces, never tabs.") && !shown.contains("Tabs, never spaces."),
        "{shown}"
    );

    // What Mimir volunteers into a session in the worktree.
    let brief = in_dir(&h, wt, &["brief"]);
    assert!(brief.contains("VAULT_ADDR"), "the brief: {brief}");
    let injected = in_dir(
        &h,
        wt,
        &[
            "recall-inject",
            "--",
            "why does terraform plan fail on staging",
        ],
    );
    assert!(
        injected.contains("VAULT_ADDR"),
        "per-prompt injection: {injected}"
    );
    let edit = serde_json::json!({
        "session_id": "s1",
        "cwd": wt,
        "tool_name": "Edit",
        "tool_input": { "file_path": wt.join("deploy.sh") },
    });
    let guard = out_text(&h.hook(&["context-guard", "pretool"], &edit).stdout);
    assert!(guard.contains("VAULT_ADDR"), "the guard anchor: {guard}");

    // A question the store cannot answer, asked in the worktree, is a miss
    // of the repository's.
    in_dir(&h, wt, &["recall", "kubernetes ingress annotations"]);
    let misses = in_dir(&h, main, &["misses"]);
    assert!(
        misses.contains("kubernetes ingress annotations"),
        "{misses}"
    );

    // `link --scan`: a memory of the repository's project names a function
    // that exists only in the worktree's graph.
    std::fs::write(
        wt.join("src/lib.rs"),
        "pub fn slugify() {}\npub fn run() { slugify(); }\npub fn only_on_the_branch() {}\n",
    )
    .unwrap();
    in_dir(&h, wt, &["graph", "build"]);
    in_dir(
        &h,
        main,
        &[
            "remember",
            "`only_on_the_branch` must stay idempotent: the scheduler calls it twice",
            "-t",
            "gotcha",
        ],
    );
    let scan = in_dir(&h, wt, &["link", "--scan"]);
    assert!(
        scan.contains("—mentions→ only_on_the_branch") && scan.contains("created 1 link(s)"),
        "{scan}"
    );

    // A worktree's name stands for the project its memories are filed under.
    let stray = in_dir(
        &h,
        main,
        &[
            "remember",
            "-g",
            "the staging bucket is versioned, the prod one is not",
        ],
    );
    let id = stray.split_whitespace().next().unwrap().to_string();
    let moved = h
        .command(&["reproject", &id, "--project", "wt"])
        .current_dir(main)
        .output()
        .unwrap();
    assert!(moved.status.success(), "{}", stderr_of(&moved));
    assert!(
        out_text(&moved.stdout).contains("pr:main"),
        "{}",
        out_text(&moved.stdout)
    );
    assert!(
        stderr_of(&moved).contains("'wt' is a linked worktree of"),
        "{}",
        stderr_of(&moved)
    );
}

/// `project init` in a linked worktree writes the marker there, and says
/// whose identity it will be: until the committed marker reaches the main
/// checkout, the worktree keeps a project of its own under the new id, as
/// every checkout with its own `.mimir` did before linked worktrees were
/// told apart; then the main checkout's project takes it.
#[test]
fn project_init_in_a_linked_worktree_says_when_the_identity_takes_effect() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    let out = in_dir(&h, &repo.wt, &["project", "init", "--sync"]);
    assert!(repo.wt.join(".mimir").is_file());
    assert!(out.contains("sync     true"), "{out}");
    assert!(
        out.contains("This checkout is a linked worktree of")
            && out.contains("keeps a project of its own")
            && out.contains("Once the committed `.mimir` is checked out there"),
        "{out}"
    );
    let rows = project_rows(&h);
    assert_eq!(
        rows.len(),
        1,
        "the main checkout's row is not made: {rows:?}"
    );
    assert_eq!(rows[0].1.as_deref(), Some(&*repo.wt.to_string_lossy()));
    assert!(
        rows[0].2.is_some(),
        "the worktree's own row holds the key: {rows:?}"
    );
    let status = in_dir(&h, &repo.wt, &["status"]);
    assert!(
        status.contains(
            "keeps a project of its own, because its `.mimir` id is not the main \
             checkout's"
        ),
        "{status}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.wt, &["status", "--json"])).unwrap();
    assert_eq!(json["checkout"], "linked_worktree");
    assert_eq!(json["memory_filing"], "own_key");
    assert_eq!(json["memory_project_path"], &*repo.wt.to_string_lossy());

    // The marker arrives in the main checkout (here: copied, as a merge
    // would bring it): the repository's project takes the identity.
    std::fs::copy(repo.wt.join(".mimir"), repo.main.join(".mimir")).unwrap();
    in_dir(&h, &repo.wt, &["status"]);
    let rows = project_rows(&h);
    let keyed: Vec<_> = rows.iter().filter(|(_, _, key)| key.is_some()).collect();
    assert_eq!(keyed.len(), 1, "{rows:?}");
    assert_eq!(
        keyed[0].1.as_deref(),
        Some(&*repo.main.to_string_lossy()),
        "{rows:?}"
    );
    let out = in_dir(&h, &repo.main, &["project", "init"]);
    assert!(!out.contains("linked worktree"), "{out}");
}

/// A daemon started in a linked worktree answers a prompt that names no
/// `cwd` (an older hook script sends none) from the scope it was started
/// in. That scope is the repository's memory project, not the worktree's
/// own row, which holds no memories.
#[test]
fn a_daemon_started_in_a_linked_worktree_injects_its_repositorys_memories() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    in_dir(
        &h,
        &repo.main,
        &[
            "remember",
            "terraform plan fails unless VAULT_ADDR is exported first",
            "-t",
            "gotcha",
        ],
    );
    let port = free_port();
    let _daemon = HubGuard(
        h.command(&["daemon"])
            .env(
                "MIMIR_INJECT_URL",
                format!("http://127.0.0.1:{port}/inject"),
            )
            .current_dir(&repo.wt)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_for_hub(port);
    let answer = ureq::get(&format!("http://127.0.0.1:{port}/inject"))
        .query("prompt", "why does terraform plan fail on staging")
        .timeout(std::time::Duration::from_secs(30))
        .call()
        .expect("/inject answers")
        .into_string()
        .unwrap();
    assert!(answer.contains("VAULT_ADDR"), "{answer:?}");
}

/// `recall` is the one surface that reads memories and code and docs in one
/// ranked list. In a linked worktree it answers with the repository's
/// memories and the worktree's own code and docs, whatever the kind filter:
/// never with the main checkout's code or docs, and never without the
/// worktree's.
#[test]
fn recall_in_a_linked_worktree_reads_its_repositorys_memories_and_its_own_code_and_docs() {
    for marker in [false, true] {
        let h = Harness::new();
        h.ok(&["init", "--no-model"]);
        let Some(repo) = WorktreeRepo::new(&h, marker) else {
            return;
        };
        let (main, wt) = (repo.main.as_path(), repo.wt.as_path());
        // The two checkouts have diverged: a function and a doc each that
        // the other does not have, and `slugify` in both.
        let checkout = |root: &Path, function: &str, doc: &str| {
            std::fs::write(
                root.join("src/lib.rs"),
                format!("pub fn slugify() {{}}\npub fn run() {{ slugify(); }}\npub fn {function}() {{}}\n"),
            )
            .unwrap();
            std::fs::create_dir_all(root.join("docs")).unwrap();
            std::fs::write(root.join("docs/guide.md"), format!("# Guide\n\n{doc}\n")).unwrap();
            in_dir(&h, root, &["graph", "build"]);
            let name = format!("{}-docs", root.file_name().unwrap().to_string_lossy());
            in_dir(&h, root, &["docs", "add", "docs", "--name", &name]);
        };
        checkout(
            main,
            "only_in_main",
            "The zorblat procedure is run from the main checkout.",
        );
        checkout(
            wt,
            "only_on_the_branch",
            "The frobnicator widget is installed from the official site.",
        );
        in_dir(&h, main, &["index"]);
        in_dir(
            &h,
            main,
            &[
                "remember",
                "slugify must keep digits; the importer relies on it",
                "-t",
                "gotcha",
            ],
        );
        let recall = |dir: &Path, args: &[&str]| {
            let mut full = vec!["recall"];
            full.extend_from_slice(args);
            in_dir(&h, dir, &full)
        };
        let case = |what: &str| format!("marker {marker}: {what}");

        // The four ways a worktree asks.
        let memory = recall(wt, &["importer relies on slugify digits"]);
        assert!(
            memory.contains("slugify must keep digits"),
            "{}: {memory}",
            case("memory")
        );
        let default_kinds = recall(wt, &["only_on_the_branch"]);
        assert!(
            default_kinds.contains("only_on_the_branch") && default_kinds.contains("pr:wt"),
            "{}: {default_kinds}",
            case("the worktree's symbol, default kinds")
        );
        let code = recall(wt, &["only_on_the_branch", "--kind", "code"]);
        assert!(
            code.contains("only_on_the_branch"),
            "{}: {code}",
            case("--kind code")
        );
        let doc = recall(wt, &["frobnicator", "--kind", "doc"]);
        assert!(doc.contains("frobnicator"), "{}: {doc}", case("--kind doc"));

        // One page, both projects: the repository's memory beside the
        // worktree's own `slugify`, and not the main checkout's.
        let page = recall(wt, &["slugify", "--json"]);
        let hits: Vec<serde_json::Value> = page
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let owners = |kind: &str| -> Vec<String> {
            hits.iter()
                .filter(|hit| hit["kind"] == kind)
                .map(|hit| hit["project"].as_str().unwrap_or("(global)").to_string())
                .collect()
        };
        assert_eq!(owners("memory"), ["main"], "{}: {page}", case("the memory"));
        assert!(!owners("symbol").is_empty(), "{}: {page}", case("a symbol"));
        assert!(
            owners("symbol").iter().all(|owner| owner == "wt"),
            "{}: {page}",
            case("symbols are the worktree's own")
        );

        // The main checkout's code and docs are not the worktree's to find,
        // nor the other way round; `--all` still reaches everything.
        for (dir, args) in [
            (wt, &["only_in_main"][..]),
            (wt, &["only_in_main", "--kind", "code"]),
            (wt, &["zorblat", "--kind", "doc"]),
            (main, &["only_on_the_branch"]),
            (main, &["frobnicator", "--kind", "doc"]),
        ] {
            assert_eq!(
                recall(dir, args).trim(),
                "no results",
                "{}",
                case(&format!("{args:?} from {}", dir.display()))
            );
        }
        assert!(recall(main, &["only_in_main"]).contains("only_in_main"));
        assert!(recall(main, &["zorblat", "--kind", "doc"]).contains("zorblat"));
        assert!(recall(wt, &["only_in_main", "--all"]).contains("only_in_main"));
    }
}

/// Set when git made the fixture's worktree (the modification time of its
/// `commondir`, which is what Mimir reads) to an hour ago, so a memory the
/// test writes now is unambiguously newer than the worktree.
fn worktree_made_an_hour_ago(repo: &WorktreeRepo, name: &str) {
    let commondir = repo
        .main
        .join(".git/worktrees")
        .join(name)
        .join("commondir");
    std::fs::File::options()
        .write(true)
        .open(&commondir)
        .unwrap()
        .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(3600))
        .unwrap();
}

/// What an older Mimir did in a linked worktree: it filed the memory under
/// the worktree's own project row. Re-files memory `short_id` there by hand,
/// optionally with an older `created_at`.
fn file_under_worktree_row(h: &Harness, wt: &Path, short_id: &str, created_at: Option<i64>) {
    let conn = db_conn(h);
    let prefix = short_id.trim_start_matches("m:");
    let changed = conn
        .execute(
            "UPDATE node SET project_id = (SELECT id FROM node WHERE kind = 'project'
                                           AND path = ?1),
                             created_at = COALESCE(?3, created_at)
             WHERE kind = 'memory' AND uid LIKE '%' || ?2",
            rusqlite::params![wt.to_string_lossy(), prefix, created_at],
        )
        .unwrap();
    assert_eq!(changed, 1, "one memory {short_id}");
}

/// Where memory `short_id` is filed: its project's path.
fn filed_at(h: &Harness, short_id: &str) -> Option<String> {
    db_conn(h)
        .query_row(
            "SELECT p.path FROM node m LEFT JOIN node p ON p.id = m.project_id
             WHERE m.kind = 'memory' AND m.uid LIKE '%' || ?1",
            [short_id.trim_start_matches("m:")],
            |r| r.get(0),
        )
        .unwrap()
}

/// A memory still filed under a linked worktree's own row (an older Mimir
/// put it there) stays put through every hook and through `status`, which
/// reports it; the first memory command run there moves it home and says
/// so in one line on stderr, once.
#[test]
fn a_worktrees_stranded_memories_move_home_on_a_memory_command_and_on_nothing_else() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    worktree_made_an_hour_ago(&repo, "wt");
    let wt_path = repo.wt.to_string_lossy().into_owned();
    let stored = in_dir(
        &h,
        &repo.wt,
        &[
            "remember",
            "the importer batches inserts by five hundred rows",
            "-t",
            "gotcha",
        ],
    );
    let id = stored.split_whitespace().next().unwrap().to_string();
    file_under_worktree_row(&h, &repo.wt, &id, None);
    assert_eq!(filed_at(&h, &id).as_deref(), Some(wt_path.as_str()));

    // Hooks read; so does status.
    let edit = serde_json::json!({
        "session_id": "s1", "cwd": repo.wt, "tool_name": "Bash",
        "tool_input": { "command": "ls" },
    });
    let reset = serde_json::json!({ "session_id": "s1", "source": "clear", "cwd": repo.wt });
    for (hook, event) in [
        (&["context-guard", "pretool"][..], &edit),
        (&["context-guard", "session-start"], &reset),
        (&["brief", "show"], &reset),
        (&["rules", "show", "--hook"], &reset),
    ] {
        let out = h.hook(hook, event);
        assert!(out.status.success(), "{hook:?}: {}", stderr_of(&out));
        assert_eq!(
            filed_at(&h, &id).as_deref(),
            Some(wt_path.as_str()),
            "{hook:?}"
        );
    }
    in_dir(
        &h,
        &repo.wt,
        &["recall-inject", "--", "how does the importer batch"],
    );
    let status = in_dir(&h, &repo.wt, &["status"]);
    assert!(
        status.contains("1 memory is still filed under this worktree's own project: not moved yet"),
        "{status}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.wt, &["status", "--json"])).unwrap();
    assert_eq!(json["memory_filing"], "home");
    assert_eq!(json["worktree_memories"]["to_move"], 1);
    assert_eq!(json["worktree_memories"]["kept"], 0);
    assert_eq!(filed_at(&h, &id).as_deref(), Some(wt_path.as_str()));
    let doctor = h
        .command(&["doctor"])
        .current_dir(&repo.wt)
        .output()
        .unwrap();
    assert!(
        !stderr_of(&doctor).contains("moved"),
        "{}",
        stderr_of(&doctor)
    );
    assert_eq!(
        filed_at(&h, &id).as_deref(),
        Some(wt_path.as_str()),
        "doctor"
    );

    // A memory command moves it, and says so once.
    let listed = h.command(&["list"]).current_dir(&repo.wt).output().unwrap();
    assert!(listed.status.success());
    assert_eq!(
        stderr_of(&listed).trim(),
        format!(
            "moved 1 memory from linked worktree {wt_path} to project main (`mimir audit` lists \
             it)"
        )
    );
    assert!(out_text(&listed.stdout).contains("importer batches"));
    assert_eq!(
        filed_at(&h, &id).as_deref(),
        Some(&*repo.main.to_string_lossy())
    );
    let again = h.command(&["list"]).current_dir(&repo.wt).output().unwrap();
    assert_eq!(stderr_of(&again), "");
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.wt, &["status", "--json"])).unwrap();
    assert_eq!(json["worktree_memories"], serde_json::Value::Null);
}

/// A memory older than the worktree under the same path was written by a
/// checkout that was there before, possibly of another repository. It is
/// never moved into this repository's project; `status` says why it stays.
#[test]
fn a_memory_older_than_the_worktree_at_its_path_stays_where_it_is() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    let wt_path = repo.wt.to_string_lossy().into_owned();
    let stored = in_dir(
        &h,
        &repo.wt,
        &[
            "remember",
            "repo A's billing reconciler runs before the ledger close",
        ],
    );
    let id = stored.split_whitespace().next().unwrap().to_string();
    file_under_worktree_row(&h, &repo.wt, &id, Some(1_000_000));

    let moved = h.command(&["list"]).current_dir(&repo.wt).output().unwrap();
    assert!(
        !stderr_of(&moved).contains("moved"),
        "{}",
        stderr_of(&moved)
    );
    assert_eq!(filed_at(&h, &id).as_deref(), Some(wt_path.as_str()));
    let status = in_dir(&h, &repo.wt, &["status"]);
    assert!(
        status.contains(
            "1 memory is filed under this worktree's own project and older than the worktree"
        ),
        "{status}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&in_dir(&h, &repo.wt, &["status", "--json"])).unwrap();
    assert_eq!(json["worktree_memories"]["to_move"], 0);
    assert_eq!(json["worktree_memories"]["kept"], 1);
}

/// When the home project syncs and the worktree's own row did not, a move
/// makes the memories part of what `mimir sync` sends; the line says so.
#[test]
fn a_move_into_a_project_that_syncs_says_the_memories_will_be_sent() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, true) else {
        return;
    };
    worktree_made_an_hour_ago(&repo, "wt");
    // The same id on both sides, now with sync on.
    let marker = std::fs::read_to_string(repo.main.join(".mimir")).unwrap();
    let synced = marker.replace("sync = false", "sync = true");
    assert_ne!(marker, synced, "{marker}");
    for root in [&repo.main, &repo.wt] {
        std::fs::write(root.join(".mimir"), &synced).unwrap();
    }
    let stored = in_dir(
        &h,
        &repo.wt,
        &["remember", "local scratch: staging rotates on Fridays"],
    );
    let id = stored.split_whitespace().next().unwrap().to_string();
    file_under_worktree_row(&h, &repo.wt, &id, None);

    let moved = h
        .command(&["recall", "staging rotates"])
        .current_dir(&repo.wt)
        .output()
        .unwrap();
    assert!(
        stderr_of(&moved)
            .trim()
            .ends_with("; that project syncs, so it will be sent at the next `mimir sync`"),
        "{}",
        stderr_of(&moved)
    );
}

/// A session handoff is one session's state. Written in worktree A, it
/// reaches no session in worktree B or in the main checkout through any
/// scoped surface: `list` (including the exact command the context guard's
/// pause nudge tells the agent to run), `recall` and per-prompt injection.
/// `--all` and `get <id>` still show it, and A sees it everywhere.
#[test]
fn a_handoff_written_in_one_worktree_reaches_no_session_in_another() {
    let h = Harness::new();
    h.ok(&["init", "--no-model"]);
    let Some(repo) = WorktreeRepo::new(&h, false) else {
        return;
    };
    let wt_b = repo.main.parent().unwrap().join("wt-b");
    let added = git(
        &repo.main,
        &["worktree", "add", "-q", "-b", "feature-b", "../wt-b"],
    )
    .expect("git runs");
    assert!(
        added.status.success(),
        "{}",
        String::from_utf8_lossy(&added.stderr)
    );
    let handoff = |dir: &Path, text: &str| {
        let out = in_dir(
            &h,
            dir,
            &[
                "remember",
                text,
                "-t",
                "decision",
                "--tags",
                "session-handoff",
            ],
        );
        out.split_whitespace().next().unwrap().to_string()
    };
    let from_b = handoff(
        &wt_b,
        "handoff: B was deleting the legacy exporter; three files left",
    );
    let from_a = handoff(
        &repo.wt,
        "handoff: A is mid-rebase onto main; resolve the importer conflict next",
    );
    // What the pause nudge tells the agent to run.
    let nudge = ["list", "--tag", "session-handoff", "-n", "1"];
    let in_a = in_dir(&h, &repo.wt, &nudge);
    assert!(
        in_a.contains("A is mid-rebase") && !in_a.contains("legacy exporter"),
        "{in_a}"
    );
    let in_b = in_dir(&h, &wt_b, &nudge);
    assert!(
        in_b.contains("legacy exporter") && !in_b.contains("mid-rebase"),
        "B's own, past A's newer one: {in_b}"
    );
    assert_eq!(in_dir(&h, &repo.main, &nudge).trim(), "no memories");

    let recalled = in_dir(&h, &wt_b, &["recall", "mid-rebase importer conflict"]);
    assert!(!recalled.contains("mid-rebase"), "{recalled}");
    let recalled = in_dir(&h, &repo.wt, &["recall", "mid-rebase importer conflict"]);
    assert!(recalled.contains("mid-rebase"), "{recalled}");
    let injected = in_dir(
        &h,
        &wt_b,
        &[
            "recall-inject",
            "--",
            "where were we with the rebase importer conflict",
        ],
    );
    assert!(!injected.contains("mid-rebase"), "{injected}");
    let injected = in_dir(
        &h,
        &repo.wt,
        &[
            "recall-inject",
            "--",
            "where were we with the rebase importer conflict",
        ],
    );
    assert!(injected.contains("mid-rebase"), "{injected}");

    let everything = in_dir(&h, &wt_b, &["list", "--all", "--tag", "session-handoff"]);
    assert!(everything.contains("mid-rebase") && everything.contains("legacy exporter"));
    let by_id = in_dir(&h, &wt_b, &["get", &from_a]);
    assert!(by_id.contains("mid-rebase"), "{by_id}");
    let _ = from_b;
}
