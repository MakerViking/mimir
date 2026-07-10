use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mimir_core::config::{Config, Paths};
use mimir_core::format::agent_line;
use mimir_core::memory::{self, Remember, RememberOutcome};
use mimir_core::model::{now_unix, short_uid, Kind, MemoryType, Node, Rel, Scope};
use mimir_core::search::SearchQuery;
use mimir_core::{db, store, Mimir};

pub fn init(no_model: bool, hooks: bool, auto_recall: bool) -> Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    config.save(&paths.config_file)?;
    // Opening creates + migrates the database.
    let _conn = db::open(&paths.db_file)?;
    println!("config  {}", paths.config_file.display());
    println!("db      {}", paths.db_file.display());
    if no_model {
        println!(
            "model   skipped (BM25-only; run `mimir embed --fetch` to enable semantic search)"
        );
    } else {
        match mimir_core::embed::Embedder::load(
            &paths,
            &config.embedding.model,
            &config.embedding.device,
            true,
        ) {
            Ok(e) => println!("model   {} ready ({}-dim)", e.name, e.dim),
            Err(err) => {
                eprintln!("model   download failed ({err}); search is BM25-only until `mimir embed --fetch` succeeds")
            }
        }
    }
    install_agent_commands(&config);
    if hooks {
        if let Err(err) = install_hooks(&config, auto_recall) {
            eprintln!("hooks   install failed: {err}");
        }
    }
    println!();
    println!("Register the MCP server once, globally:");
    println!("  claude mcp add --scope user mimir -- mimir mcp");
    if !hooks {
        println!();
        println!("Token-saving hooks (filter command output + inject project rules) are opt-in:");
        println!("  mimir init --hooks");
    } else if !auto_recall {
        println!();
        println!("Per-prompt auto-recall (inject a relevant memory into every prompt) is opt-in:");
        println!("  mimir init --hooks --auto-recall");
    }
    Ok(())
}

/// The PreToolUse hook script: delegates all rewrite logic to `mimir rewrite`,
/// so rules live in the Rust binary (single source of truth), not this file.
const MIMIR_REWRITE_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 1
# Mimir PreToolUse hook — rewrites noisy commands through `mimir run` to save
# tokens. All logic lives in `mimir rewrite`. Requires: mimir, jq.
command -v jq >/dev/null 2>&1 || exit 0
command -v mimir >/dev/null 2>&1 || exit 0
INPUT=$(cat)
CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')
[ -z "$CMD" ] && exit 0
REWRITTEN=$(mimir rewrite "$CMD" 2>/dev/null) || exit 0
[ "$CMD" = "$REWRITTEN" ] && exit 0
UPDATED=$(printf '%s' "$INPUT" | jq -c --arg cmd "$REWRITTEN" '.tool_input | .command = $cmd')
jq -n --argjson updated "$UPDATED" '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "Mimir token-saving rewrite",
    "updatedInput": $updated
  }
}'
"#;

/// The opt-in UserPromptSubmit hook script: tries the warm `/inject` HTTP
/// endpoint first (only live while `mimir mcp --http` is running — see
/// `mcp.rs::inject_router`), falling back to the cold `mimir recall-inject`
/// CLI path when that's unreachable. Both paths run the exact same
/// relevance-floor/formatting/budget logic (`mimir_core::inject::compute`),
/// so which one answers is purely a latency concern, never a behavior one.
/// Unlike `MIMIR_REWRITE_SH`, plain stdout (not a `hookSpecificOutput` JSON
/// envelope) is how Claude Code injects UserPromptSubmit context — same
/// mechanism as the SessionStart rules hook.
///
/// The warm endpoint's address is `config.hooks.inject_url`
/// (`HooksConfig::inject_url`, default `127.0.0.1:8077`, matching the port
/// documented in README.md's remote-MCP example) — baked into the script at
/// install time by `render_recall_script`. `MIMIR_INJECT_URL` still
/// overrides it at hook-invocation time, for a one-off run against a
/// different bind without touching config.toml. If no daemon answers there,
/// curl fails fast (2s timeout) and behavior is identical to before this
/// endpoint existed — just slower.
const MIMIR_RECALL_SH: &str = r#"#!/usr/bin/env bash
# mimir-hook-version: 2
# Mimir UserPromptSubmit hook — prints at most one relevant memory (or
# nothing) as extra context for this turn. Tries the warm HTTP endpoint
# (fast; requires `mimir mcp --http` to be running) first, falls back to
# the cold CLI path (slower, always available). All relevance-floor logic
# lives in mimir_core::inject::compute, shared by both. Requires: mimir, jq;
# curl is optional (falls straight to cold path without it).
command -v jq >/dev/null 2>&1 || exit 0
command -v mimir >/dev/null 2>&1 || exit 0
INPUT=$(cat)
PROMPT=$(printf '%s' "$INPUT" | jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
INJECT_URL="${MIMIR_INJECT_URL:-__MIMIR_INJECT_URL_DEFAULT__}"
if command -v curl >/dev/null 2>&1; then
    ENC_PROMPT=$(printf '%s' "$PROMPT" | jq -sRr @uri)
    if WARM=$(curl -sf --max-time 2 "${INJECT_URL}?prompt=${ENC_PROMPT}" 2>/dev/null); then
        [ -n "$WARM" ] && printf '%s\n' "$WARM"
        exit 0
    fi
fi
mimir recall-inject "$PROMPT" 2>/dev/null
exit 0
"#;

/// Bakes `config.hooks.inject_url` into `MIMIR_RECALL_SH`'s fallback default,
/// split out as a pure function so it's unit-testable without touching disk.
fn render_recall_script(inject_url: &str) -> String {
    MIMIR_RECALL_SH.replace("__MIMIR_INJECT_URL_DEFAULT__", inject_url)
}

/// Install the opt-in Claude Code hooks: a PreToolUse(Bash) rewrite hook and a
/// SessionStart hook that injects the project rules pack, plus (when
/// `auto_recall`) a UserPromptSubmit hook that injects at most one relevant
/// memory per prompt. Idempotent, backs up settings.json, and never clobbers
/// existing hooks. `auto_recall=false` leaves `hooks.UserPromptSubmit`
/// untouched — behavior is otherwise identical to before this flag existed
/// (see `merge_hook_settings`'s unit tests). Re-running after editing
/// `config.hooks.inject_url` rewrites `mimir-recall.sh` unconditionally
/// (step 2 below is a plain `fs::write`, no existence check), so the new
/// URL always lands on the next `mimir init --hooks --auto-recall`.
fn install_hooks(config: &Config, auto_recall: bool) -> Result<()> {
    if std::env::var_os("MIMIR_HOME").is_some() {
        return Ok(()); // isolated instances never touch the user's agent config
    }
    let base = directories::BaseDirs::new().context("cannot resolve home directory")?;
    let claude = base.home_dir().join(".claude");
    if !claude.is_dir() {
        println!("hooks   ~/.claude not found — skipping (is Claude Code installed?)");
        return Ok(());
    }

    let hooks_dir = claude.join("hooks");
    std::fs::create_dir_all(&hooks_dir)?;

    // 1. Write the PreToolUse delegate script (executable).
    let script = hooks_dir.join("mimir-rewrite.sh");
    std::fs::write(&script, MIMIR_REWRITE_SH)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))?;
    }
    let script_str = script.to_string_lossy().into_owned();

    // 2. Auto-recall delegate script — only written when opted in. Always
    // rewritten (not skipped if already present), so changing
    // config.hooks.inject_url and re-running takes effect immediately.
    let recall_script_str = if auto_recall {
        let recall_script = hooks_dir.join("mimir-recall.sh");
        std::fs::write(
            &recall_script,
            render_recall_script(&config.hooks.inject_url),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&recall_script, std::fs::Permissions::from_mode(0o755))?;
        }
        Some(recall_script.to_string_lossy().into_owned())
    } else {
        None
    };

    // 3. Merge into settings.json (back it up first).
    let settings_path = claude.join("settings.json");
    let root: serde_json::Value = match std::fs::read_to_string(&settings_path) {
        Ok(text) => {
            std::fs::write(claude.join("settings.json.mimir-bak"), &text)?;
            serde_json::from_str(&text).context("settings.json is not valid JSON")?
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => return Err(e.into()),
    };
    let (root, messages) = merge_hook_settings(root, &script_str, recall_script_str.as_deref())?;

    std::fs::write(&settings_path, serde_json::to_string_pretty(&root)?)?;
    println!("hooks   {}", messages.join("; "));
    println!(
        "hooks   backup at {}",
        claude.join("settings.json.mimir-bak").display()
    );
    Ok(())
}

/// Pure settings.json merge, split out of `install_hooks` so the merge
/// logic (idempotency, which keys get touched) is unit-testable without a
/// real `~/.claude` — `install_hooks` early-returns under `MIMIR_HOME`, so
/// this is the only way to test it at all. `recall_script = None` must
/// leave `hooks.UserPromptSubmit` completely untouched: that's what makes
/// `auto_recall=false` byte-identical to pre-auto-recall behavior.
fn merge_hook_settings(
    mut root: serde_json::Value,
    rewrite_script: &str,
    recall_script: Option<&str>,
) -> Result<(serde_json::Value, Vec<String>)> {
    if !root.is_object() {
        bail!("settings.json is not a JSON object");
    }
    let hooks = root
        .as_object_mut()
        .unwrap()
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("settings.json `hooks` is not an object")?;

    let mut messages: Vec<String> = Vec::new();

    // SessionStart: inject the rules pack (stdout becomes session context).
    let session = hooks
        .entry("SessionStart")
        .or_insert_with(|| serde_json::json!([]));
    let session_arr = session
        .as_array_mut()
        .context("hooks.SessionStart is not an array")?;
    if entries_mention(session_arr, "mimir rules show") {
        messages.push("SessionStart already installed".into());
    } else {
        session_arr.push(serde_json::json!({
            "hooks": [{ "type": "command", "command": "mimir rules show" }]
        }));
        messages.push("SessionStart (project rules) added".into());
    }

    // PreToolUse(Bash): the rewrite hook. Skip if another rewrite hook (e.g.
    // RTK) is present — running both would double-wrap commands.
    let pre = hooks
        .entry("PreToolUse")
        .or_insert_with(|| serde_json::json!([]));
    let pre_arr = pre
        .as_array_mut()
        .context("hooks.PreToolUse is not an array")?;
    if entries_mention(pre_arr, "mimir-rewrite") {
        messages.push("PreToolUse already installed".into());
    } else if entries_mention(pre_arr, "rtk") || entries_mention(pre_arr, "rewrite") {
        messages.push(
            "PreToolUse SKIPPED — another rewrite hook (e.g. RTK) is present. Remove it \
             from ~/.claude/settings.json, then re-run `mimir init --hooks`."
                .into(),
        );
    } else {
        pre_arr.push(serde_json::json!({
            "matcher": "Bash",
            "hooks": [{ "type": "command", "command": rewrite_script }]
        }));
        messages.push("PreToolUse (command filter) added".into());
    }

    // UserPromptSubmit: opt-in auto-recall. Only touched when asked for.
    if let Some(recall_script) = recall_script {
        let prompt = hooks
            .entry("UserPromptSubmit")
            .or_insert_with(|| serde_json::json!([]));
        let prompt_arr = prompt
            .as_array_mut()
            .context("hooks.UserPromptSubmit is not an array")?;
        if entries_mention(prompt_arr, "mimir-recall") {
            messages.push("UserPromptSubmit already installed".into());
        } else {
            prompt_arr.push(serde_json::json!({
                "hooks": [{ "type": "command", "command": recall_script }]
            }));
            messages.push("UserPromptSubmit (auto-recall) added".into());
        }
    }

    Ok((root, messages))
}


/// True if any hook entry (or its nested hooks) has a command containing `needle`.
fn entries_mention(entries: &[serde_json::Value], needle: &str) -> bool {
    entries.iter().any(|e| {
        e.get("hooks")
            .and_then(|h| h.as_array())
            .map(|arr| {
                arr.iter().any(|h| {
                    h.get("command")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.contains(needle))
                })
            })
            .unwrap_or(false)
    })
}

/// One Mimir slash command: name carries the `m-` prefix (collision safety
/// with users' own commands), `allowed` is Claude-Code-only tool pre-approval.
struct SlashCmd {
    name: &'static str,
    desc: &'static str,
    body: &'static str,
    allowed: Option<&'static str>,
}

/// Every `/m-*` slash command Mimir ships. `{args}` becomes the app's own
/// argument placeholder at render time.
const SLASH_COMMANDS: &[SlashCmd] = &[
    SlashCmd {
        name: "m-graph",
        desc: "Open the interactive Mimir graph visualization (current project)",
        body: "Run `mimir graph viz --open {args}` with your shell tool from the current \
            project root, then report the output path it prints. If it fails with \
            \"not inside a project\", relay the suggestion in the error: it needs a \
            project root (.git/.hg/.svn/.jj), or `touch .mimir` to mark one.",
        allowed: Some("Bash(mimir graph viz:*)"),
    },
    SlashCmd {
        name: "m-stats",
        desc: "Open the Mimir stats dashboard (memories, docs, code, learning)",
        body: "Run `mimir dashboard --open {args}` with your shell tool, then report the \
            output path it prints.",
        allowed: Some("Bash(mimir dashboard:*)"),
    },
    SlashCmd {
        name: "m-report",
        desc: "Mimir activity report: day / week / month / year / all-time",
        body: "Run `mimir report` with your shell tool and show its complete output \
            verbatim in a code block. Do not summarize or reformat the table.",
        allowed: Some("Bash(mimir report:*)"),
    },
    SlashCmd {
        name: "m-savings",
        desc: "Mimir token-savings report (outline/peek/command-filter/proxy)",
        body: "Run `mimir savings` with your shell tool and show its complete output \
            verbatim. It reports tokens saved today/week/month/all-time and by source.",
        allowed: Some("Bash(mimir savings:*)"),
    },
    SlashCmd {
        name: "m-scan",
        desc: "Auto-link Mimir memories to the code symbols they mention",
        body: "Run `mimir link --scan` with your shell tool from the current project \
            root and show its output. If links were created, suggest /m-graph to see \
            the new memory-to-code connections.",
        allowed: Some("Bash(mimir link:*)"),
    },
    SlashCmd {
        name: "m-recall",
        desc: "Search Mimir memory (memories, docs, code)",
        body: "Run `mimir recall {args}` with your shell tool and show the results. \
            If a hit looks like exactly what the user needs, also run \
            `mimir get <id>` on it and show the full body.",
        allowed: Some("Bash(mimir recall:*), Bash(mimir get:*)"),
    },
    SlashCmd {
        name: "m-remember",
        desc: "Save a memory to Mimir",
        body: "Store this in Mimir: {args}\n\nUse the mimir remember MCP tool (or \
            `mimir remember` via shell). Pick the fitting type (gotcha / decision / \
            insight / idea / note / person) and concise tags. If it is about specific \
            code, pass `link` with the symbol name. Confirm what was stored.",
        allowed: Some("mcp__mimir__remember, mcp__mimir__recall, Bash(mimir remember:*)"),
    },
    SlashCmd {
        name: "m-impact",
        desc: "Blast radius of the current uncommitted changes (Mimir code graph)",
        body: "Run `mimir graph impact $(git diff --name-only)` with your shell tool \
            from the current project root and show the affected symbols. If the diff \
            is empty, say there are no uncommitted changes to analyze.",
        allowed: Some("Bash(mimir graph impact:*)"),
    },
    SlashCmd {
        name: "m-doctor",
        desc: "Mimir health check (database, search index, models)",
        body: "Run `mimir doctor` and `mimir status` with your shell tool and show \
            both outputs verbatim. If any check is not ok, explain what it means and \
            how to fix it.",
        allowed: Some("Bash(mimir doctor:*), Bash(mimir status:*)"),
    },
];

/// Installed only when `[sync]` is enabled (re-run `mimir init` after enabling).
const SYNC_SLASH_COMMANDS: &[SlashCmd] = &[SlashCmd {
    name: "m-sync",
    desc: "Sync Mimir memories with the central store",
    body: "Run `mimir sync` with your shell tool and show the push/pull summary. \
        If it reports an auth or connection error, check MIMIR_SYNC_TOKEN and the \
        [sync] endpoint/dir in the Mimir config.",
    allowed: Some("Bash(mimir sync:*)"),
}];

/// Install the `/m-*` slash commands for the agent CLIs that support
/// user-level custom commands. Installed only for apps already present on
/// the machine; existing files are never overwritten (user edits win).
/// Re-running `mimir init` after an upgrade refreshes missing files.
fn install_agent_commands(config: &Config) {
    // An isolated instance (tests, scratch homes) must not touch the user's
    // agent configs — MIMIR_HOME means "everything under one directory".
    if std::env::var_os("MIMIR_HOME").is_some() {
        return;
    }
    // /m-sync is only useful (and only installed) when sync is enabled.
    let extra: &[SlashCmd] = if config.sync.enabled() {
        SYNC_SLASH_COMMANDS
    } else {
        &[]
    };

    let md = |cmd: &SlashCmd, with_allowed: bool| {
        let allowed = match (with_allowed, cmd.allowed) {
            (true, Some(a)) => format!("allowed-tools: {a}\n"),
            _ => String::new(),
        };
        format!(
            "---\ndescription: {}\n{allowed}---\n\n{}\n",
            cmd.desc,
            cmd.body.replace("{args}", "$ARGUMENTS")
        )
    };
    let toml = |cmd: &SlashCmd| {
        format!(
            "description = \"{}\"\nprompt = \"\"\"\n{}\n\"\"\"\n",
            cmd.desc,
            cmd.body.replace("{args}", "{{args}}")
        )
    };

    let Some(base) = directories::BaseDirs::new() else {
        return;
    };
    let home = base.home_dir();

    // (app, detect dir, target dir, file ext, claude-style allowed-tools?)
    const APPS: &[(&str, &str, &str, &str, bool)] = &[
        ("claude", ".claude", ".claude/commands", "md", true),
        ("codex", ".codex", ".codex/prompts", "md", false),
        (
            "opencode",
            ".config/opencode",
            ".config/opencode/command",
            "md",
            false,
        ),
        ("gemini", ".gemini", ".gemini/commands", "toml", false),
        ("cursor", ".cursor", ".cursor/commands", "md", false),
    ];

    let mut installed: Vec<String> = Vec::new();
    let mut detected = 0usize;
    for (app, detect, target, ext, with_allowed) in APPS {
        if !home.join(detect).is_dir() {
            continue;
        }
        detected += 1;
        let dir = home.join(target);
        if std::fs::create_dir_all(&dir).is_err() {
            continue;
        }
        let mut wrote = Vec::new();
        for cmd in SLASH_COMMANDS.iter().chain(extra) {
            let content = if *ext == "toml" {
                toml(cmd)
            } else {
                md(cmd, *with_allowed)
            };
            let path = dir.join(format!("{}.{ext}", cmd.name));
            if !path.exists() && std::fs::write(&path, content).is_ok() {
                wrote.push(format!("/{}", cmd.name));
            }
        }
        if !wrote.is_empty() {
            installed.push(format!("{app} ({})", wrote.join(" ")));
        }
    }
    // Always say what happened — a silent installer is undiagnosable
    // (a stale release binary once looked identical to "no agents found").
    if !installed.is_empty() {
        println!("agents  slash commands installed: {}", installed.join(", "));
    } else if detected == 0 {
        println!(
            "agents  no agent CLI config dirs found (~/.claude, ~/.codex, \
             ~/.config/opencode, ~/.gemini, ~/.cursor) — slash commands not installed"
        );
    } else {
        println!("agents  slash commands already present (nothing new to install)");
    }
}

/// Embed pending content; --fetch additionally allows the model download,
/// --rerank (with --fetch) also downloads the reranker.
pub fn embed(fetch: bool, rerank: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    if mimir.ensure_embedder(fetch).is_none() {
        bail!("embedding model unavailable; run `mimir embed --fetch` (or `mimir init`) to download it");
    }
    if rerank {
        if mimir.ensure_reranker(fetch).is_none() {
            bail!("reranker model unavailable; run `mimir embed --fetch --rerank` to download it");
        }
        println!("reranker {} ready", mimir.config.rerank.model);
    }
    let n = mimir.embed_pending()?;
    println!("embedded {n} node(s)");
    Ok(())
}

/// Count tokens in stdin (or the given text) with Mimir's bundled tokenizer —
/// the same counter the savings ledger uses, handy for measuring/benchmarking.
pub fn tokens(text: Vec<String>) -> Result<()> {
    let input = if text.is_empty() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        text.join(" ")
    };
    println!("{}", mimir_core::tokens::count(&input));
    Ok(())
}

pub fn status(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let counts = store::count_by_kind(&mimir.conn)?;
    let db_size = std::fs::metadata(&mimir.paths.db_file)
        .with_context(|| format!("stat {}", mimir.paths.db_file.display()))?
        .len();
    let (project, detection) = mimir.detect_project(&std::env::current_dir()?)?;
    let via = match &detection {
        mimir_core::scope::Detection::Found { via, .. } => Some(mimir_core::scope::via_label(via)),
        mimir_core::scope::Detection::NotFound { .. } => None,
    };

    if json {
        let counts_json: serde_json::Map<String, serde_json::Value> = counts
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v)))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "db": mimir.paths.db_file,
                "db_bytes": db_size,
                "project": project.as_ref().and_then(|p| p.title.clone()),
                "project_path": project.as_ref().and_then(|p| p.path.clone()),
                "scope": if project.is_some() { "project" } else { "global" },
                "detected_via": via,
                "counts": counts_json,
            })
        );
        return Ok(());
    }

    match (&project, &detection) {
        (Some(p), _) => println!(
            "project {} ({})  [via: {}]",
            p.title.as_deref().unwrap_or("?"),
            p.path.as_deref().unwrap_or("?"),
            via.unwrap_or("?")
        ),
        (None, mimir_core::scope::Detection::NotFound { from }) => println!(
            "project (none) — no git root or project marker found above {}; \
             using global scope (touch .mimir here to make it a project)",
            from.display()
        ),
        (None, _) => println!("project (none — global scope)"),
    }
    if counts.is_empty() {
        println!("store   empty");
    } else {
        let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
        println!("store   {}", summary.join(", "));
    }
    println!(
        "db      {} ({} KB)",
        mimir.paths.db_file.display(),
        db_size / 1024
    );
    let sc = &mimir.config.sync;
    if sc.enabled() {
        let push = mimir_core::replicate::get_watermark(&mimir.conn, "last_push").unwrap_or(0);
        let pull = mimir_core::replicate::get_watermark(&mimir.conn, "last_pull").unwrap_or(0);
        let cadence = if sc.auto {
            format!("auto every {} min", sc.interval_mins)
        } else {
            "manual".into()
        };
        println!(
            "sync    {} {} ({cadence}); local watermarks push={push} pull={pull}",
            sc.mode, sc.endpoint
        );
    } else {
        println!("sync    off (local store only)");
    }
    Ok(())
}

pub fn doctor() -> Result<()> {
    let paths = Paths::resolve()?;
    let mut failures = 0;

    let check = |name: &str, ok: bool, detail: String, failures: &mut i32| {
        let mark = if ok { "ok " } else { "FAIL" };
        if !ok {
            *failures += 1;
        }
        println!("{mark}  {name}: {detail}");
    };

    match db::open(&paths.db_file) {
        Ok(conn) => {
            check(
                "db",
                true,
                paths.db_file.display().to_string(),
                &mut failures,
            );
            let integrity: String = conn
                .query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .unwrap_or_else(|e| format!("error: {e}"));
            check("integrity", integrity == "ok", integrity, &mut failures);
            let fts = conn
                .prepare("SELECT count(*) FROM node_fts")
                .and_then(|mut s| s.query_row([], |r| r.get::<_, i64>(0)));
            check(
                "fts5",
                fts.is_ok(),
                fts.map(|n| format!("{n} rows indexed"))
                    .unwrap_or_else(|e| e.to_string()),
                &mut failures,
            );
        }
        Err(e) => check("db", false, e.to_string(), &mut failures),
    }

    println!(
        "ok   gpu: {}",
        mimir_core::embed::gpu_backend()
            .unwrap_or("not compiled in (CPU; rebuild with --features gpu-webgpu or gpu-cuda)")
    );

    let model_present = paths.models_dir.exists()
        && std::fs::read_dir(&paths.models_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
    check(
        "model",
        true, // informational until embeddings land; BM25-only is a valid state
        if model_present {
            format!("present at {}", paths.models_dir.display())
        } else {
            "not downloaded (search is BM25-only until `mimir init` fetches it)".into()
        },
        &mut failures,
    );

    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    Ok(())
}

// ---------- memory verbs ----------

#[allow(clippy::too_many_arguments)]
pub fn remember(
    json: bool,
    text: String,
    mtype: &str,
    tags: Vec<String>,
    global: bool,
    force: bool,
    link_ref: Option<String>,
) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let mtype: MemoryType = mtype.parse()?;
    let project = if global {
        None
    } else {
        mimir.project_for_cwd(&std::env::current_dir()?)?
    };
    let outcome = memory::remember(
        &mimir.conn,
        Remember {
            text,
            mtype,
            tags,
            project_id: project.as_ref().map(|p| p.id),
            force,
        },
    )?;
    let projects = store::project_titles(&mimir.conn)?;
    let snippet = mimir.config.output.snippet_chars;
    match outcome {
        RememberOutcome::Created(node) => {
            if json {
                println!("{}", node_json(&node, &projects));
            } else {
                println!("{}", line(&node, &projects, snippet));
            }
            if let Some(r) = link_ref {
                let target = store::resolve_ref(&mimir.conn, &r)?;
                store::link(&mimir.conn, node.id, target.id, Rel::Relates, 1.0)?;
                println!("linked → {}", line(&target, &projects, 0));
            }
            // Keep semantic recall fresh; harmless no-op without a model.
            if let Err(err) = mimir.embed_pending() {
                tracing::warn!(%err, "embedding new memory failed");
            }
            Ok(())
        }
        RememberOutcome::Duplicate(existing) => bail!(
            "refused: near-duplicate of\n  {}\nuse --force to store anyway",
            line(&existing, &projects, snippet)
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn recall(
    json: bool,
    text: String,
    kind: &str,
    global: bool,
    all: bool,
    since: Option<String>,
    limit: Option<usize>,
    full: bool,
    rerank: bool,
    linked: bool,
    min_score: Option<f64>,
    include_superseded: bool,
) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let query = SearchQuery {
        scope: read_scope(&mimir, global, all)?,
        kinds: parse_kind_filter(kind)?,
        since: since.map(|s| parse_since(&s)).transpose()?,
        limit: limit.unwrap_or(mimir.config.output.default_limit),
        strength_alpha: mimir.config.scoring.strength_alpha,
        recency_alpha: mimir.config.scoring.recency_alpha,
        type_prior_alpha: mimir.config.scoring.type_prior_alpha,
        code_damp: mimir.config.scoring.code_damp,
        include_superseded,
        text,
    };
    let mut hits = mimir.search_with(&query, rerank)?;
    if let Some(min) = min_score {
        hits.retain(|hit| hit.score >= min);
    }

    let query_hash = blake3::hash(query.text.as_bytes());
    let shown: Vec<(i64, i64, f64)> = hits
        .iter()
        .enumerate()
        .map(|(rank, hit)| (hit.node.id, rank as i64, hit.score))
        .collect();
    store::record_shown(&mimir.conn, query_hash.as_bytes(), &shown)?;
    if let Some(report) = mimir_core::consolidate::maybe_auto(
        &mimir.conn,
        &mimir.config.embedding.model,
        &mimir.config.consolidate.auto,
    ) {
        if !report.is_empty() {
            eprintln!(
                "(consolidated: {} superseded, {} distilled, {} archived)",
                report.superseded, report.distilled, report.archived
            );
        }
    }

    let projects = store::project_titles(&mimir.conn)?;
    if hits.is_empty() && !json {
        println!("no results");
        return Ok(());
    }
    for hit in &hits {
        if json {
            let mut value = node_json(&hit.node, &projects);
            value["score"] = serde_json::json!(hit.score);
            println!("{value}");
        } else if full {
            print_full(&hit.node, &mimir, &projects)?;
            println!();
        } else {
            println!(
                "{}",
                line(&hit.node, &projects, mimir.config.output.snippet_chars)
            );
        }
        if linked && !json {
            for edge in store::edges_of(&mimir.conn, hit.node.id)?.iter().take(4) {
                let other_id = if edge.src == hit.node.id {
                    edge.dst
                } else {
                    edge.src
                };
                let Ok(other) = store::get_node(&mimir.conn, other_id) else {
                    continue;
                };
                let l = match other.kind {
                    Kind::Symbol => mimir_graph::symbol_line(&other),
                    _ => line(&other, &projects, 60),
                };
                println!("  ~{} {}", edge.rel, l);
            }
        }
    }
    Ok(())
}

/// Print at most one relevant memory for `prompt`, or nothing if none
/// clears the relevance floor. This is the COLD fallback path: it opens a
/// fresh `Mimir` (ONNX load + full matrix rebuild) every invocation, so the
/// hook script (`MIMIR_RECALL_SH`) tries the warm `/inject` HTTP endpoint
/// first (see `mcp.rs::inject_router`, only live while `mimir mcp --http`
/// is running) and only falls back to this command when that's
/// unreachable. All relevance-floor/formatting/budget logic lives in
/// `mimir_core::inject::compute` — single-sourced so the warm and cold
/// paths can never disagree.
pub fn recall_inject(prompt: String) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let scope = read_scope(&mimir, false, false)?;
    if let Some(text) = mimir_core::inject::compute(&mut mimir, &prompt, scope)? {
        println!("{text}");
    }
    Ok(())
}

pub fn get(json: bool, refs: Vec<String>) -> Result<()> {
    let mimir = Mimir::open()?;
    let projects = store::project_titles(&mimir.conn)?;
    for (i, r) in refs.iter().enumerate() {
        if i > 0 && !json {
            println!();
        }
        if let Some(slice) = mimir_core::index::file_slice(&mimir.conn, r)? {
            println!("{slice}");
            continue;
        }
        let node = store::resolve_ref(&mimir.conn, r)?;
        mimir_core::learn::record_opened(&mimir.conn, node.id)?;
        if json {
            println!("{}", node_json(&node, &projects));
        } else {
            print_full(&node, &mimir, &projects)?;
        }
    }
    Ok(())
}

pub fn list(
    json: bool,
    mtype: Option<String>,
    tag: Option<String>,
    global: bool,
    all: bool,
    limit: usize,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let scope = read_scope(&mimir, global, all)?;
    let mtype = mtype.map(|t| t.parse::<MemoryType>()).transpose()?;
    let nodes = memory::list(&mimir.conn, scope, mtype, tag.as_deref(), limit)?;
    let projects = store::project_titles(&mimir.conn)?;
    if nodes.is_empty() && !json {
        println!("no memories");
        return Ok(());
    }
    for node in &nodes {
        if json {
            println!("{}", node_json(node, &projects));
        } else {
            println!(
                "{}",
                line(node, &projects, mimir.config.output.snippet_chars)
            );
        }
    }
    Ok(())
}

pub fn mark(reference: &str, useful: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = resolve_any(&mimir, reference)?;
    let strength = mimir_core::learn::apply_mark(&mimir.conn, node.id, useful)?;
    println!(
        "{} {} → strength {strength:.2}",
        short_uid(node.kind, &node.uid),
        if useful { "useful" } else { "noise" },
    );
    Ok(())
}

pub fn consolidate(dry_run: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let report =
        mimir_core::consolidate::consolidate(&mimir.conn, &mimir.config.embedding.model, dry_run)?;
    print_consolidate_report(&report, dry_run);
    Ok(())
}

fn print_consolidate_report(report: &mimir_core::consolidate::Report, dry_run: bool) {
    let prefix = if dry_run { "would " } else { "" };
    if report.is_empty() {
        println!("nothing to consolidate");
        return;
    }
    if report.superseded > 0 {
        println!("{prefix}supersede {} near-duplicate(s)", report.superseded);
    }
    if report.distilled > 0 {
        println!(
            "{prefix}distill {} cluster(s) into summaries",
            report.distilled
        );
    }
    if report.archived > 0 {
        println!("{prefix}archive {} decayed memorie(s)", report.archived);
    }
    for (a, b) in &report.contradictions {
        println!("possible contradiction (review by hand):\n  {a}\n  {b}");
    }
}

pub fn forget(reference: &str, hard: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = store::resolve_ref(&mimir.conn, reference)?;
    if hard {
        store::hard_delete(&mimir.conn, node.id)?;
    } else {
        store::soft_delete(&mimir.conn, node.id)?;
    }
    println!(
        "forgot {} {}{}",
        short_uid(node.kind, &node.uid),
        node.title.as_deref().unwrap_or(""),
        if hard { " (permanently)" } else { "" }
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn edit(
    json: bool,
    reference: &str,
    text: String,
    title: Option<String>,
    mtype: Option<String>,
    tags: Option<Vec<String>>,
    pin: Option<bool>,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let node = store::resolve_ref(&mimir.conn, reference)?;
    let mtype = mtype.map(|t| t.parse::<MemoryType>()).transpose()?;
    let edit = memory::Edit {
        text: if text.is_empty() { None } else { Some(text) },
        title,
        mtype,
        tags,
    };
    if edit.text.is_none()
        && edit.title.is_none()
        && edit.mtype.is_none()
        && edit.tags.is_none()
        && pin.is_none()
    {
        bail!("nothing to change: pass TEXT, --title, --type, --tags, or --pin/--unpin");
    }
    if let Some(pin) = pin {
        store::set_pinned(&mimir.conn, node.id, pin)?;
    }
    let updated = memory::edit(&mimir.conn, node.id, edit)?;
    let projects = store::project_titles(&mimir.conn)?;
    if json {
        println!("{}", node_json(&updated, &projects));
    } else {
        println!(
            "{}",
            line(&updated, &projects, mimir.config.output.snippet_chars)
        );
    }
    Ok(())
}

pub fn link(a: &str, b: &str, rel: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let rel: Rel = rel.parse()?;
    let src = resolve_any(&mimir, a)?;
    let dst = resolve_any(&mimir, b)?;
    store::link(&mimir.conn, src.id, dst.id, rel, 1.0)?;
    println!(
        "{} —{rel}→ {}",
        short_uid(src.kind, &src.uid),
        short_uid(dst.kind, &dst.uid)
    );
    Ok(())
}

/// Mark OLD as superseded by NEW: OLD stops surfacing in recall (kept as
/// history) and a `supersedes` edge is recorded.
pub fn supersede(old: &str, by: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let old = resolve_any(&mimir, old)?;
    let new = resolve_any(&mimir, by)?;
    store::set_superseded(&mimir.conn, old.id, new.id)?;
    store::link(&mimir.conn, new.id, old.id, Rel::Supersedes, 1.0)?;
    println!(
        "{} superseded by {}",
        short_uid(old.kind, &old.uid),
        short_uid(new.kind, &new.uid)
    );
    Ok(())
}

/// Auto-link memories to the code symbols their text literally mentions
/// (current project + global memories vs the current project's graph).
/// Precision-first: only code-shaped names (snake_case, ::path, CamelCase)
/// or backticked mentions link, on word boundaries; ambiguous names that
/// resolve to many symbols are skipped. Idempotent: existing edges are kept.
pub fn link_scan(dry_run: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = mimir
        .project_for_cwd(&std::env::current_dir()?)?
        .context("not inside a project (the scan links memories to this project's symbols)")?;

    // name → symbol nodes carrying it (bare name from meta, fallback title)
    let mut by_name: HashMap<String, Vec<(i64, String, String)>> = HashMap::new();
    {
        let mut stmt = mimir.conn.prepare(
            "SELECT id, uid, COALESCE(json_extract(meta,'$.name'), title) FROM node
             WHERE kind='symbol' AND project_id=?1 AND deleted_at IS NULL",
        )?;
        let mut rows = stmt.query([proj.id])?;
        while let Some(r) = rows.next()? {
            let (id, uid): (i64, String) = (r.get(0)?, r.get(1)?);
            let name: Option<String> = r.get(2)?;
            if let Some(name) = name {
                if name.len() >= 4 {
                    by_name
                        .entry(name)
                        .or_default()
                        .push((id, uid, String::new()));
                }
            }
        }
    }

    let memories: Vec<(i64, String, String)> = {
        let mut stmt = mimir.conn.prepare(
            "SELECT id, uid, COALESCE(title,'') || ' ' || COALESCE(body,'') FROM node
             WHERE kind='memory' AND deleted_at IS NULL AND superseded_by IS NULL
               AND (project_id=?1 OR project_id IS NULL)",
        )?;
        let rows = stmt.query_map([proj.id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };

    let mut existing: std::collections::HashSet<(i64, i64)> = Default::default();
    {
        let mut stmt = mimir.conn.prepare("SELECT src, dst FROM edge")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let (s, d): (i64, i64) = (r.get(0)?, r.get(1)?);
            existing.insert((s, d));
            existing.insert((d, s));
        }
    }

    let mut created = 0usize;
    for (mid, muid, text) in &memories {
        for (name, syms) in &by_name {
            if syms.len() > 3 {
                continue; // same name everywhere = ambiguous, skip
            }
            if !mentions_symbol(text, name) {
                continue;
            }
            for (sid, suid, _) in syms {
                if existing.contains(&(*mid, *sid)) {
                    continue;
                }
                if dry_run {
                    println!(
                        "would link m:{} —mentions→ {name} (c:{})",
                        tail(muid),
                        tail(suid)
                    );
                } else {
                    store::link(&mimir.conn, *mid, *sid, Rel::Mentions, 1.0)?;
                    println!("m:{} —mentions→ {name} (c:{})", tail(muid), tail(suid));
                }
                existing.insert((*mid, *sid));
                created += 1;
            }
        }
    }
    println!(
        "{} {created} link(s) ({} memories × {} distinct symbol names)",
        if dry_run { "would create" } else { "created" },
        memories.len(),
        by_name.len(),
    );
    Ok(())
}

fn tail(uid: &str) -> &str {
    &uid[uid.len().saturating_sub(6)..]
}

/// True when `text` mentions `name` as code: word-boundary matched AND
/// code-shaped (snake_case, ::path, or mixed-case with an uppercase letter
/// *after* the first — `MimirServer` yes, sentence-case `Pending` no).
/// Plain English words never match, even in backticks: `main` is usually
/// a git branch, not fn main. Precision beats recall here — a wrong link
/// pollutes recall, a missing one just waits for the next scan.
fn mentions_symbol(text: &str, name: &str) -> bool {
    let mixed_case = name.len() >= 6
        && name.chars().skip(1).any(|c| c.is_uppercase())
        && name.chars().any(|c| c.is_lowercase());
    let code_shaped = name.contains('_') || name.contains("::") || mixed_case;
    if !code_shaped {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = text[start..].find(name) {
        let i = start + pos;
        let j = i + name.len();
        let is_word = |c: Option<char>| c.map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
        let pre = text[..i].chars().next_back();
        let post = text[j..].chars().next();
        if !is_word(pre) && !is_word(post) {
            return true;
        }
        start = j;
    }
    false
}

// ---------- docs & index ----------

pub fn docs_add(path: &str, name: Option<String>, global: bool) -> Result<()> {
    add_collection_cmd(path, name, global, "docs")
}

/// Register + index a source-code collection: chunks function/method bodies
/// (not just signatures) on tree-sitter symbol boundaries for recall —
/// mirrors `docs add`, indexing source instead of markdown. Idea credit:
/// nworks3d's THOR fork of Mimir (see CHANGELOG.md).
pub fn code_add(path: &str, name: Option<String>, global: bool) -> Result<()> {
    add_collection_cmd(path, name, global, "code")
}

fn add_collection_cmd(path: &str, name: Option<String>, global: bool, kind: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let root = std::path::Path::new(path);
    let canonical = std::fs::canonicalize(root).with_context(|| format!("no such dir: {path}"))?;
    let name = name.unwrap_or_else(|| {
        canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    });
    let project = if global {
        None
    } else {
        mimir.project_for_cwd(&canonical)?
    };
    let coll = mimir_core::index::add_collection(
        &mimir.conn,
        &canonical,
        &name,
        project.as_ref().map(|p| p.id),
        kind,
    )?;
    println!(
        "{} {} {}",
        short_uid(coll.kind, &coll.uid),
        name,
        coll.path.as_deref().unwrap_or("?")
    );
    println!("run `mimir index` to scan it");
    Ok(())
}

pub fn docs_list(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let collections = mimir_core::index::list_collections(&mimir.conn)?;
    if collections.is_empty() && !json {
        println!("no collections (add one with `mimir docs add <path>`)");
        return Ok(());
    }
    let projects = store::project_titles(&mimir.conn)?;
    for coll in collections {
        let (files, chunks) = mimir_core::index::collection_stats(&mimir.conn, coll.id)?;
        if json {
            let mut v = node_json(&coll, &projects);
            v["files"] = serde_json::json!(files);
            v["chunks"] = serde_json::json!(chunks);
            println!("{v}");
        } else {
            let kind = coll
                .meta
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("docs");
            println!(
                "{} [{kind}] {} {} ({files} files, {chunks} chunks)",
                short_uid(coll.kind, &coll.uid),
                coll.title.as_deref().unwrap_or("?"),
                coll.path.as_deref().unwrap_or("?"),
            );
        }
    }
    Ok(())
}

pub fn docs_remove(name: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let coll = mimir_core::index::find_collection(&mimir.conn, name)?;
    mimir_core::index::remove_collection(&mimir.conn, coll.id)?;
    println!("removed {}", coll.title.as_deref().unwrap_or(name));
    Ok(())
}

pub fn docs_note(target: &str, text: String) -> Result<()> {
    let mimir = Mimir::open()?;
    let target_node = mimir_core::index::find_collection(&mimir.conn, target)
        .or_else(|_| store::resolve_ref(&mimir.conn, target))?;
    let note = mimir_core::index::annotate(&mimir.conn, &target_node, &text)?;
    println!(
        "{} describes {} {}",
        short_uid(note.kind, &note.uid),
        short_uid(target_node.kind, &target_node.uid),
        target_node.title.as_deref().unwrap_or("")
    );
    Ok(())
}

pub fn index(name: Option<String>) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let results = match name {
        Some(n) => {
            let coll = mimir_core::index::find_collection(&mimir.conn, &n)?;
            let stats = mimir_core::index::index_collection(&mut mimir.conn, &coll)?;
            vec![(coll.title.unwrap_or(n), stats)]
        }
        None => mimir_core::index::index_all(&mut mimir.conn)?,
    };
    if results.is_empty() {
        println!("no collections (add one with `mimir docs add <path>`)");
        return Ok(());
    }
    for (name, s) in results {
        println!(
            "{name}: {} files seen, {} indexed ({} chunks), {} unchanged, {} removed",
            s.seen, s.indexed, s.chunks, s.unchanged, s.removed
        );
    }
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

// ---------- import / export ----------

pub fn import_openbrain(file: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let text = if file == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read_to_string(file).with_context(|| format!("read {file}"))?
    };
    let stats = mimir_core::import::openbrain(&mimir.conn, &text)?;
    finish_import(&mut mimir, stats)
}

pub fn import_claude_memory(dir: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let stats = mimir_core::import::claude_memory(&mimir.conn, std::path::Path::new(dir))?;
    finish_import(&mut mimir, stats)
}

pub fn import_qmd(file: Option<String>) -> Result<()> {
    let mimir = Mimir::open()?;
    let path = match file {
        Some(f) => std::path::PathBuf::from(f),
        None => directories::BaseDirs::new()
            .context("cannot resolve home")?
            .home_dir()
            .join(".config/qmd/index.yml"),
    };
    let yml = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let collections = mimir_core::import::qmd_collections(&yml);
    if collections.is_empty() {
        bail!("no collections found in {}", path.display());
    }
    for (name, root) in &collections {
        let root_path = std::path::Path::new(root);
        if !root_path.is_dir() {
            eprintln!("skipping {name}: {root} is not a directory");
            continue;
        }
        let coll = mimir_core::index::add_collection(&mimir.conn, root_path, name, None, "docs")?;
        println!(
            "registered {} {} {}",
            short_uid(coll.kind, &coll.uid),
            name,
            root
        );
    }
    println!("run `mimir index` to scan them");
    Ok(())
}

fn finish_import(mimir: &mut Mimir, stats: mimir_core::import::ImportStats) -> Result<()> {
    println!(
        "imported {} memorie(s), skipped {} duplicate(s)",
        stats.imported, stats.skipped_duplicates
    );
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

pub fn export() -> Result<()> {
    let mimir = Mimir::open()?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let n = mimir_core::import::export_jsonl(&mimir.conn, &mut lock)?;
    eprintln!("exported {n} line(s)");
    Ok(())
}

// ---------- helpers ----------

/// Resolve ids first, then symbol names within the current project — so
/// `mimir link m:ABC123 resolve_ref --rel about` just works.
fn resolve_any(mimir: &Mimir, reference: &str) -> Result<Node> {
    match store::resolve_ref(&mimir.conn, reference) {
        Ok(node) => Ok(node),
        Err(id_err) => {
            if let Some(proj) = mimir.project_for_cwd(&std::env::current_dir()?)? {
                if let Ok(sym) = mimir_graph::resolve_symbol(&mimir.conn, proj.id, reference) {
                    return Ok(sym);
                }
            }
            Err(id_err.into())
        }
    }
}

/// Scope for read operations. Inside a project: that project + global.
/// Outside: everything (reads want breadth; -g narrows to global-only).
fn read_scope(mimir: &Mimir, global: bool, all: bool) -> Result<Scope> {
    if all {
        return Ok(Scope::All);
    }
    if global {
        return Ok(Scope::Global);
    }
    Ok(match mimir.project_for_cwd(&std::env::current_dir()?)? {
        Some(p) => Scope::Project(p.id),
        None => Scope::All,
    })
}

fn parse_kind_filter(kind: &str) -> Result<Vec<Kind>> {
    Ok(match kind {
        // No kind filter — deliberately includes CodeChunk. The point of
        // indexing function bodies (not just signatures) is that a plain
        // `recall` finds them without the caller knowing to ask for
        // `--kind code`; ScoringConfig::code_damp keeps code from drowning
        // out memories given its much larger corpus share.
        "all" => vec![],
        "memory" => vec![Kind::Memory],
        "doc" => vec![Kind::File, Kind::Chunk, Kind::Annotation],
        // Symbol = signature/doc only; CodeChunk = actual body/content text
        // (see chunker::chunk_source). Both belong under `code`.
        "code" => vec![Kind::Symbol, Kind::CodeChunk],
        other => bail!("unknown --kind '{other}' (use all|memory|doc|code)"),
    })
}

/// "12h" | "7d" | "2w" | "3m" | "1y" → unix cutoff.
fn parse_since(s: &str) -> Result<i64> {
    // Split on the last CHARACTER, not the last byte — a multibyte unit
    // (e.g. "5µ") would otherwise slice mid-codepoint and panic.
    let split = s.char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
    let (num, unit) = s.split_at(split);
    let n: i64 = num
        .parse()
        .with_context(|| format!("bad --since '{s}' (use e.g. 12h, 7d, 2w, 3m, 1y)"))?;
    let secs = match unit {
        "h" => 3_600,
        "d" => 86_400,
        "w" => 604_800,
        "m" => 2_592_000,
        "y" => 31_536_000,
        _ => bail!("bad --since unit '{unit}' (use h, d, w, m, y)"),
    };
    Ok(now_unix() - n * secs)
}

fn line(node: &Node, projects: &HashMap<i64, String>, snippet_chars: usize) -> String {
    let project = node
        .project_id
        .and_then(|id| projects.get(&id))
        .map(String::as_str);
    agent_line(node, project, snippet_chars)
}

fn print_full(node: &Node, mimir: &Mimir, projects: &HashMap<i64, String>) -> Result<()> {
    println!(
        "{}",
        mimir_core::format::full_record(&mimir.conn, node, projects)?
    );
    Ok(())
}

fn node_json(node: &Node, projects: &HashMap<i64, String>) -> serde_json::Value {
    serde_json::json!({
        "id": short_uid(node.kind, &node.uid),
        "uid": node.uid,
        "kind": node.kind.as_str(),
        "type": node.subkind,
        "project": node.project_id.and_then(|id| projects.get(&id)),
        "title": node.title,
        "body": node.body,
        "path": node.path,
        "tags": node.tags(),
        "created_at": node.created_at,
        "updated_at": node.updated_at,
        "access_count": node.access_count,
        "strength": node.strength,
    })
}

#[cfg(test)]
mod since_tests {
    use super::parse_since;

    #[test]
    fn multibyte_unit_errors_not_panics() {
        // Regression: split_at on a byte offset panicked mid-codepoint.
        assert!(parse_since("5µ").is_err());
        assert!(parse_since("7€").is_err());
        assert!(parse_since("").is_err());
        assert!(parse_since("3d").is_ok());
    }
}

#[cfg(test)]
mod scan_tests {
    use super::mentions_symbol;

    #[test]
    fn snake_case_matches_on_word_boundaries() {
        assert!(mentions_symbol(
            "the record_opened path is the entry",
            "record_opened"
        ));
        assert!(!mentions_symbol("we prerecord_opened it", "record_opened"));
        assert!(mentions_symbol(
            "learn::record_opened is the single entry",
            "record_opened"
        ));
    }

    #[test]
    fn plain_words_never_match_even_backticked() {
        assert!(!mentions_symbol("we should update the docs", "update"));
        assert!(!mentions_symbol("we changed `update` semantics", "update"));
        assert!(!mentions_symbol("force push `main` to origin", "main"));
    }

    #[test]
    fn camel_case_matches_but_sentence_case_does_not() {
        assert!(mentions_symbol(
            "the MimirServer struct owns the router",
            "MimirServer"
        ));
        assert!(!mentions_symbol("nothing here", "MimirServer"));
        assert!(!mentions_symbol("Pending tasks for tomorrow", "Pending"));
    }
}

#[cfg(test)]
mod hooks_tests {
    use super::{merge_hook_settings, render_recall_script};

    /// A custom `config.hooks.inject_url` must land verbatim as the
    /// script's `MIMIR_INJECT_URL` fallback default — the actual install
    /// path (`install_hooks`) is only exercised end-to-end by the
    /// fake-$HOME e2e test (crates/mimir-cli/tests/e2e.rs), since it needs
    /// a real `~/.claude` dir and MIMIR_HOME must be *unset* for it to run
    /// at all; this covers the pure rendering logic in isolation.
    #[test]
    fn custom_inject_url_lands_in_generated_script() {
        let script = render_recall_script("http://10.0.0.5:9999/inject");
        assert!(
            script.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://10.0.0.5:9999/inject}""#),
            "custom URL missing from script:\n{script}"
        );
        // The env-var override placeholder syntax itself must survive
        // untouched — only the default inside it gets substituted.
        assert!(script.contains("${MIMIR_INJECT_URL:-"));
        assert!(!script.contains("__MIMIR_INJECT_URL_DEFAULT__"));
    }

    #[test]
    fn default_inject_url_matches_documented_port() {
        let script = render_recall_script("http://127.0.0.1:8077/inject");
        assert!(script.contains(r#"INJECT_URL="${MIMIR_INJECT_URL:-http://127.0.0.1:8077/inject}""#));
    }

    /// `auto_recall=false` (recall_script = None) must be byte-identical to
    /// pre-auto-recall behavior: no `UserPromptSubmit` key appears at all.
    #[test]
    fn no_recall_script_leaves_user_prompt_submit_untouched() {
        let (root, messages) =
            merge_hook_settings(serde_json::json!({}), "/path/mimir-rewrite.sh", None).unwrap();
        assert!(
            root["hooks"].get("UserPromptSubmit").is_none(),
            "auto_recall=false must not add hooks.UserPromptSubmit, got: {root}"
        );
        // The pre-existing hooks still get installed as before.
        assert!(root["hooks"]["SessionStart"].is_array());
        assert!(root["hooks"]["PreToolUse"].is_array());
        assert!(messages.iter().any(|m| m.contains("SessionStart")));
        assert!(messages.iter().any(|m| m.contains("PreToolUse")));
        assert!(!messages.iter().any(|m| m.contains("UserPromptSubmit")));
    }

    /// `auto_recall=true` adds exactly one `UserPromptSubmit` entry
    /// pointing at the recall script, and re-running is a no-op (idempotent).
    #[test]
    fn recall_script_adds_one_entry_idempotently() {
        let (root, messages) = merge_hook_settings(
            serde_json::json!({}),
            "/path/mimir-rewrite.sh",
            Some("/path/mimir-recall.sh"),
        )
        .unwrap();
        let entries = root["hooks"]["UserPromptSubmit"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(messages.iter().any(|m| m.contains("UserPromptSubmit")));

        // Re-run against the already-merged settings: still exactly one entry,
        // and the message says "already installed" instead of "added".
        let (root2, messages2) = merge_hook_settings(
            root,
            "/path/mimir-rewrite.sh",
            Some("/path/mimir-recall.sh"),
        )
        .unwrap();
        let entries2 = root2["hooks"]["UserPromptSubmit"].as_array().unwrap();
        assert_eq!(entries2.len(), 1, "re-running must not duplicate the entry");
        assert!(messages2
            .iter()
            .any(|m| m.contains("UserPromptSubmit already installed")));
    }
}

