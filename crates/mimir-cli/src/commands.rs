use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mimir_core::config::{Config, Paths};
use mimir_core::format::agent_line;
use mimir_core::memory::{self, Remember, RememberOutcome};
use mimir_core::model::{now_unix, short_uid, Kind, MemoryType, Node, Rel, Scope};
use mimir_core::search::SearchQuery;
use mimir_core::{db, store, Mimir};

pub fn init(no_model: bool) -> Result<()> {
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
    install_agent_commands();
    println!();
    println!("Register the MCP server once, globally:");
    println!("  claude mcp add --scope user mimir -- mimir mcp");
    Ok(())
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
            \"not inside a project\", tell the user to run it from inside a git repository.",
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

/// Install the `/m-*` slash commands for the agent CLIs that support
/// user-level custom commands. Installed only for apps already present on
/// the machine; existing files are never overwritten (user edits win).
/// Re-running `mimir init` after an upgrade refreshes missing files.
fn install_agent_commands() {
    // An isolated instance (tests, scratch homes) must not touch the user's
    // agent configs — MIMIR_HOME means "everything under one directory".
    if std::env::var_os("MIMIR_HOME").is_some() {
        return;
    }

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
    for (app, detect, target, ext, with_allowed) in APPS {
        if !home.join(detect).is_dir() {
            continue;
        }
        let dir = home.join(target);
        if std::fs::create_dir_all(&dir).is_err() {
            continue;
        }
        let mut wrote = Vec::new();
        for cmd in SLASH_COMMANDS {
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
    if !installed.is_empty() {
        println!("agents  slash commands installed: {}", installed.join(", "));
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

pub fn status(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let counts = store::count_by_kind(&mimir.conn)?;
    let db_size = std::fs::metadata(&mimir.paths.db_file)
        .map(|m| m.len())
        .unwrap_or(0);
    let project = mimir.project_for_cwd(&std::env::current_dir()?)?;

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
                "counts": counts_json,
            })
        );
        return Ok(());
    }

    match &project {
        Some(p) => println!(
            "project {} ({})",
            p.title.as_deref().unwrap_or("?"),
            p.path.as_deref().unwrap_or("?")
        ),
        None => println!("project (none — global scope)"),
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
) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let query = SearchQuery {
        scope: read_scope(&mimir, global, all)?,
        kinds: parse_kind_filter(kind)?,
        since: since.map(|s| parse_since(&s)).transpose()?,
        limit: limit.unwrap_or(mimir.config.output.default_limit),
        strength_alpha: mimir.config.scoring.strength_alpha,
        text,
    };
    let hits = mimir.search_with(&query, rerank)?;

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
            println!("{}", node_json(&hit.node, &projects));
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
            println!(
                "{} {} {} ({files} files, {chunks} chunks)",
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
        let coll = mimir_core::index::add_collection(&mimir.conn, root_path, name, None)?;
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
        "all" => vec![],
        "memory" => vec![Kind::Memory],
        "doc" => vec![Kind::File, Kind::Chunk, Kind::Annotation],
        "code" => vec![Kind::Symbol],
        other => bail!("unknown --kind '{other}' (use all|memory|doc|code)"),
    })
}

/// "12h" | "7d" | "2w" | "3m" | "1y" → unix cutoff.
fn parse_since(s: &str) -> Result<i64> {
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
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
