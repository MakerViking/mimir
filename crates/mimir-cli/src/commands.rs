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
        match mimir_core::embed::Embedder::load(&paths, &config.embedding.model, true) {
            Ok(e) => println!("model   {} ready ({}-dim)", e.name, e.dim),
            Err(err) => {
                eprintln!("model   download failed ({err}); search is BM25-only until `mimir embed --fetch` succeeds")
            }
        }
    }
    println!();
    println!("Register the MCP server once, globally:");
    println!("  claude mcp add --scope user mimir -- mimir mcp");
    Ok(())
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
        store::touch_accessed(&mimir.conn, node.id)?;
        store::record_event(&mimir.conn, node.id, "opened", None, None, None)?;
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

pub fn edit(
    json: bool,
    reference: &str,
    text: String,
    title: Option<String>,
    mtype: Option<String>,
    tags: Option<Vec<String>>,
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
    if edit.text.is_none() && edit.title.is_none() && edit.mtype.is_none() && edit.tags.is_none() {
        bail!("nothing to change: pass TEXT, --title, --type, or --tags");
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
    let src = store::resolve_ref(&mimir.conn, a)?;
    let dst = store::resolve_ref(&mimir.conn, b)?;
    store::link(&mimir.conn, src.id, dst.id, rel, 1.0)?;
    println!(
        "{} —{rel}→ {}",
        short_uid(src.kind, &src.uid),
        short_uid(dst.kind, &dst.uid)
    );
    Ok(())
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

// ---------- helpers ----------

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
