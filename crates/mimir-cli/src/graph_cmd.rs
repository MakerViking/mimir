//! `mimir graph …` — build/query the code graph of the current project.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mimir_core::model::{short_uid, Kind, NewNode, Node, Rel};
use mimir_core::{store, Mimir};
use mimir_graph::{resolve_symbol, symbol_line, CodeGraph};

/// Current project or a clear error (the graph is per-project by nature).
fn project(mimir: &Mimir) -> Result<Node> {
    mimir.project_for_cwd(&std::env::current_dir()?)?.context(
        "not inside a project. The code graph needs a project root — a dir with \
             a VCS marker (.git/.hg/.svn/.jj) or a build file (Cargo.toml/\
             package.json/pyproject.toml/go.mod/…); otherwise `touch .mimir` to \
             mark one.",
    )
}

/// Build the code graph on first use so queries "just work" in a freshly
/// detected project — no manual `graph build`. No-op once symbols exist.
/// Shared with `graph viz`.
pub(crate) fn ensure_graph(mimir: &mut Mimir, proj: &Node) -> Result<()> {
    let symbols: i64 = mimir.conn.query_row(
        "SELECT count(*) FROM node WHERE kind='symbol' AND project_id=?1 AND deleted_at IS NULL",
        [proj.id],
        |r| r.get(0),
    )?;
    if symbols > 0 {
        return Ok(());
    }
    let Some(root) = proj.path.clone() else {
        return Ok(());
    };
    eprintln!("building the code graph (first run for this project)…");
    match mimir_graph::update(&mut mimir.conn, proj, std::path::Path::new(&root)) {
        Ok(s) => eprintln!("  {} symbols, {} call edges", s.symbols, s.calls_resolved),
        Err(e) => eprintln!("  graph build skipped: {e}"),
    }
    let _ = mimir.embed_pending();
    Ok(())
}

/// Every project that actually has a graph. The rest cost a query and can
/// match nothing.
fn graphed_projects(mimir: &Mimir) -> Result<Vec<Node>> {
    let mut stmt = mimir.conn.prepare(&format!(
        "SELECT {} FROM node p WHERE p.kind='project' AND p.deleted_at IS NULL
           AND EXISTS (SELECT 1 FROM node s WHERE s.kind='symbol'
                       AND s.project_id=p.id AND s.deleted_at IS NULL)
         ORDER BY p.title",
        store::NODE_COLS
    ))?;
    let rows = stmt.query_map([], mimir_core::store::row_to_node)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The projects a query should run against: just this one, or every graphed
/// project. `--all-projects` exists because a symbol you are asking about
/// often lives in a repo you are not standing in — and call edges never
/// cross a project boundary, so each graph is searched on its own and the
/// results are labelled, never merged.
fn targets(mimir: &mut Mimir, all_projects: bool) -> Result<Vec<Node>> {
    if !all_projects {
        let proj = project(mimir)?;
        ensure_graph(mimir, &proj)?;
        return Ok(vec![proj]);
    }
    let all = graphed_projects(mimir)?;
    if all.is_empty() {
        bail!("no project has a code graph yet — run `mimir graph build` in one first");
    }
    Ok(all)
}

fn project_header(proj: &Node) -> String {
    format!("── {} ──", proj.title.as_deref().unwrap_or("(unnamed)"))
}

pub fn build() -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let root = std::path::PathBuf::from(proj.path.as_deref().context("project has no root")?);
    let started = std::time::Instant::now();
    let stats = mimir_graph::update(&mut mimir.conn, &proj, &root)?;
    println!(
        "{}: {} files seen, {} indexed, {} unchanged, {} removed — {} symbols, {} calls resolved ({} by receiver type, +{} heuristic), {} import edges [{:.1?}]",
        proj.title.as_deref().unwrap_or("project"),
        stats.files_seen,
        stats.files_indexed,
        stats.unchanged,
        stats.removed,
        stats.symbols,
        stats.calls_resolved,
        stats.calls_typed,
        stats.calls_heuristic,
        stats.imports,
        started.elapsed(),
    );
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

/// Report whether the graph still matches the working tree, and say so with
/// an exit code so a hook or CI job can act on it. Never writes: a check
/// that quietly rebuilt would pass every time and prove nothing.
pub fn check(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let root = std::path::PathBuf::from(proj.path.as_deref().context("project has no root")?);
    let drift = mimir_graph::check(&mimir.conn, &proj, &root)?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "project": proj.title,
                "stale": drift.is_stale(),
                "added": drift.added,
                "changed": drift.changed,
                "removed": drift.removed,
                "unchanged": drift.unchanged,
            })
        );
    } else if !drift.is_stale() {
        println!(
            "{}: graph is current ({} files)",
            proj.title.as_deref().unwrap_or("project"),
            drift.unchanged
        );
    } else {
        // Enough to act on without burying a big refactor's output.
        const SHOWN: usize = 10;
        for (label, paths) in [
            ("new", &drift.added),
            ("changed", &drift.changed),
            ("gone", &drift.removed),
        ] {
            for p in paths.iter().take(SHOWN) {
                println!("{label:>7}  {p}");
            }
            if paths.len() > SHOWN {
                println!("{:>7}  … and {} more", "", paths.len() - SHOWN);
            }
        }
    }
    if drift.is_stale() {
        bail!(
            "graph is {} file(s) behind the working tree — run `mimir graph build`",
            drift.stale_count()
        );
    }
    Ok(())
}

pub fn callers(reference: &str, depth: usize, all_projects: bool) -> Result<()> {
    walk_relation(reference, depth, true, all_projects)
}

pub fn calls(reference: &str, depth: usize, all_projects: bool) -> Result<()> {
    walk_relation(reference, depth, false, all_projects)
}

fn walk_relation(reference: &str, depth: usize, reverse: bool, all_projects: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let projects = targets(&mut mimir, all_projects)?;
    let multi = projects.len() > 1;
    let mut found = 0usize;

    for proj in &projects {
        // Across projects a name may be missing or ambiguous in most of
        // them; that is not an error, it is just the wrong repo.
        let sym = match resolve_symbol(&mimir.conn, proj.id, reference) {
            Ok(s) => s,
            Err(e) if multi => {
                if matches!(e, mimir_core::error::Error::AmbiguousRef(..)) {
                    eprintln!("{}: {e}", proj.title.as_deref().unwrap_or("(unnamed)"));
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        found += 1;
        let graph = CodeGraph::load(&mimir.conn, proj.id)?;
        let reached = if reverse {
            graph.callers(sym.id, depth)
        } else {
            graph.calls(sym.id, depth)
        };
        if multi {
            println!("{}", project_header(proj));
        }
        println!("{}", symbol_line(&sym));
        if reached.is_empty() {
            println!("no {} found", if reverse { "callers" } else { "callees" });
            continue;
        }
        for r in reached {
            let node = store::get_node(&mimir.conn, r.id)?;
            println!("{}{}", "  ".repeat(r.distance), symbol_line(&node));
        }
    }
    if found == 0 {
        bail!("no project has a symbol named `{reference}`");
    }
    Ok(())
}

/// Blast radius of changing the given files (e.g. `$(git diff --name-only)`).
pub fn impact(files: Vec<String>, depth: usize, all_projects: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let projects = targets(&mut mimir, all_projects)?;
    let multi = projects.len() > 1;
    let mut hit_any = false;

    for proj in &projects {
        let graph = CodeGraph::load(&mimir.conn, proj.id)?;
        let mut seeds: Vec<i64> = Vec::new();
        for f in &files {
            let matches = store::files_by_path_suffix(&mimir.conn, f.trim_start_matches("./"))?;
            for file in matches {
                // A suffix like `src/lib.rs` matches in half the repos on
                // the machine; only this project's files seed its graph.
                if file.project_id != Some(proj.id) {
                    continue;
                }
                seeds.push(file.id);
                if let Some(symbols) = graph.file_symbols.get(&file.id) {
                    seeds.extend(symbols);
                }
            }
        }
        if seeds.is_empty() {
            continue;
        }
        hit_any = true;
        let reached = graph.impact(&seeds, depth);
        if multi {
            println!("{}", project_header(proj));
        }
        if reached.is_empty() {
            println!("no impact outside the changed files");
            continue;
        }
        for r in &reached {
            let node = store::get_node(&mimir.conn, r.id)?;
            if node.kind == Kind::Symbol {
                println!("{}{}", "  ".repeat(r.distance - 1), symbol_line(&node));
            }
        }
        println!("({} affected symbols within {depth} hops)", reached.len());
    }
    if !hit_any {
        bail!("none of those paths are in the code graph (run `mimir graph build`?)");
    }
    Ok(())
}

/// Full record for a symbol: signature/doc, edges, linked memories.
pub fn node_info(reference: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    ensure_graph(&mut mimir, &proj)?;
    let sym = resolve_symbol(&mimir.conn, proj.id, reference)?;
    println!("{}", symbol_line(&sym));
    if let Some(body) = &sym.body {
        println!("{body}");
    }
    mimir_core::learn::record_opened(&mimir.conn, sym.id)?;
    for edge in store::edges_of(&mimir.conn, sym.id)? {
        let (arrow, other_id) = if edge.src == sym.id {
            ("→", edge.dst)
        } else {
            ("←", edge.src)
        };
        let Ok(other) = store::get_node(&mimir.conn, other_id) else {
            continue;
        };
        let line = match other.kind {
            Kind::Symbol => symbol_line(&other),
            _ => format!(
                "{} {}",
                short_uid(other.kind, &other.uid),
                other.title.as_deref().unwrap_or("")
            ),
        };
        println!("{arrow} {} {line}", edge.rel);
    }
    Ok(())
}

pub fn path(a: &str, b: &str, all_projects: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let projects = targets(&mut mimir, all_projects)?;
    let multi = projects.len() > 1;
    let mut found = 0usize;

    for proj in &projects {
        // A call path needs both ends in one graph — edges never cross a
        // project — so a project missing either symbol simply isn't it.
        // Standing in one project, though, "not found" is the useful answer.
        let (sa, sb) = match (
            resolve_symbol(&mimir.conn, proj.id, a),
            resolve_symbol(&mimir.conn, proj.id, b),
        ) {
            (Ok(sa), Ok(sb)) => (sa, sb),
            (Err(e), _) | (_, Err(e)) if !multi => return Err(e.into()),
            _ => continue,
        };
        found += 1;
        let graph = CodeGraph::load(&mimir.conn, proj.id)?;
        if multi {
            println!("{}", project_header(proj));
        }
        match graph.path(sa.id, sb.id) {
            Some(ids) => {
                for (i, id) in ids.iter().enumerate() {
                    let node = store::get_node(&mimir.conn, *id)?;
                    println!("{}{}", "  ".repeat(i), symbol_line(&node));
                }
            }
            None => println!(
                "no call path {} → {}",
                sa.title.as_deref().unwrap_or(a),
                sb.title.as_deref().unwrap_or(b)
            ),
        }
    }
    if found == 0 {
        bail!("no project has both `{a}` and `{b}`");
    }
    Ok(())
}

pub fn hubs(limit: usize) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    ensure_graph(&mut mimir, &proj)?;
    let graph = CodeGraph::load(&mimir.conn, proj.id)?;
    let hubs = graph.hubs(limit);
    if hubs.is_empty() {
        println!("no call edges yet (run `mimir graph build`)");
        return Ok(());
    }
    for (id, indeg) in hubs {
        let node = store::get_node(&mimir.conn, id)?;
        println!("↑{indeg:<4} {}", symbol_line(&node));
    }
    Ok(())
}

/// Label-propagation communities with heuristic names; --persist stores
/// them as community nodes with member_of edges.
pub fn communities(persist: bool, min_size: usize) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    ensure_graph(&mut mimir, &proj)?;
    let graph = CodeGraph::load(&mimir.conn, proj.id)?;
    let groups = graph.communities(min_size);
    if groups.is_empty() {
        println!("no communities of size >= {min_size}");
        return Ok(());
    }
    if persist {
        // Replace previous communities wholesale (derived data).
        mimir.conn.execute(
            "DELETE FROM node WHERE kind = 'community' AND project_id = ?1",
            [proj.id],
        )?;
    }
    for (i, group) in groups.iter().enumerate() {
        let members: Vec<Node> = group
            .iter()
            .filter_map(|id| store::get_node(&mimir.conn, *id).ok())
            .collect();
        let name = community_name(&members);
        println!("community {} ({} symbols): {name}", i + 1, group.len());
        for m in members.iter().take(5) {
            println!("  {}", symbol_line(m));
        }
        if members.len() > 5 {
            println!("  … and {} more", members.len() - 5);
        }
        if persist {
            let mut new = NewNode::new(Kind::Community);
            new.title = Some(name);
            new.project_id = Some(proj.id);
            new.body = Some(
                members
                    .iter()
                    .filter_map(|m| m.title.clone())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            let cnode = store::insert_node(&mimir.conn, new)?;
            for m in &members {
                store::link(&mimir.conn, m.id, cnode.id, Rel::MemberOf, 1.0)?;
            }
        }
    }
    if persist {
        println!("persisted {} community node(s)", groups.len());
    }
    Ok(())
}

/// Heuristic name: dominant directory + most frequent identifier tokens.
fn community_name(members: &[Node]) -> String {
    let mut dirs: HashMap<String, usize> = HashMap::new();
    let mut tokens: HashMap<String, usize> = HashMap::new();
    for m in members {
        if let Some(p) = &m.path {
            let dir = std::path::Path::new(p)
                .parent()
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default();
            *dirs.entry(dir).or_default() += 1;
        }
        if let Some(t) = &m.title {
            for tok in t
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| s.len() > 2)
            {
                *tokens.entry(tok.to_lowercase()).or_default() += 1;
            }
        }
    }
    let dir = dirs
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(d, _)| d)
        .unwrap_or_default();
    let mut toks: Vec<(String, usize)> = tokens.into_iter().collect();
    toks.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let theme: Vec<String> = toks.into_iter().take(3).map(|(t, _)| t).collect();
    if dir.is_empty() {
        theme.join("-")
    } else {
        format!("{dir} ({})", theme.join(", "))
    }
}
