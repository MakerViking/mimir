//! `mimir graph …` — build/query the code graph of the current project.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mimir_core::model::{short_uid, Kind, NewNode, Node, Rel};
use mimir_core::{store, Mimir};
use mimir_graph::{resolve_symbol, symbol_line, CodeGraph};

/// Current project or a clear error (the graph is per-project by nature).
fn project(mimir: &Mimir) -> Result<Node> {
    mimir
        .project_for_cwd(&std::env::current_dir()?)?
        .context("not inside a project (the code graph needs a git repository)")
}

pub fn build() -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let root = std::path::PathBuf::from(proj.path.as_deref().context("project has no root")?);
    let started = std::time::Instant::now();
    let stats = mimir_graph::update(&mut mimir.conn, &proj, &root)?;
    println!(
        "{}: {} files seen, {} indexed, {} unchanged, {} removed — {} symbols, {} calls resolved (+{} heuristic), {} import edges [{:.1?}]",
        proj.title.as_deref().unwrap_or("project"),
        stats.files_seen,
        stats.files_indexed,
        stats.unchanged,
        stats.removed,
        stats.symbols,
        stats.calls_resolved,
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

pub fn callers(reference: &str, depth: usize) -> Result<()> {
    walk_relation(reference, depth, true)
}

pub fn calls(reference: &str, depth: usize) -> Result<()> {
    walk_relation(reference, depth, false)
}

fn walk_relation(reference: &str, depth: usize, reverse: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let sym = resolve_symbol(&mimir.conn, proj.id, reference)?;
    let graph = CodeGraph::load(&mimir.conn, proj.id)?;
    let reached = if reverse {
        graph.callers(sym.id, depth)
    } else {
        graph.calls(sym.id, depth)
    };
    println!("{}", symbol_line(&sym));
    if reached.is_empty() {
        println!("no {} found", if reverse { "callers" } else { "callees" });
        return Ok(());
    }
    for r in reached {
        let node = store::get_node(&mimir.conn, r.id)?;
        println!("{}{}", "  ".repeat(r.distance), symbol_line(&node));
    }
    Ok(())
}

/// Blast radius of changing the given files (e.g. `$(git diff --name-only)`).
pub fn impact(files: Vec<String>, depth: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let graph = CodeGraph::load(&mimir.conn, proj.id)?;

    let mut seeds: Vec<i64> = Vec::new();
    for f in &files {
        let matches = store::files_by_path_suffix(&mimir.conn, f.trim_start_matches("./"))?;
        for file in matches {
            seeds.push(file.id);
            if let Some(symbols) = graph.file_symbols.get(&file.id) {
                seeds.extend(symbols);
            }
        }
    }
    if seeds.is_empty() {
        bail!("none of those paths are in the code graph (run `mimir graph build`?)");
    }
    let reached = graph.impact(&seeds, depth);
    if reached.is_empty() {
        println!("no impact outside the changed files");
        return Ok(());
    }
    for r in &reached {
        let node = store::get_node(&mimir.conn, r.id)?;
        if node.kind == Kind::Symbol {
            println!("{}{}", "  ".repeat(r.distance - 1), symbol_line(&node));
        }
    }
    println!("({} affected symbols within {depth} hops)", reached.len());
    Ok(())
}

/// Full record for a symbol: signature/doc, edges, linked memories.
pub fn node_info(reference: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
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

pub fn path(a: &str, b: &str) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let sa = resolve_symbol(&mimir.conn, proj.id, a)?;
    let sb = resolve_symbol(&mimir.conn, proj.id, b)?;
    let graph = CodeGraph::load(&mimir.conn, proj.id)?;
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
    Ok(())
}

pub fn hubs(limit: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
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
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
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
