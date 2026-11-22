//! `mimir graph …` — build/query the code graph of the current project.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mimir_core::model::{Kind, NewNode, Node, Rel};
use mimir_core::{store, Mimir};
use mimir_graph::render::{self, Direction};
use mimir_graph::{resolve_symbol, symbol_line, CodeGraph, UpdateOptions, UpdateReport};

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
/// detected project — no manual `graph build`. No-op once the project has a
/// graph, which `mimir_graph::has_graph` decides: a project that was built
/// and holds no code has one, and is not built again on every query.
/// Shared with `graph viz`.
pub(crate) fn ensure_graph(mimir: &mut Mimir, proj: &Node) -> Result<()> {
    if mimir_graph::has_graph(&mimir.conn, proj.id)? {
        return Ok(());
    }
    let Some(root) = proj.path.clone() else {
        return Ok(());
    };
    eprintln!("building the code graph (first run for this project)…");
    match mimir_graph::update(&mut mimir.conn, proj, std::path::Path::new(&root)) {
        Ok(s) => eprintln!(
            "  {} symbols, {} call sites resolved",
            s.symbols, s.calls_resolved
        ),
        Err(e) => eprintln!("  graph build skipped: {e}"),
    }
    let _ = mimir.embed_pending();
    Ok(())
}

/// What an answer read from this project's graph does not show about the
/// graph itself, one line each: that it is behind the working tree, and that
/// it was resolved by an older build's rules. A query answers from what is
/// stored; without these lines an answer about code that has since changed,
/// or one resting on edges a newer build would not draw, looks exactly like
/// a current one.
///
/// The one list for the CLI (which prints it on stderr) and the MCP `graph`
/// tool (which appends it to the answer). Read-only, and a failure to work
/// either line out is no line, never a failed query.
pub(crate) fn graph_notes(conn: &rusqlite::Connection, proj: &Node) -> Vec<String> {
    let mut notes = Vec::new();
    // A project whose checkout is not on this disk has no working tree to
    // be behind, and comparing against nothing would report every file gone.
    if let Some(root) = proj
        .path
        .as_deref()
        .map(std::path::Path::new)
        .filter(|root| root.is_dir())
    {
        if let Ok(freshness) = mimir_graph::freshness(conn, proj, root) {
            notes.extend(freshness.note());
        }
    }
    if let Ok(note) = mimir_graph::rules_note(conn, proj.id) {
        notes.extend(note);
    }
    notes
}

/// Print [`graph_notes`] on stderr, so stdout stays the answer.
fn note_if_behind(mimir: &Mimir, proj: &Node, multi: bool) {
    for note in graph_notes(&mimir.conn, proj) {
        if multi {
            eprintln!("{}: {note}", proj.title.as_deref().unwrap_or("(unnamed)"));
        } else {
            eprintln!("{note}");
        }
    }
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
    let all = mimir_graph::graphed_projects(&mimir.conn)?;
    if all.is_empty() {
        bail!("no project has a code graph yet — run `mimir graph build` in one first");
    }
    Ok(all)
}

fn project_header(proj: &Node) -> String {
    format!("── {} ──", proj.title.as_deref().unwrap_or("(unnamed)"))
}

pub fn build(full: bool) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let root = std::path::PathBuf::from(proj.path.as_deref().context("project has no root")?);
    let started = std::time::Instant::now();
    // A graph an older Mimir resolved is rebuilt in full without being
    // asked when the project is small; a large one keeps its note and waits
    // for `--full`.
    let upgrade = !full && mimir_graph::wants_full_rebuild(&mimir.conn, proj.id)?;
    let report = mimir_graph::update_with(
        &mut mimir.conn,
        &proj,
        &root,
        UpdateOptions {
            full: full || upgrade,
        },
    )?;
    println!(
        "{} [{:.1?}]{}",
        build_line(proj.title.as_deref().unwrap_or("project"), &report),
        started.elapsed(),
        if upgrade {
            " (rebuilt in full: this graph was resolved by an older Mimir's rules)"
        } else {
            ""
        },
    );
    let embedded = mimir.embed_pending()?;
    if embedded > 0 {
        println!("embedded {embedded} node(s)");
    }
    Ok(())
}

/// The one line `graph build` prints. The call numbers are call *sites*,
/// which is what the update counts; edges are fewer, since several sites in
/// one function share an edge. What the update could not do is appended only
/// when there is something to say, so an ordinary build reads as before.
fn build_line(title: &str, report: &UpdateReport) -> String {
    let stats = &report.stats;
    let mut line = format!(
        "{title}: {} files seen, {} indexed, {} unchanged, {} removed — {} symbols, {} call sites resolved ({} by receiver type, +{} heuristic), {} import edges",
        stats.files_seen,
        stats.files_indexed,
        stats.unchanged,
        stats.removed,
        stats.symbols,
        stats.calls_resolved,
        stats.calls_typed,
        stats.calls_heuristic,
        stats.imports,
    );
    if report.reresolved > 0 {
        line.push_str(&format!(
            ", {} unchanged file(s) re-resolved",
            report.reresolved
        ));
    }
    if report.parse_failed > 0 {
        line.push_str(&format!(
            ", {} file(s) failed to parse (kept as last indexed)",
            report.parse_failed
        ));
    }
    if report.unreadable > 0 {
        line.push_str(&format!(
            ", {} unreadable file(s) skipped",
            report.unreadable
        ));
    }
    line
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
    walk_relation(reference, depth, Direction::Callers, all_projects)
}

pub fn calls(reference: &str, depth: usize, all_projects: bool) -> Result<()> {
    walk_relation(reference, depth, Direction::Calls, all_projects)
}

fn walk_relation(
    reference: &str,
    depth: usize,
    direction: Direction,
    all_projects: bool,
) -> Result<()> {
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
        if multi {
            println!("{}", project_header(proj));
        }
        // The answer is rendered in one place for the CLI and the MCP tool
        // alike (`mimir_graph::render`); this only frames it.
        println!(
            "{}",
            render::relation(&mimir.conn, &graph, &sym, depth, direction)?
        );
        note_if_behind(&mimir, proj, multi);
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
        // The outcome is decided in one place for the CLI and the MCP tool
        // alike (`mimir_graph::Impact`), and so are the words
        // (`mimir_graph::render`).
        let trace = graph.impact_of_paths_traced(&mimir.conn, &files, depth)?;
        let Some(answer) = render::impact(&trace, depth) else {
            continue;
        };
        hit_any = true;
        if multi {
            println!("{}", project_header(proj));
        }
        println!("{answer}");
        note_if_behind(&mimir, proj, multi);
    }
    if !hit_any {
        // When the reason is a file added since the last build, the note
        // says the graph is behind and what to run.
        if let [proj] = projects.as_slice() {
            note_if_behind(&mimir, proj, false);
        }
        bail!("{}", render::NOT_IN_GRAPH);
    }
    Ok(())
}

/// Full record for a symbol: signature/doc, edges, linked memories.
pub fn node_info(reference: &str) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    ensure_graph(&mut mimir, &proj)?;
    let sym = resolve_symbol(&mimir.conn, proj.id, reference)?;
    println!("{}", render::node(&mimir.conn, &sym)?);
    mimir_core::learn::record_opened(&mimir.conn, sym.id)?;
    note_if_behind(&mimir, &proj, false);
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
        println!("{}", render::path(&mimir.conn, &graph, &sa, &sb)?);
        note_if_behind(&mimir, proj, multi);
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
    println!("{}", render::hubs(&mimir.conn, &graph, limit)?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_graph::GraphStats;

    /// The notes a graph answer carries, in order: nothing for a graph that
    /// is current and this build's work; the staleness line once a file has
    /// changed; the older-rules line once the rules stamp is missing — and
    /// only that one for a project whose checkout is not on this disk.
    #[test]
    fn graph_notes_say_what_an_answer_cannot() {
        let home = tempfile::tempdir().unwrap();
        let code = tempfile::tempdir().unwrap();
        std::fs::write(code.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let mut mimir = Mimir::open_at(mimir_core::config::Paths::under_root(home.path())).unwrap();
        let proj =
            store::ensure_project(&mimir.conn, &code.path().to_string_lossy(), "probe").unwrap();
        mimir_graph::update(&mut mimir.conn, &proj, code.path()).unwrap();
        assert_eq!(graph_notes(&mimir.conn, &proj), Vec::<String>::new());

        std::fs::write(code.path().join("a.rs"), "pub fn a() { let _ = 1; }\n").unwrap();
        let stale = graph_notes(&mimir.conn, &proj);
        assert_eq!(stale.len(), 1, "{stale:?}");
        assert!(stale[0].starts_with("note: the code graph is 1 file(s) behind"));

        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.graph_rules') WHERE id = ?1",
                [proj.id],
            )
            .unwrap();
        let both = graph_notes(&mimir.conn, &proj);
        assert_eq!(both.len(), 2, "{both:?}");
        assert_eq!(both[0], stale[0]);
        assert!(both[1].starts_with("note: this graph was built with an older Mimir's"));

        drop(code);
        let gone = graph_notes(&mimir.conn, &proj);
        assert_eq!(gone, both[1..], "no working tree to compare with");
    }

    /// The build line names what it counts, and mentions what the update
    /// could not do only when there is something to mention.
    #[test]
    fn build_line_counts_call_sites_and_reports_skips_only_when_there_are_any() {
        let mut report = UpdateReport {
            stats: GraphStats {
                files_seen: 9,
                files_indexed: 2,
                unchanged: 7,
                removed: 1,
                symbols: 30,
                calls_resolved: 12,
                calls_typed: 4,
                calls_heuristic: 3,
                imports: 5,
            },
            ..Default::default()
        };
        assert_eq!(
            build_line("probe", &report),
            "probe: 9 files seen, 2 indexed, 7 unchanged, 1 removed — 30 symbols, \
             12 call sites resolved (4 by receiver type, +3 heuristic), 5 import edges"
        );

        report.reresolved = 2;
        report.parse_failed = 1;
        report.unreadable = 3;
        let line = build_line("probe", &report);
        assert!(
            line.ends_with(
                "5 import edges, 2 unchanged file(s) re-resolved, 1 file(s) failed to parse \
                 (kept as last indexed), 3 unreadable file(s) skipped"
            ),
            "{line}"
        );
    }
}
