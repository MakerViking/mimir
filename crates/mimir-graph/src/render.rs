//! One rendering of each graph answer, shared by `mimir graph …` and the
//! MCP `graph` tool.
//!
//! The two surfaces used to format the same answer separately, and drifted:
//! "no callees found" on one and "no calls found" on the other, a call path
//! that named its ends on one and not on the other, an impact count on one
//! only. An agent that learned the CLI's output could not read the tool's.
//! Each answer is built here, once, as the lines to show; a surface decides
//! only where they go (stdout, or a tool result) and what frames them (a
//! project header, a freshness note).
//!
//! Nothing here writes to the store.

use mimir_core::error::Result;
use mimir_core::model::{Kind, Node, Rel};
use mimir_core::store;
use rusqlite::Connection;

use crate::queries::{CodeGraph, Impact, ImpactTrace, UNCONFIRMED};
use crate::tier::CallTier;
use crate::{impact_line, symbol_line};

/// Which way `callers` / `calls` walks the call edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Who calls the symbol.
    Callers,
    /// What the symbol calls.
    Calls,
}

impl Direction {
    /// What the walk finds, for the empty answer.
    fn found(self) -> &'static str {
        match self {
            Direction::Callers => "callers",
            Direction::Calls => "callees",
        }
    }
}

/// The answer to `impact` when none of the paths is a file of this graph.
/// Not an all-clear, and worded so it cannot be read as one.
///
/// It names both reasons and recommends nothing: a `Cargo.lock` or a
/// `README.md` is not in the graph because it is not code, and no rebuild
/// will put it there. When the reason is the other one, the freshness note
/// that follows the answer says the graph is behind and what to run.
pub const NOT_IN_GRAPH: &str = "none of those paths are in the code graph: they are not source \
     files Mimir indexes, or they were added since the last build";

/// The answer to `hubs` in a graph without a single call edge.
pub const NO_CALL_EDGES: &str = "no call edges yet (run `mimir graph build`)";

fn unconfirmed(confirmed: bool) -> &'static str {
    if confirmed {
        ""
    } else {
        UNCONFIRMED
    }
}

/// `callers` / `calls`: the symbol, then everything the walk reaches,
/// indented by distance. A result reached only through a guessed edge says
/// so.
pub fn relation(
    conn: &Connection,
    graph: &CodeGraph,
    symbol: &Node,
    depth: usize,
    direction: Direction,
) -> Result<String> {
    let reached = match direction {
        Direction::Callers => graph.callers_traced(symbol.id, depth),
        Direction::Calls => graph.calls_traced(symbol.id, depth),
    };
    let mut out = vec![symbol_line(symbol)];
    if reached.is_empty() {
        out.push(format!("no {} found", direction.found()));
    }
    for r in reached {
        let node = store::get_node(conn, r.id)?;
        out.push(format!(
            "{}{}{}",
            "  ".repeat(r.distance),
            symbol_line(&node),
            unconfirmed(r.confirmed)
        ));
    }
    Ok(out.join("\n"))
}

/// `path`: the shortest call path, one hop per line, or a sentence naming
/// the two ends that have none. A hop over a guessed edge says so.
pub fn path(conn: &Connection, graph: &CodeGraph, from: &Node, to: &Node) -> Result<String> {
    let Some(ids) = graph.path(from.id, to.id) else {
        return Ok(format!(
            "no call path {} → {}",
            from.title.as_deref().unwrap_or("?"),
            to.title.as_deref().unwrap_or("?")
        ));
    };
    let mut out = Vec::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        let node = store::get_node(conn, *id)?;
        let confirmed = i == 0 || graph.call_is_confirmed(ids[i - 1], *id);
        out.push(format!(
            "{}{}{}",
            "  ".repeat(i),
            symbol_line(&node),
            unconfirmed(confirmed)
        ));
    }
    Ok(out.join("\n"))
}

/// `impact`: what reaches the changed files, nearest first, then a count.
/// `None` when the files are not in this graph at all — the caller says
/// [`NOT_IN_GRAPH`], or moves on to the next project.
pub fn impact(trace: &ImpactTrace, depth: usize) -> Option<String> {
    let hits = match &trace.impact {
        Impact::NotInGraph => return None,
        Impact::Contained => return Some("no impact outside the changed files".to_string()),
        Impact::Reached(hits) => hits,
    };
    let mut out: Vec<String> = hits
        .iter()
        .map(|hit| {
            format!(
                "{}{}{}",
                "  ".repeat(hit.distance.saturating_sub(1)),
                impact_line(hit),
                unconfirmed(trace.is_confirmed(hit))
            )
        })
        .collect();
    let symbols = hits.iter().filter(|h| h.node.kind == Kind::Symbol).count();
    let files = hits.len() - symbols;
    out.push(if files == 0 {
        format!("({symbols} affected symbols within {depth} hops)")
    } else {
        format!("({symbols} affected symbols and {files} importing file(s) within {depth} hops)")
    });
    Some(out.join("\n"))
}

/// `node`: the symbol, its signature and doc, then every edge that touches
/// it. A guessed call edge says so; an edge to something that no longer
/// exists is not listed.
pub fn node(conn: &Connection, symbol: &Node) -> Result<String> {
    let mut out = vec![symbol_line(symbol)];
    if let Some(body) = &symbol.body {
        out.push(body.clone());
    }
    for edge in store::edges_of(conn, symbol.id)? {
        let (arrow, other_id) = if edge.src == symbol.id {
            ("→", edge.dst)
        } else {
            ("←", edge.src)
        };
        let Ok(other) = store::get_node(conn, other_id) else {
            continue;
        };
        let line = match other.kind {
            Kind::Symbol if other.deleted_at.is_some() => continue,
            Kind::Symbol => symbol_line(&other),
            // A retired memory says so; a forgotten one is not listed.
            _ => match mimir_core::format::link_target_label(&other) {
                Some(label) => label,
                None => continue,
            },
        };
        let guess = edge.rel == Rel::Calls && !CallTier::weight_is_resolved(edge.weight);
        out.push(format!(
            "{arrow} {} {line}{}",
            edge.rel,
            unconfirmed(!guess)
        ));
    }
    Ok(out.join("\n"))
}

/// `hubs`: the most-called symbols with their in-degree.
pub fn hubs(conn: &Connection, graph: &CodeGraph, limit: usize) -> Result<String> {
    let hubs = graph.hubs(limit);
    if hubs.is_empty() {
        return Ok(NO_CALL_EDGES.to_string());
    }
    let mut out = Vec::with_capacity(hubs.len());
    for (id, indeg) in hubs {
        let node = store::get_node(conn, id)?;
        out.push(format!("↑{indeg:<4} {}", symbol_line(&node)));
    }
    Ok(out.join("\n"))
}
