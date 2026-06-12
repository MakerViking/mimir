//! In-memory graph queries over the persisted symbols/edges.
//!
//! Loaded on demand from SQL (cheap at ≤100k symbols), cached by the
//! caller (the MCP server holds one per process, invalidated like the
//! vector matrix).

use std::collections::{HashMap, HashSet, VecDeque};

use mimir_core::error::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use rusqlite::Connection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Calls,
    Imports,
}

pub struct CodeGraph {
    pub graph: DiGraph<i64, EdgeKind>,
    idx: HashMap<i64, NodeIndex>,
    /// file node id → symbol node ids inside it
    pub file_symbols: HashMap<i64, Vec<i64>>,
}

#[derive(Debug)]
pub struct Reached {
    pub id: i64,
    pub distance: usize,
}

impl CodeGraph {
    pub fn load(conn: &Connection, project_id: i64) -> Result<CodeGraph> {
        let mut graph = DiGraph::new();
        let mut idx = HashMap::new();
        let mut file_symbols: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = conn.prepare(
            "SELECT id, parent_id FROM node
             WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL",
        )?;
        let symbols: Vec<(i64, Option<i64>)> = stmt
            .query_map([project_id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for (id, parent) in &symbols {
            idx.insert(*id, graph.add_node(*id));
            if let Some(p) = parent {
                file_symbols.entry(*p).or_default().push(*id);
            }
        }
        // File nodes participate so `imports` edges can route impact.
        let mut stmt = conn.prepare(
            "SELECT id FROM node
             WHERE kind = 'file' AND project_id = ?1 AND collection_id IS NULL
               AND deleted_at IS NULL",
        )?;
        let files: Vec<i64> = stmt
            .query_map([project_id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for id in &files {
            idx.entry(*id).or_insert_with(|| graph.add_node(*id));
        }

        let mut stmt = conn.prepare(
            "SELECT e.src, e.dst, e.rel FROM edge e
             JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
             WHERE e.rel IN ('calls', 'imports')
               AND s.project_id = ?1 AND s.deleted_at IS NULL AND d.deleted_at IS NULL",
        )?;
        let edges: Vec<(i64, i64, String)> = stmt
            .query_map([project_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for (src, dst, rel) in edges {
            if let (Some(&a), Some(&b)) = (idx.get(&src), idx.get(&dst)) {
                let kind = if rel == "calls" {
                    EdgeKind::Calls
                } else {
                    EdgeKind::Imports
                };
                graph.add_edge(a, b, kind);
            }
        }
        Ok(CodeGraph {
            graph,
            idx,
            file_symbols,
        })
    }

    /// Who (transitively) calls `id`, up to `depth` hops, nearest first.
    pub fn callers(&self, id: i64, depth: usize) -> Vec<Reached> {
        self.bfs(&[id], depth, petgraph::Direction::Incoming, false)
    }

    /// What `id` (transitively) calls, up to `depth` hops.
    pub fn calls(&self, id: i64, depth: usize) -> Vec<Reached> {
        self.bfs(&[id], depth, petgraph::Direction::Outgoing, false)
    }

    /// Blast radius of changing the given symbols (or whole files):
    /// reverse calls + reverse imports, depth-capped.
    pub fn impact(&self, seeds: &[i64], depth: usize) -> Vec<Reached> {
        self.bfs(seeds, depth, petgraph::Direction::Incoming, true)
    }

    fn bfs(
        &self,
        seeds: &[i64],
        depth: usize,
        dir: petgraph::Direction,
        include_imports: bool,
    ) -> Vec<Reached> {
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        for s in seeds {
            if let Some(&ix) = self.idx.get(s) {
                visited.insert(ix);
                queue.push_back((ix, 0));
            }
        }
        let mut out = Vec::new();
        while let Some((ix, d)) = queue.pop_front() {
            if d > 0 {
                out.push(Reached {
                    id: self.graph[ix],
                    distance: d,
                });
            }
            if d == depth {
                continue;
            }
            for edge in self.graph.edges_directed(ix, dir) {
                if !include_imports && *edge.weight() != EdgeKind::Calls {
                    continue;
                }
                let next = match dir {
                    petgraph::Direction::Incoming => petgraph::visit::EdgeRef::source(&edge),
                    petgraph::Direction::Outgoing => petgraph::visit::EdgeRef::target(&edge),
                };
                if visited.insert(next) {
                    queue.push_back((next, d + 1));
                }
            }
        }
        out
    }

    /// Shortest call path a → b (forward over calls).
    pub fn path(&self, a: i64, b: i64) -> Option<Vec<i64>> {
        let (&start, &goal) = (self.idx.get(&a)?, self.idx.get(&b)?);
        let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue = VecDeque::from([start]);
        let mut visited = HashSet::from([start]);
        while let Some(ix) = queue.pop_front() {
            if ix == goal {
                let mut path = vec![self.graph[ix]];
                let mut cur = ix;
                while let Some(&p) = prev.get(&cur) {
                    path.push(self.graph[p]);
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            for edge in self.graph.edges_directed(ix, petgraph::Direction::Outgoing) {
                if *edge.weight() != EdgeKind::Calls {
                    continue;
                }
                let next = petgraph::visit::EdgeRef::target(&edge);
                if visited.insert(next) {
                    prev.insert(next, ix);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Most-called symbols (in-degree over calls), best first.
    pub fn hubs(&self, limit: usize) -> Vec<(i64, usize)> {
        let mut counts: Vec<(i64, usize)> = self
            .graph
            .node_indices()
            .map(|ix| {
                let indeg = self
                    .graph
                    .edges_directed(ix, petgraph::Direction::Incoming)
                    .filter(|e| *e.weight() == EdgeKind::Calls)
                    .count();
                (self.graph[ix], indeg)
            })
            .filter(|(_, n)| *n > 0)
            .collect();
        counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        counts.truncate(limit);
        counts
    }

    /// Label-propagation communities over the undirected call graph.
    /// Returns symbol-id groups of size >= min_size.
    pub fn communities(&self, min_size: usize) -> Vec<Vec<i64>> {
        let mut label: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .enumerate()
            .map(|(i, ix)| (ix, i))
            .collect();
        // Deterministic sweeps (no RNG: stable output, resumable callers).
        for _ in 0..10 {
            let mut changed = false;
            for ix in self.graph.node_indices() {
                let mut freq: HashMap<usize, usize> = HashMap::new();
                for dir in [petgraph::Direction::Incoming, petgraph::Direction::Outgoing] {
                    for e in self.graph.edges_directed(ix, dir) {
                        if *e.weight() != EdgeKind::Calls {
                            continue;
                        }
                        let other = if petgraph::visit::EdgeRef::source(&e) == ix {
                            petgraph::visit::EdgeRef::target(&e)
                        } else {
                            petgraph::visit::EdgeRef::source(&e)
                        };
                        *freq.entry(label[&other]).or_default() += 1;
                    }
                }
                if let Some((&best, _)) = freq
                    .iter()
                    .max_by_key(|(l, n)| (**n, std::cmp::Reverse(**l)))
                {
                    if best != label[&ix] {
                        label.insert(ix, best);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut groups: HashMap<usize, Vec<i64>> = HashMap::new();
        for (ix, l) in &label {
            groups.entry(*l).or_default().push(self.graph[*ix]);
        }
        let mut out: Vec<Vec<i64>> = groups
            .into_values()
            .filter(|g| g.len() >= min_size)
            .collect();
        out.sort_by_key(|g| std::cmp::Reverse(g.len()));
        out
    }
}
