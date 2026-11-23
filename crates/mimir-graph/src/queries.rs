//! In-memory graph queries over the persisted symbols/edges.
//!
//! Loaded from SQL by every query that needs it (cheap at ≤100k symbols).
//! Nothing caches a loaded graph, in the CLI or in the MCP server, so an
//! answer always reflects the store as it is when the question is asked.

use std::collections::{HashMap, HashSet, VecDeque};

use mimir_core::error::Result;
use mimir_core::model::{Kind, Node};
use mimir_core::store;
use petgraph::graph::{DiGraph, NodeIndex};
use rusqlite::{params, Connection};

use crate::store_graph::LIVE_CODE_FILE;
use crate::tier::CallTier;

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
    /// The project this graph was loaded for. Kept so a path lookup can only
    /// ever be answered from this project's files — see [`Impact`].
    project_id: i64,
    /// File node ids. The graph holds files beside symbols so `imports`
    /// edges can route impact; anything that is about symbols alone
    /// (communities) leaves these out.
    files: HashSet<i64>,
    /// Call edges (caller id, callee id) the resolver stored as a guess: one
    /// of two or three candidates for the name ([`CallTier::weight_is_resolved`]
    /// says which edges those are). Kept beside the graph rather than in
    /// [`EdgeKind`], whose two variants are what the edge *is*; this is how
    /// far to trust it.
    unresolved: HashSet<(i64, i64)>,
}

#[derive(Debug)]
pub struct Reached {
    pub id: i64,
    pub distance: usize,
}

/// A traversal result, and whether the route to it can be relied on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Traced {
    pub id: i64,
    pub distance: usize,
    /// `false` when every route to it, within the depth asked for, passes
    /// through at least one unresolved call edge: the graph cannot say this
    /// result is real, only that a name matched. A surface prints such a
    /// result with [`UNCONFIRMED`].
    pub confirmed: bool,
}

/// Suffix for a result reached only through a guess.
pub const UNCONFIRMED: &str = " (unconfirmed call)";

/// What changing a set of files touches in one project's graph.
///
/// Three outcomes, and a reader must be able to tell them apart, because one
/// of them is an all-clear and another is its opposite. They used to be
/// re-derived by each surface from whichever list happened to come back
/// empty, and the derivations disagreed: the CLI filtered path matches to the
/// project *after* a machine-wide cap, the MCP tool did not filter them at
/// all, and so an agent asking about a path that existed only in some other
/// project was told "no impact outside the changed files".
///
/// The outcome is decided once, in [`CodeGraph::impact_of_paths`]; a surface
/// only chooses the words.
#[derive(Debug)]
pub enum Impact {
    /// None of the paths name a file this project has indexed. This says
    /// nothing about what depends on them — the graph cannot answer.
    NotInGraph,
    /// The files are indexed for this project, and nothing outside them
    /// reaches them within the requested depth. A docs file lands here by
    /// construction: it is indexed, and has no call or import edges.
    Contained,
    /// What reaches the changed files, nearest first.
    Reached(Vec<ImpactHit>),
}

/// One thing affected by a change, and how many hops away it is (≥ 1).
///
/// `node` is a symbol, or — when a file depends on the change through an
/// `imports` edge and none of its symbols has a call edge into it — the file
/// itself. That second case is a file that uses a type or a constant: real
/// impact, with no call to hang it on.
#[derive(Debug)]
pub struct ImpactHit {
    pub node: Node,
    pub distance: usize,
}

/// An [`Impact`], and which of its hits are guesses.
#[derive(Debug)]
pub struct ImpactTrace {
    pub impact: Impact,
    /// Node ids of hits reached only through unresolved call edges.
    unconfirmed: HashSet<i64>,
}

impl ImpactTrace {
    /// Whether some route to `hit` is made of resolved edges only.
    pub fn is_confirmed(&self, hit: &ImpactHit) -> bool {
        !self.unconfirmed.contains(&hit.node.id)
    }
}

/// Most files a single path may seed. The lookup is by suffix, so `mod.rs`
/// can match much of a project; the cap keeps a sloppy path from turning the
/// answer into "everything".
const MAX_FILES_PER_PATH: usize = 3;

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
        let mut stmt = conn.prepare(&format!("SELECT id FROM node WHERE {LIVE_CODE_FILE}"))?;
        let files: HashSet<i64> = stmt
            .query_map([project_id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for id in &files {
            idx.entry(*id).or_insert_with(|| graph.add_node(*id));
        }

        // Whether a call edge is a guess is read from its weight, not from
        // the `resolved` flag in its meta: the same answer for every edge
        // any build has written, without parsing a JSON object per edge.
        let mut stmt = conn.prepare(
            "SELECT e.src, e.dst, e.rel, e.weight
             FROM edge e
             JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
             WHERE e.rel IN ('calls', 'imports')
               AND s.project_id = ?1 AND s.deleted_at IS NULL AND d.deleted_at IS NULL",
        )?;
        let edges: Vec<(i64, i64, String, f64)> = stmt
            .query_map([project_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        let mut unresolved: HashSet<(i64, i64)> = HashSet::new();
        for (src, dst, rel, weight) in edges {
            if let (Some(&a), Some(&b)) = (idx.get(&src), idx.get(&dst)) {
                let kind = match rel.as_str() {
                    "calls" => EdgeKind::Calls,
                    "imports" => EdgeKind::Imports,
                    // The query asks for exactly those two. Should it ever
                    // return another, that edge has no kind here, and
                    // filing it under one of them would route impact along
                    // an edge that means something else.
                    other => {
                        tracing::warn!(rel = other, "ignoring an edge of an unexpected relation");
                        continue;
                    }
                };
                if kind == EdgeKind::Calls && !CallTier::weight_is_resolved(weight) {
                    unresolved.insert((src, dst));
                }
                graph.add_edge(a, b, kind);
            }
        }
        Ok(CodeGraph {
            graph,
            idx,
            file_symbols,
            project_id,
            files,
            unresolved,
        })
    }

    /// Who (transitively) calls `id`, up to `depth` hops, nearest first.
    pub fn callers(&self, id: i64, depth: usize) -> Vec<Reached> {
        self.bfs(&[id], depth, petgraph::Direction::Incoming, false, false)
    }

    /// What `id` (transitively) calls, up to `depth` hops.
    pub fn calls(&self, id: i64, depth: usize) -> Vec<Reached> {
        self.bfs(&[id], depth, petgraph::Direction::Outgoing, false, false)
    }

    /// Blast radius of changing the given symbols (or whole files):
    /// reverse calls + reverse imports, depth-capped.
    pub fn impact(&self, seeds: &[i64], depth: usize) -> Vec<Reached> {
        self.bfs(seeds, depth, petgraph::Direction::Incoming, true, false)
    }

    /// [`CodeGraph::callers`], each result marked with whether it is
    /// confirmed.
    pub fn callers_traced(&self, id: i64, depth: usize) -> Vec<Traced> {
        self.traced(&[id], depth, petgraph::Direction::Incoming, false)
    }

    /// [`CodeGraph::calls`], each result marked with whether it is confirmed.
    pub fn calls_traced(&self, id: i64, depth: usize) -> Vec<Traced> {
        self.traced(&[id], depth, petgraph::Direction::Outgoing, false)
    }

    /// [`CodeGraph::impact`], each result marked with whether it is
    /// confirmed.
    pub fn impact_traced(&self, seeds: &[i64], depth: usize) -> Vec<Traced> {
        self.traced(seeds, depth, petgraph::Direction::Incoming, true)
    }

    /// Whether the call edge `src → dst` is a resolved one. `true` for a
    /// pair with no call edge at all: this answers "is this hop a guess",
    /// not "does this hop exist".
    pub fn call_is_confirmed(&self, src: i64, dst: i64) -> bool {
        !self.unresolved.contains(&(src, dst))
    }

    /// A traversal over every edge, with each result checked against the
    /// same traversal over resolved edges only: a result the second one does
    /// not reach is one the graph only guessed at.
    fn traced(
        &self,
        seeds: &[i64],
        depth: usize,
        dir: petgraph::Direction,
        include_imports: bool,
    ) -> Vec<Traced> {
        let confirmed: HashSet<i64> = self
            .bfs(seeds, depth, dir, include_imports, true)
            .into_iter()
            .map(|r| r.id)
            .collect();
        self.bfs(seeds, depth, dir, include_imports, false)
            .into_iter()
            .map(|r| Traced {
                id: r.id,
                distance: r.distance,
                confirmed: confirmed.contains(&r.id),
            })
            .collect()
    }

    /// Blast radius of changing the files at `paths` (as `git diff
    /// --name-only` prints them), decided once for every surface.
    ///
    /// Paths match by suffix against **this project's files only**: files of
    /// other projects are not in this graph, and letting them count is what
    /// turned "not in the graph" into "no impact". Code files seed the
    /// traversal; the shortest match sorts first, so the cap can never cut
    /// off the file the caller named exactly. A path that is only an indexed
    /// docs file seeds nothing but still counts as known.
    pub fn impact_of_paths(
        &self,
        conn: &Connection,
        paths: &[String],
        depth: usize,
    ) -> Result<Impact> {
        Ok(self.impact_of_paths_traced(conn, paths, depth)?.impact)
    }

    /// [`CodeGraph::impact_of_paths`], also saying which hits are reached
    /// only through unresolved call edges.
    pub fn impact_of_paths_traced(
        &self,
        conn: &Connection,
        paths: &[String],
        depth: usize,
    ) -> Result<ImpactTrace> {
        let outcome = |impact: Impact, unconfirmed: HashSet<i64>| ImpactTrace {
            impact,
            unconfirmed,
        };
        let mut code = conn.prepare(&format!(
            "SELECT id FROM node
             WHERE {LIVE_CODE_FILE} AND path LIKE ?2
             ORDER BY length(path), path
             LIMIT ?3"
        ))?;
        let mut docs = conn.prepare(
            "SELECT EXISTS (
                SELECT 1 FROM node
                WHERE kind = 'file' AND project_id = ?1 AND collection_id IS NOT NULL
                  AND deleted_at IS NULL AND path LIKE ?2)",
        )?;
        let mut seeds: Vec<i64> = Vec::new();
        let mut any_indexed = false;
        for path in paths {
            let pattern = format!("%{}", path.trim_start_matches("./"));
            let files: Vec<i64> = code
                .query_map(
                    params![self.project_id, pattern, MAX_FILES_PER_PATH as i64],
                    |r| r.get(0),
                )?
                .collect::<rusqlite::Result<_>>()?;
            if files.is_empty() {
                any_indexed |=
                    docs.query_row(params![self.project_id, pattern], |r| r.get::<_, bool>(0))?;
            }
            for file in files {
                seeds.push(file);
                if let Some(symbols) = self.file_symbols.get(&file) {
                    seeds.extend(symbols);
                }
            }
        }
        if seeds.is_empty() {
            let impact = if any_indexed {
                Impact::Contained
            } else {
                Impact::NotInGraph
            };
            return Ok(outcome(impact, HashSet::new()));
        }
        let reached = self.impact_traced(&seeds, depth);
        let reached_ids: HashSet<i64> = reached.iter().map(|r| r.id).collect();
        let unconfirmed: HashSet<i64> = reached
            .iter()
            .filter(|r| !r.confirmed)
            .map(|r| r.id)
            .collect();
        let mut hits = Vec::new();
        for r in reached {
            let node = store::get_node(conn, r.id)?;
            let wanted = match node.kind {
                Kind::Symbol => true,
                // A file is reached along `imports` edges. Usually one of
                // its symbols is reached too, by the call that the import
                // exists for, and that symbol is the better answer. When
                // none is, the import is the only evidence there is —
                // dropping it reports "no impact" for every change to a
                // file of types or constants.
                Kind::File => !self
                    .file_symbols
                    .get(&node.id)
                    .is_some_and(|symbols| symbols.iter().any(|s| reached_ids.contains(s))),
                _ => false,
            };
            if wanted {
                hits.push(ImpactHit {
                    node,
                    distance: r.distance,
                });
            }
        }
        let impact = if hits.is_empty() {
            Impact::Contained
        } else {
            Impact::Reached(hits)
        };
        Ok(outcome(impact, unconfirmed))
    }

    /// Breadth-first from `seeds`, nearest first. `resolved_only` leaves out
    /// call edges the resolver stored as a guess.
    fn bfs(
        &self,
        seeds: &[i64],
        depth: usize,
        dir: petgraph::Direction,
        include_imports: bool,
        resolved_only: bool,
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
                let (from, to) = (
                    petgraph::visit::EdgeRef::source(&edge),
                    petgraph::visit::EdgeRef::target(&edge),
                );
                if resolved_only
                    && *edge.weight() == EdgeKind::Calls
                    && self
                        .unresolved
                        .contains(&(self.graph[from], self.graph[to]))
                {
                    continue;
                }
                let next = match dir {
                    petgraph::Direction::Incoming => from,
                    petgraph::Direction::Outgoing => to,
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
    /// Returns symbol-id groups of size >= min_size: largest first, members
    /// in id order, so the same graph always prints the same communities.
    ///
    /// Symbols only. The graph also holds file nodes, which have no call
    /// edges and would each come out as a community of one.
    pub fn communities(&self, min_size: usize) -> Vec<Vec<i64>> {
        let symbols: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|ix| !self.files.contains(&self.graph[*ix]))
            .collect();
        let mut label: HashMap<NodeIndex, usize> =
            symbols.iter().map(|ix| (*ix, ix.index())).collect();
        // Deterministic sweeps (no RNG: stable output, resumable callers).
        for _ in 0..10 {
            let mut changed = false;
            for &ix in &symbols {
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
                        // A call edge someone authored may end at a file;
                        // that end has no label to vote with.
                        if let Some(l) = label.get(&other) {
                            *freq.entry(*l).or_default() += 1;
                        }
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
        for group in &mut out {
            group.sort_unstable();
        }
        out.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        out
    }
}
