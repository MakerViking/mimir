//! MCP stdio server (`mimir mcp`): one global registration serves every
//! repo. Project scope comes from the server's cwd (where the agent
//! launched it), overridable via `MIMIR_PROJECT=<path>`.
//!
//! Returns are token-lean agent-format text, not JSON. The engine is
//! sync; every tool hops through spawn_blocking.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use mimir_core::format::agent_line;
use mimir_core::memory::{self, RememberOutcome};
use mimir_core::model::{short_uid, Kind, MemoryType, Rel, Scope};
use mimir_core::search::SearchQuery;
use mimir_core::{store, Mimir};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Deserialize;

const INSTRUCTIONS: &str = "Mimir is the user's persistent memory across all projects: typed \
memories (gotchas, decisions, insights, ideas, notes, person-notes), indexed docs, and (soon) \
code symbols, in one searchable store. Norms: SEARCH BEFORE CAPTURE — call recall before \
remember to avoid duplicates (remember also refuses near-duplicates). Recall at the start of \
non-trivial tasks (debugging, architecture, 'how did we do X'). Capture concise reusable facts, \
not session noise. Use get to read full bodies (this also teaches Mimir what is useful).";

#[derive(Clone)]
pub struct MimirServer {
    engine: Arc<Mutex<Mimir>>,
    project_id: Option<i64>,
    // Read only inside the #[tool_handler] macro expansion.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

fn engine_err(e: impl std::fmt::Display) -> String {
    format!("error: {e}")
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// What to search for (natural language or keywords).
    pub query: String,
    /// Filter: all | memory | doc | code (default all).
    #[serde(default)]
    pub kind: Option<String>,
    /// Max results (default 8).
    #[serde(default)]
    pub limit: Option<usize>,
    /// true = search every project, not just the current one + global.
    #[serde(default)]
    pub all_projects: bool,
    /// true = rescore candidates with the cross-encoder reranker
    /// (slower, better ordering; needs the reranker model downloaded).
    #[serde(default)]
    pub rerank: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct RememberArgs {
    /// The fact to store (concise, self-contained).
    pub text: String,
    /// insight | decision | gotcha | idea | note | person (default note).
    #[serde(default)]
    pub r#type: Option<String>,
    /// Topic tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// true = cross-project (global) instead of current-project scope.
    #[serde(default)]
    pub global: bool,
    /// Optional: the code symbol or node this memory is about (name or id).
    /// Link at capture time so the memory and the code surface together.
    #[serde(default)]
    pub link: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct GetArgs {
    /// Node id (m:ABCDEF), full ULID, or indexed file path like docs/x.md:10-40.
    pub r#ref: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct MarkArgs {
    /// Node reference (id from a recall hit).
    pub r#ref: String,
    /// true = useful (+1 strength), false = noise (-1 strength).
    pub useful: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct GraphArgs {
    /// callers | calls | impact | node | path | hubs
    pub op: String,
    /// Symbol reference (name, qualified name, or id) for callers/calls/node/path.
    #[serde(default)]
    pub symbol: Option<String>,
    /// Second symbol for path.
    #[serde(default)]
    pub to: Option<String>,
    /// Changed file paths for impact.
    #[serde(default)]
    pub files: Vec<String>,
    /// Traversal depth (default 3).
    #[serde(default)]
    pub depth: Option<usize>,
    /// Max results for hubs (default 10).
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct LinkArgs {
    /// Source node reference.
    pub a: String,
    /// Target node reference.
    pub b: String,
    /// Relation: relates | about | links | mentions | supersedes (default relates).
    #[serde(default)]
    pub rel: Option<String>,
}

#[tool_router]
impl MimirServer {
    pub fn new(engine: Mimir, project_id: Option<i64>) -> Self {
        MimirServer {
            engine: Arc::new(Mutex::new(engine)),
            project_id,
            tool_router: Self::tool_router(),
        }
    }

    fn blocking<F>(&self, f: F) -> impl std::future::Future<Output = String>
    where
        F: FnOnce(&mut Mimir) -> Result<String, String> + Send + 'static,
    {
        let engine = self.engine.clone();
        async move {
            tokio::task::spawn_blocking(move || {
                let mut engine = engine.lock().unwrap_or_else(|p| p.into_inner());
                f(&mut engine).unwrap_or_else(|e| e)
            })
            .await
            .unwrap_or_else(|e| format!("error: task panicked: {e}"))
        }
    }

    #[tool(
        description = "Search the user's memories, docs and code in one ranked list (hybrid BM25 + semantic). One ~25-token line per hit; use `get` for full bodies."
    )]
    async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let kinds = match args.kind.as_deref().unwrap_or("all") {
                "all" => vec![],
                "memory" => vec![Kind::Memory],
                "doc" => vec![Kind::File, Kind::Chunk, Kind::Annotation],
                "code" => vec![Kind::Symbol],
                other => return Err(format!("unknown kind '{other}' (all|memory|doc|code)")),
            };
            let scope = if args.all_projects {
                Scope::All
            } else {
                project_id.map(Scope::Project).unwrap_or(Scope::All)
            };
            let query = SearchQuery {
                text: args.query,
                scope,
                kinds,
                since: None,
                limit: args.limit.unwrap_or(m.config.output.default_limit),
                strength_alpha: m.config.scoring.strength_alpha,
            };
            let hits = m.search_with(&query, args.rerank).map_err(engine_err)?;
            if hits.is_empty() {
                return Ok("no results".into());
            }
            let query_hash = blake3::hash(query.text.as_bytes());
            let shown: Vec<(i64, i64, f64)> = hits
                .iter()
                .enumerate()
                .map(|(rank, hit)| (hit.node.id, rank as i64, hit.score))
                .collect();
            store::record_shown(&m.conn, query_hash.as_bytes(), &shown).map_err(engine_err)?;
            let projects = store::project_titles(&m.conn).map_err(engine_err)?;
            let mut out = Vec::new();
            for hit in &hits {
                let project = hit
                    .node
                    .project_id
                    .and_then(|id| projects.get(&id))
                    .map(String::as_str);
                out.push(agent_line(
                    &hit.node,
                    project,
                    m.config.output.snippet_chars,
                ));
            }
            Ok(out.join("\n"))
        })
        .await
    }

    #[tool(
        description = "Store a typed memory (gotcha/decision/insight/idea/note/person). Search first with recall; near-duplicates are refused with the existing entry. When the fact is about specific code, pass `link` with the symbol or file so memory and code surface together."
    )]
    async fn remember(&self, Parameters(args): Parameters<RememberArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let mtype: MemoryType = args
                .r#type
                .as_deref()
                .unwrap_or("note")
                .parse()
                .map_err(engine_err)?;
            let outcome = memory::remember(
                &m.conn,
                memory::Remember {
                    text: args.text,
                    mtype,
                    tags: args.tags,
                    project_id: if args.global { None } else { project_id },
                    force: false,
                },
            )
            .map_err(engine_err)?;
            let projects = store::project_titles(&m.conn).map_err(engine_err)?;
            let snippet = m.config.output.snippet_chars;
            let line = |node: &mimir_core::model::Node| {
                let project = node
                    .project_id
                    .and_then(|id| projects.get(&id))
                    .map(String::as_str);
                agent_line(node, project, snippet)
            };
            match outcome {
                RememberOutcome::Created(node) => {
                    if let Err(err) = m.embed_pending() {
                        tracing::warn!(%err, "embedding new memory failed");
                    }
                    let mut msg = format!("stored {}", line(&node));
                    if let Some(target) = args.link.as_deref() {
                        let resolved = store::resolve_ref(&m.conn, target).ok().or_else(|| {
                            project_id.and_then(|pid| {
                                mimir_graph::resolve_symbol(&m.conn, pid, target).ok()
                            })
                        });
                        match resolved {
                            Some(t) => {
                                store::link(&m.conn, node.id, t.id, Rel::About, 1.0)
                                    .map_err(engine_err)?;
                                msg.push_str(&format!(
                                    "\nlinked —about→ {}",
                                    short_uid(t.kind, &t.uid)
                                ));
                            }
                            None => msg.push_str(&format!(
                                "\n(link target '{target}' not found — memory stored unlinked)"
                            )),
                        }
                    }
                    Ok(msg)
                }
                RememberOutcome::Duplicate(existing) => Ok(format!(
                    "refused: near-duplicate of\n{}\n(edit that entry instead, or rephrase if genuinely different)",
                    line(&existing)
                )),
            }
        })
        .await
    }

    #[tool(
        description = "Read the full record behind a recall hit: complete body, tags, links. Also accepts indexed doc paths (file.md:10-40) to read file lines."
    )]
    async fn get(&self, Parameters(args): Parameters<GetArgs>) -> String {
        self.blocking(move |m| {
            if let Some(slice) =
                mimir_core::index::file_slice(&m.conn, &args.r#ref).map_err(engine_err)?
            {
                return Ok(slice);
            }
            let node = store::resolve_ref(&m.conn, &args.r#ref).map_err(engine_err)?;
            mimir_core::learn::record_opened(&m.conn, node.id).map_err(engine_err)?;
            let projects = store::project_titles(&m.conn).map_err(engine_err)?;
            mimir_core::format::full_record(&m.conn, &node, &projects).map_err(engine_err)
        })
        .await
    }

    #[tool(
        description = "Link two nodes (memory↔memory, memory↔doc, memory↔code) so context surfaces together later."
    )]
    async fn link(&self, Parameters(args): Parameters<LinkArgs>) -> String {
        self.blocking(move |m| {
            let rel: Rel = args
                .rel
                .as_deref()
                .unwrap_or("relates")
                .parse()
                .map_err(engine_err)?;
            let src = store::resolve_ref(&m.conn, &args.a).map_err(engine_err)?;
            let dst = store::resolve_ref(&m.conn, &args.b).map_err(engine_err)?;
            store::link(&m.conn, src.id, dst.id, rel, 1.0).map_err(engine_err)?;
            Ok(format!(
                "{} —{rel}→ {}",
                short_uid(src.kind, &src.uid),
                short_uid(dst.kind, &dst.uid)
            ))
        })
        .await
    }

    #[tool(
        description = "Code-graph queries for the current project: callers/calls of a symbol, impact (blast radius of changed files — pass git diff --name-only), node (full symbol record), path between symbols, hubs (most-called). Build the graph first with `mimir graph build`."
    )]
    async fn graph(&self, Parameters(args): Parameters<GraphArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let Some(project_id) = project_id else {
                return Err("no project detected (the code graph is per-project)".into());
            };
            let depth = args.depth.unwrap_or(3);
            let graph = mimir_graph::CodeGraph::load(&m.conn, project_id).map_err(engine_err)?;
            let line = |id: i64| -> Result<String, String> {
                let node = store::get_node(&m.conn, id).map_err(engine_err)?;
                Ok(mimir_graph::symbol_line(&node))
            };
            let resolve = |r: &Option<String>| -> Result<mimir_core::model::Node, String> {
                let r = r.as_deref().ok_or("missing 'symbol' argument")?;
                mimir_graph::resolve_symbol(&m.conn, project_id, r).map_err(engine_err)
            };
            match args.op.as_str() {
                "callers" | "calls" => {
                    let sym = resolve(&args.symbol)?;
                    let reached = if args.op == "callers" {
                        graph.callers(sym.id, depth)
                    } else {
                        graph.calls(sym.id, depth)
                    };
                    if reached.is_empty() {
                        return Ok(format!("no {} found", args.op));
                    }
                    let mut out = vec![mimir_graph::symbol_line(&sym)];
                    for r in reached {
                        out.push(format!("{}{}", "  ".repeat(r.distance), line(r.id)?));
                    }
                    Ok(out.join("\n"))
                }
                "impact" => {
                    let mut seeds = Vec::new();
                    for f in &args.files {
                        for file in store::files_by_path_suffix(&m.conn, f.trim_start_matches("./"))
                            .map_err(engine_err)?
                        {
                            seeds.push(file.id);
                            if let Some(syms) = graph.file_symbols.get(&file.id) {
                                seeds.extend(syms);
                            }
                        }
                    }
                    if seeds.is_empty() {
                        return Ok("none of those paths are in the code graph".into());
                    }
                    let reached = graph.impact(&seeds, depth);
                    let mut out = Vec::new();
                    for r in &reached {
                        let node = store::get_node(&m.conn, r.id).map_err(engine_err)?;
                        if matches!(node.kind, Kind::Symbol) {
                            out.push(format!(
                                "{}{}",
                                "  ".repeat(r.distance.saturating_sub(1)),
                                mimir_graph::symbol_line(&node)
                            ));
                        }
                    }
                    if out.is_empty() {
                        return Ok("no impact outside the changed files".into());
                    }
                    Ok(out.join("\n"))
                }
                "node" => {
                    let sym = resolve(&args.symbol)?;
                    mimir_core::learn::record_opened(&m.conn, sym.id).map_err(engine_err)?;
                    let mut out = vec![mimir_graph::symbol_line(&sym)];
                    if let Some(body) = &sym.body {
                        out.push(body.clone());
                    }
                    for edge in store::edges_of(&m.conn, sym.id).map_err(engine_err)? {
                        let (arrow, other_id) = if edge.src == sym.id {
                            ("→", edge.dst)
                        } else {
                            ("←", edge.src)
                        };
                        if let Ok(other) = store::get_node(&m.conn, other_id) {
                            let l = match other.kind {
                                Kind::Symbol => mimir_graph::symbol_line(&other),
                                _ => format!(
                                    "{} {}",
                                    short_uid(other.kind, &other.uid),
                                    other.title.as_deref().unwrap_or("")
                                ),
                            };
                            out.push(format!("{arrow} {} {l}", edge.rel));
                        }
                    }
                    Ok(out.join("\n"))
                }
                "path" => {
                    let a = resolve(&args.symbol)?;
                    let b = mimir_graph::resolve_symbol(
                        &m.conn,
                        project_id,
                        args.to.as_deref().ok_or("missing 'to' argument")?,
                    )
                    .map_err(engine_err)?;
                    match graph.path(a.id, b.id) {
                        Some(ids) => {
                            let mut out = Vec::new();
                            for (i, id) in ids.iter().enumerate() {
                                out.push(format!("{}{}", "  ".repeat(i), line(*id)?));
                            }
                            Ok(out.join("\n"))
                        }
                        None => Ok("no call path".into()),
                    }
                }
                "hubs" => {
                    let hubs = graph.hubs(args.limit.unwrap_or(10));
                    if hubs.is_empty() {
                        return Ok("no call edges yet (run `mimir graph build`)".into());
                    }
                    let mut out = Vec::new();
                    for (id, indeg) in hubs {
                        out.push(format!("↑{indeg:<4} {}", line(id)?));
                    }
                    Ok(out.join("\n"))
                }
                other => Err(format!(
                    "unknown op '{other}' (callers|calls|impact|node|path|hubs)"
                )),
            }
        })
        .await
    }

    #[tool(
        description = "Explicit feedback on a recalled node: useful strengthens its future ranking, noise weakens it. Use after a memory actually helped (or misled)."
    )]
    async fn mark(&self, Parameters(args): Parameters<MarkArgs>) -> String {
        self.blocking(move |m| {
            let node = store::resolve_ref(&m.conn, &args.r#ref).map_err(engine_err)?;
            let strength =
                mimir_core::learn::apply_mark(&m.conn, node.id, args.useful).map_err(engine_err)?;
            Ok(format!(
                "{} {} → strength {strength:.2}",
                short_uid(node.kind, &node.uid),
                if args.useful { "useful" } else { "noise" }
            ))
        })
        .await
    }

    #[tool(description = "Store overview: detected project, counts by kind, database size.")]
    async fn status(&self) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let counts = store::count_by_kind(&m.conn).map_err(engine_err)?;
            let project = match project_id {
                Some(id) => store::get_node(&m.conn, id)
                    .ok()
                    .and_then(|p| p.title)
                    .unwrap_or_else(|| "?".into()),
                None => "(none — global scope)".into(),
            };
            let db_size = std::fs::metadata(&m.paths.db_file)
                .map(|md| md.len())
                .unwrap_or(0);
            let summary = if counts.is_empty() {
                "empty".to_string()
            } else {
                counts
                    .iter()
                    .map(|(k, v)| format!("{v} {k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            Ok(format!(
                "project {project}\nstore   {summary}\ndb      {} KB",
                db_size / 1024
            ))
        })
        .await
    }
}

#[tool_handler]
impl ServerHandler for MimirServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }
}

/// Entry point for `mimir mcp` (blocking; owns the tokio runtime).
pub fn run() -> Result<()> {
    let engine = Mimir::open()?;
    // Project scope: MIMIR_PROJECT env beats the server's cwd.
    let cwd = std::env::var_os("MIMIR_PROJECT")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_dir().ok());
    let project_id = cwd
        .and_then(|d| engine.project_for_cwd(&d).ok().flatten())
        .map(|p| p.id);

    // Auto-sync the project on session start: first contact builds the code
    // graph and indexes the repo's markdown; every later start is an
    // incremental no-op in milliseconds. Separate thread + connection like
    // consolidation below — never on the request path, never fatal, and
    // local-only (embed_pending silently skips when no model is on disk).
    if let Some(pid) = project_id {
        let auto = engine.config.auto.clone();
        if auto.graph || auto.docs {
            std::thread::spawn(move || {
                let Ok(mut m) = Mimir::open() else { return };
                let Ok(proj) = mimir_core::store::get_node(&m.conn, pid) else {
                    return;
                };
                let Some(root) = proj.path.clone() else {
                    return;
                };
                let root = std::path::PathBuf::from(root);
                if auto.graph {
                    match mimir_graph::update(&mut m.conn, &proj, &root) {
                        Ok(s) => tracing::info!(
                            symbols = s.symbols,
                            files_indexed = s.files_indexed,
                            "auto graph sync"
                        ),
                        Err(e) => tracing::warn!("auto graph sync failed: {e}"),
                    }
                }
                if auto.docs {
                    let name = proj.title.clone().unwrap_or_else(|| "project".into());
                    let indexed =
                        mimir_core::index::add_collection(&m.conn, &root, &name, Some(pid))
                            .and_then(|c| mimir_core::index::index_collection(&mut m.conn, &c));
                    match indexed {
                        Ok(s) => {
                            tracing::info!(files = s.seen, chunks = s.chunks, "auto docs sync")
                        }
                        Err(e) => tracing::warn!("auto docs sync failed: {e}"),
                    }
                }
                if let Ok(n) = m.embed_pending() {
                    if n > 0 {
                        tracing::info!(embedded = n, "auto embed");
                    }
                }
            });
        }
    }

    // Weekly consolidation runs off the request path: separate thread,
    // separate connection (WAL handles the concurrency).
    {
        let cadence = engine.config.consolidate.auto.clone();
        let model = engine.config.embedding.model.clone();
        let db_path = engine.paths.db_file.clone();
        std::thread::spawn(move || {
            if let Ok(conn) = mimir_core::db::open(&db_path) {
                if let Some(r) = mimir_core::consolidate::maybe_auto(&conn, &model, &cadence) {
                    tracing::info!(
                        superseded = r.superseded,
                        distilled = r.distilled,
                        archived = r.archived,
                        contradictions = r.contradictions.len(),
                        "auto-consolidation ran"
                    );
                }
            }
        });
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let service = MimirServer::new(engine, project_id)
            .serve(rmcp::transport::stdio())
            .await?;
        service.waiting().await?;
        Ok(())
    })
}
