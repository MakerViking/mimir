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
not session noise. Use get to read full bodies (this also teaches Mimir what is useful). \
To understand code cheaply, prefer `outline` (a file/dir's signature map) and `peek` (one symbol's \
body) over reading whole files — they cost a fraction of the tokens.";

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

#[derive(Deserialize, schemars::JsonSchema)]
pub struct OutlineArgs {
    /// File or directory to outline, relative to the project (default: whole project).
    #[serde(default)]
    pub target: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct PeekArgs {
    /// Symbol: short id, qualified name, or unique bare name.
    pub symbol: String,
}

/// Filesystem root for the detected project (None outside any project).
fn project_root(m: &Mimir, project_id: Option<i64>) -> Option<std::path::PathBuf> {
    let id = project_id?;
    store::get_node(&m.conn, id)
        .ok()?
        .path
        .map(std::path::PathBuf::from)
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
        description = "Dense signature map of a file or directory: one line per symbol (signature + first doc line), no bodies. Prefer this over reading a whole file when you only need its structure — it costs a fraction of the tokens."
    )]
    async fn outline(&self, Parameters(args): Parameters<OutlineArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let cwd = project_root(m, project_id)
                .or_else(|| std::env::current_dir().ok())
                .ok_or("cannot resolve a working directory")?;
            let target = args.target.as_deref().unwrap_or(".");
            let o = crate::context_cmd::outline_paths(target, &cwd).map_err(engine_err)?;
            let _ = mimir_core::savings::record(
                &m.conn,
                project_id,
                mimir_core::savings::source::OUTLINE,
                o.tokens_before,
                o.tokens_after,
                serde_json::json!({ "target": target, "files": o.files }),
            );
            Ok(o.text)
        })
        .await
    }

    #[tool(
        description = "Print one symbol's body by name (resolved via the project's code graph), instead of reading the whole file. Build the graph first with `mimir graph build`."
    )]
    async fn peek(&self, Parameters(args): Parameters<PeekArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let Some(pid) = project_id else {
                return Err("no project detected (peek needs the per-project code graph)".into());
            };
            let root = project_root(m, Some(pid)).ok_or("project has no root")?;
            let p = crate::context_cmd::peek_symbol(&m.conn, pid, &root, &args.symbol)
                .map_err(engine_err)?;
            let _ = mimir_core::savings::record(
                &m.conn,
                Some(pid),
                mimir_core::savings::source::PEEK,
                p.tokens_before,
                p.tokens_after,
                serde_json::json!({ "symbol": args.symbol, "path": p.path }),
            );
            // peek_symbol already resolved it — record the access without a re-lookup.
            let _ = mimir_core::learn::record_opened(&m.conn, p.symbol_id);
            Ok(p.text)
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
            // Read-only detection just for the "[via: …]" reason (no DB write).
            let detection = server_cwd().map(|d| mimir_core::scope::detect(&d));
            let project = match project_id {
                Some(id) => {
                    let name = store::get_node(&m.conn, id)
                        .ok()
                        .and_then(|p| p.title)
                        .unwrap_or_else(|| "?".into());
                    let via = match &detection {
                        Some(mimir_core::scope::Detection::Found { via, .. }) => {
                            mimir_core::scope::via_label(via)
                        }
                        _ => "?",
                    };
                    format!("{name}  [via: {via}]")
                }
                None => match &detection {
                    Some(mimir_core::scope::Detection::NotFound { from }) => format!(
                        "(none) — no marker above {}; using global scope",
                        from.display()
                    ),
                    _ => "(none — global scope)".into(),
                },
            };
            let db_size = std::fs::metadata(&m.paths.db_file)
                .map(|md| md.len())
                .map_err(engine_err)?;
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

/// Where the server scopes to: `MIMIR_PROJECT` env beats the launch cwd.
fn server_cwd() -> Option<std::path::PathBuf> {
    std::env::var_os("MIMIR_PROJECT")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
}

/// Entry point for `mimir mcp` (blocking; owns the tokio runtime).
/// `http = Some(addr)` serves Streamable-HTTP instead of stdio.
pub fn run(http: Option<String>, http_allow_remote: bool) -> Result<()> {
    let engine = Mimir::open()?;
    let project_id = match server_cwd() {
        Some(d) => match engine.detect_project(&d) {
            Ok((node, detection)) => {
                // Never degrade silently: always log how scope was resolved.
                match &detection {
                    mimir_core::scope::Detection::Found { root, via } => tracing::info!(
                        project = %root.display(),
                        via = mimir_core::scope::via_label(via),
                        "scoped to project"
                    ),
                    mimir_core::scope::Detection::NotFound { from } => tracing::info!(
                        from = %from.display(),
                        "no project marker found; serving GLOBAL scope"
                    ),
                }
                node.map(|p| p.id)
            }
            Err(err) => {
                // Wrong-scoped memories are corruption nobody notices.
                tracing::warn!(%err, "project detection failed; serving GLOBAL scope only");
                None
            }
        },
        None => None,
    };

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

    // Reclaim the WAL on an idle timer: `mimir mcp` is long-lived, so its read
    // marks can starve PASSIVE autocheckpoints and let the -wal grow unbounded.
    mimir_core::db::spawn_checkpoint_timer(engine.paths.db_file.clone());

    // Optional background sync (own connection, non-fatal, WAL-safe). Logs via
    // tracing, never stdout — stdout is the MCP protocol channel. Gated on the
    // [sync] config so there is zero cost when sync is off.
    if engine.config.sync.enabled() && engine.config.sync.auto {
        let interval = engine.config.sync.interval_mins.max(1);
        std::thread::spawn(move || loop {
            match mimir_core::Mimir::open() {
                Ok(mut m) => match crate::sync::perform(&mut m) {
                    Ok(summary) => tracing::info!("auto-sync: {summary}"),
                    Err(e) => tracing::warn!("auto-sync failed: {e}"),
                },
                Err(e) => tracing::warn!("auto-sync open failed: {e}"),
            }
            std::thread::sleep(std::time::Duration::from_secs(interval * 60));
        });
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        match http {
            Some(bind) => serve_http(project_id, &bind, http_allow_remote).await,
            None => {
                let service = MimirServer::new(engine, project_id)
                    .serve(rmcp::transport::stdio())
                    .await?;
                service.waiting().await?;
                Ok(())
            }
        }
    })
}

/// Serve the same MCP tools over Streamable-HTTP for remote clients reached
/// through a tunnel / reverse proxy. Each session gets its own engine (its own
/// SQLite connection; WAL handles the concurrency). This transport carries NO
/// auth — the loopback bind is the security boundary, so non-loopback binds
/// are refused unless `--http-allow-remote` says the exposure is deliberate.
async fn serve_http(project_id: Option<i64>, bind: &str, allow_remote: bool) -> Result<()> {
    if !bind_is_loopback(bind) {
        if !allow_remote {
            anyhow::bail!(
                "refusing to bind MCP to non-loopback address `{bind}`: this transport has no \
                 auth, so anyone who can reach the port gets full read/write access to the \
                 store. Bind to 127.0.0.1 and front it with TLS + an auth gate (tunnel / \
                 reverse proxy), or pass --http-allow-remote if the port is otherwise \
                 unreachable (e.g. a container port published to 127.0.0.1 only)."
            );
        }
        tracing::warn!(
            %bind,
            "serving unauthenticated MCP on a non-loopback address (--http-allow-remote); \
             make sure an auth gate fronts this port"
        );
    }
    let app = mcp_http_router(move || {
        Ok(MimirServer::new(
            Mimir::open().map_err(std::io::Error::other)?,
            project_id,
        ))
    });
    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!("mimir MCP (streamable-http) listening on http://{bind}/mcp");
    axum::serve(listener, app).await?;
    Ok(())
}

/// True only when every address `bind` resolves to is loopback. Unresolvable
/// binds return false — the TcpListener will produce the real error, and the
/// gate stays fail-closed.
fn bind_is_loopback(bind: &str) -> bool {
    use std::net::ToSocketAddrs;
    match bind.to_socket_addrs() {
        Ok(addrs) => {
            let mut any = false;
            for addr in addrs {
                if !addr.ip().is_loopback() {
                    return false;
                }
                any = true;
            }
            any
        }
        Err(_) => false,
    }
}

/// Build the axum router that serves the MCP tools over Streamable-HTTP at
/// `/mcp`. `make_server` runs once per session (its own engine/connection);
/// split out so a test can mount it on an ephemeral port with a temp store.
fn mcp_http_router<F>(make_server: F) -> axum::Router
where
    F: Fn() -> std::result::Result<MimirServer, std::io::Error> + Send + Sync + 'static,
{
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    let service = StreamableHttpService::new(
        make_server,
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    axum::Router::new().nest_service("/mcp", service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::config::Paths;

    /// A server backed by a real temp-dir DB (status reads the db file), in
    /// global scope (project_id = None), so the tool handlers exercise the
    /// same paths an agent hits.
    fn server() -> (tempfile::TempDir, MimirServer) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        (dir, MimirServer::new(engine, None))
    }

    /// The gate refuses everything that isn't provably loopback, including
    /// unresolvable input (fail-closed).
    #[test]
    fn loopback_gate() {
        assert!(bind_is_loopback("127.0.0.1:8077"));
        assert!(bind_is_loopback("[::1]:8077"));
        assert!(bind_is_loopback("localhost:8077"));
        assert!(!bind_is_loopback("0.0.0.0:8077"));
        assert!(!bind_is_loopback("192.168.1.10:8077"));
        assert!(!bind_is_loopback("not an address"));
    }

    /// The Streamable-HTTP transport serves the same tools as stdio: drive a
    /// real MCP initialize + tools/list handshake over an ephemeral port,
    /// against a temp store so the real user's db is never touched.
    #[tokio::test(flavor = "multi_thread")]
    async fn http_transport_serves_mcp_tools() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under_root(dir.path());
        let router = mcp_http_router(move || {
            Ok(MimirServer::new(
                Mimir::open_at(paths.clone()).map_err(std::io::Error::other)?,
                None,
            ))
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let url = format!("http://{addr}/mcp");

        // initialize: a stateful Streamable-HTTP server returns a session id.
        let init_url = url.clone();
        let (sid, body) = tokio::task::spawn_blocking(move || {
            let agent = ureq::AgentBuilder::new()
                .timeout_read(Duration::from_secs(10))
                .build();
            let resp = agent
                .post(&init_url)
                .set("Content-Type", "application/json")
                .set("Accept", "application/json, text/event-stream")
                .send_string(
                    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#,
                )
                .expect("initialize ok");
            let sid = resp.header("mcp-session-id").unwrap_or_default().to_string();
            (sid, resp.into_string().unwrap_or_default())
        })
        .await
        .unwrap();
        assert!(body.contains("protocolVersion"), "initialize: {body}");
        assert!(!sid.is_empty(), "stateful mode returns a session id");

        // tools/list over the same session exposes the real tool set.
        let tools = tokio::task::spawn_blocking(move || {
            let agent = ureq::AgentBuilder::new()
                .timeout_read(Duration::from_secs(10))
                .build();
            let send = |b: &str| -> String {
                agent
                    .post(&url)
                    .set("Content-Type", "application/json")
                    .set("Accept", "application/json, text/event-stream")
                    .set("Mcp-Session-Id", &sid)
                    .send_string(b)
                    .map(|r| r.into_string().unwrap_or_default())
                    .unwrap_or_default()
            };
            let _ = send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
            send(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
        })
        .await
        .unwrap();
        assert!(
            tools.contains("recall") && tools.contains("remember"),
            "tools/list: {tools}"
        );
    }

    fn remember_args(text: &str, ty: &str) -> RememberArgs {
        RememberArgs {
            text: text.into(),
            r#type: Some(ty.into()),
            tags: vec!["auth".into()],
            global: true,
            link: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_recall_get_roundtrip() {
        let (_dir, s) = server();
        let stored = s
            .remember(Parameters(remember_args(
                "SCRAM auth rejects non-ASCII passwords",
                "gotcha",
            )))
            .await;
        assert!(stored.starts_with("stored"), "remember: {stored}");
        let id = stored.split_whitespace().nth(1).unwrap().to_string();

        let hits = s
            .recall(Parameters(RecallArgs {
                query: "postgres password trouble".into(),
                kind: None,
                limit: None,
                all_projects: true,
                rerank: false,
            }))
            .await;
        assert!(hits.contains("SCRAM"), "recall: {hits}");

        let body = s.get(Parameters(GetArgs { r#ref: id })).await;
        assert!(body.contains("SCRAM"), "get: {body}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn near_duplicate_is_refused() {
        let (_dir, s) = server();
        let first = s
            .remember(Parameters(remember_args(
                "we picked SQLite for one file",
                "decision",
            )))
            .await;
        assert!(first.starts_with("stored"), "{first}");
        let dup = s
            .remember(Parameters(remember_args(
                "we picked SQLite for one file",
                "decision",
            )))
            .await;
        assert!(dup.contains("refused"), "duplicate not refused: {dup}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mark_link_and_status() {
        let (_dir, s) = server();
        let a = s
            .remember(Parameters(remember_args(
                "alpha decision about caching",
                "decision",
            )))
            .await;
        let b = s
            .remember(Parameters(remember_args(
                "beta gotcha about caching",
                "gotcha",
            )))
            .await;
        let id_a = a.split_whitespace().nth(1).unwrap().to_string();
        let id_b = b.split_whitespace().nth(1).unwrap().to_string();

        let marked = s
            .mark(Parameters(MarkArgs {
                r#ref: id_a.clone(),
                useful: true,
            }))
            .await;
        assert!(marked.contains("useful"), "mark: {marked}");

        let linked = s
            .link(Parameters(LinkArgs {
                a: id_a,
                b: id_b,
                rel: Some("relates".into()),
            }))
            .await;
        assert!(!linked.starts_with("error"), "link: {linked}");

        let status = s.status().await;
        assert!(status.contains("memory"), "status: {status}");
        assert!(status.contains("KB"), "status: {status}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bad_reference_errors_cleanly() {
        let (_dir, s) = server();
        // A missing id must return an error string, not panic or wedge.
        let got = s
            .get(Parameters(GetArgs {
                r#ref: "m:ZZZZZZ".into(),
            }))
            .await;
        assert!(got.starts_with("error"), "expected error: {got}");
    }
}
