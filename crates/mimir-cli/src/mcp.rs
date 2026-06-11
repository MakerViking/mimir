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
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct GetArgs {
    /// Node id (m:ABCDEF), full ULID, or indexed file path like docs/x.md:10-40.
    pub r#ref: String,
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
        description = "Store a typed memory (gotcha/decision/insight/idea/note/person). Search first with recall; near-duplicates are refused with the existing entry."
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
                    Ok(format!("stored {}", line(&node)))
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
            store::touch_accessed(&m.conn, node.id).map_err(engine_err)?;
            store::record_event(&m.conn, node.id, "opened", None, None, None)
                .map_err(engine_err)?;
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

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let service = MimirServer::new(engine, project_id)
            .serve(rmcp::transport::stdio())
            .await?;
        service.waiting().await?;
        Ok(())
    })
}
