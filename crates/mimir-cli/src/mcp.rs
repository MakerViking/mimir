//! MCP stdio server (`mimir mcp`): one global registration serves every
//! repo. Project scope comes from the server's cwd (where the agent
//! launched it), overridable via `MIMIR_PROJECT=<path>`.
//!
//! Returns are token-lean agent-format text, not JSON. The engine is
//! sync; every tool hops through spawn_blocking.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use mimir_core::format::{agent_line, agent_line_for_query};
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
    /// Search text (natural language or keywords).
    pub query: String,
    /// Filter: all | memory | doc | code (default all).
    #[serde(default)]
    pub kind: Option<String>,
    /// Max results (default 8).
    #[serde(default)]
    pub limit: Option<usize>,
    /// true = search every project (default: current + global).
    #[serde(default)]
    pub all_projects: bool,
    /// true = cross-encoder rerank (slower, more accurate; needs model on disk).
    #[serde(default)]
    pub rerank: bool,
    /// true = include superseded memories (hidden by default).
    #[serde(default)]
    pub include_superseded: bool,
    /// Answer as the store stood on this date (YYYY-MM-DD): memories deleted
    /// or superseded since come back, ones written later are absent. This is
    /// what you *knew* then, not what was *true* then — a memory's own
    /// validity interval is a separate thing, shown by `get`.
    #[serde(default)]
    pub as_of: Option<String>,
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
    /// true = global (cross-project), not current-project scope.
    #[serde(default)]
    pub global: bool,
    /// Code symbol or node this memory is about (name or id).
    #[serde(default)]
    pub link: Option<String>,
    /// Trigger phrases (e.g. "release checklist"); a matching future prompt
    /// auto-injects this memory even without keyword overlap. Max 8, 6 words
    /// each; oversized ones are dropped (not truncated).
    #[serde(default)]
    pub fires_when: Vec<String>,
    /// File/command glob patterns (e.g. "deploy.sh", "src/store.rs") that
    /// surface this memory at act-time via the PreToolUse guard when the
    /// agent edits a matching file or runs a matching command. Up to 8,
    /// 200 chars each; oversized/empty dropped. Optional "file:" prefix.
    #[serde(default)]
    pub anchors: Vec<String>,
    /// Relative duration after which this memory stops being true — e.g.
    /// "90d", "2w", "12h". Units h/d/w only (months/years have no fixed
    /// length; use "365d"). Once passed the memory is hidden from recall
    /// like a superseded one. Only set it when the fact genuinely has a
    /// deadline; a wrong expiry silently deletes something still true.
    #[serde(default)]
    pub expires_in: Option<String>,
    /// Free-text condition that would falsify this memory — e.g. "when
    /// upstream ships the fix". Does NOT hide the memory (nothing can
    /// evaluate it); it rides along so a future reader sees what would
    /// make it wrong. Prefer this over expires_in whenever the end
    /// condition is an event rather than a date.
    #[serde(default)]
    pub resolves_when: Option<String>,
    /// How sure you are that this is true: "certain", "likely" or
    /// "unsure". Record what you actually knew when you wrote it — this is
    /// tracked separately from how often the memory gets used, so a guess
    /// stays legible as a guess however much it is later recalled. Leave
    /// unset rather than guessing a level; absent means undeclared, not
    /// medium. Does not affect ranking.
    #[serde(default)]
    pub confidence: Option<String>,
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
    /// Symbol (name, qualified name, or id); for callers/calls/node/path.
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

#[derive(Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    /// Node reference (m:ABCDEF or ULID) to delete.
    pub r#ref: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct ConsolidateArgs {
    /// true (default) = report only. false = apply for real — take a backup first.
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct SupersedeArgs {
    /// The OLD node reference to retire.
    pub old: String,
    /// The NEW node reference that replaces it.
    pub by: String,
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
        description = "Search memories, docs and code in one ranked list (hybrid BM25+semantic). ~25-token line per hit."
    )]
    async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let project_id = self.project_id;
        self.blocking(move |m| {
            let kinds = match args.kind.as_deref().unwrap_or("all") {
                "all" => vec![],
                "memory" => vec![Kind::Memory],
                "doc" => vec![Kind::File, Kind::Chunk, Kind::Annotation],
                // Symbol = signature/doc only; CodeChunk = actual body/content
                // text (see chunker::chunk_source). Both belong under `code`.
                "code" => vec![Kind::Symbol, Kind::CodeChunk],
                other => return Err(format!("unknown kind '{other}' (all|memory|doc|code)")),
            };
            let scope = if args.all_projects {
                Scope::All
            } else {
                project_id.map(Scope::Project).unwrap_or(Scope::All)
            };
            let as_of = match args.as_of.as_deref() {
                Some(s) => Some(mimir_core::format::parse_date(s).ok_or_else(|| {
                    engine_err(mimir_core::error::Error::Invalid(format!(
                        "bad as_of '{s}': expected a date like 2026-03-01"
                    )))
                })?),
                None => None,
            };
            let query = SearchQuery {
                text: args.query,
                scope,
                kinds,
                since: None,
                limit: args.limit.unwrap_or(m.config.output.default_limit),
                strength_alpha: m.config.scoring.strength_alpha,
                recency_alpha: m.config.scoring.recency_alpha,
                type_prior_alpha: m.config.scoring.type_prior_alpha,
                code_damp: m.config.scoring.code_damp,
                include_superseded: args.include_superseded,
                as_of,
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
            let ids: Vec<i64> = hits.iter().map(|h| h.node.id).collect();
            let stale = mimir_core::grounding::stale_ids(&m.conn, &ids).unwrap_or_default();
            let mut out = Vec::new();
            for hit in &hits {
                let project = hit
                    .node
                    .project_id
                    .and_then(|id| projects.get(&id))
                    .map(String::as_str);
                out.push(agent_line_for_query(
                    &hit.node,
                    project,
                    m.config.output.snippet_chars,
                    Some(&query.text),
                    stale.contains(&hit.node.id),
                ));
            }
            Ok(out.join("\n"))
        })
        .await
    }

    #[tool(
        description = "Store a typed memory. Pass `link` to attach it to a code symbol/file. Pass `fires_when` to force-inject on trigger phrases without keyword overlap. Pass `anchors` to surface it at act-time when a matching file is edited or command is run."
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
            if let Some(kind) =
                mimir_core::secrets::scan_capture(&args.text, &args.tags, &args.fires_when)
            {
                mimir_core::secrets::record_refusal(
                    &m.conn,
                    &mimir_core::secrets::capture_hash(
                        &args.text,
                        &args.tags,
                        &args.fires_when,
                    ),
                    kind,
                    mimir_core::secrets::Surface::McpRemember,
                    mimir_core::model::now_unix(),
                );
                return Err(engine_err(mimir_core::error::Error::Secret(kind)));
            }
            let outcome = memory::remember(
                &m.conn,
                memory::Remember {
                    text: args.text,
                    mtype,
                    tags: args.tags,
                    project_id: if args.global { None } else { project_id },
                    force: false,
                    actor: mimir_core::model::Actor::Agent,
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
                    if !args.fires_when.is_empty() {
                        let phrases = memory::sanitize_fires_when(args.fires_when);
                        if !phrases.is_empty() {
                            store::set_fires_when(&m.conn, node.id, &phrases)
                                .map_err(engine_err)?;
                        }
                    }
                    if !args.anchors.is_empty() {
                        let patterns = mimir_core::anchors::sanitize_anchors(args.anchors);
                        if !patterns.is_empty() {
                            mimir_core::anchors::set_anchors(&m.conn, node.id, &patterns)
                                .map_err(engine_err)?;
                        }
                    }
                    // Reject a bad duration instead of storing a memory that
                    // silently never expires — the caller asked for a deadline
                    // and has no other way to learn it wasn't applied.
                    let expires_at = match args.expires_in.as_deref() {
                        Some(spec) => Some(
                            memory::parse_expires_in(spec, mimir_core::model::now_unix())
                                .ok_or_else(|| {
                                    engine_err(mimir_core::error::Error::Invalid(format!(
                                        "invalid expires_in {spec:?}: expected a positive \
                                         duration like 90d, 2w or 12h (units h/d/w)"
                                    )))
                                })?,
                        ),
                        None => None,
                    };
                    if expires_at.is_some() || args.resolves_when.is_some() {
                        store::set_expiry(
                            &m.conn,
                            node.id,
                            expires_at,
                            args.resolves_when.as_deref(),
                        )
                        .map_err(engine_err)?;
                    }
                    // Same contract as expires_in: an unparseable level is
                    // an error, not a silently undeclared memory.
                    if let Some(spec) = args.confidence.as_deref() {
                        let level: mimir_core::model::MemoryConfidence =
                            spec.parse().map_err(|_| {
                                engine_err(mimir_core::error::Error::Invalid(format!(
                                    "invalid confidence {spec:?}: expected certain, likely \
                                     or unsure"
                                )))
                            })?;
                        store::set_confidence(&m.conn, node.id, level).map_err(engine_err)?;
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
                RememberOutcome::Forgotten(gone) => Ok(format!(
                    "refused: a human forgot this on {}\n{}\n(it was deleted on purpose — do not re-add it \
                     unless the user asks; if they do, they can run `mimir remember --force`)",
                    mimir_core::format::full_date(gone.deleted_at.unwrap_or(gone.updated_at)),
                    line(&gone)
                )),
            }
        })
        .await
    }

    #[tool(
        description = "Full record for a recall hit (body, tags, links); or an indexed doc path for a line range."
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
        description = "Link two nodes (memory/doc/code) so context surfaces together. Note: `supersedes` is NOT accepted here — use the `supersede` tool, which also retires the old node."
    )]
    async fn link(&self, Parameters(args): Parameters<LinkArgs>) -> String {
        self.blocking(move |m| {
            let rel: Rel = args
                .rel
                .as_deref()
                .unwrap_or("relates")
                .parse()
                .map_err(engine_err)?;
            // A `supersedes` EDGE and the `superseded_by` COLUMN are separate
            // mechanisms and only the column suppresses recall. Accepting the
            // rel here produced a link that reads exactly like a retirement
            // and silently did nothing — one such edge sat in the store for
            // six weeks while the "retired" memory kept ranking first.
            if rel == Rel::Supersedes {
                return Err(engine_err(mimir_core::error::Error::Invalid(
                    "link does not accept rel=\"supersedes\": it records an edge but does \
                     NOT retire the old node, so it would look like a supersede and change \
                     nothing. Use the `supersede` tool instead (it sets both)."
                        .into(),
                )));
            }
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
        description = "Code-graph queries: callers/calls, impact (blast radius from files, e.g. git diff --name-only), node (record), path (between symbols), hubs (most-called). Run `mimir graph build` first."
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
        description = "Dense signature map of a file or directory: one line per symbol (signature + first doc line), no bodies."
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
        description = "Print one symbol's body by name via the code graph. Run `mimir graph build` first."
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
        description = "Feedback on a recalled node: useful strengthens ranking, noise weakens it."
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
            let sync = if m.config.sync.enabled() {
                let push = mimir_core::replicate::get_watermark(&m.conn, "last_push").unwrap_or(0);
                let pull = mimir_core::replicate::get_watermark(&m.conn, "last_pull").unwrap_or(0);
                let cadence = if m.config.sync.auto {
                    format!("auto every {} min", m.config.sync.interval_mins)
                } else {
                    "manual".into()
                };
                format!(
                    "\nsync    {} {} ({cadence}); local watermarks push={push} pull={pull}",
                    m.config.sync.mode, m.config.sync.endpoint
                )
            } else {
                "\nsync    off (local store only)".into()
            };
            Ok(format!(
                "project {project}\nstore   {summary}\ndb      {} KB{sync}",
                db_size / 1024
            ))
        })
        .await
    }

    #[tool(
        description = "Soft-delete (reversible tombstone: hidden from recall, synced across machines). No hard delete over MCP — use `mimir forget --hard` on the CLI."
    )]
    async fn forget(&self, Parameters(args): Parameters<ForgetArgs>) -> String {
        self.blocking(move |m| {
            let node = store::resolve_ref(&m.conn, &args.r#ref).map_err(engine_err)?;
            store::soft_delete(&m.conn, node.id, mimir_core::model::Actor::Agent, None)
                .map_err(engine_err)?;
            Ok(format!(
                "forgot {} {}",
                short_uid(node.kind, &node.uid),
                node.title.as_deref().unwrap_or("")
            ))
        })
        .await
    }

    #[tool(
        description = "Mark OLD superseded by NEW (stops surfacing in recall, kept as history) via a `supersedes` edge. Prefer over `forget` when NEW replaces OLD."
    )]
    async fn supersede(&self, Parameters(args): Parameters<SupersedeArgs>) -> String {
        self.blocking(move |m| {
            let old = store::resolve_ref(&m.conn, &args.old).map_err(engine_err)?;
            let new = store::resolve_ref(&m.conn, &args.by).map_err(engine_err)?;
            store::set_superseded(
                &m.conn,
                old.id,
                new.id,
                mimir_core::model::Actor::Agent,
                None,
            )
            .map_err(engine_err)?;
            store::link(&m.conn, new.id, old.id, Rel::Supersedes, 1.0).map_err(engine_err)?;
            Ok(format!(
                "{} superseded by {}",
                short_uid(old.kind, &old.uid),
                short_uid(new.kind, &new.uid)
            ))
        })
        .await
    }

    #[tool(
        description = "Run consolidation: dedup -> supersede near-duplicates, distill clusters, archive decayed memories. Contradictions are reported only, never auto-resolved."
    )]
    async fn consolidate(&self, Parameters(args): Parameters<ConsolidateArgs>) -> String {
        self.blocking(move |m| {
            let report = mimir_core::consolidate::consolidate(
                &m.conn,
                &m.config.embedding.model,
                args.dry_run,
            )
            .map_err(engine_err)?;
            let prefix = if args.dry_run { "would " } else { "" };
            if report.is_empty() {
                return Ok("nothing to consolidate".into());
            }
            let mut out = Vec::new();
            if report.superseded > 0 {
                out.push(format!(
                    "{prefix}supersede {} near-duplicate(s)",
                    report.superseded
                ));
            }
            if report.distilled > 0 {
                out.push(format!(
                    "{prefix}distill {} cluster(s) into summaries",
                    report.distilled
                ));
            }
            if report.archived > 0 {
                out.push(format!(
                    "{prefix}archive {} decayed memorie(s)",
                    report.archived
                ));
            }
            for (a, b) in &report.contradictions {
                out.push(format!(
                    "possible contradiction (review by hand):\n  {a}\n  {b}"
                ));
            }
            Ok(out.join("\n"))
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

/// Read `MIMIR_HTTP_TOKEN`, treating empty-but-set as unset. One definition
/// shared by `mimir mcp` and `mimir daemon` so the two entry points to the
/// same HTTP surface can't drift on auth.
pub fn http_token_from_env() -> Option<String> {
    crate::sync::env_nonempty("MIMIR_HTTP_TOKEN")
}

/// Entry point for `mimir mcp` (blocking; owns the tokio runtime).
/// `http = Some(addr)` serves Streamable-HTTP instead of stdio. `http_token`,
/// when set, requires an `Authorization: Bearer <token>` header on the HTTP
/// transport (defense-in-depth on top of the loopback bind; ignored for stdio).
pub fn run(
    http: Option<String>,
    http_allow_remote: bool,
    http_token: Option<String>,
) -> Result<()> {
    let engine = Mimir::open()?;
    // Inference delegation (default on): background engines and the stdio
    // session run their embedder on CPU and route bulk embeds/reranks to a
    // running `mimir daemon`, so GPU device create/teardown happens in ONE
    // process instead of every session and background tick.
    //
    // `http.is_none()` because a process serving `--http` IS the daemon:
    // delegating would point its own background engines back at its own
    // endpoint. That "works" (the client fails and falls back to local), but
    // it costs a doomed HTTP round-trip per tick and — because the background
    // auto-sync thread starts before the listener binds — logs a spurious
    // `daemon embed failed ... Connection refused` on every single daemon
    // start, which reads like a real fault in the journal.
    let delegate = engine.config.daemon.inference == "auto" && http.is_none();
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
                if delegate {
                    m.set_embedder_device("cpu");
                    crate::infer::attach_lazy(&mut m);
                }
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
                        mimir_core::index::add_collection(&m.conn, &root, &name, Some(pid), "docs")
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
    // Same timer also prunes old `recall_event` rows (`[learn]
    // event_retention_days`, clamped/warned if configured unsafely low).
    mimir_core::db::spawn_checkpoint_timer(
        engine.paths.db_file.clone(),
        engine.config.learn.effective_retention_days(),
    );

    // Optional background sync (own connection, non-fatal, WAL-safe). Logs via
    // tracing, never stdout — stdout is the MCP protocol channel. Gated on the
    // [sync] config so there is zero cost when sync is off.
    if engine.config.sync.enabled() && engine.config.sync.auto {
        let interval = engine.config.sync.interval_mins.max(1);
        std::thread::spawn(move || loop {
            match mimir_core::Mimir::open() {
                Ok(mut m) => {
                    if delegate {
                        m.set_embedder_device("cpu");
                        crate::infer::attach_lazy(&mut m);
                    }
                    match crate::sync::perform(&mut m) {
                        Ok(summary) => tracing::info!("auto-sync: {summary}"),
                        Err(e) => tracing::warn!("auto-sync failed: {e}"),
                    }
                }
                Err(e) => tracing::warn!("auto-sync open failed: {e}"),
            }
            std::thread::sleep(std::time::Duration::from_secs(interval * 60));
        });
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        match http {
            Some(bind) => serve_http(project_id, &bind, http_allow_remote, http_token).await,
            None => {
                // Eager-load here, not up front in `run()`: in HTTP mode this
                // `engine` is only used for the project-detection/background
                // work above and never actually serves requests, so loading
                // the reranker onto it would just be a wasted cold load.
                let mut engine = engine;
                // With a live daemon this session holds zero GPU state: CPU
                // embedder (measured faster for batch-1 query embeds than
                // GPU) and no local reranker — `[rerank] auto = "warm"`
                // fires through the daemon instead. Daemon unreachable at
                // startup (or delegation off): exactly the old behavior.
                let daemon_alive = crate::infer::attach(&mut engine);
                if delegate {
                    engine.set_embedder_device("cpu");
                }
                // Record which path this session took. Both outcomes "work",
                // so at runtime they are indistinguishable — the only visible
                // symptom of a fallback is a few hundred MB of resident
                // reranker that nobody can attribute to a cause days later.
                // The decision is made once, here, and is sticky for the life
                // of the session, which is exactly why it needs a record.
                if !delegate {
                    tracing::info!(
                        inference = %engine.config.daemon.inference,
                        "inference: local only (delegation not enabled)"
                    );
                } else if daemon_alive {
                    tracing::info!(
                        "inference: delegating to daemon (CPU embedder, no local reranker)"
                    );
                } else {
                    let eager = matches!(engine.config.rerank.auto.as_str(), "warm" | "always");
                    tracing::warn!(
                        rerank_auto = %engine.config.rerank.auto,
                        loads_local_reranker = eager,
                        "inference: daemon unreachable at session start; this session stays \
                         local for its entire life (delegation is not re-probed)"
                    );
                }
                if !daemon_alive {
                    eager_load_reranker_if_auto(&mut engine);
                }
                // Armed BEFORE `serve`, which awaits the MCP `initialize`
                // handshake and therefore blocks indefinitely against a client
                // that spawns the server and dies without ever initializing —
                // an orphan window that arming afterwards cannot see. Stdio
                // only: the HTTP path is a daemon with its own lifetime.
                #[cfg(unix)]
                arm_shutdown_watchdog();
                let service = MimirServer::new(engine, project_id)
                    .serve(rmcp::transport::stdio())
                    .await?;
                #[cfg(unix)]
                register_shutdown_token(service.cancellation_token());
                service.waiting().await?;
                Ok(())
            }
        }
    })
}

/// The parent pid as it was at process start.
///
/// Captured in `main`, microseconds after exec, and deliberately not where
/// it is used. The stdio server spends a second or two loading models before
/// it starts serving, and a client that exits during that window is already
/// gone by then — a ppid read taken at that point returns the *reaper*, which
/// then compares equal to itself forever and the watchdog never fires. That
/// is not hypothetical: it is what the first version of this did, and the
/// regression test caught it.
#[cfg(unix)]
static PARENT_AT_START: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Record the parent pid. Call once, as early in `main` as possible.
#[cfg(unix)]
pub fn record_parent_pid() {
    PARENT_AT_START.store(
        unsafe { libc::getppid() },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Set by the SIGTERM/SIGHUP handler. A relaxed store is the whole handler
/// body on purpose — it is the only thing here that is async-signal-safe.
#[cfg(unix)]
static SHUTDOWN_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn note_shutdown(_sig: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Exit when the client goes away, for the cases where stdin EOF never comes.
///
/// The stdio server's EOF handling is correct and needs no help: rmcp's
/// transport returns `None` on a zero-length read and the service task ends.
/// The problem is that the EOF can simply never arrive. Claude Code wires MCP
/// stdio over an `AF_UNIX` **socketpair**, not a pipe, and a socketpair
/// delivers EOF only once *every* descriptor for the peer end is closed. Any
/// unrelated process that inherited that descriptor — a background command,
/// another server spawned by the same client — pins it open, so when the
/// client exits the kernel delivers nothing at all and the reader blocks
/// forever. Measured before this existed: fourteen orphans, the oldest
/// fourteen days old, ~2.9 GB resident between them, each holding the SQLite
/// database open.
///
/// So the trigger cannot be a descriptor. It has to be the parent dying:
///
/// * `PR_SET_PDEATHSIG` (Linux) has the kernel send us SIGTERM the moment the
///   parent does, regardless of who else holds the socket.
/// * A `getppid()` poll covers macOS, which has no equivalent, and closes the
///   race where the parent died in the window *before* `prctl` ran — that
///   signal is never sent, so nothing but a check would catch it.
///
/// The poll compares against the ppid captured at startup rather than testing
/// for `1`, because a server legitimately started by pid 1 would otherwise
/// shut itself down immediately.
///
/// Cancelling the token ends the service task, which resolves the `waiting()`
/// the caller is already parked on, so shutdown runs through exactly the same
/// path as a clean EOF: in-flight work finishes, the transport closes, the
/// engine drops and with it the database handle.
/// Where `arm_shutdown_watchdog` finds the token once there is one. Empty
/// until the handshake completes.
#[cfg(unix)]
static SHUTDOWN_TOKEN: std::sync::Mutex<Option<rmcp::service::RunningServiceCancellationToken>> =
    std::sync::Mutex::new(None);

/// Hand the watchdog a way to shut the service down gracefully. Before this,
/// it can only exit the process — which is correct, because before the
/// handshake there is no service to drain.
#[cfg(unix)]
fn register_shutdown_token(token: rmcp::service::RunningServiceCancellationToken) {
    *SHUTDOWN_TOKEN.lock().unwrap() = Some(token);
}

#[cfg(unix)]
fn arm_shutdown_watchdog() {
    use std::sync::atomic::Ordering;

    #[cfg(target_os = "linux")]
    // SAFETY: prctl with PR_SET_PDEATHSIG only sets this process's
    // parent-death signal; it touches no memory we own.
    unsafe {
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
    }

    let handler = note_shutdown as extern "C" fn(libc::c_int);
    // SAFETY: installing a handler whose body is a single relaxed atomic
    // store, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGTERM, handler as libc::sighandler_t);
        libc::signal(libc::SIGHUP, handler as libc::sighandler_t);
    }

    // Recorded in `main`, not here — see PARENT_AT_START. A zero means the
    // recorder was never called; treat that as "cannot tell" and rely on the
    // signal paths rather than shutting down on a bogus comparison.
    let parent_at_start = PARENT_AT_START.load(Ordering::Relaxed);

    std::thread::spawn(move || {
        loop {
            if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                break;
            }
            if parent_at_start != 0 && unsafe { libc::getppid() } != parent_at_start {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        // A token exists only after the handshake. Before that there is no
        // service, no in-flight request and nothing to drain, so exiting is
        // the whole of a correct shutdown.
        match SHUTDOWN_TOKEN.lock().unwrap().take() {
            Some(token) => token.cancel(),
            None => std::process::exit(0),
        }
        // Backstop. Installing a SIGTERM handler means the default "die now"
        // action is gone, so a wedged shutdown would leave a process that
        // ignores SIGTERM — strictly worse than the orphan this fixes. Give
        // the graceful path time to finish, then leave anyway.
        std::thread::sleep(std::time::Duration::from_secs(10));
        std::process::exit(0);
    });
}

/// Serve the same MCP tools over Streamable-HTTP for remote clients reached
/// through a tunnel / reverse proxy. Each session gets its own engine (its own
/// SQLite connection; WAL handles the concurrency). The loopback bind is the
/// primary security boundary, so non-loopback binds are refused unless
/// `--http-allow-remote` says the exposure is deliberate. `token`, when set,
/// adds an in-tree `Authorization: Bearer` gate (constant-time compare) as
/// defense-in-depth — a misconfigured fronting proxy or an exposed port no
/// longer means instant full read/write access to the store. No token = the
/// prior behavior (loopback gate + fronting proxy only).
async fn serve_http(
    project_id: Option<i64>,
    bind: &str,
    allow_remote: bool,
    token: Option<String>,
) -> Result<()> {
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
        if token.is_some() {
            tracing::warn!(
                %bind,
                "serving MCP on a non-loopback address (--http-allow-remote); the bearer-token \
                 gate is active, but still front this port with TLS + an auth proxy"
            );
        } else {
            tracing::warn!(
                %bind,
                "serving unauthenticated MCP on a non-loopback address (--http-allow-remote); \
                 make sure an auth gate fronts this port"
            );
        }
    }
    let app = mcp_http_router(move || {
        Ok(MimirServer::new(
            Mimir::open().map_err(std::io::Error::other)?,
            project_id,
        ))
    })
    // `/inject` shares ONE process-wide engine across every request — unlike
    // `/mcp` above, which pays a fresh `Mimir::open()` (ONNX load + full
    // vector-matrix rebuild, ~450ms measured) per session. That's fine for
    // `/mcp`'s long-lived sessions but far too slow for a UserPromptSubmit
    // hook, which must answer inside the turn. First request still pays the
    // model-load cost (lazy); every request after that reuses the warm
    // engine (~26ms measured, vs ~480ms for the cold `recall-inject` CLI
    // path on the same store).
    .merge({
        let mut inject_engine = Mimir::open()?;
        // Worth eager-loading here specifically: this is the one engine
        // every prompt-submit hook call actually reranks through, and
        // there's only one of it for the whole process (unlike the
        // per-session `/mcp` engines above), so the memory cost doesn't
        // multiply with concurrent sessions.
        eager_load_reranker_if_auto(&mut inject_engine);
        inject_router(InjectState {
            engine: Arc::new(Mutex::new(inject_engine)),
            project_id,
        })
    });
    let app = with_bearer_auth(app, token);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!("mimir MCP (streamable-http) listening on http://{bind}/mcp");
    println!("mimir warm auto-recall listening on http://{bind}/inject");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Wrap `app` in an optional `Authorization: Bearer <token>` gate (#11).
/// `None` returns the router unchanged (prior behavior — loopback bind + any
/// fronting proxy are the only gates). `Some` requires the header on EVERY
/// route (`/mcp` and `/inject` both reach the store); the compare is
/// constant-time so a wrong token can't be recovered by timing (a length
/// mismatch short-circuits, leaking only length — same as the sync gate).
fn with_bearer_auth(app: axum::Router, token: Option<String>) -> axum::Router {
    let Some(token) = token else { return app };
    use axum::response::IntoResponse;
    // Arc<str>: the per-request clone below is a refcount bump, not a heap
    // copy — /inject sits on the warm auto-recall path.
    let expected: Arc<str> = format!("Bearer {token}").into();
    tracing::info!("MCP HTTP: bearer-token auth enabled");
    app.layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let expected = Arc::clone(&expected);
            async move {
                let ok = req
                    .headers()
                    .get(axum::http::header::AUTHORIZATION)
                    .map(|h| crate::sync::ct_eq(h.as_bytes(), expected.as_bytes()))
                    .unwrap_or(false);
                if ok {
                    next.run(req).await
                } else {
                    axum::http::StatusCode::UNAUTHORIZED.into_response()
                }
            }
        },
    ))
}

/// Eager-load the cross-encoder reranker at startup when config says
/// automatic reranking should be able to fire (`[rerank] auto = "warm"` or
/// `"always"`) — otherwise "warm" mode's first query (and every query,
/// for a session whose reranker never gets forced on) would never see it
/// resident and auto-rerank would silently never fire on a long-lived
/// server. Costs one cold ONNX load (a few hundred ms, plus the model's
/// resident memory — same footprint an explicit `--rerank` call already
/// pays) up front instead of a surprise on some later request; failure
/// degrades the same as any other reranker load (logged, un-reranked
/// results). Only called on the engines documented below — never on a
/// per-session HTTP `/mcp` engine, since each of those would eager-load its
/// own copy and memory would grow with concurrent sessions.
fn eager_load_reranker_if_auto(m: &mut Mimir) {
    if matches!(m.config.rerank.auto.as_str(), "warm" | "always") {
        m.ensure_reranker(false);
    }
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
/// Deliberately does NOT eager-load the reranker (unlike the stdio and
/// `/inject` engines) — every concurrent session gets its own `Mimir`, so
/// eager-loading here would multiply the model's resident memory by however
/// many sessions are open. "warm" auto-rerank simply won't fire on a
/// session's early queries unless something else in-process already forced
/// a load; `"always"` still loads it lazily on that session's first query.
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

/// Shared state for the `/inject` endpoint: ONE process-wide engine (see
/// `serve_http`'s comment for why this can't reuse `/mcp`'s per-session
/// factory) plus the server's fixed project scope. `Clone` is cheap — it's
/// just an `Arc` bump and a `Copy` `Option<i64>`.
#[derive(Clone)]
struct InjectState {
    engine: Arc<Mutex<Mimir>>,
    project_id: Option<i64>,
}

/// `GET /inject?prompt=...&enrich=...` — the warm counterpart to `mimir
/// recall-inject`. Read-mostly (the only write is the impression log, same
/// as a normal recall) and loopback-only (same bind guard as `/mcp`), so no
/// auth beyond that is added here. Runs the actual work on `spawn_blocking`,
/// mirroring `MimirServer::blocking`, so a slow SQLite/tokenizer call can't
/// stall the async runtime. Never errors to the caller: any failure (bad
/// DB, panic) degrades to an empty body, matching the "silence over a wrong
/// answer" rule the floor itself already follows.
///
/// `enrich` is optional working-tree signal (see `MIMIR_RECALL_SH` in
/// `commands.rs`) — passed straight through to `inject::compute_with_session`,
/// which never lets it single-handedly clear the relevance floor.
///
/// `session` is an optional per-session id (the hook script's own session
/// identifier — see `commands.rs`'s UserPromptSubmit wiring). When present,
/// a memory already injected earlier in that session is skipped in favor of
/// the next floor-clearing candidate (or silence) instead of repeating —
/// see `inject::compute_with_session`'s doc comment for the ledger and its
/// fail-open behavior. Omitted or empty: behaves exactly as before, no
/// ledger check.
async fn inject_handler(
    axum::extract::State(state): axum::extract::State<InjectState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> String {
    let prompt = params.get("prompt").cloned().unwrap_or_default();
    let enrich = params.get("enrich").cloned().unwrap_or_default();
    let session = params.get("session").filter(|s| !s.is_empty()).cloned();
    let scope = state.project_id.map(Scope::Project).unwrap_or(Scope::All);
    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_or_else(|p| p.into_inner());
        match mimir_core::inject::compute_with_session(
            &mut engine,
            &prompt,
            &enrich,
            scope,
            false,
            session.as_deref(),
        ) {
            Ok(text) => text.unwrap_or_default(),
            Err(err) => {
                tracing::warn!(%err, "inject: compute failed");
                String::new()
            }
        }
    })
    .await
    .unwrap_or_default()
}

fn inject_router(state: InjectState) -> axum::Router {
    axum::Router::new()
        .route("/inject", axum::routing::get(inject_handler))
        .route("/inference", axum::routing::get(inference_handler))
        .route("/embed", axum::routing::post(embed_handler))
        .route("/rerank", axum::routing::post(rerank_handler))
        .with_state(state)
}

/// Caps on delegated inference requests. Embed matches `TX_BATCH`'s order of
/// magnitude but clients chunk at EMBED_BATCH (64) anyway to keep the shared
/// engine mutex hold short; rerank matches realistic candidate counts.
const EMBED_MAX_TEXTS: usize = 256;
const RERANK_MAX_DOCS: usize = 64;

/// `GET /inference` — cheap probe for inference delegation: which models
/// this daemon is configured for, without force-loading either (a probe
/// must never cost a model load). Clients compare against their own config
/// before trusting `/embed` / `/rerank` vectors.
async fn inference_handler(
    axum::extract::State(state): axum::extract::State<InjectState>,
) -> axum::Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap_or_else(|p| p.into_inner());
    axum::Json(serde_json::json!({
        "embedding_model": engine.config.embedding.model,
        "rerank_model": engine.config.rerank.model,
    }))
}

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

/// `POST /embed {"texts": [...]}` — L2-normalized vectors from the daemon's
/// resident embedder, so per-session MCP servers never need their own GPU
/// copy. 503 (not 200-with-nothing) when no model is loadable: the client
/// treats that as "daemon can't help", backs off, and embeds locally.
async fn embed_handler(
    axum::extract::State(state): axum::extract::State<InjectState>,
    axum::Json(req): axum::Json<EmbedRequest>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;
    if req.texts.is_empty() || req.texts.len() > EMBED_MAX_TEXTS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("texts must be 1..={EMBED_MAX_TEXTS}"),
        ));
    }
    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_or_else(|p| p.into_inner());
        let Some(embedder) = engine.ensure_embedder(false) else {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "embedding model unavailable".to_string(),
            ));
        };
        let vectors = embedder
            .embed(req.texts)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(axum::Json(serde_json::json!({
            "model": embedder.name,
            "dim": embedder.dim,
            "vectors": vectors,
        })))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    docs: Vec<String>,
}

/// `POST /rerank {"query": ..., "docs": [...]}` — cross-encoder scores in
/// **docs order** (the local `Reranker::rerank` returns best-first pairs;
/// re-sorting here keeps the wire format positional and dead simple).
async fn rerank_handler(
    axum::extract::State(state): axum::extract::State<InjectState>,
    axum::Json(req): axum::Json<RerankRequest>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;
    if req.docs.is_empty() || req.docs.len() > RERANK_MAX_DOCS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("docs must be 1..={RERANK_MAX_DOCS}"),
        ));
    }
    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_or_else(|p| p.into_inner());
        let Some(reranker) = engine.ensure_reranker(false) else {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "rerank model unavailable".to_string(),
            ));
        };
        let n = req.docs.len();
        let ranked = reranker
            .rerank(&req.query, req.docs)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mut scores = vec![0f32; n];
        let mut seen = 0usize;
        for (idx, score) in ranked {
            if let Some(slot) = scores.get_mut(idx) {
                *slot = score;
                seen += 1;
            }
        }
        if seen != n {
            // A partial scoring would silently mis-rank the unscored docs
            // at 0.0 (cross-encoder scores can be negative) — refuse instead
            // and let the client keep its fused order.
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reranker scored {seen}/{n} docs"),
            ));
        }
        Ok(axum::Json(serde_json::json!({
            "model": reranker.name,
            "scores": scores,
        })))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
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

    /// The optional bearer gate (#11): with a token set, `/mcp` rejects a
    /// missing or wrong `Authorization` header (401) and admits the correct
    /// one. Drives real HTTP against an ephemeral port + temp store.
    #[tokio::test(flavor = "multi_thread")]
    async fn http_bearer_token_gates_requests() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under_root(dir.path());
        let inject_paths = paths.clone();
        let router = mcp_http_router(move || {
            Ok(MimirServer::new(
                Mimir::open_at(paths.clone()).map_err(std::io::Error::other)?,
                None,
            ))
        });
        // Merge the inject router like production `serve_http` does, so the
        // gate is verified on BOTH routes — a refactor that layers auth
        // before the merge would leave /inject open and only this catches it.
        let router = router.merge(inject_router(InjectState {
            engine: Arc::new(Mutex::new(Mimir::open_at(inject_paths).unwrap())),
            project_id: None,
        }));
        let router = with_bearer_auth(router, Some("s3cret".into()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let url = format!("http://{addr}/mcp");
        let inject_url = format!("http://{addr}/inject?prompt=hello");
        let embed_url = format!("http://{addr}/embed");

        const INIT: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#;
        let (no_auth, wrong, right, inj_no_auth, inj_right, emb_no_auth, emb_right) =
            tokio::task::spawn_blocking(move || {
                let agent = ureq::AgentBuilder::new()
                    .timeout_read(Duration::from_secs(10))
                    .build();
                // One sender for both routes: POST with a body (MCP) when
                // `body` is Some, plain GET (/inject) when None.
                let send = |req: ureq::Request, auth: Option<&str>, body: Option<&str>| -> u16 {
                    let req = match auth {
                        Some(a) => req.set("Authorization", a),
                        None => req,
                    };
                    let res = match body {
                        Some(b) => req.send_string(b),
                        None => req.call(),
                    };
                    match res {
                        Ok(r) => r.status(),
                        Err(ureq::Error::Status(code, _)) => code,
                        Err(_) => 0,
                    }
                };
                let mcp_req = || {
                    agent
                        .post(&url)
                        .set("Content-Type", "application/json")
                        .set("Accept", "application/json, text/event-stream")
                };
                let embed_req = || {
                    agent
                        .post(&embed_url)
                        .set("Content-Type", "application/json")
                };
                (
                    send(mcp_req(), None, Some(INIT)),
                    send(mcp_req(), Some("Bearer nope"), Some(INIT)),
                    send(mcp_req(), Some("Bearer s3cret"), Some(INIT)),
                    send(agent.get(&inject_url), None, None),
                    send(agent.get(&inject_url), Some("Bearer s3cret"), None),
                    send(embed_req(), None, Some(r#"{"texts":["hi"]}"#)),
                    send(
                        embed_req(),
                        Some("Bearer s3cret"),
                        Some(r#"{"texts":["hi"]}"#),
                    ),
                )
            })
            .await
            .unwrap();
        assert_eq!(no_auth, 401, "missing header must be rejected");
        assert_eq!(wrong, 401, "wrong token must be rejected");
        assert_eq!(right, 200, "correct token must pass, got {right}");
        assert_eq!(inj_no_auth, 401, "/inject without token must be rejected");
        assert_eq!(
            inj_right, 200,
            "/inject with token must pass, got {inj_right}"
        );
        assert_eq!(emb_no_auth, 401, "/embed without token must be rejected");
        assert_eq!(
            emb_right, 503,
            "/embed with token must clear auth and hit the model-less 503, got {emb_right}"
        );
    }

    /// The inference-delegation endpoints on a model-less temp store: the
    /// probe answers with configured names (no model load), /embed and
    /// /rerank say 503 "can't help" (the client's back-off signal), and
    /// malformed/oversized bodies are 400s, not 500s.
    #[tokio::test(flavor = "multi_thread")]
    async fn inference_endpoints_semantics() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        let router = inject_router(InjectState {
            engine: Arc::new(Mutex::new(engine)),
            project_id: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let base = format!("http://{addr}");

        let (probe, embed_503, rerank_503, embed_400_empty, embed_400_oversize) =
            tokio::task::spawn_blocking(move || {
                let agent = ureq::AgentBuilder::new()
                    .timeout_read(Duration::from_secs(10))
                    .build();
                let probe = agent
                    .get(&format!("{base}/inference"))
                    .call()
                    .unwrap()
                    .into_string()
                    .unwrap();
                let status = |res: std::result::Result<ureq::Response, ureq::Error>| match res {
                    Ok(r) => r.status(),
                    Err(ureq::Error::Status(code, _)) => code,
                    Err(_) => 0,
                };
                let post = |path: &str, body: String| {
                    status(
                        agent
                            .post(&format!("{base}{path}"))
                            .set("Content-Type", "application/json")
                            .send_string(&body),
                    )
                };
                let big = serde_json::json!({
                    "texts": vec!["x"; EMBED_MAX_TEXTS + 1]
                })
                .to_string();
                (
                    probe,
                    post("/embed", r#"{"texts":["hi"]}"#.into()),
                    post("/rerank", r#"{"query":"q","docs":["a","b"]}"#.into()),
                    post("/embed", r#"{"texts":[]}"#.into()),
                    post("/embed", big),
                )
            })
            .await
            .unwrap();

        let probe: serde_json::Value = serde_json::from_str(&probe).unwrap();
        assert_eq!(probe["embedding_model"], "bge-small-en-v1.5");
        assert_eq!(probe["rerank_model"], "jina-reranker-v1-turbo-en");
        assert_eq!(embed_503, 503, "no model on disk => 503, not an error body");
        assert_eq!(rerank_503, 503, "no reranker on disk => 503");
        assert_eq!(embed_400_empty, 400, "empty texts is a client error");
        assert_eq!(embed_400_oversize, 400, "oversized batch is a client error");
    }

    fn remember_args(text: &str, ty: &str) -> RememberArgs {
        RememberArgs {
            text: text.into(),
            r#type: Some(ty.into()),
            tags: vec!["auth".into()],
            global: true,
            link: None,
            fires_when: Vec::new(),
            anchors: Vec::new(),
            expires_in: None,
            resolves_when: None,
            confidence: None,
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
                include_superseded: false,
                as_of: None,
            }))
            .await;
        assert!(hits.contains("SCRAM"), "recall: {hits}");

        let body = s.get(Parameters(GetArgs { r#ref: id })).await;
        assert!(body.contains("SCRAM"), "get: {body}");
    }

    /// `link` must refuse `supersedes`. The edge and the `superseded_by`
    /// column are independent and only the column suppresses recall, so
    /// accepting the rel here records something that reads as a retirement
    /// and retires nothing — the exact shape of a real six-week-old bug.
    #[tokio::test(flavor = "multi_thread")]
    async fn link_refuses_the_supersedes_rel() {
        let (_dir, s) = server();
        let a = s
            .remember(Parameters(remember_args("first fact here", "note")))
            .await;
        let b = s
            .remember(Parameters(remember_args("second fact here", "note")))
            .await;
        let (a, b) = (
            a.split_whitespace().nth(1).unwrap().to_string(),
            b.split_whitespace().nth(1).unwrap().to_string(),
        );

        let refused = s
            .link(Parameters(LinkArgs {
                a: a.clone(),
                b: b.clone(),
                rel: Some("supersedes".into()),
            }))
            .await;
        assert!(
            refused.contains("supersede") && refused.contains("NOT retire"),
            "expected a pointer at the supersede tool, got: {refused}"
        );

        // The neighbouring rel still works — this is a targeted refusal, not
        // a broken link tool.
        let ok = s
            .link(Parameters(LinkArgs {
                a,
                b,
                rel: Some("relates".into()),
            }))
            .await;
        assert!(ok.contains("relates"), "relates should still link: {ok}");
    }

    /// `remember` with an unparseable `expires_in` must fail loudly rather
    /// than storing a memory whose requested deadline was silently dropped.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_rejects_a_bad_expires_in() {
        let (_dir, s) = server();
        let mut args = remember_args("this fact has a bogus deadline", "note");
        args.expires_in = Some("soon".into());
        let out = s.remember(Parameters(args)).await;
        assert!(
            out.contains("invalid expires_in"),
            "expected a rejection, got: {out}"
        );
    }

    /// Anchors passed to the MCP `remember` tool land in `meta.anchors` —
    /// the agent-facing path to the same guard the CLI `--anchor` flag sets.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_with_anchors_sets_meta() {
        let (_dir, s) = server();
        let stored = s
            .remember(Parameters(RememberArgs {
                text: "this deploy step looks safe and is not".into(),
                r#type: Some("gotcha".into()),
                tags: Vec::new(),
                global: true,
                link: None,
                fires_when: Vec::new(),
                anchors: vec!["deploy.sh".into()],
                expires_in: None,
                resolves_when: None,
                confidence: None,
            }))
            .await;
        assert!(stored.starts_with("stored"), "remember: {stored}");
        let uid = stored.split_whitespace().nth(1).unwrap();

        let engine = s.engine.lock().unwrap_or_else(|p| p.into_inner());
        let node = store::resolve_ref(&engine.conn, uid).unwrap();
        assert_eq!(
            node.meta["anchors"][0], "deploy.sh",
            "meta.anchors should hold the pattern: {:?}",
            node.meta
        );
    }

    /// The secrets guard (mimir_core::secrets::scan) is wired into the MCP
    /// `remember` tool, not just the CLI command: a real AWS key never
    /// reaches `memory::remember` and the store stays empty.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_refuses_secret_looking_text() {
        let (_dir, s) = server();
        let refused = s
            .remember(Parameters(remember_args(
                "aws_access_key_id = AKIAABCDEFGHIJKLMNOP",
                "note",
            )))
            .await;
        assert!(
            refused.contains("AWS access key"),
            "expected secret refusal: {refused}"
        );
        assert!(!refused.starts_with("stored"), "must not store: {refused}");
    }

    /// The secrets guard must also cover `tags`, not just `text` — a secret
    /// smuggled into a tag would otherwise be stored and recalled in plain
    /// text via `tags_text`.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_refuses_secret_looking_tag() {
        let (_dir, s) = server();
        let refused = s
            .remember(Parameters(RememberArgs {
                text: "harmless note about a deploy".into(),
                r#type: Some("note".into()),
                tags: vec!["AKIAABCDEFGHIJKLMNOP".into(), "deploy".into()],
                global: true,
                link: None,
                fires_when: Vec::new(),
                anchors: Vec::new(),
                expires_in: None,
                resolves_when: None,
                confidence: None,
            }))
            .await;
        assert!(
            refused.contains("AWS access key"),
            "expected secret refusal: {refused}"
        );
        assert!(!refused.starts_with("stored"), "must not store: {refused}");
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
