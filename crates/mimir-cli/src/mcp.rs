//! MCP stdio server (`mimir mcp`): one global registration serves every
//! repo. Project scope comes from the server's cwd (where the agent
//! launched it), overridable via `MIMIR_PROJECT=<path>`. It is resolved
//! once, at start-up, and frozen ([`ProjectOrigin`]): every tool scopes by
//! that one answer and `status` reports that same answer.
//!
//! Returns are token-lean agent-format text, not JSON. The engine is
//! sync; every tool hops through spawn_blocking.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use mimir_core::config::{DaemonInference, RerankAuto};
use mimir_core::format::{agent_line, agent_line_for_query, HitMarks};
use mimir_core::model::{short_uid, Kind, MemoryType, Rel, RelOrigin, Scope};
use mimir_core::search::SearchQuery;
use mimir_core::strength::MatchStrength;
use mimir_core::{store, Mimir};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Deserialize;

use crate::capture::{self, Capture, Captured, Linked, MissingLink};

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
    /// The checkout project: whose code graph `graph`, `peek` and `outline`
    /// read, and which resolves a symbol name given to `remember`'s `link`.
    project_id: Option<i64>,
    /// The memory project: where `remember` files and where `recall` reads
    /// memories (its code and docs come from the checkout project). The
    /// checkout project, except when the server was started in a git linked
    /// worktree (`Mimir::projects_for_cwd`).
    memory_project_id: Option<i64>,
    /// The line `status` adds in a linked worktree, saying where its
    /// memories go (`ProjectContext::worktree_note`), as start-up resolved
    /// it; `None` for any other checkout.
    worktree_note: Option<Arc<str>>,
    origin: Arc<ProjectOrigin>,
    // Read only inside the #[tool_handler] macro expansion.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

/// How start-up arrived at the server's `project_id`: where it looked and
/// which marker it found. Recorded once so `status` describes the project
/// the tools actually use — a later re-detection can disagree with the
/// frozen id (a marker added or removed since, a cwd that moved), and a
/// status line that contradicts what recall does is worse than none.
#[derive(Debug, Clone, Default)]
pub struct ProjectOrigin {
    /// The directory the server scopes from: `MIMIR_PROJECT`, else the
    /// launch cwd. Relative tool paths resolve against it when there is no
    /// project root to resolve against.
    pub dir: Option<std::path::PathBuf>,
    /// Label of the marker that identified the project root (`git`,
    /// `Cargo.toml`, …); `None` when no project was found.
    pub via: Option<&'static str>,
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
    /// true = include retired memories: superseded or expired (hidden by default).
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
    /// Relation: relates | about | links | mentions | describes |
    /// summarizes | derived_from (default relates). Refused here:
    /// `supersedes` (use the `supersede` tool), `co_recalled` (the store
    /// derives those from observed usage) and the relations Mimir derives
    /// from the code graph (`calls`, `imports`, ...).
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

/// The one shape a failed tool call has: text that starts with "error: ".
///
/// A tool result is a single string, so its first word is all an agent has
/// to tell an answer from a failure ("refused: " is a third outcome, and is
/// an answer). Most failures already carry the mark from `engine_err`; a
/// handful were plain sentences, and "unknown kind 'x'" read as a search
/// result. Marking them where every tool call returns means a new tool
/// cannot forget to.
fn mark_error(message: String) -> String {
    if message.starts_with("error: ") {
        message
    } else {
        format!("error: {message}")
    }
}

/// What every tool answers once the store has been migrated past this
/// binary, or `None` while the two still match.
///
/// The upgrade order is install, migrate, restart, and nothing enforces the
/// last step: the first command run with a newer `mimir` migrates the
/// store, while a server started on the older binary keeps its connection
/// and keeps answering — against a schema it was not written for. Opening
/// the store anew would fail with `SchemaTooNew`; a connection that is
/// already open never asks. So each tool call asks, with one header read,
/// and the agent is told the one thing that fixes it instead of being
/// handed results nobody can vouch for.
fn stale_server_error(conn: &rusqlite::Connection) -> Option<String> {
    let found: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .ok()?;
    let supported = mimir_core::db::migrations::SCHEMA_VERSION;
    (found > supported).then(|| {
        format!(
            "error: this Mimir server is older than its store (schema {found}, this server \
             supports {supported}) — restart it (in Claude Code: /mcp, then reconnect mimir)"
        )
    })
}

/// Whether a failed background re-open is the store having been migrated
/// past this binary — the one failure that cannot clear while this process
/// lives. Logged the first time only; the caller stops retrying, so a
/// server left running after an upgrade says it once instead of once per
/// interval.
fn store_outgrew_server(err: &mimir_core::error::Error) -> bool {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    if !matches!(err, mimir_core::error::Error::SchemaTooNew { .. }) {
        return false;
    }
    LOGGED.call_once(|| {
        tracing::warn!(
            "{err}: this server was started before the store was migrated. Background \
             work that re-opens the store has stopped; restart the server"
        );
    });
    true
}

/// What the `graph` and `peek` tools answer for a project whose code graph
/// has not been built. The server builds it in the background when it
/// starts (`[auto] graph`), so "not there yet" usually means "not finished
/// yet" — unless that is switched off, in which case nothing will build it
/// but the command.
fn no_graph_yet(auto_build: bool) -> String {
    if auto_build {
        "error: this project's code graph is not built yet — the build started with this \
         server has not finished (or failed: see the server log). Try again shortly, or run \
         `mimir graph build`."
            .into()
    } else {
        "error: this project has no code graph yet — run `mimir graph build` (automatic \
         builds are off: `[auto] graph = false`)."
            .into()
    }
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
    /// A server for the checkout project `project_id`, whose memories are
    /// filed under that same project unless [`Self::with_memory_project`]
    /// says otherwise.
    pub fn new(engine: Mimir, project_id: Option<i64>) -> Self {
        MimirServer {
            engine: Arc::new(Mutex::new(engine)),
            project_id,
            memory_project_id: project_id,
            worktree_note: None,
            origin: Arc::new(ProjectOrigin::default()),
            tool_router: Self::tool_router(),
        }
    }

    /// Record the project this server's memories are filed under, when
    /// start-up found it is not the checkout's own (a linked worktree).
    pub fn with_memory_project(mut self, memory_project_id: Option<i64>) -> Self {
        self.memory_project_id = memory_project_id;
        self
    }

    /// Record what `status` says about a linked worktree's two projects
    /// (`ProjectContext::worktree_note`).
    pub fn with_worktree_note(mut self, note: Option<String>) -> Self {
        self.worktree_note = note.map(Arc::from);
        self
    }

    /// Record how start-up resolved the project this server was built for.
    pub fn with_origin(mut self, origin: ProjectOrigin) -> Self {
        self.origin = Arc::new(origin);
        self
    }

    /// Run one tool call on the blocking pool, with the engine locked.
    ///
    /// Every tool goes through here, so this is also where a server that
    /// has been left behind by its store says so (see
    /// [`stale_server_error`]), and where a failure gets its mark (see
    /// [`mark_error`]).
    fn blocking<F>(&self, f: F) -> impl std::future::Future<Output = String>
    where
        F: FnOnce(&mut Mimir) -> Result<String, String> + Send + 'static,
    {
        let engine = self.engine.clone();
        async move {
            tokio::task::spawn_blocking(move || {
                let mut engine = engine.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(stale) = stale_server_error(&engine.conn) {
                    return stale;
                }
                f(&mut engine).unwrap_or_else(mark_error)
            })
            .await
            .unwrap_or_else(|e| format!("error: task panicked: {e}"))
        }
    }

    #[tool(
        description = "Search memories, docs and code in one ranked list (hybrid BM25+semantic). ~25-token line per hit; `weak` = little evidence it matches the query."
    )]
    async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let (project_id, memory_project_id) = (self.project_id, self.memory_project_id);
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
            // The same reach the CLI's `recall` has: memories of the
            // memory project, code and docs of the checkout's own.
            let reach = if args.all_projects {
                mimir_core::search::Reach::uniform(Scope::All)
            } else {
                mimir_core::scope::query_reach(memory_project_id, project_id)
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
                // The memories' scope: what the miss ledger keys this
                // question by. The search itself reads `reach`.
                scope: reach.memories,
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
            let mimir_core::SearchOutcome {
                mut hits,
                legs,
                rerank: reranked,
            } = m
                .search_with_outcome_within(&query, reach, args.rerank)
                .map_err(engine_err)?;
            // Another checkout's session handoff is not this session's to
            // find, the same rule as the CLI's; `all_projects` shows it.
            if !args.all_projects {
                let checkout =
                    project_root(m, project_id).map(|p| p.to_string_lossy().into_owned());
                hits.retain(|hit| {
                    mimir_core::context_guard::shown_in_checkout(&hit.node, checkout.as_deref())
                });
            }
            let strengths = mimir_core::strength::classify_page(&hits, &query.text, &legs);
            // The same gate as the CLI, by construction: a view into the
            // past or into retired memories is not a question to the store
            // as it stands.
            if mimir_core::misses::should_observe(m.config.learn.log_misses, &query) {
                mimir_core::misses::observe(
                    &m.conn,
                    &query.text,
                    mimir_core::misses::scope_project(&query.scope),
                    &strengths,
                    mimir_core::model::now_unix(),
                );
            }
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
            if m.config.learn.shadow_ranking {
                mimir_core::shadow::observe(
                    &m.conn,
                    query_hash.as_bytes(),
                    &hits,
                    mimir_core::model::now_unix(),
                );
            }
            let projects = store::project_titles(&m.conn).map_err(engine_err)?;
            let ids: Vec<i64> = hits.iter().map(|h| h.node.id).collect();
            let stale = mimir_core::grounding::stale_ids(&m.conn, &ids).unwrap_or_default();
            let mut out = Vec::new();
            for (hit, strength) in hits.iter().zip(&strengths) {
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
                    HitMarks {
                        stale_grounding: stale.contains(&hit.node.id),
                        weak_match: *strength == MatchStrength::Weak,
                    },
                ));
            }
            // `rerank: true` was asked for by name; an agent that got the
            // fused order instead has to be told, as the CLI tells a person.
            if let Some(note) = reranked.note().filter(|_| args.rerank) {
                out.push(format!("note: {note}"));
            }
            Ok(out.join("\n"))
        })
        .await
    }

    #[tool(
        description = "Store a typed memory. Pass `link` to attach it to a code symbol/file. Pass `fires_when` to force-inject on trigger phrases without keyword overlap. Pass `anchors` to surface it at act-time when a matching file is edited or command is run."
    )]
    async fn remember(&self, Parameters(args): Parameters<RememberArgs>) -> String {
        let (project_id, memory_project_id) = (self.project_id, self.memory_project_id);
        self.blocking(move |m| {
            let mtype: MemoryType = args
                .r#type
                .as_deref()
                .unwrap_or("note")
                .parse()
                .map_err(engine_err)?;
            let surface = mimir_core::secrets::Surface::McpRemember;
            let outcome = capture::capture(
                m,
                Capture {
                    text: args.text,
                    mtype,
                    tags: args.tags,
                    project_id: if args.global { None } else { memory_project_id },
                    working_project: project_id,
                    // An agent cannot override a refusal; the reply names
                    // who can.
                    force: false,
                    actor: mimir_core::model::Actor::Agent,
                    surface,
                    link: args.link,
                    missing_link: MissingLink::StoreUnlinked,
                    fires_when: args.fires_when,
                    anchors: args.anchors,
                    expires_in: args.expires_in,
                    resolves_when: args.resolves_when,
                    confidence: args.confidence,
                    valid_from: None,
                    valid_to: None,
                },
            )
            .map_err(|rejected| engine_err(rejected.message(surface)))?;
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
                Captured::Stored(stored) => {
                    let mut msg = format!("stored {}", line(&stored.node));
                    match &stored.link {
                        Linked::To { target, rel } => msg.push_str(&format!(
                            "\nlinked —{rel}→ {}",
                            short_uid(target.kind, &target.uid)
                        )),
                        Linked::Missing(target) => msg.push_str(&format!(
                            "\n(link target '{target}' not found — memory stored unlinked)"
                        )),
                        Linked::NotRequested => {}
                    }
                    Ok(msg)
                }
                Captured::Duplicate {
                    node: existing,
                    rearmed_handoff,
                } => Ok(format!(
                    "refused: near-duplicate of\n{}\n({})",
                    line(&existing),
                    if rearmed_handoff {
                        capture::REARMED_HANDOFF
                    } else {
                        "edit that entry instead, or rephrase if genuinely different"
                    }
                )),
                Captured::Forgotten { node: gone, by, on } => Ok(format!(
                    "refused: {} on {}\n{}\n(it was deleted on purpose — do not re-add it \
                     unless the user asks; if they do, they can run `mimir remember --force`)",
                    capture::forgotten_phrase(by),
                    mimir_core::format::full_date(on),
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
        description = "Link two nodes (memory/doc/code) so context surfaces together. Note: `supersedes` is NOT accepted here — use the `supersede` tool, which also retires the old node. `co_recalled` is NOT accepted either — the store writes those itself from observed usage, so a hand-authored one forges a co-activation that never happened; use `relates`."
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
            if rel.origin() == RelOrigin::Retirement {
                return Err(engine_err(mimir_core::error::Error::Invalid(
                    "link does not accept rel=\"supersedes\": it records an edge but does \
                     NOT retire the old node, so it would look like a supersede and change \
                     nothing. Use the `supersede` tool instead (it sets both)."
                        .into(),
                )));
            }
            // `co_recalled` is not a claim, it is an observation the store made
            // about its own usage: two memories opened in the same stretch of
            // work. Hand-authoring one forges a usage record that never
            // happened. Worse, `learn::record_coactivation` deliberately skips
            // any pair that already carries an authored edge, so the forgery is
            // not a harmless white lie — it permanently suppresses the genuine
            // co-activation signal for that pair.
            // Derived relations are rebuilt from what Mimir indexes; the
            // extractor wipes a file's `calls` and `imports` before writing
            // the fresh ones, so a hand-made edge would vanish on the next
            // build and pass for a resolved one until then.
            if rel.origin() == RelOrigin::Extracted {
                return Err(engine_err(mimir_core::error::Error::Invalid(format!(
                    "link does not accept rel=\"{rel}\": Mimir derives those edges from \
                     what it indexes (the code graph, communities, tags) and replaces \
                     them when it re-reads the source, so a hand-made one would not \
                     survive. Use rel=\"relates\" or rel=\"about\" for a link you \
                     actually mean."
                ))));
            }
            if rel.origin() == RelOrigin::Observed {
                return Err(engine_err(mimir_core::error::Error::Invalid(
                    "link does not accept rel=\"co_recalled\": the store derives those \
                     from observed co-recall and refuses to draw one for a pair that \
                     already has an authored edge, so hand-authoring one fakes usage \
                     that never happened AND blocks the real signal for that pair \
                     forever. Use rel=\"relates\" for a link you actually mean."
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
            use mimir_graph::render::{self, Direction};
            let Some(project_id) = project_id else {
                return Err("error: no project detected (the code graph is per-project)".into());
            };
            let project = store::get_node(&m.conn, project_id).map_err(engine_err)?;
            // An empty answer from a graph that does not exist yet would
            // read as "nothing calls this". Say what is actually the case.
            if !mimir_graph::has_graph(&m.conn, project_id).map_err(engine_err)? {
                return Err(no_graph_yet(m.config.auto.graph));
            }
            let depth = args.depth.unwrap_or(3);
            let graph = mimir_graph::CodeGraph::load(&m.conn, project_id).map_err(engine_err)?;
            // A call that cannot be answered as asked, and `hubs` (which
            // ranks the whole graph rather than describing one symbol's
            // current code), end here: neither gets a freshness line.
            let missing = |arg: &str| Err(format!("error: missing '{arg}' argument"));
            let (symbol, to) = (args.symbol.as_deref(), args.to.as_deref());
            match args.op.as_str() {
                "hubs" => {
                    return render::hubs(&m.conn, &graph, args.limit.unwrap_or(10))
                        .map_err(engine_err)
                }
                "callers" | "calls" | "node" | "path" if symbol.is_none() => {
                    return missing("symbol")
                }
                "path" if to.is_none() => return missing("to"),
                "impact" if args.files.is_empty() => return missing("files"),
                "callers" | "calls" | "node" | "path" | "impact" => {}
                other => {
                    return Err(format!(
                        "error: unknown op '{other}' (callers|calls|impact|node|path|hubs)"
                    ))
                }
            }
            let resolve = |reference: Option<&str>| {
                mimir_graph::resolve_symbol(&m.conn, project_id, reference.unwrap_or_default())
                    .map_err(engine_err)
            };
            // The answers are rendered by `mimir_graph::render`, the same
            // code `mimir graph …` prints from, so the two read alike.
            let answer = || -> Result<String, String> {
                match args.op.as_str() {
                    "impact" => {
                        // Decided once for this tool and the CLI alike; an
                        // agent reading "no impact" as an all-clear needs it
                        // to never mean "that path is not in this graph".
                        let trace = graph
                            .impact_of_paths_traced(&m.conn, &args.files, depth)
                            .map_err(engine_err)?;
                        render::impact(&trace, depth)
                            .ok_or_else(|| engine_err(render::NOT_IN_GRAPH))
                    }
                    "node" => {
                        let sym = resolve(symbol)?;
                        mimir_core::learn::record_opened(&m.conn, sym.id).map_err(engine_err)?;
                        render::node(&m.conn, &sym).map_err(engine_err)
                    }
                    "path" => {
                        let (a, b) = (resolve(symbol)?, resolve(to)?);
                        render::path(&m.conn, &graph, &a, &b).map_err(engine_err)
                    }
                    op => {
                        let direction = if op == "callers" {
                            Direction::Callers
                        } else {
                            Direction::Calls
                        };
                        let sym = resolve(symbol)?;
                        render::relation(&m.conn, &graph, &sym, depth, direction)
                            .map_err(engine_err)
                    }
                }
            };
            // What the answer cannot show about the graph it was read from
            // (behind the working tree; resolved by an older build's rules)
            // follows it, one line each — on a miss as much as on a hit,
            // since a file added after the last build is why a path is
            // "not in the graph". The same lines `mimir graph …` prints on
            // stderr.
            let notes = crate::graph_cmd::graph_notes(&m.conn, &project);
            let with_note = |text: String| {
                notes
                    .iter()
                    .fold(text, |text, note| format!("{text}\n{note}"))
            };
            answer().map(with_note).map_err(with_note)
        })
        .await
    }

    #[tool(
        description = "Dense signature map of a file or directory: one line per symbol (signature + first doc line), no bodies."
    )]
    async fn outline(&self, Parameters(args): Parameters<OutlineArgs>) -> String {
        let project_id = self.project_id;
        let origin = self.origin.clone();
        self.blocking(move |m| {
            // A relative target is relative to the project; outside one, to
            // the directory the server scopes from (which honours
            // MIMIR_PROJECT) — never to wherever the process happens to be.
            let cwd = project_root(m, project_id)
                .or_else(|| origin.dir.clone())
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
                return Err(
                    "error: no project detected (peek needs the per-project code graph)".into(),
                );
            };
            let root = project_root(m, Some(pid)).ok_or("error: project has no root")?;
            if !mimir_graph::has_graph(&m.conn, pid).map_err(engine_err)? {
                return Err(no_graph_yet(m.config.auto.graph));
            }
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
            // The body is cut by stored line numbers; when the file has
            // moved on since they were stored, the last line says so.
            Ok(match p.stale_note {
                Some(note) => format!("{}{note}\n", p.text),
                None => p.text,
            })
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
        let worktree_note = self.worktree_note.clone();
        let origin = self.origin.clone();
        self.blocking(move |m| {
            let counts = store::count_by_kind(&m.conn).map_err(engine_err)?;
            // The project is the one frozen at start-up, with the reason
            // recorded then — the same answer every tool scopes by.
            let project = match project_id {
                Some(id) => {
                    let name = store::get_node(&m.conn, id)
                        .ok()
                        .and_then(|p| p.title)
                        .unwrap_or_else(|| "?".into());
                    // In a linked worktree, where its memories go: the
                    // same line `mimir status` adds, resolved at start-up
                    // like the ids.
                    let worktree = worktree_note
                        .as_deref()
                        .map(|note| format!("\n        {note}"))
                        .unwrap_or_default();
                    format!("{name}  [via: {}]{worktree}", origin.via.unwrap_or("?"))
                }
                // Outside a project the two directions differ, so both are
                // said: what is written is global, what is asked searches
                // everything (`scope::query_scope`).
                None => match &origin.dir {
                    Some(dir) => format!(
                        "(none — global scope) — no marker above {}; new memories are \
                         global, recall searches every project",
                        dir.display()
                    ),
                    None => "(none — global scope; new memories are global, recall \
                             searches every project)"
                        .into(),
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
                "project {project}\nstore   {summary}\ndb      {} KB\nsync    {}",
                db_size / 1024,
                crate::sync::status_line(m)
            ))
        })
        .await
    }

    #[tool(
        description = "Forget a memory: hidden from recall (the deletion syncs across machines) and its earlier wordings are erased. Not undoable: `remember` then refuses the same text unless a human runs `mimir remember --force`. No hard delete over MCP — use `mimir forget --hard` on the CLI."
    )]
    async fn forget(&self, Parameters(args): Parameters<ForgetArgs>) -> String {
        self.blocking(move |m| {
            let node = store::forget(
                &m.conn,
                &args.r#ref,
                false,
                mimir_core::model::Actor::Agent,
                None,
            )
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
        description = "Retire memory OLD in favour of memory NEW: OLD is marked superseded, which hides it from recall (kept as history), and a `supersedes` link records the replacement. Not undoable. Prefer over `forget` when NEW replaces OLD."
    )]
    async fn supersede(&self, Parameters(args): Parameters<SupersedeArgs>) -> String {
        self.blocking(move |m| {
            let old = store::resolve_ref(&m.conn, &args.old).map_err(engine_err)?;
            let new = store::resolve_ref(&m.conn, &args.by).map_err(engine_err)?;
            store::supersede(
                &m.conn,
                old.id,
                new.id,
                mimir_core::model::Actor::Agent,
                None,
            )
            .map_err(engine_err)?;
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
    // A mode config.toml misspells falls back without a word wherever it is
    // read (see `Config::issues`). A server reads its config for hours, so
    // it says so here, once, rather than at every query that consults it.
    for issue in engine.config.issues() {
        tracing::warn!("config: {issue}");
    }
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
    let delegate = engine.config.daemon.inference() == DaemonInference::Auto && http.is_none();
    // Resolved here and nowhere else: the ids every tool scopes by, and the
    // origin `status` reports, come from this one detection. Two ids: the
    // checkout's own project (its code graph) and the one its memories are
    // filed under, which differ in a git linked worktree.
    let mut origin = ProjectOrigin {
        dir: server_cwd(),
        via: None,
    };
    // The stdio server is a session's own, started on purpose in its
    // checkout: like a CLI memory command, it moves what a linked
    // worktree's own row still holds to the memory project (the engine
    // logs that at info level). The HTTP server, which is the daemon, is a
    // fallback every checkout's hooks share, and only resolves.
    let resolved = |d: &std::path::Path| match http {
        None => engine
            .projects_for_cwd_moving_memories_home(d)
            .map(|(here, _)| here),
        Some(_) => engine.projects_for_cwd(d),
    };
    let mut worktree_note = None;
    let (project_id, memory_project_id) = match origin.dir.clone() {
        Some(d) => match resolved(&d) {
            Ok(here) => {
                let detection = &here.detection;
                // Never degrade silently: always log how scope was resolved.
                // Without a project the two directions differ (see
                // `scope::query_scope`): reads search every project, writes
                // are global — so the line names both.
                match detection {
                    mimir_core::scope::Detection::Found { root, via } => {
                        origin.via = Some(mimir_core::scope::via_label(via));
                        tracing::info!(
                            project = %root.display(),
                            via = mimir_core::scope::via_label(via),
                            "scoped to project"
                        );
                        worktree_note = here.worktree_note();
                        if let Some(note) = &worktree_note {
                            tracing::info!("{note}");
                        }
                    }
                    mimir_core::scope::Detection::NotFound { from } => {
                        origin.dir = Some(from.clone());
                        tracing::info!(
                            from = %from.display(),
                            "no project marker found; no project scope: recall searches \
                             every project, new memories are global"
                        )
                    }
                }
                (here.checkout_id(), here.memory_id())
            }
            Err(err) => {
                // Wrong-scoped memories are corruption nobody notices.
                tracing::warn!(
                    %err,
                    "project detection failed; running without a project: recall searches \
                     every project, new memories are global"
                );
                (None, None)
            }
        },
        None => (None, None),
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
                    // The same rule as `mimir graph build`: a small project
                    // whose graph an older Mimir resolved is rebuilt in full
                    // here, in the background, so its answers carry no
                    // "older rules" note for someone to act on.
                    let full = mimir_graph::wants_full_rebuild(&m.conn, pid).unwrap_or(false);
                    match mimir_graph::update_with(
                        &mut m.conn,
                        &proj,
                        &root,
                        mimir_graph::UpdateOptions { full },
                    ) {
                        Ok(report) => tracing::info!(
                            symbols = report.stats.symbols,
                            files_indexed = report.stats.files_indexed,
                            parse_failed = report.parse_failed,
                            unreadable = report.unreadable,
                            full,
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
        let cadence = engine.config.consolidate.auto();
        let model = engine.config.embedding.model.clone();
        let db_path = engine.paths.db_file.clone();
        std::thread::spawn(move || {
            if let Ok(conn) = mimir_core::db::open(&db_path) {
                if let Some(r) =
                    mimir_core::consolidate::maybe_auto(&conn, &model, cadence.as_str())
                {
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
    // tracing, never stdout — stdout is the MCP protocol channel. The loop
    // always runs and decides on each tick from the [sync] config as it is
    // then (see `sync::auto_tick`), so turning sync or `auto` on or off takes
    // effect within one interval, without a restart. With sync off a tick
    // reads the config file and nothing else.
    {
        let config_file = engine.paths.config_file.clone();
        std::thread::spawn(move || loop {
            // The one open failure that cannot clear while this process
            // lives: the store was migrated past this binary. Said once
            // (`store_outgrew_server`), and the loop ends instead of
            // failing the same way every interval.
            let mut outgrown = false;
            let pause = crate::sync::auto_tick(&config_file, || {
                let mut m = match mimir_core::Mimir::open() {
                    Ok(m) => m,
                    Err(e) => {
                        outgrown = store_outgrew_server(&e);
                        return Err(e.into());
                    }
                };
                if delegate {
                    m.set_embedder_device("cpu");
                    crate::infer::attach_lazy(&mut m);
                }
                Ok(m)
            });
            if outgrown {
                break;
            }
            std::thread::sleep(pause);
        });
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        match http {
            Some(bind) => {
                serve_http(
                    project_id,
                    memory_project_id,
                    worktree_note,
                    origin,
                    &bind,
                    http_allow_remote,
                    http_token,
                )
                .await
            }
            None => {
                // Eager-load here, not up front in `run()`: in HTTP mode this
                // `engine` is only used for the project-detection/background
                // work above and never actually serves requests, so loading
                // the reranker onto it would just be a wasted cold load.
                let mut engine = engine;
                // With a live daemon this session holds zero GPU state: CPU
                // embedder (measured faster for batch-1 query embeds than
                // GPU) and no local reranker — `[rerank] auto = "warm"`
                // fires through the daemon instead.
                //
                // The embedder is CPU whenever delegation is *configured*,
                // whether or not the daemon answered: the device is fixed
                // at the embedder's first load, and a session that put its
                // embedder on the GPU because the daemon was a few seconds
                // late would hold that VRAM for its whole life — the cost
                // delegation exists to remove. What the unreachable case
                // does share with delegation being off is the reranker:
                // `[rerank] auto` eager-loads a local one, on the
                // configured device (below).
                let daemon_alive = crate::infer::attach(&mut engine);
                if delegate {
                    engine.set_embedder_device("cpu");
                }
                // Record which path this session took. Both outcomes "work",
                // so at runtime they are indistinguishable — the only visible
                // symptom of a fallback is a few hundred MB of resident
                // reranker that nobody can attribute to a cause days later.
                // What is decided here for the life of the session is that
                // eager load: a reranker made resident now stays resident
                // and is preferred over the daemon from then on. Delegation
                // itself is not decided here — the client stays attached
                // and is asked again once its cooldown has passed.
                if !delegate {
                    tracing::info!(
                        inference = %engine.config.daemon.inference(),
                        "inference: local only (delegation not enabled)"
                    );
                } else if daemon_alive {
                    tracing::info!(
                        "inference: delegating to daemon (CPU embedder, no local reranker)"
                    );
                } else {
                    let rerank_auto = engine.config.rerank.auto();
                    tracing::warn!(
                        %rerank_auto,
                        loads_local_reranker = rerank_auto != RerankAuto::Off,
                        "inference: daemon unreachable at session start; working locally for \
                         now. The client stays attached and tries the daemon again after its \
                         cooldown (bulk embeds, and reranks unless a local reranker is \
                         resident — one loaded now stays for the life of the session)"
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
                    .with_memory_project(memory_project_id)
                    .with_worktree_note(worktree_note)
                    .with_origin(origin)
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
    memory_project_id: Option<i64>,
    worktree_note: Option<String>,
    origin: ProjectOrigin,
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
        Ok(
            MimirServer::new(Mimir::open().map_err(std::io::Error::other)?, project_id)
                .with_memory_project(memory_project_id)
                .with_worktree_note(worktree_note.clone())
                .with_origin(origin.clone()),
        )
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
        let checkout = project_root(&inject_engine, project_id)
            .map(|root| root.to_string_lossy().into_owned());
        inject_router_in(
            InjectState {
                engine: Arc::new(Mutex::new(inject_engine)),
                // Injection reads memories, so its fallback scope is the
                // memory project.
                project_id: memory_project_id,
            },
            checkout,
        )
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
    if m.config.rerank.auto() != RerankAuto::Off {
        m.ensure_reranker(false);
    }
}

/// True only when every address `bind` resolves to is loopback. Unresolvable
/// binds return false — the TcpListener will produce the real error, and the
/// gate stays fail-closed.
pub(crate) fn bind_is_loopback(bind: &str) -> bool {
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

/// Shared state for the daemon's routes: ONE process-wide engine (see
/// `serve_http`'s comment for why this can't reuse `/mcp`'s per-session
/// factory) plus the memory project of the directory the server was started
/// in. `Clone` is cheap — it's just an `Arc` bump and a `Copy` `Option<i64>`.
#[derive(Clone)]
struct InjectState {
    engine: Arc<Mutex<Mimir>>,
    project_id: Option<i64>,
}

/// How long a `cwd → project` answer is reused by `/inject`. Long enough
/// that a session's prompts do not each walk the directory tree, short
/// enough that `git init` or `touch .mimir` in a directory takes effect
/// within the minute.
const PROJECT_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// Entries the `cwd → project` cache holds before expired ones are dropped.
const PROJECT_CACHE_MAX: usize = 512;

/// What `/inject` and `/inference` work from: [`InjectState`], plus the two
/// things derived from it once, when the router is built.
#[derive(Clone)]
struct WarmState {
    engine: Arc<Mutex<Mimir>>,
    /// The memory project of the directory the server was started in: the
    /// scope of a request that names no `cwd` (a hook script older than
    /// version 5 sends none).
    project_id: Option<i64>,
    /// The canonical root of that directory's checkout: whose session
    /// handoffs a request that names no `cwd` may be handed
    /// (`context_guard::shown_in_checkout`).
    checkout: Option<String>,
    /// The `/inference` answer. Copied out of the engine's config, which
    /// does not change while the process lives, so the probe never has to
    /// take the engine mutex — see [`inference_handler`].
    models: Arc<serde_json::Value>,
    /// The memory project and checkout of each `cwd` a request has named,
    /// and when they were resolved.
    projects: Arc<Mutex<ProjectCache>>,
}

/// Where one `/inject` request is: the scope it searches, and the
/// checkout whose session handoffs it may be handed.
#[derive(Debug, Clone)]
struct InjectPlace {
    scope: Scope,
    checkout: Option<String>,
}

type ProjectCache = std::collections::HashMap<
    std::path::PathBuf,
    ((Option<i64>, Option<String>), std::time::Instant),
>;

impl WarmState {
    fn new(state: InjectState, checkout: Option<String>) -> Self {
        let models = {
            let engine = state.engine.lock().unwrap_or_else(|p| p.into_inner());
            serde_json::json!({
                "embedding_model": engine.config.embedding.model,
                "rerank_model": engine.config.rerank.model,
            })
        };
        WarmState {
            engine: state.engine,
            project_id: state.project_id,
            checkout,
            models: Arc::new(models),
            projects: Arc::default(),
        }
    }

    /// Where one `/inject` request is: the same
    /// [`mimir_core::scope::query_scope`] the cold `mimir recall-inject`
    /// applies to its working directory, taken from the memory project of
    /// the request's `cwd` (and that directory's checkout, for the
    /// handoffs it may be handed) — or, when the request names none, from
    /// the directory the server was started in. The resolution never moves
    /// memories (`Mimir::projects_for_cwd`).
    fn place_for(
        &self,
        engine: &Mimir,
        cwd: Option<&std::path::Path>,
    ) -> mimir_core::error::Result<InjectPlace> {
        let place = |(project, checkout): (Option<i64>, Option<String>)| InjectPlace {
            scope: mimir_core::scope::query_scope(project),
            checkout,
        };
        let Some(cwd) = cwd else {
            return Ok(place((self.project_id, self.checkout.clone())));
        };
        let now = std::time::Instant::now();
        let fresh = |at: &std::time::Instant| now.duration_since(*at) < PROJECT_CACHE_TTL;
        let mut cache = self.projects.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((found, _)) = cache.get(cwd).filter(|(_, at)| fresh(at)) {
            return Ok(place(found.clone()));
        }
        let here = engine.projects_for_cwd(cwd)?;
        let found = (here.memory_id(), here.checkout_path().map(str::to_owned));
        if cache.len() >= PROJECT_CACHE_MAX {
            cache.retain(|_, (_, at)| fresh(at));
        }
        cache.insert(cwd.to_path_buf(), (found.clone(), now));
        Ok(place(found))
    }
}

/// `GET /inject?prompt=...&enrich=...&session=...&cwd=...` — the warm
/// counterpart to `mimir recall-inject`. Read-mostly (the only writes are
/// the impression log, same as a normal recall, and the project row a new
/// `cwd` registers, same as the cold path). It sits behind the same gates
/// as every other route of this server: the loopback bind guard, and the
/// bearer token when one is set (`with_bearer_auth` covers `/inject` too,
/// which is why the hook script sends `MIMIR_HTTP_TOKEN`). Runs the actual
/// work on `spawn_blocking`, mirroring `MimirServer::blocking`, so a slow
/// SQLite/tokenizer call can't stall the async runtime. Never errors to the
/// caller: any failure (bad DB, panic) degrades to an empty body, matching
/// the "silence over a wrong answer" rule the floor itself already follows.
///
/// `cwd` is the directory the prompt was typed in (the hook event's `cwd`).
/// One daemon serves every project, so the scope is resolved from it per
/// request — see [`WarmState::place_for`]; a path that is not absolute
/// means nothing to a server and is ignored.
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
    axum::extract::State(state): axum::extract::State<WarmState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<String, axum::http::StatusCode> {
    let prompt = params.get("prompt").cloned().unwrap_or_default();
    let enrich = params.get("enrich").cloned().unwrap_or_default();
    let session = params.get("session").filter(|s| !s.is_empty()).cloned();
    let cwd = params
        .get("cwd")
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_absolute());
    tokio::task::spawn_blocking(move || {
        let mut engine = state.engine.lock().unwrap_or_else(|p| p.into_inner());
        // A daemon left running across an upgrade holds a connection to a
        // store a newer binary has since migrated. It must not answer from a
        // schema it was not written for; 503 sends the hook to its cold
        // path, which runs the binary that did the migrating.
        if stale_server_error(&engine.conn).is_some() {
            return Err(axum::http::StatusCode::SERVICE_UNAVAILABLE);
        }
        let place = match state.place_for(&engine, cwd.as_deref()) {
            Ok(place) => place,
            Err(err) => {
                // The cold path says nothing when it cannot resolve its
                // project; so does this one, rather than guess a scope.
                tracing::warn!(%err, "inject: project detection failed");
                return Ok(String::new());
            }
        };
        Ok(
            match mimir_core::inject::compute_in_checkout(
                &mut engine,
                &prompt,
                &enrich,
                place.scope,
                false,
                session.as_deref(),
                place.checkout.as_deref(),
            ) {
                Ok(text) => text.unwrap_or_default(),
                Err(err) => {
                    tracing::warn!(%err, "inject: compute failed");
                    String::new()
                }
            },
        )
    })
    .await
    .unwrap_or_else(|_| Ok(String::new()))
}

/// [`inject_router_in`] for a server started outside any checkout.
#[cfg(test)]
fn inject_router(state: InjectState) -> axum::Router {
    inject_router_in(state, None)
}

/// The daemon's routes, for a server started in `checkout` (the canonical
/// root of that checkout; see [`WarmState::checkout`]).
fn inject_router_in(state: InjectState, checkout: Option<String>) -> axum::Router {
    let inference = axum::Router::new()
        .route("/embed", axum::routing::post(embed_handler))
        .route("/rerank", axum::routing::post(rerank_handler))
        .with_state(state.clone());
    axum::Router::new()
        .route("/inject", axum::routing::get(inject_handler))
        .route("/inference", axum::routing::get(inference_handler))
        .with_state(WarmState::new(state, checkout))
        .merge(inference)
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
///
/// Answered from [`WarmState::models`], never from the engine: the probe
/// has a one-second budget on the client, and a request that queued behind
/// the engine mutex would time out — and read as "daemon down" — exactly
/// when the daemon is busy embedding for someone else.
async fn inference_handler(
    axum::extract::State(state): axum::extract::State<WarmState>,
) -> axum::Json<serde_json::Value> {
    axum::Json((*state.models).clone())
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
    use mimir_core::memory;

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

    /// The agent surface carries both new signals: a hit with no evidence
    /// behind it says `weak` on its line, and a page with nothing better
    /// lands in the miss ledger — while a real answer to the same question
    /// later resolves it.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_marks_weak_hits_and_feeds_the_miss_ledger() {
        let (_dir, s) = server();
        s.remember(Parameters(remember_args(
            "SCRAM auth rejects non-ASCII passwords",
            "gotcha",
        )))
        .await;
        let recall = |query: &str| {
            s.recall(Parameters(RecallArgs {
                query: query.into(),
                kind: None,
                limit: None,
                all_projects: true,
                rerank: false,
                include_superseded: false,
                as_of: None,
            }))
        };
        let open_misses = || {
            let engine = s.engine.clone();
            move || {
                let m = engine.lock().unwrap();
                mimir_core::misses::list(&m.conn, None, false, 10)
                    .unwrap()
                    .into_iter()
                    .map(|miss| miss.query)
                    .collect::<Vec<_>>()
            }
        };

        // No embedder in tests, so this is the lexical-only path. Nothing
        // shares a word: an empty page, which is a miss.
        assert_eq!(recall("sourdough hydration ratio").await, "no results");
        // One shared word of three: the hit comes back, marked weak.
        let partial = recall("passwords reset flow").await;
        assert!(
            partial.contains("SCRAM") && partial.contains(" weak]"),
            "{partial}"
        );
        assert_eq!(
            open_misses()(),
            vec!["sourdough hydration ratio", "passwords reset flow"]
        );

        s.remember(Parameters(remember_args(
            "Passwords reset flow sends a magic link, never a temporary password",
            "decision",
        )))
        .await;
        let answered = recall("passwords reset flow").await;
        let first = answered.lines().next().unwrap_or_default();
        assert!(
            first.contains("magic link") && !first.contains(" weak]"),
            "{answered}"
        );
        assert_eq!(
            open_misses()(),
            vec!["sourdough hydration ratio"],
            "a strong answer resolves its own miss and no other"
        );
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

    /// A relation Mimir derives from the code graph is refused too: the
    /// extractor deletes a file's `calls` edges before writing fresh ones,
    /// so a hand-made one disappears on the next build. And the argument's
    /// description offers exactly the relations the tool accepts.
    #[tokio::test(flavor = "multi_thread")]
    async fn link_refuses_a_relation_the_extractor_owns() {
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
                a,
                b,
                rel: Some("calls".into()),
            }))
            .await;
        assert!(
            refused.starts_with("error: link does not accept rel=\"calls\""),
            "expected a refusal, got: {refused}"
        );

        let schema = serde_json::to_string(&schemars::schema_for!(LinkArgs)).unwrap();
        for rel in Rel::authorable() {
            assert!(
                schema.contains(rel.as_str()),
                "the link tool does not offer `{rel}`: {schema}"
            );
        }
    }

    /// `link` must refuse `co_recalled`. It is derived from observed usage, and
    /// the store skips drawing a real one for any pair that already has an
    /// authored edge — so a hand-written one would not just fake a co-recall,
    /// it would permanently mask the genuine one for that pair.
    #[tokio::test(flavor = "multi_thread")]
    async fn link_refuses_the_co_recalled_rel() {
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
                rel: Some("co_recalled".into()),
            }))
            .await;
        assert!(
            refused.contains("co_recalled") && refused.contains("co-recall"),
            "expected a derived-edge refusal, got: {refused}"
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
            out.contains(
                "invalid expires_in \"soon\": expected a positive duration like 90d, 2w or 12h"
            ),
            "expected a rejection, got: {out}"
        );
        // "Rejected" has to mean nothing was written. If the memory went in
        // before the deadline was parsed, the caller's corrected retry is
        // refused as a duplicate of a memory that will never expire.
        let mut retry = remember_args("this fact has a bogus deadline", "note");
        retry.expires_in = Some("30d".into());
        let out = s.remember(Parameters(retry)).await;
        assert!(
            out.starts_with("stored"),
            "the rejected capture must not have been stored, got: {out}"
        );
    }

    /// Same contract for `confidence`: an unparseable level is refused
    /// before the write, so the corrected retry goes through.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_rejects_a_bad_confidence_before_writing() {
        let (_dir, s) = server();
        let mut args = remember_args("this fact has a bogus certainty", "note");
        args.confidence = Some("maybe".into());
        let out = s.remember(Parameters(args)).await;
        assert!(
            out.contains("invalid confidence \"maybe\": expected certain, likely or unsure"),
            "expected a rejection, got: {out}"
        );
        let mut retry = remember_args("this fact has a bogus certainty", "note");
        retry.confidence = Some("unsure".into());
        let out = s.remember(Parameters(retry)).await;
        assert!(
            out.starts_with("stored"),
            "the rejected capture must not have been stored, got: {out}"
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

    /// GET `url` and return the body; `None` when it does not answer in
    /// time or answers with an error status.
    fn http_get(url: &str, timeout: std::time::Duration) -> Option<String> {
        ureq::AgentBuilder::new()
            .timeout(timeout)
            .build()
            .get(url)
            .call()
            .ok()?
            .into_string()
            .ok()
    }

    /// Percent-encode a query value (everything but unreserved characters).
    fn enc(value: &str) -> String {
        value
            .bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    (b as char).to_string()
                }
                other => format!("%{other:02X}"),
            })
            .collect()
    }

    /// The delegation probe has a one-second budget and is sent exactly
    /// when the daemon may be busy embedding for someone else. It answers
    /// from configuration, so it must not queue behind the engine.
    #[tokio::test(flavor = "multi_thread")]
    async fn inference_probe_does_not_wait_for_the_engine() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(Mutex::new(
            Mimir::open_at(Paths::under_root(dir.path())).unwrap(),
        ));
        let router = inject_router(InjectState {
            engine: engine.clone(),
            project_id: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let answer = tokio::task::spawn_blocking(move || {
            // Stand in for a long embed: hold the engine for the whole probe.
            let _busy = engine.lock().unwrap();
            http_get(
                &format!("http://{addr}/inference"),
                std::time::Duration::from_secs(3),
            )
        })
        .await
        .unwrap();
        let answer = answer.expect("the probe must answer while the engine is busy");
        assert!(answer.contains("bge-small-en-v1.5"), "{answer}");
    }

    /// One daemon serves every project. A prompt's injection must come from
    /// the project the prompt was typed in — the scope the cold path gets
    /// from its working directory — not from wherever the daemon was
    /// started, and not from every project at once.
    #[tokio::test(flavor = "multi_thread")]
    async fn inject_scopes_to_the_project_of_the_requests_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path().join("home").as_path())).unwrap();
        let project_dir = |name: &str| {
            let root = dir.path().join(name);
            std::fs::create_dir_all(root.join(".git")).unwrap();
            root
        };
        let (infra, website) = (project_dir("infra"), project_dir("website"));
        let infra_id = engine.project_for_cwd(&infra).unwrap().unwrap().id;
        let website_id = engine.project_for_cwd(&website).unwrap().unwrap().id;
        memory::remember(
            &engine.conn,
            memory::Remember {
                text: "terraform plan fails unless VAULT_ADDR is exported first".into(),
                mtype: MemoryType::Gotcha,
                tags: Vec::new(),
                project_id: Some(infra_id),
                force: false,
                actor: mimir_core::model::Actor::Agent,
            },
        )
        .unwrap();

        // Two daemons over the same store: one started outside any project
        // (the systemd case), one started inside `website`.
        let serve = |project_id: Option<i64>, engine: Mimir| async move {
            let router = inject_router(InjectState {
                engine: Arc::new(Mutex::new(engine)),
                project_id,
            });
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, router).await;
            });
            addr
        };
        let paths = engine.paths.clone();
        let unscoped = serve(None, engine).await;
        let in_website = serve(Some(website_id), Mimir::open_at(paths).unwrap()).await;

        let prompt = enc("why does terraform plan fail on staging");
        let nested = infra.join("modules");
        std::fs::create_dir_all(&nested).unwrap();
        let outside = dir.path().join("scratch");
        std::fs::create_dir_all(&outside).unwrap();
        let ask = move |addr: std::net::SocketAddr, cwd: Option<std::path::PathBuf>| {
            let mut url = format!("http://{addr}/inject?prompt={prompt}");
            if let Some(cwd) = cwd {
                url.push_str(&format!("&cwd={}", enc(&cwd.to_string_lossy())));
            }
            http_get(&url, std::time::Duration::from_secs(10)).expect("/inject answers")
        };
        let answers = tokio::task::spawn_blocking(move || {
            [
                ask(unscoped, Some(nested)),
                ask(unscoped, Some(website.clone())),
                ask(unscoped, Some(outside)),
                ask(unscoped, None),
                ask(in_website, Some(infra)),
                ask(in_website, None),
            ]
        })
        .await
        .unwrap();
        let [from_infra, from_website, from_outside, no_cwd, other_daemon_infra, other_daemon_no_cwd] =
            answers;

        assert!(
            from_infra.contains("VAULT_ADDR"),
            "a prompt typed in the project gets its memory: {from_infra:?}"
        );
        assert_eq!(
            from_website, "",
            "a prompt typed in another project must not get it"
        );
        assert!(
            from_outside.contains("VAULT_ADDR"),
            "outside any project a question searches everything, as the cold path does: \
             {from_outside:?}"
        );
        assert!(
            no_cwd.contains("VAULT_ADDR"),
            "no cwd (an older hook script): the daemon's own scope, here every project: \
             {no_cwd:?}"
        );
        assert!(
            other_daemon_infra.contains("VAULT_ADDR"),
            "the request's cwd wins over the directory the daemon was started in: \
             {other_daemon_infra:?}"
        );
        assert_eq!(
            other_daemon_no_cwd, "",
            "no cwd: the daemon's start-up project, as before"
        );
    }

    /// A server started in a linked worktree holds two projects, and
    /// `recall` reads both in one page: memories of the memory project,
    /// code of the checkout's own. A miss is filed under the memory
    /// project, whichever half failed to answer.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_reads_memories_and_code_from_the_servers_two_projects() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        let project =
            |path: &str, name: &str| store::ensure_project(&engine.conn, path, name).unwrap().id;
        let (home, own) = (project("/repo/main", "main"), project("/repo/wt", "wt"));
        memory::remember(
            &engine.conn,
            memory::Remember {
                text: "quorum loss needs a manual failover".into(),
                mtype: MemoryType::Gotcha,
                tags: Vec::new(),
                project_id: Some(home),
                force: false,
                actor: mimir_core::model::Actor::Agent,
            },
        )
        .unwrap();
        for (owner, title) in [(own, "quorum_on_the_branch"), (home, "quorum_in_main")] {
            let mut symbol = mimir_core::model::NewNode::new(Kind::Symbol);
            symbol.title = Some(title.into());
            symbol.body = Some(format!("fn {title}() checks the quorum"));
            symbol.project_id = Some(owner);
            store::insert_node(&engine.conn, symbol).unwrap();
        }
        let s = MimirServer::new(engine, Some(own)).with_memory_project(Some(home));
        let recall = |query: &str, kind: Option<&str>, all_projects: bool| {
            s.recall(Parameters(RecallArgs {
                query: query.into(),
                kind: kind.map(str::to_string),
                limit: None,
                all_projects,
                rerank: false,
                include_superseded: false,
                as_of: None,
            }))
        };

        let page = recall("quorum", None, false).await;
        assert!(page.contains("manual failover"), "{page}");
        assert!(page.contains("quorum_on_the_branch"), "{page}");
        assert!(!page.contains("quorum_in_main"), "{page}");
        let code = recall("quorum", Some("code"), false).await;
        assert!(
            code.contains("quorum_on_the_branch") && !code.contains("quorum_in_main"),
            "{code}"
        );
        assert!(!code.contains("manual failover"), "{code}");
        let memories = recall("quorum", Some("memory"), false).await;
        assert!(
            memories.contains("manual failover") && !memories.contains("quorum_on"),
            "{memories}"
        );
        let everywhere = recall("quorum", None, true).await;
        assert!(
            everywhere.contains("quorum_in_main") && everywhere.contains("quorum_on_the_branch"),
            "{everywhere}"
        );

        // Nothing answers this, in code or anywhere: the repository's miss.
        recall("kubernetes ingress annotations", Some("code"), false).await;
        let engine = s.engine.clone();
        let misses = tokio::task::spawn_blocking(move || {
            let m = engine.lock().unwrap();
            mimir_core::misses::list(&m.conn, None, false, 10).unwrap()
        })
        .await
        .unwrap();
        let unanswered = misses
            .iter()
            .find(|miss| miss.query == "kubernetes ingress annotations")
            .expect("the miss was recorded");
        assert_eq!(unanswered.project_id, Some(home), "{misses:?}");
        // (Without a model no page is `strong`, so the recalls above are in
        // the ledger too.) No row is keyed by the checkout's own project:
        // the ledger is the memory project's, or nobody's under
        // `all_projects`.
        assert!(
            misses.iter().all(|miss| miss.project_id != Some(own)),
            "{misses:?}"
        );
    }

    /// A session handoff written in one worktree reaches no session in
    /// another through the server's `recall` or the daemon's `/inject`,
    /// although both read the repository's memory project, which holds it.
    /// `all_projects` still finds it, and the worktree that wrote it does.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_handoff_of_another_worktree_is_not_recalled_or_injected() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path().join("home").as_path())).unwrap();
        let main = dir.path().join("repo");
        let worktree = |name: &str| {
            let root = dir.path().join(name);
            let gitdir = main.join(".git/worktrees").join(name);
            std::fs::create_dir_all(&gitdir).unwrap();
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
            std::fs::write(root.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
            std::fs::write(
                gitdir.join("gitdir"),
                format!("{}\n", root.join(".git").display()),
            )
            .unwrap();
            root
        };
        let (wt_a, wt_b) = (worktree("repo-a"), worktree("repo-b"));
        let at_a = engine.projects_for_cwd(&wt_a).unwrap();
        let at_b = engine.projects_for_cwd(&wt_b).unwrap();
        let home = at_a.memory_id().unwrap();
        let handoff = match memory::remember(
            &engine.conn,
            memory::Remember {
                text: "handoff: terraform plan is half applied; VAULT_ADDR must be exported \
                       before step four"
                    .into(),
                mtype: MemoryType::Gotcha,
                tags: vec![mimir_core::context_guard::HANDOFF_TAG.into()],
                project_id: Some(home),
                force: false,
                actor: mimir_core::model::Actor::Agent,
            },
        )
        .unwrap()
        {
            memory::RememberOutcome::Created(node) => node,
            other => panic!("{other:?}"),
        };
        mimir_core::context_guard::set_handoff_checkout(
            &engine.conn,
            handoff.id,
            at_a.checkout_path().unwrap(),
        )
        .unwrap();

        let engine = Arc::new(Mutex::new(engine));
        let recall_in = |checkout: Option<i64>, all_projects: bool| {
            let s = MimirServer {
                engine: engine.clone(),
                project_id: checkout,
                memory_project_id: Some(home),
                worktree_note: None,
                origin: Arc::new(ProjectOrigin::default()),
                tool_router: MimirServer::tool_router(),
            };
            async move {
                s.recall(Parameters(RecallArgs {
                    query: "terraform plan VAULT_ADDR".into(),
                    kind: Some("memory".into()),
                    limit: None,
                    all_projects,
                    rerank: false,
                    include_superseded: false,
                    as_of: None,
                }))
                .await
            }
        };
        let in_b = recall_in(at_b.checkout_id(), false).await;
        assert!(!in_b.contains("VAULT_ADDR"), "{in_b}");
        let in_a = recall_in(at_a.checkout_id(), false).await;
        assert!(in_a.contains("VAULT_ADDR"), "{in_a}");
        let everywhere = recall_in(at_b.checkout_id(), true).await;
        assert!(everywhere.contains("VAULT_ADDR"), "{everywhere}");

        let router = inject_router(InjectState {
            engine: engine.clone(),
            project_id: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let inject_from = |cwd: &std::path::Path| {
            let url = format!(
                "http://{addr}/inject?prompt={}&cwd={}",
                enc("why does terraform plan need VAULT_ADDR"),
                enc(&cwd.to_string_lossy())
            );
            tokio::task::spawn_blocking(move || {
                http_get(&url, std::time::Duration::from_secs(10)).expect("/inject answers")
            })
        };
        let in_b = inject_from(&wt_b).await.unwrap();
        assert!(!in_b.contains("VAULT_ADDR"), "{in_b:?}");
        let in_a = inject_from(&wt_a).await.unwrap();
        assert!(in_a.contains("VAULT_ADDR"), "{in_a:?}");
    }

    /// A prompt typed in a linked worktree is injected from the repository's
    /// memories: the daemon scopes the request by the memory project of its
    /// `cwd`, which is the main checkout's, not by the worktree's own row
    /// (which holds the code graph and no memories).
    #[tokio::test(flavor = "multi_thread")]
    async fn inject_from_a_linked_worktree_reads_its_repositorys_memories() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path().join("home").as_path())).unwrap();
        // A main checkout and a linked worktree of it, as files.
        let main = dir.path().join("infra");
        let wt = dir.path().join("infra-feature");
        let gitdir = main.join(".git/worktrees/infra-feature");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(wt.join("modules")).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        let home_id = engine.project_for_cwd(&main).unwrap().unwrap().id;
        memory::remember(
            &engine.conn,
            memory::Remember {
                text: "terraform plan fails unless VAULT_ADDR is exported first".into(),
                mtype: MemoryType::Gotcha,
                tags: Vec::new(),
                project_id: Some(home_id),
                force: false,
                actor: mimir_core::model::Actor::Agent,
            },
        )
        .unwrap();
        let router = inject_router(InjectState {
            engine: Arc::new(Mutex::new(engine)),
            project_id: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let nested = wt.join("modules");
        let answer = tokio::task::spawn_blocking(move || {
            let url = format!(
                "http://{addr}/inject?prompt={}&cwd={}",
                enc("why does terraform plan fail on staging"),
                enc(&nested.to_string_lossy())
            );
            http_get(&url, std::time::Duration::from_secs(10)).expect("/inject answers")
        })
        .await
        .unwrap();
        assert!(answer.contains("VAULT_ADDR"), "{answer:?}");
    }

    /// A failure is marked whichever way the tool produced it. An unknown
    /// `kind` used to come back as a bare sentence, indistinguishable by its
    /// first word from a search result.
    #[tokio::test(flavor = "multi_thread")]
    async fn every_failed_tool_call_starts_with_error() {
        let (_dir, s) = server();
        let unknown_kind = s
            .recall(Parameters(RecallArgs {
                query: "anything".into(),
                kind: Some("everything".into()),
                limit: None,
                all_projects: false,
                rerank: false,
                include_superseded: false,
                as_of: None,
            }))
            .await;
        assert_eq!(
            unknown_kind,
            "error: unknown kind 'everything' (all|memory|doc|code)"
        );
        assert_eq!(
            mark_error("error: already marked".into()),
            "error: already marked"
        );
        assert_eq!(mark_error("plain".into()), "error: plain");
    }

    /// A server started before an upgrade keeps its connection after a
    /// newer binary migrates the store. Every tool has to say so, with the
    /// fix, instead of answering from a schema it was not written for.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_older_than_its_store_says_so_on_every_tool() {
        let (dir, s) = server();
        let healthy = s.status().await;
        assert!(healthy.starts_with("project "), "{healthy}");

        // What a newer binary's first `Mimir::open` does to the store this
        // server already has open.
        let newer = mimir_core::db::migrations::SCHEMA_VERSION + 1;
        rusqlite::Connection::open(Paths::under_root(dir.path()).db_file)
            .unwrap()
            .execute_batch(&format!("PRAGMA user_version = {newer}"))
            .unwrap();

        let expected = format!(
            "error: this Mimir server is older than its store (schema {newer}, this server \
             supports {}) — restart it (in Claude Code: /mcp, then reconnect mimir)",
            mimir_core::db::migrations::SCHEMA_VERSION
        );
        assert_eq!(s.status().await, expected);
        let recall = s
            .recall(Parameters(RecallArgs {
                query: "anything".into(),
                kind: None,
                limit: None,
                all_projects: false,
                rerank: false,
                include_superseded: false,
                as_of: None,
            }))
            .await;
        assert_eq!(recall, expected);
        let stored = s
            .remember(Parameters(remember_args("must not be written", "note")))
            .await;
        assert_eq!(stored, expected, "a write is refused too");
    }

    /// The same for the daemon's `/inject`: once the store is newer than
    /// the daemon, it answers 503, which sends the hook script to its cold
    /// path — the newer binary — instead of injecting from a schema this
    /// process was not written for.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_daemon_older_than_its_store_sends_the_hook_to_the_cold_path() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under_root(dir.path());
        let engine = Mimir::open_at(paths.clone()).unwrap();
        let router = inject_router(InjectState {
            engine: Arc::new(Mutex::new(engine)),
            project_id: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let url = format!("http://{addr}/inject?prompt=anything");
        let ask = move || http_get(&url, std::time::Duration::from_secs(5));

        let healthy = tokio::task::spawn_blocking(ask.clone()).await.unwrap();
        assert_eq!(healthy.as_deref(), Some(""), "answers, with nothing to say");

        let newer = mimir_core::db::migrations::SCHEMA_VERSION + 1;
        rusqlite::Connection::open(paths.db_file)
            .unwrap()
            .execute_batch(&format!("PRAGMA user_version = {newer}"))
            .unwrap();
        let stale = tokio::task::spawn_blocking(ask).await.unwrap();
        assert_eq!(stale, None, "an error status, so the hook falls back");
    }

    /// The one re-open failure a background loop stops on is the store
    /// having outgrown the binary; anything else may clear and is retried.
    #[test]
    fn only_a_newer_schema_ends_background_reopens() {
        let too_new = mimir_core::error::Error::SchemaTooNew {
            found: mimir_core::db::migrations::SCHEMA_VERSION + 1,
            supported: mimir_core::db::migrations::SCHEMA_VERSION,
        };
        assert!(store_outgrew_server(&too_new));
        assert!(store_outgrew_server(&too_new), "and keeps saying so");
        let transient = mimir_core::error::Error::Invalid("disk hiccup".into());
        assert!(!store_outgrew_server(&transient));
    }

    /// `status` describes the project the tools use: the one frozen at
    /// start-up, with the reason recorded then.
    #[tokio::test(flavor = "multi_thread")]
    async fn status_reports_the_project_frozen_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        let repo = dir.path().join("frozen-repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let project = engine.project_for_cwd(&repo).unwrap().unwrap();
        let in_project = MimirServer::new(engine, Some(project.id)).with_origin(ProjectOrigin {
            dir: Some(repo.clone()),
            via: Some("git"),
        });
        // The marker disappearing later changes nothing the tools do, so it
        // changes nothing status says.
        std::fs::remove_dir_all(repo.join(".git")).unwrap();
        let status = in_project.status().await;
        assert!(
            status.starts_with("project frozen-repo  [via: git]\n"),
            "{status}"
        );

        let bare = dir.path().join("scratch");
        std::fs::create_dir_all(&bare).unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path())).unwrap();
        let outside = MimirServer::new(engine, None).with_origin(ProjectOrigin {
            dir: Some(bare.clone()),
            via: None,
        });
        // A marker appearing later does not make status claim a project the
        // tools are not scoped to.
        std::fs::create_dir_all(bare.join(".git")).unwrap();
        let status = outside.status().await;
        let first = status.lines().next().unwrap();
        assert!(first.contains("none — global scope"), "{first}");
        assert!(first.contains(&bare.display().to_string()), "{first}");
        assert!(
            first.contains("recall searches every project"),
            "says what a read does, not only where a write goes: {first}"
        );
    }

    /// A relative `outline` target resolves against the directory the
    /// server scopes from — which is what `MIMIR_PROJECT` sets — not
    /// against the process's working directory.
    #[tokio::test(flavor = "multi_thread")]
    async fn outline_resolves_a_relative_target_against_the_server_directory() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Mimir::open_at(Paths::under_root(dir.path().join("home").as_path())).unwrap();
        let scoped = dir.path().join("elsewhere");
        std::fs::create_dir_all(&scoped).unwrap();
        std::fs::write(
            scoped.join("only_here.rs"),
            "pub fn found_through_the_server_directory() {}\n",
        )
        .unwrap();
        let s = MimirServer::new(engine, None).with_origin(ProjectOrigin {
            dir: Some(scoped),
            via: None,
        });
        let outline = s
            .outline(Parameters(OutlineArgs {
                target: Some("only_here.rs".into()),
            }))
            .await;
        assert!(
            outline.contains("found_through_the_server_directory"),
            "{outline}"
        );
    }

    fn write(dir: &std::path::Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    /// A project on disk (`main → run → slugify`) and an engine that knows
    /// it, with the graph not built yet.
    fn graph_fixture() -> (tempfile::TempDir, tempfile::TempDir, Mimir, i64) {
        let home = tempfile::tempdir().unwrap();
        let code = tempfile::tempdir().unwrap();
        write(code.path(), "src/util.rs", "pub fn slugify() {}\n");
        write(
            code.path(),
            "src/main.rs",
            "fn main() { run(); }\n\nfn run() { slugify(); }\n",
        );
        let engine = Mimir::open_at(Paths::under_root(home.path())).unwrap();
        let project =
            store::ensure_project(&engine.conn, &code.path().to_string_lossy(), "probe").unwrap();
        (home, code, engine, project.id)
    }

    fn graph_args(op: &str, symbol: Option<&str>) -> GraphArgs {
        GraphArgs {
            op: op.into(),
            symbol: symbol.map(str::to_string),
            to: None,
            files: vec![],
            depth: None,
            limit: None,
        }
    }

    /// Before the graph exists, an empty answer would read as "nothing
    /// calls this". The tools say the graph is not there, and whether
    /// anything is going to build it.
    #[tokio::test(flavor = "multi_thread")]
    async fn graph_and_peek_say_when_the_project_has_no_graph_yet() {
        let (_home, _code, mut engine, pid) = graph_fixture();
        engine.config.auto.graph = true;
        let s = MimirServer::new(engine, Some(pid));
        let got = s
            .graph(Parameters(graph_args("callers", Some("slugify"))))
            .await;
        assert!(
            got.starts_with("error: this project's code graph is not built yet")
                && got.contains("has not finished"),
            "{got}"
        );
        let got = s
            .peek(Parameters(PeekArgs {
                symbol: "slugify".into(),
            }))
            .await;
        assert!(
            got.starts_with("error: this project's code graph is not built yet"),
            "{got}"
        );

        let (_home, _code, mut engine, pid) = graph_fixture();
        engine.config.auto.graph = false;
        let s = MimirServer::new(engine, Some(pid));
        let got = s.graph(Parameters(graph_args("hubs", None))).await;
        assert!(
            got.starts_with("error: this project has no code graph yet")
                && got.contains("mimir graph build"),
            "{got}"
        );
    }

    /// The tool prints what `mimir graph …` prints: the same renderer, so
    /// the same words for the empty answers, and every failure starts with
    /// "error: " so an agent can tell an answer from a refusal by its first
    /// word.
    #[tokio::test(flavor = "multi_thread")]
    async fn graph_tool_shares_the_cli_wording_and_marks_errors() {
        let (_home, code, mut engine, pid) = graph_fixture();
        let project = store::get_node(&engine.conn, pid).unwrap();
        mimir_graph::update(&mut engine.conn, &project, code.path()).unwrap();
        let s = MimirServer::new(engine, Some(pid));
        let ask = |args: GraphArgs| s.graph(Parameters(args));

        let callers = ask(graph_args("callers", Some("slugify"))).await;
        let lines: Vec<&str> = callers.lines().collect();
        assert_eq!(lines.len(), 3, "{callers}");
        assert!(lines[0].ends_with("slugify"), "{callers}");
        assert!(
            lines[1].starts_with("  c:") && lines[1].ends_with(" run"),
            "{callers}"
        );
        assert!(
            lines[2].starts_with("    c:") && lines[2].ends_with(" main"),
            "{callers}"
        );

        let calls = ask(graph_args("calls", Some("slugify"))).await;
        assert!(calls.ends_with("slugify\nno callees found"), "{calls}");
        let none = ask(graph_args("callers", Some("main"))).await;
        assert!(none.ends_with("main\nno callers found"), "{none}");

        let mut path = graph_args("path", Some("slugify"));
        path.to = Some("main".into());
        assert_eq!(ask(path).await, "no call path slugify → main");

        let mut impact = graph_args("impact", None);
        impact.files = vec!["src/util.rs".into()];
        let hit = ask(impact).await;
        assert!(hit.ends_with("(2 affected symbols within 3 hops)"), "{hit}");

        // Every refusal is marked.
        let mut impact = graph_args("impact", None);
        impact.files = vec!["src/nope.rs".into()];
        assert_eq!(
            ask(impact).await,
            "error: none of those paths are in the code graph: they are not source files \
             Mimir indexes, or they were added since the last build"
        );
        assert_eq!(
            ask(graph_args("impact", None)).await,
            "error: missing 'files' argument"
        );
        assert_eq!(
            ask(graph_args("callers", None)).await,
            "error: missing 'symbol' argument"
        );
        assert_eq!(
            ask(graph_args("path", Some("main"))).await,
            "error: missing 'to' argument"
        );
        assert_eq!(
            ask(graph_args("frobnicate", None)).await,
            "error: unknown op 'frobnicate' (callers|calls|impact|node|path|hubs)"
        );
        assert_eq!(
            ask(graph_args("callers", Some("no_such_symbol"))).await,
            "error: not found: symbol 'no_such_symbol'"
        );

        let global = MimirServer::new(
            Mimir::open_at(Paths::under_root(tempfile::tempdir().unwrap().path())).unwrap(),
            None,
        );
        let got = global.graph(Parameters(graph_args("hubs", None))).await;
        assert!(got.starts_with("error: no project detected"), "{got}");
    }

    /// An answer read from a graph that is behind the working tree says so
    /// in its last line, and `peek` says when the one file it slices has
    /// moved on. Neither writes: the note is still there on the next call.
    #[tokio::test(flavor = "multi_thread")]
    async fn graph_and_peek_note_a_stale_graph() {
        let (_home, code, mut engine, pid) = graph_fixture();
        let project = store::get_node(&engine.conn, pid).unwrap();
        mimir_graph::update(&mut engine.conn, &project, code.path()).unwrap();
        let s = MimirServer::new(engine, Some(pid));
        const NOTE: &str =
            "note: the code graph is 1 file(s) behind the working tree — run `mimir graph build`";

        let fresh = s
            .graph(Parameters(graph_args("callers", Some("slugify"))))
            .await;
        assert!(!fresh.contains("note:"), "{fresh}");
        let peeked = s
            .peek(Parameters(PeekArgs {
                symbol: "slugify".into(),
            }))
            .await;
        assert!(!peeked.contains("note:"), "{peeked}");

        write(
            code.path(),
            "src/util.rs",
            "// moved down\n\npub fn slugify() {}\n",
        );
        for _ in 0..2 {
            let stale = s
                .graph(Parameters(graph_args("callers", Some("slugify"))))
                .await;
            assert_eq!(stale.lines().last(), Some(NOTE), "{stale}");
            assert!(
                stale.starts_with("c:"),
                "the answer still comes first: {stale}"
            );
        }
        let node = s
            .graph(Parameters(graph_args("node", Some("slugify"))))
            .await;
        assert_eq!(node.lines().last(), Some(NOTE), "{node}");
        // A miss gets the note too: it is often the reason for the miss.
        let missing = s
            .graph(Parameters(graph_args("callers", Some("not_built_yet"))))
            .await;
        assert!(
            missing.starts_with("error: not found") && missing.ends_with(NOTE),
            "{missing}"
        );

        let peeked = s
            .peek(Parameters(PeekArgs {
                symbol: "slugify".into(),
            }))
            .await;
        assert_eq!(
            peeked.lines().last(),
            Some(
                "note: src/util.rs changed since the code graph was built — these lines may be \
                 off; run `mimir graph build`"
            ),
            "{peeked}"
        );
        // A symbol in a file that did not change is peeked without comment.
        let other = s
            .peek(Parameters(PeekArgs {
                symbol: "run".into(),
            }))
            .await;
        assert!(!other.contains("note:"), "{other}");
    }

    /// A graph an older build resolved answers as before, and says in its
    /// last line that a full rebuild would apply this build's rules. When
    /// the graph is also behind the working tree, that note comes first.
    #[tokio::test(flavor = "multi_thread")]
    async fn graph_tool_notes_a_graph_built_with_older_rules() {
        const OLD_RULES: &str = "note: this graph was built with an older Mimir's call and \
            import rules — `mimir graph build --full` rebuilds it with the current ones";
        let (_home, code, mut engine, pid) = graph_fixture();
        let project = store::get_node(&engine.conn, pid).unwrap();
        mimir_graph::update(&mut engine.conn, &project, code.path()).unwrap();
        // As a build that recorded no rules version left it.
        engine
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.graph_rules') WHERE id = ?1",
                [pid],
            )
            .unwrap();
        let s = MimirServer::new(engine, Some(pid));

        let callers = s
            .graph(Parameters(graph_args("callers", Some("slugify"))))
            .await;
        assert!(callers.starts_with("c:"), "{callers}");
        assert_eq!(callers.lines().last(), Some(OLD_RULES), "{callers}");
        assert_eq!(callers.matches("note:").count(), 1, "{callers}");

        write(
            code.path(),
            "src/util.rs",
            "// moved down\n\npub fn slugify() {}\n",
        );
        let callers = s
            .graph(Parameters(graph_args("callers", Some("slugify"))))
            .await;
        let lines: Vec<&str> = callers.lines().collect();
        assert_eq!(lines[lines.len() - 1], OLD_RULES, "{callers}");
        assert!(
            lines[lines.len() - 2].starts_with("note: the code graph is 1 file(s) behind"),
            "{callers}"
        );
        // `hubs` ranks the graph as a whole and carries neither.
        let hubs = s.graph(Parameters(graph_args("hubs", None))).await;
        assert!(!hubs.contains("note:"), "{hubs}");
    }
}
