mod brief_cmd;
mod commands;
mod context_cmd;
mod context_guard_cmd;
mod dashboard;
mod filters;
mod fsutil;
mod graph_cmd;
mod graph_viz;
mod infer;
mod mcp;
mod project_cmd;
mod proxy_cmd;
mod report;
mod review_cmd;
mod rewrite_cmd;
mod rules_cmd;
mod run_cmd;
mod savings_cmd;
mod sync;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "mimir",
    version,
    about = "Unified local-first memory for AI coding agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Emit JSONL instead of the human/agent text format.
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Create config + database and print setup instructions.
    Init {
        /// Skip the embedding-model download (search stays BM25-only).
        #[arg(long)]
        no_model: bool,
        /// Also install the opt-in Claude Code hooks (command filter + rules).
        #[arg(long)]
        hooks: bool,
        /// Also install the opt-in per-prompt auto-recall hook (implies
        /// --hooks): injects a relevant memory into UserPromptSubmit, when
        /// one clears the relevance floor (see `recall-inject`).
        #[arg(long)]
        auto_recall: bool,
        /// Also install the opt-in context-window guard (implies --hooks):
        /// "off" (default, installs nothing new) | "pause" (nudge to
        /// deliberately /clear or /compact) | "handoff" (also
        /// auto-save/restore a handoff memory across the clear). See
        /// `mimir context-guard` and `[hooks] context_guard` in config.toml.
        #[arg(long)]
        context_guard: Option<String>,
    },
    /// Rewrite a shell command to use `mimir run` (used by the PreToolUse hook).
    Rewrite {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        cmd: Vec<String>,
    },
    /// Print at most one relevant memory for a prompt, or nothing if none
    /// clears the relevance floor (used by the UserPromptSubmit hook —
    /// see `mimir init --auto-recall`). Always exits 0.
    RecallInject {
        /// Working-tree enrichment signal (changed-file stems), passed
        /// separately from `prompt` — see `mimir_core::inject::compute`.
        /// Must be given before `prompt` (or after `--`): `prompt` is a
        /// trailing var-arg and would otherwise swallow it.
        #[arg(long)]
        enrich: Option<String>,
        /// Per-session dedup: skip a memory already injected earlier in
        /// this session (see `inject::compute_with_session`). Same
        /// positional constraint as `--enrich`.
        #[arg(long)]
        session: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        prompt: Vec<String>,
    },
    /// Store overview: counts by kind, scope, database size.
    Status,
    /// Activity table: day / week / month / year / all-time.
    Report,
    /// Token-savings analytics (outline/peek/filter/proxy).
    Savings {
        /// Print a compact one-line segment (for a statusline).
        #[arg(long)]
        oneline: bool,
    },
    /// Count tokens in stdin or the given text (Mimir's bundled tokenizer).
    Tokens {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        text: Vec<String>,
    },
    /// Health checks: database integrity, FTS availability, model presence.
    Doctor {
        /// Watchdog mode: silent when healthy, prints only failures and
        /// exits non-zero (for timers/cron; see contrib/mimir-watchdog.timer).
        #[arg(long)]
        check: bool,
    },
    /// Capture a memory (refuses near-duplicates unless --force).
    Remember {
        /// Text to remember.
        #[arg(required = true)]
        text: Vec<String>,
        /// insight|decision|gotcha|idea|note|person
        #[arg(short = 't', long = "type", default_value = "note")]
        mtype: String,
        /// Comma-separated tags.
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Store cross-project (global) even when inside a project.
        #[arg(short, long)]
        global: bool,
        /// Store even if a near-duplicate exists.
        #[arg(long)]
        force: bool,
        /// Link the new memory to an existing node.
        #[arg(long, value_name = "REF")]
        link: Option<String>,
        /// Author-declared trigger phrase(s) that bypass the inferred-
        /// relevance floor on auto-recall (repeatable) — see
        /// `memory::sanitize_fires_when`.
        #[arg(long = "fires-when")]
        fires_when: Vec<String>,
        /// Guard-anchor pattern(s) (repeatable): surface this memory on
        /// PreToolUse when a matching file is edited/written, or mentioned
        /// in a Bash command — e.g. `--anchor "file:store.rs"`. See
        /// `mimir_core::anchors`.
        #[arg(long = "anchor")]
        anchors: Vec<String>,
        /// Declare when this memory stops being true, relative to now —
        /// e.g. `90d`, `2w`, `12h`. Units: h/d/w only (months and years
        /// have no fixed length; use `365d`). Once passed, the memory is
        /// hidden from recall exactly like a superseded one.
        #[arg(long = "expires-in", value_name = "DURATION")]
        expires_in: Option<String>,
        /// Free-text condition that would falsify this memory — e.g.
        /// "when upstream ships the fix". Deliberately does NOT gate
        /// recall (nothing can evaluate it); it travels with the memory
        /// so the reader can see what would make it wrong.
        #[arg(long = "resolves-when", value_name = "CONDITION")]
        resolves_when: Option<String>,
        /// How sure you are: certain | likely | unsure. Records the
        /// author's certainty, which is a different thing from how often
        /// the memory gets used — so a guess stays visibly a guess no
        /// matter how much it is recalled. Does not affect ranking.
        #[arg(long, value_name = "LEVEL")]
        confidence: Option<String>,
        /// When the fact became true in the world (YYYY-MM-DD) — its *valid*
        /// time, which is not when you wrote it down. Pairs with
        /// `--valid-to`; either may be given alone.
        #[arg(long = "valid-from", value_name = "DATE")]
        valid_from: Option<String>,
        /// When the fact stopped being true (YYYY-MM-DD). Unlike
        /// `--expires-in` this does NOT hide the memory: a fact that was true
        /// until 2024 is still worth surfacing, labelled, because the reader
        /// often needs exactly that. Use `--expires-in` to retire something.
        #[arg(long = "valid-to", value_name = "DATE")]
        valid_to: Option<String>,
    },
    /// Search memories (and later docs/code) with hybrid ranking.
    Recall {
        #[arg(required = true)]
        query: Vec<String>,
        /// all|memory|doc|code
        #[arg(long, default_value = "all")]
        kind: String,
        /// Only cross-project (global) results.
        #[arg(short, long)]
        global: bool,
        /// Search every project (no scope filter).
        #[arg(long)]
        all: bool,
        /// Only results newer than e.g. 12h, 7d, 2w, 3m, 1y.
        #[arg(long)]
        since: Option<String>,
        #[arg(short = 'n', long)]
        limit: Option<usize>,
        /// Drop hits scoring below this fused relevance score (the value shown
        /// in `--json`). The score is scale-free (RRF), so tune it empirically.
        #[arg(long)]
        min_score: Option<f64>,
        /// Print full bodies instead of one-line summaries.
        #[arg(long)]
        full: bool,
        /// Rescore the top candidates with a cross-encoder (needs the
        /// reranker model: `mimir embed --fetch --rerank`).
        #[arg(long)]
        rerank: bool,
        /// Show nodes linked to each hit (memories on code, code on memories).
        #[arg(long)]
        linked: bool,
        /// Include superseded memories (default hides them).
        #[arg(long)]
        include_superseded: bool,
        /// Answer as the store stood on this date (YYYY-MM-DD) instead of
        /// now: memories deleted or superseded since come back, and ones
        /// written later are absent. This is transaction-time travel — what
        /// you *knew* then — which is a different question from what was
        /// *true* then (see `--valid-from`/`--valid-to` on `remember`).
        #[arg(long, value_name = "DATE")]
        as_of: Option<String>,
    },
    /// Show full records by reference (logs the access).
    Get {
        #[arg(required = true)]
        refs: Vec<String>,
    },
    /// Recent memories, newest first.
    List {
        /// Filter by memory type.
        #[arg(short = 't', long = "type")]
        mtype: Option<String>,
        /// Filter by tag.
        #[arg(long)]
        tag: Option<String>,
        #[arg(short, long)]
        global: bool,
        #[arg(long)]
        all: bool,
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
    },
    /// Soft-delete a node (--hard to remove permanently).
    Forget {
        reference: String,
        #[arg(long)]
        hard: bool,
        /// Why. Recorded in the mutation ledger (`mimir audit`), never the
        /// value itself — a deletion nobody can account for is the thing the
        /// ledger exists to prevent.
        #[arg(long)]
        reason: Option<String>,
    },
    /// Mark OLD as superseded by NEW (OLD stops surfacing in recall, kept as history).
    Supersede {
        /// The old node reference to retire.
        old: String,
        /// The new node reference that replaces it.
        #[arg(long)]
        by: String,
        /// Why the replacement happened. Recorded in `mimir audit`.
        #[arg(long)]
        reason: Option<String>,
    },
    /// Set guard anchors on an EXISTING memory, so it surfaces at act-time
    /// (PreToolUse) when a matching file is edited or command is run.
    /// `--anchor` on `remember` only covers capture time; this is how a
    /// memory written before you knew which file it guards gets wired up.
    Anchor {
        /// Memory reference (m:ABCDEF or ULID).
        reference: String,
        /// Pattern(s), repeatable — e.g. `--pattern "store.rs"`,
        /// `--pattern "file:deploy.sh"`. Replaces any existing set.
        #[arg(long = "pattern", required = true)]
        patterns: Vec<String>,
    },
    /// Update a memory's text, title, type, or tags.
    Edit {
        reference: String,
        /// Replacement text (optional when only changing metadata).
        text: Vec<String>,
        #[arg(long)]
        title: Option<String>,
        #[arg(short = 't', long = "type")]
        mtype: Option<String>,
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,
        /// Exempt from decay and consolidation merging.
        #[arg(long, conflicts_with = "unpin")]
        pin: bool,
        #[arg(long)]
        unpin: bool,
    },
    /// Move a memory to another project, or to global scope.
    ///
    /// A memory's project is fixed at `remember` time from the session's
    /// working directory, so a session started in the wrong directory files it
    /// under the wrong project. This re-files it after the fact; the change
    /// replicates across a sync hub like any edit.
    Reproject {
        /// The memory to move.
        reference: String,
        /// Target project by name. Prefers a local project over a synced
        /// shadow of the same name.
        #[arg(long, conflicts_with = "global")]
        project: Option<String>,
        /// Move to global scope (no project).
        #[arg(long)]
        global: bool,
    },
    /// Create an edge between two nodes.
    Link {
        a: Option<String>,
        b: Option<String>,
        /// links|about|relates|mentions|describes|supersedes…
        #[arg(long, default_value = "relates")]
        rel: String,
        /// Auto-link memories to code symbols their text mentions
        /// (current project; precision-first heuristic, idempotent).
        #[arg(long)]
        scan: bool,
        /// With --scan: show what would be linked without writing.
        #[arg(long)]
        dry_run: bool,
        /// With --scan: scan every project that has a code graph, not just
        /// the current one. Global memories (most of them) can match
        /// symbols in any project, so scanning one at a time leaves the
        /// rest unlinked.
        #[arg(long)]
        all_projects: bool,
    },
    /// Manage docs collections.
    Docs {
        #[command(subcommand)]
        cmd: DocsCmd,
    },
    /// Manage code collections: source chunked by symbol boundary (not
    /// just signatures) for recall. `docs list`/`remove`/`note` also work
    /// on code collections — this only adds `add`. Idea credit: nworks3d's
    /// THOR fork of Mimir (see CHANGELOG.md).
    Code {
        #[command(subcommand)]
        cmd: CodeCmd,
    },
    /// (Re)index docs collections incrementally.
    Index {
        /// Collection name or path; omit to index all.
        name: Option<String>,
    },
    /// Embed new/changed content for semantic recall.
    Embed {
        /// Allow downloading the model if it isn't cached yet.
        #[arg(long)]
        fetch: bool,
        /// With --fetch: also download the reranker model.
        #[arg(long)]
        rerank: bool,
    },
    /// Strengthen or weaken a memory's ranking (explicit feedback).
    Mark {
        reference: String,
        /// This was useful (+1 strength).
        #[arg(long, conflicts_with = "noise")]
        useful: bool,
        /// This was noise (-1 strength).
        #[arg(long)]
        noise: bool,
    },
    /// Run the four consolidation passes (dedup, contradictions,
    /// distillation, decay archival).
    Consolidate {
        /// Report without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Whether memories are attached to something Mimir can re-check, and
    /// which of those attachments have since broken.
    Grounding {
        /// List the memories whose linked artifact is gone.
        #[arg(long)]
        stale: bool,
        /// How many to list.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// What the secret guard turned away — fingerprints and counts, never
    /// the refused values.
    Refusals {
        /// How many rows to show (most recently offered first).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// What was done to a memory, and by whom: forgets, supersessions, edits,
    /// archival. Records the change, never the changed value.
    Audit {
        /// One memory (id or `m:ABCDEF`). Omit for the whole store, newest first.
        reference: Option<String>,
        /// How many rows to show when no reference is given.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// A memory's life: every wording it has had, and every change made to
    /// it. Correction is not amnesia — "what did this used to say?" keeps
    /// working after a supersession or an edit.
    History {
        /// The memory (id or `m:ABCDEF`).
        reference: String,
    },
    /// Findings the store noticed and deliberately will not decide for you:
    /// contradicting memories, falsified grounding links, expired facts whose
    /// end-condition nobody confirmed, and deleted facts that keep coming
    /// back. Nothing here resolves on its own — that is the point.
    Review {
        /// Decide this finding (from `mimir review`). Omit to list.
        id: Option<i64>,
        /// Both are right, or it isn't a problem. Changes nothing.
        #[arg(long, group = "disposition")]
        keep: bool,
        /// Retire the older memory in favour of the newer.
        #[arg(long, group = "disposition")]
        supersede: bool,
        /// Not worth acting on, and not worth showing again.
        #[arg(long, group = "disposition")]
        dismiss: bool,
        /// Why. This is what carries your own words into `mimir audit`.
        #[arg(long)]
        reason: Option<String>,
        /// How many findings to list.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Import memories from the tools Mimir replaces.
    Import {
        #[command(subcommand)]
        cmd: ImportCmd,
    },
    /// Dump all live nodes and edges as JSONL (backup / migration).
    Export,
    /// Generate a self-contained HTML stats dashboard.
    Dashboard {
        /// Output path (default: temp dir).
        #[arg(short, long)]
        out: Option<String>,
        /// Open in the default browser.
        #[arg(long)]
        open: bool,
    },
    /// Run the MCP server. Default: stdio (what Claude Code launches). Pass
    /// `--http <addr>` to serve Streamable-HTTP for remote clients instead.
    Mcp {
        /// Serve MCP over Streamable-HTTP at this address (e.g. 127.0.0.1:8077)
        /// instead of stdio. Carries no auth itself — bind to localhost and
        /// front it with TLS + an auth gate (tunnel / reverse proxy).
        #[arg(long)]
        http: Option<String>,
        /// Allow --http to bind a non-loopback address. Without this flag,
        /// non-loopback binds are refused: the transport has no auth, so
        /// anyone who can reach the port owns the store. Only pass it when
        /// the port is unreachable from outside (e.g. a container whose port
        /// is published to 127.0.0.1 only) or an auth gate fronts it.
        #[arg(long, requires = "http")]
        http_allow_remote: bool,
        /// Require this bearer token on the HTTP transport (checked as
        /// `Authorization: Bearer <token>`). Defense-in-depth on top of the
        /// loopback bind — a misconfigured proxy or exposed port no longer
        /// means instant full store access. Prefer the `MIMIR_HTTP_TOKEN` env
        /// var over this flag so the token stays out of process lists; the
        /// flag wins if both are set. Omit both to keep the prior behavior.
        #[arg(long, requires = "http")]
        http_token: Option<String>,
    },
    /// Run the warm daemon: a thin alias for `mimir mcp --http <addr>`
    /// where <addr> comes from `[hooks] inject_url` (host:port only, scheme
    /// and /inject path stripped) — the same config key the auto-recall
    /// hook already reads, so one setting drives both sides. Does nothing
    /// beyond that: no auto-spawn, no process supervision — see
    /// contrib/mimir-daemon.service to run it as a systemd unit.
    Daemon,
    /// Run the optional local API proxy (prompt-cache optimization; see docs/proxy.md).
    Proxy {
        /// Address to bind (default from [proxy] config, 127.0.0.1:8788).
        #[arg(long)]
        bind: Option<String>,
        /// Measure only: forward request bodies unchanged.
        #[arg(long)]
        dry_run: bool,
        /// Disable the prompt-cache breakpoint pass.
        #[arg(long)]
        no_cache: bool,
        /// TTL for the breakpoints we add: 5m (default) or 1h. 1h doubles the
        /// write cost, so it only pays off when turns idle >5min apart.
        #[arg(long, value_name = "5m|1h")]
        cache_ttl: Option<String>,
        /// Disable the (safe) repeated-block dedup pass.
        #[arg(long)]
        no_dedup: bool,
        /// Enable lossy pruning of stale tool results (off by default).
        #[arg(long)]
        prune: bool,
        /// Reject requests estimated over this many input tokens (0 = off).
        /// A runaway circuit breaker: rejects, never truncates.
        #[arg(long, value_name = "N")]
        max_request_tokens: Option<usize>,
    },
    /// Sync memories with the central store (optional; see docs/sync.md).
    Sync {
        #[command(subcommand)]
        cmd: Option<SyncCmd>,
    },
    /// Run a sync hub other installs push/pull against (`[sync]` server mode).
    Serve {
        /// Address to bind, e.g. 0.0.0.0:7777 for tailnet access.
        #[arg(long, default_value = "127.0.0.1:7777")]
        bind: String,
    },
    /// Build and query the code graph of the current project.
    Graph {
        #[command(subcommand)]
        cmd: GraphCmd,
    },
    /// Dense signature map of a file or directory (read structure, not bodies).
    Outline {
        /// File or directory (default: current directory).
        #[arg(default_value = ".")]
        target: String,
    },
    /// Print one symbol's body by name (resolved via the code graph).
    Peek {
        /// Symbol: short id, qualified name, or unique bare name.
        symbol: String,
    },
    /// Per-project rules pack (architecture/conventions surfaced at session start).
    Rules {
        #[command(subcommand)]
        cmd: RulesCmd,
    },
    /// Session brief: capped digest of drift-preventing memories (gotchas +
    /// decisions) for session start. `mimir brief` previews it with scores;
    /// `mimir brief show` is the SessionStart hook entry (stdin JSON,
    /// silent unless `[brief] enabled = true`).
    Brief {
        #[command(subcommand)]
        cmd: Option<BriefCmd>,
    },
    /// Project identity for cross-machine sync (the `.mimir` marker).
    Project {
        #[command(subcommand)]
        cmd: ProjectCmd,
    },
    /// Context-window guard hook-event entry points (see `mimir init
    /// --context-guard` and `[hooks] context_guard`). Not meant to be run
    /// by hand — Claude Code pipes each event's JSON on stdin.
    ContextGuard {
        #[command(subcommand)]
        cmd: ContextGuardCmd,
    },
    /// Run a command, printing token-lean output (drops build/progress noise).
    Run {
        /// Print raw, unfiltered output (debugging).
        #[arg(long)]
        raw: bool,
        /// The command to run, e.g. `mimir run -- cargo build`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        cmd: Vec<String>,
    },
}

#[derive(Subcommand)]
enum RulesCmd {
    /// Set/replace the rules pack (text from args, or stdin if omitted).
    Set { text: Vec<String> },
    /// Print the rules pack (what the SessionStart hook injects).
    Show,
    /// Remove the rules pack.
    Clear,
}

#[derive(Subcommand)]
enum BriefCmd {
    /// SessionStart hook entry — reads the hook event JSON on stdin.
    Show,
}

#[derive(Subcommand)]
enum ProjectCmd {
    /// Write/refresh `.mimir` with a stable id (and `--sync` to replicate this
    /// project's memories across machines). Commit the file so every clone shares it.
    Init {
        #[arg(long)]
        sync: bool,
    },
}

#[derive(Subcommand)]
enum ContextGuardCmd {
    /// UserPromptSubmit: nudge when the transcript crosses the threshold.
    Prompt,
    /// PreCompact: block an auto-compact over threshold with the guard message.
    Precompact,
    /// SessionStart: marker/nag bookkeeping, and (handoff mode) restore.
    SessionStart,
    /// PreToolUse (Bash/Edit/Write): surface a matching guard-anchored memory.
    Pretool,
}

#[derive(Subcommand)]
enum ImportCmd {
    /// OpenBrain `list_thoughts` text export (file or - for stdin).
    Openbrain { file: String },
    /// A Claude Code auto-memory directory (the per-project memory/ dir).
    ClaudeMemory { dir: String },
    /// Register the collections from a qmd index.yml, then `mimir index`.
    Qmd {
        /// Path to index.yml (default: ~/.config/qmd/index.yml).
        file: Option<String>,
    },
}

#[derive(Subcommand)]
enum SyncCmd {
    /// Send local changes to the central store.
    Push,
    /// Fetch and merge remote changes.
    Pull,
    /// Show sync configuration and pending changes.
    Status,
    /// Print the active sync token (env var, else the hub's persisted token) so
    /// it can be copied to a client. Token goes to stdout; provenance to stderr.
    Token,
}

#[derive(Subcommand)]
enum GraphCmd {
    /// Extract/refresh symbols and call edges (incremental).
    Build,
    /// Alias of build (always incremental).
    Update,
    /// Is the graph still current? Exits 1 on drift; never writes.
    Check {
        /// Machine-readable report for CI.
        #[arg(long)]
        json: bool,
    },
    /// Who calls this symbol (transitively).
    Callers {
        symbol: String,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Search every project that has a graph, not just this one.
        #[arg(long)]
        all_projects: bool,
    },
    /// What this symbol calls (transitively).
    Calls {
        symbol: String,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Search every project that has a graph, not just this one.
        #[arg(long)]
        all_projects: bool,
    },
    /// Blast radius of changing these files (try `$(git diff --name-only)`).
    Impact {
        #[arg(required = true)]
        files: Vec<String>,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Search every project that has a graph, not just this one.
        #[arg(long)]
        all_projects: bool,
    },
    /// Full record for a symbol: signature, doc, edges, linked memories.
    Node { symbol: String },
    /// Shortest call path between two symbols.
    Path {
        from: String,
        to: String,
        /// Search every project that has a graph, not just this one.
        #[arg(long)]
        all_projects: bool,
    },
    /// Most-called symbols.
    Hubs {
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
    },
    /// Call-graph communities with heuristic names.
    Communities {
        /// Store them as community nodes (member_of edges).
        #[arg(long)]
        persist: bool,
        #[arg(long, default_value_t = 4)]
        min_size: usize,
    },
    /// Interactive HTML visualization of the project graph (memories,
    /// symbols, files, docs and the edges between them).
    Viz {
        /// Output path (default: temp dir).
        #[arg(short, long)]
        out: Option<String>,
        /// Open in the default browser.
        #[arg(long)]
        open: bool,
        /// Node budget: all memories + best-connected code up to this many.
        #[arg(long, default_value_t = 1500)]
        max_nodes: usize,
    },
}

#[derive(Subcommand)]
enum DocsCmd {
    /// Register a folder of markdown docs.
    Add {
        path: String,
        /// Display name (default: folder name).
        #[arg(long)]
        name: Option<String>,
        /// Register cross-project (global) even when inside a project.
        #[arg(short, long)]
        global: bool,
    },
    /// List collections with file/chunk counts.
    List,
    /// Soft-delete a collection and everything indexed under it.
    Remove { name: String },
    /// Attach a context note to a collection or any node.
    Note {
        /// Collection name/path or node reference.
        target: String,
        #[arg(required = true)]
        text: Vec<String>,
    },
}

#[derive(Subcommand)]
enum CodeCmd {
    /// Register a folder of source code (any tree-sitter-supported
    /// language Mimir already parses for the code graph).
    Add {
        path: String,
        /// Display name (default: folder name).
        #[arg(long)]
        name: Option<String>,
        /// Register cross-project (global) even when inside a project.
        #[arg(short, long)]
        global: bool,
    },
}

/// Stack for the worker thread the CLI actually runs on. Generous on
/// purpose: it costs address space, not memory, and the alternative is an
/// abort with no diagnostic.
const WORKER_STACK: usize = 16 * 1024 * 1024;

fn main() {
    // Behave like a normal Unix tool when piped into head/grep: die on
    // SIGPIPE instead of panicking on a failed stdout write.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // Before anything slow. The stdio MCP server uses this to notice that the
    // client that spawned it has died; read any later and a client that exits
    // during startup is already gone, leaving nothing to compare against.
    #[cfg(unix)]
    mcp::record_parent_pid();

    // Windows hands the main thread a 1 MiB stack (Linux/macOS give 8).
    // clap's derived builder for our ~50 subcommands needs more than that
    // in debug builds, so `mimir --version` aborted with a stack overflow
    // before parsing a single argument. Do the work on a thread whose
    // stack size we set ourselves rather than on the one the OS picked.
    let worker = std::thread::Builder::new()
        .stack_size(WORKER_STACK)
        .spawn(cli_main)
        .expect("spawn CLI worker thread");
    // A panic already printed its own message via the default hook; 101 is
    // what rustc's own binaries exit with in that case.
    let code = worker.join().unwrap_or(101);

    // GPU builds: Dawn/onnxruntime process-teardown destructors are known
    // to segfault (ort 2.0-rc). All work is flushed and SQLite-committed
    // by now, so skip libc teardown entirely instead of crashing on exit.
    #[cfg(any(feature = "gpu-webgpu", feature = "gpu-cuda"))]
    {
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: bypasses atexit/dso destructors on purpose; no state
        // depends on them.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(any(feature = "gpu-webgpu", feature = "gpu-cuda")))]
    std::process::exit(code)
}

/// The actual entry point. Runs on the worker thread spawned by `main`.
fn cli_main() -> i32 {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("Error: {err:#}");
            1
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Init {
            no_model,
            hooks,
            auto_recall,
            context_guard,
        } => commands::init(
            no_model,
            hooks || auto_recall || context_guard.is_some(),
            auto_recall,
            context_guard.as_deref(),
        ),
        Command::Rewrite { cmd } => rewrite_cmd::rewrite(cmd),
        Command::RecallInject {
            enrich,
            session,
            prompt,
        } => commands::recall_inject(prompt.join(" "), enrich, session),
        Command::Status => commands::status(cli.json),
        Command::Doctor { check } => commands::doctor(check),
        Command::Remember {
            text,
            mtype,
            tags,
            global,
            force,
            link,
            fires_when,
            anchors,
            expires_in,
            resolves_when,
            confidence,
            valid_from,
            valid_to,
        } => commands::remember(
            cli.json,
            text.join(" "),
            &mtype,
            tags,
            global,
            force,
            link,
            fires_when,
            anchors,
            expires_in,
            resolves_when,
            confidence,
            valid_from,
            valid_to,
        ),
        Command::Recall {
            query,
            kind,
            global,
            all,
            since,
            limit,
            min_score,
            full,
            rerank,
            linked,
            include_superseded,
            as_of,
        } => commands::recall(
            cli.json,
            query.join(" "),
            &kind,
            global,
            all,
            since,
            limit,
            full,
            rerank,
            linked,
            min_score,
            include_superseded,
            as_of
                .map(|s| {
                    mimir_core::format::parse_date(&s).ok_or_else(|| {
                        anyhow::anyhow!("bad --as-of '{s}': expected a date like 2026-03-01")
                    })
                })
                .transpose()?,
        ),
        Command::Get { refs } => commands::get(cli.json, refs),
        Command::List {
            mtype,
            tag,
            global,
            all,
            limit,
        } => commands::list(cli.json, mtype, tag, global, all, limit),
        Command::Forget {
            reference,
            hard,
            reason,
        } => commands::forget(&reference, hard, reason.as_deref()),
        Command::Supersede { old, by, reason } => commands::supersede(&old, &by, reason.as_deref()),
        Command::Anchor {
            reference,
            patterns,
        } => commands::anchor(&reference, patterns),
        Command::Edit {
            reference,
            text,
            title,
            mtype,
            tags,
            pin,
            unpin,
        } => commands::edit(
            cli.json,
            &reference,
            text.join(" "),
            title,
            mtype,
            tags,
            if pin {
                Some(true)
            } else if unpin {
                Some(false)
            } else {
                None
            },
        ),
        Command::Reproject {
            reference,
            project,
            global,
        } => commands::reproject(cli.json, &reference, project, global),
        Command::Link {
            a,
            b,
            rel,
            scan,
            dry_run,
            all_projects,
        } => match (scan, a, b) {
            (true, _, _) => commands::link_scan(dry_run, all_projects),
            (false, Some(a), Some(b)) => commands::link(&a, &b, &rel),
            _ => anyhow::bail!(
                "usage: mimir link <A> <B> [--rel REL]  |  \
                 mimir link --scan [--all-projects] [--dry-run]"
            ),
        },
        Command::Docs { cmd } => match cmd {
            DocsCmd::Add { path, name, global } => commands::docs_add(&path, name, global),
            DocsCmd::List => commands::docs_list(cli.json),
            DocsCmd::Remove { name } => commands::docs_remove(&name),
            DocsCmd::Note { target, text } => commands::docs_note(&target, text.join(" ")),
        },
        Command::Code { cmd } => match cmd {
            CodeCmd::Add { path, name, global } => commands::code_add(&path, name, global),
        },
        Command::Index { name } => commands::index(name),
        Command::Embed { fetch, rerank } => commands::embed(fetch, rerank),
        Command::Mark {
            reference,
            useful,
            noise,
        } => {
            // Marking mutates ranking state — refuse to guess the intent.
            if !useful && !noise {
                anyhow::bail!("pass --useful or --noise");
            }
            commands::mark(&reference, useful)
        }
        Command::Consolidate { dry_run } => commands::consolidate(dry_run),
        Command::Refusals { limit } => commands::refusals(limit),
        Command::Audit { reference, limit } => {
            commands::audit(reference.as_deref(), limit, cli.json)
        }
        Command::History { reference } => commands::history(&reference),
        Command::Review {
            id,
            keep,
            supersede,
            dismiss,
            reason,
            limit,
        } => match id {
            Some(id) => review_cmd::decide(id, keep, supersede, dismiss, reason),
            None => review_cmd::list(cli.json, limit),
        },
        Command::Grounding { stale, limit } => commands::grounding(stale, limit),
        Command::Import { cmd } => match cmd {
            ImportCmd::Openbrain { file } => commands::import_openbrain(&file),
            ImportCmd::ClaudeMemory { dir } => commands::import_claude_memory(&dir),
            ImportCmd::Qmd { file } => commands::import_qmd(file),
        },
        Command::Export => commands::export(),
        Command::Dashboard { out, open } => dashboard::dashboard(out, open),
        Command::Report => report::report(cli.json),
        Command::Savings { oneline } => savings_cmd::savings(cli.json, oneline),
        Command::Tokens { text } => commands::tokens(text),
        Command::Mcp {
            http,
            http_allow_remote,
            http_token,
        } => {
            // Flag wins over MIMIR_HTTP_TOKEN; empty flag or env = unset (an
            // unset shell var in `--http-token "$TOKEN"` must not enable an
            // unsatisfiable gate).
            let token = http_token
                .filter(|t| !t.is_empty())
                .or_else(mcp::http_token_from_env);
            mcp::run(http, http_allow_remote, token)
        }
        Command::Daemon => commands::daemon(),
        Command::Proxy {
            bind,
            dry_run,
            no_cache,
            cache_ttl,
            no_dedup,
            prune,
            max_request_tokens,
        } => proxy_cmd::run(
            bind,
            dry_run,
            no_cache,
            cache_ttl,
            no_dedup,
            prune,
            max_request_tokens,
        ),
        Command::Sync { cmd } => match cmd {
            None => sync::sync(),
            Some(SyncCmd::Push) => sync::push(),
            Some(SyncCmd::Pull) => sync::pull(),
            Some(SyncCmd::Status) => sync::status(),
            Some(SyncCmd::Token) => sync::token(),
        },
        Command::Serve { bind } => sync::serve(bind),
        Command::Graph { cmd } => match cmd {
            GraphCmd::Build | GraphCmd::Update => graph_cmd::build(),
            GraphCmd::Check { json } => graph_cmd::check(json),
            GraphCmd::Callers {
                symbol,
                depth,
                all_projects,
            } => graph_cmd::callers(&symbol, depth, all_projects),
            GraphCmd::Calls {
                symbol,
                depth,
                all_projects,
            } => graph_cmd::calls(&symbol, depth, all_projects),
            GraphCmd::Impact {
                files,
                depth,
                all_projects,
            } => graph_cmd::impact(files, depth, all_projects),
            GraphCmd::Node { symbol } => graph_cmd::node_info(&symbol),
            GraphCmd::Path {
                from,
                to,
                all_projects,
            } => graph_cmd::path(&from, &to, all_projects),
            GraphCmd::Hubs { limit } => graph_cmd::hubs(limit),
            GraphCmd::Communities { persist, min_size } => {
                graph_cmd::communities(persist, min_size)
            }
            GraphCmd::Viz {
                out,
                open,
                max_nodes,
            } => graph_viz::viz(out, open, max_nodes),
        },
        Command::Outline { target } => context_cmd::outline(&target, cli.json),
        Command::Peek { symbol } => context_cmd::peek(&symbol, cli.json),
        Command::Rules { cmd } => match cmd {
            RulesCmd::Set { text } => rules_cmd::set(text),
            RulesCmd::Show => rules_cmd::show(),
            RulesCmd::Clear => rules_cmd::clear(),
        },
        Command::Brief { cmd } => match cmd {
            Some(BriefCmd::Show) => brief_cmd::show(),
            None => brief_cmd::dry_run(),
        },
        Command::Project { cmd } => match cmd {
            ProjectCmd::Init { sync } => project_cmd::init(sync),
        },
        Command::ContextGuard { cmd } => match cmd {
            ContextGuardCmd::Prompt => context_guard_cmd::prompt(),
            ContextGuardCmd::Precompact => context_guard_cmd::precompact(),
            ContextGuardCmd::SessionStart => context_guard_cmd::session_start(),
            ContextGuardCmd::Pretool => context_guard_cmd::pretool(),
        },
        Command::Run { raw, cmd } => run_cmd::run(raw, cmd),
    }
}
