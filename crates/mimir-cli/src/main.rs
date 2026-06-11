mod commands;

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
    Init,
    /// Store overview: counts by kind, scope, database size.
    Status,
    /// Health checks: database integrity, FTS availability, model presence.
    Doctor,
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
        /// Print full bodies instead of one-line summaries.
        #[arg(long)]
        full: bool,
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
    },
    /// Create an edge between two nodes.
    Link {
        a: String,
        b: String,
        /// links|about|relates|mentions|describes|supersedes…
        #[arg(long, default_value = "relates")]
        rel: String,
    },
    /// Manage docs collections.
    Docs {
        #[command(subcommand)]
        cmd: DocsCmd,
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Init => commands::init(),
        Command::Status => commands::status(cli.json),
        Command::Doctor => commands::doctor(),
        Command::Remember {
            text,
            mtype,
            tags,
            global,
            force,
            link,
        } => commands::remember(cli.json, text.join(" "), &mtype, tags, global, force, link),
        Command::Recall {
            query,
            kind,
            global,
            all,
            since,
            limit,
            full,
        } => commands::recall(
            cli.json,
            query.join(" "),
            &kind,
            global,
            all,
            since,
            limit,
            full,
        ),
        Command::Get { refs } => commands::get(cli.json, refs),
        Command::List {
            mtype,
            tag,
            global,
            all,
            limit,
        } => commands::list(cli.json, mtype, tag, global, all, limit),
        Command::Forget { reference, hard } => commands::forget(&reference, hard),
        Command::Edit {
            reference,
            text,
            title,
            mtype,
            tags,
        } => commands::edit(cli.json, &reference, text.join(" "), title, mtype, tags),
        Command::Link { a, b, rel } => commands::link(&a, &b, &rel),
        Command::Docs { cmd } => match cmd {
            DocsCmd::Add { path, name, global } => commands::docs_add(&path, name, global),
            DocsCmd::List => commands::docs_list(cli.json),
            DocsCmd::Remove { name } => commands::docs_remove(&name),
            DocsCmd::Note { target, text } => commands::docs_note(&target, text.join(" ")),
        },
        Command::Index { name } => commands::index(name),
        Command::Embed { fetch } => commands::embed(fetch),
    }
}
