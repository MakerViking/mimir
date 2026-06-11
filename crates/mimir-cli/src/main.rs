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
    }
}
