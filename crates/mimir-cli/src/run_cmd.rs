//! `mimir run -- <cmd>` — run a command and print token-lean output.
//!
//! Mimir's replacement for a standalone terminal-noise filter (RTK): execute
//! the command, pass stdout/stderr through the per-program filters, preserve
//! the exit code, and record the tokens saved. `--raw` bypasses filtering for
//! debugging. Streams are kept separate so a tool's stderr stays on stderr.

use std::io::Write;

use anyhow::{bail, Result};
use mimir_core::{savings, tokens, Mimir};
use serde_json::json;

use crate::filters;

pub fn run(raw: bool, cmd: Vec<String>) -> Result<()> {
    if cmd.is_empty() {
        bail!("usage: mimir run [--raw] -- <command> [args...]");
    }
    let program = cmd[0].clone();

    let output = match std::process::Command::new(&program)
        .args(&cmd[1..])
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("mimir run: cannot execute `{program}`: {e}");
            std::process::exit(127);
        }
    };

    let raw_out = String::from_utf8_lossy(&output.stdout).into_owned();
    let raw_err = String::from_utf8_lossy(&output.stderr).into_owned();

    let (out_text, err_text) = if raw {
        (raw_out.clone(), raw_err.clone())
    } else {
        (
            filters::filter_output(&program, &raw_out),
            filters::filter_output(&program, &raw_err),
        )
    };

    let _ = std::io::stdout().write_all(out_text.as_bytes());
    let _ = std::io::stderr().write_all(err_text.as_bytes());
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    if !raw {
        let before = tokens::count(&raw_out) + tokens::count(&raw_err);
        let after = tokens::count(&out_text) + tokens::count(&err_text);
        if before > after {
            // Content commands (coreutils/kubectl) are volume-capped, not
            // line-filtered — record them under a distinct ledger source.
            let source = if filters::is_content(filters::base_program(&program)) {
                savings::source::CAP
            } else {
                savings::source::FILTER
            };
            // Best-effort ledger write; never let bookkeeping fail the command.
            if let Ok(mimir) = Mimir::open() {
                let project_id = std::env::current_dir()
                    .ok()
                    .and_then(|d| mimir.project_for_cwd(&d).ok().flatten())
                    .map(|p| p.id);
                let _ = savings::record(
                    &mimir.conn,
                    project_id,
                    source,
                    before,
                    after,
                    json!({ "cmd": program }),
                );
            }
        }
    }

    std::process::exit(output.status.code().unwrap_or(1));
}
