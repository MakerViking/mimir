//! `mimir rewrite "<cmd>"` — single source of truth for the PreToolUse hook.
//!
//! Given a shell command, decide whether to wrap it in `mimir run` so its
//! output gets filtered. Mirrors RTK's exit-code protocol so the installed
//! hook stays a thin delegate:
//!   exit 0 + stdout  → rewritten command (auto-allow)
//!   exit 1           → no rewrite (pass through unchanged)
//!
//! Conservative by design: only known-noisy commands are rewritten, and any
//! command with shell operators (pipes/redirs/&&/;/subshells) is left alone —
//! rewrapping those is risky and the filters wouldn't see the real output.

/// git subcommands worth filtering (clone/fetch progress + status boilerplate).
const GIT_NOISY_SUBCMDS: &[&str] = &["clone", "pull", "fetch", "push", "status"];

use crate::filters::{base_program, is_filterable};

fn has_shell_operators(cmd: &str) -> bool {
    cmd.contains(|c| {
        matches!(
            c,
            '|' | '&' | ';' | '>' | '<' | '`' | '$' | '(' | ')' | '\n'
        )
    })
}

/// Returns the rewritten command if `cmd` should be wrapped, else None.
pub fn rewritten(cmd: &str) -> Option<String> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() || has_shell_operators(trimmed) {
        return None;
    }
    let mut parts = trimmed.split_whitespace();
    let base = base_program(parts.next()?);
    if base == "mimir" {
        return None; // already ours
    }
    // git wraps only its noisy subcommands; everything else with a dedicated
    // filter is wrapped wholesale (single source of truth: `is_filterable`).
    let eligible = if base == "git" {
        parts
            .next()
            .is_some_and(|sub| GIT_NOISY_SUBCMDS.contains(&sub))
    } else {
        is_filterable(base)
    };
    eligible.then(|| format!("mimir run -- {trimmed}"))
}

/// CLI entry: prints the rewrite and exits 0, or exits 1 for pass-through.
pub fn rewrite(cmd_parts: Vec<String>) -> ! {
    let cmd = cmd_parts.join(" ");
    match rewritten(&cmd) {
        Some(new) => {
            println!("{new}");
            std::process::exit(0)
        }
        None => std::process::exit(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_known_commands() {
        assert_eq!(
            rewritten("cargo build").as_deref(),
            Some("mimir run -- cargo build")
        );
        assert_eq!(
            rewritten("npm install").as_deref(),
            Some("mimir run -- npm install")
        );
    }

    #[test]
    fn git_only_noisy_subcommands() {
        assert!(rewritten("git clone https://x").is_some());
        assert!(rewritten("git status").is_some());
        assert!(rewritten("git log").is_none());
        assert!(rewritten("git diff").is_none());
    }

    #[test]
    fn wraps_the_expanded_toolchain() {
        for cmd in [
            "go build ./...",
            "make",
            "docker build .",
            "pip install requests",
            "jest",
            "tsc --noEmit",
            "eslint .",
        ] {
            assert!(rewritten(cmd).is_some(), "should wrap: {cmd}");
        }
    }

    #[test]
    fn skips_compound_and_already_wrapped() {
        assert!(rewritten("cargo build | grep error").is_none());
        assert!(rewritten("cargo build && cargo test").is_none());
        assert!(rewritten("mimir run -- cargo build").is_none());
        assert!(rewritten("ls -la").is_none());
        assert!(rewritten("").is_none());
    }

    #[test]
    fn strips_path_prefix() {
        assert_eq!(
            rewritten("/usr/bin/cargo test").as_deref(),
            Some("mimir run -- /usr/bin/cargo test")
        );
    }
}
