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
//! Which commands those are is not decided here: `filters::classify` answers
//! for this hook and for `mimir run` alike.

use crate::filters::{base_program, classify, Hook};

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
    let rest: Vec<&str> = parts.collect();
    match classify(base, &rest).hook {
        Hook::Wrap => Some(format!("mimir run -- {trimmed}")),
        Hook::LeaveFollowing | Hook::LeaveQuietSubcommand | Hook::LeaveUnknown => None,
    }
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
        assert!(rewritten("echo hello").is_none()); // not filterable, not content
        assert!(rewritten("").is_none());
    }

    #[test]
    fn wraps_content_commands() {
        for cmd in [
            "cat README.md",
            "grep -rn foo src",
            "rg pattern",
            "find . -name '*.rs'",
            "ls -la",
            "df -h",
            "du -sh .",
            "kubectl get pods",
        ] {
            assert!(rewritten(cmd).is_some(), "should wrap: {cmd}");
        }
    }

    #[test]
    fn does_not_wrap_follow_streams() {
        // `mimir run` waits for exit; a follow would hang forever.
        assert!(rewritten("tail -f /var/log/syslog").is_none());
        assert!(rewritten("tail -F app.log").is_none());
        assert!(rewritten("tail --follow=name x").is_none());
        assert!(rewritten("journalctl -f").is_none());
    }

    #[test]
    fn wraps_non_follow_tail_and_unrelated_dash_f() {
        // a bounded tail is fine to cap …
        assert!(rewritten("tail -n 5000 big.log").is_some());
        // … and `-f` on a non-follow program (grep pattern file) is not a follow.
        assert!(rewritten("grep -f patterns.txt data.txt").is_some());
    }

    #[test]
    fn strips_path_prefix() {
        assert_eq!(
            rewritten("/usr/bin/cargo test").as_deref(),
            Some("mimir run -- /usr/bin/cargo test")
        );
    }
}
