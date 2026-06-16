//! Per-program noise rules, as a declarative registry.
//!
//! Each program maps to a [`Noise`] description (prefixes / substrings /
//! suffixes to drop, plus an optional custom predicate for the few shapes that
//! aren't a simple match). The `is_signal` safety net — anything that looks
//! like an error/warning/failure is always kept — is applied centrally in
//! [`verdict`], so a new program's rules physically *cannot* forget it.

#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Keep,
    Drop,
}

/// A line (after left-trim for prefix/suffix tests) is dropped if it starts
/// with any `prefixes`, ends with any `ends_with`, contains any `contains`, or
/// `custom` returns true.
struct Noise {
    prefixes: &'static [&'static str],
    contains: &'static [&'static str],
    ends_with: &'static [&'static str],
    custom: Option<fn(&str) -> bool>,
}

const NONE: &[&str] = &[];

/// Lines that must never be dropped, whatever a program's rules say.
pub(crate) fn is_signal(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    const NEEDLES: &[&str] = &[
        "error",
        "warning",
        "warn:",
        "fail",
        "panic",
        "fatal",
        "exception",
        "traceback",
        "assert",
        "cannot",
        "denied",
        "refused",
        "unable",
        "missing",
        "not found",
        "conflict",
        "rejected",
        "✗",
        "✘",
        "❌",
    ];
    NEEDLES.iter().any(|n| l.contains(n))
}

/// Rules for `program` (argv[0] base name), or None to fall back to `GENERIC`.
fn rules_for(program: &str) -> Option<&'static Noise> {
    Some(match program {
        "cargo" | "rustc" => &CARGO,
        "git" => &GIT,
        // `node` is intentionally absent — filtering a running app's stdout as
        // package-manager noise would eat its logs.
        "npm" | "pnpm" | "yarn" | "npx" | "bun" => &NODE_PM,
        "pytest" | "py.test" => &PYTEST,
        "go" => &GO,
        "make" | "gmake" => &MAKE,
        "docker" | "docker-compose" | "podman" => &DOCKER,
        "pip" | "pip3" => &PIP,
        "jest" | "vitest" | "eslint" | "tsc" | "next" => &JS_TOOL,
        "terraform" | "tofu" => &TERRAFORM,
        _ => return None,
    })
}

/// True if Mimir has a dedicated handler for `program` (i.e. it's worth
/// auto-wrapping in `mimir run`). The single source of truth the PreToolUse
/// rewrite shares, so adding a handler makes the command rewrite-eligible too.
pub fn is_filterable(program: &str) -> bool {
    rules_for(program).is_some()
}

/// Decide one line. `is_signal` wins first; then the program's rules (or the
/// gentle generic fallback for unknown programs).
pub fn verdict(program: &str, line: &str) -> Verdict {
    if is_signal(line) {
        return Verdict::Keep;
    }
    let rules = rules_for(program).unwrap_or(&GENERIC);
    let t = line.trim_start();
    let drop = rules.prefixes.iter().any(|p| t.starts_with(p))
        || rules.ends_with.iter().any(|s| t.ends_with(s))
        || rules.contains.iter().any(|c| line.contains(c))
        || rules.custom.is_some_and(|f| f(line));
    if drop {
        Verdict::Drop
    } else {
        Verdict::Keep
    }
}

// ── the registry ────────────────────────────────────────────────────────────

static CARGO: Noise = Noise {
    // "Finished" stays — it's the one-line confirmation a build succeeded.
    prefixes: &[
        "Compiling",
        "Checking ",
        "Updating ",
        "Downloading ",
        "Downloaded ",
        "Installing ",
        "Blocking ",
        "Adding ",
        "Locking ",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: None,
};

static GIT: Noise = Noise {
    prefixes: &[
        // clone/fetch/pull progress counters and remote chatter.
        "Counting objects",
        "Compressing objects",
        "Receiving objects",
        "Resolving deltas",
        "remote: Counting",
        "remote: Compressing",
        "remote: Total",
        "remote: Enumerating",
        // `git status` boilerplate: the "(use \"git …\")" hints and branch
        // tracking chatter. Branch line, section headers and file lists stay.
        "(use ",
        "(commit ",
        "Your branch is up to date",
        "Your branch is ahead",
        "Your branch is behind",
        "nothing to commit",
        "no changes added to commit",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: None,
};

static NODE_PM: Noise = Noise {
    prefixes: &[
        // deprecation spam (is_signal doesn't catch "WARN deprecated").
        "npm WARN deprecated",
        "npm notice",
        "added ",
        "removed ",
        "changed ",
        "audited ",
        "found 0 vulnerabilities",
        "packages are looking for funding",
        "run `npm fund`",
        "Progress:",
        "Already up to date",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: Some(is_pnpm_progress),
};

static PYTEST: Noise = Noise {
    prefixes: &[
        "platform ",
        "rootdir:",
        "plugins:",
        "collecting ",
        "collected ",
        "cachedir:",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: Some(is_pytest_progress),
};

static GO: Noise = Noise {
    prefixes: &[
        "go: downloading",
        "go: extracting",
        "go: finding",
        "go: added",
        "go: upgraded",
        "go: downloaded",
    ],
    contains: &["[no test files]"],
    ends_with: NONE,
    custom: None,
};

static MAKE: Noise = Noise {
    prefixes: NONE,
    // Recursive-make directory chatter; compiler output is kept.
    contains: &["Entering directory", "Leaving directory"],
    ends_with: NONE,
    custom: None,
};

static DOCKER: Noise = Noise {
    // BuildKit step/graph progress: "#5 …", " => …", " => => …".
    prefixes: &["#", "=>", "Sending build context"],
    contains: NONE,
    // Classic image-layer status lines: "<sha>: Pull complete".
    ends_with: &[
        "Pull complete",
        "Already exists",
        "Download complete",
        "Pulling fs layer",
        "Verifying Checksum",
        "Waiting",
        "Extracting",
    ],
    custom: None,
};

static PIP: Noise = Noise {
    // Resolution/download/build chatter; "Successfully installed …" is kept.
    prefixes: &[
        "Collecting ",
        "Downloading ",
        "Using cached ",
        "Requirement already satisfied",
        "Preparing metadata",
        "Building wheel",
        "Created wheel",
        "Stored in directory",
        "Getting requirements",
        "Installing build dependencies",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: None,
};

static JS_TOOL: Noise = Noise {
    // jest/vitest/eslint/tsc/next: per-file PASS spam + runner chatter. Errors
    // and the summary ("Tests:", "Test Suites:") survive via is_signal/keep.
    prefixes: &[
        "PASS ",
        "Determining test suites",
        "Ran all test suites",
        "RUNS ",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: None,
};

static TERRAFORM: Noise = Noise {
    prefixes: &["Acquiring state lock", "Releasing state lock"],
    // State-refresh and "still …-ing" progress; the plan/apply diff is kept.
    contains: &[
        "Refreshing state...",
        "Reading...",
        "Read complete after",
        "Still creating...",
        "Still modifying...",
        "Still destroying...",
        "Still reading...",
    ],
    ends_with: NONE,
    custom: None,
};

static GENERIC: Noise = Noise {
    prefixes: &[
        "Progress:",
        "downloading",
        "Downloading",
        "Fetching ",
        "Resolving ",
    ],
    contains: NONE,
    ends_with: NONE,
    custom: None,
};

// ── custom predicates (the shapes a plain match can't express) ───────────────

/// pnpm progress bars like "[ 12/345] reify" — the "[N/M]" shape, not a bare
/// "[" (which would eat app logs like "[INFO] …").
fn is_pnpm_progress(line: &str) -> bool {
    let t = line.trim_start();
    let Some(inner) = t.strip_prefix('[').and_then(|r| r.split(']').next()) else {
        return false;
    };
    matches!(inner.trim().split_once('/'), Some((a, b)) if is_digits(a) && is_digits(b))
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// pytest progress: a bare "....s..  [ 42%]" line, or "tests/foo.py …" followed
/// only by progress markers. The FAILURES section and the "1 failed, …"
/// summary survive via is_signal.
fn is_pytest_progress(line: &str) -> bool {
    let t = line.trim_start();
    if let Some((head, rest)) = t.split_once(char::is_whitespace) {
        if head.ends_with(".py") && !rest.trim().is_empty() && is_progress_glyphs(rest) {
            return true;
        }
    }
    !t.is_empty() && t.contains('.') && is_progress_glyphs(t)
}

fn is_progress_glyphs(s: &str) -> bool {
    s.chars().all(|c| {
        matches!(
            c,
            '.' | 'F' | 'E' | 's' | 'x' | 'X' | 'p' | ' ' | '[' | ']' | '%'
        ) || c.is_ascii_digit()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dropped(program: &str, line: &str) -> bool {
        matches!(verdict(program, line), Verdict::Drop)
    }

    #[test]
    fn signal_overrides_every_ruleset() {
        assert!(!dropped("cargo", "Compiling: error in build script"));
        assert!(!dropped("git", "Counting objects: fatal: bad"));
        assert!(!dropped("npm", "added 5 but error occurred"));
    }

    #[test]
    fn cargo_noise_vs_signal() {
        assert!(dropped("cargo", "   Compiling serde v1.0"));
        assert!(!dropped("cargo", "    Finished dev profile"));
        assert!(!dropped("cargo", "error[E0432]: unresolved"));
    }

    #[test]
    fn pytest_progress_and_preamble() {
        assert!(dropped("pytest", "....s..x  [ 78%]"));
        assert!(dropped("pytest", "tests/test_a.py ........  [ 42%]"));
        assert!(dropped("pytest", "rootdir: /home/x"));
        assert!(dropped("pytest", "collected 248 items"));
        assert!(!dropped("pytest", "FAILED tests/test_x.py::test_y"));
    }

    #[test]
    fn npm_and_pnpm() {
        assert!(dropped("npm", "npm WARN deprecated foo@1: old"));
        assert!(!dropped("npm", "npm WARN config something risky"));
        assert!(dropped("pnpm", "[ 12/345] reify"));
        assert!(!dropped("pnpm", "[INFO] server started"));
    }

    #[test]
    fn git_status_boilerplate_vs_files() {
        assert!(dropped(
            "git",
            "  (use \"git restore --staged <file>...\" to unstage)"
        ));
        assert!(dropped(
            "git",
            "Your branch is up to date with 'origin/main'."
        ));
        assert!(!dropped("git", "\tmodified:   src/main.rs"));
        assert!(!dropped("git", "On branch main"));
    }

    #[test]
    fn expanded_toolchain() {
        assert!(dropped("go", "go: downloading github.com/x/y v1.2.3"));
        assert!(!dropped("go", "ok   pkg/foo   0.5s"));
        assert!(!dropped("go", "--- FAIL: TestX"));

        assert!(dropped("make", "make[1]: Entering directory '/x'"));
        assert!(!dropped("make", "gcc -o foo foo.c"));

        assert!(dropped("docker", "#5 [internal] load build context"));
        assert!(dropped("docker", " => => transferring context: 2kB"));
        assert!(dropped("docker", "a1b2c3: Pull complete"));
        assert!(!dropped("docker", "Successfully tagged myapp:latest"));

        assert!(dropped("pip", "Collecting requests"));
        assert!(dropped("pip", "Requirement already satisfied: idna"));
        assert!(!dropped("pip", "Successfully installed requests-2.31.0"));

        assert!(dropped("jest", "PASS src/a.test.ts"));
        assert!(!dropped("jest", "FAIL src/b.test.ts"));
        assert!(!dropped("jest", "Tests:       1 failed, 5 passed"));

        assert!(dropped(
            "terraform",
            "aws_instance.web: Refreshing state... [id=i-123]"
        ));
        assert!(dropped(
            "terraform",
            "aws_instance.web: Still creating... [10s elapsed]"
        ));
        assert!(!dropped(
            "terraform",
            "  + resource \"aws_instance\" \"web\" {"
        ));
        assert!(!dropped(
            "terraform",
            "Plan: 3 to add, 0 to change, 0 to destroy."
        ));
    }

    #[test]
    fn is_filterable_matches_registry() {
        assert!(is_filterable("cargo"));
        assert!(is_filterable("docker-compose"));
        assert!(is_filterable("terraform"));
        assert!(!is_filterable("node")); // runs user apps — not filtered
        assert!(!is_filterable("ls"));
    }
}
