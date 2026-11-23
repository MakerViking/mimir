//! Per-program noise rules, as a declarative registry.
//!
//! Each program maps to a [`Noise`] description (prefixes / substrings /
//! suffixes to drop, plus an optional custom predicate for the few shapes that
//! aren't a simple match). The `is_signal` safety net — anything that looks
//! like an error/warning/failure is always kept — is applied centrally in
//! [`Filter::verdict`], so a new program's rules physically *cannot* forget it.
//!
//! What a command *is* gets decided in one function, [`classify`]. `mimir
//! run` reads how to filter its output from the answer, the PreToolUse hook
//! (`mimir rewrite`) reads whether to wrap it, and neither keeps a table of
//! its own.

#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Keep,
    Drop,
}

/// A line (after left-trim for prefix/suffix tests) is dropped if it starts
/// with any `prefixes`, ends with any `ends_with`, contains any `contains`, or
/// `custom` returns true.
pub struct Noise {
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

/// High-volume "content" commands whose output *is* the signal (coreutils,
/// search, listings, kubectl). We deliberately do NOT run the per-line noise
/// verdict on these — dropping a matched/listed line would be lossy. `mimir
/// run` instead applies only the non-lossy volume cap (head + tail + every
/// signal line, with the bulky middle elided behind a visible marker).
const CONTENT_CMDS: &[&str] = &[
    "cat", "head", "tail", "ls", "find", "grep", "rg", "egrep", "fgrep", "ps", "df", "du", "tree",
    "kubectl",
];

/// True for a [`CONTENT_CMDS`] command — cap-only handling, never line-dropped.
fn is_content(program: &str) -> bool {
    CONTENT_CMDS.contains(&program)
}

/// git subcommands noisy enough for the hook to wrap unasked (clone/fetch
/// progress + status boilerplate).
const GIT_NOISY_SUBCMDS: &[&str] = &["clone", "pull", "fetch", "push", "status"];

/// How `mimir run` filters a program's output.
pub enum Filter {
    /// Output is the signal (see [`CONTENT_CMDS`]): no line is dropped, only
    /// runaway volume is capped.
    Content,
    /// A tool with noise rules of its own, applied to whatever it prints.
    /// The rules go by program alone: `mimir run -- git log` gets the git
    /// rules although the hook would not have wrapped it (see
    /// [`Hook::LeaveQuietSubcommand`]).
    Noisy(&'static Noise),
    /// A program nothing is known about: the gentle generic rules.
    Generic,
}

/// What the PreToolUse hook (`mimir rewrite`) does with a command.
///
/// `mimir run` filters anything it is handed, by [`Filter`]. The hook wraps
/// commands nobody asked it to, so it is choosier; each `Leave…` names one
/// way the two differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hook {
    /// Wrap it in `mimir run`: its output is noisy or large, and it ends on
    /// its own.
    Wrap,
    /// Leave it: it follows a stream until killed (`tail -f`), and `mimir
    /// run` waits for the child to exit before it prints anything — wrapped,
    /// it would hang the agent forever.
    LeaveFollowing,
    /// Leave it: the program has noise rules, but this subcommand is not one
    /// of its noisy ones (git outside [`GIT_NOISY_SUBCMDS`]).
    LeaveQuietSubcommand,
    /// Leave it: the program has no filter of its own, and the generic rules
    /// are not worth a wrapper nobody asked for.
    LeaveUnknown,
}

/// What Mimir knows about one command.
pub struct Class {
    /// How `mimir run` filters its output.
    pub filter: Filter,
    /// Whether the hook wraps it.
    pub hook: Hook,
}

/// Classify a command: `program` is argv[0]'s base name, `args` the rest.
///
/// The one place that reads the registries above. A program gets a filter
/// from its name alone; whether the hook wraps it also depends on how it is
/// being run.
pub fn classify(program: &str, args: &[&str]) -> Class {
    let filter = if is_content(program) {
        Filter::Content
    } else if let Some(rules) = rules_for(program) {
        Filter::Noisy(rules)
    } else {
        Filter::Generic
    };
    let hook = match filter {
        Filter::Generic => Hook::LeaveUnknown,
        Filter::Content | Filter::Noisy(_) if follows(program, args) => Hook::LeaveFollowing,
        Filter::Noisy(_) if !is_noisy_subcommand(program, args) => Hook::LeaveQuietSubcommand,
        Filter::Content | Filter::Noisy(_) => Hook::Wrap,
    };
    Class { filter, hook }
}

/// True when this invocation streams until something kills it: a log
/// follower, a watcher, a dev server.
///
/// `mimir run` prints once the child has exited, so a wrapped command that
/// never exits shows the agent nothing at all — not on a foreground call,
/// which sits until its timeout, and not on a background one, whose output
/// the agent polls and finds empty for as long as the server lives. Leaving
/// such a command unwrapped costs only the filtering.
///
/// Decided per program, because the same flag means different things:
/// `kubectl logs -f` follows, `kubectl apply -f x.yaml` names a file, and
/// `docker compose -f x.yml logs` names one before a subcommand that ends.
/// So each rule first finds the subcommand ([`subcommand`], which steps over
/// the global options in front of it) and reads only the flags after it.
/// Only programs the hook would otherwise wrap have a rule here (a test
/// holds the table to that), and a rule that is too eager is the safe
/// mistake.
fn follows(program: &str, args: &[&str]) -> bool {
    let has = |flags: &[&str]| args.iter().any(|a| flags.contains(a));
    match program {
        "tail" => args.iter().any(|a| *a == "--retry" || is_follow_flag(a)),
        "kubectl" => {
            has(&["-w", "--watch", "--watch-only"])
                || match subcommand(args, KUBECTL_VALUE_FLAGS) {
                    Some(("logs", rest)) => rest.iter().any(|a| is_follow_flag(a)),
                    Some(("port-forward" | "proxy" | "attach", _)) => true,
                    _ => false,
                }
        }
        "docker" | "podman" => subcommand(args, DOCKER_VALUE_FLAGS)
            .is_some_and(|(sub, rest)| container_follows(sub, rest)),
        "docker-compose" => compose_follows(args),
        "npm" | "pnpm" | "yarn" | "bun" => {
            (program == "bun" && has(&["--watch", "--hot"])) || script_follows(program, args)
        }
        "next" => matches!(subcommand(args, &[]), Some(("dev" | "start", _))),
        // `jest -w` is `--maxWorkers`; only the long forms watch.
        "jest" => args
            .iter()
            .any(|a| is_on(a, "--watch") || is_on(a, "--watchAll")),
        "vitest" => {
            args.iter().any(|a| *a == "-w" || is_on(a, "--watch"))
                || matches!(subcommand(args, &[]), Some(("watch" | "dev", _)))
        }
        "tsc" => args.iter().any(|a| *a == "-w" || is_on(a, "--watch")),
        "cargo" => {
            // `cargo +nightly watch`: the toolchain comes before everything.
            let args = match args.first() {
                Some(first) if first.starts_with('+') => &args[1..],
                _ => args,
            };
            matches!(subcommand(args, CARGO_VALUE_FLAGS), Some(("watch", _)))
        }
        _ => false,
    }
}

/// Global options that take their value as the next argument, per program:
/// what [`subcommand`] has to step over to find the subcommand in
/// `kubectl -n prod logs -f` or `npm --prefix web run dev`. The
/// `--flag=value` spelling needs no entry.
const KUBECTL_VALUE_FLAGS: &[&str] = &[
    "-n",
    "--namespace",
    "--context",
    "--cluster",
    "--user",
    "--kubeconfig",
    "-s",
    "--server",
    "--as",
    "--as-group",
    "--token",
    "--request-timeout",
    "--cache-dir",
    "--certificate-authority",
    "--client-certificate",
    "--client-key",
    "--tls-server-name",
    "-v",
];
const DOCKER_VALUE_FLAGS: &[&str] = &[
    "--context",
    "-c",
    "--config",
    "-H",
    "--host",
    "-l",
    "--log-level",
    "--tlscacert",
    "--tlscert",
    "--tlskey",
    "--connection",
    "--url",
    "--root",
    "--runroot",
];
const COMPOSE_VALUE_FLAGS: &[&str] = &[
    "-f",
    "--file",
    "-p",
    "--project-name",
    "--profile",
    "--env-file",
    "--project-directory",
    "--ansi",
    "--parallel",
    "--progress",
];
const CARGO_VALUE_FLAGS: &[&str] = &["--config", "-Z", "--color", "-C", "--manifest-path"];

/// Value-taking options of the script runners. Per program on purpose:
/// `-w` names a workspace to npm and is a plain switch to pnpm, and treating
/// pnpm's as taking a value would swallow the script name after it.
fn script_runner_value_flags(program: &str) -> &'static [&'static str] {
    match program {
        "npm" => &[
            "--prefix",
            "-C",
            "-w",
            "--workspace",
            "--userconfig",
            "--registry",
            "--cache",
        ],
        "pnpm" => &["--filter", "-F", "--dir", "-C", "--workspace-dir"],
        "yarn" => &["--cwd"],
        "bun" => &["--cwd", "--filter", "-F", "--config", "-c", "--env-file"],
        _ => &[],
    }
}

/// The subcommand of an invocation and the arguments after it: the first
/// argument that is neither an option nor the value of one of `value_flags`.
fn subcommand<'a>(args: &'a [&'a str], value_flags: &[&str]) -> Option<(&'a str, &'a [&'a str])> {
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if value_flags.contains(&arg) {
            i += 2;
        } else if arg.starts_with('-') {
            i += 1;
        } else {
            return Some((arg, &args[i + 1..]));
        }
    }
    None
}

/// `--flag` or `--flag=<anything but false>`.
fn is_on(arg: &str, flag: &str) -> bool {
    match arg.strip_prefix(flag) {
        Some("") => true,
        Some(value) => value.starts_with('=') && value != "=false",
        None => false,
    }
}

/// A "keep following" flag: `--follow` in any spelling, or a short-flag
/// bundle containing `f` or `F` (`-f`, `-F`, `-ft`, `-fn`).
fn is_follow_flag(arg: &str) -> bool {
    if arg.starts_with("--") {
        return is_on(arg, "--follow");
    }
    arg.starts_with('-') && arg.contains(['f', 'F'])
}

/// `docker <sub> …` / `podman <sub> …`, global options already stepped over.
fn container_follows(sub: &str, rest: &[&str]) -> bool {
    match sub {
        "compose" => compose_follows(rest),
        "logs" => rest.iter().any(|a| is_follow_flag(a)),
        "stats" => !rest.contains(&"--no-stream"),
        "events" | "attach" => true,
        // `docker container logs -f`, `docker service logs -f`.
        "container" | "service" => {
            subcommand(rest, &[]).is_some_and(|(sub, rest)| container_follows(sub, rest))
        }
        _ => false,
    }
}

/// The arguments of `docker compose` (or of `docker-compose`). `-f` before
/// the subcommand names a compose file; only after `logs` does it follow.
fn compose_follows(args: &[&str]) -> bool {
    match subcommand(args, COMPOSE_VALUE_FLAGS) {
        Some(("up", rest)) => !rest
            .iter()
            .any(|a| matches!(*a, "-d" | "--detach" | "--wait")),
        Some(("logs", rest)) => rest.iter().any(|a| is_follow_flag(a)),
        Some(("watch" | "events" | "attach" | "stats", _)) => true,
        _ => false,
    }
}

/// `npm run dev`, `pnpm --filter web dev`, `yarn workspace app start`: the
/// script is the word after `run`, or the subcommand itself.
fn script_follows(program: &str, args: &[&str]) -> bool {
    const SERVES: &[&str] = &["dev", "start", "serve", "watch", "preview", "storybook"];
    let value_flags = script_runner_value_flags(program);
    let script = match subcommand(args, value_flags) {
        Some(("run" | "run-script", rest)) => subcommand(rest, value_flags).map(|(s, _)| s),
        // `yarn workspace <name> <script>`.
        Some(("workspace", rest)) => {
            return subcommand(rest, value_flags)
                .is_some_and(|(_name, rest)| script_follows(program, rest));
        }
        // `yarn workspaces foreach … run dev`: any of them, anywhere.
        Some(("workspaces", rest)) => return rest.iter().any(|a| SERVES.contains(a)),
        Some((sub, _)) => Some(sub),
        None => None,
    };
    script.is_some_and(|s| SERVES.contains(&s))
}

/// git is wrapped for its noisy subcommands only; every other program with
/// rules is noisy whatever it is asked to do.
fn is_noisy_subcommand(program: &str, args: &[&str]) -> bool {
    program != "git"
        || args
            .first()
            .is_some_and(|sub| GIT_NOISY_SUBCMDS.contains(sub))
}

/// True if `program` has a filter of its own: a dedicated noise handler, or
/// the content cap.
#[cfg(test)]
pub fn is_filterable(program: &str) -> bool {
    !matches!(classify(program, &[]).filter, Filter::Generic)
}

impl Filter {
    /// Decide one line. Content keeps every line. Otherwise `is_signal` wins
    /// first; then the program's rules (or the gentle generic fallback for
    /// unknown programs).
    pub fn verdict(&self, line: &str) -> Verdict {
        let rules = match self {
            Filter::Content => return Verdict::Keep,
            Filter::Noisy(rules) => rules,
            Filter::Generic => &GENERIC,
        };
        if is_signal(line) {
            return Verdict::Keep;
        }
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
}

/// [`Filter::verdict`] for `program`'s filter.
#[cfg(test)]
pub fn verdict(program: &str, line: &str) -> Verdict {
    classify(program, &[]).filter.verdict(line)
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
                                         // content commands are filterable (cap-only), but never noise-handled
        assert!(is_filterable("ls"));
        assert!(is_content("ls"));
        assert!(rules_for("ls").is_none()); // cap-only: no per-line noise rules
        assert!(!is_content("cargo")); // noise-handled, not a content command
    }

    /// The filter's variant, without its rules.
    fn filter_name(class: &Class) -> &'static str {
        match class.filter {
            Filter::Content => "content",
            Filter::Noisy(_) => "noisy",
            Filter::Generic => "generic",
        }
    }

    #[test]
    fn classify_says_how_run_filters_and_whether_the_hook_wraps() {
        let cases: &[(&str, &[&str], &str, Hook)] = &[
            ("cargo", &["build"], "noisy", Hook::Wrap),
            ("cat", &["README.md"], "content", Hook::Wrap),
            ("tail", &["-n", "50", "app.log"], "content", Hook::Wrap),
            ("tail", &["-f", "app.log"], "content", Hook::LeaveFollowing),
            ("tail", &["-Fq", "app.log"], "content", Hook::LeaveFollowing),
            (
                "tail",
                &["--follow=name", "x"],
                "content",
                Hook::LeaveFollowing,
            ),
            // `-f` only means "follow" on a program that can.
            ("grep", &["-f", "patterns.txt", "x"], "content", Hook::Wrap),
            ("git", &["status"], "noisy", Hook::Wrap),
            // Where the two sides differ: `run` has rules for all of git, the
            // hook wraps only its noisy subcommands.
            ("git", &["log"], "noisy", Hook::LeaveQuietSubcommand),
            ("git", &[], "noisy", Hook::LeaveQuietSubcommand),
            (
                "git",
                &["-C", "x", "status"],
                "noisy",
                Hook::LeaveQuietSubcommand,
            ),
            ("echo", &["hi"], "generic", Hook::LeaveUnknown),
            ("node", &["server.js"], "generic", Hook::LeaveUnknown),
            // Not a program Mimir filters, in follow mode or out of it.
            ("journalctl", &["-f"], "generic", Hook::LeaveUnknown),
            (
                "journalctl",
                &["-u", "nginx"],
                "generic",
                Hook::LeaveUnknown,
            ),
        ];
        for (program, args, filter, hook) in cases {
            let class = classify(program, args);
            assert_eq!(filter_name(&class), *filter, "{program} {args:?}");
            assert_eq!(class.hook, *hook, "{program} {args:?}");
        }
    }

    /// A command that streams until it is killed is left alone, and the
    /// same program run to completion is still wrapped. Every row is a
    /// program the hook wraps: a follow rule for one it never wraps would
    /// decide nothing.
    #[test]
    fn a_command_that_never_exits_is_not_wrapped() {
        let streams: &[(&str, &[&str])] = &[
            ("tail", &["-f", "app.log"]),
            ("tail", &["--follow=name", "app.log"]),
            ("kubectl", &["logs", "-f", "deploy/api"]),
            ("kubectl", &["get", "pods", "-w"]),
            ("kubectl", &["port-forward", "svc/api", "8080:80"]),
            ("docker", &["logs", "--follow", "api"]),
            ("docker", &["compose", "up"]),
            ("docker", &["compose", "logs", "-f"]),
            ("docker-compose", &["up"]),
            ("podman", &["stats"]),
            ("npm", &["run", "dev"]),
            ("pnpm", &["dev"]),
            ("yarn", &["start"]),
            ("bun", &["run", "--hot", "serve"]),
            ("next", &["dev"]),
            ("vitest", &["--watch"]),
            ("tsc", &["-w"]),
            ("cargo", &["watch", "-x", "test"]),
            // The subcommand behind a global option and its value.
            ("kubectl", &["-n", "prod", "logs", "-f", "deploy/api"]),
            ("kubectl", &["--context=staging", "logs", "--follow", "api"]),
            ("docker", &["--context", "remote", "stats"]),
            ("docker", &["logs", "-ft", "api"]),
            ("docker", &["container", "logs", "-f", "api"]),
            ("docker", &["compose", "-f", "dev.yml", "logs", "-f"]),
            ("docker", &["compose", "-f", "dev.yml", "up"]),
            ("docker-compose", &["-f", "dev.yml", "logs", "--follow"]),
            ("npm", &["--prefix", "web", "run", "dev"]),
            ("npm", &["run", "-w", "web", "dev"]),
            ("pnpm", &["--filter", "web", "dev"]),
            ("pnpm", &["-w", "dev"]),
            ("yarn", &["workspace", "app", "dev"]),
            ("yarn", &["--cwd", "web", "start"]),
            ("bun", &["--watch", "test"]),
            ("jest", &["--watch"]),
            ("jest", &["--watchAll"]),
            ("vitest", &["watch"]),
            ("tsc", &["-b", "--watch"]),
            ("cargo", &["+nightly", "watch"]),
        ];
        for (program, args) in streams {
            assert_eq!(
                classify(program, args).hook,
                Hook::LeaveFollowing,
                "{program} {args:?}"
            );
            assert_eq!(classify(program, &[]).hook, Hook::Wrap, "{program}");
        }
        let ends: &[(&str, &[&str])] = &[
            ("tail", &["-n", "50", "app.log"]),
            ("kubectl", &["apply", "-f", "deploy.yaml"]),
            ("kubectl", &["logs", "deploy/api"]),
            ("docker", &["build", "-f", "Dockerfile", "."]),
            ("docker", &["compose", "up", "-d"]),
            ("docker", &["logs", "api"]),
            ("npm", &["run", "build"]),
            ("npm", &["install"]),
            ("pnpm", &["test"]),
            ("next", &["build"]),
            ("vitest", &["run"]),
            ("tsc", &["--noEmit"]),
            ("cargo", &["test"]),
            // A flag that only looks like "follow" or "watch".
            ("docker", &["compose", "-f", "dev.yml", "logs"]),
            ("docker", &["compose", "-f", "dev.yml", "ps"]),
            ("docker-compose", &["-f", "dev.yml", "up", "-d"]),
            ("docker", &["stats", "--no-stream"]),
            ("docker", &["--context", "logs", "ps"]),
            ("kubectl", &["-n", "logs", "get", "pods"]),
            ("jest", &["-w", "4"]),
            ("jest", &["--watchAll=false"]),
            ("npm", &["--prefix", "dev", "run", "build"]),
            ("pnpm", &["--filter", "start", "build"]),
            ("yarn", &["workspace", "dev", "build"]),
            ("cargo", &["+nightly", "test"]),
        ];
        for (program, args) in ends {
            assert_eq!(
                classify(program, args).hook,
                Hook::Wrap,
                "{program} {args:?}"
            );
        }
    }

    #[test]
    fn a_content_filter_keeps_every_line() {
        // Lines every other filter would drop.
        for line in [
            "   Compiling serde v1.0",
            "Downloading bar",
            "Progress: 3/9",
        ] {
            assert_eq!(Filter::Content.verdict(line), Verdict::Keep, "{line}");
            assert!(
                dropped("whatever", line) || dropped("cargo", line),
                "{line}"
            );
        }
    }
}
