//! Command-output filters — Mimir's take on "drop the terminal noise".
//!
//! These are our own line-based rules (not vendored from any tool). The
//! guiding invariant: **never hide a failure.** We drop progress spinners,
//! "Compiling …" chatter, download bars and duplicate blank lines, but any
//! line that looks like an error, warning, test result or panic is always
//! kept. When in doubt, keep the line — a wrong drop is far costlier than a
//! few wasted tokens.

mod rules;

pub(crate) use rules::is_filterable;
use rules::{is_signal, verdict, Verdict};

/// Above this many kept lines, the generic volume cap kicks in.
const VOLUME_CAP: usize = 400;
/// Lines kept verbatim at the head and tail when the cap triggers.
const VOLUME_EDGE: usize = 60;

/// Filter `raw` output from `program` (the command's argv[0]): strip ANSI, drop
/// the lines the program's [`rules`] mark as noise, collapse blank runs, and —
/// if the result is still huge — keep only head + tail + signal lines.
pub fn filter_output(program: &str, raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    let cleaned = strip_ansi(raw);
    let base = base_program(program);

    // Pass 1: drop per-program noise + collapse blank runs.
    let mut kept: Vec<&str> = Vec::new();
    let mut prev_blank = false;
    for line in cleaned.lines() {
        if matches!(verdict(base, line), Verdict::Drop) {
            continue;
        }
        let blank = line.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        kept.push(line);
        prev_blank = blank;
    }

    if kept.len() <= VOLUME_CAP {
        return join_lines(&kept);
    }
    // Pass 2 (volume cap): any command can still dump thousands of lines. Keep
    // the head, the tail, and every signal line in between; elide the rest with
    // a visible count so nothing important vanishes silently.
    let n = kept.len();
    let mut out = String::new();
    let mut elided = 0usize;
    for (i, line) in kept.iter().enumerate() {
        if i < VOLUME_EDGE || i >= n - VOLUME_EDGE || is_signal(line) {
            if elided > 0 {
                out.push_str(&format!("[… {elided} lines elided by mimir …]\n"));
                elided = 0;
            }
            out.push_str(line);
            out.push('\n');
        } else {
            elided += 1;
        }
    }
    if elided > 0 {
        out.push_str(&format!("[… {elided} lines elided by mimir …]\n"));
    }
    out
}

fn join_lines(lines: &[&str]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// argv[0] without a directory or extension: `/usr/bin/cargo` → `cargo`.
pub(crate) fn base_program(program: &str) -> &str {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    name.strip_suffix(".exe").unwrap_or(name)
}

/// Remove ANSI/VT escape sequences (CSI, OSC) and lone carriage returns from
/// progress redraws. Conservative: only well-formed escapes.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.next() {
                // CSI: ESC [ … final-byte (0x40–0x7E)
                Some('[') => {
                    for d in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … BEL or ST
                Some(']') => {
                    while let Some(&d) = chars.peek() {
                        if d == '\u{7}' {
                            chars.next();
                            break;
                        }
                        if d == '\u{1b}' {
                            chars.next();
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                        chars.next();
                    }
                }
                // Other two-char escapes: drop the next byte.
                Some(_) | None => {}
            },
            // A carriage return not followed by newline is a progress redraw;
            // keep only the last segment of such a line by clearing what we
            // buffered for it.
            '\r' => {
                if chars.peek() != Some(&'\n') {
                    // Erase back to the previous newline.
                    if let Some(nl) = out.rfind('\n') {
                        out.truncate(nl + 1);
                    } else {
                        out.clear();
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_and_progress() {
        let raw = "\u{1b}[32m   Compiling foo\u{1b}[0m\nplain line\n";
        let out = strip_ansi(raw);
        assert!(out.contains("Compiling foo"));
        assert!(!out.contains('\u{1b}'));
    }

    #[test]
    fn carriage_return_progress_keeps_last() {
        let raw = "downloading 10%\rdownloading 50%\rdownloading 100%\ndone\n";
        let out = strip_ansi(raw);
        assert!(out.contains("downloading 100%"), "{out}");
        assert!(!out.contains("10%"), "{out}");
        assert!(out.contains("done"));
    }

    #[test]
    fn cargo_drops_compiling_keeps_errors() {
        let raw = "   Compiling serde v1.0\n   Compiling foo v0.1\nerror[E0432]: unresolved import\n    Finished dev\n";
        let out = filter_output("cargo", raw);
        assert!(!out.contains("Compiling serde"), "{out}");
        assert!(out.contains("error[E0432]"), "{out}");
    }

    #[test]
    fn cargo_keeps_warnings_and_test_results() {
        let raw =
            "   Compiling x\nwarning: unused variable `y`\ntest result: ok. 12 passed; 0 failed\n";
        let out = filter_output("cargo", raw);
        assert!(out.contains("warning: unused"), "{out}");
        assert!(out.contains("test result"), "{out}");
        assert!(!out.contains("Compiling x"), "{out}");
    }

    #[test]
    fn unknown_program_uses_generic_but_keeps_errors() {
        let raw = "info: starting\n\n\n\nError: boom\n";
        let out = filter_output("whatever", raw);
        assert!(out.contains("Error: boom"));
        // collapsed blank run
        assert!(!out.contains("\n\n\n"));
    }

    #[test]
    fn volume_cap_keeps_edges_and_signals_elides_middle() {
        let mut raw = String::new();
        for i in 0..1000 {
            raw.push_str(&format!("line {i}\n"));
        }
        // a signal buried in the middle must survive
        raw.push_str("error: buried in the middle\n");
        for i in 1000..1100 {
            raw.push_str(&format!("line {i}\n"));
        }
        let out = filter_output("whatever", &raw);
        assert!(out.contains("line 0"), "head kept");
        assert!(out.contains("line 1099"), "tail kept");
        assert!(out.contains("error: buried in the middle"), "signal kept");
        assert!(out.contains("elided by mimir"), "middle elided with marker");
        assert!(out.lines().count() < 300, "much smaller than 1100 lines");
    }
}
