//! Each state the store can be in has one definition
//! (`docs/design/state-machines.md`). This test keeps it that way.
//!
//! It reads the production source of every crate and fails when code outside
//! a state's owners spells that state itself: reads a `meta` flag by its key,
//! compares a mode by its string, writes a lifecycle column directly. Every
//! bug the "explicit state machines" change fixed began as one such second
//! spelling, written by someone who did not know the first existed.
//!
//! It is a text search, so it is a tripwire and not a proof. When it fires:
//!
//! - use the function its message names, or
//! - if the new site really is part of the definition (a new writer of the
//!   flag, say), add the file to that rule's owners, with the reason.
//!
//! Do not rephrase the code until the search no longer matches it.

use std::path::{Path, PathBuf};

/// One state, how code spells it when it derives it by hand, and the only
/// files that may.
struct Rule {
    state: &'static str,
    /// Substrings; any of them on a line of production code is the spelling.
    spelled: &'static [&'static str],
    /// Paths relative to `crates/`, each with why it is allowed.
    owners: &'static [(&'static str, &'static str)],
    /// What everyone else calls instead.
    ask: &'static str,
}

const RULES: &[Rule] = &[
    Rule {
        state: "superseded (the column that retires a memory)",
        spelled: &["set_superseded(", "superseded_by = ?"],
        owners: &[
            (
                "mimir-core/src/store.rs",
                "defines supersede and set_superseded",
            ),
            (
                "mimir-core/src/replicate.rs",
                "merge_supersession: a retirement arriving over sync",
            ),
        ],
        ask: "store::supersede (column, edge and ledger row together, guarded)",
    },
    Rule {
        state: "archived (the decay pass's flag)",
        spelled: &["$.archived", "\"archived\""],
        owners: &[
            ("mimir-core/src/lifecycle.rs", "the definition"),
            ("mimir-core/src/consolidate.rs", "the decay pass sets it"),
            (
                "mimir-core/src/store.rs",
                "forgetting an archived memory clears it",
            ),
            (
                "mimir-core/src/replicate.rs",
                "archival is taken off the wire",
            ),
        ],
        ask: "Lifecycle::of, lifecycle::ARCHIVED_SQL, lifecycle::counts",
    },
    Rule {
        state: "expired",
        spelled: &["$.expires_at", "get(\"expires_at\")"],
        owners: &[
            ("mimir-core/src/lifecycle.rs", "the SQL definition"),
            ("mimir-core/src/model.rs", "Node::expires_at, its Rust twin"),
            ("mimir-core/src/store.rs", "set_expiry writes it"),
        ],
        ask: "Lifecycle::of, Node::expires_at, lifecycle::expired_sql",
    },
    Rule {
        state: "forgotten or archived (the tombstone column)",
        spelled: &["deleted_at = ?"],
        owners: &[
            ("mimir-core/src/store.rs", "forget"),
            ("mimir-core/src/consolidate.rs", "the decay pass archives"),
            (
                "mimir-core/src/index/mod.rs",
                "a docs file or chunk that left its collection",
            ),
            (
                "mimir-graph/src/store_graph.rs",
                "a symbol or file that left the tree",
            ),
            (
                "mimir-graph/src/prune.rs",
                "what a prune keeps because someone linked to it",
            ),
        ],
        ask: "store::forget for a memory",
    },
    Rule {
        state: "session handoff: restored or pending, and whose checkout",
        spelled: &[
            "$.restored_at",
            "\"restored_at\"",
            "\"session-handoff\"",
            "$.handoff_checkout",
            "\"handoff_checkout\"",
        ],
        owners: &[("mimir-core/src/context_guard.rs", "the definition")],
        ask: "context_guard::{HANDOFF_TAG, restored_at, handoff_checkout, handoff_is_for, \
              pending_handoff_in, mark_handoff_restored}",
    },
    Rule {
        state: "when a memory's text was decided",
        spelled: &["$.text_clock", "\"text_clock\""],
        owners: &[("mimir-core/src/text_clock.rs", "the definition")],
        ask: "text_clock::{read, keep, KEY}",
    },
    Rule {
        state: "a row was written (the timestamp sync orders by)",
        spelled: &["updated_at = ?"],
        owners: &[
            ("mimir-core/src/store.rs", "the store's own writers"),
            ("mimir-core/src/memory.rs", "edit"),
            ("mimir-core/src/anchors.rs", "set_anchors"),
            ("mimir-core/src/context_guard.rs", "the handoff mark"),
            (
                "mimir-core/src/index/mod.rs",
                "docs files and chunks, never memories",
            ),
            (
                "mimir-graph/src/store_graph.rs",
                "code files and symbols, never memories",
            ),
        ],
        ask: "an existing writer; a new one that can touch a memory without \
              changing its text calls text_clock::keep first",
    },
    Rule {
        state: "a project's sync identity",
        spelled: &["$.portable_key", "$.shadow"],
        owners: &[
            (
                "mimir-core/src/store.rs",
                "find_project_by_key, set_project_identity, shadows",
            ),
            (
                "mimir-core/src/replicate.rs",
                "syncable_sql and the wire record",
            ),
            ("mimir-core/src/db/migrations.rs", "the index on the key"),
        ],
        ask: "store::find_project_by_key, replicate::project_is_syncable",
    },
    Rule {
        state: "which project a linked worktree files its memories under",
        // Not the bare word: `status --json` has a field of that name.
        spelled: &["$.worktree_of", "get(\"worktree_of\")"],
        owners: &[(
            "mimir-core/src/store.rs",
            "WORKTREE_OF and its two functions",
        )],
        ask: "store::{WORKTREE_OF, set_worktree_home, worktree_home}",
    },
    Rule {
        state: "has a code graph, and by which rules",
        spelled: &["$.graph_built_at", "$.graph_rules"],
        owners: &[("mimir-graph/src/state.rs", "the definition")],
        ask: "mimir_graph::{has_graph, graph_built_at, rules_note, wants_full_rebuild}",
    },
    Rule {
        state: "a config mode, compared as a string",
        spelled: &[
            "== \"server\"",
            "== \"file\"",
            "== \"handoff\"",
            "== \"pause\"",
            "== \"warm\"",
        ],
        // Nobody: config.rs turns each string into its enum with a `match`.
        owners: &[],
        ask: "the setting's enum (config.rs), read through its accessor",
    },
    Rule {
        state: "who owns prompt caching",
        spelled: &["cache_control"],
        owners: &[("mimir-proxy/src/optimize.rs", "the definition")],
        ask: "optimize::has_cache_control, Applied::we_cached",
    },
];

/// The production lines of a source file, numbered from 1: comment lines
/// left out, and every top-level item marked `#[cfg(test)]` left out with
/// its body, wherever in the file it stands (a test module at the end, a
/// test-only helper in the middle).
///
/// An item's end is taken to be the first `}` in column 0, or the end of its
/// first line when that line ends the item (`use …;`). That is how rustfmt
/// lays out every top-level item in this workspace.
fn production_lines(source: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut lines = source.lines().enumerate().map(|(i, line)| (i + 1, line));
    while let Some((number, line)) = lines.next() {
        if line.starts_with("#[cfg(test)]") || line.starts_with("#[cfg(all(test") {
            let mut first_of_item = true;
            for (_, skipped) in lines.by_ref() {
                if skipped.starts_with("#[") {
                    continue;
                }
                if skipped == "}" || (first_of_item && skipped.trim_end().ends_with(';')) {
                    break;
                }
                first_of_item = false;
            }
            continue;
        }
        if !line.trim_start().starts_with("//") {
            out.push((number, line));
        }
    }
    out
}

/// `(file, line number, rule index)` for every spelling outside its owners.
fn violations(files: &[(String, String)]) -> Vec<(String, usize, usize)> {
    let mut found = Vec::new();
    for (path, source) in files {
        for (number, line) in production_lines(source) {
            for (index, rule) in RULES.iter().enumerate() {
                let owned = rule.owners.iter().any(|(owner, _)| owner == path);
                if !owned && rule.spelled.iter().any(|s| line.contains(s)) {
                    found.push((path.clone(), number, index));
                }
            }
        }
    }
    found
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Every production source file of the workspace as `(path under crates/,
/// text)`, or `None` when this crate is being built on its own (from a
/// published package there are no sibling crates to read).
fn workspace_sources() -> Option<Vec<(String, String)>> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .to_path_buf();
    if !crates.join("mimir-core/src/lifecycle.rs").is_file() {
        return None;
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&crates).ok()?.flatten() {
        rust_files(&entry.path().join("src"), &mut paths);
    }
    let mut files = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(&crates)
            .ok()?
            .to_string_lossy()
            .replace('\\', "/");
        // Evaluation fixtures and test-only modules build stores by hand on
        // purpose.
        if relative.contains("/eval/") || relative.ends_with("cli_tests.rs") {
            continue;
        }
        files.push((relative, std::fs::read_to_string(&path).ok()?));
    }
    files.sort();
    Some(files)
}

#[test]
fn each_state_is_spelled_only_by_its_owners() {
    let Some(files) = workspace_sources() else {
        eprintln!("skipped: the workspace's other crates are not here to read");
        return;
    };
    let found = violations(&files);
    let report: Vec<String> = found
        .iter()
        .map(|(path, line, index)| {
            let rule = &RULES[*index];
            format!(
                "crates/{path}:{line} derives \"{}\" by hand.\n    Ask: {}\n    Owners: {}",
                rule.state,
                rule.ask,
                rule.owners
                    .iter()
                    .map(|(owner, why)| format!("{owner} ({why})"))
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })
        .collect();
    assert!(
        report.is_empty(),
        "{} place(s) spell a state that already has one definition \
         (see docs/design/state-machines.md):\n\n{}\n",
        report.len(),
        report.join("\n\n")
    );
}

/// A rule nobody can break is not a rule: every owner named above exists and
/// still spells its state. A file that was renamed, or a definition that
/// moved, has to take its rule along.
#[test]
fn every_rule_still_points_at_a_definition() {
    let Some(files) = workspace_sources() else {
        return;
    };
    for rule in RULES {
        for (owner, why) in rule.owners {
            let source = files
                .iter()
                .find(|(path, _)| path == owner)
                .map(|(_, source)| source)
                .unwrap_or_else(|| panic!("{owner} ({why}) is not a source file any more"));
            assert!(
                production_lines(source)
                    .iter()
                    .any(|(_, line)| rule.spelled.iter().any(|s| line.contains(s))),
                "{owner} is listed as an owner of \"{}\" ({why}) and no longer spells it",
                rule.state
            );
        }
    }
}

/// The search itself, on sources made up for the purpose: a spelling outside
/// the owners is found, one inside them, in a comment or in a test module is
/// not.
#[test]
fn the_search_finds_a_second_spelling_and_nothing_else() {
    let line =
        "    let n = q(\"SELECT count(*) FROM node WHERE json_extract(meta,'$.archived')=1\");";
    let files = vec![
        (
            "mimir-cli/src/somewhere_new.rs".to_string(),
            format!("fn stats() {{\n{line}\n}}\n"),
        ),
        (
            "mimir-core/src/lifecycle.rs".to_string(),
            format!("fn definition() {{\n{line}\n}}\n"),
        ),
        (
            "mimir-cli/src/commented.rs".to_string(),
            "// the flag is '$.archived'\nfn nothing() {}\n".to_string(),
        ),
        (
            "mimir-cli/src/tested.rs".to_string(),
            format!("fn nothing() {{}}\n\n#[cfg(test)]\nmod tests {{\n{line}\n}}\n"),
        ),
        // A test-only helper in the middle of a file hides its own body and
        // nothing after it.
        (
            "mimir-cli/src/helper_then_code.rs".to_string(),
            format!(
                "#[cfg(test)]\nuse std::fmt;\n\n#[cfg(test)]\nfn helper() {{\n{line}\n}}\n\n\
                 fn real() {{\n{line}\n}}\n"
            ),
        ),
    ];
    let found = violations(&files);
    assert_eq!(found.len(), 2, "{found:?}");
    assert_eq!(found[0].0, "mimir-cli/src/somewhere_new.rs");
    assert_eq!(found[0].1, 2);
    assert!(RULES[found[0].2].state.starts_with("archived"));
    assert_eq!(found[1].0, "mimir-cli/src/helper_then_code.rs");
    assert_eq!(
        found[1].1, 10,
        "the line inside `real`, not the one in the helper"
    );
}
