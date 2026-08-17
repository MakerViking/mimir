//! Guard anchors: file/command triggers that surface a memory on
//! `PreToolUse` (Bash/Edit/Write) instead of waiting for a matching prompt.
//!
//! Clean-room design — the functional idea ("a memory can declare which
//! files/commands should surface it") is THOR-inspired, but this module was
//! written from that one-sentence spec only; THOR's (GPLv3) implementation
//! was never read. Storage, matching, and dedup below are Mimir-native:
//! patterns live in `meta.anchors` (same `json_set`-on-`node.meta` idiom as
//! `store::set_fires_when`), and a match is deduped per session via the v8
//! `injection_log` ledger (`store::injection_seen`/`record_injection`) —
//! the same ledger `inject::compute_with_session` already uses, so "was
//! this memory already surfaced this session" is one shared concept, not
//! two parallel ones.
//!
//! Independent of `[hooks] context_guard`: anchors have no on/off mode of
//! their own. A memory with no `meta.anchors` never matches anything, so
//! installing the PreToolUse hook that calls into this module is a no-op
//! until someone actually runs `mimir remember --anchor ...` — see
//! `crates/mimir-cli/src/context_guard_cmd.rs::pretool`.

use rusqlite::{params, Connection};

use crate::error::Result;
use crate::format::agent_line;
use crate::model::Scope;
use crate::store::{self, row_to_node, NODE_COLS};

/// Normalize author-declared guard-anchor patterns for `meta.anchors`. Each
/// entry may carry the CLI's own `file:` scheme prefix (`mimir remember
/// --anchor "file:store.rs"`) — stripped here so matching only ever sees
/// the bare glob (see [`anchor_matches_path`]). Same "reject, don't mangle"
/// cap discipline as `memory::sanitize_fires_when`: trimmed, capped to 8
/// patterns of at most 200 characters each, empties dropped.
pub fn sanitize_anchors(patterns: Vec<String>) -> Vec<String> {
    patterns
        .into_iter()
        .map(|p| {
            let t = p.trim();
            t.strip_prefix("file:").unwrap_or(t).trim().to_string()
        })
        .filter(|p| !p.is_empty() && p.len() <= 200)
        .take(8)
        .collect()
}

/// Set a memory's guard-anchor patterns (`meta.anchors`), overwriting any
/// previous list. `patterns` is expected pre-sanitized (see
/// [`sanitize_anchors`]) — a raw setter, same division of labor
/// `store::set_fires_when` uses for `meta.fires_when`. Kept in this module
/// rather than `store.rs` since guard anchors are new, self-contained
/// functionality.
pub fn set_anchors(conn: &Connection, id: i64, patterns: &[String]) -> Result<()> {
    let json = serde_json::to_string(patterns).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "UPDATE node SET meta = json_set(meta, '$.anchors', json(?2)) WHERE id = ?1",
        params![id, json],
    )?;
    Ok(())
}

/// Every live, in-scope memory that declares at least one anchor pattern.
/// Small by construction (anchors are an explicit, rare author opt-in), so
/// filtering by pattern match happens in Rust after this one query rather
/// than pushing glob logic into SQL.
fn anchored_candidates(conn: &Connection, scope: Scope) -> Result<Vec<crate::model::Node>> {
    let sql = format!(
        "SELECT {NODE_COLS} FROM node n
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL AND {}
           AND json_array_length(n.meta, '$.anchors') > 0
         ORDER BY n.id",
        crate::search::scope_sql(scope)
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// Minimal glob match, anchored to the *whole* string (like shell
/// `fnmatch`): `*` matches any run of characters (including `/` — there is
/// no `**`-vs-`*` distinction here, deliberately simpler than a full glob
/// library since anchor patterns are short path fragments, not nested
/// directory trees), `?` matches exactly one character. No external crate:
/// nothing in the workspace already depends on one, and this is ~15 lines.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    // Classic two-pointer glob match with backtracking on the last `*`.
    let (mut pi, mut ti, mut star, mut star_ti) = (0usize, 0usize, None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// True if `pattern` matches `path`. Matching is against the path's
/// **suffix** — `pattern` need not repeat the full (often absolute) path a
/// tool reports, only enough of the tail to be unambiguous (e.g.
/// `"store.rs"` or `"mimir-core/src/store.rs"`), mirroring
/// `store::files_by_path_suffix`'s existing suffix-match convention. Tried
/// against every `/`-delimited suffix of `path`, so a bare filename pattern
/// matches regardless of how deep it lives.
pub fn anchor_matches_path(pattern: &str, path: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    let normalized = path.replace('\\', "/");
    let components: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    (0..components.len()).any(|start| glob_match(pattern, &components[start..].join("/")))
}

/// True if `pattern` matches any whitespace/quote-delimited token of
/// `command` (via [`anchor_matches_path`]) — covers the common case of a
/// path-like argument appearing in a shell command, without trying to
/// actually parse shell syntax.
pub fn anchor_matches_command(pattern: &str, command: &str) -> bool {
    command
        .split(|c: char| c.is_whitespace() || c == '"' || c == '\'')
        .filter(|t| !t.is_empty())
        .any(|tok| anchor_matches_path(pattern, tok))
}

/// The PreToolUse entry point: find the first in-scope anchored memory
/// whose pattern matches `target` and hasn't already been surfaced this
/// session, record it as seen, and format it for
/// `hookSpecificOutput.additionalContext`. `is_command` selects the matcher
/// — `true` for Bash (`target` = the shell command text), `false` for
/// Edit/Write (`target` = the file path). Fail-open on any read/write
/// error: returns `Ok(None)`  rather than surfacing a DB hiccup to the hook.
pub fn find_and_record(
    conn: &Connection,
    session_id: &str,
    scope: Scope,
    target: &str,
    is_command: bool,
) -> Result<Option<String>> {
    let candidates = match anchored_candidates(conn, scope) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };
    for node in candidates {
        let Some(patterns) = node.meta.get("anchors").and_then(|v| v.as_array()) else {
            continue;
        };
        let hit = patterns.iter().filter_map(|v| v.as_str()).any(|pattern| {
            if is_command {
                anchor_matches_command(pattern, target)
            } else {
                anchor_matches_path(pattern, target)
            }
        });
        if !hit {
            continue;
        }
        if store::injection_seen(conn, session_id, node.id).unwrap_or(false) {
            continue;
        }
        let _ = store::record_injection(conn, session_id, node.id);
        let projects = store::project_titles(conn).unwrap_or_default();
        let project_name = node
            .project_id
            .and_then(|id| projects.get(&id).map(String::as_str));
        return Ok(Some(format!(
            "Guard anchor matched — relevant memory: {}",
            agent_line(&node, project_name, 400)
        )));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::{self, Remember, RememberOutcome};
    use crate::model::MemoryType;

    #[test]
    fn glob_match_basics() {
        assert!(glob_match("store.rs", "store.rs"));
        assert!(!glob_match("store.rs", "other.rs"));
        assert!(glob_match("*.rs", "store.rs"));
        assert!(glob_match("store.*", "store.rs"));
        assert!(glob_match("s?ore.rs", "store.rs"));
        assert!(glob_match(
            "mimir-core/*/store.rs",
            "mimir-core/src/store.rs"
        ));
        // `*` matches any run of characters INCLUDING '/' (no `**` distinction
        // here — see the function's doc comment), so a single `*` segment
        // spans "src/deep" too.
        assert!(glob_match(
            "mimir-core/*/store.rs",
            "mimir-core/src/deep/store.rs"
        ));
    }

    #[test]
    fn anchor_matches_path_tries_every_suffix() {
        assert!(anchor_matches_path(
            "store.rs",
            "/home/t/repo/crates/mimir-core/src/store.rs"
        ));
        assert!(anchor_matches_path(
            "mimir-core/src/store.rs",
            "/home/t/repo/crates/mimir-core/src/store.rs"
        ));
        assert!(!anchor_matches_path(
            "commands.rs",
            "/home/t/repo/crates/mimir-core/src/store.rs"
        ));
        assert!(anchor_matches_path("*.rs", "/a/b/store.rs"));
        assert!(!anchor_matches_path("*.rs", "/a/b/store.txt"));
    }

    #[test]
    fn anchor_matches_command_scans_tokens() {
        assert!(anchor_matches_command(
            "store.rs",
            "cat crates/mimir-core/src/store.rs"
        ));
        assert!(anchor_matches_command(
            "store.rs",
            r#"rg "meta.anchors" crates/mimir-core/src/store.rs"#
        ));
        assert!(!anchor_matches_command("store.rs", "cargo build --release"));
    }

    #[test]
    fn sanitize_anchors_strips_file_prefix_trims_caps() {
        let out = sanitize_anchors(vec![
            "file:store.rs".into(),
            "  crates/mimir-core/**  ".into(),
            "".into(),
            "   ".into(),
            "x".repeat(201),
        ]);
        assert_eq!(
            out,
            vec!["store.rs".to_string(), "crates/mimir-core/**".to_string()]
        );
    }

    #[test]
    fn sanitize_anchors_caps_at_eight() {
        let patterns: Vec<String> = (0..20).map(|i| format!("p{i}.rs")).collect();
        assert_eq!(sanitize_anchors(patterns).len(), 8);
    }

    fn seed_anchored(conn: &Connection, text: &str, anchors: &[&str]) -> i64 {
        let outcome = memory::remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Gotcha,
                tags: vec![],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        let RememberOutcome::Created(node) = outcome else {
            panic!("expected a fresh memory");
        };
        let patterns: Vec<String> = anchors.iter().map(|s| s.to_string()).collect();
        set_anchors(conn, node.id, &patterns).unwrap();
        node.id
    }

    #[test]
    fn find_and_record_matches_edit_and_dedupes_per_session() {
        let conn = db::open_in_memory().unwrap();
        seed_anchored(
            &conn,
            "store.rs: never call json_set twice in one UPDATE",
            &["store.rs"],
        );

        let first = find_and_record(
            &conn,
            "s1",
            Scope::All,
            "/home/t/repo/crates/mimir-core/src/store.rs",
            false,
        )
        .unwrap();
        assert!(first.unwrap().contains("json_set"));

        // Same session, same file again: silent (already surfaced).
        let second = find_and_record(
            &conn,
            "s1",
            Scope::All,
            "/home/t/repo/crates/mimir-core/src/store.rs",
            false,
        )
        .unwrap();
        assert!(second.is_none());

        // A different session sees it fresh.
        let third = find_and_record(
            &conn,
            "s2",
            Scope::All,
            "/home/t/repo/crates/mimir-core/src/store.rs",
            false,
        )
        .unwrap();
        assert!(third.is_some());
    }

    #[test]
    fn find_and_record_matches_bash_command_text() {
        let conn = db::open_in_memory().unwrap();
        seed_anchored(
            &conn,
            "migrations.rs: bump schema_version too",
            &["migrations.rs"],
        );
        let hit = find_and_record(
            &conn,
            "s1",
            Scope::All,
            "cat crates/mimir-core/src/db/migrations.rs",
            true,
        )
        .unwrap();
        assert!(hit.unwrap().contains("schema_version"));
    }

    #[test]
    fn find_and_record_is_none_without_a_match() {
        let conn = db::open_in_memory().unwrap();
        seed_anchored(&conn, "unrelated", &["store.rs"]);
        let hit = find_and_record(&conn, "s1", Scope::All, "/a/b/commands.rs", false).unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn find_and_record_is_none_when_no_memory_has_anchors() {
        let conn = db::open_in_memory().unwrap();
        memory::remember(
            &conn,
            Remember {
                text: "plain memory, no anchors".into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        let hit = find_and_record(&conn, "s1", Scope::All, "/a/b/store.rs", false).unwrap();
        assert!(hit.is_none());
    }
}
