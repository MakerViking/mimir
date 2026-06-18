//! `mimir rules` — the per-project "rules pack".
//!
//! Closes the one gap Mimir had versus a dedicated session-bootstrapper: a
//! place to keep a project's architecture rules / conventions that gets
//! surfaced automatically at session start (via the SessionStart hook that
//! `mimir init --hooks` installs) instead of the agent re-deriving them by
//! scanning the repo every time.
//!
//! Storage is deliberately boring: a project-scoped, *pinned* memory tagged
//! `mimir-rules`. Pinned ⇒ never decays, never consolidated away, and it
//! surfaces through ordinary `recall` too.

use anyhow::{bail, Context, Result};
use mimir_core::model::{now_unix, Kind, NewNode, Node};
use mimir_core::store::{self, row_to_node, NODE_COLS};
use mimir_core::Mimir;
use rusqlite::{params, Connection, OptionalExtension};

/// Tag that marks a memory as the project's rules pack.
pub const RULES_TAG: &str = "mimir-rules";

fn find_rules(conn: &Connection, project_id: i64) -> Result<Option<Node>> {
    let node = conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'memory' AND project_id = ?1 AND deleted_at IS NULL
                   AND tags_text LIKE '%{RULES_TAG}%'
                 ORDER BY updated_at DESC LIMIT 1"
            ),
            [project_id],
            row_to_node,
        )
        .optional()?;
    Ok(node)
}

fn project(mimir: &Mimir) -> Result<Node> {
    mimir.project_for_cwd(&std::env::current_dir()?)?.context(
        "not inside a project. The rules pack is per-project; it needs a \
             project root — a VCS marker (.git/.hg/.svn/.jj), a build file \
             (Cargo.toml/package.json/pyproject.toml/go.mod/…), or `touch .mimir`.",
    )
}

/// Set (or replace) the project's rules pack. Text comes from args, or stdin
/// when no args are given.
pub fn set(text_parts: Vec<String>) -> Result<()> {
    let mut mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    let proj_name = proj.title.clone().unwrap_or_else(|| "project".into());

    let text = if text_parts.is_empty() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        text_parts.join(" ")
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        bail!("empty rules text (pass it as arguments or pipe it on stdin)");
    }
    let title = format!("{proj_name} — project rules");

    if let Some(existing) = find_rules(&mimir.conn, proj.id)? {
        mimir.conn.execute(
            "UPDATE node SET body = ?2, title = ?3, updated_at = ?4, pinned = 1 WHERE id = ?1",
            params![existing.id, text, title, now_unix()],
        )?;
        println!("updated rules pack for {proj_name} ({} chars)", text.len());
    } else {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some(title);
        new.body = Some(text.clone());
        new.project_id = Some(proj.id);
        new.tags = vec![RULES_TAG.into()];
        let node = store::insert_node(&mimir.conn, new)?;
        store::set_pinned(&mimir.conn, node.id, true)?;
        println!(
            "stored rules pack for {proj_name} ({} chars) — pinned",
            text.len()
        );
    }
    // Keep semantic recall fresh; harmless no-op without a model.
    if let Err(err) = mimir.embed_pending() {
        tracing::warn!(%err, "embedding rules pack failed");
    }
    Ok(())
}

/// Print the project's rules pack to stdout. Used by the SessionStart hook, so
/// it stays silent (exit 0, no stdout) when there's no project or no pack —
/// the hook then injects nothing.
pub fn show() -> Result<()> {
    let mimir = Mimir::open()?;
    let Some(proj) = mimir.project_for_cwd(&std::env::current_dir()?)? else {
        return Ok(());
    };
    match find_rules(&mimir.conn, proj.id)? {
        Some(node) => {
            if let Some(body) = node.body {
                let name = proj.title.as_deref().unwrap_or("project");
                println!("# {name} — project rules (via Mimir)\n\n{body}");
            }
        }
        None => {
            eprintln!("no rules pack for this project (set one with `mimir rules set <text>`)");
        }
    }
    Ok(())
}

/// Remove the project's rules pack.
pub fn clear() -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    match find_rules(&mimir.conn, proj.id)? {
        Some(node) => {
            store::soft_delete(&mimir.conn, node.id)?;
            println!(
                "cleared rules pack for {}",
                proj.title.as_deref().unwrap_or("project")
            );
        }
        None => println!("no rules pack to clear"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::config::Paths;

    fn mimir_in(dir: &std::path::Path) -> Mimir {
        Mimir::open_at(Paths::under_root(dir)).unwrap()
    }

    #[test]
    fn set_then_find_is_pinned_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let proj = store::ensure_project(&mimir.conn, "/tmp/p", "p").unwrap();

        // Insert directly (set() needs cwd detection; exercise the storage shape).
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some("p — project rules".into());
        new.body = Some("Always use tabs.".into());
        new.project_id = Some(proj.id);
        new.tags = vec![RULES_TAG.into()];
        let node = store::insert_node(&mimir.conn, new).unwrap();
        store::set_pinned(&mimir.conn, node.id, true).unwrap();

        let found = find_rules(&mimir.conn, proj.id).unwrap().unwrap();
        assert!(found.pinned);
        assert_eq!(found.body.as_deref(), Some("Always use tabs."));
        assert!(found.tags_text.contains(RULES_TAG));
    }

    #[test]
    fn none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let proj = store::ensure_project(&mimir.conn, "/tmp/p", "p").unwrap();
        assert!(find_rules(&mimir.conn, proj.id).unwrap().is_none());
    }
}
