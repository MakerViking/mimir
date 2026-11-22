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
use mimir_core::model::{now_unix, Actor, Kind, NewNode, Node};
use mimir_core::store::{self, row_to_node, NODE_COLS};
use mimir_core::{memory, Mimir};
use rusqlite::{params, Connection};

/// Tag that marks a memory as the project's rules pack.
pub const RULES_TAG: &str = "mimir-rules";

/// The project's rules pack: the most recently updated *current* memory of
/// that project carrying exactly the [`RULES_TAG`] tag.
///
/// Exactly the tag — a memory tagged `no-mimir-rules-here` is not a rules
/// pack, and a substring match would let it be shown as one, replaced by
/// `rules set` and deleted by `rules clear`. And current, as
/// `Lifecycle` means it: a pack that was superseded or has expired is
/// history, and the SessionStart hook must not inject it as the project's
/// rules.
pub(crate) fn find_rules(conn: &Connection, project_id: i64) -> Result<Option<Node>> {
    Ok(current_rules(conn, project_id)?.into_iter().next())
}

/// Every current rules pack of the project, the one in force first.
///
/// There is normally one. There are several when a linked worktree had a
/// pack of its own and its memories moved to the main checkout's project,
/// which may have had one already. Then:
///
/// 1. A pack that was written in this project outranks one that was moved
///    into it. The main checkout's rules do not change because a worktree
///    of it existed.
/// 2. Among equals, the pack whose *text* was written last
///    (`text_clock::decided_at`), not the row written last: the move marks
///    every memory it carries as changed, and that mark must not be what
///    decides which rules a repository runs under.
fn current_rules(conn: &Connection, project_id: i64) -> Result<Vec<Node>> {
    // The LIKE narrows; the tag comparison below decides (LIKE ignores case
    // and treats `_` as a wildcard).
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'memory' AND project_id = ?1 AND deleted_at IS NULL
           AND (' ' || tags_text || ' ') LIKE ?2"
    ))?;
    let now = now_unix();
    let packs: Vec<Node> = stmt
        .query_map(params![project_id, format!("% {RULES_TAG} %")], row_to_node)?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .filter(|node| node.tags().contains(&RULES_TAG) && node.lifecycle(now).is_current())
        .collect();
    let mut ranked: Vec<(bool, Node)> = packs
        .into_iter()
        .map(|node| (moved_in_from_a_worktree(conn, node.id), node))
        .collect();
    ranked.sort_by_key(|(moved_in, node)| {
        (
            *moved_in,
            std::cmp::Reverse((mimir_core::text_clock::decided_at(node), node.id)),
        )
    });
    Ok(ranked.into_iter().map(|(_, node)| node).collect())
}

/// Whether the ledger says this memory came into its project with a linked
/// worktree's memories. A ledger that cannot be read says no.
fn moved_in_from_a_worktree(conn: &Connection, node_id: i64) -> bool {
    mimir_core::audit::history(conn, node_id).is_ok_and(|rows| {
        rows.iter()
            .any(|row| row.reason.as_deref() == Some(mimir_core::WORKTREE_MEMORIES_MOVED))
    })
}

/// What [`store_rules`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stored {
    Created,
    Updated,
}

/// Write `text` as the rules pack of `project`: replace the current pack's
/// text if there is one, create a pinned one if not.
///
/// A rules pack is a memory and changes like one. Replacing it goes through
/// `memory::edit`, so the previous wording is kept in its history, the
/// content hash follows the text and the ledger records who changed it —
/// the things `mimir history` and `mimir audit` show for any other memory.
fn store_rules(conn: &Connection, project: &Node, text: &str) -> Result<Stored> {
    let title = format!(
        "{} — project rules",
        project.title.as_deref().unwrap_or("project")
    );
    if let Some(existing) = find_rules(conn, project.id)? {
        memory::edit(
            conn,
            existing.id,
            memory::Edit {
                text: Some(text.to_string()),
                title: Some(title),
                ..Default::default()
            },
            Actor::Human,
            Some("mimir rules set"),
        )?;
        // A pack someone unpinned is pinned again: pinned is what keeps it
        // out of decay and consolidation. Already pinned, this is a no-op.
        store::set_pinned(conn, existing.id, true, Actor::Human, Some("rules pack"))?;
        return Ok(Stored::Updated);
    }
    let mut new = NewNode::new(Kind::Memory);
    new.subkind = Some("note".into());
    new.title = Some(title);
    new.body = Some(text.to_string());
    new.project_id = Some(project.id);
    new.tags = vec![RULES_TAG.into()];
    // Hashed like any memory. Without a hash the pack is invisible to
    // `remember`'s exact-duplicate check and counts as un-embedded forever.
    new.content_hash = Some(memory::content_hash(text));
    let node = store::insert_node(conn, new)?;
    store::set_pinned(conn, node.id, true, Actor::Human, Some("rules pack"))?;
    Ok(Stored::Created)
}

/// The project whose rules pack this is: the memory project, so every
/// linked worktree of a repository reads and writes the one pack of its
/// main checkout.
fn project(mimir: &Mimir) -> Result<Node> {
    mimir
        .memory_project_for_cwd(&std::env::current_dir()?)?
        .context(
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
    // The pack is a pinned memory injected into every session of the
    // project, and it syncs like one; it gets the guard a memory gets.
    crate::capture::refuse_secrets(
        &mimir.conn,
        &text,
        &[],
        &[],
        mimir_core::secrets::Surface::CliRemember,
    )?;
    match store_rules(&mimir.conn, &proj, &text)? {
        Stored::Updated => println!("updated rules pack for {proj_name} ({} chars)", text.len()),
        Stored::Created => println!(
            "stored rules pack for {proj_name} ({} chars) — pinned",
            text.len()
        ),
    }
    // Keep semantic recall fresh; harmless no-op without a model.
    if let Err(err) = mimir.embed_pending() {
        tracing::warn!(%err, "embedding rules pack failed");
    }
    Ok(())
}

/// How long `show --hook` waits for the hook event. Claude Code writes the
/// event as it starts the hook, so this only ever elapses when something
/// runs the hook command with a stdin that stays open and silent.
const HOOK_EVENT_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Print the project's rules pack to stdout. Used by the SessionStart hook, so
/// it stays silent (exit 0, no stdout) when there's no project or no pack —
/// the hook then injects nothing.
///
/// `hook` is how the command is told it is running as that hook, and it is
/// the only thing that makes it read stdin: the project is then the one the
/// event's `cwd` is in, the same directory the session brief and the
/// context guard scope by, and an error is logged instead of failing the
/// session start (`context_guard_cmd::fail_open`). Typed by hand, without
/// the flag, it is the project of the working directory, an error is an
/// error, and stdin is left alone — so it can sit inside a `while read` loop
/// or behind a pipe without eating the input or waiting on it.
pub fn show(hook: bool) -> Result<()> {
    use crate::context_guard_cmd::{fail_open, HookEvent};
    if !hook {
        return show_for(&std::env::current_dir()?, Reader::Person);
    }
    fail_open("rules show", || {
        let event = HookEvent::from_piped_stdin(HOOK_EVENT_WAIT);
        match event.as_ref().and_then(HookEvent::cwd) {
            Some(cwd) => show_for(cwd, Reader::Session),
            None => show_for(&std::env::current_dir()?, Reader::Session),
        }
    })
}

/// Who `rules show` is printing for: someone at a terminal, who can act on a
/// note about the store, or a session start, where every line is context.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reader {
    Person,
    Session,
}

/// Print the rules pack of the project `cwd` is in, if it has one.
fn show_for(cwd: &std::path::Path, reader: Reader) -> Result<()> {
    let mimir = Mimir::open()?;
    let Some(proj) = mimir.memory_project_for_cwd(cwd)? else {
        return Ok(());
    };
    let mut packs = current_rules(&mimir.conn, proj.id)?.into_iter();
    match packs.next() {
        Some(node) => {
            if reader == Reader::Person {
                if let Some(note) = shadowed_note(&node, packs.as_slice()) {
                    eprintln!("{note}");
                }
            }
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

/// What a person is told when a project holds more than one current rules
/// pack: which is in force, and how to be rid of the others. `None` for the
/// ordinary case of one pack.
fn shadowed_note(in_force: &Node, others: &[Node]) -> Option<String> {
    if others.is_empty() {
        return None;
    }
    let refs = |nodes: &[Node]| {
        nodes
            .iter()
            .map(|n| mimir_core::model::short_uid(n.kind, &n.uid))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Some(format!(
        "note: this project has {} rules packs. {} is in force; {} {} not shown to \
         sessions (a pack that came in with a worktree's memories ranks below one written \
         here, then the newer text wins). `mimir forget <id>` removes one, `mimir get <id>` \
         shows it.",
        others.len() + 1,
        refs(std::slice::from_ref(in_force)),
        refs(others),
        if others.len() == 1 { "is" } else { "are" },
    ))
}

/// Remove the project's rules pack.
pub fn clear() -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = project(&mimir)?;
    match find_rules(&mimir.conn, proj.id)? {
        Some(node) => {
            store::soft_delete(
                &mimir.conn,
                node.id,
                mimir_core::model::Actor::Human,
                Some("mimir rules clear"),
            )?;
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

    /// A worktree with a rules pack of its own, in a repository whose main
    /// checkout has one too: when the worktree's memories move home the
    /// project holds two. The pack in force is the one whose text is newer.
    /// The move marks the pack it carries as changed, and that mark must not
    /// be what puts it in force over rules written after it.
    #[test]
    fn a_pack_moved_in_from_a_worktree_does_not_win_by_having_been_moved() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let home = store::ensure_project(&mimir.conn, "/tmp/repo", "repo").unwrap();
        let worktree = store::ensure_project(&mimir.conn, "/tmp/repo-wt", "repo-wt").unwrap();
        store_rules(
            &mimir.conn,
            &worktree,
            "Old rules, written in the worktree.",
        )
        .unwrap();
        store_rules(
            &mimir.conn,
            &home,
            "Newer rules, written in the main checkout.",
        )
        .unwrap();
        let moved = find_rules(&mimir.conn, worktree.id).unwrap().unwrap();
        let newer = find_rules(&mimir.conn, home.id).unwrap().unwrap();
        // The worktree's pack was written a day before the main checkout's.
        // (A pack is pinned as it is created, so it carries a text clock;
        // the fixture moves that back together with the row.)
        let yesterday = now_unix() - 86_400;
        for (id, at) in [(moved.id, yesterday), (newer.id, yesterday + 3_600)] {
            mimir
                .conn
                .execute(
                    "UPDATE node SET created_at = ?2, updated_at = ?2,
                                     meta = json_set(meta, '$.text_clock.at', ?2)
                     WHERE id = ?1",
                    params![id, at],
                )
                .unwrap();
        }
        let moved = store::get_node(&mimir.conn, moved.id).unwrap();

        store::reproject(&mimir.conn, &moved, Some(home.id), Actor::System, None).unwrap();
        let moved_now = store::get_node(&mimir.conn, moved.id).unwrap();
        assert!(
            moved_now.updated_at > yesterday + 3_600,
            "the move is the later write"
        );

        let packs = current_rules(&mimir.conn, home.id).unwrap();
        assert_eq!(
            packs.iter().map(|n| n.id).collect::<Vec<_>>(),
            [newer.id, moved.id],
            "the newer text is in force"
        );
        let note = shadowed_note(&packs[0], &packs[1..]).unwrap();
        assert!(note.contains("2 rules packs"), "{note}");
        assert!(
            note.contains(&mimir_core::model::short_uid(Kind::Memory, &moved.uid)),
            "{note}"
        );
        assert_eq!(shadowed_note(&packs[0], &[]), None);

        // Rewriting the rules replaces the pack in force, and it stays so.
        store_rules(&mimir.conn, &home, "Rules as of today.").unwrap();
        assert_eq!(
            find_rules(&mimir.conn, home.id).unwrap().unwrap().id,
            newer.id
        );
    }

    /// The upgrade case: both packs were written by a build that gave a
    /// rules pack no content hash, so neither can carry a text clock, and
    /// the worktree's pack is the later text. Moved in with the worktree's
    /// memories, it still does not displace the pack written in the main
    /// checkout; with no pack there, it is the pack.
    #[test]
    fn the_main_checkouts_pack_outranks_one_moved_in_with_a_worktrees_memories() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let home = store::ensure_project(&mimir.conn, "/tmp/repo", "repo").unwrap();
        let worktree = store::ensure_project(&mimir.conn, "/tmp/repo-wt", "repo-wt").unwrap();
        let pack_of = |project: &Node, text: &str| -> Node {
            let mut new = NewNode::new(Kind::Memory);
            new.subkind = Some("note".into());
            new.body = Some(text.into());
            new.project_id = Some(project.id);
            new.tags = vec![RULES_TAG.into()];
            store::insert_node(&mimir.conn, new).unwrap()
        };
        let main_pack = pack_of(&home, "Rules of the main checkout.");
        let worktree_pack = pack_of(&worktree, "Rules someone set in the worktree, later.");
        assert!(main_pack.content_hash.is_none() && worktree_pack.content_hash.is_none());

        let moved = store::reproject_all(
            &mimir.conn,
            worktree.id,
            home.id,
            Actor::System,
            Some(mimir_core::WORKTREE_MEMORIES_MOVED),
        )
        .unwrap();
        assert_eq!(moved, 1);

        let packs = current_rules(&mimir.conn, home.id).unwrap();
        assert_eq!(
            packs.iter().map(|n| n.id).collect::<Vec<_>>(),
            [main_pack.id, worktree_pack.id]
        );

        // With no pack of its own, the repository runs under the one that
        // came in.
        store::soft_delete(&mimir.conn, main_pack.id, Actor::Human, None).unwrap();
        assert_eq!(
            find_rules(&mimir.conn, home.id).unwrap().unwrap().id,
            worktree_pack.id
        );
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
        store::set_pinned(
            &mimir.conn,
            node.id,
            true,
            mimir_core::model::Actor::Human,
            None,
        )
        .unwrap();

        let found = find_rules(&mimir.conn, proj.id).unwrap().unwrap();
        assert!(found.pinned);
        assert_eq!(found.body.as_deref(), Some("Always use tabs."));
        assert!(found.tags_text.contains(RULES_TAG));
    }

    fn revisions(mimir: &Mimir, id: i64) -> Vec<String> {
        mimir_core::revision::history(&mimir.conn, id)
            .unwrap()
            .into_iter()
            .map(|r| r.body.unwrap_or_default())
            .collect()
    }

    fn ledger(mimir: &Mimir, id: i64) -> Vec<mimir_core::model::MutationOp> {
        mimir_core::audit::history(&mimir.conn, id)
            .unwrap()
            .into_iter()
            .map(|m| m.op)
            .collect()
    }

    /// Replacing the pack used to be a raw UPDATE: the old rules vanished
    /// without a revision, the hash stayed stale and the ledger said nothing.
    #[test]
    fn replacing_the_pack_is_an_edit_like_any_other() {
        use mimir_core::model::MutationOp;
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let proj = store::ensure_project(&mimir.conn, "/tmp/p", "p").unwrap();

        assert_eq!(
            store_rules(&mimir.conn, &proj, "Always use tabs.").unwrap(),
            Stored::Created
        );
        let first = find_rules(&mimir.conn, proj.id).unwrap().unwrap();
        assert!(first.pinned);
        assert_eq!(first.title.as_deref(), Some("p — project rules"));
        assert_eq!(
            first.content_hash,
            Some(memory::content_hash("Always use tabs.")),
            "a new pack is hashed like any memory"
        );

        assert_eq!(
            store_rules(&mimir.conn, &proj, "Always use spaces.").unwrap(),
            Stored::Updated
        );
        let second = find_rules(&mimir.conn, proj.id).unwrap().unwrap();
        assert_eq!(second.id, first.id, "replaced in place, not duplicated");
        assert_eq!(second.body.as_deref(), Some("Always use spaces."));
        assert_eq!(second.title.as_deref(), Some("p — project rules"));
        assert!(second.pinned);
        assert_eq!(
            second.content_hash,
            Some(memory::content_hash("Always use spaces."))
        );
        assert_eq!(revisions(&mimir, first.id), vec!["Always use tabs."]);
        assert_eq!(
            ledger(&mimir, first.id),
            vec![MutationOp::Pin, MutationOp::Edit]
        );

        // Setting the same text again is not a change.
        store_rules(&mimir.conn, &proj, "Always use spaces.").unwrap();
        assert_eq!(ledger(&mimir, first.id).len(), 2);

        // A pack someone unpinned is pinned again by the next `set`.
        store::set_pinned(&mimir.conn, first.id, false, Actor::Human, None).unwrap();
        store_rules(&mimir.conn, &proj, "Always use spaces, four of them.").unwrap();
        assert!(find_rules(&mimir.conn, proj.id).unwrap().unwrap().pinned);
    }

    fn tagged(mimir: &Mimir, project_id: i64, body: &str, tags: &[&str]) -> Node {
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some(body.into());
        new.body = Some(body.into());
        new.project_id = Some(project_id);
        new.tags = tags.iter().map(|t| t.to_string()).collect();
        store::insert_node(&mimir.conn, new).unwrap()
    }

    #[test]
    fn the_pack_is_found_by_its_exact_tag_among_current_memories() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let proj = store::ensure_project(&mimir.conn, "/tmp/p", "p").unwrap();
        let other = store::ensure_project(&mimir.conn, "/tmp/q", "q").unwrap();

        // Tags that merely contain the rules tag, and another project's pack.
        tagged(
            &mimir,
            proj.id,
            "a note about the rules",
            &["no-mimir-rules-here"],
        );
        tagged(&mimir, proj.id, "another", &["mimir-rules-draft", "x"]);
        tagged(&mimir, other.id, "q's rules", &[RULES_TAG]);
        assert!(find_rules(&mimir.conn, proj.id).unwrap().is_none());

        let pack = tagged(&mimir, proj.id, "Always use tabs.", &["style", RULES_TAG]);
        assert_eq!(
            find_rules(&mimir.conn, proj.id).unwrap().map(|n| n.id),
            Some(pack.id)
        );

        // Superseded, it is history; `set` then starts a new pack rather
        // than quietly rewriting the retired one.
        let replacement = tagged(&mimir, proj.id, "see the handbook", &[]);
        store::supersede(&mimir.conn, pack.id, replacement.id, Actor::Human, None).unwrap();
        assert!(find_rules(&mimir.conn, proj.id).unwrap().is_none());
        assert_eq!(
            store_rules(&mimir.conn, &proj, "Always use spaces.").unwrap(),
            Stored::Created
        );
        let fresh = find_rules(&mimir.conn, proj.id).unwrap().unwrap();
        assert_ne!(fresh.id, pack.id);

        // Expired, it is history too.
        store::set_expiry(&mimir.conn, fresh.id, Some(now_unix() - 60), None).unwrap();
        assert!(find_rules(&mimir.conn, proj.id).unwrap().is_none());
    }

    #[test]
    fn none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mimir = mimir_in(dir.path());
        let proj = store::ensure_project(&mimir.conn, "/tmp/p", "p").unwrap();
        assert!(find_rules(&mimir.conn, proj.id).unwrap().is_none());
    }
}
