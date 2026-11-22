//! `mimir project init [--sync | --no-sync]` — give a project a portable,
//! machine-stable identity by writing a `.mimir` marker (a committed `id`,
//! plus an opt-in `sync` flag). With a portable key, the project's memories
//! can replicate across machines via the sync layer; without one, they stay
//! local.
//!
//! `mimir project prune [--apply]` — the other end of a project's life: list
//! the projects whose directory is gone and, when asked, remove the code
//! graph still stored for them (see `mimir_graph::prune`).

use anyhow::{bail, Result};
use mimir_core::scope::{self, MarkerState, MimirMarker};
use mimir_core::{MemoryFiling, Mimir};

/// What `project init` was asked to do with the marker's `sync` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncChange {
    /// Neither flag: leave it as the marker has it (off for a new marker).
    Keep,
    /// `--sync`
    Enable,
    /// `--no-sync`
    Disable,
}

impl SyncChange {
    /// From the two command-line flags. clap rejects both at once.
    pub fn from_flags(sync: bool, no_sync: bool) -> Self {
        match (sync, no_sync) {
            (true, _) => SyncChange::Enable,
            (false, true) => SyncChange::Disable,
            (false, false) => SyncChange::Keep,
        }
    }
}

/// Create or update `.mimir` in the current project root: ensure a stable `id`
/// (generated once if absent) and set `sync` as asked. Idempotent: re-running
/// keeps the existing id. Prints the resulting key and sync state.
///
/// A `.mimir` that exists but cannot be read is never written over: its id
/// may be the one the project's synced memories are filed under.
pub fn init(sync: SyncChange) -> Result<()> {
    let cwd = std::env::current_dir()?;
    // Resolve the project root (VCS / build-file / existing `.mimir`); fall back
    // to cwd so `init` can bootstrap a brand-new project directory.
    let root = scope::find_project_root(&cwd).unwrap_or(cwd);
    let marker = root.join(".mimir");

    let mut m = match scope::read_marker_state(&root) {
        MarkerState::Absent => MimirMarker::default(),
        MarkerState::Parsed(m) => m,
        MarkerState::Unparseable(why) => bail!(
            "{} exists but cannot be read as a project marker:\n{}\n\
             Nothing was changed. A marker holds `id = \"<text>\"` and `sync = true|false`.\n\
             If this file had the project's id in it, repair or restore it (a new id would \
             cut the project off from the memories synced under the old one). Otherwise \
             move it aside and run `mimir project init` again.",
            marker.display(),
            why.trim_end()
        ),
    };
    let had_id = m.id.is_some();
    m.id.get_or_insert_with(scope::new_id);
    match sync {
        SyncChange::Keep => {}
        SyncChange::Enable => m.sync = true,
        SyncChange::Disable => m.sync = false,
    }

    // The store before the marker: if it cannot be opened, the directory is
    // left as it was instead of marked with an identity that was never
    // recorded.
    let mimir = Mimir::open()?;

    let mut out = String::new();
    if let Some(id) = &m.id {
        out.push_str(&format!("id = \"{id}\"\n"));
    }
    out.push_str(&format!("sync = {}\n", m.sync));
    std::fs::write(&marker, out)?;

    // Record the project's identity in the store right away. In a linked
    // worktree whose main checkout has no such id yet, that is the
    // worktree's own project (`MemoryFiling::OwnKey`); once the main
    // checkout has the same marker, it is the main checkout's.
    let here = mimir.projects_for_cwd(&root)?;

    let key = scope::portable_key(&root).unwrap_or_default();
    println!("project  {}", scope::project_name(&root));
    println!(
        "marker   {}{}",
        marker.display(),
        if had_id { "" } else { " (created)" }
    );
    println!("key      {key}");
    println!("sync     {}", m.sync);
    if m.sync {
        println!(
            "\nCommit `.mimir` so every clone shares this identity. This project's \
             memories will replicate wherever sync is enabled (config.toml `[sync]`)."
        );
    } else {
        println!("\nRe-run with --sync to replicate this project's memories.");
    }
    match (here.filing, here.checkout_kind.home()) {
        (MemoryFiling::OwnKey, Some(home)) => println!(
            "\nThis checkout is a linked worktree of {}, whose `.mimir` does not carry this \
             id: until it does, this worktree keeps a project of its own under it. Once the \
             committed `.mimir` is checked out there, the two share the main checkout's \
             project.",
            home.display()
        ),
        _ => {
            if let Some(note) = here.worktree_note() {
                println!("\nThis checkout is a {note}.");
            }
        }
    }
    Ok(())
}

/// How many missing projects the listing spells out before it summarises.
const PRUNE_LISTED: usize = 15;

/// One line of the listing: what is stored for a project that is gone.
fn prune_line(m: &mimir_graph::MissingProject) -> String {
    let mut line = format!(
        "  {:<24} {:>9} symbols {:>7} files  {}",
        m.project.title.as_deref().unwrap_or("(untitled)"),
        m.symbols,
        m.files,
        m.project.path.as_deref().unwrap_or("")
    );
    if m.memories > 0 {
        line.push_str(&format!(
            "  ({} memor{} kept)",
            m.memories,
            if m.memories == 1 { "y" } else { "ies" }
        ));
    }
    line
}

/// List the projects whose directory no longer exists and, with `apply`,
/// remove their code graphs. The default is the listing alone: "missing" is
/// a filesystem check, and a project on an unmounted drive passes it too.
pub fn prune(apply: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let missing: Vec<_> = mimir_graph::missing_projects(&mimir.conn)?
        .into_iter()
        .filter(|m| m.has_graph_rows())
        .collect();
    if missing.is_empty() {
        println!("no project has a code graph left behind by a directory that is gone");
        return Ok(());
    }
    let symbols: i64 = missing.iter().map(|m| m.symbols).sum();
    let files: i64 = missing.iter().map(|m| m.files).sum();
    println!(
        "{} project(s) point at a directory that no longer exists; they hold {symbols} \
         symbols and {files} code files:",
        missing.len()
    );
    for m in missing.iter().take(PRUNE_LISTED) {
        println!("{}", prune_line(m));
    }
    if missing.len() > PRUNE_LISTED {
        println!("  … and {} smaller ones", missing.len() - PRUNE_LISTED);
    }
    if !apply {
        println!(
            "nothing was changed. `mimir project prune --apply` removes these code graphs \
             (`mimir graph build` recreates one if its directory comes back); memories are \
             never removed"
        );
        return Ok(());
    }

    // Each removed row is checked against the rows that could still point
    // at it; without the lookup indexes that check reads the whole store.
    let added = mimir_core::db::ensure_indexes(&mimir.conn)?;
    if added > 0 {
        println!(
            "added {added} lookup index(es) first, so each removal is a lookup and not a scan"
        );
    }
    let mut total = mimir_graph::Pruned::default();
    let mut removed_projects = 0usize;
    for m in &missing {
        // A linked worktree's links move to the same code at home.
        let home = mimir_core::store::worktree_home(&mimir.conn, &m.project)?;
        let pruned = mimir_graph::prune_graph_toward(&mimir.conn, &m.project, home.as_ref())?;
        total.links_moved += pruned.links_moved;
        total.symbols_removed += pruned.symbols_removed;
        total.files_removed += pruned.files_removed;
        total.symbols_kept += pruned.symbols_kept;
        total.files_kept += pruned.files_kept;
        removed_projects += usize::from(pruned.project_removed);
    }
    println!(
        "removed {} symbols and {} code files from {} project(s); {} project row(s) removed, \
         {} kept because memories or links still refer to them",
        total.symbols_removed,
        total.files_removed,
        missing.len(),
        removed_projects,
        missing.len() - removed_projects
    );
    if total.links_moved > 0 {
        println!(
            "{} link(s) from memories to a removed worktree's code now point at the same code \
             in its main checkout",
            total.links_moved
        );
    }
    if total.symbols_kept + total.files_kept > 0 {
        println!(
            "{} symbol(s) and {} code file(s) were kept as tombstones because a memory is \
             linked to them, so those memories read as stale instead of losing their link",
            total.symbols_kept, total.files_kept
        );
    }
    println!("{}", freed_space_line(&mimir));
    Ok(())
}

/// What a prune says about the space it freed. SQLite keeps freed pages in
/// the file and reuses them, so the file does not get smaller by itself.
/// The line says how much of it is free and gives the command that hands it
/// back: "it only shrinks after a VACUUM" told the reader that something
/// could be done and not how.
fn freed_space_line(mimir: &Mimir) -> String {
    // Below a tenth of a gigabyte there is nothing worth a VACUUM.
    freed_space_line_from(mimir, 100_000_000)
}

fn freed_space_line_from(mimir: &Mimir, worth_saying: u64) -> String {
    let pragma = |name: &str| -> Option<u64> {
        mimir
            .conn
            .query_row(&format!("PRAGMA {name}"), [], |r| r.get::<_, i64>(0))
            .ok()
            .and_then(|n| u64::try_from(n).ok())
    };
    let gb = |bytes: u64| format!("{:.1} GB", bytes as f64 / 1e9);
    match (
        pragma("freelist_count"),
        pragma("page_count"),
        pragma("page_size"),
    ) {
        (Some(free), Some(pages), Some(size)) if free * size >= worth_saying => format!(
            "{} of the store file's {} is free now and will be reused. To shrink the file \
             itself, run this while no agent session is writing (others wait on it, and it \
             needs that much free disk): sqlite3 '{}' 'VACUUM;'",
            gb(free * size),
            gb(pages * size),
            mimir.paths.db_file.display()
        ),
        _ => "the freed space is reused by the store".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::SyncChange;

    /// The line a prune ends on names the command that gives the space back
    /// once there is space worth giving back, and stays one short sentence
    /// when there is not.
    #[test]
    fn the_freed_space_line_gives_the_command_when_there_is_space_to_free() {
        let dir = tempfile::tempdir().unwrap();
        let mimir =
            mimir_core::Mimir::open_at(mimir_core::config::Paths::under_root(dir.path())).unwrap();
        mimir
            .conn
            .execute_batch(
                "CREATE TABLE filler (x); \
                 WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 200) \
                 INSERT INTO filler SELECT randomblob(4000) FROM n; \
                 DROP TABLE filler;",
            )
            .unwrap();
        assert_eq!(
            super::freed_space_line(&mimir),
            "the freed space is reused by the store"
        );
        let line = super::freed_space_line_from(&mimir, 1);
        assert!(line.contains("is free now and will be reused"), "{line}");
        assert!(
            line.contains(&format!(
                "sqlite3 '{}' 'VACUUM;'",
                mimir.paths.db_file.display()
            )),
            "{line}"
        );
    }

    #[test]
    fn sync_change_from_flags() {
        assert_eq!(SyncChange::from_flags(false, false), SyncChange::Keep);
        assert_eq!(SyncChange::from_flags(true, false), SyncChange::Enable);
        assert_eq!(SyncChange::from_flags(false, true), SyncChange::Disable);
    }
}
