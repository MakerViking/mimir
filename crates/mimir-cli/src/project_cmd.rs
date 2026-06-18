//! `mimir project init [--sync]` — give a project a portable, machine-stable
//! identity by writing a `.mimir` marker (a committed `id`, plus an opt-in
//! `sync` flag). With a portable key, the project's memories can replicate
//! across machines via the sync layer; without one, they stay local.

use anyhow::Result;
use mimir_core::{scope, Mimir};

/// Create or update `.mimir` in the current project root: ensure a stable `id`
/// (generated once if absent) and, with `--sync`, set `sync = true`. Idempotent:
/// re-running keeps the existing id. Prints the resulting key and sync state.
pub fn init(sync: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    // Resolve the project root (VCS / build-file / existing `.mimir`); fall back
    // to cwd so `init` can bootstrap a brand-new project directory.
    let root = scope::find_project_root(&cwd).unwrap_or(cwd);
    let marker = root.join(".mimir");

    let mut m = scope::read_mimir_marker(&root).unwrap_or_default();
    let had_id = m.id.is_some();
    m.id.get_or_insert_with(scope::new_id);
    if sync {
        m.sync = true;
    }

    let mut out = String::new();
    if let Some(id) = &m.id {
        out.push_str(&format!("id = \"{id}\"\n"));
    }
    out.push_str(&format!("sync = {}\n", m.sync));
    std::fs::write(&marker, out)?;

    // Record the project's identity in the store right away.
    let mimir = Mimir::open()?;
    let _ = mimir.project_for_cwd(&root)?;

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
    Ok(())
}
