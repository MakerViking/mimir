//! File transport: the zero-server sync path.
//!
//! Each machine writes a full snapshot of its global memories to
//! `dir/<machine-id>.json` and merges every other `*.json` it finds there.
//! The directory is one the user's file-sync tool (Syncthing/Dropbox/git/…)
//! replicates between machines. Full snapshots are convergent and idempotent,
//! so no per-peer watermark or clock comparison is needed at the transport
//! level — last-write-wins (by the records' own timestamps) governs merges.

use std::path::Path;

use anyhow::{Context, Result};
use mimir_core::replicate::{self, ApplyStats};
use mimir_core::Mimir;

pub struct FileSummary {
    pub wrote: usize,
    pub peers_merged: usize,
    pub applied: ApplyStats,
}

/// Write this machine's snapshot, then merge all peers' snapshots.
pub fn sync(mimir: &mut Mimir, dir: &Path) -> Result<FileSummary> {
    let wrote = write_snapshot(mimir, dir)?;
    let (peers_merged, applied) = merge_peers(mimir, dir)?;
    // Embed any newly-arrived memories locally (no-op without a model).
    let _ = mimir.embed_pending();
    Ok(FileSummary {
        wrote,
        peers_merged,
        applied,
    })
}

/// Write `dir/<machine-id>.json` via a temp file + rename so a peer never
/// reads a half-written snapshot.
pub fn write_snapshot(mimir: &Mimir, dir: &Path) -> Result<usize> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let id = replicate::machine_id(&mimir.conn)?;
    let batch = replicate::snapshot(&mimir.conn)?;
    let count = batch.nodes.len();
    let final_path = dir.join(format!("{id}.json"));
    let tmp_path = dir.join(format!(".{id}.json.tmp"));
    std::fs::write(&tmp_path, serde_json::to_vec(&batch)?)
        .with_context(|| format!("write {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("rename into {}", final_path.display()))?;
    Ok(count)
}

/// Merge every `*.json` in `dir` except this machine's own and temp files.
pub fn merge_peers(mimir: &mut Mimir, dir: &Path) -> Result<(usize, ApplyStats)> {
    let id = replicate::machine_id(&mimir.conn)?;
    let mine = format!("{id}.json");
    let mut total = ApplyStats::default();
    let mut peers = 0;
    if !dir.is_dir() {
        return Ok((0, total));
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".json") || name == mine || name.starts_with('.') {
            continue;
        }
        let data = std::fs::read(entry.path())?;
        let batch: replicate::SyncBatch = serde_json::from_slice(&data)
            .with_context(|| format!("parse {}", entry.path().display()))?;
        let s = replicate::apply_changes(&mimir.conn, &batch)?;
        total.nodes_upserted += s.nodes_upserted;
        total.nodes_skipped += s.nodes_skipped;
        total.edges_upserted += s.edges_upserted;
        total.edges_skipped += s.edges_skipped;
        total.tombstones += s.tombstones;
        peers += 1;
    }
    Ok((peers, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::config::Paths;
    use mimir_core::memory::{remember, Remember};
    use mimir_core::model::MemoryType;

    fn store(root: &std::path::Path) -> Mimir {
        std::fs::create_dir_all(root).unwrap();
        Mimir::open_at(Paths::under_root(root)).unwrap()
    }
    fn glob(m: &Mimir, text: &str) {
        remember(
            &m.conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
            },
        )
        .unwrap();
    }
    fn live_count(m: &Mimir) -> i64 {
        m.conn
            .query_row(
                "SELECT count(*) FROM node WHERE kind='memory' AND project_id IS NULL AND deleted_at IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap()
    }

    #[test]
    fn two_stores_converge_via_shared_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared");
        let mut a = store(&tmp.path().join("a"));
        let mut b = store(&tmp.path().join("b"));

        glob(&a, "fact from machine a");
        sync(&mut a, &shared).unwrap();
        let s = sync(&mut b, &shared).unwrap();
        assert_eq!(s.applied.nodes_upserted, 1);
        assert_eq!(live_count(&b), 1, "b received a's memory");

        // Delete on a propagates to b.
        let id: i64 = a
            .conn
            .query_row("SELECT id FROM node WHERE kind='memory'", [], |r| r.get(0))
            .unwrap();
        mimir_core::store::soft_delete(&a.conn, id).unwrap();
        sync(&mut a, &shared).unwrap();
        sync(&mut b, &shared).unwrap();
        assert_eq!(live_count(&b), 0, "delete propagated");
    }
}
