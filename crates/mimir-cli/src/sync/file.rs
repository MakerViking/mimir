//! File transport: the zero-server sync path.
//!
//! Each machine writes a full snapshot of its synced memories to
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
    /// Peer files that could not be read or parsed, and were left out.
    pub peers_skipped: usize,
    pub applied: ApplyStats,
}

/// What merging a directory of peer snapshots did.
#[derive(Default)]
pub struct Merged {
    /// Peer files read and applied.
    pub peers: usize,
    /// Peer files that could not be read or parsed, and were left out.
    pub skipped: usize,
    pub applied: ApplyStats,
}

/// Write this machine's snapshot, then merge all peers' snapshots.
pub fn sync(mimir: &mut Mimir, dir: &Path) -> Result<FileSummary> {
    let wrote = write_snapshot(mimir, dir)?;
    let merged = merge_peers(mimir, dir)?;
    // Embed any newly-arrived memories locally (no-op without a model).
    let _ = mimir.embed_pending();
    Ok(FileSummary {
        wrote,
        peers_merged: merged.peers,
        peers_skipped: merged.skipped,
        applied: merged.applied,
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
///
/// Each peer file is its own transaction, and the files are independent:
/// one that cannot be read or parsed (a peer's half-synced upload, a
/// conflict copy the file-sync tool truncated) is skipped with a warning and
/// counted, and the rest still merge. Failing the whole merge on it left the
/// peers read before it applied and the peers after it not, by directory
/// order. A failure to apply is a problem with this store, not with a peer
/// file, and still ends the merge.
pub fn merge_peers(mimir: &mut Mimir, dir: &Path) -> Result<Merged> {
    let id = replicate::machine_id(&mimir.conn)?;
    let mine = format!("{id}.json");
    let mut merged = Merged::default();
    if !dir.is_dir() {
        return Ok(merged);
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".json") || name == mine || name.starts_with('.') {
            continue;
        }
        let batch = match read_peer(&entry.path()) {
            Ok(batch) => batch,
            Err(e) => {
                tracing::warn!("sync: skipped an unreadable peer file: {e:#}");
                merged.skipped += 1;
                continue;
            }
        };
        let s = replicate::apply_changes(&mimir.conn, &batch)?;
        merged.applied.nodes_upserted += s.nodes_upserted;
        merged.applied.nodes_skipped += s.nodes_skipped;
        merged.applied.edges_upserted += s.edges_upserted;
        merged.applied.edges_skipped += s.edges_skipped;
        merged.applied.tombstones += s.tombstones;
        merged.peers += 1;
    }
    Ok(merged)
}

fn read_peer(path: &Path) -> Result<replicate::SyncBatch> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parse {}", path.display()))
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
                actor: mimir_core::model::Actor::System,
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
        mimir_core::store::soft_delete(&a.conn, id, mimir_core::model::Actor::System, None)
            .unwrap();
        sync(&mut a, &shared).unwrap();
        sync(&mut b, &shared).unwrap();
        assert_eq!(live_count(&b), 0, "delete propagated");
    }

    /// One peer file that cannot be parsed must not decide what happens to
    /// the others. It is skipped and counted; every readable peer merges,
    /// whichever side of it the directory listing puts them on.
    #[test]
    fn an_unreadable_peer_file_is_skipped_and_the_rest_merge() {
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared");
        let mut a = store(&tmp.path().join("a"));
        let mut b = store(&tmp.path().join("b"));
        let mut c = store(&tmp.path().join("c"));

        glob(&a, "fact from machine a");
        glob(&b, "fact from machine b");
        sync(&mut a, &shared).unwrap();
        sync(&mut b, &shared).unwrap();
        // Names that sort before and after any ULID-named snapshot.
        std::fs::write(shared.join("0-truncated.json"), b"{\"nodes\": [").unwrap();
        std::fs::write(shared.join("zz-not-a-batch.json"), b"[1, 2, 3]").unwrap();

        let s = sync(&mut c, &shared).unwrap();
        assert_eq!(s.peers_merged, 2);
        assert_eq!(s.peers_skipped, 2);
        assert_eq!(s.applied.nodes_upserted, 2);
        assert_eq!(live_count(&c), 2, "both readable peers merged");

        let clean = merge_peers(&mut a, &tmp.path().join("nowhere")).unwrap();
        assert_eq!((clean.peers, clean.skipped), (0, 0));
    }
}
