//! Vector leg: brute-force dot product over an in-process matrix cache.
//!
//! No vector extension — embeddings are L2-normalized f32 blobs, so
//! cosine == dot, and exact scan of ≤200k vectors takes single-digit ms.
//!
//! Scope/kind/since filters are evaluated against per-row metadata cached
//! with the matrix — no per-query candidate-set table scan.
//!
//! # One matrix per kind
//!
//! The cache holds one matrix per node kind, and a search loads only the
//! kinds it asks for, the first time it asks. On a store holding the code
//! graphs of many projects, code symbols outnumber every other embedded row
//! a thousand to one (1.97M symbol vectors, 2.9 GB, next to 1,814 memory
//! vectors on the store this was measured on). The prompt hook's recall asks
//! for memories only, and must not read the symbols: when the cache was one
//! matrix of everything, the warm daemon reread all 2.9 GB after any other
//! process's commit, 13 to 99 s per prompt.
//!
//! Invalidation contract: other connections' commits are caught via
//! `PRAGMA data_version`, and drop every loaded kind at once. Same-connection
//! embedding mutations do NOT auto-invalidate (deliberately — recall's own
//! event logging must not force a rebuild every call); the only
//! same-connection mutator is `embed_pending`, and `Mimir::embed_pending`
//! repairs the cache itself afterwards: [`MatrixCache::upsert`] patches the
//! embedded rows in for a small batch, and a bulk batch drops the cache for
//! the next search to rebuild.

use rusqlite::Connection;

use crate::error::Result;
use crate::model::{Kind, Scope};
use crate::search::{Reach, SearchQuery};

/// Row count above which the dot-product scan fans out across cores.
const PARALLEL_SCAN_THRESHOLD: usize = 8_192;

pub struct MatrixCache {
    model: String,
    /// `PRAGMA data_version` when the first of `kinds` was loaded. Every
    /// matrix in `kinds` is at least that fresh: a later one was loaded
    /// while the version still read the same.
    stamp: i64,
    /// The kinds loaded so far, each with its matrix. A kind with no
    /// embedded rows has an empty one, so it is not looked up again.
    kinds: Vec<(Kind, Matrix)>,
}

/// The embedded rows of one kind. Per-row filter metadata is parallel to
/// `node_ids`.
#[derive(Default)]
struct Matrix {
    /// The vector width, fixed by the first row admitted (0 = none yet).
    dim: usize,
    node_ids: Vec<i64>,
    data: Vec<f32>,
    project_ids: Vec<Option<i64>>,
    created_at: Vec<i64>,
}

/// Whether a stored embedding row may join a matrix, fixing the matrix
/// width from the first row offered (`*dim == 0` means none has been yet).
///
/// A row fits when the dimension it declares and the length of the vector
/// it actually holds both equal that width. The full load and the in-place
/// patch ask this one question, so a row is in the matrix or out of it
/// whichever of them saw it.
fn admits_row(dim: &mut usize, row_dim: i64, vec: &[f32]) -> bool {
    if *dim == 0 {
        *dim = row_dim as usize;
    }
    row_dim as usize == *dim && vec.len() == *dim
}

/// The kinds a query's vector leg reads: the ones it names, or every kind.
fn kinds_of(query: &SearchQuery) -> Vec<Kind> {
    if query.kinds.is_empty() {
        Kind::ALL.to_vec()
    } else {
        query.kinds.clone()
    }
}

impl Matrix {
    /// The embedded, live rows of `kind` under `model`. Read from the node
    /// side (`node_kind_scope`), so loading one kind does not visit the
    /// rows of the others.
    fn load(conn: &Connection, model: &str, kind: Kind) -> Result<Matrix> {
        let mut stmt = conn.prepare_cached(
            "SELECT e.node_id, e.dim, e.vec, n.project_id, n.created_at
             FROM node n CROSS JOIN embedding e ON e.node_id = n.id
             WHERE n.kind = ?2 AND n.deleted_at IS NULL AND e.model = ?1",
        )?;
        let mut matrix = Matrix::default();
        let mut rows = stmt.query(rusqlite::params![model, kind.as_str()])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let row_dim: i64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let vec = crate::embed::from_blob(&blob);
            if !admits_row(&mut matrix.dim, row_dim, &vec) {
                tracing::warn!(node = id, "embedding dim mismatch; skipping");
                continue;
            }
            matrix.node_ids.push(id);
            matrix.data.extend_from_slice(&vec);
            matrix.project_ids.push(row.get(3)?);
            matrix.created_at.push(row.get(4)?);
        }
        Ok(matrix)
    }

    /// Put one row in, replacing the row of the same node if there is one.
    fn patch(&mut self, id: i64, vec: &[f32], project_id: Option<i64>, created: i64) {
        // Linear scan: a patch is small (single-digit in the common
        // `remember`-one-node case), so this beats building a hash index.
        match self.node_ids.iter().position(|&x| x == id) {
            Some(i) => {
                self.data[i * self.dim..(i + 1) * self.dim].copy_from_slice(vec);
                self.project_ids[i] = project_id;
                self.created_at[i] = created;
            }
            None => {
                self.node_ids.push(id);
                self.data.extend_from_slice(vec);
                self.project_ids.push(project_id);
                self.created_at.push(created);
            }
        }
    }
}

impl MatrixCache {
    /// Load (or refresh) the matrix of every kind for `model`: what
    /// [`MatrixCache::ensure_kinds`] does for a search over all kinds.
    pub fn ensure(conn: &Connection, model: &str, cache: &mut Option<MatrixCache>) -> Result<()> {
        Self::ensure_kinds(conn, model, cache, Kind::ALL)
    }

    /// Make sure the matrix of each of `kinds` for `model` is loaded and
    /// current: when another connection committed since the cache was
    /// built, or it was built for another model, every loaded kind is
    /// dropped first. A kind already loaded is not read again.
    pub fn ensure_kinds(
        conn: &Connection,
        model: &str,
        cache: &mut Option<MatrixCache>,
        kinds: &[Kind],
    ) -> Result<()> {
        let stamp = crate::db::data_version(conn)?;
        let current = matches!(cache, Some(c) if c.model == model && c.stamp == stamp);
        if !current {
            *cache = Some(MatrixCache {
                model: model.to_string(),
                stamp,
                kinds: Vec::new(),
            });
        }
        let Some(c) = cache.as_mut() else {
            return Ok(());
        };
        for &kind in kinds {
            if !c.kinds.iter().any(|(k, _)| *k == kind) {
                let matrix = Matrix::load(conn, model, kind)?;
                c.kinds.push((kind, matrix));
            }
        }
        Ok(())
    }

    /// Patch specific rows into an already-loaded cache instead of a full
    /// reload. `embed_pending` (e.g. one `remember`) touches a handful of
    /// nodes; without this, the resident MCP server's cache would otherwise
    /// pay a reload (hundreds of ms at real size) on the very next recall,
    /// since same-connection writes don't bump `data_version` and so can't
    /// be caught by `ensure`'s normal invalidation check.
    ///
    /// A row whose kind is not loaded yet is left out: the load that first
    /// asks for that kind reads it from the store.
    ///
    /// No-op (never an error) when there's no cache to patch or its model
    /// doesn't match `model` — the next `ensure` call does a correct full
    /// rebuild in that case, same as before this existed.
    pub fn upsert(
        conn: &Connection,
        model: &str,
        ids: &[i64],
        cache: &mut Option<MatrixCache>,
    ) -> Result<()> {
        let Some(c) = cache.as_mut() else {
            return Ok(());
        };
        if c.model != model || ids.is_empty() {
            return Ok(());
        }
        let id_list = ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
        let mut stmt = conn.prepare(&format!(
            "SELECT e.node_id, e.dim, e.vec, n.project_id, n.kind, n.created_at
             FROM embedding e JOIN node n ON n.id = e.node_id
             WHERE e.model = ?1 AND n.deleted_at IS NULL AND e.node_id IN ({id_list})"
        ))?;
        let mut rows = stmt.query([model])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let kind: String = row.get(4)?;
            let Some((_, matrix)) = c.kinds.iter_mut().find(|(k, _)| k.as_str() == kind) else {
                continue;
            };
            let row_dim: i64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let vec = crate::embed::from_blob(&blob);
            if !admits_row(&mut matrix.dim, row_dim, &vec) {
                tracing::warn!(node = id, "embedding dim mismatch on cache patch; skipping");
                continue;
            }
            matrix.patch(id, &vec, row.get(3)?, row.get(5)?);
        }
        Ok(())
    }

    /// Every row the cache holds, in scan order.
    #[cfg(test)]
    fn node_ids(&self) -> Vec<i64> {
        self.kinds
            .iter()
            .flat_map(|(_, m)| m.node_ids.iter().copied())
            .collect()
    }

    /// The kinds loaded so far, in load order.
    #[cfg(test)]
    fn loaded_kinds(&self) -> Vec<Kind> {
        self.kinds.iter().map(|(k, _)| *k).collect()
    }
}

/// Ranked node ids (best first) by dot product against `query_vec`,
/// restricted to nodes passing the query's scope/kind/since filters.
pub fn leg(
    conn: &Connection,
    cache: &mut Option<MatrixCache>,
    model: &str,
    query_vec: &[f32],
    query: &SearchQuery,
    cap: usize,
) -> Result<Vec<i64>> {
    leg_within(
        conn,
        cache,
        model,
        query_vec,
        query,
        Reach::uniform(query.scope),
        cap,
    )
}

/// [`leg`] over a [`Reach`], which is read instead of `query.scope`: each
/// cached row is admitted by the scope its kind answers to.
pub fn leg_within(
    conn: &Connection,
    cache: &mut Option<MatrixCache>,
    model: &str,
    query_vec: &[f32],
    query: &SearchQuery,
    reach: Reach,
    cap: usize,
) -> Result<Vec<i64>> {
    let kinds = kinds_of(query);
    MatrixCache::ensure_kinds(conn, model, cache, &kinds)?;
    let Some(c) = cache.as_ref() else {
        return Ok(vec![]);
    };
    let mut scored: Vec<(i64, f32)> = Vec::new();
    for (kind, m) in c.kinds.iter().filter(|(k, _)| kinds.contains(k)) {
        if m.node_ids.is_empty() || m.dim != query_vec.len() {
            continue;
        }
        let scope: Scope = reach.scope_of_kind_named(kind.as_str());
        let score_row = |(i, id): (usize, &i64)| -> Option<(i64, f32)> {
            if !scope.admits(m.project_ids[i]) {
                return None;
            }
            if let Some(since) = query.since {
                if m.created_at[i] < since {
                    return None;
                }
            }
            let row = &m.data[i * m.dim..(i + 1) * m.dim];
            let dot: f32 = row.iter().zip(query_vec).map(|(a, b)| a * b).sum();
            Some((*id, dot))
        };
        // Parallel scan pays for itself only on big matrices; below the
        // threshold rayon's fork/join overhead exceeds the work.
        if m.node_ids.len() >= PARALLEL_SCAN_THRESHOLD {
            use rayon::prelude::*;
            let part: Vec<(i64, f32)> = m
                .node_ids
                .par_iter()
                .enumerate()
                .filter_map(score_row)
                .collect();
            scored.extend(part);
        } else {
            scored.extend(m.node_ids.iter().enumerate().filter_map(score_row));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(cap);
    Ok(scored.into_iter().map(|(id, _)| id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::embed::to_blob;
    use crate::model::{Kind, NewNode};
    use crate::store;

    const MODEL: &str = "test-model";

    fn add_with_vec(conn: &Connection, title: &str, vec: &[f32]) -> i64 {
        add_kind_with_vec(conn, Kind::Memory, title, vec)
    }

    fn add_kind_with_vec(conn: &Connection, kind: Kind, title: &str, vec: &[f32]) -> i64 {
        let mut new = NewNode::new(kind);
        new.title = Some(title.into());
        new.body = Some(title.into());
        let node = store::insert_node(conn, new).unwrap();
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
             VALUES (?1, ?2, ?3, ?4, x'00')",
            rusqlite::params![node.id, MODEL, vec.len() as i64, to_blob(vec)],
        )
        .unwrap();
        node.id
    }

    fn q() -> SearchQuery {
        SearchQuery {
            text: String::new(),
            scope: Scope::All,
            kinds: vec![],
            since: None,
            limit: 10,
            strength_alpha: 0.0,
            recency_alpha: 0.0,
            type_prior_alpha: 0.0,
            code_damp: 1.0,
            include_superseded: false,
            as_of: None,
        }
    }

    #[test]
    fn ranks_by_dot_product() {
        let conn = db::open_in_memory().unwrap();
        let near = add_with_vec(&conn, "near", &[1.0, 0.0, 0.0]);
        let far = add_with_vec(&conn, "far", &[0.0, 1.0, 0.0]);
        let mid = add_with_vec(&conn, "mid", &[0.7, 0.7, 0.0]);

        let mut cache = None;
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0, 0.0], &q(), 10).unwrap();
        assert_eq!(ids, vec![near, mid, far]);
    }

    #[test]
    fn respects_filters_and_deletions() {
        let conn = db::open_in_memory().unwrap();
        let a = add_with_vec(&conn, "a", &[1.0, 0.0]);
        let b = add_with_vec(&conn, "b", &[0.9, 0.1]);

        let mut cache = None;
        let mut query = q();
        query.kinds = vec![Kind::Chunk]; // no chunks exist
        assert!(leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10)
            .unwrap()
            .is_empty());

        // Fresh cache after a delete excludes the deleted node. (With a
        // stale cache the leg may still return it; fusion filters those.)
        store::soft_delete(&conn, a, crate::model::Actor::System, None).unwrap();
        cache = None;
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10).unwrap();
        assert_eq!(ids, vec![b]);
    }

    #[test]
    fn scope_and_since_filter_from_cached_metadata() {
        let conn = db::open_in_memory().unwrap();
        let proj = store::insert_node(&conn, NewNode::new(Kind::Project)).unwrap();
        let other = store::insert_node(&conn, NewNode::new(Kind::Project)).unwrap();
        let global = add_with_vec(&conn, "global", &[1.0, 0.0]);
        let scoped = add_with_vec(&conn, "scoped", &[0.9, 0.1]);
        conn.execute(
            "UPDATE node SET project_id = ?2 WHERE id = ?1",
            [scoped, proj.id],
        )
        .unwrap();

        let mut cache = None;
        let mut query = q();
        query.scope = Scope::Global;
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10).unwrap();
        assert_eq!(ids, vec![global], "global scope excludes project rows");

        query.scope = Scope::Project(proj.id);
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10).unwrap();
        assert_eq!(
            ids,
            vec![global, scoped],
            "project scope = project + global"
        );

        query.scope = Scope::Project(other.id);
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10).unwrap();
        assert_eq!(ids, vec![global], "other project sees only global");

        let mut query = q();
        query.since = Some(i64::MAX);
        assert!(leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn same_connection_writes_need_explicit_invalidation() {
        let conn = db::open_in_memory().unwrap();
        let a = add_with_vec(&conn, "a", &[1.0, 0.0]);
        let mut cache = None;
        assert_eq!(
            leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10).unwrap(),
            vec![a]
        );
        // Same-connection write: cache deliberately stays stale…
        let b = add_with_vec(&conn, "b", &[1.0, 0.0]);
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10).unwrap();
        assert_eq!(ids, vec![a], "no auto-reload on own writes");
        // …until dropped (what Mimir::embed_pending does).
        cache = None;
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10).unwrap();
        assert!(ids.contains(&b));
    }

    #[test]
    fn dim_mismatch_returns_empty() {
        let conn = db::open_in_memory().unwrap();
        add_with_vec(&conn, "a", &[1.0, 0.0, 0.0]);
        let mut cache = None;
        assert!(leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10)
            .unwrap()
            .is_empty());
    }

    /// A row whose declared `dim` disagrees with the vector it holds is not
    /// a row either path can score. The load and the patch used to answer
    /// that differently, so the same store gave a different matrix depending
    /// on whether the row arrived before or after the cache was built.
    #[test]
    fn a_row_with_a_wrong_declared_dim_is_refused_by_load_and_patch_alike() {
        let conn = db::open_in_memory().unwrap();
        let good = add_with_vec(&conn, "good", &[1.0, 0.0]);
        let add_mislabelled = |title: &str| {
            let id = add_with_vec(&conn, title, &[0.0, 1.0]);
            conn.execute("UPDATE embedding SET dim = 3 WHERE node_id = ?1", [id])
                .unwrap();
            id
        };

        let before = add_mislabelled("mislabelled before the load");
        let mut loaded = None;
        MatrixCache::ensure(&conn, MODEL, &mut loaded).unwrap();
        assert_eq!(loaded.as_ref().unwrap().node_ids(), vec![good]);

        let after = add_mislabelled("mislabelled after the load");
        MatrixCache::upsert(&conn, MODEL, &[before, after], &mut loaded).unwrap();
        assert_eq!(loaded.as_ref().unwrap().node_ids(), vec![good]);
    }

    #[test]
    fn upsert_patches_new_and_changed_rows_matching_a_full_rebuild() {
        let conn = db::open_in_memory().unwrap();
        let a = add_with_vec(&conn, "a", &[1.0, 0.0]);
        let mut cache = None;
        MatrixCache::ensure(&conn, MODEL, &mut cache).unwrap();

        // New node, added via a same-connection write `ensure` alone would
        // miss (see `same_connection_writes_need_explicit_invalidation`).
        let b = add_with_vec(&conn, "b", &[0.0, 1.0]);
        // Existing node, content changed in place (same node_id) — now a
        // partial match instead of its original orthogonal-to-query vector.
        conn.execute(
            "UPDATE embedding SET vec = ?2 WHERE node_id = ?1",
            rusqlite::params![a, to_blob(&[0.6, 0.8])],
        )
        .unwrap();

        MatrixCache::upsert(&conn, MODEL, &[a, b], &mut cache).unwrap();
        let patched = leg(&conn, &mut cache, MODEL, &[0.0, 1.0], &q(), 10).unwrap();

        // A fresh full rebuild from the same (now-committed) state must agree,
        // including tie-break order.
        let mut rebuilt = None;
        let full = leg(&conn, &mut rebuilt, MODEL, &[0.0, 1.0], &q(), 10).unwrap();
        assert_eq!(patched, full, "patched cache diverged from a full rebuild");
        assert_eq!(
            patched,
            vec![b, a],
            "b (dot=1.0) outranks patched a (dot=0.8)"
        );
    }

    /// A search for memories reads the memory vectors and no others: the
    /// prompt hook's recall must not pay for the code graph's symbols. A
    /// search over every kind then loads the rest, and finds both.
    #[test]
    fn a_search_for_memories_loads_no_other_kind() {
        let conn = db::open_in_memory().unwrap();
        let memory = add_with_vec(&conn, "memory", &[1.0, 0.0]);
        let symbol = add_kind_with_vec(&conn, Kind::Symbol, "symbol", &[0.9, 0.1]);

        let mut cache = None;
        let mut query = q();
        query.kinds = vec![Kind::Memory];
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10).unwrap();
        assert_eq!(ids, vec![memory]);
        assert_eq!(cache.as_ref().unwrap().loaded_kinds(), vec![Kind::Memory]);

        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &q(), 10).unwrap();
        assert_eq!(ids, vec![memory, symbol]);
        assert_eq!(
            cache.as_ref().unwrap().loaded_kinds().len(),
            Kind::ALL.len()
        );
    }

    /// A patch leaves out a row of a kind not loaded yet, and the load that
    /// first asks for that kind finds it in the store.
    #[test]
    fn a_patch_for_a_kind_not_loaded_waits_for_its_load() {
        let conn = db::open_in_memory().unwrap();
        let memory = add_with_vec(&conn, "memory", &[1.0, 0.0]);
        let mut cache = None;
        MatrixCache::ensure_kinds(&conn, MODEL, &mut cache, &[Kind::Memory]).unwrap();

        let symbol = add_kind_with_vec(&conn, Kind::Symbol, "symbol", &[1.0, 0.0]);
        MatrixCache::upsert(&conn, MODEL, &[symbol], &mut cache).unwrap();
        assert_eq!(cache.as_ref().unwrap().node_ids(), vec![memory]);

        let mut query = q();
        query.kinds = vec![Kind::Symbol];
        let ids = leg(&conn, &mut cache, MODEL, &[1.0, 0.0], &query, 10).unwrap();
        assert_eq!(ids, vec![symbol]);
    }

    #[test]
    fn upsert_is_noop_without_a_cache_or_on_model_mismatch() {
        let conn = db::open_in_memory().unwrap();
        let a = add_with_vec(&conn, "a", &[1.0, 0.0]);
        let mut cache = None;
        // No cache yet: no-op, doesn't fabricate one.
        MatrixCache::upsert(&conn, MODEL, &[a], &mut cache).unwrap();
        assert!(cache.is_none());

        MatrixCache::ensure(&conn, MODEL, &mut cache).unwrap();
        // Wrong model: no-op, leaves the existing cache alone.
        MatrixCache::upsert(&conn, "other-model", &[a], &mut cache).unwrap();
        assert_eq!(cache.as_ref().unwrap().model, MODEL);
    }
}
