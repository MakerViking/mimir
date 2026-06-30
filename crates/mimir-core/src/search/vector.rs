//! Vector leg: brute-force dot product over an in-process matrix cache.
//!
//! No vector extension — embeddings are L2-normalized f32 blobs, so
//! cosine == dot, and exact scan of ≤200k vectors takes single-digit ms.
//!
//! Scope/kind/since filters are evaluated against per-row metadata cached
//! with the matrix — no per-query candidate-set table scan.
//!
//! Invalidation contract: other connections' commits are caught via
//! `PRAGMA data_version`. Same-connection embedding mutations do NOT
//! auto-invalidate (deliberately — recall's own event logging must not
//! force a rebuild every call); the only same-connection mutator is
//! `embed_pending`, and `Mimir` drops its cache after calling it.

use rusqlite::Connection;

use crate::error::Result;
use crate::model::Scope;
use crate::search::SearchQuery;

/// Row count above which the dot-product scan fans out across cores.
const PARALLEL_SCAN_THRESHOLD: usize = 8_192;

pub struct MatrixCache {
    model: String,
    dim: usize,
    node_ids: Vec<i64>,
    data: Vec<f32>,
    // Per-row filter metadata, parallel to node_ids. Kinds are interned:
    // kind_of[i] indexes into kind_names (a handful of distinct strings).
    project_ids: Vec<Option<i64>>,
    kind_of: Vec<u8>,
    kind_names: Vec<String>,
    created_at: Vec<i64>,
    stamp: i64,
}

impl MatrixCache {
    /// Load (or refresh) the matrix for `model` if another connection
    /// committed since the last load.
    pub fn ensure(conn: &Connection, model: &str, cache: &mut Option<MatrixCache>) -> Result<()> {
        let stamp = crate::db::data_version(conn)?;
        if let Some(c) = cache {
            if c.model == model && c.stamp == stamp {
                return Ok(());
            }
        }
        let mut stmt = conn.prepare(
            "SELECT e.node_id, e.dim, e.vec, n.project_id, n.kind, n.created_at
             FROM embedding e
             JOIN node n ON n.id = e.node_id
             WHERE e.model = ?1 AND n.deleted_at IS NULL",
        )?;
        let mut node_ids = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        let mut project_ids = Vec::new();
        let mut kind_of = Vec::new();
        let mut kind_names: Vec<String> = Vec::new();
        let mut created_at = Vec::new();
        let mut dim = 0usize;
        let mut rows = stmt.query([model])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let row_dim: i64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let vec = crate::embed::from_blob(&blob);
            if dim == 0 {
                dim = row_dim as usize;
            }
            if vec.len() != dim {
                tracing::warn!(node = id, "embedding dim mismatch; skipping");
                continue;
            }
            let kind: String = row.get(4)?;
            let kind_idx = match kind_names.iter().position(|k| *k == kind) {
                Some(i) => i,
                None => {
                    kind_names.push(kind);
                    kind_names.len() - 1
                }
            };
            node_ids.push(id);
            data.extend_from_slice(&vec);
            project_ids.push(row.get(3)?);
            kind_of.push(kind_idx as u8);
            created_at.push(row.get(5)?);
        }
        *cache = Some(MatrixCache {
            model: model.to_string(),
            dim,
            node_ids,
            data,
            project_ids,
            kind_of,
            kind_names,
            created_at,
            stamp,
        });
        Ok(())
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
    MatrixCache::ensure(conn, model, cache)?;
    let Some(c) = cache.as_ref() else {
        return Ok(vec![]);
    };
    if c.node_ids.is_empty() || c.dim != query_vec.len() {
        return Ok(vec![]);
    }
    // Requested kinds → interned indices in this cache. A requested kind
    // absent from kind_names matches nothing, which is correct.
    let kind_filter: Option<Vec<u8>> = if query.kinds.is_empty() {
        None
    } else {
        Some(
            query
                .kinds
                .iter()
                .filter_map(|k| {
                    let name = k.to_string();
                    c.kind_names
                        .iter()
                        .position(|n| *n == name)
                        .map(|i| i as u8)
                })
                .collect(),
        )
    };
    let score_row = |(i, id): (usize, &i64)| -> Option<(i64, f32)> {
        let scope_ok = match query.scope {
            Scope::All => true,
            Scope::Global => c.project_ids[i].is_none(),
            Scope::Project(pid) => c.project_ids[i].is_none() || c.project_ids[i] == Some(pid),
        };
        if !scope_ok {
            return None;
        }
        if let Some(kinds) = &kind_filter {
            if !kinds.contains(&c.kind_of[i]) {
                return None;
            }
        }
        if let Some(since) = query.since {
            if c.created_at[i] < since {
                return None;
            }
        }
        let row = &c.data[i * c.dim..(i + 1) * c.dim];
        let dot: f32 = row.iter().zip(query_vec).map(|(a, b)| a * b).sum();
        Some((*id, dot))
    };
    // Parallel scan pays for itself only on big matrices; below the
    // threshold rayon's fork/join overhead exceeds the work.
    let mut scored: Vec<(i64, f32)> = if c.node_ids.len() >= PARALLEL_SCAN_THRESHOLD {
        use rayon::prelude::*;
        c.node_ids
            .par_iter()
            .enumerate()
            .filter_map(score_row)
            .collect()
    } else {
        c.node_ids
            .iter()
            .enumerate()
            .filter_map(score_row)
            .collect()
    };
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
        let mut new = NewNode::new(Kind::Memory);
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
            include_superseded: false,
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
        store::soft_delete(&conn, a).unwrap();
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
}
