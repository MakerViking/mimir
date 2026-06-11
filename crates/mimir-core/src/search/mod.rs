//! Hybrid search: ranked legs fused with reciprocal-rank fusion.
//!
//! Legs: FTS5 BM25 (always) + vector dot product (when a query embedding
//! is supplied). RRF is scale-free, so adding legs never requires re-tuning.

pub mod fts;
pub mod vector;

use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::Result;
use crate::model::{Kind, Node, Scope};
use crate::store;

/// Reciprocal-rank-fusion constant (standard k=60).
const RRF_K: f64 = 60.0;

/// How many candidates each leg contributes before fusion.
const LEG_CAP: usize = 100;

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub scope: Scope,
    /// Empty = all kinds.
    pub kinds: Vec<Kind>,
    /// Unix floor on created_at.
    pub since: Option<i64>,
    pub limit: usize,
    /// Weight of learned strength in the final score (config scoring.strength_alpha).
    pub strength_alpha: f64,
}

#[derive(Debug)]
pub struct Hit {
    pub node: Node,
    pub score: f64,
}

/// SQL predicate (aliased table `n`) for a scope.
/// Project scope includes unscoped (global) nodes: cross-project knowledge
/// is meant to surface inside any project.
pub(crate) fn scope_sql(scope: Scope) -> String {
    match scope {
        Scope::All => "1=1".into(),
        Scope::Global => "n.project_id IS NULL".into(),
        Scope::Project(id) => format!("(n.project_id = {id} OR n.project_id IS NULL)"),
    }
}

/// BM25-only search (no model needed).
pub fn search(conn: &Connection, query: &SearchQuery) -> Result<Vec<Hit>> {
    search_hybrid(conn, query, None, &mut None)
}

/// Hybrid search. `vector_query` = (model name, L2-normalized query
/// embedding); pass None to skip the vector leg.
pub fn search_hybrid(
    conn: &Connection,
    query: &SearchQuery,
    vector_query: Option<(&str, &[f32])>,
    cache: &mut Option<vector::MatrixCache>,
) -> Result<Vec<Hit>> {
    let mut legs: Vec<Vec<i64>> = vec![fts::leg(conn, query, LEG_CAP)?];
    if let Some((model, qvec)) = vector_query {
        legs.push(vector::leg(conn, cache, model, qvec, query, LEG_CAP)?);
    }

    let mut rrf: HashMap<i64, f64> = HashMap::new();
    for leg in &legs {
        for (rank, id) in leg.iter().enumerate() {
            *rrf.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }

    let mut hits: Vec<Hit> = rrf
        .into_iter()
        .filter_map(|(id, base)| match store::get_node(conn, id) {
            // A stale vector cache may still hold soft-deleted nodes;
            // they must never surface.
            Ok(node) if node.deleted_at.is_some() => None,
            Ok(node) => {
                // Strength is a tiebreaker multiplier, never a burier.
                let score = base * (1.0 + query.strength_alpha * (1.0 + node.strength).ln());
                Some(Ok(Hit { node, score }))
            }
            Err(e) => Some(Err(e)),
        })
        .collect::<Result<_>>()?;
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits.truncate(query.limit);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::{Kind, NewNode};

    fn add(conn: &Connection, kind: Kind, project: Option<i64>, title: &str, body: &str) -> Node {
        let mut new = NewNode::new(kind);
        new.title = Some(title.into());
        new.body = Some(body.into());
        new.project_id = project;
        store::insert_node(conn, new).unwrap()
    }

    fn q(text: &str) -> SearchQuery {
        SearchQuery {
            text: text.into(),
            scope: Scope::All,
            kinds: vec![],
            since: None,
            limit: 10,
            strength_alpha: 0.15,
        }
    }

    #[test]
    fn finds_by_body_and_ranks_title_matches_higher() {
        let conn = db::open_in_memory().unwrap();
        add(
            &conn,
            Kind::Memory,
            None,
            "about databases",
            "sqlite pragma tuning notes",
        );
        add(
            &conn,
            Kind::Memory,
            None,
            "sqlite checklist",
            "general things to remember",
        );
        let hits = search(&conn, &q("sqlite")).unwrap();
        assert_eq!(hits.len(), 2);
        // Title hits outrank body hits (bm25 column weighting).
        assert_eq!(hits[0].node.title.as_deref(), Some("sqlite checklist"));
    }

    #[test]
    fn scope_filtering() {
        let conn = db::open_in_memory().unwrap();
        let project = store::ensure_project(&conn, "/tmp/p", "p").unwrap();
        add(
            &conn,
            Kind::Memory,
            Some(project.id),
            "scoped fact",
            "alpha beaver",
        );
        add(&conn, Kind::Memory, None, "global fact", "alpha beaver");

        let mut query = q("beaver");
        query.scope = Scope::Project(project.id);
        assert_eq!(search(&conn, &query).unwrap().len(), 2); // project + global

        query.scope = Scope::Global;
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.title.as_deref(), Some("global fact"));

        query.scope = Scope::Project(project.id + 999);
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1); // only the global one leaks across projects
    }

    #[test]
    fn kind_filtering_and_deleted_exclusion() {
        let conn = db::open_in_memory().unwrap();
        add(&conn, Kind::Memory, None, "memory hit", "zebra pattern");
        let chunk = add(&conn, Kind::Chunk, None, "doc hit", "zebra pattern");

        let mut query = q("zebra");
        query.kinds = vec![Kind::Memory];
        let hits = search(&conn, &query).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.kind, Kind::Memory);

        store::soft_delete(&conn, chunk.id).unwrap();
        let hits = search(&conn, &q("zebra")).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn weird_input_does_not_break_fts_syntax() {
        let conn = db::open_in_memory().unwrap();
        add(
            &conn,
            Kind::Memory,
            None,
            "quoted",
            "the \"weird\" AND OR NOT (input)",
        );
        for text in ["\"unbalanced", "AND OR", "a* b: (c)", "   ", "résumé café"] {
            // Must not error, whatever it returns.
            search(&conn, &q(text)).unwrap();
        }
        assert_eq!(search(&conn, &q("weird input")).unwrap().len(), 1);
    }
}
