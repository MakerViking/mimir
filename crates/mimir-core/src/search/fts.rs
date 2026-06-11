//! FTS5 BM25 leg.

use rusqlite::Connection;

use crate::error::Result;
use crate::memory::tokens;
use crate::search::{scope_sql, SearchQuery};

/// Column weights for bm25(): title, body, path, tags_text.
const BM25_WEIGHTS: &str = "4.0, 1.0, 2.0, 3.0";

/// Build a safe MATCH expression: every token quoted (no user-controlled
/// FTS syntax), OR-joined so partial matches still rank via BM25.
pub fn match_expr(text: &str) -> String {
    tokens(text)
        .iter()
        // A token of only '.'/'-'/'_' chars tokenizes to nothing — skip.
        .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Ranked node ids (best first), capped.
pub fn leg(conn: &Connection, query: &SearchQuery, cap: usize) -> Result<Vec<i64>> {
    let expr = match_expr(&query.text);
    if expr.is_empty() {
        return Ok(vec![]);
    }
    let mut sql = format!(
        "SELECT n.id FROM node_fts JOIN node n ON n.id = node_fts.rowid
         WHERE node_fts MATCH ?1 AND n.deleted_at IS NULL AND {}",
        scope_sql(query.scope)
    );
    if !query.kinds.is_empty() {
        let kinds = query
            .kinds
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", ");
        sql.push_str(&format!(" AND n.kind IN ({kinds})"));
    }
    if let Some(since) = query.since {
        sql.push_str(&format!(" AND n.created_at >= {since}"));
    }
    sql.push_str(&format!(
        " ORDER BY bm25(node_fts, {BM25_WEIGHTS}) LIMIT {cap}"
    ));
    let mut stmt = conn.prepare(&sql)?;
    let ids = stmt
        .query_map([&expr], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(ids)
}
