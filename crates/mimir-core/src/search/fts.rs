//! FTS5 BM25 leg.

use rusqlite::Connection;

use crate::error::Result;
use crate::memory::tokens;
use crate::search::{scope_sql, SearchQuery};

/// Column weights for bm25(): title, body, path, tags_text.
const BM25_WEIGHTS: &str = "4.0, 1.0, 2.0, 3.0";

/// English stopwords dropped from natural-language queries so "how do I
/// authenticate" doesn't let "how/do/i" drag in noise. Small on purpose:
/// only words that are near-meaningless in a search query.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "again", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "get", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "may", "me", "might", "more", "most", "must", "my", "no", "not", "of", "on",
    "only", "or", "our", "over", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "too", "under", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
];

fn is_stopword(token: &str) -> bool {
    STOPWORDS.binary_search(&token).is_ok()
}

/// Build a safe MATCH expression: every token quoted (no user-controlled
/// FTS syntax), OR-joined so partial matches still rank via BM25.
/// Stopwords are dropped unless the query is nothing but stopwords.
pub fn match_expr(text: &str) -> String {
    let all: Vec<String> = tokens(text)
        .into_iter()
        // A token of only '.'/'-'/'_' chars tokenizes to nothing — skip.
        .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
        .collect();
    let content: Vec<&String> = all.iter().filter(|t| !is_stopword(t)).collect();
    let keep: Vec<&String> = if content.is_empty() {
        all.iter().collect()
    } else {
        content
    };
    keep.iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopword_list_is_sorted_for_binary_search() {
        let mut sorted = STOPWORDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(STOPWORDS, &sorted[..]);
    }

    #[test]
    fn stopwords_dropped_from_queries() {
        assert_eq!(match_expr("how do I authenticate"), "\"authenticate\"");
        assert_eq!(
            match_expr("the supabase migration"),
            "\"supabase\" OR \"migration\""
        );
        // All-stopword queries fall back to the raw tokens.
        assert_eq!(match_expr("what is this"), "\"what\" OR \"is\" OR \"this\"");
        // Identifiers pass through untouched.
        assert_eq!(match_expr("resolve_ref"), "\"resolve_ref\"");
    }
}
