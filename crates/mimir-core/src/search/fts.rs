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

pub fn is_stopword(token: &str) -> bool {
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
    ranked_ids(conn, query, &match_expr(&query.text), cap)
}

/// Exact-phrase FTS leg for identifier-shaped fragments of the query
/// (`Foo::bar`, `snake_case`, `camelCase`, `dotted.path`) — shapes `leg`'s
/// OR-of-every-token match dilutes. A query like "how does MatrixCache::ensure
/// work" ORs in "work" too via `match_expr`, so a doc that only mentions
/// "work" ranks in the same leg as the actual identifier hit. This leg
/// instead requires each identifier fragment's own tokens *adjacent* (an
/// FTS5 phrase), so it only ever contributes docs containing the literal
/// identifier — a pure addition to RRF fusion, never a source of noise.
/// Empty (no query run) when nothing in the text looks like an identifier,
/// so it costs nothing on ordinary prose queries.
pub fn identifier_leg(conn: &Connection, query: &SearchQuery, cap: usize) -> Result<Vec<i64>> {
    let expr = identifier_match_expr(&query.text);
    if expr.is_empty() {
        return Ok(vec![]);
    }
    ranked_ids(conn, query, &expr, cap)
}

/// Shared MATCH runner behind [`leg`] and [`identifier_leg`]: same
/// scope/kind/since filtering and BM25 ordering, different MATCH
/// expression. `expr` empty short-circuits to no query at all.
fn ranked_ids(conn: &Connection, query: &SearchQuery, expr: &str, cap: usize) -> Result<Vec<i64>> {
    if expr.is_empty() {
        return Ok(vec![]);
    }
    let mut sql = format!(
        "SELECT n.id FROM node_fts JOIN node n ON n.id = node_fts.rowid
         WHERE node_fts MATCH ?1 AND {} AND {}",
        crate::search::liveness_sql(query.as_of),
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
        .query_map([expr], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(ids)
}

/// Build a MATCH expression of quoted phrases, one per identifier-shaped
/// word in `text` — each phrase is that word's own tokens space-joined
/// inside one pair of quotes, which FTS5 reads as "these tokens adjacent",
/// not "any of these tokens". Multiple identifier words in the query
/// OR-join (a hit on any one of them counts). Empty string when `text` has
/// no identifier-shaped word.
fn identifier_match_expr(text: &str) -> String {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| {
                !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ':')
            })
        })
        .filter(|w| is_identifier_shaped(w))
        .filter_map(|w| {
            let toks = tokens(w);
            (!toks.is_empty()).then(|| format!("\"{}\"", toks.join(" ")))
        })
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// A word "looks like" a code identifier rather than an English word: a
/// `::` path, `snake_case`, a dotted path (`foo.bar`), or internal
/// capitalization (`camelCase`/`PascalCase`, checked past the first
/// character so an ordinary capitalized word like "Rust" doesn't qualify).
pub fn is_identifier_shaped(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    if word.contains("::") || word.contains('_') {
        return true;
    }
    if word.contains('.')
        && word.chars().next().is_some_and(|c| c.is_alphanumeric())
        && word.chars().last().is_some_and(|c| c.is_alphanumeric())
    {
        return true;
    }
    is_camel_case(word)
}

/// A lowercase-to-uppercase letter transition somewhere past the first
/// character — the actual defining feature of camelCase/PascalCase
/// (`matrixCache`/`MatrixCache`/`addSaturating` all have one: `x`→`C`,
/// `x`→`C`, `d`→`S`). Deliberately narrower than "mixed case anywhere":
/// that would also catch acronym-led prose words like `SQLite` or
/// `GitHub` (all-uppercase run, then lowercase — an upper-to-lower
/// transition, never a lower-to-upper one), which are ordinary nouns in a
/// query, not code identifiers, and were false-tripping this leg on
/// eval's `tuning` set before this fix.
fn is_camel_case(word: &str) -> bool {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_alphabetic() {
        return false;
    }
    let mut prev_lower = first.is_lowercase();
    for c in chars {
        if prev_lower && c.is_uppercase() {
            return true;
        }
        prev_lower = c.is_lowercase();
    }
    false
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

    #[test]
    fn identifier_shape_detection() {
        // Qualifying shapes.
        assert!(is_identifier_shaped("MatrixCache::ensure"));
        assert!(is_identifier_shaped("add_saturating"));
        assert!(is_identifier_shaped("addSaturating"));
        assert!(is_identifier_shaped("MatrixCache"));
        assert!(is_identifier_shaped("foo.bar"));
        // Ordinary English words, including capitalized ones, do not.
        assert!(!is_identifier_shaped("Rust"));
        assert!(!is_identifier_shaped("hello"));
        assert!(!is_identifier_shaped("HELLO"));
        assert!(!is_identifier_shaped(""));
        // Acronym-led proper nouns (all-caps run then lowercase — an
        // upper-to-lower transition, not camelCase's lower-to-upper) read as
        // ordinary prose, not identifiers. Regression: this used to
        // false-trip the identifier leg on eval's `tuning` set, adding an
        // unwanted extra RRF vote to a prose query that merely named one.
        assert!(!is_identifier_shaped("SQLite"));
        // A trailing sentence period isn't a dotted path (caller trims it
        // before calling, but the check alone must not be fooled either).
        assert!(!is_identifier_shaped("sentence."));
    }

    #[test]
    fn identifier_match_expr_extracts_and_phrases() {
        // `::` tokenizes apart (not a tokenchar); the phrase requires the
        // resulting tokens adjacent, so it's still an exact-identifier match.
        assert_eq!(
            identifier_match_expr("how does MatrixCache::ensure work"),
            "\"matrixcache ensure\""
        );
        // `_` and `.` are tokenchars, so these stay single tokens.
        assert_eq!(
            identifier_match_expr("the add_saturating function"),
            "\"add_saturating\""
        );
        // Multiple identifier fragments OR-join.
        assert_eq!(
            identifier_match_expr("MatrixCache::ensure vs add_saturating"),
            "\"matrixcache ensure\" OR \"add_saturating\""
        );
        // Plain prose has no identifier-shaped fragment.
        assert_eq!(identifier_match_expr("how do I authenticate"), "");
        assert_eq!(identifier_match_expr(""), "");
    }
}
