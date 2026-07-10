//! Memory capture: typed thoughts with tag support and duplicate detection.

use std::collections::HashSet;

use rusqlite::{Connection, OptionalExtension};

use crate::error::Result;
use crate::model::{Kind, MemoryType, NewNode, Node, Scope};
use crate::search::{self, SearchQuery};
use crate::store::{self, row_to_node, NODE_COLS};

/// Token-set similarity at or above which a new memory is refused
/// as a near-duplicate (without --force).
const NEAR_DUP_JACCARD: f64 = 0.85;

#[derive(Debug)]
pub enum RememberOutcome {
    Created(Node),
    /// Refused: the contained node is the existing near-duplicate.
    Duplicate(Node),
}

#[derive(Debug)]
pub struct Remember {
    pub text: String,
    pub mtype: MemoryType,
    pub tags: Vec<String>,
    pub project_id: Option<i64>,
    pub force: bool,
}

/// Store a memory unless an exact or near duplicate already exists.
pub fn remember(conn: &Connection, args: Remember) -> Result<RememberOutcome> {
    let hash = content_hash(&args.text);
    if !args.force {
        if let Some(dup) = find_duplicate(conn, &args.text, &hash)? {
            return Ok(RememberOutcome::Duplicate(dup));
        }
    }
    let mut new = NewNode::new(Kind::Memory);
    new.subkind = Some(args.mtype.to_string());
    new.title = Some(derive_title(&args.text));
    new.body = Some(args.text);
    new.tags = sanitize_tags(args.tags);
    new.project_id = args.project_id;
    new.content_hash = Some(hash);
    Ok(RememberOutcome::Created(store::insert_node(conn, new)?))
}

/// Tags surface as nodes in the graph visualization (HTML). Strip the
/// HTML-significant characters at ingest so a tag can never carry markup —
/// the viz also escapes at render time, this is defense in depth. Tags that
/// reduce to nothing are dropped.
pub fn sanitize_tags(tags: Vec<String>) -> Vec<String> {
    tags.into_iter()
        .map(|t| {
            t.chars()
                .filter(|c| !matches!(c, '<' | '>' | '&' | '"' | '\''))
                .collect::<String>()
                .trim()
                .to_string()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Normalize author-declared trigger phrases for `meta.fires_when` (see
/// `inject::clears_floor`'s fires_when path): whitespace-collapsed,
/// lowercased, capped to 8 phrases of at most 6 words each. An oversized
/// phrase is dropped rather than truncated — same "reject, don't mangle"
/// call as `sanitize_tags` makes for markup, since a truncated trigger
/// could silently mean something the author never wrote.
pub fn sanitize_fires_when(phrases: Vec<String>) -> Vec<String> {
    phrases
        .into_iter()
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase())
        .filter(|p| !p.is_empty() && p.split_whitespace().count() <= 6)
        .take(8)
        .collect()
}

fn find_duplicate(conn: &Connection, text: &str, hash: &[u8]) -> Result<Option<Node>> {
    // Exact (normalized) content match, any scope.
    let exact = conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'memory' AND content_hash = ?1 AND deleted_at IS NULL
                 LIMIT 1"
            ),
            [hash],
            row_to_node,
        )
        .optional()?;
    if exact.is_some() {
        return Ok(exact);
    }
    // Near-duplicate: high token overlap with the best FTS matches.
    let query = SearchQuery {
        text: text.to_string(),
        scope: Scope::All,
        kinds: vec![Kind::Memory],
        since: None,
        limit: 5,
        strength_alpha: 0.0,
        recency_alpha: 0.0,
        type_prior_alpha: 0.0,
        code_damp: 1.0,
        include_superseded: false,
    };
    for hit in search::search(conn, &query)? {
        // Title-only memories (some imports) have no body — compare against
        // the title so they still participate in duplicate detection.
        let existing = hit
            .node
            .body
            .as_deref()
            .filter(|b| !b.is_empty())
            .or(hit.node.title.as_deref())
            .unwrap_or_default();
        if jaccard(text, existing) >= NEAR_DUP_JACCARD {
            return Ok(Some(hit.node));
        }
    }
    Ok(None)
}

/// blake3 over lowercased, whitespace/punctuation-normalized text, so
/// formatting tweaks don't defeat exact-duplicate detection.
pub fn content_hash(text: &str) -> Vec<u8> {
    let normalized = tokens(text).join(" ");
    blake3::hash(normalized.as_bytes()).as_bytes().to_vec()
}

/// Identifier-friendly tokens: keeps `snake_case`, `kebab-case`, `file.rs` whole.
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = tokens(a).into_iter().collect();
    let sb: HashSet<String> = tokens(b).into_iter().collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    inter / union
}

/// First line of the text, truncated to 80 chars on a char boundary.
pub fn derive_title(text: &str) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    let mut title: String = first.chars().take(80).collect();
    if first.chars().count() > 80 {
        title.push('…');
    }
    title
}

#[derive(Debug, Default)]
pub struct Edit {
    pub text: Option<String>,
    pub title: Option<String>,
    pub mtype: Option<MemoryType>,
    pub tags: Option<Vec<String>>,
}

/// Apply field edits to an existing node; body edits refresh the derived
/// title (unless one is given) and the content hash.
pub fn edit(conn: &Connection, id: i64, edit: Edit) -> Result<Node> {
    let node = store::get_node(conn, id)?;
    let body = edit.text.or(node.body);
    let title = match edit.title {
        Some(t) => Some(t),
        None => body.as_deref().map(derive_title),
    };
    let subkind = edit.mtype.map(|t| t.to_string()).or(node.subkind);
    let tags_text = edit
        .tags
        .map(|t| sanitize_tags(t).join(" "))
        .unwrap_or(node.tags_text);
    let hash = body.as_deref().map(content_hash);
    conn.execute(
        "UPDATE node SET title = ?2, body = ?3, subkind = ?4, tags_text = ?5,
                         content_hash = ?6, updated_at = ?7
         WHERE id = ?1",
        rusqlite::params![
            id,
            title,
            body,
            subkind,
            tags_text,
            hash,
            crate::model::now_unix()
        ],
    )?;
    store::get_node(conn, id)
}

/// Recent memories, newest first, optionally filtered by type and tag.
pub fn list(
    conn: &Connection,
    scope: Scope,
    mtype: Option<MemoryType>,
    tag: Option<&str>,
    limit: usize,
) -> Result<Vec<Node>> {
    let mut sql = format!(
        "SELECT {NODE_COLS} FROM node n
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL AND {}",
        search::scope_sql(scope)
    );
    if let Some(t) = mtype {
        sql.push_str(&format!(" AND n.subkind = '{t}'"));
    }
    if tag.is_some() {
        sql.push_str(" AND (' ' || n.tags_text || ' ') LIKE ?2");
    }
    sql.push_str(" ORDER BY n.created_at DESC LIMIT ?1");
    let mut stmt = conn.prepare(&sql)?;
    let rows = match tag {
        Some(t) => stmt
            .query_map(
                rusqlite::params![limit as i64, format!("% {t} %")],
                row_to_node,
            )?
            .collect::<rusqlite::Result<_>>()?,
        None => stmt
            .query_map([limit as i64], row_to_node)?
            .collect::<rusqlite::Result<_>>()?,
    };
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[test]
    fn sanitize_tags_strips_markup_and_drops_empties() {
        let got = sanitize_tags(vec![
            "<img/src=x/onerror=alert(1)>".into(),
            "auth".into(),
            "<>&\"'".into(),
            "  postgres  ".into(),
        ]);
        // markup chars gone, empties dropped, whitespace trimmed
        assert_eq!(got, vec!["img/src=x/onerror=alert(1)", "auth", "postgres"]);
        assert!(got.iter().all(|t| !t.contains('<') && !t.contains('>')));
    }

    #[test]
    fn sanitize_fires_when_normalizes_caps_and_drops_oversized() {
        let got = sanitize_fires_when(vec![
            "  Release   Checklist  ".into(),
            "CUTTING a release".into(),
            "".into(),
            "   ".into(),
            "this phrase has way more than six words in it".into(),
        ]);
        assert_eq!(got, vec!["release checklist", "cutting a release"]);
    }

    #[test]
    fn sanitize_fires_when_caps_at_eight_phrases() {
        let phrases: Vec<String> = (0..12).map(|i| format!("phrase {i}")).collect();
        assert_eq!(sanitize_fires_when(phrases).len(), 8);
    }

    fn remember_text(conn: &Connection, text: &str) -> RememberOutcome {
        remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
            },
        )
        .unwrap()
    }

    #[test]
    fn remember_then_exact_duplicate_refused() {
        let conn = db::open_in_memory().unwrap();
        let first = match remember_text(&conn, "SCRAM auth rejects non-ASCII passwords") {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        // Same content, different whitespace/case → still a duplicate.
        match remember_text(&conn, "scram auth  rejects\nnon-ascii passwords") {
            RememberOutcome::Duplicate(d) => assert_eq!(d.id, first.id),
            other => panic!("expected Duplicate, got {other:?}"),
        }
    }

    #[test]
    fn near_duplicate_refused_force_overrides() {
        let conn = db::open_in_memory().unwrap();
        remember_text(
            &conn,
            "the indexer walks gitignore files using the ignore crate",
        );
        match remember_text(
            &conn,
            "the indexer walks gitignore files using the ignore crate now",
        ) {
            RememberOutcome::Duplicate(_) => {}
            other => panic!("expected Duplicate, got {other:?}"),
        }
        let forced = remember(
            &conn,
            Remember {
                text: "the indexer walks gitignore files using the ignore crate now".into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
            },
        )
        .unwrap();
        assert!(matches!(forced, RememberOutcome::Created(_)));
    }

    #[test]
    fn distinct_memories_both_stored_and_listed() {
        let conn = db::open_in_memory().unwrap();
        remember_text(&conn, "use WAL mode for concurrent readers");
        remember_text(&conn, "prefer RRF over score normalization for fusion");
        let all = list(&conn, Scope::All, None, None, 10).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn edit_updates_body_title_and_hash() {
        let conn = db::open_in_memory().unwrap();
        let node = match remember_text(&conn, "original text here") {
            RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        };
        let old_hash = node.content_hash.clone();
        let updated = edit(
            &conn,
            node.id,
            Edit {
                text: Some("completely new body".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(updated.title.as_deref(), Some("completely new body"));
        assert_ne!(updated.content_hash, old_hash);
    }

    #[test]
    fn list_filters_by_type_and_tag() {
        let conn = db::open_in_memory().unwrap();
        remember(
            &conn,
            Remember {
                text: "tagged gotcha".into(),
                mtype: MemoryType::Gotcha,
                tags: vec!["sqlite".into()],
                project_id: None,
                force: false,
            },
        )
        .unwrap();
        remember_text(&conn, "plain note");
        let gotchas = list(&conn, Scope::All, Some(MemoryType::Gotcha), None, 10).unwrap();
        assert_eq!(gotchas.len(), 1);
        let tagged = list(&conn, Scope::All, None, Some("sqlite"), 10).unwrap();
        assert_eq!(tagged.len(), 1);
        let missing = list(&conn, Scope::All, None, Some("nope"), 10).unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn derive_title_truncates() {
        assert_eq!(derive_title("short one\nsecond line"), "short one");
        let long = "x".repeat(100);
        let title = derive_title(&long);
        assert_eq!(title.chars().count(), 81); // 80 + ellipsis
    }
}
