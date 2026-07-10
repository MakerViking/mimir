//! Docs collections and the incremental, hash-driven indexer.
//!
//! A collection is a node (kind=collection) whose `path` is the canonical
//! root of a docs folder. Files and chunks under it are nodes too:
//! file identity is (collection_id, relative path); chunks hang off files
//! via parent_id. One transaction per file → interrupted scans resume
//! idempotently.

pub mod chunker;

use std::collections::HashSet;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{Error, Result};
use crate::model::{now_unix, Kind, NewNode, Node};
use crate::store::{self, row_to_node, NODE_COLS};

/// Register (or return the existing) collection for a docs root.
///
/// `kind` is the collection's content type — "docs" (markdown, chunked by
/// heading) or "code" (source, chunked by tree-sitter symbol; see
/// `chunker::chunk_source`) — recorded in `meta.kind` and read back by
/// `index_collection` to pick the walk filter and chunker.
///
/// Collections are keyed by (path, kind), not path alone: a docs collection
/// and a code collection can coexist at the same root (they walk disjoint
/// file sets — markdown vs. tree-sitter-parseable source), which is exactly
/// what `mimir code add <root>` on a root already auto-registered for docs
/// needs. Re-adding the same (path, kind) returns the existing node
/// unchanged; a different kind at an already-registered path creates a
/// second, independent collection rather than silently no-opping.
pub fn add_collection(
    conn: &Connection,
    root: &Path,
    name: &str,
    project_id: Option<i64>,
    kind: &str,
) -> Result<Node> {
    if !root.is_dir() {
        return Err(Error::Invalid(format!(
            "not a directory: {}",
            root.display()
        )));
    }
    let canonical = crate::scope::canonical_root(root);
    let existing = conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'collection' AND path = ?1 AND deleted_at IS NULL
                 AND COALESCE(json_extract(meta, '$.kind'), 'docs') = ?2"
            ),
            params![&canonical, kind],
            row_to_node,
        )
        .optional()?;
    if let Some(node) = existing {
        return Ok(node);
    }
    let mut new = NewNode::new(Kind::Collection);
    new.title = Some(name.to_string());
    new.path = Some(canonical);
    new.project_id = project_id;
    new.meta = Some(serde_json::json!({ "kind": kind }));
    store::insert_node(conn, new)
}

/// A collection's content type, as recorded by `add_collection`. Absent
/// (pre-existing collections indexed before this field existed) defaults
/// to "docs" so old collections keep behaving exactly as before.
fn collection_kind(collection: &Node) -> &str {
    collection
        .meta
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("docs")
}

pub fn list_collections(conn: &Connection) -> Result<Vec<Node>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE kind = 'collection' AND deleted_at IS NULL ORDER BY title"
    ))?;
    let rows = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// Find a collection by name or by (canonical) path.
pub fn find_collection(conn: &Connection, reference: &str) -> Result<Node> {
    let by_name = conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'collection' AND title = ?1 AND deleted_at IS NULL"
            ),
            [reference],
            row_to_node,
        )
        .optional()?;
    if let Some(node) = by_name {
        return Ok(node);
    }
    let canonical = crate::scope::canonical_root(Path::new(reference));
    conn.query_row(
        &format!(
            "SELECT {NODE_COLS} FROM node
             WHERE kind = 'collection' AND path = ?1 AND deleted_at IS NULL"
        ),
        [&canonical],
        row_to_node,
    )
    .optional()?
    .ok_or_else(|| Error::NotFound(format!("collection '{reference}'")))
}

/// (file_count, chunk_count) of live nodes in a collection.
pub fn collection_stats(conn: &Connection, collection_id: i64) -> Result<(i64, i64)> {
    let count = |kind: &str| -> Result<i64> {
        Ok(conn.query_row(
            "SELECT count(*) FROM node
             WHERE collection_id = ?1 AND kind = ?2 AND deleted_at IS NULL",
            params![collection_id, kind],
            |r| r.get(0),
        )?)
    };
    // A code collection's chunks are Kind::CodeChunk (tree-sitter symbol
    // chunks); a docs collection's are Kind::Chunk (heading chunks); a code
    // collection's plain-text/config fallback content (see plain_text_kind)
    // is also Kind::Chunk. Count both so "chunks" reflects what was actually
    // indexed regardless of which kind carried it.
    Ok((count("file")?, count("chunk")? + count("codechunk")?))
}

/// Soft-delete a collection and everything indexed under it.
pub fn remove_collection(conn: &Connection, collection_id: i64) -> Result<()> {
    let now = now_unix();
    conn.execute(
        "UPDATE node SET deleted_at = ?2
         WHERE deleted_at IS NULL AND (id = ?1 OR collection_id = ?1)",
        params![collection_id, now],
    )?;
    Ok(())
}

/// Attach a qmd-style context note to a collection or file node.
pub fn annotate(conn: &Connection, target: &Node, text: &str) -> Result<Node> {
    let mut new = NewNode::new(Kind::Annotation);
    new.title = Some(crate::memory::derive_title(text));
    new.body = Some(text.to_string());
    new.project_id = target.project_id;
    new.collection_id = match target.kind {
        Kind::Collection => Some(target.id),
        _ => target.collection_id,
    };
    let note = store::insert_node(conn, new)?;
    store::link(conn, note.id, target.id, crate::model::Rel::Describes, 1.0)?;
    Ok(note)
}

/// Resolve `docs/file.md:10-40` (or `:10`, or a bare indexed path) to the
/// numbered lines of the file on disk. Returns None when the reference
/// doesn't look like (or match) an indexed file path. Logs the access.
pub fn file_slice(conn: &Connection, reference: &str) -> Result<Option<String>> {
    let (path_part, span) = match reference.rsplit_once(':') {
        Some((p, range)) if !p.is_empty() => {
            let parse = |s: &str| s.parse::<usize>().ok();
            let span = match range.split_once('-') {
                Some((a, b)) => parse(a).zip(parse(b)),
                None => parse(range).map(|n| (n, n)),
            };
            match span {
                Some(s) => (p, Some(s)),
                None => (reference, None),
            }
        }
        _ => (reference, None),
    };
    // Only paths (with a separator or extension) qualify; ids fall through.
    if !(path_part.contains('/') || path_part.contains('.')) {
        return Ok(None);
    }
    let matches = store::files_by_path_suffix(conn, path_part)?;
    let file = match matches.len() {
        0 => return Ok(None),
        1 => &matches[0],
        n => {
            return Err(Error::AmbiguousRef(
                format!(
                    "{path_part} (matches {})",
                    matches
                        .iter()
                        .filter_map(|f| f.path.as_deref())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                n,
            ))
        }
    };
    let rel = file.path.as_deref().unwrap_or("");
    let collection_id = file
        .collection_id
        .ok_or_else(|| Error::Invalid(format!("file {rel} has no collection")))?;
    let coll = store::get_node(conn, collection_id)?;
    let abs = Path::new(coll.path.as_deref().unwrap_or("")).join(rel);
    let content = std::fs::read_to_string(&abs).map_err(|e| Error::io(&abs, e))?;
    crate::learn::record_opened(conn, file.id)?;
    let total = content.lines().count();
    let (a, b) = span.unwrap_or((1, total));
    let mut out = format!("{rel}:{a}-{} ({total} lines)", b.min(total));
    for (n, line) in content.lines().enumerate() {
        let n = n + 1;
        if n >= a && n <= b {
            out.push_str(&format!("\n{n:>5}  {line}"));
        }
    }
    Ok(Some(out))
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub seen: usize,
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub chunks: usize,
}

/// Index every live collection. Returns (collection name, stats) pairs.
pub fn index_all(conn: &mut Connection) -> Result<Vec<(String, IndexStats)>> {
    let collections = list_collections(conn)?;
    let mut out = Vec::new();
    for c in collections {
        let stats = index_collection(conn, &c)?;
        out.push((c.title.clone().unwrap_or_default(), stats));
    }
    Ok(out)
}

/// Incrementally index one collection: mtime+size short-circuit, blake3
/// change detection, re-chunk only changed files, soft-delete missing ones.
pub fn index_collection(conn: &mut Connection, collection: &Node) -> Result<IndexStats> {
    let root_str = collection
        .path
        .as_deref()
        .ok_or_else(|| Error::Invalid("collection has no path".into()))?;
    let root = Path::new(root_str);
    if !root.is_dir() {
        return Err(Error::NotFound(format!(
            "collection root missing: {root_str}"
        )));
    }

    let mut stats = IndexStats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let is_code = collection_kind(collection) == "code";
    let mut walk = ignore::WalkBuilder::new(root);
    // The plain-text/config fallback (below) whitelists a few dotfiles
    // (`.env.example`), which `ignore`'s default hidden-file filter would
    // otherwise drop before we ever see them — for both collection kinds,
    // now that both use the fallback. Unhide everything, but keep `.git`
    // out explicitly — it's dot-prefixed but never itself gitignored, and
    // walking its (often huge) object store would cost real time for zero
    // payoff (blob filenames have no extension, so plain_text_kind rejects
    // them all anyway).
    walk.hidden(false)
        .filter_entry(|e| e.file_name() != ".git");
    for entry in walk.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(%err, "skipping unreadable entry");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let meta = entry
            .metadata()
            .map_err(|e| Error::Invalid(format!("stat {}: {e}", path.display())))?;
        // -1 = mtime unavailable; must never satisfy the fast path
        // (0==0 would skip changed files forever on such filesystems).
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(-1);
        let size = meta.len() as i64;

        // Docs collections: markdown, plus the plain-text/config fallback
        // (chunker::plain_text_kind) for whitelisted non-markdown files
        // sitting in a docs root — a config how-to's actual .toml/.yml/
        // Dockerfile. Code collections: any file a tree-sitter adapter can
        // parse (Lang::from_path's extension list), plus the same fallback
        // for everything else a source tree needs indexed — Cargo.toml,
        // Dockerfile, .env.example — that tree-sitter can't parse.
        let lang = mimir_syntax::languages::Lang::from_path(&path.to_string_lossy());
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_markdown = matches!(ext.to_ascii_lowercase().as_str(), "md" | "markdown");
        let plain = if lang.is_none() && !is_markdown {
            chunker::plain_text_kind(path, size)
        } else {
            None
        };
        if is_code {
            if lang.is_none() && plain.is_none() {
                continue;
            }
        } else if !is_markdown && plain.is_none() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        seen.insert(rel.clone());
        stats.seen += 1;

        let existing = conn
            .query_row(
                &format!(
                    "SELECT {NODE_COLS} FROM node
                     WHERE kind = 'file' AND collection_id = ?1 AND path = ?2"
                ),
                params![collection.id, rel],
                row_to_node,
            )
            .optional()?;

        // Fast path: stat unchanged and node alive.
        if let Some(f) = &existing {
            if mtime >= 0
                && f.deleted_at.is_none()
                && f.meta.get("mtime").and_then(|v| v.as_i64()) == Some(mtime)
                && f.meta.get("size").and_then(|v| v.as_i64()) == Some(size)
            {
                stats.unchanged += 1;
                continue;
            }
        }

        let raw = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let content = String::from_utf8_lossy(&raw);
        let hash = blake3::hash(content.as_bytes()).as_bytes().to_vec();
        let file_meta = serde_json::json!({ "mtime": mtime, "size": size });

        // Content unchanged (file touched/moved back): refresh stat only.
        if let Some(f) = &existing {
            if f.deleted_at.is_none() && f.content_hash.as_deref() == Some(&hash[..]) {
                conn.execute(
                    "UPDATE node SET meta = ?2, updated_at = ?3 WHERE id = ?1",
                    params![f.id, file_meta.to_string(), now_unix()],
                )?;
                stats.unchanged += 1;
                continue;
            }
        }

        let stem = Path::new(&rel)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| rel.clone());
        // Plain-text/config crumbs use the full file name, not the stem:
        // `Path::file_stem` treats a leading dot as a hidden-file marker and
        // then strips the *next* dot as an "extension" — so `.env.example`'s
        // stem is `.env`, which reads in a search result as if a real .env
        // got indexed (alarming, since .env itself is blacklisted). The
        // markdown branches below intentionally keep using `stem` (e.g.
        // "guide.md" -> crumb "guide"), which is what's wanted there.
        let name = Path::new(&rel)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| rel.clone());
        // Three chunking modes, mutually exclusive by construction (the
        // walk filter above only let one of these `Some` through):
        // tree-sitter source (CodeChunk, subject to code_damp), the
        // plain-text/config fallback (Chunk — config hits must not be
        // damped like code), or markdown (docs collections; also the
        // txt/rst fallback, which packs paragraphs the same way).
        let (file_lang, chunks, chunk_kind): (String, Vec<chunker::DocChunk>, Kind) =
            match (lang, plain) {
                // CodeChunk crumbs use the full relative path, not just the
                // stem: two files named `mod.rs` or `index.ts` in different
                // directories would otherwise produce identical, ambiguous
                // breadcrumbs ("mod › Foo::bar (fn) — ...") in search results.
                (Some(l), _) => (
                    l.name().to_string(),
                    chunker::chunk_source(l, &rel, &content),
                    Kind::CodeChunk,
                ),
                (None, Some(chunker::PlainTextKind::Prose)) => (
                    "text".into(),
                    chunker::chunk_markdown(&name, &content),
                    Kind::Chunk,
                ),
                (None, Some(chunker::PlainTextKind::Config)) => (
                    "config".into(),
                    chunker::chunk_plain_text(&name, &content),
                    Kind::Chunk,
                ),
                (None, None) => (
                    "markdown".into(),
                    chunker::chunk_markdown(&stem, &content),
                    Kind::Chunk,
                ),
            };

        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let file_id = match &existing {
            Some(f) => {
                tx.execute(
                    "UPDATE node SET content_hash = ?2, meta = ?3, updated_at = ?4,
                                     deleted_at = NULL
                     WHERE id = ?1",
                    params![f.id, hash, file_meta.to_string(), now_unix()],
                )?;
                // Chunks are derived data: replace wholesale.
                tx.execute("DELETE FROM node WHERE parent_id = ?1", [f.id])?;
                f.id
            }
            None => {
                let mut new = NewNode::new(Kind::File);
                new.title = Some(
                    Path::new(&rel)
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| rel.clone()),
                );
                new.path = Some(rel.clone());
                new.lang = Some(file_lang.clone());
                new.collection_id = Some(collection.id);
                new.project_id = collection.project_id;
                new.content_hash = Some(hash);
                new.meta = Some(file_meta);
                store::insert_node(&tx, new)?.id
            }
        };
        for chunk in &chunks {
            let mut new = NewNode::new(chunk_kind);
            new.title = Some(chunk.title.clone());
            new.body = Some(chunk.body.clone());
            new.path = Some(rel.clone());
            new.lang = Some(file_lang.clone());
            new.parent_id = Some(file_id);
            new.collection_id = Some(collection.id);
            new.project_id = collection.project_id;
            new.span_start = Some(chunk.start_line as i64);
            new.span_end = Some(chunk.end_line as i64);
            new.content_hash = Some(blake3::hash(chunk.body.as_bytes()).as_bytes().to_vec());
            store::insert_node(&tx, new)?;
        }
        tx.commit()?;
        stats.indexed += 1;
        stats.chunks += chunks.len();
    }

    // Files that vanished from disk: soft-delete them and their chunks.
    let mut stmt = conn.prepare(
        "SELECT id, path FROM node
         WHERE kind = 'file' AND collection_id = ?1 AND deleted_at IS NULL",
    )?;
    let live: Vec<(i64, String)> = stmt
        .query_map([collection.id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for (id, path) in live {
        if !seen.contains(&path) {
            let now = now_unix();
            conn.execute(
                "UPDATE node SET deleted_at = ?2
                 WHERE deleted_at IS NULL AND (id = ?1 OR parent_id = ?1)",
                params![id, now],
            )?;
            stats.removed += 1;
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::model::Scope;
    use crate::search::{self, SearchQuery};

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn setup() -> (tempfile::TempDir, Connection, Node) {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "guide.md",
            "# Install\n\nrun the installer with care\n\n# Use\n\ncall the api twice\n",
        );
        write(dir.path(), "sub/notes.md", "plain notes about turnips\n");
        // Not on any whitelist (no recognized extension) — must stay
        // excluded even now that docs collections also pick up the
        // plain-text/config fallback (unlike `.txt`, which the fallback
        // now legitimately indexes; see docs_collection_indexes_whitelisted_config_too).
        write(dir.path(), "ignored.bin", "not markdown\n");
        let conn = db::open_in_memory().unwrap();
        let coll = add_collection(&conn, dir.path(), "docs", None, "docs").unwrap();
        (dir, conn, coll)
    }

    #[test]
    fn index_creates_files_and_chunks_searchably() {
        let (_dir, mut conn, coll) = setup();
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.seen, 2);
        assert_eq!(stats.indexed, 2);
        let (files, chunks) = collection_stats(&conn, coll.id).unwrap();
        assert_eq!(files, 2);
        assert!(chunks >= 2);

        let hits = search::search(
            &conn,
            &SearchQuery {
                text: "turnips".into(),
                scope: Scope::All,
                kinds: vec![Kind::Chunk],
                since: None,
                limit: 5,
                strength_alpha: 0.0,
                recency_alpha: 0.0,
                type_prior_alpha: 0.0,
                code_damp: 1.0,
                include_superseded: false,
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.path.as_deref(), Some("sub/notes.md"));
    }

    #[test]
    fn reindex_is_incremental_and_detects_changes() {
        let (dir, mut conn, coll) = setup();
        index_collection(&mut conn, &coll).unwrap();

        // Untouched second run: nothing re-indexed.
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.indexed, 0);
        assert_eq!(stats.unchanged, 2);

        // Change one file (force a different mtime via distinct content+size).
        write(
            dir.path(),
            "sub/notes.md",
            "plain notes about rutabagas instead\n",
        );
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.indexed, 1);
        assert_eq!(stats.unchanged, 1);

        let hits = search::search(
            &conn,
            &SearchQuery {
                text: "rutabagas".into(),
                scope: Scope::All,
                kinds: vec![],
                since: None,
                limit: 5,
                strength_alpha: 0.0,
                recency_alpha: 0.0,
                type_prior_alpha: 0.0,
                code_damp: 1.0,
                include_superseded: false,
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        // Old content is gone from the index.
        let old = search::search(
            &conn,
            &SearchQuery {
                text: "turnips".into(),
                scope: Scope::All,
                kinds: vec![],
                since: None,
                limit: 5,
                strength_alpha: 0.0,
                recency_alpha: 0.0,
                type_prior_alpha: 0.0,
                code_damp: 1.0,
                include_superseded: false,
            },
        )
        .unwrap();
        assert!(old.is_empty());
    }

    #[test]
    fn missing_files_soft_deleted_and_resurrected() {
        let (dir, mut conn, coll) = setup();
        index_collection(&mut conn, &coll).unwrap();

        std::fs::remove_file(dir.path().join("sub/notes.md")).unwrap();
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.removed, 1);
        let (files, _) = collection_stats(&conn, coll.id).unwrap();
        assert_eq!(files, 1);

        // File comes back → resurrected, not duplicated.
        write(dir.path(), "sub/notes.md", "plain notes about turnips\n");
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.indexed, 1);
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE kind='file' AND path='sub/notes.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn add_collection_idempotent_and_remove_cascades() {
        let (_dir, mut conn, coll) = setup();
        index_collection(&mut conn, &coll).unwrap();
        let root = Path::new(coll.path.as_deref().unwrap());
        let again = add_collection(&conn, root, "docs", None, "docs").unwrap();
        assert_eq!(again.id, coll.id);

        remove_collection(&conn, coll.id).unwrap();
        let (files, chunks) = collection_stats(&conn, coll.id).unwrap();
        assert_eq!((files, chunks), (0, 0));
        assert!(find_collection(&conn, "docs").is_err());
    }

    #[test]
    fn annotate_attaches_describes_edge() {
        let (_dir, conn, coll) = setup();
        let note = annotate(&conn, &coll, "these docs cover the v2 api only").unwrap();
        assert_eq!(note.kind, Kind::Annotation);
        assert_eq!(note.collection_id, Some(coll.id));
        let edges = store::edges_of(&conn, note.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst, coll.id);
    }

    /// A "code" collection at a path a "docs" collection already owns must
    /// not silently no-op: re-adding with a different `kind` creates a
    /// second, independent collection rather than returning the first one
    /// unchanged. This is exactly what `auto.docs` (registers a docs
    /// collection at the project root on session start) + an explicit
    /// `mimir code add <same root>` produces in practice.
    #[test]
    fn add_collection_keys_by_path_and_kind() {
        let (_dir, conn, docs_coll) = setup();
        let root = Path::new(docs_coll.path.as_deref().unwrap());

        let code_coll = add_collection(&conn, root, "docs", None, "code").unwrap();
        assert_ne!(
            code_coll.id, docs_coll.id,
            "different kind at the same path must not reuse the existing collection"
        );
        assert_eq!(collection_kind(&code_coll), "code");
        assert_eq!(collection_kind(&docs_coll), "docs");

        // Re-adding either one is still idempotent.
        let again = add_collection(&conn, root, "docs", None, "code").unwrap();
        assert_eq!(again.id, code_coll.id);
        let again = add_collection(&conn, root, "docs", None, "docs").unwrap();
        assert_eq!(again.id, docs_coll.id);
    }

    #[test]
    fn code_collection_indexes_config_as_chunk_not_codechunk() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/lib.rs",
            "fn distinctive_needle_marker() {}\n",
        );
        write(
            dir.path(),
            "Cargo.toml",
            "# a config comment, not a markdown heading\nname = \"demo\"\n",
        );
        write(dir.path(), "Cargo.lock", "this must never be indexed\n");
        write(dir.path(), ".env", "SECRET=must-never-be-indexed\n");
        write(dir.path(), ".env.example", "SECRET=placeholder\n");
        let mut conn = db::open_in_memory().unwrap();
        let coll = add_collection(&conn, dir.path(), "demo", None, "code").unwrap();
        let stats = index_collection(&mut conn, &coll).unwrap();

        // Cargo.lock and .env are excluded outright — never even "seen".
        assert_eq!(stats.seen, 3, "expected src/lib.rs, Cargo.toml, .env.example only");

        let kind_of = |path: &str| -> Option<String> {
            conn.query_row(
                "SELECT kind FROM node WHERE collection_id = ?1 AND path = ?2
                 AND kind IN ('chunk','codechunk') AND deleted_at IS NULL LIMIT 1",
                params![coll.id, path],
                |r| r.get(0),
            )
            .optional()
            .unwrap()
        };
        assert_eq!(kind_of("src/lib.rs").as_deref(), Some("codechunk"));
        assert_eq!(
            kind_of("Cargo.toml").as_deref(),
            Some("chunk"),
            "config files must be Chunk, not CodeChunk, or code_damp would suppress them"
        );
        assert_eq!(kind_of(".env.example").as_deref(), Some("chunk"));

        // Crumb regression: file_stem(".env.example") == ".env" (Rust's
        // stem-stripping only ever removes the extension *after* a leading
        // dot), which would make a config hit's title read as if a real
        // .env got indexed. Must be the full file name instead.
        let title: String = conn
            .query_row(
                "SELECT title FROM node WHERE collection_id = ?1 AND path = '.env.example'
                 AND kind = 'chunk' LIMIT 1",
                [coll.id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            title.starts_with(".env.example"),
            "crumb must use the full file name, not the truncated stem: {title}"
        );

        let lockfile_present: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE collection_id = ?1 AND path = 'Cargo.lock'",
                [coll.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lockfile_present, 0);
        let env_present: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE collection_id = ?1 AND path = '.env'",
                [coll.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(env_present, 0);
    }

    /// The plain-text/config fallback isn't code-collection-only: a docs
    /// root's actual `.toml`/`.yml`/Dockerfile (the "how do I configure
    /// this" files a user's question is often really about) must be
    /// searchable too, not just its markdown.
    #[test]
    fn docs_collection_indexes_whitelisted_config_too() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "README.md", "# Setup\n\nfollow the guide\n");
        write(
            dir.path(),
            "config.yml",
            "# deploy config\nreplicas: 3\ntimeout_seconds: 42\n",
        );
        write(dir.path(), "Dockerfile", "FROM scratch\nCMD [\"run\"]\n");
        write(dir.path(), "Cargo.lock", "must never be indexed\n");
        let mut conn = db::open_in_memory().unwrap();
        let coll = add_collection(&conn, dir.path(), "demo", None, "docs").unwrap();
        let stats = index_collection(&mut conn, &coll).unwrap();
        assert_eq!(stats.seen, 3, "expected README.md, config.yml, Dockerfile only");

        let hits = search::search(
            &conn,
            &SearchQuery {
                text: "timeout_seconds".into(),
                scope: Scope::All,
                kinds: vec![],
                since: None,
                limit: 5,
                strength_alpha: 0.0,
                recency_alpha: 0.0,
                type_prior_alpha: 0.0,
                code_damp: 1.0,
                include_superseded: false,
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.path.as_deref(), Some("config.yml"));

        let kind: String = conn
            .query_row(
                "SELECT kind FROM node WHERE collection_id = ?1 AND path = 'config.yml'
                 AND kind IN ('chunk','codechunk') LIMIT 1",
                params![coll.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kind, "chunk");

        let lockfile_present: i64 = conn
            .query_row(
                "SELECT count(*) FROM node WHERE collection_id = ?1 AND path = 'Cargo.lock'",
                [coll.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lockfile_present, 0);
    }
}
