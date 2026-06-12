//! Persist extraction results as nodes/edges, incrementally.
//!
//! Symbol identity across re-extraction: blake3(project | path | qualified
//! | kind) stored in meta.stable_id — the row (and its ULID, and every
//! memory link pointing at it) survives line shifts and body edits.
//! Line numbers live in span columns, never in identity.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mimir_core::error::{Error, Result};
use mimir_core::model::{now_unix, Kind, NewNode, Node, Rel};
use mimir_core::store::{self, row_to_node, NODE_COLS};
use rusqlite::{params, Connection, OptionalExtension};

use crate::extract::{self, FileExtract};
use crate::languages::Lang;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub files_seen: usize,
    pub files_indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub symbols: usize,
    pub calls_resolved: usize,
    pub calls_heuristic: usize,
    pub imports: usize,
}

pub fn stable_id(project_id: i64, rel_path: &str, qualified: &str, kind: &str) -> String {
    blake3::hash(format!("{project_id}|{rel_path}|{qualified}|{kind}").as_bytes())
        .to_hex()
        .to_string()
}

/// Build or incrementally update the code graph for a project rooted at
/// `root`. mtime+size short-circuit, blake3 change detection, one
/// transaction per file; calls re-resolved only for changed files.
pub fn update(conn: &mut Connection, project: &Node, root: &Path) -> Result<GraphStats> {
    let mut stats = GraphStats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut changed_files: Vec<(i64, String, FileExtract)> = Vec::new();

    for entry in ignore::WalkBuilder::new(root).build() {
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
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let Some(lang) = Lang::from_path(&rel) else {
            continue;
        };
        seen.insert(rel.clone());
        stats.files_seen += 1;

        let meta = entry
            .metadata()
            .map_err(|e| Error::Invalid(format!("stat {rel}: {e}")))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let size = meta.len() as i64;

        let existing = code_file(conn, project.id, &rel)?;
        if let Some(f) = &existing {
            if f.deleted_at.is_none()
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
        if let Some(f) = &existing {
            if f.deleted_at.is_none() && f.content_hash.as_deref() == Some(&hash[..]) {
                conn.execute(
                    "UPDATE node SET meta = json_set(meta, '$.mtime', ?2, '$.size', ?3),
                                     updated_at = ?4 WHERE id = ?1",
                    params![f.id, mtime, size, now_unix()],
                )?;
                stats.unchanged += 1;
                continue;
            }
        }

        let fx = extract::extract(lang, &content);
        let file_id = persist_file(
            conn,
            project.id,
            existing.as_ref(),
            &rel,
            lang,
            &hash,
            mtime,
            size,
            &fx,
            &mut stats,
        )?;
        changed_files.push((file_id, rel, fx));
        stats.files_indexed += 1;
    }

    // Files gone from disk: soft-delete file + its symbols.
    let mut stmt = conn.prepare(
        "SELECT id, path FROM node
         WHERE kind = 'file' AND project_id = ?1 AND collection_id IS NULL
           AND deleted_at IS NULL",
    )?;
    let live: Vec<(i64, String)> = stmt
        .query_map([project.id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for (id, path) in live {
        if !seen.contains(&path) {
            conn.execute(
                "UPDATE node SET deleted_at = ?2
                 WHERE deleted_at IS NULL AND (id = ?1 OR parent_id = ?1)",
                params![id, now_unix()],
            )?;
            stats.removed += 1;
        }
    }

    resolve_calls(conn, project.id, &changed_files, &mut stats)?;
    Ok(stats)
}

fn code_file(conn: &Connection, project_id: i64, rel: &str) -> Result<Option<Node>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'file' AND project_id = ?1 AND path = ?2
                   AND collection_id IS NULL"
            ),
            params![project_id, rel],
            row_to_node,
        )
        .optional()?)
}

#[allow(clippy::too_many_arguments)]
fn persist_file(
    conn: &mut Connection,
    project_id: i64,
    existing: Option<&Node>,
    rel: &str,
    lang: Lang,
    hash: &[u8],
    mtime: i64,
    size: i64,
    fx: &FileExtract,
    stats: &mut GraphStats,
) -> Result<i64> {
    let imports_json: Vec<serde_json::Value> = fx
        .imports
        .iter()
        .map(|i| serde_json::json!({"local": i.local, "source": i.source}))
        .collect();
    let calls_json: Vec<serde_json::Value> = fx
        .calls
        .iter()
        .filter(|c| !c.caller.is_empty())
        .map(|c| serde_json::json!({"caller": c.caller, "callee": c.callee}))
        .collect();
    let file_meta = serde_json::json!({
        "mtime": mtime, "size": size,
        "imports": imports_json, "calls": calls_json,
    });

    let tx = conn.transaction()?;
    let file_id = match existing {
        Some(f) => {
            tx.execute(
                "UPDATE node SET content_hash = ?2, meta = ?3, lang = ?4,
                                 updated_at = ?5, deleted_at = NULL WHERE id = ?1",
                params![f.id, hash, file_meta.to_string(), lang.name(), now_unix()],
            )?;
            f.id
        }
        None => {
            let mut new = NewNode::new(Kind::File);
            new.title = Some(
                Path::new(rel)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel.to_string()),
            );
            new.path = Some(rel.to_string());
            new.lang = Some(lang.name().into());
            new.project_id = Some(project_id);
            new.content_hash = Some(hash.to_vec());
            new.meta = Some(file_meta);
            store::insert_node(&tx, new)?.id
        }
    };

    // Upsert symbols by stable_id; collect survivors to prune the rest.
    let mut kept: HashSet<i64> = HashSet::new();
    for sym in &fx.symbols {
        let sid = stable_id(project_id, rel, &sym.qualified, sym.kind);
        let body = match &sym.doc {
            Some(d) => format!("{}\n{d}", sym.signature),
            None => sym.signature.clone(),
        };
        let meta = serde_json::json!({"stable_id": sid, "name": sym.name});
        let existing_id: Option<i64> = tx
            .query_row(
                "SELECT id FROM node
                 WHERE kind = 'symbol' AND json_extract(meta, '$.stable_id') = ?1",
                [&sid],
                |r| r.get(0),
            )
            .optional()?;
        let id = match existing_id {
            Some(id) => {
                tx.execute(
                    "UPDATE node SET title = ?2, body = ?3, subkind = ?4, path = ?5,
                            span_start = ?6, span_end = ?7, content_hash = ?8, meta = ?9,
                            lang = ?10, parent_id = ?11, updated_at = ?12, deleted_at = NULL
                     WHERE id = ?1",
                    params![
                        id,
                        sym.qualified,
                        body,
                        sym.kind,
                        rel,
                        sym.start_line as i64,
                        sym.end_line as i64,
                        blake3::hash(body.as_bytes()).as_bytes().to_vec(),
                        meta.to_string(),
                        lang.name(),
                        file_id,
                        now_unix()
                    ],
                )?;
                id
            }
            None => {
                let mut new = NewNode::new(Kind::Symbol);
                new.subkind = Some(sym.kind.into());
                new.title = Some(sym.qualified.clone());
                new.body = Some(body.clone());
                new.path = Some(rel.to_string());
                new.lang = Some(lang.name().into());
                new.project_id = Some(project_id);
                new.parent_id = Some(file_id);
                new.span_start = Some(sym.start_line as i64);
                new.span_end = Some(sym.end_line as i64);
                new.content_hash = Some(blake3::hash(body.as_bytes()).as_bytes().to_vec());
                new.meta = Some(meta);
                store::insert_node(&tx, new)?.id
            }
        };
        kept.insert(id);
        stats.symbols += 1;
    }
    // Symbols that vanished from this file: hard-delete (derived data;
    // edges cascade, embeddings cascade).
    {
        let mut stmt =
            tx.prepare("SELECT id FROM node WHERE kind = 'symbol' AND parent_id = ?1")?;
        let all: Vec<i64> = stmt
            .query_map([file_id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for id in all {
            if !kept.contains(&id) {
                tx.execute("DELETE FROM node WHERE id = ?1", [id])?;
            }
        }
    }
    tx.commit()?;
    Ok(file_id)
}

/// A project symbol as seen by the resolver.
struct SymRef {
    id: i64,
    name: String,
    qualified: String,
    path: String,
}

/// Re-resolve call/import edges for the changed files only. Unchanged
/// symbols keep their ids, so inbound edges stay valid automatically.
fn resolve_calls(
    conn: &Connection,
    project_id: i64,
    changed: &[(i64, String, FileExtract)],
    stats: &mut GraphStats,
) -> Result<()> {
    if changed.is_empty() {
        return Ok(());
    }
    // Project-wide symbol table (one query, in-memory buckets).
    let mut stmt = conn.prepare(
        "SELECT id, json_extract(meta, '$.name'), title, path FROM node
         WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL",
    )?;
    let symbols: Vec<SymRef> = stmt
        .query_map([project_id], |r| {
            Ok(SymRef {
                id: r.get(0)?,
                name: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                qualified: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    let mut by_name: HashMap<&str, Vec<&SymRef>> = HashMap::new();
    for s in &symbols {
        by_name.entry(s.name.as_str()).or_default().push(s);
    }
    let mut by_file_qualified: HashMap<(&str, &str), i64> = HashMap::new();
    for s in &symbols {
        by_file_qualified.insert((s.path.as_str(), s.qualified.as_str()), s.id);
    }
    let file_paths: HashSet<&str> = {
        let mut set = HashSet::new();
        for s in &symbols {
            set.insert(s.path.as_str());
        }
        set
    };
    let file_ids: HashMap<String, i64> = {
        let mut stmt = conn.prepare(
            "SELECT path, id FROM node
             WHERE kind = 'file' AND project_id = ?1 AND collection_id IS NULL
               AND deleted_at IS NULL",
        )?;
        let rows: Vec<(String, i64)> = stmt
            .query_map([project_id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        rows.into_iter().collect()
    };

    let tx = conn.unchecked_transaction()?;
    for (file_id, rel, fx) in changed {
        // Wipe edges originating from this file's symbols + its imports.
        tx.execute(
            "DELETE FROM edge WHERE rel = 'calls' AND src IN
               (SELECT id FROM node WHERE kind = 'symbol' AND parent_id = ?1)",
            [file_id],
        )?;
        tx.execute(
            "DELETE FROM edge WHERE rel = 'imports' AND src = ?1",
            [file_id],
        )?;

        // Import map for tier-2 resolution + file→file import edges.
        let mut import_target: HashMap<&str, String> = HashMap::new();
        for imp in &fx.imports {
            if let Some(target) = resolve_import(rel, &imp.source, &file_paths) {
                import_target.insert(imp.local.as_str(), target.clone());
                if let Some(dst) = file_ids.get(&target) {
                    if *dst != *file_id {
                        store::link(&tx, *file_id, *dst, Rel::Imports, 1.0)?;
                        stats.imports += 1;
                    }
                }
            }
        }

        for call in &fx.calls {
            if call.caller.is_empty() {
                continue; // top-level statements have no source symbol
            }
            let Some(&src) = by_file_qualified.get(&(rel.as_str(), call.caller.as_str())) else {
                continue;
            };
            let candidates = by_name.get(call.callee.as_str());
            let Some(candidates) = candidates else {
                continue;
            };
            // Tier 1: same file.
            if let Some(c) = candidates.iter().find(|c| c.path == *rel && c.id != src) {
                link_call(&tx, src, c.id, 1.0, true)?;
                stats.calls_resolved += 1;
                continue;
            }
            // Tier 2: imported name → resolved file.
            if let Some(target) = import_target.get(call.callee.as_str()) {
                if let Some(c) = candidates.iter().find(|c| c.path == *target) {
                    link_call(&tx, src, c.id, 1.0, true)?;
                    stats.calls_resolved += 1;
                    continue;
                }
            }
            // Tier 3: global by name — honest about ambiguity.
            let global: Vec<&&SymRef> = candidates.iter().filter(|c| c.id != src).collect();
            match global.len() {
                0 => {}
                1 => {
                    link_call(&tx, src, global[0].id, 0.8, true)?;
                    stats.calls_resolved += 1;
                }
                n if n <= 3 => {
                    for c in &global {
                        link_call(&tx, src, c.id, 1.0 / n as f64, false)?;
                        stats.calls_heuristic += 1;
                    }
                }
                _ => {} // >3 candidates: too ambiguous to be useful
            }
        }
    }
    tx.commit()?;
    Ok(())
}

fn link_call(conn: &Connection, src: i64, dst: i64, weight: f64, resolved: bool) -> Result<()> {
    conn.execute(
        "INSERT INTO edge (src, dst, rel, weight, meta, created_at)
         VALUES (?1, ?2, 'calls', ?3, json_object('resolved', ?4), ?5)
         ON CONFLICT(src, dst, rel) DO UPDATE SET
           weight = excluded.weight, meta = excluded.meta",
        params![src, dst, weight, resolved, now_unix()],
    )?;
    Ok(())
}

/// Best-effort import-source → project-file resolution.
fn resolve_import(importer: &str, source: &str, files: &HashSet<&str>) -> Option<String> {
    let dir = Path::new(importer).parent().unwrap_or(Path::new(""));
    let try_paths = |bases: Vec<String>| -> Option<String> {
        bases.into_iter().find(|b| files.contains(b.as_str()))
    };

    if source.starts_with('.') {
        if source.contains("::") {
            return None; // relative Rust paths handled below by suffix
        }
        // JS/TS relative ("./x", "../y/z") or Python relative (".util").
        if source.starts_with("./") || source.starts_with("../") {
            let joined = normalize(&dir.join(source));
            return try_paths(vec![
                format!("{joined}.ts"),
                format!("{joined}.tsx"),
                format!("{joined}.js"),
                format!("{joined}.jsx"),
                format!("{joined}/index.ts"),
                format!("{joined}/index.js"),
                joined.clone(),
            ]);
        }
        // Python relative: ".util" / "..pkg.mod"
        let dots = source.chars().take_while(|c| *c == '.').count();
        let module = &source[dots..];
        let mut base = dir.to_path_buf();
        for _ in 1..dots {
            base = base.parent().map(Path::to_path_buf).unwrap_or_default();
        }
        let joined = normalize(&base.join(module.replace('.', "/")));
        return try_paths(vec![
            format!("{joined}.py"),
            format!("{joined}/__init__.py"),
        ]);
    }

    if source.contains("::") {
        // Rust: crate::a::b / super::x — match by path suffix on segments.
        let segs: Vec<&str> = source
            .split("::")
            .filter(|s| !matches!(*s, "crate" | "super" | "self"))
            .collect();
        if segs.is_empty() {
            return None;
        }
        // The import target is usually an item; its module file is the
        // second-to-last segment (or last for module imports).
        for take in (1..=segs.len().min(3)).rev() {
            let suffix = format!("{}.rs", segs[..take].join("/"));
            if let Some(hit) = files.iter().find(|f| f.ends_with(&suffix)) {
                return Some(hit.to_string());
            }
        }
        return None;
    }

    if source.contains('.') && !source.contains('/') {
        // Python absolute module: a.b.c
        let joined = source.replace('.', "/");
        return try_paths(vec![
            format!("{joined}.py"),
            format!("{joined}/__init__.py"),
        ])
        .or_else(|| {
            files
                .iter()
                .find(|f| f.ends_with(&format!("{joined}.py")))
                .map(|f| f.to_string())
        });
    }

    // Go package path / bare python module: match a file in a dir (or with
    // a name) equal to the last segment.
    let last = source.rsplit('/').next().unwrap_or(source);
    files
        .iter()
        .find(|f| {
            Path::new(f)
                .parent()
                .and_then(|p| p.file_name())
                .map(|d| d.to_string_lossy() == last)
                .unwrap_or(false)
                || **f == format!("{last}.py")
        })
        .map(|f| f.to_string())
}

fn normalize(p: &Path) -> String {
    let mut parts: Vec<&std::ffi::OsStr> = Vec::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::CurDir => {}
            std::path::Component::Normal(s) => parts.push(s),
            _ => {}
        }
    }
    parts
        .iter()
        .map(|s| s.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}
