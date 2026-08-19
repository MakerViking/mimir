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
    /// Subset of `calls_resolved` pinned to one method by its receiver's
    /// type rather than by name alone.
    pub calls_typed: usize,
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
/// transaction for the whole update; calls re-resolved only for changed
/// files.
pub fn update(conn: &mut Connection, project: &Node, root: &Path) -> Result<GraphStats> {
    let mut stats = GraphStats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut changed_files: Vec<(i64, String, FileExtract)> = Vec::new();

    // One transaction for the whole update: file/symbol upserts and the
    // call-edge rebuild commit together. A crash mid-update used to leave
    // committed file hashes with missing call edges — and the hash
    // short-circuit then skipped those files forever. IMMEDIATE so a
    // concurrent writer waits (busy_timeout) rather than erroring on a
    // DEFERRED read→write lock upgrade.
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

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
        // -1 = mtime unavailable (some network/virtual filesystems). It
        // must never satisfy the fast path: 0==0 would skip changed files
        // forever; -1 falls through to the content-hash check instead.
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(-1);
        let size = meta.len() as i64;

        let existing = code_file(&tx, project.id, &rel)?;
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
        if let Some(f) = &existing {
            if f.deleted_at.is_none() && f.content_hash.as_deref() == Some(&hash[..]) {
                tx.execute(
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
            &tx,
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
    let mut stmt = tx.prepare(
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
            tx.execute(
                "UPDATE node SET deleted_at = ?2
                 WHERE deleted_at IS NULL AND (id = ?1 OR parent_id = ?1)",
                params![id, now_unix()],
            )?;
            stats.removed += 1;
        }
    }

    resolve_calls(&tx, project.id, &changed_files, &mut stats)?;
    tx.commit()?;
    Ok(stats)
}

/// What the graph would learn if it were rebuilt right now. Drift is
/// defined by content, not timestamps: a touched file whose bytes are
/// unchanged is not drift, because a rebuild would find nothing to do.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Drift {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: usize,
}

impl Drift {
    pub fn is_stale(&self) -> bool {
        !self.added.is_empty() || !self.changed.is_empty() || !self.removed.is_empty()
    }

    pub fn stale_count(&self) -> usize {
        self.added.len() + self.changed.len() + self.removed.len()
    }
}

/// Compare the stored graph against the working tree without touching
/// either. Deliberately never writes — not even the mtime refresh `update`
/// does — so it is safe in a pre-commit hook or a CI job running against a
/// read-only checkout.
pub fn check(conn: &Connection, project: &Node, root: &Path) -> Result<Drift> {
    let mut drift = Drift::default();
    let mut seen: HashSet<String> = HashSet::new();

    // Same traversal as `update`, or the two would disagree about which
    // files are even in scope.
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
        if Lang::from_path(&rel).is_none() {
            continue;
        }
        seen.insert(rel.clone());

        let existing = code_file(conn, project.id, &rel)?;
        let Some(f) = existing.filter(|f| f.deleted_at.is_none()) else {
            drift.added.push(rel);
            continue;
        };
        let meta = entry
            .metadata()
            .map_err(|e| Error::Invalid(format!("stat {rel}: {e}")))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(-1);
        if mtime >= 0
            && f.meta.get("mtime").and_then(|v| v.as_i64()) == Some(mtime)
            && f.meta.get("size").and_then(|v| v.as_i64()) == Some(meta.len() as i64)
        {
            drift.unchanged += 1;
            continue;
        }
        let raw = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let hash = blake3::hash(String::from_utf8_lossy(&raw).as_bytes());
        if f.content_hash.as_deref() == Some(hash.as_bytes()) {
            drift.unchanged += 1;
        } else {
            drift.changed.push(rel);
        }
    }

    let mut stmt = conn.prepare(
        "SELECT path FROM node
         WHERE kind = 'file' AND project_id = ?1 AND collection_id IS NULL
           AND deleted_at IS NULL",
    )?;
    let live: Vec<String> = stmt
        .query_map([project.id], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    drift.removed = live.into_iter().filter(|p| !seen.contains(p)).collect();

    drift.added.sort();
    drift.changed.sort();
    drift.removed.sort();
    Ok(drift)
}

fn code_file(conn: &Connection, project_id: i64, rel: &str) -> Result<Option<Node>> {    Ok(conn
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
    conn: &Connection,
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

    let file_id = match existing {
        Some(f) => {
            conn.execute(
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
            store::insert_node(conn, new)?.id
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
        let existing_id: Option<i64> = conn
            .query_row(
                "SELECT id FROM node
                 WHERE kind = 'symbol' AND json_extract(meta, '$.stable_id') = ?1",
                [&sid],
                |r| r.get(0),
            )
            .optional()?;
        let id = match existing_id {
            Some(id) => {
                conn.execute(
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
                store::insert_node(conn, new)?.id
            }
        };
        kept.insert(id);
        stats.symbols += 1;
    }
    // Symbols that vanished from this file: hard-delete (derived data;
    // edges cascade, embeddings cascade).
    {
        let mut stmt =
            conn.prepare("SELECT id FROM node WHERE kind = 'symbol' AND parent_id = ?1")?;
        let all: Vec<i64> = stmt
            .query_map([file_id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for id in all {
            if !kept.contains(&id) {
                conn.execute("DELETE FROM node WHERE id = ?1", [id])?;
            }
        }
    }
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
        // A NULL/empty name would bucket under "" and attract phantom
        // call edges from any unresolvable callee.
        if !s.name.is_empty() {
            by_name.entry(s.name.as_str()).or_default().push(s);
        }
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

    for (file_id, rel, fx) in changed {
        // Bindings are per-file by construction: a receiver is a name in
        // scope at the call site, so the file that holds the call also
        // holds whatever declared it.
        let mut binds: HashMap<(&str, &str), &str> = HashMap::new();
        for b in &fx.bindings {
            // First declaration wins — parameters and fields are seen
            // before any later shadowing, and determinism matters more
            // than chasing reassignment.
            binds
                .entry((b.scope.as_str(), b.name.as_str()))
                .or_insert(b.type_name.as_str());
        }

        // Wipe edges originating from this file's symbols + its imports.
        conn.execute(
            "DELETE FROM edge WHERE rel = 'calls' AND src IN
               (SELECT id FROM node WHERE kind = 'symbol' AND parent_id = ?1)",
            [file_id],
        )?;
        conn.execute(
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
                        store::link(conn, *file_id, *dst, Rel::Imports, 1.0)?;
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
            // Tier 0: the receiver's type names exactly one method. This is
            // the only tier that can tell two same-named methods apart, so
            // it runs before the same-file shortcut.
            if let Some(dst) = receiver_target(call, &binds, &fx.bindings, candidates, src, rel) {
                link_call(conn, src, dst, 1.0, true)?;
                stats.calls_resolved += 1;
                stats.calls_typed += 1;
                continue;
            }
            // Tier 1: same file.
            if let Some(c) = candidates.iter().find(|c| c.path == *rel && c.id != src) {
                link_call(conn, src, c.id, 1.0, true)?;
                stats.calls_resolved += 1;
                continue;
            }
            // Tier 2: imported name → resolved file.
            if let Some(target) = import_target.get(call.callee.as_str()) {
                if let Some(c) = candidates.iter().find(|c| c.path == *target) {
                    link_call(conn, src, c.id, 1.0, true)?;
                    stats.calls_resolved += 1;
                    continue;
                }
            }
            // Tier 3: global by name — honest about ambiguity.
            let global: Vec<&&SymRef> = candidates.iter().filter(|c| c.id != src).collect();
            match global.len() {
                0 => {}
                1 => {
                    link_call(conn, src, global[0].id, 0.8, true)?;
                    stats.calls_resolved += 1;
                }
                n if n <= 3 => {
                    for c in &global {
                        link_call(conn, src, c.id, 1.0 / n as f64, false)?;
                        stats.calls_heuristic += 1;
                    }
                }
                _ => {} // >3 candidates: too ambiguous to be useful
            }
        }
    }
    Ok(())
}

/// Receivers that mean "the type I am defined on" rather than a name in
/// scope. `Self` is Rust's type-position spelling; the rest are values.
const SELF_RECEIVERS: &[&str] = &["self", "this", "Self", "$this"];

/// Tier 0: turn `recv.callee(..)` into the one method it can mean, using the
/// receiver's type. Returns the target symbol id, or None when the receiver
/// is absent, its type is unknown, or the type owns no such method — every
/// one of which falls through to the name-based tiers unchanged.
fn receiver_target(
    call: &extract::CallSite,
    binds: &HashMap<(&str, &str), &str>,
    bindings: &[extract::Binding],
    candidates: &[&SymRef],
    src: i64,
    rel: &str,
) -> Option<i64> {
    let receiver = call.receiver.as_deref()?;
    let ty = receiver_type(receiver, &call.caller, binds, bindings)?;

    // The type may be written bare (`Greeter`) while the symbol is nested
    // (`App::Greeter::format`), so accept an exact match or a suffix.
    let exact = format!("{ty}::{}", call.callee);
    let suffix = format!("::{exact}");
    let owned: Vec<&&SymRef> = candidates
        .iter()
        .filter(|c| c.id != src && (c.qualified == exact || c.qualified.ends_with(&suffix)))
        .collect();
    match owned.len() {
        0 => None,
        1 => Some(owned[0].id),
        // The same type path in two files: no better than a guess unless
        // exactly one of them is right here. Otherwise fall through to the
        // name-based tiers, which are honest about the weight.
        _ => {
            let same_file: Vec<&&&SymRef> = owned.iter().filter(|c| c.path == rel).collect();
            (same_file.len() == 1).then(|| same_file[0].id)
        }
    }
}

/// Walk a receiver expression to a type name: `self` → the enclosing type,
/// `self.db` → that type's field type, `store` → a local/parameter's type,
/// `Formatter` → itself. Field chains are followed one segment at a time.
fn receiver_type(
    receiver: &str,
    caller: &str,
    binds: &HashMap<(&str, &str), &str>,
    bindings: &[extract::Binding],
) -> Option<String> {
    let segments: Vec<&str> = receiver
        .split(['.', ':', '>', '-'])
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_start_matches('$'))
        .collect();
    let (first, rest) = segments.split_first()?;

    let mut ty = if SELF_RECEIVERS.contains(first) {
        // The enclosing type is the caller's qualified name minus the
        // method itself: "App::Greeter::greet" → "App::Greeter".
        caller.rsplit_once("::")?.0.to_string()
    } else {
        lookup_binding(binds, caller, first)
            .map(str::to_string)
            // Not a name in scope: the receiver is a type already
            // (`Formatter::upper`, `Type.Method()`).
            .unwrap_or_else(|| (*first).to_string())
    };
    for field in rest {
        ty = field_type(bindings, &ty, field)?.to_string();
    }
    (!ty.is_empty()).then_some(ty)
}

/// A local or parameter, looked up from the calling scope outward — an
/// inner function sees names declared around it.
fn lookup_binding<'a>(
    binds: &HashMap<(&str, &'a str), &'a str>,
    caller: &str,
    name: &str,
) -> Option<&'a str> {
    let mut scope = caller;
    loop {
        if let Some(ty) = binds.get(&(scope, name)) {
            return Some(ty);
        }
        match scope.rsplit_once("::") {
            Some((outer, _)) => scope = outer,
            None => return binds.get(&("", name)).copied(),
        }
    }
}

/// A field's type, searched across every scope belonging to the owning type
/// — a Python field is declared inside `__init__`, not on the class. Source
/// order decides ties, so the graph is the same on every rebuild.
fn field_type<'a>(bindings: &'a [extract::Binding], ty: &str, field: &str) -> Option<&'a str> {
    let owned = || {
        bindings
            .iter()
            .filter(|b| b.name == field && scope_owned_by(&b.scope, ty))
    };
    owned()
        .find(|b| b.scope == ty)
        .or_else(|| owned().next())
        .map(|b| b.type_name.as_str())
}

/// Is `scope` the type `ty` itself, or something nested inside it? Matches
/// on the bare name too, since a receiver rarely spells the full path.
fn scope_owned_by(scope: &str, ty: &str) -> bool {
    scope == ty
        || scope.starts_with(&format!("{ty}::"))
        || scope.ends_with(&format!("::{ty}"))
        || scope.contains(&format!("::{ty}::"))
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
        // second-to-last segment (or last for module imports). `files` is a
        // set, so pick the smallest match rather than whichever the hash
        // order happens to yield — a graph that differs between rebuilds
        // makes `graph check` cry wolf.
        for take in (1..=segs.len().min(3)).rev() {
            let suffix = format!("{}.rs", segs[..take].join("/"));
            if let Some(hit) = files.iter().filter(|f| f.ends_with(&suffix)).min() {
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
                .filter(|f| f.ends_with(&format!("{joined}.py")))
                .min()
                .map(|f| f.to_string())
        });
    }

    // Go package path / bare python module: match a file in a dir (or with
    // a name) equal to the last segment.
    let last = source.rsplit('/').next().unwrap_or(source);
    files
        .iter()
        .filter(|f| {
            Path::new(f)
                .parent()
                .and_then(|p| p.file_name())
                .map(|d| d.to_string_lossy() == last)
                .unwrap_or(false)
                || **f == format!("{last}.py")
        })
        .min()
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
