//! Persist extraction results as nodes/edges, incrementally.
//!
//! Symbol identity across re-extraction: blake3(project | path | qualified
//! | kind) stored in meta.stable_id — the row (and its ULID, and every
//! memory link pointing at it) survives line shifts and body edits.
//! Line numbers live in span columns, never in identity.
//!
//! Two definitions with the same qualified name and kind in one file (a
//! `#[cfg(unix)]` / `#[cfg(windows)]` pair, one method given to a type by two
//! trait impls) share one row, the later overwriting the earlier. A row each
//! would be a row with nothing to say: a call site carries a name, not the
//! line of the definition it binds to, so no caller could be attributed to
//! one of them rather than the other, the second would have no edges at all,
//! and the name would become ambiguous to ask about. Their position in the
//! file cannot be the identity either, since swapping the two would move
//! every memory linked to one onto the other.
//!
//! # What an update owes
//!
//! After any sequence of edits, `update` leaves exactly the live symbols and
//! edges that building the final tree into an empty store would. Extraction
//! is per file, but resolution is not: the edges of an *unchanged* file depend
//! on which symbols carry the names it calls and on which files its imports
//! can reach. So an update that adds or removes a symbol, or a file,
//! re-resolves every unchanged file that calls that name or whose imports
//! now land elsewhere — found from the `calls` and `imports` each file row
//! keeps in its meta for exactly this. `tests/incremental.rs` holds the
//! promise.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use mimir_core::error::Result;
use mimir_core::model::{now_unix, Kind, NewNode, Node, Rel};
use mimir_core::store;
use rusqlite::{params, Connection, OptionalExtension};

use crate::extract::{self, ExtractError, FileExtract, SCOPE_SEP};
use crate::imports::{resolve_import, FileSet};
use crate::languages::Lang;
use crate::tier::CallTier;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub files_seen: usize,
    /// Files extracted and persisted in this run.
    pub files_indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    /// Symbol rows written (inserted or updated) for the files indexed.
    pub symbols: usize,
    /// Call sites tied to exactly one target, in every file resolved by this
    /// run. A count of call sites, not of edges: several sites in one
    /// function that call the same target share one edge.
    pub calls_resolved: usize,
    /// Subset of `calls_resolved` pinned to one method by its receiver's
    /// type rather than by name alone.
    pub calls_typed: usize,
    /// Call sites left as a guess between two or three candidates.
    pub calls_heuristic: usize,
    /// Import edges written.
    pub imports: usize,
}

/// What a caller may ask of [`update_with`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UpdateOptions {
    /// Re-extract every file and rebuild every edge, whatever the stored
    /// hashes say. The recovery path: for a graph whose edges were resolved
    /// by an older build's rules, or that is suspected wrong for any reason.
    /// Symbols keep their rows, so memory links survive it.
    pub full: bool,
}

/// Everything one update did, including what it could not do.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct UpdateReport {
    pub stats: GraphStats,
    /// Files whose parser failed. Each keeps the symbols it had and is not
    /// recorded as current, so the next update tries it again.
    pub parse_failed: usize,
    /// Files that could not be stat'ed or read. Skipped the same way.
    pub unreadable: usize,
    /// Unchanged files whose edges were rebuilt because a symbol they call,
    /// or a file they import, appeared or disappeared elsewhere.
    pub reresolved: usize,
}

pub fn stable_id(project_id: i64, rel_path: &str, qualified: &str, kind: &str) -> String {
    blake3::hash(format!("{project_id}|{rel_path}|{qualified}|{kind}").as_bytes())
        .to_hex()
        .to_string()
}

/// "A live code file of this project", as a `WHERE` fragment over `node`
/// with the project id bound to `?1`. `kind = 'file'` alone is not it: the
/// docs indexer writes file rows too, told apart by their `collection_id`.
pub(crate) const LIVE_CODE_FILE: &str =
    "kind = 'file' AND project_id = ?1 AND collection_id IS NULL AND deleted_at IS NULL";

/// What the filesystem says about a file without reading it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stat {
    /// Whole seconds; -1 when the filesystem reports no mtime.
    mtime: i64,
    /// The sub-second part, in nanoseconds; -1 alongside `mtime`.
    mtime_ns: i64,
    size: i64,
}

impl Stat {
    fn of(meta: &std::fs::Metadata) -> Stat {
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
        Stat {
            mtime: modified.map(|d| d.as_secs() as i64).unwrap_or(-1),
            mtime_ns: modified.map(|d| d.subsec_nanos() as i64).unwrap_or(-1),
            size: meta.len() as i64,
        }
    }
}

/// The columns of a live code-file row that decide whether it is current.
struct FileRow {
    id: i64,
    hash: Option<Vec<u8>>,
    mtime: Option<i64>,
    mtime_ns: Option<i64>,
    size: Option<i64>,
}

impl FileRow {
    /// The stat alone proves the bytes are the ones indexed. The one
    /// freshness test `update` and `check` share, so they cannot disagree
    /// about which files need reading.
    ///
    /// An unavailable mtime (-1, some network and virtual filesystems) never
    /// passes: 0 == 0 would skip a changed file forever. Neither does a row
    /// written before `mtime_ns` was stored — whole seconds cannot see an
    /// edit made in the same second that kept the size — so such a row falls
    /// through to the content hash once, and `update` then stores the value.
    fn stat_matches(&self, stat: &Stat) -> bool {
        stat.mtime >= 0
            && self.mtime == Some(stat.mtime)
            && self.mtime_ns == Some(stat.mtime_ns)
            && self.size == Some(stat.size)
    }

    /// [`FileRow::stat_matches`] for the check a query makes, which cannot
    /// afford to hash: a row with no `mtime_ns` is taken at whole seconds
    /// and size, the precision it was written with. Hashing it instead would
    /// read every file of a graph an older build made, on every query, until
    /// the next build stamped them.
    fn stat_matches_as_stored(&self, stat: &Stat) -> bool {
        match self.mtime_ns {
            Some(_) => self.stat_matches(stat),
            None => {
                stat.mtime >= 0 && self.mtime == Some(stat.mtime) && self.size == Some(stat.size)
            }
        }
    }

    /// Does the file at `path` still hold the bytes this row indexed?
    /// `None` when it cannot be read, which no build could act on.
    fn same_content(&self, path: &Path) -> Option<bool> {
        let raw = std::fs::read(path).ok()?;
        let hash = blake3::hash(String::from_utf8_lossy(&raw).as_bytes());
        Some(self.hash.as_deref() == Some(hash.as_bytes().as_slice()))
    }
}

/// The stored stat of a file row — `(mtime, mtime_ns, size)` — read out of
/// its meta text.
///
/// The meta of a file row is mostly its call list, several kilobytes of it,
/// and SQLite's `json_extract` parses all of that to hand back one number.
/// Three of them per row was most of what a freshness check cost. The three
/// keys are top-level, written last, and nothing else in the object can
/// spell them (a string containing `"size":` has its quotes escaped), so
/// they are found by searching the text from its end. A meta that is not in
/// the compact form this module and `json_set` write is parsed properly.
fn stored_stat(meta: &str) -> (Option<i64>, Option<i64>, Option<i64>) {
    fn number_after(meta: &str, key: &str) -> Option<i64> {
        let rest = &meta[meta.rfind(key)? + key.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '-'))
            .unwrap_or(rest.len());
        // An integer, then the next key or the end of the object: anything
        // else (a fraction, an exponent, a space) is not the compact form.
        matches!(rest[end..].chars().next(), Some(',' | '}'))
            .then(|| rest[..end].parse().ok())
            .flatten()
    }
    let found = (
        number_after(meta, "\"mtime\":"),
        number_after(meta, "\"mtime_ns\":"),
        number_after(meta, "\"size\":"),
    );
    if found.0.is_some() && found.2.is_some() {
        return found;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(meta) else {
        return (None, None, None);
    };
    let number = |key: &str| value.get(key).and_then(|v| v.as_i64());
    (number("mtime"), number("mtime_ns"), number("size"))
}

/// Live code-file rows of a project by path.
fn live_code_files(conn: &Connection, project_id: i64) -> Result<HashMap<String, FileRow>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT path, id, content_hash, meta FROM node WHERE {LIVE_CODE_FILE} ORDER BY id"
    ))?;
    let mut rows = stmt.query([project_id])?;
    let mut out: HashMap<String, FileRow> = HashMap::new();
    while let Some(r) = rows.next()? {
        let Some(path) = r.get::<_, Option<String>>(0)? else {
            continue;
        };
        let (mtime, mtime_ns, size) = stored_stat(r.get_ref(3)?.as_str().unwrap_or_default());
        // Nothing enforces one row per path; the oldest is the one earlier
        // builds kept updating.
        out.entry(path).or_insert(FileRow {
            id: r.get(1)?,
            hash: r.get(2)?,
            mtime,
            mtime_ns,
            size,
        });
    }
    Ok(out)
}

/// The files `update` indexes and `check` compares: one traversal, or the
/// two would disagree about which files are even in scope. It can be given
/// a budget of directory entries, for a caller that must not pay for a tree
/// that is mostly something other than code.
struct CodeWalk {
    inner: ignore::Walk,
    root: PathBuf,
    visited: usize,
    budget: usize,
    cut_short: bool,
}

impl CodeWalk {
    fn new(root: &Path) -> CodeWalk {
        CodeWalk::within(root, usize::MAX)
    }

    /// A walk that stops after visiting `budget` entries, code or not.
    fn within(root: &Path, budget: usize) -> CodeWalk {
        CodeWalk {
            inner: ignore::WalkBuilder::new(root).build(),
            root: root.to_path_buf(),
            visited: 0,
            budget,
            cut_short: false,
        }
    }
}

impl Iterator for CodeWalk {
    /// (entry, project-relative path with `/` separators, language)
    type Item = (ignore::DirEntry, String, Lang);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Checked per entry, not per code file: the entries between two
            // code files are what a budget is for.
            if self.visited >= self.budget {
                self.cut_short = true;
                return None;
            }
            let entry = self.inner.next()?;
            self.visited += 1;
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
            let rel = entry
                .path()
                .strip_prefix(&self.root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            let Some(lang) = Lang::from_path(&rel) else {
                continue;
            };
            return Some((entry, rel, lang));
        }
    }
}

/// Build or incrementally update the code graph for a project rooted at
/// `root`. mtime+size short-circuit, blake3 change detection, one
/// transaction for the whole update.
pub fn update(conn: &mut Connection, project: &Node, root: &Path) -> Result<GraphStats> {
    Ok(update_with(conn, project, root, UpdateOptions::default())?.stats)
}

/// [`update`] with options, reporting what was skipped as well as what was
/// done.
pub fn update_with(
    conn: &mut Connection,
    project: &Node,
    root: &Path,
    options: UpdateOptions,
) -> Result<UpdateReport> {
    run_update(conn, project, root, options, &extract::try_extract)
}

/// The parser, as `run_update` sees it. A parameter so a test can make one
/// file's parse fail, which no input can make tree-sitter do on demand.
type Parse<'a> = &'a dyn Fn(Lang, &str) -> std::result::Result<FileExtract, ExtractError>;

/// One file whose edges are to be rebuilt.
struct ResolveJob {
    file_id: i64,
    rel: String,
    lang: Lang,
    fx: FileExtract,
}

fn run_update(
    conn: &mut Connection,
    project: &Node,
    root: &Path,
    options: UpdateOptions,
    parse: Parse,
) -> Result<UpdateReport> {
    let mut report = UpdateReport::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut jobs: Vec<ResolveJob> = Vec::new();
    // Bare names whose set of definitions changed in this run: what an
    // unchanged file must not be calling if its edges are to stay as they are.
    let mut touched: HashSet<String> = HashSet::new();
    // Files this run could not stat, read or parse. Counted once, here, and
    // left out of the re-resolve pass below, which could do no better.
    let mut skipped: HashSet<String> = HashSet::new();
    // Whether a file row was last written by a build older than the rules
    // stamp: see where it is set.
    let mut older_build_wrote = false;

    // One transaction for the whole update: file/symbol upserts and the
    // call-edge rebuild commit together. A crash mid-update used to leave
    // committed file hashes with missing call edges — and the hash
    // short-circuit then skipped those files forever. IMMEDIATE so a
    // concurrent writer waits (busy_timeout) rather than erroring on a
    // DEFERRED read→write lock upgrade.
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let before = live_code_files(&tx, project.id)?;

    for (entry, rel, lang) in CodeWalk::new(root) {
        seen.insert(rel.clone());
        report.stats.files_seen += 1;

        // A file that cannot be stat'ed, read or parsed is skipped and
        // counted, never fatal: one bad file must not roll back the update
        // of every other. It stays in `seen`, so what the graph already
        // knows about it is kept rather than tombstoned.
        let stat = match entry.metadata() {
            Ok(meta) => Stat::of(&meta),
            Err(err) => {
                tracing::warn!(%err, file = %rel, "skipping a file that cannot be stat'ed");
                report.unreadable += 1;
                skipped.insert(rel);
                continue;
            }
        };
        let existing = before.get(&rel);
        if !options.full && existing.is_some_and(|f| f.stat_matches(&stat)) {
            report.stats.unchanged += 1;
            continue;
        }

        let raw = match std::fs::read(entry.path()) {
            Ok(raw) => raw,
            Err(err) => {
                tracing::warn!(%err, file = %rel, "skipping a file that cannot be read");
                report.unreadable += 1;
                skipped.insert(rel);
                continue;
            }
        };
        let content = String::from_utf8_lossy(&raw);
        let hash = blake3::hash(content.as_bytes());
        if let Some(f) = existing.filter(|_| !options.full) {
            if f.hash.as_deref() == Some(hash.as_bytes().as_slice()) {
                // A current row without `mtime_ns` was written by a build
                // that did not store it — and resolved that file's edges by
                // its own rules. Stamping the row hides the evidence, so
                // the fact is kept: this graph is not all this build's work.
                older_build_wrote |= f.mtime_ns.is_none();
                tx.execute(
                    "UPDATE node SET meta = json_set(meta, '$.mtime', ?2, '$.mtime_ns', ?3,
                                                     '$.size', ?4),
                                     updated_at = ?5 WHERE id = ?1",
                    params![f.id, stat.mtime, stat.mtime_ns, stat.size, now_unix()],
                )?;
                report.stats.unchanged += 1;
                continue;
            }
        }

        // The parser failing is not the file being empty. Persisting an empty
        // extraction would delete every symbol of the file and record the
        // new hash, so the loss would also read as current. Leave the row
        // alone: the stored hash no longer matches, and the next update
        // tries again.
        let fx = match parse(lang, &content) {
            Ok(fx) => fx,
            Err(err) => {
                tracing::warn!(%err, file = %rel, "parser failed; keeping the file as last indexed");
                report.parse_failed += 1;
                skipped.insert(rel);
                continue;
            }
        };
        let file_id = match existing {
            Some(f) => Some(f.id),
            None => tombstoned_code_file(&tx, project.id, &rel)?,
        };
        let file_id = persist_file(
            &tx,
            project.id,
            file_id,
            &rel,
            lang,
            hash.as_bytes(),
            &stat,
            &fx,
            &mut report.stats,
            &mut touched,
        )?;
        jobs.push(ResolveJob {
            file_id,
            rel,
            lang,
            fx,
        });
        report.stats.files_indexed += 1;
    }

    // Files gone from disk: soft-delete file + its symbols.
    for (path, file) in &before {
        if seen.contains(path) {
            continue;
        }
        let mut stmt = tx.prepare_cached(
            "SELECT json_extract(meta, '$.name') FROM node
             WHERE kind = 'symbol' AND parent_id = ?1 AND deleted_at IS NULL",
        )?;
        let names = stmt.query_map([file.id], |r| r.get::<_, Option<String>>(0))?;
        for name in names {
            touched.extend(name?);
        }
        tx.execute(
            "UPDATE node SET deleted_at = ?2
             WHERE deleted_at IS NULL AND (id = ?1 OR parent_id = ?1)",
            params![file.id, now_unix()],
        )?;
        report.stats.removed += 1;
    }

    // The files that exist now, by path: what an import can resolve to.
    let file_ids: HashMap<String, i64> = {
        let mut stmt = tx.prepare(&format!(
            "SELECT path, id FROM node WHERE {LIVE_CODE_FILE} ORDER BY id DESC"
        ))?;
        let rows = stmt.query_map([project.id], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut ids = HashMap::new();
        for row in rows {
            if let (Some(path), id) = row? {
                ids.insert(path, id);
            }
        }
        ids
    };
    let files = FileSet::new(file_ids.keys().cloned());

    // Unchanged files whose edges this update has made wrong. `--full`
    // already holds every file.
    if !options.full {
        let files_before = FileSet::new(before.keys().cloned());
        let moved = (!files_before.same_paths(&files)).then_some((&files_before, &files));
        if !touched.is_empty() || moved.is_some() {
            let settled: HashSet<&str> = jobs
                .iter()
                .map(|j| j.rel.as_str())
                .chain(skipped.iter().map(String::as_str))
                .collect();
            for (file_id, rel) in dependents(&tx, project.id, &touched, moved, &settled)? {
                let Some(lang) = Lang::from_path(&rel) else {
                    continue;
                };
                // Its symbols are current; only its edges are not, and the
                // stored call list has no receivers or bindings to resolve
                // them from. So read it again. If that fails its edges stay
                // as they are, which is wrong, so the row is marked as owing
                // an update rather than left looking current.
                let raw = match std::fs::read(root.join(&rel)) {
                    Ok(raw) => raw,
                    Err(err) => {
                        tracing::warn!(%err, file = %rel, "cannot re-resolve a file that cannot be read");
                        report.unreadable += 1;
                        owe_update(&tx, file_id)?;
                        continue;
                    }
                };
                let fx = match parse(lang, &String::from_utf8_lossy(&raw)) {
                    Ok(fx) => fx,
                    Err(err) => {
                        tracing::warn!(%err, file = %rel, "cannot re-resolve a file that does not parse");
                        report.parse_failed += 1;
                        owe_update(&tx, file_id)?;
                        continue;
                    }
                };
                jobs.push(ResolveJob {
                    file_id,
                    rel,
                    lang,
                    fx,
                });
                report.reresolved += 1;
            }
        }
    }

    resolve_calls(&tx, project.id, &jobs, &files, &file_ids, &mut report.stats)?;
    // Every live file went through this build when it was asked to redo
    // them all, or when there was none before it.
    let every_file = options.full || before.is_empty();
    crate::state::mark_built(&tx, project.id, every_file)?;
    if older_build_wrote && !every_file {
        crate::state::unstamp_rules(&tx, project.id)?;
    }
    tx.commit()?;
    Ok(report)
}

/// Make a file row stop passing for current: without the stored hash and
/// the stat's nanoseconds neither freshness test can match, so the next
/// update re-extracts the file and `check` counts it as changed until then.
fn owe_update(conn: &Connection, file_id: i64) -> Result<()> {
    conn.execute(
        "UPDATE node SET content_hash = NULL, meta = json_remove(meta, '$.mtime_ns')
         WHERE id = ?1",
        [file_id],
    )?;
    Ok(())
}

/// Unchanged files whose edges depend on something this update changed:
/// they call a name in `touched`, or one of their imports resolves
/// differently against the files that exist now (`moved` is the file set
/// before and after, present only when it changed).
///
/// Answered from the `calls` and `imports` each file row stores in its
/// meta, so finding them reads no source. A row without those lists was not
/// written by this extractor; it is treated as affected rather than assumed
/// independent. `settled` holds the paths this run has already dealt with.
fn dependents(
    conn: &Connection,
    project_id: i64,
    touched: &HashSet<String>,
    moved: Option<(&FileSet, &FileSet)>,
    settled: &HashSet<&str>,
) -> Result<Vec<(i64, String)>> {
    let needles = name_needles(touched);
    let mut stmt = conn.prepare(&format!(
        "SELECT id, path, meta FROM node WHERE {LIVE_CODE_FILE}"
    ))?;
    let mut rows = stmt.query([project_id])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let id: i64 = r.get(0)?;
        let Some(path) = r.get::<_, Option<String>>(1)? else {
            continue;
        };
        // Already being resolved in full, or already given up on.
        if settled.contains(path.as_str()) {
            continue;
        }
        let meta: String = r.get(2)?;
        if calls_touched(&meta, touched, needles.as_deref())
            || moved.is_some_and(|sets| imports_moved(&path, &meta, sets))
        {
            out.push((id, path));
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(out)
}

/// Most names for which a file's meta is searched as text before it is
/// parsed. Each name is one pass over the text, so past a handful the parse
/// is the cheaper way to ask.
const NEEDLE_MAX: usize = 8;

/// `touched` as the JSON strings they are stored as (`"name"`), for a text
/// search that can rule a file out without parsing its meta. `None` when
/// there are too many names for that to pay, or when one needs escaping and
/// the stored spelling is not the obvious one.
fn name_needles(touched: &HashSet<String>) -> Option<Vec<String>> {
    if touched.is_empty() || touched.len() > NEEDLE_MAX {
        return None;
    }
    touched
        .iter()
        .map(|name| {
            let quoted = serde_json::to_string(name).ok()?;
            (quoted.len() == name.len() + 2).then_some(quoted)
        })
        .collect()
}

/// Does the file whose row meta is `meta` call a name in `touched`?
///
/// On a large project this runs over every file row whenever a symbol is
/// added or removed anywhere, and the call lists are most of the store. So
/// the text is searched first: a name that appears nowhere in it, as any
/// JSON string, is not a callee in it, and the file is ruled out unparsed.
fn calls_touched(meta: &str, touched: &HashSet<String>, needles: Option<&[String]>) -> bool {
    if touched.is_empty() {
        return false;
    }
    if let Some(needles) = needles {
        if meta.contains("\"calls\":") && !needles.iter().any(|n| meta.contains(n.as_str())) {
            return false;
        }
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(meta) else {
        return true;
    };
    match value.get("calls").and_then(|c| c.as_array()) {
        Some(calls) => calls.iter().any(|c| {
            c.get("callee")
                .and_then(|v| v.as_str())
                .is_some_and(|callee| touched.contains(callee))
        }),
        None => true,
    }
}

/// Does any import of the file at `path` resolve to a different file now
/// than it did before the file set changed?
fn imports_moved(path: &str, meta: &str, (before, after): (&FileSet, &FileSet)) -> bool {
    // Most files of most languages import nothing the resolver can place;
    // an empty list is recognisable without parsing.
    if meta.contains("\"imports\":[]") {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(meta) else {
        return true;
    };
    let Some(imports) = value.get("imports").and_then(|i| i.as_array()) else {
        return true;
    };
    let Some(lang) = Lang::from_path(path) else {
        return false;
    };
    imports.iter().any(|i| {
        i.get("source")
            .and_then(|v| v.as_str())
            .is_some_and(|source| {
                resolve_import(lang, path, source, before)
                    != resolve_import(lang, path, source, after)
            })
    })
}

/// What the graph would learn from an update right now, file by file.
/// Drift is defined by content, not timestamps: a touched file whose bytes
/// are unchanged is not drift, because an update would find nothing to do.
///
/// It compares files, not edges, and it does not need to: an update leaves
/// what a from-scratch build would (see the module docs), so a graph with no
/// drift has nothing left to resolve. The one thing it cannot see is a graph
/// resolved by an older build's rules; `graph build --full` is for that.
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
    let stored = live_code_files(conn, project.id)?;

    for (entry, rel, _) in CodeWalk::new(root) {
        seen.insert(rel.clone());

        // A file `update` would skip — it cannot be opened, stat'ed or read
        // — is not drift: an update would learn nothing from it.
        let Some(f) = stored.get(&rel) else {
            if std::fs::File::open(entry.path()).is_ok() {
                drift.added.push(rel);
            }
            continue;
        };
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if f.stat_matches(&Stat::of(&meta)) {
            drift.unchanged += 1;
            continue;
        }
        match f.same_content(entry.path()) {
            Some(true) => drift.unchanged += 1,
            Some(false) => drift.changed.push(rel),
            None => {}
        }
    }

    drift.removed = stored.into_keys().filter(|p| !seen.contains(p)).collect();

    drift.added.sort();
    drift.changed.sort();
    drift.removed.sort();
    Ok(drift)
}

/// Whether an answer read from the graph describes the working tree.
///
/// `graph check` used to be the only thing that looked. Every query answered
/// from whatever was stored, so after an edit `callers` described the old
/// code and looked exactly as right as it did before. So a query asks, and
/// says so in one line when the answer is not [`Freshness::Current`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Every file is as the graph indexed it.
    Current,
    /// This many files were added, changed or removed since.
    Behind(usize),
    /// Not compared: the graph has more files than a query should pay to
    /// look at. `graph check` still does the full comparison.
    Unchecked,
}

impl Freshness {
    /// The line a query adds to its answer, or `None` when there is nothing
    /// to say.
    pub fn note(self) -> Option<String> {
        match self {
            Freshness::Current => None,
            Freshness::Behind(n) => Some(format!(
                "note: the code graph is {n} file(s) behind the working tree — run `mimir graph build`"
            )),
            Freshness::Unchecked => Some(
                "note: the code graph's freshness was not checked (the project is too large to \
                 compare on every query) — `mimir graph check` does"
                    .to_string(),
            ),
        }
    }
}

/// Largest graph, in code files, that a query compares against the working
/// tree. Each stored file costs a stat and the reading of its row, so the
/// cost grows with the graph: about 6 µs a file, measured at 29 ms for a
/// graph of this size whose files are large (9 KB of stored calls each).
/// Above it the comparison is skipped and the answer says so.
pub const FRESHNESS_MAX_FILES: usize = 5_000;

/// How many directory entries, as a multiple of the file budget, the search
/// for *new* files may visit. A tree can hold far more entries than code
/// files (assets, data, a vendored dependency tree).
const FRESHNESS_WALK_FACTOR: usize = 5;

/// How far behind the working tree a project's graph is, at a cost a query
/// can afford. Read-only, like [`check`].
pub fn freshness(conn: &Connection, project: &Node, root: &Path) -> Result<Freshness> {
    freshness_within(conn, project, root, FRESHNESS_MAX_FILES)
}

/// [`freshness`] with the file budget given.
///
/// Two passes, because the two halves of the question cost differently.
/// What became of the files the graph knows is answered by a stat of each,
/// bounded by the size of the graph and nothing else. Which files are new
/// can only be answered by walking the tree, whose size has nothing to do
/// with the graph's: fifty source files can sit beside a hundred thousand
/// assets. So the walk has a budget of entries, and running out of it is
/// not a reason to say nothing — the first pass stands, with whatever
/// additions the walk did reach.
///
/// When the walk completes, the answer is [`check`]'s count.
pub fn freshness_within(
    conn: &Connection,
    project: &Node,
    root: &Path,
    max_files: usize,
) -> Result<Freshness> {
    // Counting stops one past the budget: on the projects this turns away,
    // counting every row would itself be most of what is being avoided.
    let files: i64 = conn.query_row(
        &format!("SELECT count(*) FROM (SELECT 1 FROM node WHERE {LIVE_CODE_FILE} LIMIT ?2)"),
        params![project.id, max_files.saturating_add(1) as i64],
        |r| r.get(0),
    )?;
    if files as usize > max_files {
        return Ok(Freshness::Unchecked);
    }
    let stored = live_code_files(conn, project.id)?;

    // The files the graph knows: gone, edited, or as they were. A file that
    // is there and cannot be read is not drift; no build could act on it.
    let mut behind: HashSet<&str> = HashSet::new();
    for (rel, row) in &stored {
        let path = root.join(rel);
        let current = match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() => {
                row.stat_matches_as_stored(&Stat::of(&meta))
                    || row.same_content(&path).unwrap_or(true)
            }
            _ => false,
        };
        if !current {
            behind.insert(rel.as_str());
        }
    }

    // The files it does not know.
    let mut added = 0usize;
    let mut walked: HashSet<&str> = HashSet::new();
    let mut walk = CodeWalk::within(root, max_files.saturating_mul(FRESHNESS_WALK_FACTOR));
    for (entry, rel, _) in walk.by_ref() {
        match stored.get_key_value(&rel) {
            Some((known, _)) => {
                walked.insert(known.as_str());
            }
            None if std::fs::File::open(entry.path()).is_ok() => added += 1,
            None => {}
        }
    }
    // A stored file a complete walk did not come across is out of scope
    // (ignored since it was indexed, say): an update would drop it.
    if !walk.cut_short {
        behind.extend(
            stored
                .keys()
                .map(String::as_str)
                .filter(|rel| !walked.contains(rel)),
        );
    }

    Ok(match behind.len() + added {
        0 => Freshness::Current,
        n => Freshness::Behind(n),
    })
}

/// Whether one file still has the bytes the graph indexed: `Some(false)`
/// when it was edited or removed since, `None` when the graph has no live
/// row for it. For a reader about to slice the file by stored line numbers.
pub fn file_is_current(
    conn: &Connection,
    project_id: i64,
    root: &Path,
    rel: &str,
) -> Result<Option<bool>> {
    let row = conn
        .query_row(
            &format!(
                "SELECT id, content_hash, meta FROM node
                 WHERE {LIVE_CODE_FILE} AND path = ?2 ORDER BY id LIMIT 1"
            ),
            params![project_id, rel],
            |r| {
                let (mtime, mtime_ns, size) =
                    stored_stat(r.get_ref(2)?.as_str().unwrap_or_default());
                Ok(FileRow {
                    id: r.get(0)?,
                    hash: r.get(1)?,
                    mtime,
                    mtime_ns,
                    size,
                })
            },
        )
        .optional()?;
    let Some(row) = row else {
        return Ok(None);
    };
    let path = root.join(rel);
    let Ok(meta) = std::fs::metadata(&path) else {
        return Ok(Some(false));
    };
    if row.stat_matches(&Stat::of(&meta)) {
        return Ok(Some(true));
    }
    Ok(Some(row.same_content(&path).unwrap_or(false)))
}

/// The row of a code file that is not live — one this project had and lost
/// — so a file coming back revives it, and its symbols with their links,
/// instead of starting over.
fn tombstoned_code_file(conn: &Connection, project_id: i64, rel: &str) -> Result<Option<i64>> {
    Ok(conn
        .query_row(
            "SELECT id FROM node
             WHERE kind = 'file' AND project_id = ?1 AND path = ?2
               AND collection_id IS NULL
             ORDER BY id LIMIT 1",
            params![project_id, rel],
            |r| r.get(0),
        )
        .optional()?)
}

/// Whether anyone linked to this symbol: an edge the code did not put there.
fn has_authored_edge(conn: &Connection, id: i64) -> Result<bool> {
    // Which relations are derived from the code itself is decided once, by
    // `Rel::origin`: an edge of any other relation was put there by someone,
    // and is what makes a vanished symbol worth keeping as a tombstone.
    let derived = crate::prune::derived_rels_sql();
    Ok(conn.query_row(
        &format!(
            "SELECT EXISTS (SELECT 1 FROM edge
                            WHERE (src = ?1 OR dst = ?1) AND rel NOT IN ({derived}))"
        ),
        [id],
        |r| r.get(0),
    )?)
}

#[allow(clippy::too_many_arguments)]
fn persist_file(
    conn: &Connection,
    project_id: i64,
    existing: Option<i64>,
    rel: &str,
    lang: Lang,
    hash: &[u8],
    stat: &Stat,
    fx: &FileExtract,
    stats: &mut GraphStats,
    touched: &mut HashSet<String>,
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
    // `calls` and `imports` are what `dependents` reads to find the files an
    // update elsewhere has made stale.
    let file_meta = serde_json::json!({
        "mtime": stat.mtime, "mtime_ns": stat.mtime_ns, "size": stat.size,
        "imports": imports_json, "calls": calls_json,
    });

    let file_id = match existing {
        Some(id) => {
            conn.execute(
                "UPDATE node SET content_hash = ?2, meta = ?3, lang = ?4,
                                 updated_at = ?5, deleted_at = NULL WHERE id = ?1",
                params![id, hash, file_meta.to_string(), lang.name(), now_unix()],
            )?;
            id
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

    // Upsert symbols by stable_id; collect survivors to prune the rest. A
    // symbol whose row is new, or comes back from a tombstone, changes what
    // its bare name can resolve to — and so does one that goes away below.
    // Two definitions with one identity land on one row, in source order, so
    // the row describes the last of them (see the module docs).
    let mut kept: HashSet<i64> = HashSet::new();
    for sym in &fx.symbols {
        let sid = stable_id(project_id, rel, &sym.qualified, sym.kind);
        let body = match &sym.doc {
            Some(d) => format!("{}\n{d}", sym.signature),
            None => sym.signature.clone(),
        };
        let meta = serde_json::json!({"stable_id": sid, "name": sym.name});
        let existing: Option<(i64, bool)> = conn
            .query_row(
                "SELECT id, deleted_at IS NULL FROM node
                 WHERE kind = 'symbol' AND json_extract(meta, '$.stable_id') = ?1",
                [&sid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let id = match existing {
            Some((id, was_live)) => {
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
                if !was_live {
                    touched.insert(sym.name.clone());
                }
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
                touched.insert(sym.name.clone());
                store::insert_node(conn, new)?.id
            }
        };
        kept.insert(id);
    }
    stats.symbols += kept.len();

    // Symbols that vanished from this file. Derived data, so normally
    // hard-deleted (edges cascade, embeddings cascade). But a symbol someone
    // linked a memory to is evidence about that memory: deleting the row
    // deletes the link, and the memory would read as never having been
    // attached to anything. It is tombstoned instead, so the memory reads
    // `stale` (see `mimir_core::grounding`), and if the symbol returns under
    // the same identity the upsert above revives the row with its links.
    {
        let mut stmt = conn.prepare_cached(
            "SELECT id, json_extract(meta, '$.name'), deleted_at IS NULL FROM node
             WHERE kind = 'symbol' AND parent_id = ?1",
        )?;
        let all: Vec<(i64, Option<String>, bool)> = stmt
            .query_map([file_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for (id, name, live) in all {
            if kept.contains(&id) {
                continue;
            }
            if !has_authored_edge(conn, id)? {
                conn.execute("DELETE FROM node WHERE id = ?1", [id])?;
            } else if live {
                conn.execute(
                    "UPDATE node SET deleted_at = ?2 WHERE id = ?1",
                    params![id, now_unix()],
                )?;
            }
            if live {
                touched.extend(name);
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
    /// The symbol kind (`function`, `struct`, …), to choose between two
    /// symbols of one file that share a qualified name.
    kind: String,
}

impl SymRef {
    /// Where this symbol stands among the symbols of its file that share its
    /// qualified name; the lowest is the one calls are attributed to.
    ///
    /// Something callable comes first: when `interface Options` and
    /// `function Options()` sit in one file, a call to `Options()` is a call
    /// to the function, and a call made inside `Options` is made by it. Then
    /// the kind's name, which is arbitrary and is the point: the order must
    /// depend on nothing that can change while the symbols stay the same.
    /// Not the position in the file — swapping the two definitions adds and
    /// removes no symbol, so no caller in another file would be brought back
    /// to follow the new first one — and not the row id, which differs
    /// between a store updated in place and one built from nothing.
    fn standing(&self) -> (bool, &str) {
        let callable = matches!(
            self.kind.as_str(),
            "function" | "method" | "constructor" | "procedure"
        );
        (!callable, self.kind.as_str())
    }
}

/// Rebuild the call and import edges that start in `jobs`' files, against
/// the project's whole symbol table and the files that exist now.
///
/// Everything an edge of a file depends on is that file's own extraction,
/// the symbols carrying the names it calls, and the files its imports can
/// reach — so a file's edges are rebuilt as a unit, and `run_update` passes
/// every file for which any of the three changed.
fn resolve_calls(
    conn: &Connection,
    project_id: i64,
    jobs: &[ResolveJob],
    files: &FileSet,
    file_ids: &HashMap<String, i64>,
    stats: &mut GraphStats,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    // Project-wide symbol table (one query, in-memory buckets). Nothing
    // below may depend on the order the rows arrive in: row ids, and with
    // them the scan order, differ between an incremental store and a fresh
    // one, and the two must resolve alike.
    let mut stmt = conn.prepare(
        "SELECT id, json_extract(meta, '$.name'), title, path, subkind FROM node
         WHERE kind = 'symbol' AND project_id = ?1 AND deleted_at IS NULL",
    )?;
    let symbols: Vec<SymRef> = stmt
        .query_map([project_id], |r| {
            Ok(SymRef {
                id: r.get(0)?,
                name: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                qualified: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                kind: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    // For call edges a (file, qualified name) is one endpoint: a call site
    // knows its caller by qualified name only, and names its callee the same
    // way. Symbols of one file share a qualified name only by differing in
    // kind, and one of them stands for the name, as caller and as callee
    // (`SymRef::standing` says which).
    let mut endpoint: HashMap<(&str, &str), &SymRef> = HashMap::new();
    for s in &symbols {
        endpoint
            .entry((s.path.as_str(), s.qualified.as_str()))
            .and_modify(|held| {
                if s.standing() < held.standing() {
                    *held = s;
                }
            })
            .or_insert(s);
    }
    let mut by_name: HashMap<&str, Vec<&SymRef>> = HashMap::new();
    for s in endpoint.values() {
        // A NULL/empty name would bucket under "" and attract phantom
        // call edges from any unresolvable callee.
        if !s.name.is_empty() {
            by_name.entry(s.name.as_str()).or_default().push(s);
        }
    }

    for job in jobs {
        let (file_id, rel, fx) = (job.file_id, job.rel.as_str(), &job.fx);
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

        // Import map for the imported-name tier + file→file import edges.
        let mut import_target: HashMap<&str, String> = HashMap::new();
        let mut imported: HashSet<i64> = HashSet::new();
        for imp in &fx.imports {
            if let Some(target) = resolve_import(job.lang, rel, &imp.source, files) {
                if let Some(dst) = file_ids.get(&target) {
                    if *dst != file_id && imported.insert(*dst) {
                        store::link(conn, file_id, *dst, Rel::Imports, 1.0)?;
                        stats.imports += 1;
                    }
                }
                import_target.insert(imp.local.as_str(), target);
            }
        }

        // One function can call the same target from several sites, each
        // resolved on its own evidence. The edge carries the strongest: a
        // later guess must not overwrite an earlier receiver-typed match.
        let mut edges: BTreeMap<(i64, i64), CallTier> = BTreeMap::new();

        for call in &fx.calls {
            if call.caller.is_empty() {
                continue; // top-level statements have no source symbol
            }
            let Some(src) = endpoint.get(&(rel, call.caller.as_str())).map(|s| s.id) else {
                continue;
            };
            let candidates = by_name.get(call.callee.as_str());
            let Some(candidates) = candidates else {
                continue;
            };
            // A function is never its own callee here: a recursive call
            // says nothing about what else carries the name.
            let others = |keep: &dyn Fn(&SymRef) -> bool| -> Vec<&SymRef> {
                candidates
                    .iter()
                    .copied()
                    .filter(|c| c.id != src && keep(c))
                    .collect()
            };
            // Receiver-typed: the receiver's type names exactly one method.
            // This is the only tier that can tell two same-named methods
            // apart, so it runs before the same-file shortcut.
            if let Some(dst) = receiver_target(call, &binds, &fx.bindings, candidates, src, rel) {
                keep_strongest(&mut edges, src, dst, CallTier::ReceiverTyped);
                stats.calls_resolved += 1;
                stats.calls_typed += 1;
                continue;
            }
            // Same file. The name being defined here settles where to look,
            // not which one: several definitions are still a guess.
            let local = others(&|c| c.path == rel);
            if !local.is_empty() {
                settle(&mut edges, stats, src, &local, CallTier::SameFile);
                continue;
            }
            // Imported name → the file the import resolved to.
            if let Some(target) = import_target.get(call.callee.as_str()) {
                let from_import = others(&|c| c.path == *target);
                if !from_import.is_empty() {
                    settle(&mut edges, stats, src, &from_import, CallTier::ImportedName);
                    continue;
                }
            }
            // Global by name — honest about ambiguity.
            settle(
                &mut edges,
                stats,
                src,
                &others(&|_| true),
                CallTier::UniqueGlobal,
            );
        }

        for ((src, dst), tier) in edges {
            link_call(conn, src, dst, tier)?;
        }
    }
    Ok(())
}

/// One call site and its candidates under one rule: a single candidate is a
/// match at that rule's `tier`, two or three are a guess shared between
/// them, and more is too ambiguous to be useful.
fn settle(
    edges: &mut BTreeMap<(i64, i64), CallTier>,
    stats: &mut GraphStats,
    src: i64,
    candidates: &[&SymRef],
    tier: CallTier,
) {
    match candidates.len() {
        0 => {}
        1 => {
            keep_strongest(edges, src, candidates[0].id, tier);
            stats.calls_resolved += 1;
        }
        n if n <= 3 => {
            for c in candidates {
                keep_strongest(edges, src, c.id, CallTier::Ambiguous { candidates: n });
            }
            stats.calls_heuristic += 1;
        }
        _ => {}
    }
}

/// Record `tier` for the edge `src → dst` unless a stronger one is already
/// there.
fn keep_strongest(edges: &mut BTreeMap<(i64, i64), CallTier>, src: i64, dst: i64, tier: CallTier) {
    edges
        .entry((src, dst))
        .and_modify(|held| {
            if tier.outranks(*held) {
                *held = tier;
            }
        })
        .or_insert(tier);
}

/// Receivers that mean "the type I am defined on" rather than a name in
/// scope. `Self` is Rust's type-position spelling; the rest are values.
/// PHP's `$this` is covered by `this`: `receiver_type` strips the sigil
/// before it looks here.
const SELF_RECEIVERS: &[&str] = &["self", "this", "Self"];

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
    let exact = format!("{ty}{SCOPE_SEP}{}", call.callee);
    let suffix = format!("{SCOPE_SEP}{exact}");
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
        caller.rsplit_once(SCOPE_SEP)?.0.to_string()
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
        match scope.rsplit_once(SCOPE_SEP) {
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
        || scope.starts_with(&format!("{ty}{SCOPE_SEP}"))
        || scope.ends_with(&format!("{SCOPE_SEP}{ty}"))
        || scope.contains(&format!("{SCOPE_SEP}{ty}{SCOPE_SEP}"))
}

/// Write one call edge. Its weight and meta are the tier's, never chosen
/// here. The resolver has already kept the strongest tier per pair; the
/// conflict clause holds the same line against a row it did not wipe, so no
/// path through here can lower an edge.
fn link_call(conn: &Connection, src: i64, dst: i64, tier: CallTier) -> Result<()> {
    conn.execute(
        "INSERT INTO edge (src, dst, rel, weight, meta, created_at)
         VALUES (?1, ?2, 'calls', ?3, json_object('resolved', ?4, 'tier', ?5), ?6)
         ON CONFLICT(src, dst, rel) DO UPDATE SET
           weight = excluded.weight, meta = excluded.meta
         WHERE excluded.weight > edge.weight",
        params![
            src,
            dst,
            tier.weight(),
            tier.is_resolved(),
            tier.label(),
            now_unix()
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve_symbol;
    use mimir_core::grounding::{grounding, Grounding};
    use mimir_core::Mimir;

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn project(conn: &Connection, root: &Path) -> Node {
        store::ensure_project(conn, &root.to_string_lossy(), "test-proj").unwrap()
    }

    /// util.rs defines `slugify`; main.rs has `main` calling `run` calling it.
    fn setup() -> (tempfile::TempDir, Mimir, Node) {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/util.rs", "pub fn slugify() {}\n");
        write(
            dir.path(),
            "src/main.rs",
            "fn main() { run(); }\n\nfn run() { slugify(); }\n",
        );
        let mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        (dir, mimir, proj)
    }

    fn note(conn: &Connection, project_id: i64, text: &str) -> Node {
        match mimir_core::memory::remember(
            conn,
            mimir_core::memory::Remember {
                text: text.into(),
                mtype: mimir_core::model::MemoryType::Note,
                tags: vec![],
                project_id: Some(project_id),
                force: false,
                actor: mimir_core::model::Actor::System,
            },
        )
        .unwrap()
        {
            mimir_core::memory::RememberOutcome::Created(n) => n,
            other => panic!("expected a new memory, got {other:?}"),
        }
    }

    /// Call edges as `caller -> callee weight meta`, sorted.
    fn call_edges(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare(
                "SELECT s.title || ' -> ' || d.title || ' ' || printf('%.2f', e.weight)
                        || ' ' || e.meta
                 FROM edge e JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
                 WHERE e.rel = 'calls' AND s.deleted_at IS NULL AND d.deleted_at IS NULL
                 ORDER BY 1",
            )
            .unwrap();
        let rows = stmt.query_map([], |r| r.get(0)).unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    }

    /// Import edges as `importer -> imported`, sorted.
    fn import_edges(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare(
                "SELECT s.path || ' -> ' || d.path FROM edge e
                 JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
                 WHERE e.rel = 'imports' ORDER BY 1",
            )
            .unwrap();
        let rows = stmt.query_map([], |r| r.get(0)).unwrap();
        rows.collect::<rusqlite::Result<_>>().unwrap()
    }

    fn row_exists(conn: &Connection, id: i64) -> bool {
        conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM node WHERE id = ?1)",
            [id],
            |r| r.get(0),
        )
        .unwrap()
    }

    /// A parser that fails on any source carrying `marker`, and otherwise is
    /// the real one.
    fn failing_on(
        marker: &'static str,
    ) -> impl Fn(Lang, &str) -> std::result::Result<FileExtract, ExtractError> {
        move |lang, source| {
            if source.contains(marker) {
                Err(ExtractError::Parse)
            } else {
                extract::try_extract(lang, source)
            }
        }
    }

    /// The parser failing on a file is not the file becoming empty. It used
    /// to be persisted as exactly that: every symbol of the file deleted,
    /// the new hash stored, and the graph reporting itself current.
    #[test]
    fn a_parse_failure_keeps_the_file_as_it_was_and_is_retried() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let slug = resolve_symbol(&mimir.conn, proj.id, "slugify").unwrap();
        let before = call_edges(&mimir.conn);

        write(
            dir.path(),
            "src/util.rs",
            "// UNPARSEABLE\npub fn slugify() {}\npub fn added() {}\n",
        );
        write(
            dir.path(),
            "src/new.rs",
            "// UNPARSEABLE\npub fn fresh() {}\n",
        );
        let report = run_update(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions::default(),
            &failing_on("UNPARSEABLE"),
        )
        .unwrap();
        assert_eq!(report.parse_failed, 2, "{report:?}");
        assert_eq!(report.stats.files_indexed, 0, "{report:?}");

        // What was known is still known, under the same row.
        let kept = resolve_symbol(&mimir.conn, proj.id, "slugify").unwrap();
        assert_eq!(kept.id, slug.id);
        assert_eq!(call_edges(&mimir.conn), before);
        assert!(resolve_symbol(&mimir.conn, proj.id, "added").is_err());
        assert!(resolve_symbol(&mimir.conn, proj.id, "fresh").is_err());

        // And the graph does not claim to be current: both files are still
        // owed an update.
        let drift = check(&mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(drift.changed, vec!["src/util.rs"]);
        assert_eq!(drift.added, vec!["src/new.rs"]);

        // Once the parser works, the next ordinary update picks them up.
        let report =
            update_with(&mut mimir.conn, &proj, dir.path(), UpdateOptions::default()).unwrap();
        assert_eq!(report.parse_failed, 0);
        assert_eq!(report.stats.files_indexed, 2);
        assert!(resolve_symbol(&mimir.conn, proj.id, "added").is_ok());
        assert!(resolve_symbol(&mimir.conn, proj.id, "fresh").is_ok());
        assert!(!check(&mimir.conn, &proj, dir.path()).unwrap().is_stale());
    }

    /// An unchanged file whose edges an edit elsewhere made wrong, and which
    /// cannot be parsed when the update comes back to it: its edges cannot
    /// be fixed now, so it must not go on looking current. It is owed an
    /// update, `check` says so, and the next build that can parse it ends
    /// where a fresh build would.
    #[test]
    fn a_dependent_that_cannot_be_re_resolved_is_owed_an_update() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/util.rs", "pub fn slugify() {}\n");
        write(
            dir.path(),
            "src/main.rs",
            "// UNPARSEABLE\nfn main() { slugify(); }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let certain = call_edges(&mimir.conn);
        assert_eq!(certain.len(), 1, "{certain:?}");
        assert!(certain[0].contains("unique-global"), "{certain:?}");

        // A second `slugify` makes main.rs's call a guess; main.rs is
        // unchanged, and this time its parse fails.
        write(dir.path(), "src/more.rs", "pub fn slugify() -> u8 { 2 }\n");
        let report = run_update(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions::default(),
            &failing_on("UNPARSEABLE"),
        )
        .unwrap();
        assert_eq!(
            (report.parse_failed, report.reresolved),
            (1, 0),
            "{report:?}"
        );
        assert_eq!(call_edges(&mimir.conn), certain, "left as it was");
        assert_eq!(
            check(&mimir.conn, &proj, dir.path()).unwrap().changed,
            vec!["src/main.rs"],
            "and not passed off as current"
        );

        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let guessed = call_edges(&mimir.conn);
        assert_eq!(guessed.len(), 2, "{guessed:?}");
        assert!(
            guessed.iter().all(|e| e.contains("ambiguous")),
            "{guessed:?}"
        );
        assert!(!check(&mimir.conn, &proj, dir.path()).unwrap().is_stale());
    }

    /// A changed file that does not parse has been given up on for this run.
    /// When the same run also changes something it calls, it is a dependent
    /// by its stored call list — and must not be read, failed and counted a
    /// second time.
    #[test]
    fn a_file_that_fails_to_parse_is_counted_once_even_as_a_dependent() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        write(
            dir.path(),
            "src/main.rs",
            "// UNPARSEABLE\nfn main() { run(); }\n\nfn run() { slugify(); }\n",
        );
        write(dir.path(), "src/more.rs", "pub fn slugify() -> u8 { 2 }\n");
        let report = run_update(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions::default(),
            &failing_on("UNPARSEABLE"),
        )
        .unwrap();
        assert_eq!(
            (
                report.parse_failed,
                report.reresolved,
                report.stats.files_indexed
            ),
            (1, 0, 1),
            "{report:?}"
        );
        assert_eq!(
            check(&mimir.conn, &proj, dir.path()).unwrap().changed,
            vec!["src/main.rs"]
        );
    }

    /// An import edge is between two files, however many names one takes
    /// from the other; the count is of edges.
    #[test]
    fn two_names_imported_from_one_file_are_one_import_edge() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/util.rs", "pub fn a() {}\npub fn b() {}\n");
        write(
            dir.path(),
            "src/main.rs",
            "use crate::util::a;\nuse crate::util::b;\n\nfn main() { a(); b(); }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.imports, 1, "{stats:?}");
        assert_eq!(import_edges(&mimir.conn), ["src/main.rs -> src/util.rs"]);
    }

    /// One file that cannot be read is skipped and counted, as unreadable
    /// directory entries always were. It used to abort the update and roll
    /// back every other file with it.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_file_is_skipped_and_counted() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, mut mimir, proj) = setup();
        write(dir.path(), "src/locked.rs", "pub fn secret() {}\n");
        let locked = dir.path().join("src/locked.rs");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::File::open(&locked).is_ok() {
            // Running as a user permissions do not bind (root): there is no
            // unreadable file to test with.
            return;
        }

        let report =
            update_with(&mut mimir.conn, &proj, dir.path(), UpdateOptions::default()).unwrap();
        assert_eq!(report.unreadable, 1, "{report:?}");
        assert_eq!(report.stats.files_seen, 3);
        assert_eq!(report.stats.files_indexed, 2, "the readable files went in");
        assert!(resolve_symbol(&mimir.conn, proj.id, "slugify").is_ok());
        assert!(resolve_symbol(&mimir.conn, proj.id, "secret").is_err());

        // An update can learn nothing from it, so it is not drift either —
        // `graph check` must not fail forever over a file no build can read.
        let drift = check(&mimir.conn, &proj, dir.path()).unwrap();
        assert!(!drift.is_stale(), "{drift:?}");

        // Readable again: an ordinary update indexes it.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            check(&mimir.conn, &proj, dir.path()).unwrap().added,
            vec!["src/locked.rs"]
        );
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(resolve_symbol(&mimir.conn, proj.id, "secret").is_ok());
    }

    /// A symbol someone linked a memory to does not vanish with its code:
    /// it is tombstoned, the memory reads `stale`, and the same symbol
    /// coming back is the same row with the link still on it.
    #[test]
    fn a_removed_symbol_someone_linked_goes_stale_and_can_come_back() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let run = resolve_symbol(&mimir.conn, proj.id, "run").unwrap();
        let memory = note(&mimir.conn, proj.id, "run must stay idempotent");
        store::link(&mimir.conn, memory.id, run.id, Rel::About, 1.0).unwrap();
        assert_eq!(
            grounding(&mimir.conn, memory.id).unwrap().label(),
            "grounded"
        );

        // `run` leaves a file that stays.
        write(dir.path(), "src/main.rs", "fn main() {}\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        assert!(resolve_symbol(&mimir.conn, proj.id, "run").is_err());
        let tombstone = store::get_node(&mimir.conn, run.id).unwrap();
        assert!(tombstone.deleted_at.is_some(), "kept, as a tombstone");
        assert_eq!(store::edges_of(&mimir.conn, memory.id).unwrap().len(), 1);
        let g = grounding(&mimir.conn, memory.id).unwrap();
        assert_eq!(g.label(), "stale");
        assert_eq!(g.target(), Some((Kind::Symbol, "run")));
        assert!(matches!(g, Grounding::Stale { .. }));
        // A tombstone takes no part in the graph.
        assert!(
            call_edges(&mimir.conn).is_empty(),
            "{:?}",
            call_edges(&mimir.conn)
        );

        // Still gone on the next update: it stays a tombstone.
        write(dir.path(), "src/main.rs", "fn main() { }\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(row_exists(&mimir.conn, run.id));

        // Back under the same name and kind: the same row, grounded again.
        write(
            dir.path(),
            "src/main.rs",
            "fn main() { run(); }\n\n\nfn run() { slugify(); }\n",
        );
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let back = resolve_symbol(&mimir.conn, proj.id, "run").unwrap();
        assert_eq!(back.id, run.id);
        assert_eq!(back.uid, run.uid);
        assert_eq!(
            grounding(&mimir.conn, memory.id).unwrap().label(),
            "grounded"
        );
        assert_eq!(call_edges(&mimir.conn).len(), 2, "main → run → slugify");
    }

    /// With nothing but its own call edges on it, a removed symbol is
    /// derived data and is deleted outright, as before.
    #[test]
    fn a_removed_symbol_nobody_linked_is_deleted() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let run = resolve_symbol(&mimir.conn, proj.id, "run").unwrap();
        assert!(
            !store::edges_of(&mimir.conn, run.id).unwrap().is_empty(),
            "run has call edges, which are not a reason to keep it"
        );

        write(dir.path(), "src/main.rs", "fn main() {}\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(!row_exists(&mimir.conn, run.id));
    }

    /// A tombstone kept for a link is dropped once the link is gone.
    #[test]
    fn a_tombstone_goes_when_its_last_link_does() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let run = resolve_symbol(&mimir.conn, proj.id, "run").unwrap();
        let memory = note(&mimir.conn, proj.id, "run must stay idempotent");
        store::link(&mimir.conn, memory.id, run.id, Rel::About, 1.0).unwrap();

        write(dir.path(), "src/main.rs", "fn main() {}\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(row_exists(&mimir.conn, run.id));

        mimir
            .conn
            .execute("DELETE FROM edge WHERE src = ?1", [memory.id])
            .unwrap();
        write(dir.path(), "src/main.rs", "fn main() { }\n");
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(!row_exists(&mimir.conn, run.id));
    }

    /// Two trait impls give one type the same method twice. They are one
    /// identity (path, qualified name, kind) and one row, which describes the
    /// later of the two — and the count says one, where it used to say two.
    /// Reordering them moves nothing: a memory linked to the name stays on it.
    #[test]
    fn two_definitions_of_one_name_and_kind_share_a_row() {
        let dir = tempfile::tempdir().unwrap();
        let display = "impl A for T { fn show(&self) -> u8 { 1 } }\n";
        let debug = "impl B for T { fn show(&self) -> u16 { 2 } }\n";
        write(
            dir.path(),
            "src/t.rs",
            &format!("pub struct T;\n{display}{debug}"),
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.symbols, 2, "T and one `show`: rows written");

        let rows = |conn: &Connection| -> Vec<(i64, i64, String, String)> {
            let mut stmt = conn
                .prepare(
                    "SELECT id, span_start, body, json_extract(meta, '$.stable_id') FROM node
                     WHERE kind = 'symbol' AND title = 'T::show'",
                )
                .unwrap();
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                .unwrap();
            rows.collect::<rusqlite::Result<_>>().unwrap()
        };
        let first = rows(&mimir.conn);
        assert_eq!(first.len(), 1, "{first:?}");
        assert_eq!(
            (first[0].1, first[0].2.contains("u16")),
            (3, true),
            "the later one"
        );
        assert_eq!(
            first[0].3,
            stable_id(proj.id, "src/t.rs", "T::show", "method")
        );
        // The name is something to ask about, not an ambiguity.
        let show = resolve_symbol(&mimir.conn, proj.id, "T::show").unwrap();
        assert_eq!(show.id, first[0].0);
        let memory = note(&mimir.conn, proj.id, "show must never allocate");
        store::link(&mimir.conn, memory.id, show.id, Rel::About, 1.0).unwrap();

        // Swapped: the same row, now describing the other definition.
        write(
            dir.path(),
            "src/t.rs",
            &format!("pub struct T;\n{debug}{display}"),
        );
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.symbols, 2);
        let swapped = rows(&mimir.conn);
        assert_eq!(swapped.len(), 1);
        assert_eq!(swapped[0].0, first[0].0, "same row");
        assert!(swapped[0].2.contains("u8"), "{swapped:?}");
        assert_eq!(
            store::edges_of(&mimir.conn, memory.id).unwrap()[0].dst,
            show.id
        );

        // One of the two goes: the name is still defined, so nothing is lost.
        write(dir.path(), "src/t.rs", &format!("pub struct T;\n{debug}"));
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        let one = rows(&mimir.conn);
        assert_eq!((one.len(), one[0].0), (1, first[0].0));
        assert_eq!(
            grounding(&mimir.conn, memory.id).unwrap().label(),
            "grounded"
        );
    }

    /// A name defined twice under one identity (a `#[cfg]` pair) is one thing
    /// to call, and the call is resolved. Only a name shared by *different*
    /// qualified names makes a call a guess.
    #[test]
    fn a_call_to_a_name_defined_twice_under_one_identity_is_resolved() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/os.rs",
            "#[cfg(unix)]\npub fn home() -> u8 { 1 }\n\
             #[cfg(windows)]\npub fn home() -> u8 { 2 }\n\
             pub fn run() -> u8 { home() }\n",
        );
        write(dir.path(), "src/main.rs", "fn main() { home(); }\n");
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.symbols, 3, "home, run, main");
        assert_eq!(
            call_edges(&mimir.conn),
            [
                r#"main -> home 0.80 {"resolved":1,"tier":"unique-global"}"#,
                r#"run -> home 1.00 {"resolved":1,"tier":"same-file"}"#,
            ]
        );
    }

    /// A type and a function of one name in one file: a call to the name is
    /// a call to the function, and a call made in its body is made by the
    /// function. Which of the two comes first in the file changes nothing.
    ///
    /// The types are chosen on both sides of `function` in the alphabet.
    /// The rule falls back on the kind's name, so `interface` alone would
    /// lose to `function` with the "callable first" half deleted; `class`
    /// and `enum` would win.
    #[test]
    fn a_call_to_a_name_shared_by_a_type_and_a_function_lands_on_the_function() {
        const FUNCTION: &str = "export function Options() { return make(); }\n";
        const USE_TS: &str =
            "export function run() { return Options(); }\nexport function make() { return 1; }\n";
        let ts_types = [
            "export interface Options { a: number }\n",
            "export class Options { a = 1 }\n",
            "export enum Options { A }\n",
        ];
        let mut cases: Vec<(&str, String, &str, &str)> = Vec::new();
        for ty in ts_types {
            cases.push((
                "lib/both.ts",
                format!("{ty}{FUNCTION}"),
                "lib/use.ts",
                USE_TS,
            ));
            cases.push((
                "lib/both.ts",
                format!("{FUNCTION}{ty}"),
                "lib/use.ts",
                USE_TS,
            ));
        }
        // Legal Rust: a type and a function live in different namespaces.
        const RUST_FN: &str = "pub fn Options() -> u8 { make() }\n";
        const USE_RS: &str = "pub fn run() -> u8 { Options() }\npub fn make() -> u8 { 1 }\n";
        for ty in ["pub enum Options { A }\n", "pub struct Options { a: u8 }\n"] {
            cases.push((
                "src/both.rs",
                format!("{ty}{RUST_FN}"),
                "src/use.rs",
                USE_RS,
            ));
            cases.push((
                "src/both.rs",
                format!("{RUST_FN}{ty}"),
                "src/use.rs",
                USE_RS,
            ));
        }

        for (both_path, both, use_path, uses) in cases {
            let dir = tempfile::tempdir().unwrap();
            write(dir.path(), both_path, &both);
            write(dir.path(), use_path, uses);
            let mut mimir = Mimir::open_in_memory().unwrap();
            let proj = project(&mimir.conn, dir.path());
            update(&mut mimir.conn, &proj, dir.path()).unwrap();
            let kinds: Vec<String> = mimir
                .conn
                .prepare(
                    "SELECT subkind FROM node
                     WHERE kind = 'symbol' AND title = 'Options' ORDER BY 1",
                )
                .unwrap()
                .query_map([], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(kinds.len(), 2, "both definitions are symbols: {both}");
            let mut stmt = mimir
                .conn
                .prepare(
                    "SELECT s.title || '[' || s.subkind || '] -> ' || d.title || '[' || d.subkind || ']'
                     FROM edge e JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
                     WHERE e.rel = 'calls' ORDER BY 1",
                )
                .unwrap();
            let edges: Vec<String> = stmt
                .query_map([], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(
                edges,
                [
                    "Options[function] -> make[function]",
                    "run[function] -> Options[function]",
                ],
                "{both}"
            );
        }
    }

    /// Every stored call edge names its tier, and the weight and the
    /// `resolved` flag are that tier's.
    #[test]
    fn an_edge_stores_the_tier_that_resolved_it() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/db.rs",
            "pub struct Database;\nimpl Database { pub fn save(&self) -> u8 { 1 } }\n",
        );
        write(
            dir.path(),
            "src/util.rs",
            "pub fn lonely() {}\npub fn twin() {}\n",
        );
        write(dir.path(), "src/other.rs", "pub fn twin() {}\n");
        write(
            dir.path(),
            "web/api.ts",
            "export function fetchUser(id: number) { return id; }\n",
        );
        write(
            dir.path(),
            "web/more.ts",
            "export function fetchUser(id: string) { return id; }\n",
        );
        write(
            dir.path(),
            "web/app.ts",
            "import { fetchUser } from \"./api\";\nexport function load() { return fetchUser(1); }\n",
        );
        write(
            dir.path(),
            "src/app.rs",
            "fn helper() {}\n\
             fn go(m: Mystery) {\n    let db = Database::new();\n    db.save();\n    helper();\n    lonely();\n    m.twin();\n}\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();

        assert_eq!(
            call_edges(&mimir.conn),
            [
                r#"go -> Database::save 1.00 {"resolved":1,"tier":"receiver-typed"}"#,
                r#"go -> helper 1.00 {"resolved":1,"tier":"same-file"}"#,
                r#"go -> lonely 0.80 {"resolved":1,"tier":"unique-global"}"#,
                r#"go -> twin 0.50 {"resolved":0,"tier":"ambiguous"}"#,
                r#"go -> twin 0.50 {"resolved":0,"tier":"ambiguous"}"#,
                r#"load -> fetchUser 1.00 {"resolved":1,"tier":"imported-name"}"#,
            ]
        );
        // The flag a reader of the graph derives from the weight is the flag
        // that was stored, edge for edge.
        let mut stmt = mimir
            .conn
            .prepare(
                "SELECT weight, json_extract(meta, '$.resolved') FROM edge WHERE rel = 'calls'",
            )
            .unwrap();
        let stored: Vec<(f64, bool)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(stored.len(), 6);
        for (weight, resolved) in stored {
            assert_eq!(CallTier::weight_is_resolved(weight), resolved, "{weight}");
        }
        // Counted per call site: four resolved (one of them by receiver
        // type), and one site left as a guess — not one per candidate.
        assert_eq!(stats.calls_resolved, 4, "{stats:?}");
        assert_eq!(stats.calls_typed, 1, "{stats:?}");
        assert_eq!(stats.calls_heuristic, 1, "{stats:?}");
    }

    /// One function, two call sites, one target: first pinned by the
    /// receiver's type, then again through a receiver nobody can type. The
    /// second used to overwrite the first, so which tag the edge carried
    /// depended on the order of the lines.
    #[test]
    fn a_later_guess_does_not_downgrade_a_resolved_edge() {
        for body in [
            "    let db = Database::new();\n    db.save();\n    m.save();\n",
            "    m.save();\n    let db = Database::new();\n    db.save();\n",
        ] {
            let dir = tempfile::tempdir().unwrap();
            write(
                dir.path(),
                "src/db.rs",
                "pub struct Database;\nimpl Database { pub fn save(&self) -> u8 { 1 } }\n",
            );
            write(
                dir.path(),
                "src/cache.rs",
                "pub struct Cache;\nimpl Cache { pub fn save(&self) -> u8 { 2 } }\n",
            );
            write(
                dir.path(),
                "src/app.rs",
                &format!("fn persist(m: Mystery) {{\n{body}}}\n"),
            );
            let mut mimir = Mimir::open_in_memory().unwrap();
            let proj = project(&mimir.conn, dir.path());
            update(&mut mimir.conn, &proj, dir.path()).unwrap();
            assert_eq!(
                call_edges(&mimir.conn),
                [
                    r#"persist -> Cache::save 0.50 {"resolved":0,"tier":"ambiguous"}"#,
                    r#"persist -> Database::save 1.00 {"resolved":1,"tier":"receiver-typed"}"#,
                ],
                "call order: {body}"
            );
        }
    }

    /// A name defined twice in the calling file is still a guess: being
    /// nearby says where to look, not which one. It used to be stored as a
    /// certain edge to whichever definition the table happened to list first.
    #[test]
    fn a_same_file_name_with_two_definitions_is_not_resolved() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/app.rs",
            "pub struct A;\nimpl A { pub fn go(&self) {} }\n\
             pub struct B;\nimpl B { pub fn go(&self) {} }\n\
             fn run(m: Mystery) { m.go(); }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(
            call_edges(&mimir.conn),
            [
                r#"run -> A::go 0.50 {"resolved":0,"tier":"ambiguous"}"#,
                r#"run -> B::go 0.50 {"resolved":0,"tier":"ambiguous"}"#,
            ]
        );
        assert_eq!((stats.calls_resolved, stats.calls_heuristic), (0, 1));
    }

    /// PHP writes the self receiver as `$this`. The sigil is stripped before
    /// the receiver is classified, so `this` is the entry that matches it —
    /// and the call is pinned to the enclosing class, not left as a guess
    /// between the two `format`s in the file.
    #[test]
    fn php_this_is_a_self_receiver() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "app.php",
            "<?php\nclass Greeter {\n    function greet() { return $this->format(); }\n    \
             function format() { return 1; }\n}\n\
             class Other {\n    function format() { return 2; }\n}\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(
            call_edges(&mimir.conn),
            [r#"Greeter::greet -> Greeter::format 1.00 {"resolved":1,"tier":"receiver-typed"}"#]
        );
    }

    /// A module that imports itself and recurses: the imported-name tier
    /// found the caller in the import's target file and linked it to itself.
    #[test]
    fn the_imported_name_tier_never_links_a_function_to_itself() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "m.py",
            "from m import run\n\ndef run(n):\n    return run(n - 1)\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert!(
            call_edges(&mimir.conn).is_empty(),
            "{:?}",
            call_edges(&mimir.conn)
        );
    }

    /// A file of constants defines no symbol, and is still a file something
    /// imports. The target set used to be "files that have a symbol".
    #[test]
    fn a_file_without_symbols_can_be_an_import_target() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/limits.rs", "pub const MAX: u8 = 9;\n");
        write(
            dir.path(),
            "src/main.rs",
            "use crate::limits::MAX;\n\nfn main() { let _ = MAX; }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.imports, 1, "{stats:?}");

        let graph = crate::CodeGraph::load(&mimir.conn, proj.id).unwrap();
        match graph
            .impact_of_paths(&mimir.conn, &["src/limits.rs".to_string()], 3)
            .unwrap()
        {
            crate::Impact::Reached(hits) => {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].node.path.as_deref(), Some("src/main.rs"));
            }
            other => panic!("main.rs imports limits.rs, got {other:?}"),
        }
    }

    /// The import edge is written through the language's own rule, end to
    /// end: an `#include` and a Java import, neither of which resolved while
    /// every dotted source was read as a Python module.
    #[test]
    fn includes_and_jvm_imports_become_edges() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/util.h", "int helper(void);\n");
        write(
            dir.path(),
            "src/main.c",
            "#include \"util.h\"\n#include <stdio.h>\nint run(void) { return helper(); }\n",
        );
        write(
            dir.path(),
            "src/main/java/com/acme/util/Slug.java",
            "package com.acme.util;\npublic class Slug { }\n",
        );
        write(
            dir.path(),
            "src/main/java/com/acme/App.java",
            "package com.acme;\nimport com.acme.util.Slug;\nimport java.util.List;\nclass App { }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        assert_eq!(
            import_edges(&mimir.conn),
            [
                "src/main.c -> src/util.h",
                "src/main/java/com/acme/App.java -> src/main/java/com/acme/util/Slug.java",
            ]
        );
    }

    /// `#include <math.h>` is the system's header even when the project has
    /// a `math.h` of its own right beside the including file; `"math.h"` is
    /// the project's. With the brackets stripped the two were one, and the
    /// system include became an edge to the project file.
    #[test]
    fn an_angle_include_is_never_resolved_to_a_project_file() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "geom/math.h", "int area(int w, int h);\n");
        write(
            dir.path(),
            "geom/shape.c",
            "#include <math.h>\nint side(int a) { return a; }\n",
        );
        write(
            dir.path(),
            "geom/rect.c",
            "#include \"math.h\"\nint rect(int a) { return a; }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(import_edges(&mimir.conn), ["geom/rect.c -> geom/math.h"]);
    }

    /// A binary under `src/bin/` with a module of its own, beside a library
    /// module of the same name. `crate::helper` in the binary is the
    /// binary's: the import edge, the call edge that follows from it, and
    /// what `impact` says about the two files all depend on that.
    #[test]
    fn a_binarys_crate_path_reaches_the_binarys_own_module() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "crates/a/src/lib.rs", "pub mod helper;\n");
        write(dir.path(), "crates/a/src/helper.rs", "pub fn run() {}\n");
        write(
            dir.path(),
            "crates/a/src/bin/helper.rs",
            "pub fn run() {}\n",
        );
        write(
            dir.path(),
            "crates/a/src/bin/tool.rs",
            "mod helper;\nuse crate::helper::run;\n\nfn main() { run(); }\n",
        );
        let mut mimir = Mimir::open_in_memory().unwrap();
        let proj = project(&mimir.conn, dir.path());
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        assert_eq!(
            import_edges(&mimir.conn),
            ["crates/a/src/bin/tool.rs -> crates/a/src/bin/helper.rs"]
        );
        let target: String = mimir
            .conn
            .query_row(
                "SELECT d.path || ' ' || json_extract(e.meta, '$.tier') FROM edge e
                 JOIN node d ON d.id = e.dst WHERE e.rel = 'calls'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(target, "crates/a/src/bin/helper.rs imported-name");

        let graph = crate::CodeGraph::load(&mimir.conn, proj.id).unwrap();
        let impact = |path: &str| {
            graph
                .impact_of_paths(&mimir.conn, &[path.to_string()], 3)
                .unwrap()
        };
        assert!(
            matches!(
                impact("crates/a/src/bin/helper.rs"),
                crate::Impact::Reached(_)
            ),
            "main() in tool.rs calls into it"
        );
        assert!(
            matches!(impact("crates/a/src/helper.rs"), crate::Impact::Contained),
            "nothing uses the library's helper"
        );
    }

    /// An unchanged file is re-resolved, and only when it has to be.
    #[test]
    fn only_the_files_an_edit_can_affect_are_re_resolved() {
        let (dir, mut mimir, proj) = setup();
        write(
            dir.path(),
            "src/quiet.rs",
            "pub fn quiet() { nothing_here(); }\n",
        );
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        // A body edit: no name appears or disappears, so nobody else moves.
        write(
            dir.path(),
            "src/util.rs",
            "pub fn slugify() { let _ = 1; }\n",
        );
        let report =
            update_with(&mut mimir.conn, &proj, dir.path(), UpdateOptions::default()).unwrap();
        assert_eq!((report.stats.files_indexed, report.reresolved), (1, 0));

        // A second `slugify`: main.rs calls that name, quiet.rs does not.
        write(dir.path(), "src/more.rs", "pub fn slugify() -> u8 { 2 }\n");
        let report =
            update_with(&mut mimir.conn, &proj, dir.path(), UpdateOptions::default()).unwrap();
        assert_eq!((report.stats.files_indexed, report.reresolved), (1, 1));
    }

    /// Whole seconds and a size cannot see an edit made within the second
    /// that kept the length. The nanoseconds can.
    #[test]
    fn a_same_second_same_size_edit_is_seen() {
        let (dir, mut mimir, proj) = setup();
        let path = dir.path().join("src/util.rs");
        let at =
            |nanos: u32| std::time::UNIX_EPOCH + std::time::Duration::new(1_700_000_000, nanos);
        let stamp = |nanos: u32| {
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(at(nanos))
                .unwrap();
        };
        stamp(100);
        update(&mut mimir.conn, &proj, dir.path()).unwrap();

        std::fs::write(&path, "pub fn slugifx() {}\n").unwrap();
        stamp(200);
        assert_eq!(
            check(&mimir.conn, &proj, dir.path()).unwrap().changed,
            vec!["src/util.rs"]
        );
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(stats.files_indexed, 1, "{stats:?}");
        assert!(resolve_symbol(&mimir.conn, proj.id, "slugifx").is_ok());
    }

    /// A graph built before nanoseconds were stored has none. Its files are
    /// hashed once (not re-extracted), found unchanged, and stamped — so the
    /// upgrade costs one read per file and no symbol is rewritten.
    #[test]
    fn a_row_without_nanoseconds_is_hashed_once_not_re_extracted() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.mtime_ns') WHERE kind = 'file'",
                [],
            )
            .unwrap();

        // Read-only: decided by the hash, and nothing is drift.
        assert!(!check(&mimir.conn, &proj, dir.path()).unwrap().is_stale());

        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!((stats.files_indexed, stats.unchanged), (0, 2), "{stats:?}");
        let stamped: i64 = mimir
            .conn
            .query_row(
                "SELECT count(*) FROM node
                 WHERE kind = 'file' AND json_extract(meta, '$.mtime_ns') IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stamped, 2);
    }

    /// Two installs of different versions can share a store (the CLI and the
    /// MCP server upgrade separately). When the older one rebuilds a file, it
    /// resolves it by its own rules and writes the row without nanoseconds.
    /// The next build here finds the bytes unchanged and has nothing to
    /// redo — but the graph is no longer all this build's work, and its
    /// rules stamp must stop saying it is.
    #[test]
    fn a_row_an_older_build_rewrote_takes_the_rules_stamp_with_it() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(crate::rules_note(&mimir.conn, proj.id).unwrap(), None);

        // An ordinary update with nothing to do leaves the stamp alone.
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(crate::rules_note(&mimir.conn, proj.id).unwrap(), None);

        // As the older build leaves a file it re-extracted.
        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.mtime_ns')
                 WHERE path = 'src/util.rs'",
                [],
            )
            .unwrap();
        let stats = update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!((stats.files_indexed, stats.unchanged), (0, 2), "{stats:?}");
        assert_eq!(crate::graph_rules(&mimir.conn, proj.id).unwrap(), None);
        assert!(crate::rules_note(&mimir.conn, proj.id).unwrap().is_some());

        // Until every file has been through this build again.
        update_with(
            &mut mimir.conn,
            &proj,
            dir.path(),
            UpdateOptions { full: true },
        )
        .unwrap();
        assert_eq!(crate::rules_note(&mimir.conn, proj.id).unwrap(), None);
    }

    /// The staleness a query reports: current, then behind by a count, and
    /// never a write — a query that refreshed the graph as a side effect
    /// would also be a query that can fail on a read-only store.
    #[test]
    fn freshness_counts_drift_and_never_writes() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(
            freshness(&mimir.conn, &proj, dir.path()).unwrap(),
            Freshness::Current
        );
        assert_eq!(Freshness::Current.note(), None);

        write(
            dir.path(),
            "src/util.rs",
            "pub fn slugify() {}\npub fn n() {}\n",
        );
        write(dir.path(), "src/new.rs", "pub fn added() {}\n");
        let writes_before = mimir.conn.total_changes();
        let behind = freshness(&mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(behind, Freshness::Behind(2));
        assert_eq!(
            behind.note().as_deref(),
            Some(
                "note: the code graph is 2 file(s) behind the working tree — run `mimir graph build`"
            )
        );
        assert_eq!(
            file_is_current(&mimir.conn, proj.id, dir.path(), "src/util.rs").unwrap(),
            Some(false)
        );
        assert_eq!(
            file_is_current(&mimir.conn, proj.id, dir.path(), "src/main.rs").unwrap(),
            Some(true)
        );
        assert_eq!(
            file_is_current(&mimir.conn, proj.id, dir.path(), "src/new.rs").unwrap(),
            None,
            "not in the graph at all"
        );
        assert_eq!(mimir.conn.total_changes(), writes_before, "read-only");
    }

    /// Above the file budget the comparison is not run, and the answer says
    /// that rather than passing for "current".
    #[test]
    fn freshness_is_not_checked_above_the_file_budget() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        write(dir.path(), "src/new.rs", "pub fn added() {}\n");

        // Two files in the graph, a budget of one.
        let unchecked = freshness_within(&mimir.conn, &proj, dir.path(), 1).unwrap();
        assert_eq!(unchecked, Freshness::Unchecked);
        assert!(unchecked.note().unwrap().contains("was not checked"));
        // Within the budget it is compared as usual.
        assert_eq!(
            freshness_within(&mimir.conn, &proj, dir.path(), 2).unwrap(),
            Freshness::Behind(1)
        );
    }

    /// The text search that rules a file out unparsed must never disagree
    /// with the parse it stands in for: a missed dependent is a stale edge.
    #[test]
    fn searching_the_stored_calls_as_text_agrees_with_parsing_them() {
        let meta = serde_json::json!({
            "mtime": 1, "mtime_ns": 2, "size": 3,
            "imports": [],
            "calls": [
                {"caller": "run", "callee": "slugify"},
                {"caller": "slugify_all", "callee": "quote\"d"},
            ],
        })
        .to_string();
        let set =
            |names: &[&str]| -> HashSet<String> { names.iter().map(|n| n.to_string()).collect() };
        for (names, expected) in [
            (vec!["slugify"], true),
            (vec!["other", "slugify"], true),
            // A caller, or part of a name, is not a callee.
            (vec!["run"], false),
            (vec!["slug"], false),
            (vec!["slugify_all"], false),
            (vec!["nothing"], false),
            // A name the store writes escaped is found too.
            (vec!["quote\"d"], true),
        ] {
            let touched = set(&names);
            let needles = name_needles(&touched);
            assert_eq!(
                calls_touched(&meta, &touched, needles.as_deref()),
                expected,
                "{names:?} searched as text"
            );
            assert_eq!(
                calls_touched(&meta, &touched, None),
                expected,
                "{names:?} parsed"
            );
        }
        assert!(name_needles(&set(&["slugify"])).is_some());
        assert!(
            name_needles(&set(&["quote\"d"])).is_none(),
            "escaped: parse"
        );
        let many: Vec<String> = (0..=NEEDLE_MAX).map(|i| format!("n{i}")).collect();
        assert!(name_needles(&many.into_iter().collect()).is_none());
        assert!(!calls_touched(&meta, &HashSet::new(), None));

        // A row with no call list was not written by this extractor: it is
        // affected until proven otherwise, by either route.
        let bare = r#"{"mtime":1,"size":3}"#;
        let touched = set(&["slugify"]);
        assert!(calls_touched(
            bare,
            &touched,
            name_needles(&touched).as_deref()
        ));
        assert!(calls_touched("not json", &touched, None));

        // Imports: an empty list never moves; a missing one is unknown.
        let none = FileSet::new(Vec::new());
        let some = FileSet::new(vec!["src/util.rs".to_string()]);
        assert!(!imports_moved("src/main.rs", &meta, (&none, &some)));
        assert!(imports_moved("src/main.rs", bare, (&none, &some)));
        let importing = serde_json::json!({
            "calls": [],
            "imports": [{"local": "slugify", "source": "crate::util::slugify"}],
        })
        .to_string();
        assert!(imports_moved("src/main.rs", &importing, (&none, &some)));
        assert!(!imports_moved("src/main.rs", &importing, (&some, &some)));
    }

    /// The search for new files has a budget of directory entries, because
    /// a tree can be mostly something other than code. Running out of it
    /// must not silence the half of the answer that was already known: it
    /// used to turn every answer in such a project into "not checked … too
    /// large", fifty source files or not.
    #[test]
    fn a_tree_that_is_mostly_not_code_is_still_checked() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        for i in 0..200 {
            write(dir.path(), &format!("assets/blob{i}.bin"), "x");
        }
        // A file budget of two is a walk budget of ten entries: far short
        // of the 200 assets.
        assert_eq!(
            freshness_within(&mimir.conn, &proj, dir.path(), 2).unwrap(),
            Freshness::Current
        );
        write(
            dir.path(),
            "src/util.rs",
            "pub fn slugify() { let _ = 1; }\n",
        );
        std::fs::remove_file(dir.path().join("src/main.rs")).unwrap();
        assert_eq!(
            freshness_within(&mimir.conn, &proj, dir.path(), 2).unwrap(),
            Freshness::Behind(2),
            "an edit and a deletion are seen without walking anything"
        );
        // The budget is counted in entries, and spent on the ones between
        // code files too.
        let mut walk = CodeWalk::within(dir.path(), 10);
        let found = walk.by_ref().count();
        assert!(
            walk.cut_short && walk.visited == 10 && found <= 1,
            "{found}"
        );
        let mut walk = CodeWalk::within(dir.path(), 10_000);
        assert_eq!(walk.by_ref().count(), 1, "src/util.rs is the code left");
        assert!(!walk.cut_short);
    }

    /// A file the graph holds that the walk no longer yields — ignored since
    /// it was indexed — is one an update would drop. `check` calls it
    /// removed, and a query that walked the whole tree must agree.
    #[test]
    fn freshness_agrees_with_check_about_a_file_that_left_the_walk() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        write(dir.path(), ".ignore", "src/util.rs\n");

        let drift = check(&mimir.conn, &proj, dir.path()).unwrap();
        assert_eq!(drift.removed, vec!["src/util.rs"], "{drift:?}");
        assert_eq!(
            freshness(&mimir.conn, &proj, dir.path()).unwrap(),
            Freshness::Behind(drift.stale_count())
        );
    }

    /// A graph from a build that stored no nanoseconds: a query takes each
    /// row at the precision it has, and reads no file to do it. `check`
    /// still settles such a row by its hash.
    #[test]
    fn freshness_takes_a_row_without_nanoseconds_at_its_word() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.mtime_ns'), content_hash = x'00'
                 WHERE kind = 'file'",
                [],
            )
            .unwrap();
        // The hashes are wrong on purpose: had the query hashed, it would
        // have called both files changed, as `check` does.
        assert_eq!(
            freshness(&mimir.conn, &proj, dir.path()).unwrap(),
            Freshness::Current
        );
        assert_eq!(
            check(&mimir.conn, &proj, dir.path()).unwrap().changed.len(),
            2
        );
        // A different size is still a change at that precision.
        write(
            dir.path(),
            "src/util.rs",
            "pub fn slugify() { let _ = 1; }\n",
        );
        assert_eq!(
            freshness(&mimir.conn, &proj, dir.path()).unwrap(),
            Freshness::Behind(1)
        );
    }

    /// The stat is read out of the meta text without parsing it, and must
    /// say what a JSON parser says: for the rows a build writes, for rows
    /// SQLite has rewritten, and for text that only looks like the keys.
    #[test]
    fn the_stored_stat_read_as_text_is_what_json_says() {
        let (dir, mut mimir, proj) = setup();
        update(&mut mimir.conn, &proj, dir.path()).unwrap();
        // One row rewritten by `json_set`, as a touch does; one left without
        // nanoseconds, as an older build wrote it.
        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_set(meta, '$.mtime', 12, '$.mtime_ns', 34)
                 WHERE path = 'src/util.rs'",
                [],
            )
            .unwrap();
        mimir
            .conn
            .execute(
                "UPDATE node SET meta = json_remove(meta, '$.mtime_ns') WHERE path = 'src/main.rs'",
                [],
            )
            .unwrap();
        let mut stmt = mimir
            .conn
            .prepare(
                "SELECT meta, json_extract(meta, '$.mtime'), json_extract(meta, '$.mtime_ns'),
                        json_extract(meta, '$.size')
                 FROM node WHERE kind = 'file' ORDER BY path",
            )
            .unwrap();
        type Row = (String, Option<i64>, Option<i64>, Option<i64>);
        let rows: Vec<Row> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows.len(), 2);
        for (meta, mtime, mtime_ns, size) in &rows {
            assert_eq!(stored_stat(meta), (*mtime, *mtime_ns, *size), "{meta}");
        }
        assert_eq!(rows[0].2, None, "main.rs has no nanoseconds");
        assert_eq!((rows[1].1, rows[1].2), (Some(12), Some(34)));

        // Call and import text that spells the keys is not the keys.
        let tricky = serde_json::json!({
            "calls": [{"caller": "mtime", "callee": "\"size\":7,"}],
            "imports": [{"local": "size", "source": "\"mtime\":1,\"mtime_ns\":2}"}],
            "mtime": -1, "mtime_ns": -1, "size": 0,
        })
        .to_string();
        assert_eq!(
            stored_stat(&tricky),
            (Some(-1), Some(-1), Some(0)),
            "{tricky}"
        );
        // Not the compact form: a parser is asked instead.
        assert_eq!(
            stored_stat("{ \"mtime\" : 5, \"size\" : 9, \"mtime_ns\" : 6 }"),
            (Some(5), Some(6), Some(9))
        );
        // Not integers, or not there, or not JSON.
        assert_eq!(
            stored_stat(r#"{"mtime":1.5,"size":2}"#),
            (None, None, Some(2))
        );
        assert_eq!(stored_stat(r#"{"calls":[]}"#), (None, None, None));
        assert_eq!(stored_stat("not json"), (None, None, None));
    }
}
