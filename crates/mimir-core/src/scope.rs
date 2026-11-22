use std::path::{Path, PathBuf};

/// Markers that identify a project root, in priority order. `.git` covers
/// the common case; the other VCS dirs catch non-git repos; `.mimir` is the
/// explicit opt-in for code outside any VCS (Windows drives, unpacked
/// archives, config trees) — `touch .mimir` makes a directory a project
/// root.
pub const ROOT_MARKERS: &[&str] = &[".git", ".hg", ".svn", ".jj", ".mimir"];

/// Build-system files that mark a project root when no VCS/explicit marker is
/// found (a lower-priority second pass). A `const` so it's trivial to extend.
pub const PROJECT_MARKERS: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "go.work",
    "deno.json",
    "deno.jsonc",
    "pnpm-workspace.yaml",
];

/// Outcome of resolving a project root from a directory. `Found.via` is the
/// marker that identified the root (for "detected via …" reporting); `NotFound`
/// carries where the search started so callers can explain the fallback
/// (which one, see [`query_scope`] and [`push_scope`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Detection {
    Found { root: PathBuf, via: &'static str },
    NotFound { from: PathBuf },
}

fn absolutize(start: &Path) -> Option<PathBuf> {
    if start.is_absolute() {
        Some(start.to_path_buf())
    } else {
        Some(std::env::current_dir().ok()?.join(start))
    }
}

/// Deepest ancestor of `start` (inclusive) containing any of `markers`, with the
/// marker that matched.
fn walk_up(start: &Path, markers: &[&'static str]) -> Option<(PathBuf, &'static str)> {
    let mut dir = start.to_path_buf();
    loop {
        if let Some(&m) = markers.iter().find(|m| dir.join(m).exists()) {
            return Some((dir, m));
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Resolve the project root from `start`, first match wins:
/// 1. a VCS/explicit root (`ROOT_MARKERS`) — deepest from `start`, so a nested
///    `.mimir` overrides and a git root wins over a deeper build file;
/// 2. else the nearest build-file root (`PROJECT_MARKERS`);
/// 3. else `NotFound` (no project: new memories are global, and what a read
///    sees is [`query_scope`] or [`push_scope`], explained by the caller).
pub fn detect(start: &Path) -> Detection {
    let Some(abs) = absolutize(start) else {
        return Detection::NotFound {
            from: start.to_path_buf(),
        };
    };
    if let Some((root, via)) = walk_up(&abs, ROOT_MARKERS) {
        return Detection::Found { root, via };
    }
    if let Some((root, via)) = walk_up(&abs, PROJECT_MARKERS) {
        return Detection::Found { root, via };
    }
    Detection::NotFound { from: abs }
}

/// Just the root path (no reason). Thin wrapper over [`detect`] for callers that
/// only need to scope; all of them transparently gain build-file detection.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    match detect(start) {
        Detection::Found { root, .. } => Some(root),
        Detection::NotFound { .. } => None,
    }
}

/// The scope a **question** gets when its caller named none: `recall`,
/// `list`, `misses`, the MCP read tools, and the per-prompt auto-recall
/// (the prompt is the question, and its relevance floor decides whether
/// anything is said at all).
///
/// Inside a project that is the project plus the global memories. Outside
/// any project it is *everything*: someone asking from `~` or `/tmp` wants
/// the answer wherever it was written down, and an answer from another
/// project arrives labelled with that project's name, so nothing is passed
/// off as local.
///
/// The counterpart is [`push_scope`]. The two agree inside a project and
/// differ only outside one, and every surface picks one of them by name
/// instead of spelling its own fallback.
pub fn query_scope(project: Option<i64>) -> crate::model::Scope {
    use crate::model::Scope;
    match project {
        Some(id) => Scope::Project(id),
        None => Scope::All,
    }
}

/// What a **question** that may be answered by a memory, a doc or a piece of
/// code can reach (`recall`): [`query_scope`] taken twice, of the project the
/// asker's memories are filed under and of the asker's own checkout.
///
/// The two are one project, and the reach one scope, everywhere except in a
/// git linked worktree ([`Checkout::LinkedWorktree`]): there the memories are
/// the repository's and the code and docs are the worktree's own. Outside
/// any project both halves are *everything*, as [`query_scope`] says.
///
/// A surface that reads memories only (`list`, per-prompt injection, the
/// brief, guard anchors, the rules pack) has no second half to ask for, and
/// takes [`query_scope`] or [`push_scope`] of the memory project.
pub fn query_reach(
    memory_project: Option<i64>,
    checkout_project: Option<i64>,
) -> crate::search::Reach {
    crate::search::Reach {
        memories: query_scope(memory_project),
        code_and_docs: query_scope(checkout_project),
    }
}

/// The scope for what Mimir **volunteers** without being asked: the session
/// brief, the context guard's handoff restore, the guard-anchors hook.
///
/// Inside a project that is the project plus the global memories, same as
/// [`query_scope`]. Outside any project it is the global memories *only*:
/// nobody asked, so there is no question to judge relevance against, and a
/// gotcha written for one repository pushed into a session in an unrelated
/// directory reads as an instruction about that directory. Global memories
/// were written to apply everywhere; they are all that is safe to push.
pub fn push_scope(project: Option<i64>) -> crate::model::Scope {
    use crate::model::Scope;
    match project {
        Some(id) => Scope::Project(id),
        None => Scope::Global,
    }
}

/// Human label for a `Detection::Found.via` marker (`.git` → `git`).
pub fn via_label(via: &str) -> &str {
    match via {
        ".git" => "git",
        ".hg" => "hg",
        ".svn" => "svn",
        ".jj" => "jj",
        other => other,
    }
}

/// Canonical string form of a project root, used as project identity.
pub fn canonical_root(root: &Path) -> String {
    dunce_canonicalize(root).to_string_lossy().into_owned()
}

/// Best-effort canonicalization that avoids Windows `\\?\` extended-length
/// paths, so the resulting identity is both stable and still openable.
fn dunce_canonicalize(path: &Path) -> PathBuf {
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    PathBuf::from(simplify_verbatim(&canon.to_string_lossy()))
}

/// Turn a Windows extended-length (`\\?\`) path into its normal form.
/// `canonicalize` returns verbatim paths on Windows:
///   `\\?\C:\dir`         -> `C:\dir`
///   `\\?\UNC\srv\share`  -> `\\srv\share`
/// Stripping only `\\?\` leaves an invalid `UNC\srv\share` that fails to open,
/// which broke indexing of projects on network shares. The UNC form must be
/// rewritten to a leading `\\`. Any non-verbatim input is returned unchanged.
fn simplify_verbatim(s: &str) -> String {
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        s.to_string()
    }
}

/// Which checkout of its repository a project root is.
///
/// One directory answers two questions, and for a `git worktree add`
/// checkout the answers differ. *Whose code graph* is always the directory
/// itself: its files are on its own branch. *Whose memories* is the
/// repository: what was learned about the code is as true in a second
/// checkout of it as in the first, and has to outlive a worktree that is
/// deleted when its branch merges.
///
/// [`Checkout::of`] is the one derivation, from files alone (no git
/// subprocess: hooks ask on every event). `Mimir::projects_for_cwd` turns it
/// into the two project rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Checkout {
    /// The root is its own home, or no home can be found for it: a
    /// repository's main checkout, a plain clone, a submodule, a project
    /// that is not a git checkout at all, and a linked worktree whose main
    /// checkout is not a checkout Mimir could run in (a bare repository's
    /// worktree; a `core.worktree` naming no directory; a home whose own
    /// `.git` does not lead back to the repository). Its memories are filed
    /// under its own project.
    Standalone,
    /// A git linked worktree. `home` is the canonical root of the
    /// repository's main checkout, the project its memories are filed
    /// under: the directory the repository's `core.worktree` names, or else
    /// the parent of its `.git` directory, and in either case a directory
    /// whose own `.git` leads back to the repository.
    LinkedWorktree { home: PathBuf },
}

impl Checkout {
    /// Classify the checkout rooted at `root`.
    ///
    /// A linked worktree is recognised by what `git worktree add` writes,
    /// every part of which has to hold:
    /// - `root/.git` is a regular file reading `gitdir: <path>`, the path
    ///   absolute or relative to `root`;
    /// - `<gitdir>/commondir` names the repository's common directory,
    ///   absolute or relative to the gitdir (a submodule's gitdir has no
    ///   such file);
    /// - the common directory's `config` does not say `bare = true` (a bare
    ///   repository has no main checkout to file memories under);
    /// - the home is the directory that `config` names as `core.worktree`
    ///   (relative to the common directory, as git reads it: a submodule's
    ///   repository says where its checkout is this way), or, when it names
    ///   none, the parent of a common directory called `.git`;
    /// - the home's own `.git`, a directory or a `gitdir:` file, is the
    ///   common directory;
    /// - and the home is a directory other than `root`.
    ///
    /// Anything else, an unreadable or malformed file included, is
    /// [`Checkout::Standalone`]: the checkout keeps its own memories, which
    /// is what every checkout did before the two were told apart.
    pub fn of(root: &Path) -> Checkout {
        match linked_worktree_home(root) {
            Some(home) => Checkout::LinkedWorktree { home },
            None => Checkout::Standalone,
        }
    }

    /// The main checkout's root, when this is a linked worktree.
    pub fn home(&self) -> Option<&Path> {
        match self {
            Checkout::Standalone => None,
            Checkout::LinkedWorktree { home } => Some(home),
        }
    }

    /// Whether this is a git linked worktree.
    pub fn is_linked_worktree(&self) -> bool {
        matches!(self, Checkout::LinkedWorktree { .. })
    }

    /// The state's stable name, for JSON and logs.
    pub fn as_str(&self) -> &'static str {
        match self {
            Checkout::Standalone => "standalone",
            Checkout::LinkedWorktree { .. } => "linked_worktree",
        }
    }
}

impl std::fmt::Display for Checkout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The path a `.git` *file* points at: the text after `gitdir:` on its
/// first line. `None` for anything that is not such a pointer.
fn parse_gitdir_pointer(text: &str) -> Option<&str> {
    let path = text.lines().next()?.strip_prefix("gitdir:")?.trim();
    (!path.is_empty()).then_some(path)
}

/// The gitdir a worktree's `.git` file points at, as written (not checked
/// to exist). `None` when `root/.git` is not a regular file holding a
/// `gitdir:` pointer: the ordinary checkout ends here, on one stat, since
/// its `.git` is a directory.
fn worktree_gitdir(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    if !std::fs::metadata(&dot_git).ok()?.is_file() {
        return None;
    }
    let pointer = std::fs::read_to_string(&dot_git).ok()?;
    let gitdir = root.join(parse_gitdir_pointer(&pointer)?);
    // The repository has to know this worktree too: git writes the path of
    // the worktree's `.git` file into `<gitdir>/gitdir` when it creates it.
    // Without that back-pointer a `.git` file anyone wrote would make any
    // directory a worktree of any repository on the machine, and its
    // sessions would read and write that repository's memories.
    let back = std::fs::read_to_string(gitdir.join("gitdir")).ok()?;
    let back = Path::new(back.lines().next()?.trim());
    let back = if back.is_absolute() {
        back.to_path_buf()
    } else {
        gitdir.join(back)
    };
    (back.canonicalize().ok()? == dot_git.canonicalize().ok()?).then_some(gitdir)
}

/// The canonical common directory of a linked worktree whose gitdir is
/// `gitdir`, from its `commondir` file.
fn common_dir(gitdir: &Path) -> Option<PathBuf> {
    let commondir = std::fs::read_to_string(gitdir.join("commondir")).ok()?;
    let commondir = commondir.lines().next()?.trim();
    if commondir.is_empty() {
        return None;
    }
    // Canonical before it is taken apart: git writes `../..` here, and a
    // path ending in `..` has neither a file name nor a usable parent. It
    // also has to exist and be a directory.
    let common = gitdir.join(commondir).canonicalize().ok()?;
    common.is_dir().then_some(common)
}

/// The two `[core]` settings of a repository's `config` that say where its
/// main checkout is.
#[derive(Debug, Default, PartialEq, Eq)]
struct CoreConfig {
    /// `core.bare`: the repository has no main checkout.
    bare: bool,
    /// `core.worktree`: the main checkout is here, not beside the
    /// repository; relative paths are relative to the repository directory.
    worktree: Option<String>,
}

/// Read `core.bare` and `core.worktree` out of a git config file's text.
///
/// A reader of the file format, not of git's configuration: sections and
/// keys are matched without regard to case, as git does; a value may be
/// quoted, and a `#` or `;` outside quotes starts a comment; a key with no
/// `=` is a boolean that is true; the last occurrence wins. `include`
/// directives and per-worktree config files are not followed: a setting
/// that lives only there is not seen, and the checkout is classified as if
/// it were absent.
fn parse_core_config(text: &str) -> CoreConfig {
    let mut core = CoreConfig::default();
    let mut in_core = false;
    for line in text.lines() {
        let mut line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            // `[core]`, and nothing with a subsection (`[core "x"]`,
            // `[core.x]`). A key may follow the header on the same line.
            let Some((name, rest)) = header.split_once(']') else {
                in_core = false;
                continue;
            };
            in_core = name.trim().eq_ignore_ascii_case("core");
            line = rest.trim();
        }
        if !in_core || line.is_empty() || line.starts_with(['#', ';']) {
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some((key, value)) => (key.trim(), Some(config_value(value))),
            None => (
                line.split([' ', '\t', '#', ';']).next().unwrap_or(line),
                None,
            ),
        };
        if key.eq_ignore_ascii_case("bare") {
            core.bare = value.as_deref().is_none_or(config_bool);
        } else if key.eq_ignore_ascii_case("worktree") {
            core.worktree = value.filter(|v| !v.is_empty());
        }
    }
    core
}

/// A config value as git reads it: the comment after an unquoted `#` or `;`
/// dropped, the quotes removed, the outer whitespace trimmed.
fn config_value(raw: &str) -> String {
    let mut out = String::new();
    let mut quoted = false;
    let mut chars = raw.trim().chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => quoted = !quoted,
            '#' | ';' if !quoted => break,
            '\\' => {
                if let Some(next) = chars.next() {
                    out.push(match next {
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    });
                }
            }
            other => out.push(other),
        }
    }
    out.trim().to_string()
}

/// Whether a config value reads as true (`true`, `yes`, `on`, a non-zero
/// number), as git's booleans do.
fn config_bool(value: &str) -> bool {
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" => true,
        "false" | "no" | "off" | "" => false,
        number => number.parse::<i64>().is_ok_and(|n| n != 0),
    }
}

/// The canonical git directory of the checkout at `root`, through its
/// `.git` directory or its `gitdir:` file; `None` when it has neither.
fn checkout_git_dir(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    let meta = std::fs::metadata(&dot_git).ok()?;
    if meta.is_dir() {
        return dot_git.canonicalize().ok();
    }
    let pointer = std::fs::read_to_string(&dot_git).ok()?;
    root.join(parse_gitdir_pointer(&pointer)?)
        .canonicalize()
        .ok()
}

/// The canonical root of the main checkout `root` is a linked worktree of,
/// or `None`; see [`Checkout::of`] for the rule. The result goes through the
/// same canonicalization as [`canonical_root`], so it is the identity the
/// main checkout registers under when Mimir runs there.
fn linked_worktree_home(root: &Path) -> Option<PathBuf> {
    let common = common_dir(&worktree_gitdir(root)?)?;
    let config = std::fs::read_to_string(common.join("config"))
        .map(|text| parse_core_config(&text))
        .unwrap_or_default();
    if config.bare {
        return None;
    }
    let home = match config.worktree {
        Some(worktree) => common.join(worktree).canonicalize().ok()?,
        None if common.file_name() == Some(std::ffi::OsStr::new(".git")) => {
            common.parent()?.to_path_buf()
        }
        None => return None,
    };
    if !home.is_dir() || checkout_git_dir(&home)? != common {
        return None;
    }
    let home = canonical_root(&home);
    (home != canonical_root(root)).then(|| PathBuf::from(home))
}

/// When the linked worktree at `root` was made, in unix seconds: the
/// modification time of its `<gitdir>/commondir`, which `git worktree add`
/// writes once and nothing rewrites. `None` when `root` is not a linked
/// worktree or the time cannot be read.
///
/// Rounded *up* to the second. A memory's `created_at` has one-second
/// resolution, so a memory from the very second the worktree was made is
/// counted as older than it: of the two mistakes, filing one memory of
/// this worktree too late is the one that can be undone, filing another
/// repository's memory under this one is not
/// (`Mimir::projects_for_cwd_moving_memories_home`).
pub fn worktree_created_at(root: &Path) -> Option<i64> {
    let made = worktree_made(root)?;
    let seconds = i64::try_from(made.as_secs()).ok()?;
    Some(if made.subsec_nanos() > 0 {
        seconds + 1
    } else {
        seconds
    })
}

/// Whether the linked worktree now at `root` was made in a later second
/// than `at` (unix seconds): something dated `at` was written by an earlier
/// checkout at the same path, one since removed. `false` for a root that is
/// not a linked worktree, and when the time cannot be read. The opposite
/// rounding to [`worktree_created_at`]: this one has to be sure the thing
/// is older before it disowns it.
pub fn worktree_made_after(root: &Path, at: i64) -> bool {
    worktree_made(root)
        .and_then(|made| i64::try_from(made.as_secs()).ok())
        .is_some_and(|made| made > at)
}

/// The modification time of the `commondir` of the linked worktree at
/// `root`, since the epoch.
fn worktree_made(root: &Path) -> Option<std::time::Duration> {
    let modified = std::fs::metadata(worktree_gitdir(root)?.join("commondir"))
        .ok()?
        .modified()
        .ok()?;
    modified.duration_since(std::time::UNIX_EPOCH).ok()
}

/// Project display name: the root directory's file name.
pub fn project_name(root: &Path) -> String {
    root.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.to_string_lossy().into_owned())
}

/// Parsed `.mimir` marker. A `touch`ed / empty file yields all-defaults.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MimirMarker {
    /// Stable, machine-independent project id (the portable key, when present).
    pub id: Option<String>,
    /// Opt this project's memories into sync.
    pub sync: bool,
}

/// What reading a root's `.mimir` found. Exactly one holds.
///
/// "No marker" and "a marker that cannot be read" are different things to
/// anyone about to write the file: the first is free to create, the second
/// is a file somebody made, quite possibly holding the id the project's
/// synced memories are filed under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkerState {
    /// There is no `.mimir` here.
    Absent,
    /// A marker that reads. A `touch`ed / empty file is one, all defaults.
    Parsed(MimirMarker),
    /// A `.mimir` exists but is not a readable marker: not TOML, not text,
    /// a directory, no permission. Carries the reason, for a human.
    Unparseable(String),
}

/// Read the `.mimir` marker of `root` as TOML (`id = "..."`, `sync = true`)
/// and say which of the three [`MarkerState`]s it is in.
pub fn read_marker_state(root: &Path) -> MarkerState {
    let txt = match std::fs::read_to_string(root.join(".mimir")) {
        Ok(txt) => txt,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return MarkerState::Absent,
        Err(e) => return MarkerState::Unparseable(e.to_string()),
    };
    match toml::from_str::<toml::Value>(&txt) {
        Ok(val) => MarkerState::Parsed(MimirMarker {
            id: val.get("id").and_then(|v| v.as_str()).map(str::to_owned),
            sync: val.get("sync").and_then(|v| v.as_bool()).unwrap_or(false),
        }),
        Err(e) => MarkerState::Unparseable(e.to_string()),
    }
}

/// The marker of `root`, for a reader that only wants its fields: `None` if
/// absent or unparseable; an empty file parses to defaults. A caller that
/// may write the file needs to tell those two apart: [`read_marker_state`].
pub fn read_mimir_marker(root: &Path) -> Option<MimirMarker> {
    match read_marker_state(root) {
        MarkerState::Parsed(marker) => Some(marker),
        MarkerState::Absent | MarkerState::Unparseable(_) => None,
    }
}

/// A project's **portable key** — identity that is the same on every machine
/// (unlike the absolute path), so project-scoped memories can sync. Order:
///   1. a committed `.mimir` `id` (source of truth — survives renames/moves),
///   2. else the normalized `origin` remote from `.git/config`,
///   3. else `None` (not portable → the project's memories stay local-only).
pub fn portable_key(root: &Path) -> Option<String> {
    if let Some(id) = read_mimir_marker(root).and_then(|m| m.id) {
        return Some(id);
    }
    git_origin_key(root)
}

/// Whether this project opts its memories into sync (`sync = true` in `.mimir`).
pub fn sync_opt_in(root: &Path) -> bool {
    read_mimir_marker(root).map(|m| m.sync).unwrap_or(false)
}

/// A fresh, machine-independent project id (ULID) for a `.mimir` marker.
pub fn new_id() -> String {
    ulid::Ulid::new().to_string()
}

/// `git:<host>/<path>` from the `origin` remote in `.git/config`, read as a
/// file (no git subprocess — matching the rest of scope detection). `None` if
/// there's no repo or no origin remote.
fn git_origin_key(root: &Path) -> Option<String> {
    let cfg = std::fs::read_to_string(root.join(".git").join("config")).ok()?;
    parse_git_origin_url(&cfg).map(|url| format!("git:{}", normalize_remote(&url)))
}

/// Find `url = …` under `[remote "origin"]` in a git config file.
fn parse_git_origin_url(cfg: &str) -> Option<String> {
    let mut in_origin = false;
    for line in cfg.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_origin = t.replace(char::is_whitespace, "") == "[remote\"origin\"]";
            continue;
        }
        if in_origin {
            if let Some(rest) = t.strip_prefix("url") {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    return Some(v.trim().to_string());
                }
            }
        }
    }
    None
}

/// Normalize a git remote URL to a `host/path` key: scheme/user stripped,
/// trailing `.git` removed, lowercased. `git@github.com:Org/Repo.git` and
/// `https://github.com/Org/Repo.git` both → `github.com/org/repo`.
fn normalize_remote(url: &str) -> String {
    let u = url.trim();
    let host_path = if let Some(rest) = u.strip_prefix("git@") {
        rest.replacen(':', "/", 1) // scp-like: git@host:path
    } else {
        let after_scheme = u.split_once("://").map(|(_, r)| r).unwrap_or(u);
        after_scheme
            .split_once('@')
            .map(|(_, r)| r.to_string())
            .unwrap_or_else(|| after_scheme.to_string())
    };
    host_path
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two no-project policies agree inside a project and part ways
    /// only outside one: a question searches everything, a push stays
    /// with the global memories.
    #[test]
    fn query_and_push_scopes_differ_only_outside_a_project() {
        use crate::model::Scope;
        assert_eq!(query_scope(Some(7)), Scope::Project(7));
        assert_eq!(push_scope(Some(7)), Scope::Project(7));
        assert_eq!(query_scope(None), Scope::All);
        assert_eq!(push_scope(None), Scope::Global);
        // What that means for a memory owned by some other project.
        assert!(query_scope(None).admits(Some(3)));
        assert!(!push_scope(None).admits(Some(3)));
        assert!(push_scope(None).admits(None));
    }

    /// A question's reach is one scope wherever the two projects are one,
    /// and two only when memories and code are filed apart.
    #[test]
    fn a_reach_is_uniform_unless_memories_and_code_are_filed_apart() {
        use crate::model::{Kind, Scope};
        use crate::search::Reach;
        assert_eq!(query_reach(None, None), Reach::uniform(Scope::All));
        assert_eq!(
            query_reach(Some(7), Some(7)),
            Reach::uniform(query_scope(Some(7)))
        );
        assert!(query_reach(Some(7), Some(7)).is_uniform());

        // A linked worktree: memories under the home project, code under
        // its own.
        let split = query_reach(Some(7), Some(9));
        assert_eq!(split.memories, Scope::Project(7));
        assert_eq!(split.code_and_docs, Scope::Project(9));
        assert!(!split.is_uniform());
        assert!(split.admits(Kind::Memory, Some(7)));
        assert!(!split.admits(Kind::Memory, Some(9)));
        assert!(split.admits(Kind::Symbol, Some(9)));
        assert!(!split.admits(Kind::Symbol, Some(7)));
        assert!(split.admits(Kind::Memory, None) && split.admits(Kind::Chunk, None));
    }

    #[test]
    fn finds_root_by_any_marker_and_walks_up() {
        for marker in ROOT_MARKERS {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().join("proj");
            let nested = root.join("a/b/c");
            std::fs::create_dir_all(&nested).unwrap();
            std::fs::create_dir_all(root.join(marker)).unwrap();
            assert_eq!(
                find_project_root(&nested),
                Some(root.clone()),
                "marker {marker} should be found from a nested dir"
            );
        }
    }

    #[test]
    fn mimir_marker_can_be_a_plain_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("novcs");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".mimir"), b"").unwrap();
        assert_eq!(find_project_root(&root), Some(root.clone()));
    }

    #[test]
    fn none_outside_any_project() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(find_project_root(tmp.path()), None);
    }

    #[test]
    fn detects_git_root_via() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        assert_eq!(
            detect(&root),
            Detection::Found {
                root: root.clone(),
                via: ".git"
            }
        );
        assert_eq!(via_label(".git"), "git");
    }

    #[test]
    fn detects_non_git_build_marker() {
        for marker in ["Cargo.toml", "package.json", "pyproject.toml"] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().join("proj");
            let nested = root.join("src/deep");
            std::fs::create_dir_all(&nested).unwrap();
            std::fs::write(root.join(marker), b"").unwrap();
            assert_eq!(
                detect(&nested),
                Detection::Found {
                    root: root.clone(),
                    via: marker
                },
                "{marker} should scope the project from a nested dir"
            );
        }
    }

    #[test]
    fn nested_build_marker_deepest_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let outer = tmp.path().join("workspace");
        let inner = outer.join("packages/app");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(outer.join("package.json"), b"").unwrap();
        std::fs::write(inner.join("package.json"), b"").unwrap();
        // From the inner dir, the nearest (deepest) marker wins.
        assert_eq!(
            detect(&inner),
            Detection::Found {
                root: inner.clone(),
                via: "package.json"
            }
        );
    }

    #[test]
    fn vcs_root_wins_over_deeper_build_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let pkg = repo.join("crates/sub");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::write(pkg.join("Cargo.toml"), b"").unwrap();
        // Phase A (git) wins even though Cargo.toml is deeper/nearer.
        assert_eq!(
            detect(&pkg),
            Detection::Found {
                root: repo.clone(),
                via: ".git"
            }
        );
    }

    #[test]
    fn bare_dir_is_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let bare = tmp.path().join("scratch");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(detect(&bare), Detection::NotFound { from: bare });
    }

    #[test]
    fn portable_key_prefers_mimir_id_over_remote() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".mimir"), "id = \"PROJ123\"\nsync = true\n").unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(
            root.join(".git/config"),
            "[remote \"origin\"]\n\turl = git@github.com:Org/Repo.git\n",
        )
        .unwrap();
        assert_eq!(portable_key(root).as_deref(), Some("PROJ123"));
        assert!(sync_opt_in(root));
    }

    #[test]
    fn portable_key_falls_back_to_git_origin() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(
            root.join(".git/config"),
            "[core]\n\tbare = false\n[remote \"origin\"]\n\turl = https://github.com/MakerViking/Mimir.git\n",
        )
        .unwrap();
        assert_eq!(
            portable_key(root).as_deref(),
            Some("git:github.com/makerviking/mimir")
        );
        assert!(!sync_opt_in(root)); // no .mimir → not opted in
    }

    #[test]
    fn portable_key_none_without_id_or_remote() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".mimir"), b"").unwrap(); // touched, empty
        assert_eq!(portable_key(root), None);
        assert!(!sync_opt_in(root));
    }

    #[test]
    fn marker_state_tells_absent_from_unparseable() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert_eq!(read_marker_state(root), MarkerState::Absent);

        std::fs::write(root.join(".mimir"), b"").unwrap();
        assert_eq!(
            read_marker_state(root),
            MarkerState::Parsed(MimirMarker::default()),
            "a touched file is a marker with nothing set"
        );

        std::fs::write(root.join(".mimir"), "id = \"PROJ123\"\nsync = true\n").unwrap();
        assert_eq!(
            read_marker_state(root),
            MarkerState::Parsed(MimirMarker {
                id: Some("PROJ123".into()),
                sync: true
            })
        );

        // The closing quote is missing.
        std::fs::write(root.join(".mimir"), "id = \"PROJ123\nsync = true\n").unwrap();
        match read_marker_state(root) {
            MarkerState::Unparseable(why) => {
                assert!(why.contains("line 1"), "says where it stops parsing: {why}")
            }
            other => panic!("a file that is not TOML must not read as {other:?}"),
        }
        // The readers that only want the fields see it as they always did.
        assert_eq!(read_mimir_marker(root), None);
        assert_eq!(portable_key(root), None);
        assert!(!sync_opt_in(root));
    }

    #[test]
    fn marker_state_of_a_mimir_that_is_not_a_text_file() {
        // A directory named `.mimir` marks a root as well as a file does,
        // but there is nothing in it to read.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".mimir")).unwrap();
        assert!(matches!(
            read_marker_state(tmp.path()),
            MarkerState::Unparseable(_)
        ));
        assert_eq!(read_mimir_marker(tmp.path()), None);

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".mimir"), [0xff, 0xfe, 0x00]).unwrap();
        assert!(matches!(
            read_marker_state(tmp.path()),
            MarkerState::Unparseable(_)
        ));
    }

    /// What `git worktree add ../wt` leaves on disk, written by hand: the
    /// main checkout's `.git/worktrees/wt/` with its `commondir`, and the
    /// worktree whose `.git` is a file pointing there. `pointer` and
    /// `commondir` are the two files' contents, so each test can write them
    /// the way one git version or platform does.
    fn worktree_fixture(
        tmp: &Path,
        pointer: impl FnOnce(&Path) -> String,
        commondir: impl FnOnce(&Path) -> String,
    ) -> (PathBuf, PathBuf) {
        let main = tmp.join("main");
        let wt = tmp.join("wt");
        let gitdir = main.join(".git").join("worktrees").join("wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), commondir(&main.join(".git"))).unwrap();
        std::fs::write(wt.join(".git"), pointer(&gitdir)).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        (main, wt)
    }

    fn home_of(main: &Path) -> Checkout {
        Checkout::LinkedWorktree {
            home: PathBuf::from(canonical_root(main)),
        }
    }

    #[test]
    fn a_linked_worktree_names_its_main_checkout() {
        // As git writes it: an absolute gitdir, a relative commondir.
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\n", gitdir.display()),
            |_| "../..\n".into(),
        );
        let checkout = Checkout::of(&wt);
        assert_eq!(checkout, home_of(&main));
        assert!(checkout.is_linked_worktree());
        assert_eq!(checkout.to_string(), "linked_worktree");
        assert_eq!(Checkout::Standalone.as_str(), "standalone");
        assert_eq!(
            checkout.home(),
            Some(Path::new(&canonical_root(&main))),
            "home is the identity the main checkout registers under"
        );
        // The main checkout itself is its own home.
        assert_eq!(Checkout::of(&main), Checkout::Standalone);
        assert_eq!(Checkout::Standalone.home(), None);
    }

    #[test]
    fn a_linked_worktree_is_found_through_relative_and_absolute_pointers() {
        // `worktree.useRelativePaths`: the gitdir relative to the worktree.
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |_| "gitdir: ../main/.git/worktrees/wt\n".into(),
            |_| "../..\n".into(),
        );
        assert_eq!(Checkout::of(&wt), home_of(&main));

        // An absolute commondir.
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\n", gitdir.display()),
            |common| format!("{}\n", common.display()),
        );
        assert_eq!(Checkout::of(&wt), home_of(&main));

        // No trailing newline, and padding around the path.
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir:   {}  ", gitdir.display()),
            |_| "../..".into(),
        );
        assert_eq!(Checkout::of(&wt), home_of(&main));
    }

    #[test]
    fn a_worktree_written_with_crlf_line_endings_is_still_one() {
        assert_eq!(
            parse_gitdir_pointer("gitdir: C:/Users/x/repo/.git/worktrees/wt\r\n"),
            Some("C:/Users/x/repo/.git/worktrees/wt")
        );
        assert_eq!(
            parse_gitdir_pointer("gitdir: C:\\Users\\x\\repo\\.git\\worktrees\\wt\r\n"),
            Some("C:\\Users\\x\\repo\\.git\\worktrees\\wt")
        );
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\r\n", gitdir.display()),
            |_| "../..\r\n".into(),
        );
        assert_eq!(Checkout::of(&wt), home_of(&main));
    }

    #[test]
    fn a_gitdir_pointer_is_the_first_line_after_its_prefix() {
        assert_eq!(parse_gitdir_pointer("gitdir: /a/b\n"), Some("/a/b"));
        assert_eq!(parse_gitdir_pointer("gitdir:/a/b"), Some("/a/b"));
        assert_eq!(parse_gitdir_pointer("gitdir: /a/b\nextra\n"), Some("/a/b"));
        for not_a_pointer in ["", "\n", "gitdir:", "gitdir:   \n", "/a/b\n", "ref: x\n"] {
            assert_eq!(
                parse_gitdir_pointer(not_a_pointer),
                None,
                "{not_a_pointer:?}"
            );
        }
    }

    #[test]
    fn a_nested_directory_of_a_worktree_resolves_to_the_worktree_root() {
        // Detection finds the root by the `.git` file; classifying it is a
        // second step over that root.
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\n", gitdir.display()),
            |_| "../..\n".into(),
        );
        let nested = wt.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        let root = find_project_root(&nested).unwrap();
        assert_eq!(root, wt);
        assert_eq!(Checkout::of(&root), home_of(&main));
    }

    /// A worktree kept inside the main checkout (`main/.claude/worktrees/x`)
    /// is still its own root, found by its own `.git` file before the walk
    /// up reaches the main checkout's `.git` directory.
    /// A `.git` file anyone can write does not make a directory a worktree
    /// of a repository: the repository has to name it back, as git does in
    /// `<gitdir>/gitdir`. Without that, a directory could read and write
    /// the memories of any repository on the machine.
    #[test]
    fn a_pointer_the_repository_does_not_name_back_is_not_a_worktree() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\n", gitdir.display()),
            |_| "../..\n".to_string(),
        );
        assert!(Checkout::of(&wt).is_linked_worktree());

        // A second directory pointing at the same repository entry.
        let stranger = tmp.path().join("stranger");
        std::fs::create_dir_all(&stranger).unwrap();
        let gitdir = main.join(".git/worktrees/wt");
        std::fs::write(
            stranger.join(".git"),
            format!("gitdir: {}\n", gitdir.display()),
        )
        .unwrap();
        assert_eq!(Checkout::of(&stranger), Checkout::Standalone);

        // The entry without its back-pointer, or with one naming nobody.
        std::fs::write(gitdir.join("gitdir"), "/nowhere/.git\n").unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);
        std::fs::remove_file(gitdir.join("gitdir")).unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);
    }

    #[test]
    fn a_worktree_inside_the_main_checkout_is_its_own_root() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("main");
        let wt = main.join(".claude/worktrees/feature");
        let gitdir = main.join(".git/worktrees/feature");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(wt.join("src")).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();

        assert_eq!(find_project_root(&wt.join("src")), Some(wt.clone()));
        assert_eq!(Checkout::of(&wt), home_of(&main));
        assert_eq!(find_project_root(&main.join(".claude")), Some(main.clone()));
        assert_eq!(Checkout::of(&main), Checkout::Standalone);
    }

    #[test]
    fn an_ordinary_checkout_and_a_plain_directory_are_standalone() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        assert_eq!(Checkout::of(&repo), Checkout::Standalone);

        let plain = tmp.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("Cargo.toml"), b"").unwrap();
        assert_eq!(Checkout::of(&plain), Checkout::Standalone);
        assert_eq!(
            Checkout::of(&tmp.path().join("does-not-exist")),
            Checkout::Standalone
        );
    }

    #[test]
    fn a_submodule_is_standalone() {
        // A submodule's `.git` is a file too, but its gitdir (under the
        // superproject's `.git/modules`) has no `commondir`.
        let tmp = tempfile::tempdir().unwrap();
        let superproject = tmp.path().join("super");
        let module_gitdir = superproject.join(".git/modules/dep");
        let submodule = superproject.join("dep");
        std::fs::create_dir_all(&module_gitdir).unwrap();
        std::fs::create_dir_all(&submodule).unwrap();
        std::fs::write(module_gitdir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(submodule.join(".git"), "gitdir: ../.git/modules/dep\n").unwrap();
        assert_eq!(Checkout::of(&submodule), Checkout::Standalone);
    }

    #[test]
    fn a_worktree_of_a_bare_repository_is_standalone() {
        // The common directory is `repo.git`: there is no main checkout
        // beside it to file memories under.
        let tmp = tempfile::tempdir().unwrap();
        let bare = tmp.path().join("repo.git");
        let gitdir = bare.join("worktrees/wt");
        let wt = tmp.path().join("wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);
    }

    /// A linked worktree whose common directory is `common`, made by hand:
    /// `common/worktrees/wt/commondir` and `<tmp>/wt/.git` pointing there.
    fn worktree_of_common(tmp: &Path, common: &Path) -> PathBuf {
        let gitdir = common.join("worktrees/wt");
        let wt = tmp.join("wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        std::fs::write(
            gitdir.join("gitdir"),
            format!("{}\n", wt.join(".git").display()),
        )
        .unwrap();
        wt
    }

    /// A bare repository kept in a directory called `.git`, its worktrees
    /// beside it (`git clone --bare <url> .git`): the parent is not a
    /// checkout, and its config says so.
    #[test]
    fn a_worktree_of_a_bare_repository_called_dot_git_is_standalone() {
        let tmp = tempfile::tempdir().unwrap();
        let common = tmp.path().join(".git");
        std::fs::create_dir_all(&common).unwrap();
        let wt = worktree_of_common(tmp.path(), &common);
        // Not bare: the parent is the main checkout.
        std::fs::write(common.join("config"), "[core]\n\tbare = false\n").unwrap();
        assert_eq!(Checkout::of(&wt), home_of(tmp.path()));
        for bare in [
            "[core]\n\tbare = true\n",
            "[core]\n\tbare\n",
            "[CORE]\n\tBare = Yes\n",
            "[core] bare = 1\n",
            "[core]\n\tbare = false\n[core]\n\tbare = true ; set by clone\n",
        ] {
            std::fs::write(common.join("config"), bare).unwrap();
            assert_eq!(Checkout::of(&wt), Checkout::Standalone, "{bare:?}");
        }
    }

    /// A repository whose config names its main checkout (`core.worktree`,
    /// relative to the repository directory): a submodule's repository
    /// under the superproject's `.git/modules`, and a worktree of it.
    #[test]
    fn core_worktree_names_the_home_when_it_leads_back_to_the_repository() {
        let tmp = tempfile::tempdir().unwrap();
        let superproject = tmp.path().join("super");
        let common = superproject.join(".git/modules/dep");
        let checkout = superproject.join("dep");
        std::fs::create_dir_all(&common).unwrap();
        std::fs::create_dir_all(&checkout).unwrap();
        std::fs::write(
            common.join("config"),
            "[core]\n\tbare = false\n\tworktree = ../../../dep\n",
        )
        .unwrap();
        std::fs::write(checkout.join(".git"), "gitdir: ../.git/modules/dep\n").unwrap();
        let wt = worktree_of_common(tmp.path(), &common);
        assert_eq!(Checkout::of(&wt), home_of(&checkout));
        assert_eq!(Checkout::of(&checkout), Checkout::Standalone);

        // An absolute `core.worktree`, quoted, as `git init
        // --separate-git-dir` plus `git config core.worktree` leave it.
        std::fs::write(
            common.join("config"),
            format!("[core]\n\tworktree = \"{}\"\n", checkout.display()),
        )
        .unwrap();
        assert_eq!(Checkout::of(&wt), home_of(&checkout));

        // It names no directory.
        std::fs::write(
            common.join("config"),
            "[core]\n\tworktree = ../../../gone\n",
        )
        .unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);

        // It names a directory whose `.git` is another repository's, or
        // which has none.
        let other = tmp.path().join("other");
        std::fs::create_dir_all(other.join(".git")).unwrap();
        let bare_dir = tmp.path().join("plain");
        std::fs::create_dir_all(&bare_dir).unwrap();
        for elsewhere in [&other, &bare_dir] {
            std::fs::write(
                common.join("config"),
                format!("[core]\n\tworktree = {}\n", elsewhere.display()),
            )
            .unwrap();
            assert_eq!(
                Checkout::of(&wt),
                Checkout::Standalone,
                "{}",
                elsewhere.display()
            );
        }

        // And a `core.worktree` in a subsection is not the repository's.
        std::fs::write(
            common.join("config"),
            "[core \"x\"]\n\tworktree = ../../../dep\n",
        )
        .unwrap();
        assert_eq!(
            Checkout::of(&wt),
            Checkout::Standalone,
            "no `core.worktree`, and the repository is not called `.git`"
        );
    }

    #[test]
    fn the_core_config_reader_reads_what_git_writes() {
        assert_eq!(parse_core_config(""), CoreConfig::default());
        assert_eq!(
            parse_core_config(
                "[core]\n\trepositoryformatversion = 0\n\tbare = false\n\
                 \tworktree = ../../../dep\n[remote \"origin\"]\n\turl = x\n"
            ),
            CoreConfig {
                bare: false,
                worktree: Some("../../../dep".into())
            }
        );
        assert_eq!(
            parse_core_config("[core]\n# bare = true\n; bare = true\n\tworktree = \"/a b\" # c\n"),
            CoreConfig {
                bare: false,
                worktree: Some("/a b".into())
            }
        );
        assert_eq!(
            parse_core_config("[remote \"origin\"]\n\tbare = true\n\tworktree = /x\n"),
            CoreConfig::default(),
            "keys of another section"
        );
        assert_eq!(
            parse_core_config("[core.sub]\n\tbare = true\n[Core]\n\tWorkTree = /y\n"),
            CoreConfig {
                bare: false,
                worktree: Some("/y".into())
            }
        );
        for (value, bare) in [
            ("true", true),
            ("on", true),
            ("2", true),
            ("false", false),
            ("off", false),
            ("0", false),
            ("maybe", false),
        ] {
            assert_eq!(
                parse_core_config(&format!("[core]\nbare = {value}\n")).bare,
                bare,
                "{value}"
            );
        }
    }

    /// `git worktree add` writes `commondir` once; its modification time is
    /// when the worktree was made, rounded up to the second to bound what
    /// may move into it, down to say what is older than it.
    #[test]
    fn a_worktrees_creation_time_is_when_its_commondir_was_written() {
        let tmp = tempfile::tempdir().unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |gitdir| format!("gitdir: {}\n", gitdir.display()),
            |_| "../..\n".into(),
        );
        let commondir = main.join(".git/worktrees/wt/commondir");
        let set = |at: std::time::SystemTime| {
            std::fs::File::options()
                .write(true)
                .open(&commondir)
                .unwrap()
                .set_modified(at)
                .unwrap();
        };
        let epoch = std::time::UNIX_EPOCH;
        set(epoch + std::time::Duration::from_secs(1_700_000_000));
        assert_eq!(worktree_created_at(&wt), Some(1_700_000_000));
        set(epoch + std::time::Duration::from_millis(1_700_000_000_250));
        assert_eq!(worktree_created_at(&wt), Some(1_700_000_001), "rounded up");

        assert!(worktree_made_after(&wt, 1_699_999_999));
        assert!(!worktree_made_after(&wt, 1_700_000_000), "the same second");
        assert!(!worktree_made_after(&wt, 1_700_000_001));

        assert_eq!(worktree_created_at(&main), None, "not a linked worktree");
        assert!(!worktree_made_after(&main, 0));
        std::fs::remove_file(&commondir).unwrap();
        assert_eq!(worktree_created_at(&wt), None, "nothing to read");
        assert!(!worktree_made_after(&wt, 0));
    }

    #[test]
    fn unreadable_or_malformed_worktree_files_are_standalone() {
        let git_layout = |pointer: &dyn Fn(&Path) -> String, commondir: Option<&str>| {
            let tmp = tempfile::tempdir().unwrap();
            let (_main, wt) = worktree_fixture(
                tmp.path(),
                |gitdir| pointer(gitdir),
                |_| commondir.unwrap_or("../..\n").to_string(),
            );
            if commondir.is_none() {
                std::fs::remove_file(tmp.path().join("main/.git/worktrees/wt/commondir")).unwrap();
            }
            Checkout::of(&wt)
        };
        let good = |gitdir: &Path| format!("gitdir: {}\n", gitdir.display());
        // The fixture is a worktree until one part of it is broken.
        assert!(git_layout(&good, Some("../..\n")).is_linked_worktree());

        // `.git` is a file that is not a pointer.
        assert_eq!(
            git_layout(&|_| String::new(), Some("../..\n")),
            Checkout::Standalone
        );
        assert_eq!(
            git_layout(&|_| "not a pointer\n".into(), Some("../..\n")),
            Checkout::Standalone
        );
        // It points at a gitdir that is not there.
        assert_eq!(
            git_layout(&|_| "gitdir: /no/such/gitdir\n".into(), Some("../..\n")),
            Checkout::Standalone
        );
        // No `commondir`, an empty one, one naming nothing, and one naming
        // a directory that is not called `.git`.
        assert_eq!(git_layout(&good, None), Checkout::Standalone);
        assert_eq!(git_layout(&good, Some("")), Checkout::Standalone);
        assert_eq!(
            git_layout(&good, Some("../../../missing/.git\n")),
            Checkout::Standalone
        );
        assert_eq!(git_layout(&good, Some("..\n")), Checkout::Standalone);

        // `.git` is a file that is not text.
        let tmp = tempfile::tempdir().unwrap();
        let (_main, wt) = worktree_fixture(tmp.path(), |_| String::new(), |_| "../..\n".into());
        std::fs::write(wt.join(".git"), [0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);

        // The common directory is a file called `.git`, not a directory.
        let tmp = tempfile::tempdir().unwrap();
        let (_main, wt) = worktree_fixture(tmp.path(), good, |_| "../..\n".into());
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join(".git"), "gitdir: nowhere\n").unwrap();
        std::fs::write(
            tmp.path().join("main/.git/worktrees/wt/commondir"),
            format!("{}\n", elsewhere.join(".git").display()),
        )
        .unwrap();
        assert_eq!(Checkout::of(&wt), Checkout::Standalone);
    }

    /// The home is canonical, so it is the same string the main checkout
    /// registers under even when the worktree's pointer reaches it through
    /// a symlink.
    #[cfg(unix)]
    #[test]
    fn the_home_is_canonical_through_a_symlinked_pointer() {
        let tmp = tempfile::tempdir().unwrap();
        let alias = tmp.path().join("alias");
        std::os::unix::fs::symlink(tmp.path().join("main"), &alias).unwrap();
        let (main, wt) = worktree_fixture(
            tmp.path(),
            |_| format!("gitdir: {}\n", alias.join(".git/worktrees/wt").display()),
            |_| "../..\n".into(),
        );
        assert_eq!(Checkout::of(&wt), home_of(&main));
        assert_eq!(
            Checkout::of(&wt).home().unwrap().to_string_lossy(),
            canonical_root(&alias),
            "and the alias canonicalizes to the same identity"
        );
    }

    #[test]
    fn normalize_remote_forms_converge() {
        for url in [
            "git@github.com:Org/Repo.git",
            "https://github.com/Org/Repo.git",
            "ssh://git@github.com/Org/Repo.git",
            "https://github.com/Org/Repo",
        ] {
            assert_eq!(normalize_remote(url), "github.com/org/repo", "{url}");
        }
    }

    #[test]
    fn simplify_verbatim_rewrites_unc_and_drive_paths() {
        // Drive paths just lose the verbatim prefix.
        assert_eq!(simplify_verbatim(r"\\?\C:\dir\proj"), r"C:\dir\proj");
        // UNC paths must become a real \\server\share path, not "UNC\...".
        assert_eq!(
            simplify_verbatim(r"\\?\UNC\server\share\proj"),
            r"\\server\share\proj"
        );
        // Already-normal paths are left untouched.
        assert_eq!(
            simplify_verbatim(r"\\server\share\proj"),
            r"\\server\share\proj"
        );
        assert_eq!(simplify_verbatim("/home/user/proj"), "/home/user/proj");
    }
}
