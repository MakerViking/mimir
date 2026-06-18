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
/// carries where the search started so callers can explain the global fallback.
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
/// 3. else `NotFound` (global scope, explained by the caller).
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

/// Read and parse the `.mimir` marker as TOML (`id = "..."`, `sync = true`).
/// `None` if absent or unparseable; an empty file parses to defaults.
pub fn read_mimir_marker(root: &Path) -> Option<MimirMarker> {
    let txt = std::fs::read_to_string(root.join(".mimir")).ok()?;
    let val: toml::Value = toml::from_str(&txt).ok()?;
    Some(MimirMarker {
        id: val.get("id").and_then(|v| v.as_str()).map(str::to_owned),
        sync: val.get("sync").and_then(|v| v.as_bool()).unwrap_or(false),
    })
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
