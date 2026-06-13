use std::path::{Path, PathBuf};

/// Markers that identify a project root, in priority order. `.git` covers
/// the common case; the other VCS dirs catch non-git repos; `.mimir` is the
/// explicit opt-in for code outside any VCS (Windows drives, unpacked
/// archives, config trees) — `touch .mimir` makes a directory a project
/// root.
pub const ROOT_MARKERS: &[&str] = &[".git", ".hg", ".svn", ".jj", ".mimir"];

/// Walk up from `start` to find the enclosing project root (a directory
/// containing one of `ROOT_MARKERS`). Returns None when none is found.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(start)
    };
    loop {
        if ROOT_MARKERS.iter().any(|m| dir.join(m).exists()) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Canonical string form of a project root, used as project identity.
pub fn canonical_root(root: &Path) -> String {
    dunce_canonicalize(root).to_string_lossy().into_owned()
}

/// Best-effort canonicalization that avoids Windows \\?\ extended paths.
fn dunce_canonicalize(path: &Path) -> PathBuf {
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    // Strip Windows verbatim prefix for stable, human-readable identity.
    let s = canon.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        canon
    }
}

/// Project display name: the root directory's file name.
pub fn project_name(root: &Path) -> String {
    root.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.to_string_lossy().into_owned())
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
}
