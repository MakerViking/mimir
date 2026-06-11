use std::path::{Path, PathBuf};

/// Walk up from `start` to find the enclosing git repository root.
/// Returns None when `start` is not inside a repo.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(start)
    };
    loop {
        if dir.join(".git").exists() {
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
