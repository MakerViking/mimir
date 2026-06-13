//! Owner-only file writes.
//!
//! The dashboard and graph-viz embed private memory contents (titles,
//! bodies, decisions) into HTML written under the shared temp dir. A plain
//! `fs::write` lands at umask-default 0644, so any other local user can read
//! it. Write at 0600 instead, and re-assert the mode in case a looser file
//! was left behind by an earlier run.

use std::io::Write;
use std::path::Path;

pub fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        let mut f = opts.open(path)?;
        f.write_all(contents.as_bytes())?;
        // A pre-existing file keeps its old mode on open; force 0600.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        // Windows: the temp dir is per-user under the profile; fs::write is
        // adequate and the unix mode bits don't apply.
        let mut f = std::fs::File::create(path)?;
        f.write_all(contents.as_bytes())
    }
}
