use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("config error: {0}")]
    Config(String),

    #[error(
        "database schema version {found} is newer than this binary supports ({supported}); upgrade mimir"
    )]
    SchemaTooNew { found: i64, supported: i64 },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("ambiguous reference '{0}' matches {1} nodes; use a longer prefix")]
    AmbiguousRef(String, usize),

    #[error("{0}")]
    Invalid(String),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
