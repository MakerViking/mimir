use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

macro_rules! str_enum {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $text),+
                }
            }

            pub const ALL: &'static [$name] = &[$($name::$variant),+];
        }

        impl std::str::FromStr for $name {
            type Err = Error;
            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $($text => Ok($name::$variant),)+
                    other => Err(Error::Invalid(format!(
                        concat!("unknown ", stringify!($name), " '{}'"), other
                    ))),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

// CodeChunk: source-code content chunked on tree-sitter symbol boundaries
// (see index::chunker::chunk_source) — distinct from Symbol (signature +
// first doc line only) and from Chunk (markdown docs).
str_enum!(Kind {
    Memory => "memory",
    Collection => "collection",
    File => "file",
    Chunk => "chunk",
    Project => "project",
    Symbol => "symbol",
    Community => "community",
    Tag => "tag",
    Annotation => "annotation",
    CodeChunk => "codechunk",
});

str_enum!(MemoryType {
    Insight => "insight",
    Decision => "decision",
    Gotcha => "gotcha",
    Idea => "idea",
    Note => "note",
    Person => "person",
    Summary => "summary",
});

str_enum!(Rel {
    Links => "links",
    Mentions => "mentions",
    Tagged => "tagged",
    Describes => "describes",
    About => "about",
    Relates => "relates",
    Supersedes => "supersedes",
    Summarizes => "summarizes",
    DerivedFrom => "derived_from",
    Calls => "calls",
    Imports => "imports",
    Defines => "defines",
    MemberOf => "member_of",
});

/// Where a query or capture applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// No scope filter: global + every project.
    All,
    /// Only nodes with no project (cross-project).
    Global,
    /// Only nodes of one project (by node rowid).
    Project(i64),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: i64,
    pub uid: String,
    pub kind: Kind,
    pub subkind: Option<String>,
    pub project_id: Option<i64>,
    pub collection_id: Option<i64>,
    pub parent_id: Option<i64>,
    pub title: Option<String>,
    pub body: Option<String>,
    pub path: Option<String>,
    pub span_start: Option<i64>,
    pub span_end: Option<i64>,
    pub lang: Option<String>,
    pub tags_text: String,
    pub content_hash: Option<Vec<u8>>,
    pub meta: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
    pub access_count: i64,
    pub last_accessed: Option<i64>,
    pub strength: f64,
    pub pinned: bool,
    pub superseded_by: Option<i64>,
    pub deleted_at: Option<i64>,
}

impl Node {
    pub fn tags(&self) -> Vec<&str> {
        self.tags_text.split_whitespace().collect()
    }
}

/// Fields for inserting a new node; everything not listed gets a default.
#[derive(Debug, Clone, Default)]
pub struct NewNode {
    pub kind: Option<Kind>,
    pub subkind: Option<String>,
    pub project_id: Option<i64>,
    pub collection_id: Option<i64>,
    pub parent_id: Option<i64>,
    pub title: Option<String>,
    pub body: Option<String>,
    pub path: Option<String>,
    pub span_start: Option<i64>,
    pub span_end: Option<i64>,
    pub lang: Option<String>,
    pub tags: Vec<String>,
    pub content_hash: Option<Vec<u8>>,
    pub meta: Option<serde_json::Value>,
}

impl NewNode {
    pub fn new(kind: Kind) -> Self {
        NewNode {
            kind: Some(kind),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub id: i64,
    pub src: i64,
    pub dst: i64,
    pub rel: Rel,
    pub weight: f64,
    pub meta: serde_json::Value,
    pub created_at: i64,
}

pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Short display form of a node uid: sigil + first 6 chars after the
/// timestamp prefix would be ambiguous, so we use the LAST 6 chars of the
/// ULID (its random part), which is what makes ids visually distinct.
pub fn short_uid(kind: Kind, uid: &str) -> String {
    let sigil = match kind {
        Kind::Memory => 'm',
        Kind::Chunk | Kind::File | Kind::Annotation => 'd',
        Kind::Symbol | Kind::Community | Kind::CodeChunk => 'c',
        Kind::Project => 'p',
        Kind::Collection => 'l',
        Kind::Tag => 't',
    };
    let tail: String = uid
        .chars()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{sigil}:{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_enum_round_trips() {
        for kind in Kind::ALL {
            assert_eq!(kind.as_str().parse::<Kind>().unwrap(), *kind);
        }
        assert!("bogus".parse::<Kind>().is_err());
        assert_eq!("gotcha".parse::<MemoryType>().unwrap(), MemoryType::Gotcha);
    }

    #[test]
    fn short_uid_uses_tail() {
        let uid = "01JXGABCDEF123456789QRSTUV";
        assert_eq!(short_uid(Kind::Memory, uid), "m:QRSTUV");
        assert_eq!(short_uid(Kind::Project, uid), "p:QRSTUV");
    }
}
