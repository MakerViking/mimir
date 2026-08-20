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

// How sure the *author* was when they wrote it — never how useful it has
// since proved.
//
// Mimir already tracks how often a memory gets used (`strength`, which
// decays) and whether it has been retired (`superseded_by`, `expires_at`,
// and `grounding`). What it had no way to say is "I was guessing." Those
// are different questions, and folding them into one number is how a guess
// becomes canon: recall it enough and it outranks things that were checked,
// with nothing left to show it was ever uncertain.
//
// Author-declared at capture, deliberately — the same argument
// `expires_at` and `resolves_when` are built on. The person writing the
// memory is the only one who knows how sure they were, they know it at
// write time, and nobody ever backfills. The alternative, a confidence
// score inferred later by a model, is precisely the unfalsifiable policy
// label this is meant to replace: it would be computed from the text, so
// it could never contradict the text.
//
// Absent is a real state and is NOT the same as `Likely`. Defaulting an
// undeclared memory to the middle would put words in the author's mouth,
// the same "reject, don't mangle" call `parse_expires_in` and
// `sanitize_fires_when` already make.
str_enum!(MemoryConfidence {
    Certain => "certain",
    Likely => "likely",
    Unsure => "unsure",
});

// Who performed a mutation. Threaded explicitly to the store functions that
// change a memory rather than read from ambient state, because an audit whose
// actor can be silently wrong is worse than no audit — the same argument
// `MemoryConfidence` makes for refusing to infer a level nobody declared.
//
// `System` is a background pass (consolidation's decay, an indexer, a graph
// rebuild). `Sync` is a change that arrived from another install, where the
// actor is whoever acted on the machine it came from — recorded here as the
// transport, since this store cannot know more than that.
str_enum!(Actor {
    Human => "human",
    Agent => "agent",
    System => "system",
    Sync => "sync",
});

// What was done. A closed set: an open one drifts into free text, and then
// nothing can be counted.
str_enum!(MutationOp {
    Create => "create",
    Edit => "edit",
    Supersede => "supersede",
    Forget => "forget",
    Archive => "archive",
    HardDelete => "hard_delete",
    Reproject => "reproject",
    Pin => "pin",
    Unpin => "unpin",
    Restore => "restore",
    Review => "review",
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

    /// Unix time after which this memory stops being true, if it declared one.
    ///
    /// Stored in `meta.expires_at` rather than as a column, deliberately:
    /// `meta` already replicates (see `replicate::SyncRecord`) while new
    /// columns do not, so this crosses machines for free and needs no schema
    /// bump — which also means no `SchemaTooNew` risk for a peer running an
    /// older binary.
    pub fn expires_at(&self) -> Option<i64> {
        self.meta.get("expires_at")?.as_i64()
    }

    /// Free-text condition under which this memory stops being true, e.g.
    /// "when Deluge ships libtorrent 2.1 support".
    ///
    /// Deliberately NOT machine-evaluated and deliberately NOT a gate: we
    /// cannot know whether the condition has been met, so a memory carrying
    /// one still surfaces normally. Its job is to travel *with* the memory so
    /// the reader knows what would falsify it, instead of silently trusting a
    /// fact whose expiry nobody recorded.
    pub fn resolves_when(&self) -> Option<&str> {
        self.meta.get("resolves_when")?.as_str()
    }

    /// What the author said about their own certainty, if they said
    /// anything. `None` means undeclared, which is the common case and is
    /// not a synonym for medium confidence.
    ///
    /// Read-only as far as ranking is concerned: nothing in `search` or
    /// `learn` consults this. It exists so a reader — human or agent — can
    /// tell a checked fact from a guess that has merely been recalled a
    /// lot, which `strength` alone can never distinguish.
    pub fn confidence(&self) -> Option<MemoryConfidence> {
        self.meta.get("confidence")?.as_str()?.parse().ok()
    }

    /// When the fact this memory states began being true in the world, and
    /// when it stopped — its *valid time*, as distinct from the transaction
    /// time already on the row (`created_at`, `updated_at`, `deleted_at`,
    /// plus the dated ops in [`crate::audit`]).
    ///
    /// The two axes answer different questions and conflating them loses
    /// both. "I was a vegetarian from 2019 until 2024" and "I recorded that
    /// on Tuesday" are independent, and only the pair can answer "what was
    /// true in 2022?" separately from "what did I believe in 2022?".
    ///
    /// Author-declared and open-ended at both ends, on the same argument
    /// `expires_at` and `confidence` rest on: the writer is the only one who
    /// knows, they know it at write time, and nobody backfills. Absent is a
    /// real state — an undeclared interval is *unknown*, not `-∞..∞`, so
    /// nothing here filters recall. `expires_at` remains the separate,
    /// deliberate gate; a memory can be valid-until-2024 and still worth
    /// surfacing, which is exactly why "no longer current" is rendered rather
    /// than hidden.
    pub fn valid_from(&self) -> Option<i64> {
        self.meta.get("valid_from")?.as_i64()
    }

    /// See [`Node::valid_from`].
    pub fn valid_to(&self) -> Option<i64> {
        self.meta.get("valid_to")?.as_i64()
    }

    /// True when a declared validity interval has closed — the fact was true
    /// once and, by the author's own account, no longer is.
    ///
    /// This is the case worth a reader's attention, and the only one that
    /// earns a token on the compact recall line: an open interval says
    /// "still true as far as anyone said", which is what every undeclared
    /// memory already implies.
    pub fn validity_closed(&self, now: i64) -> bool {
        self.valid_to().is_some_and(|to| to <= now)
    }

    /// Whether the fact was true at `at`, as far as the author declared.
    /// `None` when nothing was declared — unknown, not false.
    pub fn valid_at(&self, at: i64) -> Option<bool> {
        let (from, to) = (self.valid_from(), self.valid_to());
        if from.is_none() && to.is_none() {
            return None;
        }
        Some(from.is_none_or(|f| at >= f) && to.is_none_or(|t| at < t))
    }

    /// True once a declared `expires_at` has passed.
    ///    /// This is a *gate*, unlike strength decay — see `learn::effective_strength`,
    /// where decay is explicitly "a tiebreaker multiplier, never a burier". Decay
    /// models "gradually less relevant"; it cannot model "became false on a
    /// specific date", so an expired memory would otherwise keep winning its
    /// query forever at a fractionally lower score.
    pub fn is_expired(&self, now: i64) -> bool {
        self.expires_at().is_some_and(|t| t <= now)
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
