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

impl MemoryType {
    /// The type a memory's `subkind` column names.
    ///
    /// The column is nullable, and a memory without one is a `Note` — the
    /// type `remember` gives a capture that named none. A value this build
    /// does not know (one written by a newer build) is `None`: the memory
    /// is still a memory, but nothing type-specific is claimed about it.
    pub fn from_subkind(subkind: Option<&str>) -> Option<MemoryType> {
        match subkind {
            None => Some(MemoryType::Note),
            Some(text) => text.parse().ok(),
        }
    }

    /// Gotchas and decisions: the two types that exist to stop a mistake
    /// being made twice. They are the only ones the session brief admits
    /// and per-prompt injection fires for unpinned, and the ones ranking
    /// nudges ahead of an equal match (`learn::type_prior`).
    pub fn is_preventer(self) -> bool {
        matches!(self, MemoryType::Gotcha | MemoryType::Decision)
    }
}

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

// What `recall_event.event` records about one node: the implicit-feedback
// ledger behind strength, the impression damp, co-activation and the shadow
// ranker. The column is free text; this is the whole vocabulary, so a writer
// and the readers that count its rows cannot drift apart by a typo.
//
// `Shown` is an impression (the hit was on a result page). `Opened` is an
// agent pulling the body — evidence, where an impression is not. `Useful`
// and `NotUseful` are explicit marks; the surfaces call the second "noise".
str_enum!(RecallEvent {
    Shown => "shown",
    Opened => "opened",
    Useful => "useful",
    NotUseful => "not_useful",
});

impl RecallEvent {
    /// The reader acted on the hit: opened it, or marked it useful. The
    /// dashboard's activity chart draws these two as one series ("opened /
    /// marked useful"); everything that says *opened* means
    /// [`RecallEvent::Opened`] alone.
    pub fn is_engagement(self) -> bool {
        matches!(self, RecallEvent::Opened | RecallEvent::Useful)
    }
}

// Edge relations. All are authored — by a person, by the code-graph
// extractor, or by an importer — except `CoRecalled`, which the store
// observes for itself (see `learn::record_coactivation`). That one is
// derived, per-machine and regenerable, so it is excluded from replication
// and from export rather than shipped as if it were a claim anyone made.
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
    CoRecalled => "co_recalled",
});

/// Who writes an edge of a given relation. The single list behind which
/// relations `mimir link` and the MCP `link` tool will author, what their
/// help says they accept, and which edges count as links somebody made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOrigin {
    /// A claim somebody states: `link`, `remember --link`, an importer, or
    /// the consolidation pass recording which memories a summary covers.
    Authored,
    /// Written only by retiring a memory, together with the `superseded_by`
    /// column that actually hides it. Authored on its own it would read as a
    /// retirement and retire nothing.
    Retirement,
    /// Derived by Mimir from what it indexes (the code-graph extractor's
    /// calls and imports, community membership, tags) and replaced when the
    /// source is re-read, so a hand-made one is not a link Mimir keeps.
    Extracted,
    /// Observed by the store from its own usage, per machine.
    Observed,
}

impl Rel {
    pub fn origin(self) -> RelOrigin {
        match self {
            Rel::Links
            | Rel::Mentions
            | Rel::Describes
            | Rel::About
            | Rel::Relates
            | Rel::Summarizes
            | Rel::DerivedFrom => RelOrigin::Authored,
            Rel::Supersedes => RelOrigin::Retirement,
            Rel::Tagged | Rel::Calls | Rel::Imports | Rel::Defines | Rel::MemberOf => {
                RelOrigin::Extracted
            }
            Rel::CoRecalled => RelOrigin::Observed,
        }
    }

    /// The relations a person or an agent may author by hand, in the order
    /// the vocabulary declares them.
    pub fn authorable() -> impl Iterator<Item = Rel> {
        Rel::ALL
            .iter()
            .copied()
            .filter(|rel| rel.origin() == RelOrigin::Authored)
    }
}

/// Where `now` falls against the interval a memory's author declared for the
/// fact it states (`meta.valid_from` .. `meta.valid_to`, start-inclusive and
/// end-exclusive). Valid time only: none of these hides a memory from
/// recall, which is what `expires_at` is for — they say how to read it.
///
/// Derived in one place, [`Node::validity`]; a surface that renders
/// validity matches on this instead of comparing the two timestamps itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Validity {
    /// Neither end was declared. Unknown, which is the common case and is
    /// not the same as "always true".
    Undeclared,
    /// Declared, and `now` is inside it: true as far as the author said.
    Open,
    /// Declared to begin later than `now`: recorded ahead of the fact.
    NotYet,
    /// Declared to have ended by `now`: true once, no longer.
    Closed,
}

/// Where a query or capture applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// No scope filter: global + every project.
    All,
    /// Only nodes with no project (cross-project).
    Global,
    /// One project (by node rowid) together with the global nodes: what
    /// someone working in that project can see. Never another project's.
    Project(i64),
}

impl Scope {
    /// Whether a node owned by `project_id` (`None` = global) falls inside
    /// this scope. The Rust twin of `search::scope_sql`, for the places that
    /// reach a node by following a link rather than by running a scoped
    /// query: a project scope admits its own nodes and the global ones.
    pub fn admits(self, project_id: Option<i64>) -> bool {
        match self {
            Scope::All => true,
            Scope::Global => project_id.is_none(),
            Scope::Project(id) => project_id.is_none() || project_id == Some(id),
        }
    }
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

    /// This memory's type, or `None` for a node that is not a memory (its
    /// `subkind` means something else: a symbol's is `function`, `struct`…)
    /// or whose stored type this build does not know. A memory with no
    /// `subkind` at all is a `Note` — see [`MemoryType::from_subkind`].
    ///
    /// Decay, ranking, injection and display all ask this, so "no subkind"
    /// cannot mean a note to one of them and something else to the next.
    pub fn memory_type(&self) -> Option<MemoryType> {
        if self.kind != Kind::Memory {
            return None;
        }
        MemoryType::from_subkind(self.subkind.as_deref())
    }

    /// The one word a line or record header uses for what this node is: a
    /// memory's type, otherwise the node's own `subkind`, otherwise its
    /// kind. A memory of a type this build does not know keeps the word it
    /// was stored with.
    pub fn type_label(&self) -> &str {
        match self.memory_type() {
            Some(mtype) => mtype.as_str(),
            None => self.subkind.as_deref().unwrap_or(self.kind.as_str()),
        }
    }

    /// Unix time after which this memory stops being true, if it declared one.
    ///
    /// Stored in `meta.expires_at` rather than as a column, deliberately:
    /// `meta` already replicates (see `replicate::SyncRecord`) while new
    /// columns do not, so this crosses machines for free and needs no schema
    /// bump — which also means no `SchemaTooNew` risk for a peer running an
    /// older binary.
    ///
    /// Any JSON number counts, with or without a decimal point, because the
    /// SQL that reads this key (`review`'s expired-memory sweep) compares it
    /// as a number either way. A fractional value rounds up: it is the first
    /// whole second at which `expires_at <= now` holds, so [`Node::is_expired`]
    /// and that sweep flip on the same second.
    pub fn expires_at(&self) -> Option<i64> {
        let value = self.meta.get("expires_at")?;
        value
            .as_i64()
            .or_else(|| value.as_f64().map(|secs| secs.ceil() as i64))
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

    /// This memory's [`Validity`] at `now` (unix seconds).
    ///
    /// When an interval is both over and not yet begun (an inverted one,
    /// which `store::set_valid_interval` refuses but a peer could send),
    /// the end wins: `Closed` is the reading that stops a reader acting on
    /// the fact.
    pub fn validity(&self, now: i64) -> Validity {
        let (from, to) = (self.valid_from(), self.valid_to());
        if from.is_none() && to.is_none() {
            return Validity::Undeclared;
        }
        if to.is_some_and(|to| to <= now) {
            return Validity::Closed;
        }
        if from.is_some_and(|from| from > now) {
            return Validity::NotYet;
        }
        Validity::Open
    }

    /// True when a declared validity interval has closed — the fact was true
    /// once and, by the author's own account, no longer is
    /// ([`Validity::Closed`]).
    ///
    /// This is the case worth a reader's attention on the compact recall
    /// line, along with its mirror [`Validity::NotYet`]: an open interval
    /// says "still true as far as anyone said", which is what every
    /// undeclared memory already implies.
    pub fn validity_closed(&self, now: i64) -> bool {
        self.validity(now) == Validity::Closed
    }

    /// Whether the fact was true at `at`, as far as the author declared.
    /// `None` when nothing was declared — unknown, not false.
    pub fn valid_at(&self, at: i64) -> Option<bool> {
        match self.validity(at) {
            Validity::Undeclared => None,
            Validity::Open => Some(true),
            Validity::NotYet | Validity::Closed => Some(false),
        }
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

    fn bare(kind: Kind, subkind: Option<&str>) -> Node {
        Node {
            id: 1,
            uid: "01JXGABCDEF123456789QRSTUV".into(),
            kind,
            subkind: subkind.map(str::to_string),
            project_id: None,
            collection_id: None,
            parent_id: None,
            title: None,
            body: None,
            path: None,
            span_start: None,
            span_end: None,
            lang: None,
            tags_text: String::new(),
            content_hash: None,
            meta: serde_json::json!({}),
            created_at: 0,
            updated_at: 0,
            access_count: 0,
            last_accessed: None,
            strength: 1.0,
            pinned: false,
            superseded_by: None,
            deleted_at: None,
        }
    }

    #[test]
    fn a_memory_without_a_subkind_is_a_note_everywhere() {
        let untyped = bare(Kind::Memory, None);
        assert_eq!(untyped.memory_type(), Some(MemoryType::Note));
        assert_eq!(untyped.type_label(), "note");

        for mtype in MemoryType::ALL {
            let node = bare(Kind::Memory, Some(mtype.as_str()));
            assert_eq!(node.memory_type(), Some(*mtype));
            assert_eq!(node.type_label(), mtype.as_str());
        }

        // A type from a newer build: still a memory, nothing claimed about
        // its type, and shown under the word it was stored with.
        let newer = bare(Kind::Memory, Some("hypothesis"));
        assert_eq!(newer.memory_type(), None);
        assert_eq!(newer.type_label(), "hypothesis");

        // Another kind's subkind is not a memory type, even when it spells one.
        let symbol = bare(Kind::Symbol, Some("function"));
        assert_eq!(symbol.memory_type(), None);
        assert_eq!(symbol.type_label(), "function");
        assert_eq!(bare(Kind::Chunk, Some("gotcha")).memory_type(), None);
        assert_eq!(bare(Kind::Chunk, None).type_label(), "chunk");
    }

    #[test]
    fn recall_events_keep_their_stored_spelling() {
        let stored: Vec<&str> = RecallEvent::ALL.iter().map(|e| e.as_str()).collect();
        assert_eq!(stored, ["shown", "opened", "useful", "not_useful"]);
        let engaged: Vec<RecallEvent> = RecallEvent::ALL
            .iter()
            .copied()
            .filter(|e| e.is_engagement())
            .collect();
        assert_eq!(engaged, [RecallEvent::Opened, RecallEvent::Useful]);
    }

    #[test]
    fn preventers_are_gotchas_and_decisions() {
        let preventers: Vec<MemoryType> = MemoryType::ALL
            .iter()
            .copied()
            .filter(|t| t.is_preventer())
            .collect();
        assert_eq!(preventers, vec![MemoryType::Decision, MemoryType::Gotcha]);
    }

    fn with_meta(meta: serde_json::Value) -> Node {
        let mut node = bare(Kind::Memory, Some("note"));
        node.meta = meta;
        node
    }

    /// `review` finds expired memories in SQL, where a JSON number is a
    /// number whether or not it was written with a decimal point. Rust read
    /// the same key with `as_i64`, which is `None` for `1700000000.0` — so a
    /// memory whose expiry arrived as a float was past due to the review
    /// queue and immortal to recall. Checked against the SQL expression
    /// itself, across the second on either side of the boundary.
    #[test]
    fn expiry_reads_a_json_float_the_way_sql_compares_it() {
        let conn = crate::db::open_in_memory().unwrap();
        let sql_says_expired = |meta: &serde_json::Value, now: i64| -> bool {
            conn.query_row(
                "SELECT COALESCE(json_extract(?1, '$.expires_at'), 1e18) <= ?2",
                rusqlite::params![meta.to_string(), now],
                |r| r.get(0),
            )
            .unwrap()
        };
        let t = 1_700_000_000i64;
        for meta in [
            serde_json::json!({ "expires_at": t }),
            serde_json::json!({ "expires_at": t as f64 }),
            serde_json::json!({ "expires_at": t as f64 + 0.5 }),
            serde_json::json!({ "expires_at": 1.7e9 }),
            serde_json::json!({ "expires_at": "1700000000" }),
            serde_json::json!({ "expires_at": null }),
            serde_json::json!({}),
        ] {
            let node = with_meta(meta.clone());
            for now in [t - 1, t, t + 1] {
                assert_eq!(
                    node.is_expired(now),
                    sql_says_expired(&meta, now),
                    "{meta} at {now}"
                );
            }
        }
        assert_eq!(
            with_meta(serde_json::json!({ "expires_at": t as f64 })).expires_at(),
            Some(t)
        );
        assert_eq!(
            with_meta(serde_json::json!({ "expires_at": t as f64 + 0.5 })).expires_at(),
            Some(t + 1),
            "the first whole second at which it has expired"
        );
    }

    #[test]
    fn validity_names_each_state_of_the_declared_interval() {
        let (past, now, future) = (1_000, 2_000, 3_000);
        let at = |from: Option<i64>, to: Option<i64>| {
            let mut meta = serde_json::json!({});
            if let Some(from) = from {
                meta["valid_from"] = serde_json::json!(from);
            }
            if let Some(to) = to {
                meta["valid_to"] = serde_json::json!(to);
            }
            with_meta(meta)
        };

        assert_eq!(at(None, None).validity(now), Validity::Undeclared);
        assert_eq!(at(Some(past), None).validity(now), Validity::Open);
        assert_eq!(at(None, Some(future)).validity(now), Validity::Open);
        assert_eq!(at(Some(past), Some(future)).validity(now), Validity::Open);
        assert_eq!(at(Some(future), None).validity(now), Validity::NotYet);
        assert_eq!(
            at(Some(past), Some(past + 1)).validity(now),
            Validity::Closed
        );
        assert_eq!(
            at(None, Some(now)).validity(now),
            Validity::Closed,
            "end-exclusive"
        );
        assert_eq!(
            at(Some(now), None).validity(now),
            Validity::Open,
            "start-inclusive"
        );

        // The older yes/no readings are this same derivation.
        for node in [
            at(None, None),
            at(Some(past), None),
            at(Some(future), None),
            at(Some(past), Some(past + 1)),
            at(Some(past), Some(future)),
        ] {
            let validity = node.validity(now);
            assert_eq!(node.validity_closed(now), validity == Validity::Closed);
            assert_eq!(
                node.valid_at(now),
                match validity {
                    Validity::Undeclared => None,
                    Validity::Open => Some(true),
                    Validity::NotYet | Validity::Closed => Some(false),
                }
            );
        }
    }

    #[test]
    fn short_uid_uses_tail() {
        let uid = "01JXGABCDEF123456789QRSTUV";
        assert_eq!(short_uid(Kind::Memory, uid), "m:QRSTUV");
        assert_eq!(short_uid(Kind::Project, uid), "p:QRSTUV");
    }
}
