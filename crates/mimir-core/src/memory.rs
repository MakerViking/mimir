//! Memory capture: typed thoughts with tag support and duplicate detection.

use std::collections::HashSet;

use rusqlite::{Connection, OptionalExtension};

use crate::error::Result;
use crate::model::{Kind, MemoryType, NewNode, Node, Scope};
use crate::search::{self, SearchQuery};
use crate::store::{self, row_to_node, NODE_COLS};

/// Token-set similarity at or above which a new memory is refused
/// as a near-duplicate (without --force).
const NEAR_DUP_JACCARD: f64 = 0.85;

/// Most tokens a subset match may differ by and still count as the same fact
/// reworded — see [`is_reword`].
const REWORD_SUBSET_SLACK: usize = 2;

/// How many deliberate tombstones the reword pass compares against. A bound
/// so `remember` can never turn into a table scan on a store where someone
/// has forgotten a great deal; newest-first, because a recent deletion is
/// the one a re-add is most likely to be undoing.
const FORGOTTEN_SCAN_LIMIT: usize = 500;

#[derive(Debug)]
pub enum RememberOutcome {
    Created(Node),
    /// Refused: the contained node is the existing near-duplicate.
    Duplicate(Node),
    /// Refused: this text was deliberately forgotten and the tombstone is
    /// still there. Re-adding it silently is how a deleted fact creeps
    /// back in — an extractor re-reading the same source produces the same
    /// text, and without this the only trace of the deletion is a node the
    /// dedup path can't see. Takes `--force`, which is the point: coming
    /// back has to be a decision someone made, not a default.
    Forgotten(Node),
}

#[derive(Debug)]
pub struct Remember {
    pub text: String,
    pub mtype: MemoryType,
    pub tags: Vec<String>,
    pub project_id: Option<i64>,
    pub force: bool,
    pub actor: crate::model::Actor,
}

/// Store a memory unless an exact or near duplicate already exists.
pub fn remember(conn: &Connection, args: Remember) -> Result<RememberOutcome> {
    let hash = content_hash(&args.text);
    let mut revived = false;
    if !args.force {
        // Precedence matters. An exact *live* copy is an ordinary duplicate
        // even when an older tombstone of the same text also exists — that
        // is the shape of a fact someone deliberately brought back, and
        // refusing it as "forgotten" forever would make `--force` a
        // one-way door. Only when nothing live matches does the tombstone
        // get to refuse. Near-duplicates rank last: a loose match against
        // some other live memory must not mask an exact tombstone hit.
        if let Some(dup) = find_exact(conn, &hash)? {
            return Ok(RememberOutcome::Duplicate(dup));
        }
        if let Some(gone) = find_forgotten(conn, &args.text, &hash)? {
            return Ok(RememberOutcome::Forgotten(gone));
        }
        if let Some(dup) = find_near_duplicate(conn, &args.text)? {
            return Ok(RememberOutcome::Duplicate(dup));
        }
    } else {
        // Forcing past a tombstone is the one creation worth recording.
        // Ordinary capture needs no ledger row — `created_at` says when and
        // the text is right there — but bringing back something a person
        // deliberately deleted is a decision, and the deletion it reverses
        // is already in the ledger above it.
        revived = find_forgotten(conn, &args.text, &hash)?.is_some();
    }
    let mut new = NewNode::new(Kind::Memory);
    new.subkind = Some(args.mtype.to_string());
    new.title = Some(derive_title(&args.text));
    new.body = Some(args.text);
    new.tags = sanitize_tags(args.tags);
    new.project_id = args.project_id;
    new.content_hash = Some(hash.clone());
    let node = store::insert_node(conn, new)?;
    if revived {
        crate::audit::record_for_kind(
            conn,
            Kind::Memory,
            node.id,
            crate::model::MutationOp::Restore,
            args.actor,
            Some("--force over a deliberate tombstone"),
            None,
            Some(&hash),
        );
    }
    Ok(RememberOutcome::Created(node))
}

/// Tags surface as nodes in the graph visualization (HTML). Strip the
/// HTML-significant characters at ingest so a tag can never carry markup —
/// the viz also escapes at render time, this is defense in depth. Tags that
/// reduce to nothing are dropped.
pub fn sanitize_tags(tags: Vec<String>) -> Vec<String> {
    tags.into_iter()
        .map(|t| {
            t.chars()
                .filter(|c| !matches!(c, '<' | '>' | '&' | '"' | '\''))
                .collect::<String>()
                .trim()
                .to_string()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Normalize author-declared trigger phrases for `meta.fires_when` (see
/// `inject::clears_floor`'s fires_when path): whitespace-collapsed,
/// lowercased, capped to 8 phrases of at most 6 words each. An oversized
/// phrase is dropped rather than truncated — same "reject, don't mangle"
/// call as `sanitize_tags` makes for markup, since a truncated trigger
/// could silently mean something the author never wrote.
pub fn sanitize_fires_when(phrases: Vec<String>) -> Vec<String> {
    phrases
        .into_iter()
        .map(|p| {
            p.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        })
        .filter(|p| !p.is_empty() && p.split_whitespace().count() <= 6)
        .take(8)
        .collect()
}

/// Parse a relative expiry like `"90d"` into an absolute `expires_at` unix
/// timestamp, `now + duration`.
///
/// Relative-only, and units stop at weeks, on purpose. "Months" and "years"
/// have no fixed length, so accepting them would mean either a lie (30d) or
/// calendar math and a date dependency — for a field whose whole job is to be
/// a coarse "stops being true around then", neither is worth it. A year is
/// `365d`.
///
/// Returns `None` for anything unparseable, zero, or negative: callers reject
/// loudly rather than silently storing an expiry the author didn't mean. A
/// memory that quietly expired early is indistinguishable from one that was
/// never written.
pub fn parse_expires_in(spec: &str, now: i64) -> Option<i64> {
    let s = spec.trim().to_lowercase();
    let (digits, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit())?);
    let n: i64 = digits.parse().ok()?;
    if n <= 0 {
        return None;
    }
    let secs = match unit {
        "h" | "hr" | "hrs" | "hour" | "hours" => 3_600,
        "d" | "day" | "days" => 86_400,
        "w" | "wk" | "wks" | "week" | "weeks" => 604_800,
        _ => return None,
    };
    n.checked_mul(secs)?.checked_add(now)
}

/// SQL for tombstones that represent a *deliberate* deletion.
///
/// `consolidate`'s decay pass also soft-deletes (with `meta.archived = 1`),
/// but nobody decided that — it fell below a strength threshold while idle.
/// Refusing to re-learn a fact because a background job archived it six
/// months ago would be wrong, and would teach people to pass `--force` by
/// reflex, which is exactly how the guard stops meaning anything.
const DELIBERATE_TOMBSTONE: &str = "kind = 'memory' AND deleted_at IS NOT NULL \
     AND COALESCE(json_extract(meta, '$.archived'), 0) != 1";

/// A memory someone deliberately forgot, whose text is being offered again.
///
/// Two passes, because re-extraction takes two shapes. The exact
/// `content_hash` catches a source re-read unchanged. The [`is_reword`] pass
/// catches a reword — a different extractor, or the same fact said again in
/// other words — which the hash cannot, since `tokens` keeps `file.rs` and
/// trailing punctuation whole and so treats "utc" and "utc." as different.
///
/// The reword test is deliberately more sensitive here than the live
/// near-duplicate pass below, because the costs are not symmetric: a false
/// duplicate refusal on a live memory is an annoyance with `--force` one
/// keystroke away, while a missed tombstone silently undoes a deletion
/// someone meant.
///
/// The scan is over deliberate tombstones only, which stay rare in a real
/// store (people forget on purpose seldom, and decay archival is excluded),
/// so this is a few dozen short strings, not a table sweep.
fn find_forgotten(conn: &Connection, text: &str, hash: &[u8]) -> Result<Option<Node>> {
    let exact = conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE {DELIBERATE_TOMBSTONE} AND content_hash = ?1
                 ORDER BY deleted_at DESC LIMIT 1"
            ),
            [hash],
            row_to_node,
        )
        .optional()?;
    if exact.is_some() {
        return Ok(exact);
    }

    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node
         WHERE {DELIBERATE_TOMBSTONE}
         ORDER BY deleted_at DESC LIMIT {FORGOTTEN_SCAN_LIMIT}"
    ))?;
    let rows = stmt.query_map([], row_to_node)?;
    for row in rows {
        let node = row?;
        let existing = node
            .body
            .as_deref()
            .filter(|b| !b.is_empty())
            .or(node.title.as_deref())
            .unwrap_or_default();
        if is_reword(text, existing) {
            return Ok(Some(node));
        }
    }
    Ok(None)
}

/// Exact (normalized) content match against a *live* memory, any scope.
fn find_exact(conn: &Connection, hash: &[u8]) -> Result<Option<Node>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {NODE_COLS} FROM node
                 WHERE kind = 'memory' AND content_hash = ?1 AND deleted_at IS NULL
                 LIMIT 1"
            ),
            [hash],
            row_to_node,
        )
        .optional()?)
}

/// Near-duplicate: high token overlap with the best FTS matches.
fn find_near_duplicate(conn: &Connection, text: &str) -> Result<Option<Node>> {
    let query = SearchQuery {
        text: text.to_string(),
        scope: Scope::All,
        kinds: vec![Kind::Memory],
        since: None,
        limit: 5,
        strength_alpha: 0.0,
        recency_alpha: 0.0,
        type_prior_alpha: 0.0,
        code_damp: 1.0,
        include_superseded: false,
        as_of: None,
    };
    for hit in search::search(conn, &query)? {
        // Title-only memories (some imports) have no body — compare against
        // the title so they still participate in duplicate detection.
        let existing = hit
            .node
            .body
            .as_deref()
            .filter(|b| !b.is_empty())
            .or(hit.node.title.as_deref())
            .unwrap_or_default();
        if jaccard(text, existing) >= NEAR_DUP_JACCARD {
            return Ok(Some(hit.node));
        }
    }
    Ok(None)
}

/// blake3 over lowercased, whitespace/punctuation-normalized text, so
/// formatting tweaks don't defeat exact-duplicate detection.
pub fn content_hash(text: &str) -> Vec<u8> {
    let normalized = tokens(text).join(" ");
    blake3::hash(normalized.as_bytes()).as_bytes().to_vec()
}

/// Identifier-friendly tokens: keeps `snake_case`, `kebab-case`, `file.rs` whole.
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = tokens(a).into_iter().collect();
    let sb: HashSet<String> = tokens(b).into_iter().collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    inter / union
}

/// `tokens` with leading/trailing `.`/`-`/`_` stripped, for *comparison only*.
///
/// `tokens` keeps those characters whole so `file.rs`, `snake_case` and
/// `kebab-case` survive as single tokens — which is right for the content
/// hash, and wrong when deciding whether two texts say the same thing: it
/// makes "utc" and "utc." different words. Interior punctuation is untouched,
/// so `file.rs` still differs from `file`.
fn comparison_tokens(text: &str) -> HashSet<String> {
    tokens(text)
        .into_iter()
        .map(|t| t.trim_matches(['.', '-', '_']).to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Whether `text` is `existing` said again — the question the tombstone pass
/// asks before letting a deliberately forgotten fact back in.
///
/// Three rules, because a single similarity ratio is a function of how much
/// someone wrote. At one differing token, `jaccard >= 0.85` needs 13 unique
/// tokens to clear, so the same trailing period that was caught on a long
/// memory walked straight through a terse one — and terse is the common shape
/// of a hand-written gotcha.
///
/// 1. Equal under [`comparison_tokens`] — a pure formatting reword (case,
///    whitespace, trailing punctuation). Length-independent by construction.
/// 2. One token set contains the other, differing by at most
///    [`REWORD_SUBSET_SLACK`] — the same fact with a qualifier added or
///    dropped ("redis port 6379" → "redis default port 6379").
/// 3. The ratio, which still earns its keep on diffuse rewording of long text.
///
/// Rule 2 is a *subset* test rather than a token-distance one on purpose. An
/// unconditional "differs by ≤2 tokens" would also swallow substitutions —
/// "redis port 6379 tcp" against "redis port 5432 tcp" is two tokens apart and
/// a completely different fact. Requiring containment keeps insertions and
/// deletions in scope while leaving substitutions out.
fn is_reword(text: &str, existing: &str) -> bool {
    let a = comparison_tokens(text);
    let b = comparison_tokens(existing);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let (small, large) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    if small.is_subset(large) && large.len() - small.len() <= REWORD_SUBSET_SLACK {
        return true;
    }
    jaccard(text, existing) >= NEAR_DUP_JACCARD
}

/// First line of the text, truncated to 80 chars on a char boundary.
pub fn derive_title(text: &str) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    let mut title: String = first.chars().take(80).collect();
    if first.chars().count() > 80 {
        title.push('…');
    }
    title
}

#[derive(Debug, Default)]
pub struct Edit {
    pub text: Option<String>,
    pub title: Option<String>,
    pub mtype: Option<MemoryType>,
    pub tags: Option<Vec<String>>,
}

/// Apply field edits to an existing node; body edits refresh the derived
/// title (unless one is given) and the content hash.
///
/// `actor`/`reason` go to the mutation ledger. An edit is the one mutation
/// that changes what a memory *says* while leaving it live, so the pair of
/// hashes it records is the only evidence that the text ever read differently.
pub fn edit(
    conn: &Connection,
    id: i64,
    edit: Edit,
    actor: crate::model::Actor,
    reason: Option<&str>,
) -> Result<Node> {
    let node = store::get_node(conn, id)?;
    let before = node.content_hash.clone();
    let kind = node.kind;
    // Keep what it said before the update lands — `recall --as-of` reads
    // this, and after the UPDATE there is nothing left to read.
    crate::revision::snapshot(conn, &node);
    let body = edit.text.or(node.body);
    let title = match edit.title {
        Some(t) => Some(t),
        None => body.as_deref().map(derive_title),
    };
    let subkind = edit.mtype.map(|t| t.to_string()).or(node.subkind);
    let tags_text = edit
        .tags
        .map(|t| sanitize_tags(t).join(" "))
        .unwrap_or(node.tags_text);
    let hash = body.as_deref().map(content_hash);
    conn.execute(
        "UPDATE node SET title = ?2, body = ?3, subkind = ?4, tags_text = ?5,
                         content_hash = ?6, updated_at = ?7
         WHERE id = ?1",
        rusqlite::params![
            id,
            title,
            body,
            subkind,
            tags_text,
            hash,
            crate::model::now_unix()
        ],
    )?;
    crate::audit::record_for_kind(
        conn,
        kind,
        id,
        crate::model::MutationOp::Edit,
        actor,
        reason,
        before.as_deref(),
        hash.as_deref(),
    );
    store::get_node(conn, id)
}

/// Recent memories, newest first, optionally filtered by type and tag.
pub fn list(
    conn: &Connection,
    scope: Scope,
    mtype: Option<MemoryType>,
    tag: Option<&str>,
    limit: usize,
) -> Result<Vec<Node>> {
    let mut sql = format!(
        "SELECT {NODE_COLS} FROM node n
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL AND {}",
        search::scope_sql(scope)
    );
    if let Some(t) = mtype {
        sql.push_str(&format!(" AND n.subkind = '{t}'"));
    }
    if tag.is_some() {
        sql.push_str(" AND (' ' || n.tags_text || ' ') LIKE ?2");
    }
    sql.push_str(" ORDER BY n.created_at DESC LIMIT ?1");
    let mut stmt = conn.prepare(&sql)?;
    let rows = match tag {
        Some(t) => stmt
            .query_map(
                rusqlite::params![limit as i64, format!("% {t} %")],
                row_to_node,
            )?
            .collect::<rusqlite::Result<_>>()?,
        None => stmt
            .query_map([limit as i64], row_to_node)?
            .collect::<rusqlite::Result<_>>()?,
    };
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[test]
    fn sanitize_tags_strips_markup_and_drops_empties() {
        let got = sanitize_tags(vec![
            "<img/src=x/onerror=alert(1)>".into(),
            "auth".into(),
            "<>&\"'".into(),
            "  postgres  ".into(),
        ]);
        // markup chars gone, empties dropped, whitespace trimmed
        assert_eq!(got, vec!["img/src=x/onerror=alert(1)", "auth", "postgres"]);
        assert!(got.iter().all(|t| !t.contains('<') && !t.contains('>')));
    }

    #[test]
    fn sanitize_fires_when_normalizes_caps_and_drops_oversized() {
        let got = sanitize_fires_when(vec![
            "  Release   Checklist  ".into(),
            "CUTTING a release".into(),
            "".into(),
            "   ".into(),
            "this phrase has way more than six words in it".into(),
        ]);
        assert_eq!(got, vec!["release checklist", "cutting a release"]);
    }

    #[test]
    fn sanitize_fires_when_caps_at_eight_phrases() {
        let phrases: Vec<String> = (0..12).map(|i| format!("phrase {i}")).collect();
        assert_eq!(sanitize_fires_when(phrases).len(), 8);
    }

    fn remember_text(conn: &Connection, text: &str) -> RememberOutcome {
        remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
    }

    #[test]
    fn remember_then_exact_duplicate_refused() {
        let conn = db::open_in_memory().unwrap();
        let first = match remember_text(&conn, "SCRAM auth rejects non-ASCII passwords") {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        // Same content, different whitespace/case → still a duplicate.
        match remember_text(&conn, "scram auth  rejects\nnon-ascii passwords") {
            RememberOutcome::Duplicate(d) => assert_eq!(d.id, first.id),
            other => panic!("expected Duplicate, got {other:?}"),
        }
    }

    /// The failure this closes: `forget` tombstones a memory, recall stops
    /// returning it, and then the same text is offered again — by an
    /// importer, an extractor re-reading an unchanged source, or an agent
    /// that saw the fact a second time. Dedup only ever looked at live
    /// nodes, so the deletion left nothing that could catch the re-add and
    /// the fact came back silently under a fresh id.
    #[test]
    fn forgotten_memory_is_not_silently_re_added() {
        let conn = db::open_in_memory().unwrap();
        let text = "staging db password rotation runs from the ops cron at 03:00 UTC";
        let first = match remember_text(&conn, text) {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        store::soft_delete(&conn, first.id, crate::model::Actor::System, None).unwrap();

        // Same fact, reformatted the way a different extractor would emit it.
        match remember_text(
            &conn,
            "Staging DB password rotation runs from the OPS cron at 03:00 utc",
        ) {
            RememberOutcome::Forgotten(gone) => {
                assert_eq!(gone.id, first.id);
                assert!(gone.deleted_at.is_some(), "must carry the tombstone date");
            }
            other => panic!("expected Forgotten, got {other:?}"),
        }

        // Refusing is not the same as making it impossible: a human who
        // means it can still bring it back, and gets a new live node.
        let forced = remember(
            &conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        let RememberOutcome::Created(revived) = forced else {
            panic!("--force must override the tombstone");
        };
        assert_ne!(revived.id, first.id);
        assert!(revived.deleted_at.is_none());
    }

    /// The hash pass alone would miss this: `tokens` keeps trailing
    /// punctuation attached (so `file.rs` survives), which makes "utc" and
    /// "utc." different tokens and the two texts different hashes. A
    /// re-extraction that rewords even slightly is the common case, so the
    /// guard has to survive it.
    ///
    /// Deliberately fixtured *short* (9 unique tokens). An earlier version of
    /// this test used a 14-token sentence, which passed under a ratio-only
    /// rule for the wrong reason: at one differing token, `jaccard >= 0.85`
    /// needs 13 unique tokens to clear. The terse one-line gotcha is the
    /// common shape of a hand-written memory, so it is the case that has to
    /// hold.
    #[test]
    fn forgotten_memory_is_not_re_added_under_a_reword() {
        let conn = db::open_in_memory().unwrap();
        let first =
            match remember_text(&conn, "SCRAM auth rejects non-ASCII passwords at 03:00 UTC") {
                RememberOutcome::Created(n) => n,
                other => panic!("expected Created, got {other:?}"),
            };
        store::soft_delete(&conn, first.id, crate::model::Actor::System, None).unwrap();
        match remember_text(
            &conn,
            "scram auth rejects non-ascii passwords at 03:00 utc.",
        ) {
            RememberOutcome::Forgotten(gone) => assert_eq!(gone.id, first.id),
            other => panic!("expected Forgotten, got {other:?}"),
        }
    }

    /// The guard must not be a function of how much someone wrote. A fixed
    /// similarity *ratio* is: at one differing token it needs 13 unique
    /// tokens to fire, so the same trailing period that was caught on a long
    /// memory walked straight through a short one. Both lengths, one
    /// perturbation, same verdict.
    #[test]
    fn reword_guard_does_not_depend_on_memory_length() {
        for n in [5usize, 9, 12, 13, 20] {
            let conn = db::open_in_memory().unwrap();
            let base: String = (1..n).fold(String::new(), |mut s, i| {
                s.push_str(&format!("w{i} "));
                s
            }) + "tail";
            let first = match remember_text(&conn, &base) {
                RememberOutcome::Created(node) => node,
                other => panic!("[n={n}] expected Created, got {other:?}"),
            };
            store::soft_delete(&conn, first.id, crate::model::Actor::System, None).unwrap();
            match remember_text(&conn, &format!("{base}.")) {
                RememberOutcome::Forgotten(gone) => assert_eq!(gone.id, first.id),
                other => panic!(
                    "[n={n} unique tokens] a forgotten memory walked back in \
                     under a one-token reword: {other:?}"
                ),
            }
        }
    }

    /// The absolute-distance rule must not swallow genuinely different short
    /// facts. Two tokens apart out of four is a different fact, not a reword,
    /// and refusing it would train `--force` by reflex just as surely as
    /// treating decay archival as a deletion would.
    #[test]
    fn short_distinct_facts_are_not_confused_for_a_reword() {
        let conn = db::open_in_memory().unwrap();
        let first = match remember_text(&conn, "redis port 6379") {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        store::soft_delete(&conn, first.id, crate::model::Actor::System, None).unwrap();
        assert!(
            matches!(
                remember_text(&conn, "postgres port 5432"),
                RememberOutcome::Created(_)
            ),
            "a different fact of the same shape must still be capturable"
        );
    }

    /// Decay archival is not a decision anyone made. Treating it as one
    /// would refuse re-learning facts a background job retired while idle,
    /// and train everyone to pass `--force` by reflex — at which point the
    /// guard on real deletions stops meaning anything.
    #[test]
    fn auto_archived_memory_does_not_block_re_adding() {
        let conn = db::open_in_memory().unwrap();
        let text = "cargo nextest needs a separate install on the CI image";
        let first = match remember_text(&conn, text) {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        // Exactly what consolidate's decay pass does.
        conn.execute(
            "UPDATE node SET deleted_at = 1, meta = json_set(meta, '$.archived', 1) WHERE id = ?1",
            [first.id],
        )
        .unwrap();
        assert!(
            matches!(remember_text(&conn, text), RememberOutcome::Created(_)),
            "decay archival must not masquerade as a deliberate forget"
        );
    }

    /// A tombstone must not outrank a live memory: once the fact is back,
    /// remembering it again is an ordinary duplicate, not a resurrection.
    #[test]
    fn live_duplicate_wins_over_an_older_tombstone() {
        let conn = db::open_in_memory().unwrap();
        let text = "the proxy strips cache breakpoints above four";
        let first = match remember_text(&conn, text) {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        store::soft_delete(&conn, first.id, crate::model::Actor::System, None).unwrap();
        let revived = remember(
            &conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        let RememberOutcome::Created(revived) = revived else {
            panic!("expected Created");
        };
        match remember_text(&conn, text) {
            RememberOutcome::Duplicate(d) => assert_eq!(d.id, revived.id),
            other => panic!("expected Duplicate, got {other:?}"),
        }
    }

    #[test]
    fn near_duplicate_refused_force_overrides() {
        let conn = db::open_in_memory().unwrap();
        remember_text(
            &conn,
            "the indexer walks gitignore files using the ignore crate",
        );
        match remember_text(
            &conn,
            "the indexer walks gitignore files using the ignore crate now",
        ) {
            RememberOutcome::Duplicate(_) => {}
            other => panic!("expected Duplicate, got {other:?}"),
        }
        let forced = remember(
            &conn,
            Remember {
                text: "the indexer walks gitignore files using the ignore crate now".into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        assert!(matches!(forced, RememberOutcome::Created(_)));
    }

    #[test]
    fn distinct_memories_both_stored_and_listed() {
        let conn = db::open_in_memory().unwrap();
        remember_text(&conn, "use WAL mode for concurrent readers");
        remember_text(&conn, "prefer RRF over score normalization for fusion");
        let all = list(&conn, Scope::All, None, None, 10).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn edit_updates_body_title_and_hash() {
        let conn = db::open_in_memory().unwrap();
        let node = match remember_text(&conn, "original text here") {
            RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        };
        let old_hash = node.content_hash.clone();
        let updated = edit(
            &conn,
            node.id,
            Edit {
                text: Some("completely new body".into()),
                ..Default::default()
            },
            crate::model::Actor::Human,
            None,
        )
        .unwrap();
        assert_eq!(updated.title.as_deref(), Some("completely new body"));
        assert_ne!(updated.content_hash, old_hash);
    }

    #[test]
    fn list_filters_by_type_and_tag() {
        let conn = db::open_in_memory().unwrap();
        remember(
            &conn,
            Remember {
                text: "tagged gotcha".into(),
                mtype: MemoryType::Gotcha,
                tags: vec!["sqlite".into()],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap();
        remember_text(&conn, "plain note");
        let gotchas = list(&conn, Scope::All, Some(MemoryType::Gotcha), None, 10).unwrap();
        assert_eq!(gotchas.len(), 1);
        let tagged = list(&conn, Scope::All, None, Some("sqlite"), 10).unwrap();
        assert_eq!(tagged.len(), 1);
        let missing = list(&conn, Scope::All, None, Some("nope"), 10).unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn derive_title_truncates() {
        assert_eq!(derive_title("short one\nsecond line"), "short one");
        let long = "x".repeat(100);
        let title = derive_title(&long);
        assert_eq!(title.chars().count(), 81); // 80 + ellipsis
    }

    #[test]
    fn parse_expires_in_accepts_hours_days_weeks() {
        let now = 1_000_000;
        assert_eq!(parse_expires_in("12h", now), Some(now + 43_200));
        assert_eq!(parse_expires_in("90d", now), Some(now + 7_776_000));
        assert_eq!(parse_expires_in("2w", now), Some(now + 1_209_600));
        // Forgiving about spelling and padding, since this is hand-typed.
        assert_eq!(parse_expires_in("  3 DAYS ", now), None); // inner space is not
        assert_eq!(parse_expires_in("3days", now), Some(now + 259_200));
        assert_eq!(parse_expires_in(" 1W ", now), Some(now + 604_800));
    }

    #[test]
    fn parse_expires_in_rejects_rather_than_guesses() {
        let now = 1_000_000;
        // No unit: "30" could be seconds, days, anything. Refuse.
        assert_eq!(parse_expires_in("30", now), None);
        // Ambiguous-length units are deliberately unsupported.
        assert_eq!(parse_expires_in("6m", now), None);
        assert_eq!(parse_expires_in("1y", now), None);
        // Zero/negative would store an already-passed expiry, hiding the
        // memory the instant it was written.
        assert_eq!(parse_expires_in("0d", now), None);
        assert_eq!(parse_expires_in("-5d", now), None);
        assert_eq!(parse_expires_in("", now), None);
        assert_eq!(parse_expires_in("soon", now), None);
        // Overflow must not wrap into the past.
        assert_eq!(parse_expires_in("999999999999999999999d", now), None);
    }
}
