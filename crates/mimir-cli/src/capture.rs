//! One capture pipeline behind `mimir remember` and the MCP `remember` tool.
//!
//! A capture has three endings and no others: **stored** exactly as asked,
//! **refused** (it duplicates a live memory, or one somebody forgot on
//! purpose), or **rejected** with nothing written. Each surface used to walk
//! the same nine steps itself, in its own order, and every difference between
//! them was an accident: one validated a duration after the insert, one
//! resolved a link target after it, each scanned a different set of fields for
//! secrets. The order is fixed here, once:
//!
//! 1. validate every argument — nothing is written yet
//! 2. scan every free-text field for secrets — a refusal is recorded, nothing
//!    is stored
//! 3. resolve the link target — see [`MissingLink`]
//! 4. `memory::remember` — the duplicate and forgotten refusals
//! 5. attach what was asked for: link, triggers, anchors, expiry, confidence,
//!    validity
//! 6. embed (best effort)
//!
//! Steps 4 and 5 are one transaction, so a failure part-way through cannot
//! leave a memory stored without the deadline or the certainty its author
//! asked to record — an omission nothing afterwards would reveal, and one the
//! corrected retry could not repair because it would be refused as a
//! duplicate.
//!
//! What legitimately differs between the two surfaces is a field of
//! [`Capture`], never a branch in a handler: who is acting, whether `--force`
//! exists, and what a missing link target does.

use anyhow::anyhow;
use mimir_core::error::Error;
use mimir_core::memory::{self, Remember, RememberOutcome};
use mimir_core::model::{
    now_unix, short_uid, Actor, Kind, MemoryConfidence, MemoryType, MutationOp, Node, Rel,
};
use mimir_core::secrets::{self, Surface};
use mimir_core::{anchors, context_guard, format, store, Mimir};

/// Everything a caller can ask of a capture.
pub struct Capture {
    pub text: String,
    pub mtype: MemoryType,
    pub tags: Vec<String>,
    /// The project the memory belongs to; `None` stores it globally.
    pub project_id: Option<i64>,
    /// The project the caller is working in, whose code graph resolves a
    /// symbol *name* given as the link target. Deliberately separate from
    /// `project_id`: a global memory about `retry_with_backoff` is still
    /// about the `retry_with_backoff` in the checkout it was written from.
    pub working_project: Option<i64>,
    /// Store past a duplicate or a deliberate tombstone. Only a person gets
    /// this; an agent that was refused is told who can override it.
    pub force: bool,
    pub actor: Actor,
    /// Where the request came from, for the refusal ledger and for wording
    /// an argument error in that surface's vocabulary.
    pub surface: Surface,
    pub link: Option<String>,
    pub missing_link: MissingLink,
    pub fires_when: Vec<String>,
    pub anchors: Vec<String>,
    pub expires_in: Option<String>,
    pub resolves_when: Option<String>,
    pub confidence: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
}

/// What a link target that does not exist does to the capture.
///
/// The two surfaces differ on purpose. A person at a terminal re-runs the
/// command with the name fixed, so the capture is rejected whole. An agent
/// pays for a retry by re-sending the entire memory, and an unlinked memory
/// is still worth having, so it is stored and the reply says the link was
/// not made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingLink {
    Reject,
    StoreUnlinked,
}

/// How a capture ended when nothing was rejected.
#[derive(Debug)]
pub enum Captured {
    Stored(Box<Stored>),
    /// An exact or near duplicate of this live memory already exists.
    /// `rearmed_handoff`: the duplicate is a session handoff that had already
    /// been restored once, and offering it again made it restorable again
    /// (see [`REARMED_HANDOFF`]).
    Duplicate {
        node: Node,
        rearmed_handoff: bool,
    },
    /// This text was deliberately forgotten; `by` is who did it and `on`
    /// is when, as far as the ledger still knows.
    Forgotten {
        node: Node,
        by: Option<Actor>,
        on: i64,
    },
}

/// A stored memory and everything that was attached to it.
#[derive(Debug)]
pub struct Stored {
    /// Read back after every attachment, so `meta` is what the store holds.
    pub node: Node,
    pub link: Linked,
    /// The duration as the caller wrote it, and the instant it resolved to.
    pub expires: Option<(String, i64)>,
    pub resolves_when: Option<String>,
    pub confidence: Option<MemoryConfidence>,
    pub valid: Option<(Option<i64>, Option<i64>)>,
}

/// What became of the requested link.
#[derive(Debug)]
pub enum Linked {
    NotRequested,
    To {
        target: Box<Node>,
        rel: Rel,
    },
    /// The target did not resolve and [`MissingLink::StoreUnlinked`] applied.
    Missing(String),
}

/// An argument a capture can be rejected for. One list, so the CLI flag and
/// the MCP field for the same thing share one rule and one description of
/// what was expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    ExpiresIn,
    Confidence,
    ValidFrom,
    ValidTo,
}

impl Arg {
    fn name(self, surface: Surface) -> &'static str {
        let flag = !matches!(surface, Surface::McpRemember);
        match (self, flag) {
            (Arg::ExpiresIn, true) => "--expires-in",
            (Arg::ExpiresIn, false) => "expires_in",
            (Arg::Confidence, true) => "--confidence",
            (Arg::Confidence, false) => "confidence",
            (Arg::ValidFrom, true) => "--valid-from",
            (Arg::ValidFrom, false) => "valid_from",
            (Arg::ValidTo, true) => "--valid-to",
            (Arg::ValidTo, false) => "valid_to",
        }
    }

    fn expected(self) -> &'static str {
        match self {
            Arg::ExpiresIn => {
                "a positive duration like 90d, 2w or 12h (units: h/d/w — use 365d for a year)"
            }
            Arg::Confidence => "certain, likely or unsure",
            Arg::ValidFrom | Arg::ValidTo => "a date like 2024-06-30",
        }
    }
}

/// Why a capture wrote nothing.
#[derive(Debug)]
pub enum Rejected {
    /// An argument that cannot be honoured as given.
    Invalid { arg: Arg, value: String },
    /// A validity interval that ends on or before the day it starts.
    ValidityOrder,
    /// The link target does not exist and the surface rejects on that.
    Link(anyhow::Error),
    /// The store refused (a secret) or failed (a database error).
    Store(Error),
}

impl Rejected {
    fn invalid(arg: Arg, value: &str) -> Self {
        Rejected::Invalid {
            arg,
            value: value.to_string(),
        }
    }

    /// The rejection in the vocabulary of `surface`: flags for the CLI,
    /// field names for the MCP tool.
    pub fn message(&self, surface: Surface) -> String {
        match self {
            Rejected::Invalid { arg, value } => format!(
                "invalid {} {value:?}: expected {}",
                arg.name(surface),
                arg.expected()
            ),
            Rejected::ValidityOrder => format!(
                "{} must be after {}",
                Arg::ValidTo.name(surface),
                Arg::ValidFrom.name(surface)
            ),
            Rejected::Link(err) => err.to_string(),
            Rejected::Store(err) => err.to_string(),
        }
    }

    /// For the CLI: store errors keep their type (so `main` reports them as
    /// it reports any other), argument errors carry the flag names.
    pub fn into_anyhow(self) -> anyhow::Error {
        match self {
            Rejected::Store(err) => err.into(),
            Rejected::Link(err) => err,
            other => anyhow!(other.message(Surface::CliRemember)),
        }
    }
}

impl From<Error> for Rejected {
    fn from(err: Error) -> Self {
        Rejected::Store(err)
    }
}

impl From<rusqlite::Error> for Rejected {
    fn from(err: rusqlite::Error) -> Self {
        Rejected::Store(err.into())
    }
}

/// The relation a capture-time link gets. A memory pointing at something
/// Mimir indexes from disk is *about* it (that is the link grounding reads);
/// a memory pointing at another memory merely relates to it.
fn link_rel(target: &Node) -> Rel {
    if target.kind == Kind::Memory {
        Rel::Relates
    } else {
        Rel::About
    }
}

fn parse_day(arg: Arg, value: Option<&str>) -> Result<Option<i64>, Rejected> {
    value
        .map(|s| format::parse_date(s).ok_or_else(|| Rejected::invalid(arg, s)))
        .transpose()
}

/// Who recorded the deliberate forget of `gone`, and when. Read from the
/// ledger because the row cannot say: a refusal that names the wrong party
/// ("a human forgot this", about a tombstone an agent or a sync peer wrote)
/// sends the reader to the wrong place to ask why, and `deleted_at` is the
/// day the memory decayed when it was archived first and forgotten later.
/// Without a ledger row the row's own tombstone date is the best there is.
fn forgotten_by(conn: &rusqlite::Connection, gone: &Node) -> (Option<Actor>, i64) {
    let recorded = mimir_core::audit::history(conn, gone.id)
        .ok()
        .and_then(|rows| rows.into_iter().rev().find(|m| m.op == MutationOp::Forget));
    match recorded {
        Some(row) => (Some(row.actor), row.at),
        None => (None, gone.deleted_at.unwrap_or(gone.updated_at)),
    }
}

/// The secret guard, for every surface that stores free text a person or an
/// agent typed: capture, `edit`, the rules pack, a docs note. `declared` is
/// whatever else rides along (trigger phrases, anchors, a title). A refusal
/// is recorded in the ledger by fingerprint, never by value, and nothing is
/// stored.
///
/// One function because the guard is only as good as its narrowest caller:
/// it used to be called from capture and from `edit`'s body, so a credential
/// pasted as an `--anchor`, a `--title`, a rules pack or a docs note went
/// into the store (and over sync) unchecked.
pub(crate) fn refuse_secrets(
    conn: &rusqlite::Connection,
    text: &str,
    tags: &[String],
    declared: &[String],
    surface: Surface,
) -> Result<(), Error> {
    let Some(kind) = secrets::scan_capture(text, tags, declared) else {
        return Ok(());
    };
    secrets::record_refusal(
        conn,
        &secrets::capture_hash(text, tags, declared),
        kind,
        surface,
        now_unix(),
    );
    Err(Error::Secret(kind))
}

/// What a duplicate refusal adds when the duplicate is a session handoff
/// that was made restorable again: the capture was refused, and the thing
/// the caller wanted from it will still happen.
pub const REARMED_HANDOFF: &str = "it is the session handoff already stored; it had been \
     restored once and is pending again, so the next clear restores it";

/// How a refusal names whoever forgot a memory. Only a person can override
/// the refusal, so the wording has to send the reader to the right party:
/// a tombstone an agent or a sync peer wrote was not "forgotten by a human".
pub fn forgotten_phrase(by: Option<Actor>) -> &'static str {
    match by {
        Some(Actor::Human) => "a human forgot this",
        Some(Actor::Agent) => "an agent forgot this",
        Some(Actor::Sync) => "this was forgotten on another machine",
        Some(Actor::System) | None => "this was forgotten",
    }
}

/// Run one capture. See the module documentation for the order and why it
/// is fixed.
pub fn capture(mimir: &mut Mimir, req: Capture) -> Result<Captured, Rejected> {
    let now = now_unix();

    // 1. Arguments. Parsed before anything is written: a bad duration costs
    //    the caller an error message, not a memory that silently never
    //    expires.
    let expires = match req.expires_in.as_deref() {
        Some(spec) => Some((
            spec.to_string(),
            memory::parse_expires_in(spec, now)
                .ok_or_else(|| Rejected::invalid(Arg::ExpiresIn, spec))?,
        )),
        None => None,
    };
    let confidence = req
        .confidence
        .as_deref()
        .map(|level| {
            level
                .parse::<MemoryConfidence>()
                .map_err(|_| Rejected::invalid(Arg::Confidence, level))
        })
        .transpose()?;
    let valid_from = parse_day(Arg::ValidFrom, req.valid_from.as_deref())?;
    let valid_to = parse_day(Arg::ValidTo, req.valid_to.as_deref())?;
    if let (Some(from), Some(to)) = (valid_from, valid_to) {
        if to <= from {
            return Err(Rejected::ValidityOrder);
        }
    }

    // 2. Secrets, across every free-text field this capture would store.
    //    The trigger phrases, the anchor patterns and the falsifying
    //    condition all land in `meta` and replicate with the memory, so a
    //    credential pasted into any of them leaks exactly as far as one in
    //    the body would.
    let mut declared = req.fires_when.clone();
    declared.extend(req.anchors.iter().cloned());
    declared.extend(req.resolves_when.iter().cloned());
    refuse_secrets(&mimir.conn, &req.text, &req.tags, &declared, req.surface)?;

    // 3. The link target, while rejecting is still free.
    let link = match req.link.as_deref() {
        None => Linked::NotRequested,
        Some(target) => {
            match crate::commands::resolve_link_target(&mimir.conn, target, req.working_project) {
                Ok(node) => Linked::To {
                    rel: link_rel(&node),
                    target: Box::new(node),
                },
                Err(err) => match req.missing_link {
                    MissingLink::Reject => return Err(Rejected::Link(err)),
                    MissingLink::StoreUnlinked => Linked::Missing(target.to_string()),
                },
            }
        }
    };

    // 4 and 5. The memory and everything asked of it, or nothing.
    let tx = rusqlite::Transaction::new_unchecked(
        &mimir.conn,
        rusqlite::TransactionBehavior::Immediate,
    )?;
    // What was offered, kept for the duplicate case below: `remember` takes
    // the text and the tags.
    let offered_as_handoff = req.tags.iter().any(|t| t == context_guard::HANDOFF_TAG);
    let offered_hash = memory::content_hash(&req.text);
    // The checkout a handoff is written in, which is the one it will be
    // restored in: `working_project` is the checkout's own project even
    // when the memory is filed under another (a linked worktree's goes to
    // the main checkout's), and its path is the checkout's root.
    let handoff_checkout = match req.working_project {
        Some(id) if offered_as_handoff => store::get_node(&tx, id).ok().and_then(|p| p.path),
        _ => None,
    };
    let outcome = memory::remember(
        &tx,
        Remember {
            text: req.text,
            mtype: req.mtype,
            tags: req.tags,
            project_id: req.project_id,
            force: req.force,
            actor: req.actor,
        },
    )?;
    let node = match outcome {
        RememberOutcome::Created(node) => node,
        // Committed, not rolled back: a refusal can leave a note in the
        // review queue ("offered again after being forgotten"), and that
        // note is the point of noticing.
        RememberOutcome::Duplicate(existing) => {
            // Saving the same handoff twice is refused like any duplicate,
            // but it means "this is where I am, again": the stored copy was
            // already consumed by a restore, and without this the next
            // clear would deliver nothing at all.
            //
            // Only the same handoff, offered as one. The duplicate check
            // also refuses text that is merely close, and a new handoff
            // that resembles an old one must not bring the old one's
            // contents back as the state to resume from; nor must a plain
            // note that happens to repeat a handoff's words.
            //
            // And only from the checkout it belongs to: the same words
            // saved in another worktree of the repository are that
            // session's state, and must not make this one pending again
            // for a session that already resumed from it.
            let rearmed_handoff = offered_as_handoff
                && existing.tags().contains(&context_guard::HANDOFF_TAG)
                && existing.content_hash.as_deref() == Some(offered_hash.as_slice())
                && context_guard::restored_at(&existing).is_some()
                && context_guard::handoff_is_for(&existing, handoff_checkout.as_deref());
            if rearmed_handoff {
                context_guard::rearm_handoff(&tx, existing.id)?;
            }
            tx.commit()?;
            return Ok(Captured::Duplicate {
                node: existing,
                rearmed_handoff,
            });
        }
        RememberOutcome::Forgotten(gone) => {
            tx.commit()?;
            let (by, on) = forgotten_by(&mimir.conn, &gone);
            return Ok(Captured::Forgotten { node: gone, by, on });
        }
    };
    if let Linked::To { target, rel } = &link {
        store::link(&tx, node.id, target.id, *rel, 1.0)?;
    }
    // A handoff is restored in the checkout it was written in and nowhere
    // else (`context_guard::handoff_is_for`), so it records which. Asked of
    // the stored node: the tag is what the restore looks for.
    if let Some(checkout) = handoff_checkout
        .as_deref()
        .filter(|_| node.tags().contains(&context_guard::HANDOFF_TAG))
    {
        context_guard::set_handoff_checkout(&tx, node.id, checkout)?;
    }
    let phrases = memory::sanitize_fires_when(req.fires_when);
    if !phrases.is_empty() {
        store::set_fires_when(&tx, node.id, &phrases)?;
    }
    let patterns = anchors::sanitize_anchors(req.anchors);
    if !patterns.is_empty() {
        anchors::set_anchors(&tx, node.id, &patterns)?;
    }
    if expires.is_some() || req.resolves_when.is_some() {
        store::set_expiry(
            &tx,
            node.id,
            expires.as_ref().map(|(_, at)| *at),
            req.resolves_when.as_deref(),
        )?;
    }
    if let Some(level) = confidence {
        store::set_confidence(&tx, node.id, level)?;
    }
    let valid = (valid_from.is_some() || valid_to.is_some()).then_some((valid_from, valid_to));
    if valid.is_some() {
        store::set_valid_interval(&tx, node.id, valid_from, valid_to)?;
    }
    let node = store::get_node(&tx, node.id)?;
    tx.commit()?;

    // 6. Keep semantic recall fresh; a harmless no-op without a model, and
    //    never a reason to report a stored memory as failed.
    if let Err(err) = mimir.embed_pending() {
        tracing::warn!(%err, "embedding new memory failed");
    }

    Ok(Captured::Stored(Box::new(Stored {
        node,
        link,
        expires,
        resolves_when: req.resolves_when,
        confidence,
        valid,
    })))
}

impl Stored {
    /// The lines the CLI prints under the stored memory, one per thing that
    /// was attached, in the order a reader checks them.
    pub fn detail_lines(&self, describe: impl Fn(&Node) -> String) -> Vec<String> {
        let mut lines = Vec::new();
        if let Linked::To { target, .. } = &self.link {
            lines.push(format!("linked → {}", describe(target)));
        }
        if let Some((spec, at)) = &self.expires {
            lines.push(format!("expires in {spec} (at {at})"));
        }
        if let Some(condition) = &self.resolves_when {
            lines.push(format!("resolves when: {condition}"));
        }
        if let Some(level) = self.confidence {
            lines.push(format!("confidence: {level}"));
        }
        if let Some((from, to)) = self.valid {
            lines.push(format!(
                "valid {} → {}",
                from.map(format::full_date).unwrap_or_else(|| "?".into()),
                to.map(format::full_date).unwrap_or_else(|| "now".into())
            ));
        }
        lines
    }

    /// The same attachments as fields of the memory's JSON object, so
    /// `--json` output is one object a caller can parse whole rather than a
    /// JSON line followed by prose.
    pub fn extend_json(&self, value: &mut serde_json::Value) {
        if let Linked::To { target, rel } = &self.link {
            value["linked"] = serde_json::json!({
                "id": short_uid(target.kind, &target.uid),
                "rel": rel.as_str(),
            });
        }
        if let Some((_, at)) = &self.expires {
            value["expires_at"] = serde_json::json!(at);
        }
        if let Some(condition) = &self.resolves_when {
            value["resolves_when"] = serde_json::json!(condition);
        }
        if let Some(level) = self.confidence {
            value["confidence"] = serde_json::json!(level.to_string());
        }
        if let Some((from, to)) = self.valid {
            value["valid_from"] = serde_json::json!(from);
            value["valid_to"] = serde_json::json!(to);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_core::model::NewNode;

    fn engine() -> Mimir {
        Mimir::open_in_memory().unwrap()
    }

    /// A plain global capture typed by a person; each test changes the one
    /// field it is about.
    fn request(text: &str) -> Capture {
        Capture {
            text: text.into(),
            mtype: MemoryType::Note,
            tags: Vec::new(),
            project_id: None,
            working_project: None,
            force: false,
            actor: Actor::Human,
            surface: Surface::CliRemember,
            link: None,
            missing_link: MissingLink::Reject,
            fires_when: Vec::new(),
            anchors: Vec::new(),
            expires_in: None,
            resolves_when: None,
            confidence: None,
            valid_from: None,
            valid_to: None,
        }
    }

    fn memories(m: &Mimir) -> i64 {
        m.conn
            .query_row("SELECT count(*) FROM node WHERE kind = 'memory'", [], |r| {
                r.get(0)
            })
            .unwrap()
    }

    fn stored(m: &mut Mimir, req: Capture) -> Stored {
        match capture(m, req) {
            Ok(Captured::Stored(stored)) => *stored,
            other => panic!("expected a stored memory, got {other:?}"),
        }
    }

    fn rejected(m: &mut Mimir, req: Capture) -> Rejected {
        match capture(m, req) {
            Err(rejected) => rejected,
            Ok(other) => panic!("expected a rejection, got {other:?}"),
        }
    }

    /// "Rejected" has to mean nothing was written, for every argument a
    /// capture can be rejected on — otherwise the corrected retry is refused
    /// as a duplicate of the half-made memory. And the message has to name
    /// the argument the way the caller typed it.
    #[test]
    fn a_rejected_argument_stores_nothing_and_is_named_in_the_callers_words() {
        let mut m = engine();
        let text = "the deploy key rotates every quarter";

        let mut bad = request(text);
        bad.expires_in = Some("soon".into());
        let why = rejected(&mut m, bad);
        assert_eq!(
            why.message(Surface::CliRemember),
            "invalid --expires-in \"soon\": expected a positive duration like 90d, 2w or 12h \
             (units: h/d/w — use 365d for a year)"
        );
        assert!(why
            .message(Surface::McpRemember)
            .starts_with("invalid expires_in \"soon\": expected a positive duration"));

        let mut bad = request(text);
        bad.confidence = Some("maybe".into());
        assert_eq!(
            rejected(&mut m, bad).message(Surface::McpRemember),
            "invalid confidence \"maybe\": expected certain, likely or unsure"
        );

        let mut bad = request(text);
        bad.valid_from = Some("last spring".into());
        assert_eq!(
            rejected(&mut m, bad).message(Surface::CliRemember),
            "invalid --valid-from \"last spring\": expected a date like 2024-06-30"
        );

        let mut bad = request(text);
        bad.valid_from = Some("2024-06-30".into());
        bad.valid_to = Some("2024-06-01".into());
        assert_eq!(
            rejected(&mut m, bad).message(Surface::CliRemember),
            "--valid-to must be after --valid-from"
        );

        assert_eq!(memories(&m), 0, "a rejected capture wrote a memory");
        let mut retry = request(text);
        retry.expires_in = Some("90d".into());
        stored(&mut m, retry);
    }

    /// Trigger phrases, anchor patterns and the falsifying condition are
    /// stored in `meta` and replicate with the memory, so a credential in
    /// any of them leaks as far as one in the body. Only the body, the tags
    /// and the trigger phrases used to be scanned.
    #[test]
    fn a_secret_in_any_field_the_capture_would_store_is_refused() {
        let secret = format!("AKIA{}", "IOSFODNN7EXAMPLE");
        let mut m = engine();
        for field in ["fires_when", "anchors", "resolves_when"] {
            let mut req = request("rotate the uploader credentials after the migration");
            match field {
                "fires_when" => req.fires_when = vec![secret.clone()],
                "anchors" => req.anchors = vec![secret.clone()],
                _ => req.resolves_when = Some(format!("when {secret} is revoked")),
            }
            let why = rejected(&mut m, req);
            assert!(
                matches!(why, Rejected::Store(Error::Secret(_))),
                "a secret in {field} was not refused: {why:?}"
            );
        }
        assert_eq!(memories(&m), 0);
    }

    /// The one deliberate difference in what a missing link target does:
    /// a person's capture is rejected whole, an agent's is stored and the
    /// outcome says the link was not made.
    #[test]
    fn a_missing_link_target_rejects_for_a_person_and_stores_unlinked_for_an_agent() {
        let mut m = engine();
        let mut person = request("the retry budget is three attempts");
        person.link = Some("no_such_symbol".into());
        assert!(matches!(rejected(&mut m, person), Rejected::Link(_)));
        assert_eq!(memories(&m), 0);

        let mut agent = request("the retry budget is three attempts");
        agent.link = Some("no_such_symbol".into());
        agent.actor = Actor::Agent;
        agent.surface = Surface::McpRemember;
        agent.missing_link = MissingLink::StoreUnlinked;
        let kept = stored(&mut m, agent);
        assert!(matches!(&kept.link, Linked::Missing(t) if t == "no_such_symbol"));
    }

    /// A symbol name resolves in the project the caller is working in even
    /// when the memory itself is stored globally, and the relation says what
    /// the link is: *about* something indexed from disk, *relates* to
    /// another memory.
    #[test]
    fn a_link_is_about_an_artifact_and_relates_to_a_memory() {
        let mut m = engine();
        let mut proj = NewNode::new(Kind::Project);
        proj.title = Some("probe".into());
        let project = store::insert_node(&m.conn, proj).unwrap();
        let mut sym = NewNode::new(Kind::Symbol);
        sym.title = Some("retry_with_backoff".into());
        sym.project_id = Some(project.id);
        let symbol = store::insert_node(&m.conn, sym).unwrap();

        let mut global = request("retries back off exponentially with full jitter");
        global.working_project = Some(project.id);
        global.link = Some("retry_with_backoff".into());
        let first = stored(&mut m, global);
        assert!(first.node.project_id.is_none(), "the memory is global");
        assert!(
            matches!(&first.link, Linked::To { target, rel: Rel::About } if target.id == symbol.id)
        );

        let mut second = request("billing webhooks are verified before they are queued");
        second.link = Some(first.node.uid.clone());
        let second = stored(&mut m, second);
        assert!(matches!(
            &second.link,
            Linked::To {
                rel: Rel::Relates,
                ..
            }
        ));
    }

    /// The node handed back is read after every attachment, so what a
    /// surface prints is what the store holds — not the bare memory as it
    /// was the instant after the insert.
    #[test]
    fn the_stored_node_carries_everything_that_was_attached() {
        let mut m = engine();
        let mut req = request("the staging certificate is pinned to the old intermediate");
        req.expires_in = Some("30d".into());
        req.resolves_when = Some("when the new intermediate is deployed".into());
        req.confidence = Some("unsure".into());
        req.anchors = vec!["deploy.sh".into()];
        req.fires_when = vec!["release checklist".into()];
        req.valid_from = Some("2024-01-01".into());
        req.valid_to = Some("2024-06-30".into());
        let kept = stored(&mut m, req);

        let (_, at) = kept.expires.clone().expect("an expiry was asked for");
        assert_eq!(kept.node.expires_at(), Some(at));
        assert_eq!(
            kept.node.meta["resolves_when"],
            "when the new intermediate is deployed"
        );
        assert_eq!(kept.node.meta["confidence"], "unsure");
        assert_eq!(kept.node.meta["anchors"][0], "deploy.sh");
        assert_eq!(kept.node.meta["fires_when"][0], "release checklist");

        let lines = kept.detail_lines(|n| n.uid.clone());
        assert_eq!(
            lines,
            vec![
                format!("expires in 30d (at {at})"),
                "resolves when: when the new intermediate is deployed".to_string(),
                "confidence: unsure".to_string(),
                "valid 2024-01-01 → 2024-06-30".to_string(),
            ]
        );

        let mut json = serde_json::json!({});
        kept.extend_json(&mut json);
        assert_eq!(json["expires_at"], at);
        assert_eq!(json["confidence"], "unsure");
        assert!(json["valid_from"].is_i64() && json["valid_to"].is_i64());
    }

    /// A handoff is restored once. Saving the very same handoff again is
    /// refused as a duplicate, and used to leave the stored copy consumed,
    /// so the next clear delivered nothing. Offering it again makes it
    /// pending again, and the refusal says so.
    #[test]
    fn saving_the_same_handoff_again_makes_it_pending_again() {
        use mimir_core::model::Scope;
        let mut m = engine();
        let handoff = || {
            let mut req = request("handoff: migration half applied, run step 4 next");
            req.tags = vec![context_guard::HANDOFF_TAG.into()];
            req
        };
        stored(&mut m, handoff());
        assert!(context_guard::restore_handoff_text(&m.conn, Scope::Global)
            .unwrap()
            .is_some());
        assert!(context_guard::current_handoff(&m.conn, Scope::Global)
            .unwrap()
            .is_none());

        match capture(&mut m, handoff()) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(rearmed_handoff),
            other => panic!("expected the duplicate refusal, got {other:?}"),
        }
        assert!(
            context_guard::current_handoff(&m.conn, Scope::Global)
                .unwrap()
                .is_some(),
            "the next clear has a handoff to restore"
        );

        // The same words offered as a plain note are a duplicate and nothing
        // more: nobody asked for a handoff.
        context_guard::restore_handoff_text(&m.conn, Scope::Global).unwrap();
        match capture(
            &mut m,
            request("handoff: migration half applied, run step 4 next"),
        ) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(!rearmed_handoff),
            other => panic!("expected the duplicate refusal, got {other:?}"),
        }
        // Nor is a handoff that only resembles the stored one. It is refused
        // as a near-duplicate, and the old text must not become the state
        // to resume from on the strength of that.
        let mut near = request("handoff: migration half applied, run step 4 next today");
        near.tags = vec![context_guard::HANDOFF_TAG.into()];
        match capture(&mut m, near) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(!rearmed_handoff),
            other => panic!("expected the near-duplicate refusal, got {other:?}"),
        }
        assert!(context_guard::current_handoff(&m.conn, Scope::Global)
            .unwrap()
            .is_none());

        // An ordinary duplicate is only a duplicate.
        stored(&mut m, request("the retry budget is three attempts"));
        match capture(&mut m, request("the retry budget is three attempts")) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(!rearmed_handoff),
            other => panic!("expected the duplicate refusal, got {other:?}"),
        }
    }

    /// The refusal for a deliberately forgotten memory says who forgot it,
    /// read from the ledger. It used to say "a human forgot this" for a
    /// tombstone an agent had written through the MCP `forget` tool.
    #[test]
    fn a_forgotten_refusal_names_who_forgot() {
        let mut m = engine();
        let text = "the legacy importer needs python 2";
        let kept = stored(&mut m, request(text));
        store::soft_delete(&m.conn, kept.node.id, Actor::Agent, None).unwrap();

        match capture(&mut m, request(text)) {
            Ok(Captured::Forgotten { node, by, on }) => {
                assert_eq!(node.id, kept.node.id);
                assert_eq!(by, Some(Actor::Agent));
                assert_eq!(Some(on), node.deleted_at);
                assert_eq!(forgotten_phrase(by), "an agent forgot this");
            }
            other => panic!("expected the forgotten refusal, got {other:?}"),
        }
        assert_eq!(forgotten_phrase(Some(Actor::Human)), "a human forgot this");
        assert_eq!(forgotten_phrase(None), "this was forgotten");
    }

    /// A repository's main checkout and two linked worktrees of it, as
    /// project rows: the worktrees' memories are filed under the first.
    fn checkouts(m: &Mimir) -> (Node, Node, Node) {
        let project = |path: &str, name: &str| store::ensure_project(&m.conn, path, name).unwrap();
        (
            project("/repo/main", "main"),
            project("/repo/wt-a", "wt-a"),
            project("/repo/wt-b", "wt-b"),
        )
    }

    /// A handoff offered from `checkout`, filed under the repository's
    /// memory project `home`.
    fn handoff_from(home: &Node, checkout: &Node, text: &str) -> Capture {
        let mut req = request(text);
        req.tags = vec![context_guard::HANDOFF_TAG.into()];
        req.project_id = Some(home.id);
        req.working_project = Some(checkout.id);
        req
    }

    /// A stored handoff says which checkout wrote it: the working project's
    /// root, not the project the memory is filed under. Nothing else does.
    #[test]
    fn a_handoff_records_the_checkout_it_was_written_in() {
        let mut m = engine();
        let (home, wt_a, _) = checkouts(&m);
        let kept = stored(
            &mut m,
            handoff_from(&home, &wt_a, "handoff: A is mid-rebase onto main"),
        );
        assert_eq!(kept.node.project_id, Some(home.id));
        assert_eq!(
            context_guard::handoff_checkout(&kept.node),
            Some("/repo/wt-a"),
            "the node handed back already carries it"
        );

        // A plain memory written in the same checkout records none.
        let mut note = request("the importer batches by five hundred rows");
        note.project_id = Some(home.id);
        note.working_project = Some(wt_a.id);
        assert_eq!(
            context_guard::handoff_checkout(&stored(&mut m, note).node),
            None
        );
        // Nor does a handoff written outside any project.
        let mut nowhere = request("handoff: nothing in flight, start on the changelog");
        nowhere.tags = vec![context_guard::HANDOFF_TAG.into()];
        assert_eq!(
            context_guard::handoff_checkout(&stored(&mut m, nowhere).node),
            None
        );
    }

    /// Re-arming follows the rule restoring does. The same words saved from
    /// another worktree are refused as the duplicate they are, and do not
    /// make this checkout's consumed handoff pending again; saved again
    /// from its own checkout, they do.
    #[test]
    fn the_same_handoff_saved_from_another_checkout_does_not_rearm_it() {
        use mimir_core::model::Scope;
        let mut m = engine();
        let (home, wt_a, wt_b) = checkouts(&m);
        let scope = Scope::Project(home.id);
        let text = "handoff: migration half applied, run step 4 next";
        stored(&mut m, handoff_from(&home, &wt_a, text));
        assert!(
            context_guard::restore_handoff_text_in(&m.conn, scope, wt_a.path.as_deref())
                .unwrap()
                .is_some()
        );
        let pending_in = |m: &Mimir, checkout: &Node| {
            context_guard::current_handoff_in(&m.conn, scope, checkout.path.as_deref())
                .unwrap()
                .is_some()
        };
        assert!(!pending_in(&m, &wt_a), "consumed");

        match capture(&mut m, handoff_from(&home, &wt_b, text)) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(!rearmed_handoff, "another checkout's save"),
            other => panic!("expected the duplicate refusal, got {other:?}"),
        }
        assert!(!pending_in(&m, &wt_a), "A already resumed from it");
        assert!(!pending_in(&m, &wt_b), "and it never was B's");

        match capture(&mut m, handoff_from(&home, &wt_a, text)) {
            Ok(Captured::Duplicate {
                rearmed_handoff, ..
            }) => assert!(rearmed_handoff, "its own checkout's save"),
            other => panic!("expected the duplicate refusal, got {other:?}"),
        }
        assert!(pending_in(&m, &wt_a));
        assert!(!pending_in(&m, &wt_b));
    }
}
