//! Transport-agnostic replication for the OPTIONAL sync layer.
//!
//! Scope: GLOBAL memories, plus memories of **sync-enabled projects** (those
//! with a portable key — see `scope::portable_key`), and the edges among synced
//! memories. A project memory carries its project's portable key on the wire; on
//! apply it resolves to the local project, creating a path-less *shadow* project
//! if that project hasn't been opened on this machine yet (adopted on first
//! local open). Identity is the node `uid` (a globally unique ULID), and a
//! merge keeps, per uid, the version that stands highest ([`Standing`]: when
//! the text was decided, then the last write, then a digest). That is a total
//! order on versions, so the stores converge whatever order they exchange in,
//! and applying a batch twice changes nothing. Timestamps are whole seconds;
//! two versions written in the same second are told apart by the digest, so
//! one is kept and it is the same one everywhere. Tombstones ride along as
//! `deleted_at`. Local-only signals
//! (strength, access_count, recall_event) and embeddings are never synced; a
//! peer recomputes embeddings locally. `co_recalled` edges are local-only for
//! the same reason and are filtered in BOTH directions — see `read_batch`
//! and the edge loop in `apply_changes`.
//!
//! What a memory's state means across machines, by [`Lifecycle`]:
//!
//! ```text
//!   Forgotten   replicates: the tombstone travels, and purges prior wordings
//!               wherever it lands.
//!   Superseded  replicates: the record names its replacement by uid, and a
//!               receiver that holds the replacement retires its own copy.
//!   Expired     replicates for free: `expires_at` lives in `meta`.
//!   Archived    stays local: it is derived from strength and access counts,
//!               which never leave the machine. On the wire an archived
//!               memory is the live memory it was.
//! ```
//!
//! Two clocks are in play and they answer different questions. A row's own
//! timestamps say when it was *written*, on whichever machine wrote it; they
//! decide which version wins. When a row *arrived* at this store is not on
//! the row at all: it is the mutation ledger row `apply_changes` writes with
//! actor `Sync`. A store reporting its own changes cuts by the first
//! ([`changes_since`]); a hub relaying to a peer must cut by the second as
//! well ([`relay_since`]), because a peer that was offline delivers old
//! timestamps late.

use std::borrow::Cow;

use rusqlite::{params, Connection, OptionalExtension, Row, TransactionBehavior};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::{now_unix, Actor, Kind, MemoryType, MutationOp, Rel};
use crate::store;

/// One memory node on the wire. `type` is the memory subkind; `content_hash`
/// is hex-encoded; `deleted_at` is the tombstone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRecord {
    pub uid: String,
    #[serde(rename = "type")]
    pub mtype: Option<String>,
    pub title: Option<String>,
    pub body: Option<String>,
    pub tags: Vec<String>,
    pub pinned: bool,
    pub content_hash: Option<String>,
    pub meta: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
    pub deleted_at: Option<i64>,
    /// Portable key of the owning project (`None` = global memory). Absent on the
    /// wire for global records, so existing global-only peers are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_key: Option<String>,
    /// Uid of the memory that replaced this one (`node.superseded_by`), when
    /// it was superseded. Absent on the wire otherwise — and from every peer
    /// older than the field, so absence says nothing: a receiver sets its own
    /// column from this and never clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

/// A memory↔memory edge on the wire, identified by endpoint uids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEdge {
    pub src_uid: String,
    pub dst_uid: String,
    pub rel: String,
    pub weight: f64,
    pub created_at: i64,
}

/// A unit of replication: nodes, their edges, and the high watermark the
/// sender observed (for the receiver to advance its cursor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBatch {
    pub nodes: Vec<SyncRecord>,
    pub edges: Vec<SyncEdge>,
    pub watermark: i64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ApplyStats {
    pub nodes_upserted: usize,
    pub nodes_skipped: usize,
    pub edges_upserted: usize,
    pub edges_skipped: usize,
    pub tombstones: usize,
}

/// SQL predicate selecting the memories that replicate, for a memory aliased
/// `m` LEFT JOINed to its project aliased `p`: globals (`project_id IS
/// NULL`), plus memories of a sync-enabled project that carries a portable
/// key. The only SQL spelling of the rule — the node query, the edge query
/// (once per endpoint) and the watermark all take it from here.
fn syncable_sql(m: &str, p: &str) -> String {
    format!(
        "{m}.kind='memory' AND ({m}.project_id IS NULL OR \
         (json_extract({p}.meta,'$.sync')=1 \
          AND json_extract({p}.meta,'$.portable_key') IS NOT NULL))"
    )
}

/// Rust twin of [`syncable_sql`]'s project condition: a project's memories
/// sync only when its meta has `sync` on AND a `portable_key`. Callers (e.g.
/// `reproject`'s warning) rely on the two agreeing, and the test
/// `syncable_has_one_answer_in_sql_and_rust` holds them to it, so this reads
/// the JSON the way SQLite does: `json_extract(..) = 1` is true for `true`
/// and for the number 1, and a JSON `null` key extracts as SQL NULL — which
/// is what `set_project_identity` stores for a project with no key.
pub fn project_is_syncable(meta: &serde_json::Value) -> bool {
    let on = meta
        .get("sync")
        .is_some_and(|v| v.as_bool() == Some(true) || v.as_f64() == Some(1.0));
    let keyed = meta.get("portable_key").is_some_and(|v| !v.is_null());
    on && keyed
}

/// A row's latest activity in its writer's clock: what last-write-wins
/// compares, and what [`changes_since`] cuts by.
const ACTIVITY: &str = "max(m.updated_at, COALESCE(m.deleted_at, 0))";

/// Which rows a batch carries, and which clock its watermark is in.
#[derive(Clone, Copy)]
enum Cut {
    /// Rows written after the cursor. Watermark: the newest activity in the
    /// synced set, in the clocks of whoever wrote the rows.
    Written,
    /// Rows written after the cursor, plus rows that reached this store over
    /// sync after it. Watermark: this store's own clock when the read began.
    Arrived,
}

/// "Changed since `since`" = syncable memories whose latest activity
/// (max of updated_at and deleted_at) is newer than `since`, plus all
/// *authored* edges among synced memories (sparse; shipped whole and upserted
/// idempotently).
///
/// This is what a store says about itself: a client's push, and the file
/// transport's snapshot. A hub answering a pull uses [`relay_since`].
pub fn changes_since(conn: &Connection, since: i64) -> Result<SyncBatch> {
    read_batch(conn, since, Cut::Written)
}

/// What a hub hands a peer that last pulled at `since`: everything
/// [`changes_since`] would, plus every syncable memory that *arrived here*
/// over sync after `since`, whatever its own timestamps say.
///
/// The second half is the point. The hub keeps each row's origin timestamps
/// (they decide last-write-wins), so a memory written on a machine that was
/// offline and pushed an hour later lands with a timestamp an hour old —
/// below the cursor of every peer that pulled in the meantime, which would
/// then never see it. Its arrival is in the mutation ledger (`apply_changes`
/// records every row it applies, actor `Sync`), and that is what a pull must
/// be cut by.
///
/// The watermark is this store's clock at the start of the read, not the
/// newest row timestamp: a peer's next `since` is then in the same clock as
/// the arrival times it will be compared with, so a later arrival is always
/// above it. Rows written directly on the hub have no arrival row; the
/// activity half covers them, in the same clock.
pub fn relay_since(conn: &Connection, since: i64) -> Result<SyncBatch> {
    read_batch(conn, since, Cut::Arrived)
}

/// One batch from one snapshot: the node query, the edge query and the
/// watermark must describe the same instant, or a commit landing between
/// them puts the watermark past a row the batch does not carry.
fn read_batch(conn: &Connection, since: i64, cut: Cut) -> Result<SyncBatch> {
    // Read the clock before the snapshot opens: whatever commits after the
    // snapshot is then stamped at or after this, never before it.
    let started = now_unix();
    // A deferred transaction pins one snapshot for every statement in it.
    // A caller already inside a transaction has that guarantee from its own.
    let snapshot = if conn.is_autocommit() {
        Some(conn.unchecked_transaction()?)
    } else {
        None
    };

    let syncable = syncable_sql("m", "p");
    // A replacement that does not sync is named to nobody: a peer could not
    // resolve the uid, and it would tell the hub that a memory of a project
    // kept off sync exists.
    let replacement_syncs = syncable_sql("r", "rp");
    let arrived = match cut {
        Cut::Written => String::new(),
        Cut::Arrived => format!(
            " OR EXISTS (SELECT 1 FROM mutation a
                         WHERE a.node_id = m.id AND a.actor = '{}' AND a.at > ?1)",
            Actor::Sync.as_str()
        ),
    };
    let mut stmt = conn.prepare(&format!(
        "SELECT m.uid, m.subkind, m.title, m.body, m.tags_text, m.pinned, m.content_hash, m.meta,
                m.created_at, m.updated_at, m.deleted_at,
                json_extract(p.meta,'$.portable_key') AS project_key,
                CASE WHEN {replacement_syncs} THEN r.uid END AS superseded_by
         FROM node m
         LEFT JOIN node p ON p.id = m.project_id AND p.kind='project'
         LEFT JOIN node r ON r.id = m.superseded_by
         LEFT JOIN node rp ON rp.id = r.project_id AND rp.kind='project'
         WHERE {syncable}
           AND ({ACTIVITY} > ?1{arrived})
         ORDER BY {ACTIVITY} ASC",
    ))?;
    let nodes = stmt
        .query_map([since], record_from_row)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    // Edges between two syncable memories (each endpoint global, or in a
    // sync-enabled keyed project).
    //
    // `co_recalled` is excluded because it is not a claim anyone authored: it
    // is what THIS install observed about its own opens, per-machine and
    // regenerable from local `recall_event` rows. Shipping it would forge that
    // history on the peer — and its Hebbian weight and `meta.last_fired_at`
    // are meaningless there, since the wire carries neither the meta nor the
    // peer's own decay clock. Filtered here in SQL rather than in the Rust
    // map, so the rel never reaches the batch at all (and `snapshot`, which is
    // the file transport's whole payload, can't leak it either).
    let mut estmt = conn.prepare(&format!(
        "SELECT s.uid, d.uid, e.rel, e.weight, e.created_at
         FROM edge e
         JOIN node s ON s.id = e.src
         JOIN node d ON d.id = e.dst
         LEFT JOIN node sp ON sp.id = s.project_id AND sp.kind='project'
         LEFT JOIN node dp ON dp.id = d.project_id AND dp.kind='project'
         WHERE e.rel <> '{}'
           AND {}
           AND {}",
        Rel::CoRecalled.as_str(),
        syncable_sql("s", "sp"),
        syncable_sql("d", "dp"),
    ))?;
    let edges = estmt
        .query_map([], |r: &Row| {
            Ok(SyncEdge {
                src_uid: r.get(0)?,
                dst_uid: r.get(1)?,
                rel: r.get(2)?,
                weight: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(estmt);

    let watermark = match cut {
        Cut::Written => current_high_watermark(conn)?,
        Cut::Arrived => started,
    };
    drop(snapshot);
    Ok(SyncBatch {
        nodes,
        edges,
        watermark,
    })
}

/// The record this store sends for one of its rows. One mapping for what
/// goes out in a batch and for what an arriving record is compared with
/// (`held`), so the two can never describe a row differently.
fn record_from_row(r: &Row) -> rusqlite::Result<SyncRecord> {
    let hash: Option<Vec<u8>> = r.get("content_hash")?;
    let tags_text: String = r.get("tags_text")?;
    let meta_text: String = r.get("meta")?;
    let mut record = Cow::Owned(SyncRecord {
        uid: r.get("uid")?,
        mtype: r.get("subkind")?,
        title: r.get("title")?,
        body: r.get("body")?,
        tags: tags_text.split_whitespace().map(String::from).collect(),
        pinned: r.get::<_, i64>("pinned")? != 0,
        content_hash: hash.as_deref().map(to_hex),
        meta: serde_json::from_str(&meta_text).unwrap_or(serde_json::Value::Null),
        created_at: r.get("created_at")?,
        updated_at: r.get("updated_at")?,
        deleted_at: r.get("deleted_at")?,
        project_key: r.get("project_key")?,
        superseded_by: r.get("superseded_by")?,
    });
    keep_archival_local(&mut record);
    Ok(record.into_owned())
}

/// Whether a wire record says its memory is archived — asked of the store's
/// one derivation, so "archived" can never mean one thing in a row and
/// another in a record. The node built for the question carries what that
/// derivation reads for a tombstone (`deleted_at` and `meta`); the answer
/// does not depend on the clock.
fn wire_is_archived(record: &SyncRecord) -> bool {
    record.deleted_at.is_some() && crate::lifecycle::meta_says_archived(&record.meta)
}

/// Turn an archived record back into the live memory it was.
///
/// Decay archival is decided by strength and access counts, which are
/// per-machine and never synced: the laptop that has not opened a note in
/// months says nothing about the desktop that opens it daily. Shipping the
/// tombstone made one machine's idleness delete the memory everywhere. So
/// archival does not travel — this runs on every record on its way out, and
/// on every record on its way in, because a peer on an older build still
/// sends the tombstone with `meta.archived` set.
fn keep_archival_local(record: &mut Cow<'_, SyncRecord>) {
    if !wire_is_archived(record) {
        return;
    }
    let live = record.to_mut();
    live.deleted_at = None;
    if let Some(meta) = live.meta.as_object_mut() {
        meta.remove("archived");
    }
}

/// The full synced set, tombstones included (for the file transport, which
/// ships a convergent snapshot rather than an incremental diff).
pub fn snapshot(conn: &Connection) -> Result<SyncBatch> {
    changes_since(conn, i64::MIN)
}

/// Newest activity timestamp across the synced set (the next watermark).
pub fn current_high_watermark(conn: &Connection) -> Result<i64> {
    let v: Option<i64> = conn
        .query_row(
            &format!(
                "SELECT max({ACTIVITY})
                 FROM node m
                 LEFT JOIN node p ON p.id = m.project_id AND p.kind='project'
                 WHERE {}",
                syncable_sql("m", "p")
            ),
            [],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(v.unwrap_or(0))
}

/// What this store holds for a uid when a record for it arrives.
struct Held {
    /// The row as this store would send it: what the arriving record is
    /// compared with. A row archived here reads as the live row it was (see
    /// `keep_archival_local`), so the decay pass's stamp is never mistaken
    /// for a write.
    record: SyncRecord,
    /// The row has `deleted_at` set, for whatever reason: forgotten, or
    /// archived here. What the ledger calls a restore when a live version
    /// replaces it.
    tombstone: bool,
    content_hash: Option<Vec<u8>>,
}

fn held(conn: &Connection, uid: &str) -> Result<Option<Held>> {
    Ok(conn
        .query_row(
            "SELECT m.uid, m.subkind, m.title, m.body, m.tags_text, m.pinned, m.content_hash,
                    m.meta, m.created_at, m.updated_at, m.deleted_at,
                    json_extract(p.meta,'$.portable_key') AS project_key,
                    NULL AS superseded_by
             FROM node m
             LEFT JOIN node p ON p.id = m.project_id AND p.kind='project'
             WHERE m.uid = ?1",
            [uid],
            |r| {
                Ok(Held {
                    tombstone: r.get::<_, Option<i64>>("deleted_at")?.is_some(),
                    content_hash: r.get("content_hash")?,
                    record: record_from_row(r)?,
                })
            },
        )
        .optional()?)
}

/// Where one version of a memory stands against another version of the same
/// memory: the one definition of which write wins. The greater one is kept,
/// on every store.
///
/// Compared in this order:
///
/// 1. **When what the memory says was decided.** For a forgotten memory,
///    when it was forgotten. For a live one, when its text was written
///    ([`crate::text_clock`]; `updated_at` when the row carries no clock,
///    which is every row before its first annotation and every row an older
///    build wrote). A later decision wins: forgetting a memory beats a pin,
///    a tag or an anchor put on it by a machine that had not heard, however
///    much later that happened, and a rewrite beats an earlier forget, as
///    it always did.
/// 2. **Forgotten beats live** when the two were decided in the same second.
/// 3. **When the row was last written at all.** Between two versions with
///    the same text, the later annotation wins.
/// 4. **A digest of what the version holds**, so that two different versions
///    written in the same second still have one winner and the stores do not
///    each keep their own for ever. Which of the two wins is arbitrary; that
///    it is the same one everywhere is the point.
///
/// It has to stay a total order on what the stores hold: the stores converge
/// because each keeps the greatest version it has seen. A rule that is not
/// one ("a forget beats a pin, a rewrite beats a forget, the later of pin
/// and rewrite wins" is a cycle, and was tried) leaves three machines
/// passing three versions round for ever.
///
/// One thing is not a write at all: **decay**. A row archived here by the
/// decay pass is compared as the live row it was (`Held::record`), so the
/// moment this machine's housekeeping ran cannot shield it from a forget or
/// an edit made elsewhere. Archival does not travel (`keep_archival_local`).
///
/// A store on a build without the text clock orders by steps 2 and 3 alone.
/// It agrees with this one whenever no annotation was made after an unseen
/// forget or rewrite, and until every machine and the hub are upgraded the
/// two can disagree about exactly those rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Standing {
    decided: i64,
    forgotten: bool,
    written: i64,
    digest: [u8; 32],
}

impl Standing {
    /// `record` is a record as it travels: archival already taken out.
    fn of(record: &SyncRecord) -> Self {
        let written = record.updated_at.max(record.deleted_at.unwrap_or(0));
        let decided = match record.deleted_at {
            Some(_) => written,
            None => crate::text_clock::decided(
                &record.meta,
                record.content_hash.as_deref(),
                record.updated_at,
            ),
        };
        Standing {
            decided,
            forgotten: record.deleted_at.is_some(),
            written,
            digest: digest(record),
        }
    }
}

/// What a version holds, reduced to something two stores can compare. Taken
/// over the record as a store would hold it after applying it (an unknown
/// type reads as none, tags are re-split, a malformed hash is no hash), so a
/// version and its own stored copy always agree and are never "different".
///
/// Left out: the timestamps (they are compared before the digest is), the
/// replacement (merged separately, and only ever set), `created_at` (never
/// overwritten once a row exists), the text clock (two versions that differ
/// only in how they spell the same decision time are one version), the
/// project key, and the content hash.
///
/// The project key is what a store currently calls the memory's project,
/// and it changes without anyone writing to the memory: a repository's git
/// remote moves, and the next resolution on that machine records the new
/// key. Two stores then send one unchanged memory under two keys. In the
/// digest that difference decided the tie, and the memories moved into a
/// shadow project on whichever store lost it.
///
/// The content hash The hash is derived from the body, which is in; and a store
/// fills in a hash a peer did not send (the embed pass backfills it), so a
/// hash in the digest made the same version look new on every round.
fn digest(record: &SyncRecord) -> [u8; 32] {
    let mtype = record
        .mtype
        .as_deref()
        .filter(|t| t.parse::<MemoryType>().is_ok());
    let tags = record
        .tags
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut meta = record.meta.clone();
    if let Some(fields) = meta.as_object_mut() {
        fields.remove(crate::text_clock::KEY);
    }
    let held = serde_json::json!([mtype, record.title, record.body, tags, record.pinned, meta]);
    *blake3::hash(held.to_string().as_bytes()).as_bytes()
}

/// Whether an arriving record replaces what this store holds for its uid.
fn incoming_wins(held: Option<&Held>, incoming: &SyncRecord) -> bool {
    held.is_none_or(|held| Standing::of(incoming) > Standing::of(&held.record))
}

/// Merge a batch in: per uid the version that stands highest is kept
/// ([`Standing`]); edges are upserted by endpoint uid (danglers skipped, never
/// fatal). Idempotent.
///
/// Every row that changes this store is recorded in the mutation ledger with
/// actor `Sync` and the op that names what the arrival did here. The row's
/// own timestamps stay the writer's, so that ledger row is the only record of
/// *when it got here* — which is what [`relay_since`] selects by.
pub fn apply_changes(conn: &Connection, batch: &SyncBatch) -> Result<ApplyStats> {
    let mut stats = ApplyStats::default();
    // Uids whose incoming record replaced what this store held, in this batch.
    let mut won: std::collections::HashSet<&str> = std::collections::HashSet::new();
    // IMMEDIATE + one transaction for the whole batch: a crash or error
    // mid-apply must not leave a partial batch (some nodes/edges upserted,
    // others not) on disk. Same lock-upgrade rationale as store::record_shown.
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    for incoming in &batch.nodes {
        // An archived record deletes nothing and creates no tombstone (see
        // `keep_archival_local`): from here on it is the live row it carries.
        let mut n = Cow::Borrowed(incoming);
        keep_archival_local(&mut n);
        let n = n.as_ref();
        let hash: Option<Vec<u8>> = n.content_hash.as_deref().and_then(from_hex);
        let tags_text = n.tags.join(" ");
        // Resolve the owning project from its portable key (None = global). A
        // key with no local project yet gets a path-less shadow project, adopted
        // when that project is later opened locally.
        let project_id: Option<i64> = match n.project_key.as_deref() {
            None => None,
            Some(key) => Some(store::resolve_or_shadow_project(&tx, key)?),
        };
        // Wire `type` must be a subkind this build knows. Unknown values
        // coerce to None (reads back as "note") rather than being stored
        // verbatim: dashboards and filters group on subkind, so a hostile
        // peer must not dictate arbitrary display strings into this store.
        // A NEWER peer's new types degrade to note here instead of dropping
        // the memory — convergence beats typing fidelity across versions.
        let mtype: Option<String> = n.mtype.clone().filter(|s| s.parse::<MemoryType>().is_ok());
        // The batch holds the write lock, so what is read here is what the
        // upsert below replaces.
        let before = held(&tx, &n.uid)?;
        if !incoming_wins(before.as_ref(), n) {
            stats.nodes_skipped += 1;
            continue;
        }
        // Upsert; on uid conflict, overwrite only the synced fields. Local
        // signals (strength, access_count, …) are never touched.
        // `superseded_by` is not in the list on purpose: it is merged after
        // the loop, and an incoming row must never clear it.
        tx.execute(
            "INSERT INTO node (uid, kind, subkind, project_id, title, body, tags_text, content_hash,
                               meta, pinned, created_at, updated_at, deleted_at)
             VALUES (?1, 'memory', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(uid) DO UPDATE SET
               subkind=excluded.subkind, project_id=excluded.project_id,
               title=excluded.title, body=excluded.body,
               tags_text=excluded.tags_text, content_hash=excluded.content_hash,
               meta=excluded.meta, pinned=excluded.pinned,
               updated_at=excluded.updated_at, deleted_at=excluded.deleted_at",
            params![
                n.uid,
                mtype,
                project_id,
                n.title,
                n.body,
                tags_text,
                hash,
                n.meta.to_string(),
                n.pinned as i64,
                n.created_at,
                n.updated_at,
                n.deleted_at,
            ],
        )?;
        stats.nodes_upserted += 1;
        won.insert(incoming.uid.as_str());
        if n.deleted_at.is_some() {
            stats.tombstones += 1;
        }
        let Ok(node) = store::get_node_by_uid(&tx, &n.uid) else {
            continue;
        };
        let was = before.as_ref().and_then(|b| b.content_hash.as_deref());
        let is = node.content_hash.as_deref();
        let (op, reason, before_hash, after_hash) = if n.deleted_at.is_some() {
            // Archival never gets this far, so a tombstone here is a
            // deliberate forget, and it owes what a local one owes: prior
            // wordings go, else the deleted text survives on this peer where
            // `export` and backups read it.
            let _ = crate::revision::purge(&tx, node.id);
            (
                MutationOp::Forget,
                "tombstone arrived over sync",
                was.or(is),
                None,
            )
        } else {
            match &before {
                None => (MutationOp::Create, "arrived over sync", None, is),
                // The one path that clears a tombstone in place.
                Some(b) if b.tombstone => (
                    MutationOp::Restore,
                    "a newer live version arrived over sync",
                    was,
                    is,
                ),
                Some(_) => (
                    MutationOp::Edit,
                    "a newer version arrived over sync",
                    was,
                    is,
                ),
            }
        };
        // `Sync` is the honest actor for all four: the transport is all this
        // install can know — who acted, and why, lives in the ledger on the
        // machine it came from.
        // Fallible on purpose: this row is how a hub knows when the memory
        // arrived, and `relay_since` hands a peer only what has one. A row
        // applied without it would be exactly the late arrival that is
        // never relayed, so the batch fails as a whole and is retried.
        crate::audit::try_record_for_kind(
            &tx,
            node.kind,
            node.id,
            op,
            Actor::Sync,
            Some(reason),
            before_hash,
            after_hash,
        )?;
    }

    // Supersession, once every node of the batch is in: a record and its
    // replacement can arrive in either order within one batch.
    for n in &batch.nodes {
        if let Some(replacement) = n.superseded_by.as_deref() {
            merge_supersession(&tx, &n.uid, replacement, won.contains(n.uid.as_str()))?;
        }
    }

    for e in &batch.edges {
        let src = store::get_node_by_uid(&tx, &e.src_uid).ok();
        let dst = store::get_node_by_uid(&tx, &e.dst_uid).ok();
        match (src, dst, e.rel.parse::<Rel>()) {
            // A co-activation edge is an observation the receiving store makes
            // about its OWN opens, so an inbound one is a claim about a history
            // that never happened here. Rejected at the door rather than left
            // to the sender's filter: a peer on an older or unfiltered build
            // still ships them, and `store::link` would write the peer's weight
            // with no `meta.last_fired_at` — which reads back as never-decayed
            // and could push a fabricated partner over the visibility floor.
            // Counted as skipped, like a dangling endpoint: it is a row this
            // store declined, not an error that should fail the batch.
            (_, _, Ok(Rel::CoRecalled)) => stats.edges_skipped += 1,
            (Some(s), Some(d), Ok(rel)) => {
                store::link(&tx, s.id, d.id, rel, e.weight)?;
                stats.edges_upserted += 1;
            }
            _ => stats.edges_skipped += 1,
        }
    }
    tx.commit()?;
    Ok(stats)
}

/// Retire the local copy of `uid` in favour of the local copy of
/// `replacement_uid`, as the peer that sent the record already has.
///
/// The `superseded_by` column is what hides a memory from recall; the
/// `supersedes` edge is only a rendering of it. So the edge alone, which is
/// all the wire used to carry, left a memory superseded on one machine live
/// on every other — and nothing here infers the column from that edge.
///
/// There is no un-supersede, so the column is only ever set, never cleared:
///
/// - `won` (the record was the newer write): the column takes the record's
///   replacement, like every other synced field.
/// - not `won`: the column is filled in if it is empty — an edit made here
///   since must not undo a retirement made there — and left alone if this
///   store already names a replacement. Two machines that retired the same
///   memory in favour of different ones therefore settle on the newer
///   write's choice instead of trading choices on every sync.
///
/// Skipped when
///
/// - the replacement is not in this store: the memory stays live rather
///   than hidden behind something a reader cannot follow;
/// - the replacement is itself superseded, here, by a chain that leads back
///   to this memory: two machines that retired a pair in opposite
///   directions would otherwise leave both hidden on every store, with no
///   current memory at either end and no way to undo it. Each store keeps
///   the one it kept;
/// - the column already says so.
///
/// A pin does not hold it off. Locally a pinned memory is never superseded
/// (`store::supersede` refuses), but a retirement made on another machine
/// was made where the memory was not pinned, or before the pin arrived, and
/// it has to land on every store or the stores stop agreeing about whether
/// the memory is current. Retirement only ever spreads; the pin is then a
/// pin on a retired memory, on every store alike.
///
/// `updated_at` is left alone — it is the writer's clock and decides
/// last-write-wins. The ledger row dates the retirement *on this store*,
/// which is the date `recall --as-of` asks for; its actor `Sync` also marks
/// the row as an arrival for [`relay_since`].
fn merge_supersession(
    conn: &Connection,
    uid: &str,
    replacement_uid: &str,
    won: bool,
) -> Result<()> {
    let (Ok(node), Ok(replacement)) = (
        store::get_node_by_uid(conn, uid),
        store::get_node_by_uid(conn, replacement_uid),
    ) else {
        return Ok(());
    };
    if node.kind != Kind::Memory
        || replacement.kind != Kind::Memory
        || replacement.id == node.id
        || node.superseded_by == Some(replacement.id)
        || (node.superseded_by.is_some() && !won)
        || chain_reaches(conn, &replacement, node.id)
    {
        return Ok(());
    }
    conn.execute(
        "UPDATE node SET superseded_by = ?2 WHERE id = ?1",
        params![node.id, replacement.id],
    )?;
    crate::audit::try_record_for_kind(
        conn,
        node.kind,
        node.id,
        MutationOp::Supersede,
        Actor::Sync,
        Some("superseded on another machine"),
        node.content_hash.as_deref(),
        replacement.content_hash.as_deref(),
    )?;
    Ok(())
}

/// Does following `from`'s `superseded_by` chain arrive at `target`? Bounded
/// like `lifecycle::resolve_current`, so a cycle already in the store ends
/// the walk instead of spinning it.
fn chain_reaches(conn: &Connection, from: &crate::model::Node, target: i64) -> bool {
    let mut next = from.superseded_by;
    for _ in 0..8 {
        match next {
            None => return false,
            Some(id) if id == target => return true,
            Some(id) => next = store::get_node(conn, id).ok().and_then(|n| n.superseded_by),
        }
    }
    false
}

const META_PREFIX: &str = "sync.";

pub fn get_watermark(conn: &Connection, key: &str) -> Result<i64> {
    let v: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            [format!("{META_PREFIX}{key}")],
            |r| r.get(0),
        )
        .optional()?;
    Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
}

pub fn set_watermark(conn: &Connection, key: &str, value: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![format!("{META_PREFIX}{key}"), value.to_string()],
    )?;
    Ok(())
}

/// A fresh random bearer token (ULID; ~80 bits of entropy).
pub fn new_token() -> String {
    ulid::Ulid::new().to_string()
}

/// Read a `sync.<key>` string from the meta table.
pub fn get_str_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            [format!("{META_PREFIX}{key}")],
            |r| r.get(0),
        )
        .optional()?)
}

/// Write a `sync.<key>` string into the meta table.
pub fn set_str_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![format!("{META_PREFIX}{key}"), value],
    )?;
    Ok(())
}

/// Stable per-install id (for naming this machine's snapshot file). Generated
/// once and persisted in meta.
pub fn machine_id(conn: &Connection) -> Result<String> {
    if let Some(id) = conn
        .query_row(
            "SELECT value FROM meta WHERE key='sync.machine_id'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()?
    {
        return Ok(id);
    }
    let id = ulid::Ulid::new().to_string();
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('sync.machine_id', ?1)",
        [&id],
    )?;
    Ok(id)
}

fn to_hex(b: &[u8]) -> String {
    crate::format::hex(b)
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::{remember, Remember};
    use crate::model::{Kind, MemoryType};

    fn mem(conn: &Connection, text: &str) -> crate::model::Node {
        match remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec!["t".into()],
                project_id: None,
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            crate::memory::RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        }
    }

    fn synced_project(conn: &Connection, key: &str) -> i64 {
        let p = store::ensure_project(conn, &format!("/local/{key}"), "proj").unwrap();
        store::set_project_identity(conn, p.id, Some(key), true).unwrap();
        p.id
    }

    fn proj_mem(conn: &Connection, pid: i64, text: &str) -> crate::model::Node {
        match remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: vec!["t".into()],
                project_id: Some(pid),
                force: true,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            crate::memory::RememberOutcome::Created(n) => n,
            _ => unreachable!(),
        }
    }

    #[test]
    fn project_memory_syncs_via_key_and_creates_shadow() {
        let a = db::open_in_memory().unwrap();
        let pid = synced_project(&a, "git:github.com/o/r");
        let g = mem(&a, "global fact");
        let pm = proj_mem(&a, pid, "project fact");

        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 2, "global + synced-project memory");
        let pr = batch.nodes.iter().find(|r| r.uid == pm.uid).unwrap();
        assert_eq!(pr.project_key.as_deref(), Some("git:github.com/o/r"));
        let gr = batch.nodes.iter().find(|r| r.uid == g.uid).unwrap();
        assert_eq!(gr.project_key, None);

        // Fresh peer: a shadow project is created and the memory attaches to it;
        // the global memory stays global.
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &batch).unwrap();
        let shadow = store::find_project_by_key(&b, "git:github.com/o/r")
            .unwrap()
            .expect("shadow project created");
        let attached: Option<i64> = b
            .query_row("SELECT project_id FROM node WHERE uid=?1", [&pm.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(attached, Some(shadow.id));
        let gpid: Option<i64> = b
            .query_row("SELECT project_id FROM node WHERE uid=?1", [&g.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(gpid, None, "global stays global on the peer");
    }

    #[test]
    fn unsynced_project_memory_stays_local() {
        let a = db::open_in_memory().unwrap();
        let p = store::ensure_project(&a, "/local/x", "x").unwrap();
        // Has a key but sync=false → excluded from replication.
        store::set_project_identity(&a, p.id, Some("git:github.com/o/r"), false).unwrap();
        proj_mem(&a, p.id, "private fact");
        mem(&a, "global fact");
        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 1, "only the global memory syncs");
        assert_eq!(batch.nodes[0].project_key, None);
    }

    #[test]
    fn record_without_project_key_deserializes_as_global() {
        // An old global-only peer omits the field entirely.
        let json = r#"{"uid":"01ABC","type":"note","title":null,"body":"x","tags":[],
            "pinned":false,"content_hash":null,"meta":{},"created_at":1,"updated_at":1,
            "deleted_at":null}"#;
        let rec: SyncRecord = serde_json::from_str(json).unwrap();
        assert_eq!(rec.project_key, None);
    }

    #[test]
    fn roundtrip_and_idempotent() {
        let a = db::open_in_memory().unwrap();
        mem(&a, "alpha fact");
        mem(&a, "beta fact");
        let batch = snapshot(&a).unwrap();
        assert_eq!(batch.nodes.len(), 2);

        let b = db::open_in_memory().unwrap();
        let s1 = apply_changes(&b, &batch).unwrap();
        assert_eq!(s1.nodes_upserted, 2);
        // Second apply is a no-op (last-write-wins guard).
        let s2 = apply_changes(&b, &batch).unwrap();
        assert_eq!(s2.nodes_upserted, 0);
        assert_eq!(s2.nodes_skipped, 2);
        assert_eq!(changes_since(&b, i64::MIN).unwrap().nodes.len(), 2);
    }

    #[test]
    fn last_write_wins_both_directions() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "shared fact");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        // Newer edit on A wins on B.
        a.execute(
            "UPDATE node SET body='edited', updated_at=updated_at+100 WHERE uid=?1",
            [&n.uid],
        )
        .unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        let body: String = b
            .query_row("SELECT body FROM node WHERE uid=?1", [&n.uid], |r| r.get(0))
            .unwrap();
        assert_eq!(body, "edited");

        // A stale older edit must NOT overwrite.
        let stale = SyncBatch {
            nodes: vec![SyncRecord {
                uid: n.uid.clone(),
                mtype: Some("note".into()),
                title: Some("x".into()),
                body: Some("STALE".into()),
                tags: vec![],
                pinned: false,
                content_hash: None,
                meta: serde_json::json!({}),
                created_at: n.created_at,
                updated_at: 1,
                deleted_at: None,
                project_key: None,
                superseded_by: None,
            }],
            edges: vec![],
            watermark: 0,
        };
        let s = apply_changes(&b, &stale).unwrap();
        assert_eq!(s.nodes_skipped, 1);
        let body: String = b
            .query_row("SELECT body FROM node WHERE uid=?1", [&n.uid], |r| r.get(0))
            .unwrap();
        assert_eq!(body, "edited", "stale write rejected");
    }

    #[test]
    fn tombstone_propagates_and_sticks() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "doomed fact");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        store::soft_delete(&a, n.id, crate::model::Actor::System, None).unwrap();
        let batch = snapshot(&a).unwrap();
        assert!(batch
            .nodes
            .iter()
            .any(|r| r.uid == n.uid && r.deleted_at.is_some()));
        let s = apply_changes(&b, &batch).unwrap();
        assert_eq!(s.tombstones, 1);
        let del: Option<i64> = b
            .query_row("SELECT deleted_at FROM node WHERE uid=?1", [&n.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(del.is_some(), "deleted on peer");

        // A stale pre-delete snapshot must not resurrect it.
        let resurrect = SyncBatch {
            nodes: vec![SyncRecord {
                uid: n.uid.clone(),
                mtype: Some("note".into()),
                title: None,
                body: Some("back".into()),
                tags: vec![],
                pinned: false,
                content_hash: None,
                meta: serde_json::json!({}),
                created_at: n.created_at,
                updated_at: n.updated_at,
                deleted_at: None,
                project_key: None,
                superseded_by: None,
            }],
            edges: vec![],
            watermark: 0,
        };
        apply_changes(&b, &resurrect).unwrap();
        let del: Option<i64> = b
            .query_row("SELECT deleted_at FROM node WHERE uid=?1", [&n.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(del.is_some(), "stale write did not resurrect");
    }

    /// A deliberate forget that arrives over sync is the same transition as a
    /// local one, so it owes the same side effects: it used to purge nothing,
    /// and a forgotten memory kept its earlier wordings on every peer that
    /// had edited it, where `export` and backups still read them. Decay
    /// archival is the other kind of tombstone and owes the opposite — it
    /// does not arrive at all.
    #[test]
    fn a_synced_tombstone_has_the_side_effects_of_a_local_one() {
        use crate::model::{Actor, MutationOp};
        let a = db::open_in_memory().unwrap();
        let forgotten = mem(&a, "the staging password hint");
        let archived = mem(&a, "an idle note nobody opens");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        // The peer holds an earlier wording of each.
        let on_b = |uid: &str| store::get_node_by_uid(&b, uid).unwrap();
        for uid in [&forgotten.uid, &archived.uid] {
            crate::revision::snapshot(&b, &on_b(uid));
            assert_eq!(crate::revision::history(&b, on_b(uid).id).unwrap().len(), 1);
        }

        store::soft_delete(&a, forgotten.id, Actor::Human, None).unwrap();
        archive(&a, archived.id);
        let stats = apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(stats.tombstones, 1, "only the forget is a tombstone here");

        // Forgotten on purpose: prior wordings go, as they do locally.
        assert_eq!(
            ledger(&b, &forgotten.uid).last(),
            Some(&(MutationOp::Forget, Actor::Sync))
        );
        assert!(
            crate::revision::history(&b, on_b(&forgotten.uid).id)
                .unwrap()
                .is_empty(),
            "a deliberate forget purges prior wordings on the peer too"
        );
        // Archived by decay on A: nothing happened to it here, so the row,
        // its wordings and its ledger are as they were.
        assert_eq!(
            lifecycle_of(&b, &archived.uid),
            crate::lifecycle::Lifecycle::Live
        );
        assert_eq!(
            ledger(&b, &archived.uid),
            vec![(MutationOp::Create, Actor::Sync)],
            "the arrival, and nothing since"
        );
        assert_eq!(
            crate::revision::history(&b, on_b(&archived.uid).id)
                .unwrap()
                .len(),
            1,
            "decay archival on a peer leaves prior wordings alone"
        );
    }

    /// The statement `consolidate`'s decay pass runs.
    fn archive(conn: &Connection, id: i64) {
        conn.execute(
            "UPDATE node SET deleted_at = ?2, meta = json_set(meta, '$.archived', 1)
             WHERE id = ?1",
            params![id, crate::model::now_unix()],
        )
        .unwrap();
    }

    fn lifecycle_of(conn: &Connection, uid: &str) -> crate::lifecycle::Lifecycle {
        store::get_node_by_uid(conn, uid)
            .unwrap()
            .lifecycle(crate::model::now_unix())
    }

    fn ledger(
        conn: &Connection,
        uid: &str,
    ) -> Vec<(crate::model::MutationOp, crate::model::Actor)> {
        let id = store::get_node_by_uid(conn, uid).unwrap().id;
        crate::audit::history(conn, id)
            .unwrap()
            .into_iter()
            .map(|m| (m.op, m.actor))
            .collect()
    }

    /// Decay is computed from strength and access counts, which never leave
    /// the machine — so the archival it causes must not leave it either. A
    /// note that sits idle on the laptop is still the one the desktop opens
    /// every day.
    #[test]
    fn archiving_stays_on_the_machine_that_decayed() {
        use crate::lifecycle::Lifecycle;
        let a = db::open_in_memory().unwrap();
        let x = mem(&a, "the runbook for the quarterly failover drill");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        archive(&a, x.id);
        let from_a = snapshot(&a).unwrap();
        let on_wire = from_a.nodes.iter().find(|r| r.uid == x.uid).unwrap();
        assert_eq!(on_wire.deleted_at, None, "an archived row travels live");
        assert!(on_wire.meta.get("archived").is_none(), "{}", on_wire.meta);

        let stats = apply_changes(&b, &from_a).unwrap();
        assert_eq!(stats.tombstones, 0);
        assert_eq!(lifecycle_of(&b, &x.uid), Lifecycle::Live);

        // B's copy never changed, so it has nothing to say to A.
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        assert_eq!(lifecycle_of(&a, &x.uid), Lifecycle::Archived);
    }

    /// A peer on a build that still ships archival as a tombstone: the flag
    /// in its meta says nobody decided this, so it deletes nothing here.
    #[test]
    fn an_archived_tombstone_from_an_older_peer_deletes_nothing() {
        use crate::lifecycle::Lifecycle;
        let a = db::open_in_memory().unwrap();
        let held = mem(&a, "a note this machine still uses");
        let unseen = mem(&a, "a note this machine never had");
        let b = db::open_in_memory().unwrap();
        let mut first = snapshot(&a).unwrap();
        first.nodes.retain(|r| r.uid == held.uid);
        apply_changes(&b, &first).unwrap();

        // What 0.17.1 puts on the wire for an archived row.
        let mut old_wire = snapshot(&a).unwrap();
        for r in &mut old_wire.nodes {
            r.deleted_at = Some(r.updated_at + 500);
            r.meta["archived"] = serde_json::json!(1);
        }
        let stats = apply_changes(&b, &old_wire).unwrap();
        assert_eq!(stats.tombstones, 0, "{stats:?}");
        assert_eq!(lifecycle_of(&b, &held.uid), Lifecycle::Live);
        assert_eq!(
            lifecycle_of(&b, &unseen.uid),
            Lifecycle::Live,
            "a row first seen archived arrives as the live memory it was"
        );
        let stored = store::get_node_by_uid(&b, &unseen.uid).unwrap();
        assert!(stored.meta.get("archived").is_none(), "{}", stored.meta);
    }

    /// The arrival of a row is the one thing its own timestamps cannot say,
    /// and the only thing a hub can relay by. Every applied row leaves a
    /// ledger row whose op names what the arrival did here.
    #[test]
    fn every_applied_row_is_recorded_with_what_it_did() {
        use crate::model::{Actor, MutationOp};
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "the cache key includes the tenant id");
        let b = db::open_in_memory().unwrap();

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(ledger(&b, &n.uid), vec![(MutationOp::Create, Actor::Sync)]);

        // Not newer: nothing happened here, so nothing is recorded.
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(ledger(&b, &n.uid).len(), 1);

        a.execute(
            "UPDATE node SET body='edited', updated_at=updated_at+100 WHERE uid=?1",
            [&n.uid],
        )
        .unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(
            ledger(&b, &n.uid).last(),
            Some(&(MutationOp::Edit, Actor::Sync))
        );

        a.execute(
            "UPDATE node SET deleted_at=updated_at+100, updated_at=updated_at+100 WHERE uid=?1",
            [&n.uid],
        )
        .unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(
            ledger(&b, &n.uid).last(),
            Some(&(MutationOp::Forget, Actor::Sync))
        );

        // A strictly newer rewrite is the one thing that clears a tombstone
        // in place, and the ledger has to say it happened. (A rewrite: new
        // text, so a new content hash. A newer write that leaves the text
        // alone does not undo a forget; see
        // `a_pin_does_not_bring_back_what_another_machine_forgot`.)
        a.execute(
            "UPDATE node SET deleted_at=NULL, body='back', content_hash=x'0badc0de',
                    updated_at=updated_at+100 WHERE uid=?1",
            [&n.uid],
        )
        .unwrap();
        let stats = apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(stats.nodes_upserted, 1);
        assert_eq!(
            ledger(&b, &n.uid).last(),
            Some(&(MutationOp::Restore, Actor::Sync))
        );
        assert_eq!(lifecycle_of(&b, &n.uid), crate::lifecycle::Lifecycle::Live);
    }

    /// The pull cursor `mimir sync` keeps (`cursor_advance` in the CLI): the
    /// batch watermark, clamped to the local clock, minus a minute of grace.
    fn pull_cursor(watermark: i64) -> i64 {
        (watermark.min(crate::model::now_unix()) - 60).max(0)
    }

    /// Move a memory's own timestamps into the past, as if it had been
    /// written `secs` ago.
    fn written_ago(conn: &Connection, uid: &str, secs: i64) {
        conn.execute(
            "UPDATE node SET created_at = created_at - ?2, updated_at = updated_at - ?2
             WHERE uid = ?1",
            params![uid, secs],
        )
        .unwrap();
    }

    /// Move the arrival of a memory at this store into the past.
    fn arrived_ago(conn: &Connection, uid: &str, secs: i64) {
        let changed = conn
            .execute(
                "UPDATE mutation SET at = at - ?2
                 WHERE actor = 'sync' AND node_id = (SELECT id FROM node WHERE uid = ?1)",
                params![uid, secs],
            )
            .unwrap();
        assert!(changed > 0, "no arrival recorded for {uid}");
    }

    fn uids(batch: &SyncBatch) -> Vec<&str> {
        batch.nodes.iter().map(|r| r.uid.as_str()).collect()
    }

    /// Three stores: A and B are clients, the hub relays. A memory A wrote an
    /// hour ago and pushes only now carries a timestamp below the cursor of
    /// a B that pulled a minute ago. Cutting the pull by the row's own
    /// timestamp lost it for good; cutting by its arrival does not.
    #[test]
    fn a_hub_relays_by_arrival_not_by_when_a_row_was_written() {
        let (a, hub, b) = (
            db::open_in_memory().unwrap(),
            db::open_in_memory().unwrap(),
            db::open_in_memory().unwrap(),
        );

        // B is in regular contact: its own note reached the hub, and its
        // pull cursor is up to the present.
        let own = mem(&b, "b's note about couplings");
        apply_changes(&hub, &changes_since(&b, 0).unwrap()).unwrap();
        let first = relay_since(&hub, 0).unwrap();
        apply_changes(&b, &first).unwrap();
        let cursor = pull_cursor(first.watermark);

        // A wrote this an hour ago and was offline until now.
        let late = mem(&a, "a's note about flywheels");
        written_ago(&a, &late.uid, 3600);
        let stats = apply_changes(&hub, &changes_since(&a, 0).unwrap()).unwrap();
        assert_eq!(stats.nodes_upserted, 1);

        // The hub kept A's timestamp (it decides last-write-wins) …
        let on_hub = store::get_node_by_uid(&hub, &late.uid).unwrap();
        assert!(on_hub.updated_at < cursor);
        // … so the row is invisible to a cut by timestamp …
        assert!(!uids(&changes_since(&hub, cursor).unwrap()).contains(&late.uid.as_str()));
        // … and visible to a cut by arrival.
        let next = relay_since(&hub, cursor).unwrap();
        assert!(
            uids(&next).contains(&late.uid.as_str()),
            "{:?}",
            uids(&next)
        );
        let stats = apply_changes(&b, &next).unwrap();
        assert_eq!(stats.nodes_upserted, 1, "only the late row is new to B");
        assert!(store::get_node_by_uid(&b, &late.uid).is_ok());

        // An arrival older than the cursor is not relayed again: a pull is
        // still a diff, not a dump.
        arrived_ago(&hub, &late.uid, 7200);
        written_ago(&hub, &own.uid, 7200);
        arrived_ago(&hub, &own.uid, 7200);
        assert!(relay_since(&hub, cursor).unwrap().nodes.is_empty());
    }

    /// The watermark a hub hands out is its own clock, so the cursor a peer
    /// derives from it is comparable with arrival times. A future-dated row
    /// (a peer with a wrong clock) inflates the newest-timestamp watermark by
    /// years; it must not move this one.
    #[test]
    fn a_relay_watermark_is_the_hubs_clock() {
        let hub = db::open_in_memory().unwrap();
        let skewed = mem(&hub, "written by a peer whose clock is ten years fast");
        let ten_years = 10 * 365 * 86_400;
        written_ago(&hub, &skewed.uid, -ten_years);

        let before = crate::model::now_unix();
        let relayed = relay_since(&hub, 0).unwrap();
        let after = crate::model::now_unix();
        assert!((before..=after).contains(&relayed.watermark));
        assert!(changes_since(&hub, 0).unwrap().watermark > after + ten_years / 2);
        assert_eq!(uids(&relayed), vec![skewed.uid.as_str()]);
    }

    /// A client records arrivals too (same code path), and must not take
    /// them for changes of its own: what it pushes is cut by what was
    /// written, so a pulled row with an old timestamp is not sent back. A
    /// row it does send back — one written inside its push cursor's grace
    /// window — changes nothing where it lands and records nothing.
    #[test]
    fn a_pulled_row_is_not_pushed_back_as_a_change() {
        let (a, hub, b) = (
            db::open_in_memory().unwrap(),
            db::open_in_memory().unwrap(),
            db::open_in_memory().unwrap(),
        );
        let old = mem(&a, "written long before b's last push");
        written_ago(&a, &old.uid, 3600);
        let fresh = mem(&a, "written a moment ago");
        apply_changes(&hub, &changes_since(&a, 0).unwrap()).unwrap();
        apply_changes(&b, &relay_since(&hub, 0).unwrap()).unwrap();
        assert_eq!(
            ledger(&b, &old.uid),
            vec![(crate::model::MutationOp::Create, crate::model::Actor::Sync)]
        );

        // B's push cursor is a minute behind the present.
        let echo = changes_since(&b, crate::model::now_unix() - 60).unwrap();
        assert_eq!(uids(&echo), vec![fresh.uid.as_str()]);
        let hub_rows = |uid: &str| ledger(&hub, uid).len();
        let (old_rows, fresh_rows) = (hub_rows(&old.uid), hub_rows(&fresh.uid));
        let stats = apply_changes(&hub, &echo).unwrap();
        assert_eq!((stats.nodes_upserted, stats.nodes_skipped), (0, 1));
        assert_eq!(
            (hub_rows(&old.uid), hub_rows(&fresh.uid)),
            (old_rows, fresh_rows),
            "an echo is not an arrival"
        );
    }

    /// What every local supersede does: the column, and the edge that
    /// renders it.
    fn supersede(conn: &Connection, old: &crate::model::Node, new: &crate::model::Node) {
        store::set_superseded(conn, old.id, new.id, crate::model::Actor::Human, None).unwrap();
        store::link(conn, new.id, old.id, Rel::Supersedes, 1.0).unwrap();
    }

    /// The uid of the memory that replaced `uid` in this store, if any.
    fn replaced_by(conn: &Connection, uid: &str) -> Option<String> {
        conn.query_row(
            "SELECT r.uid FROM node m LEFT JOIN node r ON r.id = m.superseded_by
             WHERE m.uid = ?1",
            [uid],
            |r| r.get(0),
        )
        .unwrap()
    }

    /// Two machines holding the same two memories, `old` and `new`.
    fn two_stores_sharing_a_pair() -> (
        Connection,
        Connection,
        crate::model::Node,
        crate::model::Node,
    ) {
        let a = db::open_in_memory().unwrap();
        let old = mem(&a, "deploys go out on fridays");
        let new = mem(&a, "deploys go out on tuesdays, never fridays");
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        (a, b, old, new)
    }

    /// The wire used to carry the `supersedes` edge and not the column, so a
    /// memory superseded on one machine stayed live — and kept being
    /// recalled — on every other.
    #[test]
    fn supersession_travels() {
        use crate::lifecycle::Lifecycle;
        use crate::model::{Actor, MutationOp};
        let (a, b, old, new) = two_stores_sharing_a_pair();
        supersede(&a, &old, &new);

        let batch = snapshot(&a).unwrap();
        let record = |uid: &str| batch.nodes.iter().find(|r| r.uid == uid).unwrap();
        assert_eq!(record(&old.uid).superseded_by.as_deref(), Some(&*new.uid));
        // Absent, not null, when there is nothing to say: older peers and
        // hubs see the record shape they always saw.
        let json = serde_json::to_string(record(&new.uid)).unwrap();
        assert!(!json.contains("superseded_by"), "{json}");

        apply_changes(&b, &batch).unwrap();
        assert_eq!(replaced_by(&b, &old.uid).as_deref(), Some(&*new.uid));
        assert_eq!(lifecycle_of(&b, &old.uid), Lifecycle::Superseded);
        assert_eq!(lifecycle_of(&b, &new.uid), Lifecycle::Live);
        // Dated on arrival, so `recall --as-of` has an answer here too.
        assert_eq!(
            ledger(&b, &old.uid).last(),
            Some(&(MutationOp::Supersede, Actor::Sync))
        );
        let old_on_b = store::get_node_by_uid(&b, &old.uid).unwrap();
        assert!(crate::audit::superseded_at(&b, old_on_b.id)
            .unwrap()
            .is_some());

        // Applying it again changes nothing and records nothing.
        let rows = ledger(&b, &old.uid).len();
        apply_changes(&b, &batch).unwrap();
        assert_eq!(ledger(&b, &old.uid).len(), rows);
    }

    /// There is no un-supersede, so the column merges as a set-once fact and
    /// not by last-write-wins: an edit made here since must not undo a
    /// retirement made there, and a record that does not mention it (every
    /// older peer's) must not clear it.
    #[test]
    fn supersession_merges_whichever_side_is_newer_and_is_never_cleared() {
        let (a, b, old, new) = two_stores_sharing_a_pair();
        supersede(&a, &old, &new);
        // B reworded the old memory after A retired it.
        b.execute(
            "UPDATE node SET body = 'deploys went out on fridays', updated_at = updated_at + 1000
             WHERE uid = ?1",
            [&old.uid],
        )
        .unwrap();

        let stats = apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(stats.nodes_upserted, 0, "A's rows are not newer than B's");
        assert_eq!(replaced_by(&b, &old.uid).as_deref(), Some(&*new.uid));
        let body = |conn: &Connection| store::get_node_by_uid(conn, &old.uid).unwrap().body;
        assert_eq!(body(&b).as_deref(), Some("deploys went out on fridays"));

        // B's newer wording reaches A the way an older peer would send it:
        // without the field.
        let mut from_b = snapshot(&b).unwrap();
        for r in &mut from_b.nodes {
            r.superseded_by = None;
        }
        apply_changes(&a, &from_b).unwrap();
        assert_eq!(body(&a).as_deref(), Some("deploys went out on fridays"));
        assert_eq!(replaced_by(&a, &old.uid).as_deref(), Some(&*new.uid));
    }

    /// Two machines retire the same memory in favour of different ones. The
    /// newer write's choice wins on both, and repeating the exchange changes
    /// nothing — the column must not be traded back and forth on every sync.
    #[test]
    fn rival_supersessions_settle_on_the_newer_write() {
        let (a, b, old, new) = two_stores_sharing_a_pair();
        let other = mem(&b, "deploys go out whenever the queue is green");
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();

        supersede(&a, &old, &new);
        let on_b = |uid: &str| store::get_node_by_uid(&b, uid).unwrap();
        supersede(&b, &on_b(&old.uid), &on_b(&other.uid));
        b.execute(
            "UPDATE node SET updated_at = updated_at + 100 WHERE uid = ?1",
            [&old.uid],
        )
        .unwrap();

        // Both snapshots are taken before either is applied: each machine
        // hears the other's choice as it stood, not as already settled.
        let exchange = || {
            let (from_a, from_b) = (snapshot(&a).unwrap(), snapshot(&b).unwrap());
            apply_changes(&a, &from_b).unwrap();
            apply_changes(&b, &from_a).unwrap();
        };
        exchange();
        assert_eq!(replaced_by(&a, &old.uid).as_deref(), Some(&*other.uid));
        assert_eq!(replaced_by(&b, &old.uid).as_deref(), Some(&*other.uid));
        let rows = (ledger(&a, &old.uid).len(), ledger(&b, &old.uid).len());
        exchange();
        exchange();
        assert_eq!(
            (ledger(&a, &old.uid).len(), ledger(&b, &old.uid).len()),
            rows,
            "a settled supersession is not rewritten"
        );
    }

    /// A memory is not hidden behind a replacement this store cannot show.
    /// The retirement lands when a batch carries both: the replacement, in
    /// any order, and the retired record again. A replacement arriving on
    /// its own later does not retire anything by itself.
    #[test]
    fn a_replacement_this_store_does_not_hold_leaves_the_memory_live() {
        use crate::lifecycle::Lifecycle;
        let a = db::open_in_memory().unwrap();
        let old = mem(&a, "the api key rotates yearly");
        let new = mem(&a, "the api key rotates every ninety days");
        supersede(&a, &old, &new);
        let full = snapshot(&a).unwrap();

        let b = db::open_in_memory().unwrap();
        let mut without_replacement = full.clone();
        without_replacement.nodes.retain(|r| r.uid == old.uid);
        apply_changes(&b, &without_replacement).unwrap();
        assert_eq!(lifecycle_of(&b, &old.uid), Lifecycle::Live);
        assert_eq!(replaced_by(&b, &old.uid), None);

        // The superseded record first, its replacement after it.
        let mut reordered = full.clone();
        reordered.nodes.sort_by_key(|r| r.uid != old.uid);
        assert_eq!(reordered.nodes[0].uid, old.uid);
        apply_changes(&b, &reordered).unwrap();
        assert_eq!(lifecycle_of(&b, &old.uid), Lifecycle::Superseded);
    }

    /// The edge is a rendering of the column, not a second source for it: a
    /// peer older than the field ships the edge alone, and that retires
    /// nothing. (`mimir link --rel supersedes` is refused locally for the
    /// same reason.)
    #[test]
    fn a_bare_supersedes_edge_retires_nothing() {
        let (a, b, old, new) = two_stores_sharing_a_pair();
        supersede(&a, &old, &new);
        let mut from_older_peer = snapshot(&a).unwrap();
        for r in &mut from_older_peer.nodes {
            r.superseded_by = None;
        }
        assert!(from_older_peer.edges.iter().any(|e| e.rel == "supersedes"));

        let stats = apply_changes(&b, &from_older_peer).unwrap();
        assert_eq!(stats.edges_upserted, 1);
        assert_eq!(replaced_by(&b, &old.uid), None);
        assert_eq!(
            lifecycle_of(&b, &old.uid),
            crate::lifecycle::Lifecycle::Live
        );
    }

    /// A pin does not hold off a retirement made on another machine. It did
    /// for a while, and the stores then disagreed for good about whether the
    /// memory was current: superseded where it was retired, live where it was
    /// pinned. Retirement spreads; the pin spreads too, as the later write.
    #[test]
    fn a_pin_does_not_hold_off_a_supersession_made_elsewhere() {
        use crate::lifecycle::Lifecycle;
        let (a, b, old, new) = two_stores_sharing_a_pair();
        supersede(&a, &old, &new);
        // A retired it a while ago; B, not yet synced, pinned it since.
        a.execute(
            "UPDATE node SET updated_at = updated_at - 1000 WHERE uid = ?1",
            [&old.uid],
        )
        .unwrap();
        let on_b = store::get_node_by_uid(&b, &old.uid).unwrap();
        store::set_pinned(&b, on_b.id, true, crate::model::Actor::Human, None).unwrap();

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();

        for (name, store) in [("a", &a), ("b", &b)] {
            assert_eq!(
                replaced_by(store, &old.uid).as_deref(),
                Some(new.uid.as_str()),
                "store {name}"
            );
            assert_eq!(
                lifecycle_of(store, &old.uid),
                Lifecycle::Superseded,
                "store {name}"
            );
            assert!(
                store::get_node_by_uid(store, &old.uid).unwrap().pinned,
                "store {name} lost the pin"
            );
        }
    }

    /// A retirement that reaches the hub for a row the hub already holds a
    /// newer wording of changes no timestamp there. It is still an arrival,
    /// and peers that pulled in the meantime must be told.
    #[test]
    fn a_hub_relays_a_supersession_it_merged_into_a_newer_row() {
        let (a, hub, old, new) = two_stores_sharing_a_pair();
        for uid in [&old.uid, &new.uid] {
            written_ago(&a, uid, 86_400);
            written_ago(&hub, uid, 86_400);
            arrived_ago(&hub, uid, 86_400);
        }
        // Another peer's rewording reached the hub an hour ago.
        hub.execute(
            "UPDATE node SET body = 'reworded elsewhere', updated_at = ?2 WHERE uid = ?1",
            params![old.uid, crate::model::now_unix() - 3600],
        )
        .unwrap();
        // A retired the memory two hours ago, offline, and pushes only now.
        supersede(&a, &old, &new);
        a.execute(
            "UPDATE node SET updated_at = ?2 WHERE uid = ?1",
            params![old.uid, crate::model::now_unix() - 7200],
        )
        .unwrap();
        let cursor = crate::model::now_unix() - 60;
        assert!(relay_since(&hub, cursor).unwrap().nodes.is_empty());

        let stats = apply_changes(&hub, &changes_since(&a, cursor - 86_400).unwrap()).unwrap();
        assert_eq!(stats.nodes_upserted, 0, "the hub's wording is newer");
        let relayed = relay_since(&hub, cursor).unwrap();
        assert_eq!(uids(&relayed), vec![old.uid.as_str()]);
        assert_eq!(relayed.nodes[0].superseded_by.as_deref(), Some(&*new.uid));
        assert_eq!(relayed.nodes[0].body.as_deref(), Some("reworded elsewhere"));
    }

    #[test]
    fn a_record_from_an_older_peer_says_nothing_about_supersession() {
        // The record shape 0.17.1 sends: no `superseded_by` key at all.
        let json = r#"{"uid":"01ABC","type":"note","title":null,"body":"x","tags":[],
            "pinned":false,"content_hash":null,"meta":{},"created_at":1,"updated_at":1,
            "deleted_at":null,"project_key":"git:github.com/o/r"}"#;
        let rec: SyncRecord = serde_json::from_str(json).unwrap();
        assert_eq!(rec.superseded_by, None);
    }

    /// What "this project replicates" means is asked in SQL (which rows and
    /// edges to ship) and in Rust (`reproject`'s warning). Hold both to one
    /// answer over every meta shape a project row can carry.
    #[test]
    fn syncable_has_one_answer_in_sql_and_rust() {
        let metas = [
            r#"{"sync":true,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":1,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":1.0,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":true,"portable_key":null}"#,
            r#"{"sync":true}"#,
            r#"{"sync":false,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":0,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":2,"portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":"1","portable_key":"git:github.com/o/r"}"#,
            r#"{"sync":null,"portable_key":"git:github.com/o/r"}"#,
            r#"{"portable_key":"git:github.com/o/r"}"#,
            r#"{}"#,
        ];
        for meta in metas {
            let conn = db::open_in_memory().unwrap();
            let p = store::ensure_project(&conn, "/local/p", "p").unwrap();
            conn.execute(
                "UPDATE node SET meta = ?2 WHERE id = ?1",
                params![p.id, meta],
            )
            .unwrap();
            let inside = proj_mem(&conn, p.id, "a fact of the project");
            let global = mem(&conn, "a fact of no project");
            store::link(&conn, inside.id, global.id, Rel::Relates, 1.0).unwrap();

            let rust = project_is_syncable(&serde_json::from_str(meta).unwrap());
            let batch = snapshot(&conn).unwrap();
            let node_ships = batch.nodes.iter().any(|r| r.uid == inside.uid);
            assert_eq!(node_ships, rust, "node selection vs Rust for {meta}");
            assert_eq!(
                batch.edges.len() == 1,
                rust,
                "edge selection vs Rust for {meta}"
            );
        }
        assert!(project_is_syncable(&serde_json::json!(
            {"sync": 1, "portable_key": "k"}
        )));
        assert!(!project_is_syncable(&serde_json::json!(
            {"sync": true, "portable_key": null}
        )));
    }

    /// A batch's watermark describes exactly the rows the batch carries. With
    /// the three reads on separate snapshots, a commit landing between them
    /// raised the watermark over a row the batch did not hold.
    #[test]
    fn a_batch_is_one_consistent_read() {
        // Far above any real clock, so the writer's rows are the newest.
        const BASE: i64 = 4_000_000_000;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        let reader = db::open(&path).unwrap();

        // A bounded burst of single-statement commits from a second
        // connection, racing the reads below.
        let writer = {
            let path = path.clone();
            std::thread::spawn(move || {
                let conn = db::open(&path).unwrap();
                for i in 1..=1500 {
                    conn.execute(
                        "INSERT INTO node (uid, kind, body, created_at, updated_at)
                         VALUES (?1, 'memory', 'written meanwhile', ?2, ?2)",
                        params![ulid::Ulid::new().to_string(), BASE + i],
                    )
                    .unwrap();
                }
            })
        };

        let (mut reads, mut torn) = (0, 0);
        while !writer.is_finished() || reads == 0 {
            let batch = changes_since(&reader, BASE).unwrap();
            let newest = batch.nodes.iter().map(|r| r.updated_at).max().unwrap_or(0);
            reads += 1;
            if batch.watermark != newest {
                torn += 1;
            }
        }
        writer.join().unwrap();
        assert_eq!(
            torn, 0,
            "{torn} of {reads} batches had a watermark from a later snapshot than their rows"
        );
        assert!(reader.is_autocommit(), "the read snapshot is released");
    }

    #[test]
    fn edge_skipped_until_endpoints_present() {
        let a = db::open_in_memory().unwrap();
        let x = mem(&a, "node x");
        let y = mem(&a, "node y");
        store::link(&a, x.id, y.id, Rel::Relates, 1.0).unwrap();

        // Deliver only the edge + node x → edge can't resolve y, skipped.
        let b = db::open_in_memory().unwrap();
        let full = snapshot(&a).unwrap();
        let partial = SyncBatch {
            nodes: full
                .nodes
                .iter()
                .filter(|r| r.uid == x.uid)
                .cloned()
                .collect(),
            edges: full.edges.clone(),
            watermark: full.watermark,
        };
        let s = apply_changes(&b, &partial).unwrap();
        assert_eq!(s.edges_skipped, 1);

        // Now deliver everything → edge resolves.
        let s = apply_changes(&b, &full).unwrap();
        assert_eq!(s.edges_upserted, 1);
    }

    #[test]
    fn coactivation_edge_stays_local() {
        let a = db::open_in_memory().unwrap();
        let x = mem(&a, "node x");
        let y = mem(&a, "node y");
        // Control: an authored edge between a DIFFERENT pair, which must still
        // ship. Without it, a bug that dropped every edge would pass this test.
        // It has to be another pair — `record_coactivation` never fires for a
        // pair that already has an authored edge, so the two cannot coexist.
        let p = mem(&a, "node p");
        let q = mem(&a, "node q");
        store::link(&a, p.id, q.id, Rel::Relates, 1.0).unwrap();

        // Earn the co_recalled edge the way the store does — two opens inside
        // the window — rather than hand-writing the row, so this test also
        // fails if the real writer ever starts producing something else.
        crate::learn::record_opened(&a, x.id).unwrap();
        crate::learn::record_opened(&a, y.id).unwrap();
        let local: i64 = a
            .query_row(
                "SELECT count(*) FROM edge WHERE rel='co_recalled'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(local, 1, "co-activation edge exists locally to begin with");

        let batch = snapshot(&a).unwrap();
        let rels: Vec<&str> = batch.edges.iter().map(|e| e.rel.as_str()).collect();
        assert!(
            !rels.contains(&"co_recalled"),
            "derived edge must not ship: {rels:?}"
        );
        assert_eq!(rels, vec!["relates"], "authored edge still ships: {rels:?}");
    }

    #[test]
    fn inbound_coactivation_edge_is_rejected() {
        let a = db::open_in_memory().unwrap();
        let x = mem(&a, "node x");
        let y = mem(&a, "node y");
        let mut batch = snapshot(&a).unwrap();
        // A peer running an unfiltered (older or newer) build: its nodes are
        // fine, its co-activation edge is not ours to hold.
        batch.edges.push(SyncEdge {
            src_uid: x.uid.clone(),
            dst_uid: y.uid.clone(),
            rel: "co_recalled".into(),
            weight: 0.9,
            created_at: 1,
        });

        let b = db::open_in_memory().unwrap();
        let s = apply_changes(&b, &batch).unwrap();
        assert_eq!(s.edges_upserted, 0, "{s:?}");
        assert_eq!(s.edges_skipped, 1, "{s:?}");
        let landed: i64 = b
            .query_row(
                "SELECT count(*) FROM edge WHERE rel='co_recalled'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(landed, 0, "peer's co-activation edge must not land");
    }

    #[test]
    fn soft_delete_bumps_updated_at() {
        let conn = db::open_in_memory().unwrap();
        let n = {
            let mut new = crate::model::NewNode::new(Kind::Memory);
            new.title = Some("x".into());
            new.body = Some("x".into());
            store::insert_node(&conn, new).unwrap()
        };
        let before: i64 = conn
            .query_row("SELECT updated_at FROM node WHERE id=?1", [n.id], |r| {
                r.get(0)
            })
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        store::soft_delete(&conn, n.id, crate::model::Actor::System, None).unwrap();
        let after: i64 = conn
            .query_row("SELECT updated_at FROM node WHERE id=?1", [n.id], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(after > before, "soft_delete bumped updated_at");
    }

    #[test]
    fn unknown_wire_type_coerces_to_note() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "typed elsewhere");
        let mut batch = snapshot(&a).unwrap();
        batch.nodes[0].mtype = Some("<img src=x onerror=alert(1)>".into());

        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &batch).unwrap();
        let subkind: Option<String> = b
            .query_row("SELECT subkind FROM node WHERE uid=?1", [&n.uid], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(
            subkind.is_none(),
            "hostile wire type must not land verbatim (got {subkind:?})"
        );
    }

    /// The decay pass stamps `deleted_at` with the moment it ran. That stamp
    /// is local housekeeping; compared as if it were a write, it outranked a
    /// forget another machine had made earlier, which was then skipped here
    /// and never relayed, leaving the memory live on every third machine.
    #[test]
    fn a_forget_older_than_a_local_archive_still_lands() {
        use crate::lifecycle::Lifecycle;
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "the build cache lives on the scratch volume");
        written_ago(&a, &m.uid, 100 * 86_400);
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        // B forgot it an hour ago; A's decay pass archived it just now.
        let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
        store::soft_delete(&b, on_b.id, crate::model::Actor::Human, None).unwrap();
        b.execute(
            "UPDATE node SET deleted_at = deleted_at - 3600, updated_at = updated_at - 3600
             WHERE uid = ?1",
            [&m.uid],
        )
        .unwrap();
        archive(&a, m.id);
        assert_eq!(lifecycle_of(&a, &m.uid), Lifecycle::Archived);

        let stats = apply_changes(&a, &snapshot(&b).unwrap()).unwrap();

        assert_eq!(stats.tombstones, 1, "the forget was skipped: {stats:?}");
        assert_eq!(lifecycle_of(&a, &m.uid), Lifecycle::Forgotten);
    }

    /// What one store holds of a memory, as far as agreeing with another
    /// store goes: forgotten or not, its text, its pin.
    fn holds(conn: &Connection, uid: &str) -> (bool, String, bool) {
        conn.query_row(
            "SELECT deleted_at IS NOT NULL, COALESCE(body, ''), pinned FROM node WHERE uid = ?1",
            [uid],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap()
    }

    /// Move everything a write stamped on a row into the past.
    fn stamped_ago(conn: &Connection, uid: &str, secs: i64) {
        conn.execute(
            "UPDATE node SET updated_at = updated_at - ?2, deleted_at = deleted_at - ?2
             WHERE uid = ?1",
            params![uid, secs],
        )
        .unwrap();
    }

    type Annotate = fn(&Connection, i64);

    /// The writes that annotate a memory without touching its text, as a
    /// person or an agent makes them.
    fn annotations() -> Vec<(&'static str, Annotate)> {
        use crate::model::Actor;
        vec![
            ("pin", |c, id| {
                store::set_pinned(c, id, true, Actor::Human, None).unwrap()
            }),
            ("anchor", |c, id| {
                crate::anchors::set_anchors(c, id, &["deploy.sh".into()]).unwrap()
            }),
            ("retag", |c, id| {
                crate::memory::edit(
                    c,
                    id,
                    crate::memory::Edit {
                        text: None,
                        title: None,
                        mtype: None,
                        tags: Some(vec!["ops".into()]),
                    },
                    Actor::Human,
                    None,
                )
                .map(|_| ())
                .unwrap()
            }),
            ("confidence", |c, id| {
                store::set_confidence(c, id, crate::model::MemoryConfidence::Likely).unwrap()
            }),
            ("expiry", |c, id| {
                store::set_expiry(c, id, Some(now_unix() + 86_400), None).unwrap()
            }),
            ("trigger", |c, id| {
                store::set_fires_when(c, id, &["release checklist".into()]).unwrap()
            }),
        ]
    }

    /// Forgetting a memory decides what it says; pinning, tagging or
    /// anchoring it does not. So an annotation made on a machine that has
    /// not heard of the forget does not bring the memory back, however much
    /// later it was made: it stays forgotten on both machines. (Ordering by
    /// the row's last write alone, the later annotation won and the memory
    /// came back everywhere.)
    #[test]
    fn an_annotation_made_after_an_unseen_forget_does_not_bring_it_back() {
        for (what, annotate) in annotations() {
            let a = db::open_in_memory().unwrap();
            let m = mem(&a, "the staging bucket is world-readable on purpose");
            written_ago(&a, &m.uid, 86_400);
            let b = db::open_in_memory().unwrap();
            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

            // B forgets it; ten minutes later A, not yet synced, annotates it.
            let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
            store::soft_delete(&b, on_b.id, crate::model::Actor::Human, None).unwrap();
            stamped_ago(&b, &m.uid, 600);
            annotate(&a, m.id);

            for round in 0..2 {
                apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
                apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
                for (name, store) in [("a", &a), ("b", &b)] {
                    assert!(
                        holds(store, &m.uid).0,
                        "{what}: live again on {name} in round {round}"
                    );
                }
            }
        }
    }

    /// The same annotation made after the forget *arrived* is a different
    /// thing, and cannot happen by accident: a forgotten memory refuses a
    /// pin. Bringing one back is `remember --force`, which makes a new
    /// memory that the old tombstone does not touch.
    #[test]
    fn a_forgotten_memory_cannot_be_annotated_back_to_life() {
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "the staging bucket is world-readable on purpose");
        store::soft_delete(&a, m.id, crate::model::Actor::Human, None).unwrap();
        assert!(store::set_pinned(&a, m.id, true, crate::model::Actor::Human, None).is_err());
    }

    /// An annotation made on the old wording does not replace a rewrite made
    /// elsewhere before it. The rewrite is the later decision about what the
    /// memory says, and it is what both machines end up holding.
    #[test]
    fn an_annotation_on_the_old_text_does_not_undo_a_rewrite() {
        const REWRITTEN: &str = "the worker pool has sixteen threads since the resize";
        for (what, annotate) in annotations() {
            let a = db::open_in_memory().unwrap();
            let m = mem(&a, "the worker pool has eight threads");
            written_ago(&a, &m.uid, 86_400);
            let b = db::open_in_memory().unwrap();
            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

            let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
            crate::memory::edit(
                &b,
                on_b.id,
                crate::memory::Edit {
                    text: Some(REWRITTEN.into()),
                    title: None,
                    mtype: None,
                    tags: None,
                },
                crate::model::Actor::Human,
                None,
            )
            .unwrap();
            stamped_ago(&b, &m.uid, 600);
            annotate(&a, m.id);

            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
            apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
            for (name, store) in [("a", &a), ("b", &b)] {
                assert_eq!(holds(store, &m.uid).1, REWRITTEN, "{what}: store {name}");
            }
        }
    }

    /// Between two versions that say the same thing, the later annotation
    /// wins, as before.
    #[test]
    fn the_later_of_two_annotations_wins() {
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "deploys go out on tuesdays");
        written_ago(&a, &m.uid, 86_400);
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
        crate::anchors::set_anchors(&b, on_b.id, &["deploy.sh".into()]).unwrap();
        stamped_ago(&b, &m.uid, 600);
        store::set_pinned(&a, m.id, true, crate::model::Actor::Human, None).unwrap();

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        for (name, store) in [("a", &a), ("b", &b)] {
            assert!(holds(store, &m.uid).2, "store {name} lost the later pin");
        }
    }

    /// A build without the text clock annotates a row by moving `updated_at`
    /// and nothing else. Such a row orders by its last write, exactly as it
    /// did before the clock existed: the annotation is the later write and
    /// wins. Nothing is worse than it was; it is only not better until that
    /// machine is upgraded.
    #[test]
    fn a_row_annotated_by_an_older_build_is_plain_last_write_wins() {
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "the staging bucket is world-readable on purpose");
        written_ago(&a, &m.uid, 86_400);
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
        store::soft_delete(&b, on_b.id, crate::model::Actor::Human, None).unwrap();
        stamped_ago(&b, &m.uid, 600);
        a.execute(
            "UPDATE node SET pinned = 1, updated_at = ?2 WHERE id = ?1",
            params![m.id, now_unix()],
        )
        .unwrap();

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        for store in [&a, &b] {
            assert_eq!(holds(store, &m.uid), (false, m.body.clone().unwrap(), true));
        }
    }

    /// Two machines rewrite one memory in the same second. Neither is later,
    /// and each used to keep its own wording for good. One of them wins now,
    /// the same one on both, and the round after that moves nothing.
    #[test]
    fn two_rewrites_in_the_same_second_settle_on_one() {
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "the worker pool has eight threads");
        written_ago(&a, &m.uid, 86_400);
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        let at = now_unix() - 60;
        for (store, text) in [(&a, "sixteen threads"), (&b, "thirty-two threads")] {
            let id = store::get_node_by_uid(store, &m.uid).unwrap().id;
            crate::memory::edit(
                store,
                id,
                crate::memory::Edit {
                    text: Some(format!("the worker pool has {text}")),
                    title: None,
                    mtype: None,
                    tags: None,
                },
                crate::model::Actor::Human,
                None,
            )
            .unwrap();
            store
                .execute("UPDATE node SET updated_at = ?2 WHERE id = ?1", [id, at])
                .unwrap();
        }
        assert_ne!(holds(&a, &m.uid), holds(&b, &m.uid));

        let first = apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        let second = apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        assert_eq!(
            first.nodes_upserted + second.nodes_upserted,
            1,
            "exactly one side gives way"
        );
        assert_eq!(holds(&a, &m.uid), holds(&b, &m.uid));
        for (from, to) in [(&a, &b), (&b, &a)] {
            let again = apply_changes(to, &snapshot(from).unwrap()).unwrap();
            assert_eq!(again.nodes_upserted, 0, "settled: {again:?}");
        }
    }

    /// A record and the row it becomes are one version, whatever the store
    /// had to do to hold it: an unknown type reads back as none, tags are
    /// re-split, a malformed hash is dropped. Applying the same record again
    /// changes nothing, or a hub would relay it as a new arrival for ever.
    #[test]
    fn a_record_applied_once_is_never_applied_again() {
        let a = db::open_in_memory().unwrap();
        let n = mem(&a, "typed and tagged elsewhere");
        let mut batch = snapshot(&a).unwrap();
        batch.nodes[0].mtype = Some("a-type-from-the-future".into());
        batch.nodes[0].tags = vec!["two  words".into(), " padded ".into()];
        batch.nodes[0].content_hash = Some("not-hex".into());
        batch.nodes[0].meta = serde_json::json!({ "z": 1, "a": { "b": [1, 2] } });

        let b = db::open_in_memory().unwrap();
        assert_eq!(apply_changes(&b, &batch).unwrap().nodes_upserted, 1);
        let again = apply_changes(&b, &batch).unwrap();
        assert_eq!((again.nodes_upserted, again.nodes_skipped), (0, 1));
        // And what B now sends for it is, to A's copy of that record, the
        // same version.
        let back = snapshot(&b).unwrap();
        assert_eq!(
            Standing::of(&back.nodes[0]),
            Standing::of(&batch.nodes[0]),
            "{:?}",
            n.uid
        );
    }

    /// Three machines, three writes none of them has seen of the others: one
    /// forgets the memory, one rewrites it, one pins it. In every order the
    /// three writes can have happened, and every order the machines can sync
    /// in, all three end holding the same thing, and a further round changes
    /// nothing.
    ///
    /// What they hold is the later of the two decisions about what the
    /// memory says, the forget or the rewrite. The pin never decides: it was
    /// put on the old wording.
    ///
    /// This is the property the ordering exists to keep. A rule that
    /// preferred a forget to a pin and a rewrite to a forget, and otherwise
    /// took the later write, failed it: the three versions went round the
    /// three machines for ever.
    #[test]
    fn three_machines_that_each_wrote_end_up_agreeing() {
        const REWRITTEN: &str = "the worker pool has sixteen threads since the resize";
        const ORDERS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        // The three exchanges of a full round, as (from, to).
        const LINKS: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];

        for ages in ORDERS {
            for sync_order in ORDERS {
                let stores = [
                    db::open_in_memory().unwrap(),
                    db::open_in_memory().unwrap(),
                    db::open_in_memory().unwrap(),
                ];
                let m = mem(&stores[0], "the worker pool has eight threads");
                written_ago(&stores[0], &m.uid, 86_400);
                let seed = snapshot(&stores[0]).unwrap();
                apply_changes(&stores[1], &seed).unwrap();
                apply_changes(&stores[2], &seed).unwrap();
                let id = |i: usize| store::get_node_by_uid(&stores[i], &m.uid).unwrap().id;

                // Store 0 forgets, store 1 rewrites, store 2 pins; `ages`
                // says which of the three happened longest ago.
                store::soft_delete(&stores[0], id(0), crate::model::Actor::Human, None).unwrap();
                crate::memory::edit(
                    &stores[1],
                    id(1),
                    crate::memory::Edit {
                        text: Some(REWRITTEN.into()),
                        title: None,
                        mtype: None,
                        tags: None,
                    },
                    crate::model::Actor::Human,
                    None,
                )
                .unwrap();
                store::set_pinned(&stores[2], id(2), true, crate::model::Actor::Human, None)
                    .unwrap();
                for (store, rank) in stores.iter().zip(ages) {
                    stamped_ago(store, &m.uid, 600 * (rank as i64 + 1));
                }
                // `ages[i]` is how long ago store i wrote: the smaller, the
                // later. The forget is store 0's, the rewrite store 1's.
                let decider = if ages[0] < ages[1] { 0 } else { 1 };
                let expected = holds(&stores[decider], &m.uid);

                // Pairs exchange in the order under test, three rounds over.
                let links: Vec<(usize, usize)> = sync_order
                    .iter()
                    .flat_map(|&pair| [LINKS[pair * 2], LINKS[pair * 2 + 1]])
                    .collect();
                for _ in 0..3 {
                    for &(from, to) in &links {
                        apply_changes(&stores[to], &snapshot(&stores[from]).unwrap()).unwrap();
                    }
                }
                for (i, store) in stores.iter().enumerate() {
                    assert_eq!(
                        holds(store, &m.uid),
                        expected,
                        "store {i} disagrees (ages {ages:?}, sync order {sync_order:?})"
                    );
                }
                // Settled: one more full round is all no-ops.
                for &(from, to) in &links {
                    let stats =
                        apply_changes(&stores[to], &snapshot(&stores[from]).unwrap()).unwrap();
                    assert_eq!(
                        stats.nodes_upserted + stats.tombstones,
                        0,
                        "still moving after three rounds (ages {ages:?}, \
                         sync order {sync_order:?}): {stats:?}"
                    );
                }
            }
        }
    }

    fn rewrite(conn: &Connection, id: i64, text: &str) {
        crate::memory::edit(
            conn,
            id,
            crate::memory::Edit {
                text: Some(text.into()),
                title: None,
                mtype: None,
                tags: None,
            },
            crate::model::Actor::Human,
            None,
        )
        .unwrap();
    }

    fn written_at(conn: &Connection, uid: &str, at: i64) {
        conn.execute(
            "UPDATE node SET updated_at = ?2,
                    deleted_at = CASE WHEN deleted_at IS NULL THEN NULL ELSE ?2 END
             WHERE uid = ?1",
            params![uid, at],
        )
        .unwrap();
    }

    /// A rewrite is a new decision about the text, whatever the words. Two
    /// cases where the clock's own hash cannot see that: rewording back to
    /// an earlier text (whose old clock would read as valid again), and a
    /// rewrite that changes only case or punctuation (the content hash
    /// ignores both). Each must still beat a forget made elsewhere before
    /// it.
    #[test]
    fn every_rewrite_is_a_new_decision_even_when_the_hash_comes_back() {
        let cases: [(&str, &str, &str); 2] = [
            (
                "the worker pool has eight threads",
                "the worker pool has sixteen threads",
                "the worker pool has eight threads",
            ),
            (
                "retry limit is < 5 per host",
                "retry limit is < 5 per host",
                "retry limit is > 5 per host",
            ),
        ];
        for (first, middle, last) in cases {
            let a = db::open_in_memory().unwrap();
            let m = mem(&a, first);
            written_at(&a, &m.uid, 1_000);
            store::set_pinned(&a, m.id, true, crate::model::Actor::Human, None).unwrap();
            written_at(&a, &m.uid, 1_500);
            if middle != first {
                rewrite(&a, m.id, middle);
                written_at(&a, &m.uid, 2_000);
            }
            let b = db::open_in_memory().unwrap();
            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

            // B forgets it at 2,500; A rewrites it at 3,000.
            let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
            store::soft_delete(&b, on_b.id, crate::model::Actor::Human, None).unwrap();
            written_at(&b, &m.uid, 2_500);
            rewrite(&a, m.id, last);
            written_at(&a, &m.uid, 3_000);

            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
            apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
            for (name, store) in [("a", &a), ("b", &b)] {
                assert_eq!(
                    holds(store, &m.uid).1,
                    last,
                    "{first:?} -> {last:?}: store {name} lost the later rewrite"
                );
                assert!(!holds(store, &m.uid).0, "store {name}: still forgotten");
            }
        }
    }

    /// A project's portable key changes on one machine (its git remote
    /// moved). That writes nothing to the project's memories, so syncing
    /// with a machine that still has the old key moves none of them: each
    /// store keeps them in the project it has.
    #[test]
    fn a_projects_key_changing_on_one_machine_moves_none_of_its_memories() {
        let a = db::open_in_memory().unwrap();
        let b = db::open_in_memory().unwrap();
        let pa = store::ensure_project(&a, "/local/a", "proj").unwrap();
        store::set_project_identity(&a, pa.id, Some("git:github.com/old/repo"), true).unwrap();
        for i in 0..40 {
            proj_mem(
                &a,
                pa.id,
                &format!("project fact number {i} about the build"),
            );
        }
        let pb = store::ensure_project(&b, "/local/b", "proj").unwrap();
        store::set_project_identity(&b, pb.id, Some("git:github.com/old/repo"), true).unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        store::set_project_identity(&a, pa.id, Some("git:github.com/new/repo"), true).unwrap();
        for _ in 0..3 {
            apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
            apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        }
        let held_by = |c: &Connection, pid: i64| -> i64 {
            c.query_row(
                "SELECT count(*) FROM node WHERE kind = 'memory' AND project_id = ?1",
                [pid],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(held_by(&a, pa.id), 40, "memories left A's project");
        assert_eq!(held_by(&b, pb.id), 40, "memories left B's project");
    }

    /// A content hash is only text on the wire. One that is not ASCII where
    /// the clock's prefix ends reads as "no clock"; it does not stop a merge.
    #[test]
    fn a_malformed_hash_does_not_stop_a_merge() {
        let a = db::open_in_memory().unwrap();
        mem(&a, "plain");
        let mut batch = snapshot(&a).unwrap();
        batch.nodes[0].content_hash = Some("aaaaaaaaaaaaaaa\u{e9}zz".into());
        batch.nodes[0].meta = serde_json::json!({"text_clock": {"at": 5, "h": "aaaaaaaaaaaaaaaa"}});
        apply_changes(&a, &batch).unwrap();
    }

    /// A peer that sends no content hash: the receiving store fills it in
    /// (the embed pass backfills it). The record and the backfilled row are
    /// the same version, so the next round applies nothing; a hub would
    /// otherwise relay that memory as new on every round.
    #[test]
    fn a_backfilled_hash_is_not_a_new_version() {
        for i in 0..10 {
            let a = db::open_in_memory().unwrap();
            mem(
                &a,
                &format!("a memory from a peer that sends no hash, number {i}"),
            );
            let mut batch = snapshot(&a).unwrap();
            batch.nodes[0].content_hash = None;
            let b = db::open_in_memory().unwrap();
            apply_changes(&b, &batch).unwrap();
            for round in 0..3 {
                crate::embed::embed_pending(&b, "test-model", &mut |texts: Vec<String>| {
                    Ok(texts.iter().map(|_| vec![0.5f32; 4]).collect())
                })
                .unwrap();
                let again = apply_changes(&b, &batch).unwrap();
                assert_eq!(
                    again.nodes_upserted, 0,
                    "memory {i}, round {round}: {again:?}"
                );
            }
        }
    }

    /// The property behind every test above, held against schedules nobody
    /// thought of: three stores, two hundred random histories of forgets,
    /// rewrites, annotations (by this build and by one that knows no text
    /// clock), several of them in the same second, synced in random order in
    /// between. Afterwards the stores exchange until a whole round applies
    /// nothing, and then they must hold the same version.
    ///
    /// Deterministic: the histories come from a fixed seed. A failure prints
    /// the history that produced it.
    #[test]
    fn random_histories_on_three_stores_always_converge() {
        use crate::model::Actor;
        const LINKS: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
        const TEXTS: [&str; 3] = [
            "the worker pool has sixteen threads",
            "the worker pool has thirty-two threads",
            "the worker pool is sized from the core count",
        ];
        // A small generator of its own, so the histories never change under
        // the test (xorshift64).
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % n
            }
        }
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

        for history in 0..200 {
            let stores = [
                db::open_in_memory().unwrap(),
                db::open_in_memory().unwrap(),
                db::open_in_memory().unwrap(),
            ];
            let m = mem(&stores[0], "the worker pool has eight threads");
            let start = now_unix() - 100_000;
            stores[0]
                .execute(
                    "UPDATE node SET created_at = ?2, updated_at = ?2 WHERE uid = ?1",
                    params![m.uid, start],
                )
                .unwrap();
            let seed = snapshot(&stores[0]).unwrap();
            apply_changes(&stores[1], &seed).unwrap();
            apply_changes(&stores[2], &seed).unwrap();

            let mut at = start;
            let mut log: Vec<String> = Vec::new();
            for _ in 0..(4 + rng.below(9)) {
                let i = rng.below(3) as usize;
                let store = &stores[i];
                let id = store::get_node_by_uid(store, &m.uid).unwrap().id;
                // One step in three happens in the same second as the last.
                if rng.below(3) != 0 {
                    at += 1 + rng.below(50) as i64;
                }
                let what = rng.below(8);
                let done = match what {
                    0 => store::soft_delete(store, id, Actor::Human, None).is_ok(),
                    1 | 2 => crate::memory::edit(
                        store,
                        id,
                        crate::memory::Edit {
                            text: Some(TEXTS[rng.below(3) as usize].into()),
                            title: None,
                            mtype: None,
                            tags: None,
                        },
                        Actor::Human,
                        None,
                    )
                    .is_ok(),
                    3 => {
                        store::set_pinned(store, id, rng.below(2) == 0, Actor::Human, None).is_ok()
                    }
                    4 => crate::anchors::set_anchors(store, id, &[format!("f{}.rs", rng.below(3))])
                        .is_ok(),
                    // A build without the text clock: the row is marked
                    // changed and nothing else is recorded.
                    5 => store
                        .execute(
                            "UPDATE node SET pinned = 1 - pinned, updated_at = ?2
                             WHERE id = ?1 AND deleted_at IS NULL",
                            params![id, now_unix()],
                        )
                        .is_ok(),
                    _ => {
                        let (from, to) = LINKS[rng.below(6) as usize];
                        apply_changes(&stores[to], &snapshot(&stores[from]).unwrap()).unwrap();
                        log.push(format!("sync {from}->{to}"));
                        continue;
                    }
                };
                if !done {
                    continue;
                }
                // The write stamped the real clock; move it to the history's.
                store
                    .execute(
                        "UPDATE node SET updated_at = ?2,
                                         deleted_at = CASE WHEN deleted_at IS NULL THEN NULL
                                                           ELSE max(?2, ?3) END
                         WHERE id = ?1 AND updated_at >= ?4",
                        params![id, at, at, now_unix() - 5],
                    )
                    .unwrap();
                log.push(format!("store {i} op {what} at +{}", at - start));
            }

            let mut rounds = 0;
            loop {
                let mut moved = 0;
                for &(from, to) in &LINKS {
                    let stats =
                        apply_changes(&stores[to], &snapshot(&stores[from]).unwrap()).unwrap();
                    moved += stats.nodes_upserted;
                }
                if moved == 0 {
                    break;
                }
                rounds += 1;
                assert!(rounds <= 4, "history {history} never settles: {log:?}");
            }
            let versions: Vec<serde_json::Value> = stores
                .iter()
                .map(|store| {
                    let mut record =
                        serde_json::to_value(&snapshot(store).unwrap().nodes[0]).unwrap();
                    // Each store keeps the text clock a row arrived with.
                    record["meta"]
                        .as_object_mut()
                        .map(|meta| meta.remove(crate::text_clock::KEY));
                    record
                })
                .collect();
            assert!(
                versions[0] == versions[1] && versions[1] == versions[2],
                "history {history} ends in disagreement: {log:?}\n{versions:#?}"
            );
        }
    }

    /// Someone rewriting a memory after it was forgotten elsewhere is
    /// ordinary last-write-wins, as it always was.
    #[test]
    fn a_rewrite_after_a_forget_elsewhere_is_still_last_write_wins() {
        use crate::lifecycle::Lifecycle;
        let a = db::open_in_memory().unwrap();
        let m = mem(&a, "the worker pool has eight threads");
        written_ago(&a, &m.uid, 86_400);
        let b = db::open_in_memory().unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        let on_b = store::get_node_by_uid(&b, &m.uid).unwrap();
        store::soft_delete(&b, on_b.id, crate::model::Actor::Human, None).unwrap();
        stamped_ago(&b, &m.uid, 600);
        crate::memory::edit(
            &a,
            m.id,
            crate::memory::Edit {
                text: Some("the worker pool has sixteen threads since the resize".into()),
                title: None,
                mtype: None,
                tags: None,
            },
            crate::model::Actor::Human,
            None,
        )
        .unwrap();

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        assert_eq!(lifecycle_of(&b, &m.uid), Lifecycle::Live);
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        assert_eq!(lifecycle_of(&a, &m.uid), Lifecycle::Live);
    }

    /// Two machines retire a pair in opposite directions before they sync.
    /// Applied naively, every store ends with each superseded by the other:
    /// nothing current on either end, and no way to undo it. Each store
    /// keeps the memory it kept.
    #[test]
    fn opposite_supersessions_never_hide_both() {
        let (a, b, old, new) = two_stores_sharing_a_pair();
        supersede(&a, &old, &new);
        let (old_on_b, new_on_b) = (
            store::get_node_by_uid(&b, &old.uid).unwrap(),
            store::get_node_by_uid(&b, &new.uid).unwrap(),
        );
        supersede(&b, &new_on_b, &old_on_b);

        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();
        apply_changes(&a, &snapshot(&b).unwrap()).unwrap();
        apply_changes(&b, &snapshot(&a).unwrap()).unwrap();

        for (name, store) in [("a", &a), ("b", &b)] {
            let hidden = [&old.uid, &new.uid]
                .iter()
                .filter(|uid| replaced_by(store, uid).is_some())
                .count();
            assert_eq!(
                hidden, 1,
                "store {name} has {hidden} of the pair superseded"
            );
        }
    }

    /// The uid of a replacement goes on the wire only when the replacement
    /// itself syncs. One filed under a project kept off sync is named to
    /// nobody: a peer could not resolve it, and the hub would learn that a
    /// memory of that project exists.
    #[test]
    fn a_replacement_that_does_not_sync_is_not_named() {
        let a = db::open_in_memory().unwrap();
        let old = mem(&a, "release notes go in the wiki");
        let private = crate::store::ensure_project(&a, "/tmp/private-repo", "private").unwrap();
        let new = proj_mem(&a, private.id, "release notes go in CHANGELOG.md now");
        supersede(&a, &old, &new);

        let batch = snapshot(&a).unwrap();

        assert!(
            batch.nodes.iter().all(|r| r.uid != new.uid),
            "the private memory itself must not sync"
        );
        let record = batch.nodes.iter().find(|r| r.uid == old.uid).unwrap();
        assert_eq!(record.superseded_by, None);
        assert!(!serde_json::to_string(&batch).unwrap().contains(&new.uid));
    }
}
