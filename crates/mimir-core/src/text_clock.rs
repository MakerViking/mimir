//! When the text of a memory was last decided, kept apart from when the row
//! was last written.
//!
//! A memory row has one `updated_at`, and every write moves it: a rewrite,
//! but also a pin, a retag, an anchor, an expiry. Sync used to order two
//! versions of a memory by that one number, so "pinned it on the laptop"
//! outranked "forgot it on the desktop an hour earlier" and the forgotten
//! memory came back everywhere, and a pin made on the old wording replaced a
//! rewrite made elsewhere a minute before.
//!
//! Those are two different kinds of write. One decides what the memory
//! *says* (it is written, it is reworded, it is forgotten); the other only
//! annotates it. The text clock records the first kind: `meta.text_clock`
//! holds the time the current text was decided, and the start of the content
//! hash it was decided for.
//!
//! # How it stays right without every writer knowing about it
//!
//! Only writes that leave the text alone call [`keep`], and only before they
//! move `updated_at`. `keep` records "the text is as old as the last write
//! before this one" when no valid clock is there. Nothing else stamps
//! anything:
//!
//! - A new memory has no clock. Its text is as old as the row, which is what
//!   a missing clock means ([`read`] returns `None`, and the reader falls
//!   back on `updated_at`).
//! - A rewrite removes the clock (`memory::edit` does, whenever the text
//!   changes at all), so the text is as old as the rewrite, again
//!   `updated_at`. The hash in the clock is a second line of defence for
//!   writers that change the text without knowing about the clock. It is
//!   not enough on its own: the content hash ignores case and punctuation,
//!   and a rewrite back to an earlier wording brings the old hash back.
//! - A writer that never heard of the clock (an older build on another
//!   machine, a code path that forgot to call [`keep`]) leaves a row that
//!   orders exactly as it did before the clock existed.
//!
//! So a mistake here degrades to plain last-write-wins for that row. It
//! cannot make two stores disagree: the clock travels in `meta` with the
//! row, and every store reads the same row the same way.

use rusqlite::Connection;
use serde_json::Value;

use crate::error::Result;

/// The `meta` key: `{"at": <unix seconds>, "h": "<start of the content hash>"}`.
pub const KEY: &str = "text_clock";

/// How much of the hex content hash the clock carries. Enough to tell one
/// wording of a memory from the next; it is not a security boundary.
const HASH_CHARS: usize = 16;

/// The same prefix, as SQL over a row's `content_hash` column.
const HASH_SQL: &str = "substr(lower(hex(content_hash)), 1, 16)";

/// When the text of a memory was decided, if the row says so and the clock
/// is still about this text. `content_hash` is the hash as it travels (hex);
/// `None` for a row without one, which can carry no clock.
///
/// `None` means "as old as the row's last write": the caller uses
/// `updated_at`.
pub fn read(meta: &Value, content_hash: Option<&str>) -> Option<i64> {
    let clock = meta.get(KEY)?;
    let at = clock.get("at")?.as_i64()?;
    let about = clock.get("h")?.as_str()?;
    let hash = content_hash?.to_ascii_lowercase();
    // `get`, not indexing: a hash from another machine is only text, and a
    // malformed one must read as "no clock", not panic mid-merge.
    (hash.get(..HASH_CHARS) == Some(about)).then_some(at)
}

/// When what a memory says was last decided: the clock when there is a valid
/// one, `updated_at` otherwise. Never later than `updated_at` (a clock from
/// a machine whose own clock ran ahead cannot claim the text is newer than
/// the row).
///
/// This is the time to order two versions of a memory by, and the age to
/// show a reader: a pin or a tag put on a year-old note does not make the
/// note a day old.
pub fn decided(meta: &Value, content_hash: Option<&str>, updated_at: i64) -> i64 {
    read(meta, content_hash).map_or(updated_at, |at| at.min(updated_at))
}

/// [`decided`] for a row of this store.
pub fn decided_at(node: &crate::model::Node) -> i64 {
    let hash: Option<String> = node
        .content_hash
        .as_deref()
        .map(|bytes| bytes.iter().map(|byte| format!("{byte:02x}")).collect());
    decided(&node.meta, hash.as_deref(), node.updated_at)
}

/// Call before a write that leaves a memory's text alone moves its
/// `updated_at`: a pin, a tag, an anchor, an expiry, a move to another
/// project.
///
/// Records that the text is as old as the row's last write, unless a valid
/// clock already says when it was decided. Does nothing for a row that is
/// not a memory or has no content hash.
pub fn keep(conn: &Connection, id: i64) -> Result<()> {
    conn.execute(
        &format!(
            "UPDATE node
             SET meta = json_set(meta, '$.{KEY}', json_object('at', updated_at, 'h', {HASH_SQL}))
             WHERE id = ?1 AND kind = 'memory' AND content_hash IS NOT NULL
               AND NOT (typeof(json_extract(meta, '$.{KEY}.at')) = 'integer'
                        AND COALESCE(json_extract(meta, '$.{KEY}.h'), '') = {HASH_SQL})"
        ),
        [id],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::{self, Remember, RememberOutcome};
    use crate::model::{Actor, MemoryType, Node};
    use crate::store;

    fn remembered(conn: &Connection, text: &str) -> Node {
        match memory::remember(
            conn,
            Remember {
                text: text.into(),
                mtype: MemoryType::Note,
                tags: Vec::new(),
                project_id: None,
                force: false,
                actor: Actor::Human,
            },
        )
        .unwrap()
        {
            RememberOutcome::Created(node) => node,
            other => panic!("expected a new memory, got {other:?}"),
        }
    }

    fn hex(node: &Node) -> Option<String> {
        node.content_hash
            .as_deref()
            .map(|b| b.iter().map(|x| format!("{x:02x}")).collect())
    }

    fn clock(conn: &Connection, id: i64) -> Option<i64> {
        let node = store::get_node(conn, id).unwrap();
        read(&node.meta, hex(&node).as_deref())
    }

    fn written_at(conn: &Connection, id: i64, at: i64) {
        conn.execute("UPDATE node SET updated_at = ?2 WHERE id = ?1", [id, at])
            .unwrap();
    }

    /// A memory nobody has annotated carries no clock: its text is as old as
    /// the row. The first write that leaves the text alone records that
    /// moment before it moves `updated_at`, and later ones leave it be.
    #[test]
    fn an_annotation_records_when_the_text_was_last_written() {
        let conn = db::open_in_memory().unwrap();
        let m = remembered(&conn, "the staging bucket is world-readable on purpose");
        assert_eq!(clock(&conn, m.id), None);

        written_at(&conn, m.id, 1_000);
        store::set_pinned(&conn, m.id, true, Actor::Human, None).unwrap();
        assert_eq!(clock(&conn, m.id), Some(1_000));
        assert!(store::get_node(&conn, m.id).unwrap().updated_at > 1_000);

        store::set_confidence(&conn, m.id, crate::model::MemoryConfidence::Likely).unwrap();
        crate::anchors::set_anchors(&conn, m.id, &["deploy.sh".into()]).unwrap();
        assert_eq!(clock(&conn, m.id), Some(1_000), "the text has not changed");
        // And that is the age a reader is shown, not the time of the pin.
        assert_eq!(decided_at(&store::get_node(&conn, m.id).unwrap()), 1_000);
    }

    /// Without a clock the text is as old as the row; a clock that claims to
    /// be newer than the row is not believed past it.
    #[test]
    fn the_decided_time_is_never_later_than_the_last_write() {
        let hash = "00112233445566778899aabbccddeeff";
        let clocked = |at: i64| serde_json::json!({ KEY: { "at": at, "h": &hash[..HASH_CHARS] } });
        assert_eq!(decided(&serde_json::json!({}), Some(hash), 500), 500);
        assert_eq!(decided(&clocked(100), Some(hash), 500), 100);
        assert_eq!(decided(&clocked(900), Some(hash), 500), 500);
        assert_eq!(decided(&clocked(100), None, 500), 500);
    }

    /// A rewrite needs no stamp: the clock it leaves behind is about the old
    /// wording and reads as missing, so the text is as old as the rewrite.
    /// The next annotation records exactly that.
    #[test]
    fn a_rewrite_makes_the_old_clock_read_as_missing() {
        let conn = db::open_in_memory().unwrap();
        let m = remembered(&conn, "the worker pool has eight threads");
        written_at(&conn, m.id, 1_000);
        store::set_pinned(&conn, m.id, true, Actor::Human, None).unwrap();
        assert_eq!(clock(&conn, m.id), Some(1_000));

        memory::edit(
            &conn,
            m.id,
            memory::Edit {
                text: Some("the worker pool has sixteen threads since the resize".into()),
                title: None,
                mtype: None,
                tags: None,
            },
            Actor::Human,
            None,
        )
        .unwrap();
        assert_eq!(clock(&conn, m.id), None, "the clock was about the old text");

        written_at(&conn, m.id, 5_000);
        store::set_pinned(&conn, m.id, false, Actor::Human, None).unwrap();
        assert_eq!(clock(&conn, m.id), Some(5_000));
    }

    /// An edit that changes only the tags is an annotation too.
    #[test]
    fn a_retag_keeps_the_clock() {
        let conn = db::open_in_memory().unwrap();
        let m = remembered(&conn, "deploys go out on tuesdays");
        written_at(&conn, m.id, 1_000);
        memory::edit(
            &conn,
            m.id,
            memory::Edit {
                text: None,
                title: None,
                mtype: None,
                tags: Some(vec!["release".into()]),
            },
            Actor::Human,
            None,
        )
        .unwrap();
        assert_eq!(clock(&conn, m.id), Some(1_000));
    }

    /// What `read` accepts is what `keep` leaves alone: a clock of the wrong
    /// shape, or about another text, is missing to both.
    #[test]
    fn read_and_keep_agree_on_what_a_valid_clock_is() {
        let conn = db::open_in_memory().unwrap();
        let m = remembered(&conn, "the cache lives on the scratch volume");
        let hash = hex(&m).unwrap();
        let cases = [
            (
                serde_json::json!({ "at": 7, "h": &hash[..HASH_CHARS] }),
                true,
            ),
            (
                serde_json::json!({ "at": 7, "h": "0000000000000000" }),
                false,
            ),
            (
                serde_json::json!({ "at": "7", "h": &hash[..HASH_CHARS] }),
                false,
            ),
            (serde_json::json!({ "at": 7 }), false),
            (serde_json::json!("soon"), false),
        ];
        for (value, valid) in cases {
            conn.execute(
                "UPDATE node SET meta = json_set('{}', '$.text_clock', json(?2)), updated_at = 900
                 WHERE id = ?1",
                rusqlite::params![m.id, value.to_string()],
            )
            .unwrap();
            assert_eq!(clock(&conn, m.id).is_some(), valid, "read: {value}");
            keep(&conn, m.id).unwrap();
            let expected = if valid { 7 } else { 900 };
            assert_eq!(clock(&conn, m.id), Some(expected), "keep: {value}");
        }
    }

    /// Nothing but a memory with a content hash gets a clock.
    #[test]
    fn keep_leaves_other_rows_alone() {
        let conn = db::open_in_memory().unwrap();
        let project = store::ensure_project(&conn, "/tmp/somewhere", "somewhere").unwrap();
        keep(&conn, project.id).unwrap();
        let m = remembered(&conn, "a note whose hash went missing");
        conn.execute("UPDATE node SET content_hash = NULL WHERE id = ?1", [m.id])
            .unwrap();
        keep(&conn, m.id).unwrap();
        for id in [project.id, m.id] {
            assert!(store::get_node(&conn, id).unwrap().meta.get(KEY).is_none());
        }
    }
}
