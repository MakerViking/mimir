//! The miss ledger: recall queries the store had no answer for.
//!
//! `recall_event` can say which memories were shown, opened and marked
//! useful. It cannot say what was *missing*, because a recall that found
//! nothing wrote no row — and "what do agents keep asking that nobody has
//! written down?" is the list that tells a person which memory to write
//! next. This keeps that list.
//!
//! A miss is a page with no `strong` hit ([`strength::is_miss`]): either
//! nothing came back, or everything that did is only there because a ranked
//! list always has something in it. Only *explicit* recalls feed it (CLI
//! `recall`, MCP `recall`). The per-prompt inject hook is deliberately not a
//! feeder: it runs on every prompt, most of which are not questions for the
//! store, and silence is its normal answer — logging those would mean
//! storing nearly every prompt anyone types.

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::Result;
use crate::model::Scope;
use crate::strength::{self, MatchStrength};

/// Longest query kept. Recall queries are a sentence; something much longer
/// is a pasted prompt or log, and the head says what it was about.
const MAX_QUERY_CHARS: usize = 300;

/// One ledger row.
#[derive(Debug, Clone)]
pub struct Miss {
    pub id: i64,
    pub query: String,
    pub project_id: Option<i64>,
    /// How many (weak) hits the last miss returned; 0 = nothing at all.
    pub hits: i64,
    pub count: i64,
    pub first_at: i64,
    pub last_at: i64,
    pub resolved_at: Option<i64>,
}

/// The project a recall's scope belongs to, for keying. `All` and `Global`
/// both mean "not one project's question".
pub fn scope_project(scope: &Scope) -> Option<i64> {
    match scope {
        Scope::Project(id) => Some(*id),
        Scope::All | Scope::Global => None,
    }
}

/// The comparison form: case and spacing are not part of the question, so
/// "WAL growth" and "wal  growth" are one row with a count of two.
fn normalise(query: &str) -> String {
    query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Record what one explicit recall page says about the store: a miss bumps
/// (or reopens) the query's row, an answer resolves it. The single entry
/// point both recall surfaces call.
///
/// Never writes a query the secrets scanner flags — the ledger keeps text,
/// and a secret pasted into a search box must not become the one copy of it
/// that outlives the session. Best-effort like the other ledgers: a failed
/// write is a warning, never a failed recall.
pub fn observe(
    conn: &Connection,
    query: &str,
    project_id: Option<i64>,
    strengths: &[MatchStrength],
    now: i64,
) {
    let normalised = normalise(query);
    if normalised.is_empty() || crate::secrets::scan(query).is_some() {
        return;
    }
    let key = blake3::hash(normalised.as_bytes());
    let res = if strength::is_miss(strengths) {
        record(
            conn,
            query,
            key.as_bytes(),
            project_id,
            strengths.len(),
            now,
        )
    } else {
        resolve(conn, key.as_bytes(), project_id, now)
    };
    if let Err(err) = res {
        tracing::warn!(%err, "could not update the miss ledger");
    }
}

fn record(
    conn: &Connection,
    query: &str,
    key: &[u8],
    project_id: Option<i64>,
    hits: usize,
    now: i64,
) -> Result<()> {
    // Spacing collapsed, case kept: the row should read as the question was
    // asked, minus the stray whitespace `normalise` already ignores.
    let text: String = query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MAX_QUERY_CHARS)
        .collect();
    // A resolved row that misses again is reopened, not duplicated: the
    // memory that answered it was deleted or superseded, and the question
    // is open again with all of its history.
    conn.execute(
        "INSERT INTO recall_miss (query, query_key, project_id, hits, count, first_at, last_at)
         VALUES (?1, ?2, ?3, ?4, 1, ?5, ?5)
         ON CONFLICT(query_key, COALESCE(project_id, -1)) DO UPDATE SET
             query = ?1,
             hits = ?4,
             count = count + 1,
             last_at = ?5,
             resolved_at = NULL",
        params![text, key, project_id, hits as i64, now],
    )?;
    Ok(())
}

fn resolve(conn: &Connection, key: &[u8], project_id: Option<i64>, now: i64) -> Result<()> {
    conn.execute(
        "UPDATE recall_miss SET resolved_at = ?3
         WHERE query_key = ?1 AND COALESCE(project_id, -1) = COALESCE(?2, -1)
           AND resolved_at IS NULL",
        params![key, project_id, now],
    )?;
    Ok(())
}

/// Ledger rows, most-asked first, then most recent. `project_id = Some`
/// keeps that project's misses and the cross-project ones (the same reach a
/// recall from that project has); `None` keeps everything.
pub fn list(
    conn: &Connection,
    project_id: Option<i64>,
    include_resolved: bool,
    limit: usize,
) -> Result<Vec<Miss>> {
    let mut stmt = conn.prepare(
        "SELECT id, query, project_id, hits, count, first_at, last_at, resolved_at
         FROM recall_miss
         WHERE (?1 IS NULL OR project_id IS NULL OR project_id = ?1)
           AND (?2 OR resolved_at IS NULL)
         ORDER BY count DESC, last_at DESC, id
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![project_id, include_resolved, limit as i64], |r| {
        Ok(Miss {
            id: r.get(0)?,
            query: r.get(1)?,
            project_id: r.get(2)?,
            hits: r.get(3)?,
            count: r.get(4)?,
            first_at: r.get(5)?,
            last_at: r.get(6)?,
            resolved_at: r.get(7)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// `(open, resolved)` among queries last missed since `since`.
pub fn counts(conn: &Connection, since: i64) -> Result<(i64, i64)> {
    Ok(conn.query_row(
        "SELECT COALESCE(SUM(resolved_at IS NULL), 0), COALESCE(SUM(resolved_at IS NOT NULL), 0)
         FROM recall_miss WHERE last_at >= ?1",
        [since],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?)
}

/// Remove one row — a question that was noise, not a gap.
pub fn dismiss(conn: &Connection, id: i64) -> Result<bool> {
    Ok(conn.execute("DELETE FROM recall_miss WHERE id = ?1", [id])? > 0)
}

/// Drop rows nobody has asked about since `cutoff`. Runs with the
/// `recall_event` prune, on the same retention window.
pub fn prune(conn: &Connection, cutoff: i64) -> Result<usize> {
    Ok(conn.execute("DELETE FROM recall_miss WHERE last_at < ?1", [cutoff])?)
}

/// The row for `query` in `project_id`, if one exists.
pub fn find(conn: &Connection, query: &str, project_id: Option<i64>) -> Result<Option<Miss>> {
    let key = blake3::hash(normalise(query).as_bytes());
    Ok(conn
        .query_row(
            "SELECT id, query, project_id, hits, count, first_at, last_at, resolved_at
             FROM recall_miss
             WHERE query_key = ?1 AND COALESCE(project_id, -1) = COALESCE(?2, -1)",
            params![key.as_bytes(), project_id],
            |r| {
                Ok(Miss {
                    id: r.get(0)?,
                    query: r.get(1)?,
                    project_id: r.get(2)?,
                    hits: r.get(3)?,
                    count: r.get(4)?,
                    first_at: r.get(5)?,
                    last_at: r.get(6)?,
                    resolved_at: r.get(7)?,
                })
            },
        )
        .optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use MatchStrength::{Strong, Weak};

    const T: i64 = 1_700_000_000;

    #[test]
    fn a_repeated_miss_is_one_row_counted_twice_regardless_of_case_and_spacing() {
        let conn = db::open_in_memory().unwrap();
        observe(&conn, "How do we rotate the hub token", None, &[Weak], T);
        observe(&conn, "how do we  rotate the HUB token", None, &[], T + 5);
        let rows = list(&conn, None, false, 10).unwrap();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].count, 2);
        assert_eq!(rows[0].hits, 0, "the latest miss came back empty");
        assert_eq!((rows[0].first_at, rows[0].last_at), (T, T + 5));
    }

    #[test]
    fn a_strong_answer_resolves_and_a_later_miss_reopens() {
        let conn = db::open_in_memory().unwrap();
        observe(&conn, "hub token rotation", None, &[Weak], T);
        observe(&conn, "hub token rotation", None, &[Weak, Strong], T + 10);
        assert!(list(&conn, None, false, 10).unwrap().is_empty(), "answered");
        let resolved = find(&conn, "hub token rotation", None).unwrap().unwrap();
        assert_eq!(resolved.resolved_at, Some(T + 10));

        // The answering memory went away; the question is open again, with
        // its history rather than as a stranger.
        observe(&conn, "hub token rotation", None, &[Weak], T + 20);
        let reopened = find(&conn, "hub token rotation", None).unwrap().unwrap();
        assert_eq!(reopened.resolved_at, None);
        assert_eq!(reopened.count, 2);
        assert_eq!(reopened.first_at, T);
    }

    #[test]
    fn an_answer_never_creates_a_row() {
        let conn = db::open_in_memory().unwrap();
        observe(&conn, "wal growth", None, &[Strong], T);
        assert!(find(&conn, "wal growth", None).unwrap().is_none());
    }

    #[test]
    fn the_same_question_in_two_projects_is_two_gaps() {
        let conn = db::open_in_memory().unwrap();
        observe(&conn, "release steps", Some(1), &[Weak], T);
        observe(&conn, "release steps", Some(2), &[Weak], T);
        observe(&conn, "release steps", Some(1), &[Strong], T + 1);
        assert_eq!(
            find(&conn, "release steps", Some(1))
                .unwrap()
                .unwrap()
                .resolved_at,
            Some(T + 1)
        );
        assert_eq!(
            find(&conn, "release steps", Some(2))
                .unwrap()
                .unwrap()
                .resolved_at,
            None
        );
        // A project sees its own misses and the cross-project ones only.
        observe(&conn, "global question", None, &[Weak], T);
        let seen: Vec<String> = list(&conn, Some(2), false, 10)
            .unwrap()
            .into_iter()
            .map(|m| m.query)
            .collect();
        assert_eq!(seen, vec!["release steps", "global question"], "{seen:?}");
    }

    #[test]
    fn a_query_carrying_a_secret_is_never_written() {
        let conn = db::open_in_memory().unwrap();
        let token = format!("ghp_{}", "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8");
        assert!(
            crate::secrets::scan(&token).is_some(),
            "fixture must trip the scanner"
        );
        observe(&conn, &format!("why does {token} fail"), None, &[Weak], T);
        let n: i64 = conn
            .query_row("SELECT count(*) FROM recall_miss", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn long_queries_are_capped_and_prune_drops_the_unasked() {
        let conn = db::open_in_memory().unwrap();
        observe(&conn, &"word ".repeat(200), None, &[Weak], T);
        observe(&conn, "recent", None, &[Weak], T + 100);
        let long = list(&conn, None, false, 10)
            .unwrap()
            .into_iter()
            .find(|m| m.query != "recent")
            .unwrap();
        assert!(long.query.chars().count() <= MAX_QUERY_CHARS);
        assert_eq!(prune(&conn, T + 50).unwrap(), 1);
        assert_eq!(list(&conn, None, true, 10).unwrap().len(), 1);
        assert_eq!(counts(&conn, 0).unwrap(), (1, 0));
    }
}
