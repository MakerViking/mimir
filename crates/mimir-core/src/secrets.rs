//! High-precision secret detection for memory capture.
//!
//! Only patterns with a near-zero false-positive rate are matched here —
//! a generic assignment/entropy heuristic (`key = "..."`-style scanning)
//! was deliberately excluded: it's the one pattern class prone to false
//! positives (flagging ordinary prose or code that merely mentions
//! "api key" / "token" / "password" without an actual secret value), and
//! false positives are worse than a missed detection for a capture guard
//! that must stay out of the way of normal notes.

use once_cell::sync::Lazy;
use regex::Regex;

/// Scan `text` for a high-precision secret pattern. Returns the matched
/// pattern's human label (with article, e.g. "an AWS access key") so it
/// slots into the `Error::Secret` message grammatically, or `None` if
/// nothing looks like a secret.
pub fn scan(text: &str) -> Option<&'static str> {
    PATTERNS
        .iter()
        .find(|p| p.re.is_match(text))
        .map(|p| p.kind)
}

/// The single definition of which parts of a capture get scanned: body
/// text, tags, and `fires_when` trigger phrases — all persisted, and
/// `fires_when` is later auto-injected into prompts. Every guarded capture
/// path (CLI `remember`, CLI `edit`, MCP `remember`) calls this one choke
/// point; `mimir import` and sync replication deliberately do not (restoring
/// or replicating your own store is not a capture). Anchors are excluded:
/// they are glob patterns, not free text.
pub fn scan_capture(text: &str, tags: &[String], fires_when: &[String]) -> Option<&'static str> {
    scan(text)
        .or_else(|| scan(&tags.join(" ")))
        .or_else(|| scan(&fires_when.join(" ")))
}

struct Pattern {
    kind: &'static str,
    re: Regex,
}

static PATTERNS: Lazy<Vec<Pattern>> = Lazy::new(|| {
    vec![
        Pattern {
            kind: "a private key block",
            re: Regex::new(r"-----BEGIN [^\n]*PRIVATE KEY[^\n]*").unwrap(),
        },
        Pattern {
            kind: "an AWS access key",
            // `{16,}` (not `{16}`): with a fixed count, a key followed by more
            // word characters puts the trailing `\b` between two word chars
            // and the whole match fails — an open-ended run always ends on a
            // real boundary, so keys embedded in longer tokens still match.
            re: Regex::new(r"\b(AKIA|ASIA)[0-9A-Z]{16,}\b").unwrap(),
        },
        Pattern {
            kind: "a GitHub token",
            re: Regex::new(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b").unwrap(),
        },
        Pattern {
            kind: "a GitHub token",
            re: Regex::new(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b").unwrap(),
        },
        Pattern {
            kind: "a Slack token",
            re: Regex::new(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b").unwrap(),
        },
        Pattern {
            kind: "a JWT",
            re: Regex::new(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
                .unwrap(),
        },
    ]
});

/// Which capture path a refusal came from, for the ledger. Not free text,
/// so a surface can't be silently renamed out of existing rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    CliRemember,
    CliEdit,
    McpRemember,
}

impl Surface {
    pub fn label(self) -> &'static str {
        match self {
            Surface::CliRemember => "cli:remember",
            Surface::CliEdit => "cli:edit",
            Surface::McpRemember => "mcp:remember",
        }
    }
}

/// Fingerprint of everything [`scan_capture`] looked at, so a secret
/// smuggled through tags is recognized on the next offer just as well as
/// one in the body. Same normalization as `memory::content_hash`, so a
/// reformat of the same value still collides.
pub fn capture_hash(text: &str, tags: &[String], fires_when: &[String]) -> Vec<u8> {
    crate::memory::content_hash(&format!(
        "{text}\u{1f}{}\u{1f}{}",
        tags.join(" "),
        fires_when.join(" ")
    ))
}

/// One row of the refusal ledger.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub kind: String,
    pub surface: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub count: i64,
}

/// Record that a capture was turned away, keeping a fingerprint of the
/// text and never the text.
///
/// The hash is the same normalized blake3 `memory::content_hash` uses, so
/// the ledger recognizes the identical value offered again through a
/// different surface, or after a reformat. What it deliberately cannot do
/// is give the value back: there is no path from these rows to the secret,
/// which is the entire reason the ledger is safe to keep at all. A
/// rejection record that stored what it rejected would just be a second
/// copy of the leak, in a table nobody thinks to audit.
///
/// Best-effort by design: a failure to write the ledger must never turn a
/// successful refusal into an error, because the refusal is the part that
/// protects the user and the bookkeeping is not.
pub fn record_refusal(
    conn: &rusqlite::Connection,
    text_hash: &[u8],
    kind: &str,
    surface: Surface,
    now: i64,
) {
    let res = conn.execute(
        "INSERT INTO refusal (hash, kind, surface, first_seen, last_seen, count)
         VALUES (?1, ?2, ?3, ?4, ?4, 1)
         ON CONFLICT(hash) DO UPDATE SET
             last_seen = ?4,
             count = count + 1,
             kind = ?2,
             surface = ?3",
        rusqlite::params![text_hash, kind, surface.label(), now],
    );
    if let Err(err) = res {
        tracing::warn!(%err, "could not record refusal in the ledger");
    }
}

/// Ledger rows, most recently offered first.
pub fn refusals(conn: &rusqlite::Connection, limit: usize) -> crate::error::Result<Vec<Refusal>> {
    let mut stmt = conn.prepare(
        "SELECT kind, surface, first_seen, last_seen, count FROM refusal
         ORDER BY last_seen DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit as i64], |r| {
        Ok(Refusal {
            kind: r.get(0)?,
            surface: r.get(1)?,
            first_seen: r.get(2)?,
            last_seen: r.get(3)?,
            count: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// (distinct values refused, total offers) since `since`. The two differ
/// exactly when something is being retried, which is the signal worth
/// surfacing: one secret offered forty times is an agent in a loop, not
/// forty careless captures.
pub fn refusal_counts(conn: &rusqlite::Connection, since: i64) -> crate::error::Result<(i64, i64)> {
    Ok(conn.query_row(
        "SELECT count(*), COALESCE(sum(count), 0) FROM refusal WHERE last_seen >= ?1",
        [since],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole ledger exists for: a record that something
    /// was refused must not be a second copy of the thing refused. If this
    /// ever fails, the ledger is a liability rather than an audit trail —
    /// a table of plaintext secrets that nobody thinks to look in.
    #[test]
    fn ledger_never_contains_the_refused_value() {
        let conn = crate::db::open_in_memory().unwrap();
        let secret = "AKIAIOSFODNN7EXAMPLE";
        let text = format!("deploy key is {secret}");
        let kind = scan(&text).expect("fixture must look like a secret");
        record_refusal(
            &conn,
            &capture_hash(&text, &[], &[]),
            kind,
            Surface::CliRemember,
            1_700_000_000,
        );

        // Sweep every text value in every column of every row.
        let mut found = Vec::new();
        let tables: Vec<String> = {
            let mut s = conn
                .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
                .unwrap();
            let v = s
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            v
        };
        for t in tables {
            let mut stmt = match conn.prepare(&format!("SELECT * FROM \"{t}\"")) {
                Ok(s) => s,
                Err(_) => continue, // virtual/shadow tables we can't SELECT *
            };
            let cols = stmt.column_count();
            let mut rows = stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                for i in 0..cols {
                    if let Ok(v) = row.get::<_, String>(i) {
                        if v.contains(secret) {
                            found.push(format!("{t}[{i}]"));
                        }
                    }
                }
            }
        }
        assert!(
            found.is_empty(),
            "refused secret leaked into the store at {found:?}"
        );

        // ...and the row is still genuinely there.
        let (distinct, offers) = refusal_counts(&conn, 0).unwrap();
        assert_eq!((distinct, offers), (1, 1));
        assert_eq!(refusals(&conn, 10).unwrap()[0].kind, "an AWS access key");
    }

    /// A looping agent must show up as one row with a high count, not as a
    /// flood of rows — that difference is the entire diagnostic value.
    #[test]
    fn repeat_offers_increment_rather_than_insert() {
        let conn = crate::db::open_in_memory().unwrap();
        let text = "token ghp_000000000000000000000000000000000000";
        let kind = scan(text).expect("fixture must look like a secret");
        let hash = capture_hash(text, &[], &[]);
        for i in 0..5 {
            record_refusal(&conn, &hash, kind, Surface::McpRemember, 1_700_000_000 + i);
        }
        let (distinct, offers) = refusal_counts(&conn, 0).unwrap();
        assert_eq!((distinct, offers), (1, 5), "one value, five attempts");
        let row = &refusals(&conn, 10).unwrap()[0];
        assert_eq!(row.first_seen, 1_700_000_000);
        assert_eq!(row.last_seen, 1_700_000_004);
    }

    /// The same value reformatted, or moved from the body into a tag, is
    /// still the same value — otherwise a retry loop looks like fresh
    /// offers and the count stops meaning anything.
    #[test]
    fn fingerprint_survives_reformatting_and_tags() {
        let secret = "AKIAIOSFODNN7EXAMPLE";
        let a = capture_hash(&format!("deploy key is {secret}"), &[], &[]);
        let b = capture_hash(&format!("Deploy   key  is  {secret}"), &[], &[]);
        assert_eq!(
            a, b,
            "whitespace and case must not create a new fingerprint"
        );

        let in_tag = capture_hash("deploy notes", &[secret.to_string()], &[]);
        let in_body = capture_hash("deploy notes", &[], &[]);
        assert_ne!(
            in_tag, in_body,
            "a secret in tags must be fingerprinted too"
        );
    }

    #[test]
    fn detects_private_key_block() {
        let text =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK...\n-----END RSA PRIVATE KEY-----";
        let f = scan(text).expect("should detect private key block");
        assert_eq!(f, "a private key block");
    }

    #[test]
    fn detects_aws_access_key() {
        let f =
            scan("aws_access_key_id = AKIAABCDEFGHIJKLMNOP").expect("should detect AWS access key");
        assert_eq!(f, "an AWS access key");
    }

    #[test]
    fn detects_aws_key_followed_by_more_word_chars() {
        // A key concatenated with more alphanumerics (missing delimiter,
        // embedded in a longer opaque token) must still match.
        assert_eq!(
            scan("key: AKIAABCDEFGHIJKLMNOPQRST end"),
            Some("an AWS access key")
        );
    }

    #[test]
    fn detects_github_token() {
        let f = scan("token: ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1234")
            .expect("should detect GitHub token");
        assert_eq!(f, "a GitHub token");
    }

    #[test]
    fn detects_github_pat() {
        let f = scan("github_pat_11ABCDEFG0FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE")
            .expect("should detect GitHub fine-grained PAT");
        assert_eq!(f, "a GitHub token");
    }

    /// Assembled at runtime: a Slack-shaped literal in source trips GitHub
    /// push protection (and downstream forks' scanners) even as a fixture.
    fn fake_slack_token() -> String {
        format!("xoxb-{}-fakefaketokenvalue", "1234567890")
    }

    #[test]
    fn detects_slack_token() {
        let f = scan(&format!("SLACK_BOT_TOKEN={}", fake_slack_token()))
            .expect("should detect Slack token");
        assert_eq!(f, "a Slack token");
    }

    #[test]
    fn detects_jwt() {
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let f = scan(&format!("Authorization: Bearer {jwt}")).expect("should detect JWT");
        assert_eq!(f, "a JWT");
    }

    #[test]
    fn ignores_existing_scram_fixture() {
        assert!(scan("SCRAM auth rejects non-ASCII passwords").is_none());
    }

    #[test]
    fn ignores_prose_mentioning_secrets_without_values() {
        assert!(
            scan("remember to rotate the api keys and tokens before the password expires")
                .is_none()
        );
    }

    #[test]
    fn ignores_short_base64_ish_word() {
        assert!(scan("the value was QUJDREVGRw== in the log").is_none());
    }

    #[test]
    fn ignores_eyj_without_full_jwt_shape() {
        assert!(scan("the config key eyJust_a_prefix has no dots").is_none());
    }

    #[test]
    fn scan_capture_covers_tags_and_fires_when() {
        let benign = "note about the deploy process";
        let no_tags: &[String] = &[];
        assert!(scan_capture(benign, no_tags, no_tags).is_none());
        let tags = vec!["AKIAABCDEFGHIJKLMNOP".to_string(), "deploy".to_string()];
        assert_eq!(
            scan_capture(benign, &tags, no_tags),
            Some("an AWS access key")
        );
        let fires = vec![fake_slack_token()];
        assert_eq!(scan_capture(benign, no_tags, &fires), Some("a Slack token"));
    }
}
