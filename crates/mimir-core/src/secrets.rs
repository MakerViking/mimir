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

#[cfg(test)]
mod tests {
    use super::*;

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
