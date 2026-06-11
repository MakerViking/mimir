//! Agent-format output: one token-lean line per hit, shared by the CLI
//! and the MCP server. Shape: `m:ABCDEF [gotcha pr:mimir 06-11 ↑7] title — snippet`

use crate::model::{short_uid, Node};

/// `MM-DD` of a unix timestamp (UTC). Compact by design: recall output is
/// read by agents where every token counts; the year lives in `get`.
pub fn month_day(unix: i64) -> String {
    let (_, m, d) = civil_from_days(unix.div_euclid(86_400));
    format!("{m:02}-{d:02}")
}

/// `YYYY-MM-DD` of a unix timestamp (UTC).
pub fn full_date(unix: i64) -> String {
    let (y, m, d) = civil_from_days(unix.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's civil_from_days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// One-line agent format for a node.
/// `project` is the display name of the node's project, if scoped.
pub fn agent_line(node: &Node, project: Option<&str>, snippet_chars: usize) -> String {
    let id = short_uid(node.kind, &node.uid);
    let tag = node.subkind.as_deref().unwrap_or(node.kind.as_str());
    let scope = project.map(|p| format!(" pr:{p}")).unwrap_or_default();
    let date = month_day(node.created_at);
    let uses = if node.access_count > 0 {
        format!(" ↑{}", node.access_count)
    } else {
        String::new()
    };
    let title = node.title.as_deref().unwrap_or("(untitled)");
    let mut line = format!("{id} [{tag}{scope} {date}{uses}] {title}");
    if let Some(body) = node.body.as_deref() {
        let flat = collapse_ws(body);
        // Skip the part of the body the (possibly truncated) title covers.
        let covered = title.trim_end_matches('…');
        let rest = flat.strip_prefix(covered).map(str::trim_start);
        match rest {
            Some("") => {} // title covers everything
            Some(rest) => {
                line.push_str(" — ");
                line.push_str(&truncate_chars(rest, snippet_chars));
            }
            None if flat.len() > title.len() => {
                line.push_str(" — ");
                line.push_str(&truncate_chars(&flat, snippet_chars));
            }
            None => {}
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode};
    use crate::store;

    #[test]
    fn dates() {
        assert_eq!(full_date(0), "1970-01-01");
        assert_eq!(full_date(1_780_000_000), "2026-05-28");
        assert_eq!(month_day(1_780_000_000), "05-28");
    }

    #[test]
    fn agent_line_shape() {
        let conn = crate::db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("gotcha".into());
        new.title = Some("watch out".into());
        new.body = Some("watch out\nfor the    thing".into());
        let node = store::insert_node(&conn, new).unwrap();
        let line = agent_line(&node, Some("mimir"), 120);
        let tail = &node.uid[node.uid.len() - 6..];
        assert_eq!(
            line,
            format!(
                "m:{tail} [gotcha pr:mimir {}] watch out — for the thing",
                month_day(node.created_at)
            )
        );
    }

    #[test]
    fn agent_line_skips_redundant_snippet() {
        let conn = crate::db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("note".into());
        new.title = Some("short".into());
        new.body = Some("short".into());
        let node = store::insert_node(&conn, new).unwrap();
        let line = agent_line(&node, None, 120);
        assert!(!line.contains('—'), "no snippet expected: {line}");
    }
}
