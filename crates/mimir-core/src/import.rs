//! Importers for the systems Mimir replaces, plus a JSONL export.
//!
//! Imports go through `memory::remember` (force=false), so re-running an
//! import never duplicates: near-dups are skipped and counted.

use rusqlite::Connection;

use crate::error::{Error, Result};
use crate::memory::{self, Remember, RememberOutcome};
use crate::model::MemoryType;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImportStats {
    pub imported: usize,
    pub skipped_duplicates: usize,
}

/// One parsed memory ready for insertion.
struct Incoming {
    text: String,
    mtype: MemoryType,
    tags: Vec<String>,
    /// Original capture time (preserves honest decay).
    created_at: Option<i64>,
}

fn store_all(conn: &Connection, items: Vec<Incoming>) -> Result<ImportStats> {
    let mut stats = ImportStats::default();
    for item in items {
        let outcome = memory::remember(
            conn,
            Remember {
                text: item.text,
                mtype: item.mtype,
                tags: item.tags,
                project_id: None, // imports land in the global scope
                force: false,
            },
        )?;
        match outcome {
            RememberOutcome::Created(node) => {
                if let Some(at) = item.created_at {
                    conn.execute(
                        "UPDATE node SET created_at = ?2, updated_at = ?2 WHERE id = ?1",
                        rusqlite::params![node.id, at],
                    )?;
                }
                stats.imported += 1;
            }
            RememberOutcome::Duplicate(_) => stats.skipped_duplicates += 1,
        }
    }
    Ok(stats)
}

// ---------- OpenBrain ----------

/// Parse an OpenBrain `list_thoughts` text export:
/// `N. [M/D/YYYY] (type - topic, topic)\n   content…`
pub fn openbrain(conn: &Connection, text: &str) -> Result<ImportStats> {
    let re_header = |line: &str| -> Option<(i64, MemoryType, Vec<String>)> {
        // "12. [6/4/2026] (observation - go2rtc, WebSocket)"
        let rest = line.trim_start();
        let dot = rest.find(". [")?;
        rest[..dot].parse::<usize>().ok()?;
        let after = &rest[dot + 3..];
        let close = after.find(']')?;
        let date = parse_us_date(&after[..close]);
        let paren = after[close..].find('(')? + close;
        let end = after[paren..].find(')')? + paren;
        let inside = &after[paren + 1..end];
        let (ty, topics) = match inside.split_once(" - ") {
            Some((t, rest)) => (t.trim(), rest),
            None => (inside.trim(), ""),
        };
        let mtype = match ty {
            "observation" => MemoryType::Insight,
            "reference" => MemoryType::Note,
            "idea" => MemoryType::Idea,
            "task" => MemoryType::Note,
            "person_note" => MemoryType::Person,
            _ => MemoryType::Note,
        };
        let tags = topics
            .split(',')
            .map(|t| t.trim().replace(' ', "-").to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        Some((date.unwrap_or(0), mtype, tags))
    };

    let mut items: Vec<Incoming> = Vec::new();
    let mut current: Option<(Incoming, Vec<String>)> = None;
    for line in text.lines() {
        if let Some((created, mtype, tags)) = re_header(line) {
            if let Some((mut item, body)) = current.take() {
                item.text = body.join("\n").trim().to_string();
                if !item.text.is_empty() {
                    items.push(item);
                }
            }
            current = Some((
                Incoming {
                    text: String::new(),
                    mtype,
                    tags,
                    created_at: if created > 0 { Some(created) } else { None },
                },
                Vec::new(),
            ));
        } else if let Some((_, body)) = current.as_mut() {
            body.push(line.strip_prefix("   ").unwrap_or(line).to_string());
        }
    }
    if let Some((mut item, body)) = current.take() {
        item.text = body.join("\n").trim().to_string();
        if !item.text.is_empty() {
            items.push(item);
        }
    }
    if items.is_empty() {
        return Err(Error::Invalid(
            "no thoughts found — expected OpenBrain list_thoughts text format".into(),
        ));
    }
    store_all(conn, items)
}

/// "6/4/2026" → unix.
fn parse_us_date(s: &str) -> Option<i64> {
    let mut parts = s.trim().split('/');
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    let y: i64 = parts.next()?.parse().ok()?;
    // days_from_civil (Hinnant), inverse of format::civil_from_days.
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

// ---------- Claude auto-memory ----------

/// Import a Claude Code auto-memory directory: one markdown file per
/// memory with YAML-ish frontmatter (`name`, `description`,
/// `metadata.type`). MEMORY.md (the index) is skipped.
pub fn claude_memory(conn: &Connection, dir: &std::path::Path) -> Result<ImportStats> {
    let mut items = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".md") || name == "MEMORY.md" {
            continue;
        }
        let raw = std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        let (kind, body) = split_frontmatter(&raw);
        let mtype = match kind.as_deref() {
            Some("feedback") => MemoryType::Insight,
            Some("user") => MemoryType::Person,
            _ => MemoryType::Note, // project, reference
        };
        let text = body.trim();
        if text.is_empty() {
            continue;
        }
        items.push(Incoming {
            text: text.to_string(),
            mtype,
            tags: vec![
                "claude-memory".into(),
                name.trim_end_matches(".md").to_lowercase(),
            ],
            created_at: None,
        });
    }
    if items.is_empty() {
        return Err(Error::Invalid(format!(
            "no memory files found in {}",
            dir.display()
        )));
    }
    store_all(conn, items)
}

/// Returns (metadata.type, body-after-frontmatter).
fn split_frontmatter(raw: &str) -> (Option<String>, String) {
    let Some(rest) = raw.strip_prefix("---") else {
        return (None, raw.to_string());
    };
    let Some(end) = rest.find("\n---") else {
        return (None, raw.to_string());
    };
    let front = &rest[..end];
    let body = rest[end + 4..].to_string();
    let ty = front
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("type:"))
        .map(|v| v.trim().to_string());
    (ty, body)
}

// ---------- QMD ----------

/// Register the collections from a qmd index.yml (two-level YAML subset:
/// `collections:` → `name:` → `path:`). Indexing still happens via
/// `mimir index`.
pub fn qmd_collections(yml: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    let mut in_collections = false;
    for line in yml.lines() {
        let trimmed = line.trim_end();
        if trimmed == "collections:" {
            in_collections = true;
            continue;
        }
        if !in_collections || trimmed.trim().is_empty() {
            continue;
        }
        let indent = trimmed.len() - trimmed.trim_start().len();
        let body = trimmed.trim();
        if indent == 2 && body.ends_with(':') {
            current = Some(body.trim_end_matches(':').to_string());
        } else if indent >= 4 {
            if let (Some(name), Some(path)) = (&current, body.strip_prefix("path:")) {
                out.push((name.clone(), path.trim().to_string()));
            }
        } else if indent == 0 {
            in_collections = false;
        }
    }
    out
}

// ---------- Export ----------

/// Everything (except embeddings, which are derived) as JSONL:
/// node lines then edge lines, suitable for backup or migration.
pub fn export_jsonl(conn: &Connection, out: &mut dyn std::io::Write) -> Result<usize> {
    use crate::store::{row_to_node, NODE_COLS};
    let mut count = 0usize;
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLS} FROM node WHERE deleted_at IS NULL ORDER BY id"
    ))?;
    let nodes: Vec<crate::model::Node> = stmt
        .query_map([], row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for n in &nodes {
        let line = serde_json::json!({
            "type": "node", "uid": n.uid, "kind": n.kind.as_str(),
            "subkind": n.subkind, "title": n.title, "body": n.body,
            "path": n.path, "lang": n.lang, "tags": n.tags(),
            "span": n.span_start.zip(n.span_end).map(|(a, b)| [a, b]),
            "meta": n.meta, "created_at": n.created_at,
            "updated_at": n.updated_at, "strength": n.strength,
            "pinned": n.pinned, "access_count": n.access_count,
        });
        writeln!(out, "{line}").map_err(|e| Error::Invalid(format!("write: {e}")))?;
        count += 1;
    }
    let mut stmt = conn.prepare(
        "SELECT s.uid, d.uid, e.rel, e.weight FROM edge e
         JOIN node s ON s.id = e.src JOIN node d ON d.id = e.dst
         WHERE s.deleted_at IS NULL AND d.deleted_at IS NULL ORDER BY e.id",
    )?;
    let edges: Vec<(String, String, String, f64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    for (src, dst, rel, weight) in edges {
        let line = serde_json::json!({
            "type": "edge", "src": src, "dst": dst, "rel": rel, "weight": weight,
        });
        writeln!(out, "{line}").map_err(|e| Error::Invalid(format!("write: {e}")))?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[test]
    fn openbrain_round_trip() {
        let conn = db::open_in_memory().unwrap();
        let export = "104 recent thought(s):\n\n\
1. [6/11/2026] (observation - rust, mcp)\n   rmcp gotcha: ServerInfo is non_exhaustive.\n   Use the builder instead.\n\n\
2. [6/4/2026] (reference - go2rtc, WebSocket)\n   go2rtc /api/ws protocol notes.\n\n\
3. [5/1/2026] (idea)\n   Build a thing that does stuff.\n";
        let stats = openbrain(&conn, export).unwrap();
        assert_eq!(stats.imported, 3);
        // Re-import: everything is a duplicate.
        let again = openbrain(&conn, export).unwrap();
        assert_eq!(again.imported, 0);
        assert_eq!(again.skipped_duplicates, 3);

        // Types, tags, dates mapped.
        let (subkind, tags, created): (String, String, i64) = conn
            .query_row(
                "SELECT subkind, tags_text, created_at FROM node
                 WHERE kind = 'memory' AND body LIKE 'rmcp gotcha%'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(subkind, "insight");
        assert!(tags.contains("rust") && tags.contains("mcp"));
        assert_eq!(crate::format::full_date(created), "2026-06-11");
    }

    #[test]
    fn claude_memory_import() {
        let conn = db::open_in_memory().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("MEMORY.md"), "# Memory index\n- skip me\n").unwrap();
        std::fs::write(
            dir.path().join("prefers-rebase.md"),
            "---\nname: prefers-rebase\ndescription: d\nmetadata:\n  type: feedback\n---\n\nThe user prefers rebase over merge.\n",
        )
        .unwrap();
        let stats = claude_memory(&conn, dir.path()).unwrap();
        assert_eq!(stats.imported, 1);
        let (subkind, tags): (String, String) = conn
            .query_row(
                "SELECT subkind, tags_text FROM node WHERE kind = 'memory'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(subkind, "insight");
        assert!(tags.contains("prefers-rebase"));
    }

    #[test]
    fn qmd_yaml_subset() {
        let yml = "collections:\n  book1:\n    path: /home/x/drafts\n    pattern: ch*.md\n  notes:\n    path: /home/x/notes\n    pattern: \"**/*.md\"\n";
        let cols = qmd_collections(yml);
        assert_eq!(
            cols,
            vec![
                ("book1".to_string(), "/home/x/drafts".to_string()),
                ("notes".to_string(), "/home/x/notes".to_string())
            ]
        );
    }

    #[test]
    fn export_emits_nodes_and_edges() {
        let conn = db::open_in_memory().unwrap();
        let a = crate::store::insert_node(&conn, {
            let mut n = crate::model::NewNode::new(crate::model::Kind::Memory);
            n.title = Some("a".into());
            n.body = Some("a".into());
            n
        })
        .unwrap();
        let b = crate::store::insert_node(&conn, {
            let mut n = crate::model::NewNode::new(crate::model::Kind::Memory);
            n.title = Some("b".into());
            n.body = Some("b".into());
            n
        })
        .unwrap();
        crate::store::link(&conn, a.id, b.id, crate::model::Rel::Relates, 1.0).unwrap();
        let mut buf = Vec::new();
        let count = export_jsonl(&conn, &mut buf).unwrap();
        assert_eq!(count, 3);
        let lines: Vec<serde_json::Value> = String::from_utf8(buf)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["type"], "node");
        assert_eq!(lines[2]["type"], "edge");
        assert_eq!(lines[2]["rel"], "relates");
    }
}
