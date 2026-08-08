//! Agent-format output: one token-lean line per hit, shared by the CLI
//! and the MCP server. Shape: `m:ABCDEF [gotcha pr:mimir 06-11 ↑7] title — snippet`

use std::cmp::Reverse;
use std::collections::HashMap;

use rusqlite::Connection;

use crate::error::Result;
use crate::memory::tokens;
use crate::model::{short_uid, Node};
use crate::search::fts::is_stopword;
use crate::store;

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

/// Chars of lead-in kept before a matched term when the snippet window is
/// shifted, so the match lands in context rather than flush at the edge.
const MATCH_LEAD_CHARS: usize = 32;

/// Shortest query token worth centring on. Below this, prefix matching is
/// noise ("id" hits "identity", "video", …).
const MIN_MATCH_TOKEN: usize = 3;

/// Prefix length used to approximate the FTS5 porter stemmer: "splitting"
/// and "splits" both reduce to "split". Cheap and one-directional — it can
/// over-match slightly, which only costs us a differently-placed window.
const STEM_PREFIX: usize = 5;

/// How far into a body we hunt for a window. Recall is a hot path and the
/// scan is O(body × stems): measured at 7.4 ms on the largest chunk in a
/// real store (215k chars), enough on its own to blow a single-digit-ms
/// warm recall. 20k covers all but 83 of ~19k chunks whole; past it we
/// simply head-truncate, which is what this code replaced anyway.
const MATCH_SCAN_CHARS: usize = 20_000;

/// Content-bearing query stems, sharing `tokens`/`is_stopword` with the FTS
/// leg so the snippet can't drift from what actually matched.
fn query_stems(query: &str) -> Vec<String> {
    tokens(query)
        .into_iter()
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .filter(|t| !is_stopword(t))
        .filter(|t| t.chars().count() >= MIN_MATCH_TOKEN)
        .map(|t| t.chars().take(STEM_PREFIX).collect())
        .collect()
}

/// Lowercased char view of `text`'s first `limit` chars, index-aligned 1:1
/// with the original. Deliberately ASCII-only: `char::to_lowercase` can emit
/// more chars than it consumes (U+0130 'İ' becomes two), which desyncs every
/// index we then slice the original with. Non-ASCII uppercase simply fails to
/// match, which costs a differently-placed window — not a panic, and not a
/// wrong one.
fn lower_chars(text: &str, limit: usize) -> Vec<char> {
    text.chars()
        .take(limit)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Every char offset in `hay` where `needle` begins.
///
/// Deliberately plain substring matching, not word-anchored. Anchoring was
/// tried and measured worse: it removes the ~3% of matches that land inside
/// an unrelated word ("use" in "because"), but also drops legitimate ones
/// inside compound identifiers, costing 2pp of memory previews for no gain
/// on docs. Density scoring already tolerates the occasional stray match.
fn occurrences(hay: &[char], needle: &[char]) -> Vec<usize> {
    let Some(last) = hay.len().checked_sub(needle.len()) else {
        return Vec::new();
    };
    if needle.is_empty() {
        return Vec::new();
    }
    (0..=last)
        .filter(|&i| hay[i..].starts_with(needle))
        .collect()
}

/// Whether any offset in the sorted list `occ` places a `needle_len`-long
/// match wholly inside `[start, end)`. Binary search, so scoring a candidate
/// costs O(log occurrences) rather than a fresh O(window × needle) rescan of
/// text we already indexed — the difference between 1.6 ms and ~60 µs on a
/// repetitive body where a stem recurs thousands of times.
fn covers(occ: &[usize], needle_len: usize, start: usize, end: usize) -> bool {
    let Some(last) = end.checked_sub(needle_len) else {
        return false;
    };
    let i = occ.partition_point(|&at| at < start);
    occ.get(i).is_some_and(|&at| at <= last)
}

/// The `snippet_chars`-wide window over `text` covering the most *distinct*
/// query stems, ties going to the earliest window. Density beats
/// first-match: a window holding three of the query's terms tells the reader
/// far more than one holding the earliest single term, which is often an
/// incidental mention well before the passage that actually answers.
///
/// Returns `None` when the leading window already wins, so hits that read
/// fine today are left byte-identical.
fn match_window(text: &str, stems: &[String], snippet_chars: usize) -> Option<String> {
    if stems.is_empty() || snippet_chars == 0 {
        return None;
    }
    let lc = lower_chars(text, MATCH_SCAN_CHARS);
    // Each stem indexed once as (match length, sorted offsets), then reused
    // for both candidate generation and scoring.
    let index: Vec<(usize, Vec<usize>)> = stems
        .iter()
        .map(|s| {
            let needle: Vec<char> = s.chars().collect();
            (needle.len(), occurrences(&lc, &needle))
        })
        .collect();

    // Candidates: the head, plus a lead-in ahead of each stem occurrence.
    let mut starts = vec![0usize];
    for (_, offsets) in &index {
        starts.extend(offsets.iter().map(|at| at.saturating_sub(MATCH_LEAD_CHARS)));
    }
    // Scoring below is O(candidates × stems), so it is worth paying a sort to
    // drop duplicates — nearby stems collapse to the same start.
    starts.sort_unstable();
    starts.dedup();

    let best = starts.into_iter().max_by_key(|&s| {
        let end = (s + snippet_chars).min(lc.len());
        let hits = index
            .iter()
            .filter(|(len, offsets)| covers(offsets, *len, s, end))
            .count();
        (hits, Reverse(s))
    })?;
    if best == 0 {
        return None;
    }

    // `lower_chars` is index-aligned, so `best` indexes the original safely.
    // Collect only as far as the window plus its word-boundary lead-in can
    // reach — bodies run to hundreds of thousands of chars.
    let chars: Vec<char> = text
        .chars()
        .take(best + MATCH_LEAD_CHARS + snippet_chars)
        .collect();
    // Step forward to a word boundary so the window doesn't open mid-word.
    // Bounded by the lead-in, so this can never step past the match itself.
    let start = chars[best..]
        .iter()
        .take(MATCH_LEAD_CHARS)
        .position(|c| c.is_whitespace())
        .map_or(best, |off| best + off + 1);
    // Already bounded to `snippet_chars` by the take, so no truncation here.
    let window: String = chars[start..].iter().take(snippet_chars).collect();
    Some(format!("…{window}"))
}

/// One-line agent format for a node.
/// `project` is the display name of the node's project, if scoped.
pub fn agent_line(node: &Node, project: Option<&str>, snippet_chars: usize) -> String {
    agent_line_for_query(node, project, snippet_chars, None)
}

/// [`agent_line`], but with the search query in hand so the snippet can be
/// centred on what matched instead of always starting at the body's first
/// character. Only worth passing a query where one exists — ranked recall
/// results; a `remember` echo or an anchor trigger has none.
///
/// Why this matters for docs: a memory's title is a self-contained claim,
/// but a doc chunk's is a breadcrumb ("Doc › Section › Sub"), so the body
/// snippet is the only thing telling the reader why the hit came back. Head
/// truncation frequently spends it on the section's opening sentence.
///
/// Deliberately left alone: the breadcrumb titles themselves, which spend
/// ~75 chars on location rather than content. Fixing that belongs in
/// `index::chunker`, where they're built — and would shift ranking, since
/// `title` carries the heaviest BM25 weight. Revisit if snippet-only proves
/// insufficient.
pub fn agent_line_for_query(
    node: &Node,
    project: Option<&str>,
    snippet_chars: usize,
    query: Option<&str>,
) -> String {
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
    // Only `unsure` rides the compact line, and it costs one word. The
    // asymmetry is deliberate: this line is the one an agent reads before
    // acting, so the case worth spending tokens on is the one where acting
    // without checking would be a mistake. Printing "certain" everywhere
    // would be reassurance nobody asked for, at a cost per recall.
    let doubt = match node.confidence() {
        Some(crate::model::MemoryConfidence::Unsure) => " unsure",
        _ => "",
    };
    let mut line = format!("{id} [{tag}{scope} {date}{uses}{doubt}] {title}");
    if let Some(body) = node.body.as_deref() {
        let flat = collapse_ws(body);
        let stems = query.map(query_stems).unwrap_or_default();
        // Centre on the match when there is one; otherwise head-truncate.
        let excerpt = |text: &str| {
            match_window(text, &stems, snippet_chars)
                .unwrap_or_else(|| truncate_chars(text, snippet_chars))
        };
        // Skip the part of the body the (possibly truncated) title covers.
        let covered = title.trim_end_matches('…');
        let rest = flat.strip_prefix(covered).map(str::trim_start);
        match rest {
            Some("") => {} // title covers everything
            Some(rest) => {
                line.push_str(" — ");
                line.push_str(&excerpt(rest));
            }
            None if flat.len() > title.len() => {
                line.push_str(" — ");
                line.push_str(&excerpt(&flat));
            }
            None => {}
        }
    }
    line
}

/// Multi-line full record: header, title, body, tags, edges.
pub fn full_record(
    conn: &Connection,
    node: &Node,
    projects: &HashMap<i64, String>,
) -> Result<String> {
    let project = node
        .project_id
        .and_then(|id| projects.get(&id))
        .map(String::as_str);
    let mut out = format!(
        "{} {}",
        short_uid(node.kind, &node.uid),
        node.subkind.as_deref().unwrap_or(node.kind.as_str())
    );
    if let Some(p) = project {
        out.push_str(&format!(" pr:{p}"));
    }
    out.push_str(&format!(" created {}", full_date(node.created_at)));
    if node.updated_at != node.created_at {
        out.push_str(&format!(" updated {}", full_date(node.updated_at)));
    }
    if node.access_count > 0 {
        out.push_str(&format!(" ↑{}", node.access_count));
    }
    if let Some(t) = &node.title {
        out.push('\n');
        out.push_str(t);
    }
    if let Some(b) = &node.body {
        if node.title.as_deref() != Some(b.trim()) {
            out.push('\n');
            out.push_str(b);
        }
    }
    if !node.tags_text.is_empty() {
        out.push_str(&format!("\ntags: {}", node.tags_text));
    }
    for edge in store::edges_of(conn, node.id)? {
        let (arrow, other_id) = if edge.src == node.id {
            ("→", edge.dst)
        } else {
            ("←", edge.src)
        };
        if let Ok(other) = store::get_node(conn, other_id) {
            out.push_str(&format!(
                "\n{arrow} {} {} {}",
                edge.rel,
                short_uid(other.kind, &other.uid),
                other.title.as_deref().unwrap_or("")
            ));
        }
    }
    // Only annotate memories, and only when there is something to say. A
    // "grounded" badge on a doc chunk is noise (of course it is — it *is*
    // the artifact), and stamping "ungrounded" on every ordinary note would
    // read as a defect rather than the normal case that it is.
    if node.kind == crate::model::Kind::Memory {
        // The three axes a reader has to keep apart, and which a single
        // `strength` number silently merges: what the author claimed, how
        // much the store has used it, and whether the link it rests on
        // still holds. Only printed when declared — see MemoryConfidence
        // on why absent is not the middle.
        if let Some(c) = node.confidence() {
            out.push_str(&format!(
                "\nconfidence: {c} (author-declared; not a ranking signal)"
            ));
        }
        if let crate::grounding::Grounding::Stale { kind, label } =
            crate::grounding::grounding(conn, node.id)?
        {
            out.push_str(&format!(
                "\n! grounding stale: {} {label} is no longer indexed \
                 (the note may still be right; nothing has revisited it since)",
                kind.as_str()
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Kind, NewNode};

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

    /// Only doubt is worth the tokens on the line an agent acts from.
    /// Certainty is the assumption already; spending a word per recall to
    /// restate it would be pure cost.
    #[test]
    fn only_unsure_reaches_the_compact_line() {
        let conn = crate::db::open_in_memory().unwrap();
        let make = |level: Option<crate::model::MemoryConfidence>| {
            let mut new = NewNode::new(Kind::Memory);
            new.subkind = Some("note".into());
            new.title = Some("dns is the cause".into());
            new.body = Some("dns is the cause".into());
            let node = store::insert_node(&conn, new).unwrap();
            if let Some(l) = level {
                store::set_confidence(&conn, node.id, l).unwrap();
            }
            let node = store::get_node(&conn, node.id).unwrap();
            agent_line(&node, None, 120)
        };
        assert!(make(Some(crate::model::MemoryConfidence::Unsure)).contains("unsure"));
        assert!(!make(Some(crate::model::MemoryConfidence::Certain)).contains("certain"));
        assert!(!make(Some(crate::model::MemoryConfidence::Likely)).contains("likely"));
        let bare = make(None);
        assert!(!bare.contains("unsure") && !bare.contains("certain"));
    }

    /// Confidence must be readable in full, and must be legibly the
    /// author's claim rather than something Mimir computed.
    #[test]
    fn full_record_attributes_confidence_to_the_author() {
        let conn = crate::db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("insight".into());
        new.title = Some("the flake is dns".into());
        new.body = Some("the flake is dns".into());
        let node = store::insert_node(&conn, new).unwrap();
        store::set_confidence(&conn, node.id, crate::model::MemoryConfidence::Unsure).unwrap();
        let node = store::get_node(&conn, node.id).unwrap();
        let out = full_record(&conn, &node, &HashMap::new()).unwrap();
        assert!(out.contains("confidence: unsure"), "{out}");
        assert!(out.contains("author-declared"), "{out}");
        assert!(out.contains("not a ranking signal"), "{out}");
    }

    /// The separation that makes this worth having: recall a guess as much
    /// as you like and it is still marked a guess. `strength` moves,
    /// `confidence` does not — one records use, the other a claim.
    #[test]
    fn recall_raises_strength_without_touching_confidence() {
        let conn = crate::db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Memory);
        new.subkind = Some("insight".into());
        new.title = Some("probably a cache issue".into());
        new.body = Some("probably a cache issue".into());
        let node = store::insert_node(&conn, new).unwrap();
        store::set_confidence(&conn, node.id, crate::model::MemoryConfidence::Unsure).unwrap();

        let before = store::get_node(&conn, node.id).unwrap();
        for _ in 0..8 {
            crate::learn::record_opened(&conn, node.id).unwrap();
        }
        let after = store::get_node(&conn, node.id).unwrap();

        assert!(
            after.strength > before.strength,
            "being used should move strength ({} -> {})",
            before.strength,
            after.strength
        );
        assert_eq!(
            after.confidence(),
            Some(crate::model::MemoryConfidence::Unsure),
            "being used must NOT promote a guess"
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

    /// The defect this exists to prevent: a doc chunk whose breadcrumb title
    /// says only where it lives, and whose matching text sits past the head
    /// window — so plain truncation shows nothing about why it was returned.
    fn breadcrumb_chunk() -> Node {
        let conn = crate::db::open_in_memory().unwrap();
        let mut new = NewNode::new(Kind::Chunk);
        new.title = Some("proxy › mimir proxy › What it actually does".into());
        new.body = Some(format!(
            "proxy › mimir proxy › What it actually does {}the proxy adds ephemeral \
             breakpoints on the system prompt and the last message.",
            "You cannot compress text and have the API bill fewer tokens. ".repeat(4)
        ));
        store::insert_node(&conn, new).unwrap()
    }

    #[test]
    fn query_aware_snippet_centres_on_the_match() {
        let node = breadcrumb_chunk();

        let blind = agent_line(&node, None, 120);
        assert!(
            !blind.contains("breakpoints"),
            "head truncation should miss the match: {blind}"
        );

        let aware = agent_line_for_query(&node, None, 120, Some("prompt cache breakpoints"));
        assert!(
            aware.contains("breakpoints"),
            "snippet should be centred on the match: {aware}"
        );
        assert!(aware.contains("— …"), "shifted window is elided: {aware}");
    }

    #[test]
    fn query_aware_snippet_falls_back_when_nothing_matches() {
        let node = breadcrumb_chunk();
        assert_eq!(
            agent_line_for_query(&node, None, 120, Some("kubernetes ingress")),
            agent_line(&node, None, 120),
            "a query with no lexical match must not change the line"
        );
    }

    #[test]
    fn query_aware_snippet_leaves_early_matches_alone() {
        let node = breadcrumb_chunk();
        // "compress" is inside the head window, which already shows it.
        assert_eq!(
            agent_line_for_query(&node, None, 120, Some("compress")),
            agent_line(&node, None, 120),
        );
    }

    #[test]
    fn query_stems_drop_stopwords_and_short_tokens() {
        assert_eq!(query_stems("how is it   split"), vec!["split"]);
        // Stemmed to a shared prefix, so "splitting" finds "splits".
        assert_eq!(query_stems("splitting"), vec!["split"]);
    }

    #[test]
    fn match_window_is_char_safe_on_multibyte_text() {
        let text = format!("{}naïve café — chunking rules apply", "ø".repeat(200));
        let stems = query_stems("chunking");
        let win = match_window(&text, &stems, 40).expect("match past the window");
        assert!(win.contains("chunking"), "{win}");
        assert!(win.starts_with('…'), "{win}");
    }

    #[test]
    fn match_window_survives_chars_that_grow_when_lowercased() {
        // U+0130 lowercases to TWO chars, so a char index computed in the
        // lowercased text can run past the original's char count.
        let text = format!("{}chunking rules apply", "İ".repeat(200));
        let stems = query_stems("chunking");
        let win = match_window(&text, &stems, 40).expect("match past the window");
        assert!(win.contains("chunking"), "{win}");
    }

    #[test]
    fn match_window_prefers_density_over_the_earliest_match() {
        // "cache" appears early and alone; the passage that answers the
        // query — all three terms together — sits much later.
        let text = format!(
            "the cache is warm. {} prompt cache breakpoints are set here.",
            "filler prose that says nothing useful. ".repeat(12)
        );
        let stems = query_stems("prompt cache breakpoints");
        let win = match_window(&text, &stems, 80).expect("a denser window exists");
        assert!(win.contains("breakpoints"), "{win}");
        assert!(win.contains("prompt"), "{win}");
    }

    #[test]
    fn match_window_keeps_the_head_when_it_is_already_densest() {
        let text = format!(
            "prompt cache breakpoints explained up front. {}",
            "later filler with no query terms at all. ".repeat(12)
        );
        let stems = query_stems("prompt cache breakpoints");
        assert_eq!(
            match_window(&text, &stems, 80),
            None,
            "head already wins — must not churn the line"
        );
    }
}
