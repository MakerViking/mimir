//! Markdown chunker: heading-aware, fence-safe, sized for retrieval.
//!
//! Sections split at H1–H3; each chunk carries a breadcrumb title
//! ("Doc › Section › Sub") and a 1-based line span. Fenced code blocks
//! are atomic — never split, never carried into overlap.

use std::ops::Range;

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};

/// Sizing in words (≈ 0.75 tokens/word, so 280 words ≈ 250–400 tokens).
const TARGET_WORDS: usize = 280;
const MAX_WORDS: usize = 400;
const MIN_WORDS: usize = 60;
const OVERLAP_WORDS: usize = 40;

#[derive(Debug, Clone)]
pub struct DocChunk {
    /// Breadcrumb: "doc › heading › subheading".
    pub title: String,
    pub body: String,
    /// 1-based, inclusive.
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone)]
struct Para {
    text: String,
    start_line: usize,
    end_line: usize,
    words: usize,
    is_fence: bool,
}

struct Section {
    crumb: String,
    paras: Vec<Para>,
}

pub fn chunk_markdown(doc_title: &str, text: &str) -> Vec<DocChunk> {
    let lines: Vec<(usize, &str)> = line_spans(text);
    let (headings, fences) = scan_structure(text);

    // Build sections: [content-before-first-heading] + one per heading.
    let mut sections: Vec<Section> = Vec::new();
    let mut crumb_stack: Vec<(u32, String)> = Vec::new();
    let mut boundaries: Vec<(usize, usize, Option<&Heading>)> = Vec::new(); // (start, end, heading)
    {
        let first = headings
            .first()
            .map(|h| h.range.start)
            .unwrap_or(text.len());
        if first > 0 {
            boundaries.push((0, first, None));
        }
        for (i, h) in headings.iter().enumerate() {
            let end = headings
                .get(i + 1)
                .map(|n| n.range.start)
                .unwrap_or(text.len());
            boundaries.push((h.range.end, end, Some(h)));
        }
    }
    for (start, end, heading) in boundaries {
        if let Some(h) = heading {
            while crumb_stack
                .last()
                .map(|(l, _)| *l >= h.level)
                .unwrap_or(false)
            {
                crumb_stack.pop();
            }
            crumb_stack.push((h.level, h.text.clone()));
        }
        let crumb = std::iter::once(doc_title.to_string())
            .chain(crumb_stack.iter().map(|(_, t)| t.clone()))
            .collect::<Vec<_>>()
            .join(" › ");
        let paras = split_paras(&lines, start, end, &fences);
        if paras.is_empty() {
            continue;
        }
        sections.push(Section { crumb, paras });
    }

    // Merge undersized sections into their predecessor (heading text kept
    // as a leading line so the content stays findable).
    let mut merged: Vec<Section> = Vec::new();
    for section in sections {
        let words: usize = section.paras.iter().map(|p| p.words).sum();
        match merged.last_mut() {
            Some(prev) if words < MIN_WORDS => {
                let heading_text = section
                    .crumb
                    .rsplit(" › ")
                    .next()
                    .unwrap_or_default()
                    .to_string();
                let first = section.paras.first().unwrap();
                prev.paras.push(Para {
                    text: heading_text.clone(),
                    start_line: first.start_line,
                    end_line: first.start_line,
                    words: heading_text.split_whitespace().count(),
                    is_fence: false,
                });
                prev.paras.extend(section.paras);
            }
            _ => merged.push(section),
        }
    }

    let mut chunks = Vec::new();
    for section in &merged {
        pack_section(section, &mut chunks);
    }
    chunks
}

/// Greedy paragraph packing with bounded overlap between adjacent chunks.
fn pack_section(section: &Section, out: &mut Vec<DocChunk>) {
    let mut cur: Vec<Para> = Vec::new();
    let mut cur_words = 0usize;
    let mut carried = 0usize; // how many leading paras of `cur` are overlap

    let flush = |cur: &[Para], out: &mut Vec<DocChunk>, crumb: &str| {
        if cur.is_empty() {
            return;
        }
        out.push(DocChunk {
            title: crumb.to_string(),
            body: cur
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
            start_line: cur.first().unwrap().start_line,
            end_line: cur.last().unwrap().end_line,
        });
    };

    for para in &section.paras {
        let over_max = cur_words > 0 && cur_words + para.words > MAX_WORDS;
        if (over_max && cur_words >= MIN_WORDS) || cur_words >= TARGET_WORDS {
            flush(&cur, out, &section.crumb);
            // Overlap: carry trailing prose (never fences) up to the budget.
            let mut keep: Vec<Para> = Vec::new();
            let mut kept_words = 0;
            for p in cur.iter().rev() {
                if p.is_fence || kept_words + p.words > OVERLAP_WORDS {
                    break;
                }
                kept_words += p.words;
                keep.push(p.clone());
            }
            keep.reverse();
            carried = keep.len();
            cur_words = kept_words;
            cur = keep;
        }
        cur_words += para.words;
        cur.push(para.clone());
    }
    // Remainder: only if it holds anything beyond carried overlap.
    if cur.len() > carried {
        if cur_words < MIN_WORDS && !out.is_empty() {
            // Tiny tail in the same section → merge into the previous chunk.
            let prev = out.last_mut().unwrap();
            if prev.title == section.crumb {
                for p in &cur[carried..] {
                    prev.body.push_str("\n\n");
                    prev.body.push_str(&p.text);
                    prev.end_line = p.end_line;
                }
                return;
            }
        }
        flush(&cur, out, &section.crumb);
    }
}

struct Heading {
    level: u32,
    text: String,
    /// Byte range of the whole heading element.
    range: Range<usize>,
}

/// One parser pass: H1–H3 headings (with byte ranges) + fenced/indented
/// code block byte ranges.
fn scan_structure(text: &str) -> (Vec<Heading>, Vec<Range<usize>>) {
    let mut headings = Vec::new();
    let mut fences = Vec::new();
    let mut heading_buf: Option<(u32, Range<usize>, String)> = None;
    for (event, range) in Parser::new_ext(text, Options::all()).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) if level <= HeadingLevel::H3 => {
                heading_buf = Some((heading_level(level), range, String::new()));
            }
            Event::Text(t) | Event::Code(t) => {
                if let Some((_, _, buf)) = heading_buf.as_mut() {
                    buf.push_str(&t);
                }
            }
            Event::End(pulldown_cmark::TagEnd::Heading(_)) => {
                if let Some((level, range, text)) = heading_buf.take() {
                    headings.push(Heading {
                        level,
                        text: text.trim().to_string(),
                        range,
                    });
                }
            }
            Event::Start(Tag::CodeBlock(_)) => fences.push(range),
            _ => {}
        }
    }
    (headings, fences)
}

fn heading_level(level: HeadingLevel) -> u32 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// (byte_offset, line_text) for every line.
fn line_spans(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        out.push((offset, line.trim_end_matches(['\n', '\r'])));
        offset += line.len();
    }
    out
}

/// Paragraphs in byte range [start, end): blank-line separated, except
/// inside code fences (which stay atomic, blank lines included).
fn split_paras(
    lines: &[(usize, &str)],
    start: usize,
    end: usize,
    fences: &[Range<usize>],
) -> Vec<Para> {
    let in_fence = |offset: usize| fences.iter().any(|f| f.contains(&offset));
    let mut paras = Vec::new();
    let mut cur: Vec<(usize, &str, bool)> = Vec::new(); // (line_no, text, fenced)

    let mut flush = |cur: &mut Vec<(usize, &str, bool)>| {
        if cur.is_empty() {
            return;
        }
        let text = cur
            .iter()
            .map(|(_, t, _)| *t)
            .collect::<Vec<_>>()
            .join("\n");
        let words = text.split_whitespace().count();
        paras.push(Para {
            start_line: cur.first().unwrap().0,
            end_line: cur.last().unwrap().0,
            is_fence: cur.iter().any(|(_, _, f)| *f),
            text,
            words,
        });
        cur.clear();
    };

    for (i, (offset, line)) in lines.iter().enumerate() {
        if *offset < start || *offset >= end {
            continue;
        }
        let fenced = in_fence(*offset);
        if line.trim().is_empty() && !fenced {
            flush(&mut cur);
        } else {
            cur.push((i + 1, line, fenced));
        }
    }
    flush(&mut cur);
    paras
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_trivial_docs() {
        assert!(chunk_markdown("doc", "").is_empty());
        let chunks = chunk_markdown("doc", "just one line\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].title, "doc");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 1));
    }

    #[test]
    fn breadcrumbs_follow_heading_tree() {
        let md = "\
intro text before any heading

# Setup

setup paragraph

## Linux

linux details

## Windows

windows details

# Usage

usage paragraph
";
        let chunks = chunk_markdown("guide", md);
        let titles: Vec<&str> = chunks.iter().map(|c| c.title.as_str()).collect();
        // Small sections merge forward, but the first chunk keeps the root crumb.
        assert_eq!(titles[0], "guide");
        let joined = chunks
            .iter()
            .map(|c| c.body.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for text in [
            "intro text",
            "setup paragraph",
            "linux details",
            "windows details",
            "usage paragraph",
        ] {
            assert!(joined.contains(text), "missing {text}");
        }
    }

    #[test]
    fn big_sections_get_their_own_crumbs() {
        let para = "word ".repeat(80);
        let md = format!("# Alpha\n\n{para}\n\n{para}\n\n## Beta\n\n{para}\n\n{para}\n");
        let chunks = chunk_markdown("doc", &md);
        assert!(chunks.iter().any(|c| c.title == "doc › Alpha"));
        assert!(chunks.iter().any(|c| c.title == "doc › Alpha › Beta"));
    }

    #[test]
    fn fenced_code_never_splits() {
        let prose = "word ".repeat(350);
        let code = format!("```rust\n{}\n```", "let x = 1;\n".repeat(50));
        let md = format!("# Code\n\n{prose}\n\n{code}\n\n{prose}\n");
        let chunks = chunk_markdown("doc", &md);
        let with_fence: Vec<&DocChunk> = chunks
            .iter()
            .filter(|c| c.body.contains("```rust"))
            .collect();
        assert_eq!(with_fence.len(), 1);
        // The whole fence must be inside that one chunk.
        assert_eq!(with_fence[0].body.matches("```").count(), 2);
    }

    #[test]
    fn blank_lines_inside_fences_do_not_split() {
        let md = "# X\n\nbefore\n\n```\nline1\n\nline2\n```\n\nafter\n";
        let chunks = chunk_markdown("doc", md);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].body.contains("line1\n\nline2"));
    }

    #[test]
    fn oversized_sections_split_with_overlap() {
        // Paragraphs small enough (≈27 words) to fit the 40-word overlap budget.
        let para = |i: usize| format!("paragraph {i} {}", "filler ".repeat(25));
        let md = format!(
            "# Long\n\n{}\n",
            (0..24).map(para).collect::<Vec<_>>().join("\n\n")
        );
        let chunks = chunk_markdown("doc", &md);
        assert!(
            chunks.len() >= 2,
            "expected several chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            let words = c.body.split_whitespace().count();
            assert!(words <= MAX_WORDS + OVERLAP_WORDS, "chunk too big: {words}");
        }
        // Adjacent chunks overlap: chunk 1 ends with what chunk 2 starts with.
        let first_para_of_second = chunks[1].body.split("\n\n").next().unwrap();
        assert!(
            chunks[0].body.contains(first_para_of_second),
            "no overlap between adjacent chunks"
        );
        // Line spans are sane and ordered.
        assert!(chunks[0].start_line <= chunks[0].end_line);
        assert!(chunks[0].start_line < chunks[1].start_line);
    }

    #[test]
    fn line_spans_match_source() {
        let md = "# H\n\nalpha\nbeta\n\ngamma\n";
        let chunks = chunk_markdown("doc", md);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 3);
        assert_eq!(chunks[0].end_line, 6);
    }
}
