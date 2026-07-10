//! Markdown and source-code chunkers: heading/symbol-aware, sized for
//! retrieval.
//!
//! `chunk_markdown` splits at H1–H3; each chunk carries a breadcrumb title
//! ("Doc › Section › Sub") and a 1-based line span. Fenced code blocks
//! are atomic — never split, never carried into overlap.
//!
//! `chunk_source` splits on tree-sitter symbol boundaries instead of
//! headings — indexing function/method bodies (not just their signatures)
//! for recall is an idea from nworks3d's THOR fork of Mimir; see
//! CHANGELOG.md.

use std::ops::Range;
use std::path::Path;

use mimir_syntax::{extract::extract, languages::Lang};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};

/// Sizing in words (≈ 0.75 tokens/word, so 280 words ≈ 250–400 tokens).
const TARGET_WORDS: usize = 280;
const MAX_WORDS: usize = 400;
const MIN_WORDS: usize = 60;
const OVERLAP_WORDS: usize = 40;

/// Well-known lockfiles / generated manifests: never indexed even though
/// their extension would otherwise match the plain-text whitelist (e.g.
/// `package-lock.json` matches `.json`). Machine-generated, often huge,
/// and low-signal for recall.
const LOCKFILE_NAMES: &[&str] = &[
    "cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "gemfile.lock",
    "poetry.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
    "pipfile.lock",
];

/// 256 KiB — generous for hand-written config, well below anything
/// generated. Lockfiles are blacklisted by name above; this is a backstop
/// for whatever else slips through (e.g. an oversized fixture.json).
const PLAIN_TEXT_SIZE_CAP: u64 = 256 * 1024;

/// How a whitelisted plain-text/config file should be chunked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlainTextKind {
    /// toml, yaml, json, sh, ini, Dockerfile, Makefile, `.env.example`, ...
    /// — never through `chunk_markdown` (comment lines would misparse as
    /// headings).
    Config,
    /// txt, rst — free-form prose close enough to markdown that
    /// `chunk_markdown`'s paragraph packing (it just won't find headings)
    /// is the right tool.
    Prose,
}

/// Classify a non-source file (one `Lang::from_path` didn't recognize) for
/// the plain-text/config fallback. `None` means: skip it, same as before
/// this fallback existed.
///
/// Order matters: secrets and lockfiles are excluded *before* the
/// extension whitelist runs, so a future whitelist addition can never
/// accidentally re-admit `.env` or a generated lockfile.
pub fn plain_text_kind(path: &Path, size: i64) -> Option<PlainTextKind> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();

    // Never a real .env (secrets) — `.env.example` (a checked-in template)
    // is fine and explicitly whitelisted below.
    if name == ".env" || (name.starts_with(".env.") && name != ".env.example") {
        return None;
    }
    if LOCKFILE_NAMES.contains(&name.as_str()) {
        return None;
    }
    if size < 0 || size as u64 > PLAIN_TEXT_SIZE_CAP {
        return None;
    }

    if name == "dockerfile" || name == "makefile" || name == ".env.example" {
        return Some(PlainTextKind::Config);
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "toml" | "yaml" | "yml" | "json" | "sh" | "bash" | "ini" => Some(PlainTextKind::Config),
        "txt" | "rst" => Some(PlainTextKind::Prose),
        _ => None,
    }
}

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
        pack_section(section, "\n\n", &mut chunks);
    }
    chunks
}

/// Greedy paragraph packing with bounded overlap between adjacent chunks.
/// `sep` joins consecutive units in a flushed chunk's body — "\n\n" for
/// markdown paragraphs, "\n" for chunk_source's one-unit-per-line sections
/// (a blank-line join would visibly corrupt code).
fn pack_section(section: &Section, sep: &str, out: &mut Vec<DocChunk>) {
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
                .join(sep),
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
                    prev.body.push_str(sep);
                    prev.body.push_str(&p.text);
                    prev.end_line = p.end_line;
                }
                return;
            }
        }
        flush(&cur, out, &section.crumb);
    }
}

/// Chunk a whitelisted config/plain-text file (toml, yaml, json, sh, ini,
/// Dockerfile, Makefile, `.env.example`, ...) the same way `chunk_markdown`
/// packs paragraphs — but skipping markdown parsing entirely. Running these
/// through `chunk_markdown`'s pulldown_cmark pass would misread comment
/// lines (`# this is a toml comment`, `# syntax=docker/dockerfile:1`, ...)
/// as H1 headings and shred the file into garbage sections.
///
/// One flat section (no heading tree), blank-line-delimited paragraphs,
/// packed to the same word budget/overlap as everything else.
pub fn chunk_plain_text(doc_title: &str, text: &str) -> Vec<DocChunk> {
    let lines = line_spans(text);
    let paras = split_paras(&lines, 0, text.len(), &[]);
    if paras.is_empty() {
        return Vec::new();
    }
    let section = Section {
        crumb: doc_title.to_string(),
        paras,
    };
    let mut chunks = Vec::new();
    pack_section(&section, "\n\n", &mut chunks);
    chunks
}

/// Chunk source code on tree-sitter symbol boundaries: one section per
/// symbol, plus line-window sections for whatever isn't inside any symbol
/// (imports, top-level statements). Oversized symbol bodies are sub-split
/// by the same word-budget/overlap packer as markdown.
///
/// Symbols nest (a class's methods sit inside the class's own line span),
/// so lines are claimed innermost-first: a symbol only emits sections for
/// the sub-ranges its own nested symbols haven't already claimed. Without
/// that, a class chunk would duplicate the full text of every method
/// already covered by its own chunk.
pub fn chunk_source(lang: Lang, doc_title: &str, source: &str) -> Vec<DocChunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut symbols = extract(lang, source).symbols;
    // Innermost first: a shorter span is never the container of a longer
    // one, so this always claims a class's methods before the class itself.
    symbols.sort_by_key(|s| s.end_line.saturating_sub(s.start_line));

    let mut covered = vec![false; lines.len()];
    let mut chunks = Vec::new();
    for sym in &symbols {
        let start = sym.start_line.max(1);
        let end = sym.end_line.min(lines.len());
        if start > end {
            continue;
        }
        let sig_line = sym
            .signature
            .lines()
            .next()
            .unwrap_or(&sym.signature)
            .trim();
        let crumb = format!("{doc_title} › {} ({}) — {sig_line}", sym.qualified, sym.kind);
        for (run_start, run_end) in uncovered_runs(&covered, start, end) {
            let paras = line_paras(&lines, run_start, run_end);
            if !paras.is_empty() {
                pack_section(
                    &Section {
                        crumb: crumb.clone(),
                        paras,
                    },
                    "\n",
                    &mut chunks,
                );
            }
        }
        covered[start - 1..end].fill(true);
    }

    // Remainder: lines no symbol claimed. Each contiguous run gets a
    // line-range-unique crumb so pack_section's tiny-tail merge (keyed on
    // crumb equality) never splices two discontiguous runs together.
    for (run_start, run_end) in uncovered_runs(&covered, 1, lines.len()) {
        let paras = line_paras(&lines, run_start, run_end);
        if paras.is_empty() {
            continue;
        }
        let crumb = format!("{doc_title} › (lines {run_start}-{run_end})");
        pack_section(&Section { crumb, paras }, "\n", &mut chunks);
    }

    chunks
}

/// Contiguous 1-based inclusive sub-ranges of `[start, end]` not yet marked
/// covered.
fn uncovered_runs(covered: &[bool], start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;
    for l in start..=end {
        if covered[l - 1] {
            if let Some(s) = run_start.take() {
                runs.push((s, l - 1));
            }
        } else if run_start.is_none() {
            run_start = Some(l);
        }
    }
    if let Some(s) = run_start {
        runs.push((s, end));
    }
    runs
}

/// One `Para` per non-blank line in a 1-based inclusive range.
fn line_paras(lines: &[&str], start: usize, end: usize) -> Vec<Para> {
    (start..=end)
        .filter_map(|l| {
            let text = lines[l - 1];
            if text.trim().is_empty() {
                return None;
            }
            Some(Para {
                words: text.split_whitespace().count(),
                text: text.to_string(),
                start_line: l,
                end_line: l,
                is_fence: false,
            })
        })
        .collect()
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

    #[test]
    fn chunk_source_finds_function_body_and_imports() {
        let src = "use std::fmt;\n\nfn distinctive_needle_marker() {\n    println!(\"hi\");\n}\n";
        let chunks = chunk_source(Lang::Rust, "crate/lib.rs", src);
        assert!(
            chunks
                .iter()
                .any(|c| c.body.contains("distinctive_needle_marker")),
            "function body missing from any chunk"
        );
        assert!(
            chunks.iter().any(|c| c.body.contains("use std::fmt;")),
            "remainder (import) line missing from any chunk"
        );
    }

    #[test]
    fn chunk_source_nested_class_does_not_duplicate_method_body() {
        // TS: class + method are both symbols with overlapping spans —
        // the method must claim its own body; the class chunk must not
        // also contain the full method body verbatim.
        let src = "\
class Widget {
    render() {
        return marker_only_in_method_body();
    }
}
";
        let chunks = chunk_source(Lang::TypeScript, "widget.ts", src);
        let with_marker: Vec<_> = chunks
            .iter()
            .filter(|c| c.body.contains("marker_only_in_method_body"))
            .collect();
        assert_eq!(
            with_marker.len(),
            1,
            "method body should appear in exactly one chunk, got {}: {:#?}",
            with_marker.len(),
            chunks
        );
    }

    #[test]
    fn chunk_source_empty_is_empty() {
        assert!(chunk_source(Lang::Rust, "empty.rs", "").is_empty());
    }

    #[test]
    fn chunk_plain_text_does_not_treat_comments_as_headings() {
        // A markdown pass would read `# syntax=...` as an H1 and split the
        // crumb tree on it; chunk_plain_text must not.
        let toml = "# syntax=docker/dockerfile:1\nname = \"mimir\"\nversion = \"0.13.0\"\n";
        let chunks = chunk_plain_text("Cargo.toml", toml);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].title, "Cargo.toml");
        assert!(chunks[0].body.contains("# syntax=docker/dockerfile:1"));
        assert!(chunks[0].body.contains("version = \"0.13.0\""));
    }

    #[test]
    fn chunk_plain_text_splits_on_blank_lines_like_markdown() {
        let ini = "[core]\nrepositoryformatversion = 0\n\n[remote \"origin\"]\nurl = https://example.com\n";
        let chunks = chunk_plain_text("config", ini);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].body.contains("[core]"));
        assert!(chunks[0].body.contains("[remote \"origin\"]"));
    }

    #[test]
    fn chunk_plain_text_empty_is_empty() {
        assert!(chunk_plain_text("empty.txt", "").is_empty());
    }

    #[test]
    fn plain_text_kind_whitelist() {
        for name in ["Cargo.toml", "config.yaml", "config.yml", "data.json", "run.sh", "run.bash", "settings.ini"] {
            assert_eq!(
                plain_text_kind(Path::new(name), 100),
                Some(PlainTextKind::Config),
                "{name} should be Config"
            );
        }
        for name in ["Dockerfile", "Makefile", ".env.example"] {
            assert_eq!(
                plain_text_kind(Path::new(name), 100),
                Some(PlainTextKind::Config),
                "{name} should be Config"
            );
        }
        for name in ["notes.txt", "readme.rst"] {
            assert_eq!(
                plain_text_kind(Path::new(name), 100),
                Some(PlainTextKind::Prose),
                "{name} should be Prose"
            );
        }
    }

    #[test]
    fn plain_text_kind_never_real_env() {
        assert_eq!(plain_text_kind(Path::new(".env"), 100), None);
        assert_eq!(plain_text_kind(Path::new(".env.local"), 100), None);
        assert_eq!(plain_text_kind(Path::new(".env.production"), 100), None);
        // The template is fine — it's the one meant to be checked in.
        assert_eq!(
            plain_text_kind(Path::new(".env.example"), 100),
            Some(PlainTextKind::Config)
        );
    }

    #[test]
    fn plain_text_kind_blacklists_lockfiles() {
        for name in [
            "Cargo.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "go.sum",
        ] {
            assert_eq!(plain_text_kind(Path::new(name), 100), None, "{name} should be excluded");
        }
    }

    #[test]
    fn plain_text_kind_enforces_size_cap() {
        assert_eq!(
            plain_text_kind(Path::new("big.json"), (PLAIN_TEXT_SIZE_CAP + 1) as i64),
            None
        );
        assert_eq!(
            plain_text_kind(Path::new("small.json"), PLAIN_TEXT_SIZE_CAP as i64),
            Some(PlainTextKind::Config)
        );
    }

    #[test]
    fn plain_text_kind_ignores_unlisted_extensions() {
        // Binary/unrelated files a tree-sitter adapter also can't parse —
        // must not be swept up by the fallback either.
        assert_eq!(plain_text_kind(Path::new("photo.png"), 100), None);
        assert_eq!(plain_text_kind(Path::new("data.bin"), 100), None);
    }
}
