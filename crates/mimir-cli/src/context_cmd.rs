//! `mimir outline` / `mimir peek` — the dense code-context layer.
//!
//! Reuses the same tree-sitter extraction as the code graph, but instead of
//! persisting symbols it serves the agent a *signature map* (outline) or a
//! single symbol's body (peek) so it reads dense structure instead of whole
//! files. Every call records a `savings_event` (full-file tokens vs what was
//! actually returned).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mimir_core::{savings, tokens, Mimir};
use mimir_graph::extract::extract;
use mimir_graph::languages::Lang;
use mimir_graph::resolve_symbol;
use serde_json::json;

/// Result of outlining a target (one or more files).
pub struct Outlined {
    pub text: String,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub files: usize,
}

/// Result of peeking a single symbol.
pub struct Peeked {
    pub text: String,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub path: String,
    /// Node id of the resolved symbol, so callers can record the access
    /// without resolving it a second time.
    pub symbol_id: i64,
}

/// Build a signature map for `target` (a file or directory, relative to
/// `cwd`). Pure filesystem work — reads + parses on the fly, so it's always
/// fresh and needs no prior `graph build`.
pub fn outline_paths(target: &str, cwd: &Path) -> Result<Outlined> {
    let abs = {
        let p = Path::new(target);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            cwd.join(p)
        }
    };
    if !abs.exists() {
        bail!("no such file or directory: {target}");
    }

    let files = if abs.is_dir() {
        collect_source_files(&abs)
    } else {
        vec![abs.clone()]
    };
    if files.is_empty() {
        bail!("no supported source files under {target}");
    }

    let mut before = 0usize;
    let mut sections: Vec<String> = Vec::new();
    let mut shown = 0usize;
    for file in &files {
        let Some(kind) = Outliner::for_path(&file.to_string_lossy()) else {
            continue;
        };
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        before += tokens::count(&src);
        let rel = file
            .strip_prefix(cwd)
            .unwrap_or(file)
            .to_string_lossy()
            .into_owned();
        sections.push(kind.render(&rel, &src));
        shown += 1;
    }
    if shown == 0 {
        bail!("no outlineable files under {target}");
    }
    let text = sections.join("\n");
    let after = tokens::count(&text);
    Ok(Outlined {
        text,
        tokens_before: before,
        tokens_after: after,
        files: shown,
    })
}

/// What kind of structural outline a file gets.
enum Outliner {
    Code(Lang),
    Markdown,
    Json,
    Yaml,
}

impl Outliner {
    /// Code via tree-sitter, else markdown/json/yaml by extension. None = not
    /// outlineable (binary, plain text, unknown).
    fn for_path(path: &str) -> Option<Outliner> {
        if let Some(lang) = Lang::from_path(path) {
            return Some(Outliner::Code(lang));
        }
        match path.rsplit('.').next()?.to_ascii_lowercase().as_str() {
            "md" | "markdown" | "mdx" => Some(Outliner::Markdown),
            "json" => Some(Outliner::Json),
            "yaml" | "yml" => Some(Outliner::Yaml),
            _ => None,
        }
    }

    /// True for the kinds worth including in a *directory* sweep. JSON/YAML are
    /// only outlined when targeted directly (configs are everywhere).
    fn dir_eligible(path: &str) -> bool {
        matches!(
            Self::for_path(path),
            Some(Outliner::Code(_) | Outliner::Markdown)
        )
    }

    fn render(&self, rel: &str, src: &str) -> String {
        match self {
            Outliner::Code(lang) => render_code(rel, *lang, src),
            Outliner::Markdown => render_markdown(rel, src),
            Outliner::Json => render_json(rel, src),
            Outliner::Yaml => render_yaml(rel, src),
        }
    }
}

/// One code file's outline: a header plus one line per symbol (signature +
/// first doc line), no bodies.
fn render_code(rel: &str, lang: Lang, src: &str) -> String {
    let fx = extract(lang, src);
    let mut out = format!("{rel} · {} · {} symbols\n", lang.name(), fx.symbols.len());
    if fx.symbols.is_empty() {
        out.push_str("  (no top-level symbols)\n");
        return out;
    }
    for s in &fx.symbols {
        if let Some(doc) = &s.doc {
            if let Some(first) = doc.lines().map(str::trim).find(|l| !l.is_empty()) {
                out.push_str(&format!("  ⌐ {first}\n"));
            }
        }
        // The signature already carries the keyword (fn/struct/def/func/…),
        // so we don't repeat `s.kind` — every token counts here.
        out.push_str(&format!("  {:>4}  {}\n", s.start_line, s.signature));
    }
    out
}

/// Markdown outline: the heading tree (indented by level), with line numbers.
fn render_markdown(rel: &str, src: &str) -> String {
    let mut out = format!("{rel} · markdown\n");
    let mut in_fence = false;
    let mut any = false;
    for (i, line) in src.lines().enumerate() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(rest) = t.strip_prefix('#') {
            let level = 1 + rest.chars().take_while(|&c| c == '#').count();
            let title = t.trim_start_matches('#').trim();
            if !title.is_empty() {
                out.push_str(&format!(
                    "  {:>4}  {}{}\n",
                    i + 1,
                    "  ".repeat(level - 1),
                    title
                ));
                any = true;
            }
        }
    }
    if !any {
        out.push_str("  (no headings)\n");
    }
    out
}

/// JSON outline: the key skeleton (keys + types, no scalar values).
fn render_json(rel: &str, src: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(src) {
        Ok(v) => {
            let mut out = format!("{rel} · json\n");
            json_skeleton(&v, 1, &mut out);
            out
        }
        Err(e) => format!("{rel} · json (unparseable: {e})\n"),
    }
}

/// YAML outline: keys with their nesting (values elided). Line-based — no YAML
/// parser dependency; good enough for a structural map.
fn render_yaml(rel: &str, src: &str) -> String {
    let mut out = format!("{rel} · yaml\n");
    let mut any = false;
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // A mapping key: "key:" or "key: value" (value elided).
        if let Some(colon) = trimmed.find(':') {
            let key = &trimmed[..colon];
            if !key.is_empty() && !key.contains(' ') && !trimmed.starts_with('-') {
                let indent = line.len() - trimmed.len();
                out.push_str(&format!("  {}{}\n", " ".repeat(indent), key));
                any = true;
            }
        }
    }
    if !any {
        out.push_str("  (no keys)\n");
    }
    out
}

const JSON_MAX_DEPTH: usize = 4;

fn json_skeleton(v: &serde_json::Value, depth: usize, out: &mut String) {
    use serde_json::Value;
    let pad = "  ".repeat(depth);
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                match val {
                    Value::Object(inner) if depth < JSON_MAX_DEPTH => {
                        out.push_str(&format!("{pad}{k}:\n"));
                        if inner.is_empty() {
                            out.push_str(&format!("{pad}  {{}}\n"));
                        }
                        json_skeleton(val, depth + 1, out);
                    }
                    Value::Object(inner) => {
                        out.push_str(&format!("{pad}{k}: {{{} keys}}\n", inner.len()));
                    }
                    Value::Array(arr) => {
                        out.push_str(&format!("{pad}{k}: [{} items]\n", arr.len()));
                    }
                    _ => out.push_str(&format!("{pad}{k}: {}\n", json_scalar_type(val))),
                }
            }
        }
        Value::Array(arr) => out.push_str(&format!("{pad}[{} items]\n", arr.len())),
        _ => out.push_str(&format!("{pad}{}\n", json_scalar_type(v))),
    }
}

fn json_scalar_type(v: &serde_json::Value) -> &'static str {
    use serde_json::Value;
    match v {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "bool",
        Value::Null => "null",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Files under a directory worth outlining (gitignore-aware): code + markdown.
fn collect_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in ignore::WalkBuilder::new(dir).build().flatten() {
        let path = entry.path();
        if path.is_file() && Outliner::dir_eligible(&path.to_string_lossy()) {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    files
}

/// Return a single symbol's body (resolved against the project's code graph),
/// measuring savings versus reading the whole file.
pub fn peek_symbol(
    conn: &rusqlite::Connection,
    project_id: i64,
    root: &Path,
    reference: &str,
) -> Result<Peeked> {
    let sym = resolve_symbol(conn, project_id, reference)?;
    let rel = sym.path.clone().context("symbol has no file path")?;
    let (start, end) = match (sym.span_start, sym.span_end) {
        (Some(a), Some(b)) => (a as usize, b as usize),
        _ => bail!("symbol has no source span (rebuild the graph?)"),
    };
    let file = root.join(&rel);
    let src =
        std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;

    let before = tokens::count(&src);
    // 1-based inclusive lines → slice. Clamp defensively if the file shifted.
    let lines: Vec<&str> = src.lines().collect();
    let lo = start.saturating_sub(1).min(lines.len());
    let hi = end.min(lines.len());
    let snippet = lines[lo..hi].join("\n");
    let header = format!("{rel}:{start}-{end}\n");
    let text = format!("{header}{snippet}\n");
    let after = tokens::count(&text);
    Ok(Peeked {
        text,
        tokens_before: before,
        tokens_after: after,
        path: rel,
        symbol_id: sym.id,
    })
}

// ── CLI entry points ────────────────────────────────────────────────────────

pub fn outline(target: &str, json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let cwd = std::env::current_dir()?;
    let project_id = mimir.project_for_cwd(&cwd)?.map(|p| p.id);
    let o = outline_paths(target, &cwd)?;

    let _ = savings::record(
        &mimir.conn,
        project_id,
        savings::source::OUTLINE,
        o.tokens_before,
        o.tokens_after,
        json!({ "target": target, "files": o.files }),
    );

    if json {
        println!(
            "{}",
            json!({
                "target": target,
                "files": o.files,
                "tokens_before": o.tokens_before,
                "tokens_after": o.tokens_after,
                "saved": o.tokens_before.saturating_sub(o.tokens_after),
                "outline": o.text,
            })
        );
    } else {
        print!("{}", o.text);
        eprintln!(
            "— {} file(s), {} → {} tokens (saved {})",
            o.files,
            o.tokens_before,
            o.tokens_after,
            o.tokens_before.saturating_sub(o.tokens_after),
        );
    }
    Ok(())
}

pub fn peek(reference: &str, json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let cwd = std::env::current_dir()?;
    let proj = mimir.project_for_cwd(&cwd)?.context(
        "not inside a project. `peek` resolves symbols from the code graph, \
         which is per-project (a VCS marker, a build file like Cargo.toml/\
         package.json, or `touch .mimir`).",
    )?;
    let root = PathBuf::from(proj.path.as_deref().context("project has no root")?);
    let p = peek_symbol(&mimir.conn, proj.id, &root, reference)?;

    let _ = savings::record(
        &mimir.conn,
        Some(proj.id),
        savings::source::PEEK,
        p.tokens_before,
        p.tokens_after,
        json!({ "symbol": reference, "path": p.path }),
    );
    // Reading one symbol is a genuine access — feed the learning signal
    // (peek_symbol already resolved it; no second lookup).
    let _ = mimir_core::learn::record_opened(&mimir.conn, p.symbol_id);

    if json {
        println!(
            "{}",
            json!({
                "symbol": reference,
                "path": p.path,
                "tokens_before": p.tokens_before,
                "tokens_after": p.tokens_after,
                "saved": p.tokens_before.saturating_sub(p.tokens_after),
                "body": p.text,
            })
        );
    } else {
        print!("{}", p.text);
        eprintln!(
            "— {} → {} tokens (saved {})",
            p.tokens_before,
            p.tokens_after,
            p.tokens_before.saturating_sub(p.tokens_after),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn outline_renders_signatures_not_bodies() {
        let dir = tempfile::tempdir().unwrap();
        // Bodies are substantial so the signature map is genuinely smaller.
        let body = "    let mut acc = String::new();\n".repeat(40);
        write(
            dir.path(),
            "src/util.rs",
            &format!(
                "/// Slugs.\npub fn slugify(s: &str) -> String {{\n{body}    s.to_lowercase()\n}}\n\nstruct Big {{ secret_field: u64 }}\n"
            ),
        );
        let o = outline_paths("src/util.rs", dir.path()).unwrap();
        assert!(o.text.contains("slugify"), "{}", o.text);
        assert!(o.text.contains("Slugs."), "doc line shown");
        assert!(!o.text.contains("to_lowercase"), "body excluded");
        assert!(!o.text.contains("secret_field"), "struct body excluded");
        assert!(
            o.tokens_after < o.tokens_before,
            "after={} before={}",
            o.tokens_after,
            o.tokens_before
        );
        assert_eq!(o.files, 1);
    }

    #[test]
    fn outline_directory_walks_supported_files() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.rs", "fn one() {}\n");
        write(dir.path(), "b.py", "def two():\n    pass\n");
        write(dir.path(), "notes.txt", "ignore me\n");
        let o = outline_paths(".", dir.path()).unwrap();
        assert_eq!(o.files, 2, "rs + py, not txt");
        assert!(o.text.contains("one"));
        assert!(o.text.contains("two"));
    }

    #[test]
    fn outline_missing_target_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(outline_paths("nope.rs", dir.path()).is_err());
    }

    #[test]
    fn outline_markdown_headings() {
        let md = "# Title\n\nintro prose\n\n## Section\n\n```\n# not a heading\n```\n\n### Sub\n";
        let out = render_markdown("doc.md", md);
        assert!(out.contains("Title"));
        assert!(out.contains("Section"));
        assert!(out.contains("Sub"));
        assert!(!out.contains("not a heading"), "fenced code ignored");
        assert!(!out.contains("intro prose"), "prose excluded");
    }

    #[test]
    fn outline_json_skeleton_has_keys_not_values() {
        let js = r#"{"name":"x","version":"1.0","deps":{"a":"1","b":"2"},"items":[1,2,3]}"#;
        let out = render_json("p.json", js);
        assert!(out.contains("name: string"));
        assert!(out.contains("items: [3 items]"));
        assert!(out.contains("deps"));
        assert!(!out.contains("1.0"), "scalar values excluded");
    }

    #[test]
    fn outline_yaml_keys() {
        let yaml = "name: x\nversion: 1.0\njobs:\n  build:\n    runs-on: linux\n";
        let out = render_yaml("c.yaml", yaml);
        assert!(out.contains("name"));
        assert!(out.contains("jobs"));
        assert!(out.contains("build"));
        assert!(!out.contains("linux"), "values elided");
    }
}
