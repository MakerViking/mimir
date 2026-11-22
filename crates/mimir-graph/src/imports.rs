//! Import source → the project file it names, by language.
//!
//! An `imports` edge says "this file depends on that file", and `impact`
//! follows it. It also decides call edges: a name imported from a file is
//! resolved to that file's definition at full weight. So a wrong import edge
//! is the worst thing this module can produce, and a missing one is merely a
//! gap. The rule is precision first: an edge is written only when the
//! language's own lookup rules, or a match that exactly one file in the
//! project satisfies, name a single file. Two candidates is no edge — a
//! coin-flip between `core/util.rs` and `cli/util.rs` would be right half the
//! time and trusted all of it — and a name that is as likely to be the
//! standard library's or a dependency's as the project's own (`logging`,
//! `java.util.Optional`, `<math.h>`, `Illuminate\Http\Request`) is no edge
//! either, however neatly some project file happens to fit it.
//!
//! Which rules apply is decided by the importing file's [`Lang`], never by
//! the shape of the string: `util.h`, `a.b.C` and `bootstrap.php` all contain
//! a dot and no slash, and mean three different lookups.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::languages::Lang;

/// The project's code files, indexed for the lookups import resolution does:
/// exact path, "the file whose path ends in this" and "the files directly in
/// this directory".
pub(crate) struct FileSet {
    paths: HashSet<String>,
    /// File name → every path with that name. A suffix match starts here, so
    /// it costs the handful of same-named files rather than a scan of the
    /// project per import.
    by_name: HashMap<String, Vec<String>>,
    /// Directory (`""` for the project root) → the files directly inside it.
    by_dir: HashMap<String, Vec<String>>,
    /// Directory name → every directory with that name.
    dirs_by_name: HashMap<String, Vec<String>>,
}

impl FileSet {
    pub(crate) fn new(paths: impl IntoIterator<Item = String>) -> FileSet {
        let paths: HashSet<String> = paths.into_iter().collect();
        let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
        let mut by_dir: HashMap<String, Vec<String>> = HashMap::new();
        for path in &paths {
            let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
            by_name
                .entry(name.to_string())
                .or_default()
                .push(path.clone());
            by_dir
                .entry(dir.to_string())
                .or_default()
                .push(path.clone());
        }
        let mut dirs_by_name: HashMap<String, Vec<String>> = HashMap::new();
        for dir in by_dir.keys().filter(|d| !d.is_empty()) {
            let name = dir.rsplit('/').next().unwrap_or(dir);
            dirs_by_name
                .entry(name.to_string())
                .or_default()
                .push(dir.clone());
        }
        FileSet {
            paths,
            by_name,
            by_dir,
            dirs_by_name,
        }
    }

    pub(crate) fn contains(&self, path: &str) -> bool {
        self.paths.contains(path)
    }

    pub(crate) fn same_paths(&self, other: &FileSet) -> bool {
        self.paths == other.paths
    }

    /// Every file whose path is `suffix`, or ends in it at a path boundary:
    /// `util.rs` matches `src/util.rs`, never `src/db_util.rs`.
    fn with_suffix(&self, suffix: &str) -> Vec<&str> {
        let name = suffix.rsplit('/').next().unwrap_or(suffix);
        self.by_name
            .get(name)
            .into_iter()
            .flatten()
            .filter(|p| ends_at_boundary(p, suffix))
            .map(String::as_str)
            .collect()
    }

    /// Every directory whose path is `suffix` or ends in it at a boundary.
    fn dirs_with_suffix(&self, suffix: &str) -> Vec<&str> {
        let name = suffix.rsplit('/').next().unwrap_or(suffix);
        self.dirs_by_name
            .get(name)
            .into_iter()
            .flatten()
            .filter(|d| ends_at_boundary(d, suffix))
            .map(String::as_str)
            .collect()
    }

    fn in_dir(&self, dir: &str) -> &[String] {
        self.by_dir.get(dir).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The first candidate that is a file, in the order given — for lookups
    /// where the language itself defines the order (`./x` is `x.ts` before
    /// `x/index.ts`).
    fn first_existing(&self, candidates: impl IntoIterator<Item = String>) -> Option<String> {
        candidates.into_iter().find(|c| self.contains(c))
    }

    /// Suffix lookup, most specific first. Each group is one level of
    /// specificity; the first group that matches anything decides, and it
    /// decides for a file only when it matched exactly one. A less specific
    /// group is never consulted to break a tie in a more specific one.
    fn unique_suffix(&self, groups: impl IntoIterator<Item = Vec<String>>) -> Option<String> {
        for group in groups {
            let mut hits: Vec<&str> = group.iter().flat_map(|s| self.with_suffix(s)).collect();
            hits.sort_unstable();
            hits.dedup();
            match hits.as_slice() {
                [] => continue,
                [one] => return Some((*one).to_string()),
                _ => return None,
            }
        }
        None
    }
}

fn ends_at_boundary(path: &str, suffix: &str) -> bool {
    path == suffix
        || path
            .strip_suffix(suffix)
            .is_some_and(|head| head.ends_with('/'))
}

/// The project file `source` names when written in `importer`, or `None`
/// when it names none (a package, the standard library) or more than one.
pub(crate) fn resolve_import(
    lang: Lang,
    importer: &str,
    source: &str,
    files: &FileSet,
) -> Option<String> {
    let dir = importer.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    match lang {
        Lang::TypeScript | Lang::Tsx => script_module(dir, source, files),
        Lang::Python => python_module(dir, source, files),
        Lang::Rust => rust_path(importer, source, files),
        Lang::Go => go_package(source, files),
        Lang::Java | Lang::Kotlin => jvm_class(source, files),
        Lang::C | Lang::Cpp => c_include(dir, source, files),
        Lang::Php => php_source(dir, source, files),
        // `using System.Text` and `import Foundation` name a namespace or a
        // module, which is any number of files; Ruby and SQL emit no imports.
        Lang::CSharp | Lang::Swift | Lang::Ruby | Lang::Sql => None,
    }
}

/// JS/TS: only a relative specifier names a project file (`react` and
/// `@/lib/x` are a package and an alias). The candidates are tried in the
/// order a TypeScript build resolves them, so the first hit is the file.
fn script_module(dir: &str, source: &str, files: &FileSet) -> Option<String> {
    if !(source == "." || source == ".." || source.starts_with("./") || source.starts_with("../")) {
        return None;
    }
    let joined = join(dir, source);
    const EXTS: [&str; 8] = ["ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"];
    const INDEX: [&str; 4] = ["ts", "tsx", "js", "jsx"];
    let direct = EXTS.iter().map(|e| format!("{joined}.{e}"));
    let index = INDEX.iter().map(|e| join(&joined, &format!("index.{e}")));
    let found = files.first_existing(direct.chain(index).chain([joined.clone()]));
    // An ESM TypeScript source imports its sibling as `./x.js`: the name of
    // the file the compiler will emit, not of the one on disk.
    found.or_else(|| {
        let (stem, ext) = joined.rsplit_once('.')?;
        let twins: &[&str] = match ext {
            "js" => &["ts", "tsx"],
            "jsx" => &["tsx"],
            "mjs" => &["mts"],
            "cjs" => &["cts"],
            _ => return None,
        };
        files.first_existing(twins.iter().map(|e| format!("{stem}.{e}")))
    })
}

/// Python: `.util` / `..pkg.mod` relative to the importing package, or an
/// absolute `a.b.c` found from the project root, or below the one other
/// directory imports can start from.
///
/// That last lookup is where a project file can pass for somebody else's
/// module, so it is narrow. The directory a match hangs under must not be a
/// package itself: `app/logging.py` is `app.logging`, and `import logging`
/// inside `app` is the standard library's. And a single-segment import
/// matches a package directory only (`src/mypkg/__init__.py` for
/// `import mypkg`), never a lone `name.py` somewhere in the tree.
fn python_module(dir: &str, source: &str, files: &FileSet) -> Option<String> {
    if source.starts_with('.') {
        let dots = source.chars().take_while(|c| *c == '.').count();
        let module = source[dots..].replace('.', "/");
        let mut base = dir.to_string();
        for _ in 1..dots {
            base = base
                .rsplit_once('/')
                .map(|(d, _)| d.to_string())
                .unwrap_or_default();
        }
        if module.is_empty() {
            // `from . import x`: the package itself.
            return files.first_existing([join(&base, "__init__.py")]);
        }
        let joined = join(&base, &module);
        return files.first_existing([format!("{joined}.py"), format!("{joined}/__init__.py")]);
    }
    let module = source.replace('.', "/");
    let (as_file, as_package) = (format!("{module}.py"), format!("{module}/__init__.py"));
    if let Some(hit) = files.first_existing([as_file.clone(), as_package.clone()]) {
        return Some(hit);
    }
    let forms = if source.contains('.') {
        vec![as_file, as_package]
    } else {
        vec![as_package]
    };
    let mut hits: Vec<&str> = forms
        .iter()
        .flat_map(|form| {
            files.with_suffix(form).into_iter().filter(move |path| {
                let root = path.strip_suffix(form.as_str()).unwrap_or_default();
                !files.contains(&join(root, "__init__.py"))
            })
        })
        .collect();
    hits.sort_unstable();
    hits.dedup();
    match hits.as_slice() {
        [one] => Some((*one).to_string()),
        _ => None,
    }
}

/// Rust: `crate::`, `self::` and `super::` are resolved by the module rules
/// (`a::b` lives in `a/b.rs` or `a/b/mod.rs`) from where the importer sits,
/// so two crates of a workspace that both have an `error.rs` never get each
/// other's. Any other path is first tried as a child module of the importer,
/// then as the one file in the project the path ends in. A `crate::` path
/// its own crate cannot place gets that same last resort: it is a module
/// re-exported from another crate of the workspace (`pub use other::m;`),
/// and the file is wherever the one `m.rs` is. A `self::` or `super::` path
/// that misses does not — it names an item of a file already known, and the
/// commonest case is `super::x` inside an inline `mod tests`.
///
/// `crate::` starts at the importer's own crate root, and a binary under
/// `src/bin/` is its own crate: see [`rust_crate_roots`].
///
/// The tail of a `use` path is usually an item, not a module, so each lookup
/// tries the whole path first and drops trailing segments until a file
/// matches.
fn rust_path(importer: &str, source: &str, files: &FileSet) -> Option<String> {
    let mut segs: Vec<&str> = source.split("::").filter(|s| !s.is_empty()).collect();
    // `use a::b::{self}` arrives as `a::b::self`: the module `a::b`.
    while segs.len() > 1 && segs.last() == Some(&"self") {
        segs.pop();
    }
    let module_forms = |base: &str, segs: &[&str]| -> Vec<Vec<String>> {
        (1..=segs.len())
            .rev()
            .map(|take| {
                let path = join(base, &segs[..take].join("/"));
                vec![format!("{path}.rs"), format!("{path}/mod.rs")]
            })
            .collect()
    };
    // Module rules: the first depth at which a file exists decides, and
    // `x.rs` beside `x/mod.rs` is not a decision.
    let by_module_rules = |base: &str, segs: &[&str]| -> Option<Option<String>> {
        for forms in module_forms(base, segs) {
            let hits: Vec<&String> = forms.iter().filter(|f| files.contains(f)).collect();
            match hits.as_slice() {
                [] => continue,
                [one] => return Some(Some((*one).clone())),
                _ => return Some(None),
            }
        }
        None
    };

    let own = rust_module_dir(importer);
    match segs.first().copied() {
        Some("self") => by_module_rules(&own, &segs[1..]).flatten(),
        Some("super") => {
            let supers = segs.iter().take_while(|s| **s == "super").count();
            let mut base = own;
            for _ in 0..supers {
                base = base
                    .rsplit_once('/')
                    .map(|(d, _)| d.to_string())
                    .unwrap_or_default();
            }
            by_module_rules(&base, &segs[supers..]).flatten()
        }
        Some("crate") => {
            // No `src/` above the importer (an integration test, a build
            // script) means the crate root is not knowable from the path.
            let placed = rust_crate_roots(importer)
                .iter()
                .find_map(|root| by_module_rules(root, &segs[1..]));
            match placed {
                Some(decided) => decided,
                None => files.unique_suffix(module_forms("", &segs[1..])),
            }
        }
        Some(_) => match by_module_rules(&own, &segs) {
            Some(decided) => decided,
            None => files.unique_suffix(module_forms("", &segs)),
        },
        None => None,
    }
}

/// The directory a Rust file's child modules live in: `src/a/b.rs` → `src/a/b`,
/// while `mod.rs`, `lib.rs` and `main.rs` own the directory they sit in.
fn rust_module_dir(importer: &str) -> String {
    let (dir, name) = importer.rsplit_once('/').unwrap_or(("", importer));
    match name {
        "mod.rs" | "lib.rs" | "main.rs" => dir.to_string(),
        _ => join(dir, name.strip_suffix(".rs").unwrap_or(name)),
    }
}

/// Where `crate::` can start for `importer`, most specific first: the
/// directories its crate root's child modules live in.
///
/// That is the nearest `src` directory above it — unless the file sits under
/// `src/bin/`, where every binary is a crate of its own. `src/bin/tool.rs`
/// keeps its modules beside it (`src/bin/helper.rs`); `src/bin/tool/main.rs`
/// keeps them in `src/bin/tool/`. Looking only in `src` sent `crate::helper`
/// from a binary to the library's `src/helper.rs`: an import edge to a file
/// the binary cannot even see, and a call edge on top of it. The library's
/// root stays as the second place to look.
fn rust_crate_roots(importer: &str) -> Vec<String> {
    let Some((dir, _)) = importer.rsplit_once('/') else {
        return Vec::new();
    };
    let mut end = dir.len();
    let src = loop {
        let head = &dir[..end];
        if head.rsplit('/').next() == Some("src") {
            break head;
        }
        match head.rfind('/') {
            Some(slash) => end = slash,
            None => return Vec::new(),
        }
    };
    let mut roots = Vec::new();
    let below = importer[src.len()..].trim_start_matches('/');
    if let Some(in_bin) = below.strip_prefix("bin/") {
        roots.push(match in_bin.split_once('/') {
            Some((binary, _)) => format!("{src}/bin/{binary}"),
            None => format!("{src}/bin"),
        });
    }
    roots.push(src.to_string());
    roots
}

/// Go: an import names a package, which is a directory. The directory is
/// found by the longest tail of the import path that is a directory here
/// (the module prefix is not on disk); the edge exists only when that
/// directory holds exactly one non-test source file, since a file-to-file
/// edge cannot stand for "any of these".
fn go_package(source: &str, files: &FileSet) -> Option<String> {
    let segs: Vec<&str> = source.split('/').filter(|s| !s.is_empty()).collect();
    for start in 0..segs.len() {
        let dirs = files.dirs_with_suffix(&segs[start..].join("/"));
        match dirs.as_slice() {
            [] => continue,
            [dir] => {
                let mut sources = files
                    .in_dir(dir)
                    .iter()
                    .filter(|f| f.ends_with(".go") && !f.ends_with("_test.go"));
                let first = sources.next()?;
                return sources.next().is_none().then(|| first.clone());
            }
            _ => return None,
        }
    }
    None
}

/// Packages of the JVM platform and its standard libraries. No project file
/// is one of these, whatever its path looks like: a source root is commonly
/// *named* `java` or `kotlin`, so `java.util.Optional` would otherwise match
/// a project's own `src/main/java/util/Optional.java`.
const JVM_PLATFORM: [&str; 7] = [
    "java.",
    "javax.",
    "kotlin.",
    "kotlinx.",
    "android.",
    "androidx.",
    "scala.",
];

/// Java and Kotlin: `a.b.C` is `a/b/C.java` (or `.kt`) under some source
/// root, so the one file whose path ends in the whole package path. A member
/// import (`a.b.C.m`) falls back to its owner. A name with no package at all
/// is not looked for: it would match on the file name alone.
fn jvm_class(source: &str, files: &FileSet) -> Option<String> {
    if JVM_PLATFORM.iter().any(|p| source.starts_with(p)) {
        return None;
    }
    let segs: Vec<&str> = source.split('.').filter(|s| !s.is_empty()).collect();
    let forms = |take: usize| {
        let path = segs[..take].join("/");
        vec![format!("{path}.java"), format!("{path}.kt")]
    };
    match segs.len() {
        0 | 1 => None,
        2 => files.unique_suffix([forms(2)]),
        n => files.unique_suffix([forms(n), forms(n - 1)]),
    }
}

/// C and C++: `#include "x.h"` is the file beside the including one. Failing
/// that the compiler's include path decides, which is not known here, so a
/// header found that way must be written with a directory (`"proj/api.h"`)
/// and be the only file in the project its path ends in. A bare name is not
/// searched for across the project: `"config.h"` in one corner is not the
/// `config.h` in another.
///
/// Only quoted includes get here; the extractor does not report `<x>`.
fn c_include(dir: &str, source: &str, files: &FileSet) -> Option<String> {
    let beside = join(dir, source);
    if files.contains(&beside) {
        return Some(beside);
    }
    let through_include_path = join("", source);
    if !through_include_path.contains('/') {
        return None;
    }
    files.unique_suffix([vec![through_include_path]])
}

/// PHP: `require 'x.php'` is a path: beside the file, from the project root,
/// or (written with a directory) the one file it ends in.
///
/// `use App\Models\User` is a class whose file is its namespace path below
/// some PSR-4 root, so leading segments are dropped until a file matches —
/// but never down to the class name alone. `Illuminate\Http\Request` is a
/// dependency's class, and the project's only `Request.php` is some other
/// `Request`: at least two trailing segments must be the end of the path.
fn php_source(dir: &str, source: &str, files: &FileSet) -> Option<String> {
    if source.contains('/') || source.ends_with(".php") {
        let beside = join(dir, source);
        if files.contains(&beside) {
            return Some(beside);
        }
        let from_root = join("", source);
        if files.contains(&from_root) {
            return Some(from_root);
        }
        if !from_root.contains('/') {
            return None;
        }
        return files.unique_suffix([vec![from_root]]);
    }
    let segs: Vec<&str> = source.split('\\').filter(|s| !s.is_empty()).collect();
    files.unique_suffix(
        (0..segs.len().saturating_sub(1))
            .map(|start| vec![format!("{}.php", segs[start..].join("/"))]),
    )
}

/// `base/rel` with `.` and `..` resolved; `base` may be `""` (project root).
fn join(base: &str, rel: &str) -> String {
    let mut parts: Vec<&std::ffi::OsStr> = Vec::new();
    let joined = Path::new(base).join(rel);
    for c in joined.components() {
        match c {
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::Normal(s) => parts.push(s),
            _ => {}
        }
    }
    parts
        .iter()
        .map(|s| s.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(paths: &[&str]) -> FileSet {
        FileSet::new(paths.iter().map(|p| p.to_string()))
    }

    fn resolve(lang: Lang, importer: &str, source: &str, paths: &[&str]) -> Option<String> {
        resolve_import(lang, importer, source, &set(paths))
    }

    #[test]
    fn typescript_resolves_relative_specifiers_only() {
        let files = [
            "web/app.ts",
            "web/api.ts",
            "web/api.js",
            "web/ui/index.tsx",
            "web/esm.ts",
            "react.ts",
        ];
        let ts = |source: &str| resolve(Lang::TypeScript, "web/app.ts", source, &files);
        // `.ts` before the emitted `.js` beside it: the build's own order.
        assert_eq!(ts("./api").as_deref(), Some("web/api.ts"));
        assert_eq!(ts("./ui").as_deref(), Some("web/ui/index.tsx"));
        assert_eq!(ts("../web/api").as_deref(), Some("web/api.ts"));
        // ESM spelling: the `.js` that `esm.ts` will become.
        assert_eq!(ts("./esm.js").as_deref(), Some("web/esm.ts"));
        // A package, even when a file happens to carry its name.
        assert_eq!(ts("react"), None);
        assert_eq!(ts("./missing"), None);
    }

    #[test]
    fn python_resolves_relative_and_absolute_modules() {
        let files = [
            "main.py",
            "util.py",
            "app/main.py",
            "app/pkg/__init__.py",
            "app/pkg/mod.py",
            "app/pkg/sibling.py",
            "lib/helpers/text.py",
            "src/mypkg/__init__.py",
            "src/mypkg/core.py",
            "tools/a/dup/__init__.py",
            "tools/b/dup/__init__.py",
        ];
        let py = |importer: &str, source: &str| resolve(Lang::Python, importer, source, &files);
        assert_eq!(
            py("app/pkg/mod.py", ".sibling").as_deref(),
            Some("app/pkg/sibling.py")
        );
        assert_eq!(
            py("app/pkg/mod.py", "..main").as_deref(),
            Some("app/main.py")
        );
        assert_eq!(
            py("app/pkg/mod.py", ".").as_deref(),
            Some("app/pkg/__init__.py")
        );
        // Absolute from the project root: a module, and a package's `__init__`.
        assert_eq!(py("app/main.py", "util").as_deref(), Some("util.py"));
        assert_eq!(
            py("main.py", "app.pkg").as_deref(),
            Some("app/pkg/__init__.py")
        );
        // Below a source root that is not the project root.
        assert_eq!(
            py("app/main.py", "helpers.text").as_deref(),
            Some("lib/helpers/text.py")
        );
        assert_eq!(
            py("main.py", "mypkg").as_deref(),
            Some("src/mypkg/__init__.py")
        );
        assert_eq!(
            py("main.py", "mypkg.core").as_deref(),
            Some("src/mypkg/core.py")
        );
        // Two packages answer to the name: no edge.
        assert_eq!(py("app/main.py", "dup"), None);
        assert_eq!(py("app/main.py", "os.path"), None);
    }

    /// `import logging` inside a package that has its own `logging.py` is the
    /// standard library's: the sibling is `app.logging`. Nothing matches a
    /// module name against a file that sits inside a package, and nothing
    /// matches a single-segment import against a lone file below the root.
    #[test]
    fn python_never_takes_a_package_member_for_a_top_level_module() {
        let files = [
            "app/__init__.py",
            "app/main.py",
            "app/logging.py",
            "app/json/__init__.py",
            "app/util/text.py",
            "scripts/run.py",
            "scripts/helpers.py",
        ];
        let py = |importer: &str, source: &str| resolve(Lang::Python, importer, source, &files);
        assert_eq!(py("app/main.py", "logging"), None);
        assert_eq!(py("app/main.py", "json"), None, "app.json, not json");
        assert_eq!(py("app/main.py", "util.text"), None, "app.util.text");
        assert_eq!(py("scripts/run.py", "helpers"), None);
        // Spelled the way it is importable, each of them resolves.
        assert_eq!(
            py("scripts/run.py", "app.logging").as_deref(),
            Some("app/logging.py")
        );
        assert_eq!(
            py("app/main.py", ".logging").as_deref(),
            Some("app/logging.py")
        );
    }

    /// The boundary bug: a suffix match without one made `util.rs` match
    /// `db_util.rs`, and the alphabetically smallest candidate won.
    #[test]
    fn rust_matches_a_module_file_on_a_path_boundary() {
        let files = ["src/main.rs", "src/db_util.rs", "src/net/util.rs"];
        let rs = |source: &str| resolve(Lang::Rust, "src/main.rs", source, &files);
        assert_eq!(rs("util::slugify").as_deref(), Some("src/net/util.rs"));
        assert_eq!(rs("db_util::open").as_deref(), Some("src/db_util.rs"));
        assert_eq!(rs("std::collections::HashMap"), None);
    }

    /// `crate::`, `self::` and `super::` follow the module tree from where
    /// the importer sits; a same-named file in another crate is not a
    /// candidate, and without that anchor two candidates mean no edge.
    #[test]
    fn rust_anchors_crate_self_and_super_to_the_importers_crate() {
        let files = [
            "crates/core/src/lib.rs",
            "crates/core/src/error.rs",
            "crates/core/src/index/mod.rs",
            "crates/core/src/index/chunker.rs",
            "crates/core/src/index/walk.rs",
            "crates/cli/src/main.rs",
            "crates/cli/src/error.rs",
            "crates/cli/tests/e2e.rs",
        ];
        let rs = |importer: &str, source: &str| resolve(Lang::Rust, importer, source, &files);
        assert_eq!(
            rs("crates/core/src/index/chunker.rs", "crate::error::Result").as_deref(),
            Some("crates/core/src/error.rs")
        );
        assert_eq!(
            rs("crates/cli/src/main.rs", "crate::error::Result").as_deref(),
            Some("crates/cli/src/error.rs")
        );
        assert_eq!(
            rs("crates/core/src/index/chunker.rs", "super::walk::walk_dir").as_deref(),
            Some("crates/core/src/index/walk.rs")
        );
        assert_eq!(
            rs("crates/core/src/index/mod.rs", "self::chunker::chunk").as_deref(),
            Some("crates/core/src/index/chunker.rs")
        );
        assert_eq!(
            rs("crates/core/src/lib.rs", "crate::index::chunker::chunk").as_deref(),
            Some("crates/core/src/index/chunker.rs")
        );
        assert_eq!(
            rs("crates/core/src/lib.rs", "crate::index::Indexer").as_deref(),
            Some("crates/core/src/index/mod.rs")
        );
        // A child module named without an anchor.
        assert_eq!(
            rs("crates/core/src/lib.rs", "index::Indexer").as_deref(),
            Some("crates/core/src/index/mod.rs")
        );
        // `super::helper` inside an inline `mod tests` is an item of the
        // file itself: nothing in the parent's directory, so no edge.
        assert_eq!(rs("crates/core/src/index/walk.rs", "super::helper"), None);
        // No `src/` above an integration test, and two `error.rs`: no edge.
        assert_eq!(rs("crates/cli/tests/e2e.rs", "crate::error::Result"), None);
        assert_eq!(rs("crates/cli/tests/e2e.rs", "error::Result"), None);
    }

    /// `pub use other_crate::languages;` makes `crate::languages` a path of
    /// this crate with no file in it. The module is wherever the project's
    /// one `languages.rs` is — and nowhere, once there are two.
    #[test]
    fn rust_follows_a_re_exported_module_to_its_only_file() {
        let mut files = vec![
            "crates/graph/src/lib.rs",
            "crates/graph/src/store.rs",
            "crates/syntax/src/lib.rs",
            "crates/syntax/src/languages.rs",
        ];
        let importer = "crates/graph/src/store.rs";
        assert_eq!(
            resolve(Lang::Rust, importer, "crate::languages::Lang", &files).as_deref(),
            Some("crates/syntax/src/languages.rs")
        );
        // An item at the crate root is not a module anywhere.
        assert_eq!(resolve(Lang::Rust, importer, "crate::Error", &files), None);
        files.push("crates/other/src/languages.rs");
        assert_eq!(
            resolve(Lang::Rust, importer, "crate::languages::Lang", &files),
            None
        );
    }

    /// A binary under `src/bin/` is a crate of its own, and `crate::` in it
    /// starts there. Resolved from `src`, `crate::helper` landed on the
    /// library's `helper.rs` — a file the binary cannot see.
    #[test]
    fn rust_crate_paths_in_a_binary_start_at_the_binarys_own_root() {
        let files = [
            "crates/a/src/lib.rs",
            "crates/a/src/helper.rs",
            "crates/a/src/shared.rs",
            "crates/a/src/bin/tool.rs",
            "crates/a/src/bin/helper.rs",
            "crates/a/src/bin/big/main.rs",
            "crates/a/src/bin/big/helper.rs",
            "crates/a/src/bin/big/sub/deep.rs",
        ];
        let rs = |importer: &str, source: &str| resolve(Lang::Rust, importer, source, &files);
        assert_eq!(
            rs("crates/a/src/bin/tool.rs", "crate::helper::run").as_deref(),
            Some("crates/a/src/bin/helper.rs")
        );
        // A binary with a directory of its own.
        assert_eq!(
            rs("crates/a/src/bin/big/main.rs", "crate::helper::run").as_deref(),
            Some("crates/a/src/bin/big/helper.rs")
        );
        assert_eq!(
            rs("crates/a/src/bin/big/sub/deep.rs", "crate::helper::run").as_deref(),
            Some("crates/a/src/bin/big/helper.rs")
        );
        // The library's own files are unaffected.
        assert_eq!(
            rs("crates/a/src/lib.rs", "crate::helper::run").as_deref(),
            Some("crates/a/src/helper.rs")
        );
        // Not among the binary's modules: the library root is still tried.
        assert_eq!(
            rs("crates/a/src/bin/tool.rs", "crate::shared::x").as_deref(),
            Some("crates/a/src/shared.rs")
        );
    }

    #[test]
    fn go_resolves_a_package_that_is_one_file() {
        let files = [
            "cmd/app/main.go",
            "internal/slug/slug.go",
            "internal/slug/slug_test.go",
            "internal/store/a.go",
            "internal/store/b.go",
            "pkg/x/util/u.go",
            "pkg/y/util/u.go",
        ];
        let go = |source: &str| resolve(Lang::Go, "cmd/app/main.go", source, &files);
        // The module prefix is not a directory; the test file is not the package.
        assert_eq!(
            go("example.com/app/internal/slug").as_deref(),
            Some("internal/slug/slug.go")
        );
        // Two files in the package: a file edge cannot say which.
        assert_eq!(go("example.com/app/internal/store"), None);
        // Two directories answer to `util`; the longer tail picks one.
        assert_eq!(go("example.com/app/util"), None);
        assert_eq!(
            go("example.com/app/pkg/x/util").as_deref(),
            Some("pkg/x/util/u.go")
        );
        assert_eq!(go("fmt"), None);
    }

    #[test]
    fn java_resolves_a_class_by_its_package_path() {
        let files = [
            "src/main/java/com/acme/App.java",
            "src/main/java/com/acme/util/Slug.java",
            "src/test/java/com/acme/AppTest.java",
            "other/Slug.java",
        ];
        let java = |source: &str| {
            resolve(
                Lang::Java,
                "src/main/java/com/acme/App.java",
                source,
                &files,
            )
        };
        assert_eq!(
            java("com.acme.util.Slug").as_deref(),
            Some("src/main/java/com/acme/util/Slug.java")
        );
        // `import static com.acme.util.Slug.of;` names a member of the class.
        assert_eq!(
            java("com.acme.util.Slug.of").as_deref(),
            Some("src/main/java/com/acme/util/Slug.java")
        );
        assert_eq!(java("org.junit.Test"), None);
    }

    /// A source root is usually *called* `java` or `kotlin`. With the
    /// project's own `util.Optional` under it, `java.util.Optional` ended in
    /// the same path and was resolved to it.
    #[test]
    fn jvm_platform_packages_are_never_project_files() {
        let files = [
            "src/main/java/util/Optional.java",
            "src/main/java/app/Main.java",
            "src/main/kotlin/collections/List.kt",
            "src/main/kotlin/app/Main.kt",
            "lib/Helper.java",
        ];
        let java =
            |source: &str| resolve(Lang::Java, "src/main/java/app/Main.java", source, &files);
        assert_eq!(java("java.util.Optional"), None);
        assert_eq!(java("javax.util.Optional"), None);
        // The project's own class of that name, by its real package.
        assert_eq!(
            java("util.Optional").as_deref(),
            Some("src/main/java/util/Optional.java")
        );
        let kt =
            |source: &str| resolve(Lang::Kotlin, "src/main/kotlin/app/Main.kt", source, &files);
        assert_eq!(kt("kotlin.collections.List"), None);
        assert_eq!(kt("kotlinx.collections.List"), None);
        assert_eq!(kt("android.util.Optional"), None);
        assert_eq!(kt("androidx.util.Optional"), None);
        assert_eq!(kt("scala.collections.List"), None);
        // A bare class name would match on the file name alone.
        assert_eq!(java("Helper"), None);
        assert_eq!(java("Helper.make"), None);
    }

    #[test]
    fn kotlin_resolves_a_class_in_either_jvm_language() {
        let files = [
            "app/src/main/kotlin/com/acme/Main.kt",
            "app/src/main/kotlin/com/acme/data/Repo.kt",
            "app/src/main/java/com/acme/legacy/Old.java",
        ];
        let kt = |source: &str| {
            resolve(
                Lang::Kotlin,
                "app/src/main/kotlin/com/acme/Main.kt",
                source,
                &files,
            )
        };
        assert_eq!(
            kt("com.acme.data.Repo").as_deref(),
            Some("app/src/main/kotlin/com/acme/data/Repo.kt")
        );
        assert_eq!(
            kt("com.acme.legacy.Old").as_deref(),
            Some("app/src/main/java/com/acme/legacy/Old.java")
        );
        assert_eq!(kt("kotlin.collections.List"), None);
    }

    #[test]
    fn c_resolves_an_include_beside_the_file_or_by_a_path_with_a_directory() {
        let files = [
            "src/main.c",
            "src/util.h",
            "include/proj/api.h",
            "vendor/a/inc/config.h",
            "vendor/b/inc/config.h",
            "src/db_util.h",
        ];
        let c = |source: &str| resolve(Lang::C, "src/main.c", source, &files);
        assert_eq!(c("util.h").as_deref(), Some("src/util.h"));
        assert_eq!(c("./util.h").as_deref(), Some("src/util.h"));
        // Found through an include path: the only file the path ends in.
        assert_eq!(c("proj/api.h").as_deref(), Some("include/proj/api.h"));
        // Two files end in it: no edge.
        assert_eq!(c("inc/config.h"), None);
        assert_eq!(c("stdio.h"), None);
    }

    /// A bare header name that is not beside the including file is not
    /// searched for across the project: the only `math.h` in the tree, in
    /// some other directory, is not what `"math.h"` means here.
    #[test]
    fn c_does_not_match_a_bare_header_name_project_wide() {
        let files = ["src/main.c", "geom/math.h", "include/proj/api.h"];
        let c = |importer: &str, source: &str| resolve(Lang::C, importer, source, &files);
        assert_eq!(c("src/main.c", "math.h"), None);
        assert_eq!(c("src/main.c", "api.h"), None);
        // Beside the including file, it is that file.
        assert_eq!(c("geom/shape.c", "math.h").as_deref(), Some("geom/math.h"));
        assert_eq!(resolve(Lang::Cpp, "src/main.cpp", "math.h", &files), None);
    }

    #[test]
    fn cpp_resolves_a_relative_include() {
        let files = ["src/app/main.cpp", "src/core/engine.hpp", "src/app/view.h"];
        let cpp = |source: &str| resolve(Lang::Cpp, "src/app/main.cpp", source, &files);
        assert_eq!(
            cpp("../core/engine.hpp").as_deref(),
            Some("src/core/engine.hpp")
        );
        assert_eq!(cpp("view.h").as_deref(), Some("src/app/view.h"));
        assert_eq!(cpp("vector"), None);
    }

    #[test]
    fn php_resolves_a_required_file_and_a_psr4_class() {
        let files = [
            "public/index.php",
            "public/bootstrap.php",
            "config/app.php",
            "lib/tools/fmt.php",
            "src/Models/User.php",
            "src/Http/Request.php",
            "tests/Http/Request.php",
        ];
        let php = |source: &str| resolve(Lang::Php, "public/index.php", source, &files);
        assert_eq!(
            php("bootstrap.php").as_deref(),
            Some("public/bootstrap.php")
        );
        assert_eq!(php("../config/app.php").as_deref(), Some("config/app.php"));
        assert_eq!(php("config/app.php").as_deref(), Some("config/app.php"));
        assert_eq!(php("tools/fmt.php").as_deref(), Some("lib/tools/fmt.php"));
        // A bare file name that is neither beside nor at the root.
        assert_eq!(php("fmt.php"), None);
        // `App\` maps to `src/`: the path below the prefix is what is on disk.
        assert_eq!(
            php("App\\Models\\User").as_deref(),
            Some("src/Models/User.php")
        );
        // Two files end in `Http/Request.php`.
        assert_eq!(php("App\\Http\\Request"), None);
        assert_eq!(php("Psr\\Log\\LoggerInterface"), None);
    }

    /// A dependency's class is not the project's only file of that name:
    /// `Illuminate\Http\Request` was resolved to `app/Http/Requests/Request.php`
    /// because nothing else was called `Request.php`.
    #[test]
    fn php_does_not_match_a_class_by_its_file_name_alone() {
        let files = [
            "app/Http/Controllers/HomeController.php",
            "app/Http/Requests/Request.php",
            "app/Models/User.php",
        ];
        let php = |source: &str| {
            resolve(
                Lang::Php,
                "app/Http/Controllers/HomeController.php",
                source,
                &files,
            )
        };
        assert_eq!(php("Illuminate\\Http\\Request"), None);
        assert_eq!(php("Request"), None);
        assert_eq!(php("User"), None);
        // With its own namespace it is found.
        assert_eq!(
            php("App\\Http\\Requests\\Request").as_deref(),
            Some("app/Http/Requests/Request.php")
        );
        assert_eq!(
            php("App\\Models\\User").as_deref(),
            Some("app/Models/User.php")
        );
    }

    /// A namespace or a framework is not a file, whatever is on disk.
    #[test]
    fn csharp_and_swift_imports_name_no_file() {
        let files = ["App/Text.cs", "Sources/Foundation.swift"];
        assert_eq!(
            resolve(Lang::CSharp, "App/Main.cs", "System.Text", &files),
            None
        );
        assert_eq!(
            resolve(Lang::Swift, "Sources/main.swift", "Foundation", &files),
            None
        );
    }
}
