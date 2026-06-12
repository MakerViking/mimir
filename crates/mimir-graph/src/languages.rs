//! Per-language tree-sitter adapters: which AST nodes are definitions,
//! scopes, calls, and imports — and how to read docs/signatures off them.

use tree_sitter::Node;

use crate::extract::ImportRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    TypeScript,
    Tsx,
    Python,
    Go,
}

impl Lang {
    pub fn from_path(path: &str) -> Option<Lang> {
        let ext = path.rsplit('.').next()?;
        Some(match ext {
            "rs" => Lang::Rust,
            "ts" | "mts" | "cts" => Lang::TypeScript,
            "tsx" | "jsx" | "js" | "mjs" | "cjs" => Lang::Tsx,
            "py" | "pyi" => Lang::Python,
            "go" => Lang::Go,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::TypeScript | Lang::Tsx => "typescript",
            Lang::Python => "python",
            Lang::Go => "go",
        }
    }

    pub fn language(&self) -> tree_sitter::Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }

    pub fn separator(&self) -> String {
        "::".into()
    }

    /// Is this node a symbol definition? Returns (name, kind). The name may
    /// be pre-qualified with `::` (Go methods carry their receiver).
    pub fn definition(&self, node: Node, src: &str) -> Option<(String, &'static str)> {
        let text = |n: Node| src[n.byte_range()].to_string();
        match self {
            Lang::Rust => match node.kind() {
                "function_item" => Some((text(node.child_by_field_name("name")?), "function")),
                "struct_item" => Some((text(node.child_by_field_name("name")?), "struct")),
                "enum_item" => Some((text(node.child_by_field_name("name")?), "enum")),
                "trait_item" => Some((text(node.child_by_field_name("name")?), "trait")),
                "union_item" => Some((text(node.child_by_field_name("name")?), "struct")),
                _ => None,
            },
            Lang::TypeScript | Lang::Tsx => match node.kind() {
                "function_declaration" | "generator_function_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                "class_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "method_definition" => {
                    let name = text(node.child_by_field_name("name")?);
                    if name == "constructor" {
                        return None;
                    }
                    Some((name, "method"))
                }
                "interface_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "interface"))
                }
                "enum_declaration" => Some((text(node.child_by_field_name("name")?), "enum")),
                "type_alias_declaration" => Some((text(node.child_by_field_name("name")?), "type")),
                // const f = (..) => ..  /  const f = function(..) {..}
                "variable_declarator" => {
                    let value = node.child_by_field_name("value")?;
                    if matches!(value.kind(), "arrow_function" | "function_expression") {
                        let name = node.child_by_field_name("name")?;
                        if name.kind() == "identifier" {
                            return Some((text(name), "function"));
                        }
                    }
                    None
                }
                _ => None,
            },
            Lang::Python => match node.kind() {
                "function_definition" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                "class_definition" => Some((text(node.child_by_field_name("name")?), "class")),
                _ => None,
            },
            Lang::Go => match node.kind() {
                "function_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                "method_declaration" => {
                    let name = text(node.child_by_field_name("name")?);
                    let recv = node
                        .child_by_field_name("receiver")
                        .and_then(|r| receiver_type(r, src));
                    Some((
                        match recv {
                            Some(t) => format!("{t}::{name}"),
                            None => name,
                        },
                        "method",
                    ))
                }
                "type_spec" => {
                    let name = text(node.child_by_field_name("name")?);
                    let kind = match node.child_by_field_name("type").map(|t| t.kind()) {
                        Some("struct_type") => "struct",
                        Some("interface_type") => "interface",
                        _ => "type",
                    };
                    Some((name, kind))
                }
                _ => None,
            },
        }
    }

    /// Containers that qualify children without being symbols themselves.
    pub fn scope_only(&self, node: Node, src: &str) -> Option<String> {
        match self {
            Lang::Rust => match node.kind() {
                // impl Foo { .. } / impl Trait for Foo { .. } → scope "Foo"
                "impl_item" => {
                    let ty = node.child_by_field_name("type")?;
                    Some(base_type_name(ty, src))
                }
                "mod_item" => Some(src[node.child_by_field_name("name")?.byte_range()].to_string()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Field holding the body (cut point for signatures), per node kind.
    pub fn body_field(&self) -> Option<&'static str> {
        // All supported definition kinds use "body" except TS declarators,
        // which signature_text handles via the generic fallback.
        Some("body")
    }

    /// If this node is a call, return the bare callee name.
    pub fn call(&self, node: Node, src: &str) -> Option<String> {
        let text = |n: Node| src[n.byte_range()].to_string();
        match self {
            Lang::Rust => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    "field_expression" => f.child_by_field_name("field").map(text),
                    "scoped_identifier" => f.child_by_field_name("name").map(text),
                    "generic_function" => {
                        let inner = f.child_by_field_name("function")?;
                        match inner.kind() {
                            "identifier" => Some(text(inner)),
                            "scoped_identifier" => inner.child_by_field_name("name").map(text),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            Lang::TypeScript | Lang::Tsx => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    "member_expression" => f.child_by_field_name("property").map(text),
                    _ => None,
                }
            }
            Lang::Python => {
                if node.kind() != "call" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    "attribute" => f.child_by_field_name("attribute").map(text),
                    _ => None,
                }
            }
            Lang::Go => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    "selector_expression" => f.child_by_field_name("field").map(text),
                    _ => None,
                }
            }
        }
    }

    /// Collect imports declared by this node.
    pub fn imports(&self, node: Node, src: &str, out: &mut Vec<ImportRef>) {
        let text = |n: Node| src[n.byte_range()].to_string();
        match self {
            Lang::Rust => {
                if node.kind() == "use_declaration" {
                    if let Some(arg) = node.child_by_field_name("argument") {
                        rust_use_tree(arg, src, "", out);
                    }
                }
            }
            Lang::TypeScript | Lang::Tsx => {
                if node.kind() != "import_statement" {
                    return;
                }
                let Some(source) = node
                    .child_by_field_name("source")
                    .map(|s| text(s).trim_matches(['"', '\'']).to_string())
                else {
                    return;
                };
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "import_clause" {
                        continue;
                    }
                    let mut c2 = child.walk();
                    for part in child.children(&mut c2) {
                        match part.kind() {
                            "identifier" => out.push(ImportRef {
                                local: text(part),
                                source: source.clone(),
                            }),
                            "named_imports" => {
                                let mut c3 = part.walk();
                                for spec in part.children(&mut c3) {
                                    if spec.kind() != "import_specifier" {
                                        continue;
                                    }
                                    let local = spec
                                        .child_by_field_name("alias")
                                        .or_else(|| spec.child_by_field_name("name"))
                                        .map(text);
                                    if let Some(local) = local {
                                        out.push(ImportRef {
                                            local,
                                            source: source.clone(),
                                        });
                                    }
                                }
                            }
                            "namespace_import" => {
                                // import * as ns from "x"
                                let mut c3 = part.walk();
                                for id in part.children(&mut c3) {
                                    if id.kind() == "identifier" {
                                        out.push(ImportRef {
                                            local: text(id),
                                            source: source.clone(),
                                        });
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Lang::Python => match node.kind() {
                "import_statement" => {
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        match child.kind() {
                            "dotted_name" => out.push(ImportRef {
                                local: text(child)
                                    .rsplit('.')
                                    .next()
                                    .unwrap_or_default()
                                    .to_string(),
                                source: text(child),
                            }),
                            "aliased_import" => {
                                let name = child.child_by_field_name("name").map(text);
                                let alias = child.child_by_field_name("alias").map(text);
                                if let (Some(name), Some(alias)) = (name, alias) {
                                    out.push(ImportRef {
                                        local: alias,
                                        source: name,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "import_from_statement" => {
                    let Some(module) = node.child_by_field_name("module_name").map(text) else {
                        return;
                    };
                    let mut cursor = node.walk();
                    let mut past_import = false;
                    for child in node.children(&mut cursor) {
                        if child.kind() == "import" {
                            past_import = true;
                            continue;
                        }
                        if !past_import {
                            continue;
                        }
                        match child.kind() {
                            "dotted_name" => out.push(ImportRef {
                                local: text(child),
                                source: module.clone(),
                            }),
                            "aliased_import" => {
                                if let Some(alias) = child.child_by_field_name("alias").map(text) {
                                    out.push(ImportRef {
                                        local: alias,
                                        source: module.clone(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            Lang::Go => {
                if node.kind() != "import_spec" {
                    return;
                }
                let Some(path) = node
                    .child_by_field_name("path")
                    .map(|p| text(p).trim_matches('"').to_string())
                else {
                    return;
                };
                let local = node
                    .child_by_field_name("name")
                    .map(text)
                    .unwrap_or_else(|| path.rsplit('/').next().unwrap_or(&path).to_string());
                out.push(ImportRef {
                    local,
                    source: path,
                });
            }
        }
    }

    /// Doc comment attached to a definition node.
    pub fn doc_comment(&self, node: Node, src: &str) -> Option<String> {
        match self {
            Lang::Python => {
                // Docstring: first statement of the body is a string literal.
                let body = node.child_by_field_name("body")?;
                let first = body.named_child(0)?;
                if first.kind() != "expression_statement" {
                    return None;
                }
                let s = first.named_child(0)?;
                if s.kind() != "string" {
                    return None;
                }
                let raw = &src[s.byte_range()];
                let cleaned = raw
                    .trim_start_matches(['r', 'b', 'f', 'u', 'R', 'B', 'F', 'U'])
                    .trim_matches(['"', '\''])
                    .trim();
                Some(cleaned.lines().next().unwrap_or("").trim().to_string())
                    .filter(|s| !s.is_empty())
            }
            Lang::Rust | Lang::Go | Lang::TypeScript | Lang::Tsx => {
                // Contiguous comment siblings directly above the node
                // (a blank line breaks the chain; `//!` belongs to the
                // module, not this item).
                let mut lines: Vec<String> = Vec::new();
                let mut expect_row = node.start_position().row;
                let mut prev = node.prev_sibling();
                while let Some(p) = prev {
                    if !p.kind().contains("comment")
                        || expect_row.saturating_sub(p.end_position().row) > 1
                        || src[p.byte_range()].starts_with("//!")
                    {
                        break;
                    }
                    lines.push(src[p.byte_range()].to_string());
                    expect_row = p.start_position().row;
                    prev = p.prev_sibling();
                }
                if lines.is_empty() {
                    return None;
                }
                lines.reverse();
                let cleaned: Vec<String> = lines
                    .iter()
                    .flat_map(|c| c.lines())
                    .map(|l| {
                        l.trim()
                            .trim_start_matches("///")
                            .trim_start_matches("//!")
                            .trim_start_matches("//")
                            .trim_start_matches("/**")
                            .trim_start_matches("/*")
                            .trim_end_matches("*/")
                            .trim_start_matches('*')
                            .trim()
                            .to_string()
                    })
                    .filter(|l| !l.is_empty())
                    .collect();
                if cleaned.is_empty() {
                    None
                } else {
                    Some(cleaned.join(" ").chars().take(300).collect())
                }
            }
        }
    }
}

/// `impl Foo`, `impl Foo<T>`, `impl Trait for Foo<T>` → "Foo".
fn base_type_name(ty: Node, src: &str) -> String {
    match ty.kind() {
        "generic_type" => ty
            .child_by_field_name("type")
            .map(|t| src[t.byte_range()].to_string())
            .unwrap_or_else(|| src[ty.byte_range()].to_string()),
        _ => src[ty.byte_range()].to_string(),
    }
}

/// Go receiver `(s *Server)` → "Server".
fn receiver_type(receiver: Node, src: &str) -> Option<String> {
    let mut cursor = receiver.walk();
    for child in receiver.children(&mut cursor) {
        if child.kind() == "parameter_declaration" {
            let ty = child.child_by_field_name("type")?;
            let base = match ty.kind() {
                "pointer_type" => ty.named_child(0)?,
                _ => ty,
            };
            return Some(src[base.byte_range()].to_string());
        }
    }
    None
}

/// Rust use-tree walker: `use a::{b::C, d as E};` → C←a::b::C, E←a::d.
fn rust_use_tree(node: Node, src: &str, prefix: &str, out: &mut Vec<ImportRef>) {
    let text = |n: Node| src[n.byte_range()].to_string();
    let join = |prefix: &str, seg: &str| {
        if prefix.is_empty() {
            seg.to_string()
        } else {
            format!("{prefix}::{seg}")
        }
    };
    match node.kind() {
        "identifier" | "crate" | "self" | "super" => {
            let seg = text(node);
            out.push(ImportRef {
                local: seg.clone(),
                source: join(prefix, &seg),
            });
        }
        "scoped_identifier" => {
            let full = join(prefix, &text(node));
            let local = node
                .child_by_field_name("name")
                .map(text)
                .unwrap_or_default();
            if !local.is_empty() {
                out.push(ImportRef {
                    local,
                    source: full,
                });
            }
        }
        "use_as_clause" => {
            let alias = node.child_by_field_name("alias").map(text);
            let path = node.child_by_field_name("path").map(text);
            if let (Some(alias), Some(path)) = (alias, path) {
                out.push(ImportRef {
                    local: alias,
                    source: join(prefix, &path),
                });
            }
        }
        "scoped_use_list" => {
            let new_prefix = node
                .child_by_field_name("path")
                .map(|p| join(prefix, &text(p)))
                .unwrap_or_else(|| prefix.to_string());
            if let Some(list) = node.child_by_field_name("list") {
                let mut cursor = list.walk();
                for child in list.named_children(&mut cursor) {
                    rust_use_tree(child, src, &new_prefix, out);
                }
            }
        }
        "use_list" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                rust_use_tree(child, src, prefix, out);
            }
        }
        // use_wildcard and attributes: nothing useful to bind.
        _ => {}
    }
}
