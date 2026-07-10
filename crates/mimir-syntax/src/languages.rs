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
    Java,
    Ruby,
    C,
    CSharp,
    Sql,
    Cpp,
    Kotlin,
    Swift,
    Php,
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
            "java" => Lang::Java,
            "rb" | "rake" => Lang::Ruby,
            "c" => Lang::C,
            // `.h` is deliberately Cpp, not C: the C++ grammar parses
            // C-style headers acceptably, but the reverse isn't true (C
            // can't parse `class`/`namespace`/templates at all). `.c` stays
            // its own bucket since a `.c` file is never C++.
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "h" => Lang::Cpp,
            "kt" | "kts" => Lang::Kotlin,
            "swift" => Lang::Swift,
            "php" => Lang::Php,
            "cs" | "csx" => Lang::CSharp,
            "sql" => Lang::Sql,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::TypeScript | Lang::Tsx => "typescript",
            Lang::Python => "python",
            Lang::Go => "go",
            Lang::Java => "java",
            Lang::Ruby => "ruby",
            Lang::C => "c",
            Lang::CSharp => "csharp",
            Lang::Sql => "sql",
            Lang::Cpp => "cpp",
            Lang::Kotlin => "kotlin",
            Lang::Swift => "swift",
            Lang::Php => "php",
        }
    }

    pub fn language(&self) -> tree_sitter::Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
            Lang::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Lang::Sql => tree_sitter_sequel_tsql::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Lang::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Lang::Swift => tree_sitter_swift::LANGUAGE.into(),
            Lang::Php => tree_sitter_php::LANGUAGE_PHP.into(),
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
            Lang::Java => match node.kind() {
                "class_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "interface_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "interface"))
                }
                "enum_declaration" => Some((text(node.child_by_field_name("name")?), "enum")),
                "record_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "method_declaration" => Some((text(node.child_by_field_name("name")?), "method")),
                "constructor_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "method"))
                }
                _ => None,
            },
            Lang::Ruby => match node.kind() {
                "class" => Some((const_name(node.child_by_field_name("name")?, src), "class")),
                "module" => Some((const_name(node.child_by_field_name("name")?, src), "module")),
                "method" => Some((text(node.child_by_field_name("name")?), "method")),
                // def self.foo — a class method.
                "singleton_method" => Some((text(node.child_by_field_name("name")?), "method")),
                _ => None,
            },
            Lang::C => match node.kind() {
                "function_definition" => Some((
                    c_declarator_name(node.child_by_field_name("declarator")?, src)?,
                    "function",
                )),
                "struct_specifier" => Some((text(node.child_by_field_name("name")?), "struct")),
                "union_specifier" => Some((text(node.child_by_field_name("name")?), "struct")),
                "enum_specifier" => Some((text(node.child_by_field_name("name")?), "enum")),
                _ => None,
            },
            Lang::CSharp => match node.kind() {
                // Both `namespace Foo { .. }` and file-scoped `namespace Foo;`
                // — a scope that qualifies the types nested under it.
                "namespace_declaration" | "file_scoped_namespace_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "namespace"))
                }
                "class_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "interface_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "interface"))
                }
                "struct_declaration" => Some((text(node.child_by_field_name("name")?), "struct")),
                "enum_declaration" => Some((text(node.child_by_field_name("name")?), "enum")),
                // Records are reference types by default; treat as a class.
                "record_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "record_struct_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "struct"))
                }
                "method_declaration" => Some((text(node.child_by_field_name("name")?), "method")),
                "property_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "property"))
                }
                // Constructor's name field is the enclosing type identifier.
                "constructor_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "constructor"))
                }
                _ => None,
            },
            Lang::Sql => match node.kind() {
                "create_table" => sql_def_name(node, src).map(|n| (n, "table")),
                "create_view" => sql_def_name(node, src).map(|n| (n, "view")),
                "create_function" => sql_def_name(node, src).map(|n| (n, "function")),
                "create_procedure" => sql_def_name(node, src).map(|n| (n, "procedure")),
                _ => None,
            },
            Lang::Cpp => match node.kind() {
                // Bare declarator name, or (out-of-line) `Type::method` —
                // the qualifier is kept so `Greeter::Format` defined outside
                // its class body still nests under "Greeter" and reads as a
                // method regardless of enclosing namespace scope, mirroring
                // Go's receiver-type prefix for the same reason.
                "function_definition" => {
                    let name =
                        c_declarator_name(node.child_by_field_name("declarator")?, src)?;
                    let kind = if name.contains("::") { "method" } else { "function" };
                    Some((name, kind))
                }
                "class_specifier" => Some((text(node.child_by_field_name("name")?), "class")),
                "struct_specifier" => Some((text(node.child_by_field_name("name")?), "struct")),
                "union_specifier" => Some((text(node.child_by_field_name("name")?), "struct")),
                "enum_specifier" => Some((text(node.child_by_field_name("name")?), "enum")),
                "namespace_definition" => {
                    Some((text(node.child_by_field_name("name")?), "namespace"))
                }
                _ => None,
            },
            // kotlin-ng's declaration nodes carry a `name` field (only its
            // calls/imports/navigation are fieldless — see call()/imports()
            // below), so definitions read the same as any other adapter.
            Lang::Kotlin => match node.kind() {
                "class_declaration" => {
                    let name = text(node.child_by_field_name("name")?);
                    let mut cursor = node.walk();
                    let kind = if node
                        .children(&mut cursor)
                        .any(|c| c.kind() == "enum_class_body")
                    {
                        "enum"
                    } else {
                        "class"
                    };
                    Some((name, kind))
                }
                "object_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "object"))
                }
                "function_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                _ => None,
            },
            // Swift's `class_declaration` covers class/struct/actor/extension,
            // disambiguated by its `declaration_kind` field (the literal
            // keyword). `function_declaration`'s `name` field matches twice
            // in this grammar (once for the identifier, once for an
            // optionally-present return type reusing the same field id) —
            // `child_by_field_name` returns the first match in document
            // order, which is always the identifier since it precedes any
            // `-> ReturnType`.
            Lang::Swift => match node.kind() {
                "class_declaration" => {
                    let kind = match node
                        .child_by_field_name("declaration_kind")
                        .map(text)
                        .as_deref()
                    {
                        Some("struct") => "struct",
                        Some("enum") => "enum",
                        _ => "class",
                    };
                    Some((text(node.child_by_field_name("name")?), kind))
                }
                "protocol_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "interface"))
                }
                "function_declaration" | "protocol_function_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                // `init`'s `name` field is the literal `init` keyword token.
                "init_declaration" => Some(("init".to_string(), "constructor")),
                _ => None,
            },
            Lang::Php => match node.kind() {
                "function_definition" => {
                    Some((text(node.child_by_field_name("name")?), "function"))
                }
                "method_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "method"))
                }
                "class_declaration" => Some((text(node.child_by_field_name("name")?), "class")),
                "interface_declaration" => {
                    Some((text(node.child_by_field_name("name")?), "interface"))
                }
                "trait_declaration" => Some((text(node.child_by_field_name("name")?), "trait")),
                "enum_declaration" => Some((text(node.child_by_field_name("name")?), "enum")),
                "namespace_definition" => {
                    Some((text(node.child_by_field_name("name")?), "namespace"))
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
            Lang::Java => {
                if node.kind() != "method_invocation" {
                    return None;
                }
                node.child_by_field_name("name").map(text)
            }
            Lang::Ruby => {
                // `foo(...)`, `obj.foo(...)`, `obj.foo` — the method name.
                if node.kind() != "call" {
                    return None;
                }
                node.child_by_field_name("method").map(text)
            }
            Lang::C => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    _ => None,
                }
            }
            Lang::CSharp => {
                if node.kind() != "invocation_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" => Some(text(f)),
                    // obj.Method() / Type.Method() — the rightmost name.
                    "member_access_expression" => f.child_by_field_name("name").map(text),
                    _ => None,
                }
            }
            Lang::Sql => {
                // SQL has no calls; reuse the call edge for table dependencies.
                // A table reference is an `object_reference` sitting in a query
                // relation (FROM/JOIN), a DELETE/UPDATE `from` target, or a
                // column's `REFERENCES` (foreign key) clause. The enclosing
                // CREATE … is the caller, so the edge is view/proc/table → table.
                if node.kind() != "object_reference" {
                    return None;
                }
                match node.parent()?.kind() {
                    "relation" | "from" | "column_definition" => {
                        Some(text(node.child_by_field_name("name").unwrap_or(node)))
                    }
                    _ => None,
                }
            }
            Lang::Cpp => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let f = node.child_by_field_name("function")?;
                match f.kind() {
                    "identifier" | "field_identifier" => Some(text(f)),
                    // obj.method() / obj->method() — the field name.
                    "field_expression" => f.child_by_field_name("field").map(text),
                    // Foo::bar() — a namespace/static-qualified call.
                    "qualified_identifier" => f.child_by_field_name("name").map(text),
                    _ => None,
                }
            }
            // kotlin-ng's call_expression and navigation_expression are
            // fieldless — the callee is always the first named child
            // (identifier for a bare call, navigation_expression for a
            // `recv.method(...)` call, whose own last named child is the
            // member name).
            Lang::Kotlin => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let callee = node.named_child(0)?;
                match callee.kind() {
                    "identifier" => Some(text(callee)),
                    "navigation_expression" => {
                        let n = u32::try_from(callee.named_child_count())
                            .ok()?
                            .checked_sub(1)?;
                        let last = callee.named_child(n)?;
                        (last.kind() == "identifier").then(|| text(last))
                    }
                    _ => None,
                }
            }
            Lang::Swift => {
                if node.kind() != "call_expression" {
                    return None;
                }
                let callee = node.named_child(0)?;
                match callee.kind() {
                    "simple_identifier" => Some(text(callee)),
                    // `recv.method(...)` — navigation_expression's `suffix`
                    // field is a navigation_suffix, itself carrying the
                    // member name under its own `suffix` field.
                    "navigation_expression" => {
                        let suffix = callee.child_by_field_name("suffix")?;
                        suffix.child_by_field_name("suffix").map(text)
                    }
                    _ => None,
                }
            }
            Lang::Php => match node.kind() {
                "function_call_expression" => node.child_by_field_name("function").map(text),
                // obj->method(...) — the method name.
                "member_call_expression" => node.child_by_field_name("name").map(text),
                // Type::method(...) — a static/scoped call.
                "scoped_call_expression" => node.child_by_field_name("name").map(text),
                _ => None,
            },
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
            Lang::Java => {
                // import a.b.C;  /  import static a.b.C.m;  → bind the last segment.
                if node.kind() != "import_declaration" {
                    return;
                }
                let mut cursor = node.walk();
                let Some(scoped) = node
                    .children(&mut cursor)
                    .find(|c| c.kind() == "scoped_identifier")
                else {
                    return;
                };
                let source = text(scoped);
                let local = source.rsplit('.').next().unwrap_or(&source).to_string();
                out.push(ImportRef { local, source });
            }
            // C's preprocessor grammar is reused verbatim by C++, so one
            // arm covers both `#include` forms.
            Lang::C | Lang::Cpp => {
                // #include "foo.h" / <foo.h> → a file→file edge by path.
                if node.kind() != "preproc_include" {
                    return;
                }
                let Some(path_node) = node.child_by_field_name("path") else {
                    return;
                };
                let source = text(path_node).trim_matches(['"', '<', '>']).to_string();
                let local = source
                    .rsplit('/')
                    .next()
                    .unwrap_or(&source)
                    .trim_end_matches(".h")
                    .to_string();
                out.push(ImportRef { local, source });
            }
            Lang::CSharp => {
                // `using A.B;` / `using static A.B.C;` / `using X = A.B;` —
                // bind the last namespace segment, or the alias when present.
                if node.kind() != "using_directive" {
                    return;
                }
                let mut cursor = node.walk();
                let names: Vec<String> = node
                    .children(&mut cursor)
                    .filter(|c| {
                        matches!(
                            c.kind(),
                            "identifier" | "qualified_name" | "alias_qualified_name"
                        )
                    })
                    .map(text)
                    .collect();
                match names.as_slice() {
                    // `using Alias = Some.Namespace;`
                    [alias, source, ..] => out.push(ImportRef {
                        local: alias.clone(),
                        source: source.clone(),
                    }),
                    // `using Some.Namespace;` — local is the last segment.
                    [source] => out.push(ImportRef {
                        local: source.rsplit('.').next().unwrap_or(source).to_string(),
                        source: source.clone(),
                    }),
                    [] => {}
                }
            }
            // Ruby's `require` is a method call, not an import node; calls
            // still resolve same-file (tier 1) and globally by name (tier 3).
            Lang::Ruby => {}
            // SQL has no import construct.
            Lang::Sql => {}
            Lang::Kotlin => {
                if node.kind() != "import" {
                    return;
                }
                let mut cursor = node.walk();
                let children: Vec<Node> = node.children(&mut cursor).collect();
                if children.iter().any(|c| c.kind() == "*") {
                    return; // Wildcard import — no single symbol to bind.
                }
                let Some(path) = children.iter().find(|c| c.kind() == "qualified_identifier")
                else {
                    return;
                };
                let source = text(*path);
                let local = children
                    .iter()
                    .find(|c| c.kind() == "identifier")
                    .map(|c| text(*c))
                    .unwrap_or_else(|| {
                        source.rsplit('.').next().unwrap_or(&source).to_string()
                    });
                out.push(ImportRef { local, source });
            }
            Lang::Swift => {
                if node.kind() != "import_declaration" {
                    return;
                }
                let mut cursor = node.walk();
                let Some(path) = node.children(&mut cursor).find(|c| c.kind() == "identifier")
                else {
                    return;
                };
                let source = text(path);
                let local = source.rsplit('.').next().unwrap_or(&source).to_string();
                out.push(ImportRef { local, source });
            }
            Lang::Php => match node.kind() {
                "namespace_use_clause" => {
                    let mut cursor = node.walk();
                    let Some(path) = node
                        .children(&mut cursor)
                        .find(|c| matches!(c.kind(), "name" | "qualified_name"))
                    else {
                        return;
                    };
                    let source = text(path);
                    let local = node.child_by_field_name("alias").map(text).unwrap_or_else(|| {
                        source.rsplit('\\').next().unwrap_or(&source).to_string()
                    });
                    out.push(ImportRef { local, source });
                }
                "require_expression" | "require_once_expression" | "include_expression"
                | "include_once_expression" => {
                    let Some(expr) = node.named_child(0) else {
                        return;
                    };
                    if expr.kind() != "string" {
                        return;
                    }
                    let source = text(expr).trim_matches(['\'', '"']).to_string();
                    let local = source
                        .rsplit('/')
                        .next()
                        .unwrap_or(&source)
                        .trim_end_matches(".php")
                        .to_string();
                    out.push(ImportRef { local, source });
                }
                _ => {}
            },
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
            Lang::Rust
            | Lang::Go
            | Lang::TypeScript
            | Lang::Tsx
            | Lang::Java
            | Lang::C
            | Lang::CSharp
            | Lang::Ruby
            | Lang::Sql
            | Lang::Cpp
            | Lang::Kotlin
            | Lang::Swift
            | Lang::Php => {
                // Contiguous comment siblings directly above the node
                // (a blank line breaks the chain; `//!` belongs to the
                // module, not this item).
                // SQL wraps each `CREATE …` in a `statement`, so the comment
                // is a sibling of that wrapper — climb to it first.
                let mut anchor = node;
                if matches!(self, Lang::Sql) {
                    while let Some(p) = anchor.parent() {
                        if p.kind() == "statement" {
                            anchor = p;
                        } else {
                            break;
                        }
                    }
                }
                let mut lines: Vec<String> = Vec::new();
                let mut expect_row = anchor.start_position().row;
                let mut prev = anchor.prev_sibling();
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
                            .trim_start_matches("--") // SQL line comments
                            .trim_start_matches("/**")
                            .trim_start_matches("/*")
                            .trim_end_matches("*/")
                            .trim_start_matches('*')
                            .trim_start_matches('#') // Ruby line comments
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

/// SQL `CREATE TABLE`/`VIEW`/`FUNCTION`/`PROCEDURE` name: the first
/// `object_reference` child; keep its bare `name` field (drops any schema
/// qualifier so `dbo.users` and `users` resolve to the same bucket).
fn sql_def_name(node: Node, src: &str) -> Option<String> {
    let mut cursor = node.walk();
    let obj = node
        .children(&mut cursor)
        .find(|c| c.kind() == "object_reference")?;
    let name = obj.child_by_field_name("name").unwrap_or(obj);
    Some(src[name.byte_range()].to_string())
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

/// Ruby class/module name: a bare `constant` or `A::B` scope_resolution —
/// keep the last segment as the local name.
fn const_name(node: Node, src: &str) -> String {
    let full = src[node.byte_range()].to_string();
    full.rsplit("::").next().unwrap_or(&full).to_string()
}

/// C function name: unwrap pointer/function declarators down to the
/// identifier — `*foo(...)`, `foo(...)`, `(*foo)(...)` all yield "foo".
fn c_declarator_name(node: Node, src: &str) -> Option<String> {
    match node.kind() {
        // `field_identifier` is C++-only: an in-class inline method's
        // declarator (e.g. `std::string greet() { … }` inside a
        // `class_specifier` body) uses this kind instead of plain
        // `identifier`, and it's a leaf with no children, so without this
        // arm the fallback below would walk into nothing and drop the
        // symbol entirely.
        "identifier" | "field_identifier" => Some(src[node.byte_range()].to_string()),
        "function_declarator" | "pointer_declarator" | "parenthesized_declarator" => {
            c_declarator_name(node.child_by_field_name("declarator")?, src)
        }
        // C++ out-of-line member definition: `Type::method(...)`. Keep the
        // qualifier so the symbol still nests under its class
        // ("Type::method"), mirroring Go's receiver-type prefix for the
        // same reason — without it, the fallback below would walk into
        // `scope` first and return the class name instead of the method.
        "qualified_identifier" => {
            let name = c_declarator_name(node.child_by_field_name("name")?, src)?;
            match node.child_by_field_name("scope") {
                Some(scope) => Some(format!("{}::{name}", &src[scope.byte_range()])),
                None => Some(name),
            }
        }
        _ => {
            // Fall back to the first identifier descendant.
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(name) = c_declarator_name(child, src) {
                    return Some(name);
                }
            }
            None
        }
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
