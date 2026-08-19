//! Tree-sitter symbol/call/import extraction. No LLM, no type checker —
//! honest static extraction with explicit confidence tiers downstream.

use tree_sitter::{Node, Parser};

use crate::languages::Lang;

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolDef {
    /// Bare name (resolution bucket), e.g. "resolve_ref".
    pub name: String,
    /// Nesting-qualified, e.g. "MatrixCache::ensure" / "ClassName.method".
    pub qualified: String,
    /// function | method | struct | class | trait | enum | interface | type
    pub kind: &'static str,
    /// Signature line(s) — what gets embedded alongside the doc comment.
    pub signature: String,
    pub doc: Option<String>,
    /// 1-based, inclusive.
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallSite {
    /// Qualified name of the enclosing definition ("" = file top level).
    pub caller: String,
    /// Bare callee name as written (rightmost path segment).
    pub callee: String,
    /// Receiver expression as written, for `recv.callee(..)` forms —
    /// `self`, `this`, `self.db`, a variable, or a type name. `None` for a
    /// bare `callee(..)`. Resolution turns this into a type downstream.
    pub receiver: Option<String>,
}

/// A name observed to hold a value of a named type: a typed parameter, a
/// local `let x: T` / `x := T{}` / `x = T()`, or a struct/class field.
/// Static extraction, so it is a hint, not a type checker's verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// Enclosing scope as qualified name ("" = file top level). A field's
    /// scope is its type; a local's scope is its function.
    pub scope: String,
    /// The bound name (`db`, `self` never appears here).
    pub name: String,
    /// Base type name, wrappers unwrapped (`Arc<Db>` → `Db`).
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportRef {
    /// Name bound locally (rightmost segment or alias).
    pub local: String,
    /// Module/path text as written ("./util", "foo::bar", "pkg.mod").
    pub source: String,
}

#[derive(Debug, Default)]
pub struct FileExtract {
    pub symbols: Vec<SymbolDef>,
    pub calls: Vec<CallSite>,
    pub imports: Vec<ImportRef>,
    pub bindings: Vec<Binding>,
}

pub fn extract(lang: Lang, source: &str) -> FileExtract {
    let mut parser = Parser::new();
    if parser.set_language(&lang.language()).is_err() {
        return FileExtract::default();
    }
    let Some(tree) = parser.parse(source, None) else {
        return FileExtract::default();
    };
    let mut out = FileExtract::default();
    walk(lang, tree.root_node(), source, &mut Vec::new(), &mut out);
    out
}

/// Recursive walk keeping a stack of enclosing definition names.
fn walk(lang: Lang, node: Node, src: &str, scope: &mut Vec<String>, out: &mut FileExtract) {
    let mut pushed = false;

    if let Some((name, kind)) = lang.definition(node, src) {
        let qualified = qualify(scope, &name);
        // Methods: a function nested inside a type/class scope.
        let kind = if kind == "function" && !scope.is_empty() {
            "method"
        } else {
            kind
        };
        out.symbols.push(SymbolDef {
            signature: signature_text(lang, node, src),
            doc: lang.doc_comment(node, src),
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            name: qualified_tail(&name),
            qualified: qualified.clone(),
            kind,
        });
        // Push the name as returned (it may carry a `::` receiver prefix,
        // e.g. Go methods) so scope.join("::") == qualified for children.
        scope.push(name);
        pushed = true;
    } else if let Some(scope_name) = lang.scope_only(node, src) {
        // Containers that qualify children but aren't symbols themselves
        // (Rust impl blocks, modules).
        scope.push(scope_name);
        pushed = true;
    }

    if let Some(callee) = lang.call(node, src) {
        out.calls.push(CallSite {
            caller: scope.join(&lang.separator()),
            callee,
            receiver: lang.call_receiver(node, src),
        });
    }
    lang.imports(node, src, &mut out.imports);
    {
        // The Lang layer answers "what does this node bind?"; the scope it
        // was seen in is ours to attach.
        let mut bound: Vec<(String, String)> = Vec::new();
        lang.bindings(node, src, &mut bound);
        let here = scope.join(&lang.separator());
        for (name, type_name) in bound {
            out.bindings.push(Binding {
                scope: here.clone(),
                name,
                type_name,
            });
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(lang, child, src, scope, out);
    }
    if pushed {
        scope.pop();
    }
}

fn qualify(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}::{name}", scope.join("::"))
    }
}

fn qualified_tail(qualified: &str) -> String {
    qualified
        .rsplit("::")
        .next()
        .unwrap_or(qualified)
        .to_string()
}

/// Text from the definition start to its body — the signature.
fn signature_text(lang: Lang, node: Node, src: &str) -> String {
    let full = &src[node.byte_range()];
    let cut = lang
        .body_field()
        .and_then(|f| node.child_by_field_name(f))
        .map(|b| b.start_byte().saturating_sub(node.start_byte()))
        .unwrap_or(full.len());
    let sig: String = full[..cut].split_whitespace().collect::<Vec<_>>().join(" ");
    // Defensive cap: pathological one-line definitions.
    sig.chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(fx: &FileExtract) -> Vec<(&str, &str)> {
        fx.symbols
            .iter()
            .map(|s| (s.qualified.as_str(), s.kind))
            .collect()
    }

    #[test]
    fn rust_extraction() {
        let src = r#"
//! module docs

/// Adds things.
pub fn add(a: i32, b: i32) -> i32 { helper(a) + b }

fn helper(x: i32) -> i32 { x }

pub struct Counter { n: u64 }

impl Counter {
    /// Bump it.
    pub fn bump(&mut self) { self.n += 1; validate(self.n); }
}

pub trait Resettable { fn reset(&mut self); }

pub enum Mode { A, B }

use std::collections::HashMap;
use crate::store::resolve_ref as rr;
"#;
        let fx = extract(Lang::Rust, src);
        let n = names(&fx);
        assert!(n.contains(&("add", "function")), "{n:?}");
        assert!(n.contains(&("helper", "function")), "{n:?}");
        assert!(n.contains(&("Counter", "struct")), "{n:?}");
        assert!(n.contains(&("Counter::bump", "method")), "{n:?}");
        assert!(n.contains(&("Resettable", "trait")), "{n:?}");
        assert!(n.contains(&("Mode", "enum")), "{n:?}");

        let add = fx.symbols.iter().find(|s| s.qualified == "add").unwrap();
        assert_eq!(add.doc.as_deref(), Some("Adds things."));
        assert!(add.signature.contains("pub fn add(a: i32, b: i32) -> i32"));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("add", "helper")), "{calls:?}");
        assert!(calls.contains(&("Counter::bump", "validate")), "{calls:?}");

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(
            imports.contains(&("HashMap", "std::collections::HashMap")),
            "{imports:?}"
        );
        assert!(
            imports.contains(&("rr", "crate::store::resolve_ref")),
            "{imports:?}"
        );
    }

    #[test]
    fn typescript_extraction() {
        let src = r#"
import { fetchUser, postUser as pu } from "./api";
import db from "../db";

/** Greets. */
export function greet(name: string): string { return hello(name); }

const shout = (s: string) => s.toUpperCase();

export class UserService {
    find(id: number) { return fetchUser(id); }
}

interface Shape { area(): number; }
"#;
        let fx = extract(Lang::TypeScript, src);
        let n = names(&fx);
        assert!(n.contains(&("greet", "function")), "{n:?}");
        assert!(n.contains(&("shout", "function")), "{n:?}");
        assert!(n.contains(&("UserService", "class")), "{n:?}");
        assert!(n.contains(&("UserService::find", "method")), "{n:?}");
        assert!(n.contains(&("Shape", "interface")), "{n:?}");

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("greet", "hello")), "{calls:?}");
        assert!(
            calls.contains(&("UserService::find", "fetchUser")),
            "{calls:?}"
        );

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("fetchUser", "./api")), "{imports:?}");
        assert!(imports.contains(&("pu", "./api")), "{imports:?}");
        assert!(imports.contains(&("db", "../db")), "{imports:?}");
    }

    #[test]
    fn python_extraction() {
        let src = r#"
import os
from collections import OrderedDict as OD
from .util import slugify

def top(x):
    """Top-level docstring."""
    return slugify(x)

class Repo:
    def save(self, item):
        validate(item)
        return persist(item)
"#;
        let fx = extract(Lang::Python, src);
        let n = names(&fx);
        assert!(n.contains(&("top", "function")), "{n:?}");
        assert!(n.contains(&("Repo", "class")), "{n:?}");
        assert!(n.contains(&("Repo::save", "method")), "{n:?}");

        let top_sym = fx.symbols.iter().find(|s| s.qualified == "top").unwrap();
        assert_eq!(top_sym.doc.as_deref(), Some("Top-level docstring."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("top", "slugify")), "{calls:?}");
        assert!(calls.contains(&("Repo::save", "validate")), "{calls:?}");

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("os", "os")), "{imports:?}");
        assert!(imports.contains(&("OD", "collections")), "{imports:?}");
        assert!(imports.contains(&("slugify", ".util")), "{imports:?}");
    }

    #[test]
    fn go_extraction() {
        let src = r#"
package main

import (
    "fmt"
    alias "net/http"
)

// Greet says hi.
func Greet(name string) string { return fmt.Sprintf("hi %s", name) }

type Server struct{ port int }

func (s *Server) Start() error { return listen(s.port) }
"#;
        let fx = extract(Lang::Go, src);
        let n = names(&fx);
        assert!(n.contains(&("Greet", "function")), "{n:?}");
        assert!(n.contains(&("Server", "struct")), "{n:?}");
        assert!(n.contains(&("Server::Start", "method")), "{n:?}");

        let greet = fx.symbols.iter().find(|s| s.qualified == "Greet").unwrap();
        assert_eq!(greet.doc.as_deref(), Some("Greet says hi."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("Greet", "Sprintf")), "{calls:?}");
        assert!(calls.contains(&("Server::Start", "listen")), "{calls:?}");

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("fmt", "fmt")), "{imports:?}");
        assert!(imports.contains(&("alias", "net/http")), "{imports:?}");
    }

    #[test]
    fn broken_source_does_not_panic() {
        for lang in [
            Lang::Rust,
            Lang::TypeScript,
            Lang::Python,
            Lang::Go,
            Lang::CSharp,
            Lang::Sql,
            Lang::Cpp,
            Lang::Kotlin,
            Lang::Swift,
            Lang::Php,
        ] {
            extract(lang, "fn class def func ((((");
            extract(lang, "");
        }
    }

    #[test]
    fn csharp_extraction() {
        let src = r#"
using System;
using Data = App.Models;

namespace App
{
    // Greets users.
    public class Greeter
    {
        public string Name { get; set; }

        public Greeter(string name) { Name = name; }

        public string Greet() { return Format(Name); }

        private string Format(string n) { return n.ToUpper(); }
    }

    public interface IRunnable { void Run(); }

    public struct Point { public int X; }

    public record Person(string First, string Last);

    public enum Mode { On, Off }
}
"#;
        let fx = extract(Lang::CSharp, src);
        let n = names(&fx);
        assert!(n.contains(&("App", "namespace")), "{n:?}");
        assert!(n.contains(&("App::Greeter", "class")), "{n:?}");
        assert!(n.contains(&("App::Greeter::Greet", "method")), "{n:?}");
        assert!(n.contains(&("App::Greeter::Format", "method")), "{n:?}");
        assert!(n.contains(&("App::Greeter::Name", "property")), "{n:?}");
        assert!(
            n.contains(&("App::Greeter::Greeter", "constructor")),
            "{n:?}"
        );
        assert!(n.contains(&("App::IRunnable", "interface")), "{n:?}");
        assert!(n.contains(&("App::Point", "struct")), "{n:?}");
        assert!(n.contains(&("App::Person", "class")), "{n:?}");
        assert!(n.contains(&("App::Mode", "enum")), "{n:?}");

        let greeter = fx
            .symbols
            .iter()
            .find(|s| s.qualified == "App::Greeter")
            .unwrap();
        assert_eq!(greeter.doc.as_deref(), Some("Greets users."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(
            calls.contains(&("App::Greeter::Greet", "Format")),
            "{calls:?}"
        );
        assert!(
            calls.contains(&("App::Greeter::Format", "ToUpper")),
            "{calls:?}"
        );

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("System", "System")), "{imports:?}");
        assert!(imports.contains(&("Data", "App.Models")), "{imports:?}");
    }

    #[test]
    fn sql_extraction() {
        let src = r#"
-- People who use the system.
CREATE TABLE users (
  id INT PRIMARY KEY,
  name NVARCHAR(50)
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT REFERENCES users(id)
);

CREATE VIEW active_orders AS
  SELECT o.id FROM orders o JOIN users u ON o.user_id = u.id;

CREATE FUNCTION order_count() RETURNS INT AS
BEGIN
  RETURN (SELECT COUNT(*) FROM orders);
END;

CREATE PROCEDURE purge AS
BEGIN
  DELETE FROM orders;
END;
"#;
        let fx = extract(Lang::Sql, src);
        let n = names(&fx);
        assert!(n.contains(&("users", "table")), "{n:?}");
        assert!(n.contains(&("orders", "table")), "{n:?}");
        assert!(n.contains(&("active_orders", "view")), "{n:?}");
        assert!(n.contains(&("order_count", "function")), "{n:?}");
        assert!(n.contains(&("purge", "procedure")), "{n:?}");

        let users = fx.symbols.iter().find(|s| s.qualified == "users").unwrap();
        assert_eq!(users.doc.as_deref(), Some("People who use the system."));

        // Dependency edges ride the call edge: dependent → table.
        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("orders", "users")), "FK: {calls:?}");
        assert!(
            calls.contains(&("active_orders", "orders")),
            "view: {calls:?}"
        );
        assert!(
            calls.contains(&("active_orders", "users")),
            "view: {calls:?}"
        );
        assert!(
            calls.contains(&("order_count", "orders")),
            "function: {calls:?}"
        );
        assert!(calls.contains(&("purge", "orders")), "procedure: {calls:?}");

        // SQL has no import construct.
        assert!(fx.imports.is_empty(), "{:?}", fx.imports);
    }

    #[test]
    fn h_extension_maps_to_cpp() {
        // `.h` is deliberately routed to the C++ grammar (see from_path's
        // comment); `.c` stays plain C.
        assert_eq!(Lang::from_path("widget.h"), Some(Lang::Cpp));
        assert_eq!(Lang::from_path("legacy.c"), Some(Lang::C));

        // A `.h` header with a class (invalid in plain C) still extracts
        // cleanly under the C++ grammar.
        let src = r#"
#ifndef WIDGET_H
#define WIDGET_H

class Widget {
public:
    int size();
};

#endif
"#;
        let fx = extract(Lang::from_path("widget.h").unwrap(), src);
        let n = names(&fx);
        assert!(n.contains(&("Widget", "class")), "{n:?}");
    }

    #[test]
    fn cpp_extraction() {
        let src = r#"
#include "util.h"
#include <vector>

// Greets people.
class Greeter {
public:
    std::string Format(const std::string& name);

    std::string Greet(const std::string& name) {
        return Format(name);
    }
};

std::string Greeter::Format(const std::string& name) {
    return normalize(name);
}

namespace app {
    struct Point { int x; int y; };
}
"#;
        let fx = extract(Lang::Cpp, src);
        let n = names(&fx);
        assert!(n.contains(&("Greeter", "class")), "{n:?}");
        assert!(n.contains(&("Greeter::Greet", "method")), "{n:?}");
        assert!(n.contains(&("Greeter::Format", "method")), "{n:?}");
        assert!(n.contains(&("app::Point", "struct")), "{n:?}");

        let greeter = fx
            .symbols
            .iter()
            .find(|s| s.qualified == "Greeter")
            .unwrap();
        assert_eq!(greeter.doc.as_deref(), Some("Greets people."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("Greeter::Greet", "Format")), "{calls:?}");
        assert!(
            calls.contains(&("Greeter::Format", "normalize")),
            "{calls:?}"
        );

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("util", "util.h")), "{imports:?}");
        assert!(imports.contains(&("vector", "vector")), "{imports:?}");
    }

    #[test]
    fn kotlin_extraction() {
        let src = r#"
package com.example

import java.util.List
import com.example.util.Formatter as Fmt

// Greets people.
class Greeter(val name: String) {
    fun greet(): String {
        return format(name)
    }

    fun format(input: String): String {
        return Fmt.upper(input)
    }
}

enum class Mode { On, Off }
"#;
        let fx = extract(Lang::Kotlin, src);
        let n = names(&fx);
        assert!(n.contains(&("Greeter", "class")), "{n:?}");
        assert!(n.contains(&("Greeter::greet", "method")), "{n:?}");
        assert!(n.contains(&("Greeter::format", "method")), "{n:?}");
        assert!(n.contains(&("Mode", "enum")), "{n:?}");

        let greeter = fx
            .symbols
            .iter()
            .find(|s| s.qualified == "Greeter")
            .unwrap();
        assert_eq!(greeter.doc.as_deref(), Some("Greets people."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("Greeter::greet", "format")), "{calls:?}");
        assert!(calls.contains(&("Greeter::format", "upper")), "{calls:?}");

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(imports.contains(&("List", "java.util.List")), "{imports:?}");
        assert!(
            imports.contains(&("Fmt", "com.example.util.Formatter")),
            "{imports:?}"
        );
    }

    #[test]
    fn swift_extraction() {
        let src = r#"
import Foundation

// Greets people.
class Greeter {
    let name: String

    init(name: String) {
        self.name = name
    }

    func greet() -> String {
        return format(name)
    }

    func format(_ input: String) -> String {
        return input.uppercased()
    }
}

protocol Runnable {
    func run()
}
"#;
        let fx = extract(Lang::Swift, src);
        let n = names(&fx);
        assert!(n.contains(&("Greeter", "class")), "{n:?}");
        assert!(n.contains(&("Greeter::init", "constructor")), "{n:?}");
        assert!(n.contains(&("Greeter::greet", "method")), "{n:?}");
        assert!(n.contains(&("Greeter::format", "method")), "{n:?}");
        assert!(n.contains(&("Runnable", "interface")), "{n:?}");

        let greeter = fx
            .symbols
            .iter()
            .find(|s| s.qualified == "Greeter")
            .unwrap();
        assert_eq!(greeter.doc.as_deref(), Some("Greets people."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(calls.contains(&("Greeter::greet", "format")), "{calls:?}");
        assert!(
            calls.contains(&("Greeter::format", "uppercased")),
            "{calls:?}"
        );

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(
            imports.contains(&("Foundation", "Foundation")),
            "{imports:?}"
        );
    }

    #[test]
    fn php_extraction() {
        let src = r#"<?php

namespace App {
    use App\Util\Formatter;
    require_once 'bootstrap.php';

    // Greets people.
    class Greeter
    {
        public function greet($name)
        {
            return $this->format($name);
        }

        private function format($name)
        {
            return Formatter::upper($name);
        }
    }

    interface Runnable
    {
        public function run();
    }
}
"#;
        let fx = extract(Lang::Php, src);
        let n = names(&fx);
        assert!(n.contains(&("App", "namespace")), "{n:?}");
        assert!(n.contains(&("App::Greeter", "class")), "{n:?}");
        assert!(n.contains(&("App::Greeter::greet", "method")), "{n:?}");
        assert!(n.contains(&("App::Greeter::format", "method")), "{n:?}");
        assert!(n.contains(&("App::Runnable", "interface")), "{n:?}");

        let greeter = fx
            .symbols
            .iter()
            .find(|s| s.qualified == "App::Greeter")
            .unwrap();
        assert_eq!(greeter.doc.as_deref(), Some("Greets people."));

        let calls: Vec<(&str, &str)> = fx
            .calls
            .iter()
            .map(|c| (c.caller.as_str(), c.callee.as_str()))
            .collect();
        assert!(
            calls.contains(&("App::Greeter::greet", "format")),
            "{calls:?}"
        );
        assert!(
            calls.contains(&("App::Greeter::format", "upper")),
            "{calls:?}"
        );

        let imports: Vec<(&str, &str)> = fx
            .imports
            .iter()
            .map(|i| (i.local.as_str(), i.source.as_str()))
            .collect();
        assert!(
            imports.contains(&("Formatter", "App\\Util\\Formatter")),
            "{imports:?}"
        );
        assert!(
            imports.contains(&("bootstrap", "bootstrap.php")),
            "{imports:?}"
        );
    }

    #[test]
    fn php_without_tag_extracts_nothing() {
        // Without a `<?php` tag, the whole file parses as plain HTML/text —
        // no symbols, calls, or imports.
        let fx = extract(Lang::Php, "class Foo { function bar() {} }");
        assert!(fx.symbols.is_empty(), "{:?}", fx.symbols);
        assert!(fx.calls.is_empty(), "{:?}", fx.calls);
        assert!(fx.imports.is_empty(), "{:?}", fx.imports);
    }
}
