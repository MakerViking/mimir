//! Tree-sitter language adapters and symbol/call/import extraction.
//!
//! Deliberately dependency-free from mimir-core: mimir-graph (which depends
//! on mimir-core to persist symbols/edges) and mimir-core (which depends on
//! this crate to chunk source for recall — see mimir_core::index::chunker)
//! both consume it, and mimir-core -> mimir-graph would be a cycle.

pub mod extract;
pub mod languages;
