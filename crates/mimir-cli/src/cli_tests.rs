//! Parser-level tests: which flag combinations `mimir` accepts.
//!
//! A flag that is silently ignored is worse than one that is refused. The
//! sharpest case was `mimir link A B --dry-run`: `--dry-run` only applies to
//! `--scan`, so the command parsed, ignored the flag, and wrote the link the
//! user had asked to preview. These combinations are declared on the parser
//! instead of being checked (or not) by hand in each command.

use clap::Parser;

use crate::Cli;

fn parses(args: &[&str]) -> bool {
    Cli::try_parse_from(std::iter::once("mimir").chain(args.iter().copied())).is_ok()
}

#[test]
fn link_dry_run_only_exists_for_a_scan() {
    assert!(!parses(&["link", "m:AAAAAA", "m:BBBBBB", "--dry-run"]));
    assert!(!parses(&["link", "m:AAAAAA", "m:BBBBBB", "--all-projects"]));
    assert!(parses(&["link", "--scan", "--dry-run"]));
    assert!(parses(&["link", "--scan", "--all-projects", "--dry-run"]));
}

#[test]
fn link_scan_takes_no_endpoints_and_no_relation() {
    assert!(!parses(&["link", "--scan", "m:AAAAAA", "m:BBBBBB"]));
    assert!(!parses(&["link", "--scan", "--rel", "about"]));
    // The relation's default must not count as "given".
    assert!(parses(&["link", "--scan"]));
    assert!(parses(&["link", "m:AAAAAA", "m:BBBBBB"]));
    assert!(parses(&["link", "m:AAAAAA", "m:BBBBBB", "--rel", "about"]));
}

#[test]
fn a_review_decision_needs_the_finding_it_decides() {
    assert!(!parses(&["review", "--keep"]));
    assert!(!parses(&["review", "--dismiss", "--reason", "noise"]));
    assert!(!parses(&["review", "--reason", "noise"]));
    assert!(!parses(&["review", "3", "--keep", "--dismiss"]));
    assert!(parses(&["review"]));
    assert!(parses(&["review", "--limit", "5"]));
    assert!(parses(&["review", "3", "--keep", "--reason", "both hold"]));
}

#[test]
fn dismissing_a_miss_is_not_a_listing() {
    assert!(!parses(&["misses", "--dismiss", "12", "--all"]));
    assert!(!parses(&["misses", "--dismiss", "12", "--resolved"]));
    assert!(parses(&["misses", "--dismiss", "12"]));
    assert!(parses(&["misses", "--all", "--resolved"]));
}

/// `--all` used to win over `--global` without a word, so a caller who asked
/// for global-only results got every project's.
#[test]
fn a_read_has_one_scope() {
    assert!(!parses(&["recall", "query", "--all", "--global"]));
    assert!(!parses(&["list", "--all", "-g"]));
    assert!(parses(&["recall", "query", "--all"]));
    assert!(parses(&["recall", "query", "-g"]));
    assert!(parses(&["list", "--all"]));
    assert!(parses(&["list", "-g"]));
}
