//! Deterministic retrieval-eval harness.
//!
//! Not part of the product: a dev-only measurement backbone so later
//! ranking/index changes can prove wins and guard against regressions.
//!
//! Two modes, because determinism and semantic realism conflict:
//! - **hermetic** (this module; compiled whenever `cfg(test)` is on, so
//!   plain `cargo test` covers it): a small hand-authored corpus
//!   ([`fixtures`]) scored with deterministic synthetic vectors — no
//!   network, no model. It exercises RRF fusion and ranking-config changes
//!   deterministically; it is a regression guard, not a semantic-quality
//!   signal (see [`fixtures`] docs).
//! - **real-model** (`--features eval`, `#[ignore]`d): the identical corpus
//!   and questions, embedded with the actual default model. This is the
//!   honest number to quote for accuracy work. Skips (does not fail) when
//!   the model isn't downloaded locally.
//!
//! Reports come in two cuts: by [`Category`] (the four question types) and
//! by [`QuestionSet`] (core / tuning / holdout — see [`fixtures`] docs for
//! why the split exists).
//!
//! Run:
//! ```text
//! cargo test -p mimir-mem-core eval:: -- --ignored --nocapture
//! cargo test -p mimir-mem-core --features eval eval:: -- --ignored --nocapture
//! ```

pub mod fixtures;

use std::collections::HashSet;

use rusqlite::Connection;

use crate::db;
use crate::error::Result;
use crate::model::Scope;
use crate::search::{self, SearchQuery};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    CodeStructure,
    CodeBehavior,
    GotchaDecision,
    DriftPreventer,
    /// A curated memory competing against raw `Kind::CodeChunk` hits on the
    /// same topic — regression guard for `scoring.code_damp` (see
    /// `fixtures` docs, scenario family 5).
    CodeVsMemory,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::CodeStructure => "code-structure",
            Category::CodeBehavior => "code-behavior",
            Category::GotchaDecision => "gotcha-decision",
            Category::DriftPreventer => "drift-preventer",
            Category::CodeVsMemory => "code-vs-memory",
        }
    }
}

const ALL_CATEGORIES: [Category; 5] = [
    Category::CodeStructure,
    Category::CodeBehavior,
    Category::GotchaDecision,
    Category::DriftPreventer,
    Category::CodeVsMemory,
];

/// Which slice of the question set a [`fixtures::Question`] belongs to.
/// `Core` is the original WS0 regression set (always scored, not split).
/// `Tuning`/`Holdout` are the newer headroom scenarios: tune ranking config
/// against `Tuning`, confirm the change generalizes on `Holdout` before
/// trusting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionSet {
    Core,
    Tuning,
    Holdout,
}

impl QuestionSet {
    fn label(self) -> &'static str {
        match self {
            QuestionSet::Core => "core",
            QuestionSet::Tuning => "tuning",
            QuestionSet::Holdout => "holdout",
        }
    }
}

const ALL_SETS: [QuestionSet; 3] = [QuestionSet::Core, QuestionSet::Tuning, QuestionSet::Holdout];

/// precision@k / recall@k / reciprocal rank for one question at cutoff k.
#[derive(Debug, Clone, Copy, Default)]
struct Metrics {
    precision: f64,
    recall: f64,
    mrr: f64,
}

fn score(retrieved: &[i64], relevant: &HashSet<i64>, k: usize) -> Metrics {
    let top = &retrieved[..retrieved.len().min(k)];
    let hits = top.iter().filter(|id| relevant.contains(id)).count();
    Metrics {
        precision: hits as f64 / k as f64,
        recall: if relevant.is_empty() {
            0.0
        } else {
            hits as f64 / relevant.len() as f64
        },
        mrr: top
            .iter()
            .position(|id| relevant.contains(id))
            .map(|i| 1.0 / (i as f64 + 1.0))
            .unwrap_or(0.0),
    }
}

/// Averaged metrics for one category/set, or the "overall" row.
#[derive(Debug, Clone)]
pub struct CategoryReport {
    pub label: String,
    pub n: usize,
    pub precision: f64,
    pub recall: f64,
    pub mrr: f64,
}

fn mean_report(label: String, rows: &[Metrics]) -> CategoryReport {
    let n = rows.len();
    let mean = |f: fn(&Metrics) -> f64| rows.iter().map(f).sum::<f64>() / n as f64;
    CategoryReport {
        label,
        n,
        precision: mean(|m| m.precision),
        recall: mean(|m| m.recall),
        mrr: mean(|m| m.mrr),
    }
}

/// One row per distinct key present in `rows`, plus a trailing "overall"
/// row averaged across all of them.
fn aggregate_by<T: Copy + PartialEq>(
    k: usize,
    rows: &[(Category, QuestionSet, Metrics)],
    keys: &[T],
    key_of: impl Fn(Category, QuestionSet) -> T,
    label_of: impl Fn(T) -> String,
) -> Vec<CategoryReport> {
    let mut reports: Vec<CategoryReport> = keys
        .iter()
        .filter_map(|&key| {
            let sel: Vec<Metrics> = rows
                .iter()
                .filter(|(c, s, _)| key_of(*c, *s) == key)
                .map(|(_, _, m)| *m)
                .collect();
            (!sel.is_empty()).then(|| mean_report(label_of(key), &sel))
        })
        .collect();
    let all: Vec<Metrics> = rows.iter().map(|(_, _, m)| *m).collect();
    reports.push(mean_report(format!("overall (k={k})"), &all));
    reports
}

pub fn print_report(title: &str, reports: &[CategoryReport]) {
    println!("\n{title}");
    println!(
        "{:<20} {:>3}  {:>11} {:>8} {:>7}",
        "category", "n", "precision", "recall", "mrr"
    );
    for r in reports {
        println!(
            "{:<20} {:>3}  {:>11.3} {:>8.3} {:>7.3}",
            r.label, r.n, r.precision, r.recall, r.mrr
        );
    }
}

/// Both report cuts for one eval run.
pub struct EvalResult {
    pub by_category: Vec<CategoryReport>,
    pub by_set: Vec<CategoryReport>,
}

/// Shared scoring loop: runs every question through `search_hybrid`, given
/// a way to produce a query vector (synthetic in hermetic mode, real-model
/// embedding in real-model mode). Both modes were structurally identical
/// duplicated loops before the corpus grew for the headroom scenarios;
/// this is the one place that changed to support it.
fn evaluate_all(
    conn: &Connection,
    ids: &fixtures::FixtureIds,
    k: usize,
    model: &str,
    mut query_vec: impl FnMut(&fixtures::Question) -> Result<Vec<f32>>,
) -> Result<Vec<(Category, QuestionSet, Metrics)>> {
    // Score with the same weights a real user gets (config defaults), so a
    // ranking-config change (recency_alpha, type_prior_alpha, ...) actually
    // shows up here instead of the eval scoring silently diverging from it.
    let scoring = crate::config::ScoringConfig::default();
    let mut cache = None;
    let mut rows = Vec::new();
    for q in fixtures::questions() {
        let qvec = query_vec(q)?;
        let query = SearchQuery {
            text: q.query.to_string(),
            scope: Scope::All,
            kinds: vec![],
            since: None,
            limit: k,
            strength_alpha: scoring.strength_alpha,
            recency_alpha: scoring.recency_alpha,
            type_prior_alpha: scoring.type_prior_alpha,
            code_damp: scoring.code_damp,
            include_superseded: false,
        };
        let hits = search::search_hybrid(conn, &query, Some((model, &qvec)), &mut cache)?;
        let retrieved: Vec<i64> = hits.iter().map(|h| h.node.id).collect();
        let relevant = fixtures::relevant_ids(ids, q);
        rows.push((q.category, q.set, score(&retrieved, &relevant, k)));
    }
    Ok(rows)
}

fn build_result(k: usize, rows: Vec<(Category, QuestionSet, Metrics)>) -> EvalResult {
    let by_category = aggregate_by(k, &rows, &ALL_CATEGORIES, |c, _| c, |c| c.label().to_string());
    let by_set = aggregate_by(k, &rows, &ALL_SETS, |_, s| s, |s| s.label().to_string());
    EvalResult { by_category, by_set }
}

/// Hermetic baseline: synthetic vectors, no model, no network. Safe for
/// normal `cargo test`.
pub fn run_hermetic(k: usize) -> Result<EvalResult> {
    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    fixtures::seed_synthetic_vectors(&conn, &ids)?;

    let rows = evaluate_all(&conn, &ids, k, fixtures::SYNTHETIC_MODEL, |q| {
        Ok(fixtures::synthetic_query_vector(q.topic, q.query))
    })?;
    Ok(build_result(k, rows))
}

/// Real-model baseline: identical corpus, embedded with the configured
/// default model. `Ok(None)` (not an error) when the model isn't
/// downloaded locally — this mode never touches the network itself.
#[cfg(feature = "eval")]
pub fn run_real_model(k: usize) -> Result<Option<EvalResult>> {
    use rusqlite::params;

    use crate::config::{Config, Paths};
    use crate::embed::{self, Embedder};

    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    if !embed::model_ready(&paths, &config.embedding.model) {
        println!(
            "eval: model '{}' not downloaded (run `mimir init` first); skipping real-model eval",
            config.embedding.model
        );
        return Ok(None);
    }
    let mut embedder =
        Embedder::load(&paths, &config.embedding.model, &config.embedding.device, false)?;

    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    for (node_id, text) in fixtures::fixture_texts(&ids) {
        let vec = embedder.embed(vec![text])?.remove(0);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash) \
             VALUES (?1, ?2, ?3, ?4, x'00')",
            params![node_id, embedder.name, embedder.dim as i64, embed::to_blob(&vec)],
        )?;
    }

    let model_name = embedder.name.clone();
    let rows = evaluate_all(&conn, &ids, k, &model_name, |q| {
        Ok(embedder.embed(vec![q.query.to_string()])?.remove(0))
    })?;
    Ok(Some(build_result(k, rows)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fast and deterministic — the actual regression guard. Not a
    /// semantic-quality signal (see module docs): just proves the harness,
    /// fixtures, and RRF fusion wiring stay sane after a ranking change.
    ///
    /// This does NOT assert MRR/recall near 1.0 — several scenarios in
    /// `fixtures.rs` are deliberately hard so a future ranking change has
    /// headroom to prove a win. The floor here only catches "the harness
    /// itself is broken" (empty results, ids not resolving, etc.).
    #[test]
    fn hermetic_scores_are_well_formed() {
        let result = run_hermetic(5).unwrap();
        assert_eq!(result.by_category.len(), 6, "5 categories + overall");
        assert_eq!(result.by_set.len(), 4, "core/tuning/holdout + overall");
        for r in result.by_category.iter().chain(&result.by_set) {
            assert!(r.n > 0, "{} has no questions", r.label);
            assert!((0.0..=1.0).contains(&r.precision), "{}", r.label);
            assert!((0.0..=1.0).contains(&r.recall), "{}", r.label);
            assert!((0.0..=1.0).contains(&r.mrr), "{}", r.label);
        }
        let overall = result.by_category.last().unwrap();
        assert_eq!(overall.n, fixtures::questions().len());
        assert!(
            overall.mrr > 0.2,
            "hermetic MRR implausibly low ({}); harness or fixtures likely broken",
            overall.mrr
        );
    }

    #[test]
    #[ignore]
    fn eval_baseline_report() {
        let result = run_hermetic(5).unwrap();
        print_report("hermetic — by category (synthetic vectors, k=5)", &result.by_category);
        print_report("hermetic — by question set (k=5)", &result.by_set);
    }

    #[cfg(feature = "eval")]
    #[test]
    #[ignore]
    fn eval_baseline_report_real_model() {
        if let Some(result) = run_real_model(5).unwrap() {
            print_report("real model — by category (k=5)", &result.by_category);
            print_report("real model — by question set (k=5)", &result.by_set);
        }
    }

    /// Shared diagnostic loop backing the hermetic/real-model detail reports
    /// below: prints the ranked hit list per question, marking which hits
    /// are relevant, so a scenario that ceilings out (or should show
    /// headroom but doesn't) can be inspected directly. Not a regression
    /// check. Same config-defaults wiring as `evaluate_all`.
    fn print_question_detail(
        conn: &Connection,
        ids: &fixtures::FixtureIds,
        k: usize,
        model: &str,
        mut query_vec: impl FnMut(&fixtures::Question) -> Result<Vec<f32>>,
    ) -> Result<()> {
        let scoring = crate::config::ScoringConfig::default();
        let mut cache = None;
        for q in fixtures::questions() {
            let qvec = query_vec(q)?;
            let query = SearchQuery {
                text: q.query.to_string(),
                scope: Scope::All,
                kinds: vec![],
                since: None,
                limit: k,
                strength_alpha: scoring.strength_alpha,
                recency_alpha: scoring.recency_alpha,
                type_prior_alpha: scoring.type_prior_alpha,
                code_damp: scoring.code_damp,
                include_superseded: false,
            };
            let hits = search::search_hybrid(conn, &query, Some((model, &qvec)), &mut cache)?;
            let relevant = fixtures::relevant_ids(ids, q);
            println!("\n[{:?}/{:?}] {}", q.category, q.set, q.query);
            for (rank, h) in hits.iter().enumerate() {
                let mark = if relevant.contains(&h.node.id) { "*" } else { " " };
                println!(
                    "  {mark} #{rank} {} (score {:.4})",
                    fixtures::key_for_id(ids, h.node.id),
                    h.score
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn eval_question_detail_report() {
        let conn = db::open_in_memory().unwrap();
        let ids = fixtures::insert_fixture_nodes(&conn).unwrap();
        fixtures::seed_synthetic_vectors(&conn, &ids).unwrap();
        print_question_detail(&conn, &ids, 8, fixtures::SYNTHETIC_MODEL, |q| {
            Ok(fixtures::synthetic_query_vector(q.topic, q.query))
        })
        .unwrap();
    }

    /// Real-model counterpart to `eval_question_detail_report`: same
    /// diagnostic, actual embeddings. Skips (does not fail) when the model
    /// isn't downloaded locally, same as `eval_baseline_report_real_model`.
    #[cfg(feature = "eval")]
    #[test]
    #[ignore]
    fn eval_question_detail_report_real_model() {
        use crate::config::{Config, Paths};
        use crate::embed::{self, Embedder};

        let paths = Paths::resolve().unwrap();
        let config = Config::load(&paths.config_file).unwrap();
        if !embed::model_ready(&paths, &config.embedding.model) {
            println!(
                "eval: model '{}' not downloaded (run `mimir init` first); skipping",
                config.embedding.model
            );
            return;
        }
        let mut embedder =
            Embedder::load(&paths, &config.embedding.model, &config.embedding.device, false)
                .unwrap();

        let conn = db::open_in_memory().unwrap();
        let ids = fixtures::insert_fixture_nodes(&conn).unwrap();
        for (node_id, text) in fixtures::fixture_texts(&ids) {
            let vec = embedder.embed(vec![text]).unwrap().remove(0);
            conn.execute(
                "INSERT INTO embedding (node_id, model, dim, vec, content_hash) \
                 VALUES (?1, ?2, ?3, ?4, x'00')",
                rusqlite::params![
                    node_id,
                    embedder.name,
                    embedder.dim as i64,
                    embed::to_blob(&vec)
                ],
            )
            .unwrap();
        }
        let model_name = embedder.name.clone();
        print_question_detail(&conn, &ids, 8, &model_name, |q| {
            Ok(embedder.embed(vec![q.query.to_string()])?.remove(0))
        })
        .unwrap();
    }
}
