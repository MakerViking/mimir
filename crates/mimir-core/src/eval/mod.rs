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

pub mod brief;
pub mod fixtures;

use std::collections::HashSet;

use rusqlite::Connection;

use crate::db;
use crate::error::Result;
use crate::inject;
use crate::model::{Kind, Scope};
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
    /// A query naming a literal identifier (`retry_with_backoff`, not "the
    /// retry waiting strategy") — regression guard for the identifier-exact
    /// RRF leg (see `fixtures` docs, scenario family 5b). Ground truth is
    /// the function body: an identifier-explicit mechanics question has a
    /// different human-correct answer than `CodeVsMemory`'s conceptual
    /// phrasing of the same topic, so the two guards must stay separate
    /// rather than sharing one fixture pair.
    IdentifierLookup,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::CodeStructure => "code-structure",
            Category::CodeBehavior => "code-behavior",
            Category::GotchaDecision => "gotcha-decision",
            Category::DriftPreventer => "drift-preventer",
            Category::CodeVsMemory => "code-vs-memory",
            Category::IdentifierLookup => "identifier-lookup",
        }
    }
}

const ALL_CATEGORIES: [Category; 6] = [
    Category::CodeStructure,
    Category::CodeBehavior,
    Category::GotchaDecision,
    Category::DriftPreventer,
    Category::CodeVsMemory,
    Category::IdentifierLookup,
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
    scoring: &crate::config::ScoringConfig,
    mut query_vec: impl FnMut(&fixtures::Question) -> Result<Vec<f32>>,
) -> Result<Vec<(Category, QuestionSet, Metrics)>> {
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
    let by_category = aggregate_by(
        k,
        &rows,
        &ALL_CATEGORIES,
        |c, _| c,
        |c| c.label().to_string(),
    );
    let by_set = aggregate_by(k, &rows, &ALL_SETS, |_, s| s, |s| s.label().to_string());
    EvalResult {
        by_category,
        by_set,
    }
}

/// Confusion counts for the **preventer end-to-end eval**: unlike
/// [`EvalResult`] above (which scores a raw ranked list against a relevant
/// set), this scores the actual product decision —
/// `inject::select_injection`'s call to inject or stay silent — against
/// [`fixtures::InjectQuestion`]'s labels. `injected_wrong` is the cell that
/// matters most: an injected memory actively misleads the agent, which is
/// worse than staying silent (see `inject.rs` module docs), so a floor
/// change that raises `injected_wrong` at all must be backed out regardless
/// of what it does to the other three cells.
#[derive(Debug, Clone, Copy, Default)]
pub struct InjectConfusion {
    /// A labeled preventer existed and the right one was injected.
    pub injected_correct: usize,
    /// No relevant preventer existed (or the only relevant hit was a
    /// non-preventer kind) and the floor correctly stayed silent.
    pub silent_correct: usize,
    /// Something was injected that wasn't the labeled preventer, or nothing
    /// was labeled relevant at all — the worst cell.
    pub injected_wrong: usize,
    /// A labeled preventer existed but the floor stayed silent.
    pub silent_wrong: usize,
}

impl InjectConfusion {
    pub fn print(&self, title: &str) {
        println!("\n{title}");
        println!(
            "  injected-correct: {:>3}    silent-correct: {:>3}",
            self.injected_correct, self.silent_correct
        );
        println!(
            "  injected-wrong:   {:>3}    silent-wrong:   {:>3}   (injected-wrong must never rise)",
            self.injected_wrong, self.silent_wrong
        );
    }
}

/// Runs every [`fixtures::InjectQuestion`] through the same search shape
/// `inject::compute` uses (`kinds: [Memory]`, `limit: 5`, config-default
/// scoring weights), then hands the raw hits/legs to
/// `inject::select_injection` — the exact pure decision core the hook uses
/// in production — and classifies the outcome against the question's
/// label. Shared by hermetic and real-model modes, same split as
/// `evaluate_all`.
fn evaluate_inject_all(
    conn: &Connection,
    ids: &fixtures::FixtureIds,
    model: &str,
    mut query_vec: impl FnMut(&str) -> Result<Vec<f32>>,
) -> Result<InjectConfusion> {
    let scoring = crate::config::ScoringConfig::default();
    let mut cache = None;
    let mut confusion = InjectConfusion::default();
    for q in fixtures::inject_questions() {
        let qvec = query_vec(q.query)?;
        let enrich = q.enrich.join(" ");
        let search_text = if enrich.is_empty() {
            q.query.to_string()
        } else {
            format!("{} {enrich}", q.query)
        };
        let query = SearchQuery {
            text: search_text,
            scope: Scope::All,
            kinds: vec![Kind::Memory],
            since: None,
            limit: 5,
            strength_alpha: scoring.strength_alpha,
            recency_alpha: scoring.recency_alpha,
            type_prior_alpha: scoring.type_prior_alpha,
            code_damp: scoring.code_damp,
            include_superseded: false,
        };
        let (hits, legs) =
            search::search_hybrid_with_legs(conn, &query, Some((model, &qvec)), &mut cache)?;
        match inject::select_injection(&hits, q.query, &enrich, &legs) {
            Some(hit) => {
                let key = fixtures::key_for_id(ids, hit.node.id);
                if q.expects_injection && q.acceptable.contains(&key) {
                    confusion.injected_correct += 1;
                } else {
                    confusion.injected_wrong += 1;
                }
            }
            None => {
                if q.expects_injection {
                    confusion.silent_wrong += 1;
                } else {
                    confusion.silent_correct += 1;
                }
            }
        }
    }
    Ok(confusion)
}

/// Hermetic preventer eval: synthetic vectors, keyed off `InjectQuestion::topic`
/// same as `run_hermetic`. Safe for normal `cargo test`.
pub fn run_inject_eval_hermetic() -> Result<InjectConfusion> {
    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    fixtures::seed_synthetic_vectors(&conn, &ids)?;
    let topic_by_query: std::collections::HashMap<&str, &str> = fixtures::inject_questions()
        .iter()
        .map(|q| (q.query, q.topic))
        .collect();
    evaluate_inject_all(&conn, &ids, fixtures::SYNTHETIC_MODEL, |query| {
        Ok(fixtures::synthetic_query_vector(
            topic_by_query[query],
            query,
        ))
    })
}

/// Model name for real-model eval runs: `MIMIR_EVAL_MODEL` overrides the
/// configured default, so an A/B (e.g. granite vs bge-small, wave-2 item #8)
/// can run against a second model without touching `mimir.toml`. Unset →
/// identical behavior to before this existed (the configured default).
#[cfg(feature = "eval")]
fn eval_model_name(config: &crate::config::Config) -> String {
    std::env::var("MIMIR_EVAL_MODEL").unwrap_or_else(|_| config.embedding.model.clone())
}

/// Real-model counterpart to [`run_inject_eval_hermetic`]: identical
/// corpus/questions, embedded with the configured default model. `None`
/// (not an error) when the model isn't downloaded locally.
#[cfg(feature = "eval")]
pub fn run_inject_eval_real_model() -> Result<Option<InjectConfusion>> {
    use rusqlite::params;

    use crate::config::{Config, Paths};
    use crate::embed::{self, Embedder};

    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    let model_name = eval_model_name(&config);
    if !embed::model_ready(&paths, &model_name) {
        println!(
            "eval: model '{model_name}' not downloaded; set `embedding.model = \"{model_name}\"` \
             in mimir.toml then run `mimir init` (or `mimir embed --fetch`); skipping \
             real-model preventer eval"
        );
        return Ok(None);
    }
    let mut embedder = Embedder::load(&paths, &model_name, &config.embedding.device, false)?;

    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    for (node_id, text) in fixtures::fixture_texts(&ids) {
        let vec = embedder.embed(vec![text])?.remove(0);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash) \
             VALUES (?1, ?2, ?3, ?4, x'00')",
            params![
                node_id,
                embedder.name,
                embedder.dim as i64,
                embed::to_blob(&vec)
            ],
        )?;
    }
    let model_name = embedder.name.clone();
    let confusion = evaluate_inject_all(&conn, &ids, &model_name, |query| {
        Ok(embedder.embed(vec![query.to_string()])?.remove(0))
    })?;
    Ok(Some(confusion))
}

/// Hermetic baseline: synthetic vectors, no model, no network. Safe for
/// normal `cargo test`.
pub fn run_hermetic(k: usize) -> Result<EvalResult> {
    // The same weights a real user gets, so a ranking-config change
    // (recency_alpha, type_prior_alpha, ...) actually shows up here instead
    // of the eval scoring silently diverging from the product's.
    run_hermetic_with(k, &crate::config::ScoringConfig::default())
}

/// [`run_hermetic`] with the scoring weights supplied, so an ablation can
/// ask what a knob is actually buying. Nothing in the product calls this
/// with a non-default config — if it did, the defaults would no longer be
/// what the baseline measures.
pub fn run_hermetic_with(k: usize, scoring: &crate::config::ScoringConfig) -> Result<EvalResult> {
    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    fixtures::seed_synthetic_vectors(&conn, &ids)?;

    let rows = evaluate_all(&conn, &ids, k, fixtures::SYNTHETIC_MODEL, scoring, |q| {
        Ok(fixtures::synthetic_query_vector(q.topic, q.query))
    })?;
    Ok(build_result(k, rows))
}

/// Real-model baseline: identical corpus, embedded with the configured
/// default model (or `MIMIR_EVAL_MODEL`, see [`eval_model_name`]). `Ok(None)`
/// (not an error) when the model isn't downloaded locally — this mode never
/// touches the network itself.
#[cfg(feature = "eval")]
pub fn run_real_model(k: usize) -> Result<Option<EvalResult>> {
    use rusqlite::params;

    use crate::config::{Config, Paths};
    use crate::embed::{self, Embedder};

    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    let model_name = eval_model_name(&config);
    if !embed::model_ready(&paths, &model_name) {
        println!(
            "eval: model '{model_name}' not downloaded; set `embedding.model = \"{model_name}\"` \
             in mimir.toml then run `mimir init` (or `mimir embed --fetch`); skipping \
             real-model eval"
        );
        return Ok(None);
    }
    let mut embedder = Embedder::load(&paths, &model_name, &config.embedding.device, false)?;

    let conn = db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    for (node_id, text) in fixtures::fixture_texts(&ids) {
        let vec = embedder.embed(vec![text])?.remove(0);
        conn.execute(
            "INSERT INTO embedding (node_id, model, dim, vec, content_hash) \
             VALUES (?1, ?2, ?3, ?4, x'00')",
            params![
                node_id,
                embedder.name,
                embedder.dim as i64,
                embed::to_blob(&vec)
            ],
        )?;
    }

    let model_name = embedder.name.clone();
    let scoring = crate::config::ScoringConfig::default();
    let rows = evaluate_all(&conn, &ids, k, &model_name, &scoring, |q| {
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
    /// Landed hermetic numbers at k=5, by question set. Committed so a
    /// ranking change has to show up as a deliberate edit to this table in
    /// the diff, rather than as a number nobody reads drifting quietly.
    ///
    /// The hermetic mode is fully deterministic (synthetic vectors, fixed
    /// corpus), so these reproduce exactly — which is why the gate below
    /// compares in *both* directions. An improvement failing the build is
    /// the point: it means someone made ranking better and should say so
    /// here, where the next person can see what the number used to be.
    const HERMETIC_BASELINE: &[(&str, f64, f64)] = &[
        // (set, mrr, recall)
        ("core", 1.0000, 1.0000),
        ("tuning", 0.9000, 0.9000),
        ("holdout", 1.0000, 1.0000),
        ("overall (k=5)", 0.9565, 0.9565),
    ];

    const BASELINE_TOL: f64 = 5e-4;

    #[test]
    fn hermetic_baseline_reproduces_exactly() {
        let result = run_hermetic(5).unwrap();
        for (label, want_mrr, want_recall) in HERMETIC_BASELINE {
            let got = result
                .by_set
                .iter()
                .find(|r| &r.label == label)
                .unwrap_or_else(|| panic!("no '{label}' row in by_set"));
            for (metric, want, have) in [
                ("mrr", *want_mrr, got.mrr),
                ("recall", *want_recall, got.recall),
            ] {
                assert!(
                    (have - want).abs() <= BASELINE_TOL,
                    "{label} {metric}: baseline {want:.4}, now {have:.4}.\n\
                     If this change was intended, update HERMETIC_BASELINE in this file \
                     and say why in the commit — a ranking change that edits no baseline \
                     is a ranking change nobody reviewed."
                );
            }
        }
    }

    /// The ablation memsem-style: every scoring knob must pay for itself on
    /// the corpus, and the stack as a whole must beat scoring with all of
    /// them off. Without this, a knob can rot into a no-op — or worse, into
    /// a net negative — while its config field and docs still claim a job.
    #[test]
    fn every_ranking_knob_earns_its_place() {
        use crate::config::ScoringConfig;
        let d = ScoringConfig::default();
        let mrr = |cfg: &ScoringConfig| {
            run_hermetic_with(5, cfg)
                .unwrap()
                .by_category
                .last()
                .unwrap()
                .mrr
        };
        let base = mrr(&d);

        for (knob, cfg) in [
            (
                "strength_alpha",
                ScoringConfig {
                    strength_alpha: 0.0,
                    ..d.clone()
                },
            ),
            (
                "recency_alpha",
                ScoringConfig {
                    recency_alpha: 0.0,
                    ..d.clone()
                },
            ),
            (
                "type_prior_alpha",
                ScoringConfig {
                    type_prior_alpha: 0.0,
                    ..d.clone()
                },
            ),
        ] {
            let without = mrr(&cfg);
            assert!(
                base > without,
                "{knob} buys nothing: MRR {base:.4} with it, {without:.4} without. \
                 Either it stopped working or the corpus stopped testing it — \
                 both need a human, neither should pass silently."
            );
        }

        let all_off = mrr(&ScoringConfig {
            strength_alpha: 0.0,
            recency_alpha: 0.0,
            type_prior_alpha: 0.0,
            code_damp: 1.0,
            ..d.clone()
        });
        assert!(
            base > all_off,
            "the whole scoring stack buys nothing: {base:.4} vs {all_off:.4} unscored"
        );
    }

    /// `code_damp` is deliberately absent from the ablation above, because
    /// it currently changes *nothing* on this corpus — including the
    /// `code-vs-memory` category that exists to guard it. That is a gap in
    /// the fixtures, not proof the knob is useless: the one scenario there
    /// ranks the memory first with or without the damp, so it never
    /// exercises the tie the damp is meant to break.
    ///
    /// Pinned rather than ignored, so the day someone authors a fixture
    /// that does exercise it, this test fails and forces the knob into the
    /// ablation above where it belongs.
    #[test]
    fn code_damp_is_still_inert_on_this_corpus() {
        use crate::config::ScoringConfig;
        let d = ScoringConfig::default();
        let with = run_hermetic_with(5, &d).unwrap();
        let without = run_hermetic_with(
            5,
            &ScoringConfig {
                code_damp: 1.0,
                ..d.clone()
            },
        )
        .unwrap();
        let overall = |r: &EvalResult| r.by_category.last().unwrap().mrr;
        assert!(
            (overall(&with) - overall(&without)).abs() <= BASELINE_TOL,
            "code_damp now moves the corpus ({:.4} vs {:.4}) — good. Move it into \
             every_ranking_knob_earns_its_place and delete this test.",
            overall(&with),
            overall(&without)
        );
    }

    #[test]
    fn hermetic_scores_are_well_formed() {
        let result = run_hermetic(5).unwrap();
        assert_eq!(result.by_category.len(), 7, "6 categories + overall");
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

    /// Regression guard for the preventer end-to-end eval itself: every
    /// question must be accounted for in exactly one confusion cell, and
    /// `injected_wrong` must be zero on the current (already-landed) floor.
    /// This is the harness sanity check; `eval_inject_baseline_report`
    /// below is the number to actually read/compare across floor changes.
    #[test]
    fn inject_eval_confusion_is_well_formed() {
        let c = run_inject_eval_hermetic().unwrap();
        let total = c.injected_correct + c.silent_correct + c.injected_wrong + c.silent_wrong;
        assert_eq!(total, fixtures::inject_questions().len());
        // Known baseline gap, not a floor bug: `mem_decision_pool_old` vs.
        // `mem_decision_pool_new` (scenario family 1) share type Decision,
        // topic, and overlap — the floor has no recency logic of its own
        // (that's `scoring.recency_alpha`, owned by search/mod.rs, not
        // this module), so an un-superseded stale decision ranked fused
        // #1 clears the floor just as legitimately as the current one
        // would. This is the one pre-existing `injected_wrong`; the bound
        // is a regression guard against a *second* one appearing.
        assert!(
            c.injected_wrong <= 1,
            "landed floor injected {} wrong preventer(s), expected at most the known \
             recency-ranking gap (1)",
            c.injected_wrong
        );
    }

    #[test]
    #[ignore]
    fn eval_baseline_report() {
        let result = run_hermetic(5).unwrap();
        print_report(
            "hermetic — by category (synthetic vectors, k=5)",
            &result.by_category,
        );
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

    /// The number to actually read/compare across floor changes — run with
    /// `cargo test -p mimir-mem-core eval:: -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn eval_inject_baseline_report() {
        run_inject_eval_hermetic()
            .unwrap()
            .print("hermetic preventer end-to-end (synthetic vectors)");
    }

    #[cfg(feature = "eval")]
    #[test]
    #[ignore]
    fn eval_inject_baseline_report_real_model() {
        if let Some(c) = run_inject_eval_real_model().unwrap() {
            c.print("real model preventer end-to-end");
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
                let mark = if relevant.contains(&h.node.id) {
                    "*"
                } else {
                    " "
                };
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
        let model_name = eval_model_name(&config);
        if !embed::model_ready(&paths, &model_name) {
            println!(
                "eval: model '{model_name}' not downloaded; set `embedding.model = \
                 \"{model_name}\"` in mimir.toml then run `mimir init`; skipping"
            );
            return;
        }
        let mut embedder =
            Embedder::load(&paths, &model_name, &config.embedding.device, false).unwrap();

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
