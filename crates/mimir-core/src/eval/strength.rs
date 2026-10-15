//! Calibration of the recall match-strength label (`strength::classify`)
//! against the labelled retrieval corpus.
//!
//! A label is only worth its token if it means something, so this asks the
//! two questions an agent relies on it for:
//!
//! 1. **Does `sure` mean relevant more often than `weak` does?** Every top-k
//!    hit of every [`fixtures::Question`] is labelled and checked against
//!    the question's relevant set, giving P(relevant | label).
//! 2. **Does a miss mean the store had nothing?** The answerable questions
//!    should almost never come back as a miss, and the [`UNANSWERABLE`]
//!    probes — questions the corpus says nothing about — always should.
//!
//! Hermetic mode runs in plain `cargo test` and gates the ordering; the
//! real-model numbers are the honest ones and are printed by the ignored
//! test under `--features eval`.

use std::collections::HashSet;

use rusqlite::Connection;

use super::fixtures;
use crate::error::Result;
use crate::model::Scope;
use crate::search::{self, SearchQuery};
use crate::strength::{self, MatchStrength};

/// Questions nothing in the corpus answers. Each has its own topic anchor so
/// the hermetic query vector lands near no fixture's cluster — the semantic
/// leg still returns a top-k (it always does), which is exactly the case the
/// label must not mistake for evidence.
pub const UNANSWERABLE: &[(&str, &str)] = &[
    (
        "how do I configure TLS certificates for a kubernetes ingress",
        "probe_k8s",
    ),
    (
        "what hydration ratio works best for sourdough bread",
        "probe_bread",
    ),
    (
        "which terraform provider version pins the aws region",
        "probe_tf",
    ),
    (
        "how should I tune the garbage collector in the JVM",
        "probe_jvm",
    ),
    (
        "what is the airspeed velocity of an unladen swallow",
        "probe_swallow",
    ),
    ("how do I center a div with flexbox in css", "probe_css"),
];

/// `(relevant, total)` per label, indexed by `MatchStrength as usize`.
#[derive(Debug, Default, Clone)]
pub struct Calibration {
    pub by_label: [(usize, usize); 2],
    /// Answerable questions whose page came back as a miss.
    pub false_misses: Vec<&'static str>,
    pub answerable: usize,
    /// Unanswerable probes whose page did NOT come back as a miss.
    pub false_answers: Vec<&'static str>,
    pub unanswerable: usize,
}

impl Calibration {
    pub fn precision(&self, label: MatchStrength) -> Option<f64> {
        let (relevant, total) = self.by_label[label as usize];
        (total > 0).then(|| relevant as f64 / total as f64)
    }

    pub fn print(&self, title: &str) {
        println!("\n=== {title} ===");
        for label in [MatchStrength::Strong, MatchStrength::Weak] {
            let (relevant, total) = self.by_label[label as usize];
            let p = self
                .precision(label)
                .map_or("   -".to_string(), |p| format!("{p:.3}"));
            println!(
                "{:<7} P(relevant)={p}  ({relevant}/{total})",
                label.as_str()
            );
        }
        println!(
            "answerable questions reported as a miss: {}/{} {:?}",
            self.false_misses.len(),
            self.answerable,
            self.false_misses
        );
        println!(
            "unanswerable probes NOT reported as a miss: {}/{} {:?}",
            self.false_answers.len(),
            self.unanswerable,
            self.false_answers
        );
    }
}

fn query(text: &str, k: usize) -> SearchQuery {
    let scoring = crate::config::ScoringConfig::default();
    SearchQuery {
        text: text.to_string(),
        scope: Scope::All,
        kinds: vec![],
        since: None,
        limit: k,
        strength_alpha: scoring.strength_alpha,
        recency_alpha: scoring.recency_alpha,
        type_prior_alpha: scoring.type_prior_alpha,
        code_damp: scoring.code_damp,
        include_superseded: false,
        as_of: None,
    }
}

/// Label every top-`k` hit of every question (and probe) and tally.
/// `query_vec(text, topic)` produces the query embedding — synthetic in
/// hermetic mode, the real model otherwise.
pub fn calibrate(
    conn: &Connection,
    ids: &fixtures::FixtureIds,
    k: usize,
    model: &str,
    mut query_vec: impl FnMut(&str, &str) -> Result<Vec<f32>>,
) -> Result<Calibration> {
    let mut cache = None;
    let mut cal = Calibration::default();
    for q in fixtures::questions() {
        let qvec = query_vec(q.query, q.topic)?;
        let (hits, legs) = search::search_hybrid_with_legs(
            conn,
            &query(q.query, k),
            Some((model, &qvec)),
            &mut cache,
        )?;
        let relevant: HashSet<i64> = fixtures::relevant_ids(ids, q);
        let labels = strength::classify_page(&hits, q.query, &legs);
        for (hit, label) in hits.iter().zip(&labels) {
            let slot = &mut cal.by_label[*label as usize];
            slot.1 += 1;
            if relevant.contains(&hit.node.id) {
                slot.0 += 1;
            }
        }
        cal.answerable += 1;
        if strength::is_miss(&labels) {
            cal.false_misses.push(q.query);
        }
    }
    for (text, topic) in UNANSWERABLE {
        let qvec = query_vec(text, topic)?;
        let (hits, legs) = search::search_hybrid_with_legs(
            conn,
            &query(text, k),
            Some((model, &qvec)),
            &mut cache,
        )?;
        cal.unanswerable += 1;
        if !strength::is_miss(&strength::classify_page(&hits, text, &legs)) {
            cal.false_answers.push(text);
        }
    }
    Ok(cal)
}

pub fn run_hermetic(k: usize) -> Result<Calibration> {
    let conn = crate::db::open_in_memory()?;
    let ids = fixtures::insert_fixture_nodes(&conn)?;
    fixtures::seed_synthetic_vectors(&conn, &ids)?;
    calibrate(&conn, &ids, k, fixtures::SYNTHETIC_MODEL, |text, topic| {
        Ok(fixtures::synthetic_query_vector(topic, text))
    })
}

/// Real-model counterpart: identical corpus, embedded with the configured
/// default model. `None` when the model isn't downloaded locally.
#[cfg(feature = "eval")]
pub fn run_real_model(k: usize) -> Result<Option<Calibration>> {
    use rusqlite::params;

    use crate::config::{Config, Paths};
    use crate::embed::{self, Embedder};

    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    let model_name = config.embedding.model.clone();
    if !embed::model_ready(&paths, &model_name) {
        println!("eval: model '{model_name}' not downloaded; skipping strength calibration");
        return Ok(None);
    }
    let mut embedder = Embedder::load(&paths, &model_name, &config.embedding.device, false)?;
    let conn = crate::db::open_in_memory()?;
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
    let cal = calibrate(&conn, &ids, k, &model_name, |text, _| {
        Ok(embedder.embed(vec![text.to_string()])?.remove(0))
    })?;
    Ok(Some(cal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermetic_labels_are_ordered_and_misses_mean_nothing_matched() {
        let cal = run_hermetic(8).unwrap();
        cal.print("strength calibration (hermetic, k=8)");
        let strong = cal
            .precision(MatchStrength::Strong)
            .expect("some hit is strong");
        let weak = cal
            .precision(MatchStrength::Weak)
            .expect("some hit is weak");
        // A label earns its word on the line only if it separates: the
        // real-model run has strong at 25x the weak rate; 10x is the floor
        // this hermetic gate holds the rule to.
        assert!(
            strong >= 10.0 * weak,
            "strong must mean relevant far more often than weak: \
             strong {strong:.3}, weak {weak:.3}"
        );
        // Three vague drift prompts ("how do we cut a release", "anything I
        // should watch out for ...", "what should I know about past mistakes
        // ...") share too few words with their answers to be anything but
        // weak; the same three miss with the real model. More is the rule
        // getting worse.
        assert!(
            cal.false_misses.len() <= 3,
            "answerable questions reported as misses: {:?}",
            cal.false_misses
        );
        assert!(
            cal.false_answers.is_empty(),
            "a question nothing answers must be a miss: {:?}",
            cal.false_answers
        );
    }

    #[cfg(feature = "eval")]
    #[test]
    #[ignore]
    fn real_model_strength_calibration() {
        if let Some(cal) = run_real_model(8).unwrap() {
            cal.print("strength calibration (real model, k=8)");
        }
    }
}
