//! How much evidence stands behind one recall hit: `strong` or `weak`.
//!
//! The fused score cannot answer that. RRF ranks by *position* in each leg,
//! so the best hit on a query the store knows nothing about scores exactly
//! what the best hit on a perfect match scores — every ranked list has a #1.
//! An agent reading the list cannot tell "this is the answer" from "this is
//! the least-bad thing in the store", and the second case is the one where
//! acting on the hit is a mistake.
//!
//! So the label is built from evidence that does not depend on rank alone,
//! the same two signals `inject::clears_floor` already trusts before it puts
//! a memory in front of an agent unasked:
//!
//! - **term overlap** — how many of the query's content words the hit's
//!   title, body and tags actually contain;
//! - **leg agreement** — whether the hit sits in the lexical leg's own
//!   top [`LEG_TOP_K`] *and* the semantic leg's, i.e. two independent
//!   retrievers both put it near the top.
//!
//! What is deliberately NOT here: inject's rule 1 (only gotchas and
//! decisions). That rule is about whether a memory is worth injecting
//! unprompted, not about whether it matches. An explicit recall for a note
//! deserves an honest label for the note.
//!
//! **Two levels, not three, and that is measured.** A middle `likely` tier
//! (words present, legs disagreeing) was built first and checked against the
//! labelled corpus with the real embedding model (`eval::strength`, k=8):
//! `sure` 23/42 relevant, `likely` 0/8, `weak` 3/134. A tier that is right
//! less often than the one below it is noise with a reassuring name, so it
//! folded into `weak`. The same run is why the upper level says `strong`
//! rather than `sure`: 55% relevant is strong evidence — 25× the weak rate —
//! and not certainty.

use std::collections::HashSet;

use crate::inject::{LEG_TOP_K, MIN_OVERLAP};
use crate::memory;
use crate::model::Kind;
use crate::search::{fts, Hit};

/// Evidence behind one hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchStrength {
    Weak,
    Strong,
}

impl MatchStrength {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchStrength::Weak => "weak",
            MatchStrength::Strong => "strong",
        }
    }
}

/// The query's content words: tokenized the way the inject floor tokenizes,
/// stopwords dropped.
///
/// Exact tokens, deliberately unstemmed. The snippet window's 5-char prefix
/// stem (`format::STEM_PREFIX`) was tried here so "password" would meet
/// "passwords", and measured with the real model it found no relevant hit
/// the exact match missed (23 strong-and-relevant either way) while adding
/// 11 irrelevant strong ones — prefix collisions, precision 55% → 43%.
pub fn query_terms(query: &str) -> HashSet<String> {
    memory::tokens(query)
        .into_iter()
        .filter(|t| !fts::is_stopword(t))
        .collect()
}

/// Label one hit. `legs` is what [`crate::Mimir::search_with_evidence`]
/// returns: `legs[0]` the lexical leg, `legs[1]` the semantic leg iff an
/// embedding model ran for this query.
///
/// `strong` when the query names this symbol exactly, or when the hit
/// contains the query's words AND both legs rank it in their own top-K.
/// Without a semantic leg there is nothing to agree with, so covering
/// *every* content word (at least two) stands in for the agreement.
/// Everything else is `weak`: in the list because something had to be.
///
/// Semantic rank alone is deliberately NOT evidence. The vector leg returns
/// a top-k for every query, nonsense included, so "in the semantic top 10"
/// is rank again, not relevance: counted on its own, it let all six
/// unanswerable calibration probes (`eval::strength::UNANSWERABLE`) through.
/// It still counts as agreement alongside matching words, where it confirms
/// rather than stands in for a match. The cost is that a pure paraphrase —
/// right answer, no shared words — reads `weak`; that is the honest reading
/// of the evidence this has, and why the label says "weak", not "wrong".
pub fn classify(hit: &Hit, terms: &HashSet<String>, legs: &[Vec<i64>]) -> MatchStrength {
    if terms.is_empty() {
        return MatchStrength::Weak;
    }
    if names_symbol(hit, terms) {
        return MatchStrength::Strong;
    }
    let id = hit.node.id;
    let in_top = |leg: Option<&Vec<i64>>| {
        leg.is_some_and(|leg| leg.iter().take(LEG_TOP_K).any(|other| *other == id))
    };
    let lexical = in_top(legs.first());
    let semantic = in_top(legs.get(1));

    let hay = format!(
        "{} {} {}",
        hit.node.title.as_deref().unwrap_or(""),
        hit.node.body.as_deref().unwrap_or(""),
        hit.node.tags_text,
    );
    let hay: HashSet<String> = memory::tokens(&hay).into_iter().collect();
    let overlap = terms.iter().filter(|t| hay.contains(*t)).count();
    // A one-word query cannot reach MIN_OVERLAP; matching its one word is
    // all the overlap it can show.
    //
    // A flat count rather than a share of the query, and that is measured
    // too. "Half the query's words" was tried after two real-store hits came
    // back `strong` on 2 of 5 shared words (`watchdog.sh` for a TLS
    // certificate question, a UI row helper for a centring question — both
    // sharing the query's two meaningful words, so partial topical matches
    // rather than junk). With the real model it lifted strong's precision
    // 55% → 73% but tripled what matters more, the relevant hits labelled
    // weak (3 → 10), and doubled answerable questions reported as misses
    // (3 → 6 of 23): natural questions carry words their answers never use.
    // `weak` is the label that promises "safe to discount", so it is the one
    // kept honest.
    let covers = overlap >= MIN_OVERLAP.min(terms.len());

    let agrees = if legs.len() > 1 {
        lexical && semantic
    } else {
        lexical && overlap == terms.len() && overlap >= MIN_OVERLAP
    };
    tracing::debug!(
        node = id,
        title = hit.node.title.as_deref().unwrap_or(""),
        overlap,
        terms = terms.len(),
        lexical,
        semantic,
        "match evidence"
    );
    if covers && agrees {
        MatchStrength::Strong
    } else {
        MatchStrength::Weak
    }
}

/// A query that names a code symbol exactly (`retry_with_backoff`,
/// `Mimir::search_with`) and got that symbol back. The identifier leg exists
/// for this case and is not returned in `legs`, so it is recognised here
/// from the hit itself: the title's last path segment must be one of the
/// query's tokens, and must look like an identifier rather than a word —
/// otherwise a symbol called `open` would be `strong` for any query saying
/// "open".
fn names_symbol(hit: &Hit, terms: &HashSet<String>) -> bool {
    if hit.node.kind != Kind::Symbol {
        return false;
    }
    let Some(title) = hit.node.title.as_deref() else {
        return false;
    };
    let name = title
        .rsplit([':', '.'])
        .next()
        .unwrap_or(title)
        .to_lowercase();
    let identifier_shaped = name.contains('_') || title.chars().any(|c| c.is_uppercase());
    identifier_shaped && terms.contains(&name)
}

/// Label a whole page, in order.
pub fn classify_page(hits: &[Hit], query: &str, legs: &[Vec<i64>]) -> Vec<MatchStrength> {
    let terms = query_terms(query);
    hits.iter().map(|hit| classify(hit, &terms, legs)).collect()
}

/// Did this page answer the query at all? An empty page, or one with no
/// `strong` hit, is a miss: the store had nothing that matched, it only had
/// something to rank.
pub fn is_miss(strengths: &[MatchStrength]) -> bool {
    !strengths.contains(&MatchStrength::Strong)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NewNode;
    use crate::{db, store};

    /// Hits keep their real rowids so the `legs` below can name them: a
    /// fresh store numbers from 1, so the `id`-th insert has rowid `id`.
    fn hit(id: i64, kind: Kind, title: &str, body: &str) -> Hit {
        let conn = db::open_in_memory().unwrap();
        let mut node = None;
        for _ in 0..id {
            let mut new = NewNode::new(kind);
            new.title = Some(title.into());
            new.body = Some(body.into());
            node = Some(store::insert_node(&conn, new).unwrap());
        }
        let node = node.expect("id >= 1");
        assert_eq!(node.id, id);
        Hit { node, score: 1.0 }
    }

    fn terms(q: &str) -> HashSet<String> {
        query_terms(q)
    }

    #[test]
    fn both_legs_and_coverage_is_strong() {
        let h = hit(
            1,
            Kind::Memory,
            "WAL file growth",
            "checkpoint starvation keeps the wal large",
        );
        let legs = vec![vec![1, 2], vec![3, 1]];
        assert_eq!(
            classify(&h, &terms("why does the wal file grow"), &legs),
            MatchStrength::Strong
        );
    }

    #[test]
    fn coverage_without_semantic_agreement_is_weak() {
        let h = hit(1, Kind::Memory, "WAL file growth", "checkpoint starvation");
        let legs = vec![vec![1], vec![9, 8, 7]];
        assert_eq!(classify(&h, &terms("wal file"), &legs), MatchStrength::Weak);
    }

    #[test]
    fn semantic_rank_alone_is_not_evidence() {
        // The vector leg has a top-k for every query; ranking first there
        // with none of the query's words is what a nonsense query looks like.
        let h = hit(1, Kind::Memory, "Connection pooling", "one writer");
        let semantic_only = vec![vec![], vec![1]];
        assert_eq!(
            classify(&h, &terms("sqlite concurrency limits"), &semantic_only),
            MatchStrength::Weak
        );
    }

    #[test]
    fn without_a_semantic_leg_only_full_coverage_is_strong() {
        let h = hit(1, Kind::Memory, "WAL file growth", "checkpoint starvation");
        let lexical_only = vec![vec![1]];
        assert_eq!(
            classify(&h, &terms("wal file growth"), &lexical_only),
            MatchStrength::Strong
        );
        assert_eq!(
            classify(&h, &terms("wal file growth policy"), &lexical_only),
            MatchStrength::Weak
        );
    }

    #[test]
    fn an_exact_identifier_is_strong_but_a_plain_word_symbol_is_not() {
        let named = hit(
            1,
            Kind::Symbol,
            "retry_with_backoff",
            "fn retry_with_backoff()",
        );
        assert_eq!(
            classify(&named, &terms("retry_with_backoff"), &[vec![]]),
            MatchStrength::Strong
        );
        let path = hit(
            2,
            Kind::Symbol,
            "Mimir::search_with",
            "pub fn search_with()",
        );
        assert_eq!(
            classify(&path, &terms("what does search_with do"), &[vec![]]),
            MatchStrength::Strong
        );
        // `open` is a word, not an identifier: no shortcut.
        let word = hit(3, Kind::Symbol, "open", "pub fn open()");
        assert_eq!(
            classify(&word, &terms("open the file"), &[vec![]]),
            MatchStrength::Weak
        );
    }

    #[test]
    fn a_stopword_only_query_has_no_evidence() {
        let h = hit(1, Kind::Memory, "anything", "the a of");
        assert_eq!(
            classify(&h, &terms("what is the"), &[vec![1], vec![1]]),
            MatchStrength::Weak
        );
    }

    #[test]
    fn a_page_is_a_miss_only_when_nothing_is_strong() {
        assert!(is_miss(&[]));
        assert!(is_miss(&[MatchStrength::Weak, MatchStrength::Weak]));
        assert!(!is_miss(&[MatchStrength::Weak, MatchStrength::Strong]));
    }
}
