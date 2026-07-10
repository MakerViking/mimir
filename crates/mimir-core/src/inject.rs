//! Per-prompt auto-recall injection: the relevance floor, formatting, and
//! token budget shared by the cold CLI path (`mimir recall-inject`) and the
//! warm HTTP endpoint (`mimir mcp --http`'s `GET /inject`). Single-sourced
//! here so the two call sites can't drift — see `mimir-cli/src/commands.rs`
//! for the UserPromptSubmit hook script that tries warm, falls back to cold.

use std::collections::HashSet;

use crate::error::Result;
use crate::format::agent_line;
use crate::memory;
use crate::model::{Kind, Scope};
use crate::search::{fts, Hit, SearchQuery};
use crate::{store, Mimir};

/// ~200 tokens — team target, comparable to THOR's ~212. Injecting a wrong
/// memory is worse than injecting nothing, so callers never see partial
/// sentences, just a shorter (or absent) line.
pub const TOKEN_BUDGET: usize = 200;

/// Minimum number of non-stopword prompt tokens that must also appear in
/// the candidate's title+body+tags. Always enforced — the cheapest,
/// always-available half of the relevance floor (works with or without an
/// embedding model).
pub const MIN_OVERLAP: usize = 2;

/// How deep into the vector leg's own ranking (not the fused rank) a hit
/// must appear to count as "semantically agreed" for rule 3 below.
pub const LEG_TOP_K: usize = 10;

/// Compute at most one relevant-memory injection for `prompt` in `scope`,
/// or `None` if nothing clears the relevance floor. Also logs the
/// impression (same as an explicit `recall`) so it feeds future ranking.
/// Runs on every single prompt via one of two call sites, so it must stay
/// fast and side-effect-light: RRF's fused score isn't a calibrated
/// probability, so it can't be thresholded directly, and a wrong injected
/// memory actively misleads the agent — worse than saying nothing.
///
/// Relevance floor (erring toward silence):
///   1. Candidate must be `Kind::Memory` with subkind gotcha/decision (the
///      two "preventer" types, see `learn::type_prior`) OR pinned. Pinned
///      is a deliberate, durable importance marker the user set explicitly
///      — unlike `strength` (a usage pump, +0.3/day on open), which would
///      auto-inject chatty notes just because they're opened often, so it
///      is deliberately NOT used as a bypass here.
///   2. At least `MIN_OVERLAP` non-stopword prompt tokens must also appear
///      in the hit's title+body+tags. Tags are included because the FTS
///      leg ranks on `tags_text` at weight 3.0 — a hit can reach #1
///      *because of* its tags, and excluding them from the overlap check
///      let such hits clear the floor without actually matching the
///      prompt's words.
///   3. When a vector leg was actually computed (an embedding model is
///      loaded), the hit must ALSO appear in that leg's own top
///      `LEG_TOP_K` — lexical+semantic agreement, not just a lucky BM25
///      match. Skipped (not required) with no model: degrades cleanly to
///      rules 1+2 alone rather than refusing to ever fire.
pub fn compute(mimir: &mut Mimir, prompt: &str, scope: Scope) -> Result<Option<String>> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(None);
    }
    let query = SearchQuery {
        text: prompt.to_string(),
        scope,
        kinds: vec![Kind::Memory],
        since: None,
        limit: 5,
        strength_alpha: mimir.config.scoring.strength_alpha,
        recency_alpha: mimir.config.scoring.recency_alpha,
        type_prior_alpha: mimir.config.scoring.type_prior_alpha,
        code_damp: mimir.config.scoring.code_damp,
        include_superseded: false,
    };
    let (hits, legs) = mimir.search_with_legs(&query)?;
    let Some(top) = hits.first() else {
        return Ok(None);
    };
    if !clears_floor(top, prompt, &legs) {
        return Ok(None);
    }

    let projects = store::project_titles(&mimir.conn)?;
    let project_name = top
        .node
        .project_id
        .and_then(|id| projects.get(&id).map(String::as_str));
    let text = truncate_to_token_budget(
        &format!(
            "Relevant memory: {}",
            agent_line(&top.node, project_name, mimir.config.output.snippet_chars)
        ),
        TOKEN_BUDGET,
    );

    // Log the impression like a normal recall, so it feeds future ranking
    // (learn::record_shown / strength decay) same as an explicit recall.
    let query_hash = blake3::hash(query.text.as_bytes());
    store::record_shown(
        &mimir.conn,
        query_hash.as_bytes(),
        &[(top.node.id, 0i64, top.score)],
    )?;

    Ok(Some(text))
}

fn clears_floor(top: &Hit, prompt: &str, legs: &[Vec<i64>]) -> bool {
    // Rule 1: preventer types, or explicitly pinned.
    let is_preventer = matches!(
        top.node.subkind.as_deref(),
        Some("gotcha") | Some("decision")
    );
    if !is_preventer && !top.node.pinned {
        return false;
    }

    // Rule 2: minimum term overlap against title+body+tags (case/
    // punctuation-insensitive, via the same tokenizer the near-duplicate
    // detector uses).
    let prompt_tokens: HashSet<String> = memory::tokens(prompt)
        .into_iter()
        .filter(|t| !fts::is_stopword(t))
        .collect();
    let hay = format!(
        "{} {} {}",
        top.node.title.as_deref().unwrap_or(""),
        top.node.body.as_deref().unwrap_or(""),
        top.node.tags_text,
    );
    let hay_tokens: HashSet<String> = memory::tokens(&hay).into_iter().collect();
    let overlap = prompt_tokens.intersection(&hay_tokens).count();
    if overlap < MIN_OVERLAP {
        return false;
    }

    // Rule 3: dual-leg agreement, only when a vector leg exists (legs[0] is
    // always FTS, legs[1] is vector iff a model embedded this query).
    if let Some(vector_leg) = legs.get(1) {
        let agrees = vector_leg
            .iter()
            .take(LEG_TOP_K)
            .any(|id| *id == top.node.id);
        if !agrees {
            return false;
        }
    }

    true
}

/// Trim `text` to at most `budget` tokens, using a cheap chars/4 heuristic
/// instead of exact BPE counting (`mimir_core::tokens::count`). This runs
/// on every single prompt (cold CLI process, or warm HTTP request), and the
/// bundled tokenizer's ~100ms one-time table-parse cost is a real chunk of
/// turn latency for a 200-token budget that doesn't need BPE precision.
/// `tokens::count` stays the source of truth where precision matters (the
/// savings ledger); ~±10% imprecision here only ever costs a slightly
/// short or slightly long injected line.
fn truncate_to_token_budget(text: &str, budget: usize) -> String {
    let limit_chars = budget * 4; // o200k_base averages ~4 chars/token for English
    if text.chars().count() <= limit_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(limit_chars).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Remember, RememberOutcome};
    use crate::model::MemoryType;

    fn seed(mimir: &mut Mimir, mtype: MemoryType, title: &str, body: &str, tags: &[&str]) -> i64 {
        let out = memory::remember(
            &mimir.conn,
            Remember {
                text: format!("{title}. {body}"),
                mtype,
                tags: tags.iter().map(|s| s.to_string()).collect(),
                project_id: None,
                force: false,
            },
        )
        .unwrap();
        match out {
            RememberOutcome::Created(node) | RememberOutcome::Duplicate(node) => node.id,
        }
    }

    #[test]
    fn note_type_never_injects_even_with_overlap() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Note,
            "SCRAM auth notes",
            "some notes about scram authentication normalization",
            &[],
        );
        let out = compute(
            &mut mimir,
            "how does scram authentication normalization work",
            Scope::All,
        )
        .unwrap();
        assert!(out.is_none(), "note-type hit must never auto-inject");
    }

    #[test]
    fn pinned_note_passes_rule_one() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let id = seed(
            &mut mimir,
            MemoryType::Note,
            "SCRAM auth pinned note",
            "pinned notes about scram authentication normalization details",
            &[],
        );
        store::set_pinned(&mimir.conn, id, true).unwrap();
        let out = compute(
            &mut mimir,
            "how does scram authentication normalization work",
            Scope::All,
        )
        .unwrap();
        assert!(out.is_some(), "pinned hit should clear rule 1 like a gotcha/decision");
    }

    #[test]
    fn tags_count_toward_overlap() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        // Title/body deliberately share almost nothing with the prompt;
        // only the tags do. Without tags in the haystack this must fail
        // the overlap floor even though FTS (tags_text weight 3.0) could
        // rank it #1.
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "auth edge case",
            "remember to check this before shipping",
            &["scram", "nfkc", "normalization"],
        );
        let out = compute(&mut mimir, "scram nfkc normalization", Scope::All).unwrap();
        assert!(out.is_some(), "tag overlap alone should clear rule 2");
    }

    #[test]
    fn unrelated_prompt_stays_silent() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "SCRAM auth rejects non-ASCII passwords",
            "always normalize with NFKC before hashing",
            &["auth", "scram"],
        );
        let out = compute(
            &mut mimir,
            "please refactor this function to be more readable",
            Scope::All,
        )
        .unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn budget_truncation_is_bounded_and_cheap() {
        let long = "word ".repeat(500);
        let out = truncate_to_token_budget(&long, 50);
        // 50 tokens * 4 chars/token + 1 ellipsis char, generously bounded.
        assert!(out.chars().count() <= 50 * 4 + 1);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn short_text_is_untouched() {
        let out = truncate_to_token_budget("short line", 200);
        assert_eq!(out, "short line");
    }
}
