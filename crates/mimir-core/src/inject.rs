//! Per-prompt auto-recall injection: the relevance floor, formatting, and
//! token budget shared by the cold CLI path (`mimir recall-inject`) and the
//! warm HTTP endpoint (`mimir mcp --http`'s `GET /inject`). Single-sourced
//! here so the two call sites can't drift — see `mimir-cli/src/commands.rs`
//! for the UserPromptSubmit hook script that tries warm, falls back to cold.

use std::collections::HashSet;

use rusqlite::Connection;

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
///
/// **Short-prompt relaxation was tried and backed out** (measured on the
/// hermetic eval, `eval::run_inject_eval_hermetic`): relaxing to a 1-token
/// overlap for prompts under 4 non-stopword tokens — gated on rule 3
/// (vector-leg agreement) also holding — fixed the intended case
/// ("force-push risk?" now injects `mem_drift_force_push`) but also let an
/// unrelated short prompt ("let's get started") clear the floor on a
/// single coincidental token plus hermetic-vector agreement, injecting the
/// wrong gotcha. Net effect on the same fixture snapshot: injected-wrong
/// rose 1→2. Backed out per the product contract (silence beats wrong
/// injection). A future attempt should tighten rule 3's bar (e.g. a
/// vector-rank cutoff, not just top-`LEG_TOP_K` membership) before
/// relaxing rule 2, rather than relaxing rule 2 alone.
pub const MIN_OVERLAP: usize = 2;

/// How deep into the vector leg's own ranking (not the fused rank) a hit
/// must appear to count as "semantically agreed" for rule 3 below.
pub const LEG_TOP_K: usize = 10;

/// How many of the (fused-ranked) candidate hits `select_injection` scans
/// before giving up. Measured at 3 (scanning past the fused #1 hit for one
/// that clears the floor's stricter rules): it flipped one fixture case
/// from silent-wrong to injected-wrong (a stale `pool_old` decision at rank
/// 2 cleared the floor once the correct `pool_new` at rank 1 didn't) —
/// worse per the product contract, so it was backed out. Kept at 1: the
/// fused #1 hit is what the floor gates, full stop.
pub const TOP_CANDIDATES: usize = 1;

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
///      prompt's words. `enrich` (client-side git-diff stems, see
///      `mimir-cli/src/commands.rs`) tokens count toward this overlap too,
///      but can't carry it alone — see the `enrich` note on `compute`.
///   3. When a vector leg was actually computed (an embedding model is
///      loaded), the hit must ALSO appear in that leg's own top
///      `LEG_TOP_K` — lexical+semantic agreement, not just a lucky BM25
///      match. Skipped (not required) with no model: degrades cleanly to
///      rules 1+2 alone rather than refusing to ever fire.
///
/// `enrich`: space-separated stems from the caller's working tree (e.g. changed
/// file names from `git diff --name-only`), passed separately from `prompt`.
/// It strengthens the search query text and can help meet rule 2's overlap —
/// but a hit may never clear the floor on enrichment terms alone: at least
/// one raw-prompt-token overlap or rule 3's vector agreement must also hold.
/// Without this guard, enrichment would let *any* prompt on a branch that
/// touches file X auto-inject every gotcha whose title happens to mention
/// X, regardless of what the prompt actually asked.
///
/// **Fires-when bypass:** a candidate carrying author-declared trigger
/// phrases in `meta.fires_when` (set via `remember`'s `fires_when` arg, see
/// `memory::sanitize_fires_when`) skips rules 1 and 2 entirely when the
/// prompt (+ enrich) matches one of its phrases — see `fires_when_matches`.
/// This is a deliberate, explicit opt-in from whoever wrote the memory, not
/// an inferred signal, so it's held to a different bar than the rest of the
/// floor. Rule 3 (vector-leg agreement) still applies when a vector leg
/// exists: a declared trigger says "this topic matters", not "ignore a
/// contradicting embedding signal".
pub fn compute(
    mimir: &mut Mimir,
    prompt: &str,
    enrich: &str,
    scope: Scope,
) -> Result<Option<String>> {
    compute_with_session(mimir, prompt, enrich, scope, false, None)
}

/// Same as [`compute`], but when `bm25_only` is true, never touches the
/// embedder (`Mimir::search_with_legs_bm25_only` instead of
/// `search_with_legs`) — no `ensure_embedder` call, so a cold process
/// skips the ONNX-load + matrix-load cost entirely. Used by the cold
/// `mimir recall-inject` CLI path when config `[hooks] cold_mode = "fast"`
/// (default). `compute` is a thin `bm25_only = false` wrapper so the warm
/// `/inject` HTTP endpoint (`mcp.rs`, which stays resident and should
/// always use the best available signal) keeps calling the unchanged
/// 4-argument `compute` and behaves identically to before this existed.
///
/// With `bm25_only = true`, `legs` never has a `legs[1]` (vector) entry —
/// the same shape `clears_floor`'s rule 3 and the self-licensing guard
/// already handle for a model-less machine today (both check
/// `legs.get(1)`, which is `None` here), so no floor-rule changes were
/// needed for this mode; only routing which search variant is called.
pub fn compute_with_mode(
    mimir: &mut Mimir,
    prompt: &str,
    enrich: &str,
    scope: Scope,
    bm25_only: bool,
) -> Result<Option<String>> {
    compute_with_session(mimir, prompt, enrich, scope, bm25_only, None)
}

/// Same as [`compute_with_mode`], plus an optional per-session dedup ledger:
/// when `session_id` is `Some`, a candidate that already clears the floor is
/// still skipped if it was already injected earlier in that same session
/// (`store::injection_seen`) — the search moves on to the next
/// floor-clearing candidate instead of repeating it (see
/// `select_unseen_injection`'s doc comment for why this doesn't reopen the
/// regression `TOP_CANDIDATES` was pinned against). On a fresh pick, the
/// injection is recorded (`store::record_injection`) so it won't repeat
/// later in the same session. `session_id: None` reproduces today's
/// behavior exactly — no ledger read, no ledger write.
///
/// Ledger I/O is fail-open: a read or write error is logged at `debug` and
/// treated as if the ledger were empty/absent, so a store hiccup degrades
/// to "inject as if this session has no history" rather than blocking an
/// otherwise-valid injection.
pub fn compute_with_session(
    mimir: &mut Mimir,
    prompt: &str,
    enrich: &str,
    scope: Scope,
    bm25_only: bool,
    session_id: Option<&str>,
) -> Result<Option<String>> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(None);
    }
    let search_text = if enrich.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt} {enrich}")
    };
    let query = SearchQuery {
        text: search_text,
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
    let (hits, legs) = if bm25_only {
        mimir.search_with_legs_bm25_only(&query)?
    } else {
        mimir.search_with_legs(&query)?
    };
    let top = match session_id {
        Some(session_id) => {
            match select_unseen_injection(&mimir.conn, session_id, &hits, prompt, enrich, &legs) {
                Some(hit) => hit,
                None => return Ok(None),
            }
        }
        None => match select_injection(&hits, prompt, enrich, &legs) {
            Some(hit) => hit,
            None => return Ok(None),
        },
    };

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
    // Hashed on the raw prompt, not the (possibly enriched) search text —
    // enrichment is per-invocation working-tree state, not part of what the
    // user asked, so it shouldn't fragment the impression key for the same
    // question asked from different working-tree states.
    let query_hash = blake3::hash(prompt.as_bytes());
    store::record_shown(
        &mimir.conn,
        query_hash.as_bytes(),
        &[(top.node.id, 0i64, top.score)],
    )?;

    if let Some(session_id) = session_id {
        if let Err(err) = store::record_injection(&mimir.conn, session_id, top.node.id) {
            tracing::debug!(%err, "inject: failed to record injection ledger row");
        }
    }

    Ok(Some(text))
}

/// Pure selection core: scans the top [`TOP_CANDIDATES`] fused hits and
/// returns the first that clears [`clears_floor`], or `None`. Split out of
/// [`compute`] so the eval harness (`eval::mod`'s preventer end-to-end
/// scoring) can drive the exact same decision logic over fixture search
/// results without a real `Mimir` engine (ONNX model, on-disk store) —
/// only `compute`'s I/O shell (the search call, formatting, impression
/// logging) needs one.
///
/// Before that top-1 scan, every returned hit (not just the fused #1 —
/// `TOP_CANDIDATES` doesn't apply here, see [`fires_when_clears`]) is
/// checked for an author-declared `fires_when` trigger match; a hit is
/// what actually got scored least by inferred relevance is exactly the
/// case a paraphrase-missing-rule-2 trigger exists to rescue.
pub fn select_injection<'h>(
    hits: &'h [Hit],
    prompt: &str,
    enrich: &str,
    legs: &[Vec<i64>],
) -> Option<&'h Hit> {
    if let Some(hit) = hits
        .iter()
        .find(|hit| fires_when_clears(hit, prompt, enrich, legs))
    {
        return Some(hit);
    }
    hits.iter()
        .take(TOP_CANDIDATES)
        .find(|hit| clears_floor(hit, prompt, enrich, legs))
}

/// Same as [`select_injection`], but skips any candidate already recorded
/// in the per-session dedup ledger (`store::injection_seen`) and, unlike
/// `select_injection`'s top-1 fallback scan, keeps scanning past the fused
/// #1 hit there when it does.
///
/// The fallback scan intentionally goes further than [`TOP_CANDIDATES`]
/// normally allows (see that constant's doc comment for the regression
/// it's pinned against), but it's not the same relaxation: that regression
/// came from letting a *lower-quality* hit qualify only because a *better*
/// hit at rank 1 had already failed [`clears_floor`] — a quality fallback
/// that let a stale, wrong pick through. Here every candidate this
/// function returns, at any rank, still has to independently pass the
/// exact same unrelaxed `clears_floor`; the only reason to move past rank
/// 1 is that it already qualified and was already shown this session, so
/// returning `None` for a rank-2 hit that *also* legitimately qualifies
/// would just mean staying silent for no quality reason. A store read
/// error on a given candidate is treated as "not seen" (fail-open) rather
/// than skipping it.
fn select_unseen_injection<'h>(
    conn: &Connection,
    session_id: &str,
    hits: &'h [Hit],
    prompt: &str,
    enrich: &str,
    legs: &[Vec<i64>],
) -> Option<&'h Hit> {
    let unseen =
        |hit: &&Hit| !store::injection_seen(conn, session_id, hit.node.id).unwrap_or(false);
    if let Some(hit) = hits
        .iter()
        .filter(|hit| fires_when_clears(hit, prompt, enrich, legs))
        .find(unseen)
    {
        return Some(hit);
    }
    hits.iter()
        .filter(|hit| clears_floor(hit, prompt, enrich, legs))
        .find(unseen)
}

/// True if `phrase` (one already-lowercased trigger from a candidate's
/// `meta.fires_when`, see `memory::sanitize_fires_when`) fires against the
/// prompt: either `phrase` appears as a literal substring of the lowercased
/// `prompt`+`enrich` text, or every non-stopword word in `phrase` is present
/// somewhere in `prompt_tokens` (already the union of prompt + enrich
/// tokens, stopwords stripped) — order-free, so a trigger of "release
/// checklist" also fires on "checklist for the release".
fn fires_when_matches(phrase: &str, prompt_lower: &str, prompt_tokens: &HashSet<String>) -> bool {
    if !phrase.is_empty() && prompt_lower.contains(phrase) {
        return true;
    }
    let mut words = phrase
        .split_whitespace()
        .filter(|w| !fts::is_stopword(w))
        .peekable();
    if words.peek().is_none() {
        return false;
    }
    words.all(|w| prompt_tokens.contains(w))
}

/// True if `top` carries an author-declared `meta.fires_when` trigger
/// matching `prompt` (+ `enrich`) — see [`fires_when_matches`] — AND, when
/// a vector leg exists, `top` still appears in that leg's own top
/// [`LEG_TOP_K`] (the same rule 3 [`clears_floor`] enforces: a declared
/// trigger says "this topic matters", not "ignore a contradicting
/// embedding signal"). Deliberately independent of `clears_floor`'s rules
/// 1 and 2 (type/pinned gate, min overlap) and of [`TOP_CANDIDATES`] — an
/// explicit author opt-in is checked against *every* returned candidate,
/// not just the fused #1 hit, since the whole point is to catch a
/// paraphrase that inferred relevance ranked low.
fn fires_when_clears(top: &Hit, prompt: &str, enrich: &str, legs: &[Vec<i64>]) -> bool {
    let Some(phrases) = top.node.meta.get("fires_when").and_then(|v| v.as_array()) else {
        return false;
    };
    if phrases.is_empty() {
        return false;
    }
    let prompt_lower = format!("{} {}", prompt.to_lowercase(), enrich.to_lowercase());
    let prompt_tokens: HashSet<String> = memory::tokens(prompt)
        .into_iter()
        .chain(memory::tokens(enrich))
        .filter(|t| !fts::is_stopword(t))
        .collect();
    let matched = phrases
        .iter()
        .filter_map(|p| p.as_str())
        .any(|phrase| fires_when_matches(phrase, &prompt_lower, &prompt_tokens));
    if !matched {
        return false;
    }
    match legs.get(1) {
        Some(vector_leg) => vector_leg
            .iter()
            .take(LEG_TOP_K)
            .any(|id| *id == top.node.id),
        None => true,
    }
}

fn clears_floor(top: &Hit, prompt: &str, enrich: &str, legs: &[Vec<i64>]) -> bool {
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
    // detector uses). `enrich` tokens are folded into the overlap count
    // (see the self-licensing guard below, after rule 3).
    let raw_tokens: HashSet<String> = memory::tokens(prompt)
        .into_iter()
        .filter(|t| !fts::is_stopword(t))
        .collect();
    let enrich_tokens: HashSet<String> = memory::tokens(enrich)
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
    let raw_overlap = raw_tokens.intersection(&hay_tokens).count();
    let combined_overlap = raw_tokens
        .union(&enrich_tokens)
        .filter(|t| hay_tokens.contains(*t))
        .count();
    if combined_overlap < MIN_OVERLAP {
        return false;
    }

    // Rule 3: dual-leg agreement, only when a vector leg exists (legs[0] is
    // always FTS, legs[1] is vector iff a model embedded this query).
    let vector_agrees = match legs.get(1) {
        Some(vector_leg) => vector_leg
            .iter()
            .take(LEG_TOP_K)
            .any(|id| *id == top.node.id),
        None => false,
    };
    if legs.get(1).is_some() && !vector_agrees {
        return false;
    }

    // Self-licensing guard: if enrichment was load-bearing for rule 2 (raw
    // prompt tokens alone didn't reach MIN_OVERLAP), enrichment can't also
    // be the only thing backing up rule 3 — at least one raw-prompt-token
    // overlap or confirmed vector agreement must independently hold. Silent
    // beats wrong: a hit shouldn't clear the floor purely because the
    // caller's working tree happens to touch a file whose name matches an
    // unrelated gotcha's title.
    //
    // "Confirmed" vector agreement requires the vector leg to actually
    // discriminate: a store small enough that every node fits within the
    // leg's own top-`LEG_TOP_K` (e.g. a fresh project with a handful of
    // memories) makes membership there true of *any* candidate, sighted or
    // not — sandbox-tested (2-memory store): without this check, an
    // unrelated prompt ("let's get started") on a branch that happened to
    // touch two files matching an unrelated gotcha's title tokens cleared
    // the floor purely on that trivial "agreement". Requiring the leg to
    // be strictly larger than `LEG_TOP_K` before its agreement counts here
    // closes that gap without touching rule 3's own (pre-existing,
    // out-of-scope) behavior for the raw_overlap-driven path above.
    let vector_leg_discriminates = legs.get(1).is_some_and(|l| l.len() > LEG_TOP_K);
    let enrichment_was_needed = raw_overlap < MIN_OVERLAP;
    if enrichment_was_needed && raw_overlap == 0 && !(vector_agrees && vector_leg_discriminates) {
        return false;
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
            RememberOutcome::Created(node)
            | RememberOutcome::Duplicate(node)
            | RememberOutcome::Forgotten(node) => node.id,
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
            "",
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
            "",
            Scope::All,
        )
        .unwrap();
        assert!(
            out.is_some(),
            "pinned hit should clear rule 1 like a gotcha/decision"
        );
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
        let out = compute(&mut mimir, "scram nfkc normalization", "", Scope::All).unwrap();
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
            "",
            Scope::All,
        )
        .unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn enrichment_alone_cannot_clear_floor() {
        // Prompt shares zero tokens with the hit; only the (simulated
        // git-diff) enrichment terms overlap. Without a real embedding
        // model (as here) rule 3 never agrees, so the self-licensing guard
        // must block this — enrichment can strengthen a real match, not
        // manufacture one on its own.
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "WAL autocheckpoint pathological growth",
            "always run PRAGMA wal_checkpoint before assuming corruption",
            &[],
        );
        let out = compute(
            &mut mimir,
            "let's get started",
            "wal autocheckpoint",
            Scope::All,
        )
        .unwrap();
        assert!(
            out.is_none(),
            "enrichment-only overlap must not self-license an injection"
        );
    }

    #[test]
    fn enrichment_extends_a_real_overlap() {
        // Prompt already shares one real token ("checkpoint") with the
        // hit; enrichment supplies a second. Since raw overlap is
        // non-zero, the self-licensing guard doesn't apply — enrichment is
        // allowed to carry the hit the rest of the way to MIN_OVERLAP.
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "WAL checkpoint pathological growth",
            "always run PRAGMA wal_checkpoint before assuming corruption",
            &[],
        );
        let out = compute(
            &mut mimir,
            "why does checkpoint never happen?",
            "pathological",
            Scope::All,
        )
        .unwrap();
        assert!(
            out.is_some(),
            "enrichment should extend a real single-token overlap"
        );
    }

    #[test]
    fn bm25_only_mode_still_injects_on_a_clean_lexical_match() {
        // `search_with_legs_bm25_only` never computes a vector leg, so
        // `legs.get(1)` is always None here — the same shape rule 3 and the
        // self-licensing guard already handle for a model-less machine
        // (both degrade to skipping when there's no vector leg). This
        // confirms `cold_mode = "fast"` still fires on a real, unambiguous
        // lexical match once rules 1+2 alone are asked to carry the floor —
        // same fixture as `tags_count_toward_overlap` (exact tag overlap,
        // no ambiguity about whether MIN_OVERLAP is cleared).
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "auth edge case",
            "remember to check this before shipping",
            &["scram", "nfkc", "normalization"],
        );
        let out = compute_with_mode(&mut mimir, "scram nfkc normalization", "", Scope::All, true)
            .unwrap();
        assert!(
            out.is_some(),
            "bm25_only mode must still inject on a clean lexical match \
             (no vector leg to require agreement from)"
        );
    }

    #[test]
    fn bm25_only_mode_still_blocks_enrichment_only_self_licensing() {
        // Same enrichment-only scenario as `enrichment_alone_cannot_clear_floor`,
        // run through the bm25_only path: with no vector leg at all,
        // `vector_leg_discriminates` is false and `vector_agrees` is false,
        // so the self-licensing guard's `!(vector_agrees &&
        // vector_leg_discriminates)` term is true regardless — the guard
        // must still block, exactly as it does with a real absent model.
        let mut mimir = Mimir::open_in_memory().unwrap();
        seed(
            &mut mimir,
            MemoryType::Gotcha,
            "WAL autocheckpoint pathological growth",
            "always run PRAGMA wal_checkpoint before assuming corruption",
            &[],
        );
        let out = compute_with_mode(
            &mut mimir,
            "let's get started",
            "wal autocheckpoint",
            Scope::All,
            true,
        )
        .unwrap();
        assert!(
            out.is_none(),
            "bm25_only mode must not let enrichment-only overlap self-license an injection"
        );
    }

    #[test]
    fn fires_when_bypasses_type_and_overlap_rules() {
        // A Note (never a preventer type) sharing only one token with the
        // prompt ("release" — below MIN_OVERLAP) would fail both rule 1 and
        // rule 2 today. A declared fires_when trigger should let it clear
        // the floor anyway on a prompt that matches the trigger, even in
        // paraphrased word order. One shared token is kept so the FTS leg
        // still surfaces this node as a search candidate in the first
        // place — the floor rules only ever gate a candidate that search
        // already returned, they don't influence retrieval itself.
        let mut mimir = Mimir::open_in_memory().unwrap();
        let id = seed(
            &mut mimir,
            MemoryType::Note,
            "Release day operational notes",
            "double check environment variables before shipping",
            &[],
        );
        store::set_fires_when(&mimir.conn, id, &["release checklist".to_string()]).unwrap();
        let out = compute(
            &mut mimir,
            "is there a checklist to run through before we release?",
            "",
            Scope::All,
        )
        .unwrap();
        assert!(out.is_some(), "fires_when match must bypass rules 1 and 2");
    }

    #[test]
    fn fires_when_present_but_not_matched_stays_silent() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let id = seed(
            &mut mimir,
            MemoryType::Note,
            "Release day operational notes",
            "double check environment variables before shipping",
            &[],
        );
        store::set_fires_when(&mimir.conn, id, &["release checklist".to_string()]).unwrap();
        let out = compute(&mut mimir, "why is the CI docs job failing", "", Scope::All).unwrap();
        assert!(
            out.is_none(),
            "an unrelated prompt must not fire an unmatched trigger"
        );
    }

    #[test]
    fn session_ledger_dedups_within_a_session_but_not_across() {
        let mut mimir = Mimir::open_in_memory().unwrap();
        let id = seed(
            &mut mimir,
            MemoryType::Note,
            "Release day operational notes",
            "double check environment variables before shipping",
            &[],
        );
        store::set_fires_when(&mimir.conn, id, &["release checklist".to_string()]).unwrap();

        let first = compute_with_session(
            &mut mimir,
            "is there a checklist to run through before we release?",
            "",
            Scope::All,
            false,
            Some("session-a"),
        )
        .unwrap();
        assert!(first.is_some(), "first call in a session should inject");

        let second = compute_with_session(
            &mut mimir,
            "is there a checklist to run through before we release?",
            "",
            Scope::All,
            false,
            Some("session-a"),
        )
        .unwrap();
        assert!(
            second.is_none(),
            "the same session must not re-inject the same memory"
        );

        let other_session = compute_with_session(
            &mut mimir,
            "is there a checklist to run through before we release?",
            "",
            Scope::All,
            false,
            Some("session-b"),
        )
        .unwrap();
        assert!(
            other_session.is_some(),
            "a different session should still inject"
        );
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
