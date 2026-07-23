//! Session brief: a hard-capped digest of the most drift-preventing
//! memories (gotchas + decisions), composed for injection at session start
//! and after a context clear/compaction. Design: docs/design/session-brief.md.
//!
//! Selection is deliberately query-free and model-free — session start is a
//! cold one-shot process, so everything here is one indexed SQL fetch over
//! the memory subset plus pure-Rust scoring that reuses `learn.rs`
//! verbatim. The per-prompt silence-first `/inject` floor is a different
//! surface and is untouched by this module.
//!
//! The split (candidates → rank → compose) mirrors `inject::select_injection`'s
//! pure-function shape so the eval fixtures can drive selection directly.

use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

use crate::config::{BriefConfig, ScoringConfig};
use crate::error::Result;
use crate::format::{collapse_ws, truncate_chars};
use crate::learn::{self, ImpressionStats};
use crate::model::{short_uid, Node, Scope};
use crate::store;

/// Fixed preamble; costs ~7 tokens and is only emitted when at least one
/// line survives selection (an empty brief prints nothing at all,
/// matching `rules show`'s silent-empty precedent).
pub const HEADER: &str = "Mimir guards (do not violate):";

/// Items not touched for longer than this render a visible age tag —
/// pin exempts decay everywhere else (and must keep doing so), so a
/// stale-but-pinned rule is at least visibly old instead of being shown
/// with implicit full confidence forever.
const STALE_AFTER_DAYS: i64 = 90;

/// A scored candidate, ready for budget fill.
pub struct Ranked {
    pub node: Node,
    pub score: f64,
}

/// The composed brief: rendered text (empty = print nothing) and the
/// rowids actually rendered, for the caller's suppression bookkeeping.
pub struct Brief {
    pub text: String,
    pub shown: Vec<i64>,
}

/// Stage 1 — candidate fetch. Hits the kind/scope index, then scans only
/// the memory subset (~1k rows at 100k-node scale, sub-ms). Deliberately
/// no LIMIT: a cap here would have to sort by some proxy key, and any
/// proxy that isn't the real Stage-2 score silently drops old-but-vital
/// candidates (adversarial-review blocker). `subkind` is the admission
/// gate; `pinned` is only a scoring boost, never an admission path — a
/// pinned note/idea is not a drift preventer.
pub fn candidates(conn: &Connection, scope: Scope) -> Result<Vec<Node>> {
    let sql = format!(
        "SELECT {} FROM node n
         WHERE n.kind = 'memory' AND n.deleted_at IS NULL
           AND n.superseded_by IS NULL
           AND n.subkind IN ('gotcha','decision')
           AND {}",
        store::NODE_COLS,
        crate::search::scope_sql(scope)
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map([], store::row_to_node)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Stage 2 — score in Rust, reusing `learn.rs` and the production gating
/// idioms from `search/mod.rs` (recency only for decaying kinds, alphas
/// applied as capped multiplicative nudges). There is no RRF base term —
/// no query exists at session start — so pin and type prior carry the
/// tier structure instead. `current_project` gives same-project nodes a
/// 1.5x affinity edge so a large personal global pool cannot crowd out a
/// small project's own gotchas.
pub fn rank(
    nodes: Vec<Node>,
    current_project: Option<i64>,
    scoring: &ScoringConfig,
    impressions: &HashMap<i64, ImpressionStats>,
    now: i64,
) -> Vec<Ranked> {
    let mut out: Vec<Ranked> = nodes
        .into_iter()
        .map(|node| {
            let effective = learn::effective_strength(&node, now);
            let recency_term =
                if learn::half_life_days(node.kind, node.subkind.as_deref()).is_some() {
                    1.0 + scoring.recency_alpha * learn::recency_factor(&node, now)
                } else {
                    1.0
                };
            let prior = learn::type_prior(node.subkind.as_deref());
            let stats = impressions.get(&node.id).copied().unwrap_or_default();
            let damp = learn::impression_damp(&stats, scoring.impression_alpha, now);
            let affinity = if current_project.is_some() && node.project_id == current_project {
                1.5
            } else {
                1.0
            };
            let pin = if node.pinned { 2.0 } else { 1.0 };
            let score = pin
                * (1.0 + scoring.strength_alpha * (1.0 + effective).ln())
                * recency_term
                * (1.0 + scoring.type_prior_alpha * (prior - 1.0))
                * damp
                * affinity;
            Ranked { node, score }
        })
        .collect();
    // Deterministic order: score, then freshness, then rowid.
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(b.node.updated_at.cmp(&a.node.updated_at))
            .then(a.node.id.cmp(&b.node.id))
    });
    out
}

/// Everything the CLI needs in one call: fetch, filter the caller's
/// exclusions (session suppression, handoff overlap) and rules-pack
/// content duplicates, batch the impression stats, rank.
pub fn select(
    conn: &Connection,
    scope: Scope,
    exclude: &HashSet<i64>,
    rules_body: Option<&str>,
    scoring: &ScoringConfig,
    now: i64,
) -> Result<Vec<Ranked>> {
    let rules_norm = rules_body.map(normalize);
    let pool: Vec<Node> = candidates(conn, scope)?
        .into_iter()
        .filter(|n| !exclude.contains(&n.id))
        .filter(|n| {
            rules_norm
                .as_deref()
                .is_none_or(|r| !duplicates_rules(n, r))
        })
        .collect();
    let ids: Vec<i64> = pool.iter().map(|n| n.id).collect();
    let impressions = store::impression_stats(conn, &ids)?;
    let current_project = match scope {
        Scope::Project(id) => Some(id),
        _ => None,
    };
    Ok(rank(pool, current_project, scoring, &impressions, now))
}

/// Stage 3 — budget fill + render. Per-item truncation happens here,
/// BEFORE the fill decision, so the loop only ever asks "does this whole
/// bounded line fit" — a mid-item cut could read as a different (wrong)
/// claim. Fill is ranked-order with best-fit fallback: when the
/// next-ranked line doesn't fit, keep scanning shorter lower-ranked lines
/// instead of stopping — strictly more value under the same hard cap.
/// `budget_tokens` is `max_tokens` for a startup fire and `recap_tokens`
/// for a clear/compact re-fire.
pub fn compose(ranked: &[Ranked], cfg: &BriefConfig, budget_tokens: usize, now: i64) -> Brief {
    let mut text = String::new();
    let mut shown = Vec::new();
    for r in ranked {
        if shown.len() >= cfg.max_items {
            break;
        }
        let line = render_line(&r.node, cfg.line_chars, now);
        if line.is_empty() {
            continue;
        }
        let candidate = if shown.is_empty() {
            format!("{HEADER}\n{line}")
        } else {
            format!("{text}\n{line}")
        };
        if crate::tokens::count(&candidate) <= budget_tokens {
            text = candidate;
            shown.push(r.node.id);
        }
    }
    if shown.is_empty() {
        text.clear();
    }
    Brief { text, shown }
}

/// One line per memory: literal GOTCHA/DECISION prefix (imperative
/// framing outlives symbols), compact node ref (a few tokens buys `mimir
/// get` traceability), age tag past the staleness horizon, body bounded
/// to `line_chars`.
fn render_line(node: &Node, line_chars: usize, now: i64) -> String {
    let Some(raw) = node.body.as_deref().or(node.title.as_deref()) else {
        return String::new();
    };
    let body = truncate_chars(&collapse_ws(raw), line_chars);
    if body.is_empty() {
        return String::new();
    }
    let label = match node.subkind.as_deref() {
        Some("decision") => "DECISION",
        _ => "GOTCHA",
    };
    let refid = short_uid(node.kind, &node.uid);
    let age_days = (now - node.updated_at) / 86_400;
    let age = if age_days > STALE_AFTER_DAYS {
        let months = age_days / 30;
        if months >= 12 {
            format!(" ({}y)", months / 12)
        } else {
            format!(" ({months}mo)")
        }
    } else {
        String::new()
    };
    format!("- {label} [{refid}]{age}: {body}")
}

/// Content-level rules-pack dedup (adversarial-review blocker: node-identity
/// dedup alone lets the same fact ride in twice when it appears verbatim
/// inside the rules-pack body). Normalized ~80-char prefix substring test —
/// cheap, and a prefix that long matching inside the rules body is not a
/// coincidence.
pub fn duplicates_rules(node: &Node, rules_norm: &str) -> bool {
    let Some(body) = node.body.as_deref() else {
        return false;
    };
    let prefix: String = normalize(body).chars().take(80).collect();
    !prefix.is_empty() && rules_norm.contains(&prefix)
}

/// Case- and whitespace-folded form used for the rules dedup test.
pub fn normalize(s: &str) -> String {
    collapse_ws(s).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BriefConfig;
    use crate::db;
    use crate::memory::{self, Remember, RememberOutcome};
    use crate::model::{now_unix, MemoryType};

    fn remember(conn: &Connection, text: &str, mtype: MemoryType, project_id: Option<i64>) -> Node {
        match memory::remember(
            conn,
            Remember {
                text: text.into(),
                mtype,
                tags: vec![],
                project_id,
                force: true,
            },
        )
        .unwrap()
        {
            RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        }
    }

    fn pin(conn: &Connection, id: i64) {
        conn.execute("UPDATE node SET pinned = 1 WHERE id = ?1", [id])
            .unwrap();
    }

    fn select_ids(conn: &Connection, scope: Scope) -> Vec<i64> {
        let ranked = select(
            conn,
            scope,
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        ranked.iter().map(|r| r.node.id).collect()
    }

    /// Admission gate: only gotcha/decision qualify — a pinned note is
    /// still not a drift preventer (pin is a boost, not an admission path).
    #[test]
    fn pool_admits_only_gotcha_and_decision() {
        let conn = db::open_in_memory().unwrap();
        let g = remember(&conn, "gotcha body", MemoryType::Gotcha, None);
        let d = remember(&conn, "decision body", MemoryType::Decision, None);
        let n = remember(&conn, "note body", MemoryType::Note, None);
        pin(&conn, n.id);
        let ids = select_ids(&conn, Scope::Global);
        assert!(ids.contains(&g.id) && ids.contains(&d.id));
        assert!(!ids.contains(&n.id), "pinned note must not be admitted");
    }

    /// Forbidden family: superseded and soft-deleted memories never
    /// surface — the brief has no relevance check, so this exclusion can
    /// never be optional.
    #[test]
    fn superseded_and_deleted_never_surface() {
        let conn = db::open_in_memory().unwrap();
        let old = remember(&conn, "old gotcha", MemoryType::Gotcha, None);
        let new = remember(&conn, "new gotcha replacing old", MemoryType::Gotcha, None);
        store::set_superseded(&conn, old.id, new.id).unwrap();
        let dead = remember(&conn, "deleted gotcha", MemoryType::Gotcha, None);
        conn.execute(
            "UPDATE node SET deleted_at = ?1 WHERE id = ?2",
            [now_unix(), dead.id],
        )
        .unwrap();
        let ids = select_ids(&conn, Scope::Global);
        assert!(ids.contains(&new.id));
        assert!(!ids.contains(&old.id), "superseded must never render");
        assert!(!ids.contains(&dead.id), "deleted must never render");
    }

    /// Crowd-out family: a project's own gotcha outranks an equal-signal
    /// global flood via the 1.5x affinity term.
    #[test]
    fn project_gotcha_beats_equal_global_flood() {
        let conn = db::open_in_memory().unwrap();
        let project = store::ensure_project(&conn, "/tmp/p", "p").unwrap();
        for i in 0..8 {
            remember(
                &conn,
                &format!("global gotcha {i}"),
                MemoryType::Gotcha,
                None,
            );
        }
        let ours = remember(
            &conn,
            "project gotcha",
            MemoryType::Gotcha,
            Some(project.id),
        );
        let ranked = select(
            &conn,
            Scope::Project(project.id),
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        assert_eq!(
            ranked.first().map(|r| r.node.id),
            Some(ours.id),
            "same-project affinity must put the project's own gotcha first"
        );
    }

    /// Budget family: both caps bind; token adherence holds at every
    /// fixture; best-fit keeps scanning after a too-long line.
    #[test]
    fn budget_caps_bind_and_best_fit_rescues_short_lines() {
        let conn = db::open_in_memory().unwrap();
        let long = remember(
            &conn,
            &format!("long gotcha {}", "verbose padding phrase ".repeat(12)),
            MemoryType::Gotcha,
            None,
        );
        pin(&conn, long.id); // rank it first
        let short = remember(&conn, "short gotcha", MemoryType::Gotcha, None);
        let ranked = select(
            &conn,
            Scope::Global,
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        assert_eq!(ranked[0].node.id, long.id, "precondition: long ranks first");
        let cfg = BriefConfig::default();
        // Budget sized so the long (pinned, truncated-to-line_chars) line
        // cannot fit but the short one can: header ~7 + short line fits
        // inside 22 tokens; the 100-char long line does not.
        let brief = compose(&ranked, &cfg, 22, now_unix());
        assert_eq!(
            brief.shown,
            vec![short.id],
            "best-fit must rescue the short line"
        );
        assert!(
            crate::tokens::count(&brief.text) <= 22,
            "token cap must hold"
        );

        // max_items binds independently of a generous token budget.
        for i in 0..10 {
            remember(&conn, &format!("tiny {i}"), MemoryType::Gotcha, None);
        }
        let ranked = select(
            &conn,
            Scope::Global,
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        let brief = compose(&ranked, &cfg, 10_000, now_unix());
        assert_eq!(brief.shown.len(), cfg.max_items);
    }

    /// Silent-empty precedent: zero candidates ⇒ empty text, not a lone
    /// header; and a zero-length store selects nothing without error.
    #[test]
    fn empty_selection_renders_nothing() {
        let conn = db::open_in_memory().unwrap();
        let ranked = select(
            &conn,
            Scope::Global,
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        let brief = compose(&ranked, &BriefConfig::default(), 150, now_unix());
        assert!(brief.text.is_empty() && brief.shown.is_empty());
    }

    /// Must-not-surface family: session suppression (exclude set) and
    /// rules-pack content dedup both filter before ranking.
    #[test]
    fn exclusions_and_rules_dedup_filter_candidates() {
        let conn = db::open_in_memory().unwrap();
        let a = remember(&conn, "already shown gotcha", MemoryType::Gotcha, None);
        let b = remember(
            &conn,
            "Never run cargo publish without fmt check first",
            MemoryType::Gotcha,
            None,
        );
        let c = remember(&conn, "fresh unrelated gotcha", MemoryType::Gotcha, None);
        let rules_body =
            "House rules:\n- never run cargo publish without fmt check first\n- be kind";
        let mut exclude = HashSet::new();
        exclude.insert(a.id);
        let ranked = select(
            &conn,
            Scope::Global,
            &exclude,
            Some(rules_body),
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        let ids: Vec<i64> = ranked.iter().map(|r| r.node.id).collect();
        assert!(
            !ids.contains(&a.id),
            "excluded (already shown) must be dropped"
        );
        assert!(
            !ids.contains(&b.id),
            "content duplicated in rules pack must be dropped"
        );
        assert!(ids.contains(&c.id));
    }

    /// Rendering: labels, ref, stale-age tag, pre-fill truncation bound.
    #[test]
    fn rendering_labels_refs_and_age() {
        let conn = db::open_in_memory().unwrap();
        let g = remember(&conn, "watch the borrow checker", MemoryType::Gotcha, None);
        let _d = remember(&conn, "we chose sqlite", MemoryType::Decision, None);
        let old_ts = now_unix() - 200 * 86_400;
        conn.execute(
            "UPDATE node SET updated_at = ?1, created_at = ?1 WHERE id = ?2",
            [old_ts, g.id],
        )
        .unwrap();
        let ranked = select(
            &conn,
            Scope::Global,
            &HashSet::new(),
            None,
            &crate::config::ScoringConfig::default(),
            now_unix(),
        )
        .unwrap();
        let brief = compose(&ranked, &BriefConfig::default(), 150, now_unix());
        assert!(brief.text.starts_with(HEADER));
        assert!(brief.text.contains("- GOTCHA [m:"));
        assert!(brief.text.contains("- DECISION [m:"));
        assert!(
            brief.text.contains("(6mo)"),
            "stale item must carry an age tag: {}",
            brief.text
        );
        assert!(
            !brief.text.contains("we chose sqlite ("),
            "fresh item must not carry an age tag"
        );
        for line in brief.text.lines().skip(1) {
            assert!(
                line.chars().count() <= BriefConfig::default().line_chars + 40,
                "line must be bounded before fill: {line}"
            );
        }
    }
}
