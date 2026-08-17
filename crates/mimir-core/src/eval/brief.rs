//! Drift eval for the session brief — the corpus + gates from
//! docs/design/session-brief.md §9. This is the harness that adjudicates
//! the `[brief] enabled` default-flip; the feature does not default on
//! until these gates pass with citable numbers.
//!
//! Unlike the retrieval eval (`super::fixtures`), there is no query and no
//! model: brief selection is query-free SQL + pure-Rust scoring, so every
//! fixture here is a *store shape* — a project/global memory pool with
//! pins, strengths, supersedes, deletions, a rules pack, and a suppression
//! state — and a [`BriefQuestion`] labels what the composed brief must and
//! must never contain. `brief::select` + `brief::compose` are driven as
//! pure functions (the same split `inject::select_injection` exposes for
//! the preventer eval), so the whole thing is deterministic and runs in
//! plain `cargo test`.
//!
//! ## Ground truth
//!
//! Labels are authored from the user-labeled dogfood corpus, not from how
//! the current selector behaves:
//! - **BookForge = silence**: a project whose own footprint is unrelated
//!   to every global received other projects' gotchas as its whole brief;
//!   the right brief was nothing. Silence beats filler, always.
//! - **ARIA = the project's canonical gotcha first**: the project's own
//!   strongest guard outranks everything else shown.
//! - **Rotation-to-weak-tail = wrong**: once the strong guards are
//!   suppressed, serving the weak tail as "best available" is a labeling
//!   error, not a feature.
//! - A pinned global is the user saying "never miss this" — it must
//!   surface even where the lexical relevance gate would drop it.
//!
//! ## What this harness cannot measure
//!
//! §9's repeated-exposure family has two halves. The *selection* half —
//! does the top guard keep surfacing across sessions instead of rotating
//! away, does a recap re-anchor it, does a within-window sibling session
//! stay silent — is asserted here ([`replay_sessions`]). The *compliance*
//! half — does the agent go boilerplate-blind to a line it has seen N
//! sessions in a row — needs a real agent reading real briefs and cannot
//! be asserted in Rust. The `#[ignore]`d report prints each simulated
//! session's rendered brief verbatim so that measurement can be run
//! against them (dogfood labeling); any number published from this file
//! is scoped to hook-running clients (§9 gate 5), not a general
//! session-drift claim.

use std::collections::{HashMap, HashSet};

use rusqlite::{params, Connection};

use crate::brief;
use crate::config::{BriefConfig, Config};
use crate::db;
use crate::error::Result;
use crate::memory::{self, Remember, RememberOutcome};
use crate::model::{now_unix, MemoryType, Scope};
use crate::store;
use crate::tokens;

/// The §9 marginal-catch caps: rerun composition at each, then let
/// [`calibrated_cap`] pick the winner.
pub const CURVE_CAPS: [usize; 4] = [100, 150, 250, 400];

/// §9's calibration rule: "each +100 tokens must buy ≥3 pts of catch or
/// the smaller cap wins." Applied pro-rata to the uneven step sizes in
/// [`CURVE_CAPS`] (100→150 must buy ≥1.5 pts, 250→400 ≥4.5).
pub const MIN_PTS_PER_100_TOKENS: f64 = 3.0;

/// Which store shape a question runs against. Several questions share one
/// shape; the store is rebuilt in-memory per question (cheap, and keeps
/// every question independent of run order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreFixture {
    /// A project with real guards of its own plus a global pool: the
    /// crowd-out / fresh-capture / must-not-surface families, and the
    /// replay corpus. Encodes the ARIA dogfood label.
    Aria,
    /// A project with an unrelated footprint plus irrelevant globals and
    /// one pinned global: the silence family + pin-bypass. Encodes the
    /// BookForge dogfood label (the colored-3MF global is verbatim the
    /// labeled case).
    Bookforge,
    /// Same as [`StoreFixture::Bookforge`] minus every project memory: a
    /// project with no signal of its own admits nothing at all.
    BookforgeEmpty,
    /// Global scope, one strong pinned guard over a weak tail: the
    /// rotation-to-weak-tail family.
    WeakTail,
    /// A project with more real guards than any cap can hold: the
    /// over-budget truncation-order family and the backbone of the
    /// marginal-catch curve. Includes one guard the user hand-promoted
    /// into the rules pack, so the rules-baseline comparison (§9 gate 1)
    /// has a case the baseline genuinely catches.
    Budget,
}

/// One labeled brief scenario (§9's fixture shape). `expected_ids` /
/// `forbidden_ids` are fixture keys resolved against the built store;
/// `exclude` simulates the session-suppression state (`brief_shown:*`)
/// the CLI would pass in.
pub struct BriefQuestion {
    pub name: &'static str,
    pub family: &'static str,
    pub store_fixture: StoreFixture,
    /// Fixture keys already shown this session (the caller-side exclude
    /// set `brief_cmd::suppressed` would produce).
    pub exclude: &'static [&'static str],
    /// Keys the composed brief must render (at a sufficient budget; the
    /// curve measures how many survive each cap). Empty = the correct
    /// brief is silence.
    pub expected_ids: &'static [&'static str],
    /// When set, the FIRST rendered line must be this key at every cap —
    /// the truncation-order / crowd-out assertion.
    pub expected_first: Option<&'static str>,
    /// Keys that must never be selected, let alone rendered, at any cap.
    pub forbidden_ids: &'static [&'static str],
}

pub const BRIEF_QUESTIONS: &[BriefQuestion] = &[
    // ARIA startup: the project's own guards, canonical gotcha first;
    // superseded/deleted/rules-dup/gated/floored candidates never appear.
    BriefQuestion {
        name: "aria-startup",
        family: "crowd-out+fresh-capture+must-not-surface",
        store_fixture: StoreFixture::Aria,
        exclude: &[],
        expected_ids: &["aria_checkout", "aria_fresh", "aria_new"],
        expected_first: Some("aria_checkout"),
        forbidden_ids: &[
            "aria_old",
            "aria_deleted",
            "aria_rules_dup",
            "glob_yield",
            "glob_far",
        ],
    },
    // ARIA with the top guard already shown: comparable peers still
    // render (the floor is measured pre-suppression, so it cannot exile
    // them), but the shown guard must not repeat.
    BriefQuestion {
        name: "aria-already-shown",
        family: "must-not-surface(already-shown)",
        store_fixture: StoreFixture::Aria,
        exclude: &["aria_checkout"],
        expected_ids: &["aria_fresh", "aria_new"],
        expected_first: None,
        forbidden_ids: &[
            "aria_checkout",
            "aria_old",
            "aria_deleted",
            "aria_rules_dup",
            "glob_yield",
            "glob_far",
        ],
    },
    // BookForge with its own footprint: own guard plus the pinned global
    // (pin bypasses the gate); the labeled-irrelevant globals stay out.
    BriefQuestion {
        name: "bookforge-own",
        family: "pinned-must-include+silence",
        store_fixture: StoreFixture::Bookforge,
        exclude: &[],
        expected_ids: &["bf_own", "bf_pin_force_push"],
        expected_first: None,
        forbidden_ids: &["bf_glob_3mf", "bf_glob_nvenc"],
    },
    // The literal BookForge dogfood label: no signal of its own ⇒ the
    // right brief is NOTHING, not other projects' gotchas as filler.
    BriefQuestion {
        name: "bookforge-empty-silence",
        family: "silence",
        store_fixture: StoreFixture::BookforgeEmpty,
        exclude: &[],
        expected_ids: &[],
        expected_first: None,
        forbidden_ids: &["bf_glob_3mf", "bf_glob_nvenc"],
    },
    // Fresh session over the weak-tail store: the pinned guard renders,
    // the tail stays below the floor even with nothing suppressed.
    BriefQuestion {
        name: "weak-tail-fresh-session",
        family: "pinned-must-include",
        store_fixture: StoreFixture::WeakTail,
        exclude: &[],
        expected_ids: &["wt_pinned"],
        expected_first: Some("wt_pinned"),
        forbidden_ids: &["wt_weak_a", "wt_weak_b"],
    },
    // The rotation dogfood label: with the strong guard suppressed, the
    // weak tail must NOT be served as "best available" — silence wins.
    BriefQuestion {
        name: "weak-tail-suppressed",
        family: "rotation-to-weak-tail",
        store_fixture: StoreFixture::WeakTail,
        exclude: &["wt_pinned"],
        expected_ids: &[],
        expected_first: None,
        forbidden_ids: &["wt_pinned", "wt_weak_a", "wt_weak_b"],
    },
    // Over-budget: five labeled-vital guards against caps that cannot
    // hold them all. `bud_tabs` is ALSO verbatim in the rules pack, so
    // the brief must dedup it (a permanent, honest catch miss for the
    // brief — and the one case the rules baseline genuinely catches).
    BriefQuestion {
        name: "budget-truncation",
        family: "over-budget-truncation",
        store_fixture: StoreFixture::Budget,
        exclude: &[],
        expected_ids: &["bud_v1", "bud_v2", "bud_v3", "bud_v4", "bud_tabs"],
        expected_first: Some("bud_v1"),
        forbidden_ids: &[],
    },
];

/// A built store fixture, ready to drive `brief::select`.
pub struct BuiltStore {
    pub conn: Connection,
    pub ids: HashMap<&'static str, i64>,
    pub scope: Scope,
    pub rules_body: Option<&'static str>,
}

fn capture(
    conn: &Connection,
    ids: &mut HashMap<&'static str, i64>,
    key: &'static str,
    text: &str,
    mtype: MemoryType,
    project_id: Option<i64>,
) -> Result<i64> {
    let node = match memory::remember(
        conn,
        Remember {
            text: text.into(),
            mtype,
            tags: vec![],
            project_id,
            force: true,
            actor: crate::model::Actor::System,
        },
    )? {
        RememberOutcome::Created(n) => n,
        other => panic!("fixture capture must create, got {other:?}"),
    };
    ids.insert(key, node.id);
    Ok(node.id)
}

/// Learned-usage strength, set directly: the dogfood stores' guard
/// orderings come from accumulated `mark`/open signals, which fixtures
/// model as a strength value rather than replaying N mark calls.
fn set_strength(conn: &Connection, id: i64, strength: f64) -> Result<()> {
    conn.execute(
        "UPDATE node SET strength = ?2 WHERE id = ?1",
        params![id, strength],
    )?;
    Ok(())
}

fn backdate(conn: &Connection, id: i64, days: i64, now: i64) -> Result<()> {
    let ts = now - days * 86_400;
    conn.execute(
        "UPDATE node SET created_at = ?2, updated_at = ?2 WHERE id = ?1",
        params![id, ts],
    )?;
    Ok(())
}

/// The ARIA rules pack. Contains `aria_rules_dup`'s body verbatim (the
/// user hand-copied that guard into the pack), so the brief must dedup it.
const ARIA_RULES: &str = "ARIA project rules:\n\
     - Use the workspace prettier config; never a per-package one.\n\
     - All API handlers must validate the session token before touching the db.\n\
     - Branch names use the aria/<ticket> prefix convention for CI routing.";

/// The Budget-store rules pack: `bud_tabs`'s body verbatim.
const BUDGET_RULES: &str = "Atlas conventions:\n\
     - Indentation is tabs in Go files and four spaces everywhere else; gofmt is the arbiter of layout.\n\
     - PR titles follow conventional commits.";

impl StoreFixture {
    /// Build this store shape in a fresh in-memory database. `now` is the
    /// eval clock — creation times are backdated relative to it so decay
    /// behaves like a store that has lived for a few weeks.
    pub fn build(self, now: i64) -> Result<BuiltStore> {
        let conn = db::open_in_memory()?;
        let mut ids = HashMap::new();
        let (scope, rules_body) = match self {
            StoreFixture::Aria => {
                let proj = store::ensure_project(&conn, "/eval/aria", "aria")?;
                let checkout = capture(
                    &conn,
                    &mut ids,
                    "aria_checkout",
                    "Always clone through the canonical checkout script; a bare git clone skips \
                     the submodule pinning and LFS hooks.",
                    MemoryType::Gotcha,
                    Some(proj.id),
                )?;
                // The project's strongest guard by accumulated usage —
                // the ARIA dogfood label says this must rank first.
                set_strength(&conn, checkout, 3.0)?;
                backdate(&conn, checkout, 40, now)?;
                capture(
                    &conn,
                    &mut ids,
                    "aria_fresh",
                    "The staging S3 bucket rotates credentials nightly at 03:00 UTC; cached \
                     presigned URLs go stale after rotation.",
                    MemoryType::Gotcha,
                    Some(proj.id),
                )?;
                let new = capture(
                    &conn,
                    &mut ids,
                    "aria_new",
                    "Retry policy: exponential backoff with jitter, max five attempts, then \
                     dead-letter — replaces the old fixed-delay loop.",
                    MemoryType::Decision,
                    Some(proj.id),
                )?;
                set_strength(&conn, new, 1.6)?;
                backdate(&conn, new, 30, now)?;
                let old = capture(
                    &conn,
                    &mut ids,
                    "aria_old",
                    "Retry policy: fixed 2 second delay between attempts, retry forever until \
                     the operation succeeds.",
                    MemoryType::Decision,
                    Some(proj.id),
                )?;
                backdate(&conn, old, 60, now)?;
                store::set_superseded(&conn, old, new, crate::model::Actor::System, None)?;
                let deleted = capture(
                    &conn,
                    &mut ids,
                    "aria_deleted",
                    "Old CI runner leaks docker volumes; prune them manually after each release \
                     build finishes.",
                    MemoryType::Gotcha,
                    Some(proj.id),
                )?;
                store::soft_delete(&conn, deleted, crate::model::Actor::System, None)?;
                let dup = capture(
                    &conn,
                    &mut ids,
                    "aria_rules_dup",
                    "All API handlers must validate the session token before touching the db.",
                    MemoryType::Gotcha,
                    Some(proj.id),
                )?;
                backdate(&conn, dup, 20, now)?;
                // Shares "checkout" with the project signature: admitted
                // by the lexical gate, but an unpinned global yields to a
                // project with guards of its own (score floor) — the
                // "unpinned globals yield" dogfood decision.
                capture(
                    &conn,
                    &mut ids,
                    "glob_yield",
                    "A stale checkout of a pinned toolchain silently builds against the wrong \
                     std; rerun the pin script after rebasing.",
                    MemoryType::Gotcha,
                    None,
                )?;
                // No signature overlap at all: the lexical gate drops it.
                capture(
                    &conn,
                    &mut ids,
                    "glob_far",
                    "Kubernetes ingress rewrites strip the auth header unless the rewrite \
                     annotation is set explicitly.",
                    MemoryType::Gotcha,
                    None,
                )?;
                (Scope::Project(proj.id), Some(ARIA_RULES))
            }
            StoreFixture::Bookforge | StoreFixture::BookforgeEmpty => {
                let proj = store::ensure_project(&conn, "/eval/bookforge", "bookforge")?;
                if self == StoreFixture::Bookforge {
                    let own = capture(
                        &conn,
                        &mut ids,
                        "bf_own",
                        "Chapter typesetting breaks on long epigraphs; wrap them in the quote \
                         environment before the layout pass.",
                        MemoryType::Gotcha,
                        Some(proj.id),
                    )?;
                    set_strength(&conn, own, 3.0)?;
                }
                let pinned = capture(
                    &conn,
                    &mut ids,
                    "bf_pin_force_push",
                    "Never force-push a shared branch; rewriting published history strands \
                     every checked-out clone.",
                    MemoryType::Gotcha,
                    None,
                )?;
                store::set_pinned(&conn, pinned, true, crate::model::Actor::System, None)?;
                backdate(&conn, pinned, 60, now)?;
                // Verbatim the BookForge dogfood label: this global (from
                // another project's world) must never be BookForge filler.
                capture(
                    &conn,
                    &mut ids,
                    "bf_glob_3mf",
                    "Colored 3MF exports from the python mesher drop vertex colors unless the \
                     slicer profile enables painting.",
                    MemoryType::Gotcha,
                    None,
                )?;
                capture(
                    &conn,
                    &mut ids,
                    "bf_glob_nvenc",
                    "NVENC sessions leak on consumer GPUs when the encoder is reopened without \
                     destroying the old context.",
                    MemoryType::Gotcha,
                    None,
                )?;
                let scope = Scope::Project(proj.id);
                if self == StoreFixture::BookforgeEmpty {
                    // A pinned global still surfaces on an empty project
                    // (pin bypasses the gate) — remove it so this store
                    // isolates the "no signal ⇒ no globals" label.
                    let id = ids["bf_pin_force_push"];
                    store::soft_delete(&conn, id, crate::model::Actor::System, None)?;
                    ids.remove("bf_pin_force_push");
                }
                (scope, None)
            }
            StoreFixture::WeakTail => {
                let pinned = capture(
                    &conn,
                    &mut ids,
                    "wt_pinned",
                    "Database migrations must run inside a transaction; a half-applied \
                     migration bricks the deploy.",
                    MemoryType::Gotcha,
                    None,
                )?;
                store::set_pinned(&conn, pinned, true, crate::model::Actor::System, None)?;
                set_strength(&conn, pinned, 2.0)?;
                capture(
                    &conn,
                    &mut ids,
                    "wt_weak_a",
                    "The vendored spellcheck wordlist is sorted case-sensitively; resorting it \
                     with plain sort breaks the diff.",
                    MemoryType::Gotcha,
                    None,
                )?;
                capture(
                    &conn,
                    &mut ids,
                    "wt_weak_b",
                    "Sample fixtures in the demo folder use CRLF endings on purpose; the \
                     import test depends on them staying CRLF.",
                    MemoryType::Gotcha,
                    None,
                )?;
                (Scope::Global, None)
            }
            StoreFixture::Budget => {
                let proj = store::ensure_project(&conn, "/eval/atlas", "atlas")?;
                let vitals: [(&'static str, f64, &str); 4] = [
                    (
                        "bud_v1",
                        3.0,
                        "The payments sandbox shares rate limits with production; a runaway \
                         test loop locks out real checkouts.",
                    ),
                    (
                        "bud_v2",
                        2.6,
                        "Schema migrations must ship one release before the code that needs \
                         them; rollback has to stay possible.",
                    ),
                    (
                        "bud_v3",
                        2.2,
                        "The feature-flag service caches for five minutes; toggling a flag \
                         during an incident needs a cache purge.",
                    ),
                    (
                        "bud_v4",
                        1.8,
                        "Load tests against the shared cluster must be announced in ops chat \
                         first; autoscaling bills are per-minute.",
                    ),
                ];
                for (key, strength, text) in vitals {
                    let id = capture(
                        &conn,
                        &mut ids,
                        key,
                        text,
                        MemoryType::Gotcha,
                        Some(proj.id),
                    )?;
                    set_strength(&conn, id, strength)?;
                }
                let tabs = capture(
                    &conn,
                    &mut ids,
                    "bud_tabs",
                    "Indentation is tabs in Go files and four spaces everywhere else; gofmt is \
                     the arbiter of layout.",
                    MemoryType::Gotcha,
                    Some(proj.id),
                )?;
                set_strength(&conn, tabs, 2.0)?;
                for (key, text) in [
                    (
                        "bud_fill_a",
                        "Local dev seeds twenty demo users; the admin one is always \
                         demo-admin@example.test with password in .env.sample.",
                    ),
                    (
                        "bud_fill_b",
                        "The changelog generator reads squashed commit subjects only; bodies \
                         are ignored when drafting release notes.",
                    ),
                    (
                        "bud_fill_c",
                        "Grafana dashboards live in the ops repo, not this one; edits made in \
                         the UI are overwritten on deploy.",
                    ),
                ] {
                    capture(
                        &conn,
                        &mut ids,
                        key,
                        text,
                        MemoryType::Gotcha,
                        Some(proj.id),
                    )?;
                }
                (Scope::Project(proj.id), Some(BUDGET_RULES))
            }
        };
        Ok(BuiltStore {
            conn,
            ids,
            scope,
            rules_body,
        })
    }
}

/// One question's run at one cap: which fixture keys were selected
/// (post-gate/floor/exclusion, pre-budget) and which were rendered.
pub struct QuestionOutcome {
    pub selected: Vec<&'static str>,
    pub shown: Vec<&'static str>,
    pub text: String,
}

fn keys_for(ids: &HashMap<&'static str, i64>, node_ids: &[i64]) -> Vec<&'static str> {
    node_ids
        .iter()
        .map(|id| {
            ids.iter()
                .find(|(_, v)| *v == id)
                .map(|(k, _)| *k)
                .unwrap_or("?")
        })
        .collect()
}

/// Drive the production selection + composition path for one question at
/// one budget cap. Identical wiring to `brief_cmd::show`'s startup fire:
/// config-default scoring/brief knobs, rules-pack dedup, caller exclusions.
pub fn run_question(
    built: &BuiltStore,
    q: &BriefQuestion,
    cap: usize,
    now: i64,
) -> Result<QuestionOutcome> {
    let config = Config::default();
    let exclude: HashSet<i64> = q.exclude.iter().map(|k| built.ids[k]).collect();
    let ranked = brief::select(
        &built.conn,
        built.scope,
        &exclude,
        built.rules_body,
        &config,
        now,
    )?;
    let selected_ids: Vec<i64> = ranked.iter().map(|r| r.node.id).collect();
    let composed = brief::compose(&ranked, &config.brief, cap, now);
    Ok(QuestionOutcome {
        selected: keys_for(&built.ids, &selected_ids),
        shown: keys_for(&built.ids, &composed.shown),
        text: composed.text,
    })
}

/// catch@cap for one question: the fraction of `expected_ids` rendered.
/// `None` when the question labels silence (nothing expected) — those
/// questions are scored by the forbidden assertion instead, so they don't
/// inflate the catch mean.
pub fn catch_rate(q: &BriefQuestion, outcome: &QuestionOutcome) -> Option<f64> {
    if q.expected_ids.is_empty() {
        return None;
    }
    let shown: HashSet<&str> = outcome.shown.iter().copied().collect();
    let hits = q
        .expected_ids
        .iter()
        .filter(|k| shown.contains(**k))
        .count();
    Some(hits as f64 / q.expected_ids.len() as f64)
}

/// §9 gate 1 baseline: what `mimir rules show` alone surfaces — the
/// fraction of this question's expected guards whose content the store's
/// rules pack already carries (same containment test the brief's dedup
/// uses, so the two channels are compared on identical terms).
pub fn rules_baseline_catch(built: &BuiltStore, q: &BriefQuestion) -> Result<Option<f64>> {
    if q.expected_ids.is_empty() {
        return Ok(None);
    }
    let Some(rules) = built.rules_body else {
        return Ok(Some(0.0));
    };
    let rules_norm = brief::normalize(rules);
    let mut hits = 0;
    for key in q.expected_ids {
        let node = store::get_node(&built.conn, built.ids[key])?;
        if brief::duplicates_rules(&node, &rules_norm) {
            hits += 1;
        }
    }
    Ok(Some(hits as f64 / q.expected_ids.len() as f64))
}

/// Mean catch across the catch-scored questions at one cap, plus the
/// per-question outcomes for the gates that inspect every fixture.
pub struct CapRun {
    pub cap: usize,
    pub mean_catch: f64,
    pub outcomes: Vec<QuestionOutcome>,
}

pub fn run_at_cap(cap: usize, now: i64) -> Result<CapRun> {
    let mut catches = Vec::new();
    let mut outcomes = Vec::new();
    for q in BRIEF_QUESTIONS {
        let built = q.store_fixture.build(now)?;
        let outcome = run_question(&built, q, cap, now)?;
        if let Some(c) = catch_rate(q, &outcome) {
            catches.push(c);
        }
        outcomes.push(outcome);
    }
    let mean_catch = catches.iter().sum::<f64>() / catches.len() as f64;
    Ok(CapRun {
        cap,
        mean_catch,
        outcomes,
    })
}

/// §9 gate 4: walk the curve from the smallest cap; a bigger cap wins only
/// if the step buys at least [`MIN_PTS_PER_100_TOKENS`] catch points per
/// 100 tokens (pro-rata for the uneven steps). Returns the winning cap.
pub fn calibrated_cap(curve: &[CapRun]) -> usize {
    let mut winner = &curve[0];
    for candidate in &curve[1..] {
        let step_tokens = (candidate.cap - winner.cap) as f64;
        let gain_pts = (candidate.mean_catch - winner.mean_catch) * 100.0;
        if gain_pts >= MIN_PTS_PER_100_TOKENS * step_tokens / 100.0 {
            winner = candidate;
        }
    }
    winner.cap
}

/// One simulated session fire in the repeated-exposure replay.
pub struct ReplayFire {
    pub label: &'static str,
    /// Simulated wall-clock offset from the replay start, seconds.
    pub at_offset: i64,
    pub session: &'static str,
    pub recap: bool,
    pub shown: Vec<&'static str>,
    pub text: String,
}

/// §9 repeated-exposure family, selection half: replay a fixed fire
/// schedule over the ARIA store, mirroring `brief_cmd::suppressed`'s
/// exclusion semantics exactly (startup: own session's shows plus any
/// session's shows inside the 6 h wall-clock window; recap: only OTHER
/// sessions' recent shows). Assertions live in the tests; the report
/// prints each rendered brief for the agent-compliance half.
pub fn replay_sessions(now: i64) -> Result<Vec<ReplayFire>> {
    const RECENT_SHOWN_SECS: i64 = 6 * 3600;
    let config = Config::default();
    let built = StoreFixture::Aria.build(now)?;
    // (session, node_id, shown_at) — the session_state log this replay
    // stands in for.
    let mut log: Vec<(&'static str, i64, i64)> = Vec::new();
    let schedule: [(&'static str, i64, &'static str, bool); 4] = [
        ("s1-startup", 0, "s1", false),
        ("s1-clear-recap", 3600, "s1", true),
        ("s2-startup-within-window", 2 * 3600, "s2", false),
        ("s3-startup-next-day", 9 * 3600, "s3", false),
    ];
    let mut fires = Vec::new();
    for (label, at_offset, session, recap) in schedule {
        let t = now + at_offset;
        let exclude: HashSet<i64> = log
            .iter()
            .filter(|(s, _, shown_at)| {
                let recent = *shown_at > t - RECENT_SHOWN_SECS;
                if recap {
                    *s != session && recent
                } else {
                    *s == session || recent
                }
            })
            .map(|(_, id, _)| *id)
            .collect();
        let ranked = brief::select(
            &built.conn,
            built.scope,
            &exclude,
            built.rules_body,
            &config,
            t,
        )?;
        let budget = if recap {
            config.brief.recap_tokens
        } else {
            config.brief.max_tokens
        };
        let composed = brief::compose(&ranked, &config.brief, budget, t);
        for id in &composed.shown {
            log.push((session, *id, t));
        }
        fires.push(ReplayFire {
            label,
            at_offset,
            session,
            recap,
            shown: keys_for(&built.ids, &composed.shown),
            text: composed.text,
        });
    }
    Ok(fires)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §9 gate 2: zero `forbidden_ids` ever rendered — asserted at the
    /// stronger selection level too (every forbidden class is a
    /// select-stage filter, so a forbidden key surviving selection is a
    /// bug even if the budget happens to hide it), at every curve cap.
    #[test]
    fn gate2_forbidden_never_selected_or_rendered() {
        let now = now_unix();
        for cap in CURVE_CAPS {
            for q in BRIEF_QUESTIONS {
                let built = q.store_fixture.build(now).unwrap();
                let outcome = run_question(&built, q, cap, now).unwrap();
                for forbidden in q.forbidden_ids {
                    assert!(
                        !outcome.selected.contains(forbidden),
                        "[{}@{cap}] forbidden '{forbidden}' survived selection",
                        q.name
                    );
                    assert!(
                        !outcome.shown.contains(forbidden),
                        "[{}@{cap}] forbidden '{forbidden}' was rendered",
                        q.name
                    );
                }
            }
        }
    }

    /// §9 gate 3: budget adherence at every fixture × every cap, measured
    /// on the rendered text with the production token counter.
    #[test]
    fn gate3_budget_adherence_at_every_fixture() {
        let now = now_unix();
        for cap in CURVE_CAPS {
            let run = run_at_cap(cap, now).unwrap();
            for (q, outcome) in BRIEF_QUESTIONS.iter().zip(&run.outcomes) {
                assert!(
                    tokens::count(&outcome.text) <= cap,
                    "[{}@{cap}] rendered {} tokens",
                    q.name,
                    tokens::count(&outcome.text)
                );
            }
        }
    }

    /// §9 gate 1: brief catch at the default budget beats the rules-pack
    /// baseline on the same fixtures. The margin is structural — rules
    /// packs are hand-maintained, so auto-captured guards are exactly
    /// what they miss — and that structural gap is the feature's reason
    /// to exist; the one hand-promoted guard (`bud_tabs`) keeps the
    /// baseline from being a strawman zero.
    #[test]
    fn gate1_brief_beats_rules_pack_baseline() {
        let now = now_unix();
        let run = run_at_cap(BriefConfig::default().max_tokens, now).unwrap();
        let mut baseline = Vec::new();
        for q in BRIEF_QUESTIONS {
            let built = q.store_fixture.build(now).unwrap();
            if let Some(b) = rules_baseline_catch(&built, q).unwrap() {
                baseline.push(b);
            }
        }
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        assert!(
            baseline_mean > 0.0,
            "baseline must not be a strawman zero — bud_tabs should be rules-covered"
        );
        assert!(
            run.mean_catch > baseline_mean,
            "brief catch {:.3} must beat rules baseline {:.3}",
            run.mean_catch,
            baseline_mean
        );
    }

    /// §9 gate 4: the marginal-catch curve calibrates the default cap.
    /// The curve must be monotone non-decreasing (more budget can only
    /// render more), and the pro-rata ≥3pts/100tok rule must pick exactly
    /// the shipped default (150).
    #[test]
    fn gate4_marginal_catch_curve_calibrates_default_cap() {
        let now = now_unix();
        let curve: Vec<CapRun> = CURVE_CAPS
            .iter()
            .map(|&cap| run_at_cap(cap, now).unwrap())
            .collect();
        for pair in curve.windows(2) {
            assert!(
                pair[1].mean_catch >= pair[0].mean_catch - 1e-9,
                "catch must be monotone in cap: {} pts at {} vs {} pts at {}",
                pair[0].mean_catch * 100.0,
                pair[0].cap,
                pair[1].mean_catch * 100.0,
                pair[1].cap
            );
        }
        assert_eq!(
            calibrated_cap(&curve),
            BriefConfig::default().max_tokens,
            "marginal-catch curve no longer calibrates the shipped default; \
             recalibrate [brief] max_tokens or re-adjudicate the corpus \
             (curve: {:?})",
            curve
                .iter()
                .map(|r| (r.cap, (r.mean_catch * 100.0).round()))
                .collect::<Vec<_>>()
        );
    }

    /// Truncation order / crowd-out: where a question labels a first
    /// guard, the first rendered line is that guard at every cap — the
    /// ARIA "canonical gotcha first" label and the Budget ordering.
    #[test]
    fn expected_first_holds_at_every_cap() {
        let now = now_unix();
        for cap in CURVE_CAPS {
            for q in BRIEF_QUESTIONS {
                let Some(first) = q.expected_first else {
                    continue;
                };
                let built = q.store_fixture.build(now).unwrap();
                let outcome = run_question(&built, q, cap, now).unwrap();
                assert_eq!(
                    outcome.shown.first().copied(),
                    Some(first),
                    "[{}@{cap}] first rendered guard",
                    q.name
                );
            }
        }
    }

    /// Fresh-capture family: yesterday's capture is in today's startup
    /// brief at the default cap alongside the older stock.
    #[test]
    fn fresh_capture_is_included_at_default_cap() {
        let now = now_unix();
        let built = StoreFixture::Aria.build(now).unwrap();
        let q = &BRIEF_QUESTIONS[0];
        let outcome = run_question(&built, q, BriefConfig::default().max_tokens, now).unwrap();
        assert!(
            outcome.shown.contains(&"aria_fresh"),
            "fresh capture missing from startup brief: {:?}",
            outcome.shown
        );
    }

    /// §9 repeated-exposure, selection half: across the replay schedule
    /// the top guard keeps its seat instead of rotating away, the recap
    /// re-anchors it after a clear, and a sibling session inside the
    /// suppression window gets silence rather than the weak tail.
    #[test]
    fn repeated_exposure_replay_holds_selection() {
        let now = now_unix();
        let fires = replay_sessions(now).unwrap();
        let by_label: HashMap<&str, &ReplayFire> = fires.iter().map(|f| (f.label, f)).collect();

        let s1 = by_label["s1-startup"];
        assert_eq!(s1.shown.first().copied(), Some("aria_checkout"));

        // Recap re-anchors THIS session's top guard after the context wipe
        // (the old rotate-to-unseen behavior is the labeled failure).
        let recap = by_label["s1-clear-recap"];
        assert_eq!(
            recap.shown.first().copied(),
            Some("aria_checkout"),
            "recap must re-anchor the top guard, not rotate to the unseen tail"
        );

        // A sibling session inside the 6 h window: everything strong was
        // just shown; silence beats re-serving or weak-tail filler.
        let s2 = by_label["s2-startup-within-window"];
        assert!(
            s2.shown.is_empty(),
            "within-window sibling session must stay silent, got {:?}",
            s2.shown
        );

        // Outside the window: the top guard is back, first — repeated
        // exposure must not degrade selection of the #1 guard.
        let s3 = by_label["s3-startup-next-day"];
        assert_eq!(
            s3.shown.first().copied(),
            Some("aria_checkout"),
            "top guard must keep surfacing across sessions"
        );
    }

    /// The citable numbers (§9). Prints the marginal-catch curve, the
    /// rules-baseline comparison, per-question outcomes at the default
    /// cap, and every replay brief verbatim (the corpus for the
    /// agent-compliance half of the repeated-exposure family). Run:
    /// `cargo test -p mimir-mem-core eval::brief -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn eval_brief_report() {
        let now = now_unix();
        println!("\nsession-brief drift eval — §9 report");
        println!("(scope: hook-running clients only; not a general session-drift claim)");

        let curve: Vec<CapRun> = CURVE_CAPS
            .iter()
            .map(|&cap| run_at_cap(cap, now).unwrap())
            .collect();
        println!("\nmarginal-catch curve (mean catch across labeled questions):");
        for run in &curve {
            println!(
                "  cap {:>4} tokens  catch {:>5.1} pts",
                run.cap,
                run.mean_catch * 100.0
            );
        }
        println!(
            "  calibrated cap (≥{MIN_PTS_PER_100_TOKENS} pts per +100 tokens): {} \
             (shipped default: {})",
            calibrated_cap(&curve),
            BriefConfig::default().max_tokens
        );

        let default_cap = BriefConfig::default().max_tokens;
        println!("\nper-question at default cap ({default_cap} tokens):");
        for q in BRIEF_QUESTIONS {
            let built = q.store_fixture.build(now).unwrap();
            let outcome = run_question(&built, q, default_cap, now).unwrap();
            let catch = catch_rate(q, &outcome)
                .map(|c| format!("{:>5.1} pts", c * 100.0))
                .unwrap_or_else(|| "  (silence-labeled)".into());
            let baseline = rules_baseline_catch(&built, q)
                .unwrap()
                .map(|b| format!("{:>5.1} pts", b * 100.0))
                .unwrap_or_else(|| "      —".into());
            println!(
                "  {:<28} catch {catch}  rules-baseline {baseline}  shown: {:?}",
                q.name, outcome.shown
            );
        }

        println!("\nrepeated-exposure replay (compliance-labeling corpus):");
        for fire in replay_sessions(now).unwrap() {
            println!(
                "\n--- {} (session {}, +{}h, {}) ---",
                fire.label,
                fire.session,
                fire.at_offset / 3600,
                if fire.recap { "recap" } else { "startup" }
            );
            if fire.text.is_empty() {
                println!("(silent)");
            } else {
                println!("{}", fire.text);
            }
        }
        println!(
            "\nNOTE: the agent-compliance half (boilerplate-blindness by session N) \
             cannot be asserted here — label it against the briefs above with a real \
             agent (dogfood corpus)."
        );
    }
}
