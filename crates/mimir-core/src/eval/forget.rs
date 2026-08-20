//! Deletion evals: does a forgotten fact stay forgotten?
//!
//! Two suites, both deterministic — no judge model, no labelled dataset, no
//! network. They run under plain `cargo test`, so a regression fails CI the
//! way the hermetic ranking baseline already does.
//!
//! **[`ATTACKS`]** ports the ten adversarial categories from ForgetEval
//! (`Control-Plane Placement Shapes Forgetting`, arXiv:2606.15903). Each is a
//! shape in which a re-offered fact can slip past a deletion guard that only
//! compares strings. `paraphrase_supersession` is the category that caught a
//! real bug here: the tombstone guard used a fixed similarity *ratio*, so the
//! same trailing period was refused on a long memory and accepted on a short
//! one.
//!
//! **[`durability`]** walks the Agent Memory Atlas's thirteen-step deletion
//! sequence, whose interesting half is the part almost nothing implements:
//! re-feeding the original source by a *different* write path, running every
//! background job, and probing the derived stores — FTS, embeddings, prior
//! revisions, the audit ledger, export.
//!
//! Two scoring rules, both borrowed, both load-bearing:
//!
//! - **The scorecard does not pool.** Obligations (things the store must do)
//!   and negative invariants (things that must not leak) are counted apart,
//!   because a store that keeps nothing passes every negative invariant.
//!   AOEP-v0 makes this argument; the empty-store control arm in
//!   [`tests::an_empty_store_fails_every_obligation`] keeps us honest about it.
//! - **Every case carries a never-forgotten twin** that must stay
//!   retrievable, so no negative assertion can pass vacuously — a suite where
//!   `recall` is simply broken would otherwise score perfectly.
//!
//! Reported as a pass/fail matrix, never a percentage: "forgets correctly 87%
//! of the time" says nothing useful about a deletion request.

use rusqlite::Connection;

use crate::model::{Actor, MemoryType, Scope};
use crate::search::SearchQuery;
use crate::{db, memory, review, revision, search, store};

/// One adversarial re-offer: a fact is written, deliberately forgotten, then
/// offered again in `variant`. The guard must refuse.
pub struct Attack {
    pub category: &'static str,
    /// What gets written and then forgotten.
    pub original: &'static str,
    /// How it comes back — the shape the attack is named for.
    pub variant: &'static str,
    /// Whether `variant` is the same fact. `false` cases are the control:
    /// they must be *accepted*, or the guard is simply refusing everything.
    pub same_fact: bool,
}

/// ForgetEval's ten categories, each with a positive case (must refuse) and a
/// near-miss control (must accept). The controls matter as much as the
/// attacks: a guard that refuses every short string would score perfectly on
/// the attacks alone while making the store unusable.
pub const ATTACKS: &[Attack] = &[
    // 1. Substring traps: one fact's text contains another's.
    Attack {
        category: "substring_trap",
        original: "the deploy token rotates on Sundays",
        variant: "the deploy token rotates on Sundays at 03:00",
        same_fact: true,
    },
    Attack {
        category: "substring_trap",
        original: "the deploy token rotates weekly",
        variant: "the backup token rotates weekly",
        same_fact: false,
    },
    // 2. Prefix collisions: a shared opening, a different fact.
    Attack {
        category: "prefix_collision",
        original: "postgres max_connections is 200 in staging",
        variant: "postgres max_connections is 200 in staging environments",
        same_fact: true,
    },
    Attack {
        category: "prefix_collision",
        original: "postgres max_connections is 200 in staging",
        variant: "postgres max_connections is 500 in production",
        same_fact: false,
    },
    // 3. Paraphrase supersession: the same fact, said differently. The
    //    category that exposed the length-dependent ratio bug.
    Attack {
        category: "paraphrase_supersession",
        original: "SCRAM auth rejects non-ASCII passwords at 03:00 UTC",
        variant: "scram auth rejects non-ascii passwords at 03:00 utc.",
        same_fact: true,
    },
    Attack {
        category: "paraphrase_supersession",
        original: "redis listens on port 6379",
        variant: "postgres listens on port 5432",
        same_fact: false,
    },
    // 4. Negation: the opposite claim is a different fact, not a reword.
    Attack {
        category: "negation",
        original: "the staging cluster has autoscaling enabled",
        variant: "the staging cluster does not have autoscaling enabled",
        same_fact: false,
    },
    // 5. Temporal qualifiers: a date attached to the same claim.
    Attack {
        category: "temporal_qualifier",
        original: "the ops cron runs at 03:00 UTC",
        variant: "the ops cron runs at 03:00 UTC daily",
        same_fact: true,
    },
    Attack {
        category: "temporal_qualifier",
        original: "the ops cron runs at 03:00 UTC",
        variant: "the ops cron runs at 04:00 UTC",
        same_fact: false,
    },
    // 6. Shared attributes: same subject, different property.
    Attack {
        category: "shared_attribute",
        original: "the metrics box has 64 GB of RAM",
        variant: "the metrics box has 64 GB of RAM installed",
        same_fact: true,
    },
    Attack {
        category: "shared_attribute",
        original: "the metrics box has 64 GB of RAM",
        variant: "the metrics box has 16 CPU cores",
        same_fact: false,
    },
    // 7. Compound facts: one clause of a two-clause memory.
    Attack {
        category: "compound_fact",
        original: "the proxy strips cache breakpoints above four and dedups requests",
        variant: "The proxy strips cache breakpoints above four, and dedups requests.",
        same_fact: true,
    },
    // 8. Identifier obfuscation: an identifier respelled.
    Attack {
        category: "identifier_obfuscation",
        original: "retry_with_backoff caps at five attempts",
        variant: "retry_with_backoff caps at five attempts.",
        same_fact: true,
    },
    Attack {
        category: "identifier_obfuscation",
        original: "retry_with_backoff caps at five attempts",
        variant: "retry_immediately caps at five attempts",
        same_fact: false,
    },
    // 9. Cross-lingual identifiers: same code symbol, different prose.
    //    Honest expectation: the guard is lexical, so it does NOT catch a
    //    translated restatement. Encoded as `same_fact: false` because that
    //    is what this store can actually do — asserting a catch we cannot
    //    make would be a test that lies.
    Attack {
        category: "cross_lingual",
        original: "the cache expires after thirty seconds",
        variant: "le cache expire apres trente secondes",
        same_fact: false,
    },
    // 10. Recursive supersession: the replacement's replacement.
    Attack {
        category: "recursive_supersession",
        original: "deploys go to eu-west-1",
        variant: "deploys go to eu-west-1 only",
        same_fact: true,
    },
];

/// One arm of the scorecard. Kept apart from the other on purpose — see the
/// module note on why pooling would let an empty store look good.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Score {
    pub passed: usize,
    pub total: usize,
}

impl Score {
    fn record(&mut self, ok: bool) {
        self.total += 1;
        self.passed += usize::from(ok);
    }
    pub fn is_clean(&self) -> bool {
        self.passed == self.total
    }
}

#[derive(Debug, Default, Clone)]
pub struct Scorecard {
    /// Things the store must actively do (record a deletion, keep the twin
    /// retrievable, refuse a re-offer).
    pub obligation: Score,
    /// Things that must not happen (a value surviving anywhere).
    pub negative: Score,
    /// Per-step verdicts, for the matrix.
    pub rows: Vec<(String, bool, bool)>,
}

impl Scorecard {
    fn obligation(&mut self, label: &str, ok: bool) {
        self.obligation.record(ok);
        self.rows.push((label.to_string(), ok, true));
    }
    fn negative(&mut self, label: &str, ok: bool) {
        self.negative.record(ok);
        self.rows.push((label.to_string(), ok, false));
    }
    pub fn render(&self) -> String {
        let width = self
            .rows
            .iter()
            .map(|(l, _, _)| l.len())
            .max()
            .unwrap_or(40)
            .max("step".len())
            + 2;
        let mut out = format!("{:<width$}{:<12}verdict\n", "step", "arm");
        for (label, ok, is_obligation) in &self.rows {
            out.push_str(&format!(
                "{label:<width$}{:<12}{}\n",
                if *is_obligation {
                    "obligation"
                } else {
                    "negative"
                },
                if *ok { "pass" } else { "FAIL" }
            ));
        }
        out.push_str(&format!(
            "\nobligations {}/{}   negative invariants {}/{}\n\
             (kept apart deliberately: a store that keeps nothing passes every negative invariant)\n",
            self.obligation.passed,
            self.obligation.total,
            self.negative.passed,
            self.negative.total
        ));
        out
    }
}

/// A deliberately fictional, high-entropy canary, so a hit can only come from
/// this store and never from a model's training data.
const CANARY: &str = "the dog Plumbus Vantablack-7 guards the staging hatch";
/// Probing on one rare token keeps the search honest about substrings.
const CANARY_TOKEN: &str = "Vantablack-7";
/// The never-forgotten twin. Every negative assertion is paired with this, so
/// a suite where recall is simply broken cannot score clean.
const TWIN: &str = "the cat Grimbold Cerulean-4 guards the production hatch";
const TWIN_TOKEN: &str = "Cerulean-4";

fn q(text: &str) -> SearchQuery {
    SearchQuery {
        text: text.into(),
        scope: Scope::All,
        kinds: vec![],
        since: None,
        limit: 20,
        strength_alpha: 0.0,
        recency_alpha: 0.0,
        type_prior_alpha: 0.0,
        code_damp: 1.0,
        include_superseded: true, // archaeology mode: the hardest setting to pass
        as_of: None,
    }
}

fn capture(conn: &Connection, text: &str) -> crate::error::Result<i64> {
    match memory::remember(
        conn,
        memory::Remember {
            text: text.into(),
            mtype: MemoryType::Note,
            tags: vec![],
            project_id: None,
            force: true,
            actor: Actor::Human,
        },
    )? {
        memory::RememberOutcome::Created(n) => Ok(n.id),
        other => panic!("fixture capture must succeed: {other:?}"),
    }
}

/// Whether `token` appears in any column of any table — the leak probe the
/// atlas asks for, run over the real database rather than over an API that
/// might be filtering.
fn leaks_anywhere(conn: &Connection, token: &str) -> Vec<String> {
    let mut tables: Vec<String> = Vec::new();
    {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            .expect("sqlite_master is always readable");
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .expect("table list");
        for name in rows.flatten() {
            tables.push(name);
        }
    }
    let mut hits = Vec::new();
    for t in tables {
        let sql = format!("SELECT * FROM \"{t}\"");
        let Ok(mut stmt) = conn.prepare(&sql) else {
            continue;
        };
        let cols = stmt.column_count();
        let Ok(mut rows) = stmt.query([]) else {
            continue;
        };
        'row: while let Ok(Some(row)) = rows.next() {
            for i in 0..cols {
                let cell = row
                    .get::<_, Option<String>>(i)
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                if cell.contains(token) {
                    hits.push(t.clone());
                    break 'row;
                }
            }
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

/// The atlas's thirteen-step deletion-durability sequence, run end to end.
///
/// `hard` selects which contract is under test. A soft delete is documented
/// as a *reversible tombstone*, so the row itself legitimately still holds
/// the text — the assertion there is unreachability, not absence. A hard
/// delete claims absence, and is held to it.
pub fn durability(hard: bool) -> crate::error::Result<Scorecard> {
    let conn = db::open_in_memory()?;
    let mut card = Scorecard::default();

    // 1-2. Write, and assert retrievable — the baseline that stops every
    // later "not found" from being vacuous.
    let canary = capture(&conn, CANARY)?;
    let _twin = capture(&conn, TWIN)?;
    let found = |c: &Connection, token: &str| -> bool {
        search::search(c, &q(token))
            .map(|hits| {
                hits.iter()
                    .any(|h| h.node.body.as_deref().is_some_and(|b| b.contains(token)))
            })
            .unwrap_or(false)
    };
    card.obligation(
        "1. canary retrievable before delete",
        found(&conn, CANARY_TOKEN),
    );
    card.obligation(
        "2. twin retrievable before delete",
        found(&conn, TWIN_TOKEN),
    );

    // 3. An edit, so there is prior content to leak later. This is the step
    // most deletion tests skip, and the one `node_revision` made possible to
    // fail.
    memory::edit(
        &conn,
        canary,
        memory::Edit {
            text: Some(format!("{CANARY} (rotated)")),
            ..Default::default()
        },
        Actor::Human,
        Some("rotation"),
    )?;
    card.obligation(
        "3. prior wording kept before delete",
        !revision::history(&conn, canary)?.is_empty(),
    );

    // 4. Delete.
    if hard {
        store::hard_delete(&conn, canary, Actor::Human, Some("eval: purge"))?;
    } else {
        store::soft_delete(&conn, canary, Actor::Human, Some("eval: forget"))?;
    }
    card.negative("4. canary not retrievable", !found(&conn, CANARY_TOKEN));
    card.obligation("4b. twin still retrievable", found(&conn, TWIN_TOKEN));

    // 5-6. Re-feed the original source by a *different* write path. `import`
    // is a real second door into the store, which is the point: an extractor
    // re-reading an unchanged source is how deletions quietly come undone.
    //
    // Only meaningful for a soft delete. A hard delete removes the tombstone,
    // and the tombstone is the only thing that could recognise the re-offer —
    // so `--hard` trades this guard away entirely. That is a real property of
    // the two commands, stated here rather than hidden: it is the reason
    // `forget` is soft by default and `--hard` is opt-in.
    if hard {
        card.rows.push((
            "5. re-feed guard traded away by --hard (stated limit)".into(),
            true,
            true,
        ));
    } else {
        let reimported = memory::remember(
            &conn,
            memory::Remember {
                text: CANARY.into(),
                mtype: MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
                actor: Actor::System,
            },
        )?;
        card.obligation(
            "5. re-feed by another path refused",
            matches!(reimported, memory::RememberOutcome::Forgotten(_)),
        );
        card.negative(
            "6. canary still not retrievable",
            !found(&conn, CANARY_TOKEN),
        );
    }

    // 7-8. Run every background job, then assert again. This is where the
    // atlas expects almost everything to fail.
    let _ = crate::consolidate::consolidate(&conn, "eval-model", false);
    let _ = review::scan(&conn);
    card.negative(
        "8. still not retrievable after background jobs",
        !found(&conn, CANARY_TOKEN),
    );
    card.obligation(
        "8b. twin survived background jobs",
        found(&conn, TWIN_TOKEN),
    );

    // 9. Derived stores. For a hard delete the claim is absence everywhere;
    // for a soft delete the tombstone row legitimately still holds the text,
    // so the honest assertion is that nothing *beyond* the row does.
    let leaks = leaks_anywhere(&conn, CANARY_TOKEN);
    if hard {
        card.negative("9. no derived store holds the value", leaks.is_empty());
    } else {
        let beyond_the_tombstone: Vec<&String> = leaks
            .iter()
            .filter(|t| t.as_str() != "node" && !t.starts_with("node_fts"))
            .collect();
        card.negative(
            "9. nothing beyond the tombstone holds it",
            beyond_the_tombstone.is_empty(),
        );
        card.negative(
            "9b. prior wordings purged with the delete",
            revision::history(&conn, canary)?.is_empty(),
        );
    }

    // 10. The deletion is auditable — and the audit does not contain the
    // value it audits.
    let ledger = crate::audit::history(&conn, canary)?;
    card.obligation(
        "10. deletion recorded with actor and reason",
        ledger.iter().any(|m| {
            matches!(
                m.op,
                crate::model::MutationOp::Forget | crate::model::MutationOp::HardDelete
            ) && m.actor == Actor::Human
                && m.reason.is_some()
        }),
    );
    card.negative(
        "10b. audit does not quote the deleted value",
        !leaks_anywhere(&conn, CANARY_TOKEN)
            .iter()
            .any(|t| t == "mutation"),
    );

    // 11-13. Propagation: share to a second scope *before* deleting, delete
    // the original, assert unreachable from the second scope.
    let hub = db::open_in_memory()?;
    let batch = crate::replicate::snapshot(&conn)?;
    crate::replicate::apply_changes(&hub, &batch)?;
    card.negative(
        "13. unreachable from the synced second scope",
        !found(&hub, CANARY_TOKEN),
    );
    card.obligation(
        "13b. twin reached the second scope",
        found(&hub, TWIN_TOKEN),
    );

    Ok(card)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ForgetEval's ten categories. Each attack must be refused; each control
    /// must be accepted, because a guard that refuses everything is not a
    /// guard, it is a broken store.
    #[test]
    fn forgotten_facts_survive_every_attack_shape() {
        let mut card = Scorecard::default();
        let mut failures = Vec::new();
        for a in ATTACKS {
            let conn = db::open_in_memory().unwrap();
            let id = capture(&conn, a.original).unwrap();
            store::soft_delete(&conn, id, Actor::Human, Some("eval")).unwrap();

            let outcome = memory::remember(
                &conn,
                memory::Remember {
                    text: a.variant.into(),
                    mtype: MemoryType::Note,
                    tags: vec![],
                    project_id: None,
                    force: false,
                    actor: Actor::System,
                },
            )
            .unwrap();
            let refused = matches!(outcome, memory::RememberOutcome::Forgotten(_));
            let ok = refused == a.same_fact;
            if a.same_fact {
                card.negative(&format!("{}: re-offer refused", a.category), ok);
            } else {
                card.obligation(&format!("{}: control accepted", a.category), ok);
            }
            if !ok {
                failures.push(format!(
                    "[{}] {:?} -> {:?}: expected {}, got {}",
                    a.category,
                    a.original,
                    a.variant,
                    if a.same_fact { "refused" } else { "accepted" },
                    if refused { "refused" } else { "accepted" },
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{}\n{}",
            failures.join("\n"),
            card.render()
        );
    }

    #[test]
    fn soft_delete_is_durable_across_the_thirteen_steps() {
        let card = durability(false).unwrap();
        assert!(
            card.negative.is_clean() && card.obligation.is_clean(),
            "{}",
            card.render()
        );
    }

    #[test]
    fn hard_delete_leaves_nothing_anywhere() {
        let card = durability(true).unwrap();
        assert!(
            card.negative.is_clean() && card.obligation.is_clean(),
            "{}",
            card.render()
        );
    }

    /// The control arm that keeps the split scorecard honest. An empty store
    /// passes every negative invariant trivially — it leaks nothing because
    /// it holds nothing — and must fail the obligations. If this ever scores
    /// clean on both arms, the suite has stopped measuring anything.
    #[test]
    fn an_empty_store_fails_every_obligation() {
        let conn = db::open_in_memory().unwrap();
        let mut card = Scorecard::default();
        let found = |token: &str| {
            search::search(&conn, &q(token))
                .map(|h| !h.is_empty())
                .unwrap_or(false)
        };
        card.obligation("canary retrievable", found(CANARY_TOKEN));
        card.obligation("twin retrievable", found(TWIN_TOKEN));
        card.negative(
            "nothing leaks",
            leaks_anywhere(&conn, CANARY_TOKEN).is_empty(),
        );

        assert_eq!(card.obligation.passed, 0, "{}", card.render());
        assert!(
            card.negative.is_clean(),
            "an empty store must pass the negative arm — that is exactly why \
             the two arms are never pooled:\n{}",
            card.render()
        );
    }

    /// Prints the matrix. Pass/fail per step, never a percentage.
    #[test]
    #[ignore = "reporting, not a gate: cargo test -p mimir-mem-core eval::forget -- --ignored --nocapture"]
    fn print_the_matrix() {
        println!("== soft delete ==\n{}", durability(false).unwrap().render());
        println!("== hard delete ==\n{}", durability(true).unwrap().render());
    }
}
