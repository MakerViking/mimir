//! How a call edge was resolved, named once.
//!
//! Extraction is static: no type checker, so a call site is tied to its
//! target by one of five rules of decreasing certainty. The rule used to
//! survive only as a side effect — three of them wrote the same weight, the
//! `resolved` flag folded four into one, and a reader of the edge table could
//! not tell a receiver-typed edge from a same-file one, or see that a later,
//! weaker match for the same pair had overwritten a stronger one.
//!
//! [`CallTier`] is that rule. It owns the edge's weight, its `resolved` flag
//! and its stored label; the resolver picks a tier and everything written to
//! the edge follows from it.

/// The rule that tied a call site to a target, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallTier {
    /// `recv.callee(..)` where the receiver's type owns exactly one method of
    /// that name. The only tier that can tell two same-named methods apart.
    ReceiverTyped,
    /// The one definition of that name in the calling file.
    SameFile,
    /// The name was imported, and the imported file defines it.
    ImportedName,
    /// The only definition of that name in the whole project.
    UniqueGlobal,
    /// One of `candidates` definitions the name could mean (two or three;
    /// more than that is dropped). Every candidate gets an edge, and none of
    /// them is a claim that the call happens.
    Ambiguous { candidates: usize },
}

impl CallTier {
    /// The weight stored on the edge: 1.0 for a match with evidence beyond
    /// the name, 0.8 for a name that happens to be unique, and an even split
    /// across the candidates of a guess.
    pub fn weight(self) -> f64 {
        match self {
            CallTier::ReceiverTyped | CallTier::SameFile | CallTier::ImportedName => 1.0,
            CallTier::UniqueGlobal => 0.8,
            CallTier::Ambiguous { candidates } => 1.0 / candidates.max(1) as f64,
        }
    }

    /// Whether the edge names one target. `false` is a guess, and every
    /// surface that walks the edge says so.
    pub fn is_resolved(self) -> bool {
        !matches!(self, CallTier::Ambiguous { .. })
    }

    /// [`CallTier::is_resolved`] for an edge as stored, read from its weight.
    ///
    /// The weight is the one thing every build has written on a call edge,
    /// from the first release on: 1.0 or 0.8 for a match, an even share of
    /// 1.0 between two or three candidates for a guess. So "heavier than the
    /// best guess" is the `resolved` flag, without opening the edge's JSON —
    /// which is what loading a graph of a million edges would otherwise
    /// spend a fifth of its time on. An edge somebody authored by hand has
    /// the default weight, 1.0, and reads as resolved, as it always has.
    pub fn weight_is_resolved(weight: f64) -> bool {
        weight > CallTier::Ambiguous { candidates: 2 }.weight()
    }

    /// The name stored in the edge's `meta.tier`.
    pub fn label(self) -> &'static str {
        match self {
            CallTier::ReceiverTyped => "receiver-typed",
            CallTier::SameFile => "same-file",
            CallTier::ImportedName => "imported-name",
            CallTier::UniqueGlobal => "unique-global",
            CallTier::Ambiguous { .. } => "ambiguous",
        }
    }

    /// Position in the order above; lower is stronger. A guess among two
    /// outranks a guess among three.
    fn rank(self) -> usize {
        match self {
            CallTier::ReceiverTyped => 0,
            CallTier::SameFile => 1,
            CallTier::ImportedName => 2,
            CallTier::UniqueGlobal => 3,
            CallTier::Ambiguous { candidates } => 3 + candidates.max(1),
        }
    }

    /// Whether this tier is stronger evidence than `other` for the same
    /// (caller, target) pair. One function can call the same target from
    /// several sites; the edge keeps the best of them.
    pub fn outranks(self, other: CallTier) -> bool {
        self.rank() < other.rank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDERED: [CallTier; 6] = [
        CallTier::ReceiverTyped,
        CallTier::SameFile,
        CallTier::ImportedName,
        CallTier::UniqueGlobal,
        CallTier::Ambiguous { candidates: 2 },
        CallTier::Ambiguous { candidates: 3 },
    ];

    /// The three things an edge stores all come from the tier, and they
    /// agree with each other: only a guess is unresolved, and a guess never
    /// weighs as much as a match.
    #[test]
    fn a_tier_owns_weight_flag_and_label() {
        for tier in ORDERED {
            let guess = matches!(tier, CallTier::Ambiguous { .. });
            assert_eq!(tier.is_resolved(), !guess, "{tier:?}");
            assert_eq!(tier.label() == "ambiguous", guess, "{tier:?}");
            if guess {
                assert!(tier.weight() <= 0.5, "{tier:?}");
            } else {
                assert!(tier.weight() >= 0.8, "{tier:?}");
            }
        }
        assert_eq!(CallTier::Ambiguous { candidates: 3 }.weight(), 1.0 / 3.0);
        // The stored weight alone says which side of the line an edge is on.
        for tier in ORDERED {
            assert_eq!(
                CallTier::weight_is_resolved(tier.weight()),
                tier.is_resolved(),
                "{tier:?}"
            );
        }
        let labels: std::collections::HashSet<_> = ORDERED.iter().map(|t| t.label()).collect();
        assert_eq!(
            labels.len(),
            5,
            "one label per tier, shared by both guesses"
        );
    }

    /// Strictly ordered: a stronger tier outranks every weaker one, and
    /// nothing outranks itself (so an equal match never rewrites the edge).
    #[test]
    fn outranks_is_a_strict_order() {
        for (i, a) in ORDERED.iter().enumerate() {
            for (j, b) in ORDERED.iter().enumerate() {
                assert_eq!(a.outranks(*b), i < j, "{a:?} vs {b:?}");
            }
        }
    }
}
