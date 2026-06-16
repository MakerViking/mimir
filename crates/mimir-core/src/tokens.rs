//! Token counting for the savings ledger.
//!
//! Claude has no exact public local tokenizer, so we use a bundled BPE
//! (tiktoken `o200k_base`) as a deterministic, offline, close-enough proxy.
//! Every savings producer (outline/peek/filter/proxy) measures `tokens_before`
//! and `tokens_after` through [`count`], so the *relative* numbers in the
//! dashboard are consistent even if the absolute count drifts a few percent
//! from Anthropic's billed tokens.

use once_cell::sync::Lazy;
use tiktoken_rs::{o200k_base, CoreBPE};

/// The encoder is built once (it parses an embedded rank table) and reused.
static BPE: Lazy<Option<CoreBPE>> = Lazy::new(|| match o200k_base() {
    Ok(bpe) => Some(bpe),
    Err(err) => {
        tracing::warn!(%err, "tokenizer unavailable; falling back to char heuristic");
        None
    }
});

/// Count tokens in `text`. Falls back to a ~chars/4 heuristic only if the
/// bundled tokenizer fails to load (it never should — the table is embedded).
pub fn count(text: &str) -> usize {
    match BPE.as_ref() {
        Some(bpe) => bpe.encode_ordinary(text).len(),
        None => text.chars().count().div_ceil(4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(count(""), 0);
    }

    #[test]
    fn counts_scale_with_length() {
        let short = count("hello world");
        let long = count("hello world ".repeat(50).as_str());
        assert!(short > 0);
        assert!(long > short * 10, "long={long} short={short}");
    }

    #[test]
    fn code_counts_are_reasonable() {
        // A line of code should be a handful of tokens, not characters.
        let n = count("pub fn count(text: &str) -> usize {");
        assert!(n > 3 && n < 20, "n={n}");
    }
}
