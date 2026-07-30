//! Request-body transforms applied before forwarding to the Anthropic API.
//!
//! Three passes, all pure and independently testable:
//!  - **cache** (default on, safe): if the client set NO `cache_control`, add
//!    ephemeral breakpoints on the system prompt and the last message — the
//!    stable prefix that repeats across turns. Actual savings are measured from
//!    the *response* (`usage.cache_read_input_tokens`), not estimated here.
//!  - **dedup** (default on, safe): replace later identical large content
//!    blocks with a short placeholder. The model still sees the content once
//!    (the first occurrence). Lossless in practice; big in long sessions.
//!  - **prune** (default off, lossy): replace large `tool_result` blocks in
//!    older turns with a placeholder. Off by default — it changes what the
//!    model sees.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use mimir_core::tokens;
use serde_json::{json, Value};

/// Lifetime of the breakpoints we add. The API bills a cache *write* at 1.25x
/// input for `FiveMin` and 2x for `OneHour`, and a read at ~0.1x either way —
/// so 5m breaks even on the 2nd request and 1h only on the 3rd. 1h pays off
/// solely when the session idles longer than 5 minutes between turns (an agent
/// waiting on a human); for back-to-back work it is strictly more expensive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheTtl {
    #[default]
    FiveMin,
    OneHour,
}

impl CacheTtl {
    /// Parse the `[proxy] cache_ttl` value. Returns `None` on anything else so
    /// callers can reject it loudly rather than silently falling back.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "5m" => Some(CacheTtl::FiveMin),
            "1h" => Some(CacheTtl::OneHour),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CacheTtl::FiveMin => "5m",
            CacheTtl::OneHour => "1h",
        }
    }

    /// The `cache_control` object to stamp on a block. 5m is the API default,
    /// so we omit `ttl` entirely there and keep the emitted JSON unchanged.
    fn control(self) -> Value {
        match self {
            CacheTtl::FiveMin => json!({ "type": "ephemeral" }),
            CacheTtl::OneHour => json!({ "type": "ephemeral", "ttl": "1h" }),
        }
    }
}

#[derive(Clone, Copy)]
pub struct OptimizeOpts {
    pub cache: bool,
    pub cache_ttl: CacheTtl,
    pub dedup: bool,
    pub prune: bool,
}

/// Tokens removed by the request-side passes, for the savings ledger. Cache
/// savings are NOT here — they're measured from the response.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Optimization {
    pub deduped: usize,
    pub pruned: usize,
}

/// Keep tool results from at least this many trailing messages intact.
const KEEP_RECENT_MESSAGES: usize = 6;
/// Only prune a tool_result whose text is larger than this.
const PRUNE_MIN_TOKENS: usize = 200;
/// Only dedup a block whose text is larger than this.
const DEDUP_MIN_TOKENS: usize = 100;
const PRUNE_PLACEHOLDER: &str = "[older tool result elided by mimir proxy]";
const DEDUP_PLACEHOLDER: &str = "[identical to an earlier block — elided by mimir proxy]";

pub fn optimize_request(mut req: Value, opts: OptimizeOpts) -> (Value, Optimization) {
    let mut opt = Optimization::default();
    if opts.cache {
        add_cache_breakpoints(&mut req, opts.cache_ttl);
    }
    if opts.dedup {
        dedup_blocks(&mut req, &mut opt);
    }
    if opts.prune {
        prune_old_tool_results(&mut req, &mut opt);
    }
    (req, opt)
}

fn blocks_have_cache_control(v: &Value) -> bool {
    matches!(v, Value::Array(a) if a.iter().any(|b| b.get("cache_control").is_some()))
}

/// True if the request already carries any cache_control breakpoint.
pub(crate) fn has_cache_control(req: &Value) -> bool {
    if blocks_have_cache_control(req.get("system").unwrap_or(&Value::Null)) {
        return true;
    }
    req.get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| {
            msgs.iter()
                .any(|m| blocks_have_cache_control(m.get("content").unwrap_or(&Value::Null)))
        })
        .unwrap_or(false)
}

/// Mark the stable prefix for caching: the system prompt and the last message's
/// final block. No-op if the client already manages caching.
fn add_cache_breakpoints(req: &mut Value, ttl: CacheTtl) {
    if has_cache_control(req) {
        return;
    }
    if let Some(system) = req.get_mut("system") {
        mark_cache_control(system, ttl);
    }
    if let Some(content) = req
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
        .and_then(|m| m.last_mut())
        .and_then(|m| m.get_mut("content"))
    {
        mark_cache_control(content, ttl);
    }
}

/// Put an ephemeral cache breakpoint on a `system`/`content` value, converting
/// a bare string into a one-block array so the marker has somewhere to live.
fn mark_cache_control(v: &mut Value, ttl: CacheTtl) {
    match v {
        Value::String(s) => {
            let text = std::mem::take(s);
            *v = json!([{
                "type": "text", "text": text, "cache_control": ttl.control()
            }]);
        }
        Value::Array(arr) => {
            if let Some(obj) = arr.last_mut().and_then(|b| b.as_object_mut()) {
                obj.insert("cache_control".into(), ttl.control());
            }
        }
        _ => {}
    }
}

/// Comparable text of a content block (text blocks and tool_result blocks).
fn block_text(block: &Value) -> Option<String> {
    match block.get("type").and_then(|t| t.as_str()) {
        Some("text") => block
            .get("text")
            .and_then(|t| t.as_str())
            .map(str::to_owned),
        Some("tool_result") => block.get("content").map(|c| match c {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }),
        _ => None,
    }
}

fn set_block_text(block: &mut Value, placeholder: &str) {
    if let Some(obj) = block.as_object_mut() {
        if obj.contains_key("text") {
            obj.insert("text".into(), json!(placeholder));
        } else {
            obj.insert("content".into(), json!(placeholder));
        }
    }
}

/// Replace later identical large blocks with a placeholder (keep the first).
fn dedup_blocks(req: &mut Value, opt: &mut Optimization) {
    let Some(msgs) = req.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    // Store digests, not the full block texts — those can be KB–MB each.
    let mut seen: HashSet<u64> = HashSet::new();
    for msg in msgs.iter_mut() {
        let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for b in blocks.iter_mut() {
            let Some(text) = block_text(b) else { continue };
            let toks = tokens::count(&text);
            if toks < DEDUP_MIN_TOKENS {
                continue;
            }
            if !seen.insert(digest(&text)) {
                opt.deduped += toks.saturating_sub(tokens::count(DEDUP_PLACEHOLDER));
                set_block_text(b, DEDUP_PLACEHOLDER);
            }
        }
    }
}

fn digest(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn prune_old_tool_results(req: &mut Value, opt: &mut Optimization) {
    let Some(msgs) = req.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    let keep_from = msgs.len().saturating_sub(KEEP_RECENT_MESSAGES);
    for msg in msgs.iter_mut().take(keep_from) {
        let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for b in blocks.iter_mut() {
            if b.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                continue;
            }
            let Some(text) = block_text(b) else { continue };
            let toks = tokens::count(&text);
            if toks > PRUNE_MIN_TOKENS && text != DEDUP_PLACEHOLDER {
                opt.pruned += toks.saturating_sub(tokens::count(PRUNE_PLACEHOLDER));
                set_block_text(b, PRUNE_PLACEHOLDER);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(cache: bool, dedup: bool, prune: bool) -> OptimizeOpts {
        OptimizeOpts {
            cache,
            cache_ttl: CacheTtl::FiveMin,
            dedup,
            prune,
        }
    }

    #[test]
    fn five_min_ttl_omits_the_ttl_field() {
        let req = json!({
            "system": "You are careful. ".repeat(20),
            "messages": [{"role":"user","content":"hi"}]
        });
        let (out, _) = optimize_request(req, opts(true, false, false));
        // 5m is the API default — emit the bare marker, unchanged from before.
        assert_eq!(
            out["system"][0]["cache_control"],
            json!({"type":"ephemeral"})
        );
    }

    #[test]
    fn one_hour_ttl_stamps_both_breakpoints() {
        let req = json!({
            "system": "You are careful. ".repeat(20),
            "messages": [{"role":"user","content":"do the thing"}]
        });
        let (out, _) = optimize_request(
            req,
            OptimizeOpts {
                cache: true,
                cache_ttl: CacheTtl::OneHour,
                dedup: false,
                prune: false,
            },
        );
        let expected = json!({"type":"ephemeral","ttl":"1h"});
        assert_eq!(out["system"][0]["cache_control"], expected);
        assert_eq!(
            out["messages"][0]["content"][0]["cache_control"], expected,
            "the last message must carry the same TTL as the system prefix"
        );
    }

    #[test]
    fn cache_ttl_parse_rejects_unknown_values() {
        assert_eq!(CacheTtl::parse("5m"), Some(CacheTtl::FiveMin));
        assert_eq!(CacheTtl::parse(" 1h "), Some(CacheTtl::OneHour));
        assert_eq!(CacheTtl::parse("1hour"), None);
        assert_eq!(CacheTtl::parse("300"), None);
        assert_eq!(CacheTtl::parse(""), None);
    }

    #[test]
    fn caches_system_and_last_message() {
        let req = json!({
            "system": "You are careful. ".repeat(20),
            "messages": [
                {"role":"user","content":"hi"},
                {"role":"user","content":"do the thing"}
            ]
        });
        let (out, _) = optimize_request(req, opts(true, false, false));
        assert_eq!(out["system"][0]["cache_control"]["type"], "ephemeral");
        let msgs = out["messages"].as_array().unwrap();
        let last = msgs.last().unwrap();
        assert_eq!(last["content"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn respects_existing_cache_control() {
        let req = json!({
            "system": [{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}],
            "messages": []
        });
        let (out, opt) = optimize_request(req.clone(), opts(true, true, false));
        assert_eq!(opt, Optimization::default());
        assert_eq!(out, req);
    }

    #[test]
    fn dedup_elides_later_identical_blocks() {
        let big = "x ".repeat(200); // > 100 tokens
        let req = json!({
            "messages": [
                {"role":"user","content":[{"type":"text","text": big}]},
                {"role":"user","content":"middle"},
                {"role":"user","content":[{"type":"text","text": big}]}
            ]
        });
        let (out, opt) = optimize_request(req, opts(false, true, false));
        assert!(opt.deduped > 0);
        // first kept, last elided
        assert_ne!(out["messages"][0]["content"][0]["text"], DEDUP_PLACEHOLDER);
        assert_eq!(out["messages"][2]["content"][0]["text"], DEDUP_PLACEHOLDER);
    }

    #[test]
    fn dedup_keeps_small_repeats() {
        let req = json!({
            "messages": [
                {"role":"user","content":[{"type":"text","text":"short"}]},
                {"role":"user","content":[{"type":"text","text":"short"}]}
            ]
        });
        let (out, opt) = optimize_request(req, opts(false, true, false));
        assert_eq!(opt.deduped, 0);
        assert_eq!(out["messages"][1]["content"][0]["text"], "short");
    }

    #[test]
    fn prune_elides_old_large_tool_results() {
        let big = "x ".repeat(400);
        let mut messages = vec![json!({
            "role":"user",
            "content":[{"type":"tool_result","tool_use_id":"a","content": big}]
        })];
        for _ in 0..KEEP_RECENT_MESSAGES {
            messages.push(json!({"role":"user","content":"recent"}));
        }
        let req = json!({ "messages": messages });
        let (out, opt) = optimize_request(req, opts(false, false, true));
        assert!(opt.pruned > 0);
        assert_eq!(
            out["messages"][0]["content"][0]["content"],
            PRUNE_PLACEHOLDER
        );
    }
}
