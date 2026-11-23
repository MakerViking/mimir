//! Request-body transforms applied before forwarding to the Anthropic API.
//!
//! Three passes, all pure and independently testable:
//!  - **cache** (default on, safe): if the client set NO `cache_control`
//!    anywhere the API accepts one, add ephemeral breakpoints on the system
//!    prompt and the last message — the stable prefix that repeats across
//!    turns. Actual savings are measured from the *response*
//!    (`usage.cache_read_input_tokens`), not estimated here.
//!  - **dedup** (default on, safe): replace later identical large content
//!    blocks with a short placeholder. The model still sees the content once
//!    (the first occurrence). Lossless in practice; big in long sessions.
//!  - **prune** (default off, lossy): replace large `tool_result` blocks in
//!    older turns with a placeholder. Off by default — it changes what the
//!    model sees.
//!
//! Every request of a conversation is the previous one plus a turn, and the
//! API's prompt cache reads a prefix only if its bytes come out the same
//! again. So a pass may never rewrite a block it left alone on an earlier
//! turn, except at a boundary it chose on purpose: dedup only ever replaces a
//! block that has an identical one *before* it, and prune moves its window in
//! steps ([`PRUNE_STEP_MESSAGES`]).
//!
//! API rules relied on here are from the prompt-caching guide
//! (platform.claude.com/docs/en/build-with-claude/prompt-caching), the
//! Messages reference (…/docs/en/api/messages/create) and the
//! mid-conversation system messages guide
//! (…/docs/en/build-with-claude/mid-conversation-system-messages).

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

/// Everything the passes did to one request: what the forwarder needs to
/// decide which bytes to send and what it may claim for them afterwards.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Applied {
    /// Breakpoints the cache pass put in the body. Zero when the pass is off,
    /// when the client manages its own caching, and when the request had no
    /// block that can take one.
    pub(crate) breakpoints: usize,
    /// Blocks dedup and prune replaced with a placeholder.
    pub(crate) replaced: usize,
    /// Tokens those replacements removed.
    pub(crate) saved: Optimization,
}

impl Applied {
    /// The body is no longer what the client sent.
    pub(crate) fn changed(&self) -> bool {
        self.breakpoints > 0 || self.replaced > 0
    }

    /// A breakpoint of ours is in the body, so the cache reads the response
    /// reports are ours to claim.
    pub(crate) fn we_cached(&self) -> bool {
        self.breakpoints > 0
    }
}

/// Keep tool results from at least this many trailing messages intact.
const KEEP_RECENT_MESSAGES: usize = 6;
/// The prune window moves this many messages at a time. A window that slides
/// with every turn rewrites one more old block on every request: the request
/// prefix never repeats, and the prompt cache (ours or the client's) writes
/// the whole conversation again each turn instead of reading it. In steps,
/// the prefix is byte-stable for a whole step and the cache pays for one
/// re-write per step. 20 messages is about ten tool round-trips.
const PRUNE_STEP_MESSAGES: usize = 20;
/// Only prune a tool_result whose text is larger than this.
const PRUNE_MIN_TOKENS: usize = 200;
/// Only dedup a block whose text is larger than this.
const DEDUP_MIN_TOKENS: usize = 100;
/// What one image costs in input tokens. The API bills an image at about
/// width*height/750 after resizing its long edge to at most 1568 px, which
/// tops out near 1,600. Its base64 `data` is NOT text: read as text, one
/// screenshot came out at hundreds of thousands of tokens.
const IMAGE_TOKENS: usize = 1_600;
const PRUNE_PLACEHOLDER: &str = "[older tool result elided by mimir proxy]";
const DEDUP_PLACEHOLDER: &str = "[identical to an earlier block — elided by mimir proxy]";

pub fn optimize_request(req: Value, opts: OptimizeOpts) -> (Value, Optimization) {
    let (req, applied) = apply_passes(req, opts);
    (req, applied.saved)
}

/// Run the enabled passes over `req` and report what each one did.
pub(crate) fn apply_passes(mut req: Value, opts: OptimizeOpts) -> (Value, Applied) {
    let mut applied = Applied::default();
    if opts.cache {
        applied.breakpoints = add_cache_breakpoints(&mut req, opts.cache_ttl);
    }
    if opts.dedup {
        dedup_blocks(&mut req, &mut applied);
    }
    if opts.prune {
        prune_old_tool_results(&mut req, &mut applied);
    }
    (req, applied)
}

/// True if the client manages its own caching: a breakpoint anywhere the API
/// accepts one. That is the request's own top-level `cache_control`
/// (automatic caching) and any block of `tools`, `system` or `messages`,
/// nested ones included (the content of a `tool_result`, a document's source,
/// a tool defined inside a `tool_addition` block).
///
/// It has to be all of them. The API allows four breakpoints per request and
/// no longer TTL after a shorter one, so ours on top of a client's we did not
/// see can turn a request that worked into a 400.
pub(crate) fn has_cache_control(req: &Value) -> bool {
    req.get("cache_control").is_some_and(|c| !c.is_null())
        || ["tools", "system", "messages"]
            .iter()
            .any(|section| req.get(*section).is_some_and(carries_breakpoint))
}

/// True if `v` or anything nested in it has a `cache_control` that is set.
/// `null` is the API's spelling of "not set".
///
/// `input`, `input_schema` and `input_examples` are not searched: they hold
/// free-form JSON (a tool call's arguments, a tool's schema), where a key
/// named `cache_control` is somebody's data and the API reads no breakpoint.
fn carries_breakpoint(v: &Value) -> bool {
    match v {
        Value::Object(obj) => obj.iter().any(|(key, v)| match key.as_str() {
            "cache_control" => !v.is_null(),
            "input" | "input_schema" | "input_examples" => false,
            _ => carries_breakpoint(v),
        }),
        Value::Array(items) => items.iter().any(carries_breakpoint),
        _ => false,
    }
}

/// Mark the stable prefix for caching: the system prompt and the end of the
/// conversation. Returns how many breakpoints it added — none if the client
/// already manages caching.
fn add_cache_breakpoints(req: &mut Value, ttl: CacheTtl) -> usize {
    if has_cache_control(req) {
        return 0;
    }
    let mut added = 0;
    if let Some(system) = req.get_mut("system") {
        added += usize::from(mark_cache_control(system, ttl));
    }
    if let Some(msgs) = req.get_mut("messages").and_then(|m| m.as_array_mut()) {
        // The last message, or the nearest one before it that can take a
        // breakpoint: everything up to the marked block is still cached.
        for msg in msgs.iter_mut().rev() {
            if is_turn_scoped(msg) {
                continue;
            }
            if msg
                .get_mut("content")
                .is_some_and(|content| mark_cache_control(content, ttl))
            {
                added += 1;
                break;
            }
        }
    }
    added
}

/// A `role: "system"` message that renders for one turn only: until a user
/// message follows it, after which the API clears it. The passes keep off
/// it. The API rejects `cache_control` on its blocks (the breakpoint belongs
/// on the turn before it), and dedup has its own reason, given there.
fn is_turn_scoped(msg: &Value) -> bool {
    msg.get("clear_at").and_then(|c| c.as_str()) == Some("next_user_message")
}

/// Put an ephemeral cache breakpoint at the end of a `system`/`content`
/// value: on its last block that can take one. A bare string becomes a
/// one-block array so the marker has somewhere to live. Returns whether a
/// breakpoint was placed; if not, `v` is untouched.
fn mark_cache_control(v: &mut Value, ttl: CacheTtl) -> bool {
    match v {
        // As a block this would be an empty text block, which the API rejects.
        Value::String(s) if s.is_empty() => false,
        Value::String(s) => {
            let text = std::mem::take(s);
            *v = json!([{
                "type": "text", "text": text, "cache_control": ttl.control()
            }]);
            true
        }
        Value::Array(arr) => {
            let target = arr
                .iter_mut()
                .rev()
                .find(|b| takes_breakpoint(b))
                .and_then(|b| b.as_object_mut());
            match target {
                Some(obj) => {
                    obj.insert("cache_control".into(), ttl.control());
                    true
                }
                None => false,
            }
        }
        _ => false,
    }
}

/// What a content block is, as far as these passes care. Parsed from the
/// block in this one place, so the questions a pass asks of a block — what
/// text identifies it, what it costs, where a placeholder goes, whether it
/// can carry a breakpoint — are all answered about the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    /// `text`. Its text is the `text` field. A bare string is one too: the
    /// API reads string content as shorthand for a single text block.
    Text,
    /// `tool_result`. Its text is `content`: a string, or nested blocks.
    ToolResult,
    /// `image`. No text; a fixed cost.
    Image,
    /// `thinking` and `redacted_thinking`: the model's own reasoning, sent
    /// back as it came.
    Thinking,
    /// Every other type. Left alone and counted as nothing.
    Other,
}

impl BlockKind {
    fn of(block: &Value) -> BlockKind {
        if block.is_string() {
            return BlockKind::Text;
        }
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => BlockKind::Text,
            Some("tool_result") => BlockKind::ToolResult,
            Some("image") => BlockKind::Image,
            Some("thinking" | "redacted_thinking") => BlockKind::Thinking,
            _ => BlockKind::Other,
        }
    }
}

/// The text of a [`BlockKind::Text`] block, in either of its two forms.
fn text_of(block: &Value) -> Option<&str> {
    match block {
        Value::String(s) => Some(s),
        other => other.get("text").and_then(|t| t.as_str()),
    }
}

/// Whether a block of a content array may carry `cache_control`. The API
/// says no for an empty text block and for thinking blocks; a marker there
/// is an error where the request had none. A type this file does not know
/// is marked: most block types take one, and nothing documented says that
/// one does not.
fn takes_breakpoint(block: &Value) -> bool {
    if !block.is_object() {
        return false;
    }
    match BlockKind::of(block) {
        BlockKind::Text => text_of(block).is_some_and(|t| !t.is_empty()),
        BlockKind::Thinking => false,
        BlockKind::ToolResult | BlockKind::Image | BlockKind::Other => true,
    }
}

/// Comparable text of a content block (text blocks and tool_result blocks).
fn block_text(block: &Value) -> Option<String> {
    match BlockKind::of(block) {
        BlockKind::Text => text_of(block).map(str::to_owned),
        BlockKind::ToolResult => block.get("content").map(|c| match c {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }),
        BlockKind::Image | BlockKind::Thinking | BlockKind::Other => None,
    }
}

/// Billed-token estimate for one content block: text as text, an image as
/// [`IMAGE_TOKENS`], never the base64 behind it. `block_text` stays the
/// block's IDENTITY (two different screenshots must never dedup); this is its
/// COST. Other block types count 0, keeping `count_request_tokens` an
/// under-estimate.
fn block_tokens(block: &Value) -> usize {
    match BlockKind::of(block) {
        BlockKind::Text => text_of(block).map_or(0, tokens::count),
        BlockKind::Image => IMAGE_TOKENS,
        BlockKind::ToolResult => match block.get("content") {
            Some(Value::String(s)) => tokens::count(s),
            Some(Value::Array(items)) => items.iter().map(block_tokens).sum(),
            _ => 0,
        },
        BlockKind::Thinking | BlockKind::Other => 0,
    }
}

/// Replace a block's text with `placeholder`, where [`block_text`] read it
/// from. A block without text is left as it is.
fn set_block_text(block: &mut Value, placeholder: &str) {
    match BlockKind::of(block) {
        BlockKind::Text => match block {
            Value::String(s) => *s = placeholder.to_owned(),
            other => other["text"] = json!(placeholder),
        },
        BlockKind::ToolResult => block["content"] = json!(placeholder),
        BlockKind::Image | BlockKind::Thinking | BlockKind::Other => {}
    }
}

/// The blocks of a message's `content`: the elements of an array, or the
/// string itself, which stands for one text block. Dedup sees both, so
/// whether a message is eligible does not depend on which form the client
/// chose, or on the cache pass having turned one into the other first.
fn content_blocks_mut(content: &mut Value) -> std::slice::IterMut<'_, Value> {
    match content {
        Value::Array(blocks) => blocks.iter_mut(),
        Value::String(_) => std::slice::from_mut(content).iter_mut(),
        _ => Default::default(),
    }
}

/// Approximate billed input tokens for a `/v1/messages` body: the `system`
/// prompt, every message's text/tool_result content, and the `tools` array
/// (whose descriptions and schemas are input too — often tens of thousands of
/// tokens before the user types anything).
///
/// Deliberately an **under**-estimate: it counts text, not JSON scaffolding or
/// per-message overhead. This is only used as a runaway circuit breaker, where
/// a false positive (rejecting a request that was actually fine) is much worse
/// than firing slightly late. Never use it for billing — the real numbers come
/// from the response's `usage`.
pub fn count_request_tokens(req: &Value) -> usize {
    let mut total = 0usize;

    total += match req.get("system") {
        Some(Value::String(s)) => tokens::count(s),
        Some(Value::Array(blocks)) => blocks.iter().map(block_tokens).sum(),
        _ => 0,
    };

    if let Some(msgs) = req.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            total += match msg.get("content") {
                Some(Value::String(s)) => tokens::count(s),
                Some(Value::Array(blocks)) => blocks.iter().map(block_tokens).sum(),
                _ => 0,
            };
        }
    }

    // Tool definitions ride along on every single request.
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        for tool in tools {
            for key in ["name", "description"] {
                if let Some(s) = tool.get(key).and_then(|v| v.as_str()) {
                    total += tokens::count(s);
                }
            }
            if let Some(schema) = tool.get("input_schema") {
                total += tokens::count(&schema.to_string());
            }
        }
    }

    total
}

/// True for a `tool_result` whose nested content carries a breakpoint of the
/// client's. A placeholder replaces `content` whole, as a plain string, and
/// the breakpoint would go with the blocks it stood on. Dedup and prune leave
/// such a block exactly as it is.
fn nests_a_breakpoint(block: &Value) -> bool {
    BlockKind::of(block) == BlockKind::ToolResult
        && block.get("content").is_some_and(carries_breakpoint)
}

/// Replace later identical large blocks with a placeholder (keep the first).
fn dedup_blocks(req: &mut Value, applied: &mut Applied) {
    let Some(msgs) = req.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    // Store digests, not the full block texts — those can be KB–MB each.
    let mut seen: HashSet<u64> = HashSet::new();
    for msg in msgs.iter_mut() {
        // A turn-scoped message is rendered until the next user message and
        // cleared by the API from then on, so of several copies of a reminder
        // the model only ever reads the newest. It takes no part in dedup:
        // replaced, the model would have no copy left; remembered as a first
        // occurrence, a later block would be "identical to" one the model
        // cannot see.
        if is_turn_scoped(msg) {
            continue;
        }
        let Some(content) = msg.get_mut("content") else {
            continue;
        };
        for b in content_blocks_mut(content) {
            if nests_a_breakpoint(b) {
                continue;
            }
            let Some(text) = block_text(b) else { continue };
            let toks = block_tokens(b);
            if toks < DEDUP_MIN_TOKENS {
                continue;
            }
            if !seen.insert(digest(&text)) {
                applied.saved.deduped += toks.saturating_sub(tokens::count(DEDUP_PLACEHOLDER));
                applied.replaced += 1;
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

/// Index of the first message prune leaves alone, for a conversation of
/// `len` messages: everything before it is old. At least
/// [`KEEP_RECENT_MESSAGES`] trailing messages are always kept, and the
/// boundary only ever sits on a multiple of [`PRUNE_STEP_MESSAGES`].
fn prune_boundary(len: usize) -> usize {
    len.saturating_sub(KEEP_RECENT_MESSAGES) / PRUNE_STEP_MESSAGES * PRUNE_STEP_MESSAGES
}

/// Replace the large tool results of old messages with a placeholder. A
/// block already replaced by either pass is far below [`PRUNE_MIN_TOKENS`]
/// and is not touched again.
fn prune_old_tool_results(req: &mut Value, applied: &mut Applied) {
    let Some(msgs) = req.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    let boundary = prune_boundary(msgs.len());
    for msg in msgs.iter_mut().take(boundary) {
        let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for b in blocks.iter_mut() {
            if BlockKind::of(b) != BlockKind::ToolResult || nests_a_breakpoint(b) {
                continue;
            }
            let toks = block_tokens(b);
            if toks > PRUNE_MIN_TOKENS {
                applied.saved.pruned += toks.saturating_sub(tokens::count(PRUNE_PLACEHOLDER));
                applied.replaced += 1;
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
    fn count_request_tokens_covers_system_messages_and_tools() {
        let req = json!({
            "system": "You are careful. ".repeat(50),
            "messages": [
                {"role":"user","content":"hello there"},
                {"role":"user","content":[
                    {"type":"text","text":"x ".repeat(100)},
                    {"type":"tool_result","tool_use_id":"a","content":"y ".repeat(100)}
                ]}
            ],
            "tools": [{
                "name":"get_weather",
                "description":"Get the weather. ".repeat(30),
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}}}
            }]
        });
        let n = count_request_tokens(&req);

        // Each section must actually contribute — a counter that silently
        // skipped `tools` would still look plausible on the total alone.
        let no_tools = count_request_tokens(&json!({
            "system": req["system"], "messages": req["messages"]
        }));
        let only_tools = count_request_tokens(&json!({ "tools": req["tools"] }));
        assert!(only_tools > 0, "tool definitions must be counted");
        assert_eq!(n, no_tools + only_tools);
    }

    #[test]
    fn count_request_tokens_is_zero_for_an_empty_body() {
        assert_eq!(count_request_tokens(&json!({})), 0);
        assert_eq!(count_request_tokens(&json!({"messages": []})), 0);
    }

    #[test]
    fn count_request_tokens_ignores_unknown_block_types() {
        // An image costs a bounded IMAGE_TOKENS, never its base64 payload read
        // as text. Not 0: an image at 0 falls under DEDUP_MIN_TOKENS, and a
        // repeated screenshot would stop being elided.
        let req = json!({
            "messages": [{"role":"user","content":[
                {"type":"image","source":{"type":"base64","data":"AAAA".repeat(500)}}
            ]}]
        });
        assert_eq!(count_request_tokens(&req), IMAGE_TOKENS);
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
        // Enough newer messages for the prune boundary to have taken its
        // first step past the old one.
        for _ in 1..KEEP_RECENT_MESSAGES + PRUNE_STEP_MESSAGES {
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

    /// A screenshot tool_result: base64 PNG data, the shape Read returns for an image.
    fn screenshot_result(seed: char) -> Value {
        json!({"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
             "data": seed.to_string().repeat(400_000)}}
        ]})
    }

    #[test]
    fn an_image_counts_as_an_image_not_as_its_base64() {
        // 400 KB of base64 is ~100k "tokens" read as text; the API bills an
        // image at most IMAGE_TOKENS. Measured 2026-10-15: the ledger logged
        // single requests "saving" 798k tokens from deduped screenshots.
        let req = json!({"messages": [{"role": "user", "content": [screenshot_result('A')]}]});
        assert_eq!(count_request_tokens(&req), IMAGE_TOKENS);
    }

    #[test]
    fn dedup_of_a_repeated_screenshot_saves_an_image_worth() {
        let req = json!({"messages": [
            {"role": "user", "content": [screenshot_result('A')]},
            {"role": "user", "content": [screenshot_result('A')]},
        ]});
        let (_, opt) = optimize_request(req, opts(false, true, false));
        assert!(opt.deduped > 0 && opt.deduped <= IMAGE_TOKENS, "{opt:?}");
    }

    #[test]
    fn two_different_screenshots_are_not_duplicates() {
        let req = json!({"messages": [
            {"role": "user", "content": [screenshot_result('A')]},
            {"role": "user", "content": [screenshot_result('B')]},
        ]});
        let (out, opt) = optimize_request(req, opts(false, true, false));
        assert_eq!(opt.deduped, 0);
        assert!(
            out.to_string().contains("BBBB"),
            "the second image must survive"
        );
    }

    fn ephemeral() -> Value {
        json!({"type": "ephemeral"})
    }

    #[test]
    fn an_empty_system_prompt_gets_no_breakpoint() {
        // `system: ""` is a legal request. As a block it would be
        // `{"type":"text","text":""}`, which the API rejects: a text block's
        // `text` has minLength 1.
        let req = json!({
            "system": "",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (out, _) = optimize_request(req, opts(true, false, false));
        assert_eq!(out["system"], json!(""), "left as the client sent it");
        assert_eq!(
            out["messages"][0]["content"][0]["cache_control"],
            ephemeral(),
            "the message breakpoint does not depend on the system one"
        );
    }

    #[test]
    fn breakpoint_lands_on_the_nearest_earlier_block_that_can_take_one() {
        let thinking = json!({"type": "thinking", "thinking": "hm", "signature": "sig"});
        for (name, last) in [
            ("an empty text block", json!({"type": "text", "text": ""})),
            ("a thinking block", thinking),
            (
                "a redacted thinking block",
                json!({"type": "redacted_thinking", "data": "xx"}),
            ),
        ] {
            let req = json!({"messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "the answer so far"},
                last.clone(),
            ]}]});
            let (out, _) = optimize_request(req, opts(true, false, false));
            let blocks = &out["messages"][0]["content"];
            assert_eq!(blocks[0]["cache_control"], ephemeral(), "{name}");
            assert_eq!(blocks[1], last, "{name}: must reach the API unchanged");
        }
    }

    #[test]
    fn breakpoint_moves_to_an_earlier_message_when_the_last_cannot_take_one() {
        // A turn-scoped system message rejects `cache_control` on its blocks
        // (HTTP 400); the documented place is the turn before it.
        let scoped = json!({
            "role": "system", "clear_at": "next_user_message",
            "content": "Keep this reply under 50 words."
        });
        // An effort-only system message has no block at all.
        let effort_only = json!({
            "role": "system", "content": [], "output_config": {"effort": "low"}
        });
        for last in [
            scoped,
            effort_only,
            json!({"role": "assistant", "content": ""}),
        ] {
            let req = json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "the question"}]},
                last.clone(),
            ]});
            let (out, _) = optimize_request(req, opts(true, false, false));
            assert_eq!(
                out["messages"][0]["content"][0]["cache_control"],
                ephemeral(),
                "{last}"
            );
            assert_eq!(out["messages"][1], last, "must reach the API unchanged");
        }
    }

    #[test]
    fn a_request_with_nothing_to_mark_is_left_alone() {
        for req in [
            json!({}),
            json!({"messages": []}),
            json!({"system": "", "messages": [{"role": "user", "content": ""}]}),
            json!({"system": [], "messages": [{"role": "user", "content": []}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}]}),
        ] {
            let (out, _) = optimize_request(req.clone(), opts(true, true, true));
            assert_eq!(out, req);
        }
    }

    /// Everywhere the Messages API accepts `cache_control`. A client with a
    /// breakpoint in any one of them manages its own caching, and ours on top
    /// can push the request over the four-breakpoint limit or put a longer TTL
    /// after a shorter one — both rejected.
    #[test]
    fn a_client_breakpoint_anywhere_keeps_ours_out() {
        let system = "You are careful. ".repeat(20);
        let user = json!({"role": "user", "content": "hi"});
        let tool = json!({"name": "t", "input_schema": {"type": "object"}});
        let mut cached_tool = tool.clone();
        cached_tool["cache_control"] = ephemeral();

        let cases = [
            (
                "the request itself (automatic caching)",
                json!({"cache_control": ephemeral(), "system": system, "messages": [user]}),
            ),
            (
                "a tool definition",
                json!({"tools": [tool, cached_tool], "system": system, "messages": [user]}),
            ),
            (
                "a system block",
                json!({
                    "system": [{"type": "text", "text": system, "cache_control": ephemeral()}],
                    "messages": [user]
                }),
            ),
            (
                "a message block",
                json!({"system": system, "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "hi", "cache_control": ephemeral()}
                    ]},
                    {"role": "assistant", "content": "hello"},
                    user,
                ]}),
            ),
            (
                "a block nested in a tool_result",
                json!({"system": system, "messages": [
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": [
                            {"type": "text", "text": "out", "cache_control": ephemeral()}
                        ]}
                    ]},
                    {"role": "assistant", "content": "hello"},
                    user,
                ]}),
            ),
            (
                "a tool defined inside a tool_addition block",
                json!({"system": system, "messages": [
                    user,
                    {"role": "system", "content": [
                        {"type": "tool_addition", "tool": {
                            "type": "tool_definition", "definition": cached_tool
                        }}
                    ]},
                    {"role": "assistant", "content": "hello"},
                    user,
                ]}),
            ),
        ];
        for (name, req) in cases {
            let (out, _) = optimize_request(req.clone(), opts(true, false, false));
            assert_eq!(out, req, "a breakpoint on {name} is the client's caching");
        }
    }

    #[test]
    fn a_key_named_cache_control_in_free_form_json_is_not_a_breakpoint() {
        // A tool's schema and a tool call's arguments are the client's and the
        // model's own JSON. A property that happens to be called
        // `cache_control` there marks nothing.
        let req = json!({
            "tools": [{
                "name": "send",
                "input_schema": {"type": "object", "properties": {
                    "cache_control": {"type": "object"}
                }},
                "input_examples": [{"cache_control": {"type": "ephemeral"}}]
            }],
            "system": "You are careful. ".repeat(20),
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "a", "name": "send",
                     "input": {"cache_control": {"type": "ephemeral"}}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "sent"}
                ]}
            ]
        });
        let (out, _) = optimize_request(req, opts(true, false, false));
        assert_eq!(out["system"][0]["cache_control"], ephemeral());
        assert_eq!(
            out["messages"][1]["content"][0]["cache_control"],
            ephemeral()
        );
    }

    #[test]
    fn a_null_cache_control_is_not_a_breakpoint() {
        // The field is "optional … or null": a serializer that writes its
        // unset options as null has set no breakpoint.
        let req = json!({
            "cache_control": null,
            "system": [{"type": "text", "text": "You are careful.", "cache_control": null}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (out, _) = optimize_request(req, opts(true, false, false));
        assert_eq!(out["system"][0]["cache_control"], ephemeral());
        assert_eq!(
            out["messages"][0]["content"][0]["cache_control"],
            ephemeral()
        );
    }

    /// A request whose last message repeats an earlier large block, with that
    /// last message in both forms the API treats as the same thing.
    fn repeat_in_last_message(as_string: bool) -> Value {
        let big = "x ".repeat(200);
        let last = if as_string {
            json!(big)
        } else {
            json!([{"type": "text", "text": big}])
        };
        json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": big}]},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": last},
        ]})
    }

    #[test]
    fn dedup_does_not_depend_on_the_cache_pass() {
        for as_string in [true, false] {
            let (_, alone) =
                optimize_request(repeat_in_last_message(as_string), opts(false, true, false));
            let (_, after_cache) =
                optimize_request(repeat_in_last_message(as_string), opts(true, true, false));
            assert!(alone.deduped > 0, "string content: {as_string}");
            assert_eq!(alone, after_cache, "string content: {as_string}");
        }
    }

    #[test]
    fn dedup_treats_string_content_as_the_text_block_it_stands_for() {
        let big = "x ".repeat(200);
        // First occurrence is a string, the repeat a block …
        let req = json!({"messages": [
            {"role": "user", "content": big},
            {"role": "user", "content": [{"type": "text", "text": big}]},
        ]});
        let (out, opt) = optimize_request(req, opts(false, true, false));
        assert!(opt.deduped > 0);
        assert_eq!(out["messages"][0]["content"], json!(big));
        assert_eq!(out["messages"][1]["content"][0]["text"], DEDUP_PLACEHOLDER);

        // … and the other way round. An elided string stays a string.
        let (out, opt) = optimize_request(repeat_in_last_message(true), opts(false, true, false));
        assert!(opt.deduped > 0);
        assert_eq!(out["messages"][2]["content"], json!(DEDUP_PLACEHOLDER));
    }

    #[test]
    fn a_placeholder_is_never_large_enough_to_be_replaced_again() {
        for placeholder in [PRUNE_PLACEHOLDER, DEDUP_PLACEHOLDER] {
            let toks = tokens::count(placeholder);
            assert!(
                toks < DEDUP_MIN_TOKENS.min(PRUNE_MIN_TOKENS),
                "{placeholder}"
            );
        }
    }

    #[test]
    fn apply_passes_reports_what_each_pass_did() {
        let all = opts(true, true, true);
        let system = "You are careful. ".repeat(20);
        let user = json!({"role": "user", "content": "hi"});

        let (_, both) = apply_passes(json!({"system": system, "messages": [user]}), all);
        assert_eq!(both.breakpoints, 2);
        assert!(both.we_cached() && both.changed());

        let (_, one) = apply_passes(json!({"messages": [user]}), all);
        assert_eq!(one.breakpoints, 1);

        // Caching is on, and still nothing of ours went in.
        let own = json!({
            "system": [{"type": "text", "text": system, "cache_control": ephemeral()}],
            "messages": [user]
        });
        for req in [own, json!({"messages": []}), json!({"system": ""})] {
            let (_, none) = apply_passes(req, all);
            assert_eq!(none, Applied::default());
            assert!(!none.we_cached() && !none.changed());
        }

        let (_, deduped) = apply_passes(repeat_in_last_message(false), opts(false, true, false));
        assert_eq!((deduped.breakpoints, deduped.replaced), (0, 1));
        assert!(deduped.changed() && !deduped.we_cached());
        assert!(deduped.saved.deduped > 0);
    }

    /// `n` messages of a tool loop: every user turn carries a large tool
    /// result, each one different so dedup has nothing to say.
    fn tool_loop(n: usize) -> Value {
        let messages: Vec<Value> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    json!({"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": format!("t{i}"),
                        "content": format!("result {i} ").repeat(300),
                    }]})
                } else {
                    json!({"role": "assistant", "content": [{
                        "type": "tool_use", "id": format!("t{}", i + 1), "name": "run",
                        "input": {"step": i},
                    }]})
                }
            })
            .collect();
        json!({ "messages": messages })
    }

    fn pruned_messages(n: usize) -> Vec<Value> {
        let (out, _) = optimize_request(tool_loop(n), opts(false, false, true));
        out["messages"].as_array().unwrap().clone()
    }

    fn is_pruned(msg: &Value) -> bool {
        msg["content"][0]["content"] == PRUNE_PLACEHOLDER
    }

    #[test]
    fn prune_keeps_the_prefix_byte_identical_between_steps() {
        // Every request in a conversation extends the one before it. Prompt
        // caching reads the earlier request's prefix only if those bytes come
        // out the same again.
        let first = KEEP_RECENT_MESSAGES + PRUNE_STEP_MESSAGES;
        assert!(
            pruned_messages(first).iter().any(is_pruned),
            "the step under test must be one where pruning is active"
        );
        for n in first..first + PRUNE_STEP_MESSAGES - 1 {
            let earlier = serde_json::to_vec(&pruned_messages(n)).unwrap();
            let later = serde_json::to_vec(&pruned_messages(n + 1)).unwrap();
            // Drop the earlier array's closing bracket: what is left is
            // exactly the bytes the later request has to start with.
            let prefix = &earlier[..earlier.len() - 1];
            assert!(
                later.starts_with(prefix),
                "request of {} messages does not start with the pruned request of {n}",
                n + 1
            );
        }
    }

    #[test]
    fn prune_boundary_advances_a_whole_step_at_a_time() {
        let pruned_count = |n: usize| pruned_messages(n).iter().filter(|m| is_pruned(m)).count();
        let first = KEEP_RECENT_MESSAGES + PRUNE_STEP_MESSAGES;
        assert_eq!(pruned_count(first - 1), 0, "inside the first step");
        // Half the messages of a tool loop are tool results.
        assert_eq!(pruned_count(first), PRUNE_STEP_MESSAGES / 2);
        assert_eq!(
            pruned_count(first + PRUNE_STEP_MESSAGES),
            PRUNE_STEP_MESSAGES
        );
        // Whatever the length, the recent window is never touched.
        for n in 0..3 * PRUNE_STEP_MESSAGES {
            let msgs = pruned_messages(n);
            let recent = msgs.iter().rev().take(KEEP_RECENT_MESSAGES);
            assert!(!recent.into_iter().any(is_pruned), "{n} messages");
        }
    }

    /// A turn-scoped reminder, long enough for dedup, in the string form and
    /// in the block form.
    fn reminders() -> [Value; 2] {
        let text = "Run the test suite before you report, and never push. ".repeat(20);
        assert!(tokens::count(&text) >= DEDUP_MIN_TOKENS);
        [json!(text), json!([{"type": "text", "text": text}])]
            .map(|content| json!({"role": "system", "clear_at": "next_user_message", "content": content}))
    }

    #[test]
    fn dedup_never_elides_a_turn_scoped_reminder() {
        // One copy per tool round. The API renders only the newest: once a
        // user message follows, the earlier one is cleared for good. So the
        // newest copy is the only one the model ever reads.
        for reminder in reminders() {
            let req = json!({"messages": [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok 1"}]},
                reminder.clone(),
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "run", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok 2"}]},
                reminder.clone(),
            ]});
            let (out, applied) = apply_passes(req.clone(), opts(false, true, false));
            assert_eq!(out, req, "both copies reach the API as written");
            assert_eq!(applied, Applied::default());
        }
    }

    #[test]
    fn a_turn_scoped_reminder_is_not_an_earlier_block_to_point_at() {
        // The same text again in an ordinary message. "Identical to an
        // earlier block" would send the model to a block the API has cleared.
        for reminder in reminders() {
            let text = match &reminder["content"] {
                Value::String(s) => s.clone(),
                blocks => blocks[0]["text"].as_str().unwrap().to_owned(),
            };
            let repeat = json!({"role": "user", "content": [{"type": "text", "text": text}]});

            let after = json!({"messages": [
                {"role": "user", "content": "go"},
                reminder.clone(),
                {"role": "assistant", "content": "done"},
                repeat.clone(),
            ]});
            let (out, applied) = apply_passes(after.clone(), opts(false, true, false));
            assert_eq!(out, after);
            assert_eq!(applied, Applied::default());

            // And the other way round: an ordinary block first does not make
            // the reminder a repeat of it.
            let before = json!({"messages": [repeat.clone(), reminder.clone()]});
            let (out, _) = apply_passes(before.clone(), opts(false, true, false));
            assert_eq!(out, before);

            // Two ordinary copies are still deduped, reminder in between or not.
            let twice = json!({"messages": [repeat.clone(), reminder.clone(), repeat.clone()]});
            let (out, applied) = apply_passes(twice, opts(false, true, false));
            assert_eq!(applied.replaced, 1);
            assert_eq!(out["messages"][2]["content"][0]["text"], DEDUP_PLACEHOLDER);
        }
    }

    /// A large tool result whose content is blocks, the last of them carrying
    /// a breakpoint the client put there.
    fn result_with_a_nested_breakpoint(id: &str, text: &str) -> Value {
        json!({"type": "tool_result", "tool_use_id": id, "content": [
            {"type": "text", "text": text, "cache_control": ephemeral()}
        ]})
    }

    #[test]
    fn dedup_leaves_a_tool_result_that_nests_a_client_breakpoint() {
        // The placeholder replaces `content` whole, as a string. The
        // breakpoint inside would go with it.
        let big = "x ".repeat(400);
        let req = json!({"messages": [
            {"role": "user", "content": [result_with_a_nested_breakpoint("a", &big)]},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": [result_with_a_nested_breakpoint("a", &big)]},
        ]});
        let (out, applied) = apply_passes(req.clone(), opts(true, true, false));
        assert_eq!(out, req);
        assert_eq!(applied, Applied::default());

        // The same repeat without the nested breakpoints is elided, with the
        // block's own breakpoint left where it is.
        let plain = |marked: bool| {
            let mut block = json!({"type": "tool_result", "tool_use_id": "a", "content": [
                {"type": "text", "text": big}
            ]});
            if marked {
                block["cache_control"] = ephemeral();
            }
            json!({"role": "user", "content": [block]})
        };
        let req = json!({"messages": [plain(false), plain(true)]});
        let (out, applied) = apply_passes(req, opts(true, true, false));
        assert_eq!(applied.replaced, 1);
        let repeat = &out["messages"][1]["content"][0];
        assert_eq!(repeat["content"], DEDUP_PLACEHOLDER);
        assert_eq!(repeat["cache_control"], ephemeral());
    }

    #[test]
    fn prune_leaves_a_tool_result_that_nests_a_client_breakpoint() {
        let mut req = tool_loop(KEEP_RECENT_MESSAGES + PRUNE_STEP_MESSAGES);
        let kept = result_with_a_nested_breakpoint("t0", &"result 0 ".repeat(300));
        req["messages"][0]["content"][0] = kept.clone();

        let (out, applied) = apply_passes(req, opts(false, false, true));
        assert_eq!(out["messages"][0]["content"][0], kept);
        assert!(is_pruned(&out["messages"][2]), "its neighbours still go");
        assert_eq!(applied.replaced, PRUNE_STEP_MESSAGES / 2 - 1);
    }
}
