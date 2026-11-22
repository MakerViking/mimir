//! Apply what the passes changed to the client's own JSON text.
//!
//! The passes work on a parsed `serde_json::Value`, and writing a `Value`
//! back out is not the inverse of parsing it: object keys come out sorted,
//! and a number too large for 64 bits comes out as a float. Neither is
//! something a pass decided to do. Key order is part of what the API's
//! prompt cache matches on (the caching guide warns about unstable key order
//! in `tool_use` blocks), so a body that is sorted on the turns a pass
//! changed something and in the client's order on the turns none did would
//! miss the cache every time it switches.
//!
//! So a changed request is written as the client's text with only the
//! changed values replaced: [`rewrite`] walks the original text and the
//! optimized value side by side and copies every value the passes left alone
//! byte for byte.

use std::collections::HashMap;

use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;
use serde_json::Value;

/// `original` with every value that differs in `after` replaced, and nothing
/// else touched: fields stay in the client's order, untouched values keep
/// their exact text. `after` must be what the passes made of `original`.
///
/// `None` when that cannot be done faithfully — the result is checked by
/// parsing it back, and anything but exactly `after` is discarded. The caller
/// then serializes `after` whole, which is always correct, only not
/// order-preserving.
pub(crate) fn rewrite(original: &[u8], after: &Value) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(original).ok()?;
    let before: Value = serde_json::from_str(text).ok()?;
    let raw: &RawValue = serde_json::from_str(text).ok()?;
    let mut out = String::with_capacity(text.len() + 256);
    write_changed(raw, &before, after, &mut out)?;
    verified(out, after)
}

/// `out` as bytes, if it reads back as exactly `after`. The one check
/// between whatever [`write_changed`] wrote and the API: a body that fails
/// it is never sent.
fn verified(out: String, after: &Value) -> Option<Vec<u8>> {
    let reread: Value = serde_json::from_str(&out).ok()?;
    (reread == *after).then(|| out.into_bytes())
}

/// Write `after` to `out`, taking the text of `raw` wherever `after` still
/// equals `before` (the parsed form of `raw`). `None` if the three do not
/// line up, which a pass that only adds fields and replaces values never
/// causes.
fn write_changed(raw: &RawValue, before: &Value, after: &Value, out: &mut String) -> Option<()> {
    if before == after {
        out.push_str(raw.get());
        return Some(());
    }
    match (before, after) {
        (Value::Object(old), Value::Object(new)) => {
            let fields: Fields = serde_json::from_str(raw.get()).ok()?;
            // Where each key is written last. A key written twice has one
            // value, the last one's: that is what `before` and `after` hold,
            // and the only occurrence there is anything to compare with.
            let last: HashMap<&str, usize> = fields
                .0
                .iter()
                .enumerate()
                .map(|(i, (key, _))| (key.as_str(), i))
                .collect();
            out.push('{');
            let mut first = true;
            let mut separate = |out: &mut String| {
                if !std::mem::take(&mut first) {
                    out.push(',');
                }
            };
            for (i, (key, child)) in fields.0.iter().enumerate() {
                separate(out);
                out.push_str(&serde_json::to_string(key).ok()?);
                out.push(':');
                if last[key.as_str()] == i {
                    write_changed(child, old.get(key)?, new.get(key)?, out)?;
                } else {
                    // An earlier, shadowed occurrence: the client's text, as
                    // it is. Whatever the API makes of a repeated key, it
                    // makes the same of it with the proxy in between.
                    out.push_str(child.get());
                }
            }
            // Fields a pass added go after the client's own.
            for (key, value) in new.iter().filter(|(key, _)| !old.contains_key(*key)) {
                separate(out);
                out.push_str(&serde_json::to_string(key).ok()?);
                out.push(':');
                out.push_str(&serde_json::to_string(value).ok()?);
            }
            out.push('}');
        }
        (Value::Array(old), Value::Array(new)) if old.len() == new.len() => {
            let items: Vec<&RawValue> = serde_json::from_str(raw.get()).ok()?;
            if items.len() != new.len() {
                return None;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_changed(item, &old[i], &new[i], out)?;
            }
            out.push(']');
        }
        // A different kind of value, or a different one of the same kind:
        // nothing of the original is left to keep.
        _ => out.push_str(&serde_json::to_string(after).ok()?),
    }
    Some(())
}

/// A JSON object as its text has it: fields in the order written, each value
/// as its own untouched text.
struct Fields<'a>(Vec<(String, &'a RawValue)>);

impl<'de> Deserialize<'de> for Fields<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InOrder;

        impl<'de> Visitor<'de> for InOrder {
            type Value = Fields<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut fields = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(field) = map.next_entry::<String, &'de RawValue>()? {
                    fields.push(field);
                }
                Ok(Fields(fields))
            }
        }

        deserializer.deserialize_map(InOrder)
    }
}

#[cfg(test)]
mod tests {
    use super::{rewrite, verified};
    use serde_json::{json, Value};

    /// `rewrite` as text, with `edit` standing in for the passes.
    fn rewritten(original: &str, edit: impl FnOnce(&mut Value)) -> Option<String> {
        let mut after: Value = serde_json::from_str(original).unwrap();
        edit(&mut after);
        rewrite(original.as_bytes(), &after).map(|bytes| String::from_utf8(bytes).unwrap())
    }

    #[test]
    fn untouched_values_keep_their_order_and_their_text() {
        // Keys out of alphabetical order at every level, a number no 64-bit
        // type holds, an escape and a float a serializer would respell.
        let original = r#"{"model":"m","max_tokens":1.0,"big":12345678901234567890123,"system":"be careful","alpha":{"z": 1, "a": [3,  2]},"messages":[{"role":"user","content":"café"}]}"#;
        let out = rewritten(original, |v| {
            v["system"] = json!([{"type": "text", "text": "be careful", "cache_control": {"type": "ephemeral"}}]);
        })
        .unwrap();
        assert_eq!(
            out,
            r#"{"model":"m","max_tokens":1.0,"big":12345678901234567890123,"system":[{"cache_control":{"type":"ephemeral"},"text":"be careful","type":"text"}],"alpha":{"z": 1, "a": [3,  2]},"messages":[{"role":"user","content":"café"}]}"#
        );
    }

    #[test]
    fn a_changed_block_keeps_its_other_fields_as_they_were() {
        // The arguments of a tool call are the model's own JSON: their key
        // order is part of the prompt.
        let original = r#"{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"run","input":{"zeta":1,"alpha":{"b":2,"a":1}}}]}]}"#;
        let out = rewritten(original, |v| {
            v["messages"][0]["content"][0]["cache_control"] = json!({"type": "ephemeral"});
        })
        .unwrap();
        assert_eq!(
            out,
            r#"{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"run","input":{"zeta":1,"alpha":{"b":2,"a":1}},"cache_control":{"type":"ephemeral"}}]}]}"#
        );
    }

    #[test]
    fn a_replaced_value_is_the_only_thing_that_changes() {
        let original = r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a long result"},{"type":"text","text":"and a question"}]},{"role":"user","content":"again"}]}"#;
        let out = rewritten(original, |v| {
            v["messages"][0]["content"][0]["content"] = json!("[elided]");
            v["messages"][1]["content"] = json!("[elided]");
        })
        .unwrap();
        assert_eq!(
            out,
            r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"[elided]"},{"type":"text","text":"and a question"}]},{"role":"user","content":"[elided]"}]}"#
        );
    }

    #[test]
    fn an_unset_field_that_a_pass_sets_is_replaced_in_place() {
        let original = r#"{"system":[{"type":"text","cache_control":null,"text":"x"}]}"#;
        let out = rewritten(original, |v| {
            v["system"][0]["cache_control"] = json!({"type": "ephemeral"});
        })
        .unwrap();
        assert_eq!(
            out,
            r#"{"system":[{"type":"text","cache_control":{"type":"ephemeral"},"text":"x"}]}"#
        );
    }

    #[test]
    fn a_repeated_key_is_passed_on_as_the_client_wrote_it() {
        // Whatever the API makes of a repeated key, it makes the same of it
        // with the proxy in between.
        let original = r#"{"a":1,"a":2,"b":0}"#;
        let out = rewritten(original, |v| v["b"] = json!(5)).unwrap();
        assert_eq!(out, r#"{"a":1,"a":2,"b":5}"#);
    }

    /// What the cache pass makes of `original`, written back over its text.
    fn after_the_cache_pass(original: &str) -> Option<String> {
        let before: Value = serde_json::from_str(original).unwrap();
        let opts = crate::OptimizeOpts {
            cache: true,
            cache_ttl: crate::CacheTtl::FiveMin,
            dedup: true,
            prune: false,
        };
        let (after, applied) = crate::optimize::apply_passes(before, opts);
        assert!(applied.changed());
        rewrite(original.as_bytes(), &after).map(|bytes| String::from_utf8(bytes).unwrap())
    }

    #[test]
    fn a_shadowed_repeat_of_the_changed_key_is_left_as_written() {
        // `content` twice in one message. The value everyone reads is the
        // last one; the first is text nobody parsed, and stays what it was.
        let original = r#"{"messages":[{"role":"user","content":"A","content":"hello"}]}"#;
        assert_eq!(
            after_the_cache_pass(original).as_deref(),
            Some(
                r#"{"messages":[{"role":"user","content":"A","content":[{"cache_control":{"type":"ephemeral"},"text":"hello","type":"text"}]}]}"#
            )
        );
    }

    #[test]
    fn a_shadowed_repeat_of_another_shape_does_not_cost_the_whole_body_its_text() {
        // The shadowed `content` has one block, the live one two. Refusing
        // here would mean serializing the body whole: keys sorted, `big` a
        // float.
        let head = r#"{"model":"m","tools":[{"name":"t","input_schema":{"type":"object","properties":{"zeta":{},"alpha":{}}}}],"big":12345678901234567890123,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}],"content":[{"type":"text","text":"go"},{"type":"text","text":"now""#;
        let original = format!("{head}}}]}}]}}");
        assert_eq!(
            after_the_cache_pass(&original),
            Some(format!(
                r#"{head},"cache_control":{{"type":"ephemeral"}}}}]}}]}}"#
            ))
        );
    }

    #[test]
    fn an_array_of_another_length_is_written_whole() {
        // Positions no longer line up, so none of its text is reused. A
        // pass never does this; the answer is still exact.
        let out = rewritten(r#"{"k":0,"a":[{"z":1,"y":2}]}"#, |v| {
            v["a"].as_array_mut().unwrap().push(json!(3));
        });
        assert_eq!(out.as_deref(), Some(r#"{"k":0,"a":[{"y":2,"z":1},3]}"#));
    }

    #[test]
    fn text_that_does_not_read_back_as_the_optimized_value_is_never_sent() {
        let after = json!({"a": 1, "b": [true]});
        assert_eq!(
            verified(r#"{"b":[true],"a":1}"#.to_owned(), &after).as_deref(),
            Some(&br#"{"b":[true],"a":1}"#[..]),
            "the same value in another order is the point of the exercise"
        );
        // A wrong patch: well-formed, and not what the passes produced.
        assert_eq!(verified(r#"{"a":1,"b":[false]}"#.to_owned(), &after), None);
        assert_eq!(verified(r#"{"a":1}"#.to_owned(), &after), None);
        // A broken one.
        assert_eq!(verified(r#"{"a":1,"b":[true]"#.to_owned(), &after), None);
    }

    #[test]
    fn what_cannot_be_rewritten_faithfully_is_refused() {
        // A field gone: its text can neither be kept nor replaced.
        assert_eq!(
            rewritten(r#"{"a":1,"b":2}"#, |v| {
                v.as_object_mut().unwrap().remove("b");
            }),
            None
        );
        // An original that is not JSON at all.
        assert_eq!(rewrite(b"not json", &json!({"a": 1})), None);
    }
}
