# `mimir proxy` — optional local API proxy

Mimir's proxy sits between your AI client and `api.anthropic.com` and optimizes
the **request body** to cut token cost, then records what it saved. It is
**off by default** — you start it explicitly — and it is our own
implementation, not derived from any other tool.

## What it actually does (and doesn't)

You cannot compress text and have the API bill fewer tokens — the API tokenizes
whatever it receives. So the proxy only saves money by *changing request
content*:

- **Prompt-cache breakpoints** (default **on**, safe). If your client set **no**
  `cache_control` anywhere, the proxy adds ephemeral breakpoints on the system
  prompt and the last message — the stable prefix that repeats across turns.
  Cached input is billed at ~10%. Savings are **measured**, not estimated: the
  proxy reads `usage.cache_read_input_tokens` from the response and records the
  real reduction — and only on requests where it actually introduced the
  caching (if your client already caches, e.g. Claude Code, it does nothing).
- **Cache TTL** (`--cache-ttl`, `[proxy] cache_ttl`, default `"5m"`). The
  breakpoints above are written with the API's default 5-minute lifetime. Set
  `"1h"` to keep them alive across longer gaps. This is **not** a free upgrade:
  a cache write costs 1.25x input at 5m and **2x at 1h**, while a read is ~0.1x
  either way — so 5m breaks even on the 2nd request and 1h only on the 3rd.
  Choose `"1h"` only when your turns idle more than 5 minutes apart (an agent
  waiting on a human, a bursty batch job); for back-to-back work it is strictly
  more expensive. Because the proxy measures real `cache_read_input_tokens`,
  you can A/B the two against `mimir savings` instead of guessing. Only applies
  to breakpoints *we* add — a client that manages its own caching is untouched.
- **Block dedup** (default **on**, safe/lossless). When the same large content
  block (a re-read file, a repeated tool result) appears more than once, later
  copies are replaced with `[identical to an earlier block …]`. The model still
  sees the content once. Disable with `--no-dedup` / `[proxy] dedup = false`.
- **Prune stale tool results** (default **off**, lossy). Replaces large
  `tool_result` blocks in older turns with a short placeholder. This changes
  what the model sees, so it's opt-in (`--prune` or `[proxy] prune = true`).
- **Runaway circuit breaker** (`--max-request-tokens N`, `[proxy]
  max_request_tokens`, default `0` = off). Rejects a `/v1/messages` request
  whose estimated input exceeds `N` tokens, returning an Anthropic-shaped
  `request_too_large` error so your SDK raises a readable exception. The
  request is **not** forwarded and you are not billed for it. It **rejects,
  never truncates** — silently dropping content would change what the model
  sees without telling anyone, and could drop the actual question. Checked
  *after* dedup/prune, so those get a chance to bring a request back under the
  line first. The estimate counts `system`, message text, and the `tools` array
  (tool definitions are input too), and deliberately errs **low** — a false
  rejection is worse than firing slightly late. Leave it off unless you're
  guarding against a specific runaway loop.

Everything else — every other path, method, header and the entire streaming
(SSE) response — is forwarded **verbatim**. Your `x-api-key` / auth headers are
passed through untouched; the proxy never reads or stores your key.

## Security

The proxy is a **man-in-the-middle on your prompts and completions**. Run it
only on `127.0.0.1` (the default). It buffers the `POST /v1/messages` body to
rewrite it; responses are streamed, never buffered.

## Usage

```sh
# start it (own process; leave it running)
mimir proxy                      # 127.0.0.1:8788 — cache on, dedup on, prune off
mimir proxy --dry-run            # measure only — forward bodies unchanged
mimir proxy --prune              # also enable lossy tool-result pruning
mimir proxy --no-cache           # disable the cache pass
mimir proxy --cache-ttl 1h       # 1-hour breakpoints (2x write cost — see above)
mimir proxy --no-dedup           # disable the dedup pass
mimir proxy --max-request-tokens 400000   # reject runaway requests (0 = off)

# point your client at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
```

Then watch the effect:

```sh
mimir savings              # proxy_cache / proxy_dedup / proxy_prune rows, in $
mimir savings --oneline    # compact segment for a statusline
```

## Config (`[proxy]` in config.toml)

```toml
[proxy]
bind     = "127.0.0.1:8788"
upstream = "https://api.anthropic.com"
cache    = true     # add cache breakpoints when the client set none
cache_ttl = "5m"    # "5m" (default) or "1h" — 1h doubles the write cost
dedup    = true     # elide later identical large blocks (lossless)
prune    = false    # lossy: elide stale tool results
max_request_tokens = 0   # 0 = off; above this, reject (never truncate)

[savings]
input_price_per_mtok = 3.0   # set to your model's input price (e.g. 15.0 for Opus)
```

## Caveats

- `proxy_cache` savings are **measured** from `usage.cache_read_input_tokens`
  (90% of the cached input), recorded only when the proxy introduced the
  caching. `proxy_dedup`/`proxy_prune` are measured at the request.
- It mirrors the Anthropic request shape; a future breaking API change could
  require an update. The proxy fails safe — on any parse error it forwards the
  original body unchanged.
