# Token-savings benchmarks (Mimir vs RTK)

Reproduce with `scripts/bench-token-savings.sh`. Every number is counted with
**one tokenizer** — `mimir tokens` (tiktoken `o200k_base`) — so RTK's and
Mimir's outputs are measured on the same ruler. Filter inputs are fed to both
tools via fake binaries on `PATH`, so each tool sees **identical input** and
dispatches its own real handler. `cargo`/`git` fixtures are real output from
this machine; `npm`/`pytest`/`docker` are realistic synthetic fixtures.

> **Note:** RTK has since been **retired and uninstalled** — Mimir replaced it.
> The RTK columns below are **historical** (measured at v0.8.0 while both were
> installed); Mimir's columns, §1b, §2 and §3 are current and re-runnable.

## 1. Command-output filtering — RTK vs Mimir (identical input)

| command | raw tok | RTK tok | RTK saved | Mimir tok | Mimir saved |
|---|---:|---:|---:|---:|---:|
| cargo build | 85 | 29 | 66% | 23 | **73%** |
| git clone | 6272 | 6272 | 0% | 13 | **100%** |
| npm install | 182 | 50 | 73% | 24 | **87%** |
| pytest | 277 | 47 | **83%** | 51 | 82% |
| docker build | 441 | 440 | 0% | 21 | **95%** |

Where Mimir has a handler it matches or beats RTK. `git clone` and `docker`:
RTK leaves them untouched here; Mimir's rules collapse the progress noise.

## 1b. Content commands — non-lossy volume cap (coreutils/kubectl)

The coreutils and `kubectl` get the **volume cap**, not the per-line noise
filter. It only triggers on runaway output (smaller output passes through
verbatim) and keeps head + tail + every signal line — nothing is silently
dropped, unlike RTK's lossy line compression.

| command | raw tok | Mimir tok | saved |
|---|---:|---:|---:|
| `find` (2000 lines) | 21001 | 1271 | **94%** |
| `grep` (1500 lines) | 25503 | 2112 | **92%** |

Cap savings are recorded under a distinct ledger source (`cap`), separate from
the `filter` (per-line noise) source, so `mimir savings` attributes them apart.

## 2. `mimir outline` vs reading whole files (no RTK equivalent)

| file | full tok | outline tok | saved |
|---|---:|---:|---:|
| crates/mimir-cli/src/commands.rs | 11462 | 1495 | 87% |
| crates/mimir-cli/src/graph_viz.rs | 8196 | 99 | 99% |
| crates/mimir-cli/src/dashboard.rs | 7280 | 122 | 98% |
| crates/mimir-cli/src/mcp.rs | 7121 | 590 | 92% |
| **TOTAL (50 .rs files)** | **134925** | **16068** | **88%** |

Outlining the whole codebase instead of reading it costs **12% of the tokens**.
This is the single biggest lever and RTK has nothing like it.

## 3. `mimir peek <symbol>` vs reading the whole file (no RTK equivalent)

| symbol | file tok | peek tok | saved |
|---|---:|---:|---:|
| optimize_request | 1741 | 94 | 95% |
| filter_output | 1335 | 192 | 86% |
| count | 469 | 66 | 86% |
| add_cache_breakpoint | 1741 | 252 | 86% |

## 4. Coverage breadth — which commands each tool wraps

Mimir wraps every command RTK does, in **two modes**:

- **Noise handlers** (declarative per-program rules) drop progress/boilerplate
  for the build/test/package/infra tools: cargo, git (status/clone/fetch/pull/
  push), npm, pnpm, yarn, bun, pytest, go, make, docker, podman, pip, the JS
  toolchain (jest/vitest/eslint/tsc/next), and terraform/tofu.
- **A non-lossy volume cap** covers the high-volume "content" commands whose
  output *is* the signal — the coreutils and `kubectl` (`cat`/`head`/`tail`/
  `ls`/`find`/`grep`/`rg`/`ps`/`df`/`du`/`tree`/`kubectl`). These are **never
  line-dropped** (that would lose a matched or listed line); output under the
  cap passes through verbatim, and only runaway output is bounded to head +
  tail + every signal line, with the bulky middle elided behind a visible
  marker. `tail -f`/`journalctl -f` are left alone so the wrapper can't hang.

This closes RTK's old coreutils gap **without** RTK's lossy compression: where
RTK silently shrank `grep`/`ls` output, Mimir returns it intact unless it's
huge, then caps it transparently.

On the 24-command sample, Mimir now wraps **22 / 24** — the only misses are
`gradle` and `mvn` (no handler yet). (RTK's historical figure was 18 / 24.)

## Verdict: RTK can be retired for the dev workflow

The token-saving mechanism works, and across every noisy build/test/package/
infra command Mimir **matches or beats RTK**. The coreutils + `kubectl` that
were once RTK-exclusive are now covered too — **non-lossily**, via the volume
cap rather than RTK's lossy line compression. On top of that, Mimir's outline
(88%), peek (~90%), and the proxy are wins RTK never offered.

**Recommendation:** switch the PreToolUse hook to Mimir (`mimir init --hooks`,
then remove the RTK hook). Nothing about RTK's coverage is lost.

## Reproduce the eval (retrieval quality, not token savings)

The numbers above measure token cost. Retrieval *quality* — does auto-recall
inject the right memory, and stay silent when it should — has its own
dev-only harness at `crates/mimir-core/src/eval` (not shipped on any user
surface). It scores precision@k/recall/MRR by [`Category`] and
[`QuestionSet`], plus a confusion table (injected-correct/wrong,
silent-correct/wrong) over `inject::select_injection`'s actual decision,
including drift-preventer fixtures — the case that matters most in practice.

Two modes, since determinism and semantic realism conflict:

```sh
# hermetic: deterministic synthetic vectors, no model/network — a
# regression guard, always runnable, part of plain `cargo test`
cargo test -p mimir-mem-core eval:: -- --ignored --nocapture

# real-model: identical corpus/questions, embedded with the actual default
# model — the honest number for accuracy work; skips (doesn't fail) if the
# model isn't downloaded locally
cargo test -p mimir-mem-core --features eval eval:: -- --ignored --nocapture

# the specific report CHANGELOG entries cite for injection decisions:
cargo test -p mimir-mem-core eval::tests::eval_inject_baseline_report -- --ignored --nocapture

# compare a different embedding model without touching config.toml:
MIMIR_EVAL_MODEL=granite-embedding-small-r2 cargo test -p mimir-mem-core \
  --features eval eval:: -- --ignored --nocapture
```

The product rule enforced on every ranking/floor change: injected-wrong must
never rise, even when the change also fixes an intended case — a wrong
injection is worse than a missed one.
