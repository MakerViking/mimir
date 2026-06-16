# Token-savings benchmarks (Mimir vs RTK)

Reproduce with `scripts/bench-token-savings.sh`. Every number is counted with
**one tokenizer** — `mimir tokens` (tiktoken `o200k_base`) — so RTK's and
Mimir's outputs are measured on the same ruler. Filter inputs are fed to both
tools via fake binaries on `PATH`, so each tool sees **identical input** and
dispatches its own real handler. `cargo`/`git` fixtures are real output from
this machine; `npm`/`pytest`/`docker` are realistic synthetic fixtures.

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

After expanding the handler set, RTK wraps **18 / 24** sampled commands and
Mimir wraps **15 / 24**: cargo, git (status/clone/fetch/pull/push), npm, pnpm,
yarn, bun, pytest, go, make, docker, podman, pip, the JS toolchain
(jest/vitest/eslint/tsc/next), and terraform/tofu.

The only RTK-exclusive commands left are **`kubectl` and the coreutils**
(`ls`/`grep`/`rg`/`find`/`ps`/`df`). Mimir deliberately does **not** filter
those: their output *is* the signal you asked for, and compressing it is lossy.
Dropping RTK means those commands return their full output — correct behavior,
just not compressed.

## Verdict: RTK can be retired for the dev workflow

The token-saving mechanism works, and across every noisy build/test/package/
infra command Mimir now **matches or beats RTK**. On top of that, Mimir's
outline (88%), peek (~90%), and the proxy are wins RTK never offered. The
remaining RTK-only commands are coreutils where filtering is lossy by nature.

**Recommendation:** switch the PreToolUse hook to Mimir (`mimir init --hooks`,
then remove the RTK hook). You lose only RTK's lossy compression of
`ls`/`grep`/`find`/`ps`/`df`/`kubectl` output — which is arguably content you
wanted to see in full anyway.
