# Mimir launch video — script v1 (approved draft)

**Format:** 1920×1080, ~80 seconds, motion-graphics style (no screen recording —
animated terminal and graph scenes built to match the logo's look).
Voiceover + music + light SFX.
**Audience:** developers using AI coding agents. **Goal:** "I want this" → install.

| # | Time | Scene | Voiceover |
|---|------|-------|-----------|
| 1 | 0:00–0:09 | **The amnesia.** Black screen, blinking terminal cursor. `new session started` types out. A day-counter flips Mon→Tue→Wed while the same question re-types: "so what does this codebase do?" | "Every new session, your AI coding agent wakes up with amnesia. Yesterday's gotcha — gone. Last month's architecture decision — gone." |
| 2 | 0:09–0:20 | **The duct tape.** Three boxes pop in — *notes*, *doc search*, *code graph* — each with its own database cylinder. Dashed wires connect them and visibly tangle. | "So you duct-tape a memory together: one tool for notes, another to search your docs, a third to map your code. Three stores, three indexes — and none of them talk to each other." |
| 3 | 0:20–0:31 | **Reveal.** The tangle collapses into a single glowing node; the Mimir logo resolves out of it. Tagline fades in: **"Memory for AI agents."** Music opens up. | "Mimir is one memory. Notes, docs, and the code itself — one graph, one local file, one small binary." |
| 4 | 0:31–0:50 | **How it works.** Animated graph: nodes tagged `memory` / `doc` / `symbol` light up and link. A query slides in — *"why did we pick SQLite?"* — beams traverse the graph, a result card pops out connecting a decision → a function → a doc chunk. Badges tick on: `BM25 + vectors` · `learns from use` · `MCP`. | "Ask in plain language. Hybrid keyword-and-semantic search finds it; the graph connects it — the decision, the function it touched, the doc that explains why. And it learns: what helps gets stronger, what doesn't fades away. It plugs into Claude Code — or any MCP agent — with one line." |
| 5 | 0:50–1:04 | **Fast & private.** The benchmark bars animate up (re-using `assets/benchmark.svg` styling); stat cards punch in: **7 ms recall** · **up to 360× faster**. Cut to a house outline around the database: **no cloud · no API keys · zero telemetry**. | "It's Rust. Recall lands in milliseconds — up to three hundred and sixty times faster than the tools it replaces. And it's completely local. No cloud, no API keys, zero telemetry. Your knowledge stays yours." |
| 6 | 1:04–1:20 | **CTA.** Terminal types the real install one-liner, green ✓, then `mimir recall` returns instantly. End card: logo + **github.com/MakerViking/mimir** + "Give your agent a memory that lasts." | "One command, any platform. Give your agent a memory that lasts. Mimir — on GitHub today." |

**Audio:** low ambient synth pulse through scenes 1–2, opens into a warmer
progression at the reveal, ducks under VO, bright resolve on the end card.
SFX: soft keyboard ticks (scene 1), whoosh at the reveal, subtle UI pops for
badges/stats.

**Claims check:** everything spoken is already public in the README — 7 ms
recall and 360× come from measured benchmarks (`assets/benchmark.svg`);
"zero telemetry" and "any MCP agent" are README claims verbatim.

**Open production decisions (pending user):**
1. Voice: edge-tts (neural, online) vs Piper (local). Lean: edge-tts.
2. Music: procedural ambient (license-clean) vs user-supplied CC0 track.
3. Optional Norse flavor line at reveal: "Named for the Norse giant who
   guarded the well of wisdom."
