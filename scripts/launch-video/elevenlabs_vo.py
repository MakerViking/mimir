#!/usr/bin/env python3
"""Generate the Mimir launch-video voiceover with ElevenLabs.

Voice: "Josh" (nzFihrBIvB34imQBuxub), model eleven_multilingual_v2.
Key: read from BookForge backend/.env (ELEVENLABS_API_KEY).
Output: gvo1.mp3 … gvo6.mp3 — consumed by synth.py.
NOTE: if line texts change, re-measure durations and retime video.html + synth.py.
"""
import json, urllib.request, subprocess, os

ENV = os.path.expanduser("~/Koding/projects/BookForge/backend/.env")
key = [l.strip().split("=", 1)[1] for l in open(ENV) if l.startswith("ELEVENLABS_API_KEY=")][0]
VID = "nzFihrBIvB34imQBuxub"
OUT = os.path.dirname(os.path.abspath(__file__))

LINES = {
    1: "Every new session, your AI coding agent wakes up with amnesia. Yesterday's gotcha — gone. Last month's architecture decision — gone.",
    2: "So you duct-tape a memory together: one tool for notes, another to search your docs, a third to map your code. Three stores, three indexes — and none of them talk to each other.",
    3: "Mimir is one memory — named for the Norse giant who guarded the well of wisdom. Notes, docs, and the code itself: one graph, one local file, one small binary.",
    4: "Ask in plain language. Hybrid keyword-and-semantic search finds it; the graph connects it — the decision, the function it touched, the doc that explains why. And it learns: what helps gets stronger, what doesn't fades away. It plugs into Claude Code — or any MCP agent — with one line.",
    5: "It's Rust. Recall lands in milliseconds — up to three hundred and sixty times faster than the tools it replaces. And it's completely local. No cloud, no API keys, zero telemetry. Your knowledge stays yours.",
    6: "One command, any platform. Give your agent a memory that lasts. Mimir — on GitHub today.",
}

for i, text in LINES.items():
    body = json.dumps({"text": text, "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3}}).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VID}?output_format=mp3_44100_128",
        data=body, headers={"xi-api-key": key, "Content-Type": "application/json"})
    path = f"{OUT}/gvo{i}.mp3"
    open(path, "wb").write(urllib.request.urlopen(req, timeout=120).read())
    d = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    print(f"gvo{i}: {float(d):.2f}s")
