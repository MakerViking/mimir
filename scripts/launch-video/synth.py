#!/usr/bin/env python3
"""Procedural soundtrack for the Mimir launch video (ElevenLabs 'Josh' timeline, 93.5s)."""
import numpy as np, wave, subprocess, os

SR = 44100
DUR = 93.5
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(42)

def env_ar(n, a, r):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na)
    e[n - nr:] = np.linspace(1, 0, nr)
    return e

def place(buf, sig, at):
    i = int(at * SR)
    j = min(i + len(sig), N)
    if i < N:
        buf[i:j] += sig[: j - i]

def tone(freq, dur, harmonics=((1, 1.0), (2, .35), (3, .15)), detune=0.15):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for mult, amp in harmonics:
        for d in (-detune, detune):
            s += amp * np.sin(2 * np.pi * (freq * mult + d) * tt)
    return s / np.abs(s).max()

# ---------------- music ----------------
music = np.zeros(N)

# dark intro drone (0 - 27.5s): A1 + A2 with slow pulse
drone_n = int(27.5 * SR)
dt_ = np.arange(drone_n) / SR
drone = (np.sin(2 * np.pi * 55 * dt_) * .6 + np.sin(2 * np.pi * 110.15 * dt_) * .35
         + np.sin(2 * np.pi * 164.9 * dt_) * .12)
pulse = 0.72 + 0.28 * np.sin(2 * np.pi * 0.55 * dt_ - np.pi / 2)
drone *= pulse * env_ar(drone_n, 2.0, 2.5)
place(music, drone * .30, 0)

# chord pads from the reveal (27.0s): Am F C G, looped to ~81; then A maj outro
NOTE = {'A2':110.0,'C3':130.81,'E3':164.81,'F2':87.31,'A3':220.0,'C4':261.63,
        'F3':174.61,'G2':98.0,'B2':123.47,'D3':146.83,'G3':196.0,'E4':329.63,
        'C#3':138.59}
CHORDS = [('A2','C3','E3','A3'), ('F2','F3','A2','C3'), ('C3','E3','G3','C4'), ('G2','B2','D3','G3')]
pos = 27.0
ci = 0
while pos < 81.0:
    notes = CHORDS[ci % 4]
    dur = 4.1
    chord = np.zeros(int(dur * SR))
    for nm in notes:
        chord += tone(NOTE[nm], dur) * .25
    chord *= env_ar(len(chord), 1.2, 1.4)
    place(music, chord * .32, pos)
    pos += 4.0
    ci += 1
# outro: bright A major (83.7 - 93.3)
outro = np.zeros(int(9.6 * SR))
for nm, amp in (('A2', .3), ('C#3', .22), ('E3', .22), ('A3', .18), ('E4', .10)):
    outro += tone(NOTE[nm], 9.6) * amp
outro *= env_ar(len(outro), 1.8, 3.8)
place(music, outro * .36, 83.7)

# gentle arp during the graph scene (42 - 64.5)
arp_notes = ['A3', 'C4', 'E3', 'G3', 'A3', 'E3', 'C4', 'G3']
for k, at in enumerate(np.arange(42.0, 64.5, 0.5)):
    f = NOTE[arp_notes[k % 8]] * 2
    n = int(.42 * SR)
    tt = np.arange(n) / SR
    pluck = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9)
    place(music, pluck * .05, at)

# ---------------- sfx ----------------
sfx = np.zeros(N)

def ticks(start, end):
    for at in np.arange(start, end, 0.085):
        n = int(.012 * SR)
        click = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.002 * SR))
        place(sfx, click * .035, at + float(rng.uniform(-0.01, 0.01)))

for a, b in ((0.9, 2.0), (3.3, 4.6), (5.9, 7.2), (8.3, 9.6), (84.3, 86.7)):
    ticks(a, b)

# whoosh into the reveal (26.8 - 28.6)
n = int(1.8 * SR)
tt = np.arange(n) / SR
noise = rng.standard_normal(n)
sweep = np.sin(np.pi * tt / 1.8) ** 2
body = np.convolve(noise, np.ones(48) / 48, mode='same')
place(sfx, body * sweep * .16, 26.8)

# soft bell when the logo lands (28.7)
n = int(2.2 * SR)
tt = np.arange(n) / SR
bell = (np.sin(2 * np.pi * 880 * tt) + .5 * np.sin(2 * np.pi * 1318.5 * tt)) * np.exp(-tt * 2.4)
place(sfx, bell * .07, 28.7)

# UI pops (badges, stats, privacy lines)
for at in (56.0, 58.5, 62.1, 67.6, 68.4, 69.2, 70.8, 76.6, 78.0, 79.4):
    n = int(.09 * SR)
    tt = np.arange(n) / SR
    blip = np.sin(2 * np.pi * (700 + 500 * tt / .09) * tt) * np.exp(-tt * 40)
    place(sfx, blip * .06, at)

# ---------------- voiceover (ElevenLabs gvoN.mp3) ----------------
vo = np.zeros(N)
VO_AT = [(1, 0.8), (2, 13.1), (3, 27.4), (4, 41.8), (5, 66.1), (6, 84.2)]
durs = {}
for idx, at in VO_AT:
    wavp = f"/tmp/mimir-video/gvo{idx}.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", f"/tmp/mimir-video/gvo{idx}.mp3",
                    "-ar", str(SR), "-ac", "1", wavp], check=True)
    with wave.open(wavp) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    durs[idx] = len(data) / SR
    place(vo, data * .95, at)

# duck music under VO
duck = np.ones(N)
ramp = int(0.35 * SR)
for idx, at in VO_AT:
    i, j = int(at * SR), int((at + durs[idx]) * SR)
    duck[i:j] = 0.42
kernel = np.ones(ramp) / ramp
duck = np.convolve(duck, kernel, mode='same')

# ---------------- mix ----------------
mix = music * duck + sfx + vo
mix = mix / np.abs(mix).max() * 0.85
stereo = np.stack([mix, mix], axis=1)
pcm = (stereo * 32767).astype(np.int16)
with wave.open("/tmp/mimir-video/audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("audio.wav written:", os.path.getsize("/tmp/mimir-video/audio.wav") // 1024, "KB")
