"""Sampling döngüsünü KOD ÜZERİNDE adım adım gösterir: hangi satır çalıştı, ne değişti.

Sadece görselleştirme. makemore.py'ye dokunmaz, veri hazırlığını kendi içinde kurar.
Çıktı: output/code_sampling_animation.gif (+ .mp4)
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
SEED = 2147483647
N_NAMES = 2

# ---------------------------------------------------------------- veri
with open(os.path.join(BASE_DIR, "names.txt")) as f:
    words = [line.strip() for line in f]

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

P = N.float()
P = P / P.sum(1, keepdim=True)

# ---------------------------------------------------------------- gösterilecek kod
CODE = [
    "g = torch.Generator().manual_seed(2147483647)",
    "",
    "for i in range(5):",
    "    ix = 0",
    "    out = []",
    "    while True:",
    "        p = P[ix]",
    "        ix = torch.multinomial(p, num_samples=1,",
    "                               replacement=True,",
    "                               generator=g).item()",
    "        out.append(itos[ix])",
    "        if ix == 0:",
    "            break",
    "    print(\"\".join(out))",
]
L_GEN, L_FOR, L_IX0, L_OUT0, L_WHILE = 0, 2, 3, 4, 5
L_P, L_MULTI, L_APPEND, L_IF, L_BREAK, L_PRINT = 6, 7, 10, 11, 12, 13

# ------------------------------------------------- yürütme izini önden çıkar
steps = []          # her kare: hangi satır(lar), state, ne değişti
g = torch.Generator().manual_seed(SEED)


def snap(lines, state, changed, note=""):
    steps.append(dict(lines=lines, state=dict(state), changed=changed, note=note))


state = dict(i="-", ix="-", p="-", out="[]", name="")
snap([L_GEN], state, "g", "uretec tohumlandi, akis basa sarildi")

for i in range(N_NAMES):
    state["i"] = str(i)
    snap([L_FOR], state, "i", f"{i}. isim baslıyor")

    ix = 0
    state["ix"] = "0  ('.')"
    state["p"] = "-"
    snap([L_IX0], state, "ix", "baslangic: '.' satirindan gidilecek")

    out = []
    state["out"] = "[]"
    state["name"] = ""
    snap([L_OUT0], state, "out", "harfler burada birikecek")

    while True:
        snap([L_WHILE], state, "", "dongu basi")

        p = P[ix]
        top = torch.topk(p, 2)
        top_s = ", ".join(f"'{itos[j]}' {v:.3f}"
                          for v, j in zip(top.values.tolist(), top.indices.tolist()))
        state["p"] = f"P[{ix}] -> 27 olasilik\n         en yuksek: {top_s}"
        snap([L_P], state, "p", f"'{itos[ix]}' satiri okundu (P degismedi, sadece okundu)")

        nix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        old = ix
        ix = nix
        state["ix"] = f"{ix}  ('{itos[ix]}')"
        snap([L_MULTI], state, "ix",
             f"zar atildi: {old} -> {ix}   olasilik {p[ix]:.4f}")

        out.append(itos[ix])
        state["out"] = "[" + ", ".join(f"'{c}'" for c in out) + "]"
        state["name"] = "".join(out)
        snap([L_APPEND], state, "out", f"'{itos[ix]}' isme eklendi")

        if ix == 0:
            snap([L_IF], state, "", "ix == 0  ->  DOGRU")
            snap([L_BREAK], state, "", "dongu kirildi")
            break
        snap([L_IF], state, "", f"ix == 0  ->  YANLIS (ix={ix}), donguye devam")

    snap([L_PRINT], state, "", f'ekrana basildi: "{state["name"]}"')

steps += [dict(steps[-1]) for _ in range(2)]

# ---------------------------------------------------------------- figür
fig = plt.figure(figsize=(16, 9), dpi=100)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.55],
              wspace=0.08, hspace=0.20, left=0.03, right=0.985, top=0.90, bottom=0.05)

ax_code = fig.add_subplot(gs[:, 0])
ax_state = fig.add_subplot(gs[0, 1])
ax_note = fig.add_subplot(gs[1, 1])
for a in (ax_code, ax_state, ax_note):
    a.axis("off")

ax_code.set_xlim(0, 1)
ax_code.set_ylim(0, 1)
ax_code.set_title("makemore.py — sampling dongusu", fontsize=13, loc="left")

TOP, DY = 0.96, 0.062
code_texts = []
for k, line in enumerate(CODE):
    t = ax_code.text(0.06, TOP - k * DY, line, va="top", ha="left",
                     fontsize=13.5, family="monospace", color="#555555")
    code_texts.append(t)
    ax_code.text(0.005, TOP - k * DY, f"{k + 1:2d}", va="top", ha="left",
                 fontsize=11, family="monospace", color="#bbbbbb")

hl = Rectangle((0.03, 0), 0.96, DY * 0.92, facecolor="#ffe08a", alpha=0.75, zorder=0)
ax_code.add_patch(hl)

state_txt = ax_state.text(0.0, 0.98, "", va="top", ha="left", fontsize=14,
                          family="monospace", linespacing=2.0)
name_txt = ax_note.text(0.0, 0.72, "", va="top", ha="left", fontsize=26,
                        family="monospace", color="#0b3d62")
note_txt = ax_note.text(0.0, 0.30, "", va="top", ha="left", fontsize=15,
                        family="monospace", color="#1a7f37")
suptitle = fig.suptitle("", fontsize=16)


def update(k):
    st = steps[k]
    line = st["lines"][0]

    for j, t in enumerate(code_texts):
        t.set_color("#111111" if j == line else "#999999")
        t.set_fontweight("bold" if j == line else "normal")
    hl.set_bounds(0.03, TOP - line * DY - DY * 0.88, 0.96, DY * 0.92)

    s = st["state"]
    rows = [("i", s["i"]), ("ix", s["ix"]), ("p", s["p"]), ("out", s["out"])]
    body = "\n".join(
        f"{'>' if name == st['changed'] else ' '} {name:<4}= {val}"
        for name, val in rows
    )
    state_txt.set_text("DEGISKENLER\n\n" + body)

    name_txt.set_text(f'isim: {s["name"]}_' if s["name"] else "isim: _")
    note_txt.set_text(st["note"])
    suptitle.set_text(f"Satir {line + 1} calisti   —   adim {k + 1}/{len(steps)}")
    return []


anim = FuncAnimation(fig, update, frames=len(steps), interval=1400, blit=False)

os.makedirs(OUT_DIR, exist_ok=True)
gif_path = os.path.join(OUT_DIR, "code_sampling_animation.gif")
anim.save(gif_path, writer=PillowWriter(fps=0.72))
print("yazildi:", gif_path, f"({len(steps)} kare)")

try:
    from matplotlib.animation import FFMpegWriter

    mp4_path = os.path.join(OUT_DIR, "code_sampling_animation.mp4")
    anim.save(mp4_path, writer=FFMpegWriter(fps=1, bitrate=2400))
    print("yazildi:", mp4_path)
except Exception as e:
    print("mp4 atlandi:", e)
