"""Negative log likelihood'un bigram bigram nasıl biriktiğini gösteren animasyon.

Sadece görselleştirme. makemore.py'ye dokunmaz, veri hazırlığını kendi içinde kurar.
Son bölümde smoothing gerekçesi: olasılığı 0 olan tek bir bigram NLL'i sonsuza götürüyor.

Çıktı: output/nll_animation.gif (+ .mp4)
"""

import math
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
WORDS_SHOWN = 3          # words[0:3] — makemore.py'deki ile aynı
BAD_WORD = "andrejq"     # sıfır olasılıklı bigram içeren örnek

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

# tüm veri kümesinin NLL'i — karşılaştırma çizgisi
_ll, _n = 0.0, 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        _ll += math.log(P[stoi[ch1], stoi[ch2]].item())
        _n += 1
NLL_ALL = -_ll / _n

# ------------------------------------------------- kareleri önden üret
frames = []
rows = []            # (bigram, prob, logprob)
ll, n = 0.0, 0

for w in words[:WORDS_SHOWN]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]].item()
        lp = math.log(prob)
        ll += lp
        n += 1
        rows.append((ch1 + ch2, prob, lp))
        frames.append(dict(kind="step", word=w, chs=chs, cur=(ch1, ch2),
                           prob=prob, lp=lp, ll=ll, n=n,
                           rows=list(rows), note=""))

frames.append(dict(frames[-1], kind="summary",
                   note=f"{WORDS_SHOWN} kelime bitti"))
frames.append(dict(frames[-1], kind="compare",
                   note="tum veri kumesi ile karsilastirma"))

# --- sıfır olasılık örneği
bad_rows = []
bll, bn = 0.0, 0
bchs = ["."] + list(BAD_WORD) + ["."]
for ch1, ch2 in zip(bchs, bchs[1:]):
    prob = P[stoi[ch1], stoi[ch2]].item()
    lp = math.log(prob) if prob > 0 else float("-inf")
    bll += lp
    bn += 1
    bad_rows.append((ch1 + ch2, prob, lp))
    frames.append(dict(kind="bad", word=BAD_WORD, chs=bchs, cur=(ch1, ch2),
                       prob=prob, lp=lp, ll=bll, n=bn, rows=list(bad_rows),
                       note="" if prob > 0 else "olasilik 0  ->  log(0) = -sonsuz"))

frames.append(dict(frames[-1], kind="bad_end",
                   note="tek bir imkansiz bigram butun NLL'i sonsuz yapti"))
frames += [dict(frames[-1]) for _ in range(2)]

# ---------------------------------------------------------------- figür
fig = plt.figure(figsize=(16, 9), dpi=100)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.05], height_ratios=[0.42, 1.0],
              wspace=0.12, hspace=0.22, left=0.05, right=0.97, top=0.88, bottom=0.06)

ax_word = fig.add_subplot(gs[0, :])
ax_tbl = fig.add_subplot(gs[1, 0])
ax_acc = fig.add_subplot(gs[1, 1])
for a in (ax_word, ax_tbl, ax_acc):
    a.axis("off")
ax_word.set_xlim(0, 1)
ax_word.set_ylim(0, 1)

word_texts = []
hl_word = Rectangle((0, 0), 0, 0, facecolor="#ffe08a", alpha=0.85, zorder=0)
ax_word.add_patch(hl_word)

tbl_txt = ax_tbl.text(0.0, 1.0, "", va="top", ha="left", fontsize=14.5,
                      family="monospace", linespacing=1.75)
acc_txt = ax_acc.text(0.0, 1.0, "", va="top", ha="left", fontsize=16,
                      family="monospace", linespacing=2.0)
note_txt = ax_word.text(0.52, 0.50, "", va="center", ha="left", fontsize=17,
                        family="monospace", color="#b3261e")
suptitle = fig.suptitle("", fontsize=17)


def draw_word(chs, cur):
    for t in word_texts:
        t.remove()
    word_texts.clear()
    x0, dx = 0.06, 0.055
    hit = None
    for k, c in enumerate(chs):
        col = "#111111"
        word_texts.append(ax_word.text(x0 + k * dx, 0.45, c, fontsize=34,
                                       family="monospace", ha="center", va="center",
                                       color=col, zorder=2))
        if hit is None and k + 1 < len(chs) and (chs[k], chs[k + 1]) == cur:
            hit = k
    if hit is not None:
        hl_word.set_bounds(x0 + hit * dx - dx * 0.5, 0.18, dx * 2, 0.55)


def fmt_rows(rs):
    head = "bigram    olasilik    log(olasilik)\n" + "-" * 38
    body = []
    for bg, pr, lp in rs[-12:]:
        lp_s = "  -sonsuz" if lp == float("-inf") else f"{lp:9.4f}"
        body.append(f"  {bg:<6}  {pr:8.4f}    {lp_s}")
    return head + "\n" + "\n".join(body)


def update(k):
    fr = frames[k]
    draw_word(fr["chs"], fr["cur"])
    tbl_txt.set_text(fmt_rows(fr["rows"]))

    ll, n = fr["ll"], fr["n"]
    avg = ll / n
    nll = -avg
    a, b = fr["cur"]

    lp_s = "-sonsuz" if fr["lp"] == float("-inf") else f"{fr['lp']:.4f}"
    lines = [
        f"su anki bigram : '{a}{b}'",
        f"P['{a}' -> '{b}']  = {fr['prob']:.4f}",
        f"log(P)         = {lp_s}",
        "",
        f"log_likelihood = {'-sonsuz' if ll == float('-inf') else f'{ll:.4f}'}",
        f"n              = {n}",
        f"ortalama       = {'-sonsuz' if avg == float('-inf') else f'{avg:.4f}'}",
        "",
        f"NLL = -ortalama = {'sonsuz' if nll == float('inf') else f'{nll:.4f}'}",
    ]
    if fr["kind"] in ("compare",):
        lines += ["", f"tum veri kumesi NLL = {NLL_ALL:.4f}",
                  "(hedef olcut bu — Gorev 3)"]
    acc_txt.set_text("\n".join(lines))
    note_txt.set_text(fr["note"])

    if fr["kind"] in ("bad", "bad_end"):
        title = f"Neden smoothing gerekiyor  —  '{fr['word']}'"
    else:
        title = f"NLL nasil birikiyor  —  '{fr['word']}'"
    suptitle.set_text(title)
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=1500, blit=False)

os.makedirs(OUT_DIR, exist_ok=True)
gif_path = os.path.join(OUT_DIR, "nll_animation.gif")
anim.save(gif_path, writer=PillowWriter(fps=0.67))
print("yazildi:", gif_path, f"({len(frames)} kare)")

try:
    from matplotlib.animation import FFMpegWriter

    mp4_path = os.path.join(OUT_DIR, "nll_animation.mp4")
    anim.save(mp4_path, writer=FFMpegWriter(fps=1, bitrate=2400))
    print("yazildi:", mp4_path)
except Exception as e:
    print("mp4 atlandi:", e)
