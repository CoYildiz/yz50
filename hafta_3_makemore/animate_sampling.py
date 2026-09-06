"""Bigram sampling döngüsünün 27x27 olasılık matrisi üzerindeki yürüyüşünü animasyona çevirir.

Sadece görselleştirme: makemore.py'ye dokunmaz, import etmez, veri hazırlığını
kendi içinde tekrar kurar. Çıktı: output/sampling_animation.gif (+ varsa .mp4)
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
N_NAMES = 3

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

LABELS = [itos[i] for i in range(27)]

# ------------------------------------------------- sampling izini önden çıkar
# Her adım iki kareye bölünür: (1) satır okunuyor, (2) hücre seçildi.
frames = []
g = torch.Generator().manual_seed(SEED)

for name_no in range(N_NAMES):
    ix = 0
    out = []
    step = 0
    while True:
        step += 1
        row = P[ix]
        nix = torch.multinomial(row, num_samples=1, replacement=True, generator=g).item()

        frames.append(
            dict(name_no=name_no, step=step, ix=ix, pick=None,
                 prob=None, name="".join(out), done=False)
        )
        out.append(itos[nix])
        frames.append(
            dict(name_no=name_no, step=step, ix=ix, pick=nix,
                 prob=row[nix].item(), name="".join(out), done=False)
        )
        ix = nix
        if ix == 0:
            frames[-1]["done"] = True
            break

# son karede bekle
frames += [dict(frames[-1]) for _ in range(2)]

# ---------------------------------------------------------------- figür
fig = plt.figure(figsize=(16, 9), dpi=100)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.75],
              wspace=0.18, hspace=0.28, left=0.05, right=0.98, top=0.90, bottom=0.07)

ax_mat = fig.add_subplot(gs[:, 0])
ax_bar = fig.add_subplot(gs[0, 1])
ax_txt = fig.add_subplot(gs[1, 1])

# --- matris
ax_mat.imshow(P, cmap="YlGnBu", vmin=0.0, vmax=0.22)
ax_mat.set_xticks(range(27))
ax_mat.set_yticks(range(27))
ax_mat.set_xticklabels(LABELS, fontsize=9)
ax_mat.set_yticklabels(LABELS, fontsize=9)
ax_mat.set_xlabel("sonraki harf", fontsize=11)
ax_mat.set_ylabel("su anki harf  (ix)", fontsize=11)
ax_mat.set_title("P — satir: su anki harf, sutun: sonraki harf", fontsize=12)
ax_mat.tick_params(length=0)

# aktif satırın dışını soluklaştıran iki maske
mask_top = Rectangle((-0.5, -0.5), 27, 0, facecolor="white", alpha=0.72, zorder=3)
mask_bot = Rectangle((-0.5, -0.5), 27, 0, facecolor="white", alpha=0.72, zorder=3)
ax_mat.add_patch(mask_top)
ax_mat.add_patch(mask_bot)

row_box = Rectangle((-0.5, -0.5), 27, 1, fill=False, edgecolor="#d62728", lw=2.2, zorder=5)
ax_mat.add_patch(row_box)

cell_box = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111111", lw=3.0, zorder=6)
cell_box.set_visible(False)
ax_mat.add_patch(cell_box)

# --- bar chart
bars = ax_bar.bar(range(27), [0] * 27, color="#7fb3d5")
ax_bar.set_xticks(range(27))
ax_bar.set_xticklabels(LABELS, fontsize=9)
ax_bar.set_ylabel("olasilik", fontsize=10)
ax_bar.tick_params(length=0)
for side in ("top", "right"):
    ax_bar.spines[side].set_visible(False)

# --- metin paneli
ax_txt.axis("off")
txt = ax_txt.text(0.02, 0.95, "", va="top", ha="left", fontsize=15,
                  family="monospace", linespacing=1.9)
name_txt = ax_txt.text(0.02, 0.34, "", va="top", ha="left", fontsize=30,
                       family="monospace", color="#0b3d62")
done_txt = ax_txt.text(0.02, 0.04, "", va="top", ha="left", fontsize=15,
                       family="monospace", color="#1a7f37")

suptitle = fig.suptitle("", fontsize=16)


def update(k):
    fr = frames[k]
    ix, pick = fr["ix"], fr["pick"]
    row = P[ix]

    # satır maskesi
    mask_top.set_bounds(-0.5, -0.5, 27, ix)
    mask_bot.set_bounds(-0.5, ix + 0.5, 27, 26 - ix)
    row_box.set_bounds(-0.5, ix - 0.5, 27, 1)

    if pick is None:
        cell_box.set_visible(False)
    else:
        cell_box.set_visible(True)
        cell_box.set_bounds(pick - 0.5, ix - 0.5, 1, 1)

    # bar chart
    ax_bar.set_ylim(0, max(row.max().item() * 1.15, 0.05))
    for j, b in enumerate(bars):
        b.set_height(row[j].item())
        b.set_color("#111111" if (pick is not None and j == pick) else "#7fb3d5")
    ax_bar.set_title(f"P[{ix}]  —  '{itos[ix]}' harfinden sonra ne gelir",
                     fontsize=12)

    # metin
    lines = [f"adim   : {fr['step']}",
             f"ix     : {ix}  ('{itos[ix]}')"]
    if pick is None:
        lines.append("secilen: ...  (zar atiliyor)")
    else:
        lines.append(f"secilen: {pick}  ('{itos[pick]}')")
        lines.append(f"olasilik: {fr['prob']:.4f}")
    txt.set_text("\n".join(lines))
    done_txt.set_text("'.' cekildi -> isim bitti" if fr["done"] else "")

    shown = fr["name"] if fr["name"] else ""
    name_txt.set_text(f"isim: {shown}" + ("" if fr["done"] else "_"))

    suptitle.set_text(f"Bigram sampling  —  isim {fr['name_no'] + 1}/{N_NAMES}")
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=1200, blit=False)

os.makedirs(OUT_DIR, exist_ok=True)
gif_path = os.path.join(OUT_DIR, "sampling_animation.gif")
anim.save(gif_path, writer=PillowWriter(fps=0.83))
print("yazildi:", gif_path, f"({len(frames)} kare)")

try:
    from matplotlib.animation import FFMpegWriter

    mp4_path = os.path.join(OUT_DIR, "sampling_animation.mp4")
    anim.save(mp4_path, writer=FFMpegWriter(fps=1, bitrate=2400))
    print("yazildi:", mp4_path)
except Exception as e:  # ffmpeg yoksa GIF yeter
    print("mp4 atlandi:", e)
