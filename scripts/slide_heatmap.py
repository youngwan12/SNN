"""발표 슬라이드용 그림: 원본 weight → 분포 기반(k-means) 3비트 양자화 weight (흰 바탕, 검은 글씨)

사용법 (저장소 루트에서): python scripts/slide_heatmap.py data/seed4 figures/slide_kmeans_seed4.png
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RUN, OUT = sys.argv[1], sys.argv[2]

spec = importlib.util.spec_from_file_location("qc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_compare.py"))
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)

BLACK, WHITE = "#000000", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f2f2f2", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
    "figure.facecolor": WHITE, "savefig.facecolor": WHITE, "text.color": BLACK,
    "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
})

raw = pd.read_csv(os.path.join(RUN, "weights_01_raw.csv"), index_col=0).values.astype(np.float64)
codebook = qc.kmeans_codebook(raw)
km, stage = qc.apply_codebook(raw, codebook)
vmax = raw.max()

fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.55))
fig.subplots_adjust(left=0.07, right=0.9, bottom=0.13, top=0.88, wspace=0.22)


def draw(ax, data, title, stage=None):
    im = ax.imshow(data, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=13, color=BLACK, pad=8)
    ax.set_xticks(range(10), [str(j) for j in range(10)])
    ax.set_yticks(range(16), [str(i) for i in range(16)])
    ax.set_xlabel("Output Class 0~9")
    ax.set_ylabel("Hidden Node 0~15")
    ax.tick_params(length=0, labelsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 16), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    if stage is not None:
        rgba = im.cmap(im.norm(data))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                lum = 0.2126 * rgba[i, j, 0] + 0.7152 * rgba[i, j, 1] + 0.0722 * rgba[i, j, 2]
                ax.text(j, i, str(stage[i, j]), ha="center", va="center", fontsize=8.5,
                        color=WHITE if lum < 0.5 else BLACK)
    return im


im = draw(axes[0], raw, "원본 Weight")
draw(axes[1], km, "분포 기반 3-bit 양자화 Weight (칸 숫자 = 단계 1~8)", stage)

cax = fig.add_axes([0.915, 0.13, 0.013, 0.75])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Weight")
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=9.5)

fig.savefig(OUT, dpi=200)
print("saved", OUT)
print("codebook:", ", ".join(f"{v:.2f}" for v in codebook))
