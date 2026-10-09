"""출력층 weight 히트맵: 실수 / 균등 / k-means (칸 숫자 = 1~8 단계) + 양자화 오차

사용법 (저장소 루트에서): python scripts/heatmap_report.py [data/seed42] [results] [figures]
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/seed42"
RESULTS = sys.argv[2] if len(sys.argv) > 2 else "results"
FIGURES = sys.argv[3] if len(sys.argv) > 3 else "figures"

spec = importlib.util.spec_from_file_location("qc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_compare.py"))
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f4b8b3", "#e34948", "#9b2226"])

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "pdf.fonttype": 42,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
})

raw = pd.read_csv(os.path.join(RUN, "weights_01_raw.csv"), index_col=0).values.astype(np.float64)
w_min, w_max = raw.min(), raw.max()
step = (w_max - w_min) / qc.LEVELS
uni_stage = (np.round((raw - w_min) / step).clip(0, qc.LEVELS - 1) + 1).astype(int)
uni = qc.quantize_weights(raw)
km, km_stage = qc.apply_codebook(raw, qc.kmeans_codebook(raw))

# 정확도 (quant_compare 결과에서)
res = pd.read_csv(os.path.join(RESULTS, "results_per_run.csv"))
run_name = os.path.basename(os.path.normpath(RUN))
acc = res[res.run == run_name].set_index("method").accuracy * 100

fig = plt.figure(figsize=(15, 13))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.28, wspace=0.18)


def draw(ax, data, cmap, norm, title, stage=None, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xticks(range(data.shape[1]), [str(j) for j in range(data.shape[1])])
    ax.set_yticks(range(data.shape[0]), [str(i) for i in range(data.shape[0])])
    ax.set_xlabel("출력 클래스")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 칸 사이 2px 여백
    ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    if stage is not None:
        rgba = im.cmap(im.norm(data))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                lum = 0.2126 * rgba[i, j, 0] + 0.7152 * rgba[i, j, 1] + 0.0722 * rgba[i, j, 2]
                ax.text(j, i, str(stage[i, j]), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if lum < 0.5 else INK)
    return im


# ── 1행: weight 값 (같은 색 범위) ──
ax0 = fig.add_subplot(gs[0, 0])
im = draw(ax0, raw, SEQ, None, f"실수 weight (양자화 전)\n정확도 {acc['float']:.2f}%", vmin=0, vmax=w_max)
ax0.set_ylabel("은닉 노드")
ax1 = fig.add_subplot(gs[0, 1])
draw(ax1, uni, SEQ, None, f"균등 양자화 (원본 코드)  — 칸 숫자 = 단계\n정확도 {acc['uniform']:.2f}%", stage=uni_stage, vmin=0, vmax=w_max)
ax2 = fig.add_subplot(gs[0, 2])
draw(ax2, km, SEQ, None, f"분포 기반 k-means  — 칸 숫자 = 단계\n정확도 {acc['kmeans']:.2f}%", stage=km_stage, vmin=0, vmax=w_max)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
cb.set_label("weight 값")
cb.outline.set_visible(False)

# ── 2행: 양자화 오차 = 양자화값 - 실수값 (같은 대칭 범위) ──
err_uni, err_km = uni - raw, km - raw
lim = max(np.abs(err_uni).max(), np.abs(err_km).max())
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

ax_txt = fig.add_subplot(gs[1, 0])
ax_txt.axis("off")
cb_uni = qc.quantize_codebook_levels("uniform", raw)
cb_km = qc.kmeans_codebook(raw)
lines = [
    f"{run_name}  ·  출력층 16×10 = {raw.size}개 weight",
    "",
    "단계별 레벨 값 (1 → 8)",
    "  균등   : " + ", ".join(f"{v:.2f}" for v in cb_uni),
    "  k-means: " + ", ".join(f"{v:.2f}" for v in cb_km),
    "",
    "단계별 weight 개수 (1 → 8)",
    "  균등   : " + ", ".join(str(int((uni_stage == s).sum())) for s in range(1, 9)),
    "  k-means: " + ", ".join(str(int((km_stage == s).sum())) for s in range(1, 9)),
    "",
    "양자화 오차",
    f"  균등   : MSE {np.mean(err_uni ** 2):.4f},  최대 |오차| {np.abs(err_uni).max():.3f}",
    f"  k-means: MSE {np.mean(err_km ** 2):.4f},  최대 |오차| {np.abs(err_km).max():.3f}",
    "",
    f"실수 weight 범위: {w_min:.3f} ~ {w_max:.3f}",
    f"균등 최상위 레벨 {cb_uni[-1]:.2f} 초과 weight: {int((raw > cb_uni[-1] + step / 2).sum())}개",
]
ax_txt.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5, color=INK2, linespacing=1.6,
            transform=ax_txt.transAxes)

ax3 = fig.add_subplot(gs[1, 1])
im_e = draw(ax3, err_uni, DIV, norm, "균등 양자화 오차 (양자화값 - 실수값)")
ax3.set_ylabel("은닉 노드")
ax4 = fig.add_subplot(gs[1, 2])
draw(ax4, err_km, DIV, norm, "k-means 양자화 오차 (양자화값 - 실수값)")
cb2 = fig.colorbar(im_e, cax=fig.add_subplot(gs[1, 3]))
cb2.set_label("오차  (파랑 = 작아짐, 빨강 = 커짐)")
cb2.outline.set_visible(False)

for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    path = os.path.join(FIGURES, f"fig2_heatmaps_{run_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print("saved", path)
