"""quant_compare.py 결과 → 비교 그림 (weight 분포 + 레벨 위치 / 방식별 정확도)

사용법 (저장소 루트에서): python scripts/plot_report.py [results] [figures]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
FIGURES = sys.argv[2] if len(sys.argv) > 2 else "figures"
DATA = "data"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "pdf.fonttype": 42,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "text.color": INK, "font.size": 10,
})

per_run = pd.read_csv(os.path.join(RESULTS, "results_per_run.csv"))
first_run = per_run.run.iloc[0]
with open(os.path.join(FIGURES, "summary.json"), encoding="utf-8") as f:
    codebooks = json.load(f)["codebooks"][first_run]
w = pd.read_csv(os.path.join(DATA, first_run, "weights_01_raw.csv"), index_col=0).values.ravel()

order = ["kmeans_zero", "kmeans", "quantile", "uniform_std", "uniform", "uniform_pipeline"]
labels = {"uniform": "균등 (원본 코드)", "uniform_pipeline": "균등 (원본 복원 경로)",
          "uniform_std": "균등 (양 끝 포함)", "quantile": "분포: 분위수", "kmeans": "분포: k-means",
          "kmeans_zero": "분포: 0 고정 + k-means"}

fig = plt.figure(figsize=(12.5, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.5], width_ratios=[1, 1.05], hspace=0.08, wspace=0.45)

# ── (a) weight 분포 ──
ax_h = fig.add_subplot(gs[0, 0])
ax_h.hist(w, bins=40, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1)
ax_h.set_ylabel("weight 개수")
ax_h.set_title(f"출력층 weight 분포 (16×10 = {w.size}개, {first_run})", loc="left", color=INK, fontsize=11)
ax_h.tick_params(labelbottom=False)
ax_h.grid(axis="y", color=GRID, linewidth=0.8)
ax_h.set_axisbelow(True)

# ── (a') 방식별 8개 레벨 위치 ──
ax_l = fig.add_subplot(gs[1, 0], sharex=ax_h)
cb_order = ["kmeans_zero", "kmeans", "quantile", "uniform_std", "uniform"]
for y, m in enumerate(cb_order):
    lv = np.array(codebooks[m])
    ax_l.axhline(y, color=GRID, linewidth=0.8, zorder=0)
    ax_l.scatter(lv, np.full_like(lv, y), s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
ax_l.set_yticks(range(len(cb_order)), [labels[m] for m in cb_order])
ax_l.set_ylim(-0.6, len(cb_order) - 0.4)
ax_l.set_xlabel("weight 값  (점 = 양자화 레벨)")
for s in ("top", "right", "left"):
    ax_h.spines[s].set_visible(False)
    ax_l.spines[s].set_visible(False)
ax_l.tick_params(axis="y", length=0)

# ── (b) 방식별 정확도 ──
ax_a = fig.add_subplot(gs[:, 1])
float_mean = per_run[per_run.method == "float"].accuracy.mean() * 100
ax_a.axvline(float_mean, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax_a.text(float_mean, len(order) - 0.45, f"실수(양자화 전) {float_mean:.2f}% ", color=INK2, fontsize=9,
          va="bottom", ha="right")
n_runs = per_run.run.nunique()
for y, m in enumerate(order):
    acc = per_run[per_run.method == m].accuracy.values * 100
    if n_runs > 1:
        ax_a.scatter(acc, np.full_like(acc, y), s=22, color=MUTED, alpha=0.55, zorder=2, linewidth=0)
    mean = acc.mean()
    ax_a.scatter([mean], [y], s=80, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax_a.text(mean, y + 0.28, f"{mean:.2f}%", color=INK, fontsize=9.5, ha="center", va="bottom")
    ax_a.axhline(y, color=GRID, linewidth=0.8, zorder=0)
ax_a.set_yticks(range(len(order)), [labels[m] for m in order])
ax_a.set_ylim(-0.6, len(order) - 0.1)
ax_a.set_xlabel("테스트 정확도 (%)")
sub = f"큰 점 = {n_runs}개 시드 평균, 작은 점 = 각 시드" if n_runs > 1 else "단일 실행"
ax_a.set_title(f"8단계(3비트) 양자화 후 정확도 — {sub}", loc="left", color=INK, fontsize=11)
for s in ("top", "right", "left"):
    ax_a.spines[s].set_visible(False)
ax_a.tick_params(axis="y", length=0)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out = os.path.join(FIGURES, f"fig1_quant_comparison.{ext}")
    fig.savefig(out, bbox_inches="tight", **kw)
    print("saved", out)
