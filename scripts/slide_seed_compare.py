"""발표 슬라이드용: 시드별 균등 vs 분포 기반(k-means) 3비트 양자화 정확도 (덤벨 차트)

사용법 (저장소 루트에서): python scripts/slide_seed_compare.py results/results_per_run.csv figures/slide_seed_compare.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
BLACK, WHITE, GRAY, LINE, BLUE = "#000000", "#ffffff", "#9a9a9a", "#d0d0d0", "#1c5cab"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12,
                     "text.color": BLACK, "xtick.color": BLACK, "ytick.color": BLACK})

df = pd.read_csv(SRC)
acc = df.pivot(index="run", columns="method", values="accuracy") * 100
order = ["seed42", "seed1", "seed2", "seed3", "seed4"]
rows = [(f"seed {r[4:]}", acc.loc[r, "uniform"], acc.loc[r, "kmeans"]) for r in order]
rows.append(("평균", acc.loc[order, "uniform"].mean(), acc.loc[order, "kmeans"].mean()))

fig, ax = plt.subplots(figsize=(4.8, 4.2))
fig.patch.set_facecolor(WHITE)
for i, (name, u, k) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([u, k], [y, y], color=LINE, linewidth=3, zorder=1, solid_capstyle="round")
    ax.scatter([u], [y], s=90, color=GRAY, zorder=2, edgecolor=WHITE, linewidth=1.5)
    ax.scatter([k], [y], s=90, color=BLUE, zorder=3, edgecolor=WHITE, linewidth=1.5)
    if name == "평균":
        ax.text(u - 0.15, y, f"{u:.2f}", ha="right", va="center", fontsize=12, color=BLACK)
        ax.text(k + 0.15, y, f"{k:.2f}", ha="left", va="center", fontsize=12, color=BLACK, fontweight="bold")
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "평균":
        lbl.set_fontweight("bold")
ax.axhline(0.5, color=LINE, linewidth=1)
ax.set_xlim(88.8, 95.8)
ax.set_xlabel("테스트 정확도 (%)")
ax.tick_params(length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.grid(axis="x", color="#eeeeee", linewidth=1)
ax.set_axisbelow(True)
ax.scatter([], [], s=90, color=GRAY, label="균등")
ax.scatter([], [], s=90, color=BLUE, label="분포 기반 (k-means)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=12, handletextpad=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=220)
print("saved", OUT)
