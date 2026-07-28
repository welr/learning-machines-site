"""Hero figure for the Chapter 1 companion page — treatment B, per FIGURE_SPEC.md.

The baseline (a flat line at the mean) against the least-squares line, with the
gaps the baseline leaves drawn in. Generated from the page's own computation: the
same twenty Frankfurt sales the page's live cells use.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

area = np.array([52, 82, 120, 95, 140, 160, 180, 200, 210, 225,
                 150, 110, 130, 170, 190, 205, 240, 60, 75, 100.])
price = np.array([164, 384, 465, 420, 520, 610, 700, 760, 820, 900,
                  560, 430, 500, 650, 720, 790, 980, 260, 330, 410.])

baseline = price.mean()
ac, pc = area - area.mean(), price - price.mean()
slope = (ac * pc).sum() / (ac ** 2).sum()
intercept = price.mean() - slope * area.mean()

mse_base = ((price - baseline) ** 2).mean()
mse_line = ((price - (intercept + slope * area)) ** 2).mean()
print(f"baseline = {baseline:.1f}  MSE {mse_base:.1f}")
print(f"line: {intercept:.1f} + {slope:.2f}*area  MSE {mse_line:.1f}")
print(f"error removed = {1 - mse_line / mse_base:.3%}")

fig, ax = plt.subplots(figsize=(7.0, 4.2))

# the gaps the baseline leaves — what every later chapter is trying to shrink
for a, p in zip(area, price):
    ax.plot([a, a], [baseline, p], color=mt.GRAY, lw=0.8, alpha=0.45, zorder=1,
            label="_nolegend_")

ax.axhline(baseline, color=mt.RED, lw=2.0, ls="--", zorder=2,
           label=f"baseline: predict {baseline:.0f}")
grid = np.linspace(area.min() - 6, area.max() + 6, 50)
ax.plot(grid, intercept + slope * grid, color=mt.BLUE, lw=2.2, zorder=3,
        label="least-squares line")
ax.scatter(area, price, s=46, color=mt.FS_BLUE, edgecolors="white", linewidths=0.8,
           zorder=4, label="Frankfurt sales")

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(SPINE)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors="#666666", labelsize=11)
ax.set_xlabel("living area, m²", fontsize=11)
ax.set_ylabel("price, € thousands", fontsize=11)
ax.legend(loc="upper left", frameon=False, fontsize=10.5, handlelength=1.7,
          labelspacing=0.35)

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch01_01_baseline.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
