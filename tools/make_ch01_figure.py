"""Hero figure for the Chapter 1 companion page — treatment B, per FIGURE_SPEC.md.

The baseline (a flat line at the mean) against the least-squares line, with the
gaps the baseline leaves drawn in. Generated from the page's own computation: the
same Frankfurt sales the page's live cells use (read from the page).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

# The sales come from the page's own first {pyodide} cell, so hero and page cannot drift apart.
import contextlib, io, re
_page = (pathlib.Path(__file__).resolve().parents[1] / "chapters" / "ch01_01_baseline.qmd").read_text()
_cell = re.search(r"```\{pyodide\}\n(.*?)```", _page, re.S).group(1)
_env = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(_cell, _env)
area, price = _env["area"], _env["price"]

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

ax.axhline(baseline, color=mt.FS_BLUE, lw=1.8, ls="--", zorder=2,
           label=f"baseline: predict {baseline:.0f}")
grid = np.linspace(area.min() - 6, area.max() + 6, 50)
ax.plot(grid, intercept + slope * grid, color=mt.FS_BLUE, lw=2.2, zorder=3,
        label="least-squares line")
ax.scatter(area, price, s=40, color=mt.BLUE, edgecolors="white", linewidths=0.8,
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

out = pathlib.Path(__file__).resolve().parents[1] / "figures" / "ch01_01_baseline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
