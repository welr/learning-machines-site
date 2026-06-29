# Hero-figure spec — TREATMENT B (legend-based, conventional)

The site moved from the editorial "direct label" look (treatment A) to **treatment B**:
a legend instead of labels-on-the-lines, a light frame, and a faint grid. This makes
the pre-rendered hero figures match the live cells (which already use legends). Gold-
standard references already in B: `figures/ch05_bias_variance.png` (multi-line) and
`figures/ch03_01_gradient_descent.png` (2-D contour).

Goal per page: ONE hero figure, generated from the page's OWN computation, in treatment
B, placed before the first code tabset; AND the page's live plotting cells given the same
light B polish so what the reader runs matches the hero.

## Palette (unchanged)

`mt.BLUE` `#076FA1` · `mt.FS_BLUE` `#31417A` · `mt.RED` `#E3120B` ·
`mt.CYAN` `#2FC1D3` · `mt.ORANGE` `#9E4F00` · `mt.GRAY` `#666666`.
Frame `#cfccc2`, grid `#e6e3da`. White background. NO green.

## Hero figures (build-time, mlone_theme) — by chart type

Start every generator with:
```python
import sys; sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np
mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"
```
Do **NOT** call `mt.apply_book_style` and do **NOT** use `mt.add_direct_label` (those are
the A look). Give every plotted series a `label=` and use a legend. Save with
`fig.savefig(".../figures/<stem>.png", dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)`.

**Multi-series line / scatter** (poly fits, bias–variance, ridge/LASSO paths, trees
train/test, logistic curve, OLS+residuals, Bayesian band, applied regression) —
`figsize=(7.0,4.2)`:
```python
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(SPINE)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors="#666666", labelsize=11)
ax.set_xlabel(...); ax.set_ylabel(...)
ax.legend(loc="best", frameon=False, fontsize=10.5, handlelength=1.7, labelspacing=0.35)
```
Pick `loc` so the legend lands in empty space (often "upper right"); for the many-line
ridge/LASSO paths use ONE entry per group ("informative", "spurious") rather than every
coefficient, via a representative labelled line + `label="_nolegend_"` on the rest.

**Single-series line** (the two ROC curves, backprop loss) — same frame+grid; a legend
only if there is a second element to name (ROC vs chance diagonal), else just label the axes.

**2-D field — contour** (gradient-descent loss surface) — `figsize=(7.0,5.0)`, all four
spines `SPINE`, BOTH feature axes, a **colorbar** for the field, and a **white-box legend**
for the path/points:
```python
cf = ax.contourf(B0,B1,Loss, levels=levels, cmap="Blues", alpha=0.92)
cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03); cb.set_label("loss (MSE)", fontsize=11)
cb.ax.tick_params(labelsize=9); cb.outline.set_edgecolor(SPINE)
ax.plot(..., label="descent path"); ax.scatter(..., label="optimum")
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor=SPINE, fontsize=10)
```

**2-D categorical — regions / decision boundaries** (multiclass, kernel rings, moons) —
`figsize=(6.8,5.2)`, soft `contourf` fills + the boundary, points colored by TRUE class,
all four spines `SPINE`, BOTH feature axes, and a **white-box class legend** (no colorbar):
```python
ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor=SPINE, fontsize=10)
```

**Bar charts** (OLS residuals figure, odds ratios) — keep as they are now (the odds chart
is already a log-axis diverging chart); just ensure top/right spines off, left/bottom = `SPINE`.

## Live cells — light B polish (Python + R)

The cells already build legends; align their frame/grid to the hero. Do NOT rewrite the
plotting logic — keep the data, colors, and `label=`/`legend()` they already have.

**Python** — just before `plt.show()`, ensure:
```python
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e3da", lw=0.8); ax.set_axisbelow(True)
```
(2-D feature-space cells: skip the y-grid; add `fig.colorbar(cf, ax=ax, label="...")` to the
contour/​field cells where one applies. Leave categorical region/boundary cells without a colorbar.)

**R** — give `plot(...)` an L-frame and a faint grid:
```r
plot(..., bty = "l")                 # was the default full box
grid(col = "#e6e3da", lty = 1, lwd = 0.6)
```
(Add `grid()` after the `plot()`/`barplot()`; keep the existing `legend(...)`. base R has no
easy colorbar, so 2-D R cells keep their legend only — that's fine.)

## Faithfulness, verification

- Regenerate each hero from the page's OWN computation (fixed seed). It must match what the
  cell produces.
- Run each figure script (`MPLBACKEND=Agg /Users/gregorywheeler/otter_env/bin/python`); PNG > 20 KB.
- Every edited cell must still run: Python via `~/otter_env/bin/python` (MPLBACKEND=Agg),
  R via `Rscript --vanilla`. Do NOT run `quarto render` (the orchestrator renders the whole
  site at the end).
- The `<figure class="lm-figure">` embed and caption stay as they are; only the PNG changes.
