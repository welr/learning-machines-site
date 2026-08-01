"""Hero figure for ch05_01: the bias-variance decomposition.

Reproduces the page's own `decompose` cell exactly -- same truth, same noise,
same 100 datasets of 20 points, same clip -- so the caption's claim that this
is what the code below produces is true. Treatment B per FIGURE_SPEC.md.
"""
import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import mlone_theme as mt
mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

def true_f(x):
    return np.sin(2 * np.pi * x)

def decompose(degree, n_datasets=100, n=20, noise=0.3):
    grid = np.linspace(0, 1, 100)
    truth = true_f(grid)
    preds = []
    for s in range(n_datasets):
        np.random.seed(s)
        x = np.random.uniform(0, 1, n)
        y = true_f(x) + np.random.normal(0, noise, n)
        coef = np.polyfit(x, y, degree)
        preds.append(np.clip(np.polyval(coef, grid), -10, 10))
    preds = np.array(preds)
    bias2 = ((preds.mean(0) - truth) ** 2).mean()
    var = preds.var(0).mean()
    return bias2, var, noise ** 2, bias2 + var + noise ** 2

degrees = list(range(1, 13))
rows = [decompose(d) for d in degrees]
bias2 = [r[0] for r in rows]; var = [r[1] for r in rows]
irr = rows[0][2];            total = [r[3] for r in rows]
best = degrees[int(np.argmin(total))]
print(f"minimum total error at degree {best}; irreducible {irr:.3f}; "
      f"total at degree 12 {total[-1]:.2f}")

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(degrees, bias2, "o-", color=mt.BLUE,    lw=2, label="bias$^2$")
ax.plot(degrees, var,   "s-", color=mt.FS_BLUE, lw=2, label="variance")
ax.plot(degrees, total, "^-", color=mt.RED,     lw=2, label="total error")
ax.axhline(irr, color=mt.GRAY, ls="--", lw=1.2,
           label=f"irreducible ($\\sigma^2$ = {irr:.2f})")
ax.axvline(best, color=mt.GRAY, ls=":", lw=1.2, alpha=0.8)
ax.set_yscale("log")          # the tail spans three orders of magnitude
ax.set_xlabel("polynomial degree (model complexity)")
ax.set_ylabel("error (log scale)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color(SPINE)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors="#666666", labelsize=11)
ax.set_xticks(degrees)
ax.legend(loc="upper left", frameon=False, fontsize=10.5,
          handlelength=1.7, labelspacing=0.35)
fig.savefig("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/companion-site/figures/ch05_01_bias_variance.png",
            dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
