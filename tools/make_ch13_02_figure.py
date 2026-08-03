"""Hero figure for the Chapter 13 applied page — treatment B, per FIGURE_SPEC.md.

The moment the exercise is built around: the seal comes off. Left, the two leading
principal components of 10,000 Fashion-MNIST images, colored by the labels that were
withheld for the whole analysis. Right, the K-means clusters cross-tabulated against
those labels. Generated from the same computation as the companion notebook
(ch13_01_unsupervised.ipynb) — same source, same subsample, same K.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_all, y_all = fetch_openml("Fashion-MNIST", version=1, as_frame=False,
                            return_X_y=True, parser="auto")
m = 10_000
X = X_all[:m].astype(np.float64) / 255.0
y_sealed = y_all[:m].astype(int)            # not used until the cross-tab below
del X_all, y_all

# PCA exactly as the notebook does it: eigendecomposition of the sample covariance,
# not sklearn's PCA, so the page and the notebook cannot drift apart.
x_mean = X.mean(axis=0)
Xc = X - x_mean
S = Xc.T @ Xc / m                           # 784 x 784 sample covariance
evals, evecs = np.linalg.eigh(S)            # ascending
evals, evecs = evals[::-1], evecs[:, ::-1]  # largest first
frac = evals / evals.sum()

Z = Xc @ evecs[:, :50]
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Z)   # notebook clusters on all 50

print("explained variance, first 2 components:", frac[:2].round(4))
print("explained variance, first 50:", frac[:50].sum().round(4))

# Cross-tabulate clusters against the withheld labels, then order clusters by their
# dominant class so the diagonal structure (where it exists) is visible.
table = np.zeros((10, 10), dtype=int)
for c, lab in zip(km.labels_, y_sealed):
    table[c, lab] += 1
order = np.argsort(table.argmax(axis=1), kind="stable")
table = table[order]
purity = table.max(axis=1).sum() / table.sum()
print(f"cluster purity (best-class share): {purity:.3f}")
for c in range(10):
    row = table[c]
    print(f"  cluster {c}: n={row.sum():5d}  dominant={CLASSES[row.argmax()]:11}"
          f" ({row.max() / row.sum():.0%})")

fig, (axP, axT) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# --- the projection, colored by the labels that were sealed ---------------------
cmap = plt.get_cmap("tab10")
for k in range(10):
    sel = y_sealed == k
    axP.scatter(Z[sel, 0], Z[sel, 1], s=3, alpha=0.45, color=cmap(k),
                linewidths=0, label=CLASSES[k])
axP.set_title("PC2 vs PC1, colored by the withheld labels", fontsize=11)
axP.set_xlabel("PC 1"); axP.set_ylabel("PC 2")
axP.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92,
           edgecolor=SPINE, fontsize=7.5, markerscale=3, labelspacing=0.25,
           handletextpad=0.2, ncol=2)
for s in ["top", "right"]:
    axP.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    axP.spines[s].set_color(SPINE)
axP.grid(color=GRID, lw=0.8); axP.set_axisbelow(True)

# --- clusters against classes ---------------------------------------------------
im = axT.imshow(table, cmap="Blues", vmin=0, vmax=table.max(), interpolation="nearest")
axT.set_title("$K$-means clusters against those labels", fontsize=11)
axT.set_xticks(range(10)); axT.set_yticks(range(10))
axT.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
axT.set_yticklabels([f"cluster {i}" for i in range(10)], fontsize=8)
axT.set_xticks(np.arange(-0.5, 10, 1), minor=True)
axT.set_yticks(np.arange(-0.5, 10, 1), minor=True)
axT.grid(which="minor", color="white", lw=0.8)
axT.tick_params(which="minor", length=0)
for s in axT.spines.values():
    s.set_color(SPINE)
axT.tick_params(colors="#666666", length=3)
cb = fig.colorbar(im, ax=axT, fraction=0.046, pad=0.03)
cb.set_label("images", fontsize=10)
cb.ax.tick_params(labelsize=9, colors="#666666")
cb.outline.set_edgecolor(SPINE)

fig.tight_layout()
out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch13_02_applied.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
