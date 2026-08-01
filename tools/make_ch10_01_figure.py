"""Hero figure for the Chapter 10 companion page — treatment B, per FIGURE_SPEC.md.

Input, kernel, feature map: the whole convolution in one row. Generated from the
page's own computation — the same 10x10 square, the same 3x3 vertical-edge kernel,
and the same from-scratch `conv2d` the live {pyodide}/{webr} cell runs, so what the
reader executes reproduces this figure exactly.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Rectangle

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"


def conv2d(image, kernel):
    """2D convolution, no padding, stride 1."""
    H, W = image.shape
    kH, kW = kernel.shape
    out = np.zeros((H - kH + 1, W - kW + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i + kH, j:j + kW]
            out[i, j] = np.sum(patch * kernel)
    return out


image = np.zeros((10, 10))
image[2:8, 2:8] = 1.0                       # a square of ones on a field of zeros

kernel = np.array([[-1., 0., 1.],
                   [-1., 0., 1.],
                   [-1., 0., 1.]])          # +ve on dark -> light, left to right

feature_map = conv2d(image, kernel)

# One position, spelled out: the window whose top-left corner sits at input (2, 1)
si, sj = 2, 1
patch = image[si:si + 3, sj:sj + 3]
print("input", image.shape, "kernel", kernel.shape, "feature map", feature_map.shape)
print("response range", feature_map.min(), "to", feature_map.max())
print("zero entries:", int((feature_map == 0).sum()), "of", feature_map.size)
print("patch at (2,1):\n", patch)
print("patch . kernel =", np.sum(patch * kernel), "= feature_map[2,1] =", feature_map[si, sj])
print("column means of |response|:", np.round(np.abs(feature_map).mean(axis=0), 3))

fig, (axI, axK, axF) = plt.subplots(1, 3, figsize=(10.6, 3.7))

# --- input -------------------------------------------------------------------
axI.imshow(image, cmap="Blues", vmin=0, vmax=1)
axI.add_patch(Rectangle((sj - 0.5, si - 0.5), 3, 3, fill=False,
                        edgecolor=mt.RED, lw=2.0, label="filter window at (2, 1)"))
axI.set_title("input image  (10 × 10)", fontsize=11)
axI.set_xticks(range(0, 10, 3))
axI.set_yticks(range(0, 10, 3))
axI.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.92,
           edgecolor=SPINE, fontsize=9, handlelength=1.2)

# --- kernel ------------------------------------------------------------------
axK.imshow(kernel, cmap="RdBu_r", vmin=-1.6, vmax=1.6)
for i in range(3):
    for j in range(3):
        axK.text(j, i, f"{kernel[i, j]:+.0f}", ha="center", va="center",
                 fontsize=13, color="#333333")
axK.set_title("vertical-edge kernel  (3 × 3)", fontsize=11)
axK.set_xticks([])
axK.set_yticks([])

# --- feature map -------------------------------------------------------------
im = axF.imshow(feature_map, cmap="RdBu_r", vmin=-3, vmax=3)
axF.scatter([sj], [si], s=70, facecolors="none", edgecolors=mt.RED, linewidths=2.0,
            label=f"response = {feature_map[si, sj]:+.0f}")
axF.set_title("feature map  (8 × 8)", fontsize=11)
axF.set_xticks(range(0, 8, 2))
axF.set_yticks(range(0, 8, 2))
axF.legend(loc="center", frameon=True, facecolor="white", framealpha=0.92,
           edgecolor=SPINE, fontsize=9, handlelength=1.2)
cb = fig.colorbar(im, ax=axF, fraction=0.046, pad=0.03)
cb.set_label("response", fontsize=10)
cb.set_ticks([-3, 0, 3])
cb.ax.tick_params(labelsize=9, colors="#666666")
cb.outline.set_edgecolor(SPINE)

for ax in (axI, axK, axF):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(SPINE)
    ax.tick_params(colors="#666666", labelsize=9, length=3)

fig.tight_layout()

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch10_01_convnets.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
