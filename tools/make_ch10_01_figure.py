"""Hero figure for the Chapter 10 companion page — treatment B, per FIGURE_SPEC.md.

Input, kernel, feature map: the whole convolution in one row, read as the equation
the defbox states above it. Generated from the page's own computation — the same
10x10 square, the same 3x3 vertical-edge kernel, and the same from-scratch `conv2d`
the live {pyodide}/{webr} cell runs, so what the reader executes reproduces this
figure exactly.

Two encoding rules this figure keeps, because the eye has to track values across
three panels:
  * the input is GRAY (sequential), so blue is free to mean "negative" everywhere;
  * the highlighted window carries its nine values, so the response is derived on
    the page rather than asserted.
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


def weight_label(v):
    """Zero is neither positive nor negative: '+1', '0', '-1'."""
    return f"{v:+.0f}" if v else "0"


fig = plt.figure(figsize=(11.2, 4.3))
gs = fig.add_gridspec(1, 3, wspace=0.36)
axI, axK, axF = (fig.add_subplot(gs[0, k]) for k in range(3))

# --- input -------------------------------------------------------------------
# Gray, not Blues: in the other two panels blue means negative, and the same ink
# cannot also mean "pixel is on".
# vmin/vmax are stretched past the data range so that 1 lands on a mid-dark gray
# rather than pure black, and 0 on a pale field rather than paper white.
axI.imshow(image, cmap="Greys", vmin=-0.13, vmax=1.48, interpolation="nearest")
axI.add_patch(Rectangle((sj - 0.5, si - 0.5), 3, 3, fill=False, edgecolor=mt.RED,
                        lw=2.0, label="filter window at row 2, col 1", zorder=4))
for i in range(si, si + 3):                 # the nine numbers actually multiplied
    for j in range(sj, sj + 3):
        axI.text(j, i, f"{image[i, j]:.0f}", ha="center", va="center", fontsize=9,
                 color="white" if image[i, j] > 0.5 else "#333333", zorder=5)
axI.set_title("input image  (10 × 10)", fontsize=11)
axI.set_xticks(range(0, 10, 2))
axI.set_yticks(range(0, 10, 2))
axI.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.92,
           edgecolor=SPINE, fontsize=8.5, handlelength=1.2)

# --- kernel ------------------------------------------------------------------
axK.imshow(kernel, cmap="RdBu_r", vmin=-1.6, vmax=1.6, interpolation="nearest")
# Cell borders on the minor ticks, so the panel reads as a 3x3 and not three bars.
axK.set_xticks(np.arange(-0.5, 3, 1), minor=True)
axK.set_yticks(np.arange(-0.5, 3, 1), minor=True)
axK.grid(which="minor", color=SPINE, lw=1.4)   # visible on the pale center column too
axK.tick_params(which="minor", length=0)
for i in range(3):
    for j in range(3):
        axK.text(j, i, weight_label(kernel[i, j]), ha="center", va="center",
                 fontsize=13, color="#333333")
axK.set_title("vertical-edge kernel  (3 × 3)", fontsize=11)
# Say the scale: these are weights on their own scale, not responses on the colorbar.
axK.set_xlabel("weights −1, 0, +1", fontsize=9, color="#666666", labelpad=6)
axK.set_xticks([])
axK.set_yticks([])

# --- feature map -------------------------------------------------------------
im = axF.imshow(feature_map, cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
axF.scatter([sj], [si], s=70, facecolors="none", edgecolors=mt.RED, linewidths=2.0)
# Annotated inline rather than in a legend box: the flat zero band through the
# middle is half the map's story and should not be sat on.
axF.annotate(f"{feature_map[si, sj]:+.0f}", xy=(sj + 0.55, si), fontsize=11,
             fontweight="bold", color=mt.RED, ha="left", va="center")
axF.set_title("feature map  (8 × 8)", fontsize=11)
axF.set_xticks(range(0, 8, 2))
axF.set_yticks(range(0, 8, 2))

for ax in (axI, axK, axF):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(SPINE)
    ax.tick_params(colors="#666666", labelsize=9, length=3)

# Positions are only final once aspect='equal' has been applied, so draw first.
# The colorbar then gets its own axes at exactly the feature map's height, instead
# of being carved out of it — otherwise the result panel is the smallest of the three.
fig.canvas.draw()
pI, pK, pF = (ax.get_position() for ax in (axI, axK, axF))

cax = fig.add_axes([pF.x1 + 0.014, pF.y0, 0.014, pF.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label("response", fontsize=10)
cb.set_ticks([-3, 0, 3])
cb.ax.tick_params(labelsize=9, colors="#666666")
cb.outline.set_edgecolor(SPINE)

# Read the row as the equation in the defbox above it.
y_mid = (pI.y0 + pI.y1) / 2
fig.text((pI.x1 + pK.x0) / 2, y_mid, "∗", ha="center", va="center",
         fontsize=19, color="#666666")
fig.text((pK.x1 + pF.x0) / 2, y_mid, "=", ha="center", va="center",
         fontsize=19, color="#666666")

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch10_01_convnets.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)

for name, ax in zip(("input", "kernel", "featmap"), (axI, axK, axF)):
    bb = ax.get_position()
    print(f"  panel {name:8} width={bb.width:.4f} height={bb.height:.4f}")
