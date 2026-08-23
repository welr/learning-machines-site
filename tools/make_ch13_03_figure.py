"""Hero figure for the Chapter 13 LSA page.

The latent plane: 36 documents and 12 terms of the constructed two-topic corpus,
plotted in the coordinates of the two leading singular directions. The synonym
pairs (loan/credit, oven/stove) land on top of each other although they never
co-occur -- the moment the page is built around. Same deterministic corpus as the
page's cells and the companion notebook (ch13_03_lsa.ipynb), so the figure shows
exactly what the reader's own run will show.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mlone_theme as mt

mt.set_book_mode()

VOCAB = ["loan", "credit", "bank", "interest", "rate", "payment",
         "oven", "stove", "flour", "bake", "dough", "recipe"]
COL = {t: j for j, t in enumerate(VOCAB)}


def finance_doc(i):
    c = {}
    c["loan" if i % 2 == 0 else "credit"] = 1 + (i % 3 == 0)
    c["bank"] = 2 + (i % 5 == 0)
    c["interest"] = 2
    if i % 3 == 1:
        c["rate"] = 1
    if i % 4 == 2:
        c["payment"] = 1
    return c


def cooking_doc(i):
    c = {}
    c["oven" if i % 2 == 0 else "stove"] = 1 + (i % 3 == 0)
    c["flour"] = 2 + (i % 4 == 0)
    c["bake"] = 2
    if i % 3 == 1:
        c["dough"] = 1
    if i % 4 == 2:
        c["recipe"] = 1
    return c


docs = [finance_doc(i) for i in range(20)] + [cooking_doc(i) for i in range(16)]
topic = ["finance"] * 20 + ["cooking"] * 16
X = np.zeros((len(docs), len(VOCAB)))
for i, counts in enumerate(docs):
    for term, c in counts.items():
        X[i, COL[term]] = c

U, d, Vt = np.linalg.svd(X, full_matrices=False)
k = 2
for j in range(k):                      # canonical signs: topic loadings positive
    if Vt[j].sum() < 0:
        Vt[j] *= -1
        U[:, j] *= -1
doc_coords = U[:, :k] * d[:k]
term_coords = Vt[:k, :].T * d[:k]

fig, ax = plt.subplots(figsize=(9, 6))
for name, marker, color in [("finance", "o", mt.FS_BLUE), ("cooking", "s", mt.GREEN)]:
    idx = [i for i in range(len(docs)) if topic[i] == name]
    ax.scatter(doc_coords[idx, 0], doc_coords[idx, 1], marker=marker,
               color=color, alpha=0.55, s=55, label=f"{name} documents")
ax.scatter(term_coords[:, 0], term_coords[:, 1], marker="x", color=mt.RED,
           s=60, linewidths=2, label="terms", zorder=3)
OFFSETS = {"loan": (6, 8), "credit": (6, -16), "oven": (7, 7), "stove": (7, -17),
"rate": (-4, 9), "payment": (-4, -18), "dough": (6, 6), "recipe": (6, -16)}
for j, term in enumerate(VOCAB):
    ax.annotate(term, term_coords[j], textcoords="offset points",
                xytext=OFFSETS.get(term, (5, 5)), fontsize=10)
ax.set_xlabel("latent direction 1 (finance)")
ax.set_ylabel("latent direction 2 (cooking)")
ax.legend(loc="upper right")
mt.apply_book_style(ax)
plt.tight_layout()

out = pathlib.Path(__file__).resolve().parents[1] / "figures" / "ch13_03_lsa.png"
plt.savefig(out, dpi=200)
print("wrote", out)
