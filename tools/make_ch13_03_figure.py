"""Hero figure for the Chapter 13 LSA page.

Topics surface as blocks: the document-term count matrix before and after rank-2
truncation. Left, the raw counts with rows and columns in an arbitrary order --
what the algorithm actually receives. Right, the rank-2 reconstruction with rows
and columns sorted by their leading latent coordinates: two topic blocks emerge,
and the loan/credit columns (outlined), each half-empty in the raw counts because
the two synonyms never co-occur, come back filled across every finance document.
Same deterministic corpus as the page's cells and the companion notebook.
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


X_hat = U[:, :k] @ np.diag(d[:k]) @ Vt[:k, :]

# Left panel: arbitrary (but deterministic) row and column order
row_shuffle = [(7 * j) % 36 for j in range(36)]        # stride permutation
col_shuffle = ["flour", "loan", "recipe", "bank", "stove", "payment",
               "interest", "oven", "credit", "dough", "bake", "rate"]

# Right panel: rows and columns sorted by leading latent coordinate, per topic
fin_rows  = sorted(range(20), key=lambda i: -doc_coords[i, 0])
cook_rows = sorted(range(20, 36), key=lambda i: -doc_coords[i, 1])
fin_terms  = sorted([t for t in VOCAB[:6]], key=lambda t: -term_coords[COL[t], 0])
cook_terms = sorted([t for t in VOCAB[6:]], key=lambda t: -term_coords[COL[t], 1])
row_sorted, col_sorted = fin_rows + cook_rows, fin_terms + cook_terms

left  = X[np.ix_(row_shuffle, [COL[t] for t in col_shuffle])]
right = X_hat[np.ix_(row_sorted, [COL[t] for t in col_sorted])]

from matplotlib.patches import Rectangle
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
vmax = X.max()
for ax, M, terms, title in [
        (axes[0], left, col_shuffle, "raw counts $\\mathbf{X}$ (arbitrary order)"),
        (axes[1], right, col_sorted,
         "rank-2 reconstruction $\\hat{\\mathbf{X}}$ (sorted by latent coordinate)")]:
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(12))
    ax.set_xticklabels(terms, rotation=60, ha="right", fontsize=10)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=10)
    for j, t in enumerate(terms):                      # track the synonym columns
        if t in ("loan", "credit"):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 36, fill=False,
                                   edgecolor=mt.RED, linewidth=1.8))
axes[0].set_ylabel("36 documents")
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cb.set_label("count")

out = pathlib.Path(__file__).resolve().parents[1] / "figures" / "ch13_03_lsa.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
