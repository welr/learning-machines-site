"""Hero figure for the Chapter 11 companion page — treatment B, per FIGURE_SPEC.md.

One head of self-attention over the nine-token sentence "the cat drank the milk
because it was thirsty", from hand-built embeddings chosen so the pattern is
readable rather than grey mush.

Everything between the two rule comments below is copied verbatim into the page's
live `{pyodide}` cell, so a reader who clicks Run reproduces this picture.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

# --- shared with the live cell -------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Eight semantic roles. A real model learns its embeddings; we hand-build them so
# every number below can be read off the sentence.
ROLES = ["ANIMATE", "OBJECT", "DET", "VERB", "CONN", "PROP", "PRON", "COP"]
ROLE = {r: i for i, r in enumerate(ROLES)}
d_k = len(ROLES)

sentence = [("the", "DET"), ("cat", "ANIMATE"), ("drank", "VERB"),
            ("the", "DET"), ("milk", "OBJECT"), ("because", "CONN"),
            ("it", "PRON"), ("was", "COP"), ("thirsty", "PROP")]
tokens = [tok for tok, _ in sentence]
X = np.eye(d_k)[[ROLE[r] for _, r in sentence]]   # one-hot role embeddings

# The query projection is the only thing we shape: row r says what a token of
# role r goes looking for. Keys and values stay raw, so a score is read straight
# off this table.
W_Q = np.zeros((d_k, d_k))
W_Q[ROLE["ANIMATE"], ROLE["VERB"]] = 6.0      # a subject looks for its verb
W_Q[ROLE["ANIMATE"], ROLE["ANIMATE"]] = 3.0
W_Q[ROLE["OBJECT"], ROLE["VERB"]] = 6.0       # so does an object
W_Q[ROLE["OBJECT"], ROLE["OBJECT"]] = 3.0
W_Q[ROLE["DET"], ROLE["ANIMATE"]] = 5.0       # a determiner looks for a noun
W_Q[ROLE["DET"], ROLE["OBJECT"]] = 5.0
W_Q[ROLE["VERB"], ROLE["ANIMATE"]] = 4.5      # a verb looks for its arguments
W_Q[ROLE["VERB"], ROLE["OBJECT"]] = 4.5
W_Q[ROLE["CONN"], ROLE["VERB"]] = 3.0         # a connective joins clause to clause
W_Q[ROLE["CONN"], ROLE["PROP"]] = 4.0
W_Q[ROLE["PROP"], ROLE["ANIMATE"]] = 5.0      # a property looks for what it describes
W_Q[ROLE["PRON"], ROLE["ANIMATE"]] = 8.0      # a pronoun hunts its referent
W_Q[ROLE["PRON"], ROLE["OBJECT"]] = 2.0
W_Q[ROLE["PRON"], ROLE["PROP"]] = 2.0
W_Q[ROLE["COP"], ROLE["PROP"]] = 5.0          # a copula links subject to property
W_Q[ROLE["COP"], ROLE["ANIMATE"]] = 4.0
W_K = np.eye(d_k)
W_V = np.eye(d_k)


def softmax_rows(S):
    """Row-wise softmax. Subtracting the row max first keeps exp() from overflowing."""
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


Q, K, V = X @ W_Q, X @ W_K, X @ W_V
scores = Q @ K.T / np.sqrt(d_k)               # scaled query-key similarity
A = softmax_rows(scores)                      # the kernel weights of Chapter 8
context = A @ V                               # weighted average of the values

assert np.allclose(A.sum(axis=1), 1.0), "each row must be a distribution"
n = len(tokens)
print("attention weights (row attends to column)")
print(f"{'':>8} " + " ".join(f"{t:>7}" for t in tokens))
for i, t in enumerate(tokens):
    print(f"{t:>8} " + " ".join(f"{v:7.3f}" for v in A[i]))
print(f"\nrow sums: {np.round(A.sum(axis=1), 12)}")
print(f"'it' -> 'cat' {A[6, 1]:.3f}; next largest in that row {np.sort(A[6])[-2]:.3f}")
print(f"the two 'the' rows differ by at most {np.abs(A[0] - A[3]).max():.1e}")
unscaled = softmax_rows(Q @ K.T)
p, p_u = A.max(), unscaled.max()
print(f"largest weight anywhere: scaled {p:.3f}, unscaled {p_u:.3f}")
print(f"softmax slope p(1-p) there: scaled {p * (1 - p):.3f}, unscaled {p_u * (1 - p_u):.4f}")

fig, ax = plt.subplots(figsize=(7.0, 5.0))
im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=A.max())
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("attention weight", fontsize=11)
cb.ax.tick_params(labelsize=9)
cb.outline.set_edgecolor("#cfccc2")
for i in range(n):
    for j in range(n):
        if A[i, j] >= 0.10:
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if A[i, j] > 0.45 else "#31417A")
ax.set_xticks(range(n)); ax.set_xticklabels(tokens, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(tokens)
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="#e6e3da", lw=0.8)
ax.tick_params(which="minor", length=0)
ax.tick_params(colors="#666666", labelsize=10)
for s in ax.spines.values():
    s.set_color("#cfccc2")
ax.set_xlabel("key — token attended to", fontsize=11)
ax.set_ylabel("query — token attending", fontsize=11)
# --- end shared block ----------------------------------------------------------

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch11_01_attention_transformers.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
print("context vector for 'it':", np.round(context[6], 3))
print("ROLES order:", ROLES)
