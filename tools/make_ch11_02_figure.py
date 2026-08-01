"""Hero figure for the Chapter 11 applied page — treatment B, per FIGURE_SPEC.md.

Reproduces the page's own computation from ../notebooks/ch11_02_applied.ipynb: an
untrained causal-attention head's weight matrix (cell 6, block_size=8, n_embd=32,
random seed 0), beside the attention matrix learned by the one-head RecallModel
after 600 training steps on the "recall the first token" task (cells 10-11). The
notebook has no sinusoidal positional encoding to plot — its positions are a
learned nn.Embedding — so the hero pairs the two attention heatmaps the notebook
itself produces: the mask's structure before training, and where the head learns
to look after.

No dataset download is needed; this is pure PyTorch on synthetic sequences, so
there is no network-failure fallback to report.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

torch.manual_seed(0)
np.random.seed(0)


class Head(nn.Module):
    """One head of causal self-attention; stores its weights so we can inspect them."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_weights = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        self.attn_weights = weights.detach()
        return weights @ self.value(x)


# ---- left panel: an untrained head, same as the notebook's "See it compute" cell ----
block_size, n_embd = 8, 32
demo_head = Head(n_embd, n_embd, block_size)
x_demo = torch.randn(1, block_size, n_embd)
demo_head(x_demo)
A_untrained = demo_head.attn_weights[0].numpy()

# ---- right panel: the RecallModel, trained to look back to position 0 ----
V, T = 12, 8


def get_batch(batch=64):
    xb = torch.randint(0, V, (batch, T))
    yb = xb[:, 0:1].expand(batch, T).contiguous()
    return xb, yb


class RecallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, n_embd)
        self.pos = nn.Embedding(T, n_embd)
        self.head = Head(n_embd, n_embd, T)
        self.norm = nn.LayerNorm(n_embd)
        self.out = nn.Linear(n_embd, V)

    def forward(self, x, y=None):
        B, Tt = x.shape
        h = self.tok(x) + self.pos(torch.arange(Tt))
        h = self.norm(self.head(h))
        logits = self.out(h)
        loss = None if y is None else F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        return logits, loss


model = RecallModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(601):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 150 == 0:
        with torch.no_grad():
            xv, yv = get_batch(256)
            logits, _ = model(xv)
            acc = (logits.argmax(-1) == yv).float().mean().item()
        print(f"step {step:3d} | loss {loss.item():.3f} | accuracy {acc:.3f}")

with torch.no_grad():
    xv, yv = get_batch(512)
    logits, _ = model(xv)
    final_acc = (logits.argmax(-1) == yv).float().mean().item()
print(f"final recall accuracy: {final_acc:.3f}")

A_learned = model.head.attn_weights.mean(0).numpy()

# ---- figure: two attention-matrix panels, treatment B ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.6))

for ax, A, title in (
    (axL, A_untrained, "untrained head: causal mask only"),
    (axR, A_learned, "trained head: learns to look at position 0"),
):
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.set_xlabel("key position", fontsize=11)
    ax.set_ylabel("query position", fontsize=11)
    ax.set_xticks(range(A.shape[1]))
    ax.set_yticks(range(A.shape[0]))
    ax.tick_params(colors="#666666", labelsize=9)
    ax.set_title(title, fontsize=11, color=mt.FS_BLUE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("attention weight", fontsize=9.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor(SPINE)

fig.tight_layout()

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch11_02_applied.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
