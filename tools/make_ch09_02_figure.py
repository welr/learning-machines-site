"""Hero figure for the Chapter 9 applied page — treatment B, per FIGURE_SPEC.md.

Reproduces the page's own computation: the MLP of ../notebooks/ch09_02_applied.ipynb,
trained on the same 15,000-image Fashion-MNIST subsample with the same seed, same
architecture (784-256-128-10, ReLU, dropout 0.2), same AdamW/cross-entropy loop
(batch 128, 12 epochs). Plots the per-epoch training loss and held-out test accuracy
that the notebook's own cell 10 produces.

Requires network access on first run (fetch_openml caches ~30 MB locally). If the
download fails, synthetic Gaussian-blob data of the same shape (784-d, 10 classes)
is substituted so the figure still renders; this is noted in stdout if it happens.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

torch.manual_seed(0)
np.random.seed(0)

# ---- data: same subsample and split as the notebook ----
try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)
    X = (X / 255.0).astype("float32")
    y = y.astype(int)
    subset = np.random.permutation(len(X))[:15000]
    X, y = X[subset], y[subset]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    print(f"Fashion-MNIST loaded: {len(X_tr)} train, {len(X_te)} test")
except Exception as exc:  # pragma: no cover - network fallback
    print(f"fetch_openml failed ({exc}); substituting synthetic data of the same shape")
    rng = np.random.default_rng(0)
    n_tr, n_te = 12000, 3000
    centers = rng.normal(scale=1.5, size=(10, 784)).astype("float32")
    y_tr = rng.integers(0, 10, n_tr)
    y_te = rng.integers(0, 10, n_te)
    X_tr = (centers[y_tr] + rng.normal(scale=1.0, size=(n_tr, 784))).astype("float32")
    X_te = (centers[y_te] + rng.normal(scale=1.0, size=(n_te, 784))).astype("float32")
    X_tr, X_te = np.clip(X_tr, 0, 1), np.clip(X_te, 0, 1)

X_tr_t, y_tr_t = torch.tensor(X_tr), torch.tensor(y_tr)
X_te_t, y_te_t = torch.tensor(X_te), torch.tensor(y_te)


# ---- model: identical to the notebook ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


device = "cpu"
model = MLP().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params / 1e3:.0f}k parameters")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size, epochs = 128, 12


@torch.no_grad()
def accuracy(X, Y):
    model.eval()
    preds = model(X).argmax(dim=1)
    model.train()
    return (preds == Y).float().mean().item()


history = []
for epoch in range(epochs):
    order = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), batch_size):
        batch = order[i : i + batch_size]
        logits = model(X_tr_t[batch])
        loss = loss_fn(logits, y_tr_t[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    acc = accuracy(X_te_t, y_te_t)
    history.append((epoch + 1, loss.item(), acc))
    print(f"epoch {epoch + 1:2d} | train loss {loss.item():.3f} | test accuracy {acc:.3f}")

ep, tr_loss, te_acc = zip(*history)
print(f"final test accuracy: {te_acc[-1]:.3f}")

# ---- figure: two single-series panels, treatment B ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

axL.plot(ep, tr_loss, color=mt.BLUE, lw=2.2, marker="o", ms=4)
axR.plot(ep, te_acc, color=mt.CYAN, lw=2.2, marker="o", ms=4)

for ax, title, ylabel in (
    (axL, "training loss", "cross-entropy loss"),
    (axR, "test accuracy", "accuracy on held-out images"),
):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(SPINE)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#666666", labelsize=11)
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, color=mt.FS_BLUE)

fig.tight_layout()

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch09_02_applied.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
