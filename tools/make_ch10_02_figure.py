"""Hero figure for the Chapter 10 applied page — treatment B, per FIGURE_SPEC.md.

Reproduces the page's own computation: the CNN of ../notebooks/ch10_02_applied.ipynb,
trained on the same 15,000-image Fashion-MNIST subsample with the same seed, same
architecture (two conv+pool stages, one residual block, linear head), same
AdamW/cross-entropy loop (batch 128, 10 epochs). Shows the sixteen learned
first-layer 3x3 filters as a grid, and a row of held-out test images with the
trained model's predicted class and confidence.

Requires network access on first run (fetch_openml caches ~30 MB locally). If the
download fails, synthetic Gaussian-blob "image" data of the same shape (1x28x28,
10 classes) is substituted so the figure still renders; this is noted in stdout
if it happens.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

torch.manual_seed(0)
np.random.seed(0)

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---- data: same subsample, split, and image reshape as the notebook ----
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
    to_images = lambda A: torch.tensor(A).reshape(-1, 1, 28, 28)
    X_tr_t, X_te_t = to_images(X_tr), to_images(X_te)
    y_tr_t, y_te_t = torch.tensor(y_tr), torch.tensor(y_te)
    print(f"Fashion-MNIST loaded: {len(X_tr_t)} train, {len(X_te_t)} test")
except Exception as exc:  # pragma: no cover - network fallback
    print(f"fetch_openml failed ({exc}); substituting synthetic data of the same shape")
    rng = np.random.default_rng(0)
    n_tr, n_te = 12000, 3000
    centers = rng.normal(scale=1.5, size=(10, 784)).astype("float32")
    y_tr_np = rng.integers(0, 10, n_tr)
    y_te_np = rng.integers(0, 10, n_te)
    X_tr_np = np.clip(centers[y_tr_np] + rng.normal(scale=1.0, size=(n_tr, 784)), 0, 1).astype("float32")
    X_te_np = np.clip(centers[y_te_np] + rng.normal(scale=1.0, size=(n_te, 784)), 0, 1).astype("float32")
    X_tr_t = torch.tensor(X_tr_np).reshape(-1, 1, 28, 28)
    X_te_t = torch.tensor(X_te_np).reshape(-1, 1, 28, 28)
    y_tr_t, y_te_t = torch.tensor(y_tr_np), torch.tensor(y_te_np)


# ---- model: identical to the notebook ----
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            ResidualBlock(32),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


model = CNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params / 1e3:.0f}k parameters")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size, epochs = 128, 10


@torch.no_grad()
def accuracy(X, Y):
    model.eval()
    correct = torch.cat([
        (model(X[i : i + 512]).argmax(1) == Y[i : i + 512])
        for i in range(0, len(X), 512)
    ])
    model.train()
    return correct.float().mean().item()


for epoch in range(epochs):
    order = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), batch_size):
        batch = order[i : i + batch_size]
        loss = loss_fn(model(X_tr_t[batch]), y_tr_t[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    acc = accuracy(X_te_t, y_te_t)
    print(f"epoch {epoch + 1:2d} | train loss {loss.item():.3f} | test accuracy {acc:.3f}")

final_acc = accuracy(X_te_t, y_te_t)
print(f"final test accuracy: {final_acc:.3f}")

# ---- figure: learned first-layer filters + a row of labelled predictions ----
conv1 = model.features[0]
filters = conv1.weight.detach().numpy()[:, 0, :, :]  # (16, 3, 3)
vmax = np.abs(filters).max()

model.eval()
with torch.no_grad():
    n_show = 8
    logits = model(X_te_t[:n_show])
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
model.train()

for i in range(n_show):
    truth = classes[y_te_t[i].item()]
    guess = classes[pred[i].item()]
    mark = "OK" if truth == guess else "WRONG"
    print(f"  shown[{i}] true={truth:14s} pred={guess:14s} conf={conf[i].item():.2f}  {mark}")

fig = plt.figure(figsize=(11.0, 4.6))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.55], wspace=0.28)

# left: 4x4 grid of the 16 learned 3x3 filters
filt_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[0], wspace=0.12, hspace=0.12)
for k in range(16):
    ax = fig.add_subplot(filt_gs[k // 4, k % 4])
    ax.imshow(filters[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.6)
fig.text(0.06, 0.94, "first-layer filters (3×3, learned)", fontsize=11.5, color=mt.FS_BLUE)

# right: a row of test predictions with confidence
pred_gs = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=outer[1], wspace=0.15)
for i in range(n_show):
    ax = fig.add_subplot(pred_gs[0, i])
    ax.imshow(X_te_t[i, 0], cmap="gray_r")
    ax.set_xticks([]); ax.set_yticks([])
    correct = pred[i].item() == y_te_t[i].item()
    color = mt.BLUE if correct else mt.RED
    ax.set_title(f"{classes[pred[i]]}\n{conf[i].item():.0%}", fontsize=8.3, color=color)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.6)
fig.text(0.365, 0.94, "test predictions, with confidence (blue = correct, red = wrong)",
          fontsize=11.5, color=mt.FS_BLUE)

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch10_02_applied.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.18)
print("saved:", out)
