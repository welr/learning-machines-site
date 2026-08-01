"""Hero figure for the Chapter 12 capstone page — treatment B, per FIGURE_SPEC.md.

The capstone's own training run: the character-level GPT of
`notebooks/ch12_01_build_a_gpt.ipynb`, same architecture and hyper-parameters,
trained on Karpathy's *Tiny Shakespeare* with `torch.manual_seed(0)`. The run is
shortened to 600 steps (the notebook does 3000) because this is a figure, not a
benchmark, and the interesting part of the curve is the first few hundred steps.

Sampled text is drawn from the live model at three points along the curve and
annotated onto the plot, so the reader can see the output change as the
cross-entropy falls.

The corpus is downloaded from the same URL the notebook uses and cached under the
system temp directory; nothing is written into the repository except the PNG.
"""

import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/"
                   "Lecture NOTES/MLone book/notebooks")

import math
import os
import tempfile
import urllib.request

import matplotlib
matplotlib.use("Agg")
import mlone_theme as mt, matplotlib.pyplot as plt, numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

mt.set_book_mode()
SPINE, GRID = "#cfccc2", "#e6e3da"

# ---------------------------------------------------------------- the corpus
DATA_URL = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
CACHE = os.path.join(tempfile.gettempdir(), "tiny_shakespeare_input.txt")
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(DATA_URL, CACHE)
text = open(CACHE).read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
print(f"{len(text):,} characters, vocabulary of {vocab_size}")

# ------------------------------------------------- configuration (notebook's)
batch_size, block_size = 32, 64
n_embd, n_head, n_layer = 96, 4, 3
dropout = 0.1
learning_rate = 1e-3
max_iters, eval_interval, eval_iters = 600, 100, 40

torch.manual_seed(0)
device = "cpu"


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------ the model (notebook's)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.norm_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.norm_final(self.blocks(tok + pos))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, vocab_size), targets.view(B * T))
        return logits, loss


model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params / 1e3:.0f}k parameters")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    model.train()
    return idx


# -------------------------------------------------------------- training loop
SAMPLE_AT = (0, 300, 600)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
history, samples = [], {}

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        L = estimate_loss()
        history.append((step, L["train"], L["val"]))
        print(f"step {step:4d} | train {L['train']:.3f} | val {L['val']:.3f} "
              f"| val perplexity {math.exp(L['val']):.2f}")
    if step in SAMPLE_AT:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        s = decode(generate(ctx, 90, temperature=0.8)[0].tolist())[1:]
        samples[step] = s
        print(f"  sample @ {step}: {s!r}")
    if step == max_iters:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

steps, tr, va = (np.array(a) for a in zip(*history))
print(f"\nheld-out loss {va[0]:.3f} -> {va[-1]:.3f} nats "
      f"(perplexity {math.exp(va[0]):.1f} -> {math.exp(va[-1]):.1f})")


def snippet(s, width=30):
    """One line of sampled text, newlines made visible, clipped to `width`."""
    flat = " ".join(s.replace("\n", " / ").split())
    return (flat[:width] + "…") if len(flat) > width else flat


for k in SAMPLE_AT:
    print(f"snippet @ {k}: {snippet(samples[k])!r}")

# -------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.0, 4.6))

ax.plot(steps, tr, color=mt.BLUE, lw=2.0, label="training")
ax.plot(steps, va, color=mt.FS_BLUE, lw=2.0, ls="--", label="held-out")

# sampled text at three points on the held-out curve
mark_x = np.array(SAMPLE_AT)
mark_y = np.array([va[list(steps).index(s)] for s in SAMPLE_AT])
ax.scatter(mark_x, mark_y, s=44, color=mt.RED, zorder=5,
           edgecolors="white", linewidths=1.0, label="text sampled here")

boxes = [(0, 0.40, 0.94), (300, 0.40, 0.75), (600, 0.40, 0.56)]
for step_i, bx, by in boxes:
    txt = f"step {step_i}:  {snippet(samples[step_i])}"
    ax.annotate(
        txt,
        xy=(step_i, va[list(steps).index(step_i)]),
        xytext=(bx, by), textcoords="axes fraction",
        fontsize=7.2, family="monospace", color="#333333", va="center",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                  edgecolor=SPINE, linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color=mt.GRAY, lw=0.8, alpha=0.7,
                        shrinkA=2, shrinkB=4))

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(SPINE)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors="#666666", labelsize=11)
ax.set_xlabel("training step", fontsize=11)
ax.set_ylabel("next-token cross-entropy, nats", fontsize=11)
ax.set_xlim(-25, max_iters + 25)
ax.legend(loc="lower left", frameon=False, fontsize=10.5, handlelength=1.7,
          labelspacing=0.35)

out = ("/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/"
       "MLone book/companion-site/figures/ch12_01_build_a_gpt.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("saved:", out)
