# Hero-figure spec for the companion site

Goal: give each classical page ONE pre-rendered hero figure — the concept-carrying
visual — generated from that page's own computation, in the book's editorial style,
placed early (before the first code tabset). The figure is what the reader then
reproduces with the live cells. Gold-standard reference: `scratchpad/pilot_fig.py`
(the Ch 5 bias–variance U-curve) and the result in `figures/ch05_bias_variance.png`.

## Style recipe (use EXACTLY — this is what keeps 16 figures consistent)

```python
import sys
sys.path.insert(0, "/Users/gregorywheeler/Dropbox/A_COURSES/Machine Learning I/Lecture NOTES/MLone book/notebooks")
import mlone_theme as mt          # the book's plot toolkit
import matplotlib.pyplot as plt
import numpy as np
mt.set_book_mode()                # blue editorial palette + base rcParams
rng = np.random.default_rng(0)    # FIXED seed — the figure must be reproducible
```

- **Palette (use these, no others):** `mt.BLUE` `#076FA1` (primary series / data),
  `mt.FS_BLUE` `#31417A` (second series / fit line), `mt.RED` `#E3120B`
  (error / the "bad" / emphasis), `mt.CYAN` `#2FC1D3` and `mt.ORANGE` `#9E4F00`
  (third/fourth series only if needed), `mt.GRAY` `#666666` (baselines, reference
  lines). White background. NO green.
- **Size:** line/scatter charts `figsize=(7.0, 4.0)`. 2-D feature-space plots
  (decision boundaries, loss-surface contours) `figsize=(6.6, 5.0)`.
- **Axes:** `mt.apply_book_style(ax, y_label_right=False)` for metric charts (left
  labels, light horizontal grid, only the bottom spine). For 2-D feature-space plots,
  don't force that — keep both feature axes, white background, a light frame, minimal
  ticks; just use the palette.
- **NO baked-in title or subtitle on the figure.** The title lives in the page's
  `<figcaption>` (below). Do set an `ax.set_xlabel(...)` (and y-label if a 2-D plot).
- **Labels, not legends.** For multiple series, label each at its right end with
  `mt.add_direct_label(ax, x, y, "name", color)`, and widen `xlim` ~15% to make room.
  Avoid `ax.legend()`.
- **Save:**
  ```python
  fig.tight_layout()
  fig.savefig(".../companion-site/figures/<PAGE_STEM>.png", dpi=150,
              bbox_inches="tight", facecolor="white", pad_inches=0.12)
  ```
  where `<PAGE_STEM>` is the page filename without extension (e.g.
  `ch03_01_gradient_descent`).

## Faithfulness

Generate the figure from the page's OWN computation. Read the page's key `{pyodide}`
cell, reuse its data-generation and model code (re-styled per the recipe), so the
figure is the same result the reader reproduces. Keep `rng`/seed fixed so it's stable.
Don't invent a different dataset or a prettier-but-fake curve.

## Embedding (in the `.qmd`)

Insert this block AFTER the hero/intro/`defbox` and BEFORE the first
`::: {.panel-tabset ...}`:

```
` ``{=html}
<figure class="lm-figure">
<img src="../figures/<PAGE_STEM>.png" alt="<one-line factual description of the plot>">
<figcaption><strong><Short title>.</strong> <One sentence on what it shows.> This is the result the code below reproduces.</figcaption>
</figure>
` ``
```

(`.lm-figure` is already styled in `theme.scss`.) Keep the caption to a bold lead +
one descriptive sentence + the "reproduces" line.

## Verify

Run your figure script with `MPLBACKEND=Agg /Users/gregorywheeler/otter_env/bin/python`
(it must create the PNG), then `quarto render chapters/<PAGE>.qmd` (must exit 0). Confirm
the PNG is non-trivial (> 20 KB) and the page embeds it.
