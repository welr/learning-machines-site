# Learning Machines — interactive companion site

The web companion to *Learning Machines: A Statistical Introduction*. Each chapter
pairs the book's exposition with code the reader can **run in the browser** — Python
via [Pyodide](https://pyodide.org) and R via [WebR](https://docs.r-wasm.org/webr/) —
or **open in Google Colab** for the deep-learning chapters.

Built with [Quarto](https://quarto.org) + the
[`quarto-live`](https://github.com/r-wasm/quarto-live) extension. **No server, no
author compute.** The site is static HTML on GitHub Pages; every code cell executes
on the reader's own machine, in their browser.

## Design

The visual identity is aligned to the printed book:

- **Palette** — Frankfurt School blue `#31417A` (primary), accent blues
  `#076FA1` / `#2FC1D3`, with the Economist red `#E3120B` used sparingly (the
  top-left tab, the Run button, the active tab underline).
- **Type** — Fraunces (display), Source Serif 4 (body), Inconsolata (code).
- **Charts** — Economist editorial style: right-hand axis labels, minimal
  gridlines, a red title tab.

All of this lives in `theme.scss` (a Quarto SCSS theme: `scss:defaults` for the
Bootstrap variables, `scss:rules` for the component styling).

## Local preview

```bash
quarto preview        # live-reloading local server
quarto render         # build the static site into _site/
```

Requires Quarto ≥ 1.5. The `quarto-live` extension is already vendored under
`_extensions/`; no extra install is needed.

## Adding a chapter

Each chapter is one `.qmd` under `chapters/`. The quickest path is to copy
`chapters/02-regression-models.qmd` (the fully worked exemplar) and adapt it. The
parts that matter:

```markdown
---
title: "Ordinary Least Squares"
engine: markdown          # <-- required: keeps Quarto from invoking a build-time
---                       #     kernel. The {pyodide}/{webr} cells run in-browser.

::: {.lm-hero}            <!-- B-style chapter opener -->
[Chapter 2 · Regression Models]{.eyebrow}
# Ordinary Least Squares
[One-line dek in italic Fraunces.]{.dek}
:::

::: {.panel-tabset}      <!-- dual-language: Python + R -->
## Python
` ``{pyodide}
...python...
` ``
## R
` ``{webr}
...R...
` ``
:::
```

Custom components available in `theme.scss`: `.lm-hero` (chapter opener),
`.defbox` (definition box, echoes the book's `myDefinition`), `.explore`
(exploration callout), `.lm-chart` (Economist-style inline-SVG figure),
`.colab`/`.colab-btn` (Colab affordances).

Then register the page in `_quarto.yml` under `website.sidebar.contents`.

### Which runtime for which chapter?

| Chapters | Stack | How it runs |
|----------|-------|-------------|
| 1–8 (classical ML) | NumPy / scikit-learn, base R | `{pyodide}` / `{webr}` — in the browser |
| 9–12 (deep learning) | PyTorch | **Open in Colab** button (`.colab-btn`) |

PyTorch is too heavy for a browser sandbox, so the deep-learning chapters link out
to Colab instead of running inline. See `chapters/09-neural-networks.qmd` for the
pattern.

## Publishing

`.github/workflows/publish.yml` renders the site and deploys it to GitHub Pages on
every push to `main`. To turn it on:

1. Push this directory to a GitHub repository (it is laid out as the repo root).
2. In **Settings → Pages**, set **Source → GitHub Actions**.
3. Push to `main`. The site builds and goes live at the Pages URL.

Because nothing executes at build time, the Action only needs Quarto — it installs
in seconds and the render is reproducible.

## A harmless build message

`quarto render` prints one warning:

```
Error adding css vars block SCSSParsingError: Expecting punctuation: "}"
The resulting CSS file will not have SCSS color variables exported as CSS.
```

This is a **false positive** in Quarto's regex-based color-variable analyzer, not a real
error. The compiled CSS is valid (braces balance exactly; the site renders correctly),
and the only stated effect — Bootstrap's `--bs-*` CSS custom properties not being
auto-derived from the SCSS — does not matter here, because `theme.scss` compiles its
`$variables` to literal color values rather than relying on runtime CSS variables. The
render completes and `_site/` is correct, so the message can be ignored.
