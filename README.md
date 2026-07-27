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
  `#076FA1` / `#2FC1D3`, with a sharp red `#E3120B` used sparingly (the brand
  circle, the Run button, and the active tab underline).
- **Type** — Fraunces (display), Source Serif 4 (body), Inconsolata (code).
  **Self-hosted** from `fonts/` (see `fonts.css`), not loaded from the Google Fonts
  CDN: all three are OFL-1.1, so serving them ourselves is permitted, and it keeps
  reader IP addresses off a third-party host. ~304 KB for an English reader; the
  latin-ext and Greek subsets load only if a page uses their characters.
- **Charts** — a clean editorial style: right-hand axis labels, minimal
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

Each page is one `.qmd` under `chapters/`, named for the notebook it accompanies
(`ch02_01_polynomial_regression.qmd`, `ch07_02_applied.qmd`, and so on). The quickest
path is to copy `chapters/ch02_02_linear_regression_ols.qmd` (the fully worked
exemplar) and adapt it. The parts that matter:

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
(exploration callout), `.lm-chart` (editorial inline-SVG figure),
`.colab`/`.colab-btn` (Colab affordances).

Then register the page in `_quarto.yml` under `website.sidebar.contents`.

### Which runtime for which chapter?

The split is by *stack*, not by chapter number: anything that needs PyTorch goes to
Colab, everything else runs in the browser.

| Pages | Stack | How it runs |
|-------|-------|-------------|
| ch02–ch08, `ch09_01_backpropagation` | NumPy / scikit-learn, base R | `{pyodide}` / `{webr}` — in the browser |
| `ch09_02`–`ch12_01` | PyTorch | **Open in Colab** button (`.colab-btn`) |
| `ch02_03_bayesian_regression`, `ch13_01_unsupervised` | mixed | live cells **and** a Colab link |

Two pages do not follow the chapter number. `ch09_01_backpropagation` is browser-run
because it builds backpropagation from NumPy, not PyTorch. `ch02_03` and `ch13_01`
carry both: live cells for the part that runs in base Python and R, plus a Colab
button for the PyMC and Fashion-MNIST work that does not.

PyTorch is too heavy for a browser sandbox, so the PyTorch pages link out to Colab
instead of running inline. See `chapters/ch10_01_convnets.qmd` for the pattern.

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

## License

The site's own content (`.qmd` pages, `theme.scss`, figures) is MIT licensed; see
`LICENSE`.

**Redistributed here** (so their notices must travel with the build):

- **quarto-live** — vendored under `_extensions/r-wasm/live/` (792 KB of Lua, JS and
  OJS templates), MIT, notice at `_extensions/r-wasm/live/LICENSE`.
- **WebR's JavaScript API** — quarto-live bundles it into
  `site_libs/quarto-contrib/live-runtime/live-runtime.js`, which the published site
  serves. This is the wrapper only (its webR/R version constants and worker-channel
  classes are identifiable in the bundle); it is *not* the R binary. Per webR's
  `LICENCE.md`, the repository's non-binary software is MIT and none of the JS API
  sources carry a contrary per-file header, so this portion is MIT. Notice at
  `THIRD_PARTY_NOTICES.md`.
- The usual Quarto front-end libraries Quarto emits into `site_libs/` — Bootstrap
  and Bootstrap Icons, clipboard.js, anchor.js, popper, tippy.js, Algolia
  autocomplete (all MIT), the Observable runtime (ISC), and Fuse.js (Apache-2.0).

**Not redistributed** (fetched by the reader's browser at page load):

- **R itself.** The webR distribution binaries — `R.wasm` is ~18 MB, plus the R
  standard library — are **GPL-3.0**: webR states it licenses its binaries under
  GPL-3 to stay compatible with the GPL code they contain (R, libgfortran). They are
  served from `webr.r-wasm.org`, not from here; there is no `.wasm` anywhere in this
  repository or in `_site/`.
- **Pyodide** (MPL-2.0), loaded from `cdn.jsdelivr.net` by a 23 KB worker shim that
  `importScripts()` it. No Pyodide code is bundled, so MPL-2.0 §3.1 source
  obligations are not triggered here.

The practical upshot: the copyleft components are executed in the reader's browser
from their upstream publishers, never conveyed by this repository, so the site's own
MIT licensing stands. If the site were ever changed to self-host the runtimes — to
drop the CDN dependency, say — that would become conveying GPL-3 material and would
need a fresh look.

## Toolchain and maintenance

Pinned, reproducible build (verified versions):

| Tool | Version | Role |
|------|---------|------|
| Quarto | 1.9.38 | site generator |
| `quarto-live` (vendored in `_extensions/`) | upstream `main` @ `12fb30a5dd5e` (2026-06-08) | the `{pyodide}`/`{webr}` live cells |
| Pyodide / WebR | loaded from CDN at runtime by `quarto-live` | the in-browser Python / R |

The `quarto-live` extension is **committed**, so the build is reproducible and won't drift
on its own. Note that the version string inside `_extensions/r-wasm/live/_extension.yml`
reads `0.1.3-dev` and is **not** a useful identifier: upstream ships that same string at
every release, including v0.2.0. The vendored copy is pinned by commit instead, recorded
in the table above; it is at or ahead of v0.2.0 (2026-05-22). Because `quarto add
r-wasm/quarto-live` pulls whatever `main` currently holds rather than a release tag,
re-vendoring is a deliberate act: run it, then `git diff _extensions/` to see exactly what
moved, update the commit in the table above, re-render, and re-run the cross-language
check before publishing. Pyodide/WebR are fetched
from a CDN by the extension, so a major upstream change there can surface only at runtime —
after any `quarto-live` bump, click through a Python and an R cell on a real page to confirm
they still execute. Cross-language cell verification (the porting spec's `Rscript`/Pyodide
check) is the standing maintenance task: every classical page is two code paths that must
agree, kept tractable by restricting R to base R + `stats`.
