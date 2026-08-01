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
| `index`, `ch01_01_baseline` | NumPy, base R | `{pyodide}` / `{webr}` — in the browser |
| ch02–ch08, `ch09_01_backpropagation` | NumPy / scikit-learn, base R | `{pyodide}` / `{webr}` — in the browser |
| `ch09_02`, `ch10_02`, `ch11_02` | PyTorch | **Open in Colab** button (`.colab-btn`) |
| `ch10_01`, `ch11_01`, `ch12_01` | PyTorch, plus one NumPy idea | live cell **and** a Colab link |
| `ch02_03_bayesian_regression`, `ch13_01_unsupervised` | mixed | live cells **and** a Colab link |

Every one of the six deep-learning pages carries a pre-rendered hero figure generated from
that page's own notebook (`tools/make_ch*_figure.py`); before 2026-08-01 they had no images
at all. Three of them also carry one live cell, because the idea at their core does not
actually need PyTorch: convolution by hand (`ch10_01`), scaled dot-product attention
(`ch11_01`), and a character bigram language model (`ch12_01`). The PyTorch work on those
pages still goes to Colab.

Two pages do not follow the chapter number. `ch09_01_backpropagation` is browser-run
because it builds backpropagation from NumPy, not PyTorch. `ch02_03` and `ch13_01`
carry both: live cells for the part that runs in base Python and R, plus a Colab
button for the PyMC and Fashion-MNIST work that does not.

PyTorch is too heavy for a browser sandbox, so the PyTorch pages link out to Colab
instead of running inline. See `chapters/ch10_01_convnets.qmd` for the pattern.

## Cells that run themselves, and reader-driven sliders

Most cells wait for the reader to click Run. Two kinds do not, and both carry a
constraint worth knowing before you add more.

**`#| autorun: true`** runs a cell as soon as the runtime is ready, so the page shows
live output rather than inert code on arrival. It is set on one Python plot cell each on
`index`, `ch01_01`, `ch02_01`, `ch03_01`, `ch05_01`, `ch08_01`, `ch10_01`, `ch11_01` and
`ch12_01` (plus the four slider cells below). Measured cold, the plot appears about four
seconds after load. It costs no
extra network traffic, because the runtimes download on page open regardless (see
"Third-party requests"). Only the Python cell of a tabset autoruns: Python is the default
tab, and autorunning the R twin as well would execute both runtimes on every load for
output nobody is looking at.

An autorun cell must not depend on cells above it — nothing guarantees they have run, and
on `index` and `ch01_01` they have not. Both autorun cells there are written to stand
alone, restating the few lines of data they need rather than inheriting them.

**Sliders** are OJS `viewof` cells wired into a `{pyodide}` cell with `#| input:`. They
are on `ch03_01` (step size), `ch05_01` (noise), `ch06_02` (threshold), and `ch08_02`
(RBF bandwidth). Three rules, each learned the hard way:

- **A slider cell must be self-contained.** A cell with `autorun: false` is *not* re-run
  when its input changes — the reader would have to click Run for the slider to do
  anything — so slider cells need `autorun: true`, which means they run before any cell
  above them and cannot rely on variables those cells define. Where re-deriving the
  inputs would just repeat the page (`ch06_02`), the upstream cell's output is inlined
  instead, with a comment saying so.
- **Re-run on release, not on drag.** The control updates its numeric readout on `input`
  events but only assigns `wrap.value` and dispatches on `change`, so dragging is smooth
  and the Python cell re-runs once, when the reader lets go. Without this, a drag queues
  a redraw per pixel.
- **No `Inputs.range`.** Observable's input library would be one line, but it pulls
  `@observablehq/inputs` and `htl` from `cdn.jsdelivr.net` on page load. The hand-rolled
  `<input type=range>` in `.lm-slider` adds nothing external.

## Exercise cells

`ch01_01` converts its "Try it" prose into real exercise cells: a cell with
`#| exercise: <key>` renders with blanks (`______`) the reader fills in, plus **Start Over**,
**Show Hint** and **Show Solution** buttons. Hints and solutions are sibling cells carrying
the same key and `#| hint: true` / `#| solution: true`. The machinery ships in the vendored
extension and works; nothing needs enabling.

Grading is deliberately not used. `gradethis`-style `#| check:` cells exist in the extension,
but the companion is expository — the book's own guidelines keep assessment out of the
companion material — so exercises here reveal a solution rather than mark an answer.

Exercise keys must be unique per page; the Lua filter raises a build error if two cells share
one, which is the good kind of failure.

## Do not enable `persist`

`#| persist: true` makes an editor remember the reader's edits in `localStorage`. It works,
and it should stay off. Tested 2026-08-01:

1. Load a page with `persist: true` and edit a cell. The edit is saved under the key
   `editor-<full page URL>#pyodide-<N>-contents`.
2. Fix the cell in the `.qmd`, re-render, redeploy.
3. Reload as that returning reader: **the editor still shows the old edit, and the corrected
   source is nowhere on the page.** Verified directly — the restored text was the stale reader
   edit, and the new source string was absent from the editor.

There is a **Start Over** button that would recover the new source, but the reader has no
reason to press it: nothing signals that the page changed. Worse, the storage key is the
block's *sequential index*, so inserting any cell earlier on a page shifts every later block's
identity and restores saved edits into the wrong cells.

The upside — a reader's experiments surviving a reload — is not worth shipping a site that can
silently serve corrected code to nobody.

## Publishing

`.github/workflows/publish.yml` renders the site and deploys it to GitHub Pages on
every push to `main`. To turn it on:

1. Push this directory to a GitHub repository (it is laid out as the repo root).
2. In **Settings → Pages**, set **Source → GitHub Actions**.
3. Push to `main`. The site builds and goes live at the Pages URL.

Because nothing executes at build time, the Action only needs Quarto — it installs
in seconds and the render is reproducible.

## Third-party requests

No third party is named in the *markup*: the webfonts and MathJax are self-hosted, and
Quarto's hardcoded cdnjs ES6 polyfill is removed after render by
`tools/strip_polyfill.py` (that pass is deliberately loud if it ever stops matching, so
it cannot rot into a silent no-op).

The runtimes are a different matter, and an earlier version of this section overstated
the position. **Pyodide and WebR begin downloading when the page opens, not when the
reader clicks Run.** `quarto-live` starts its workers eagerly so the Run button can go
from grey to red — which is exactly what the landing page's "How to read this site" box
describes. Measured on an untouched chapter page with no clicks, a cold load makes about
47 requests to three hosts:

| host | what |
|---|---|
| `cdn.jsdelivr.net` | Pyodide runtime and wheels |
| `webr.r-wasm.org` | WebR runtime |
| `repo.r-wasm.org` | WebR package binaries |

They are left on their CDNs because self-hosting Pyodide means tens of megabytes and
self-hosting WebR would mean redistributing GPL-3 binaries (see "License" below). The
practical consequence to keep in mind: opening any chapter page reveals the reader's IP
to those three hosts, whether or not they run anything.

What this section *can* still promise is that nothing else is added. Reader-facing
controls are built to keep it that way — see the note above `.lm-slider` in `theme.scss`
for why the parameter sliders are hand-rolled rather than built with Observable's
`Inputs.range`, which would pull two more files from `cdn.jsdelivr.net` on load.

To confirm after a change, grep the markup:

```bash
grep -rhoE 'https?://[a-z0-9.-]+' _site --include='*.html' --include='*.css' \
  | sort -u | grep -vE 'w3\.org|schema\.org|pandoc\.org|quarto-dev|getbootstrap'
```

Only `colab.research.google.com` and `github.com` should appear, and both are
links the reader clicks, not resources the page loads.

That grep is necessary but not sufficient: it reads static markup only, so it cannot see
a URL fetched at runtime by the OJS runtime or from inside a Web Worker — which is how
all three hosts above are reached, and why they never show up in it. To see what a page
actually requests, watch the network with workers attached (DevTools → Network with
"Selected context only" off, or a CDP session using `Target.setAutoAttach`).

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
| MathJax (self-hosted in `mathjax/`) | 3.2.2, Apache-2.0 | mathematics on the pages |
| Fonts (self-hosted in `fonts/`) | OFL-1.1 | Fraunces / Source Serif 4 / Inconsolata |
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
