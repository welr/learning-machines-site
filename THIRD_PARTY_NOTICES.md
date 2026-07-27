# Third-party notices

Components redistributed by this repository or by the site it builds, with the
notices their licenses require. Verified 2026-07-27.

---

## quarto-live

Vendored under `_extensions/r-wasm/live/`. Full license text at
`_extensions/r-wasm/live/LICENSE`.

> The MIT Licence
> Copyright (c) 2024 quarto-live authors

---

## webR (JavaScript API only)

quarto-live bundles webR's JavaScript API into
`site_libs/quarto-contrib/live-runtime/live-runtime.js`, which the published site
serves. **Only the wrapper is redistributed** — the R binaries (`R.wasm` and the R
standard library, which webR licenses under GPL-3.0) are fetched at runtime from
`https://webr.r-wasm.org/` and appear nowhere in this repository.

webR's `LICENCE.md` licenses its non-binary software under MIT, and none of the JS
API sources under `src/webR/` carry a contrary per-file header.

> The MIT Licence
> Copyright (c) 2023 webR authors

---

## Fonts (self-hosted)

The three webfonts are served from `fonts/`, not from the Google Fonts CDN. All are
**SIL OFL-1.1**, and none declares a Reserved Font Name, so unmodified redistribution
is unrestricted. OFL condition 2 requires the copyright notice and licence to travel
with every copy; they are at `fonts/OFL-Fraunces.txt`, `fonts/OFL-SourceSerif4.txt`
and `fonts/OFL-Inconsolata.txt`, and are published alongside the fonts.

> Copyright 2018 The Fraunces Project Authors (https://github.com/undercasetype/Fraunces)
> Copyright 2014 The Source Serif 4 Project Authors (https://github.com/adobe-fonts/source-serif)
> Copyright 2006 The Inconsolata Project Authors

The `.woff2` files are Google Fonts' subset builds of the upstream families, taken
from the same OFL-licensed originals in `google/fonts`. Each is a variable font, so
one file covers a whole weight range.

---

## Quarto front-end libraries

Emitted by Quarto into `_site/site_libs/` at build time:

| Component | License |
|---|---|
| Bootstrap, Bootstrap Icons | MIT, © Twitter/The Bootstrap Authors |
| clipboard.js | MIT, © Zeno Rocha |
| anchor.js | MIT, © Bryan Braun |
| popper / floating-ui | MIT, © Floating UI contributors |
| tippy.js | MIT, © atomiks |
| Algolia autocomplete | MIT, © Algolia |
| Fuse.js | Apache-2.0, © Kiro Risk |
| Observable runtime | ISC, © Observable, Inc. |
| Quarto's own JS/CSS | MIT, © 2020–2024 Posit Software, PBC |

---

## Loaded at runtime, not redistributed

| Component | License | Source |
|---|---|---|
| R / webR distribution binaries | **GPL-3.0** | `webr.r-wasm.org` |
| Pyodide | MPL-2.0 | `cdn.jsdelivr.net/pyodide` |
| MathJax | Apache-2.0 | `cdn.jsdelivr.net` |

These are executed in the reader's browser, served by their upstream publishers.
Nothing under a copyleft license is conveyed by this repository. Self-hosting any
of them would change that analysis and should be reviewed before it is done.

## Build-time only

Pandoc (GPL-2.0) and the `quarto-dev/quarto-actions` GitHub Action (GPL-2.0) run in
CI. Neither contributes code to the published site; pandoc's output is not a
derivative of pandoc.
