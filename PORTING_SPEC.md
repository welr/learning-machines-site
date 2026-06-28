# Notebook → companion-site porting spec

Goal: turn each source notebook in `MLone book/notebooks/` into ONE Quarto page under
`companion-site/chapters/`, faithful to the notebook's content, runnable **live in the
browser in both Python and R**. The site is an expository companion, not a 1:1 dump:
keep the notebook's real ideas, examples, and key figures; condense repetition.

## File + front matter

- Filename mirrors the notebook: `chapters/ch02_02_linear_regression_ols.qmd`.
- Front matter MUST be:
  ```
  ---
  title: "<short human title, e.g. Ordinary Least Squares>"
  engine: markdown
  ---
  ```
  `engine: markdown` is REQUIRED (the `{pyodide}`/`{webr}` cells run in the browser, not
  at build time). Without it the build fails.

## Page skeleton (use these exact custom classes — they are styled in theme.scss)

```
::: {.lm-hero}
[Chapter 2 · Regression Models]{.eyebrow}

# Ordinary Least Squares

[One-line italic dek that captures the notebook's thesis.]{.dek}
:::

<faithful exposition in the book's voice: economy, active voice, concrete before
abstract. Use the notebook's real narrative and examples, condensed. Mark key terms
with [term]{.term}. Use a [definition box]{.defbox} for a central formula:>

::: {.defbox}
[Ordinary Least Squares Estimate]{.lbl}
[ &theta;&#770; = (X&#8868;X)&#8722;&sup1; X&#8868;y ]{.math}
:::

<the live code, as a synced dual-language tabset:>

::: {.panel-tabset group="lang"}

## Python
```{pyodide}
# faithful Python, browser-portable (see rules)
```

## R
```{webr}
# faithful R translation (see rules)
```

:::

<closing prose; then an exploration prompt:>

::: {.explore}
[Try it]{.lbl}
<one concrete thing the reader should modify and observe>
:::
```

A page may have MORE than one tabset (e.g. one per major demo). Every tabset gets
`group="lang"` so language choice syncs across the page and persists across pages.

## Code rules — Python (`{pyodide}`)

- Pyodide has: numpy, pandas, scipy, scikit-learn, matplotlib. Use them freely.
- **Remove the local theme dependency.** Delete `import mlone_theme as mt` and any `mt.*`
  calls. Use plain matplotlib. A couple of `plt.rcParams` tweaks are fine; don't depend on
  `mlone_style.mplstyle`.
- **No network.** If the notebook calls `fetch_openml`, `fetch_california_housing`, or
  downloads a file, substitute a small **inline** dataset (define arrays in the cell) or a
  **bundled** sklearn dataset that needs no network (`load_breast_cancer`, `load_iris`,
  `load_diabetes`, `make_*`). State the substitution in one prose sentence.
- Keep cells **self-contained** (a reader can run each on its own): re-create the data at
  the top of each cell that needs it.
- Reproduce the notebook's KEY figure(s) with matplotlib where the point is visual; print
  numeric results otherwise. Don't port every minor plot — pick the one or two that carry
  the idea.
- **Plot colors: use the site's BLUE identity, NOT the notebooks' green.** Data/primary
  `#076FA1`, dark/fit-line `#31417A`, red accent `#E3120B`, neutral gray `#666666`. Do not
  use `#4F9E00` or `#2D5A00` (the old notebook green — the site dropped it). This applies to
  both matplotlib and base-R plots.

## Code rules — R (`{webr}`)

- WebR cells must run in **base R + the `stats` graphics/stats** that ship with R. The
  translation must be **verifiable with `Rscript --vanilla`** (no contributed packages),
  because (a) you can verify it locally and (b) it keeps the WebR download light.
- Translate the **computation faithfully**, not line-by-line. Map:
  - OLS / ridge → `lm`, or matrix algebra (`solve`, `qr.solve`).
  - logistic / GLM → `glm(..., family = binomial)`.
  - polynomials → `outer(x, 0:d, ` + "`^`" + `)` design matrix + `qr.solve`.
  - kernel smoothing → `ksmooth` or manual Gaussian weights.
  - k-fold / bias-variance / gradient descent → base-R loops.
  - simulation/Bayes-conjugate → base-R linear algebra + `rnorm`/`dnorm`.
- For methods base R can't do (trees, SVM, random forest, glmnet-LASSO): implement a
  **small base-R version of the core idea** (a single decision stump; ridge via linear
  algebra; a 1-D soft-threshold for LASSO; a tiny kernel computation) faithful to the
  pedagogical point, and say in one sentence that the production tool is the package
  (`rpart`/`e1071`/`glmnet`). Do NOT emit `{webr}` code that needs an uninstalled package.
- Plots: base-R `plot()` is fine in WebR.

## Verification (REQUIRED before returning)

Extract each `{pyodide}` cell and run it with `/Users/gregorywheeler/otter_env/bin/python`;
extract each `{webr}` cell and run it with `Rscript --vanilla`. Both must exit 0. Where the
two languages compute the same quantity, their printed numbers should agree (note any
divergence and why — e.g. different RNG). Report the actual outputs as evidence.

## Voice

Match the book: economy, active voice, intellectual honesty. American spelling. At most one
em-dash pair per paragraph. No "let's dive in" filler. The reader should get 2–3× more from
the page than from reading it cold, but it should stand alone.
```
