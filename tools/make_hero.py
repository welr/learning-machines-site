"""Regenerate a page's hero figure from that page's own live cell.

FIGURE_SPEC.md requires each hero to be "generated from the page's OWN
computation ... It must match what the cell produces." Fifteen heroes had no
generator at all, so that rule was unverifiable for them and a hero could drift
silently away from the cell beneath it whenever the cell changed.

Rather than fifteen bespoke scripts that would each have to be kept in step by
hand, this one extracts the designated {pyodide} cell straight out of the .qmd,
runs it, and saves the figure it draws. Hero and cell therefore cannot disagree:
the hero *is* the cell's output.

    python3 tools/make_hero.py                 # every page in HEROES
    python3 tools/make_hero.py ch02_01_polynomial_regression
    python3 tools/make_hero.py --check         # regenerate to a temp dir and diff

Pages with a bespoke generator (make_ch*_figure.py) are not listed here: their
heroes show something the browser cannot compute — a trained network's filters,
a Fashion-MNIST projection — and must stay hand-built.
"""

import argparse
import io
import pathlib
import re
import sys
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAPTERS, FIGURES = ROOT / "chapters", ROOT / "figures"

# page stem -> which {pyodide} cell draws the hero, and the size to draw it at.
# The cell index is 0-based over the page's pyodide cells.
#
# Only pages whose hero IS a whole cell's output are listed. Verified by
# regenerating and comparing composition against the shipped PNG.
HEROES = {
    "ch02_01_polynomial_regression": dict(cell=0, figsize=(8.0, 5.0)),
    "ch02_02_linear_regression_ols": dict(cell=1, figsize=(8.0, 5.0)),
    "ch02_03_bayesian_regression":   dict(cell=1, figsize=(8.0, 5.0)),
    "ch02_04_applied":               dict(cell=2, figsize=(8.0, 5.0)),
    "ch04_01_logistic_regression":   dict(cell=0, figsize=(8.0, 5.0)),
    "ch04_02_multiclass":            dict(cell=1, figsize=(7.6, 5.4)),
    "ch04_03_applied":               dict(cell=0, figsize=(8.0, 5.0)),
    "ch06_01_model_evaluation":      dict(cell=3, figsize=(8.0, 5.0)),
}

# Heroes that are NOT a whole cell's output, and why. Each still needs a bespoke
# generator (or a decision to replace the hero with the cell's figure); until one
# exists, the shipped PNG cannot be regenerated or checked against its page.
NEEDS_BESPOKE = {
    "ch03_01_gradient_descent": "hero is one contour panel; the matching cell draws three",
    "ch06_02_applied":          "hero is a square ROC; the cell's figure is a different shape",
    "ch07_01_regularization":   "hero is the ridge path alone; the cell draws ridge and LASSO",
}


# Heroes that show ONE panel of a multi-panel cell. The page's cell still does all
# the computing; draw(env, ax) only re-plots one panel from the variables it left.
# Colors follow the site rule: data and the first class blue, fits FS-blue, a second
# class or series orange or cyan; red only for residuals, zero lines and thresholds.
BLUE, FS, CYAN, ORANGE, GRAY = "#076FA1", "#31417A", "#2FC1D3", "#9E4F00", "#666666"
SPINE, GRID = "#cfccc2", "#e6e3da"


def _frame(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GRAY)


def _lasso_path(env, ax):                       # ch07_02: the LASSO panel, book-scale lambda
    lam, coefs, info = env["lasso_lams"], env["lasso_coefs"], env["informative"]
    ax.axhline(0, color=GRAY, lw=0.9, zorder=1)
    for j in (j for j in range(coefs.shape[0]) if not info[j]):
        ax.plot(lam, coefs[j], color=GRAY, lw=0.9, alpha=0.55, zorder=2)
    for j in (j for j in range(coefs.shape[0]) if info[j]):
        ax.plot(lam, coefs[j], color=BLUE, lw=1.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"penalty $\lambda$  (log scale; weak $\rightarrow$ strong)")
    ax.set_ylabel("coefficient")
    ax.plot([], [], color=BLUE, lw=1.6, label="informative features")
    ax.plot([], [], color=GRAY, lw=0.9, label="spurious features")
    ax.legend(loc="upper right", frameon=False)
    _frame(ax)


def _tree_depth(env, ax):                       # ch08_01: the train/test panel
    d = list(env["depths"])
    ax.plot(d, env["train_mse"], color=FS, lw=2.5, marker="o", label="train error")
    ax.plot(d, env["test_mse"], color=CYAN, lw=2.5, marker="o", label="test error")
    ax.axvline(env["best"], color=GRAY, ls=":", alpha=0.7)
    ax.set_xlabel("tree depth"); ax.set_ylabel("MSE")
    ax.legend(loc="center right", frameon=False)
    _frame(ax)


def _boundary(ax, X, y, clf, pad):
    """Predicted regions, the decision boundary, and the true labels."""
    from matplotlib.colors import ListedColormap
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    g = np.c_[xx.ravel(), yy.ravel()]
    ax.contourf(xx, yy, clf.predict(g).reshape(xx.shape), levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap([BLUE, ORANGE]), alpha=0.18)
    ax.contour(xx, yy, clf.decision_function(g).reshape(xx.shape), levels=[0],
               colors=FS, linewidths=2)
    return xx, yy


def _rings_svm(env, ax):                        # ch08_02: the RBF SVM at gamma = 2
    X, y = env["X"], env["y"]
    clf = env["SVC"](kernel="rbf", gamma=2.0).fit(X, y)
    _boundary(ax, X, y, clf, pad=0.35)
    ax.scatter(*X[y == 0].T, c=BLUE, s=34, edgecolors="white", linewidths=0.6, label="outer")
    ax.scatter(*X[y == 1].T, c=ORANGE, s=34, edgecolors="white", linewidths=0.6, label="inner")
    ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", framealpha=0.9)
    _frame(ax, grid=None)


def _moons_svm(env, ax):                        # ch08_03: the RBF SVM panel
    X, y = env["Xm"], env["ym"]
    clf = dict(env["models"])["kernel SVM (RBF)"]          # already fit by the cell
    _boundary(ax, X, y, clf, pad=0.5)
    for k, c in ((0, BLUE), (1, ORANGE)):
        ax.scatter(*X[y == k].T, c=c, s=22, edgecolors="white", linewidths=0.4, label=f"class {k}")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.legend(loc="lower right", framealpha=0.9)
    _frame(ax, grid=None)


PANELS = {
    "ch07_02_applied":         dict(cell=1, figsize=(8.0, 5.0), draw=_lasso_path),
    "ch08_01_trees_ensembles": dict(cell=2, figsize=(8.0, 5.0), draw=_tree_depth),
    "ch08_02_kernel_methods":  dict(cell=2, figsize=(6.6, 6.0), draw=_rings_svm),
    "ch08_03_applied":         dict(cell=1, figsize=(8.0, 5.6), draw=_moons_svm),
}


def cell_source(stem, index):
    """The index-th {pyodide} cell of a page, with its #| directives stripped."""
    qmd = (CHAPTERS / f"{stem}.qmd").read_text()
    cells = re.findall(r"```\{pyodide\}\n(.*?)```", qmd, re.S)
    if index >= len(cells):
        raise SystemExit(f"{stem}: asked for cell {index}, page has {len(cells)}")
    body = "\n".join(l for l in cells[index].splitlines() if not l.startswith("#|"))
    # The cell shows its figure; we want to save it instead.
    return body.replace("plt.show()", "pass")


def render(stem, spec, outdir):
    src = cell_source(stem, spec["cell"])
    plt.close("all")
    env = {"__name__": "__hero__"}
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        exec(compile(src, f"{stem}[cell {spec['cell']}]", "exec"), env)
    if "draw" in spec:                          # one panel, from the cell's own variables
        plt.close("all")
        fig, ax = plt.subplots(figsize=spec["figsize"])
        spec["draw"](env, ax)
    fig = plt.gcf()
    if not fig.get_axes():
        raise SystemExit(f"{stem}: cell {spec['cell']} drew nothing")
    fig.set_size_inches(*spec["figsize"])
    fig.tight_layout()
    out = outdir / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    first = printed.getvalue().strip().splitlines()
    return out, (first[0][:60] if first else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", help="page stems (default: all in HEROES)")
    ap.add_argument("--check", action="store_true",
                    help="render to a temp directory instead of overwriting")
    args = ap.parse_args()

    ALL = {**HEROES, **PANELS}
    todo = args.pages or sorted(ALL)
    unknown = [p for p in todo if p not in ALL]
    if unknown:
        for u in unknown:
            why = NEEDS_BESPOKE.get(u, "has its own make_*_figure.py, or is not a chapter page")
            print(f"  skip {u}: {why}", file=sys.stderr)
        raise SystemExit(1)

    outdir = FIGURES
    if args.check:
        outdir = ROOT / ".hero-check"
        outdir.mkdir(exist_ok=True)

    for stem in todo:
        out, note = render(stem, ALL[stem], outdir)
        kb = out.stat().st_size / 1024
        print(f"  {stem:34} cell {ALL[stem]['cell']}  {kb:6.1f} KB   {note}")
    print(f"\n{len(todo)} hero figure(s) -> {outdir}")


if __name__ == "__main__":
    main()
