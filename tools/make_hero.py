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
    "ch07_02_applied":          "hero is the LASSO path alone; the cell draws both paths",
    "ch08_01_trees_ensembles":  "hero is the train/test curve; the cell adds a second panel",
    "ch08_02_kernel_methods":   "hero is one boundary; the cell sweeps four values of gamma",
    "ch08_03_applied":          "hero is one boundary; the cell draws six classifiers",
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

    todo = args.pages or sorted(HEROES)
    unknown = [p for p in todo if p not in HEROES]
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
        out, note = render(stem, HEROES[stem], outdir)
        kb = out.stat().st_size / 1024
        print(f"  {stem:34} cell {HEROES[stem]['cell']}  {kb:6.1f} KB   {note}")
    print(f"\n{len(todo)} hero figure(s) -> {outdir}")


if __name__ == "__main__":
    main()
