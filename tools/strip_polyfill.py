#!/usr/bin/env python3
"""Remove the cdnjs ES6 polyfill that Quarto hardcodes into MathJax pages.

Quarto's HTML template emits

    <script src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?features=es6"></script>

whenever a page uses MathJax (see share/formats/html/pandoc/template.html, gated on
$if(mathjax)$). There is no configuration option to suppress it, hence this
post-render pass.

Why it is safe to drop here: the polyfill backfills ES6 for browsers that predate
it, but this site's whole purpose is running Python and R in the browser through
WebAssembly. WASM shipped across the major browsers in 2017, ES6 in 2015-2016, so
any browser that can use this site already has ES6 natively and any browser that
cannot is not helped by the polyfill. Removing it eliminates a third-party request
on every page that contains mathematics.

Idempotent, and deliberately loud: if it stops matching -- because Quarto changed
the template or dropped the polyfill upstream -- it says so rather than passing
silently, so the check does not rot into a no-op nobody notices.
"""

import pathlib
import re
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent / "_site"
PATTERN = re.compile(
    r'[ \t]*<script src="https://cdnjs\.cloudflare\.com/polyfill/[^"]*"></script>\n?'
)


def main() -> int:
    if not SITE.is_dir():
        print(f"strip_polyfill: no {SITE}, nothing to do")
        return 0

    pages = sorted(SITE.rglob("*.html"))
    stripped = 0
    for page in pages:
        text = page.read_text(encoding="utf-8")
        new, n = PATTERN.subn("", text)
        if n:
            page.write_text(new, encoding="utf-8")
            stripped += 1

    remaining = [p for p in pages if "cdnjs.cloudflare.com" in p.read_text(encoding="utf-8")]
    print(f"strip_polyfill: {stripped}/{len(pages)} pages cleaned")

    if remaining:
        print(
            "strip_polyfill: WARNING -- cdnjs references survive in "
            f"{len(remaining)} page(s); the pattern needs updating:",
            file=sys.stderr,
        )
        for p in remaining[:5]:
            print(f"  {p.relative_to(SITE)}", file=sys.stderr)
        return 1

    if stripped == 0:
        print(
            "strip_polyfill: note -- nothing matched. Either no page uses MathJax, "
            "or Quarto no longer emits the polyfill and this pass can be retired."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
