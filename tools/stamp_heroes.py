"""Version every figure link in the pages by its file's contents.

A regenerated figure keeps its file name, so a browser that has cached the old
image shows it under the new page. Each <img src="../figures/NAME.png"> gets
?v=<first 8 hex digits of the file's SHA-1>; the link changes exactly when the
image does. make_hero.py runs this after writing; run it by hand after any other
figure generator.

    python3 tools/stamp_heroes.py
"""
import hashlib, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK = re.compile(r'src="\.\./figures/([A-Za-z0-9_.-]+\.(?:png|svg|jpg))(?:\?v=[0-9a-f]+)?"')


def stamp(text):
    def repl(m):
        f = ROOT / "figures" / m.group(1)
        if not f.exists():
            return m.group(0)
        v = hashlib.sha1(f.read_bytes()).hexdigest()[:8]
        return f'src="../figures/{m.group(1)}?v={v}"'
    return LINK.sub(repl, text)


def main():
    changed = []
    for qmd in sorted((ROOT / "chapters").glob("*.qmd")):
        old = qmd.read_text()
        new = stamp(old)
        if new != old:
            qmd.write_text(new)
            changed.append(qmd.name)
    print(f"{len(changed)} page(s) restamped" + (": " + ", ".join(changed) if changed else ""))


if __name__ == "__main__":
    main()
