#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

# Build a categorized, Colab-like folder VIEW of the Unsloth notebooks.
#
# The canonical notebooks live flat under DEST/nb/<file>.ipynb (mirror of
# unslothai/notebooks, kept by unsloth_sync_notebooks.sh). This builds a sibling
# dir of *relative symlinks* grouped into folders mirroring the README headers:
#   <VIEW>/01 Main Notebooks/Llama3_2_(1B_and_3B)_Conversational.ipynb
#   <VIEW>/99 Other Notebooks/<anything on disk not linked from the README>
# Symlinks so the real files never move (the sync state machine skips symlinks);
# the VIEW is a disposable sibling of DEST, rebuilt from scratch on every boot.
#
# Categorization rules:
#   * Section = nearest preceding `###` header in README.md; a header repeated
#     across Fine-tuning/Kaggle/AMD domains merges into one folder (first order).
#   * Folder names cleaned (dashes/slashes -> spaces) and numbered `NN ` by first
#     appearance so JupyterLab's alpha sort keeps README order; "Other" is last.
#   * A notebook linked under several sections lands in its first.
#   * AMD-*.ipynb hidden unless --amd; unlinked nb/*.ipynb go to "Other Notebooks".
#
# Usage:
#   unsloth_nb_view.py <DEST> <VIEW> [--amd]      build the symlink view
#   unsloth_nb_view.py <DEST> --print [--amd]     print "section\tfile" rows
# Exits 0 on success; on error prints to stderr and exits nonzero so the caller
# can fall back to the raw tree.
import argparse
import os
import re
import sys
import urllib.parse

# nb/<file>.ipynb in any link form (markdown badge, HTML href, plain link,
# Kaggle ?src= form). Filenames use [\w.()-] plus %-escapes (%28/%29 for parens).
_NB_RE = re.compile(r"nb/([\w.()%\-]+?\.ipynb)")
_OTHER = "Other Notebooks"


def clean_section(title):
    """README header text -> a filesystem-friendly folder label."""
    title = title.strip().strip("#").strip()
    # Strip a leading run of emoji/symbols some domain headers lead with so the
    # folder label is clean text.
    title = re.sub(r"^[^\w]+", "", title)
    title = title.replace("-", " ").replace("/", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def parse_readme(readme_path):
    """Return an ordered list of (section_label, filename) pairs.

    A notebook is intentionally cross-listed under several `###` headers in the
    README (e.g. ModernBert under both "Embedding" and "BERT"), so that every
    header becomes a populated folder. We therefore dedup per (section, file) --
    a file shows up once in EACH section that lists it -- rather than globally.
    Repeated headers across the Fine-tuning / Kaggle / AMD domains share a label
    and so merge into one folder downstream.

    filename is the urldecoded basename under nb/ (literal parens, matching disk).
    """
    with open(readme_path, "r", encoding = "utf-8") as f:
        text = f.read()

    rows = []
    seen_pairs = set()  # (section, filename) already emitted
    section = None
    # Reset on ANY markdown heading, not just `###`: `#`/`##` domain headers carry
    # their own nb/*.ipynb tables with no intervening `###`, so matching only `###`
    # left `section` stale and mis-filed those links under the previous section.
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            section = clean_section(m.group(1))
            continue
        if section is None:
            continue
        for raw in _NB_RE.findall(line):
            fname = urllib.parse.unquote(raw)
            key = (section, fname)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rows.append((section, fname))
    return rows


def _ordered_sections(rows):
    """Section labels in first-appearance order, with Other Notebooks last."""
    order = []
    for section, _ in rows:
        if section not in order:
            order.append(section)
    # Force the catch-all to the end even if the README defines it earlier.
    order = [s for s in order if s != _OTHER] + [_OTHER]
    return order


def build_view(
    dest,
    view,
    amd = False,
):
    nb_dir = os.path.join(dest, "nb")
    readme = os.path.join(dest, "README.md")
    if not os.path.isdir(nb_dir):
        raise SystemExit(f"no nb/ dir under {dest}")

    # An operator may route the VIEW through a symlink to persistent/mounted
    # storage. Build inside its target instead of unlinking the routing.
    if os.path.islink(view):
        resolved = os.path.realpath(view)
        if not os.path.isdir(resolved):
            raise SystemExit(f"view symlink has no directory target: {view} -> {resolved}")
        view = resolved

    rows = parse_readme(readme) if os.path.isfile(readme) else []

    def allowed(fname):
        return amd or not fname.startswith("AMD-")

    # section -> [filenames], preserving README order, AMD-filtered, on-disk only.
    by_section = {}
    placed = set()
    for section, fname in rows:
        if not allowed(fname):
            continue
        if not os.path.isfile(os.path.join(nb_dir, fname)):
            continue
        by_section.setdefault(section, []).append(fname)
        placed.add(fname)

    # Everything on disk that the README never linked -> Other Notebooks.
    for fname in sorted(os.listdir(nb_dir)):
        if not fname.endswith(".ipynb"):
            continue
        if fname in placed or not allowed(fname):
            continue
        by_section.setdefault(_OTHER, []).append(fname)

    order = [s for s in _ordered_sections(rows) if s in by_section]
    if _OTHER in by_section and _OTHER not in order:
        order.append(_OTHER)

    # Rebuild VIEW: drop the symlinks/empty folders we made last boot, but never
    # the user's own files (VIEW is also JupyterLab's landing dir, so a user may
    # have saved real notebooks here).
    _clear_view(view, os.path.realpath(dest))
    os.makedirs(view, exist_ok = True)

    n_links = 0
    for i, section in enumerate(order, start = 1):
        folder = os.path.join(view, f"{i:02d} {section}")
        os.makedirs(folder, exist_ok = True)
        for fname in by_section[section]:
            link = os.path.join(folder, fname)
            target = os.path.join(nb_dir, fname)
            rel = os.path.relpath(target, folder)  # ../../unsloth-notebooks/nb/<file>
            try:
                if os.path.islink(link) and _points_into(link, os.path.realpath(dest)):
                    os.remove(link)  # replace our own stale symlink
                elif os.path.islink(link) or os.path.exists(link):
                    # a real user file/dir already occupies this name -- never
                    # clobber it; leave it and skip linking this notebook.
                    print(f"[unsloth-nb] view: keep user file, skip link {fname}", file = sys.stderr)
                    continue
                os.symlink(rel, link)
                n_links += 1
            except OSError as e:
                print(f"[unsloth-nb] view: skip {fname}: {e}", file = sys.stderr)
    return len(order), n_links


def _points_into(link, dest_real):
    """True when a symlink resolves into the notebooks tree we link from.

    Every link this tool creates points at DEST/nb/<file>, so this is the
    ownership test for cleanup: a user's own symlink (to a dataset, project,
    mounted dir, ...) resolves elsewhere and must survive a rebuild. realpath
    resolves a broken link's path string too, so stale links to since-removed
    notebooks are still recognised as ours.
    """
    try:
        target = os.path.realpath(link)
    except OSError:
        return False
    return target == dest_real or target.startswith(dest_real + os.sep)


def _clear_view(path, dest_real):
    # Tear down a previously built VIEW in place. It is also JupyterLab's landing
    # dir, so user files/symlinks MUST survive: unlink only the symlinks we own
    # (resolve into DEST, see _points_into) and rmdir only emptied folders. The
    # VIEW root is never unlinked (build_view already resolved a symlinked root).
    if os.path.islink(path) or not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path, topdown = False):
        for name in files:
            p = os.path.join(root, name)
            if os.path.islink(p) and _points_into(p, dest_real):  # our notebook symlinks only
                try:
                    os.remove(p)
                except OSError:
                    pass
            # a regular file / user symlink here is user-created -> keep it
        for name in dirs:
            p = os.path.join(root, name)
            try:
                if os.path.islink(p):
                    if _points_into(p, dest_real):
                        os.remove(p)  # our symlinked dir: unlink, never recurse
                else:
                    os.rmdir(p)  # succeeds only if we emptied it
            except OSError:
                pass  # holds user files -> keep


def main(argv):
    ap = argparse.ArgumentParser(description = "Build the categorized notebook view.")
    ap.add_argument("dest", help = "notebooks dir (contains README.md and nb/)")
    ap.add_argument("view", nargs = "?", help = "output view dir (omit with --print)")
    ap.add_argument("--amd", action = "store_true", help = "include AMD-* notebooks")
    ap.add_argument(
        "--print",
        dest = "do_print",
        action = "store_true",
        help = "print section<TAB>file rows instead of building",
    )
    args = ap.parse_args(argv)

    if args.do_print:
        for section, fname in parse_readme(os.path.join(args.dest, "README.md")):
            if args.amd or not fname.startswith("AMD-"):
                print(f"{section}\t{fname}")
        return 0

    if not args.view:
        ap.error("view dir is required unless --print is given")
    n_sections, n_links = build_view(args.dest, args.view, amd = args.amd)
    print(f"[unsloth-nb] view: {n_links} notebooks in {n_sections} folders -> {args.view}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
