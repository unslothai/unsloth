#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

# Build a categorized, Colab-like folder VIEW of the Unsloth notebooks: a disposable
# sibling of DEST holding relative SYMLINKS, so real files never move and the sync
# state machine skips them. Section = nearest preceding markdown header, folders
# numbered `NN ` by first appearance so JupyterLab's sort keeps README order.
#
# Usage:
#   unsloth_nb_view.py <DEST> <VIEW> [--amd]      build the symlink view
#   unsloth_nb_view.py <DEST> --print [--amd]     print "section\tfile" rows
# Exits nonzero on error (caller falls back to the raw tree).
import argparse
import os
import re
import sys
import urllib.parse

_NB_RE = re.compile(r"nb/([\w.()%\-]+?\.ipynb)")
_OTHER = "Other Notebooks"


def _load_makedirs_as_host():
    """unsloth_run's directory maker, or None.

    mkdir(2) gives a new directory the CALLER's uid and only setgid carries down, so
    building the view as root under a host bind mount leaves the view root and every
    category folder root:root: the host user cannot add their own link to a category,
    or delete the generated view, without escalating. unsloth_sync_notebooks.sh grew
    mkdir_keep_owner and unsloth_run has _makedirs_as_host for exactly this; the view
    was the sibling both of them missed.

    Imported rather than copied a third time. __file__ is the /usr/local/bin symlink
    in the image, so resolve it before deriving the directory, and
    test_the_view_uses_the_same_directory_maker_as_unsloth_run fails if this stops
    resolving rather than letting it degrade quietly back to the bug.
    """
    here = os.path.dirname(os.path.realpath(__file__))
    added = here not in sys.path
    if added:
        sys.path.insert(0, here)
    try:
        from unsloth_run import _makedirs_as_host

        return _makedirs_as_host
    except Exception:
        return None
    finally:
        if added:
            try:
                sys.path.remove(here)
            except ValueError:
                pass


_MAKEDIRS_AS_HOST = _load_makedirs_as_host()


def _makedirs(path):
    if _MAKEDIRS_AS_HOST is not None:
        _MAKEDIRS_AS_HOST(path)
    else:
        os.makedirs(path, exist_ok = True)


def clean_section(title):
    title = title.strip().strip("#").strip()
    title = re.sub(r"^[^\w]+", "", title)
    title = title.replace("-", " ").replace("/", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def parse_readme(readme_path):
    """Ordered (section_label, filename) pairs, filename urldecoded. Dedup is per
    (section, file), not global: the README deliberately cross-lists a notebook so
    every header becomes a populated folder."""
    with open(readme_path, "r", encoding = "utf-8") as f:
        text = f.read()

    rows = []
    seen_pairs = set()
    section = None
    # any heading level: `#`/`##` domain headers carry their own nb/*.ipynb tables
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
    order = []
    for section, _ in rows:
        if section not in order:
            order.append(section)
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

    if os.path.islink(view):
        resolved = os.path.realpath(view)
        if not os.path.isdir(resolved):
            raise SystemExit(f"view symlink has no directory target: {view} -> {resolved}")
        view = resolved

    rows = parse_readme(readme) if os.path.isfile(readme) else []

    def allowed(fname):
        return amd or not fname.startswith("AMD-")

    by_section = {}
    placed = set()
    for section, fname in rows:
        if not allowed(fname):
            continue
        if not os.path.isfile(os.path.join(nb_dir, fname)):
            continue
        by_section.setdefault(section, []).append(fname)
        placed.add(fname)

    for fname in sorted(os.listdir(nb_dir)):
        if not fname.endswith(".ipynb"):
            continue
        if fname in placed or not allowed(fname):
            continue
        by_section.setdefault(_OTHER, []).append(fname)

    order = [s for s in _ordered_sections(rows) if s in by_section]
    if _OTHER in by_section and _OTHER not in order:
        order.append(_OTHER)

    # keyed on DEST/nb, the only place our links point, so a user's own shortcut
    # elsewhere survives the rebuild
    nb_real = os.path.realpath(nb_dir)
    _clear_view(view, nb_real)
    _makedirs(view)

    n_links = 0
    for i, section in enumerate(order, start = 1):
        folder = os.path.join(view, f"{i:02d} {section}")
        _makedirs(folder)
        for fname in by_section[section]:
            link = os.path.join(folder, fname)
            target = os.path.join(nb_dir, fname)
            rel = os.path.relpath(target, folder)
            try:
                if os.path.islink(link) and _points_into(link, nb_real):
                    os.remove(link)
                elif os.path.islink(link) or os.path.exists(link):
                    print(f"[unsloth-nb] view: keep user file, skip link {fname}", file = sys.stderr)
                    continue
                os.symlink(rel, link)
                n_links += 1
            except OSError as e:
                print(f"[unsloth-nb] view: skip {fname}: {e}", file = sys.stderr)
    return len(order), n_links


def _points_into(link, nb_real):
    """The ownership test for cleanup: our links all point into DEST/nb, so a user's
    own symlink resolves outside it and survives. realpath resolves a broken link's
    path string too, so stale links still count as ours."""
    try:
        target = os.path.realpath(link)
    except OSError:
        return False
    return target == nb_real or target.startswith(nb_real + os.sep)


def _clear_view(path, nb_real):
    # VIEW is also JupyterLab's landing dir, so unlink only symlinks we own and rmdir
    # only emptied folders; the VIEW root is never removed
    if os.path.islink(path) or not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path, topdown = False):
        for name in files:
            p = os.path.join(root, name)
            if os.path.islink(p) and _points_into(p, nb_real):
                try:
                    os.remove(p)
                except OSError:
                    pass
        for name in dirs:
            p = os.path.join(root, name)
            try:
                if os.path.islink(p):
                    if _points_into(p, nb_real):
                        os.remove(p)  # unlink, never recurse
                else:
                    os.rmdir(p)  # succeeds only if we emptied it
            except OSError:
                pass


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
