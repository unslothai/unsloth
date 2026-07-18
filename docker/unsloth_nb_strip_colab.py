#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

# Remove the Colab-only "how to run" sentence from Unsloth notebooks for Docker.
#
# Every generated notebook's first markdown cell opens with a Colab instruction
# ("To run this, press Runtime > Run all on a free Tesla T4 ...", plus A100/L4/AMD
# variants). Inside Docker there is no such menu or Colab GPU, so it is wrong;
# strip ONLY that leading sentence and keep the rest of the cell (badge row,
# local-install link, "You will learn ..." line). Docker-only, applied at sync
# time; NOT pushed upstream (on Colab the sentence is correct).
#
# Two modes:
#   unsloth_nb_strip_colab.py <a.ipynb> [b.ipynb ...]   strip in place (idempotent)
#   unsloth_nb_strip_colab.py --state <STATE> --dest <DEST>
#       STATE-aware sync migration: for each .ipynb in the STATE file that still
#       hashes to its recorded value (owned + unedited), strip the intro and update
#       the hash; user-edited notebooks are left untouched. Runs after every STATE
#       write (populate, restore, refresh, in-place upgrade).
#
# Safe with refresh decisions: content_sig already classifies the intro cell as
# boilerplate, so the body digest is identical with or without the sentence.
# Exit code is always 0.
import argparse
import hashlib
import json
import os
import sys

# The stable identifier for the offending line (covers every GPU/Cloud variant).
_INTRO_PREFIX = "to run this, press"

# The baked notebooks ship example tqdm widget outputs + a metadata.widgets state
# block; JupyterLab can't always rebuild the Colab-saved state, so they render as
# a stuck "Loading widget..." placeholder. Dropping the widget outputs + orphan
# state removes it (running the cell recreates a fresh widget). Outputs aren't in
# the refresh signature (content_sig hashes cell type+source), so this is safe.
_WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"


def _strip_lines(lines):
    """Drop the intro line (and an immediately-following blank). Return new list
    or None if there was nothing to strip."""
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith(_INTRO_PREFIX):
            out = lines[:i] + lines[i + 1 :]
            if i < len(out) and out[i].strip() == "":
                out = out[:i] + out[i + 1 :]
            return out
    return None


def _strip_intro(nb):
    """Strip the Colab intro sentence from cells[0]. Return True if changed."""
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        return False
    cell = cells[0]
    if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
        return False
    src = cell.get("source")
    if isinstance(src, str):
        lines = src.splitlines(keepends = True)
        as_str = True
    elif isinstance(src, list):
        lines = list(src)
        as_str = False
    else:
        return False
    new_lines = _strip_lines(lines)
    if new_lines is None:
        return False
    cell["source"] = "".join(new_lines) if as_str else new_lines
    return True


def _clean_widgets(nb):
    """Drop baked ipywidget outputs + the orphan widget-state metadata that
    otherwise render as "Loading widget...". Return True if changed."""
    changed = False
    cells = nb.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            outs = cell.get("outputs")
            if not isinstance(outs, list):
                continue
            kept = [
                o
                for o in outs
                if not (isinstance(o, dict) and _WIDGET_VIEW_MIME in (o.get("data") or {}))
            ]
            if len(kept) != len(outs):
                cell["outputs"] = kept
                changed = True
    md = nb.get("metadata")
    if isinstance(md, dict) and "widgets" in md:
        del md["widgets"]
        changed = True
    return changed


def strip_notebook(path):
    """Return True if the notebook was modified and written back."""
    try:
        with open(path, "r", encoding = "utf-8") as f:
            nb = json.load(f)
    except Exception:
        return False

    # Apply both transforms; write back if either changed.
    changed = _strip_intro(nb)
    changed = _clean_widgets(nb) or changed
    if not changed:
        return False

    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(nb, f, indent = 1, ensure_ascii = False)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate(state_path, dest):
    """Strip owned+unedited notebooks listed in STATE and update their hashes."""
    try:
        with open(state_path, "r", encoding = "utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return 0

    out = []
    changed = 0
    for line in lines:
        parts = line.split("  ", 1)  # "<sha256>  <relpath>"
        if len(parts) != 2:
            out.append(line)
            continue
        rec, rel = parts
        path = os.path.join(dest, rel)
        if rel.endswith(".ipynb") and os.path.isfile(path):
            try:
                if _sha256(path) == rec:  # we own it and it is unedited
                    if strip_notebook(path):
                        rec = _sha256(path)
                        changed += 1
            except OSError:
                pass
        out.append("%s  %s" % (rec, rel))

    if changed:
        tmp = state_path + ".tmp"
        try:
            with open(tmp, "w", encoding = "utf-8") as f:
                f.write("\n".join(out) + "\n")
            os.replace(tmp, state_path)
        except OSError:
            pass
        print(f"[unsloth-nb] cleaned {changed} notebook(s) (Colab intro + widget outputs)")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description = "Strip the Colab-only intro sentence.")
    ap.add_argument("--state", help = "sync state file (enables migration mode)")
    ap.add_argument("--dest", help = "notebooks dir (with --state)")
    ap.add_argument("paths", nargs = "*", help = "notebooks to strip in place")
    args = ap.parse_args(argv)

    if args.state:
        if not args.dest:
            ap.error("--state requires --dest")
        return migrate(args.state, args.dest)

    changed = sum(1 for p in args.paths if strip_notebook(p))
    if changed:
        print(f"[unsloth-nb] cleaned {changed} notebook(s) (Colab intro + widget outputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
