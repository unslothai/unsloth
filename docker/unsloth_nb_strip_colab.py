#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

# Remove the Colab-only "To run this, press Runtime > Run all ..." sentence, which is
# wrong inside Docker, keeping the rest of the header cell. Two modes:
#
#   unsloth_nb_strip_colab.py <a.ipynb> ...              strip in place (idempotent)
#   unsloth_nb_strip_colab.py --state <STATE> --dest <DEST>
#       strip + rehash each notebook whose hash still matches STATE
#
# Safe with refresh: content_sig classifies the intro cell as boilerplate, so the
# body digest is unchanged. Exit code is always 0.
import argparse
import hashlib
import json
import os
import stat
import sys

_INTRO_PREFIX = "to run this, press"

# Baked tqdm widget outputs render as a stuck "Loading widget...", and outputs are not
# in the refresh signature, so dropping them is safe.
_WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"


def _is_intro_line(line):
    """Matches the Colab intro, bare or in one-line HTML comments; never an unclosed opener."""
    stripped = line.strip()
    low = stripped.lower()
    if low.startswith(_INTRO_PREFIX):
        return True
    if low.startswith("<!--") and stripped.endswith("-->"):
        return stripped[4:-3].strip().lower().startswith(_INTRO_PREFIX)
    return False


def _stage_metadata(staged, dest):
    """os.replace swaps the directory entry, so the staged inode's root owner would
    become the published file's, undoing the sync script's care to leave a
    bind-mounted notebook host-owned. Best effort."""
    try:
        st = os.stat(dest)
    except OSError:
        return
    try:
        os.chmod(staged, stat.S_IMODE(st.st_mode))
    except OSError:
        pass
    try:
        os.chown(staged, st.st_uid, st.st_gid)
    except (OSError, AttributeError):
        pass


def _strip_lines(lines):
    for i, line in enumerate(lines):
        if _is_intro_line(line):
            out = lines[:i] + lines[i + 1 :]
            if i < len(out) and out[i].strip() == "":
                out = out[:i] + out[i + 1 :]
            return out
    return None


def _strip_cell(cell):
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


def _strip_intro(nb):
    """Strips the Colab intro from leading markdown cells (badge and sentence may be split)."""
    cells = nb.get("cells")
    if not isinstance(cells, list):
        return False
    changed = False
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            break  # the first code cell ends the header block
        if _strip_cell(cell):
            changed = True
    return changed


def _clean_widgets(nb):
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


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _stage_clean(path):
    """Returns (tmp, hash_before, hash_after) for a cleaned copy staged, not published, beside the
    notebook, or None if unchanged or failed; split so migrate can record the hash first."""
    try:
        before = _sha256(path)
        with open(path, "r", encoding = "utf-8") as f:
            nb = json.load(f)
    except Exception:
        return None

    changed = _strip_intro(nb)
    changed = _clean_widgets(nb) or changed
    if not changed:
        return None

    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(nb, f, indent = 1, ensure_ascii = False)
            f.write("\n")
        return tmp, before, _sha256(tmp)
    except Exception:
        _unlink(tmp)
        return None


def _publish(tmp, path, before):
    """Move the staged copy onto the notebook. False if it was not published."""
    try:
        # JupyterLab is serving the tree, so recheck before replacing.
        if _sha256(path) != before:
            _unlink(tmp)
            return False
        _stage_metadata(tmp, path)
        os.replace(tmp, path)
    except Exception:
        _unlink(tmp)
        return False
    return True


def _write_state(state_path, lines):
    """Returns False on failure; fsync before rename so no crash exposes unwritten content."""
    tmp = state_path + ".tmp"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_path)
    except OSError:
        _unlink(tmp)
        return False
    return True


def _resume(path, rec):
    """Finishes a publish interrupted between the state and notebook renames, True if it did; else
    that stop reads as a user edit, and only this case reproduces the record byte for byte."""
    st = _stage_clean(path)
    if st is None:
        return False
    tmp, before, after = st
    if after != rec:
        _unlink(tmp)
        return False
    return _publish(tmp, path, before)


def strip_notebook(path, staged = None):
    """True if modified and written back; staged, when given, gets sha256 of the bytes written."""
    st = _stage_clean(path)
    if st is None:
        return False
    tmp, before, after = st
    if not _publish(tmp, path, before):
        return False
    if staged is not None:
        staged["sha256"] = after
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate(state_path, dest):
    """Cleans every owned notebook, recording each in the state file before it is published, so one
    lost state write cannot strand the set; a stop between the renames is reconciled by _resume."""
    try:
        with open(state_path, "r", encoding = "utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return 0

    out = list(lines)  # malformed lines and untouched records survive verbatim
    changed = 0
    stopped = False
    for i, line in enumerate(lines):
        parts = line.split("  ", 1)  # "<sha256>  <relpath>"
        if len(parts) != 2:
            continue
        rec, rel = parts
        path = os.path.join(dest, rel)
        if not (rel.endswith(".ipynb") and os.path.isfile(path)):
            continue
        try:
            digest = _sha256(path)
        except OSError:
            continue
        if digest != rec:  # not ours, the user has edited it, or a publish was cut off
            if _resume(path, rec):
                changed += 1
            continue
        staged = _stage_clean(path)
        if staged is None:
            continue
        tmp, before, after = staged
        out[i] = "%s  %s" % (after, rel)
        if not _write_state(state_path, out):
            _unlink(tmp)
            out[i] = line
            stopped = True
            break
        if _publish(tmp, path, before):
            changed += 1
        else:
            out[i] = line
            _write_state(state_path, out)

    if changed:
        print(f"[unsloth-nb] cleaned {changed} notebook(s) (Colab intro + widget outputs)")
    if stopped:
        print(
            "[unsloth-nb] could not record the cleaned notebooks; leaving the rest for "
            "the next start"
        )
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
