#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import hashlib
import json
import re
import sys

# Lowercased substrings that mark a markdown cell as top/bottom boilerplate.
_BOILERPLATE_MD = (
    "to run this, press",  # Colab/AMD run announcement
    'press "*runtime*"',
    "### news",  # News heading
    "introducing **unsloth studio**",  # rotating announcement body
    "you will learn how to do",  # announcement tail
    "this notebook is licensed",  # announcement license line
    "and we're done",  # footer opener
    "this notebook and all unsloth notebooks are licensed",  # footer license
    "join discord if you need help",  # footer
    "star us on",  # footer
    "some other resources",  # footer resources block
)


def _text(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.replace("\r\n", "\n").replace("\r", "\n")


# Command fragments that mark a cell as the generated install cell.
_INSTALL_MARKERS = (
    "pip install",
    "pip3-autoremove",
    "uv pip install",
    "conda install",
    "apt-get install",
    "apt install",
)


def _is_install_code(cell):
    if cell.get("cell_type") != "code":
        return False
    t = _text(cell)
    low = t.lower()
    if any(m in low for m in _INSTALL_MARKERS):
        return True
    # A %%capture / %%bash cell is boilerplate only if it also carries an install
    # command (caught above); a bare one doing real setup is substantive, so hash
    # it to avoid a false SAME on the boot refresh.
    return False


def _is_boilerplate_md(cell):
    if cell.get("cell_type") != "markdown":
        return False
    low = _text(cell).lower()
    return any(m in low for m in _BOILERPLATE_MD)


# A `#` that opens a comment: at line start, or preceded by whitespace. A `#`
# glued to the previous token is data, not a comment -- notably the fragment in
# `pip install "git+https://host/repo#subdirectory=pkg"`, which selects the
# package and has to stay in the signature.
_COMMENT_RE = re.compile(r"(?:^|(?<=\s))#.*$", re.MULTILINE)


def _normalize_install(text):
    """Drop the cosmetic half of an install cell, keep what it installs.

    Upstream regenerates this cell on every build, so its comments, blank lines
    and spacing churn across every notebook for no functional reason. That churn
    is why the cell used to be skipped outright -- but its package specs are the
    other half and they are functional: a transformers bump or an added
    dependency is a fix, and the image keys its transformers sidecar off this
    cell (unsloth_pip_shim.py reads the pin from it). Skipping the whole cell
    made such a fix invisible to the refresh forever, because SAME keeps the old
    file AND re-records its old hash, so it never converges. Hash the cell with
    only the cosmetics normalized away instead.
    """
    lines = []
    for line in _COMMENT_RE.sub("", text).split("\n"):
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def middle_digest(path):
    """sha256 over the (type, source) of every non-boilerplate cell, or None.

    Install cells are hashed in normalized form rather than skipped.
    """
    try:
        with open(path, "r", encoding = "utf-8") as f:
            nb = json.load(f)
    except Exception:
        return None
    cells = nb.get("cells")
    if not isinstance(cells, list):
        return None
    h = hashlib.sha256()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if _is_boilerplate_md(cell):
            continue
        text = _text(cell)
        if _is_install_code(cell):
            text = _normalize_install(text)
        h.update(b"\x00")
        h.update(str(cell.get("cell_type", "")).encode("utf-8"))
        h.update(b"\x01")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def main(argv):
    if len(argv) == 2:
        d = middle_digest(argv[1])
        if d is None:
            print("ERR")
            return 0
        print(d)
        return 0
    if len(argv) == 3:
        a = middle_digest(argv[1])
        b = middle_digest(argv[2])
        if a is None or b is None:
            print("ERR")
        elif a == b:
            print("SAME")
        else:
            print("DIFF")
        return 0
    print("ERR")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
