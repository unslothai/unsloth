#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import hashlib
import json
import re
import sys

_BOILERPLATE_MD = (
    "to run this, press",
    'press "*runtime*"',
    "### news",
    "introducing **unsloth studio**",
    "you will learn how to do",
    "this notebook is licensed",
    "and we're done",
    "this notebook and all unsloth notebooks are licensed",
    "join discord if you need help",
    "star us on",
    "some other resources",
)


def _text(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.replace("\r\n", "\n").replace("\r", "\n")


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
    return False


def _is_boilerplate_md(cell):
    if cell.get("cell_type") != "markdown":
        return False
    low = _text(cell).lower()
    return any(m in low for m in _BOILERPLATE_MD)


# A `#` glued to the previous token is data, not a comment: the fragment in
# `pip install "git+https://host/repo#subdirectory=pkg"` selects the package.
_COMMENT_RE = re.compile(r"(?:^|(?<=\s))#.*$", re.MULTILINE)


def _normalize_install(text):
    """Drop the cosmetic half of an install cell, keep what it installs. Its package
    specs are functional and the image keys its transformers sidecar off them, so
    skipping the whole cell hid an upstream fix forever: SAME keeps the old file AND
    re-records its old hash."""
    lines = []
    for line in _COMMENT_RE.sub("", text).split("\n"):
        body = " ".join(line.split())
        if not body:
            continue
        # Keep the INDENTATION: the cell is an `if "COLAB_" not in ...:` block, so a
        # line's indent decides which runtime it runs on. Only the spacing WITHIN a
        # line churns, and tabs expand so a tab/space rewrite stays cosmetic.
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(" " * len(indent.expandtabs(4)) + body)
    return "\n".join(lines)


def middle_digest(path):
    """sha256 over the (type, source) of every non-boilerplate cell, or None."""
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
