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


_MAGIC_INSTALL = ("%pip", "%uv", "%conda")


def _is_shell_cell(text):
    """A `%%bash` / `%%sh` cell magic makes every line of the cell a shell command."""
    for line in text.split("\n"):
        s = line.strip()
        if s:
            return s.startswith(("%%bash", "%%sh", "%%script"))
    return False


def _unquoted(line):
    """`line` with every quoted stretch blanked, so a marker inside a string literal
    cannot be read as a command. Length is preserved; only the characters change."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(" ")
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# Shell command separators. A marker right after one of these begins a command; the
# same words anywhere else on the segment are an argument or an operand.
_SEG_RE = re.compile(r"\|\||&&|[;&|(){}`]")

# Words that may stand in front of a command without changing what the command is.
_CMD_PREFIXES = ("sudo", "env", "time", "nohup", "xargs", "then", "do", "else", "!")

_PY_DASH_M = re.compile(r"^\S*python[0-9.]* -m ")


def _starts_command(segment):
    """True when `segment` invokes an install marker rather than merely naming one."""
    s = " ".join(segment.split())
    while True:
        head = s.split(" ", 1)
        if len(head) != 2:
            break
        # a leading `VAR=value` is a per-command environment, not the command
        if head[0] in _CMD_PREFIXES or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head[0]):
            s = head[1]
            continue
        break
    return _PY_DASH_M.sub("", s).startswith(_INSTALL_MARKERS)


def _is_install_line(line, shell):
    """A real install invocation rather than a mention of one: `!`, a `%pip`-family
    magic, or a command position inside a shell cell. On an ordinary Python line the
    same words are prose or data, and treating the cell as an install cell on that
    basis put four shipped notebooks through the flattening below over a comment (see
    tests/python/test_docker_nb_install_cell_sig.py).

    A `%%bash` cell used to qualify on the whole line, which is the same mistake one
    level down: `msg="pip install foo # bar"` counted, and _normalize_install then
    stripped the quoted `# bar` and collapsed the rest, so editing that string upstream
    produced SAME and the refresh kept the old shell behaviour. Quoted stretches are
    blanked and the marker has to open a command, which every one of the 1064 marker
    lines in the 561 shipped notebooks already does."""
    low = _unquoted(line.lower())
    if not any(m in low for m in _INSTALL_MARKERS):
        return False
    s = low.lstrip()
    if s.startswith("!") or s.startswith(_MAGIC_INSTALL):
        return True
    return shell and any(_starts_command(seg) for seg in _SEG_RE.split(low))


def _is_install_code(cell):
    if cell.get("cell_type") != "code":
        return False
    t = _text(cell)
    shell = _is_shell_cell(t)
    return any(_is_install_line(line, shell) for line in t.split("\n"))


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
    re-records its old hash.

    Only the install COMMANDS are flattened. Three quarters of the install cells in
    unslothai/notebooks carry ordinary Python or shell around the install, and
    flattening that as well made a `#` or a run of spaces inside a string literal
    cosmetic. A non-install line therefore keeps its bytes, unless it is blank or a
    whole-line comment, or it contains no quote and no `#` at all and so has nowhere
    for whitespace to be data."""
    shell = _is_shell_cell(text)
    lines = []
    for raw in text.split("\n"):
        if _is_install_line(raw, shell):
            line = _COMMENT_RE.sub("", raw)
        else:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(ch in raw for ch in ('"', "'", "#")):
                lines.append(raw)
                continue
            line = raw
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
