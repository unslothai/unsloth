# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copy the shared Windows preflight helpers from install.ps1 into studio/setup.ps1.

install.ps1 has to run standalone, before any repo exists, so it cannot dot-source
setup.ps1 and instead carries byte-identical copies of a handful of helpers.
tests/studio/install/test_denied_llama_cpp_preflight.py fails when the two drift.
Editing one and running this keeps them together, rather than hand-copying and
finding out from CI.

Usage:
    python3 scripts/sync_shared_ps1_helpers.py [--check]

--check exits 1 without writing when setup.ps1 is out of date, for use in CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = ROOT / "install.ps1"
SETUP_PS1 = ROOT / "studio" / "setup.ps1"

SHARED_BEGIN = "# ── BEGIN SHARED WITH studio/setup.ps1 ──"
SHARED_END = "# ── END SHARED WITH studio/setup.ps1 ──"


def function_span(text: str, name: str) -> tuple[int, int]:
    """Half-open character span of a PowerShell function, by balanced braces."""
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    if not match:
        raise SystemExit(f"{name} is not defined")
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1
    raise SystemExit(f"unbalanced braces in {name}")


def dedented(source: str) -> str:
    """Drop the common indentation, which is the only difference allowed."""
    lines = source.splitlines()
    body = [line for line in lines if line.strip()]
    indent = min(len(line) - len(line.lstrip(" ")) for line in body)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines)


def shared_names(install_text: str) -> list[str]:
    """The functions declared inside the shared block, in file order."""
    block = install_text.split(SHARED_BEGIN, 1)[1].split(SHARED_END, 1)[0]
    return re.findall(r"(?m)^[ \t]*function[ \t]+([A-Za-z-]+)", block)


def leading_comment_start(text: str, start: int) -> int:
    """Index of the contiguous comment block directly above a function."""
    lines = text[:start].splitlines(keepends = True)
    cut = len(lines)
    while cut > 0 and lines[cut - 1].lstrip().startswith("#"):
        cut -= 1
    return sum(len(line) for line in lines[:cut])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action = "store_true", help = "report drift, write nothing")
    args = parser.parse_args()

    install_text = INSTALL_PS1.read_text(encoding = "utf-8")
    setup_text = SETUP_PS1.read_text(encoding = "utf-8")

    stale: list[str] = []
    previous: str | None = None
    for name in shared_names(install_text):
        start, end = function_span(install_text, name)
        wanted = dedented(install_text[start:end])
        if re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", setup_text):
            setup_start, setup_end = function_span(setup_text, name)
            if setup_text[setup_start:setup_end] == wanted:
                previous = name
                continue
            stale.append(name)
            setup_text = setup_text[:setup_start] + wanted + setup_text[setup_end:]
            previous = name
            continue
        # New helper. Land it where install.ps1 keeps it, after the one before
        # it, with the comment block that explains it.
        if previous is None:
            raise SystemExit(f"{name} is new and has nothing to follow in setup.ps1")
        # dedented() drops the trailing newline, so the comment would otherwise
        # run into the function keyword and hide the declaration from every
        # reader of this file, this script included.
        comment = dedented(install_text[leading_comment_start(install_text, start) : start])
        comment = comment.rstrip("\n") + "\n"
        _, after = function_span(setup_text, previous)
        setup_text = setup_text[:after] + "\n\n" + comment + wanted + setup_text[after:]
        stale.append(name)
        previous = name

    if not stale:
        print("shared helpers already match")
        return 0
    if args.check:
        print("setup.ps1 is behind install.ps1: " + ", ".join(stale))
        print("run: python3 scripts/sync_shared_ps1_helpers.py")
        return 1
    SETUP_PS1.write_text(setup_text, encoding = "utf-8")
    print("synced into studio/setup.ps1: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
