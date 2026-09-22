#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Switch mamba-ssm's CUDA build from -std=c++17 to -std=c++20, which torch 2.13 requires.

state-spaces/mamba compiles its Mamba-1 selective-scan kernels at C++17. torch 2.13 raised its
own headers to C++20, and nvcc stops at the first construct it cannot parse at 17, so the build
fails before it produces anything. flash-attn already made the same change upstream
(Dao-AILab/flash-attention#2899); mamba has not, which is why this file exists and there is no
equivalent for the other two packages.

Only the four `-std=c++17` occurrences inside setup.py's `if KEEP_CUDA_BUILD:` block are
touched, in both the HIP and the CUDA branch, because both are inside it and leaving one behind
would mean the patch silently stops applying if upstream reorders them.

It is deliberately NOT a `git apply` of a diff. A diff carries line numbers and context, so an
unrelated edit above it turns "upstream changed the flags" into "the hunk did not apply", and
the two want different responses. This asserts on the text instead: every occurrence must be one
of the two known states, an already-patched file is a success with nothing to do, and anything
else -- a new flag, a third branch, a c++23 bump upstream -- fails loudly so a human decides,
rather than building something nobody chose.

Usage: patch_mamba_cxx20.py <path to mamba setup.py>
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD = '"-std=c++17"'
NEW = '"-std=c++20"'

# Two in the HIP branch, two in the CUDA branch: the "cxx" list and the nvcc list of each.
EXPECTED_OCCURRENCES = 4


def patch(path: Path) -> int:
    source = path.read_text(encoding = "utf-8")

    already = source.count(NEW)
    found = source.count(OLD)

    if found == 0 and already >= EXPECTED_OCCURRENCES:
        print(f"{path}: already at c++20 ({already} occurrences), nothing to do")
        return 0

    if found != EXPECTED_OCCURRENCES:
        print(
            f"::error::{path}: expected {EXPECTED_OCCURRENCES} occurrences of {OLD}, "
            f"found {found} (and {already} of {NEW}). Upstream changed the compile flags; "
            "re-read setup.py before bumping the pin.",
            file = sys.stderr,
        )
        return 1

    path.write_text(source.replace(OLD, NEW), encoding = "utf-8")
    print(f"{path}: {found} occurrences of {OLD} -> {NEW}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: patch_mamba_cxx20.py <setup.py>", file = sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"::error::{path} does not exist", file = sys.stderr)
        return 1
    return patch(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
