#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make mamba-ssm's selective-scan kernels preprocess with MSVC as nvcc's host compiler.

Windows only. An `#ifndef USE_ROCM` block sits inside the lambda passed to BOOL_SWITCH (a macro
argument); gcc accepts that, MSVC does not (`error: "#" not expected here`). It is replaced by its
CUDA branch, which is what a CUDA build preprocesses to anyway. Same contract as
patch_mamba_cxx20.py: asserts the text, a patched tree is a no-op, anything else fails loudly.

Usage: patch_mamba_msvc.py <mamba checkout root>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = (
    "csrc/selective_scan/selective_scan_fwd_kernel.cuh",
    "csrc/selective_scan/selective_scan_bwd_kernel.cuh",
)

# The in-lambda block: deeper than function scope (the function-scope ones sit at 4 spaces).
NESTED_ROCM = re.compile(
    r"^[ \t]{12,}#ifndef USE_ROCM[ \t]*\n"
    r"(?P<cuda>.*?)"
    r"^[ \t]+#else[ \t]*\n"
    r".*?"
    r"^[ \t]+#endif[ \t]*\n",
    re.MULTILINE | re.DOTALL,
)

# What the CUDA branch sets, so a patched file is recognisable without a marker comment.
CUDA_CALL = "cudaFuncSetAttribute(\n"


def patch_source(source: str, name: str) -> tuple[str | None, str]:
    """(patched text, or None when already patched; message). Raises ValueError on drift."""
    nested = list(NESTED_ROCM.finditer(source))
    if not nested and CUDA_CALL in source:
        return None, f"{name}: already patched, nothing to do"
    if len(nested) != 1:
        raise ValueError(
            f"{name}: expected 1 USE_ROCM block inside a launch lambda, found {len(nested)}"
        )
    match = nested[0]
    patched = source[: match.start()] + match.group("cuda") + source[match.end() :]
    return patched, f"{name}: in-lambda USE_ROCM block reduced to its CUDA branch"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: patch_mamba_msvc.py <mamba checkout root>", file = sys.stderr)
        return 2
    root = Path(argv[1])
    for rel in FILES:
        path = root / rel
        if not path.is_file():
            print(f"::error::{path} does not exist", file = sys.stderr)
            return 1
        try:
            patched, message = patch_source(path.read_text(encoding = "utf-8"), rel)
        except ValueError as exc:
            print(
                f"::error::{exc}. Upstream changed the kernels; re-read them before bumping the pin.",
                file = sys.stderr,
            )
            return 1
        if patched is not None:
            path.write_text(patched, encoding = "utf-8")
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
