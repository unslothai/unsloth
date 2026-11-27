#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give flash-attn's `varlen_fwd` binding back the 21-argument form callers built for 2.8.3 use.

The pinned revision appends `int num_splits = 0` to mha_varlen_fwd. pybind11 drops C++ default
arguments when binding a function pointer, so the compiled `varlen_fwd` demands all 22 and every
21-argument call fails with "varlen_fwd(): incompatible function arguments". xFormers' FA2 operator
makes exactly that call, so every BlockDiagonal (packed) xFormers attention breaks against our wheel.
This adds a second overload with the 2.8.3 parameter list that forwards num_splits = 0; the 22-argument
form is untouched. The overload's parameters are read from the source, not restated, so they cannot
drift from the function they forward to. Same contract as patch_mamba_msvc.py: asserts the text, a
patched tree is a no-op, anything else fails loudly.

Usage: patch_flash_attn_varlen_compat.py <flash-attention checkout root>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILE = "csrc/flash_attn/flash_api.cpp"
MARKER = '"Forward pass (variable length), without num_splits"'
DEF_LINE = re.compile(
    r'^(?P<indent>[ \t]*)m\.def\("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass \(variable length\)"\);[ \t]*$',
    re.MULTILINE,
)
SIGNATURE = re.compile(r"^mha_varlen_fwd\((?P<params>.*?)\)\s*\{", re.MULTILINE | re.DOTALL)
TRAILING = "int num_splits = 0"


def _params(block: str) -> list[str]:
    """Parameter declarations, comments stripped, in order."""
    text = re.sub(r"//[^\n]*", "", block)
    return [" ".join(part.split()) for part in text.split(",") if part.strip()]


def _name(declaration: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=.*)?$", declaration)
    if not match:
        raise ValueError(f"cannot read a parameter name from {declaration!r}")
    return match.group(1)


def patch_source(source: str, name: str = FILE) -> tuple[str | None, str]:
    """(patched text, or None when nothing is to be done; message). Raises ValueError on drift."""
    if MARKER in source:
        return None, f"{name}: already patched, nothing to do"
    signatures = list(SIGNATURE.finditer(source))
    if len(signatures) != 1:
        raise ValueError(f"{name}: expected 1 mha_varlen_fwd definition, found {len(signatures)}")
    params = _params(signatures[0].group("params"))
    if params[-1] != TRAILING:
        # A revision without num_splits already binds the 21-argument form.
        return None, f"{name}: mha_varlen_fwd takes no num_splits, nothing to do"
    defs = list(DEF_LINE.finditer(source))
    if len(defs) != 1:
        raise ValueError(f"{name}: expected 1 varlen_fwd binding, found {len(defs)}")
    legacy = params[:-1]
    names = ", ".join(_name(p) for p in legacy)
    indent = defs[0].group("indent")
    overload = (
        f"\n{indent}// Unsloth: the 2.8.3 parameter list (xFormers calls it), num_splits = 0.\n"
        f"{indent}m.def(\"varlen_fwd\", []({', '.join(legacy)}) {{\n"
        f"{indent}    return FLASH_NAMESPACE::mha_varlen_fwd({names}, 0);\n"
        f"{indent}}}, {MARKER});"
    )
    end = defs[0].end()
    patched = source[:end] + overload + source[end:]
    return patched, f"{name}: added the {len(legacy)}-argument varlen_fwd overload"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: patch_flash_attn_varlen_compat.py <flash-attention checkout root>",
            file = sys.stderr,
        )
        return 2
    path = Path(argv[1]) / FILE
    if not path.is_file():
        print(f"::error::{path} does not exist", file = sys.stderr)
        return 1
    try:
        patched, message = patch_source(path.read_text(encoding = "utf-8"))
    except ValueError as exc:
        print(
            f"::error::{exc}. Upstream changed flash_api.cpp; re-read it before bumping the pin.",
            file = sys.stderr,
        )
        return 1
    if patched is not None:
        path.write_text(patched, encoding = "utf-8")
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
