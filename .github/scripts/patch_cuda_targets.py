#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trim the device code each package's setup.py compiles to what a GPU can actually run.

Device code is almost all of every wheel, and part of it is dead weight:

- causal-conv1d and mamba-ssm pass -lineinfo, which only feeds profilers and sanitizers a source
  line table. The generated SASS is identical without it (checked with cuobjdump -sass), and it is
  more than half of every cubin.
- They also hardcode SASS for sm_87 (Jetson Orin), sm_110 (Jetson Thor) and sm_121 (GB10 / DGX
  Spark, N1X). Those are integrated Arm GPUs, so the x86 wheels drop them; the Arm wheels keep the
  full list. Whatever a platform can reach keeps SASS compiled for its own arch.
- flash-attn's setup.py (at our pin) embeds compute_<newest> PTX next to the SASS so an unlisted
  future GPU can JIT. Upstream's 2.8.1 wheels ship SASS only, every GPU flash-attn targets has
  SASS here, and compute_120 PTX cannot JIT for anything older than sm_120, so it is a second copy
  of every kernel for hardware that does not exist yet.

Like patch_mamba_cxx20.py this asserts on text rather than applying a diff: every line it touches
must be in one of two known states, an already-trimmed file is a success, and anything else fails
loudly so a pin bump is re-read by a human instead of building something nobody chose.

Usage: patch_cuda_targets.py --package NAME --archs "75;80;90;100;103;120" <setup.py>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The SASS targets causal-conv1d's and mamba-ssm's setup.py emit on CUDA 13. A pin that adds or
# removes one is a different build and fails here.
SSM_KNOWN_ARCHS = frozenset({"75", "80", "87", "90", "100", "103", "110", "120", "121"})
SSM_PACKAGES = ("causal-conv1d", "mamba-ssm")
# One in the CUDA branch of each; the HIP branch does not pass it.
LINEINFO = '"-lineinfo",'
LINEINFO_OCCURRENCES = 1

FLASH_PTX = 'cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]'
FLASH_PTX_TRIMMED = "pass  # Unsloth: SASS only, no compute_<newest> PTX"

_GENCODE = re.compile(
    r'^(?P<indent>[ \t]*)cc_flag\.append\("-gencode"\)\r?\n'
    r'(?P=indent)cc_flag\.append\("arch=compute_(?P<arch>\d+),code=sm_(?P=arch)"\)\r?\n',
    re.MULTILINE,
)


def _fail(message: str) -> int:
    print(f"::error::{message}", file = sys.stderr)
    return 1


def trim_ssm(source: str, archs: set[str]) -> tuple[str | None, str]:
    """(new source or None on failure, message) for causal-conv1d / mamba-ssm."""
    found = [m.group("arch") for m in _GENCODE.finditer(source)]
    present = set(found)
    if len(found) != len(present):
        return None, f"a gencode is listed twice: {found}"
    unknown = present - SSM_KNOWN_ARCHS
    if unknown:
        return None, f"setup.py emits SASS this script does not know: {sorted(unknown)}"
    missing = archs - present
    if missing:
        return None, f"requested SASS {sorted(missing)} is not in setup.py ({sorted(present)})"
    # Already trimmed: only requested archs left. Untrimmed: exactly the known CUDA 13 set.
    if present != archs and present != SSM_KNOWN_ARCHS:
        return None, (
            f"setup.py emits {sorted(present, key = int)}, neither the known CUDA 13 set nor the "
            f"requested one; re-read it before bumping the pin"
        )
    # A dropped pair becomes `pass`, so an `if` whose every target went keeps a body.
    trimmed = _GENCODE.sub(
        lambda m: m.group(0)
        if m.group("arch") in archs
        else f"{m.group('indent')}pass  # Unsloth: no sm_{m.group('arch')} SASS (patch_cuda_targets.py)\n",
        source,
    )
    lineinfo = trimmed.count(LINEINFO)
    if lineinfo not in (0, LINEINFO_OCCURRENCES):
        return None, f"expected {LINEINFO_OCCURRENCES} {LINEINFO} line, found {lineinfo}"
    trimmed = re.sub(
        r"^[ \t]*" + re.escape(LINEINFO) + r"[ \t]*\r?\n", "", trimmed, flags = re.MULTILINE
    )
    dropped = sorted(present - archs, key = int)
    return trimmed, (
        f"SASS {';'.join(sorted(archs, key = int))} (dropped {dropped or 'none'})"
        + (", -lineinfo removed" if lineinfo else "")
    )


def trim_flash(source: str) -> tuple[str | None, str]:
    if source.count(FLASH_PTX_TRIMMED) == 1 and FLASH_PTX not in source:
        return source, "already SASS only"
    if source.count(FLASH_PTX) != 1:
        return None, f"expected one PTX gencode line ({FLASH_PTX}), found {source.count(FLASH_PTX)}"
    return source.replace(FLASH_PTX, FLASH_PTX_TRIMMED), "compute_<newest> PTX removed, SASS kept"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--package", required = True)
    parser.add_argument("--archs", default = "", help = "SASS targets to keep, ';'-separated")
    parser.add_argument("setup_py", type = Path)
    args = parser.parse_args(argv)
    if not args.setup_py.is_file():
        return _fail(f"{args.setup_py} does not exist")
    source = args.setup_py.read_text(encoding = "utf-8")
    if args.package == "flash-attn":
        new, message = trim_flash(source)
    elif args.package in SSM_PACKAGES:
        archs = {a for a in args.archs.split(";") if a}
        if not archs or not archs <= SSM_KNOWN_ARCHS:
            return _fail(f"--archs must be a subset of {sorted(SSM_KNOWN_ARCHS, key = int)}")
        new, message = trim_ssm(source, archs)
    else:
        return _fail(f"unknown package {args.package!r}")
    if new is None:
        return _fail(f"{args.setup_py}: {message}")
    try:
        compile(new, str(args.setup_py), "exec")
    except SyntaxError as exc:
        return _fail(f"{args.setup_py}: the trimmed setup.py does not parse: {exc}")
    if new != source:
        args.setup_py.write_text(new, encoding = "utf-8")
    print(f"{args.setup_py}: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
