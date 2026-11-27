#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pack a built library into ``unsloth_studio_i8gemm-<version>-py3-none-<platform>.whl``.

python make_wheel.py --lib build/libunsloth_i8gemm.so --platform manylinux_2_28_x86_64 --out dist/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import zipfile
from pathlib import Path

NAME = "unsloth_studio_i8gemm"
DIST = "unsloth-studio-i8gemm"
VERSION = "1.0.0"
ABI_VERSION = 1


def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def build(
    lib: Path,
    platform: str,
    out: Path,
    cutlass_sha: str = "",
) -> Path:
    tag = f"py3-none-{platform}"
    dist_info = f"{NAME}-{VERSION}.dist-info"
    init = (
        '"""Prebuilt int8 GEMM for Unsloth Studio (loaded with ctypes by diffusion_int8_cutlass)."""\n'
        "import os\n\n"
        f"ABI_VERSION = {ABI_VERSION}\n"
        f"CUTLASS_SHA = {cutlass_sha!r}\n"
        f"LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', {lib.name!r})\n"
    ).encode()
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {DIST}\n"
        f"Version: {VERSION}\n"
        "Summary: Prebuilt CUTLASS int8 GEMM with a fused torchao dequant epilogue for Unsloth Studio\n"
        "License: AGPL-3.0-only\n"
        "Requires-Python: >=3.9\n"
    ).encode()
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: unsloth-studio make_wheel.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    ).encode()
    files = {
        f"{NAME}/__init__.py": init,
        f"{NAME}/lib/{lib.name}": lib.read_bytes(),
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel,
        f"{dist_info}/top_level.txt": f"{NAME}\n".encode(),
    }
    record = [_record_line(p, d) for p, d in files.items()] + [f"{dist_info}/RECORD,,"]
    files[f"{dist_info}/RECORD"] = ("\n".join(record) + "\n").encode()
    out.mkdir(parents = True, exist_ok = True)
    path = out / f"{NAME}-{VERSION}-{tag}.whl"
    with zipfile.ZipFile(path, "w", compression = zipfile.ZIP_DEFLATED, compresslevel = 9) as zf:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time = (2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if name.endswith((".so", ".dll")) else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required = True)
    ap.add_argument(
        "--platform",
        required = True,
        help = "wheel platform tag, e.g. manylinux_2_28_x86_64, win_amd64",
    )
    ap.add_argument("--out", required = True)
    ap.add_argument("--cutlass-sha", default = "")
    args = ap.parse_args()
    path = build(Path(args.lib), args.platform, Path(args.out), args.cutlass_sha)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
