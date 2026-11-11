# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUDA major of an sd.cpp build is the libcudart it links (DT_NEEDED), not the first one a string names.

The CUDA 13 prebuilt's ``libggml-cuda.so`` names "libcudart.so.12" in a dlopen fallback list too; read by string
alone it looked like a CUDA 12 build and was paired with the cu12 cuDNN, which aborts the process on a CUDA 13-only
host ("Invalid handle. Cannot load symbol cublasLtGetVersion", seen on a Colab L4)."""

from __future__ import annotations

import struct
from pathlib import Path

from core.inference import sd_cpp_cudnn as cd


def _elf_with_needed(
    path: Path,
    needed: list[str],
    extra: bytes = b"",
) -> str:
    """A minimal ELF64 LE: a .dynstr, a .dynamic with DT_NEEDED entries, and ``extra`` bytes (rodata strings)."""
    dynstr = b"\0"
    offsets = []
    for name in needed:
        offsets.append(len(dynstr))
        dynstr += name.encode() + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, off) for off in offsets) + struct.pack("<qQ", 0, 0)
    body_off = 64
    dynstr_off = body_off + len(extra)
    dynamic_off = dynstr_off + len(dynstr)
    shoff = dynamic_off + len(dynamic)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HHH", header, 0x3A, 64, 3, 0)
    null = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sec_dynstr = struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, dynstr_off, len(dynstr), 0, 0, 1, 0)
    sec_dynamic = struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dynamic_off, len(dynamic), 1, 0, 8, 16)
    path.write_bytes(bytes(header) + extra + dynstr + dynamic + null + sec_dynstr + sec_dynamic)
    return str(path)


def test_cuda13_build_naming_cudart12_as_fallback_reads_as_13(tmp_path):
    lib = _elf_with_needed(
        tmp_path / "libggml-cuda.so.0",
        ["libcublasLt.so.13", "libcudart.so.13", "libcublas.so.13"],
        extra = b"GGML_CUDA_CUDNN_LIB\0libcudart.so.12\0libcudart.so.13\0",
    )
    assert cd._elf_needed(Path(lib)) == ["libcublasLt.so.13", "libcudart.so.13", "libcublas.so.13"]
    exe = tmp_path / "sd-server"
    exe.write_bytes(b"\x7fELF" + b"\0" * 64)
    assert cd.binary_cudnn_build(str(exe)) == (True, 13)
    plan = cd.plan_cudnn_attention(
        str(exe), (8, 9), allow_install = False, platform = "linux", environ = {}
    )
    assert plan.state == cd.STATE_UNAVAILABLE and "CUDA 13" in plan.reason


def test_cuda12_build_still_reads_as_12(tmp_path):
    _elf_with_needed(
        tmp_path / "libggml-cuda.so.0",
        ["libcudart.so.12", "libcublas.so.12"],
        extra = b"GGML_CUDA_CUDNN_LIB\0libcudart.so.12\0libcudart.so.13\0",
    )
    exe = tmp_path / "sd-cli"
    exe.write_bytes(b"\x7fELF" + b"\0" * 64)
    assert cd.binary_cudnn_build(str(exe)) == (True, 12)


def test_not_an_elf_keeps_the_string_scan(tmp_path):
    path = tmp_path / "sd-cli"
    path.write_bytes(b"MZ GGML_CUDA_CUDNN_LIB\0libcudart.so.12\0")
    assert cd._elf_needed(path) is None
    assert cd.binary_cudnn_build(str(path)) == (True, 12)
