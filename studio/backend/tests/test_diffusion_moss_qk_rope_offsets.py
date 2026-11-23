# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS fused q/k RMSNorm + RoPE row offsets must be int64 (checked in sm80 TTIR, CPU only)."""

from __future__ import annotations

import re

import pytest

from core.inference import diffusion_moss_fused as M

kernel = M._qk_rope_kernel()
if kernel is None:
    pytest.skip("Triton is not installed", allow_module_level = True)

try:
    import inspect

    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    _TARGET = GPUTarget("cuda", 80, 32)
    make_backend(_TARGET)
    # Triton 3.2 takes ``constants=``, 3.3 a ``make_ir`` without the target: lower only through the API used below.
    _api_ok = "constexprs" in inspect.signature(ASTSource.__init__).parameters and (
        "target" in inspect.signature(ASTSource.make_ir).parameters
    )
except Exception:  # noqa: BLE001 -- a Triton without the NVIDIA backend or this API
    _api_ok = False
if not _api_ok:
    pytest.skip("this Triton cannot lower to TTIR offline", allow_module_level = True)

_STRIDES = {"sqb", "sqs", "skb", "sks", "svb", "svs", "scs"}


def _ttir():
    sig = {}
    for p in kernel.params:
        if p.is_constexpr:
            sig[p.name] = "constexpr"
        elif p.name.endswith("_ptr"):
            sig[p.name] = (
                "*fp32" if p.name in ("qw_ptr", "kw_ptr", "cos_ptr", "sin_ptr") else "*bf16"
            )
        else:
            sig[p.name] = "i32"
    consts = {"EPS": 1e-6, "D": 128, "HB": 8}
    src = ASTSource(
        kernel,
        signature = sig,
        constexprs = {(kernel.arg_names.index(k),): v for k, v in consts.items()},
    )
    backend = make_backend(_TARGET)
    opts = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    return str(
        src.make_ir(
            _TARGET, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), ctx
        )
    )


def test_row_offsets_are_int64():
    names = _STRIDES | {"row", "b", "s"}
    text = _ttir()
    assert "arith.muli" in text
    bad = []
    for line in text.splitlines():
        m = re.search(r"arith\.muli (%[\w.]+), (%[\w.]+) : i32\b", line)
        if m and any(op.lstrip("%") in names for op in m.groups()):
            bad.append(line.strip())
    assert bad == []
