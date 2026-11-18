# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention Triton kernels form their batch / head offsets in int64. Strides arrive as i32, so
``b * x_stride_b`` in i32 wraps once the product passes 2**31 (a batch stride of 2**30 at batch 2 lands at -2**31).
CPU only: every kernel is lowered to TTIR for an sm80 target (no GPU, no PTX) and no scalar i32 multiply may take a
stride argument or the (batch, head) program index as an operand."""

from __future__ import annotations

import re

import pytest

from core.inference import diffusion_int8_attention_triton as T

if T.triton is None:
    pytest.skip("Triton is not installed", allow_module_level = True)

try:
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    _TARGET = GPUTarget("cuda", 80, 32)
    make_backend(_TARGET)
except Exception:  # noqa: BLE001 -- a Triton without the NVIDIA backend or this API
    pytest.skip("this Triton cannot lower to TTIR offline", allow_module_level = True)

_STRIDE = re.compile(r"(_stride_[bsh]|_s[bsh])$")
_FLOATS = ("mul", "kmul", "qk_mul")


def _ttir(kernel, consts, ptrs):
    sig = {}
    for p in kernel.params:
        if p.is_constexpr:
            sig[p.name] = "constexpr"
        elif p.name.endswith("_ptr"):
            sig[p.name] = ptrs.get(p.name, "*fp16")
        else:
            sig[p.name] = "fp32" if p.name in _FLOATS else "i32"
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


def _narrow_offsets(text, kernel):
    names = {p.name for p in kernel.params if _STRIDE.search(p.name)} | {"bh", "b", "h"}
    bad = []
    for line in text.splitlines():
        m = re.search(r"arith\.muli (%[\w.]+), (%[\w.]+) : i32\b", line)
        if m and any(op.lstrip("%") in names for op in m.groups()):
            bad.append(line.strip())
    return bad


_F32 = {
    n: "*fp32"
    for n in ("c_ptr", "g_ptr", "s_ptr", "qs_ptr", "kc_ptr", "vc_ptr", "vg_ptr", "ks_ptr", "vs_ptr")
}
_I8 = {n: "*i8" for n in ("q_ptr", "qi_ptr", "ki_ptr", "vi_ptr")}

CASES = (
    [("stats_kernel", {"STEP": 8, "R": 64, "D": 64, "MODE": m}, _F32) for m in (0, 1)]
    + [
        ("quant_rows_kernel", {"R": 64, "D": 128, "CENTRE": c, "TILE": t}, {**_F32, "q_ptr": "*i8"})
        for c in (False, True)
        for t in (False, True)
    ]
    + [
        ("quant_v_kernel", {"T": 32, "D": 128, "TRANS": t}, {**_F32, "q_ptr": "*i8"})
        for t in (False, True)
    ]
    + [("transpose_v_kernel", {"T": 64, "D": 128, "TO_F16": f}, {}) for f in (False, True)]
    + [
        (
            "prep_kernel",
            {"R": 64, "STEP": 8, "SR": 64, "D": 128, "V_STATS": s, "V_T": t},
            {**_F32, "qi_ptr": "*i8"},
        )
        for s, t in ((0, 0), (1, 0), (0, 1), (0, 2))
    ]
    + [
        ("kv_kernel", {"RK": 64, "T": 32, "D": 128, "K_TILE": kt, "V_Q": q}, {**_F32, **_I8})
        for kt, q in ((False, 0), (True, 1), (True, 2))
    ]
)


@pytest.mark.parametrize("name,consts,ptrs", CASES)
def test_quantiser_offsets_are_int64(name, consts, ptrs):
    kernel = getattr(T, name)
    assert _narrow_offsets(_ttir(kernel, consts, ptrs), kernel) == []


@pytest.mark.parametrize("pv", sorted(T.PV_MODES))
def test_attention_offsets_are_int64(monkeypatch, pv):
    import torch

    monkeypatch.setattr(T, "prefetch_fix", lambda: False)
    q = torch.empty(1, 1000, 4, 128, device = "meta", dtype = torch.float16)
    plan = T._make_plan(q, q, q, dict(T.DEFAULT_CONFIGS[128], pv = pv), None, None)
    meta = plan["attn_meta"]
    consts = {p.name: meta[p.name] for p in T.attn_kernel.params if p.is_constexpr}
    ptrs = {
        **_F32,
        "q_ptr": "*i8",
        "k_ptr": "*i8",
        "v_ptr": "*i8" if meta["PV_MODE"] in (2, 4) else "*fp16",
    }
    assert _narrow_offsets(_ttir(T.attn_kernel, consts, ptrs), T.attn_kernel) == []
