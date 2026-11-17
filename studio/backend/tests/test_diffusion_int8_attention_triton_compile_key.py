# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fused INT8 attention quantiser launches (``prep_kernel``, ``kv_kernel``) and the attention kernel compile once per
configuration, not once per sequence length or head count class: Triton specialises an int argument on ``% 16 == 0``
and ``== 1`` unless told not to, so the token, block and head counts must stay out of the compile key (ROCm builds keep
the attention kernel's sequence lengths in it). CPU only: the key is computed from the launch arguments with Triton's
own specialisation function, nothing is compiled."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_int8_attention_triton as T

if T.triton is None:
    pytest.skip("Triton is not installed", allow_module_level = True)

try:
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import BaseBackend
except Exception:  # noqa: BLE001 -- older Triton without the native specialiser
    pytest.skip("this Triton has no native_specialize_impl", allow_module_level = True)

# Protocol token counts plus the FLUX / Wan sizes the kernels serve: every residue class mod 16, and 1-block cases.
LENGTHS = (
    126,
    1000,
    1024,
    1039,
    2048,
    2560,
    3000,
    4096,
    4112,
    4176,
    6144,
    10011,
    16464,
    32760,
    75600,
)


def _key(kernel, args):
    # The runtime int arguments of a launch, in order, against the kernel's non-constexpr parameters.
    params = [p for p in kernel.params if not p.is_constexpr]
    ints = []
    for p, a in zip(params, args):
        if isinstance(a, int) and not isinstance(a, bool):
            spec = not p.do_not_specialize
            align = not p.do_not_specialize_on_alignment
            ints.append((p.name, native_specialize_impl(BaseBackend, a, False, spec, align)))
    return tuple(ints)


# Head counts of the served models (Wan 2.2 5B 24, Z-Image 30, FLUX / Qwen-Image / LTX 32, Wan A14B 40,
# HunyuanVideo 56): both residue classes mod 16.
HEADS = (24, 30, 32, 40, 56)


@pytest.mark.parametrize("D", [128, 64])
def test_fused_quantiser_kernels_compile_once_across_lengths(monkeypatch, D):
    monkeypatch.setattr(T, "prefetch_fix", lambda: False)
    cfg = dict(T.DEFAULT_CONFIGS[D])
    keys = {"prep": set(), "kv": set()}
    n_ptrs = {"prep": 9, "kv": 9}
    for S, H in ((S, H) for S in LENGTHS for H in HEADS):
        q = torch.empty(1, S, H, D, device = "meta", dtype = torch.bfloat16)
        plan = T._make_plan(q, q, q, cfg, None, None)
        for name, kernel in (("prep", T.prep_kernel), ("kv", T.kv_kernel)):
            args = plan[f"{name}_args"]
            params = [p for p in kernel.params if not p.is_constexpr]
            assert len(params) == n_ptrs[name] + len(args)
            keys[name].add(_key(kernel, (None,) * n_ptrs[name] + tuple(args)))
    assert len(keys["prep"]) == 1, keys["prep"]
    assert len(keys["kv"]) == 1, keys["kv"]


@pytest.mark.parametrize("D", [128, 64])
def test_attention_kernel_compiles_once_across_lengths(monkeypatch, D):
    monkeypatch.setattr(T, "prefetch_fix", lambda: False)
    cfg = dict(T.DEFAULT_CONFIGS[D])
    keys = set()
    params = [p for p in T.attn_kernel.params if not p.is_constexpr]
    for S, H in ((S, H) for S in LENGTHS for H in HEADS):
        q = torch.empty(1, S, H, D, device = "meta", dtype = torch.bfloat16)
        args = T._make_plan(q, q, q, cfg, None, None)["attn_args"]
        assert len(params) == 9 + len(args)
        keys.add(_key(T.attn_kernel, (None,) * 9 + tuple(args)))
    if T._rocm_build():
        # the % 16 facts on sq / sk are worth 3% at 75k tokens on gfx1151: one variant per residue class there
        assert len(keys) > 1, keys
    else:
        assert len(keys) == 1, keys
