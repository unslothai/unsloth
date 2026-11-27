# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DoRA factored norm (unsloth/kernels/dora.py): float64 rank terms only on devices that have float64.

Intel Arc XPUs (and MPS) have no float64: a float64 tensor there raises "Required aspect fp64 is not supported
on the device", so DoRA training would crash in the norm on every forward. CPU only."""

import importlib.util
import os
import sys

import pytest
import torch
from torch.overrides import TorchFunctionMode

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "kernels", "dora.py")


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_kernels_dora_fp64", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    print("dora module:", mod.__file__)
    return mod


@pytest.mark.parametrize(
    "device_type, expected",
    [
        ("cuda", torch.float64),
        ("cpu", torch.float64),
        ("xpu", torch.float32),
        ("mps", torch.float32),
    ],
)
def test_rank_dtype_per_device(device_type, expected):
    mod = _load()
    assert mod._rank_dtype(torch.device(device_type)) == expected


class _NoFloat64(TorchFunctionMode):
    """Fails like an Arc XPU on any float64 tensor produced inside the norm."""

    def __torch_function__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for t in outs:
            if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
                raise RuntimeError(f"Required aspect fp64 is not supported on the device ({func})")
        return out


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cancel", [False, True])
def test_float32_rank_path_has_no_float64_and_stays_accurate(monkeypatch, w_dtype, cancel):
    mod = _load()
    monkeypatch.setattr(mod, "_rank_dtype", lambda device, rank = 0: torch.float32)
    torch.manual_seed(0)
    d_out, d_in, r, s = 256, 512, 16, 2.0
    if cancel:
        U, V = torch.randn(d_out, r) * 0.02, torch.randn(r, d_in) / r**0.5
        W, A, B = (U @ V).to(w_dtype), V.float(), -U * 0.97 / s
    else:
        W = (torch.randn(d_out, d_in) * 0.02).to(w_dtype)
        A, B = torch.randn(r, d_in) / d_in**0.5, torch.randn(d_out, r) * 1e-3
    with _NoFloat64():
        norm = mod.factored_dora_weight_norm(W, A, B, s, torch.float32)
    ref = torch.linalg.vector_norm(W.double() + s * (B.double() @ A.double()), dim = 1)
    rel = ((norm.double() - ref).abs() / ref).max().item()
    # Well inside one 16-bit ulp of the output (bf16 2**-8, fp16 2**-11).
    assert rel < (1e-3 if cancel else 1e-5), rel


@pytest.mark.parametrize(
    "device_type, rank, expected",
    [
        ("cuda", 16, torch.float64),
        ("cuda", 32, torch.float64),
        ("cuda", 64, torch.float32),
        ("cuda", 256, torch.float32),
        ("cpu", 256, torch.float64),
        ("xpu", 16, torch.float32),
    ],
)
def test_rank_dtype_per_rank(device_type, rank, expected):
    # float64 runs at 1/64 rate on T4 / L4 / RTX: 4 r^2 (d_in + d_out) float64 FLOPs per call made a rank-256 DoRA
    # step 2.7x slower on an RTX 3090. High ranks use float32 on GPUs.
    mod = _load()
    assert mod._rank_dtype(torch.device(device_type), rank) == expected


@pytest.mark.parametrize("rank", [64, 256])
def test_float32_rank_terms_at_high_rank_stay_accurate(rank):
    mod = _load()
    torch.manual_seed(0)
    d_out, d_in, s = 256, 512, 2.0
    W = (torch.randn(d_out, d_in) * 0.02).to(torch.bfloat16)
    A, B = torch.randn(rank, d_in) / d_in**0.5, torch.randn(d_out, rank) * 1e-3
    mod_rank = mod._rank_dtype
    try:
        mod._rank_dtype = lambda device, r = 0: torch.float32
        norm = mod.factored_dora_weight_norm(W, A, B, s, torch.float32)
    finally:
        mod._rank_dtype = mod_rank
    ref = torch.linalg.vector_norm(W.double() + s * (B.double() @ A.double()), dim = 1)
    assert ((norm.double() - ref).abs() / ref).max().item() < 1e-5
