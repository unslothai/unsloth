# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Per-(GPU, width) num_warps table for the LayerNorm / RMSNorm kernels: routing, and the table's warps keep
outputs as accurate as the default ones against an fp64 reference (forward, frozen-W and dW backward)."""

import pytest
import torch
from real_accelerator import has_real_cuda

from unsloth.kernels import layernorm as LN
from unsloth.kernels import rms_layernorm as RMS
from unsloth.kernels import utils as U
from unsloth.kernels.utils import calculate_settings

T4 = (7, 5)
# Read before any fixture runs: the `capability` fixture sets torch.version.hip to None to reach the CUDA tables.
IS_HIP = bool(torch.version.hip)


def _uncached(mod):
    # The eager cache keys on (width, device, dw); these tests change the capability, flag and tables.
    original = mod._settings

    def settings(*args, **kwargs):
        mod._cached_arch_settings.cache_clear()
        RMS._dw_row_config.cache_clear()
        return original(*args, **kwargs)

    return settings


@pytest.fixture
def capability(monkeypatch):
    cap = {"value": T4}
    monkeypatch.setattr(U, "_cuda_capability", lambda index: cap["value"])
    monkeypatch.setattr(U, "_NORM_ARCH_WARPS", True)
    monkeypatch.setattr(U.torch.version, "hip", None)
    for mod in (RMS, LN):
        monkeypatch.setattr(mod, "_settings", _uncached(mod))
    yield cap
    for mod in (RMS, LN):
        mod._cached_arch_settings.cache_clear()
    RMS._dw_row_config.cache_clear()


@pytest.mark.parametrize("dw", [False, True], ids = ["row", "dw"])
@pytest.mark.parametrize("mod", [RMS, LN], ids = ["rms", "ln"])
def test_table_routing(mod, dw, capability, monkeypatch):
    dev = torch.device("cuda", 0)
    table = mod._DW_NUM_WARPS if dw else mod._NUM_WARPS
    other = mod._NUM_WARPS if dw else mod._DW_NUM_WARPS
    for (cap, n_cols), warps in table.items():
        capability["value"] = cap
        assert mod._settings(n_cols, dev, dw = dw) == (calculate_settings(n_cols)[0], warps)
        # the other table only: its own entry or the default
        assert mod._settings(n_cols, dev, dw = not dw)[1] == other.get(
            (cap, n_cols), calculate_settings(n_cols)[1]
        )
        # no device (callers outside the table's kernels), other GPUs: calculate_settings unchanged
        assert mod._settings(n_cols, dw = dw) == calculate_settings(n_cols)
        capability["value"] = (6, 1)
        assert mod._settings(n_cols, dev, dw = dw) == calculate_settings(n_cols)
        capability["value"] = cap
        assert mod._settings(n_cols, torch.device("cpu"), dw = dw) == calculate_settings(n_cols)
        monkeypatch.setattr(U.torch.version, "hip", "6.4")
        assert mod._settings(n_cols, dev, dw = dw) == calculate_settings(n_cols)
        monkeypatch.setattr(U.torch.version, "hip", None)
        monkeypatch.setattr(U, "_NORM_ARCH_WARPS", False)
        assert mod._settings(n_cols, dev, dw = dw) == calculate_settings(n_cols)
        monkeypatch.setattr(U, "_NORM_ARCH_WARPS", True)
        # widths outside the table keep the default
        for width in (1000, 3072, 4095, 14336):
            assert mod._settings(width, dev, dw = dw) == calculate_settings(width)


def test_tables_are_sane():
    for mod in (RMS, LN):
        for table in (mod._NUM_WARPS, mod._DW_NUM_WARPS):
            for (cap, n_cols), warps in table.items():
                block = calculate_settings(n_cols)[0]
                assert warps in (1, 2, 4, 8, 16, 32) and 32 * warps <= block <= 64 * 32 * warps
                assert warps != calculate_settings(n_cols)[1]


def _run(mod, X, W, b, dY, train_w):
    X = X.clone().requires_grad_(True)
    W = W.clone().requires_grad_(train_w)
    if mod is LN:
        b = b.clone().requires_grad_(train_w)
        Y = LN.Fast_Layernorm.apply(X, W, b, 1e-5)
    else:
        Y = RMS.Fast_RMS_Layernorm.apply(X, W, 1e-6, False, False, False)
    Y.backward(dY)
    out = {"y": Y.detach(), "dx": X.grad}
    if train_w:
        out["dw"] = W.grad
        if mod is LN:
            out["db"] = b.grad
    return out


def _ref(mod, X, W, b, dY):
    Xd = X.double().requires_grad_(True)
    Wd = W.double().requires_grad_(True)
    bd = b.double().requires_grad_(True)
    if mod is LN:
        Y = torch.nn.functional.layer_norm(Xd, (X.shape[-1],), Wd, bd, 1e-5)
    else:
        Y = Xd * torch.rsqrt(Xd.pow(2).mean(-1, keepdim = True) + 1e-6) * Wd
    Y.backward(dY.double())
    return {"y": Y.detach(), "dx": Xd.grad, "dw": Wd.grad, "db": bd.grad}


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("mod", [RMS, LN], ids = ["rms", "ln"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids = ["fp16", "bf16"])
def test_table_warps_match_default_accuracy(mod, dtype, capability, monkeypatch):
    # Run every table entry on this GPU (the capability is faked) against the default warps.
    if IS_HIP:
        # The fixture fakes a CUDA identity, so the dW launch would pass maxnreg, which HIP Triton rejects.
        monkeypatch.setattr(RMS, "_max_sm_threads", lambda device: 0)
    entries = sorted(set(mod._NUM_WARPS) | set(mod._DW_NUM_WARPS))
    gen = torch.Generator(device = "cuda").manual_seed(3407)
    for cap, n_cols in entries:
        for rows in (1, 777, 4096):
            X = torch.randn(rows, n_cols, device = "cuda", generator = gen).to(dtype)
            W = (1 + 0.1 * torch.randn(n_cols, device = "cuda", generator = gen)).to(dtype)
            b = (0.1 * torch.randn(n_cols, device = "cuda", generator = gen)).to(dtype)
            dY = torch.randn(rows, n_cols, device = "cuda", generator = gen).to(dtype)
            ref = _ref(mod, X, W, b, dY)
            results = {}
            for arm in ("default", "table"):
                capability["value"] = cap if arm == "table" else (6, 1)
                RMS._dw_row_config.cache_clear()
                row = mod._NUM_WARPS.get((cap, n_cols)) if arm == "table" else None
                dw = mod._DW_NUM_WARPS.get((cap, n_cols)) if arm == "table" else None
                assert mod._settings(n_cols, X.device)[1] == (row or calculate_settings(n_cols)[1])
                assert mod._settings(n_cols, X.device, dw = True)[1] == (
                    dw or calculate_settings(n_cols)[1]
                )
                frozen = _run(mod, X, W, b, dY, False)
                trained = _run(mod, X, W, b, dY, True)
                results[arm] = {**frozen, **{k + "_w": v for k, v in trained.items()}}
            RMS._dw_row_config.cache_clear()
            for key, value in results["table"].items():
                base = key[:-2] if key.endswith("_w") else key
                default = results["default"][key]
                err = (value.double() - ref[base]).abs().max().item()
                err_default = (default.double() - ref[base]).abs().max().item()
                ulp = torch.finfo(dtype).eps * max(ref[base].abs().max().item(), 1.0)
                assert torch.isfinite(value).all()
                # A different reduction tree: allow one output ulp over the default's max error.
                assert err <= err_default + ulp, (cap, n_cols, rows, key, err, err_default)
