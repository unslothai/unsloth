# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit ``flash`` on ROCm: honored only when flash_attn imports and the per-device check passes; ``auto`` and every
NVIDIA path unchanged; the CUDA flash-attn wheel is never pip installed onto ROCm. Hermetic, CPU only."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import select_attention_backend


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.setattr(att, "_ROCM_FLASH_PROBE_CACHE", {}, raising = False)
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d == "cuda" else d, raising = False
    )
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())


def _target(device = "cuda", dtype = "torch.bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _rocm(
    monkeypatch,
    *,
    flash_attn = True,
    probe = True,
):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setitem(
        sys.modules, "flash_attn", types.ModuleType("flash_attn") if flash_attn else None
    )
    calls = []

    def fake_probe(device, dtype):
        calls.append((device, dtype))
        return probe

    monkeypatch.setattr(att, "_run_rocm_flash_probe", fake_probe, raising = False)
    return calls


@pytest.mark.parametrize("alias", ["flash", "flash2", "FLASH"])
def test_rocm_flash_honored_when_imported_and_check_passes(monkeypatch, alias):
    calls = _rocm(monkeypatch)
    assert select_attention_backend(_target(), alias, speed_active = False) == "flash"
    assert select_attention_backend(_target(), alias, speed_active = True) == "flash"
    assert calls == [("cuda:0", "torch.bfloat16")]  # cached per (device, dtype)


def test_rocm_flash_check_cached_per_device_and_dtype(monkeypatch):
    calls = _rocm(monkeypatch)
    select_attention_backend(_target(), "flash", speed_active = True)
    select_attention_backend(_target("cuda", "torch.float16"), "flash", speed_active = True)
    select_attention_backend(
        types.SimpleNamespace(device = "cuda", torch_device = "cuda:1", dtype = "torch.float16"),
        "flash",
        speed_active = True,
    )
    assert calls == [
        ("cuda:0", "torch.bfloat16"),
        ("cuda:0", "torch.float16"),
        ("cuda:1", "torch.float16"),
    ]


def test_rocm_flash_without_flash_attn_is_native(monkeypatch):
    calls = _rocm(monkeypatch, flash_attn = False)
    assert select_attention_backend(_target(), "flash", speed_active = True) is None
    assert calls == []


def test_rocm_flash_check_failure_is_native(monkeypatch):
    _rocm(monkeypatch, probe = False)
    assert select_attention_backend(_target(), "flash", speed_active = True) is None


def test_rocm_unaskable_check_is_native_and_not_cached(monkeypatch):
    _rocm(monkeypatch)

    def boom(device, dtype):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(att, "_run_rocm_flash_probe", boom)
    assert select_attention_backend(_target(), "flash", speed_active = True) is None
    assert att._ROCM_FLASH_PROBE_CACHE == {}


def test_rocm_auto_unchanged(monkeypatch):
    calls = _rocm(monkeypatch)
    assert select_attention_backend(_target(), "auto", speed_active = True) is None
    assert select_attention_backend(_target(), "auto", speed_active = False) is None
    assert calls == []


@pytest.mark.parametrize("alias", ["flash3", "flash4", "sage", "cudnn", "xformers"])
def test_rocm_other_nvidia_kernels_still_dropped(monkeypatch, alias):
    _rocm(monkeypatch)
    assert select_attention_backend(_target(), alias, speed_active = True) is None


@pytest.mark.parametrize("device", ["mps", "cpu", "xpu"])
def test_flash_off_cuda_still_native(monkeypatch, device):
    _rocm(monkeypatch)
    assert select_attention_backend(_target(device), "flash", speed_active = True) is None


# origin/main behaviour on an NVIDIA target at SM90 (flash3 in range, flash4 out). Frozen: must never move.
_NVIDIA_EXPECTED = {
    "auto": ("_native_cudnn", None),
    "native": (None, None),
    "sdpa": (None, None),
    "cudnn": ("_native_cudnn", "_native_cudnn"),
    "flash": ("flash", "flash"),
    "flash2": ("flash", "flash"),
    "flash3": ("_flash_3_hub", "_flash_3_hub"),
    "flash4": (None, None),
    "sage": ("sage", "sage"),
    "xformers": ("xformers", "xformers"),
    "aiter": (None, None),
}


@pytest.mark.parametrize("alias", sorted(_NVIDIA_EXPECTED))
def test_nvidia_selection_identical_to_main(monkeypatch, alias):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)

    def never(*a, **k):
        raise AssertionError("ROCm flash check must not run on NVIDIA")

    monkeypatch.setattr(att, "_run_rocm_flash_probe", never, raising = False)
    on, off = _NVIDIA_EXPECTED[alias]
    assert select_attention_backend(_target(), alias, speed_active = True) == on
    assert select_attention_backend(_target(), alias, speed_active = False) == off


def test_alias_table_covered():
    assert set(_NVIDIA_EXPECTED) == set(att.ATTN_ALIASES)


def _fake_torch(monkeypatch, *, hip):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = hip)
    torch.__version__ = "2.9.0+rocm6.4" if hip else "2.9.0+cu128"
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_installer_never_pip_installs_flash_attn_on_rocm(monkeypatch):
    _fake_torch(monkeypatch, hip = "6.4.0")
    monkeypatch.delitem(sys.modules, "flash_attn", raising = False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    monkeypatch.delenv(att._ATTENTION_INSTALL_ENV, raising = False)
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    reason = att._ensure_attention_backend_installed("flash")
    assert ran == []
    assert reason and "CUDA" in reason
    assert att._INSTALL_ATTEMPTED == set()


def test_installer_still_installs_flash_attn_on_nvidia(monkeypatch):
    _fake_torch(monkeypatch, hip = None)
    monkeypatch.delitem(sys.modules, "flash_attn", raising = False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    monkeypatch.delenv(att._ATTENTION_INSTALL_ENV, raising = False)
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a[0]))
    assert att._ensure_attention_backend_installed("flash") is None
    assert len(ran) == 1 and ran[0][-1] == "flash-attn"


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("good", [True, False])
def test_probe_compares_against_reference(monkeypatch, dtype_name, good):
    torch = pytest.importorskip("torch")
    fa = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v):
        assert q.dtype in (torch.float16, torch.bfloat16)
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                *(t.float().transpose(1, 2) for t in (q, k, v))
            )
            .transpose(1, 2)
            .to(q.dtype)
        )
        return out if good else out + 0.1

    fa.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", fa)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    expected = good and dtype_name != "float32"
    assert att._run_rocm_flash_probe("cpu", getattr(torch, dtype_name)) is expected


def test_apply_reverifies_rocm_flash_at_run_dtype(monkeypatch):
    calls = _rocm(monkeypatch, probe = False)
    set_to = []
    dit = types.SimpleNamespace(set_attention_backend = set_to.append)
    pipe = types.SimpleNamespace(transformer = dit)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: att.ATTN_NATIVE)
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    run_target = _target(dtype = "torch.float32")
    assert att.apply_attention_backend(pipe, "flash", target = run_target) is None
    assert calls == [("cuda:0", "torch.float32")]
    assert set_to == []


def test_probe_kernel_error_is_false(monkeypatch):
    torch = pytest.importorskip("torch")
    fa = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v):
        raise RuntimeError("HIP error: invalid device function")

    fa.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", fa)
    assert att._run_rocm_flash_probe("cpu", torch.bfloat16) is False


def test_cudnn_head_dim_probe_still_answers_fp32_at_bf16(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.backends.cuda as tbc

    if not hasattr(tbc, "can_use_cudnn_attention"):
        pytest.skip("torch without can_use_cudnn_attention")
    seen = []
    monkeypatch.setattr(
        tbc, "can_use_cudnn_attention", lambda p, debug: seen.append(p.query.dtype) or True
    )
    assert att._run_cudnn_head_dim_probe("cpu", torch.float32, 128) is True
    assert seen == [torch.bfloat16]
