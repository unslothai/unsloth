# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit ``flash`` on ROCm: honored only when flash_attn imports and the per-device check passes; ``auto`` and every
NVIDIA path unchanged; the CUDA flash-attn wheel is never pip installed onto ROCm. Hermetic, CPU only."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import select_attention_backend

try:  # before any test puts a bare flash_attn module in sys.modules (diffusers' import probes its spec)
    importlib.import_module("diffusers.models.attention_dispatch")
except Exception:  # noqa: BLE001
    pass

_REAL_TOOLCHAIN_KEY = att._rocm_flash_toolchain_key
_REAL_PROBE_FITS = att._rocm_flash_probe_fits
_REAL_GUARD = att._install_rocm_flash_dispatch_guard


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(att, "_ROCM_FLASH_PROBE_CACHE", {}, raising = False)
    monkeypatch.setenv(att.ROCM_FLASH_CACHE_DIR_ENV, str(tmp_path / "rocm_flash"))
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: "tc-A", raising = False)
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d == "cuda" else d, raising = False
    )
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_ROCM_FLASH_LOCAL", __import__("threading").local(), raising = False)
    monkeypatch.setattr(att, "_ROCM_FLASH_BUCKETS", {}, raising = False)
    monkeypatch.setattr(att, "_ROCM_FLASH_STORED_MISSES", set(), raising = False)
    monkeypatch.setattr(att, "_ROCM_FLASH_BASELINE", [None], raising = False)
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: True, raising = False)
    monkeypatch.setattr(att, "_ROCM_FLASH_ROUTED", {})


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
    assert calls == [("cuda:0", "torch.bfloat16")]


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


def test_rocm_auto_is_a_timed_candidate_not_flash(monkeypatch):
    calls = _rocm(monkeypatch)
    assert (
        select_attention_backend(_target(), "auto", speed_active = True) == att.ROCM_FLASH_CANDIDATE
    )
    assert select_attention_backend(_target(), "auto", speed_active = False) is None
    assert calls == [("cuda:0", "torch.bfloat16")]


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


def _klein():
    return types.SimpleNamespace(name = "flux.2-klein")


@pytest.mark.parametrize("family", ["flux.2-klein", "flux.1", "z-image", "qwen-image", None])
def test_rocm_auto_candidate_for_every_family(monkeypatch, family):
    _rocm(monkeypatch)
    fam = None if family is None else types.SimpleNamespace(name = family)
    assert (
        select_attention_backend(_target(), "auto", speed_active = True, family = fam)
        == att.ROCM_FLASH_CANDIDATE
    )
    assert (
        select_attention_backend(_target(), None, speed_active = False, family = fam, speed_unset = True)
        == att.ROCM_FLASH_CANDIDATE
    )
    assert select_attention_backend(_target(), "auto", speed_active = False, family = fam) is None


def test_rocm_auto_without_flash_attn_is_silent(monkeypatch):
    calls = _rocm(monkeypatch, flash_attn = False)
    logged = []
    monkeypatch.setattr(
        att, "_module_logger", lambda: types.SimpleNamespace(warning = lambda *a: logged.append(a))
    )
    assert select_attention_backend(_target(), "auto", speed_active = True, family = _klein()) is None
    assert calls == [] and logged == []


def test_rocm_auto_failed_probe_stays_native(monkeypatch):
    _rocm(monkeypatch, probe = False)
    assert select_attention_backend(_target(), "auto", speed_active = True, family = _klein()) is None


def test_rocm_auto_explicit_native_wins(monkeypatch):
    calls = _rocm(monkeypatch)
    for alias in ("native", "sdpa"):
        assert (
            select_attention_backend(_target(), alias, speed_active = True, family = _klein()) is None
        )
    assert calls == []


def test_rocm_kill_switch_zero_keeps_auto_native(monkeypatch):
    calls = _rocm(monkeypatch)
    monkeypatch.setenv(att.ROCM_FLASH_ENV, "0")
    assert select_attention_backend(_target(), "auto", speed_active = True, family = _klein()) is None
    assert calls == []
    assert select_attention_backend(_target(), "flash", speed_active = True) == "flash"


def test_rocm_gfx_arch_reads_the_selected_card(monkeypatch):
    import torch

    asked = []

    def props(index):
        asked.append(index)
        return types.SimpleNamespace(gcnArchName = "gfx1100" if index == 1 else "gfx90a")

    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda:1", dtype = "torch.bfloat16")
    assert att._rocm_gfx_arch(target) == "gfx1100"
    assert asked == [1]


def _pipe_with_attention(
    hidden = 3072,
    heads = 24,
    head_dim = 128,
    n = 3,
):
    set_to = []
    mods = [
        types.SimpleNamespace(
            heads = heads,
            to_q = types.SimpleNamespace(in_features = hidden, out_features = heads * head_dim),
        )
        for _ in range(n)
    ]
    dit = types.SimpleNamespace(set_attention_backend = set_to.append, modules = lambda: iter(mods))
    return types.SimpleNamespace(transformer = dit), set_to


def _apply_env(monkeypatch):
    monkeypatch.setattr(att, "_active_attention_backend", lambda: att.ATTN_NATIVE)
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    monkeypatch.setattr(att, "_configure_native_attention", lambda *a, **k: None)
    monkeypatch.setattr(att, "ROCM_FLASH_ADMISSIONS", {})
    monkeypatch.setattr(att, "_install_rocm_flash_dispatch_guard", lambda: True, raising = False)


@pytest.mark.parametrize(
    "flash_ms, rel, engaged",
    [(70.0, 1e-3, "flash"), (78.0, 1e-3, None), (90.0, 1e-3, None), (60.0, 0.5, None)],
)
def test_apply_resolves_candidate_by_block_timing(monkeypatch, flash_ms, rel, engaged):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
    ):
        timed.append((device, hidden, heads, head_dim))
        return {
            "native_ms": 80.0,
            "flash_ms": flash_ms,
            "rel_l1": rel,
            "max_rel": rel,
            "finite": True,
            "tokens": 4096,
        }

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    pipe, set_to = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert timed == [("cuda:0", 3072, 24, 128)]
    assert set_to == (["flash"] if engaged else [])
    pipe2, _ = _pipe_with_attention()
    att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = _target())
    assert len(timed) == 1
    rec = next(iter(att.ROCM_FLASH_ADMISSIONS.values()))
    assert rec["admitted"] is (engaged == "flash") and rec["geometry"] == {
        "hidden": 3072,
        "heads": 24,
        "head_dim": 128,
    }


def test_apply_candidate_kill_switch_one_skips_timing(monkeypatch):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setenv(att.ROCM_FLASH_ENV, "1")
    monkeypatch.setattr(
        att, "_time_rocm_flash_block", lambda *a, **k: pytest.fail("timed under =1")
    )
    pipe, set_to = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"


def test_apply_candidate_probe_error_is_native_and_retried(monkeypatch):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    n = []

    def boom(*a, **k):
        n.append(1)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(att, "_time_rocm_flash_block", boom)
    pipe, set_to = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert len(n) == 2 and set_to == []


def test_apply_candidate_without_geometry_is_native(monkeypatch):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(
        att, "_time_rocm_flash_block", lambda *a, **k: pytest.fail("no geometry, nothing to time")
    )
    set_to = []
    dit = types.SimpleNamespace(set_attention_backend = set_to.append, modules = lambda: iter(()))
    pipe = types.SimpleNamespace(transformer = dit)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) is None


def test_candidate_never_pip_installs(monkeypatch):
    assert att._ensure_backend_package(att.ROCM_FLASH_CANDIDATE) is None


def _diffusers_flash(monkeypatch, flash_attn_func):
    """diffusers' real FLASH backend (unguarded, restored after the test) calling ``flash_attn_func``."""
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    backends = ad._AttentionBackendRegistry._backends
    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, ad._flash_attention)
    monkeypatch.setattr(ad, "flash_attn_func", flash_attn_func)
    return ad


def test_block_timer_runs_on_cpu_shapes(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p = 0.0,
        softmax_scale = None,
        causal = False,
        window_size = (-1, -1),
        return_attn_probs = False,
    ):
        seen.append((dropout_p, softmax_scale, causal, tuple(window_size), return_attn_probs))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    _diffusers_flash(monkeypatch, flash_attn_func)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    att._ROCM_FLASH_LOCAL.tally = {}
    rec = att._time_rocm_flash_block("cpu", torch.bfloat16, 64, 4, 16, tokens = 32)
    assert rec["finite"] and rec["rel_l1"] == 0.0 and rec["native_ms"] > 0 and rec["flash_ms"] > 0
    assert len(seen) == 9 and set(seen) == {(0.0, None, False, (-1, -1), False)}
    assert att._ROCM_FLASH_LOCAL.tally == {"flash": 9}


def test_flash_attn_rejecting_the_diffusers_arguments_is_not_admitted(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda d: "cpu")

    def old_api(q, k, v):
        return q

    _diffusers_flash(monkeypatch, old_api)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(att, "ROCM_FLASH_PROBE_TOKENS", 32)
    pipe, set_to = _pipe_with_attention(hidden = 64, heads = 4, head_dim = 16)
    with pytest.raises(TypeError):
        att._time_rocm_flash_block("cpu", torch.bfloat16, 64, 4, 16, tokens = 32)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert set_to == [] and att.ROCM_FLASH_ADMISSIONS == {} and _cache_files(tmp_path) == []


def test_nvidia_auto_klein_unchanged(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cudnn_attention_supported", lambda: True)
    assert (
        select_attention_backend(_target(), "auto", speed_active = True, family = _klein())
        == "_native_cudnn"
    )
    for unset in (False, True):
        assert (
            select_attention_backend(
                _target(), "auto", speed_active = False, family = _klein(), speed_unset = unset
            )
            is None
        )


def test_image_load_passes_the_family():
    import ast
    import pathlib

    src = pathlib.Path(att.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "select_attention_backend"
    ]
    with_family = [c for c in calls if any(k.arg == "family" for k in c.keywords)]
    assert len(with_family) == 2
    assert sum(any(k.arg == "speed_unset" for k in c.keywords) for c in with_family) == 1


def test_auto_attention_reason_names_the_engaged_kernel():
    assert att.auto_attention_reason(None) == "diffusers default"
    assert att.auto_attention_reason("_native_cudnn") == "cuDNN fused attention upgrade"
    assert "cuDNN" not in att.auto_attention_reason("flash")
    assert "ROCm flash" in att.auto_attention_reason("flash")


def test_status_reason_is_derived_from_the_engaged_backend():
    def read(name):
        return open(att.__file__.replace("diffusion_attention.py", name), encoding = "utf-8").read()

    image, video = read("diffusion.py"), read("video.py")
    assert image.count("attention_status_reason(attention_backend, attention_engaged)") == 1
    assert image.count("auto_attention_reason(attention_engaged)") == 1
    assert '"cuDNN fused attention upgrade"' not in image
    assert video.count("attention_status_reason(attention_backend, attention_engaged)") == 2
    assert "cuDNN fused attention on NVIDIA" not in video


# Masked attention (HunyuanVideo 1.5 mixed padding) must never reach diffusers' flash backend, which raises on
# attn_mask; and every attention geometry that receives the backend must pass the timing, not only the most common one.


def test_auto_flash_engages_only_with_the_mask_guard(monkeypatch):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(
        att,
        "_time_rocm_flash_block",
        lambda *a, **k: {
            "native_ms": 80.0,
            "flash_ms": 60.0,
            "rel_l1": 1e-3,
            "max_rel": 2e-3,
            "finite": True,
            "tokens": 4096,
        },
    )
    installed = []
    monkeypatch.setattr(
        att, "_install_rocm_flash_dispatch_guard", lambda: installed.append(1) or True
    )
    pipe, set_to = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    assert installed and set_to == ["flash"]
    # a diffusers build whose registry cannot be guarded: auto stays native rather than risk a masked call
    monkeypatch.setattr(att, "_install_rocm_flash_dispatch_guard", lambda: False)
    pipe2, set_to2 = _pipe_with_attention()
    assert att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert set_to2 == []
    pipe3, set_to3 = _pipe_with_attention()
    assert att.apply_attention_backend(pipe3, "flash", target = _target()) is None
    assert set_to3 == []


def test_rocm_flash_guard_routes_masked_calls_to_native(monkeypatch):
    torch = pytest.importorskip("torch")
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    backends = ad._AttentionBackendRegistry._backends
    calls = []

    def fake_flash(
        query,
        key,
        value,
        attn_mask = None,
        **kw,
    ):
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not supported for flash-attn 2.")
        calls.append("flash")
        return query

    def fake_native(
        query,
        key,
        value,
        attn_mask = None,
        **kw,
    ):
        calls.append("native")
        return query

    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, fake_flash)
    monkeypatch.setitem(backends, ad.AttentionBackendName.NATIVE, fake_native)
    assert att._install_rocm_flash_dispatch_guard() is True
    guarded = backends[ad.AttentionBackendName.FLASH]
    q = torch.zeros(1, 8, 2, 16, dtype = torch.bfloat16)
    mask = torch.ones(1, 1, 8, 8, dtype = torch.bool)
    guarded(q, q, q, attn_mask = mask)
    guarded(q, q, q, None)
    assert calls == ["native", "flash"]
    assert att._install_rocm_flash_dispatch_guard() is True
    assert backends[ad.AttentionBackendName.FLASH] is guarded


def _two_geometry_pipe():
    set_to = []

    def dit(hidden, heads, head_dim, n):
        mods = [
            types.SimpleNamespace(
                heads = heads,
                to_q = types.SimpleNamespace(in_features = hidden, out_features = heads * head_dim),
            )
            for _ in range(n)
        ]
        return types.SimpleNamespace(
            set_attention_backend = set_to.append, modules = lambda: iter(mods)
        )

    first, second = dit(3072, 24, 128, 5), dit(1536, 12, 128, 1)
    monkey_dits = [first, second]
    return types.SimpleNamespace(transformer = first, transformer_2 = second), set_to, monkey_dits


@pytest.mark.parametrize("minority_wins, engaged", [(True, "flash"), (False, None)])
def test_every_attention_geometry_is_timed(monkeypatch, minority_wins, engaged):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    pipe, set_to, dits = _two_geometry_pipe()
    monkeypatch.setattr(att, "_attention_dits", lambda p: dits)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
    ):
        timed.append((hidden, heads, head_dim))
        flash = 60.0 if (hidden == 3072 or minority_wins) else 85.0
        return {
            "native_ms": 80.0,
            "flash_ms": flash,
            "rel_l1": 1e-3,
            "max_rel": 2e-3,
            "finite": True,
            "tokens": 4096,
        }

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert (1536, 12, 128) in timed
    if engaged:
        assert sorted(timed) == [(1536, 12, 128), (3072, 24, 128)]
        assert set_to == ["flash", "flash"] and len(att.ROCM_FLASH_ADMISSIONS) == 2
    else:
        assert set_to == []
        n = len(timed)
        pipe2, set_to2, dits2 = _two_geometry_pipe()
        monkeypatch.setattr(att, "_attention_dits", lambda p: dits2)
        assert (
            att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
        )
        assert len(timed) == n


def test_num_heads_attention_modules_are_measured():
    # Krea 2's attention exposes num_heads, not heads: without the fallback auto could never time it.
    import types

    from torch import nn

    class Krea2LikeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 24
            self.to_q = nn.Linear(3072, 3072, bias = False)

    dit = nn.Module()
    dit.attn = Krea2LikeAttention()
    pipe = types.SimpleNamespace()
    original = att._attention_dits
    try:
        att._attention_dits = lambda p: [dit]
        assert att._dit_attention_geometries(pipe) == ((3072, 24, 128),)
    finally:
        att._attention_dits = original


def _timer(timed, flash_ms = 70.0):
    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
    ):
        timed.append((hidden, heads, head_dim))
        return {
            "native_ms": 80.0,
            "flash_ms": flash_ms,
            "rel_l1": 1e-3,
            "max_rel": 2e-3,
            "finite": True,
            "tokens": 4096,
        }

    return fake_time


def _restart(monkeypatch):
    """A new process: the in-memory verdicts are gone, the cache dir is not."""
    monkeypatch.setattr(att, "ROCM_FLASH_ADMISSIONS", {})
    monkeypatch.setattr(att, "_ROCM_FLASH_PROBE_CACHE", {})


def _cache_files(tmp_path):
    root = tmp_path / "rocm_flash"
    return sorted(root.glob("probe_*.json")) if root.is_dir() else []


@pytest.mark.parametrize("flash_ms, engaged", [(70.0, "flash"), (90.0, None)])
def test_block_verdict_persists_across_processes(monkeypatch, tmp_path, flash_ms, engaged):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed, flash_ms))
    pipe, _ = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert len(timed) == 1 and len(_cache_files(tmp_path)) == 1
    _restart(monkeypatch)
    monkeypatch.setattr(
        att,
        "_time_rocm_flash_block",
        lambda *a, **k: pytest.fail("re-timed despite a stored verdict"),
    )
    pipe2, set_to2 = _pipe_with_attention()
    assert att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert set_to2 == (["flash"] if engaged else [])
    rec = next(iter(att.ROCM_FLASH_ADMISSIONS.values()))
    assert rec["cached"] is True and rec["flash_ms"] == flash_ms


@pytest.mark.parametrize("change", ["torch", "flash_attn", "probe_version", "geometry", "dtype"])
def test_block_verdict_key_mismatch_retimes(monkeypatch, tmp_path, change):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    toolchain = {"torch": "torch2.9", "flash_attn": "flash_attn2.8.3"}
    monkeypatch.setattr(
        att,
        "_rocm_flash_toolchain_key",
        lambda device: f"v{att.ROCM_FLASH_PROBE_VERSION}|{toolchain['torch']}|{toolchain['flash_attn']}",
    )
    pipe, _ = _pipe_with_attention()
    att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target())
    assert len(timed) == 1
    _restart(monkeypatch)
    target, geometry = _target(), {}
    if change == "torch":
        toolchain["torch"] = "torch2.10"
    elif change == "flash_attn":
        toolchain["flash_attn"] = "flash_attn2.8.4"
    elif change == "probe_version":
        monkeypatch.setattr(att, "ROCM_FLASH_PROBE_VERSION", att.ROCM_FLASH_PROBE_VERSION + 1)
    elif change == "geometry":
        geometry = {"hidden": 2048, "heads": 16, "head_dim": 128}
    else:
        target = _target(dtype = "torch.float16")
    pipe2, _ = _pipe_with_attention(**geometry)
    att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = target)
    assert len(timed) == 2


@pytest.mark.parametrize(
    "payload", ["{not json", "", "[]", '{"key": "tc-A", "verdicts": {"x": 1}}']
)
def test_corrupt_cache_file_retimes(monkeypatch, tmp_path, payload):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    pipe, _ = _pipe_with_attention()
    att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target())
    (path,) = _cache_files(tmp_path)
    path.write_text(payload)
    _restart(monkeypatch)
    pipe2, _ = _pipe_with_attention()
    assert att.apply_attention_backend(pipe2, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    assert len(timed) == 2
    _restart(monkeypatch)
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 2


def test_probe_error_is_not_persisted(monkeypatch, tmp_path):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(att, "_time_rocm_flash_block", boom)
    pipe, _ = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert _cache_files(tmp_path) == []


def test_unidentifiable_toolchain_stays_in_memory(monkeypatch, tmp_path):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: None)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    pipe, _ = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    assert _cache_files(tmp_path) == [] and not (tmp_path / "rocm_flash").exists()


def test_unwritable_cache_dir_still_admits(monkeypatch, tmp_path):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(att.ROCM_FLASH_CACHE_DIR_ENV, str(blocker / "sub"))
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    pipe, _ = _pipe_with_attention()
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"


@pytest.mark.parametrize("mode, engaged", [("0", None), ("1", "flash")])
def test_forced_modes_neither_read_nor_write_the_cache(monkeypatch, tmp_path, mode, engaged):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    # a stored verdict that would say the opposite of the forced mode
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed, 70.0 if mode == "0" else 90.0))
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    (path,) = _cache_files(tmp_path)
    before = path.read_bytes()
    _restart(monkeypatch)
    monkeypatch.setenv(att.ROCM_FLASH_ENV, mode)
    monkeypatch.setattr(att, "_load_rocm_flash_verdicts", lambda *a: pytest.fail("cache read"))
    monkeypatch.setattr(att, "_save_rocm_flash_verdict", lambda *a: pytest.fail("cache write"))
    monkeypatch.setattr(att, "_time_rocm_flash_block", lambda *a, **k: pytest.fail("timed"))
    pipe, _ = _pipe_with_attention(hidden = 2048, heads = 16)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert path.read_bytes() == before and att.ROCM_FLASH_ADMISSIONS == {}


def test_cache_dir_defaults_under_the_studio_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(att.ROCM_FLASH_CACHE_DIR_ENV)
    import utils.paths.storage_roots as roots

    monkeypatch.setattr(roots, "cache_root", lambda: tmp_path / "cache")
    path = att._rocm_flash_cache_path("tc-A")
    assert path.startswith(str(tmp_path / "cache" / "diffusion_rocm_flash" / "probe_"))
    assert att._rocm_flash_cache_path("tc-B") != path and att._rocm_flash_cache_path(None) is None


def test_toolchain_key_names_the_flash_attn_build(monkeypatch):
    pytest.importorskip("diffusers.models.attention_dispatch")
    import torch

    props = types.SimpleNamespace(name = "AMD Radeon 8060S", gcnArchName = "gfx1151:sramecc-:xnack-")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising = False)
    fa = types.ModuleType("flash_attn")
    fa.__version__ = "2.8.3"
    fa.__file__ = "/x/site-packages/flash_attn/__init__.py"
    monkeypatch.setitem(sys.modules, "flash_attn", fa)
    # A real flash_attn build already imported (a ROCm host) would answer the backend through its USE_TRITON_ROCM.
    monkeypatch.delitem(sys.modules, "flash_attn.flash_attn_interface", raising = False)
    monkeypatch.delenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", raising = False)
    ck = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert ck is not None and ck.startswith(f"v{att.ROCM_FLASH_PROBE_VERSION}|")
    for part in (
        "AMD Radeon 8060S",
        "|gfx1151|",
        "|hip7.1.0|",
        "|flash_attn2.8.3|",
        os.sep + "flash_attn|ck|",
    ):
        assert part in ck, part
    fa.__version__ = "2.8.4"
    assert _REAL_TOOLCHAIN_KEY("cuda:0") != ck
    fa.__version__ = "2.8.3"
    monkeypatch.setattr(att, "ROCM_FLASH_PROBE_VERSION", att.ROCM_FLASH_PROBE_VERSION + 1)
    assert _REAL_TOOLCHAIN_KEY("cuda:0") != ck
    monkeypatch.setattr(att, "ROCM_FLASH_PROBE_VERSION", att.ROCM_FLASH_PROBE_VERSION - 1)
    assert _REAL_TOOLCHAIN_KEY("cuda:0") == ck
    pytest.importorskip("triton")
    monkeypatch.setenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    assert "|triton" in _REAL_TOOLCHAIN_KEY("cuda:0")


# A flash_attn rebuilt in place (same version, same path) must re-time: the key fingerprints the loaded kernel files.


def _fake_build(
    monkeypatch,
    tmp_path,
    *,
    triton = False,
):
    import torch

    props = types.SimpleNamespace(name = "AMD Radeon 8060S", gcnArchName = "gfx1151")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising = False)
    pkg = tmp_path / "site" / "flash_attn"
    pkg.mkdir(parents = True)
    fa = types.ModuleType("flash_attn")
    fa.__version__ = "2.8.3"
    fa.__file__ = str(pkg / "__init__.py")
    iface = types.ModuleType("flash_attn.flash_attn_interface")
    iface.__file__ = str(pkg / "flash_attn_interface.py")
    iface.USE_TRITON_ROCM = triton
    for f in (fa.__file__, iface.__file__):
        open(f, "w").write("# py\n")
    if triton:
        kdir = tmp_path / "site" / "triton_amd"
        kdir.mkdir()
        (kdir / "fwd_prefill.py").write_text("# kernel v1\n")
        binary = kdir / "flash_attn_2.py"
    else:
        binary = tmp_path / "site" / "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    binary.write_bytes(b"\x7fELF v1")
    gpu = types.ModuleType("flash_attn_gpu")
    gpu.__file__ = str(binary)
    iface.flash_attn_gpu = gpu
    monkeypatch.setitem(sys.modules, "flash_attn", fa)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", iface)
    if triton:
        pytest.importorskip("triton")
        return kdir / "fwd_prefill.py"
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", gpu)
    return binary


@pytest.mark.parametrize("triton", [False, True])
@pytest.mark.parametrize("how", ["content", "mtime"])
def test_rebuilt_flash_attn_binary_changes_the_key(monkeypatch, tmp_path, triton, how):
    pytest.importorskip("diffusers.models.attention_dispatch")
    import os

    kernel = _fake_build(monkeypatch, tmp_path, triton = triton)
    key = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert key is not None and ("|triton" in key) is triton and _REAL_TOOLCHAIN_KEY("cuda:0") == key
    st = os.stat(kernel)
    if how == "content":
        kernel.write_bytes(kernel.read_bytes() + b" rebuilt")
        os.utime(kernel, ns = (st.st_atime_ns, st.st_mtime_ns))
    else:
        os.utime(kernel, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert _REAL_TOOLCHAIN_KEY("cuda:0") != key


def test_rebuilt_flash_attn_binary_forces_retiming(monkeypatch, tmp_path):
    pytest.importorskip("diffusers.models.attention_dispatch")
    import os

    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    binary = _fake_build(
        monkeypatch, tmp_path
    )  # after _rocm, which installs a bare flash_attn module
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", _REAL_TOOLCHAIN_KEY)
    assert _REAL_TOOLCHAIN_KEY("cuda:0") is not None
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    _restart(monkeypatch)
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 1
    binary.write_bytes(b"\x7fELF v2 rebuilt")
    st = os.stat(binary)
    os.utime(binary, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _restart(monkeypatch)
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 2


# Malformed cached records are a cache miss (re-timed), never a verdict, and never break the verdict log.

_GOOD = {
    "native_ms": 80.0,
    "flash_ms": 70.0,
    "rel_l1": 1e-3,
    "max_rel": 2e-3,
    "finite": True,
    "tokens": 4096,
}


@pytest.mark.parametrize(
    "field, value",
    [
        ("native_ms", "80"),
        ("flash_ms", "70"),
        ("rel_l1", "1e-3"),
        ("finite", "true"),
        ("finite", 1),
        ("tokens", "4096"),
        ("tokens", 4096.5),
        ("tokens", True),
        ("native_ms", {"ms": 80}),
        ("flash_ms", [70.0]),
        ("rel_l1", None),
        ("native_ms", float("nan")),
        ("flash_ms", float("inf")),
        ("rel_l1", float("nan")),
        ("native_ms", -80.0),
        ("flash_ms", 0.0),
        ("rel_l1", -1e-3),
        ("tokens", -4096),
        ("admitted", "yes"),
        ("geometry", {"hidden": "3072", "heads": 24, "head_dim": 128}),
        ("geometry", [3072, 24, 128]),
        ("native_ms", True),
        ("drop", "flash_ms"),
        ("drop", "max_rel"),
        ("max_rel", "2e-3"),
        ("max_rel", float("inf")),
        ("max_rel", -1.0),
        ("block_rel_l1", "x"),
    ],
)
def test_malformed_cached_record_is_retimed(monkeypatch, tmp_path, field, value):
    import json

    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed, 90.0))
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    (path,) = _cache_files(tmp_path)
    data = json.loads(path.read_text())
    (slot,) = data["verdicts"]
    rec = dict(_GOOD)  # a stored win: a malformed copy of it must not admit flash
    if field == "drop":
        rec.pop(value)
    else:
        rec[field] = value
    data["verdicts"][slot] = rec
    path.write_text(json.dumps(data))
    _restart(monkeypatch)
    logs = []

    class Log:
        def info(self, *a):
            logs.append(a[0] % a[1:])

    assert (
        att.apply_attention_backend(
            _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, logger = Log(), target = _target()
        )
        is None
    )
    assert len(timed) == 2 and any("probe" in line and "-> native" in line for line in logs)


def test_non_finite_probe_verdict_is_stored_and_reused(monkeypatch, tmp_path):
    # a flash output with NaN is a real (native) answer: stored, read back, never re-timed and never raising
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []

    def nan_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
    ):
        timed.append(1)
        return {**_GOOD, "finite": False, "rel_l1": float("nan"), "max_rel": float("nan")}

    monkeypatch.setattr(att, "_time_rocm_flash_block", nan_time)
    assert (
        att.apply_attention_backend(
            _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
        )
        is None
    )
    _restart(monkeypatch)
    assert (
        att.apply_attention_backend(
            _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
        )
        is None
    )
    assert len(timed) == 1


def test_implausible_probe_timing_is_native_and_not_stored(monkeypatch, tmp_path):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(
        att, "_time_rocm_flash_block", lambda *a, **k: {**_GOOD, "native_ms": 0.0, "flash_ms": 0.0}
    )
    assert (
        att.apply_attention_backend(
            _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
        )
        is None
    )
    assert _cache_files(tmp_path) == []


@pytest.mark.parametrize(
    "rec",
    [
        {},
        {"flash_ms": "x", "native_ms": {"a": 1}, "rel_l1": [1], "tokens": "n", "admitted": "yes"},
        {**_GOOD, "flash_ms": float("nan"), "rel_l1": float("inf"), "admitted": True},
    ],
)
def test_verdict_log_never_raises(rec):
    class Log:
        def info(self, fmt, *a):
            fmt % a

    att._log_rocm_flash_verdict(Log(), "cuda:0", (3072, 24, 128), rec, "cached")
    att._log_rocm_flash_verdict(Log(), "cuda:0", (), rec, "cached")


# Accuracy is judged on the raw attention output: a wrong attention the residual + FFN would hide is still rejected.


def test_wrong_attention_is_rejected_though_the_block_would_pass(monkeypatch):
    torch = pytest.importorskip("torch")

    def off_by_20pct(q, k, v, **kw):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return out * 1.2

    _diffusers_flash(monkeypatch, off_by_20pct)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    # inner == hidden as in a real DiT; at 1024 tokens the block output moves only ~1.7%
    rec = att._time_rocm_flash_block("cpu", torch.bfloat16, 256, 4, 64, tokens = 1024)
    # the old block-level check would have admitted it
    assert rec["block_rel_l1"] <= att.ROCM_FLASH_BLOCK_TOL
    assert rec["rel_l1"] > 0.15 and rec["max_rel"] > 0.1
    assert att._rocm_flash_record_ok(rec) and not att._rocm_flash_wins(
        {**rec, "native_ms": 80.0, "flash_ms": 40.0}
    )


@pytest.mark.parametrize("max_rel, wins", [(1e-2, True), (0.2, False)])
def test_max_abs_error_alone_rejects(max_rel, wins):
    assert att._rocm_flash_wins({**_GOOD, "max_rel": max_rel}) is wins


# Lengths: calls are gated per power-of-two length bucket; an unseen bucket is timed on its first eager call and never
# inherits the 4096 verdict.


@pytest.mark.parametrize(
    "tokens, bucket",
    [
        (1, 256),
        (300, 256),
        (900, 512),
        (4096, 4096),
        (4608, 4096),
        (8191, 4096),
        (27280, 16384),
        (50000, 32768),
        (140000, 131072),
    ],
)
def test_length_buckets(tokens, bucket):
    assert att._rocm_flash_bucket(tokens) == bucket


def _length_env(monkeypatch, verdicts):
    """An auto load admitted at 4096; ``verdicts[tokens]`` is the flash_ms (native 80) the bucket probe measures."""
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
        kv_tokens = None,
        ctx_dim = None,
        batch = 1,
    ):
        size = tokens if kv_tokens is None else (tokens, kv_tokens)
        if batch != 1:
            size = (size, f"b{batch}")
        if ctx_dim is not None:
            size = (size, f"ctx{ctx_dim}")
        timed.append(size)
        return {
            **_GOOD,
            "flash_ms": verdicts[size],
            "tokens": tokens,
            **({} if kv_tokens is None else {"kv_tokens": kv_tokens}),
            **({} if batch == 1 else {"batch": batch}),
        }

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    dit = _Dit()
    assert (
        att.apply_attention_backend(
            types.SimpleNamespace(transformer = dit), att.ROCM_FLASH_CANDIDATE, target = _target()
        )
        == "flash"
    )
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d in ("cuda", "cpu") else d
    )
    return dit, guarded, ran, timed


def test_unprobed_long_bucket_does_not_inherit_the_4096_verdict(monkeypatch):
    dit, guarded, ran, timed = _length_env(
        monkeypatch, {4096: 70.0, 16384: 90.0, (256, 1024): 70.0}
    )
    assert timed == [4096]
    dit.run(guarded, [(4608, None), (27280, None), (20000, None), (16, 1200)])
    assert timed == [4096, 16384, (256, 1024)]
    assert ran == ["flash", "native", "native", "flash"]
    assert att._ROCM_FLASH_ROUTED == {"slower or inexact at this size": 2}


def test_50k_call_runs_flash_only_if_its_own_bucket_admitted_it(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 90.0, 65536: 60.0})
    dit.run(guarded, [(50000, None), (70000, None), (140000, None)])
    assert timed == [4096, 32768, 65536, 131072]
    assert ran == ["native", "flash", "native"]


def test_bucket_probe_error_stays_native_without_retrying_every_call(monkeypatch):
    verdicts = {4096: 70.0}
    dit, guarded, ran, timed = _length_env(monkeypatch, verdicts)
    dit.run(guarded, [(40000, None), (40000, None)])
    assert timed == [4096, 32768] and ran == ["native", "native"]
    deferred = getattr(dit, att.ROCM_FLASH_DEFERRED_ATTR)
    assert len(deferred) == 1 and next(iter(deferred)) not in att._ROCM_FLASH_BUCKETS
    verdicts[32768] = 60.0
    att.begin_attention_run(types.SimpleNamespace(pipe = types.SimpleNamespace(transformer = dit)))
    dit.run(guarded, [(40000, None)])
    assert timed == [4096, 32768, 32768] and ran == ["native", "native", "flash"]


# An in-forward probe sits on top of the live request's tensors; no room for it defers, never rejects.


def test_bucket_probe_without_room_beside_the_forward_is_deferred_not_rejected(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 60.0})
    needs = []
    monkeypatch.setattr(
        att, "_rocm_flash_probe_fits", lambda device, need: needs.append(need) or False
    )
    dit.run(guarded, [(40000, None), (40000, None)])
    assert timed == [4096] and ran == ["native", "native"] and len(needs) == 1
    assert att._ROCM_FLASH_BUCKETS == {} and len(getattr(dit, att.ROCM_FLASH_DEFERRED_ATTR)) == 1
    assert att._ROCM_FLASH_ROUTED == {"timing deferred": 2}
    assert not any(k[-1] == 32768 or 32768 in k for k in att.ROCM_FLASH_ADMISSIONS)
    # Room again at the next generation: timed and admitted.
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: True)
    att.begin_attention_run(types.SimpleNamespace(pipe = types.SimpleNamespace(transformer = dit)))
    dit.run(guarded, [(40000, None)])
    assert timed == [4096, 32768] and ran[-1] == "flash"


def test_deferral_belongs_to_its_dit_not_to_other_generations(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 60.0})
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: False)
    dit.run(guarded, [(40000, None)])
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: True)
    other = _Dit()
    att.apply_attention_backend(
        types.SimpleNamespace(transformer = other), att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    att.begin_attention_run(types.SimpleNamespace(pipe = types.SimpleNamespace(transformer = other)))
    dit.run(guarded, [(40000, None)])
    assert timed == [4096] and ran == ["native", "native"]
    att.begin_attention_run(types.SimpleNamespace(pipe = types.SimpleNamespace(transformer = dit)))
    dit.run(guarded, [(40000, None)])
    assert timed == [4096, 32768] and ran[-1] == "flash"


def test_cfg_replica_retries_its_deferred_bucket_at_the_next_generation(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 60.0})
    replica = _Dit()
    att.track_rocm_flash_replica(dit, replica, "cpu")
    pipe = types.SimpleNamespace(transformer = dit)
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: False)
    att.begin_attention_run(types.SimpleNamespace(pipe = pipe))
    replica.run(guarded, [(40000, None)])
    assert timed == [4096] and ran == ["native"]
    assert len(getattr(replica, att.ROCM_FLASH_DEFERRED_ATTR)) == 1
    # Room on the replica card at the next generation: the same (kept) replica times the bucket again.
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: True)
    att.begin_attention_run(types.SimpleNamespace(pipe = pipe))
    replica.run(guarded, [(40000, None)])
    assert timed == [4096, 32768] and ran[-1] == "flash"


def test_cfg_replica_routes_count_in_the_generation_report(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 90.0})
    replica = _Dit()
    att.track_rocm_flash_replica(dit, replica, "cpu")
    pipe = types.SimpleNamespace(transformer = dit)
    att.begin_attention_run(types.SimpleNamespace(pipe = pipe))
    dit.run(guarded, [(5000, None)])
    replica.run(guarded, [(40000, None)])
    assert ran == ["flash", "native"]
    value, reason = att.rocm_flash_ran(pipe)
    assert value == "flash" and "on 1 of 2 attention calls" in reason
    att.begin_attention_run(types.SimpleNamespace(pipe = pipe))
    replica.run(guarded, [(40000, None)])
    assert att.rocm_flash_ran(pipe)[0] == "native"


def test_a_hooked_replica_is_freed_without_a_cycle_collection():
    import gc
    import weakref

    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 4)
    att._hook_rocm_flash_dit(module)
    assert getattr(module, att._ROCM_FLASH_HOOKS_ATTR)
    ref = weakref.ref(module)
    enabled = gc.isenabled()
    gc.disable()
    try:
        del module
        assert (
            ref() is None
        )  # no hook -> module cycle: a dropped CFG replica frees its card memory at once
    finally:
        if enabled:
            gc.enable()


def test_traced_segments_read_the_stored_verdicts_once_per_absent_bucket(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    loads = []
    real_load = att._load_rocm_flash_verdicts
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: "tc")
    monkeypatch.setattr(att, "_rocm_flash_cache_path", lambda tc: str(tmp_path / "probe.json"))
    monkeypatch.setattr(
        att, "_load_rocm_flash_verdicts", lambda *a: loads.append(a) or real_load(*a)
    )
    g = ((3072, 24, 128),)
    args = ("cuda:0", torch.bfloat16, g, 4096, 4096, 1)
    assert att._rocm_flash_bucket_known(*args) is None
    first = len(loads)
    assert first == 1
    assert att._rocm_flash_bucket_known(*args) is None
    assert len(loads) == first  # an absent slot is not re-read on every traced call


def test_a_moved_sdpa_baseline_drops_the_in_memory_verdicts(monkeypatch):
    baseline = ["sdpafem,aotriton-exp1"]
    monkeypatch.setattr(att, "_native_sdpa_baseline", lambda: baseline[0])
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 16384: 60.0})
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: "tc-" + baseline[0])
    dit.run(guarded, [(20000, None), (20000, None)])
    assert timed == [4096, 16384] and ran == ["flash", "flash"]
    baseline[0] = "sdpa--m,aotriton-exp1"
    dit.run(guarded, [(20000, None)])
    assert timed == [4096, 16384, 16384] and len(att.ROCM_FLASH_ADMISSIONS) == 1
    dit.run(guarded, [(20000, None)])
    assert timed == [4096, 16384, 16384]


def test_probe_bound_adds_math_sdpa_scores_when_fused_sdpa_is_off(monkeypatch):
    torch = pytest.importorskip("torch")
    g = (3072, 24, 128)
    monkeypatch.setattr(att, "_native_sdpa_fused_serves", lambda *a: True)
    fused = att._rocm_flash_probe_bytes(g, 16384, 16384, 1, torch.bfloat16)
    monkeypatch.setattr(att, "_native_sdpa_fused_serves", lambda *a: False)
    math = att._rocm_flash_probe_bytes(g, 16384, 16384, 1, torch.bfloat16)
    assert math - fused == 2 * 24 * 16384 * 16384 * 4


def test_fused_sdpa_eligibility_is_asked_for_the_probe_shape(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []

    def can_flash(params, debug):
        seen.append((tuple(params.query.shape), tuple(params.key.shape)))
        return params.query.shape[2] < 32768

    monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", can_flash)
    monkeypatch.setattr(torch.backends.cuda, "can_use_efficient_attention", lambda p, d: False)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    g = (3072, 24, 128)
    short = att._rocm_flash_probe_bytes(g, 16384, 16384, 1, torch.bfloat16, "cpu")
    long = att._rocm_flash_probe_bytes(g, 32768, 512, 2, torch.bfloat16, "cpu")
    assert seen == [
        ((1, 24, 16384, 128), (1, 24, 16384, 128)),
        ((2, 24, 32768, 128), (2, 24, 512, 128)),
    ]
    assert short < 4 * 2**30
    assert long > 2 * 2 * 24 * 32768 * 512 * 4
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)
    assert (
        att._rocm_flash_probe_bytes(g, 16384, 16384, 1, torch.bfloat16, "cpu") - short
        == 2 * 24 * 16384**2 * 4
    )


def test_probe_tensors_are_drawn_on_the_device():
    import inspect
    src = inspect.getsource(att._time_rocm_flash_block)
    assert ".to(device, dtype)" not in src and "torch.Generator(device = device)" in src


def test_native_sdpa_baseline_keys_the_verdicts(monkeypatch, tmp_path):
    pytest.importorskip("diffusers.models.attention_dispatch")
    torch = pytest.importorskip("torch")
    _fake_build(monkeypatch, tmp_path)
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", _REAL_TOOLCHAIN_KEY)
    flags = {"flash": True, "mem": True}
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: flags["flash"])
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: flags["mem"])
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: True)
    monkeypatch.setenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    fused = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert fused is not None and "|sdpafem,aotriton-exp1|" in fused
    flags.update(flash = False, mem = False)
    math_only = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert math_only != fused and "|sdpa--m,aotriton-exp1|" in math_only
    flags.update(flash = True, mem = True)
    monkeypatch.delenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    assert _REAL_TOOLCHAIN_KEY("cuda:0") not in (fused, math_only)


def test_probe_byte_bound_covers_the_measured_peak(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(att, "_native_sdpa_fused_serves", lambda *a: True)
    for tokens, peak in ((1024, 0.31), (65536, 5.46)):
        need = (
            att._rocm_flash_probe_bytes((3072, 24, 128), tokens, tokens, 1, torch.bfloat16) / 2**30
        )
        assert peak <= need < 2 * peak


def test_probe_fits_counts_cached_blocks_and_fails_closed(monkeypatch):
    torch = pytest.importorskip("torch")
    gib = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (2 * gib, 80 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 10 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 7 * gib)
    assert _REAL_PROBE_FITS("cuda:0", 3 * gib)
    assert not _REAL_PROBE_FITS("cuda:0", 4 * gib)

    def boom(d):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    assert not _REAL_PROBE_FITS("cuda:0", 1)


def test_unprobed_bucket_under_capture_runs_native_untimed(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 8192: 70.0})
    monkeypatch.setattr(att, "_rocm_flash_capturing", lambda: True)
    dit.run(guarded, [(9000, None)])
    assert timed == [4096] and ran == ["native"]
    monkeypatch.setattr(att, "_rocm_flash_capturing", lambda: False)
    dit.run(guarded, [(9000, None)])
    assert timed == [4096, 8192] and ran == ["native", "flash"]


def test_bucket_verdicts_persist(monkeypatch, tmp_path):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 16384: 60.0})
    dit.run(guarded, [(20000, None)])
    assert timed == [4096, 16384]
    _restart(monkeypatch)
    monkeypatch.setattr(att, "_ROCM_FLASH_BUCKETS", {})
    dit2, guarded2, ran2, timed2 = _length_env(monkeypatch, {})
    dit2.run(guarded2, [(20000, None)])
    assert timed2 == [] and ran2 == ["flash"]


@pytest.mark.parametrize("order", ["auto_first", "explicit_first"])
@pytest.mark.parametrize("explicit", ["flash", "mode1"])
def test_explicit_flash_is_never_length_gated(monkeypatch, order, explicit):
    # an auto model and an explicitly selected one share (heads, head_dim): only the auto one's calls are gated
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
    ):
        timed.append(tokens)
        return {**_GOOD, "flash_ms": 70.0 if tokens == 4096 else 90.0, "tokens": tokens}

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    auto_dit, explicit_dit = _Dit(), _Dit()

    def load_auto():
        assert (
            att.apply_attention_backend(
                types.SimpleNamespace(transformer = auto_dit),
                att.ROCM_FLASH_CANDIDATE,
                target = _target(),
            )
            == "flash"
        )

    def load_explicit():
        if explicit == "flash":
            backend = att.apply_attention_backend(
                types.SimpleNamespace(transformer = explicit_dit), "flash", target = _target()
            )
        else:
            monkeypatch.setenv(att.ROCM_FLASH_ENV, "1")
            backend = att.apply_attention_backend(
                types.SimpleNamespace(transformer = explicit_dit),
                att.ROCM_FLASH_CANDIDATE,
                target = _target(),
            )
            monkeypatch.delenv(att.ROCM_FLASH_ENV)
        assert backend == "flash"

    for load in (load_auto, load_explicit) if order == "auto_first" else (load_explicit, load_auto):
        load()
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d in ("cuda", "cpu") else d
    )
    explicit_dit.run(guarded, [(40000, None)])
    assert ran == ["flash"] and timed == [4096]
    auto_dit.run(guarded, [(40000, None)])
    assert ran == ["flash", "native"] and timed == [4096, 32768]
    explicit_dit.run(guarded, [(40000, None)])
    assert ran[-1] == "flash" and getattr(att._ROCM_FLASH_LOCAL, "gate", None) is None


def test_reapplying_explicit_flash_disarms_the_gate(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, 32768: 90.0})
    assert (
        att.apply_attention_backend(
            types.SimpleNamespace(transformer = dit), "flash", target = _target()
        )
        == "flash"
    )
    dit.run(guarded, [(40000, None)])
    assert ran == ["flash"] and timed == [4096]


def test_failed_forward_does_not_leave_its_gate_active(monkeypatch):
    torch = pytest.importorskip("torch")
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))

    class Boom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.heads = 2
            self.attn.to_q = torch.nn.Linear(32, 32, bias = False)

        def set_attention_backend(self, name):
            pass

        def forward(self, x):
            raise RuntimeError("cancelled")

    dit = Boom()
    att.apply_attention_backend(
        types.SimpleNamespace(transformer = dit), att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    with pytest.raises(RuntimeError):
        dit(torch.zeros(1))
    assert getattr(att._ROCM_FLASH_LOCAL, "gate", None) is None


class _Dit:
    """A DiT whose forward makes attention calls (masked or not) through the guarded FLASH backend."""

    def __init__(self):
        torch = pytest.importorskip("torch")
        self.inner = torch.nn.Module()
        self.backends = []
        self.inner.heads = 2
        self.inner.to_q = torch.nn.Linear(32, 32, bias = False)
        self.hooks = {"pre": [], "post": []}

    def modules(self):
        return iter([self.inner])

    def set_attention_backend(self, name):
        self.backends.append(name)

    def register_forward_pre_hook(self, fn):
        self.hooks["pre"].append(fn)
        return fn

    def register_forward_hook(
        self,
        fn,
        always_call = False,
    ):
        self.hooks["post"].append(fn)
        return fn

    def run(self, guarded, specs):
        """One forward making a call per ``(q_tokens, kv_tokens or None)``, unmasked."""
        import torch

        for fn in self.hooks["pre"]:
            fn(self, ())
        for tokens, kv in specs:
            q = torch.zeros(1, tokens, 2, 16, dtype = torch.bfloat16)
            k = torch.zeros(1, kv or tokens, 2, 16, dtype = torch.bfloat16)
            guarded(q, k, k, None)
        for fn in self.hooks["post"]:
            fn(self, (), None)

    def __call__(self, guarded, masks):
        import torch

        for fn in self.hooks["pre"]:
            fn(self, ())
        q = torch.zeros(1, 4096, 2, 16, dtype = torch.bfloat16)
        for masked in masks:
            guarded(q, q, q, attn_mask = torch.ones(1, 1, 8, 8, dtype = torch.bool) if masked else None)
        for fn in self.hooks["post"]:
            fn(self, (), None)


def _guarded_flash(monkeypatch):
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    backends = ad._AttentionBackendRegistry._backends
    ran = []

    def fake_flash(
        query,
        key,
        value,
        attn_mask = None,
        **kw,
    ):
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not supported for flash-attn 2.")
        ran.append("flash")
        return query

    def fake_native(
        query,
        key,
        value,
        attn_mask = None,
        **kw,
    ):
        ran.append("native")
        return query

    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, fake_flash)
    monkeypatch.setitem(backends, ad.AttentionBackendName.NATIVE, fake_native)
    assert att._install_rocm_flash_dispatch_guard() is True
    return backends[ad.AttentionBackendName.FLASH], ran


def _state(pipe, backend):
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class State:
        pipe: object
        attention_backend: object
        resolved: dict

    resolved = {
        "attention_backend": {"source": "auto", "value": backend or "native", "reason": "load"}
    }
    return State(pipe, backend, resolved)


def _load(monkeypatch):
    dit = _Dit()
    pipe = types.SimpleNamespace(transformer = dit)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    return dit, pipe, _state(pipe, "flash")


@pytest.mark.parametrize(
    "masks, value, phrase",
    [
        (
            [True, True],
            "native",
            "every attention call of the last generation ran native SDPA (attn_mask: 2)",
        ),
        ([False, False], "flash", "on every attention call"),
        (
            [True, False],
            "flash",
            "on 1 of 2 attention calls of the last generation; the rest ran native SDPA (attn_mask: 1)",
        ),
    ],
)
def test_badge_reports_the_routes_the_run_took(monkeypatch, masks, value, phrase):
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    dit, pipe, state = _load(monkeypatch)
    att.begin_attention_run(state)
    dit(guarded, masks)
    att.report_attention_run(state)
    assert ran == ["native" if m else "flash" for m in masks]
    entry = state.resolved["attention_backend"]
    assert entry["value"] == value and phrase in entry["reason"] and entry["source"] == "auto"
    # report only: the engaged backend (recipes, calibration and compile keys) is untouched, nothing was switched
    assert state.attention_backend == "flash" and dit.backends == ["flash"]
    ran.clear()
    dit(guarded, [False])
    assert ran == ["flash"]


def test_mixed_mask_model_keeps_flash_for_unmasked_calls_across_loads(monkeypatch, tmp_path):
    # HunyuanVideo 1.5 style: padded batches mask, unpadded ones do not. A masked-only first run changes nothing for
    # the next load, which serves its unmasked calls on flash.
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    dit, pipe, state = _load(monkeypatch)
    dit(guarded, [True, True])
    _restart(monkeypatch)
    dit2, pipe2, state2 = _load(monkeypatch)
    assert len(timed) == 1 and dit2.backends == ["flash"]
    ran.clear()
    dit2(guarded, [False, True])
    assert ran == ["flash", "native"]
    assert not list((tmp_path / "rocm_flash").glob("masked*"))


def test_each_generation_reports_its_own_calls(monkeypatch):
    # run 1 flash, run 2 masked only, run 3 flash: each badge describes its own run, never a lifetime mix
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    dit, pipe, state = _load(monkeypatch)
    for masks, value in (([False, False], "flash"), ([True, True], "native"), ([False], "flash")):
        att.begin_attention_run(state)
        dit(guarded, masks)
        att.report_attention_run(state)
        assert state.resolved["attention_backend"]["value"] == value
        assert state.attention_backend == "flash"


def test_compiled_run_reports_routing_untracked(monkeypatch):
    # a compiled / graphed run never re-enters the guard's Python: the badge says routing is not tracked
    _rocm(monkeypatch)
    _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    dit, pipe, state = _load(monkeypatch)
    att.begin_attention_run(state)
    dit.run(lambda *a, **k: None, [])
    att.report_attention_run(state)
    entry = state.resolved["attention_backend"]
    assert (
        entry["value"] == "flash"
        and "not tracked for compiled or graphed attention" in entry["reason"]
    )


def test_route_tallies_are_per_thread(monkeypatch):
    # an image and a video forward on two threads at once: each DiT's tally gets only its own calls, and each thread's
    # gate is its own
    import threading

    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    a, _, _ = _load(monkeypatch)
    b = _Dit()
    assert (
        att.apply_attention_backend(types.SimpleNamespace(transformer = b), "flash", target = _target())
        == "flash"
    )
    inside = threading.Barrier(2)
    gates = {}

    class Pausing:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kwargs):
            if self.name not in gates:
                inside.wait(timeout = 10)
                gates[self.name] = att._active_rocm_flash_gate()
            return guarded(*args, **kwargs)

    ta = threading.Thread(target = a, args = (Pausing("a"), [True, True, True]))
    tb = threading.Thread(target = b, args = (Pausing("b"), [False]))
    ta.start(), tb.start()
    ta.join(10), tb.join(10)
    tally_a = getattr(a, att.ROCM_FLASH_TALLY_ATTR)
    tally_b = getattr(b, att.ROCM_FLASH_TALLY_ATTR)
    assert tally_a == {"attn_mask": 3} and tally_b == {"flash": 1}
    assert gates["a"] == getattr(a, att.ROCM_FLASH_GATE_ATTR) and gates["b"] is None


# The activation memory guard must not trust the flash pick when calls can run native SDPA.


@pytest.mark.parametrize("routed, quadratic", [(True, True), (False, False)])
def test_memory_guard_budgets_quadratic_when_flash_can_hand_calls_to_native(
    monkeypatch, routed, quadratic
):
    import core.inference.diffusion as diffusion

    _rocm(monkeypatch)
    _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    monkeypatch.setattr(diffusion, "sdpa_math_only", lambda target, *_: True)
    import core.inference.diffusion_qwenimage21_math as qm

    monkeypatch.setattr(qm, "bounded_math_attention", lambda pipe: False)
    dit, pipe, state = _load(monkeypatch)
    if not routed:
        setattr(dit, att._ROCM_FLASH_HOOKS_ATTR, None)
    assert diffusion._quadratic_attention(object(), state.attention_backend, pipe) is quadratic


def test_diffusers_dispatch_change_retimes(monkeypatch, tmp_path):
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    binary = _fake_build(monkeypatch, tmp_path)
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", _REAL_TOOLCHAIN_KEY)
    backends = ad._AttentionBackendRegistry._backends
    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, ad._flash_attention)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    key = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert key is not None and "|diffusers" in key and _REAL_TOOLCHAIN_KEY("cuda:0") == key
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    _restart(monkeypatch)

    def _flash_attention(
        query,
        key,
        value,
        attn_mask = None,
        dropout_p = 0.0,
        is_causal = False,
        scale = None,
    ):
        return query  # a diffusers release whose flash wrapper changed

    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, _flash_attention)
    assert _REAL_TOOLCHAIN_KEY("cuda:0") != key
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 2
    # a Studio guard around the same function does not change it
    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, ad._flash_attention)
    assert _REAL_GUARD() and _REAL_TOOLCHAIN_KEY("cuda:0") == key
    del binary


@pytest.mark.parametrize("module", ["core.inference.diffusion", "core.inference.video"])
def test_both_generate_paths_mark_the_generation_boundary(module):
    import ast
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    tree = ast.parse(src)
    gens = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate"]
    called = {
        getattr(c.func, "id", None) for g in gens for c in ast.walk(g) if isinstance(c, ast.Call)
    }
    assert gens and {"begin_attention_run", "report_attention_run"} <= called


# Compiled execution (Studio compiles the denoiser's blocks on ROCm since its regional compile probe): an auto-gated
# DiT's traced call takes its bucket's verdict at trace time, timing the bucket then if it is unknown, and the answer is
# baked into the graph without a graph break. Real hooks, real torch.compile on CPU.


class _CompiledDit(__import__("torch").nn.Module):
    def __init__(self, guarded):
        import torch

        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.heads = 2
        self.attn.to_q = torch.nn.Linear(32, 32, bias = False)
        self._guarded = guarded

    def set_attention_backend(self, name):
        pass

    def forward(self, q):
        return self._guarded(q, q, q, None)


def _compiled_env(monkeypatch, auto, flash_ms):
    torch = pytest.importorskip("torch")
    _rocm(monkeypatch)
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    backends = ad._AttentionBackendRegistry._backends
    monkeypatch.setitem(
        backends,
        ad.AttentionBackendName.FLASH,
        lambda query, key, value, attn_mask = None, **k: query + 1,
    )
    monkeypatch.setitem(
        backends,
        ad.AttentionBackendName.NATIVE,
        lambda query, key, value, attn_mask = None, **k: query - 1,
    )
    assert _REAL_GUARD()
    guarded = backends[ad.AttentionBackendName.FLASH]
    _apply_env(monkeypatch)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
        **kw,
    ):
        timed.append(tokens)
        return {**_GOOD, "flash_ms": flash_ms.get(tokens, 90.0), "tokens": tokens}

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    dit = _CompiledDit(guarded)
    backend = att.ROCM_FLASH_CANDIDATE if auto else "flash"
    assert (
        att.apply_attention_backend(
            types.SimpleNamespace(transformer = dit), backend, target = _target()
        )
        == "flash"
    )
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d in ("cuda", "cpu") else d
    )
    return torch, dit, timed


@pytest.mark.parametrize("auto", [True, False])
def test_compiled_auto_dit_takes_the_bucket_verdict_under_real_torch_compile(monkeypatch, auto):
    # 4096 admitted at load; 8192 untimed until a traced call needs it, and it wins; 16384 loses
    torch, dit, timed = _compiled_env(monkeypatch, auto, {4096: 70.0, 8192: 70.0})
    q = torch.zeros(1, 4608, 2, 16, dtype = torch.bfloat16)
    assert float(dit(q).flatten()[0]) == 1.0
    torch._dynamo.reset()
    counters = __import__("torch._dynamo.utils", fromlist = ["counters"]).counters
    counters.clear()
    compiled = torch.compile(dit, backend = "eager", fullgraph = True)
    assert float(compiled(q).flatten()[0]) == 1.0  # traced: the known verdict, no graph break
    q8 = torch.zeros(1, 9000, 2, 16, dtype = torch.bfloat16)
    assert float(compiled(q8).flatten()[0]) == 1.0  # timed at trace time, then flash
    q16 = torch.zeros(1, 17000, 2, 16, dtype = torch.bfloat16)
    assert float(compiled(q16).flatten()[0]) == (-1.0 if auto else 1.0)
    # explicit flash is never gated, so nothing past the load is timed for it
    assert timed == ([4096, 8192, 16384] if auto else [])
    assert not counters["graph_break"]
    assert getattr(att._ROCM_FLASH_LOCAL, "gate", None) is None
    torch._dynamo.reset()


def test_compiled_dynamic_lengths_recompile_per_bucket_not_per_length(monkeypatch):
    torch, dit, timed = _compiled_env(monkeypatch, True, {4096: 70.0, 8192: 90.0})
    torch._dynamo.reset()
    frames = []
    compiled = torch.compile(
        dit,
        backend = lambda gm, inputs: frames.append(gm) or gm.forward,
        dynamic = True,
        fullgraph = True,
    )
    bf16 = torch.bfloat16
    out = [
        float(compiled(torch.zeros(1, n, 2, 16, dtype = bf16)).flatten()[0])
        for n in (4352, 4608, 5000)
    ]
    assert out == [1.0, 1.0, 1.0] and len(frames) == 1
    assert float(compiled(torch.zeros(1, 9000, 2, 16, dtype = bf16)).flatten()[0]) == -1.0
    assert len(frames) == 2 and timed == [4096, 8192]
    torch._dynamo.reset()


@pytest.mark.parametrize("native_8192", [False, True])
def test_compiled_lengths_across_known_buckets_guard_only_where_the_route_changes(
    monkeypatch, native_8192
):
    # every bucket already timed (persisted, read back after a restart): neighbouring buckets with the same verdict
    # share one graph; a bucket whose verdict differs still gets its own
    flash_ms = {4096: 70.0, 8192: 90.0 if native_8192 else 70.0, 16384: 70.0}
    torch, dit, timed = _compiled_env(monkeypatch, True, flash_ms)
    bf16 = torch.bfloat16
    lengths = (4352, 9000, 17000, 5000, 9500)
    eager = [float(dit(torch.zeros(1, n, 2, 16, dtype = bf16)).flatten()[0]) for n in lengths]
    att._ROCM_FLASH_BUCKETS.clear()
    att.ROCM_FLASH_ADMISSIONS.clear()
    att._ROCM_FLASH_STORED_MISSES.clear()
    torch._dynamo.reset()
    frames = []
    compiled = torch.compile(
        dit,
        backend = lambda gm, inputs: frames.append(gm) or gm.forward,
        dynamic = True,
        fullgraph = True,
    )
    out, counts = [], []
    for n in lengths:
        out.append(float(compiled(torch.zeros(1, n, 2, 16, dtype = bf16)).flatten()[0]))
        counts.append(len(frames))
    assert out == eager == ([1.0, -1.0, 1.0, 1.0, -1.0] if native_8192 else [1.0] * 5)
    assert counts == ([1, 2, 3, 3, 3] if native_8192 else [1, 1, 1, 1, 1])
    assert timed == [4096, 8192, 16384]  # read back, never re-timed
    torch._dynamo.reset()


def test_cfg_replica_keeps_the_rocm_flash_gate_on_its_worker_thread(monkeypatch):
    mg = pytest.importorskip("core.inference.diffusion_multigpu")
    torch, dit, timed = _compiled_env(monkeypatch, True, {4096: 70.0, 8192: 110.0})
    q = torch.zeros(1, 9000, 2, 16, dtype = torch.bfloat16)
    assert float(dit(q).flatten()[0]) == -1.0  # 8192 lost its timing: native on the source
    replica = mg.build_replica(dit, "cpu")
    worker = mg._ReplicaWorker(torch.device("cpu"))
    try:
        out = worker.call(mg._replica_call(replica, (q,), {}))
    finally:
        worker.close()
    assert float(out.flatten()[0]) == -1.0 and timed == [4096, 8192]
    assert getattr(att._ROCM_FLASH_LOCAL, "gate", None) is None
    # a rebuilt replica (weights changed) is tracked too; the copy carries no replica set of its own
    again = mg.build_replica(dit, "cpu")
    assert att._ROCM_FLASH_REPLICAS_ATTR not in again.__dict__
    assert set(getattr(dit, att._ROCM_FLASH_REPLICAS_ATTR)) == {replica, again}
    # on another card the gate is that card's, so its buckets are timed there
    copy = torch.nn.Module()
    att.track_rocm_flash_replica(dit, copy, "cuda:1")
    gate = getattr(copy, att.ROCM_FLASH_GATE_ATTR)
    assert {k[0] for k in gate} == {"cuda:1"} and len(copy._forward_pre_hooks) == 1


def test_traced_call_with_no_room_for_its_probe_runs_native(monkeypatch):
    torch, dit, timed = _compiled_env(monkeypatch, True, {4096: 70.0, 8192: 70.0})
    monkeypatch.setattr(att, "_rocm_flash_probe_fits", lambda device, need: False)
    torch._dynamo.reset()
    compiled = torch.compile(dit, backend = "eager", fullgraph = True)
    assert float(compiled(torch.zeros(1, 9000, 2, 16, dtype = torch.bfloat16)).flatten()[0]) == -1.0
    assert (
        timed == [4096]
        and att._ROCM_FLASH_BUCKETS.get(("cuda:0", "torch.bfloat16", 32, 2, 16, 8192, 8192, 1))
        is None
    )
    torch._dynamo.reset()


# Cross-attention is timed as the rectangle that runs (long video query x short text key), never as a square.


def test_cross_attention_is_not_decided_by_a_square_probe(monkeypatch):
    dit, guarded, ran, timed = _length_env(
        monkeypatch, {4096: 70.0, 16384: 60.0, (16384, 512): 90.0}
    )
    dit.run(guarded, [(27280, None), (27280, 512), (27280, 600)])
    assert timed == [4096, 16384, (16384, 512)]
    assert ran == ["flash", "native", "native"]
    rec = att.ROCM_FLASH_ADMISSIONS[("cuda:0", "torch.bfloat16", (32, 2, 16), 16384, 512, 1)]
    assert rec["kv_tokens"] == 512 and rec["admitted"] is False


def test_rectangular_probe_runs_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")

    def flash_attn_func(q, k, v, **kw):
        assert q.shape[1] == 64 and k.shape[1] == 16
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    _diffusers_flash(monkeypatch, flash_attn_func)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    rec = att._time_rocm_flash_block("cpu", torch.bfloat16, 64, 4, 16, tokens = 64, kv_tokens = 16)
    assert rec["tokens"] == 64 and rec["kv_tokens"] == 16 and rec["rel_l1"] == 0.0
    assert att._rocm_flash_record_ok(rec)


def test_named_partition_is_in_run_accounting(monkeypatch):
    from core.inference.video import _NamedDiTView

    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    ref = _Dit()
    pipe = types.SimpleNamespace(transformer_ref = ref)
    assert att._attention_dits(pipe) == []  # the stock enumeration cannot see it
    view = _NamedDiTView(pipe, "transformer_ref")
    assert att.apply_attention_backend(view, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    state = _state(pipe, "flash")
    assert att.attention_may_run_native(pipe)
    att.begin_attention_run(state)
    ref(guarded, [True, True])
    att.report_attention_run(state)
    assert state.resolved["attention_backend"]["value"] == "native"
    att.begin_attention_run(state)
    assert getattr(ref, att.ROCM_FLASH_TALLY_ATTR) == {}


def test_batch_two_is_not_decided_by_a_batch_one_verdict(monkeypatch):
    dit, guarded, ran, timed = _length_env(monkeypatch, {4096: 70.0, (4096, "b2"): 90.0})
    import torch

    def call(batch):
        for fn in dit.hooks["pre"]:
            fn(dit, ())
        q = torch.zeros(batch, 4608, 2, 16, dtype = torch.bfloat16)
        guarded(q, q, q, None)
        for fn in dit.hooks["post"]:
            fn(dit, (), None)

    call(1)
    call(2)
    call(3)
    assert timed == [4096, (4096, "b2")] and ran == ["flash", "native", "native"]
    rec = att.ROCM_FLASH_ADMISSIONS[("cuda:0", "torch.bfloat16", (32, 2, 16), 4096, 4096, 2)]
    assert rec["batch"] == 2 and rec["admitted"] is False


def test_batched_probe_runs_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    shapes = []

    def flash_attn_func(q, k, v, **kw):
        shapes.append((tuple(q.shape), tuple(k.shape)))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    _diffusers_flash(monkeypatch, flash_attn_func)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    rec = att._time_rocm_flash_block(
        "cpu", torch.bfloat16, 64, 4, 16, tokens = 64, kv_tokens = 16, ctx_dim = 48, batch = 2
    )
    assert shapes[0] == ((2, 64, 4, 16), (2, 16, 4, 16))
    assert (
        rec["batch"] == 2
        and rec["kv_tokens"] == 16
        and rec["rel_l1"] == 0.0
        and att._rocm_flash_record_ok(rec)
    )


def test_context_width_is_part_of_the_geometry(monkeypatch):
    torch = pytest.importorskip("torch")

    class Cross(torch.nn.Module):
        def __init__(self, kin):
            super().__init__()
            self.heads = 2
            self.to_q = torch.nn.Linear(32, 32, bias = False)
            self.to_k = torch.nn.Linear(kin, 32, bias = False)

    dit = torch.nn.Module()
    dit.self_attn, dit.cross = Cross(32), Cross(48)
    original = att._attention_dits
    try:
        att._attention_dits = lambda p: [dit]
        assert att._dit_attention_geometries(types.SimpleNamespace()) == (
            (32, 2, 16),
            (32, 2, 16, 48),
        )
    finally:
        att._attention_dits = original
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    seen = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
        kv_tokens = None,
        ctx_dim = None,
        batch = 1,
    ):
        seen.append(ctx_dim)
        return {**_GOOD, "tokens": tokens}

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    assert att._rocm_flash_geometry_admitted("cuda:0", "torch.bfloat16", (32, 2, 16, 48))
    assert att._rocm_flash_geometry_admitted("cuda:0", "torch.bfloat16", (32, 2, 16))
    assert seen == [48, None]
    rec = att.ROCM_FLASH_ADMISSIONS[("cuda:0", "torch.bfloat16", (32, 2, 16, 48), 4096, 4096, 1)]
    assert rec["geometry"]["context"] == 48 and att._rocm_flash_record_ok(rec)


# MiniMax-H3's resident INT8 load fuses to_q / to_k / to_v into to_qkv before the backend is applied.


def test_fused_qkv_geometry_is_collected():
    torch = pytest.importorskip("torch")

    class Fused(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.heads = 2
            self.to_qkv = torch.nn.Linear(32, out, bias = False)

    dit = torch.nn.Module()
    dit.fused, dit.odd = Fused(96), Fused(95)
    original = att._attention_dits
    try:
        att._attention_dits = lambda p: [dit]
        assert att._dit_attention_geometries(types.SimpleNamespace()) == ((32, 2, 16),)
    finally:
        att._attention_dits = original


# The key carries the ROCm runtime / kernel driver the process runs on, not only the wheel's build.


def test_driver_change_retimes(monkeypatch, tmp_path):
    pytest.importorskip("diffusers.models.attention_dispatch")
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    _fake_build(monkeypatch, tmp_path)
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", _REAL_TOOLCHAIN_KEY)
    driver = ["driver:hipdriver70051234,amdgpu6.12.12"]
    monkeypatch.setattr(att, "_rocm_driver_version", lambda: driver[0])
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    key = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert key is not None and "|driver:hipdriver70051234,amdgpu6.12.12|" in key
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    _restart(monkeypatch)
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 1
    driver[0] = "driver:hipdriver70151234,amdgpu6.14.4"
    _restart(monkeypatch)
    att.apply_attention_backend(
        _pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    assert len(timed) == 2


def test_driver_version_reads_without_raising(monkeypatch):
    monkeypatch.setattr(att, "_ROCM_DRIVER_VERSION", [])
    v = att._rocm_driver_version()
    assert v.startswith("driver:") and att._rocm_driver_version() is v


def test_attention_status_reason():
    assert att.attention_status_reason(None, "flash") == "verified ROCm flash attention"
    assert att.attention_status_reason("auto", "flash") == "verified ROCm flash attention"
    assert att.attention_status_reason(None, "_native_cudnn") == "cuDNN fused attention upgrade"
    assert att.attention_status_reason(None, None) == "diffusers default"
    assert att.attention_status_reason("flash", "flash") == "requested"
    assert att.attention_status_reason("flash", None) == "diffusers default"


# The in-memory admission key keeps geometry, query length, key length and batch in separate fields: a cross-attention
# geometry whose context width equals a bucket edge must not alias a narrower geometry at rectangular lengths.


def test_admission_key_does_not_alias_geometry_and_lengths(monkeypatch):
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: None)
    timed = []

    def fake_time(
        device,
        dtype,
        hidden,
        heads,
        head_dim,
        tokens = 4096,
        kv_tokens = None,
        **kw,
    ):
        timed.append((hidden, heads, head_dim, tokens, kv_tokens, kw.get("ctx_dim")))
        wins = kv_tokens is None
        return {
            "native_ms": 80.0,
            "flash_ms": 60.0 if wins else 90.0,
            "rel_l1": 1e-3,
            "max_rel": 1e-3,
            "finite": True,
            "tokens": tokens,
            **({} if kv_tokens is None else {"kv_tokens": kv_tokens}),
        }

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    assert (
        att._rocm_flash_geometry_admitted("cuda:0", "bf16", (32, 2, 16, 256), tokens = 4096) is True
    )
    # (32, 2, 16) at 256 x 4096 flattens to the same tuple as (32, 2, 16, 256) at 4096: it must be timed on its own
    assert (
        att._rocm_flash_geometry_admitted("cuda:0", "bf16", (32, 2, 16), tokens = 256, kv_tokens = 4096)
        is False
    )
    assert timed == [(32, 2, 16, 4096, None, 256), (32, 2, 16, 256, 4096, None)]
    assert len(att.ROCM_FLASH_ADMISSIONS) == 2


# A re-apply on the same pipe that no longer picks ROCm flash takes the tracking off with it: the badge and the memory
# guard must not keep reading the pipe as flash.


def test_reapply_to_native_removes_rocm_flash_tracking(monkeypatch):
    torch, dit, timed = _compiled_env(monkeypatch, True, {4096: 70.0})
    pipe = types.SimpleNamespace(transformer = dit)
    att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target())
    assert att.attention_may_run_native(pipe) and att.rocm_flash_ran(pipe) is not None
    hooks = len(dit._forward_pre_hooks) + len(dit._forward_hooks)
    assert att.apply_attention_backend(pipe, None, target = _target()) is None
    assert not att.attention_may_run_native(pipe) and att.rocm_flash_ran(pipe) is None
    assert len(dit._forward_pre_hooks) + len(dit._forward_hooks) == hooks - 2
    entry = {"value": "native", "reason": "diffusers default"}
    state = types.SimpleNamespace(pipe = pipe, resolved = {"attention_backend": entry})
    att.begin_attention_run(state)
    att.report_attention_run(state)
    assert state.resolved["attention_backend"] == entry
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    assert att.attention_may_run_native(pipe)
