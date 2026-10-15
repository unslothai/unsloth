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

_REAL_TOOLCHAIN_KEY = att._rocm_flash_toolchain_key


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(att, "_ROCM_FLASH_PROBE_CACHE", {}, raising = False)
    # The persisted block verdicts: a per-test dir and a fixed toolchain (the real one needs a ROCm card).
    monkeypatch.setenv(att.ROCM_FLASH_CACHE_DIR_ENV, str(tmp_path / "rocm_flash"))
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", lambda device: "tc-A", raising = False)
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: "cuda:0" if d == "cuda" else d, raising = False
    )
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_ROCM_FLASH_MASKED", [None])
    monkeypatch.setattr(att, "_ROCM_FLASH_RAN", [0])
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


# ``auto`` on ROCm: a timed candidate for every family and arch.


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

    def fake_time(device, dtype, hidden, heads, head_dim):
        timed.append((device, hidden, heads, head_dim))
        return {
            "native_ms": 80.0,
            "flash_ms": flash_ms,
            "rel_l1": rel,
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
    rec = att._time_rocm_flash_block("cpu", torch.bfloat16, 64, 4, 16, tokens = 32)
    assert rec["finite"] and rec["rel_l1"] == 0.0 and rec["native_ms"] > 0 and rec["flash_ms"] > 0
    # 1 accuracy run + 2 warm-ups + 5 timed, every one through diffusers' FLASH wrapper with its keyword arguments
    assert len(seen) == 8 and set(seen) == {(0.0, None, False, (-1, -1), False)}
    assert att._ROCM_FLASH_RAN[0] == 8  # and through Studio's mask guard


def test_flash_attn_rejecting_the_diffusers_arguments_is_not_admitted(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda d: "cpu")

    def old_api(q, k, v):  # a build without dropout_p / softmax_scale / window_size / return_attn_probs
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
    # the load-time pick and the eager re-pick; the pre-lock preinstall needs no family (no ROCm pip install)
    assert len(with_family) == 2
    assert sum(any(k.arg == "speed_unset" for k in c.keywords) for c in with_family) == 1


def test_auto_attention_reason_names_the_engaged_kernel():
    assert att.auto_attention_reason(None) == "diffusers default"
    assert att.auto_attention_reason("_native_cudnn") == "cuDNN fused attention upgrade"
    assert "cuDNN" not in att.auto_attention_reason("flash")
    assert "ROCm flash" in att.auto_attention_reason("flash")


def test_status_reason_is_derived_from_the_engaged_backend():
    src = open(
        att.__file__.replace("diffusion_attention.py", "diffusion.py"), encoding = "utf-8"
    ).read()
    assert src.count("auto_attention_reason(attention_engaged)") == 2
    assert '"cuDNN fused attention upgrade"' not in src


# Review: masked attention (HunyuanVideo 1.5 mixed padding) must never reach diffusers' flash backend, which raises on
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
    assert att._install_rocm_flash_dispatch_guard() is True  # idempotent
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

    def fake_time(device, dtype, hidden, heads, head_dim):
        timed.append((hidden, heads, head_dim))
        flash = 60.0 if (hidden == 3072 or minority_wins) else 85.0
        return {
            "native_ms": 80.0,
            "flash_ms": flash,
            "rel_l1": 1e-3,
            "finite": True,
            "tokens": 4096,
        }

    monkeypatch.setattr(att, "_time_rocm_flash_block", fake_time)
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == engaged
    assert (1536, 12, 128) in timed  # the minority geometry (transformer_2) is timed too
    if engaged:
        assert sorted(timed) == [(1536, 12, 128), (3072, 24, 128)]
        assert set_to == ["flash", "flash"] and len(att.ROCM_FLASH_ADMISSIONS) == 2
    else:
        assert set_to == []
        # one losing geometry is enough: the next load reads the cached answer and stays native without timing
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
    def fake_time(device, dtype, hidden, heads, head_dim):
        timed.append((hidden, heads, head_dim))
        return {
            "native_ms": 80.0,
            "flash_ms": flash_ms,
            "rel_l1": 1e-3,
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
        att, "_time_rocm_flash_block", lambda *a, **k: pytest.fail("re-timed despite a stored verdict")
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


@pytest.mark.parametrize("payload", ["{not json", "", "[]", '{"key": "tc-A", "verdicts": {"x": 1}}'])
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
    # the re-timed verdict replaced the torn file
    _restart(monkeypatch)
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
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
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
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
    import torch

    props = types.SimpleNamespace(name = "AMD Radeon 8060S", gcnArchName = "gfx1151:sramecc-:xnack-")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising = False)
    fa = types.ModuleType("flash_attn")
    fa.__version__ = "2.8.3"
    fa.__file__ = "/x/site-packages/flash_attn/__init__.py"
    monkeypatch.setitem(sys.modules, "flash_attn", fa)
    monkeypatch.delenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", raising = False)
    ck = _REAL_TOOLCHAIN_KEY("cuda:0")
    assert ck is not None and ck.startswith(f"v{att.ROCM_FLASH_PROBE_VERSION}|")
    for part in ("AMD Radeon 8060S", "|gfx1151|", "|hip7.1.0|", "|flash_attn2.8.3|", "/flash_attn|ck|"):
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


def _fake_build(monkeypatch, tmp_path, *, triton = False):
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
    import os

    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    binary = _fake_build(monkeypatch, tmp_path)  # after _rocm, which installs a bare flash_attn module
    monkeypatch.setattr(att, "_rocm_flash_toolchain_key", _REAL_TOOLCHAIN_KEY)
    assert _REAL_TOOLCHAIN_KEY("cuda:0") is not None
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
    _restart(monkeypatch)
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
    assert len(timed) == 1  # same build: the stored verdict is used
    binary.write_bytes(b"\x7fELF v2 rebuilt")
    st = os.stat(binary)
    os.utime(binary, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _restart(monkeypatch)
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
    assert len(timed) == 2


# Malformed cached records are a cache miss (re-timed), never a verdict, and never break the verdict log.

_GOOD = {"native_ms": 80.0, "flash_ms": 70.0, "rel_l1": 1e-3, "finite": True, "tokens": 4096}


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
    ],
)
def test_malformed_cached_record_is_retimed(monkeypatch, tmp_path, field, value):
    import json

    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed, 90.0))
    att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target())
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

    def nan_time(device, dtype, hidden, heads, head_dim):
        timed.append(1)
        return {**_GOOD, "finite": False, "rel_l1": float("nan")}

    monkeypatch.setattr(att, "_time_rocm_flash_block", nan_time)
    assert att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    _restart(monkeypatch)
    assert att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()) is None
    assert len(timed) == 1


def test_implausible_probe_timing_is_native_and_not_stored(monkeypatch, tmp_path):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(
        att, "_time_rocm_flash_block", lambda *a, **k: {**_GOOD, "native_ms": 0.0, "flash_ms": 0.0}
    )
    assert att.apply_attention_backend(_pipe_with_attention()[0], att.ROCM_FLASH_CANDIDATE, target = _target()) is None
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


# A model whose attention calls all carry a mask never runs the flash kernel (the guard routes them native): it must not
# stay tagged flash, and later loads neither probe nor report flash.


class _Dit:
    """A DiT whose forward makes ``calls`` attention calls through the guarded FLASH backend."""

    def __init__(self, masked, calls = 2):
        torch = pytest.importorskip("torch")
        self.inner = torch.nn.Module()
        self.masked, self.calls, self.backends = masked, calls, []
        self.inner.heads = 4
        self.inner.to_q = torch.nn.Linear(64, 64, bias = False)

    def modules(self):
        return iter([self.inner])

    def set_attention_backend(self, name):
        self.backends.append(name)

    def register_forward_pre_hook(self, fn):
        return self._hooks.setdefault("pre", []).append(fn) or _Handle(self._hooks["pre"], fn)

    def register_forward_hook(self, fn):
        return self._hooks.setdefault("post", []).append(fn) or _Handle(self._hooks["post"], fn)

    @property
    def _hooks(self):
        return self.__dict__.setdefault("_hook_lists", {})

    def __call__(self, guarded):
        import torch

        for fn in list(self._hooks.get("pre", ())):
            fn(self, ())
        q = torch.zeros(1, 8, 2, 16, dtype = torch.bfloat16)
        mask = torch.ones(1, 1, 8, 8, dtype = torch.bool) if self.masked else None
        for _ in range(self.calls):
            guarded(q, q, q, attn_mask = mask)
        for fn in list(self._hooks.get("post", ())):
            fn(self, (), None)


class _Handle:
    def __init__(self, hooks, fn):
        self.hooks, self.fn = hooks, fn

    def remove(self):
        if self.fn in self.hooks:
            self.hooks.remove(self.fn)


def _guarded_flash(monkeypatch):
    ad = pytest.importorskip("diffusers.models.attention_dispatch")
    backends = ad._AttentionBackendRegistry._backends
    ran = []

    def fake_flash(query, key, value, attn_mask = None, **kw):
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not supported for flash-attn 2.")
        ran.append("flash")
        return query

    def fake_native(query, key, value, attn_mask = None, **kw):
        ran.append("native")
        return query

    monkeypatch.setitem(backends, ad.AttentionBackendName.FLASH, fake_flash)
    monkeypatch.setitem(backends, ad.AttentionBackendName.NATIVE, fake_native)
    assert att._install_rocm_flash_dispatch_guard() is True
    return backends[ad.AttentionBackendName.FLASH], ran


@pytest.mark.parametrize("masked", [True, False])
def test_masked_only_model_is_not_kept_on_flash(monkeypatch, tmp_path, masked):
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    timed = []
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer(timed))
    dit = _Dit(masked)
    pipe = types.SimpleNamespace(transformer = dit)
    # the first load cannot see the call pattern yet: it is admitted by timing
    assert att.apply_attention_backend(pipe, att.ROCM_FLASH_CANDIDATE, target = _target()) == "flash"
    assert len(timed) == 1 and getattr(dit, att.ATTENTION_BACKEND_ATTR) == "flash"
    dit(guarded)
    dit(guarded)
    if masked:
        assert ran == ["native"] * 4
        # after the first forward it runs native and is no longer tagged or watched
        assert dit.backends == ["flash", att.ATTN_NATIVE]
        assert getattr(dit, att.ATTENTION_BACKEND_ATTR) is None and dit._hooks == {"pre": [], "post": []}
    else:
        assert ran == ["flash"] * 4
        assert dit.backends == ["flash"] and getattr(dit, att.ATTENTION_BACKEND_ATTR) == "flash"
        assert dit._hooks == {"pre": [], "post": []}  # decided after one forward, then unwatched
    # a later load in a new process: a masked-only model is native with no probe at all, an unmasked one is unchanged
    _restart(monkeypatch)
    monkeypatch.setattr(att, "_ROCM_FLASH_MASKED", [None])
    probes = _rocm(monkeypatch)
    dit2 = _Dit(masked)
    engaged = att.apply_attention_backend(
        types.SimpleNamespace(transformer = dit2), att.ROCM_FLASH_CANDIDATE, target = _target()
    )
    if masked:
        assert engaged is None and probes == [] and len(timed) == 1
        assert dit2.backends == [] and getattr(dit2, att.ATTENTION_BACKEND_ATTR) is None
        assert (tmp_path / "rocm_flash" / "masked_only.json").is_file()
    else:
        assert engaged == "flash" and len(timed) == 1  # stored timing verdict
        assert not (tmp_path / "rocm_flash" / "masked_only.json").exists()


def test_mixed_mask_model_stays_flash(monkeypatch):
    # a padded batch masks some calls; the unmasked ones run flash, so the model keeps it
    _rocm(monkeypatch)
    guarded, ran = _guarded_flash(monkeypatch)
    _apply_env(monkeypatch)
    monkeypatch.setattr(att, "_time_rocm_flash_block", _timer([]))
    dit = _Dit(masked = False)
    att.apply_attention_backend(types.SimpleNamespace(transformer = dit), att.ROCM_FLASH_CANDIDATE, target = _target())
    for fn in list(dit._hooks["pre"]):
        fn(dit, ())
    import torch

    q = torch.zeros(1, 8, 2, 16, dtype = torch.bfloat16)
    guarded(q, q, q, attn_mask = torch.ones(1, 1, 8, 8, dtype = torch.bool))
    guarded(q, q, q, None)
    for fn in list(dit._hooks["post"]):
        fn(dit, (), None)
    assert ran == ["native", "flash"] and dit.backends == ["flash"]
    assert getattr(dit, att.ATTENTION_BACKEND_ATTR) == "flash" and att._rocm_flash_masked_only() == set()


def test_explicit_flash_is_not_watched(monkeypatch):
    _rocm(monkeypatch)
    _apply_env(monkeypatch)
    dit = _Dit(masked = True)
    assert att.apply_attention_backend(types.SimpleNamespace(transformer = dit), "flash", target = _target()) == "flash"
    assert dit._hooks == {}
