# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for attention-backend selection. No torch/diffusers needed:
``_is_cuda_nvidia`` is monkeypatched for the policy tests, and the apply path uses a fake
transformer that records / raises on ``set_attention_backend``.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import (
    ATTN_AUTO,
    apply_attention_backend,
    normalize_attention_backend,
    select_attention_backend,
)


def _target(device = "cuda"):
    return types.SimpleNamespace(device = device)


# ── normalize ────────────────────────────────────────────────────────────────────
def test_normalize_defaults_and_aliases():
    assert normalize_attention_backend(None) == ATTN_AUTO
    assert normalize_attention_backend("") == ATTN_AUTO
    assert normalize_attention_backend("auto") == ATTN_AUTO
    assert normalize_attention_backend("CuDNN") == "cudnn"
    assert normalize_attention_backend("FLASH3") == "flash3"
    assert normalize_attention_backend("sdpa") == "sdpa"


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_attention_backend("bogus")
    # dashes are no longer silently rewritten to underscores, so a dashed alias is rejected.
    with pytest.raises(ValueError):
        normalize_attention_backend("flash-3")


def test_sdpa_alias_maps_to_native():
    # sdpa is an alias for native, so nothing to set on the dispatcher.
    assert select_attention_backend(_target(), "sdpa", speed_active = True) is None


# ── select policy ─────────────────────────────────────────────────────────────────
def test_auto_upgrades_to_cudnn_on_nvidia_when_speed_active(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))  # Ampere+: cuDNN ok
    assert select_attention_backend(_target(), "auto", speed_active = True) == "_native_cudnn"


def test_auto_does_not_pin_cudnn_below_sm80(monkeypatch):
    # cuDNN fused SDPA fails at run time on pre-SM80 (T4 / V100), so auto must stay native rather than pin a backend that crashes.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))  # Turing T4
    assert select_attention_backend(_target(), "auto", speed_active = True) is None


def test_auto_stays_native_when_speed_off(monkeypatch):
    # off must stay bit-identical, so no backend change even on NVIDIA.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "auto", speed_active = False) is None


def test_auto_stays_native_off_nvidia(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "auto", speed_active = True) is None


def test_explicit_backend_honored_regardless_of_speed(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    # Pin a high capability so the arch-gated flash4 isn't dropped by the runtime check.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "sage", speed_active = False) == "sage"
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


def test_explicit_backend_dropped_off_nvidia_cuda(monkeypatch):
    # Explicit cuDNN/flash/sage on ROCm / MPS / CPU passes diffusers' set-time check then crashes at first generation, so selection drops to native.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    for alias in ("sage", "flash", "flash4", "cudnn"):
        assert select_attention_backend(_target(device = "mps"), alias, speed_active = True) is None


def test_aiter_honored_on_rocm(monkeypatch):
    # AITER is the AMD ROCm kernel, so a ROCm CUDA target must honor it rather than drop it via the NVIDIA-only guard.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)  # hip build
    assert select_attention_backend(_target(), "aiter", speed_active = False) == "aiter"


def test_aiter_dropped_off_rocm(monkeypatch):
    # aiter on NVIDIA CUDA (or MPS / CPU) is not usable, so it drops to the native default.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)  # NVIDIA
    assert select_attention_backend(_target(), "aiter", speed_active = False) is None
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "aiter", speed_active = False) is None


def test_explicit_native_returns_none():
    # native is the default, so nothing to set.
    assert select_attention_backend(_target(), "native", speed_active = True) is None


# ── arch gating (flash3/flash4 need a specific CUDA capability) ─────────────────────
def test_flash3_dropped_below_hopper(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 9))  # Ada / consumer
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None


def test_flash4_dropped_below_blackwell(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))  # Hopper, but FA4 needs SM100
    assert select_attention_backend(_target(), "flash4", speed_active = False) is None
    # flash3 still allowed on Hopper.
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_arch_gate_does_not_block_when_capability_unknown(monkeypatch):
    # Unknown capability must not block; diffusers' set-time check still guards.
    monkeypatch.setattr(att, "_cuda_capability", lambda: None)
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"


def test_flash3_dropped_on_blackwell(monkeypatch):
    # FlashAttention 3 is a Hopper-SM90 rewrite with no Blackwell kernel, so explicit flash3 on a B200 drops to native rather than set fine then crash.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None
    # FA4 is still honored on Blackwell.
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    # flash3 is allowed exactly on Hopper SM90.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_explicit_cudnn_dropped_below_sm80(monkeypatch):
    # An explicit cuDNN request on pre-Ampere drops to native, the same gate the auto path applies.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) is None
    # Ampere+ still honors it.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


# ── apply ─────────────────────────────────────────────────────────────────────────
class _FakeTransformer:
    def __init__(self, *, fail = False):
        self.fail = fail
        self.set_to = None

    def set_attention_backend(self, name):
        if self.fail:
            raise RuntimeError(f"{name} kernel unavailable")
        self.set_to = name


def _pipe(transformer):
    return types.SimpleNamespace(transformer = transformer)


def test_apply_none_leaves_native_when_global_already_native(monkeypatch):
    # Global already native, so no redundant set call.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to is None


def test_apply_none_restores_native_when_global_polluted(monkeypatch):
    # A previous load pinned cuDNN process-wide; a native load must reset it so it cannot silently inherit cuDNN (the bit-identical guarantee).
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to == "native"


def test_apply_sets_backend():
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"


def test_apply_sets_backend_on_both_dits():
    # A dual-DiT family (Ideogram) runs both DiTs each step, so the backend must be set on BOTH or status reports a kernel the second never uses.
    t1, t2 = _FakeTransformer(), _FakeTransformer()
    pipe = types.SimpleNamespace(transformer = t1, unconditional_transformer = t2)
    engaged = apply_attention_backend(pipe, "_native_cudnn")
    assert engaged == "_native_cudnn"
    assert t1.set_to == "_native_cudnn" and t2.set_to == "_native_cudnn"


def test_apply_falls_back_on_unavailable_kernel(monkeypatch):
    # An unavailable kernel must not fail the load: returns None (diffusers default).
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


def test_apply_failed_kernel_restores_native_when_polluted(monkeypatch):
    # Requested kernel fails AND the global is polluted: restore native before returning.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")

    class _FailOnceTransformer:
        def __init__(self):
            self.calls = []

        def set_attention_backend(self, name):
            self.calls.append(name)
            if name != "native":
                raise RuntimeError(f"{name} kernel unavailable")

    t = _FailOnceTransformer()
    assert apply_attention_backend(_pipe(t), "sage") is None
    assert t.calls == ["sage", "native"]


def test_apply_handles_missing_method():
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert apply_attention_backend(pipe, "_native_cudnn") is None


def test_apply_resets_global_registry_after_success(monkeypatch):
    # After a successful per-transformer set, the process-wide registry resets to native so a later component cannot inherit this kernel.
    called = {"reset": False}
    monkeypatch.setattr(
        att, "_reset_global_backend_to_native", lambda logger: called.__setitem__("reset", True)
    )
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"
    assert called["reset"] is True


def test_active_attention_backend_reads_tuple_return():
    # get_active_backend() returns a (AttentionBackendName, fn) tuple, so the helper must read the name's .value, not stringify the tuple.
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
    assert att._active_attention_backend() == "native"


# ── on-demand wheel-only install of optional kernels ─────────────────────────────
@pytest.fixture(autouse = True)
def _no_real_installs(monkeypatch):
    # Unit tests must never shell out to pip: hard-disable the install gate; the install tests re-enable it with a stubbed subprocess.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    # The install once-per-process memo is module state; clear it so each test starts fresh.
    att._INSTALL_ATTEMPTED.clear()
    # So is the resolved xFormers wheel. Left set, one test's stub decides the next one's
    # answer -- and unset, the resolver shells out to probe the real torch.
    monkeypatch.setattr(att, "_XFORMERS_WHEEL_TARGET", None)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(returncode = 0)


def _stub_subprocess(monkeypatch, run):
    import subprocess
    monkeypatch.setattr(subprocess, "run", run)


def test_install_skipped_when_gate_disabled(monkeypatch):
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls == []


def test_install_skipped_when_module_present(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "sageattention" else None
    )
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls == []


def test_install_runs_wheel_only_for_missing_kernel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert "--only-binary" in cmd and ":all:" in cmd
    assert any(a.startswith("sageattention") for a in cmd)


def test_sage_install_carries_the_dispatcher_version_floor(monkeypatch):
    # PyPI's newest sageattention wheel is 1.0.6 but diffusers refuses anything below 2.1.1, so an unpinned install "succeeds",
    # writes an unusable 1.0.6 into the user's venv and is then rejected. The requirement carries the floor so pip resolves nothing.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    req = next(a for a in run.calls[0] if a.startswith("sageattention"))
    assert req == "sageattention>=2.1.1", req
    # Unversioned kernels are unaffected.
    assert att._pip_requirement("xformers", "xformers") == "xformers"


_XFORMERS_WHEEL = (
    "https://download.pytorch.org/whl/cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl"
)


def _stub_xformers_wheel(monkeypatch, url = _XFORMERS_WHEEL, reason = None):
    """Pin the resolved xFormers wheel so no test probes the real torch."""
    monkeypatch.setattr(att, "_xformers_wheel_target", lambda: (url, reason))


def test_install_uses_no_deps_to_protect_core_deps(monkeypatch):
    # A kernel add-on pins an exact torch, so --no-deps installs only the kernel wheel.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("xformers")
    assert len(run.calls) == 1
    assert "--no-deps" in run.calls[0]


def test_xformers_installs_the_cuda_matched_wheel_not_the_package_name(monkeypatch):
    """`pip install xformers` resolves the PyPI build, which exists only in the CUDA-12.8
    flavour: beside a cu130 torch its extension fails to load and xformers/_cpp_lib.py
    logs a warning instead of raising, so memory-efficient attention vanishes silently
    while the import still succeeds. Only a URL resolved against the running torch is safe."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)

    assert att._ensure_attention_backend_installed("xformers") is None
    assert len(run.calls) == 1
    assert _XFORMERS_WHEEL in run.calls[0]
    assert "xformers" not in [arg for arg in run.calls[0] if arg != _XFORMERS_WHEEL]


def test_xformers_install_refused_when_no_matching_wheel_exists(monkeypatch):
    """No matched wheel must mean NO install. Falling back to an unpinned resolve is the
    bug; the caller stays on torch SDPA and gets the reason back."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(
        monkeypatch, url = None, reason = "no xFormers wheel is published for torch 2.11.0"
    )
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)

    reason = att._ensure_attention_backend_installed("xformers")
    assert reason == "no xFormers wheel is published for torch 2.11.0"
    assert run.calls == []


def test_xformers_refusal_is_logged_with_its_reason(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch, url = None, reason = "torch could not be probed")
    _stub_subprocess(monkeypatch, _Recorder())

    warnings = []

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, msg, *args, **kwargs):
            warnings.append(msg % args if args else msg)

    att._ensure_attention_backend_installed("xformers", _Logger())
    assert len(warnings) == 1
    assert "torch could not be probed" in warnings[0]
    assert "not installing" in warnings[0]


def test_xformers_refusal_records_no_attempt(monkeypatch):
    """Refusal is a POLICY decision, like the kernels/hub gate, so it must not burn the
    one-shot install slot: once the resolver can answer, the install still happens."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch, url = None, reason = "nope")
    _stub_subprocess(monkeypatch, _Recorder())

    att._ensure_attention_backend_installed("xformers")
    assert att._INSTALL_ATTEMPTED == set()

    # Resolver can now answer -> the next request installs.
    _stub_xformers_wheel(monkeypatch)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("xformers")
    assert len(run.calls) == 1
    assert _XFORMERS_WHEEL in run.calls[0]


def test_xformers_resolution_skipped_when_already_installed(monkeypatch):
    """find_spec runs first, so a working xformers never pays for the torch probe."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    resolved = []
    monkeypatch.setattr(
        att, "_xformers_wheel_target", lambda: resolved.append(1) or (None, "x")
    )
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)

    assert att._ensure_attention_backend_installed("xformers") is None
    assert resolved == []
    assert run.calls == []


def test_matched_wheel_path_does_not_touch_other_backends(monkeypatch):
    """sage / flash / kernels still go to pip by name -- only xFormers has the silent
    ABI failure that forces URL resolution."""
    assert att._MATCHED_WHEEL_BACKENDS == frozenset({"xformers"})
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        att, "_xformers_wheel_target", lambda: pytest.fail("resolver must not run for sage")
    )
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1
    assert "sageattention>=2.1.1" in run.calls[0]


def test_failed_install_not_retried_in_same_process(monkeypatch):
    # The loader pre-installs the kernel OUTSIDE its locks and re-resolves under _generate_lock; a failed pre-install must NOT
    # re-run pip in the lock (a second 600s install would block unload/cancel), so the once-per-process memo no-ops the retry.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # stays missing

    calls: list[list[str]] = []

    def _boom(cmd, **kwargs):
        calls.append(list(cmd))
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    att._ensure_attention_backend_installed("sage")  # pre-install attempt (outside lock)
    att._ensure_attention_backend_installed("sage")  # in-lock retry -> must be skipped
    assert len(calls) == 1


def test_install_invalidates_import_caches_on_success(monkeypatch):
    # A wheel written to site-packages after the finder cached that directory can be missed by the very next import, so a successful install invalidates the caches.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_subprocess(monkeypatch, _Recorder())
    invalidated = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))
    att._ensure_attention_backend_installed("sage")
    assert invalidated == [True]


def test_install_failure_skips_cache_invalidation(monkeypatch):
    # A failed install left nothing to import, so the finder caches must be left alone.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    invalidated = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))
    att._ensure_attention_backend_installed("sage")
    assert invalidated == []


def test_install_never_attempted_for_builtin_backends(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("_native_cudnn")
    att._ensure_attention_backend_installed("native")
    assert run.calls == []


def test_install_failure_logs_pip_stderr(monkeypatch):
    # A CalledProcessError's str() hides the pip reason, so the warning must surface the captured stderr.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(
            returncode = 1, cmd = cmd, stderr = b"ERROR: No matching distribution found"
        )

    _stub_subprocess(monkeypatch, _boom)

    warnings: list[str] = []

    class _Logger:
        def info(self, *a, **k):
            pass

        def warning(self, msg, *args):
            warnings.append(msg % args if args else msg)

    att._ensure_attention_backend_installed("sage", _Logger())
    assert warnings and "No matching distribution found" in warnings[-1]


def test_install_failure_falls_back_to_native(monkeypatch):
    # pip failing (no wheel for this platform) must not break the load: apply proceeds, set_attention_backend raises, and the dispatcher is restored to native.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


# ── kernels-package install gate (huggingface_hub compatibility) ─────────────────


def test_kernels_install_skipped_on_old_hub(monkeypatch):
    # Current `kernels` wheels require huggingface_hub >= 1.10, and with an older hub the damage
    # is NOT contained to the requested backend: `import kernels` raises at module scope and
    # diffusers imports kernels whenever it is installed, so a single auto-install would break
    # every later pipeline import on the box. The installer must refuse.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert run.calls == []
    # The refusal is a policy decision, not a failed attempt: nothing is memoised, so a later
    # request on a fixed environment can still install.
    assert "kernels" not in att._INSTALL_ATTEMPTED


def test_kernels_install_allowed_on_supported_hub(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert len(run.calls) == 1 and "kernels" in run.calls[0]


def test_kernels_gate_only_applies_to_kernels_package(monkeypatch):
    # sage / xformers / flash-attn wheels do not import huggingface_hub at module scope, so the
    # hub gate must not block them.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1 and any("sageattention" in part for part in run.calls[0])


def test_kernels_hub_compatible_reads_hub_version(monkeypatch):
    import importlib.metadata

    # The gate is (major, minor) >= 1.10, the floor the kernels wheels declare -- a major-only
    # check would pass every 1.x. Measured with kernels 0.16.0: hub 1.0.0-1.2.4 fail
    # `import kernels` outright (strict dataclasses gained `str | None` in hub 1.3.0), and
    # 1.3-1.9 sit below the supported floor, so the whole sub-1.10 range is refused.
    for bad in ("0.36.2", "1.0.0", "1.2.4", "1.3.5", "1.9.2", "1.9"):
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v = bad: v)
        assert att._kernels_hub_compatible() is False, bad
    for good in ("1.10.0", "1.10.0rc0", "1.23.0", "2.0.0"):
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v = good: v)
        assert att._kernels_hub_compatible() is True, good

    def _boom(name):
        raise importlib.metadata.PackageNotFoundError(name)

    # Undeterminable hub -> keep the previous (permissive) behaviour.
    monkeypatch.setattr(importlib.metadata, "version", _boom)
    assert att._kernels_hub_compatible() is True


def test_transient_resolution_failure_is_not_memoised(monkeypatch):
    """A probe that times out on a loaded box is transient. Caching it would turn one
    hiccup into "no xFormers for the rest of this Studio session", so only DETERMINISTIC
    answers (a URL, or a refusal that depends purely on the resident torch) are cached."""
    import utils.wheel_utils as wu

    calls = []

    def _probe(**kwargs):
        calls.append(kwargs)
        return None if len(calls) == 1 else {
            "platform_tag": "win_amd64",
            "python_tag": "cp313",
            "torch_version": "2.10.0+cu130",
            "cuda_version": "13.0",
        }

    monkeypatch.setattr(wu, "probe_torch_wheel_env", _probe)

    url, reason = att._xformers_wheel_target()
    assert url is None and "could not be probed" in reason
    assert att._XFORMERS_WHEEL_TARGET is None, "a transient failure must not be cached"

    url, reason = att._xformers_wheel_target()
    assert reason is None and url == _XFORMERS_WHEEL
    assert att._XFORMERS_WHEEL_TARGET == (_XFORMERS_WHEEL, None)
    assert len(calls) == 2


def test_deterministic_refusal_is_memoised(monkeypatch):
    """torch cannot change under a running interpreter, so "no wheel for this torch" is
    settled; re-probing it on every request would re-pay the subprocess under the lock."""
    import utils.wheel_utils as wu

    calls = []

    def _probe(**kwargs):
        calls.append(kwargs)
        return {
            "platform_tag": "win_amd64",
            "python_tag": "cp313",
            "torch_version": "2.11.0+cu130",
            "cuda_version": "13.0",
        }

    monkeypatch.setattr(wu, "probe_torch_wheel_env", _probe)

    for _ in range(3):
        url, reason = att._xformers_wheel_target()
        assert url is None
        assert "no xFormers wheel is published for torch 2.11.0+cu130" in reason
    assert len(calls) == 1


def test_resolution_does_no_network_io(monkeypatch):
    """It can run under _generate_lock (the video loader has no out-of-lock pre-install),
    so it must not add a HEAD to a path that already blocks unload/cancel."""
    import utils.wheel_utils as wu

    monkeypatch.setattr(
        wu, "probe_torch_wheel_env",
        lambda **kw: {
            "platform_tag": "win_amd64",
            "python_tag": "cp313",
            "torch_version": "2.10.0+cu130",
            "cuda_version": "13.0",
        },
    )
    monkeypatch.setattr(
        wu, "url_exists", lambda url: pytest.fail("resolution must not hit the network")
    )
    assert att._xformers_wheel_target() == (_XFORMERS_WHEEL, None)


def test_probe_timeout_matches_the_other_wheel_callers(monkeypatch):
    import utils.wheel_utils as wu

    seen = {}

    def _probe(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(wu, "probe_torch_wheel_env", _probe)
    att._xformers_wheel_target()
    assert seen == {"timeout": 30, "include_windows": True}
