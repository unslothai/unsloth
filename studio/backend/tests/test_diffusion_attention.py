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
    # dashes are no longer silently rewritten to underscores -> a dashed alias is rejected.
    with pytest.raises(ValueError):
        normalize_attention_backend("flash-3")


def test_sdpa_alias_maps_to_native():
    # sdpa is an alias for native -> nothing to set on the dispatcher.
    assert select_attention_backend(_target(), "sdpa", speed_active = True) is None


# ── select policy ─────────────────────────────────────────────────────────────────
def test_auto_upgrades_to_cudnn_on_nvidia_when_speed_active(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))  # Ampere+: cuDNN ok
    assert select_attention_backend(_target(), "auto", speed_active = True) == "_native_cudnn"


def test_auto_does_not_pin_cudnn_below_sm80(monkeypatch):
    # cuDNN fused SDPA fails at run time on pre-SM80 (T4 SM75 / V100 SM70); auto must stay
    # on the native default there rather than pin a backend that crashes on first generation.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))  # Turing T4
    assert select_attention_backend(_target(), "auto", speed_active = True) is None


def test_auto_stays_native_when_speed_off(monkeypatch):
    # off must stay bit-identical -> no backend change even on NVIDIA.
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
    # Explicit cuDNN/flash/sage on ROCm / MPS / CPU passes diffusers' set-time check
    # and crashes at the first generation, so selection drops to the native default.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    for alias in ("sage", "flash", "flash4", "cudnn"):
        assert select_attention_backend(_target(device = "mps"), alias, speed_active = True) is None


def test_aiter_honored_on_rocm(monkeypatch):
    # AITER is the AMD ROCm kernel; on a ROCm CUDA target it must be honored, not dropped by
    # the NVIDIA-only guard -- it is the one explicit backend that only ever works on ROCm.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)  # hip build
    assert select_attention_backend(_target(), "aiter", speed_active = False) == "aiter"


def test_aiter_dropped_off_rocm(monkeypatch):
    # aiter on NVIDIA CUDA (or MPS / CPU) is not usable, so it drops to the native default.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)  # NVIDIA
    assert select_attention_backend(_target(), "aiter", speed_active = False) is None
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "aiter", speed_active = False) is None


def test_explicit_native_returns_none():
    # native is the default -> nothing to set.
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
    # Unknown capability (e.g. no CUDA) must not block -> diffusers' set-time check still guards.
    monkeypatch.setattr(att, "_cuda_capability", lambda: None)
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"


def test_flash3_dropped_on_blackwell(monkeypatch):
    # FlashAttention 3 is a Hopper-SM90 rewrite with no Blackwell kernel: an explicit
    # flash3 on a B200 (SM100) must drop to native rather than set fine then crash.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None
    # FA4 is still honored on Blackwell.
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    # flash3 is allowed exactly on Hopper SM90.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_explicit_cudnn_dropped_below_sm80(monkeypatch):
    # An explicit cuDNN request on pre-Ampere (T4 SM75 / V100 SM70) must drop to native,
    # not set fine and crash at first generation -- the same gate the auto path applies.
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
    # Global already native -> no redundant set call, returns None.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to is None


def test_apply_none_restores_native_when_global_polluted(monkeypatch):
    # A previous load pinned cuDNN process-wide; a native load must reset it so it can't
    # silently inherit cuDNN (the bit-identical/off guarantee).
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to == "native"


def test_apply_sets_backend():
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"


def test_apply_sets_backend_on_both_dits():
    # A dual-DiT family (Ideogram) runs transformer + unconditional_transformer each step, so the
    # backend must be set on BOTH; otherwise the second DiT keeps the native default while status
    # reports the requested kernel as engaged.
    t1, t2 = _FakeTransformer(), _FakeTransformer()
    pipe = types.SimpleNamespace(transformer = t1, unconditional_transformer = t2)
    engaged = apply_attention_backend(pipe, "_native_cudnn")
    assert engaged == "_native_cudnn"
    assert t1.set_to == "_native_cudnn" and t2.set_to == "_native_cudnn"


def test_apply_falls_back_on_unavailable_kernel(monkeypatch):
    # an unavailable kernel must not fail the load -> returns None (diffusers default).
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
    # After a successful per-transformer set, the process-wide registry must be reset to
    # native so a later component (unconfigured processors) can't inherit this kernel --
    # while the transformer's own backend stays the engaged one.
    called = {"reset": False}
    monkeypatch.setattr(
        att, "_reset_global_backend_to_native", lambda logger: called.__setitem__("reset", True)
    )
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"
    assert called["reset"] is True


def test_active_attention_backend_reads_tuple_return():
    # get_active_backend() returns a (AttentionBackendName, fn) tuple; the helper must read
    # the name's .value, not stringify the tuple (which never compares equal to a name).
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
    # Unit tests must never shell out to pip: the apply path probes installable
    # backends (sage/flash*), so hard-disable the gate; install tests re-enable it
    # with a stubbed subprocess.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    # The install once-per-process memo is module state; clear it so each test starts
    # with a fresh "not yet attempted" set (otherwise an earlier test's attempt would
    # make a later install a no-op).
    att._INSTALL_ATTEMPTED.clear()


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
    assert "--only-binary" in cmd and ":all:" in cmd and "sageattention" in cmd


def test_install_uses_no_deps_to_protect_core_deps(monkeypatch):
    # A kernel add-on (xformers/flash-attn) pins an exact torch, so a normal install would
    # upgrade/replace the running torch/triton. --no-deps installs only the kernel wheel;
    # an ABI-incompatible one fails to import and falls back to native rather than clobbering
    # the environment's core deps.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("xformers")
    assert len(run.calls) == 1
    assert "--no-deps" in run.calls[0]


def test_failed_install_not_retried_in_same_process(monkeypatch):
    # The loader pre-installs the kernel OUTSIDE its locks and then re-resolves the same
    # backend under _generate_lock; if the pre-install failed (no wheel / offline) the
    # in-lock apply path must NOT re-run pip (a second up-to-600s install holding the load
    # lock blocks unload/cancel). The once-per-process memo makes the retry a no-op.
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
    # A wheel written to site-packages after the finder cached that directory can be
    # missed by the very next import, so a successful install must invalidate the caches
    # (otherwise set_attention_backend imports the missing package and falls back).
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
    # A CalledProcessError's str() hides the pip reason; the warning must surface the
    # captured stderr (decoding bytes) so a fallback to native is diagnosable.
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
    # pip failing (no wheel for this platform) must not break the load: the apply
    # path proceeds, set_attention_backend raises on the missing package, and the
    # dispatcher is restored to native -- same contract as before the hook.
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


def test_kernels_install_skipped_on_pre_1x_hub(monkeypatch):
    # Every current `kernels` release needs huggingface_hub >= 1.0, and with an older
    # hub the damage is NOT contained: `import kernels` raises at module scope and
    # diffusers imports kernels whenever it is installed, so a single auto-install
    # would brick every later pipeline import (measured: hub 0.36 + kernels 0.13/0.16
    # both break the HunyuanVideo-1.5 pipeline import). The installer must refuse.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda logger = None: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert run.calls == []
    # The refusal is a policy decision, not a failed attempt: nothing memoised, so a
    # later request on a fixed environment can still install.
    assert "kernels" not in att._INSTALL_ATTEMPTED


def test_kernels_install_allowed_on_hub_1x(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda logger = None: True)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert len(run.calls) == 1 and "kernels" in run.calls[0]


def test_kernels_gate_only_applies_to_kernels_package(monkeypatch):
    # sage/xformers/flash-attn wheels do not import huggingface_hub at module scope,
    # so the hub gate must not block them.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda logger = None: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1 and "sageattention" in run.calls[0]


def test_kernels_hub_compatible_reads_hub_version(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.36.2")
    assert att._kernels_hub_compatible() is False
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.23.0")
    assert att._kernels_hub_compatible() is True

    def _boom(name):
        raise importlib.metadata.PackageNotFoundError(name)

    # Undeterminable hub -> keep the previous (permissive) behaviour.
    monkeypatch.setattr(importlib.metadata, "version", _boom)
    assert att._kernels_hub_compatible() is True


# ── per-device backend guard (CFG-parallel heterogeneous replica) ─────────────────
def _stub_cuda_capability(monkeypatch, caps):
    """Stub torch.cuda.get_device_capability(idx) from a {idx: (major, minor)} map."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        get_device_capability = lambda idx: caps[idx],
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)


def test_backend_supported_on_device_none_is_always_ok(monkeypatch):
    # None = native: nothing to arch-gate, so any device is fine (even unqueryable).
    assert att.attention_backend_supported_on_device(None, 0) is True


def test_backend_supported_on_device_flash3_hopper_only(monkeypatch):
    # FA3 is SM90 (Hopper) only: supported on the Hopper primary, NOT on a Blackwell replica.
    _stub_cuda_capability(monkeypatch, {0: (9, 0), 1: (10, 0)})
    assert att.attention_backend_supported_on_device("_flash_3_hub", 0) is True
    assert att.attention_backend_supported_on_device("_flash_3_hub", 1) is False


def test_backend_supported_on_device_flash4_blackwell_only(monkeypatch):
    # FA4 needs SM100 (Blackwell): rejected on a Hopper replica.
    _stub_cuda_capability(monkeypatch, {0: (10, 0), 1: (9, 0)})
    assert att.attention_backend_supported_on_device("flash_4_hub", 0) is True
    assert att.attention_backend_supported_on_device("flash_4_hub", 1) is False


def test_backend_supported_on_device_cudnn_needs_ampere(monkeypatch):
    # cuDNN fused SDPA needs Ampere+ (SM80): rejected on a pre-Ampere (T4/SM75) replica.
    _stub_cuda_capability(monkeypatch, {0: (9, 0), 1: (7, 5)})
    assert att.attention_backend_supported_on_device("_native_cudnn", 0) is True
    assert att.attention_backend_supported_on_device("_native_cudnn", 1) is False


def test_backend_supported_on_device_unqueryable_is_permissive(monkeypatch):
    # An unqueryable device must not block on a guess (best-effort, like _backend_arch_supported).
    torch = types.ModuleType("torch")

    def _boom(_idx):
        raise RuntimeError("no device props")

    torch.cuda = types.SimpleNamespace(get_device_capability = _boom)
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    assert att.attention_backend_supported_on_device("_flash_3_hub", 3) is True
