# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Attention-backend tests run without torch: _is_cuda_nvidia is patched, the apply path faked."""

from __future__ import annotations

import sys
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
    with pytest.raises(ValueError):
        normalize_attention_backend("flash-3")


def test_sdpa_alias_maps_to_native():
    assert select_attention_backend(_target(), "sdpa", speed_active = True) is None


def test_auto_upgrades_to_cudnn_on_nvidia_when_speed_active(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))
    assert select_attention_backend(_target(), "auto", speed_active = True) == "_native_cudnn"


def test_auto_does_not_pin_cudnn_below_sm80(monkeypatch):
    # cuDNN fused SDPA fails at run time on pre-SM80 (T4 / V100).
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))
    assert select_attention_backend(_target(), "auto", speed_active = True) is None


def test_auto_stays_native_when_speed_off(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "auto", speed_active = False) is None


def test_auto_stays_native_off_nvidia(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "auto", speed_active = True) is None


def test_explicit_backend_honored_regardless_of_speed(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "sage", speed_active = False) == "sage"
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


def test_explicit_backend_dropped_off_nvidia_cuda(monkeypatch):
    # Explicit cuDNN/flash/sage on ROCm/MPS/CPU passes diffusers' set-time check, then crashes.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    for alias in ("sage", "flash", "flash4", "cudnn"):
        assert select_attention_backend(_target(device = "mps"), alias, speed_active = True) is None


def test_aiter_honored_on_rocm(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(), "aiter", speed_active = False) == "aiter"


def test_aiter_dropped_off_rocm(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "aiter", speed_active = False) is None
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "aiter", speed_active = False) is None


def test_explicit_native_returns_none():
    assert select_attention_backend(_target(), "native", speed_active = True) is None


def test_flash3_dropped_below_hopper(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 9))
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None


def test_flash4_dropped_below_blackwell(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    assert select_attention_backend(_target(), "flash4", speed_active = False) is None
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_arch_gate_does_not_block_when_capability_unknown(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: None)
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"


def test_flash3_dropped_on_blackwell(monkeypatch):
    # FA3 is Hopper-only (no Blackwell kernel): it would set fine on B200, then crash.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_explicit_cudnn_dropped_below_sm80(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) is None
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


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
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to is None


def test_apply_none_restores_native_when_global_polluted(monkeypatch):
    # A previous load pinned cuDNN process-wide; a native load must reset it.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to == "native"


def test_apply_sets_backend():
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"


def test_apply_sets_backend_on_both_dits():
    # Ideogram runs both DiTs each step, so the backend must be set on both.
    t1, t2 = _FakeTransformer(), _FakeTransformer()
    pipe = types.SimpleNamespace(transformer = t1, unconditional_transformer = t2)
    engaged = apply_attention_backend(pipe, "_native_cudnn")
    assert engaged == "_native_cudnn"
    assert t1.set_to == "_native_cudnn" and t2.set_to == "_native_cudnn"


def test_apply_falls_back_on_unavailable_kernel(monkeypatch):
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


def test_apply_failed_kernel_restores_native_when_polluted(monkeypatch):
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")
    monkeypatch.setattr(att, "_sage_version_too_old", lambda: None)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: True)
    monkeypatch.setattr(att, "_install_sage_dispatch_guard", lambda: True)

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
    called = {"reset": False}
    monkeypatch.setattr(
        att, "_reset_global_backend_to_native", lambda logger: called.__setitem__("reset", True)
    )
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"
    assert called["reset"] is True


def test_active_attention_backend_reads_tuple_return():
    # get_active_backend() returns (AttentionBackendName, fn); read the name's .value.
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
    assert att._active_attention_backend() == "native"


@pytest.fixture(autouse = True)
def _no_real_installs(monkeypatch):
    # Never shell out to pip in unit tests; install tests re-enable with a stubbed subprocess.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    att._INSTALL_ATTEMPTED.clear()
    # The resolved xFormers wheel is cached too; left set, one test's stub decides the next.
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
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: True)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls == []


def test_install_runs_wheel_only_for_missing_kernel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert "--only-binary" in cmd and ":all:" in cmd


def test_sage_install_never_asks_pypi_for_sageattention(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls and not any("sageattention" in part for cmd in run.calls for part in cmd)
    assert run.calls[0][-1] == "kernels"


_XFORMERS_WHEEL = "https://download.pytorch.org/whl/cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl"


def _stub_xformers_wheel(
    monkeypatch,
    url = _XFORMERS_WHEEL,
    reason = None,
):
    """Pin the resolved xFormers wheel so no test probes the real torch."""
    monkeypatch.setattr(att, "_xformers_wheel_target", lambda: (url, reason))


def test_install_uses_no_deps_to_protect_core_deps(monkeypatch):
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
    """Install xformers from a URL matched to the running torch; the PyPI wheel is CUDA 12.8 only."""
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


def test_a_hidden_xformers_is_not_reinstalled_in_the_same_process(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    monkeypatch.setitem(sys.modules, "xformers", None)

    def _no_resolve():
        raise AssertionError("resolved a wheel for a hidden xformers")

    monkeypatch.setattr(att, "_xformers_wheel_target", _no_resolve)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    logger = _CapturingLogger()

    reason = att._ensure_attention_backend_installed("xformers", logger)

    assert reason and "requires a different torch" in reason
    assert run.calls == []
    assert any("restart Studio" in line for line in logger.lines)


def test_the_hidden_xformers_refusal_records_no_attempt(monkeypatch):
    """Policy, not a failed attempt: once the entry is gone the same process may still install."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    monkeypatch.setitem(sys.modules, "xformers", None)
    att._ensure_attention_backend_installed("xformers")
    monkeypatch.delitem(sys.modules, "xformers")

    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("xformers")
    assert len(run.calls) == 1


_CREDENTIALED_WHEEL = (
    "https://svc:s3cr3t@mirror.internal/whl/cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl"
)
_REDACTED_WHEEL = "https://mirror.internal/whl/cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl"


class _CapturingLogger:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.lines.append(msg % args if args else msg)


def test_a_private_mirrors_credentials_never_reach_the_log(monkeypatch):
    """UNSLOTH_PYTORCH_MIRROR is allowed to be an authenticated index, and the wheel URL is
    built from it -- so logging the URL as the package name wrote the token into the backend
    log on every install. pip is still handed the real URL; only the log is redacted, and
    pip's own stderr echoes the URL back, so that is redacted too."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_xformers_wheel(monkeypatch, url = _CREDENTIALED_WHEEL)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(
            returncode = 1,
            cmd = cmd,
            stderr = f"ERROR: Could not install {_CREDENTIALED_WHEEL} (404)".encode(),
        )

    _stub_subprocess(monkeypatch, _boom)
    logger = _CapturingLogger()

    att._ensure_attention_backend_installed("xformers", logger)

    assert logger.lines
    for line in logger.lines:
        assert "s3cr3t" not in line
        assert _REDACTED_WHEEL in line


def test_a_query_string_token_is_stripped_too():
    redacted = att._redacted_for_log(
        "ERROR: HTTP 403 for https://mirror.internal/whl/cu130/x.whl?token=abc123 -- retry"
    )
    assert "abc123" not in redacted
    assert "https://mirror.internal/whl/cu130/x.whl" in redacted


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
    monkeypatch.setattr(att, "_xformers_wheel_target", lambda: resolved.append(1) or (None, "x"))
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)

    assert att._ensure_attention_backend_installed("xformers") is None
    assert resolved == []
    assert run.calls == []


def test_matched_wheel_path_does_not_touch_other_backends(monkeypatch):
    """sage (via kernels) / flash / kernels still go to pip by name -- only xFormers has the silent
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
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1
    assert run.calls[0][-1] == "kernels"


def test_failed_install_not_retried_in_same_process(monkeypatch):
    # A failed pre-install must not re-run pip under _generate_lock (600s would block unload).
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    calls: list[list[str]] = []

    def _boom(cmd, **kwargs):
        calls.append(list(cmd))
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    att._ensure_attention_backend_installed("flash")
    att._ensure_attention_backend_installed("flash")
    assert len(calls) == 1


def test_install_invalidates_import_caches_on_success(monkeypatch):
    # A wheel written after the finder cached site-packages can be missed: invalidate caches.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_subprocess(monkeypatch, _Recorder())
    invalidated = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))
    att._ensure_attention_backend_installed("flash")
    assert invalidated == [True]


def test_install_failure_skips_cache_invalidation(monkeypatch):
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
    # CalledProcessError's str() hides the pip reason; surface the captured stderr.
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

    att._ensure_attention_backend_installed("flash", _Logger())
    assert warnings and "No matching distribution found" in warnings[-1]


def test_install_failure_falls_back_to_native(monkeypatch):
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


def test_kernels_install_skipped_on_old_hub(monkeypatch):
    # kernels needs huggingface_hub >= 1.10; older hubs make `import kernels` raise at module
    # scope, and diffusers imports kernels whenever installed, breaking every pipeline import.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("flash_4_hub")
    assert run.calls == []
    # A policy refusal is not memoised, so a fixed environment can still install later.
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


def test_kernels_gate_also_covers_sage(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: False)
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    assert att._ensure_attention_backend_installed("sage") is not None
    assert run.calls == []
    att._ensure_attention_backend_installed("flash")
    assert len(run.calls) == 1 and any("flash-attn" in part for part in run.calls[0])


def test_kernels_hub_compatible_reads_hub_version(monkeypatch):
    import importlib.metadata

    # Gate is (major, minor) >= 1.10: hub 1.0-1.2 fail `import kernels`, 1.3-1.9 are below the floor.
    for bad in ("0.36.2", "1.0.0", "1.2.4", "1.3.5", "1.9.2", "1.9"):
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v = bad: v)
        assert att._kernels_hub_compatible() is False, bad
    for good in ("1.10.0", "1.10.0rc0", "1.23.0", "2.0.0"):
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v = good: v)
        assert att._kernels_hub_compatible() is True, good

    def _boom(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _boom)
    assert att._kernels_hub_compatible() is True


def test_transient_resolution_failure_is_not_memoised(monkeypatch):
    """A probe that times out on a loaded box is transient. Caching it would turn one
    hiccup into "no xFormers for the rest of this Unsloth session", so only DETERMINISTIC
    answers (a URL, or a refusal that depends purely on the resident torch) are cached."""
    import utils.wheel_utils as wu

    calls = []

    def _probe(**kwargs):
        calls.append(kwargs)
        return (
            None
            if len(calls) == 1
            else {
                "platform_tag": "win_amd64",
                "python_tag": "cp313",
                "torch_version": "2.10.0+cu130",
                "cuda_version": "13.0",
            }
        )

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
            "torch_version": "2.11.0+cu124",
            "cuda_version": "12.4",
        }

    monkeypatch.setattr(wu, "probe_torch_wheel_env", _probe)

    for _ in range(3):
        url, reason = att._xformers_wheel_target()
        assert url is None
        assert "no xFormers wheel is published for torch 2.11.0+cu124" in reason
    assert len(calls) == 1


def test_resolution_does_no_network_io(monkeypatch):
    """It can run under _generate_lock (the video loader has no out-of-lock pre-install),
    so it must not add a HEAD to a path that already blocks unload/cancel."""
    import utils.wheel_utils as wu

    monkeypatch.setattr(
        wu,
        "probe_torch_wheel_env",
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


@pytest.fixture(autouse = True)
def _clear_sdpa_probe_cache():
    """The probe memoises per (device, dtype) for the process; tests must not inherit it."""
    att._SDPA_PROBE_CACHE.clear()
    yield
    att._SDPA_PROBE_CACHE.clear()


def _stub_probe(
    monkeypatch,
    kernels,
    *,
    record = None,
):
    def _probe(device, dtype):
        if record is not None:
            record.append((device, dtype))
        return tuple(kernels)

    monkeypatch.setattr(att, "_probe_sdpa_kernels", _probe)
    # Importing unsloth on SM100 turns cuDNN SDPA off process-wide; these tests pin the probe alone.
    monkeypatch.setattr(att, "_enabled_sdpa_kernels", lambda found: found)


def test_math_only_is_read_off_a_probe_not_the_flags(monkeypatch):
    # Some ROCm builds (gfx1200) report flash/mem-efficient enabled but every dispatch raises,
    # so dispatch degrades to math (N x N scores). Only a probe sees that.
    _stub_probe(monkeypatch, ("math",))
    assert att.sdpa_math_only(_target()) is True
    assert att.available_sdpa_kernels(_target()) == ("math",)


@pytest.mark.parametrize(
    "kernels",
    [
        ("flash", "mem_efficient", "cudnn", "math"),
        ("flash", "math"),
        ("mem_efficient", "math"),
        ("cudnn", "math"),
    ],
)
def test_any_subquadratic_kernel_means_not_math_only(kernels, monkeypatch):
    _stub_probe(monkeypatch, kernels)
    assert att.sdpa_math_only(_target()) is False


def test_a_fused_launch_that_fails_late_is_not_reported_available(monkeypatch):
    # Windows ROCm gfx1151: hipErrorInvalidValue surfaces on the next checked kernel; probe takes it.
    torch = pytest.importorskip("torch")

    class _Pending:
        def float(self):
            raise RuntimeError("CUDA error: invalid argument")

    real = torch.nn.functional.scaled_dot_product_attention

    def _sdpa(q, k, v, *a, **kw):
        if not torch.backends.cuda.math_sdp_enabled():
            return _Pending()
        return real(q, k, v, *a, **kw)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _sdpa)
    assert att._probe_sdpa_kernels("cpu", torch.float32) == ("math",)


def test_an_unanswerable_probe_is_not_a_math_only_verdict(monkeypatch):
    _stub_probe(monkeypatch, ())
    assert att.sdpa_math_only(_target()) is False
    assert att.warn_if_sdpa_math_only(_target(), None) is False


def test_a_probe_that_raises_is_swallowed(monkeypatch):
    def _boom(device, dtype):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(att, "_probe_sdpa_kernels", _boom)
    assert att.available_sdpa_kernels(_target()) == ()
    assert att.sdpa_math_only(_target()) is False


def test_a_target_with_no_device_is_never_probed(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, ("math",), record = seen)
    assert att.available_sdpa_kernels(types.SimpleNamespace(device = "", dtype = None)) == ()
    assert att.available_sdpa_kernels(types.SimpleNamespace(device = None, dtype = None)) == ()
    assert seen == []


def test_the_probe_runs_once_per_device_and_dtype(monkeypatch):
    # A kernel cannot appear or vanish under a running interpreter, and this sits on the load
    # path, so the probe is memoised. each indexed GPU has its own capability result.
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    seen: list = []
    _stub_probe(monkeypatch, ("math",), record = seen)
    for device in ("cuda", "cuda:0", "cuda:1"):
        assert att.sdpa_math_only(types.SimpleNamespace(device = device, dtype = "fp16")) is True
    assert len(seen) == 2
    # A different dtype is a different question: fused kernels are half-precision only.
    att.sdpa_math_only(types.SimpleNamespace(device = "cuda", dtype = "fp32"))
    assert len(seen) == 3


def test_a_dtypeless_target_probes_in_half_precision(monkeypatch):
    # fp32 has no fused kernel anywhere, so probing at fp32 would always report math only.
    import torch

    seen: list = []
    _stub_probe(monkeypatch, ("flash", "math"), record = seen)
    att.available_sdpa_kernels(types.SimpleNamespace(device = "cuda", dtype = None))
    assert seen == [(att._indexed_cuda_device("cuda"), torch.float16)]


def test_warn_names_the_quadratic_cost(monkeypatch):
    _stub_probe(monkeypatch, ("math",))
    logged: list = []
    logger = types.SimpleNamespace(warning = lambda fmt, *a: logged.append(fmt % a))

    assert att.warn_if_sdpa_math_only(_target(), logger) is True
    assert len(logged) == 1
    assert "math_only" in logged[0]
    assert "score matrix" in logged[0] and "SQUARE" in logged[0]


def test_warn_is_silent_on_a_healthy_device(monkeypatch):
    _stub_probe(monkeypatch, ("flash", "math"))
    logged: list = []
    logger = types.SimpleNamespace(warning = lambda fmt, *a: logged.append(fmt % a))
    assert att.warn_if_sdpa_math_only(_target(), logger) is False
    assert logged == []


def test_apply_native_reports_a_math_only_device(monkeypatch):
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    _stub_probe(monkeypatch, ("math",))
    logged: list = []
    logger = types.SimpleNamespace(
        warning = lambda fmt, *a: logged.append(fmt % a),
        info = lambda *a, **k: None,
    )

    assert (
        apply_attention_backend(_pipe(_FakeTransformer()), None, logger = logger, target = _target())
        is None
    )
    assert len(logged) == 1 and "math_only" in logged[0]


def test_apply_reports_a_math_only_device_for_a_unet_pipeline(monkeypatch):
    # SDXL's pipe.unet has no dispatcher setter but runs the same SDPA, so still diagnose it.
    _stub_probe(monkeypatch, ("math",))
    logged: list = []
    logger = types.SimpleNamespace(
        warning = lambda fmt, *a: logged.append(fmt % a),
        info = lambda *a, **k: None,
    )
    unet_pipe = types.SimpleNamespace(unet = object())

    assert apply_attention_backend(unet_pipe, None, logger = logger, target = _target()) is None
    assert len(logged) == 1 and "math_only" in logged[0]


def test_apply_without_a_target_probes_nothing(monkeypatch):
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    seen: list = []
    _stub_probe(monkeypatch, ("math",), record = seen)
    assert apply_attention_backend(_pipe(_FakeTransformer()), None) is None
    assert seen == []


def test_apply_does_not_warn_when_a_real_backend_engaged(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, ("math",), record = seen)
    logged: list = []
    logger = types.SimpleNamespace(
        warning = lambda fmt, *a: logged.append(fmt % a),
        info = lambda *a, **k: None,
    )
    t = _FakeTransformer()
    assert (
        apply_attention_backend(_pipe(t), "_native_cudnn", logger = logger, target = _target())
        == "_native_cudnn"
    )
    assert seen == [] and logged == []


def test_an_unanswerable_probe_is_not_memoised(monkeypatch):
    """A probe that could not RUN is not an answer about the hardware. Caching it disabled the
    warning for the rest of the process, including after the memory that broke it was freed --
    and a device under memory pressure is exactly when this warning matters."""
    calls: list = []

    def _flaky(device, dtype):
        calls.append(dtype)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory while allocating the probe tensor")
        return ("flash", "math")

    monkeypatch.setattr(att, "_probe_sdpa_kernels", _flaky)

    assert att.available_sdpa_kernels(_target()) == ()
    assert att.sdpa_math_only(_target()) is False
    assert att.available_sdpa_kernels(_target()) == ("flash", "math")
    assert att.available_sdpa_kernels(_target()) == ("flash", "math")
    assert len(calls) == 2, "the answer must be memoised; the failure must not be"
