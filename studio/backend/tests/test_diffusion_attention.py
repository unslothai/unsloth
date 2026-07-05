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
