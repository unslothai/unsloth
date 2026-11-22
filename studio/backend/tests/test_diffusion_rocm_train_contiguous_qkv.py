# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""gfx1151 training: contiguous q/k/v at or over the token floor. CPU tensors stand in for the card (index None)."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.attention_dispatch import (  # noqa: E402
    AttentionBackendName,
    _AttentionBackendRegistry,
)

from core.inference import diffusion_attention as da  # noqa: E402
from core.training import diffusion_train_common as dtc  # noqa: E402


@pytest.fixture
def registry(monkeypatch):
    backends = _AttentionBackendRegistry._backends
    monkeypatch.setitem(
        backends, AttentionBackendName.NATIVE, backends[AttentionBackendName.NATIVE]
    )
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {})
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {})
    monkeypatch.delenv(da.ROCM_CONTIGUOUS_QKV_TRAIN_ENV, raising = False)
    monkeypatch.delenv(da.ROCM_CONTIGUOUS_KV_ENV, raising = False)
    return backends


def _fake_cards(
    monkeypatch,
    archs = ("gfx1151",),
    version = "2.11.0+rocm7.13.0",
    hip = None,
):
    """An AMD SDK wheel by default: ``version.hip`` unset, the ROCm tag only in ``__version__``."""
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__version__", version, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(archs))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(gcnArchName = archs[i] + ":sramecc-:xnack-", L2_cache_size = 2 << 20),
    )


@pytest.mark.parametrize(
    "archs, version, hip",
    [
        (("gfx1100",), "2.11.0+rocm7.13.0", None),
        (("gfx1201",), "2.11.0", "7.1"),
        (("gfx942",), "2.11.0+rocm7.13.0", None),
        (("gfx1151",), "2.11.0+cu130", None),  # NVIDIA build: arch strings never consulted
    ],
)
def test_other_gpus_and_builds_keep_the_stock_native_backend(
    registry, monkeypatch, archs, version, hip
):
    _fake_cards(monkeypatch, archs, version, hip)
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_qkv_training() == ()
    assert registry[AttentionBackendName.NATIVE] is native
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {}


def test_no_gpu_keeps_the_stock_native_backend(registry, monkeypatch):
    _fake_cards(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_qkv_training() == ()
    assert registry[AttentionBackendName.NATIVE] is native


def test_tag_only_rocm_wheel_installs_on_gfx1151_only(registry, monkeypatch):
    _fake_cards(monkeypatch, archs = ("gfx1100", "gfx1151"))
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_qkv_training() == (1,)
    wrapped = registry[AttentionBackendName.NATIVE]
    assert wrapped is not native and wrapped.__wrapped__ is native
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {1: da.ROCM_CONTIGUOUS_QKV_TRAIN_MIN_TOKENS}
    assert da.install_rocm_contiguous_qkv_training() == (1,)  # idempotent: one wrapper
    assert registry[AttentionBackendName.NATIVE] is wrapped
    da.uninstall_rocm_contiguous_qkv_training((1,))
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {}


@pytest.mark.parametrize("off", ["0", "off", "false", "no", " OFF "])
def test_kill_switch(registry, monkeypatch, off):
    _fake_cards(monkeypatch)
    monkeypatch.setenv(da.ROCM_CONTIGUOUS_QKV_TRAIN_ENV, off)
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_qkv_training() == ()
    assert registry[AttentionBackendName.NATIVE] is native


@pytest.mark.parametrize("raw, floor", [("", 4096), ("auto", 4096), ("8192", 8192), ("1", 1)])
def test_env_moves_the_token_floor(registry, monkeypatch, raw, floor):
    _fake_cards(monkeypatch)
    monkeypatch.setenv(da.ROCM_CONTIGUOUS_QKV_TRAIN_ENV, raw)
    assert da.install_rocm_contiguous_qkv_training() == (0,)
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {0: floor}


def test_threshold_keeps_short_and_cross_attention_on_the_current_path(monkeypatch):
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {None: 4096})
    at, below, text = (torch.zeros(1, s, 2, 8) for s in (4096, 4095, 512))
    assert da._rocm_contiguous_qkv_wanted(at, at, at)
    assert not da._rocm_contiguous_qkv_wanted(
        below, below, below
    )  # a 1,024 px still: 1,024 to 4,095 tokens
    assert not da._rocm_contiguous_qkv_wanted(at, text, text)  # cross-attention to a short text
    assert not da._rocm_contiguous_qkv_wanted(at.flatten(2), at, at)
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {5: 4096})
    assert not da._rocm_contiguous_qkv_wanted(
        at, at, at
    )  # the tensor's card is not a covered gfx1151
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {})
    assert not da._rocm_contiguous_qkv_wanted(at, at, at)


def test_wrapper_hands_sdpa_contiguous_qkv_with_identical_outputs_and_grads(registry, monkeypatch):
    _fake_cards(monkeypatch)
    da.install_rocm_contiguous_qkv_training()
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {None: 64})
    wrapped = registry[AttentionBackendName.NATIVE]
    seen = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        q, k, v = args[:3] if len(args) >= 3 else (kwargs["query"], kwargs["key"], kwargs["value"])
        seen.append((q.is_contiguous(), k.is_contiguous(), v.is_contiguous()))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)

    def leaves(s):
        g = torch.Generator().manual_seed(0)
        return [torch.randn(1, s, 2, 16, generator = g).requires_grad_() for _ in range(3)]

    def run(fn, s, **kw):
        q, k, v = leaves(s)
        out = fn(q, k, v, **kw) if not kw.pop("by_name", False) else fn(query = q, key = k, value = v)
        out.square().sum().backward()
        return out.detach(), [t.grad for t in (q, k, v)]

    ref, ref_grads = run(wrapped.__wrapped__, 64)
    assert seen.pop() == (False, False, False)
    for by_name in (False, True):
        out, grads = run(wrapped, 64, by_name = by_name)
        assert seen.pop() == (True, True, True)
        assert torch.equal(out, ref)
        assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))
    run(wrapped, 63)
    assert seen.pop() == (False, False, False)  # under the floor: untouched


def test_inference_kv_gate_still_copies_only_kv(registry, monkeypatch):
    _fake_cards(monkeypatch)
    da.install_rocm_contiguous_qkv_training()
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_QKV_TRAIN", {})
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {None: 1 << 12})
    wrapped = registry[AttentionBackendName.NATIVE]
    seen = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(tuple(kwargs[n].is_contiguous() for n in ("query", "key", "value")))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    q, k, v = (torch.randn(1, 64, 2, 16) for _ in range(3))
    wrapped(q, k, v)
    assert seen.pop() == (False, True, True)


def test_trainer_perf_flags_install_and_restore(registry, monkeypatch):
    _fake_cards(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = SimpleNamespace(enable_tf32 = True)
    snap = dtc._apply_perf_flags(cfg, "cuda")
    assert snap.get("rocm_contiguous_qkv") == (0,)
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {0: da.ROCM_CONTIGUOUS_QKV_TRAIN_MIN_TOKENS}
    dtc._restore_perf_flags(snap)
    assert da._ROCM_CONTIGUOUS_QKV_TRAIN == {}
    assert dtc._apply_perf_flags(cfg, "cpu").get("rocm_contiguous_qkv") is None
