# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""gfx1151 contiguous k/v wrapper. CPU tensors stand in for the card (L2 table key is None on CPU)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.attention_dispatch import (  # noqa: E402
    AttentionBackendName,
    _AttentionBackendRegistry,
)

from core.inference import diffusion_attention as da  # noqa: E402

TARGET = SimpleNamespace(device = "cuda", torch_device = "cuda:0", ordinal = 0)


@pytest.fixture
def registry(monkeypatch):
    backends = _AttentionBackendRegistry._backends
    monkeypatch.setitem(
        backends, AttentionBackendName.NATIVE, backends[AttentionBackendName.NATIVE]
    )
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {})
    monkeypatch.delenv(da.ROCM_CONTIGUOUS_KV_ENV, raising = False)
    return backends


def _fake_card(
    monkeypatch,
    arch = "gfx1151",
    rocm = True,
    l2 = 2 * 1024 * 1024,
):
    monkeypatch.setattr(da, "_is_cuda_rocm", lambda target: rocm)
    monkeypatch.setattr(da, "_rocm_gfx_arch", lambda target: arch)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(L2_cache_size = l2)
    )


@pytest.mark.parametrize(
    "arch, rocm", [("gfx1100", True), ("gfx1201", True), ("gfx942", True), ("", False)]
)
def test_other_gpus_keep_the_stock_native_backend(registry, monkeypatch, arch, rocm):
    _fake_card(monkeypatch, arch = arch, rocm = rocm)
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_kv(TARGET) is False
    assert registry[AttentionBackendName.NATIVE] is native


def test_kill_switch(registry, monkeypatch):
    _fake_card(monkeypatch)
    monkeypatch.setenv(da.ROCM_CONTIGUOUS_KV_ENV, "0")
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_kv(TARGET) is False
    assert registry[AttentionBackendName.NATIVE] is native


def test_installs_once_on_gfx1151(registry, monkeypatch):
    _fake_card(monkeypatch)
    native = registry[AttentionBackendName.NATIVE]
    assert da.install_rocm_contiguous_kv(TARGET) is True
    wrapped = registry[AttentionBackendName.NATIVE]
    assert wrapped is not native and wrapped.__wrapped__ is native
    assert da.install_rocm_contiguous_kv(TARGET) is True
    assert registry[AttentionBackendName.NATIVE] is wrapped
    assert da._ROCM_CONTIGUOUS_KV_L2 == {0: 2 * 1024 * 1024}


def test_size_gate_matches_one_head_of_strided_kv_against_l2(monkeypatch):
    l2 = 1 << 16
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {None: l2})
    d = 32
    edge = l2 // (2 * d * 4)  # fp32 k + v of one head exactly fill L2
    at = torch.zeros(1, edge, 2, d)
    over = torch.zeros(1, edge + 1, 2, d)
    assert not da._rocm_contiguous_kv_wanted(at, at)
    assert da._rocm_contiguous_kv_wanted(over, over)
    dense = over.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    assert not da._rocm_contiguous_kv_wanted(dense, dense)
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {5: l2})
    assert not da._rocm_contiguous_kv_wanted(over, over)  # no L2 entry: not gfx1151


def test_wrapper_hands_sdpa_contiguous_kv_and_keeps_outputs(registry, monkeypatch):
    _fake_card(monkeypatch)
    da.install_rocm_contiguous_kv(TARGET)
    monkeypatch.setattr(da, "_ROCM_CONTIGUOUS_KV_L2", {None: 1 << 12})
    wrapped = registry[AttentionBackendName.NATIVE]
    seen = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append((kwargs["key"].is_contiguous(), kwargs["value"].is_contiguous()))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 64, 2, 16, generator = g) for _ in range(3))
    ref = wrapped.__wrapped__(q, k, v)
    assert seen.pop() == (False, False)
    assert torch.equal(wrapped(query = q, key = k, value = v), ref)
    assert seen.pop() == (True, True)
    assert torch.equal(wrapped(q, k, v), ref)  # positional, as Studio's dispatch guards call it
    assert seen.pop() == (True, True)
    small = (t[:, :8] for t in (q, k, v))
    wrapped(*small)
    assert seen.pop() == (False, False)
