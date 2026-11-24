# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image or video load on Apple Silicon (MPS) leaves torch's SDPA wrapped by the large-attention guard: torch below
2.13 returns wrong MPS attention past about 2^29 score elements, and only Qwen-Image-2.1 had its own bound.
Other devices keep the stock function."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_attention as A  # noqa: E402

F = torch.nn.functional


@pytest.fixture
def mac(monkeypatch):
    original = F.scaled_dot_product_attention
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available = lambda: True))
    monkeypatch.setattr(A, "guard_rocm_fused_sdpa", lambda *a, **k: None)
    monkeypatch.setattr(A, "install_rocm_contiguous_kv", lambda *a, **k: False)
    yield original
    F.scaled_dot_product_attention = original


def _load(device):
    target = types.SimpleNamespace(device = device, backend = device, torch_device = device)
    A._apply_attention_backend_impl(types.SimpleNamespace(transformer = None), None, target = target)


def test_mps_load_wraps_sdpa(mac):
    _load("mps")
    wrapped = F.scaled_dot_product_attention
    assert wrapped is not mac
    assert getattr(wrapped, "__wrapped__", None) is mac
    # A later load does not wrap twice.
    _load("mps")
    assert F.scaled_dot_product_attention is wrapped


def test_guarded_sdpa_is_exact_off_mps(mac):
    _load("mps")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, 300, 32, generator = g) for _ in range(3))
    assert torch.equal(F.scaled_dot_product_attention(q, k, v), mac(q, k, v))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_other_devices_keep_stock_sdpa(mac, device):
    _load(device)
    assert F.scaled_dot_product_attention is mac
