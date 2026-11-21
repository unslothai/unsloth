# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The canonical fingerprint is the same whether the weights are hashed serially or on the thread pool, and the
zero-copy digest equals md5 over ``_canonical_bytes`` (the recorded md5-canonical-v2 bytes)."""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
torchao_quant = pytest.importorskip("torchao.quantization")

from core.inference import diffusion_prequant as dp  # noqa: E402


def _int8_state_dict(n = 6, d = 512):
    from core.inference.diffusion_transformer_quant import _make_quant_config

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(d, d, bias = False, dtype = torch.bfloat16) for _ in range(n)]
    )
    torchao_quant.quantize_(model, _make_quant_config("int8"))
    return model.state_dict()


def test_threaded_fingerprint_equals_serial(monkeypatch):
    sd = _int8_state_dict()
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, "1")
    serial = dp.canonical_weight_fingerprint(sd)
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, "4")
    threaded = dp.canonical_weight_fingerprint(sd)
    assert serial["count"] == 6 and not serial["skipped"]
    assert threaded == serial
    assert list(threaded[dp.CANONICAL_FINGERPRINT_KEY]) == list(
        serial[dp.CANONICAL_FINGERPRINT_KEY]
    )


def test_threaded_packed_v1_fingerprint_equals_serial(monkeypatch):
    sd = _int8_state_dict()
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, "1")
    serial = dp.packed_weight_fingerprint(sd)
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, "4")
    threaded = dp.packed_weight_fingerprint(sd)
    assert serial["count"] == 6 and not serial["skipped"]
    assert threaded == serial


def test_fingerprint_runs_on_the_pool(monkeypatch):
    import threading

    sd = _int8_state_dict(n = 4, d = 256)
    seen = set()
    real = dp.canonical_weight_digest

    def _spy(tensor):
        seen.add(threading.current_thread().name)
        return real(tensor)

    monkeypatch.setattr(dp, "canonical_weight_digest", _spy)
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, "2")
    dp.canonical_weight_fingerprint(sd)
    assert seen and all(name.startswith("prequant-fp") for name in seen)


def test_streamed_digest_equals_canonical_bytes():
    sd = _int8_state_dict(n = 1)
    weight = next(v for k, v in sd.items() if k.endswith(".weight"))
    parts = dp._canonical_parts(weight)
    ref = hashlib.md5()
    for role in dp._CANONICAL_ROLE_ORDER:
        value = parts.get(role)
        if value is None or (role in dp._ZERO_POINT_ROLES and dp._all_zero(value, torch)):
            continue
        ref.update(dp._canonical_bytes(role, value, torch))
    assert dp.canonical_weight_digest(weight) == ref.hexdigest()


@pytest.mark.parametrize("raw, want", [("1", 1), ("bogus", None), ("", None)])
def test_thread_count_env(monkeypatch, raw, want):
    monkeypatch.setenv(dp.FINGERPRINT_THREADS_ENV, raw)
    got = dp._fingerprint_threads()
    assert got >= 1 and (want is None or got == want)
