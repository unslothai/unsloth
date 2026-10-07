"""Regression test for unslothai/unsloth#4631: xformers must not be blanket-disabled
on sm_120 GPUs where its kernel actually runs (a ~57% attention-memory saving over the
SDPA packed-mask fallback). The gate now probes the real op instead of guessing by the
compute-capability major version."""

import pytest
from real_accelerator import (
    has_real_cuda,
)
import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


@pytest.mark.parametrize(
    "capability, probe_result, expect_disabled",
    [
        ((8, 9), None, False),
        ((9, 0), None, False),
        ((10, 0), None, False),
        ((12, 0), True, False),
        ((12, 0), False, True),
    ],
)
def test_capability_gate(capability, probe_result, expect_disabled):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert ad._xformers_disabled_for_capability(capability, probe = probe) is expect_disabled
    # Below sm_120 the probe must not run (no import-time kernel launch).
    assert calls["n"] == (0 if capability[0] < 12 else 1)


@pytest.mark.skipif(
    not (has_real_cuda() and ad.HAS_XFORMERS),
    reason = "needs a CUDA GPU with a working xformers build",
)
@pytest.mark.skipif(
    has_real_cuda() and torch.cuda.get_device_capability()[0] >= 12,
    reason = "on real sm_120+ the probe legitimately returns False when the build ships no "
    "sm_120 kernel, so asserting True there would be a false failure",
)
def test_probe_shapes_are_valid_on_working_gpu():
    # A probe that raises everywhere would silently disable xformers on working GPUs.
    assert ad._xformers_runs_on_device() is True


@pytest.mark.parametrize(
    "supports_bf16, expected_dtype",
    [(True, torch.bfloat16), (False, torch.float16)],
)
def test_probe_dtype_follows_bf16_support(monkeypatch, supports_bf16, expected_dtype):
    # Pre-Ampere GPUs lack bf16 attention, so the probe dtype must follow SUPPORTS_BFLOAT16.
    captured = {}

    def fake_zeros(
        *args,
        dtype = None,
        **kwargs,
    ):
        captured["dtype"] = dtype
        raise RuntimeError("stop after capturing the probe dtype")

    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", supports_bf16)
    monkeypatch.setattr(ad.torch, "zeros", fake_zeros)
    ad._xformers_runs_on_device()
    assert captured["dtype"] is expected_dtype


def test_probe_syncs_and_fails_on_deferred_async_error(monkeypatch):
    # Kernel launches are async; the probe must synchronize to catch deferred errors.
    _bias = type(
        "B",
        (),
        {
            "BlockDiagonalCausalMask": type(
                "M", (), {"from_seqlens": staticmethod(lambda seqlens: None)}
            )
        },
    )
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    monkeypatch.setattr(ad.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(ad, "xformers", type("X", (), {"attn_bias": _bias}))
    monkeypatch.setattr(ad, "xformers_attention", lambda *a, **k: None)

    def deferred_cuda_error():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(ad.torch.cuda, "synchronize", deferred_cuda_error)
    assert ad._xformers_runs_on_device() is False
