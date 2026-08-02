"""Regression test for unslothai/unsloth#4631: xformers must not be blanket-disabled
on sm_120 GPUs where its kernel actually runs (a ~57% attention-memory saving over the
SDPA packed-mask fallback). The gate now probes the real op instead of guessing by the
compute-capability major version."""

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


@pytest.mark.parametrize(
    "capability, probe_result, expect_disabled",
    [
        ((8, 9), None, False),  # Ada: below sm_120, never probed, always kept
        ((9, 0), None, False),  # Hopper: below sm_120, kept
        ((10, 0), None, False),  # Blackwell B200 (sm_100): below sm_120, kept
        ((12, 0), True, False),  # sm_120 where the kernel runs: keep xformers
        ((12, 0), False, True),  # sm_120 where the kernel can't run: fall back to SDPA
    ],
)
def test_capability_gate(capability, probe_result, expect_disabled):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert ad._xformers_disabled_for_capability(capability, probe = probe) is expect_disabled
    # Below sm_120 the probe must not run at all (no import-time kernel launch there).
    assert calls["n"] == (0 if capability[0] < 12 else 1)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ad.HAS_XFORMERS),
    reason = "needs a CUDA GPU with a working xformers build",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12,
    reason = "on real sm_120+ the probe legitimately returns False when the build ships no "
    "sm_120 kernel, so asserting True there would be a false failure",
)
def test_probe_shapes_are_valid_on_working_gpu():
    # Guards against a malformed probe that raises on every GPU and would silently
    # disable xformers on Blackwell even where it works. On a pre-sm_120 GPU with a
    # functional xformers the real probe must succeed; sm_120+ is skipped above because
    # there a False is a correct answer, not a malformed probe.
    assert ad._xformers_runs_on_device() is True


@pytest.mark.parametrize(
    "supports_bf16, expected_dtype",
    [(True, torch.bfloat16), (False, torch.float16)],
)
def test_probe_dtype_follows_bf16_support(monkeypatch, supports_bf16, expected_dtype):
    # Pre-Ampere GPUs (sm < 80: Turing/Volta, e.g. T4/V100) run xformers fine in
    # float16 but have no bfloat16 attention kernel, so a hardcoded bf16 probe would
    # raise there, get swallowed to False, and misreport a working xformers as broken.
    # The probe must pick its dtype from SUPPORTS_BFLOAT16 (no Turing GPU needed here).
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
    ad._xformers_runs_on_device()  # RuntimeError is swallowed; only the dtype matters
    assert captured["dtype"] is expected_dtype


def test_probe_syncs_and_fails_on_deferred_async_error(monkeypatch):
    # A CUDA kernel launch is async: xformers_attention can return before the GPU
    # reports a failure. The probe must synchronize so a deferred launch/runtime error
    # is caught and disables xformers here, instead of surfacing later on an unrelated
    # CUDA call (unslothai/unsloth#6828 review). No GPU needed: everything is stubbed.
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
    monkeypatch.setattr(ad, "xformers_attention", lambda *a, **k: None)  # "succeeds"

    def deferred_cuda_error():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(ad.torch.cuda, "synchronize", deferred_cuda_error)
    # Without the synchronize the stubbed op returns cleanly and the probe wrongly
    # reports True; the sync surfaces the deferred error so the probe returns False.
    assert ad._xformers_runs_on_device() is False
