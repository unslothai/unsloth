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
def test_probe_shapes_are_valid_on_working_gpu():
    # Guards against a malformed probe that raises on every GPU and would silently
    # disable xformers on Blackwell even where it works. On any GPU with a functional
    # xformers (the CI/dev boxes are pre-sm_120), the real probe must succeed.
    assert ad._xformers_runs_on_device() is True
