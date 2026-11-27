# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""On CPU unsloth_zoo routes TRL's fused LM head kernel through a device dispatch with no `forward`. Unsloth's
fused head adapter then wrapped the dispatch, read no contract, and returned five outputs where TRL >= 1.17
unpacks six: every FastModel SFT step on CPU raised `not enough values to unpack`. CPU."""

import pytest
import torch

import unsloth  # noqa: F401

trl_utils = pytest.importorskip("trl.trainer.utils")
cpu_ops = pytest.importorskip("unsloth_zoo.cpu.ops")
from unsloth.models import rl  # noqa: E402


def test_cpu_dispatch_keeps_trl_output_count(monkeypatch):
    kernel = getattr(trl_utils, "_ChunkedLogProbFunction", None)
    for _ in range(4):  # back to TRL's own kernel under any Unsloth / unsloth_zoo wrapper
        inner = kernel.__dict__.get("function") if not isinstance(kernel, type) else None
        inner = inner or (getattr(kernel, "_kernel", None) if isinstance(kernel, type) else None)
        if inner is None:
            break
        kernel = inner
    if kernel is None or getattr(kernel, "forward", None) is None:
        pytest.skip("TRL without the fused LM head kernel")
    _, n_outputs, _ = cpu_ops._trl_logprob_contract(kernel)
    monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", kernel)
    assert cpu_ops.patch_trl_fused_lm_head()
    rl.patch_trl_fused_lm_head_kernel()
    torch.manual_seed(0)
    hidden, weight = torch.randn(6, 16), torch.randn(32, 16)
    targets = torch.randint(0, 32, (6,))
    outputs = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
    result = trl_utils._ChunkedLogProbFunction.apply(
        hidden, weight, None, targets, 1.0, 4096, None, 1.0, outputs
    )
    assert len(result) == n_outputs
    reference = torch.log_softmax(hidden @ weight.t(), -1).gather(-1, targets[:, None])[:, 0]
    torch.testing.assert_close(result[0], reference, rtol = 1e-4, atol = 1e-4)
