# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO padded scoring loop: a chunk's [B, L, V] logits are released before the next chunk's forward.

The slices taken of ``outputs.logits`` are views, so while ``logits_chunk`` stays bound the whole
logits tensor stays alive and the next forward allocates its own on top of it (one chunk more at peak).
Runs the real loop lifted out of rl_replacements.py, on CPU."""

from __future__ import annotations

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_padded_raw_logits as _loop  # noqa: E402


class _TrackingModel(_loop._StubModel):
    """Records, at every forward, whether the previous forward's output tensor is still alive."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None
        self.previous_alive = []

    def __call__(self, *args, **kwargs):
        if self.previous is not None:
            self.previous_alive.append(self.previous() is not None)
        out = super().__call__(*args, **kwargs)
        # The storage owner: a view of a view keeps it alive, not the intermediate view.
        base = out.logits._base if out.logits._base is not None else out.logits
        self.previous = weakref.ref(base)
        return SimpleNamespace(logits = out.logits)


@pytest.mark.parametrize("returns_hidden_states", [False, True], ids = ["logits", "hidden"])
@pytest.mark.parametrize("is_vlm", [False, True], ids = ["text", "vlm"])
def test_previous_chunk_released_before_next_forward(returns_hidden_states, is_vlm):
    data = _loop._make_data()
    stub = _TrackingModel(data.embedding, data.lm_head, returns_hidden_states)
    namespace = _loop._build_namespace(
        data,
        stub,
        is_vlm = is_vlm,
        temperature = 1.0,
        logit_softcapping = 0.0,
        logit_scale_multiply = 0.0,
        logit_scale_divide = 0.0,
    )
    exec(_loop._BLOCK_CODE, namespace)
    assert len(stub.calls) == _loop._BATCH >= 2
    assert stub.previous_alive == [False] * (_loop._BATCH - 1)
    # Same result as the per-row reference.
    expected = _loop._reference_logprobs(data, is_vlm = is_vlm)
    torch.testing.assert_close(namespace["logprobs"], expected, rtol = 1e-5, atol = 1e-5)
