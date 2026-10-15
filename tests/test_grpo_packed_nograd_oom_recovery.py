# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An OOM in the GRPO packed (no-grad) logprob forward caps the packed length instead of
disabling packing for the run. Runs the real block via test_grpo_packed_raw_logits_nograd."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as harness  # noqa: E402


class _OOMOnceModel(harness._Model):
    def __init__(self, error):
        super().__init__(hidden_states = False)
        self.error = error

    def forward(self, input_ids = None, packed_seq_lengths = None, **kwargs):
        if packed_seq_lengths is not None and self.error is not None:
            error, self.error = self.error, None
            raise error
        return super().forward(
            input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs
        )


def _shorter_batch():
    pad = harness.PAD_ID
    return torch.tensor(
        [
            [pad, pad, pad, 5, 7, 9, 11, 13],
            [pad, pad, 4, 6, 8, 10, 12, 14],
        ]
    )


def test_oom_keeps_packing_for_batches_that_fit(monkeypatch):
    model = _OOMOnceModel(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    ns, *_ = harness._run_packed_block(model = model)
    assert ns["_pk_use"] is False
    assert getattr(model, "_unsloth_seq_packing_nograd_ok", None) is not False
    oom_T = ns["_pk_T"]
    assert model._unsloth_seq_packing_nograd_unsafe_T == oom_T

    model.calls.clear()
    ns, *_ = harness._run_packed_block(model = model)
    assert ns["_pk_use"] is False
    assert not any(call.packed for call in model.calls)

    monkeypatch.setattr(harness, "_batch", _shorter_batch)
    model.calls.clear()
    ns, _, input_ids, max_left_pad = harness._run_packed_block(model = model)
    assert ns["_pk_T"] < oom_T
    assert ns["_pk_use"] is True
    assert any(call.packed for call in model.calls)
    mask = harness._completion_mask_of(
        input_ids[:, -(harness.KEEP + max_left_pad) :],
        harness._left_pad_of(input_ids, harness.KEEP, harness.PAD_ID),
        max_left_pad,
        harness.PAD_ID,
    ).float()
    expected = harness._reference_logprobs(model, input_ids, max_left_pad)
    assert torch.allclose(ns["_pk_result"].float() * mask, expected * mask, atol = 1e-5)


def test_non_oom_failure_still_disables_packing():
    model = _OOMOnceModel(RuntimeError("kernel rejected the packed layout"))
    ns, *_ = harness._run_packed_block(model = model)
    assert ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False
