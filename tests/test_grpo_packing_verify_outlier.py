# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO packing self-verify: one narrow bf16 outlier on packing that already verified falls back for that batch
only; a broad mismatch, a second consecutive one, or one before any verify still disables packing. Runs the real
no-grad block via test_grpo_packed_raw_logits_nograd and the zoo grad rewrite on its installed source."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as harness  # noqa: E402


def _short_batch():
    pad = harness.PAD_ID
    return torch.tensor([[pad, pad, pad, 5, 7, 9, 11, 13], [pad, pad, 4, 6, 8, 10, 12, 14]])


class _OutlierModel(harness._Model):
    """The packed forward is exact unless armed: "narrow" drops one completion logit by 9 nats, "broad" every
    position of the second segment (what contamination across packed rows looks like)."""

    def __init__(self):
        super().__init__(hidden_states = False)
        self.mode = None

    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
        **kwargs,
    ):
        out = super().forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)
        if packed_seq_lengths is None or self.mode is None:
            return out
        logits = out.logits.clone()
        T = input_ids.shape[1]
        if self.mode == "narrow":
            positions = [T - 2]
        else:
            first = int(packed_seq_lengths[0])
            positions = list(range(first, T - 1))
        for p in positions:
            logits[0, p, int(input_ids[0, p + 1])] -= 9.0
        out.logits = logits
        return out


def _verified_model(monkeypatch):
    model = _OutlierModel()
    monkeypatch.setattr(harness, "_batch", _short_batch)
    ns, *_ = harness._run_packed_block(model = model)
    assert ns["_pk_use"] is True and ns["_pk_diff"] < 7e-1
    assert model._unsloth_seq_packing_nograd_verified_T == ns["_pk_T"]
    monkeypatch.undo()
    return model


def _run(model, mode):
    model.mode = mode
    ns, *_ = harness._run_packed_block(model = model)
    model.mode = None
    return ns


def test_one_narrow_outlier_keeps_verified_packing(monkeypatch):
    model = _verified_model(monkeypatch)
    ns = _run(model, "narrow")
    assert ns["_pk_diff"] >= 1.5
    assert ns["_pk_use"] is False  # this batch falls back
    assert model._unsloth_seq_packing_nograd_ok is True
    assert getattr(model, "_unsloth_seq_packing_nograd_unsafe_T", None) is None
    # The next clean batch packs again, re-verifies and widens the envelope.
    ns = _run(model, None)
    assert ns["_pk_use"] is True
    assert model._unsloth_seq_packing_nograd_verified_T == ns["_pk_T"]
    assert model._unsloth_seq_packing_nograd_strikes == 0


def test_repeated_outlier_disables_packing(monkeypatch):
    model = _verified_model(monkeypatch)
    _run(model, "narrow")
    assert model._unsloth_seq_packing_nograd_ok is True
    ns = _run(model, "narrow")
    assert ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False


@pytest.mark.parametrize("next_mode", [None, "narrow"], ids = ["clean", "second-outlier"])
def test_recheck_previously_verified_batch_after_outlier(monkeypatch, next_mode):
    model = _verified_model(monkeypatch)
    assert _run(model, "narrow")["_pk_use"] is False
    monkeypatch.setattr(harness, "_batch", _short_batch)
    ns = _run(model, next_mode)
    if next_mode is None:
        assert ns["_pk_use"] is True and ns["_pk_diff"] < 0.7
        assert model._unsloth_seq_packing_nograd_strikes == 0
        assert "_pk_diff" not in _run(model, None)  # trusted again
    else:
        assert ns["_pk_use"] is False
        assert ns["_pk_diff"] >= 1.5
        assert model._unsloth_seq_packing_nograd_ok is False


def test_broad_mismatch_disables_verified_packing(monkeypatch):
    model = _verified_model(monkeypatch)
    ns = _run(model, "broad")
    assert ns["_pk_diff"] >= 1.5
    assert model._unsloth_seq_packing_nograd_ok is False


def test_outlier_before_any_verify_disables_packing():
    model = _OutlierModel()
    ns = _run(model, "narrow")
    assert ns["_pk_diff"] >= 1.5
    assert model._unsloth_seq_packing_nograd_ok is False


def test_zoo_grad_verify_rewrite():
    rl = pytest.importorskip("unsloth.models.rl_replacements")
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    source = inspect.getsource(zoo.grpo_accumulated_loss)
    if "if _pack_diff >= 1.5:" not in source:
        pytest.skip("installed unsloth_zoo has no grad packing kill rule")
    patched = rl._patch_grpo_accumulated_loss_seq_packing_kill(source)
    assert patched != source
    assert "if _pack_diff >= 1.5:" not in patched
    assert '_unsloth_seq_packing_verify_verdict(unwrapped_model, "grad"' in patched
    assert "_unsloth_seq_packing_grad_strikes = 0" in patched
    compile(patched, "<patched grpo_accumulated_loss>", "exec")
    assert rl._patch_grpo_accumulated_loss_seq_packing_kill(patched) == patched
    assert any(
        isinstance(item, str)
        and '_unsloth_seq_packing_verify_verdict(unwrapped_model, "grad"' in item
        for item in rl.RL_PRE_ITEMS["grpo_trainer"]
    )
