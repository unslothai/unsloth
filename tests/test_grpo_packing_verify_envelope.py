# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO packing self-verify cost: a passing verify trusts up to 2x its shape (never across a LongRoPE / dynamic
RoPE switch), and a re-verify of already trusted packing scores at most 3 rows alone, always including the last
document of the packed stream. The first verify still scores every row, and a document-boundary leak still fails
both the first verify and a bounded re-verify. Runs the helpers, the real no-grad block and the zoo grad body."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as harness  # noqa: E402
import test_grpo_packing_verify_outlier as outlier  # noqa: E402
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402


_HELPERS = load_dispatch_helpers()
verify_rows = _HELPERS["_unsloth_seq_packing_verify_rows"]
trusted_bound = _HELPERS["_unsloth_seq_packing_trusted_bound"]


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY_GROWTH", raising = False)


def test_untrusted_verify_scores_every_row():
    lengths = [40, 1, 55, 30, 70, 20, 33, 61]
    counts = [10, 0, 20, 5, 30, 4, 8, 25]
    assert verify_rows(lengths, counts, trusted = False) == [0, 2, 3, 4, 5, 6, 7]


def test_trusted_reverify_scores_at_most_three_rows_including_the_last():
    lengths = [40, 1, 90, 30, 70, 20, 33, 61]
    counts = [10, 0, 20, 5, 60, 4, 8, 25]
    rows = verify_rows(lengths, counts, trusted = True)
    assert len(rows) == 3
    assert 7 in rows  # last document: sees every earlier one in the packed stream
    assert 2 in rows  # longest row: the new position range
    assert 4 in rows  # most completion tokens
    # the last row with completion tokens, not a trailing prompt-only row
    rows = verify_rows([40, 50, 60, 45, 30], [9, 8, 7, 6, 0], trusted = True)
    assert rows == [0, 2, 3] and 4 not in rows


def test_small_batches_and_force_flag_score_every_row(monkeypatch):
    assert verify_rows([5, 6, 7], [1, 1, 1], trusted = True) == [0, 1, 2]
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", "1")
    assert verify_rows([5, 6, 7, 8, 9], [1] * 5, trusted = True) == [0, 1, 2, 3, 4]


def _model(rope = None, **config):
    return SimpleNamespace(config = SimpleNamespace(rope_scaling = rope, **config))


def test_trusted_bound_doubles_and_respects_rope_switches(monkeypatch):
    assert trusted_bound(_model(), 300) == 600
    longrope = {"rope_type": "longrope", "original_max_position_embeddings": 4096}
    assert (
        trusted_bound(_model(longrope), 3000) == 4096
    )  # never widens across the short / long switch
    assert trusted_bound(_model(longrope), 5000) == 5000  # past it: exact shape only
    dynamic = {"type": "dynamic"}
    assert trusted_bound(_model(dynamic, max_position_embeddings = 2048), 1500) == 2048
    per_layer = SimpleNamespace(
        config = SimpleNamespace(
            rope_parameters = {
                "full_attention": {
                    "rope_type": "longrope",
                    "original_max_position_embeddings": 1000,
                },
            }
        )
    )
    assert trusted_bound(per_layer, 800) == 1000
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY_GROWTH", "1")
    assert trusted_bound(_model(), 300) == 300


def _rows_batch(rows):
    return outlier._big_batch(rows = rows)


def _run(model, rows, mode, monkeypatch):
    monkeypatch.setattr(harness, "_batch", lambda: _rows_batch(rows))
    model.mode = mode
    model.calls.clear()
    ns, *_ = harness._run_packed_block(model = model)
    model.mode = None
    per_row = sum(1 for c in model.calls if not c.packed)
    return ns, per_row


def _trusted_model(monkeypatch):
    model = outlier._OutlierModel()
    ns, per_row = _run(model, 3, None, monkeypatch)
    assert ns["_pk_use"] is True and ns["_pk_diff"] < 7e-1
    assert per_row == 3  # first verify: every row
    assert model._unsloth_seq_packing_nograd_verified_T == 2 * ns["_pk_T"]
    return model, ns["_pk_T"]


def test_within_twice_the_verified_shape_skips_the_verify(monkeypatch):
    model, T = _trusted_model(monkeypatch)
    ns, per_row = _run(model, 4, None, monkeypatch)  # 4 / 3 of the verified T
    assert ns["_pk_T"] <= 2 * T
    assert ns["_pk_use"] is True and per_row == 0 and "_pk_diff" not in ns


def test_reverify_past_the_bound_scores_three_rows(monkeypatch):
    model, T = _trusted_model(monkeypatch)
    ns, per_row = _run(model, 8, None, monkeypatch)
    assert ns["_pk_T"] > 2 * T
    assert ns["_pk_use"] is True and per_row == 3
    assert int((ns["_pk_cm"].sum(dim = 1) > 0).sum()) == 3  # unscored rows are out of the compare
    assert model._unsloth_seq_packing_nograd_verified_T == 2 * ns["_pk_T"]


def test_boundary_leak_fails_the_bounded_reverify(monkeypatch):
    """A packed forward that lets documents see each other ("broad": every document after the first moves) must
    still disable packing when only 3 rows are scored."""
    model, _ = _trusted_model(monkeypatch)
    ns, per_row = _run(model, 8, "broad", monkeypatch)
    assert per_row == 3
    assert ns["_pk_diff"] >= 1.5 and ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False


def test_boundary_leak_fails_the_first_verify(monkeypatch):
    model = outlier._OutlierModel()
    ns, per_row = _run(model, 8, "broad", monkeypatch)
    assert per_row == 8
    assert ns["_pk_diff"] >= 1.5 and ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False


def test_leak_in_the_last_document_is_always_scored(monkeypatch):
    """ "narrow" moves one token of the last document only; the bounded re-verify must still see it."""
    model, _ = _trusted_model(monkeypatch)
    ns, per_row = _run(model, 8, "narrow", monkeypatch)
    assert per_row == 3 and ns["_pk_diff"] >= 1.5 and ns["_pk_use"] is False


def test_zoo_grad_verify_is_bounded():
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    if not hasattr(zoo, "grpo_packing_verify_rows"):
        pytest.skip("installed unsloth_zoo predates the bounded grad verify")
    source = inspect.getsource(zoo.grpo_accumulated_loss)
    assert "for _pack_i in _pack_vrows:" in source
    assert "grpo_packing_trusted_bound(unwrapped_model, _pack_T)" in source
    rl = pytest.importorskip("unsloth.models.rl_replacements")
    patched = rl._patch_grpo_accumulated_loss_moe_noise(
        rl._patch_grpo_accumulated_loss_seq_packing_kill(source)
    )
    assert (
        "_unsloth_seq_packing_shape_noise(unwrapped_model, _pack_result, _pack_ref, _pack_cm)"
        in patched
    )
    compile(patched, "<patched grpo_accumulated_loss>", "exec")
    assert zoo.grpo_packing_verify_rows([5, 6, 7, 8, 9], [1] * 5, True) == [0, 1, 4]
    assert zoo.grpo_packing_trusted_bound(SimpleNamespace(config = SimpleNamespace()), 10) == 20


def test_length_dependent_rope_reverifies_every_row(monkeypatch):
    """Rows on either side of a LongRoPE switch run different frequencies alone, so a 3-row sample could miss the
    short side: such models keep scoring every row."""
    longrope = _model({"rope_type": "longrope", "original_max_position_embeddings": 4096})
    assert verify_rows([5, 6, 7, 8, 9], [1] * 5, trusted = True, model = longrope) == [0, 1, 2, 3, 4]
    assert verify_rows([5, 6, 7, 8, 9], [1] * 5, trusted = True, model = _model()) == [0, 1, 4]
    model = outlier._OutlierModel()
    model.config = longrope.config
    ns, per_row = _run(model, 3, None, monkeypatch)
    assert per_row == 3 and model._unsloth_seq_packing_nograd_verified_T == 2 * ns["_pk_T"]
    ns, per_row = _run(model, 8, None, monkeypatch)
    assert per_row == 8 and ns["_pk_use"] is True
