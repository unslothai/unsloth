# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO packing self-verify on dense models: bf16 GEMM shape noise (a row scored inside a packed stream vs alone,
a few single tokens past 0.7 nats on an L4) passes, while a cross-document leak, a broken document or a broad shift
still fails. Runs the helper, the real no-grad block and the zoo grad rewrite."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as harness  # noqa: E402
import test_grpo_packing_moe_noise as moe  # noqa: E402
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402


noise = load_dispatch_helpers()["_unsloth_seq_packing_shape_noise"]

DENSE = SimpleNamespace(hidden_size = 8)


def _owner():
    return SimpleNamespace(config = DENSE)


def _shape_noise(
    rows = 8,
    cols = 128,
    seed = 0,
):
    """Per-token bf16 jitter (mean |d| ~0.016) plus two single-token outliers past the 0.7 max-abs gate."""
    g = torch.Generator().manual_seed(seed)
    ref = -torch.rand(rows, cols, generator = g) * 4
    result = ref + torch.randn(rows, cols, generator = g) * 0.02
    result[2 % rows, 40 % cols] -= 0.79
    result[5 % rows, 7 % cols] += 0.75
    return result, ref, torch.ones(rows, cols)


def test_shape_noise_passes_and_counts():
    result, ref, cm = _shape_noise()
    assert float((result - ref).abs().max()) >= 0.7  # the max-abs gate alone rejects this batch
    model = _owner()
    assert noise(model, result, ref, cm) is True
    assert model._unsloth_seq_packing_shape_noise_passes == 1


@pytest.mark.parametrize(
    "bug",
    [
        "leak_after_first_document",
        "one_document_leaks",
        "every_token_shifted",
        "too_many_outliers",
        "nan",
    ],
)
def test_real_packing_bugs_still_fail(bug):
    result, ref, cm = _shape_noise()
    if bug == "leak_after_first_document":
        result[1:] -= 1.5  # what a boundary-ignoring mask does to every later document
    elif bug == "one_document_leaks":
        result[3, 10:] -= 0.4
        assert float((result - ref).abs().mean()) < 0.1  # a batch-wide mean would miss it
    elif bug == "every_token_shifted":
        result -= 0.15
    elif bug == "too_many_outliers":
        result.view(-1)[::50] -= 1.1  # 2% of tokens
    else:
        result[0, 0] = float("nan")
    assert noise(_owner(), result, ref, cm) is False


def test_small_batches_short_documents_and_masked_columns():
    result, ref, cm = _shape_noise(rows = 2, cols = 30)
    assert (
        noise(_owner(), result, ref, cm) is False
    )  # 60 scored tokens: too few to tell noise from a bug
    result, ref, cm = _shape_noise(rows = 16, cols = 7)
    assert noise(_owner(), result, ref, cm) is False  # no document with 8+ scored tokens
    result, ref, cm = _shape_noise()
    cm[:, 100:] = 0
    result[:, 100:] += 50.0  # garbage outside the loss mask is ignored
    assert noise(_owner(), result, ref, cm) is True


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PACKING_SHAPE_NOISE", "0")
    result, ref, cm = _shape_noise()
    assert noise(_owner(), result, ref, cm) is False


class _ShapeNoiseModel(moe._MoeModel):
    """Dense model whose packed forward moves one completion logit by 0.8 nats (L4 shape noise) when armed."""

    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
        **kwargs,
    ):
        if self.mode != "shape":
            return super().forward(
                input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs
            )
        self.mode = None
        out = super().forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)
        self.mode = "shape"
        if packed_seq_lengths is None:
            return out
        logits = out.logits.clone()
        p = int(packed_seq_lengths.cumsum(0)[1]) - 4
        logits[0, p, int(input_ids[0, p + 1])] -= 0.8
        out.logits = logits
        return out


@pytest.fixture
def big_batches(monkeypatch):
    def _batch(rows = 4, prompt = 3):
        return moe._big_batch(rows = 32, prompt = prompt)

    monkeypatch.setattr(harness, "_batch", _batch)
    monkeypatch.setattr(harness, "KEEP", moe.BIG_KEEP)


def _run(mode):
    model = _ShapeNoiseModel(DENSE)
    model.mode = mode
    ns, *_ = harness._run_packed_block(model = model)
    return ns, model


def test_nograd_block_packs_through_shape_noise(big_batches):
    ns, model = _run("shape")
    assert 0.7 <= ns["_pk_diff"] < 1.5
    assert ns["_pk_use"] is True
    assert model._unsloth_seq_packing_nograd_ok is True
    assert model._unsloth_seq_packing_shape_noise_passes == 1


def test_nograd_block_contamination_still_disables(big_batches):
    ns, model = _run("broad")
    assert ns["_pk_diff"] >= 1.5 and ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False
    assert getattr(model, "_unsloth_seq_packing_shape_noise_passes", 0) == 0


def test_zoo_grad_verify_rewrite():
    rl = pytest.importorskip("unsloth.models.rl_replacements")
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    source = inspect.getsource(zoo.grpo_accumulated_loss)
    if "_pack_diff < 7e-1" not in source:
        pytest.skip("installed unsloth_zoo has no grad packing verify")
    patched = rl._patch_grpo_accumulated_loss_moe_noise(source)
    assert (
        "_unsloth_seq_packing_shape_noise(unwrapped_model, _pack_result, _pack_ref, _pack_cm)"
        in patched
    )
    compile(patched, "<patched grpo_accumulated_loss>", "exec")
    items = [i for i in rl.RL_PRE_ITEMS["grpo_trainer"] if isinstance(i, str)]
    assert any("def _unsloth_seq_packing_shape_noise(" in i for i in items)
    assert any("_unsloth_seq_packing_shape_noise(unwrapped_model, _pack_result" in i for i in items)
