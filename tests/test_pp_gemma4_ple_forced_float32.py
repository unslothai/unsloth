# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma 4 per-layer inputs under pipeline parallel keep the one-GPU dtype on forced float32 (float16 weights).

Unsloth Zoo's forced float32 patch makes every Gemma 4 scaled embedding return float32, so on one GPU the per-layer
inputs are float32 and `project_per_layer_inputs` adds the float16 projection to them in float32. The pipeline
stage used to return its table slice in the weight dtype and cast the per-layer inputs to the projection dtype,
rounding every layer's input to float16 (~3e-4 relative); on T4 that moved Gemma-4-E2B's step-1 gradient norm from
12.1 (one GPU) to 27. CPU only, no process group."""

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _tiny_mm as tiny  # noqa: E402
import test_pp_vlm_text_only_gloo as vlm  # noqa: E402

pytest.importorskip("transformers.models.gemma4")

ppn = vlm.ppn
LAYERS = 6
WIDTH = 8


def _forced_float32_decoder():
    """float16 Gemma 4 E2B decoder; per-layer table is float32 under the forced float32 patch."""
    model = tiny.build("gemma4_e2b", layers = LAYERS, lora = False, causal_lm = True).to(torch.float16)
    decoder = model.model
    table = decoder.embed_tokens_per_layer

    def forward(input_ids):
        rows = F.embedding(input_ids, table.weight, table.padding_idx)
        return rows.to(torch.float32) * table.embed_scale.to(torch.float32)

    table.forward = forward
    return decoder


def test_table_slice_returns_the_table_dtype():
    decoder = _forced_float32_decoder()
    ids, _ = tiny.batch()
    full = decoder.get_per_layer_inputs(ids, None)
    owned = [2, 3, 4, 5]
    sliced = ppn._PPLayerTableSlice(decoder.embed_tokens_per_layer, owned, LAYERS, WIDTH)(ids)
    assert full.dtype == torch.float32
    assert sliced.dtype == full.dtype, (sliced.dtype, full.dtype)
    assert torch.equal(sliced[:, :, 2:], full[:, :, 2:])
    assert not sliced[:, :, :2].any()


@pytest.mark.parametrize("end", [3, LAYERS])
def test_first_stage_projection_matches_the_model(end):
    decoder = _forced_float32_decoder()
    ids, _ = tiny.batch()
    embeds = decoder.embed_tokens(ids).to(torch.float16)
    per_layer = decoder.get_per_layer_inputs(ids, None)
    original = decoder.project_per_layer_inputs
    expected = original(embeds, per_layer)
    ctx = types.SimpleNamespace(
        active = True,
        is_first = True,
        is_last = end == LAYERS,
        start = 0,
        end = end,
        ple = types.SimpleNamespace(layers = LAYERS, width = WIDTH),
        ple_output = None,
    )
    got = ppn._ple_project(ctx, decoder, original, embeds, per_layer)
    assert got.dtype == expected.dtype == torch.float32, (got.dtype, expected.dtype)
    assert torch.equal(got, expected)


def test_weight_dtype_tables_are_unchanged():
    """No forced float32: the slice returns the weight dtype, as the unpatched scaled embedding does."""
    model = tiny.build("gemma4_e2b", layers = LAYERS, lora = False, causal_lm = True).to(torch.float16)
    decoder = model.model
    ids, _ = tiny.batch()
    full = decoder.get_per_layer_inputs(ids, None)
    sliced = ppn._PPLayerTableSlice(decoder.embed_tokens_per_layer, [0, 1, 2], LAYERS, WIDTH)(ids)
    assert sliced.dtype == full.dtype == torch.float16
    assert torch.equal(sliced[:, :, :3], full[:, :, :3])
