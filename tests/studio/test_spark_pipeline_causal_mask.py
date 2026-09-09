# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pipeline stage calls decoder layers directly, so causality is its own responsibility.

sdpa and flash read `is_causal` when `attention_mask` is None, but transformers' eager path
adds the mask under `if attention_mask is not None` and so applies none at all. Passing None
there trains a bidirectional model at a flattering loss. Measured before the fix: editing the
last token moved hidden states at earlier positions by 6.4e-02 on eager and by exactly 0 on
sdpa.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "spark_pipeline", REPO / "studio" / "spark_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_a_stage_is_causal_on_every_attention_implementation(impl: str) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
    )
    config._attn_implementation = impl
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()

    pipeline = _pipeline_module()
    owner, layers = pipeline.find_layers(model)
    stage = pipeline.stage_module_cls()(
        model,
        owner,
        list(range(len(layers))),
        is_first = True,
        is_last = True,
        grad_checkpoint = False,
    ).eval()

    length = 8
    ids = torch.randint(0, config.vocab_size, (1, length))
    edited = ids.clone()
    edited[0, -1] = (edited[0, -1] + 1) % config.vocab_size

    with torch.no_grad():
        before = stage(ids)
        after = stage(edited)

    # Every position but the last may only see tokens at or before it, so editing the final
    # token must leave them bit-identical.
    moved = (before[0, :-1] - after[0, :-1]).abs().max().item()
    assert moved == 0.0, f"{impl}: the last token moved earlier positions by {moved:.3e}"
