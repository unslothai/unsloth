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


@pytest.mark.parametrize(
    "builder,embed_attr,norm_attr",
    [
        ("llama", "embed_tokens", "norm"),
        ("opt", "embed_tokens", "final_layer_norm"),
        ("gpt2", "wte", "ln_f"),
    ],
)
def test_a_stage_resolves_every_layout_find_layers_accepts(
    builder: str, embed_attr: str, norm_attr: str
) -> None:
    """find_layers accepts OPT and GPT-2, so the stage wrapper must not assume Llama's names.

    OPT is the dangerous one: its final normalisation is `final_layer_norm`, so looking only
    for `norm` left it None and dropped it from the last stage without a word. GPT-2 keeps its
    layers in `transformer.h`, so `owner.layers` raised AttributeError.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    if builder == "llama":
        config = transformers.LlamaConfig(
            vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
            num_attention_heads = 4, num_key_value_heads = 4, max_position_embeddings = 32,
        )
        model = transformers.LlamaForCausalLM(config)
    elif builder == "opt":
        config = transformers.OPTConfig(
            vocab_size = 64, hidden_size = 32, ffn_dim = 64, num_hidden_layers = 2,
            num_attention_heads = 4, max_position_embeddings = 32, word_embed_proj_dim = 32,
        )
        model = transformers.OPTForCausalLM(config)
    else:
        config = transformers.GPT2Config(
            vocab_size = 64, n_embd = 32, n_layer = 2, n_head = 4, n_positions = 32,
        )
        model = transformers.GPT2LMHeadModel(config)

    pipeline = _pipeline_module()
    top, owner = pipeline.unwrap_stack(model)
    _, layers = pipeline.find_layers(model)
    stage = pipeline.stage_module_cls()(
        top, owner, list(range(len(layers))),
        is_first = True, is_last = True, grad_checkpoint = False,
    )

    assert len(stage.layers) == len(layers)
    assert stage.embed_tokens is getattr(owner, embed_attr)
    # The one that was silently skipped rather than raising.
    assert stage.norm is getattr(owner, norm_attr)
