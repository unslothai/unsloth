# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pipeline stage calls decoder layers directly, so causality is its own responsibility."""

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

    # Every position but the last may only see tokens at or before it, so editing the final token
    # must leave them bit-identical.
    moved = (before[0, :-1] - after[0, :-1]).abs().max().item()
    assert moved == 0.0, f"{impl}: the last token moved earlier positions by {moved:.3e}"


def _tiny(transformers, builder: str):
    if builder == "llama":
        return transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 4,
                max_position_embeddings = 32,
            )
        )
    if builder == "gpt_neox":
        return transformers.GPTNeoXForCausalLM(
            transformers.GPTNeoXConfig(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                max_position_embeddings = 32,
            )
        )
    if builder == "gemma3":
        return transformers.Gemma3ForCausalLM(
            transformers.Gemma3TextConfig(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 4,
                head_dim = 8,
                sliding_window = 8,
            )
        )
    if builder == "gemma2":
        return transformers.Gemma2ForCausalLM(
            transformers.Gemma2Config(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 4,
                head_dim = 8,
                sliding_window = 8,
                final_logit_softcapping = 0.5,
            )
        )
    if builder == "gptj":
        return transformers.GPTJForCausalLM(
            transformers.GPTJConfig(
                vocab_size = 64,
                n_embd = 32,
                n_layer = 2,
                n_head = 4,
                rotary_dim = 4,
                n_positions = 32,
            )
        )
    if builder == "opt":
        return transformers.OPTForCausalLM(
            transformers.OPTConfig(
                vocab_size = 64,
                hidden_size = 32,
                ffn_dim = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                max_position_embeddings = 32,
                word_embed_proj_dim = 32,
            )
        )
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size = 64,
            n_embd = 32,
            n_layer = 2,
            n_head = 4,
            n_positions = 32,
        )
    )


def _build_stage(pipeline, model, **kwargs):
    top, owner = pipeline.unwrap_stack(model)
    _, layers = pipeline.find_layers(model)
    return pipeline.stage_module_cls()(
        top,
        owner,
        list(range(len(layers))),
        is_first = True,
        is_last = True,
        grad_checkpoint = False,
        **kwargs,
    )


@pytest.mark.parametrize("builder", ["llama", "gpt_neox", "gemma3", "gemma2", "gptj"])
def test_a_stage_computes_the_same_model_it_was_split_from(builder: str) -> None:
    """The contract is not that the names resolve, it is that the arithmetic agrees."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model = _tiny(transformers, builder)
    model.eval()
    model.config._attn_implementation = "eager"
    pipeline = _pipeline_module()
    stage = _build_stage(pipeline, model)

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        want = model(ids).logits
        got = stage(ids)
    moved = (got - want).abs().max().item()
    assert moved < 1e-4, f"{builder}: the stage is a different model by {moved:.3e}"


@pytest.mark.parametrize("builder,dropped", [("gpt2", "wpe"), ("opt", "embed_positions")])
def test_a_stage_refuses_a_layout_whose_positions_it_would_drop(builder: str, dropped: str) -> None:
    """GPT-2 and OPT keep learned positions in a module outside every decoder layer."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    pipeline = _pipeline_module()
    with pytest.raises(RuntimeError, match = dropped):
        _build_stage(pipeline, _tiny(transformers, builder))


def test_a_stage_refuses_a_batch_longer_than_the_sliding_window() -> None:
    """Below the window a sliding mask and a causal mask are the same matrix, because every query
    already reaches every earlier token."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model = _tiny(transformers, "gemma3")
    model.eval()
    pipeline = _pipeline_module()
    stage = _build_stage(pipeline, model)
    assert stage.sliding_window == 8

    with torch.no_grad():
        stage(torch.randint(0, 64, (1, 8)))  # at the window, still exact
        with pytest.raises(RuntimeError, match = "sliding window"):
            stage(torch.randint(0, 64, (1, 9)))


def test_the_legacy_backend_actually_checkpoints_when_asked() -> None:
    """`--pp-backend legacy --grad-checkpoint` set the transformers flag and reported success."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch.utils.checkpoint as torch_checkpoint

    pipeline = _pipeline_module()
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 6,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    model.train()
    model.config._attn_implementation = "eager"

    def measure(grad_checkpoint: bool):
        calls, saved = [0], [0]
        real = torch_checkpoint.checkpoint

        def counting(*a, **k):
            calls[0] += 1
            return real(*a, **k)

        stage = pipeline._Stage(model, config, 0, 1, "cpu", torch.float32, 1)
        stage.grad_checkpoint = grad_checkpoint
        ids = torch.randint(0, 64, (2, 32))
        posid = torch.arange(32).unsqueeze(0).expand(2, -1)

        def pack(t):
            saved[0] += 1
            return t

        torch_checkpoint.checkpoint = counting
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                stage.forward(ids, None, posid)
        finally:
            torch_checkpoint.checkpoint = real
        return calls[0], saved[0]

    off_calls, off_saved = measure(False)
    on_calls, on_saved = measure(True)

    assert off_calls == 0, "checkpointing ran when it was not asked for"
    assert (
        on_calls == config.num_hidden_layers
    ), f"expected one checkpoint per decoder layer, got {on_calls}"
    assert on_saved < off_saved / 2, (
        f"checkpointing saved {on_saved} tensors against {off_saved} without it, "
        f"which is not a reduction: the option is inert"
    )


def test_gemma_embeddings_are_scaled_when_the_embedding_does_not_scale_itself() -> None:
    """transformers 4.x multiplies Gemma 1/2 embeddings by sqrt(hidden_size) in the model forward,
    which a stage skips; 5.x moved the scale into the embedding module."""
    torch = pytest.importorskip("torch")
    pipeline = _pipeline_module()

    class Cfg:
        model_type = "gemma2"
        hidden_size = 16

    plain = torch.nn.Embedding(4, 16)
    assert pipeline._embed_scale(Cfg, plain) == 4.0
    plain.embed_scale = 4.0
    assert pipeline._embed_scale(Cfg, plain) is None
    Cfg.model_type = "llama"
    assert pipeline._embed_scale(Cfg, torch.nn.Embedding(4, 16)) is None


def test_a_gpt_neox_first_stage_keeps_its_embedding_dropout() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPTNeoXForCausalLM(
        transformers.GPTNeoXConfig(
            vocab_size = 64,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            max_position_embeddings = 32,
            hidden_dropout = 0.5,
        )
    )
    stage = _build_stage(_pipeline_module(), model)
    assert isinstance(stage.embed_dropout, torch.nn.Dropout) and stage.embed_dropout.p == 0.5
    stage.train()
    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(0)
    a = stage(ids)
    torch.manual_seed(1)
    assert not torch.equal(a, stage(ids))


def test_a_gptj_first_stage_keeps_its_embedding_dropout() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPTJForCausalLM(
        transformers.GPTJConfig(
            vocab_size = 64, n_embd = 32, n_layer = 2, n_head = 4, rotary_dim = 4, embd_pdrop = 0.5
        )
    )
    stage = _build_stage(_pipeline_module(), model)
    assert stage.embed_dropout is model.transformer.drop and stage.embed_dropout.p == 0.5
