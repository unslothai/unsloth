# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selective prompt-encoder hidden states: the requested entries are the stock tensors bitwise, nothing else is kept,
and anything unexpected falls back to the stock call. Runs on CPU with a random tiny Qwen3-VL."""

from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")

from core.inference import diffusion_te_select as sel  # noqa: E402

VOCAB, HIDDEN, LAYERS = 512, 64, 9
FLUX3_LAYERS = (
    2,
    4,
    6,
    8,
)  # FLUX.3's (4, 8, ..., 32) pattern on a 9-layer stack: every one an intermediate state


def _encoder(layers: int = LAYERS):
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    config = Qwen3VLConfig(
        vision_config = Qwen3VLVisionConfig(
            depth = 1,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 4,
            out_hidden_size = HIDDEN,
            deepstack_visual_indexes = [],
        ).to_dict(),
        text_config = Qwen3VLTextConfig(
            hidden_size = HIDDEN,
            intermediate_size = 128,
            num_hidden_layers = layers,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 16,
            vocab_size = VOCAB,
            use_cache = False,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        ).to_dict(),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def _inputs(length: int = 37):
    ids = (torch.arange(length)[None] * 13 % (VOCAB - 10)) + 10
    mask = torch.ones_like(ids)
    mask[:, -5:] = 0  # right padding, as FLUX.3 pads its window
    pos = torch.arange(length)[None].masked_fill(mask == 0, 1)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos}


def _flux3_embeds(encoder, layers = FLUX3_LAYERS):
    """diffusers' ``_get_qwen3_vl_prompt_embeds`` read of the encoder output, verbatim."""
    output = encoder(**_inputs(), output_hidden_states = True, use_cache = False)
    out = torch.stack([output.hidden_states[k] for k in layers], dim = 1)
    b, c, s, h = out.shape
    return out.permute(0, 2, 1, 3).reshape(b, s, c * h)


def _qwen21_final(encoder):
    """The Qwen-Image-2.1 pipeline's read: ``hidden_states[-1]`` with the final norm neutralized for the call."""
    text_model = getattr(encoder.model, "language_model", encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = encoder(**_inputs(), output_hidden_states = True)
    finally:
        handle.remove()
    return outputs.hidden_states[-1]


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(sel.SELECT_ENV, raising = False)


def test_flux3_layers_are_the_stock_tensors_bitwise():
    encoder = _encoder()
    with torch.no_grad():
        stock = _flux3_embeds(encoder)
        before = sel.call_count()
        got = sel.run_selected(encoder, FLUX3_LAYERS, lambda: _flux3_embeds(encoder))
    assert sel.call_count() == before + 1  # the selective path ran, not the fallback
    assert torch.equal(got.view(torch.int32), stock.view(torch.int32))


def test_qwen21_final_pre_norm_state_is_the_stock_tensor_bitwise():
    encoder = _encoder()
    with torch.no_grad():
        stock = _qwen21_final(encoder)
        before = sel.call_count()
        got = sel.run_selected(encoder, (-1,), lambda: _qwen21_final(encoder), last_pre_norm = True)
    assert sel.call_count() == before + 1
    assert torch.equal(got.view(torch.int32), stock.view(torch.int32))


def test_only_the_requested_states_stay_alive():
    """Every unrequested layer output is freed during the forward: the recorded states are the only survivors."""
    encoder = _encoder()
    alive: list = []
    hooks = [
        layer.register_forward_hook(
            lambda m, a, out: alive.append(weakref.ref(out[0] if isinstance(out, tuple) else out))
        )
        for layer in encoder.model.language_model.layers
    ]
    seen = {}
    try:
        with torch.no_grad(), sel.selected_hidden_states(encoder, FLUX3_LAYERS) as armed:
            assert armed
            output = encoder(**_inputs(), output_hidden_states = True, use_cache = False)
            gc.collect()
            seen = {i for i, ref in enumerate(alive) if ref() is not None}
            kept = [output.hidden_states[k] for k in FLUX3_LAYERS]
    finally:
        for h in hooks:
            h.remove()
    assert seen == {k - 1 for k in FLUX3_LAYERS}
    assert len(kept) == len(FLUX3_LAYERS)
    with pytest.raises(sel.SelectionError):
        output.hidden_states[1]


def test_an_unrecorded_read_falls_back_to_the_stock_call():
    encoder = _encoder()
    calls = []

    def read():
        calls.append(1)
        output = encoder(**_inputs(), output_hidden_states = True, use_cache = False)
        return output.hidden_states[3]  # not requested

    with torch.no_grad():
        stock = encoder(**_inputs(), output_hidden_states = True, use_cache = False).hidden_states[3]
        got = sel.run_selected(encoder, FLUX3_LAYERS, read)
    assert len(calls) == 2  # selective attempt, then the stock re-run
    assert torch.equal(got, stock)


def test_unsupported_requests_and_the_kill_switch_run_stock(monkeypatch):
    encoder = _encoder()
    assert sel.plan(encoder, (0, 2)) is None  # embeddings: not handled
    assert (
        sel.plan(encoder, (LAYERS,)) is None
    )  # the final entry without the caller's pre-norm contract
    assert sel.plan(encoder, (-1,), last_pre_norm = True) is not None
    assert sel.plan(torch.nn.Linear(2, 2), (1,)) is None
    monkeypatch.setenv(sel.SELECT_ENV, "0")
    assert sel.plan(encoder, FLUX3_LAYERS) is None
    with torch.no_grad():
        before = sel.call_count()
        stock = _flux3_embeds(encoder)
        got = sel.run_selected(encoder, FLUX3_LAYERS, lambda: _flux3_embeds(encoder))
    assert sel.call_count() == before
    assert torch.equal(got, stock)


def test_dead_tail_passthroughs_keep_the_selection_exact():
    """After Studio's dead-tail truncation the trailing slots are passthroughs: the selection still matches."""
    from core.inference.diffusion_te_truncate import _make_passthrough

    encoder = _encoder()
    layers = encoder.model.language_model.layers
    fmt = {"tuple_len": None, "seen": True}

    def record(_m, _a, out):
        fmt["tuple_len"] = len(out) if isinstance(out, tuple) else None

    handle = layers[0].register_forward_hook(record)
    with torch.no_grad():
        _flux3_embeds(encoder)
    handle.remove()
    for i in range(max(FLUX3_LAYERS), LAYERS):
        layers[i] = _make_passthrough(layers[i], fmt)
    with torch.no_grad():
        stock = _flux3_embeds(encoder)
        got = sel.run_selected(encoder, FLUX3_LAYERS, lambda: _flux3_embeds(encoder))
    assert torch.equal(got.view(torch.int32), stock.view(torch.int32))


def test_hooks_are_removed_after_the_call():
    encoder = _encoder()
    counts = (
        len(encoder._forward_pre_hooks),
        len(encoder._forward_hooks),
        sum(len(layer._forward_hooks) for layer in encoder.model.language_model.layers),
    )
    with torch.no_grad():
        sel.run_selected(encoder, FLUX3_LAYERS, lambda: _flux3_embeds(encoder))
    assert counts == (
        len(encoder._forward_pre_hooks),
        len(encoder._forward_hooks),
        sum(len(layer._forward_hooks) for layer in encoder.model.language_model.layers),
    )
    with torch.no_grad():
        full = encoder(**_inputs(), output_hidden_states = True, use_cache = False)
    assert len(full.hidden_states) == LAYERS + 1


def test_flux3_prompt_helper_matches_bitwise():
    """diffusers' own FLUX.3 prompt helper (when this diffusers has it), stock vs selective."""
    encoders = pytest.importorskip("diffusers.modular_pipelines.flux3.encoders")
    helper = encoders.Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = True,
        ):
            return messages[0]["content"]

        def tokenizer(
            self,
            text,
            return_tensors = "pt",
            padding = None,
            padding_side = None,
            truncation = True,
            max_length = None,
        ):
            ids = (torch.arange(len(text) + 3)[None] * 11 % (VOCAB - 10)) + 10
            mask = torch.ones_like(ids)
            if padding == "max_length" and max_length > ids.shape[1]:
                pad = max_length - ids.shape[1]
                ids = torch.cat([ids, torch.zeros(1, pad, dtype = ids.dtype)], 1)
                mask = torch.cat([mask, torch.zeros(1, pad, dtype = mask.dtype)], 1)
            return {"input_ids": ids[:, :max_length], "attention_mask": mask[:, :max_length]}

    tok = _Tok()
    tok.tokenizer = tok.tokenizer
    encoder = _encoder()
    kwargs = dict(
        text_encoder = encoder,
        tokenizer = tok,
        prompt = "a red fox in snow",
        device = "cpu",
        max_sequence_length = 64,
        pad_multiple = 8,
        hidden_states_layers = FLUX3_LAYERS,
    )
    with torch.no_grad():
        stock = helper(**kwargs)
        before = sel.call_count()
        got = sel.run_selected(encoder, FLUX3_LAYERS, lambda: helper(**kwargs))
    assert sel.call_count() == before + 1
    assert torch.equal(got.view(torch.int32), stock.view(torch.int32))
