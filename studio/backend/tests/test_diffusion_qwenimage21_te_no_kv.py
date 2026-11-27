# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 prompt encoding builds no text-encoder K/V cache and returns the same embeddings. Runs on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")
q21 = pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")

from core.inference import diffusion_qwenimage21_te as te  # noqa: E402

VOCAB, HIDDEN, IMG_PAD = 512, 128, 7


def _encoder():
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
            intermediate_size = 256,
            num_hidden_layers = 3,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 32,
            vocab_size = VOCAB,
            use_cache = True,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [8, 4, 4],
                "mrope_interleaved": True,
            },
        ).to_dict(),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


class _Inputs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to(self, device):
        return self


class _Processor:
    def __call__(self, text, **kwargs):
        ids = torch.stack([(torch.arange(40) * (7 + len(t)) % (VOCAB - 10)) + 10 for t in text])
        return _Inputs(input_ids = ids, attention_mask = torch.ones_like(ids))


class _Pipe:
    """The real pipeline's encoding methods on a hand-built instance (no transformer / VAE / scheduler)."""

    _get_qwen_prompt_embeds = q21.QwenImage21Pipeline._get_qwen_prompt_embeds
    _extract_masked_hidden = q21.QwenImage21Pipeline._extract_masked_hidden

    def __init__(self):
        self.text_encoder = _encoder()
        self.processor = _Processor()
        self.prompt_template_t2i = "<s>{}</s>"
        self._drop_idx = 4
        self._img_token_id = IMG_PAD


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(te._ENV, raising = False)


def _encode(pipe):
    caches = []
    hook = pipe.text_encoder.register_forward_hook(
        lambda module, args, output: caches.append(getattr(output, "past_key_values", None))
    )
    try:
        with torch.no_grad():
            out = pipe._get_qwen_prompt_embeds(
                ["a red fox", "a long prompt about a city"], device = "cpu"
            )
    finally:
        hook.remove()
    return out, caches


def test_prompt_encoding_builds_no_kv_cache_and_keeps_the_embeddings_bitwise():
    pipe = _Pipe()
    (embeds, mask, pad), caches = _encode(pipe)
    assert caches[0] is not None  # the stock call builds a cache the pipeline never reads

    assert te.skip_text_encoder_kv_cache(pipe) is True
    (embeds2, mask2, pad2), caches2 = _encode(pipe)
    assert caches2 == [None]
    assert torch.equal(embeds2.view(torch.int32), embeds.view(torch.int32))
    assert torch.equal(mask2, mask) and torch.equal(pad2, pad)
    # only this call: a direct forward of the same encoder still caches as configured
    with torch.no_grad():
        direct = pipe.text_encoder(input_ids = torch.tensor([[11, 12, 13]]))
    assert direct.past_key_values is not None


def test_install_is_idempotent_and_the_kill_switch_keeps_the_stock_call(monkeypatch):
    pipe = _Pipe()
    assert te.skip_text_encoder_kv_cache(pipe) is True
    wrapped = pipe._get_qwen_prompt_embeds
    assert te.skip_text_encoder_kv_cache(pipe) is True and pipe._get_qwen_prompt_embeds is wrapped

    monkeypatch.setenv(te._ENV, "0")
    stock = _Pipe()
    assert te.skip_text_encoder_kv_cache(stock) is False
    assert _encode(stock)[1][0] is not None


def test_other_pipelines_are_left_alone():
    class _Other:
        text_encoder = torch.nn.Linear(2, 2)

        def _get_qwen_prompt_embeds(self):
            return None

    other = _Other()
    assert te.skip_text_encoder_kv_cache(other) is False
    assert te.skip_text_encoder_kv_cache(object()) is False


def test_final_state_selection_keeps_the_embeddings_bitwise(monkeypatch):
    from core.inference import diffusion_te_select as sel

    monkeypatch.delenv(sel.SELECT_ENV, raising = False)
    pipe = _Pipe()
    (embeds, mask, pad), _ = _encode(pipe)
    assert te.skip_text_encoder_kv_cache(pipe) is True
    assert te.select_final_hidden_state(pipe) is True
    wrapped = pipe._get_qwen_prompt_embeds
    assert te.select_final_hidden_state(pipe) is True and pipe._get_qwen_prompt_embeds is wrapped
    assert te.skip_text_encoder_kv_cache(pipe) is True and pipe._get_qwen_prompt_embeds is wrapped
    before = sel.call_count()
    (embeds2, mask2, pad2), caches = _encode(pipe)
    assert sel.call_count() == before + 1  # the selective encode ran (no stock fallback)
    assert caches == [None]
    assert torch.equal(embeds2.view(torch.int32), embeds.view(torch.int32))
    assert torch.equal(mask2, mask) and torch.equal(pad2, pad)


def test_final_state_selection_kill_switch(monkeypatch):
    from core.inference import diffusion_te_select as sel

    monkeypatch.setenv(sel.SELECT_ENV, "0")
    pipe = _Pipe()
    assert te.select_final_hidden_state(pipe) is False
    assert te.select_final_hidden_state(object()) is False
