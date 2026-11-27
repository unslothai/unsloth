# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Z-Image and Krea 2 prompt encodes build no text-encoder K/V cache, keep only the hidden states they read, and
return bit-identical embeddings. Tiny random encoders on CPU; the real pipelines' encoding methods."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")
try:
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
except Exception as exc:  # noqa: BLE001 - an older or broken diffusers: nothing to wrap
    pytest.skip(f"diffusers Z-Image / Krea 2 pipelines unavailable: {exc}", allow_module_level = True)

from core.inference import diffusion_te_lean as lean  # noqa: E402
from core.inference import diffusion_te_select as sel  # noqa: E402
from core.inference.diffusion_prompt_cache import is_transparent_wrapper  # noqa: E402

VOCAB, HIDDEN, LAYERS = 512, 64, 6


class _Batch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to(self, device):
        return self


class _Tokenizer:
    """Deterministic ids from the text; pads to ``max_length`` like the real one (Krea 2 pads mid-template)."""

    def apply_chat_template(self, messages, **kwargs):
        return "<u>" + messages[0]["content"] + "</u>"

    def __call__(
        self,
        text,
        padding = None,
        max_length = None,
        truncation = None,
        return_tensors = None,
    ):
        rows, masks = [], []
        for t in text:
            ids = [10 + (ord(c) * 7 + i) % (VOCAB - 20) for i, c in enumerate(t)]
            if max_length is not None:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length" and max_length is not None:
                mask += [0] * (max_length - len(ids))
                ids += [0] * (max_length - len(ids))
            rows.append(ids)
            masks.append(mask)
        width = max(len(r) for r in rows)
        rows = [r + [0] * (width - len(r)) for r in rows]
        masks = [m + [0] * (width - len(m)) for m in masks]
        return _Batch(input_ids = torch.tensor(rows), attention_mask = torch.tensor(masks))


def _zimage_encoder():
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(
        hidden_size = HIDDEN,
        intermediate_size = 128,
        num_hidden_layers = LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        vocab_size = VOCAB,
        use_cache = True,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3Model(config).eval()


def _krea2_encoder():
    from transformers import Qwen3VLModel
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
            num_hidden_layers = LAYERS,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 16,
            vocab_size = VOCAB,
            use_cache = True,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        ).to_dict(),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3VLModel(config).eval()


class _ZImage:
    """ZImagePipeline's own ``_encode_prompt`` on a hand-built instance (no transformer / VAE / scheduler)."""

    _encode_prompt = ZImagePipeline._encode_prompt

    def __init__(self):
        self.text_encoder = _zimage_encoder()
        self.tokenizer = _Tokenizer()

    def encode(self):
        return self._encode_prompt(
            ["a red fox", "a long prompt about a rainy city"], device = "cpu", max_sequence_length = 32
        )


class _Krea2:
    """Krea2Pipeline's own ``get_text_hidden_states`` on a hand-built instance."""

    get_text_hidden_states = Krea2Pipeline.get_text_hidden_states

    def __init__(self):
        self.text_encoder = _krea2_encoder()
        self.tokenizer = _Tokenizer()
        self.text_encoder_select_layers = (1, 2, 4, 5)
        self.prompt_template_encode_prefix = "<sys>describe</sys><u>"
        self.prompt_template_encode_suffix = "</u><a>"
        self.prompt_template_encode_start_idx = 6
        self.prompt_template_encode_num_suffix_tokens = 7

    def encode(self):
        return self.get_text_hidden_states(
            ["a red fox", "a long prompt about a rainy city"], max_sequence_length = 40, device = "cpu"
        )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(lean._ENV, raising = False)
    monkeypatch.delenv(sel.SELECT_ENV, raising = False)


def _run(pipe):
    caches = []
    hook = pipe.text_encoder.register_forward_hook(
        lambda module, args, output: caches.append(getattr(output, "past_key_values", None))
    )
    try:
        with torch.no_grad():
            out = pipe.encode()
    finally:
        hook.remove()
    return out, caches


def _flat(out):
    return [t for t in (out if isinstance(out, (list, tuple)) else [out])]


def _bitwise(a, b):
    a, b = _flat(a), _flat(b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape
        if x.is_floating_point():
            assert torch.equal(x.view(torch.int32), y.view(torch.int32))
        else:
            assert torch.equal(x, y)


@pytest.mark.parametrize("make", [_ZImage, _Krea2], ids = ["z-image", "krea2"])
def test_encode_builds_no_kv_cache_keeps_only_read_states_and_is_bitwise(make):
    pipe = make()
    stock, caches = _run(pipe)
    assert caches[0] is not None  # the stock call builds a cache the pipeline never reads

    assert lean.lean_text_encode(pipe) is True
    before = sel.call_count()
    lean_out, lean_caches = _run(pipe)
    assert sel.call_count() == before + 1  # the selective encode ran, not the stock fallback
    assert lean_caches == [None]
    _bitwise(lean_out, stock)
    # only this call: a direct forward of the same encoder still caches as configured
    with torch.no_grad():
        direct = pipe.text_encoder(input_ids = torch.tensor([[11, 12, 13]]))
    assert direct.past_key_values is not None


@pytest.mark.parametrize("make", [_ZImage, _Krea2], ids = ["z-image", "krea2"])
def test_install_is_idempotent_transparent_and_each_kill_switch_keeps_its_half(make, monkeypatch):
    pipe = make()
    stock, _ = _run(pipe)
    assert lean.lean_text_encode(pipe) is True
    name = "_encode_prompt" if make is _ZImage else "get_text_hidden_states"
    wrapped = pipe.__dict__[name]
    assert is_transparent_wrapper(wrapped)  # the prompt cache still sees through it
    assert lean.lean_text_encode(pipe) is True and pipe.__dict__[name] is wrapped

    monkeypatch.setenv(lean._ENV, "0")
    off = make()
    assert lean.lean_text_encode(off) is False
    assert _run(off)[1][0] is not None
    monkeypatch.delenv(lean._ENV)

    # selection off: the cache is still skipped and the output unchanged
    monkeypatch.setenv(sel.SELECT_ENV, "0")
    no_select = make()
    assert lean.lean_text_encode(no_select) is True
    before = sel.call_count()
    out, caches = _run(no_select)
    assert sel.call_count() == before and caches == [None]
    _bitwise(out, stock)


def test_other_pipelines_are_left_alone():
    class _Other:
        text_encoder = torch.nn.Linear(2, 2)

        def _encode_prompt(self):
            return self.text_encoder(torch.zeros(1, 2))

    assert lean.lean_text_encode(_Other()) is False
    assert lean.lean_text_encode(object()) is False
    assert lean.lean_text_encode(None) is False
