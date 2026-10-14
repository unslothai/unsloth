# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generic text-encoder dead-tail truncation: the probe finds what encode_prompt reads, the rest never runs."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.inference.diffusion_precision import _keep_bf16_block_fqns  # noqa: E402
from core.inference.diffusion_te_truncate import (  # noqa: E402
    TRUNCATE_ENV,
    is_passthrough,
    probe_consumption,
    truncate_text_encoders,
)

LAYERS = 8


def _qwen3(causal: bool = True, tie: bool = False):
    cfg = transformers.Qwen3Config(
        vocab_size = 97,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        tie_word_embeddings = tie,
    )
    torch.manual_seed(0)
    cls = transformers.Qwen3ForCausalLM if causal else transformers.Qwen3Model
    model = cls(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _ids(prompt: str, device = None):
    torch.manual_seed(abs(hash(prompt)) % 1000)
    n = 3 + len(prompt) % 7
    return torch.randint(1, 97, (1, n), device = device)


class _Pipe:
    """Stands in for a diffusers pipeline: ``encode_prompt`` is the only contract the probe uses."""

    def __init__(self, encoder, read):
        self.text_encoder = encoder
        self._read = read

    def encode_prompt(self, prompt, device = None, num_images_per_prompt = 1):
        out = self.text_encoder(input_ids = _ids(prompt, device), output_hidden_states = True)
        return self._read(out), torch.ones(3)


READERS = {
    # FLUX.2-klein: stack of three intermediate hidden states.
    "klein": (lambda o: torch.stack([o.hidden_states[k] for k in (2, 4, 6)], 1), True, 6, True),
    # Z-Image: penultimate hidden state of a head-less model.
    "z-image": (lambda o: o.hidden_states[-2], False, LAYERS - 1, False),
    # HunyuanVideo-1.5: hidden_states[-3].
    "hv15": (lambda o: o.hidden_states[-3], True, LAYERS - 2, True),
    # Qwen-Image: the final hidden state, never the head.
    "qwen-image": (lambda o: o.hidden_states[-1], True, LAYERS, True),
    # Every layer and the logits (a consumer this must refuse to trim).
    "logits": (lambda o: o.logits, True, LAYERS, False),
}


PROMPTS = ("a photo of a cat", "two robots playing chess in a library", "x", "a much longer prompt " * 3)


def _encode_all(pipe):
    with torch.inference_mode():
        return [pipe.encode_prompt(p)[0].clone() for p in PROMPTS]


@pytest.mark.parametrize("name", list(READERS))
def test_truncation_is_byte_identical(name, monkeypatch):
    monkeypatch.delenv(TRUNCATE_ENV, raising = False)
    read, causal, kept, head_dropped = READERS[name]
    pipe = _Pipe(_qwen3(causal = causal), read)
    ref = copy.deepcopy(pipe)
    report = truncate_text_encoders(pipe)
    stack = pipe.text_encoder.model.layers if causal else pipe.text_encoder.layers
    assert sum(not is_passthrough(m) for m in stack) == kept, report
    assert len(stack) == LAYERS
    if causal:
        assert (type(pipe.text_encoder.lm_head).__name__ == "NoLogitsHead") is head_dropped
    for a, b in zip(_encode_all(ref), _encode_all(pipe)):
        assert a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    if kept < LAYERS or head_dropped:
        entry = report["truncated"]["text_encoder"]
        assert entry["kept_layers"] == kept and entry["bytes_freed"] > 0


def test_kill_switch_keeps_encoder(monkeypatch):
    monkeypatch.setenv(TRUNCATE_ENV, "0")
    pipe = _Pipe(_qwen3(), READERS["klein"][0])
    assert truncate_text_encoders(pipe) == {"skipped": TRUNCATE_ENV}
    assert not any(is_passthrough(m) for m in pipe.text_encoder.model.layers)


def test_probe_failure_restores_everything(monkeypatch):
    monkeypatch.delenv(TRUNCATE_ENV, raising = False)
    pipe = _Pipe(_qwen3(), READERS["klein"][0])
    ref = copy.deepcopy(pipe)
    originals = list(pipe.text_encoder.model.layers)
    head = pipe.text_encoder.lm_head
    calls = {"n": 0}
    real = pipe.encode_prompt

    def flaky(prompt, device = None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(prompt, device)

    pipe.encode_prompt = flaky
    out = truncate_text_encoders(pipe)
    assert out["skipped"].startswith("probe: boom")
    assert list(pipe.text_encoder.model.layers) == originals and pipe.text_encoder.lm_head is head
    pipe.encode_prompt = real
    # The probe ran the model (installing any once-per-model output recorders) while slots were swapped; the restored
    # layers must still record every hidden state.
    for a, b in zip(_encode_all(ref), _encode_all(pipe)):
        assert torch.equal(a, b)


def test_uncallable_encode_prompt_declines(monkeypatch):
    monkeypatch.delenv(TRUNCATE_ENV, raising = False)

    class NeedsImage(_Pipe):
        def encode_prompt(self, prompt, image):  # a required argument the probe cannot fill
            return super().encode_prompt(prompt)

    pipe = NeedsImage(_qwen3(), READERS["klein"][0])
    assert truncate_text_encoders(pipe)["skipped"].startswith("probe:")
    assert not any(is_passthrough(m) for m in pipe.text_encoder.model.layers)


def test_encoder_the_probe_never_reaches_is_untouched(monkeypatch):
    monkeypatch.delenv(TRUNCATE_ENV, raising = False)
    pipe = _Pipe(_qwen3(), READERS["klein"][0])
    pipe.text_encoder_2 = _qwen3()  # e.g. HunyuanVideo-1.5's byT5, only run for quoted text
    report = truncate_text_encoders(pipe)
    assert "text_encoder_2" not in report["plan"]
    assert not any(is_passthrough(m) for m in pipe.text_encoder_2.model.layers)
    assert type(pipe.text_encoder_2.lm_head).__name__ != "NoLogitsHead"


def test_probe_is_side_effect_free():
    pipe = _Pipe(_qwen3(), READERS["hv15"][0])
    layers = list(pipe.text_encoder.model.layers)
    plan = probe_consumption(pipe)
    assert plan["text_encoder"]["last_consumed"] == LAYERS - 3
    assert list(pipe.text_encoder.model.layers) == layers


def test_keep_list_counts_live_slots(monkeypatch):
    monkeypatch.delenv(TRUNCATE_ENV, raising = False)
    pipe = _Pipe(_qwen3(), READERS["klein"][0])
    truncate_text_encoders(pipe)
    keep = _keep_bf16_block_fqns(pipe.text_encoder, 1, 2)
    assert keep == {"model.layers.0", "model.layers.4", "model.layers.5"}
