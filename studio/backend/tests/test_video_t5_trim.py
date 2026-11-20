# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan's trimmed umT5 encode returns what diffusers' padded encode returns, and only where it is allowed to run."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from diffusers import WanPipeline  # noqa: E402

from core.inference import video_t5_trim  # noqa: E402

MAX_LEN = 64


class _Tokenizer:
    """Right-padded ids from the word count, the shape contract of the real tokenizer call."""

    def __call__(
        self,
        prompts,
        max_length = MAX_LEN,
        **_,
    ):
        ids = torch.zeros(len(prompts), max_length, dtype = torch.long)
        mask = torch.zeros(len(prompts), max_length, dtype = torch.long)
        for row, prompt in enumerate(prompts):
            n = min(len(prompt.split()) + 1, max_length)
            generator = torch.Generator().manual_seed(n)
            ids[row, : n - 1] = torch.randint(2, 200, (n - 1,), generator = generator)
            ids[row, n - 1] = 1
            mask[row, :n] = 1
        return types.SimpleNamespace(input_ids = ids, attention_mask = mask)


def _encoder():
    torch.manual_seed(0)
    config = transformers.UMT5Config(
        vocab_size = 256,
        d_model = 64,
        d_kv = 16,
        d_ff = 128,
        num_layers = 2,
        num_heads = 4,
        dropout_rate = 0.0,
    )
    return transformers.UMT5EncoderModel(config).eval()


class _Pipe:
    """Borrows diffusers' own Wan prompt encode so the test follows the pinned diffusers, not a copy of it."""

    _get_t5_prompt_embeds = WanPipeline._get_t5_prompt_embeds

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.text_encoder = _encoder()


class WanStubPipeline(_Pipe):
    pass


class OtherPipeline(_Pipe):
    pass


def _encode(pipe, prompts):
    with torch.inference_mode():
        return pipe._get_t5_prompt_embeds(
            prompt = prompts,
            max_sequence_length = MAX_LEN,
            device = torch.device("cpu"),
            dtype = torch.float32,
        )


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.delenv(video_t5_trim.ENV, raising = False)
    video_t5_trim.STATS.update(trimmed = 0, passed = 0)


@pytest.mark.parametrize(
    "prompts",
    [
        ["a red fox in fresh snow"],
        ["a red fox in fresh snow at golden hour, detailed fur", "blurry"],
    ],
)
def test_trimmed_encode_matches_padded(prompts):
    pipe = WanStubPipeline()
    reference = _encode(pipe, prompts)
    assert video_t5_trim.install(pipe)
    trimmed = _encode(pipe, prompts)
    assert video_t5_trim.STATS["trimmed"] == 1
    assert trimmed.shape == reference.shape == (len(prompts), MAX_LEN, 64)
    assert torch.allclose(trimmed, reference, atol = 1e-5, rtol = 0)
    longest = max(len(p.split()) + 1 for p in prompts)
    assert torch.count_nonzero(trimmed[:, longest:]) == 0
    assert torch.equal(trimmed[:, longest:], reference[:, longest:])
    assert "forward" not in pipe.text_encoder.__dict__


def test_kill_switch_and_other_pipelines(monkeypatch):
    assert not video_t5_trim.install(OtherPipeline())
    monkeypatch.setenv(video_t5_trim.ENV, "0")
    assert not video_t5_trim.install(WanStubPipeline())


def test_install_is_idempotent():
    pipe = WanStubPipeline()
    assert video_t5_trim.install(pipe)
    assert not video_t5_trim.install(pipe)


def test_full_length_prompt_passes_through():
    pipe = WanStubPipeline()
    assert video_t5_trim.install(pipe)
    prompt = " ".join(["word"] * (MAX_LEN + 5))
    _encode(pipe, [prompt])
    assert video_t5_trim.STATS == {"trimmed": 0, "passed": 1}


def test_left_padded_mask_passes_through():
    encoder = _encoder()
    forward = video_t5_trim._trimmed_forward(encoder, encoder.forward)
    ids = torch.randint(2, 200, (1, 16))
    mask = torch.zeros(1, 16, dtype = torch.long)
    mask[0, 8:] = 1
    with torch.inference_mode():
        out = forward(ids, mask)
    assert out.last_hidden_state.shape == (1, 16, 64)
    assert video_t5_trim.STATS == {"trimmed": 0, "passed": 1}


def test_existing_instance_forward_is_restored():
    pipe = WanStubPipeline()
    encoder = pipe.text_encoder
    original = encoder.forward

    def hooked(*args, **kwargs):
        return original(*args, **kwargs)

    encoder.forward = hooked
    assert video_t5_trim.install(pipe)
    _encode(pipe, ["a red fox"])
    assert encoder.__dict__["forward"] is hooked
    assert video_t5_trim.STATS["trimmed"] == 1
