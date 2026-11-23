# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The H3 trainer loads only the weights its cache build reads, and keeps caption rows in their bf16 dtype."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch = pytest.importorskip("torch")


def _needs_tiny_stack():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    for name in (
        "AutoencoderKLMiniMaxH3",
        "AutoencoderKLMiniMaxH3Audio",
        "MiniMaxH3Transformer3DModel",
    ):
        if not hasattr(diffusers, name):
            pytest.skip(f"diffusers has no {name}")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers has no Qwen3-VL")
    pytest.importorskip("av")


@pytest.fixture(scope = "module")
def release(tmp_path_factory):
    _needs_tiny_stack()
    from _minimax_h3_video_tiny import make_tiny_h3_video_release, write_clip

    root = make_tiny_h3_video_release(tmp_path_factory.mktemp("h3") / "release")
    write_clip(root.parent / "clip.mp4", seed = 0)
    return root


@pytest.fixture(scope = "module")
def loaded(release):
    from core.training import diffusion_h3_trainer as h3
    return h3._load_conditioners(
        types.SimpleNamespace(base_model = str(release), hf_token = None), "cpu"
    )


def test_the_conditioner_holds_only_the_layers_before_the_read_state(loaded):
    from core.training import diffusion_h3_trainer as h3
    from _minimax_h3_video_tiny import TE_LAYERS

    te = loaded.text_encoder
    layers = te.model.language_model.layers
    read = h3._H3_TEXT_ENCODER_LAYER
    assert TE_LAYERS > read + 1
    # 50 real layers plus a weightless pass-through slot; layers 50+ were never loaded.
    assert len(layers) == read + 1
    assert not list(layers[read].parameters())
    assert te.config.text_config.num_hidden_layers == read + 1
    assert isinstance(te.model.language_model.norm, torch.nn.Identity)
    assert isinstance(te.lm_head, torch.nn.Identity)


def test_the_vaes_keep_only_their_encoders(loaded):
    assert loaded.vae.decoder is None and loaded.vae.post_quant_conv is None
    assert loaded.audio_vae.decoder is None and loaded.audio_vae.dec_in_proj is None
    assert loaded.vae.encoder is not None and loaded.audio_vae.encoder is not None


def test_the_trimmed_load_encodes_bitwise_like_the_full_one(release, loaded):
    from core.training import diffusion_h3_trainer as h3
    from core.training.diffusion_h3_clips import decode_clip
    from _minimax_h3_video_tiny import full_reference

    if not hasattr(loaded.processor, "create_mm_token_type_ids"):
        pytest.skip("this transformers' Qwen3VLProcessor has no create_mm_token_type_ids")
    ref = full_reference(release)
    caption = "a cat the dog runs"
    hidden = h3._encode_prompt(loaded, caption, "cpu")
    ids = loaded.tokenizer(caption, add_special_tokens = False)["input_ids"]
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        out = ref["text_encoder"].model(
            input_ids = input_ids,
            attention_mask = torch.ones_like(input_ids),
            mm_token_type_ids = torch.tensor(loaded.processor.create_mm_token_type_ids([ids])),
            use_cache = False,
            output_hidden_states = True,
        )
    expected = out.hidden_states[h3._H3_TEXT_ENCODER_LAYER]
    assert hidden.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(hidden, expected)

    frames, waveform = decode_clip(
        release.parent / "clip.mp4", num_frames = h3.H3_TRAIN_NUM_FRAMES, width = 64, height = 64
    )
    got = h3._encode_video_stats(loaded.vae, frames, "cpu")
    want = h3._encode_video_stats(ref["vae"], frames, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert torch.equal(
        h3._encode_audio_latents(loaded.audio_vae, waveform, "cpu"),
        h3._encode_audio_latents(ref["audio_vae"], waveform, "cpu"),
    )


def test_an_unreadable_conditioner_config_falls_back_to_the_whole_encoder():
    from core.training import diffusion_h3_trainer as h3
    assert h3._read_layers_config(types.SimpleNamespace(), None, {}) is None


def _bf16_pipe():
    tokenizer = lambda caption, add_special_tokens = False: {"input_ids": [1, 2, 3]}  # noqa: E731
    processor = types.SimpleNamespace(create_mm_token_type_ids = lambda ids: [[0] * len(ids[0])])
    text_config = types.SimpleNamespace(num_hidden_layers = 51)
    encoder = types.SimpleNamespace(config = types.SimpleNamespace(text_config = text_config))
    return types.SimpleNamespace(tokenizer = tokenizer, processor = processor, text_encoder = encoder)


def test_caption_rows_stay_bf16_on_the_host(monkeypatch):
    """fp32 rows doubled the caption bytes for values that are bf16 to begin with."""
    from core.training import diffusion_h3_trainer as h3

    produced = torch.randn(1, 3, 8).to(torch.bfloat16)
    monkeypatch.setattr(h3, "_early_exit_hidden_state", lambda *_a, **_k: produced)
    hidden = h3._encode_prompt(_bf16_pipe(), "a caption", "cpu")
    assert hidden.dtype == torch.bfloat16 and hidden.device.type == "cpu"
    assert torch.equal(hidden, produced)
    # What reaches the denoiser is unchanged: the loop casts to the weight dtype.
    for dtype in (torch.bfloat16, torch.float32):
        assert torch.equal(hidden.to(dtype), produced.float().to(dtype))


def test_h3_image_caption_rows_stay_in_the_encoder_dtype(monkeypatch):
    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training import diffusion_h3_image_trainer as trainer

    produced = torch.randn(1, 3, 8).to(torch.bfloat16)
    monkeypatch.setattr(h3i, "qwen3vl_hidden_state", lambda *_a, **_k: produced)
    tokenizer = lambda caption, add_special_tokens = False: {"input_ids": [1, 2, 3]}  # noqa: E731
    out = trainer.encode_captions(None, tokenizer, None, ["a caption"], "cpu")
    assert out["a caption"].dtype == torch.bfloat16
    assert torch.equal(out["a caption"], produced)
