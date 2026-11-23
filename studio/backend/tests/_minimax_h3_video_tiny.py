# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tiny random MiniMax-H3 (video) modular release on disk: a 54-layer Qwen3-VL conditioner (the trainer reads
hidden_states[50]), both VAEs on the released grid, and a two-block denoiser, all small enough for CPU tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

TEXT_DIM = 48
TE_LAYERS = 54

TINY_VAE = {
    "latent_channels": 24,
    "block_out_channels": [8, 8, 16, 16, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "decoder_num_layers": 1,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "latents_mean": [0.05 * (i % 7) for i in range(24)],
    "latents_std": [1.0 + 0.03 * i for i in range(24)],
}

TINY_AUDIO_VAE = {
    "encoder_dim": 2,
    "latent_dim": 64,
    "latent_channels": 32,
    "decoder_dim": 256,
    "num_attention_heads": 2,
    "latents_mean": [0.01 * i for i in range(32)],
    "latents_std": [1.0 + 0.02 * i for i in range(32)],
}

TINY_TRANSFORMER = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 32,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 64,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": TEXT_DIM,
    "freq_dim": 16,
    "time_embed_hidden_dim": 32,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}


def _spec(cls_lib: str, cls_name: str, root: Path, subfolder: str) -> list:
    return [
        cls_lib,
        cls_name,
        {
            "type_hint": [cls_lib, cls_name],
            "pretrained_model_name_or_path": str(root),
            "subfolder": subfolder,
            "variant": None,
            "revision": None,
        },
    ]


def make_tokenizer_and_processor(out: Path) -> Any:
    """Qwen2TokenizerFast + Qwen3VLProcessor, the classes the H3 index names (4.57 refuses other tokenizers)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from _minimax_h3_image_tiny import SPECIAL

    words = [chr(c) for c in range(33, 127)] + ["the", "a", "cat", "dog", "runs", "sings", "[UNK]"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(SPECIAL + words))}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = Qwen2TokenizerFast(tokenizer_object = tok, unk_token = "[UNK]", eos_token = "<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": SPECIAL})
    fast.save_pretrained(str(out / "tokenizer"))
    image = Qwen2VLImageProcessor(
        patch_size = 16,
        temporal_patch_size = 2,
        merge_size = 2,
        min_pixels = 32 * 32,
        max_pixels = 2048 * 2048,
    )
    processor = Qwen3VLProcessor(
        image_processor = image, tokenizer = fast, video_processor = Qwen3VLVideoProcessor()
    )
    processor.save_pretrained(str(out / "processor"))
    return fast


def make_tiny_h3_video_release(out: Path, seed: int = 0) -> Path:
    """Writes the release under ``out`` (bf16 conditioner, fp32 VAEs, bf16 denoiser) and returns it."""
    import torch
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Transformer3DModel,
    )

    from _minimax_h3_image_tiny import make_text_encoder, randomize

    out.mkdir(parents = True, exist_ok = True)
    tokenizer = make_tokenizer_and_processor(out)
    te = make_text_encoder(tokenizer, text_dim = TEXT_DIM, layers = TE_LAYERS, seed = seed + 3)
    te.to(torch.bfloat16).save_pretrained(str(out / "text_encoder"))

    torch.manual_seed(seed)
    vae = AutoencoderKLMiniMaxH3(**TINY_VAE).eval()
    randomize(vae, seed + 1, std = 0.05)
    vae.save_pretrained(str(out / "vae"))
    audio_vae = AutoencoderKLMiniMaxH3Audio(**TINY_AUDIO_VAE).eval()
    randomize(audio_vae, seed + 2, std = 0.05)
    audio_vae.save_pretrained(str(out / "audio_vae"))
    transformer = MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER).eval()
    randomize(transformer, seed + 4, std = 0.05)
    transformer.to(torch.bfloat16).save_pretrained(str(out / "transformer"))

    index = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_blocks_class_name": "MiniMaxH3Blocks",
        "text_encoder": _spec(
            "transformers", "Qwen3VLForConditionalGeneration", out, "text_encoder"
        ),
        "tokenizer": _spec("transformers", "Qwen2TokenizerFast", out, "tokenizer"),
        "processor": _spec("transformers", "Qwen3VLProcessor", out, "processor"),
        "vae": _spec("diffusers", "AutoencoderKLMiniMaxH3", out, "vae"),
        "audio_vae": _spec("diffusers", "AutoencoderKLMiniMaxH3Audio", out, "audio_vae"),
        "transformer": _spec("diffusers", "MiniMaxH3Transformer3DModel", out, "transformer"),
    }
    (out / "modular_model_index.json").write_text(json.dumps(index, indent = 2))
    return out


def write_clip(
    path: Path,
    *,
    seed: int,
    seconds: float = 1.2,
    width: int = 64,
    height: int = 64,
) -> Path:
    """A short mp4 with a moving pattern and a stereo tone, enough for one training window."""
    import av
    import numpy as np

    rng = np.random.default_rng(seed)
    fps, rate = 24, 32000
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate = fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = container.add_stream("aac", rate = rate)
        audio.layout = "stereo"
        base = rng.integers(0, 255, (height, width, 3), dtype = np.uint8)
        for i in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.roll(base, i * 2, axis = 1), format = "rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        t = np.arange(int(seconds * rate)) / rate
        freq = 220.0 + 40.0 * seed
        wave = np.stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * 1.5 * t)]) * 0.3
        wave = wave.astype(np.float32)
        step = 1024
        for start in range(0, wave.shape[1], step):
            chunk = np.ascontiguousarray(wave[:, start : start + step])
            frame = av.AudioFrame.from_ndarray(chunk, format = "fltp", layout = "stereo")
            frame.sample_rate = rate
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return path


def full_reference(root: Path) -> dict[str, Any]:
    """The untrimmed modules the stock load produces, for exactness checks."""
    import torch
    import transformers
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    te = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        str(root / "text_encoder"), torch_dtype = torch.bfloat16
    ).eval()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        str(root / "vae"), torch_dtype = torch.float32
    ).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        str(root / "audio_vae"), torch_dtype = torch.float32
    ).eval()
    return {"text_encoder": te, "vae": vae, "audio_vae": audio_vae}
