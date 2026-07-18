# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3 pipeline assembly for diffusers 0.39.

diffusers 0.39 ships every LTX-2.3 model class but its single-file loader maps every LTX-2
checkpoint to the 2.0 config, so 2.3 checkpoints fail a shape check at load. The community
transformer-only GGUFs also carry the DiT + connectors but NOT the text projections, VAEs, or
vocoder that 2.3 moved out of the transformer. This assembles the full 2.3 pipeline:

- transformer: from the checkpoint via ``from_single_file`` with the 2.3 config overrides and the
  ``prompt_adaln_single`` keys pre-renamed (the library converter doesn't know them).
- connectors: from the checkpoint's connector keys plus the ``text_embedding_projection`` tensors,
  fetched from the companion file in ``unsloth/LTX-2.3-GGUF`` when not bundled.
- video/audio VAE, vocoder: from the checkpoint when bundled, else the companion files.
- scheduler, text encoder (Gemma3), tokenizer: from the LTX-2.0 base repo, which 2.3 shares.

Every config and rename table mirrors diffusers' ``scripts/convert_ltx2_to_diffusers.py`` (the
authoritative 2.3 mapping the loader hasn't absorbed). Assembled through the constructor, not
``from_pretrained``, because the vocoder class differs from the base pin (``LTX2VocoderWithBWE`` vs
``LTX2Vocoder``) and the type gate would reject it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Companion files (text projections, VAEs incl. vocoder) next to the quants in unsloth's GGUF repo:
# the official Lightricks weights split out of the combined checkpoint. Keyed by variant.
LTX23_EXTRAS_REPO = "unsloth/LTX-2.3-GGUF"
_EXTRAS_TEXT_PROJ = "text_encoders/ltx-2.3-22b-{variant}_embeddings_connectors.safetensors"
_EXTRAS_VIDEO_VAE = "vae/ltx-2.3-22b-{variant}_video_vae.safetensors"
_EXTRAS_AUDIO_VAE = "vae/ltx-2.3-22b-{variant}_audio_vae.safetensors"

# ── configs + rename tables, verbatim from scripts/convert_ltx2_to_diffusers.py ──

# from_single_file config overrides on top of the base 2.0 transformer config.
LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES: dict[str, Any] = {
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

# Keys the 2.0-era converter doesn't know; renamed before from_single_file. Audio prefix first.
_TRANSFORMER_PRERENAME = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
)

_CONNECTOR_KEY_PREFIXES = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "text_embedding_projection",
    "connectors.",
    "video_connector",
    "audio_connector",
    "text_proj_in",
)

_CONNECTORS_RENAME = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.audio_aggregate_embed": "audio_text_proj_in",
    "text_embedding_projection.video_aggregate_embed": "video_text_proj_in",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

_CONNECTORS_CONFIG: dict[str, Any] = {
    "caption_channels": 3840,
    "text_proj_in_factor": 49,
    "video_connector_num_attention_heads": 32,
    "video_connector_attention_head_dim": 128,
    "video_connector_num_layers": 8,
    "video_connector_num_learnable_registers": 128,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 32,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_layers": 8,
    "audio_connector_num_learnable_registers": 128,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 4096,
    "audio_hidden_dim": 2048,
    "proj_bias": True,
}

_VIDEO_VAE_RENAME = {
    # Encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    # Decoder (2.3 adds up_blocks.7/8: a 4th decoder stage)
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    # Common
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

_VIDEO_VAE_REMOVE_SUFFIXES = (
    "per_channel_statistics.channel",
    "per_channel_statistics.mean-of-stds",
)

_VIDEO_VAE_CONFIG: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "block_out_channels": (256, 512, 1024, 1024),
    "down_block_types": (
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
    ),
    "decoder_block_out_channels": (256, 512, 512, 1024),
    "layers_per_block": (4, 6, 4, 2, 2),
    "decoder_layers_per_block": (4, 6, 4, 2, 2),
    "spatio_temporal_scaling": (True, True, True, True),
    "decoder_spatio_temporal_scaling": (True, True, True, True),
    "decoder_inject_noise": (False, False, False, False, False),
    "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    "upsample_residual": (False, False, False, False),
    "upsample_factor": (2, 2, 1, 2),
    "timestep_conditioning": False,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "encoder_causal": True,
    "decoder_causal": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "zeros",
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
}

_AUDIO_VAE_RENAME = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

# Same config as LTX-2.0 (upstream's comment); the weights are still 2.3-specific.
_AUDIO_VAE_CONFIG: dict[str, Any] = {
    "base_channels": 128,
    "output_channels": 2,
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": None,
    "in_channels": 2,
    "resolution": 256,
    "latent_channels": 8,
    "norm_type": "pixel",
    "causality_axis": "height",
    "dropout": 0.0,
    "mid_block_add_attention": False,
    "sample_rate": 16000,
    "mel_hop_length": 160,
    "is_causal": True,
    "mel_bins": 64,
    "double_z": True,
}

_VOCODER_RENAME = {
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
    "act_post": "act_out",
    "downsample.lowpass": "downsample",
}

_VOCODER_CONFIG: dict[str, Any] = {
    "in_channels": 128,
    "hidden_channels": 1536,
    "out_channels": 2,
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "upsample_factors": [5, 2, 2, 2, 2, 2],
    "resnet_kernel_sizes": [3, 7, 11],
    "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "act_fn": "snakebeta",
    "leaky_relu_negative_slope": 0.1,
    "antialias": True,
    "antialias_ratio": 2,
    "antialias_kernel_size": 12,
    "final_act_fn": None,
    "final_bias": False,
    "bwe_in_channels": 128,
    "bwe_hidden_channels": 512,
    "bwe_out_channels": 2,
    "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
    "bwe_upsample_factors": [6, 5, 2, 2, 2],
    "bwe_resnet_kernel_sizes": [3, 7, 11],
    "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "bwe_act_fn": "snakebeta",
    "bwe_leaky_relu_negative_slope": 0.1,
    "bwe_antialias": True,
    "bwe_antialias_ratio": 2,
    "bwe_antialias_kernel_size": 12,
    "bwe_final_act_fn": None,
    "bwe_final_bias": False,
    "filter_length": 512,
    "hop_length": 80,
    "window_length": 512,
    "num_mel_channels": 64,
    "input_sampling_rate": 16000,
    "output_sampling_rate": 48000,
}

_DIT_PREFIX = "model.diffusion_model."


# ── checkpoint inspection ────────────────────────────────────────────────────


def read_checkpoint_header(checkpoint_path: Path | str) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape from the checkpoint HEADER only (no weight data). GGUF shapes come back
    in GGML (reversed) order, so callers should membership-test, not assume a dimension position."""
    names_shapes: dict[str, tuple[int, ...]] = {}
    path = str(checkpoint_path)
    if path.lower().endswith(".gguf"):
        from gguf import GGUFReader
        for tensor in GGUFReader(path).tensors:
            names_shapes[str(tensor.name)] = tuple(int(x) for x in tensor.shape)
    else:
        from safetensors import safe_open
        with safe_open(path, framework = "pt") as handle:
            for name in handle.keys():
                names_shapes[name] = tuple(handle.get_slice(name).get_shape())
    return names_shapes


def is_ltx23_checkpoint(checkpoint_path: Path | str) -> bool:
    """True when the checkpoint carries the 9-row LTX-2.3 modulation tables (2.0 has 6-row
    per-block scale/shift tables; 2.3 widens them to 9). An unreadable header returns False so the
    caller falls back to the stock 2.0 path."""
    try:
        header = read_checkpoint_header(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("video.ltx2_header_probe_failed: %s", exc)
        return False
    for name, shape in header.items():
        if name.endswith("transformer_blocks.0.scale_shift_table"):
            return 9 in shape
    return False


# ── state-dict plumbing ──────────────────────────────────────────────────────


def _apply_rename(state: dict[str, Any], rename: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for old, new in rename.items():
            new_key = new_key.replace(old, new)
        out[new_key] = value
    return out


def _to_plain_dtype(state: dict[str, Any], torch_dtype: Any) -> dict[str, Any]:
    """Materialise every tensor as a plain torch tensor in torch_dtype. GGUF tensors arrive as
    block-quantized GGUFParameter; the small non-DiT components run dense, so dequantize here."""
    import torch

    try:
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    except Exception:  # noqa: BLE001 -- gguf support not installed; plain tensors only
        GGUFParameter, dequantize_gguf_tensor = (), None

    out: dict[str, Any] = {}
    for key, value in state.items():
        if dequantize_gguf_tensor is not None and isinstance(value, GGUFParameter):
            value = dequantize_gguf_tensor(value)
        out[key] = value.to(torch_dtype) if isinstance(value, torch.Tensor) else value
    return out


def _split_checkpoint(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition a combined LTX checkpoint into per-component state dicts. Handles both layouts: the
    official combined single file (``vae.*`` / ``audio_vae.*`` / ``vocoder.*`` / DiT + projections)
    and transformer-only GGUFs (bare DiT + connector keys)."""
    groups: dict[str, dict[str, Any]] = {
        "dit": {},
        "connectors": {},
        "vae": {},
        "audio_vae": {},
        "vocoder": {},
    }
    for key, value in state.items():
        bare = key[len(_DIT_PREFIX) :] if key.startswith(_DIT_PREFIX) else key
        if bare.startswith("vae."):
            groups["vae"][bare[len("vae.") :]] = value
        elif bare.startswith("audio_vae."):
            groups["audio_vae"][bare[len("audio_vae.") :]] = value
        elif bare.startswith("vocoder."):
            groups["vocoder"][bare[len("vocoder.") :]] = value
        elif bare.startswith(_CONNECTOR_KEY_PREFIXES):
            groups["connectors"][bare] = value
        else:
            groups["dit"][bare] = value
    return groups


def _load_extras_file(filename: str, hf_token: Optional[str]) -> dict[str, Any]:
    from safetensors.torch import load_file

    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    path = hf_hub_download_with_xet_fallback(LTX23_EXTRAS_REPO, filename, hf_token)
    return load_file(path)


def checkpoint_variant(checkpoint_path: Path | str) -> str:
    """Which companion-weight set a checkpoint pairs with ("dev"/"distilled"). The distilled-1.1
    refresh only retrained the DiT, so it shares the distilled companions."""
    return "dev" if "dev" in Path(checkpoint_path).name.lower() else "distilled"


# ── component builders ───────────────────────────────────────────────────────


def _build_from_config(
    model_cls: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    rename: dict[str, str],
    torch_dtype: Any,
    remove_suffixes: tuple[str, ...] = (),
) -> Any:
    from accelerate import init_empty_weights

    state = _apply_rename(_to_plain_dtype(state, torch_dtype), rename)
    for key in [k for k in state if k.endswith(remove_suffixes)] if remove_suffixes else []:
        state.pop(key)
    with init_empty_weights():
        model = model_cls.from_config(config)
    model.load_state_dict(state, strict = True, assign = True)
    return model.to(torch_dtype)


def load_ltx23_transformer(
    dit_state: dict[str, Any],
    *,
    base_repo: str,
    torch_dtype: Any,
    is_gguf: bool,
    hf_token: Optional[str],
) -> Any:
    import diffusers
    from diffusers import LTX2VideoTransformer3DModel

    # Pre-rename the 2.3-only keys the converter doesn't know, then from_single_file merges the
    # config overrides into the base 2.0 config and runs the stock key conversion.
    for old, new in _TRANSFORMER_PRERENAME:
        for key in [k for k in dit_state if k.startswith(old)]:
            dit_state[new + key[len(old) :]] = dit_state.pop(key)
    kwargs: dict[str, Any] = {
        "config": base_repo,
        "subfolder": "transformer",
        "torch_dtype": torch_dtype,
        "token": hf_token,
        **LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES,
    }
    if is_gguf:
        kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(compute_dtype = torch_dtype)
    return LTX2VideoTransformer3DModel.from_single_file(dit_state, **kwargs)


def load_ltx23_connectors(
    connector_state: dict[str, Any], *, variant: str, torch_dtype: Any, hf_token: Optional[str]
) -> Any:
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    # Transformer-only checkpoints carry the connector stacks but not the per-modality text
    # projections; fetch those from the companion file.
    if not any(k.startswith("text_embedding_projection") for k in connector_state):
        connector_state = dict(connector_state)
        connector_state.update(
            _load_extras_file(_EXTRAS_TEXT_PROJ.format(variant = variant), hf_token)
        )
    return _build_from_config(
        LTX2TextConnectors,
        _CONNECTORS_CONFIG,
        connector_state,
        _CONNECTORS_RENAME,
        torch_dtype,
    )


def load_ltx23_vae(
    vae_state: dict[str, Any], *, variant: str, torch_dtype: Any, hf_token: Optional[str]
) -> Any:
    from diffusers import AutoencoderKLLTX2Video
    if not vae_state:
        vae_state = _load_extras_file(_EXTRAS_VIDEO_VAE.format(variant = variant), hf_token)
    return _build_from_config(
        AutoencoderKLLTX2Video,
        _VIDEO_VAE_CONFIG,
        vae_state,
        _VIDEO_VAE_RENAME,
        torch_dtype,
        remove_suffixes = _VIDEO_VAE_REMOVE_SUFFIXES,
    )


def load_ltx23_audio_vae_and_vocoder(
    audio_vae_state: dict[str, Any],
    vocoder_state: dict[str, Any],
    *,
    variant: str,
    torch_dtype: Any,
    hf_token: Optional[str],
) -> tuple[Any, Any]:
    from diffusers import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    if not audio_vae_state or not vocoder_state:
        combined = _load_extras_file(_EXTRAS_AUDIO_VAE.format(variant = variant), hf_token)
        audio_vae_state = {
            k[len("audio_vae.") :]: v for k, v in combined.items() if k.startswith("audio_vae.")
        }
        vocoder_state = {
            k[len("vocoder.") :]: v for k, v in combined.items() if k.startswith("vocoder.")
        }
    audio_vae = _build_from_config(
        AutoencoderKLLTX2Audio,
        _AUDIO_VAE_CONFIG,
        audio_vae_state,
        _AUDIO_VAE_RENAME,
        torch_dtype,
    )
    # The 2.3 vocoder is a composite (base + bandwidth-extension stack + mel STFT buffers); keys
    # line up module-for-module after the renames.
    vocoder_state = _apply_rename(_to_plain_dtype(vocoder_state, torch_dtype), _VOCODER_RENAME)
    for key in [k for k in vocoder_state if ".ups." in k]:
        vocoder_state[key.replace(".ups.", ".upsamplers.")] = vocoder_state.pop(key)
    from accelerate import init_empty_weights

    with init_empty_weights():
        vocoder = LTX2VocoderWithBWE.from_config(_VOCODER_CONFIG)
    vocoder.load_state_dict(vocoder_state, strict = True, assign = True)
    return audio_vae, vocoder.to(torch_dtype)


# ── pipeline assembly ────────────────────────────────────────────────────────


def load_ltx23_pipeline(
    checkpoint_path: Path | str,
    *,
    base_repo: str,
    torch_dtype: Any,
    is_gguf: bool,
    hf_token: Optional[str] = None,
    transformer_override: Any = None,
) -> Any:
    """Full LTX-2.3 pipeline from a single-file/GGUF checkpoint. Assembled per-component
    (constructor, not from_pretrained) because the base model_index pins LTX2Vocoder while 2.3
    needs LTX2VocoderWithBWE, which the type gate would reject.

    ``transformer_override`` supplies a pre-built DiT (e.g. a pre-quantized checkpoint); the
    single file then contributes only the connectors / VAEs / vocoder groups and its DiT
    tensors are dropped unread."""
    import transformers
    from diffusers import LTX2Pipeline
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    variant = checkpoint_variant(checkpoint_path)
    logger.info(
        "video.ltx23_assembly: variant=%s gguf=%s extras=%s",
        variant,
        is_gguf,
        LTX23_EXTRAS_REPO,
    )
    state = load_single_file_checkpoint(str(checkpoint_path))
    groups = _split_checkpoint(state)
    del state

    # The Lightricks fp8 single files store SCALED float8 weights (.weight_scale/.input_scale
    # companions). Casting without the scales corrupts every quantized layer, so refuse loudly;
    # use the GGUF quants (Q8_0 for highest fidelity) instead.
    if any(k.endswith((".weight_scale", ".input_scale")) for k in groups["dit"]):
        raise ValueError(
            "This LTX checkpoint stores scaled fp8 weights, which this loader does "
            "not dequantize yet. Use the GGUF quants from unsloth/LTX-2.3-GGUF "
            "instead (Q8_0 for the highest fidelity) or the official bf16 checkpoint."
        )

    if transformer_override is not None:
        transformer = transformer_override
        groups.pop("dit", None)
    else:
        transformer = load_ltx23_transformer(
            groups["dit"],
            base_repo = base_repo,
            torch_dtype = torch_dtype,
            is_gguf = is_gguf,
            hf_token = hf_token,
        )
    connectors = load_ltx23_connectors(
        groups["connectors"],
        variant = variant,
        torch_dtype = torch_dtype,
        hf_token = hf_token,
    )
    vae = load_ltx23_vae(groups["vae"], variant = variant, torch_dtype = torch_dtype, hf_token = hf_token)
    audio_vae, vocoder = load_ltx23_audio_vae_and_vocoder(
        groups["audio_vae"],
        groups["vocoder"],
        variant = variant,
        torch_dtype = torch_dtype,
        hf_token = hf_token,
    )

    # Shared 2.0/2.3 components from the base repo, via model_index so upstream class renames break
    # loudly here rather than drift silently.
    index = LTX2Pipeline.load_config(base_repo, token = hf_token)

    def _sub(name: str, **extra: Any) -> Any:
        library, class_name = index[name]
        module = transformers if library == "transformers" else __import__("diffusers")
        return getattr(module, class_name).from_pretrained(
            base_repo, subfolder = name, token = hf_token, **extra
        )

    scheduler = _sub("scheduler")
    tokenizer = _sub("tokenizer")
    text_encoder = _sub("text_encoder", torch_dtype = torch_dtype)

    return LTX2Pipeline(
        scheduler = scheduler,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        connectors = connectors,
        transformer = transformer,
        vae = vae,
        audio_vae = audio_vae,
        vocoder = vocoder,
    )
