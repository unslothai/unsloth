# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The sampling recipe below (canvas rule, packed [text | references | image] layout, rotary grid, reference noising,
# velocity-form Euler step, presentation of edit references, decode) is a port of the MiniMax-H3-Image Modular
# Diffusers blocks of huggingface/diffusers PR 15230 (Apache-2.0, commit in vendor/diffusers_minimax_h3_image_manifest.json),
# and the native-checkpoint key mapping follows that PR's scripts/convert_minimax_h3_to_diffusers.py.

"""MiniMax-H3-Image (33B DiT + Qwen3-VL-32B conditioner) in Studio's image backend.

Runs on today's pinned diffusers: the transformer and VAE classes come from diffusers when it ships them, else from
the vendored PR copy. The released checkpoint is in MiniMax's native layout, so the DiT and VAE are converted while
they stream off the safetensors shards, never as a second full copy. The 6-step PDD grid is fixed, so every block's
AdaLN projection (13B of the 33B parameters) collapses to a lookup table over the 7 timesteps a render can ask for
(6 steps + the 0.999 reference level), computed from the dense weights at load: exact to the dense bf16 math and
20B parameters left to quantize."""

from __future__ import annotations

import functools
import gc
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

H3_IMAGE_FAMILY_NAME = "minimax-h3-image"
H3_IMAGE_BASE_REPO = "MiniMaxAI/MiniMax-H3-Image"
# The image release conditions on MiniMax-H3's text encoder without carrying it (model_index.json names this repo).
H3_IMAGE_TEXT_ENCODER_REPO = "MiniMaxAI/MiniMax-H3"
H3_IMAGE_PIPELINE_CLASS = "MiniMaxH3ImagePipeline"
H3_IMAGE_TRANSFORMER_CLASS = "MiniMaxH3ImageTransformer2DModel"
H3_IMAGE_NATIVE_TRANSFORMER_CLASS = "MiniMaxH3DiTModel"
H3_IMAGE_VAE_CLASS = "AutoencoderKLMiniMaxH3Image"
H3_IMAGE_NATIVE_VAE_CLASS = "MiniMaxH3ImageVAE"

# Model card / pdd_config.json: distilled to 6 steps with one output head each, no CFG, no negative prompt.
H3_IMAGE_STEPS = 6
H3_IMAGE_GUIDANCE = 1.0
H3_IMAGE_SIGMAS: tuple[float, ...] = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
# Smaller requests are generated on a canvas of this area at the same aspect and resized back (LANCZOS).
H3_IMAGE_MIN_GENERATION_AREA = 2048 * 2048
H3_IMAGE_MIN_GENERATION_AREA_RANGE = (1024, 4096 * 4096)
H3_IMAGE_MAX_SEQUENCE_LENGTH = 4096
H3_IMAGE_TEXT_ENCODER_LAYER = 50
H3_IMAGE_MAX_REFERENCES = 16
H3_IMAGE_CANVAS_MULTIPLE = 32
H3_IMAGE_VAE_SCALE = 16
H3_IMAGE_PIXEL_MEAN = (0.485, 0.456, 0.406)
H3_IMAGE_PIXEL_STD = (0.229, 0.224, 0.225)
H3_IMAGE_REFERENCE_NOISE_AUG = 0.999
H3_IMAGE_REFERENCE_SHORT_EDGE = (1024, 2048)
H3_IMAGE_VISION_SHORT_EDGE = (1024, 1024)
# Per-row modality tags index the AdaLN table: a checkpoint contract.
H3_IMAGE_IMAGE_TAG = 0
H3_IMAGE_TEXT_TAG = 1
H3_IMAGE_MODALITY_NUM = 3
# Model card "Sizes": the recommended ~16 MP canvases (width, height); 4 MP and up is the working range.
H3_IMAGE_16MP_SIZES: tuple[tuple[int, int], ...] = (
    (4096, 4096),
    (4736, 3552),
    (3552, 4736),
    (5024, 3360),
    (3360, 5024),
    (5472, 3072),
    (3072, 5472),
    (6272, 2688),
)
H3_IMAGE_PRESETS: tuple[tuple[int, int], ...] = (
    (2048, 2048),
    (2368, 1760),
    (1760, 2368),
    (2496, 1664),
    (1664, 2496),
    (2720, 1536),
    (1536, 2720),
) + H3_IMAGE_16MP_SIZES

H3_IMAGE_DIT_BF16_GB = 66.3
H3_IMAGE_DIT_TABLE_BF16_GB = 40.3
H3_IMAGE_DIT_TABLE_INT8_GB = 21.9
H3_IMAGE_VAE_GB = 2.34
H3_IMAGE_TE_BF16_GB = 66.7
H3_IMAGE_TE_INT8_GB = 27.2

PDD_HEADS_FILE = "pdd_fused_heads.safetensors"
PDD_CONFIG_FILE = "pdd_config.json"
DIFFUSERS_TRANSFORMER_SUBFOLDER = "transformer_diffusers"
DIFFUSERS_VAE_SUBFOLDER = "vae_diffusers"

# "0" keeps the dense AdaLN projections (diagnostic: the table is the default and exact to the dense math).
ADALN_TABLE_ENV = "UNSLOTH_H3_IMAGE_ADALN_TABLE"
# Force a placement for tests and A/B runs: resident | sequential | stream.
PLACEMENT_ENV = "UNSLOTH_H3_IMAGE_PLACEMENT"
VRAM_BUDGET_ENV = "UNSLOTH_H3_IMAGE_VRAM_GB"
HOST_BUDGET_ENV = "UNSLOTH_H3_IMAGE_HOST_GB"
# "0": stream from pageable memory (ModuleListStreamer) instead of the pinned prefetch streamer
PINNED_STREAM_ENV = "UNSLOTH_H3_IMAGE_PINNED_STREAM"

QUANT_SCHEMES = ("int8", "fp8")
# ConvRot group: the one the H3 video denoiser and its hosted encoder use (divides 5376, 7168 and 14336).
CONVROT_GROUP = 256
# The block linears the on-the-fly quantizer converts. The text-only token refiner and context embedder run at
# M = prompt tokens (can be under the int8 GEMM floor) and stay bf16 (0.8B parameters).
QUANT_LINEAR_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _env_off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "diffusers_minimax_h3_image"
_VENDOR_PACKAGE = "unsloth_studio_vendor_minimax_h3_image"


def _load_vendored(module: str) -> Any:
    """The vendored module by file path, registered under a private name (never on sys.path)."""
    name = f"{_VENDOR_PACKAGE}.{module}"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    path = _VENDOR_DIR / f"{module}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"vendored MiniMax-H3-Image source missing: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _diffusers_class(name: str) -> Any:
    try:
        import diffusers
    except Exception:  # noqa: BLE001 - no diffusers at all: the caller reports it
        return None
    try:
        cls = getattr(diffusers, name, None)
    except Exception:  # noqa: BLE001 - lazy-import failure inside diffusers
        return None
    return (
        cls if cls is not None and hasattr(cls, "from_config") and hasattr(cls, "forward") else None
    )


@functools.lru_cache(maxsize = None)
def transformer_class() -> Any:
    """``MiniMaxH3ImageTransformer2DModel``: diffusers' when installed, else the vendored copy."""
    return _diffusers_class(H3_IMAGE_TRANSFORMER_CLASS) or getattr(
        _load_vendored("transformer_minimax_h3_image"), H3_IMAGE_TRANSFORMER_CLASS
    )


@functools.lru_cache(maxsize = None)
def vae_class() -> Any:
    """``AutoencoderKLMiniMaxH3Image``: diffusers' when installed, else the vendored copy."""
    return _diffusers_class(H3_IMAGE_VAE_CLASS) or getattr(
        _load_vendored("autoencoder_kl_minimax_h3_image"), H3_IMAGE_VAE_CLASS
    )


def classes_source() -> str:
    """``"diffusers"`` when the installed diffusers ships both classes, else ``"vendored"``."""
    both = _diffusers_class(H3_IMAGE_TRANSFORMER_CLASS) and _diffusers_class(H3_IMAGE_VAE_CLASS)
    return "diffusers" if both else "vendored"


def in_tree_class(name: str) -> Any:
    """Resolve a class name the family table or a model_index names to Studio's in-tree class."""
    if name in (H3_IMAGE_PIPELINE_CLASS,):
        return MiniMaxH3ImagePipeline
    if name in (H3_IMAGE_TRANSFORMER_CLASS, H3_IMAGE_NATIVE_TRANSFORMER_CLASS):
        return transformer_class()
    if name in (H3_IMAGE_VAE_CLASS, H3_IMAGE_NATIVE_VAE_CLASS):
        return vae_class()
    if name == "MiniMaxH3ImageScheduler":
        return MiniMaxH3ImageScheduler
    raise KeyError(name)


def transformer_config_from_native(native: dict, num_pdd_heads: int = H3_IMAGE_STEPS) -> dict:
    """``MiniMaxH3ImageTransformer2DModel`` kwargs from the release's ``transformer/config.json``."""
    if native.get("audio_enabled"):
        raise ValueError(
            "MiniMax-H3-Image transformer config has audio_enabled; this is the video DiT."
        )
    patch = native.get("patch_size", [1, 2, 2])
    return {
        "num_attention_heads": int(native.get("num_attention_heads", 56)),
        "attention_head_dim": int(native.get("attention_head_dim", 128)),
        "hidden_size": int(native.get("hidden_size", 5376)),
        "num_layers": int(native.get("num_layers", 50)),
        "num_refiner_layers": int(native.get("token_refiner_num_layers", 2)),
        "ffn_dim": int(native.get("ffn_hidden_size", 14336)),
        "in_channels": int(native.get("latents_dim", 24)),
        "patch_size": tuple(int(p) for p in patch),
        "text_dim": int(native.get("text_dim", 5120)),
        "freq_dim": int(native.get("timestep_input_dim", 256)),
        "time_embed_hidden_dim": int(native.get("time_embed_hidden_size", 5376)),
        "time_embed_dim": int(native.get("time_embed_dim", 2688)),
        "rope_freq_dim": int(native.get("rope_inv_freq_len", 16)),
        "rope_theta": float(native.get("rope_theta", 10000.0)),
        "norm_eps": float(native.get("norm_eps", 1e-5)),
        "qk_norm_eps": float(native.get("qk_norm_eps", 1e-5)),
        "final_norm_eps": float(native.get("final_norm_eps", 1e-5)),
        "num_pdd_heads": int(num_pdd_heads),
    }


def vae_config_from_native(native: dict) -> dict:
    """``AutoencoderKLMiniMaxH3Image`` kwargs from the release's ``vae/config.json`` (converter checks included)."""
    if native.get("vit_decoder_2d") is not True:
        raise ValueError(
            "MiniMax-H3-Image VAE config is not the 2D ViT decoder (vit_decoder_2d is not true)."
        )
    kwargs = dict(native.get("vit_decoder_kwargs") or {})
    expected = {
        "vae_encoder_tiling": 1,
        "vae_decoder_tiling": 1,
        "vae_tile_size": 256,
        "vae_tile_overlap_min": 64,
    }
    bad = {k: native.get(k) for k, v in expected.items() if native.get(k, v) != v}
    if bad:
        raise ValueError(f"MiniMax-H3-Image VAE tile geometry differs from the released one: {bad}")
    geometry = {
        k: list(native[k])
        for k in ("block_out_channels", "spatial_downsample_factors", "temporal_downsample_factors")
        if k in native
    }
    if "layers_per_block" in native:
        geometry["layers_per_block"] = int(native["layers_per_block"])
    return {
        **geometry,
        "latent_channels": int(native.get("latent_channels", 24)),
        "decoder_num_layers": int(kwargs.get("num_layers", 24)),
        "decoder_num_attention_heads": int(kwargs.get("heads", 16)),
        "decoder_attention_head_dim": int(kwargs.get("dim_head", 64)),
        "decoder_rope_theta": float(kwargs.get("rope_theta", 100.0)),
        "decoder_rope_dim_ratio": float(kwargs.get("rope_dim_ratio", 0.5)),
        "latents_mean": tuple(float(v) for v in native["latents_mean"]),
        "latents_std": tuple(float(v) for v in native["latents_std"]),
    }


def reorder_interleaved_qkv(weight: Any, heads: int, head_dim: int) -> Any:
    """Raw shards store fused QKV per head ``[h0: q k v, h1: q k v, ...]``; returns ``[q_all; k_all; v_all]``."""
    import torch

    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected {heads} heads * 3 * {head_dim}"
        )
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim = 1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim = 0)


_TRANSFORMER_RENAMES = (
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
    ("video_patch_proj.", "proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".mlp.fc2.", ".ff.net.2."),
)
_TRANSFORMER_DROPPED = ("rope.inv_freq",)


def native_transformer_key(key: str) -> Optional[str]:
    """The diffusers name of one native DiT key (before the qkv / fc1 tensor transforms), None when dropped."""
    if key in _TRANSFORMER_DROPPED or key.startswith("final_layer.video_out."):
        # The per-step PDD heads replace the release's own output layer.
        return None
    if key.startswith("audio_") or key.startswith("final_layer.audio_out."):
        return None
    name = key
    if name.startswith("token_refiner.blocks."):
        name = "token_refiner.refiner_blocks." + name[len("token_refiner.blocks.") :]
    elif name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    for old, new in _TRANSFORMER_RENAMES:
        name = name.replace(old, new)
    return name


def convert_native_transformer_tensor(key: str, tensor: Any, heads: int, head_dim: int) -> list:
    """``[(diffusers key, tensor), ...]`` for one native DiT key/tensor (qkv split, SwiGLU half swap)."""
    import torch

    name = native_transformer_key(key)
    if name is None:
        return []
    if name.endswith(".attn.qkv_proj.weight"):
        fused = reorder_interleaved_qkv(tensor, heads, head_dim)
        q, k, v = fused.split(heads * head_dim, dim = 0)
        prefix = name[: -len("qkv_proj.weight")]
        return [
            (f"{prefix}to_q.weight", q.contiguous()),
            (f"{prefix}to_k.weight", k.contiguous()),
            (f"{prefix}to_v.weight", v.contiguous()),
        ]
    if name.endswith(".mlp.fc1.weight"):
        # Reference fc2(silu(gate) * value) from [gate; value]; diffusers SwiGLU reads [value; gate].
        gate, value = tensor.chunk(2, dim = 0)
        return [
            (
                name.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"),
                torch.cat([value, gate]).contiguous(),
            )
        ]
    return [(name, tensor)]


def split_pdd_heads(heads: dict, num_steps: int) -> dict:
    """``video_out.{weight,bias}`` stacked ``(steps, ...)`` -> ``pdd_proj_out.{i}.{weight,bias}``."""
    if set(heads) != {"video_out.weight", "video_out.bias"}:
        raise KeyError(
            f"{PDD_HEADS_FILE} must hold exactly video_out.weight and video_out.bias, got {sorted(heads)}"
        )
    out = {}
    for key, tensor in heads.items():
        if tensor.shape[0] != num_steps:
            raise ValueError(
                f"{PDD_HEADS_FILE} {key} has {tensor.shape[0]} heads, pdd_config says {num_steps}"
            )
        suffix = key[len("video_out.") :]
        for step in range(num_steps):
            out[f"pdd_proj_out.{step}.{suffix}"] = tensor[step].contiguous()
    return out


def native_vae_key(key: str) -> Optional[str]:
    """Rename one native image-VAE key onto the diffusers module path (no tensor transform); None when dropped."""
    if key == "decoder.mask_token":
        return None
    name = key
    if name.startswith("encoder.down."):
        level, rest = name[len("encoder.down.") :].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        name = f"encoder.down_blocks.{level}.{rest}"
    name = name.replace("decoder.x_embedder.", "decoder.proj_in.")
    name = name.replace(".attn.to_out.", ".attn.to_out.0.")
    name = name.replace(".ff.w1.", ".ff.net.0.proj.")
    name = name.replace(".ff.w2.", ".ff.net.2.")
    return name


def convert_native_vae_tensor(key: str, tensor: Any, heads: int, head_dim: int) -> list:
    import torch

    if key == "decoder.mask_token":
        return []
    if ".attn.to_qkv." in key:
        fused = reorder_interleaved_qkv(tensor, heads, head_dim)
        q, k, v = fused.split(heads * head_dim, dim = 0)
        prefix, suffix = key.split(".attn.to_qkv.")
        return [
            (f"{prefix}.attn.to_q.{suffix}", q.contiguous()),
            (f"{prefix}.attn.to_k.{suffix}", k.contiguous()),
            (f"{prefix}.attn.to_v.{suffix}", v.contiguous()),
        ]
    name = native_vae_key(key)
    if ".ff.w1." in key:
        gate, up = tensor.chunk(2, dim = 0)
        return [(name, torch.cat([up, gate]).contiguous())]
    return [(name, tensor)]


def grid_timesteps(sigmas: Sequence[float] = H3_IMAGE_SIGMAS) -> list[float]:
    """Every timestep a render can ask the DiT for: the float32 ``1 - sigma`` of each step, then the reference
    level. Float32 values, as the pipeline builds them."""
    import torch

    steps = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype = torch.float32).tolist()
    ref = float(torch.tensor(H3_IMAGE_REFERENCE_NOISE_AUG, dtype = torch.float32))
    out = []
    for t in steps + [ref]:
        if t not in out:
            out.append(t)
    return out


def adaln_table_enabled() -> bool:
    return not _env_off(ADALN_TABLE_ENV)


def _grid_embedder_class() -> Any:
    import torch
    from torch import nn

    class H3ImageGridTimestepEmbedder(nn.Module):
        """Maps each timestep to a one-hot row over the fixed grid; the table Linears then select exact rows."""

        def __init__(self, grid: Sequence[float]) -> None:
            super().__init__()
            self.register_buffer(
                "grid", torch.tensor(list(grid), dtype = torch.float32), persistent = False
            )

        def forward(self, timestep: Any) -> Any:
            t = timestep.to(torch.float32).reshape(-1, 1)
            grid = self.grid.to(t.device)
            dist = (t - grid[None, :]).abs()
            best, index = dist.min(dim = 1)
            # A timestep off the grid has no table row: refuse rather than return a neighbour's modulation.
            if not torch.compiler.is_compiling() and bool((best > 1e-6).any()):
                raise ValueError(
                    "MiniMax-H3-Image samples on its fixed 6-step grid (the AdaLN table holds only those timesteps); "
                    f"got timesteps {timestep.tolist()}"
                )
            return torch.nn.functional.one_hot(index, grid.numel()).to(torch.float32)

    return H3ImageGridTimestepEmbedder


def _raw_timestep_module() -> Any:
    from torch import nn
    class H3ImageRawTimestep(nn.Module):
        def forward(self, timestep: Any) -> Any:
            return timestep

    return H3ImageRawTimestep()


def _table_modulation_forward(self: Any, temb: Any) -> tuple:
    """One-hot rows in, the dense projection's rows out: no SiLU (already folded into the table)."""
    out = self.linear(temb.to(self.linear.weight.dtype))
    out = out.view(-1, 6 * self.hidden_size)
    return out.chunk(6, dim = -1)


def _table_norm_out_forward(self: Any, hidden_states: Any, temb: Any, timestep_indices: Any) -> Any:
    shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim = -1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
        0, timestep_indices
    )


def dense_adaln_rows(
    temb: Any,
    weight: Any,
    bias: Any,
    dtype: Any,
    device: Any = None,
) -> Any:
    """``linear(silu(temb).to(dtype))`` as the dense module computes it, rows per grid timestep; computed on
    ``device`` (the card the render runs on) and returned on the CPU."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device) if device is not None else weight.device
    x = F.silu(temb.to(dev, torch.float32)).to(dtype)
    rows = F.linear(x, weight.to(dev, dtype), None if bias is None else bias.to(dev, dtype))
    return rows.to("cpu")


def table_linear(rows: Any) -> Any:
    """A ``Linear(K -> out)`` whose one-hot input selects ``rows[k]`` exactly (weight = rows.T, no bias)."""
    import torch
    from torch import nn

    k, out_features = rows.shape
    linear = nn.Linear(k, out_features, bias = False, device = "meta")
    linear.weight = nn.Parameter(rows.t().contiguous(), requires_grad = False)
    return linear


def install_grid_time_embedding(transformer: Any, grid: Sequence[float]) -> None:
    """Swap the timestep sinusoid + MLP for the one-hot grid embedder (the AdaLN Linears must already be tables)."""
    transformer.time_proj = _raw_timestep_module()
    transformer.time_embedder = _grid_embedder_class()(grid)
    transformer._unsloth_adaln_table = tuple(float(t) for t in grid)


def convert_block_adaln_to_table(block: Any, rows: Any) -> None:
    block.adaln_proj.linear = table_linear(rows)
    block.adaln_proj.forward = types.MethodType(_table_modulation_forward, block.adaln_proj)


def convert_norm_out_to_table(norm_out: Any, rows: Any) -> None:
    norm_out.linear = table_linear(rows)
    norm_out.forward = types.MethodType(_table_norm_out_forward, norm_out)


def grid_temb(transformer_dense_time: Any, grid: Sequence[float]) -> Any:
    """``time_embedder(time_proj(t))`` for every grid timestep, float32, on CPU."""
    import torch

    time_proj, time_embedder = transformer_dense_time
    t = torch.tensor(list(grid), dtype = torch.float32)
    with torch.no_grad():
        return time_embedder(time_proj(t).to(torch.float32)).to(torch.float32)


def convert_dense_transformer_to_table(
    transformer: Any, grid: Optional[Sequence[float]] = None
) -> int:
    """In place on an already-loaded dense model (tests, small builds): every AdaLN projection -> its grid table."""
    import torch

    grid = list(grid or grid_timesteps())
    temb = grid_temb((transformer.time_proj, transformer.time_embedder), grid)
    converted = 0
    with torch.no_grad():
        for block in transformer.transformer_blocks:
            lin = block.adaln_proj.linear
            rows = dense_adaln_rows(temb, lin.weight, lin.bias, lin.weight.dtype)
            convert_block_adaln_to_table(block, rows)
            converted += 1
        lin = transformer.norm_out.linear
        convert_norm_out_to_table(
            transformer.norm_out, dense_adaln_rows(temb, lin.weight, lin.bias, lin.weight.dtype)
        )
    install_grid_time_embedding(transformer, grid)
    return converted


def canvas_size(
    height: int,
    width: int,
    min_generation_area: int = H3_IMAGE_MIN_GENERATION_AREA,
) -> tuple[int, int]:
    """The ``(height, width)`` canvas a ``height x width`` request is generated on: scaled up to
    ``min_generation_area`` at the same aspect when smaller, both sides rounded to multiples of 32."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    lo, hi = H3_IMAGE_MIN_GENERATION_AREA_RANGE
    if not lo <= int(min_generation_area) <= hi:
        raise ValueError(
            f"min_generation_area must be within [{lo}, {hi}], got {min_generation_area}"
        )
    m = H3_IMAGE_CANVAS_MULTIPLE
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (max(m, round(height * scale / m) * m), max(m, round(width * scale / m) * m))


def bound_short_edge(
    width: int,
    height: int,
    lo: int,
    hi: int,
    multiple: int = H3_IMAGE_CANVAS_MULTIPLE,
) -> tuple:
    """Reference resize rule: keep a short edge in ``[lo, hi]``, else scale to the nearer bound (multiples of 32)."""
    short = min(width, height)
    if lo <= short <= hi:
        return width, height
    scale = (lo if short < lo else hi) / short
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def patchify(latents: Any, patch: tuple) -> Any:
    """``(B, C, T, H, W)`` -> rows ``(B * patches, C * prod(patch))``, frame-major then row-major."""
    pt, ph, pw = patch
    b, c, t, h, w = latents.shape
    if t % pt or h % ph or w % pw:
        raise ValueError(f"latents {tuple(latents.shape)} not divisible by patch {patch}")
    x = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw).contiguous()


def unpatchify(
    rows: Any, latent_height: int, latent_width: int, channels: int, patch: tuple
) -> Any:
    _, ph, pw = patch
    x = rows.reshape(-1, latent_height // ph, latent_width // pw, channels, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(-1, channels, latent_height, latent_width).contiguous()


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> Any:
    import numpy as np
    import torch

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint = False) * 32
    return torch.from_numpy(grid).to(torch.float64)


def _frame_grid(latent_height: int, latent_width: int, ph: int, pw: int) -> Any:
    import numpy as np
    import torch

    sqrt_area = np.sqrt(latent_height * latent_width)
    hg = _spatial_grid(latent_height, ph, sqrt_area)
    wg = _spatial_grid(latent_width, pw, sqrt_area)
    grids = torch.meshgrid(hg, wg, indexing = "ij")
    return torch.stack([g.reshape(-1) for g in grids], dim = -1)


def build_packed_layout(
    text_token_tags: Any,
    reference_grids: Sequence[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch: tuple = (1, 2, 2),
) -> tuple:
    """``[text | references | generated image]``: (position_ids f64 (S,3), token_tags, image_indices, text_indices,
    number of reference rows)."""
    import torch

    _, ph, pw = patch
    n_text = int(text_token_tags.shape[0])
    n_ref = sum((h // ph) * (w // pw) for h, w in reference_grids)
    n_target = (latent_height // ph) * (latent_width // pw)
    seq = n_text + n_ref + n_target
    pos = torch.zeros(seq, 3, dtype = torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype = torch.float64)
    cursor = n_text
    clock = float(n_text)
    for h, w in reference_grids:
        rows = slice(cursor, cursor + (h // ph) * (w // pw))
        cursor = rows.stop
        pos[rows, 0] = clock
        pos[rows, 1:] = _frame_grid(h, w, ph, pw)
        clock += 1.0
    pos[cursor:, 0] = clock
    pos[cursor:, 1:] = _frame_grid(latent_height, latent_width, ph, pw)
    text_indices = torch.arange(n_text)
    image_indices = torch.arange(n_text, seq)
    tags = torch.empty(seq, dtype = torch.long)
    tags[text_indices] = text_token_tags.to(torch.long)
    tags[image_indices] = H3_IMAGE_IMAGE_TAG
    return pos, tags, image_indices, text_indices, n_ref


def row_timestep_plan(timesteps: Any, seq_len: int, reference_rows: Any) -> list:
    """Per step: (distinct timesteps, per-row index); reference rows stay at 0.999 for every step."""
    import torch

    plan = []
    for t in timesteps:
        row = torch.full((seq_len,), float(t), dtype = torch.float32)
        if reference_rows is not None and reference_rows.numel():
            row[reference_rows] = H3_IMAGE_REFERENCE_NOISE_AUG
        unique, index = torch.unique(row, sorted = True, return_inverse = True)
        plan.append((unique, index))
    return plan


def _close_diffusers_import_window() -> None:
    """The download plan and load preflight import this module before any load closes the window."""
    try:
        from utils.torch_warmup import close_dynamo_import_window
        close_dynamo_import_window(logger)
    except Exception:  # noqa: BLE001 - ordering only; the import still runs
        pass


def _scheduler_bases() -> tuple:
    _close_diffusers_import_window()
    try:
        from diffusers.configuration_utils import ConfigMixin, register_to_config
        from diffusers.schedulers.scheduling_utils import SchedulerMixin
        return SchedulerMixin, ConfigMixin, register_to_config
    except Exception:  # noqa: BLE001 - diffusers missing: distinct empty bases (object twice is a TypeError)

        class _NoSchedulerMixin:
            pass

        class _NoConfigMixin:
            pass

        return _NoSchedulerMixin, _NoConfigMixin, (lambda f: f)


_SchedulerMixin, _ConfigMixin, _register_to_config = _scheduler_bases()


class MiniMaxH3ImageScheduler(_SchedulerMixin, _ConfigMixin):
    """The release's sampler: explicit sigma grid taken verbatim, ``t = 1 - sigma``, Euler in velocity form
    ``x_next = x + (sigma - sigma_next) * v`` in float32 (MiniMaxH3Scheduler with ``euler_update_form="velocity"``,
    ``shift_provided_sigmas=False``)."""

    config_name = "scheduler_config.json"
    order = 1

    @_register_to_config
    def __init__(self, sample_sigmas: Optional[list] = None) -> None:
        self.sigmas = None
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = None

    def set_timesteps(
        self,
        sigmas: Optional[Sequence[float]] = None,
        device: Any = None,
        **_: Any,
    ) -> None:
        import torch

        base = list(
            sigmas if sigmas is not None else (self.config.sample_sigmas or H3_IMAGE_SIGMAS)
        )
        # The released sampler rounds every sigma through its float32 timestep 1 - sigma and back.
        timesteps = torch.tensor([1.0 - s for s in base[:-1]], dtype = torch.float32)
        sig = torch.cat([1.0 - timesteps, torch.zeros(1)])
        if sig.numel() < 2 or not bool((sig[1:] < sig[:-1]).all()):
            raise ValueError("sigmas must be strictly decreasing and end at 0.0")
        self.sigmas = sig.to(device)
        self.timesteps = (1.0 - sig[:-1]).to(device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = 0

    def step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        return_dict: bool = False,
        **_: Any,
    ) -> tuple:
        import torch

        i = self._step_index
        sigma = self.sigmas[i].to(device = sample.device, dtype = torch.float32)
        sigma_next = self.sigmas[i + 1].to(device = sample.device, dtype = torch.float32)
        prev = sample.to(torch.float32) + (sigma - sigma_next) * model_output.to(torch.float32)
        self._step_index = i + 1
        return (prev.to(sample.dtype),)

    @staticmethod
    def scale_noise(sample: Any, timestep: Any, noise: Any) -> Any:
        """``x_t = t * x0 + (1 - t) * noise``; a tensor ``t`` mixes in float32 as the release does."""
        import torch

        if not isinstance(timestep, torch.Tensor):
            return timestep * sample + (1.0 - timestep) * noise
        t = timestep.to(device = sample.device, dtype = sample.dtype)
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        return t * sample + (1.0 - t) * noise


PLACE_RESIDENT = "resident"
PLACE_DIT_RESIDENT = "dit_resident"
PLACE_SEQUENTIAL = "sequential"
PLACE_STREAM = "stream"
PLACEMENTS = (PLACE_RESIDENT, PLACE_DIT_RESIDENT, PLACE_SEQUENTIAL, PLACE_STREAM)


@dataclass
class H3ImagePlan:
    """What the memory planner decided; recorded on the pipe and in the load's resolved controls."""

    placement: str
    dit_gb: float
    te_gb: float
    vae_gb: float
    vram_gb: Optional[float]
    host_gb: Optional[float]
    dit_resident_blocks: Optional[int] = None
    te_resident_layers: Optional[int] = None
    # False: the encoder is loaded for each encode and freed after (host RAM cannot hold it beside the DiT)
    te_kept_on_host: bool = True
    reasons: list = field(default_factory = list)
    # host GB held once placed (host DiT blocks + encoder + 4 GB); None without a host budget
    host_need_gb: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "placement": self.placement,
            "dit_gb": round(self.dit_gb, 2),
            "te_gb": round(self.te_gb, 2),
            "vae_gb": round(self.vae_gb, 2),
            "vram_gb": None if self.vram_gb is None else round(self.vram_gb, 2),
            "host_gb": None if self.host_gb is None else round(self.host_gb, 2),
            "dit_resident_blocks": self.dit_resident_blocks,
            "te_resident_layers": self.te_resident_layers,
            "te_kept_on_host": self.te_kept_on_host,
            "reasons": list(self.reasons),
        }


def pinned_stream_enabled() -> bool:
    return not _env_off(PINNED_STREAM_ENV)


def stream_inflight() -> int:
    if not pinned_stream_enabled():
        return 1
    from .diffusion_offload_prefetch import LIST_STREAM_DEPTH
    return 1 + LIST_STREAM_DEPTH


def activation_gb(
    width: int,
    height: int,
    references: int = 0,
) -> float:
    """Peak denoise + decode working set at a canvas (bf16 DiT, fp32 tiled VAE); generous by design."""
    mp = max(width * height, H3_IMAGE_MIN_GENERATION_AREA) / 1e6
    return 1.5 + 0.35 * mp * (1 + references)


def dit_gb_for(scheme: Optional[str], table: bool = True) -> float:
    dense_adaln_gb = 0.0 if table else H3_IMAGE_DIT_BF16_GB - H3_IMAGE_DIT_TABLE_BF16_GB
    if scheme in QUANT_SCHEMES:
        # The AdaLN projections are never quantized: without the table they stay dense bf16.
        return H3_IMAGE_DIT_TABLE_INT8_GB + dense_adaln_gb
    return H3_IMAGE_DIT_TABLE_BF16_GB + dense_adaln_gb


def te_gb_for(scheme: Optional[str]) -> float:
    return H3_IMAGE_TE_INT8_GB if scheme == "int8" else H3_IMAGE_TE_BF16_GB


# Device-side pieces that are not in the streamed lists: the DiT's refiner, embedders and heads (bf16 refiner,
# 0.8B), the encoder's token embedding + vision tower (1.56 + ~1.2 GB).
DIT_TOP_GB = 2.0
TE_TOP_GB = 3.0


def plan_placement(
    *,
    vram_gb: Optional[float],
    host_gb: Optional[float],
    dit_gb: float,
    te_gb: float,
    vae_gb: float = H3_IMAGE_VAE_GB,
    width: int = 2048,
    height: int = 2048,
    references: int = 0,
    num_blocks: int = 50,
    te_layers: int = 50,
    forced: Optional[str] = None,
    inflight: int = 1,
) -> H3ImagePlan:
    """Pure: VRAM and host budgets (GB, None = unknown/CPU) -> placement, best first:

    - resident: DiT + encoder + VAE + working set fit;
    - dit_resident: the DiT and VAE stay, the encoder's top stays and as many of its layers as fit, the rest stream
      through once per prompt (A100 40 GB with both at int8);
    - sequential: one big component on the card at a time (a DiT larger than the encoder room);
    - stream: DiT blocks stream past the ones that fit, the encoder streams every layer (L4 24 GB).

    Host: the encoder is kept only when it fits beside the DiT, else it loads per prompt and is freed after.
    ``inflight``: streamed members on the device at once (2 with the prefetch streamer: one runs, one copies)."""
    reasons = []
    act = activation_gb(width, height, references)
    margin = 1.5
    plan = H3ImagePlan(
        placement = PLACE_RESIDENT,
        dit_gb = dit_gb,
        te_gb = te_gb,
        vae_gb = vae_gb,
        vram_gb = vram_gb,
        host_gb = host_gb,
    )
    if vram_gb is None:
        plan.placement = PLACE_RESIDENT
        reasons.append("no accelerator budget: everything on the compute device")
        plan.reasons = reasons
        return plan
    layer_gb = max(te_gb - TE_TOP_GB, 0.1) / max(1, te_layers)
    block_gb = max(dit_gb - DIT_TOP_GB, 0.1) / max(1, num_blocks)
    inflight = max(1, int(inflight))
    if forced in PLACEMENTS:
        choice = forced
        reasons.append(f"placement forced to {forced} by {PLACEMENT_ENV}")
    elif dit_gb + te_gb + vae_gb + act + margin <= vram_gb:
        choice = PLACE_RESIDENT
        reasons.append(
            f"DiT + encoder + VAE ({dit_gb + te_gb + vae_gb:.1f} GB) fit {vram_gb:.1f} GB"
        )
    elif dit_gb + vae_gb + act + margin + TE_TOP_GB + layer_gb * inflight <= vram_gb:
        choice = PLACE_DIT_RESIDENT
        reasons.append(
            f"DiT + VAE ({dit_gb + vae_gb:.1f} GB) stay on {vram_gb:.1f} GB; the encoder streams"
        )
    elif max(dit_gb, te_gb) + vae_gb + act + margin <= vram_gb:
        choice = PLACE_SEQUENTIAL
        reasons.append(
            f"one component at a time: largest {max(dit_gb, te_gb):.1f} GB + working set fits {vram_gb:.1f} GB"
        )
    else:
        choice = PLACE_STREAM
        reasons.append(f"DiT {dit_gb:.1f} GB + working set exceeds {vram_gb:.1f} GB: blocks stream")
    plan.placement = choice
    if choice == PLACE_DIT_RESIDENT:
        spare = vram_gb - dit_gb - vae_gb - act - margin - TE_TOP_GB - layer_gb * inflight
        plan.te_resident_layers = max(0, min(te_layers, int(spare / layer_gb)))
        reasons.append(f"{plan.te_resident_layers}/{te_layers} encoder layers stay on the device")
    elif choice == PLACE_STREAM:
        spare = vram_gb - act - margin - vae_gb - DIT_TOP_GB - TE_TOP_GB - block_gb
        spare -= (inflight - 1) * max(block_gb, layer_gb)
        plan.dit_resident_blocks = max(0, min(num_blocks, int(spare / block_gb)))
        # The resident blocks hold their room through the encode, so every encoder layer streams.
        plan.te_resident_layers = 0
        reasons.append(
            f"{plan.dit_resident_blocks}/{num_blocks} DiT blocks stay on the device; encoder streams"
        )
    if host_gb is not None:
        dit_host = {
            PLACE_RESIDENT: 0.0,
            PLACE_DIT_RESIDENT: 0.0,
            PLACE_SEQUENTIAL: dit_gb + vae_gb,
            PLACE_STREAM: block_gb * (num_blocks - int(plan.dit_resident_blocks or 0)),
        }[choice]
        need = dit_host + te_gb + 4.0
        plan.host_need_gb = need
        plan.te_kept_on_host = host_gb >= need
        # Loading per prompt only frees the encoder between prompts: each load still holds it whole beside the
        # host-side DiT blocks, so a host below that peak would be OOM-killed on the first prompt.
        peak = dit_host + te_gb + 1.0
        if host_gb < peak:
            raise RuntimeError(
                f"MiniMax-H3-Image needs about {peak:.0f} GB of free system memory on this GPU "
                f"({choice} placement: {dit_host:.1f} GB of DiT blocks on the host plus the {te_gb:.1f} GB "
                f"text encoder while it loads), and {host_gb:.1f} GB is available. Close other applications or "
                f"use a machine with more memory."
            )
        if not plan.te_kept_on_host:
            reasons.append(
                f"host {host_gb:.1f} GB < {need:.1f} GB for the host-side DiT + encoder: the encoder loads per prompt"
            )
    plan.reasons = reasons
    return plan


def replan_for_encoder(
    plan: H3ImagePlan,
    te_gb: float,
    plan_args: dict,
    slack_gb: float = 1.0,
) -> H3ImagePlan:
    """The plan again with the encoder's real size when it is larger than the one budgeted (a bf16 fallback for the
    int8 encoder, a local text_encoder/ folder); the plan unchanged otherwise. Raises when no tier fits."""
    if te_gb <= plan.te_gb + slack_gb:
        return plan
    new = plan_placement(te_gb = te_gb, **plan_args)
    vram = plan_args.get("vram_gb")
    if vram is not None and new.placement == PLACE_STREAM and not plan_args.get("forced"):
        floor = activation_gb(plan_args.get("width", 2048), plan_args.get("height", 2048), 0) + 1.5
        floor += H3_IMAGE_VAE_GB + DIT_TOP_GB + TE_TOP_GB
        if vram < floor:
            raise RuntimeError(
                f"MiniMax-H3-Image: the text encoder loaded is {te_gb:.1f} GB (the bf16 one, not Studio's 27.2 GB "
                f"int8 encoder) and no placement fits {vram:.1f} GB of GPU memory with it. Load it with the int8 "
                f"text encoder."
            )
    new.reasons = [
        f"encoder is {te_gb:.1f} GB, not the {plan.te_gb:.1f} GB planned: planned again"
    ] + list(new.reasons)
    logger.warning("diffusion.minimax_h3_image: %s", new.reasons[0])
    return new


class ModuleListStreamer:
    """Streams the non-resident members of a ModuleList through the device: a forward pre-hook swaps each tensor for
    a device copy, the post-hook swaps the host original back (no device-to-host copy). Works for torchao
    tensor-subclass weights, whose ``.to(device)`` returns a device subclass tensor."""

    def __init__(self, modules: Any, device: Any, resident: int) -> None:
        import torch

        self.device = torch.device(device)
        self.handles = []
        self.streamed = 0
        self._host: dict = {}
        for index, module in enumerate(modules):
            if index < resident:
                module.to(self.device)
                continue
            module.to("cpu")
            self._host[id(module)] = [
                (owner, kind, name, tensor)
                for owner in module.modules()
                for kind, table in (("p", owner._parameters), ("b", owner._buffers))
                for name, tensor in table.items()
                if tensor is not None
            ]
            self.handles.append(module.register_forward_pre_hook(self._onload))
            self.handles.append(module.register_forward_hook(self._offload))
            self.streamed += 1

    def _onload(self, module: Any, args: Any) -> None:
        import torch
        for owner, kind, name, tensor in self._host[id(module)]:
            moved = tensor.to(self.device, non_blocking = True)
            if kind == "p":
                owner._parameters[name] = torch.nn.Parameter(moved, requires_grad = False)
            else:
                owner._buffers[name] = moved

    def _offload(self, module: Any, args: Any, output: Any) -> None:
        for owner, kind, name, tensor in self._host[id(module)]:
            (owner._parameters if kind == "p" else owner._buffers)[name] = tensor

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def torch_no_grad(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import torch
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


def _pipeline_base() -> Any:
    _close_diffusers_import_window()
    try:
        from diffusers import DiffusionPipeline
    except (
        ImportError
    ):  # diffusers missing: the module still imports (planner, family tables); loading raises
        return object
    return DiffusionPipeline


def _image_output(images: list) -> Any:
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
    return ImagePipelineOutput(images = images)


class MiniMaxH3ImagePipeline(_pipeline_base()):
    """Text-to-image and reference editing (1-16 ``<Picture N>`` references) for MiniMax-H3-Image."""

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]
    _optional_components = ["processor"]

    def __init__(
        self,
        transformer: Any,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        scheduler: Any,
        processor: Any = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer = transformer,
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            scheduler = scheduler,
            processor = processor,
        )
        self._interrupt = False
        self._num_timesteps = H3_IMAGE_STEPS
        self._unsloth_h3_plan: Optional[H3ImagePlan] = None
        self._unsloth_h3_device: Any = None
        self._unsloth_h3_te_loader: Optional[Callable[[], Any]] = None
        self._unsloth_h3_streamers: list = []
        self._unsloth_h3_last_stats: dict = {}

    @property
    def interrupt(self) -> bool:
        return bool(self._interrupt)

    def _device(self) -> Any:
        import torch
        if self._unsloth_h3_device is not None:
            return torch.device(self._unsloth_h3_device)
        try:
            return self._execution_device
        except Exception:  # noqa: BLE001 - no registered module on a device
            return torch.device("cpu")

    def _placement(self) -> str:
        plan = self._unsloth_h3_plan
        return plan.placement if plan is not None else PLACE_RESIDENT

    def _enter(self, name: str) -> Any:
        """Put component ``name`` where its phase runs; returns it."""
        module = getattr(self, name, None)
        if module is None and name == "text_encoder" and self._unsloth_h3_te_loader is not None:
            module = self._unsloth_h3_te_loader()
            self.text_encoder = module
        if module is None:
            return None
        if self._placement() == PLACE_SEQUENTIAL:
            module.to(self._device())
        return module

    def _leave(self, name: str) -> None:
        module = getattr(self, name, None)
        if module is None:
            return
        placement = self._placement()
        if placement == PLACE_SEQUENTIAL:
            module.to("cpu")
            _empty_cache()
        if name == "text_encoder" and self._unsloth_h3_te_loader is not None:
            self.text_encoder = None
            for streamer in list(self._unsloth_h3_streamers):
                if getattr(streamer, "_unsloth_role", None) == "text_encoder":
                    streamer.remove()
                    streamer._host.clear()
                    self._unsloth_h3_streamers.remove(streamer)
            free_module_storage(module)
            del module
            gc.collect()
            _empty_cache()

    def release(self) -> None:
        """Drop every component and streaming hook so an unload frees the card and the host at once (the hooks hold
        the host copies of the streamed blocks)."""
        for streamer in list(self._unsloth_h3_streamers):
            streamer.remove()
            streamer._host.clear()
        self._unsloth_h3_streamers.clear()
        self._unsloth_h3_te_loader = None
        owned = getattr(self, "_unsloth_h3_owned", ())
        for name in ("transformer", "text_encoder", "vae"):
            module = getattr(self, name, None)
            if module is not None:
                if name in owned:
                    free_module_storage(module)
                setattr(self, name, None)
        gc.collect()
        _empty_cache()

    def _processor(self) -> Any:
        if self.processor is not None:
            return self.processor
        raise ValueError(
            "MiniMax-H3-Image editing needs the Qwen3-VL processor (processor/ of the repo)."
        )

    def encode_prompt(
        self,
        prompt: str,
        vision_images: Optional[list] = None,
    ) -> tuple:
        """(prompt_embeds, per-row text tags). Named like diffusers' encoder entry point so ``diffusion_prompt_cache``
        wraps it; VAE reference latents are never cached (seeded sample)."""
        import torch

        tokenizer = self.tokenizer
        max_len = H3_IMAGE_MAX_SEQUENCE_LENGTH
        token_ids: list[int] = []
        tags: list[int] = []
        vision_inputs: dict = {}
        if vision_images:
            processor = self._processor()
            vision = processor.image_processor(images = vision_images, return_tensors = "pt")
            merge = int(processor.image_processor.merge_size) ** 2
            start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            for index, grid in enumerate(vision["image_grid_thw"]):
                label = tokenizer(f"Picture {index + 1}: ", add_special_tokens = False)["input_ids"][
                    :max_len
                ]
                block = [start] + [pad] * (int(grid.prod()) // merge) + [end]
                token_ids += label + block
                tags += [H3_IMAGE_TEXT_TAG] * len(label) + [H3_IMAGE_IMAGE_TAG] * len(block)
            vision_inputs = {
                "pixel_values": vision["pixel_values"],
                "image_grid_thw": vision["image_grid_thw"],
            }
        prompt_ids = tokenizer(prompt, add_special_tokens = False)["input_ids"][:max_len]
        token_ids += prompt_ids
        tags += [H3_IMAGE_TEXT_TAG] * len(prompt_ids)
        if not token_ids:
            raise ValueError("MiniMax-H3-Image needs a non-empty prompt.")
        encoder = self._enter("text_encoder")
        try:
            embeds = qwen3vl_hidden_state(
                encoder,
                self.processor,
                token_ids,
                vision_inputs,
                layer = H3_IMAGE_TEXT_ENCODER_LAYER,
                device = self._device(),
            )
        finally:
            self._leave("text_encoder")
        return embeds, torch.tensor(tags, dtype = torch.long)

    def _encode(self, prompt: str, vision_images: Optional[list]) -> tuple:
        return self.encode_prompt(prompt, vision_images)

    def _streamer_stats(self) -> dict:
        """Per role: pinned prefetch counters, or {"pageable": n}."""
        out = {}
        for streamer in self._unsloth_h3_streamers:
            role = getattr(streamer, "_unsloth_role", "?")
            read = getattr(streamer, "stats", None)
            out[role] = read() if callable(read) else {"pageable": int(streamer.streamed)}
        return out

    def _prompt_cache_hits(self) -> int:
        cache = getattr(self, "_unsloth_prompt_cache", None)
        stats = getattr(cache, "stats", None)
        return int(stats.get("hits", 0)) if isinstance(stats, dict) else 0

    def _encode_references(self, images: list, generator: Any) -> list:
        import numpy as np
        import torch

        device = self._device()
        vae = self._enter("vae")
        try:
            vae_dtype = _module_dtype(vae)
            mean = torch.tensor(H3_IMAGE_PIXEL_MEAN, device = device).view(1, -1, 1, 1)
            std = torch.tensor(H3_IMAGE_PIXEL_STD, device = device).view(1, -1, 1, 1)
            lmean = torch.tensor(vae.config.latents_mean, device = device).view(1, -1, 1, 1)
            lstd = torch.tensor(vae.config.latents_std, device = device).view(1, -1, 1, 1)
            out = []
            r = H3_IMAGE_VAE_SCALE
            for image in images:
                pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
                pixels = pixels[..., : pixels.shape[-2] // r * r, : pixels.shape[-1] // r * r]
                pixels = (pixels.to(torch.float32).div(255.0) - mean) / std
                posterior = vae.encode(pixels.to(vae_dtype), return_dict = False)[0]
                latents = posterior.sample(generator = generator).to(torch.float16).float()
                latents = latents[..., : latents.shape[-2] // 2 * 2, : latents.shape[-1] // 2 * 2]
                out.append((latents - lmean) / lstd)
            return out
        finally:
            self._leave("vae")

    @staticmethod
    def prepare_references(images: Any) -> tuple[list, list]:
        """(VAE copies, encoder copies) of the references: upright, RGB, short edge bounded per consumer."""
        from PIL import Image, ImageOps

        if images is None:
            return [], []
        items = images if isinstance(images, (list, tuple)) else [images]
        if len(items) > H3_IMAGE_MAX_REFERENCES:
            raise ValueError(
                f"MiniMax-H3-Image takes at most {H3_IMAGE_MAX_REFERENCES} reference images, got {len(items)}."
            )
        prepared = []
        for image in items:
            if not isinstance(image, Image.Image):
                raise ValueError("MiniMax-H3-Image references must be PIL images.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < H3_IMAGE_CANVAS_MULTIPLE:
                raise ValueError(
                    f"A reference image needs at least {H3_IMAGE_CANVAS_MULTIPLE} pixels on each side, got {image.size}."
                )
            prepared.append(image)

        def resize(image: Any, lo: int, hi: int) -> Any:
            size = bound_short_edge(*image.size, lo, hi)
            return image if image.size == size else image.resize(size, resample = Image.LANCZOS)

        vae_images = [resize(i, *H3_IMAGE_REFERENCE_SHORT_EDGE) for i in prepared]
        vision_images = [resize(i, *H3_IMAGE_VISION_SHORT_EDGE) for i in prepared]
        return vae_images, vision_images

    def __call__(
        self,
        prompt: Any = None,
        image: Any = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = H3_IMAGE_STEPS,
        guidance_scale: float = H3_IMAGE_GUIDANCE,
        negative_prompt: Any = None,
        num_images_per_prompt: int = 1,
        generator: Any = None,
        latents: Any = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: Optional[list] = None,
        min_generation_area: int = H3_IMAGE_MIN_GENERATION_AREA,
        max_sequence_length: int = H3_IMAGE_MAX_SEQUENCE_LENGTH,
    ) -> Any:
        import torch

        if num_inference_steps not in (None, H3_IMAGE_STEPS):
            raise ValueError(
                f"MiniMax-H3-Image is distilled to {H3_IMAGE_STEPS} steps with one output head each; "
                f"got num_inference_steps={num_inference_steps}."
            )
        if guidance_scale is not None and float(guidance_scale) != H3_IMAGE_GUIDANCE:
            raise ValueError(
                "MiniMax-H3-Image is guidance-distilled: guidance_scale must be 1 (no CFG)."
            )
        if negative_prompt:
            logger.info(
                "diffusion.minimax_h3_image: negative prompt ignored (the model has no unconditional branch)"
            )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt or [])
        if not prompts or any(not isinstance(p, str) or not p for p in prompts):
            raise ValueError("MiniMax-H3-Image needs a non-empty prompt.")
        jobs = [p for p in prompts for _ in range(max(1, int(num_images_per_prompt or 1)))]
        generators = generator if isinstance(generator, list) else [generator] * len(jobs)
        if len(generators) != len(jobs):
            raise ValueError(f"got {len(generators)} generators for {len(jobs)} images")
        height = int(height or 2048)
        width = int(width or 2048)
        vae_refs, vision_refs = self.prepare_references(image)
        self._interrupt = False
        images = []
        for index, (text, gen) in enumerate(zip(jobs, generators)):
            pixels = self._render_one(
                text,
                vae_refs,
                vision_refs,
                height,
                width,
                gen,
                latents if index == 0 else None,
                min_generation_area,
                callback_on_step_end,
                callback_on_step_end_tensor_inputs or ["latents"],
                output_type,
            )
            if pixels is None:
                break
            images.extend(pixels)
        if output_type == "pt" and images and isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        if not return_dict:
            return (images,)
        return _image_output(images)

    @torch_no_grad
    def _render_one(
        self,
        prompt: str,
        vae_refs: list,
        vision_refs: list,
        height: int,
        width: int,
        generator: Any,
        latents: Any,
        min_generation_area: int,
        callback: Optional[Callable],
        callback_inputs: list,
        output_type: str,
    ) -> Optional[list]:
        import torch

        stats: dict = {"canvas": None}
        t0 = time.perf_counter()
        device = self._device()
        canvas_h, canvas_w = canvas_size(height, width, min_generation_area)
        stats["canvas"] = (canvas_w, canvas_h)
        hits = self._prompt_cache_hits()
        embeds, text_tags = self.encode_prompt(prompt, vision_refs)
        stats["encode_s"] = time.perf_counter() - t0
        stats["encode_cached"] = self._prompt_cache_hits() > hits
        patch = tuple(self.transformer.config.patch_size)
        channels = int(self.vae.config.latent_channels) if self.vae is not None else 24
        lh, lw = canvas_h // H3_IMAGE_VAE_SCALE, canvas_w // H3_IMAGE_VAE_SCALE
        shape = (1, channels, 1, lh, lw)
        if latents is None:
            from diffusers.utils.torch_utils import randn_tensor
            noise = randn_tensor(shape, generator = generator, device = device, dtype = torch.float32)
        else:
            noise = latents.to(device, torch.float32).reshape(shape)
        target_rows = patchify(noise, patch)
        reference_latents = self._encode_references(vae_refs, generator) if vae_refs else []
        grids = [tuple(r.shape[-2:]) for r in reference_latents]
        pos, tags, image_idx, text_idx, n_ref = build_packed_layout(text_tags, grids, lh, lw, patch)
        if reference_latents:
            from diffusers.utils.torch_utils import randn_tensor

            cond = []
            for ref in reference_latents:
                ref = ref.to(device, torch.float32)
                ref_noise = randn_tensor(
                    ref.shape, generator = generator, device = device, dtype = torch.float32
                )
                noised = self.scheduler.scale_noise(
                    ref, torch.tensor(H3_IMAGE_REFERENCE_NOISE_AUG), ref_noise
                )
                cond.append(patchify(noised[:, :, None], patch))
            cond = torch.cat(cond)
            if cond.shape[0] != n_ref:
                raise ValueError(
                    f"layout reserved {n_ref} reference rows, references packed {cond.shape[0]}"
                )
            rows = torch.cat([cond, target_rows])
        else:
            rows = target_rows
        self.scheduler.set_timesteps(device = device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        seq = int(image_idx.numel() + text_idx.numel())
        plan = row_timestep_plan(timesteps.cpu(), seq, image_idx[:n_ref])
        pos_dtype = torch.float64 if device.type in ("cpu", "cuda") else torch.float32
        fields = {
            "position_ids": pos.to(device, pos_dtype),
            "token_tags": tags.to(device),
            "image_indices": image_idx.to(device),
            "text_indices": text_idx.to(device),
        }
        transformer = self._enter("transformer")
        t1 = time.perf_counter()
        try:
            dtype = _module_dtype(transformer, skip_fp32 = True)
            prompt_embeds = embeds.to(device = device, dtype = dtype)
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    return None
                unique, index = plan[i]
                velocity = transformer(
                    hidden_states = rows[None],
                    encoder_hidden_states = prompt_embeds,
                    timestep = unique.to(device),
                    timestep_indices = index.to(device),
                    pdd_step_index = i,
                    return_dict = False,
                    **fields,
                )[0]
                rows[n_ref:] = self.scheduler.step(velocity[0, n_ref:].float(), t, rows[n_ref:])[0]
                if callback is not None:
                    view = unpatchify(rows[n_ref:], lh, lw, channels, patch)
                    payload = {"latents": view} if "latents" in callback_inputs else {}
                    callback(self, i, t, payload)
        finally:
            self._leave("transformer")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats["denoise_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        final = unpatchify(rows[n_ref:], lh, lw, channels, patch)
        if output_type == "latent":
            self._unsloth_h3_last_stats = stats
            return [final[0]]
        images = self._decode(final, height, width, output_type)
        stats["decode_s"] = time.perf_counter() - t2
        stats["total_s"] = time.perf_counter() - t0
        stats["streamers"] = self._streamer_stats()
        self._unsloth_h3_last_stats = stats
        return images

    def _decode(self, latents: Any, height: int, width: int, output_type: str) -> list:
        import numpy as np
        import torch
        from PIL import Image

        device = self._device()
        vae = self._enter("vae")
        try:
            lmean = torch.tensor(vae.config.latents_mean, device = device).view(1, -1, 1, 1)
            lstd = torch.tensor(vae.config.latents_std, device = device).view(1, -1, 1, 1)
            z = latents.to(device, torch.float32) * lstd + lmean
            image = vae.decode(z.to(_module_dtype(vae)), return_dict = False)[0]
        finally:
            self._leave("vae")
        mean = torch.tensor(H3_IMAGE_PIXEL_MEAN, device = image.device).view(1, -1, 1, 1)
        std = torch.tensor(H3_IMAGE_PIXEL_STD, device = image.device).view(1, -1, 1, 1)
        image = (image.float() * std + mean).clamp(0, 1)
        # The release truncates to 8 bits, then resizes the 8-bit image (LANCZOS).
        u8 = (image * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for frame in u8:
            pil = Image.fromarray(frame)
            if pil.size != (width, height):
                pil = pil.resize((width, height), resample = Image.LANCZOS)
            out.append(pil)
        if output_type == "pil":
            return out
        arrays = [np.asarray(p).astype(np.float32) / 255.0 for p in out]
        if output_type == "np":
            return arrays
        return [torch.from_numpy(a).permute(2, 0, 1) for a in arrays]


def _module_dtype(module: Any, skip_fp32: bool = False) -> Any:
    import torch
    for name, p in module.named_parameters():
        if not p.is_floating_point():
            continue
        if (
            skip_fp32
            and p.dtype == torch.float32
            and any(k in name for k in ("proj_in", "time_embedder", "pdd_proj_out", "rope"))
        ):
            continue
        return p.dtype
    return torch.float32


# "1" lets the loader run on any device (the CPU tests, diagnostics); Studio refuses everything but CUDA / ROCm.
ALLOW_ANY_DEVICE_ENV = "UNSLOTH_H3_IMAGE_ALLOW_ANY_DEVICE"


def h3_image_device_refusal(device: Any) -> Optional[str]:
    """The load refusal for a device this model cannot run on, else None. The 33B DiT and the 33B conditioner are
    streamed and quantized through CUDA (NVIDIA, or AMD through ROCm); Apple Silicon and the CPU have no route yet."""
    kind = str(getattr(device, "type", device) or "").split(":")[0].strip().lower()
    if kind == "cuda" or os.environ.get(ALLOW_ANY_DEVICE_ENV, "").strip() == "1":
        return None
    where = {"mps": "Apple Silicon", "cpu": "the CPU", "": "this machine"}.get(
        kind, f"this {kind} device"
    )
    return (
        f"MiniMax-H3-Image needs an NVIDIA or AMD GPU and cannot run on {where} yet: its 33B image model and 33B "
        f"text encoder are streamed and quantized through CUDA or ROCm. Load it on a machine with an NVIDIA or AMD "
        f"GPU."
    )


def _free_tensor_storage(tensor: Any) -> bool:
    """Swap ``tensor`` in place for an empty one, so its storage goes even while some other object still holds the
    tensor (plain tensors through ``.data``, torchao subclasses through ``swap_tensors``)."""
    import torch

    empty = torch.empty(0, dtype = tensor.dtype)
    try:
        if type(tensor) in (torch.Tensor, torch.nn.Parameter) and type(tensor.data) is torch.Tensor:
            tensor.data = empty
            return True
        replacement = (
            torch.nn.Parameter(empty, requires_grad = False)
            if isinstance(tensor, torch.nn.Parameter) or type(tensor) is not torch.Tensor
            else empty
        )
        torch.utils.swap_tensors(tensor, replacement)
        return True
    except Exception:  # noqa: BLE001 - teardown is best effort; dropping the module still frees an unshared tensor
        return False


def free_module_storage(module: Any) -> int:
    """Empty every parameter and buffer of a component this pipeline owns, in place: an unload must not depend on
    every outside reference (a retained frame, a cached closure) having let go of the module first."""
    freed = 0
    for sub in module.modules():
        for table in (sub._parameters, sub._buffers):
            for tensor in list(table.values()):
                if tensor is not None and tensor.numel():
                    freed += int(_free_tensor_storage(tensor))
    return freed


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def qwen3vl_hidden_state(
    text_encoder: Any,
    processor: Any,
    token_ids: list,
    vision_inputs: dict,
    *,
    layer: int,
    device: Any,
) -> Any:
    """``hidden_states[layer]`` of the Qwen3-VL conditioner for one tokenized presentation. Calls
    ``text_encoder.model`` (the head is never needed) and builds ``mm_token_type_ids`` only where the installed
    transformers takes them (5.x); 4.57 derives the vision layout from the pad tokens."""
    import torch

    config = getattr(text_encoder.config, "text_config", text_encoder.config)
    if int(config.num_hidden_layers) <= layer:
        raise ValueError(
            f"MiniMax-H3 conditions on hidden_states[{layer}], which needs more than {layer} decoder layers; "
            f"the encoder has {config.num_hidden_layers}."
        )
    model = text_encoder.model
    hook = getattr(text_encoder, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(text_encoder)
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = device)
    kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "use_cache": False,
        "output_hidden_states": False,
    }
    params = inspect.signature(model.forward).parameters
    if (
        "mm_token_type_ids" in params
        and processor is not None
        and hasattr(processor, "create_mm_token_type_ids")
    ):
        kwargs["mm_token_type_ids"] = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype = torch.long, device = device
        )
    te_dtype = _module_dtype(text_encoder)
    for name, value in (vision_inputs or {}).items():
        kwargs[name] = value.to(device, te_dtype) if name.startswith("pixel_") else value.to(device)
    # hidden_states[layer] is the output of decoder layer ``layer - 1``: captured by a hook instead of keeping every
    # layer's states (~10 GB with 16 references), and the layers past it are cut for the call.
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None or len(layers) < layer:
        kwargs["output_hidden_states"] = True
        return model(**kwargs).hidden_states[layer]
    captured: dict = {}

    def _capture(_module: Any, _args: Any, output: Any) -> None:
        captured["h"] = output[0] if isinstance(output, (tuple, list)) else output
        raise _ReadLayerReached()

    handle = layers[layer - 1].register_forward_hook(_capture)
    try:
        model(**kwargs)
    except _ReadLayerReached:
        pass
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"the conditioner never ran decoder layer {layer - 1}")
    return captured["h"]


class _ReadLayerReached(Exception):
    """Stops the conditioner once the read layer has run: nothing after it is used."""


# Metadata a hosted table-form checkpoint declares (the prequant_repos extension point), mirroring video's curve form.
ADALN_FORM_KEY = "adaln_form"
ADALN_GRID_FORM = "grid_table"
ADALN_GRID_KEY = "adaln_grid"


def _hub_files(
    repo_id: str,
    patterns: Sequence[str],
    *,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
) -> Path:
    """Local root holding ``patterns`` of ``repo_id`` (a local folder is used as is)."""
    if os.path.isdir(str(repo_id)):
        return Path(repo_id)
    from huggingface_hub import snapshot_download

    from .diffusion_families import pinned_revision_kwargs

    root = snapshot_download(
        repo_id,
        allow_patterns = list(patterns),
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        max_workers = 8,
        **pinned_revision_kwargs(repo_id),
    )
    return Path(root)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def native_shard_map(transformer_dir: Path) -> dict[str, Path]:
    """``{native key: shard path}`` from the index, else from every shard's header."""
    index = transformer_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map") or {}
        return {k: transformer_dir / v for k, v in weight_map.items()}
    from safetensors import safe_open

    out: dict[str, Path] = {}
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework = "pt", device = "cpu") as f:
            for key in f.keys():
                out[key] = shard
    if not out:
        raise FileNotFoundError(f"no transformer safetensors under {transformer_dir}")
    return out


class _ShardReader:
    """Keeps one mmap'd ``safe_open`` per shard; ``get`` materialises a single tensor."""

    def __init__(self, shard_map: dict[str, Path]) -> None:
        self.map = shard_map
        self._open: dict[Path, Any] = {}
        self.bytes_read = 0

    def get(self, key: str) -> Any:
        from safetensors import safe_open

        path = self.map[key]
        handle = self._open.get(path)
        if handle is None:
            handle = safe_open(str(path), framework = "pt", device = "cpu")
            self._open[path] = handle
        tensor = handle.get_tensor(key)
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def close(self) -> None:
        self._open.clear()


def _assign(module: Any, state: dict, *, where: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict = False, assign = True)
    if unexpected:
        raise KeyError(f"{where}: unexpected keys {sorted(unexpected)[:5]}")
    stranded = [
        n for n, p in list(module.named_parameters()) + list(module.named_buffers()) if p.is_meta
    ]
    if stranded:
        raise KeyError(f"{where}: no checkpoint tensor for {sorted(stranded)[:5]}")


def _cast(tensor: Any, key: str, dtype: Any) -> Any:
    """Mixed precision as released: the fp32 modules stay fp32, the block stack follows ``dtype``."""
    import torch

    if not tensor.is_floating_point():
        return tensor
    if key.startswith(("proj_in.", "time_embedder.", "pdd_proj_out.")):
        return tensor.to(torch.float32)
    return tensor.to(dtype)


def quantize_block_(block: Any, scheme: str, *, device: Any) -> int:
    """Quantize the six big Linears of one block in place on ``device`` (ConvRot-rotated for int8)."""
    from torch import nn
    from torchao.quantization import quantize_

    from .diffusion_transformer_quant import _make_quant_config

    names = [
        n
        for n, m in block.named_modules()
        if isinstance(m, nn.Linear) and n in QUANT_LINEAR_SUFFIXES
    ]
    if not names:
        return 0
    block.to(device)
    if scheme == "int8":
        from .diffusion_convrot import rotate_linears_

        modules = dict(block.named_modules())
        rotatable = [n for n in names if modules[n].in_features % CONVROT_GROUP == 0]
        rotate_linears_(block, rotatable, CONVROT_GROUP)
    wanted = set(names)
    quantize_(block, _make_quant_config(scheme), filter_fn = lambda module, fqn: fqn in wanted)
    wrap_row_multiple(block, names)
    return len(names)


# cuBLASLt int8 GEMMs (torch._int_mm) refuse some row counts with CUBLAS_STATUS_NOT_SUPPORTED, and torchao then
# silently falls back to a much slower fp32 matmul. A packed sequence rarely lands on a supported count; multiples
# of 32 always ran and also suit the fp8 row-wise GEMM.
ROW_MULTIPLE = 32


def _row_pad_class() -> Any:
    from torch import nn
    class H3ImageRowPad(nn.Module):
        """Pads a quantized Linear's rows to a multiple of ``ROW_MULTIPLE`` by repeating row 0, then drops them. Exact:
        every row's activation scale is its own (per-row dynamic quantization), so the kept rows are unchanged."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            import torch

            lead = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            m = flat.shape[0]
            pad = (-m) % ROW_MULTIPLE
            if pad and m:
                flat = torch.cat([flat, flat[:1].expand(pad, -1)], dim = 0)
                out = self.inner(flat)[:m]
            else:
                out = self.inner(flat)
            return out.reshape(*lead, out.shape[-1])

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                return getattr(self._modules["inner"], name)

    return H3ImageRowPad


_ROW_PAD_CLASS: Any = None


def wrap_row_multiple(module: Any, names: Sequence[str]) -> int:
    """Wrap each named quantized Linear (per-row activation scales only) in the row pad; returns the count."""
    global _ROW_PAD_CLASS
    from .diffusion_quant_pad import activation_granularity_is_per_row, is_quantized_linear

    if _ROW_PAD_CLASS is None:
        _ROW_PAD_CLASS = _row_pad_class()
    wrapped = 0
    for fqn in names:
        parent_name, _, leaf = fqn.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        linear = getattr(parent, leaf)
        if not is_quantized_linear(linear) or activation_granularity_is_per_row(linear) is not True:
            continue
        setattr(parent, leaf, _ROW_PAD_CLASS(linear))
        wrapped += 1
    return wrapped


def resolve_dit_scheme(
    requested: Optional[str],
    *,
    device: Any,
    vram_gb: Optional[float],
    width: int = 2048,
    height: int = 2048,
) -> tuple[Optional[str], str]:
    """(scheme or None for bf16, reason). Auto: bf16 when the table-form bf16 DiT fits the card beside its working
    set, else int8. fp8 needs SM 8.9+ (falls back to int8 elsewhere)."""
    import torch

    req = (requested or "auto").strip().lower()
    if req in ("bf16", "none", "off", "dense"):
        return None, "bf16 requested"
    dev = torch.device(device)
    if dev.type != "cuda":
        if req in QUANT_SCHEMES:
            return None, f"{req} needs a CUDA device; {dev.type} runs bf16"
        return None, f"{dev.type}: bf16"
    try:
        import torchao  # noqa: F401
    except Exception:  # noqa: BLE001 - torchao absent
        return None, "torchao unavailable: bf16"
    if req == "fp8":
        cap = torch.cuda.get_device_capability(dev)
        if tuple(cap) < (8, 9):
            return (
                "int8",
                f"fp8 needs compute capability 8.9+ (this card is {cap[0]}.{cap[1]}): int8 instead",
            )
        return "fp8", "fp8 requested"
    if req == "int8":
        return "int8", "int8 requested"
    need = H3_IMAGE_DIT_TABLE_BF16_GB + activation_gb(width, height) + 1.5
    if vram_gb is not None and vram_gb >= need:
        return None, f"auto: bf16 DiT ({H3_IMAGE_DIT_TABLE_BF16_GB} GB) fits {vram_gb:.1f} GB"
    return (
        "int8",
        f"auto: bf16 DiT would not fit {vram_gb if vram_gb is not None else 0:.1f} GB, int8 ConvRot",
    )


def device_budgets(device: Any) -> tuple[Optional[float], Optional[float]]:
    """(free VRAM GB on ``device`` or None, available host GB or None); env overrides win."""
    import torch

    vram = None
    dev = torch.device(device)
    if os.environ.get(VRAM_BUDGET_ENV):
        vram = float(os.environ[VRAM_BUDGET_ENV])
    elif dev.type == "cuda" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(dev)
        vram = free / 1e9
    host = None
    if os.environ.get(HOST_BUDGET_ENV):
        host = float(os.environ[HOST_BUDGET_ENV])
    else:
        try:
            import psutil
            host = psutil.virtual_memory().available / 1e9
        except Exception:  # noqa: BLE001 - psutil missing
            host = None
    return vram, host


def build_transformer_skeleton(config: dict, *, table_grid: Optional[Sequence[float]]) -> Any:
    """Meta-device model (buffers real), with the AdaLN table structure installed when ``table_grid`` is set."""
    from accelerate import init_empty_weights

    cls = transformer_class()
    with init_empty_weights(include_buffers = False):
        model = cls.from_config(config)
    if table_grid is not None:
        install_grid_time_embedding(model, table_grid)
    return model


def h3_image_prepare_prequant_model(logger_: Any = None) -> Callable:
    """``prepare_model`` for ``load_prequantized_transformer``: reshapes the skeleton to a hosted checkpoint's AdaLN
    form before ``load_state_dict``: the H3 video curve form (``adaln_form=curve``) or this module's table form
    (``adaln_form=grid_table`` + its grid)."""

    def _prepare(transformer: Any, metadata: Optional[dict]) -> None:
        import torch

        declared = metadata.get("attention") if isinstance(metadata, dict) else None
        if isinstance(declared, str) and declared.strip():
            setattr(transformer, CHECKPOINT_ATTENTION_ATTR, declared.strip().lower())

        from .video_minimax_h3_adaln import apply_h3_adaln_curve, is_curve_checkpoint

        if is_curve_checkpoint(metadata):
            # The hosted H3-Image files use the H3 video curve form: the raw timestep indexes the curve, so the
            # pipeline's float32 timesteps (0.999 reference rows included) drive it unchanged.
            apply_h3_adaln_curve(transformer, metadata, logger = logger_)
            return
        if not isinstance(metadata, dict) or metadata.get(ADALN_FORM_KEY) != ADALN_GRID_FORM:
            return
        grid = metadata.get(ADALN_GRID_KEY)
        if isinstance(grid, str):
            grid = json.loads(grid)
        grid = [float(t) for t in grid]
        k = len(grid)
        dtype = torch.bfloat16
        for block in transformer.transformer_blocks:
            out = block.adaln_proj.linear.out_features
            convert_block_adaln_to_table(block, torch.empty(k, out, dtype = dtype))
        out = transformer.norm_out.linear.out_features
        convert_norm_out_to_table(transformer.norm_out, torch.empty(k, out, dtype = dtype))
        install_grid_time_embedding(transformer, grid)
        if logger_ is not None:
            logger_.info(
                "diffusion.minimax_h3_image: hosted checkpoint is table form over %d timesteps", k
            )

    return _prepare


@functools.lru_cache(maxsize = None)
def native_config_transformer_class() -> Any:
    """The transformer class whose ``from_config`` also takes the release's native ``transformer/config.json``, for
    loaders that read the base repo's config (``diffusion_prequant.load_prequantized_transformer``)."""
    base = transformer_class()

    class MiniMaxH3ImageNativeConfigTransformer(base):
        @classmethod
        def from_config(
            cls,
            config: Any = None,
            **kwargs: Any,
        ) -> Any:
            cfg = dict(config or {})
            if (
                cfg.get("_class_name") == H3_IMAGE_NATIVE_TRANSFORMER_CLASS
                or "ffn_hidden_size" in cfg
            ):
                cfg = transformer_config_from_native(cfg)
            return base.from_config(cfg, **kwargs)

    MiniMaxH3ImageNativeConfigTransformer.__name__ = base.__name__
    return MiniMaxH3ImageNativeConfigTransformer


# The hosted files also quantize the text-conditioning path (context_embedder and the two token_refiner blocks), which
# the on-the-fly path keeps bf16: per-row dynamic activation quantization of the prompt embeddings moves the
# conditioning every block reads. "1" keeps them quantized (A/B only).
QUANT_CONDITIONING_ENV = "UNSLOTH_H3_IMAGE_QUANT_CONDITIONING"
CONDITIONING_PREFIXES = ("context_embedder", "token_refiner.")


def densify_conditioning_path(transformer: Any, dtype: Any = None) -> int:
    """Replace the quantized weights of the text-conditioning Linears with their dequantized copies (bf16), so the
    prompt embeddings are never activation-quantized. Rotated (ConvRot) Linears keep rotating the activation: the
    dequantized weight is the rotated one, so the product is unchanged. Returns the count."""
    import torch

    from .diffusion_quant_pad import is_quantized_linear

    if os.environ.get(QUANT_CONDITIONING_ENV, "").strip() == "1":
        return 0
    dtype = dtype or torch.bfloat16
    count = 0
    for name, module in transformer.named_modules():
        if not name.startswith(CONDITIONING_PREFIXES) or not isinstance(module, torch.nn.Linear):
            continue
        if not is_quantized_linear(module):
            continue
        weight = module.weight
        dense = weight.dequantize() if hasattr(weight, "dequantize") else None
        if dense is None:
            continue
        module.weight = torch.nn.Parameter(dense.to(dtype), requires_grad = False)
        count += 1
    return count


def hosted_prequant_available(
    repo: str,
    filename: Optional[str],
    *,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
) -> bool:
    """Whether the hosted file exists: the local cache first, then (online) one Hub lookup per process. A failed
    lookup reads as absent, so the load quantizes on the fly instead of failing."""
    if not filename:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        if isinstance(try_to_load_from_cache(repo, filename, cache_dir = cache_dir), str):
            return True
    except Exception:  # noqa: BLE001 - a cache probe only
        pass
    if local_files_only:
        return False
    return _hub_file_exists(repo, filename, hf_token)


@functools.lru_cache(maxsize = 32)
def _hub_file_exists(repo: str, filename: str, hf_token: Optional[str]) -> bool:
    try:
        from huggingface_hub import file_exists
        return bool(file_exists(repo, filename, token = hf_token))
    except Exception:  # noqa: BLE001 - offline / gated / not yet created: quantize on the fly
        return False


def load_hosted_prequant_transformer(
    scheme: str,
    *,
    base: str,
    device: Any,
    dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
    keep_on_device: bool,
) -> Optional[Any]:
    """A hosted pre-quantized DiT for ``scheme`` when the family declares one (``prequant_repos`` /
    ``prequant_filenames``, the H3 video mechanism), else None and the caller quantizes on the fly.

    The checkpoint is Studio's prequant container over the diffusers-layout model INCLUDING ``pdd_proj_out.*``, with a
    curve-form or table-form AdaLN that ``h3_image_prepare_prequant_model`` installs on the skeleton before the state
    dict is assigned. Nothing is fetched unless ``hosted_prequant_available`` finds the file."""
    try:
        from .diffusion_families import (
            detect_family,
            family_prequant_filename,
            family_prequant_repo,
        )

        fam = detect_family(base) or detect_family("", override = H3_IMAGE_FAMILY_NAME)
        if fam is None:
            return None
        repo = family_prequant_repo(fam, scheme, base)
        if not repo:
            return None
        from .diffusion_prequant import PrequantSource, load_prequantized_transformer

        filename = family_prequant_filename(fam, scheme)
        if not hosted_prequant_available(
            repo,
            filename,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        ):
            logger.info(
                "diffusion.minimax_h3_image: no hosted %s DiT at %s/%s yet; quantizing the release on the fly",
                scheme,
                repo,
                filename,
            )
            return None
        source = PrequantSource(
            kind = "repo",
            location = repo,
            filename = filename,
            declared_filenames = (filename,) if filename else (),
        )
        return load_prequantized_transformer(
            native_config_transformer_class(),
            base,
            source,
            device = str(device),
            dtype = dtype,
            hf_token = hf_token,
            scheme = scheme,
            cache_dir = cache_dir,
            prepare_model = h3_image_prepare_prequant_model(logger),
            local_files_only = local_files_only,
            logger = logger,
            placement_device = str(device) if keep_on_device else "cpu",
            family = H3_IMAGE_FAMILY_NAME,
        )
    except Exception as exc:  # noqa: BLE001 - a hosted artifact is an optimisation: quantize on the fly instead
        logger.warning(
            "diffusion.minimax_h3_image: hosted %s DiT unusable (%s); quantizing on the fly",
            scheme,
            exc,
        )
        return None


@dataclass
class H3ImageLoadReport:
    scheme: Optional[str]
    scheme_reason: str
    te_scheme: Optional[str]
    plan: H3ImagePlan
    classes: str
    layout: str
    table: bool
    load_s: float = 0.0
    quantized_linears: int = 0
    bytes_read: int = 0
    speed_levers: tuple = ()

    def as_dict(self) -> dict:
        return {
            "transformer_quant": self.scheme or "bf16",
            "transformer_quant_reason": self.scheme_reason,
            "text_encoder_quant": self.te_scheme or "bf16",
            "classes": self.classes,
            "layout": self.layout,
            "adaln_table": self.table,
            "load_s": round(self.load_s, 1),
            "quantized_linears": self.quantized_linears,
            "bytes_read": self.bytes_read,
            "speed_levers": list(self.speed_levers),
            **{f"plan_{k}": v for k, v in self.plan.as_dict().items()},
        }


def load_h3_image_transformer(
    root: Path,
    *,
    dtype: Any,
    scheme: Optional[str],
    device: Any,
    keep_on_device: bool,
    table: bool = True,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> tuple[Any, dict]:
    """The DiT from the native ``transformer/`` + ``pdd_fused_heads.safetensors`` (or a converted
    ``transformer_diffusers/``), one block at a time: convert, table the AdaLN, quantize on ``device``, then park on
    the host unless ``keep_on_device``. Host peak = the finished blocks + one dense block."""
    import torch
    from safetensors import safe_open

    check = check_cancelled or (lambda: None)
    converted_dir = root / DIFFUSERS_TRANSFORMER_SUBFOLDER
    native = not (converted_dir / "config.json").is_file()
    tdir = root / ("transformer" if native else DIFFUSERS_TRANSFORMER_SUBFOLDER)
    raw_config = _read_json(tdir / "config.json")
    if native:
        pdd = _read_json(root / PDD_CONFIG_FILE) if (root / PDD_CONFIG_FILE).is_file() else {}
        config = transformer_config_from_native(
            raw_config, int(pdd.get("num_inference_steps", H3_IMAGE_STEPS))
        )
    else:
        config = {k: v for k, v in raw_config.items() if not k.startswith("_")}
    heads, head_dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    grid = grid_timesteps() if table else None
    model = build_transformer_skeleton(config, table_grid = None)
    shard_map = native_shard_map(tdir)
    reader = _ShardReader(shard_map)
    target = torch.device(device)
    park = torch.device("cpu") if not keep_on_device else target
    stats = {"quantized_linears": 0, "layout": "native" if native else "diffusers"}

    def converted(prefix_native: str) -> dict:
        out = {}
        for key in [k for k in shard_map if k.startswith(prefix_native)]:
            tensor = reader.get(key)
            pairs = (
                convert_native_transformer_tensor(key, tensor, heads, head_dim)
                if native
                else [(key, tensor)]
            )
            for name, t in pairs:
                out[name] = _cast(t, name, dtype)
        return out

    try:
        top: dict = {}
        block_prefix = "blocks." if native else "transformer_blocks."
        for key in shard_map:
            if key.startswith(block_prefix):
                continue
            tensor = reader.get(key)
            pairs = (
                convert_native_transformer_tensor(key, tensor, heads, head_dim)
                if native
                else [(key, tensor)]
            )
            for name, t in pairs:
                top[name] = _cast(t, name, dtype)
        if native:
            heads_path = root / PDD_HEADS_FILE
            index = root / "model_index.json"
            if index.is_file():
                name = (
                    (_read_json(index).get("_minimax_h3") or {}).get("pdd_fused_heads")
                ) or PDD_HEADS_FILE
                heads_path = root / name
            with safe_open(str(heads_path), framework = "pt", device = "cpu") as f:
                pdd = {k: f.get_tensor(k) for k in f.keys()}
            for name, t in split_pdd_heads(pdd, int(config["num_pdd_heads"])).items():
                top[name] = t.to(torch.float32)
        blocks = model.transformer_blocks
        model.transformer_blocks = torch.nn.ModuleList()
        _assign(model, top, where = "transformer (top level)")
        model.transformer_blocks = blocks
        temb = grid_temb((model.time_proj, model.time_embedder), grid) if grid is not None else None
        if temb is not None:
            lin = model.norm_out.linear
            rows = dense_adaln_rows(temb, lin.weight, lin.bias, dtype, target)
            convert_norm_out_to_table(model.norm_out, rows)
        check()
        for index, block in enumerate(model.transformer_blocks):
            native_prefix = f"blocks.{index}." if native else f"transformer_blocks.{index}."
            state = converted(native_prefix)
            local = {k[len(f"transformer_blocks.{index}.") :]: v for k, v in state.items()}
            if temb is not None:
                weight = local.pop("adaln_proj.linear.weight")
                bias = local.pop("adaln_proj.linear.bias", None)
                rows = dense_adaln_rows(temb, weight, bias, dtype, target)
                del weight, bias
                convert_block_adaln_to_table(block, rows)
            _assign(block, local, where = f"transformer_blocks.{index}")
            del state, local
            if scheme in QUANT_SCHEMES:
                stats["quantized_linears"] += quantize_block_(block, scheme, device = target)
            block.to(park)
            check()
        if grid is not None:
            install_grid_time_embedding(model, grid)
        model.eval()
        model.requires_grad_(False)
        if keep_on_device:
            model.to(target)
        else:
            for name, child in model.named_children():
                if name != "transformer_blocks":
                    child.to(target if keep_on_device else "cpu")
        stats["bytes_read"] = reader.bytes_read
    finally:
        reader.close()
    return model, stats


def load_h3_image_vae(root: Path, *, dtype: Any = None) -> Any:
    """The image VAE from the native ``vae/`` (or a converted ``vae_diffusers/``), fp32 by default."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open

    converted_dir = root / DIFFUSERS_VAE_SUBFOLDER
    native = not (converted_dir / "config.json").is_file()
    vdir = root / ("vae" if native else DIFFUSERS_VAE_SUBFOLDER)
    raw = _read_json(vdir / "config.json")
    config = (
        vae_config_from_native(raw)
        if native
        else {k: v for k, v in raw.items() if not k.startswith("_")}
    )
    cls = vae_class()
    with init_empty_weights(include_buffers = False):
        vae = cls.from_config(config)
    heads = int(config.get("decoder_num_attention_heads", 16))
    head_dim = int(config.get("decoder_attention_head_dim", 64))
    state = {}
    files = sorted(vdir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no VAE safetensors under {vdir}")
    for path in files:
        with safe_open(str(path), framework = "pt", device = "cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                pairs = (
                    convert_native_vae_tensor(key, tensor, heads, head_dim)
                    if native
                    else [(key, tensor)]
                )
                for name, t in pairs:
                    state[name] = t.to(dtype or torch.float32) if t.is_floating_point() else t
    _assign(vae, state, where = "vae")
    vae.eval()
    vae.requires_grad_(False)
    return vae


def load_h3_image_text_encoder(
    scheme: Optional[str],
    *,
    dtype: Any,
    text_encoder_repo: str,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
    fallback_check: Optional[Callable[[], None]] = None,
) -> tuple[Any, Optional[str]]:
    """(encoder on the host, scheme engaged). int8 = Studio's hosted ConvRot conditioner (the video family's file),
    bf16 = the released encoder; a broken hosted file falls back to bf16 (66.7 GB). ``fallback_check`` runs before
    that fallback allocates anything and raises when the plan cannot hold it."""
    if scheme == "int8":
        from .video_minimax_h3_te import load_h3_quantized_text_encoder

        # __wrapped__: this backend owns the encoder's lifetime (the low-host tier frees it after each prompt).
        loader = getattr(
            load_h3_quantized_text_encoder, "__wrapped__", load_h3_quantized_text_encoder
        )
        encoder = loader(
            text_encoder_repo,
            "int8",
            dtype = dtype,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
        )
        if encoder is not None:
            return encoder, "int8"
        if fallback_check is not None:
            fallback_check()
        logger.warning(
            "diffusion.minimax_h3_image: hosted int8 encoder unusable, loading the bf16 encoder"
        )
    import transformers

    from .diffusion_families import pinned_revision_kwargs

    encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        text_encoder_repo,
        subfolder = "text_encoder",
        torch_dtype = dtype,
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        low_cpu_mem_usage = True,
        **pinned_revision_kwargs(text_encoder_repo),
    )
    encoder.eval()
    return encoder, None


def pin_room_bytes(plan: H3ImagePlan, pinned: Sequence[Any] = ()) -> Optional[int]:
    """Host bytes left for pins: host budget less what the plan holds less ``pinned`` streamers' locks. Pins count on
    top of the pageable copies they replace (those may stay cached). None without a host budget."""
    if plan.host_gb is None or plan.host_need_gb is None:
        return None
    room = int((float(plan.host_gb) - float(plan.host_need_gb)) * 1e9)
    room -= sum(int(getattr(s, "pinned_bytes", 0) or 0) for s in pinned if s is not None)
    return max(0, room)


def make_list_streamer(
    owner: Any,
    members: Any,
    device: Any,
    resident: int,
    role: str,
    *,
    pinned: bool = True,
    reserve_bytes: int = 0,
    max_pin_bytes: Optional[int] = None,
) -> Any:
    """A streamer for ``members[resident:]``: the pinned prefetch one where it fits, else ``ModuleListStreamer``."""
    streamer = None
    if pinned and pinned_stream_enabled():
        from .diffusion_offload_prefetch import stream_module_list
        streamer = stream_module_list(
            owner,
            members,
            device,
            resident,
            logger = logger,
            reserve_bytes = reserve_bytes,
            max_pin_bytes = max_pin_bytes,
        )
    if streamer is None:
        streamer = ModuleListStreamer(members, device, resident)
    streamer._unsloth_role = role
    return streamer


def stream_list(
    pipe: Any,
    owner: Any,
    members: Any,
    device: Any,
    resident: int,
    role: str,
    *,
    pinned: bool = True,
    reserve_bytes: int = 0,
    max_pin_bytes: Optional[int] = None,
) -> Any:
    """``make_list_streamer``, registered on the pipe by role."""
    streamer = make_list_streamer(
        owner,
        members,
        device,
        resident,
        role,
        pinned = pinned,
        reserve_bytes = reserve_bytes,
        max_pin_bytes = max_pin_bytes,
    )
    pipe._unsloth_h3_streamers.append(streamer)
    return streamer


def _place_streamed_dit(
    transformer: Any,
    vae: Any,
    device: Any,
    plan: H3ImagePlan,
    pinned_already: Sequence[Any] = (),
) -> Any:
    """Placed before the encoder so the DiT blocks (streamed every step) get the pin budget first."""
    vae.to(device)
    for name, child in transformer.named_children():
        if name != "transformer_blocks":
            child.to(device)
    resident = int(plan.dit_resident_blocks or 0)
    streamer = make_list_streamer(
        transformer,
        transformer.transformer_blocks,
        device,
        resident,
        "transformer",
        max_pin_bytes = pin_room_bytes(plan, pinned_already),
    )
    streamer._unsloth_resident = resident
    return streamer


# "0": native attention; "1": probe-gated SageAttention on the DiT blocks; unset: as the checkpoint declares
INT8_ATTENTION_ENV = "UNSLOTH_H3_IMAGE_INT8_ATTENTION"
CHECKPOINT_ATTENTION_ATTR = "_unsloth_checkpoint_attention"
# the INT8 DiT checkpoint runs INT8 attention on its DiT blocks unless its file says otherwise
SCHEME_ATTENTION = {"int8": "int8"}
ATTENTION_SAVED_ATTR = "_unsloth_h3_attention_saved"
_LOG = logger


def checkpoint_attention_wanted(transformer: Any, scheme: Optional[str]) -> Optional[str]:
    """The DiT blocks' attention backend ("sage" or None) per the env switch or checkpoint."""
    from .diffusion_attention import checkpoint_attention_backend

    raw = (os.environ.get(INT8_ATTENTION_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return None
    if raw in ("1", "on", "true", "yes"):
        return "sage"
    declared = getattr(transformer, CHECKPOINT_ATTENTION_ATTR, None)
    if not declared:
        declared = SCHEME_ATTENTION.get(scheme or "")
    return checkpoint_attention_backend(declared)


def _attention_processors(module: Any) -> list:
    return [
        m for m in module.modules() if hasattr(getattr(m, "processor", None), "_attention_backend")
    ]


def install_checkpoint_attention(
    pipe: Any,
    device: Any,
    *,
    scheme: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Route the DiT blocks to the declared backend (SageAttention must pass its probe here first); the token refiner
    keeps its own. Returns the engaged backend, or None with every processor unchanged."""
    import torch

    log = logger if logger is not None else _LOG
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or getattr(pipe, ATTENTION_SAVED_ATTR, None):
        return None
    want = checkpoint_attention_wanted(transformer, scheme)
    dev = torch.device(device)
    if want is None or dev.type != "cuda":
        return None
    from .diffusion_attention import ATTENTION_BACKEND_ATTR, apply_attention_backend

    modules = _attention_processors(transformer)
    saved = {m: m.processor._attention_backend for m in modules}
    tag = getattr(transformer, ATTENTION_BACKEND_ATTR, None)

    def restore(only: Any = None) -> None:
        for m, backend in saved.items():
            if only is None or m in only:
                m.processor._attention_backend = backend

    dtype = _module_dtype(transformer, skip_fp32 = True)
    target = types.SimpleNamespace(device = dev.type, dtype = dtype, torch_device = str(dev))
    try:
        engaged = apply_attention_backend(pipe, want, logger = log, target = target)
    except Exception as exc:  # noqa: BLE001 - native attention stays
        engaged = None
        log.warning("diffusion.minimax_h3_image: INT8 attention unavailable (%s)", exc)
    if not engaged or "sage" not in str(engaged):
        restore()
        setattr(transformer, ATTENTION_BACKEND_ATTR, tag)
        return None
    blocks = {id(m) for b in getattr(transformer, "transformer_blocks", ()) for m in b.modules()}
    restore(only = [m for m in modules if id(m) not in blocks])
    setattr(pipe, ATTENTION_SAVED_ATTR, (saved, tag))
    log.info(
        "diffusion.minimax_h3_image: INT8 attention (%s) on %d DiT blocks, as the checkpoint declares",
        engaged,
        sum(1 for m in modules if id(m) in blocks),
    )
    return engaged


def remove_checkpoint_attention(pipe: Any) -> int:
    """Every attention processor back to the backend it had before ``install_checkpoint_attention``."""
    entry = getattr(pipe, ATTENTION_SAVED_ATTR, None)
    if not entry:
        return 0
    from .diffusion_attention import ATTENTION_BACKEND_ATTR

    saved, tag = entry
    for m, backend in saved.items():
        m.processor._attention_backend = backend
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        setattr(transformer, ATTENTION_BACKEND_ATTR, tag)
    setattr(pipe, ATTENTION_SAVED_ATTR, None)
    return len(saved)


def _place_text_encoder(
    pipe: Any,
    encoder: Any,
    plan: H3ImagePlan,
    device: Any,
    *,
    per_prompt: bool = False,
    reserve_bytes: int = 0,
    pinned_already: Sequence[Any] = (),
) -> Any:
    """``per_prompt`` (low-host tier, encoder reloaded each prompt) streams pageable: pinning would copy 27 GB per
    prompt. ``pinned_already``: streamers whose pins count against the host room first."""
    if encoder is None:
        return None
    if plan.placement == PLACE_RESIDENT:
        encoder.to(device)
    elif plan.placement in (PLACE_STREAM, PLACE_DIT_RESIDENT):
        layers = encoder.model.language_model.layers
        for name, child in encoder.model.named_children():
            if name != "language_model":
                child.to(device)
        for name, child in encoder.model.language_model.named_children():
            if name != "layers":
                child.to(device)
        stream_list(
            pipe,
            encoder.model.language_model,
            layers,
            device,
            int(plan.te_resident_layers or 0),
            "text_encoder",
            pinned = not per_prompt,
            reserve_bytes = reserve_bytes,
            max_pin_bytes = pin_room_bytes(plan, pinned_already),
        )
    return encoder


def load_minimax_h3_image_pipeline(
    repo_id: str = H3_IMAGE_BASE_REPO,
    *,
    dtype: Any = None,
    device: Any = "cuda",
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = "int8",
    text_encoder_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
    text_encoder: Any = None,
    transformer: Any = None,
    plan_width: int = 2048,
    plan_height: int = 2048,
) -> Any:
    """A ready ``MiniMaxH3ImagePipeline`` for ``repo_id`` (Hub id or local folder) placed for ``device``.

    ``transformer_quant``: None/"auto" (bf16 when it fits, else int8), "int8", "fp8", "bf16". ``text_encoder_quant``:
    "int8" (Studio's hosted ConvRot conditioner) or None (bf16). ``text_encoder`` / ``transformer`` hand in prebuilt
    components (tests, a hosted prequant DiT)."""
    import torch

    t0 = time.perf_counter()
    # What release() may empty in place: never a component the caller handed in.
    owned = (
        {"vae"}
        | ({"transformer"} if transformer is None else set())
        | ({"text_encoder"} if text_encoder is None else set())
    )
    check = check_cancelled or (lambda: None)
    dtype = dtype or torch.bfloat16
    device = torch.device(device)
    vram, host = device_budgets(device)
    patterns = [
        "model_index.json",
        PDD_CONFIG_FILE,
        "transformer/config.json",
        "vae/*",
        "tokenizer/*",
        "processor/*",
        f"{DIFFUSERS_TRANSFORMER_SUBFOLDER}/config.json",
        f"{DIFFUSERS_VAE_SUBFOLDER}/*",
    ]
    hub = {"hf_token": hf_token, "cache_dir": cache_dir, "local_files_only": local_files_only}
    root = _hub_files(repo_id, patterns, **hub)
    check()
    index = _read_json(root / "model_index.json") if (root / "model_index.json").is_file() else {}
    te_entry = index.get("text_encoder")
    te_repo = text_encoder_repo
    if (
        te_repo is None
        and isinstance(te_entry, list)
        and len(te_entry) > 2
        and isinstance(te_entry[2], dict)
    ):
        te_repo = te_entry[2].get("pretrained_model_name_or_path")
    te_repo = te_repo or H3_IMAGE_TEXT_ENCODER_REPO
    scheme, scheme_reason = resolve_dit_scheme(
        transformer_quant, device = device, vram_gb = vram, width = plan_width, height = plan_height
    )
    te_scheme = "int8" if (text_encoder_quant or "").strip().lower() == "int8" else None
    table = adaln_table_enabled()
    plan_args = {
        "vram_gb": vram if device.type == "cuda" else None,
        "host_gb": host,
        "dit_gb": dit_gb_for(scheme, table),
        "width": plan_width,
        "height": plan_height,
        "forced": (os.environ.get(PLACEMENT_ENV) or "").strip().lower() or None,
        "inflight": stream_inflight() if device.type == "cuda" else 1,
    }
    plan = plan_placement(
        te_gb = te_gb_for(te_scheme) if text_encoder is None else _module_gb(text_encoder),
        **plan_args,
    )
    logger.info(
        "diffusion.minimax_h3_image: dit=%s (%s) te=%s placement=%s reasons=%s",
        scheme or "bf16",
        scheme_reason,
        te_scheme or "bf16",
        plan.placement,
        "; ".join(plan.reasons),
    )
    stats: dict = {"layout": "supplied", "quantized_linears": 0, "bytes_read": 0}
    if transformer is None and scheme in QUANT_SCHEMES and not os.path.isdir(str(repo_id)):
        transformer = load_hosted_prequant_transformer(
            scheme,
            base = repo_id,
            device = device,
            dtype = dtype,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            keep_on_device = plan.placement in (PLACE_RESIDENT, PLACE_DIT_RESIDENT),
        )
        if transformer is not None:
            for block in transformer.transformer_blocks:
                wrap_row_multiple(block, QUANT_LINEAR_SUFFIXES)
            densified = densify_conditioning_path(transformer, dtype)
            logger.info(
                "diffusion.minimax_h3_image: hosted DiT, text-conditioning path kept dense (%d Linears)",
                densified,
            )
            stats = {
                "layout": "hosted_prequant",
                "quantized_linears": 0,
                "bytes_read": 0,
                "dense_conditioning_linears": densified,
            }
    if transformer is None:
        root = _hub_files(
            repo_id,
            [PDD_HEADS_FILE, "transformer/*", f"{DIFFUSERS_TRANSFORMER_SUBFOLDER}/*"],
            **hub,
        )
        transformer, stats = load_h3_image_transformer(
            root,
            dtype = dtype,
            scheme = scheme,
            device = device,
            keep_on_device = plan.placement in (PLACE_RESIDENT, PLACE_DIT_RESIDENT),
            table = table,
            check_cancelled = check,
        )
    check()
    vae = load_h3_image_vae(root)
    check()
    early_streamer = None
    if plan.placement == PLACE_STREAM:
        # Place the streamed tier before any encoder is materialised: the host-memory plan counts only the DiT
        # blocks left on the host, not the whole DiT + VAE the loaders park there.
        early_streamer = _place_streamed_dit(transformer, vae, device, plan)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    processor = None
    if (root / "processor").is_dir():
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(root / "processor"))
        except Exception as exc:  # noqa: BLE001 - t2i still works; editing reports the missing processor
            logger.warning(
                "diffusion.minimax_h3_image: processor unavailable, editing disabled (%s)", exc
            )
    te_kwargs = {
        "dtype": dtype,
        "text_encoder_repo": te_repo,
        "hf_token": hf_token,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    engaged_te = te_scheme if text_encoder is None else "supplied"

    def _encoder_fits(te_gb: float, what: str) -> None:
        again = replan_for_encoder(plan, te_gb, plan_args)
        cpu_short = (
            plan.vram_gb is None
            and plan.host_gb is not None
            and plan.host_gb < plan.dit_gb + te_gb + 1.0
        )
        if cpu_short or not again.te_kept_on_host:
            raise RuntimeError(
                f"MiniMax-H3-Image: {what} ({te_gb:.1f} GB) does not fit this machine's "
                f"{plan.host_gb:.1f} GB of free system memory beside the model."
            )

    local_te = root / "text_encoder"
    if text_encoder is None and (local_te / "config.json").is_file():
        local_gb = sum(f.stat().st_size for f in local_te.glob("*.safetensors")) / 1e9
        if local_gb > plan.te_gb + 1.0:
            _encoder_fits(local_gb, "the text encoder in this model folder")
        import transformers

        text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            str(root / "text_encoder"), torch_dtype = dtype, low_cpu_mem_usage = True
        ).eval()
        engaged_te = None
    if text_encoder is None and plan.te_kept_on_host:

        def _bf16_fits() -> None:
            _encoder_fits(
                H3_IMAGE_TE_BF16_GB,
                "Studio's hosted int8 text encoder could not be loaded (check the network or the Hugging Face "
                "cache), and the bf16 encoder it would fall back to",
            )

        text_encoder, engaged_te = load_h3_image_text_encoder(
            te_scheme, fallback_check = _bf16_fits, **te_kwargs
        )
    check()
    if text_encoder is not None:
        plan = replan_for_encoder(plan, _module_gb(text_encoder), plan_args)
    pipe = MiniMaxH3ImagePipeline(
        transformer = transformer,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        scheduler = MiniMaxH3ImageScheduler(sample_sigmas = list(H3_IMAGE_SIGMAS)),
        processor = processor,
    )
    pipe._unsloth_h3_plan = plan
    pipe._unsloth_h3_device = str(device)
    pipe._unsloth_h3_owned = frozenset(owned)
    if text_encoder is None:

        def _refuse_bf16() -> None:
            raise RuntimeError(
                "MiniMax-H3-Image: Studio's hosted int8 text encoder could not be loaded, and the 66.7 GB bf16 "
                "encoder it would fall back to does not fit this machine's memory plan (it loads the encoder "
                "for each prompt because host memory is short). Check the network or the Hugging Face cache "
                "and load the model again."
            )

        def _te_loader() -> Any:
            encoder, engaged = load_h3_image_text_encoder(
                te_scheme, fallback_check = _refuse_bf16, **te_kwargs
            )
            if te_scheme is not None and engaged != te_scheme:
                # Placed under this plan, the bf16 fallback would overrun the budget the plan made for the int8 one.
                free_module_storage(encoder)
                raise RuntimeError(
                    "MiniMax-H3-Image: Studio's hosted int8 text encoder could not be loaded, and the 66.7 GB bf16 "
                    "encoder it would fall back to does not fit this machine's memory plan (it loads the encoder "
                    "for each prompt because host memory is short). Check the network or the Hugging Face cache "
                    "and load the model again."
                )
            return _place_text_encoder(pipe, encoder, plan, device, per_prompt = True)

        pipe._unsloth_h3_te_loader = _te_loader
    else:
        _place_text_encoder(pipe, text_encoder, plan, device, pinned_already = [early_streamer])
    if plan.placement != PLACE_SEQUENTIAL:
        vae.to(device)
    if plan.placement in (PLACE_RESIDENT, PLACE_DIT_RESIDENT):
        transformer.to(device)
    elif plan.placement == PLACE_SEQUENTIAL:
        transformer.to("cpu")
    elif plan.placement == PLACE_STREAM:
        resident = int(plan.dit_resident_blocks or 0)
        if early_streamer is None or early_streamer._unsloth_resident != resident:
            if early_streamer is not None:
                early_streamer.remove()
            early_streamer = _place_streamed_dit(
                transformer, vae, device, plan, pinned_already = pipe._unsloth_h3_streamers
            )
        pipe._unsloth_h3_streamers.append(early_streamer)
        early_streamer = None
    if early_streamer is not None:
        early_streamer.remove()
    from .diffusion_minimax_h3_image_speed import install_h3_image_speed

    speed_levers = install_h3_image_speed(pipe, plan, device, logger)
    int8_attention = install_checkpoint_attention(pipe, device, scheme = scheme, logger = logger)
    if int8_attention:
        speed_levers = tuple(speed_levers) + (f"checkpoint_attention={int8_attention}",)
    report = H3ImageLoadReport(
        scheme = scheme,
        scheme_reason = scheme_reason,
        te_scheme = engaged_te,
        plan = plan,
        classes = classes_source(),
        layout = stats.get("layout", "native"),
        table = table and stats.get("layout") not in ("supplied", "hosted_prequant"),
        load_s = time.perf_counter() - t0,
        quantized_linears = int(stats.get("quantized_linears", 0)),
        bytes_read = int(stats.get("bytes_read", 0)),
        speed_levers = speed_levers,
    )
    pipe._unsloth_h3_report = report
    try:
        transformer._unsloth_runtime_quant = scheme
    except Exception:  # noqa: BLE001 - marker only
        pass
    logger.info("diffusion.minimax_h3_image: loaded %s", report.as_dict())
    return pipe


def _module_gb(module: Any) -> float:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total / 1e9
