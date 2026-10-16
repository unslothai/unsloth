# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 on diffusers' LTX-2 pipelines.

LTX-2.5 keeps the LTX-2.3 DiT layout (minus the video FFN bias, plus a keyframe position embedding) and the 2.3 conv
VAE, but conditions on a fine-tuned Gemma 4 12B (``Gemma4UnifiedForConditionalGeneration``) that only transformers
>= 5.10.1 provides. Studio's server runs an older transformers, so the encoder runs in a spawned worker on the
``.venv_t5_510`` sidecar (``video_ltx25_te_worker``) and hands back the packed prompt embeddings; ``pipe.encode_prompt``
is replaced by a call into that worker, and everything after it (connectors, DiT, VAE, vocoder) runs here.

Generation follows the reference two-stage distilled recipe: stage 1 at half resolution on the 8 distilled sigmas,
the latent x2 spatial upsampler, then stage 2 at full resolution on 0.85 / 0.725 / 0.421875 with the stage-1 audio
latents carried over, CFG 1 and every other guidance off, sampled euler-ancestral. Image-to-video conditions the first
latent frame through ``LTX2ConditionPipeline`` at CRF 18, strength 0.7 in stage 1 and 1.0 in stage 2. The dev (full)
DiT runs the pipeline's own single-stage SFT recipe instead.

Everything here is gated on the ``ltx-2.5`` family; no other family reaches this module.
"""

from __future__ import annotations

import os
import queue as _queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Reference recipe. Stage 1 is diffusers' DISTILLED_SIGMA_VALUES; stage 2 is the reference workflow's tail (diffusers'
# own STAGE_2_DISTILLED_SIGMA_VALUES starts at 0.909375 instead).
LTX25_STAGE1_SIGMAS: tuple[float, ...] = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
)
LTX25_STAGE2_SIGMAS: tuple[float, ...] = (0.85, 0.725, 0.421875)
LTX25_IMAGE_CRF = 18
LTX25_STAGE1_IMAGE_STRENGTH = 0.7
LTX25_STAGE2_IMAGE_STRENGTH = 1.0
# Dev (full) DiT: the README's override of the distilled scheduler config.
LTX25_DEV_SCHEDULER_OVERRIDES: dict[str, Any] = {
    "use_dynamic_shifting": True,
    "shift_terminal": 0.1,
}

# Kill switches: "0" runs one full-resolution stage on the distilled sigmas; "euler" swaps the ancestral sampler for
# diffusers' deterministic Euler.
TWO_STAGE_ENV = "UNSLOTH_LTX25_TWO_STAGE"
SAMPLER_ENV = "UNSLOTH_LTX25_SAMPLER"
_OFF = ("0", "false", "no", "off")

# Pack components a load never reads: the dev DiT (single-file dev picks bring their own), the prompt enhancer and its
# processor (Studio never enhances), the diffusion decoder (conv decode is the pipeline default), the temporal
# upsampler (DFR only) and the duration head (Studio always passes num_frames).
_PACK_SKIP_PREFIXES = (
    "transformer_full/",
    "prompt_enhancer/",
    "processor/",
    "diffusion_decoder/",
    "temporal_latent_upsampler/",
    "duration_head/",
)
_PACK_MARKERS = ("duration_head/config.json", "diffusion_decoder/config.json")
# Components handed to from_pretrained as None: the Gemma 4 encoder lives in the worker, the rest is unused.
LTX25_NONE_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "prompt_enhancer",
    "duration_head",
)
LATENT_UPSAMPLER_SUBFOLDER = "latent_upsampler"
TEXT_ENCODER_MIN_TRANSFORMERS = "5.10.1"
# What the encoder worker holds on the card beyond its weights, on every plan tier: its CUDA context (~0.6 GB on
# Hopper / Blackwell) plus allocator slack. The weights are the family table's text-encoder term.
LTX25_WORKER_RESIDENT_GB = 1.0
# int8 encoder is opt-in: LPIPS 0.310 vs bf16, above the 0.256 compile floor.
LTX25_TE_QUANT_DEFAULT = "int8"
LTX25_TE_QUANT_ENV = "UNSLOTH_LTX25_TE_INT8"
# Measured int8 encoder weights (12,424 MiB vs 22,811 MiB bf16).
LTX25_TE_INT8_GB = 13.1
# The pack's latent_upsampler/ (995,743,320 bytes), part of the family table's companions; only two-stage loads hold it.
LTX25_LATENT_UPSAMPLER_GB = 1.0

# The split single-file release. A DiT picked from it assembles from the companion files beside it (no Diffusers pack):
# connectors from the DiT file plus the encoder file's text projections, the conv VAE, the audio VAE + vocoder and the
# spatial upscaler. The encoder file embeds its own config and tokenizer.
LTX25_SINGLE_FILE_REPO = "Lightricks/LTX-2.5"
LTX25_SINGLE_TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
LTX25_SINGLE_VIDEO_VAE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
LTX25_SINGLE_AUDIO_VAE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
LTX25_SINGLE_UPSAMPLER = (
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
LTX25_SINGLE_COMPANIONS = (
    LTX25_SINGLE_TEXT_ENCODER,
    LTX25_SINGLE_VIDEO_VAE,
    LTX25_SINGLE_AUDIO_VAE,
    LTX25_SINGLE_UPSAMPLER,
)
# Encoder file keys -> Gemma4UnifiedForConditionalGeneration keys (transformers ``key_mapping`` regexes).
LTX25_TEXT_ENCODER_KEY_MAPPING = {
    r"^model\.(layers|embed_tokens|norm)\b": r"model.language_model.\1",
    r"^audio_projector\.": "model.embed_audio.",
    r"^multi_modal_projector\.": "model.embed_vision.",
    r"^vision_model\.": "model.vision_embedder.",
}
LTX25_TEXT_PROJECTION_PREFIX = "text_embedding_projection."

# Lightricks/LTX-2.5-Diffusers transformer/config.json, latent_upsampler/config.json and scheduler config, for
# assembling without the pack. The connectors, conv VAE, audio VAE and vocoder configs equal the 2.3 ones in
# video_ltx2 (checked against the pack).
LTX25_TRANSFORMER_CONFIG: dict[str, Any] = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_out_bias": True,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "audio_cross_attn_mod": True,
    "audio_ff_bias": True,
    "audio_gated_attn": True,
    "audio_hop_length": 160,
    "audio_in_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_out_channels": 128,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_scale_factor": 4,
    "base_height": 2048,
    "base_width": 2048,
    "caption_channels": 3840,
    "causal_offset": 1,
    "cross_attention_dim": 4096,
    "cross_attn_mod": True,
    "cross_attn_timestep_scale_multiplier": 1000,
    "ff_bias": False,
    "gated_attn": True,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 48,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "perturbed_attn": True,
    "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads",
    "rope_double_precision": True,
    "rope_theta": 10000.0,
    "rope_type": "split",
    "timestep_scale_multiplier": 1000,
    "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True,
    "use_prompt_embeddings": False,
    "vae_scale_factors": [8, 32, 32],
}
LTX25_UPSAMPLER_CONFIG: dict[str, Any] = {
    "dims": 3,
    "in_channels": 128,
    "mid_channels": 1024,
    "num_blocks_per_stage": 4,
    "rational_spatial_scale": 2.0,
    "spatial_upsample": True,
    "temporal_upsample": False,
    "use_rational_resampler": False,
}
LTX25_SCHEDULER_CONFIG: dict[str, Any] = {
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def two_stage_enabled() -> bool:
    return (os.environ.get(TWO_STAGE_ENV) or "").strip().lower() not in _OFF


def ancestral_enabled() -> bool:
    return (os.environ.get(SAMPLER_ENV) or "").strip().lower() != "euler"


# ── image-to-video canvas ────────────────────────────────────────────────────

# "Match source" canvas: the default 1280x704 area on a 704 px short edge, 64 px units (two 32 px stages). Same shape
# of rule as MiniMax-H3's, so the frontend's preview (canvas_short_edge / canvas_max_pixels) applies unchanged.
LTX25_CANVAS_SHORT_EDGE = 704
LTX25_CANVAS_MAX_PIXELS = 1280 * 704
LTX25_CANVAS_MULTIPLE = 64
_MIN_ASPECT, _MAX_ASPECT = 0.25, 4.0


def ltx25_canvas_for_aspect(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    import math

    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The source image has no usable aspect ratio ({aspect_width}x{aspect_height})."
        )
    ratio = aspect_width / aspect_height
    if not _MIN_ASPECT <= ratio <= _MAX_ASPECT:
        raise ValueError(
            f"LTX-2.5 image-to-video takes aspect ratios from 1:4 to 4:1; this image is "
            f"{aspect_width:g}x{aspect_height:g} ({ratio:.2f}:1). Crop it first or pick a resolution preset."
        )
    short = float(LTX25_CANVAS_SHORT_EDGE)
    width, height = (short * ratio, short) if ratio >= 1.0 else (short, short / ratio)
    area = width * height
    if area > LTX25_CANVAS_MAX_PIXELS:
        scale = math.sqrt(LTX25_CANVAS_MAX_PIXELS / area)
        width, height = width * scale, height * scale

    from .video_families import snap_canvas_within_cap

    return snap_canvas_within_cap(width, height, LTX25_CANVAS_MULTIPLE, LTX25_CANVAS_MAX_PIXELS)


# The reference image-to-video graph scales the start image's longer side to 1536 px (Lanczos) before the CRF 18 round
# trip; LTX2ConditionPipeline then applies the CRF and fits the result to each stage's canvas (bilinear cover + centre
# crop), in that order.
LTX25_FIRST_FRAME_LONG_EDGE = 1536


def prepare_ltx25_first_frame(image: Any) -> Any:
    """The start image as the condition pipeline should receive it: longer side at ``LTX25_FIRST_FRAME_LONG_EDGE``,
    aspect kept, not yet cropped to the canvas."""
    from PIL import Image

    source_w, source_h = image.size
    scale = LTX25_FIRST_FRAME_LONG_EDGE / max(source_w, source_h)
    size = (max(1, round(source_w * scale)), max(1, round(source_h * scale)))
    return image if size == image.size else image.resize(size, Image.LANCZOS)


# ── base-pack file selection ─────────────────────────────────────────────────


def is_ltx25_pack(names: Any) -> bool:
    """Whether a Diffusers repo listing is an LTX-2.5 pack (it ships a duration head / diffusion decoder)."""
    return any(name in _PACK_MARKERS for name in names)


def _component(name: str) -> str:
    return name.split("/", 1)[0] if "/" in name else ""


def _shard_set(name: str) -> Optional[str]:
    """``-of-00008`` style shard-set key of a sharded weight file, or None."""
    leaf = name.rsplit("/", 1)[-1]
    if "-of-" not in leaf:
        return None
    return leaf.rsplit("-of-", 1)[1].split(".", 1)[0]


def ltx25_pack_files(
    files: list[tuple[str, int]],
    shards: Optional[dict[str, frozenset[str]]] = None,
    dev: bool = False,
) -> list[tuple[str, int]]:
    """The files of an LTX-2.5 pack a load reads.

    Drops the unused components above, and the duplicate weight sets the pack ships: ``transformer/`` carries both a
    4-shard and an 8-shard set and ``connectors/`` a sharded and an unsharded copy. With ``shards`` (component ->
    file names its ``*.index.json`` lists) only the indexed set stays. Without it (an index that could not be read)
    the unsharded copy goes when an index exists and the coarsest shard set is kept, the one the official indexes
    name. A ``dev`` load (single stage) also drops the latent upsampler, which only the two-stage recipe attaches."""
    names = {name for name, _ in files}
    kept: list[tuple[str, int]] = []
    by_component: dict[str, dict[str, list[tuple[str, int]]]] = {}
    skip = _PACK_SKIP_PREFIXES
    if not ltx25_uses_upsampler(dev):
        skip = skip + (f"{LATENT_UPSAMPLER_SUBFOLDER}/",)
    for name, size in files:
        if name.startswith(skip):
            continue
        component = _component(name)
        leaf = name.rsplit("/", 1)[-1]
        weight = leaf.endswith(".safetensors")
        if component and weight:
            index_name = None
            for candidate in names:
                if candidate.startswith(component + "/") and candidate.endswith(
                    ".safetensors.index.json"
                ):
                    index_name = candidate
                    break
            if index_name is not None:
                if shards is not None and component in shards:
                    if leaf in shards[component]:
                        kept.append((name, size))
                    continue
                key = _shard_set(name)
                if key is None:
                    # an unsharded duplicate of an indexed component
                    continue
                by_component.setdefault(component, {}).setdefault(key, []).append((name, size))
                continue
        kept.append((name, size))
    for sets in by_component.values():
        # No index read: one set, the coarsest split (the official pack indexes its 4-shard transformer, not the 8).
        best = min(sets.items(), key = lambda item: (len(item[1]), item[0]))[1]
        kept.extend(best)
    order = {name: i for i, (name, _) in enumerate(files)}
    return sorted(kept, key = lambda item: order.get(item[0], 0))


def ltx25_shards_from_indexes(indexes: dict[str, Any]) -> dict[str, frozenset[str]]:
    """component -> the shard file names its parsed ``*.index.json`` (``{path: json}``) maps weights to."""
    out: dict[str, frozenset[str]] = {}
    for path, payload in indexes.items():
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict):
            continue
        out[_component(path)] = frozenset(str(v) for v in weight_map.values())
    return out


# ── transformer from a single file / GGUF ────────────────────────────────────


def load_ltx25_transformer(
    checkpoint_path: Path | str,
    *,
    base_repo: str,
    torch_dtype: Any,
    is_gguf: bool,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    device: Optional[Any] = None,
) -> Any:
    """The LTX-2.5 DiT of a split single file or GGUF. Only the DiT keys are read; the connectors in the file are the
    pack's shared ``connectors/`` (loaded from the base like the VAEs). The base's ``transformer/config.json`` already
    carries the 2.5 deltas (``ff_bias=False``, ``use_keyframes_abs_pos_embedding=True``), so no overrides; the keys
    take the same ``prompt_adaln_single`` pre-rename the 2.3 assembly applies."""
    import diffusers
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    from .video_ltx2 import _ltx23_pre_convert, ltx23_is_dit_key

    state = load_single_file_checkpoint(str(checkpoint_path))
    dit_state = _ltx23_pre_convert({k: v for k, v in state.items() if ltx23_is_dit_key(k)})
    del state
    if any(k.endswith((".weight_scale", ".input_scale")) for k in dit_state):
        raise ValueError(
            "This LTX-2.5 checkpoint stores scaled fp8 weights, which this loader does not dequantize. Use the bf16 "
            "file, the int8-convrot file or a GGUF quant instead."
        )
    kwargs: dict[str, Any] = {
        "config": _transformer_config_source(base_repo),
        "subfolder": "transformer",
        "torch_dtype": torch_dtype,
        "token": hf_token,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    from .diffusion_families import pinned_revision_kwargs

    # The load plan's commit for a hub-id config (none for the bundled config dir or an unpinned load).
    kwargs.update(pinned_revision_kwargs(kwargs["config"]))
    if is_gguf:
        kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(compute_dtype = torch_dtype)
    elif device is not None:
        kwargs["device"] = device
    return LTX2VideoTransformer3DModel.from_single_file(dit_state, **kwargs)


def load_ltx25_comfy_transformer(
    checkpoint_path: Path | str,
    scan: Any,
    *,
    base_repo: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    int8_backend: Optional[str] = None,
    fp8_backend: Optional[str] = None,
    family: Optional[str] = None,
    target: Any = None,
    logger: Any = None,
) -> Any:
    """The DiT of an int8-convrot LTX-2.5 file (``...-int8-convrot``), keeping int8 / fp8 codes where
    ``load_comfy_quant_transformer`` can; same key handling as ``load_ltx25_transformer``."""
    from diffusers import LTX2VideoTransformer3DModel

    from .diffusion_comfy_quant import load_comfy_quant_transformer
    from .video_ltx2 import _ltx23_pre_convert, ltx23_is_dit_key

    return load_comfy_quant_transformer(
        LTX2VideoTransformer3DModel,
        str(checkpoint_path),
        scan,
        {
            "torch_dtype": torch_dtype,
            "config": _transformer_config_source(base_repo),
            "subfolder": "transformer",
            "token": hf_token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        },
        int8_backend = int8_backend,
        fp8_backend = fp8_backend,
        family = family,
        target = target,
        logger = logger,
        keep_key = ltx23_is_dit_key,
        pre_convert = _ltx23_pre_convert,
    )


def load_latent_upsampler(
    base_repo: str,
    *,
    torch_dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Any:
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    if is_ltx25_single_file_base(base_repo):
        from safetensors.torch import load_file

        from .video_ltx2 import _build_from_config

        path = ltx25_companion_path(base_repo, LTX25_SINGLE_UPSAMPLER, hf_token, local_files_only)
        return _build_from_config(
            LTX2LatentUpsamplerModel, LTX25_UPSAMPLER_CONFIG, load_file(path), {}, torch_dtype
        )
    from .diffusion_families import pinned_revision_kwargs

    return LTX2LatentUpsamplerModel.from_pretrained(
        base_repo,
        **pinned_revision_kwargs(base_repo),
        subfolder = LATENT_UPSAMPLER_SUBFOLDER,
        torch_dtype = torch_dtype,
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )


# ── the split single-file layout as its own base ────────────────────────────


def is_ltx25_single_file_base(base: Optional[str], *, fold_mirror: bool = True) -> bool:
    """Whether ``base`` is the split single-file release (hub id) or a local folder with its layout (the encoder file
    and the conv VAE beside the DiTs), rather than a Diffusers pack."""
    text = (base or "").strip()
    if not text:
        return False
    path = Path(text).expanduser()
    if path.is_dir():
        return (path / LTX25_SINGLE_TEXT_ENCODER).is_file() and (
            path / LTX25_SINGLE_VIDEO_VAE
        ).is_file()
    if not fold_mirror:
        return text.lower() == LTX25_SINGLE_FILE_REPO.lower()
    # The fetch sites hand the assembly the repo carrying the release's byte-identical files (unsloth/LTX-2.5-GGUF).
    # ``fold_mirror=False`` is for a PICK's own repo: a GGUF picked from that repo keeps the Diffusers base.
    from .diffusion_families import canonical_base

    return canonical_base(text).lower() == LTX25_SINGLE_FILE_REPO.lower()


def is_ltx25_single_file_listing(names: Any) -> bool:
    """Whether a repo listing is the split single-file release (it carries every companion file)."""
    present = set(names)
    return all(name in present for name in LTX25_SINGLE_COMPANIONS)


def ltx25_uses_upsampler(dev: bool) -> bool:
    """Whether a load attaches the latent upsampler: only the two-stage distilled recipe runs it."""
    return not dev and two_stage_enabled()


def ltx25_single_file_companions(
    files: list[tuple[str, int]], dev: bool = False
) -> list[tuple[str, int]]:
    """The (rfilename, size) entries a load from the single-file release reads besides the picked DiT."""
    wanted = set(LTX25_SINGLE_COMPANIONS)
    if not ltx25_uses_upsampler(dev):
        wanted.discard(LTX25_SINGLE_UPSAMPLER)
    return [(name, size) for name, size in files if name in wanted]


def ltx25_default_base(repo_id: Optional[str]) -> Optional[str]:
    """A DiT picked from the single-file release (or a local copy of it) uses that release as its base."""
    return repo_id if is_ltx25_single_file_base(repo_id, fold_mirror = False) else None


def ltx25_companion_path(
    base: str,
    filename: str,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> str:
    """Local path of one companion file of a single-file base (downloaded into the hub cache when needed)."""
    path = Path(base).expanduser()
    if path.is_dir():
        target = path / filename
        if not target.is_file():
            raise FileNotFoundError(
                f"LTX-2.5 needs '{filename}' beside the DiT in '{base}', and it is missing."
            )
        return str(target)
    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    from .diffusion_families import pinned_revision_kwargs

    return hf_hub_download_with_xet_fallback(
        base,
        filename,
        hf_token,
        **pinned_revision_kwargs(base),
        reuse_other_cache_root = True,
        local_files_only = local_files_only,
    )


def ltx25_transformer_config_dir() -> str:
    """A local folder holding ``transformer/config.json`` (the pack's 2.5 transformer config), for
    ``from_single_file(config=...)`` when there is no pack to read it from."""
    import json
    import tempfile

    root = Path(tempfile.gettempdir()) / "unsloth_ltx25_config"
    target = root / "transformer" / "config.json"
    payload = json.dumps(
        {"_class_name": "LTX2VideoTransformer3DModel", **LTX25_TRANSFORMER_CONFIG},
        indent = 1,
        sort_keys = True,
    )
    if not target.is_file() or target.read_text(encoding = "utf-8") != payload:
        target.parent.mkdir(parents = True, exist_ok = True)
        # A per-call scratch name: two loads in one process (a superseded one still building) may both get here.
        fd, scratch = tempfile.mkstemp(dir = target.parent, prefix = "config.", suffix = ".tmp")
        try:
            with os.fdopen(fd, "w", encoding = "utf-8") as fh:
                fh.write(payload)
            os.replace(scratch, target)
        except BaseException:
            Path(scratch).unlink(missing_ok = True)
            raise
    return str(root)


def _transformer_config_source(base_repo: str) -> str:
    return ltx25_transformer_config_dir() if is_ltx25_single_file_base(base_repo) else base_repo


def _read_tensors(path: str, keep: Callable[[str], bool]) -> dict[str, Any]:
    """The tensors of ``path`` whose names pass ``keep``, read selectively (GGUF files are read whole)."""
    if not str(path).lower().endswith(".safetensors"):
        from diffusers.loaders.single_file_utils import load_single_file_checkpoint
        return {k: v for k, v in load_single_file_checkpoint(str(path)).items() if keep(k)}
    from safetensors import safe_open

    with safe_open(str(path), framework = "pt", device = "cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys() if keep(key)}


def assemble_ltx25_single_file_pipeline(
    transformer: Any,
    checkpoint_path: Path | str,
    *,
    base: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> Any:
    """``LTX2Pipeline`` from the split single-file release: the given DiT, the connectors of the DiT file plus the
    encoder file's text projections, the conv VAE and the audio VAE + vocoder. Text encoder and tokenizer stay None
    (the worker owns them); the upsampler is loaded by ``attach_to_pipeline``."""
    from .ltx2_import_compat import ensure_ltx2_pipelines_importable

    ensure_ltx2_pipelines_importable(logger)
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline
    from safetensors.torch import load_file

    from .video_ltx2 import (
        _CONNECTORS_CONFIG,
        _CONNECTORS_RENAME,
        _build_from_config,
        _checkpoint_group,
        load_ltx23_audio_vae_and_vocoder,
        load_ltx23_vae,
    )
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    connector_state = {
        _checkpoint_group(key)[1]: value
        for key, value in _read_tensors(
            str(checkpoint_path), lambda key: _checkpoint_group(key)[0] == "connectors"
        ).items()
    }
    if not connector_state:
        raise ValueError(f"'{Path(str(checkpoint_path)).name}' carries no LTX-2.5 text connectors.")
    if any(
        name.endswith((".weight_scale", ".input_scale", ".comfy_quant")) for name in connector_state
    ):
        raise ValueError(
            "This LTX-2.5 file stores its text connectors quantized, which the single-file assembly does not "
            "dequantize. Load it on the Lightricks/LTX-2.5-Diffusers base (base_repo), or pick the bf16 DiT or a GGUF "
            "quant."
        )
    encoder_path = ltx25_companion_path(base, LTX25_SINGLE_TEXT_ENCODER, hf_token, local_files_only)
    connector_state.update(
        _read_tensors(encoder_path, lambda key: key.startswith(LTX25_TEXT_PROJECTION_PREFIX))
    )
    connectors = _build_from_config(
        LTX2TextConnectors, _CONNECTORS_CONFIG, connector_state, _CONNECTORS_RENAME, torch_dtype
    )
    vae = load_ltx23_vae(
        load_file(ltx25_companion_path(base, LTX25_SINGLE_VIDEO_VAE, hf_token, local_files_only)),
        variant = "distilled",
        torch_dtype = torch_dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
    )
    combined = load_file(
        ltx25_companion_path(base, LTX25_SINGLE_AUDIO_VAE, hf_token, local_files_only)
    )
    audio_vae, vocoder = load_ltx23_audio_vae_and_vocoder(
        {k[len("audio_vae.") :]: v for k, v in combined.items() if k.startswith("audio_vae.")},
        {k[len("vocoder.") :]: v for k, v in combined.items() if k.startswith("vocoder.")},
        variant = "distilled",
        torch_dtype = torch_dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
    )
    del combined
    logger.info("video.ltx25_single_file: assembled from %s", base)
    return LTX2Pipeline(
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(LTX25_SCHEDULER_CONFIG),
        vae = vae,
        audio_vae = audio_vae,
        text_encoder = None,
        tokenizer = None,
        connectors = connectors,
        transformer = transformer,
        vocoder = vocoder,
    )


# ── the Gemma 4 encoder worker ───────────────────────────────────────────────


def encoder_resident_for_plan(plan: Any) -> bool:
    """Whether the encoder worker keeps Gemma 4 on the card, by the tier the memory plan budgeted.

    The plan prices the family's text encoder (the table's 23.9 GB) as if it were in-process: resident under ``none``
    and under ``group`` with resident companions, streamed under ``group`` with streamed text encoders and under
    ``streaming`` / ``sequential``, alone on the card under ``model``. The worker cannot stream, so on every tier that did
    not budget it resident it keeps the encoder on the host and onloads it only for an encode, when the card has room
    (else it encodes on the CPU; see ``video_ltx25_te_worker._encode_device``)."""
    from .diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE

    policy = getattr(plan, "offload_policy", None)
    if policy == OFFLOAD_NONE:
        return True
    return policy == OFFLOAD_GROUP and not bool(getattr(plan, "stream_text_encoders", False))


def ltx25_te_quant_scheme(text_encoder_quant: Optional[str]) -> Optional[str]:
    """``"int8"`` or None (bf16); any other scheme raises ValueError (the worker has only the ConvRot INT8 build)."""
    from .diffusion_precision import normalize_te_quant, te_quant_is_auto

    if te_quant_is_auto(text_encoder_quant):
        if (os.environ.get(LTX25_TE_QUANT_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
            return LTX25_TE_QUANT_DEFAULT
        return None
    scheme = normalize_te_quant(text_encoder_quant)
    if scheme is None or scheme == LTX25_TE_QUANT_DEFAULT:
        return scheme
    raise ValueError(
        f"LTX-2.5's Gemma 4 text encoder runs int8 (ConvRot, as in ComfyUI's LTX-2.5 workflows) or bf16, not "
        f"'{scheme}'. Pick int8 for int8, or leave it unset / Off for bf16."
    )


def ltx25_te_budget_scale(text_encoder_quant: Optional[str], bf16_gb: float) -> float:
    try:
        scheme = ltx25_te_quant_scheme(text_encoder_quant)
    except ValueError:
        return 1.0
    if scheme == LTX25_TE_QUANT_DEFAULT and bf16_gb > 0:
        return min(1.0, LTX25_TE_INT8_GB / float(bf16_gb))
    return 1.0


def ltx25_te_outcome(client: Any) -> Any:
    from .diffusion_precision import TEQuantOutcome

    scheme = (getattr(client, "config", None) or {}).get("te_quant")
    if scheme:
        return TEQuantOutcome(
            scheme, "int8 ConvRot weight-only Gemma 4 encoder (ComfyUI's LTX-2.5 encoder)"
        )
    return TEQuantOutcome(None)


def sidecar_missing_message(venv_dir: str) -> str:
    return (
        "LTX-2.5's text encoder is a Gemma 4 model (Gemma4UnifiedForConditionalGeneration) that needs transformers "
        f">= {TEXT_ENCODER_MIN_TRANSFORMERS}. Studio keeps that in its transformers 5.10 sidecar at '{venv_dir}', "
        "which is missing and could not be installed. Re-run the Studio installer (or check the network so Studio "
        "can rebuild the sidecar), then load LTX-2.5 again."
    )


def ensure_text_encoder_sidecar() -> str:
    """The ``.venv_t5_510`` dir, created / repaired the way the chat and training workers do it; RuntimeError naming
    the sidecar when it cannot be made usable."""
    from utils.transformers_version import _VENV_T5_510_DIR, _ensure_venv_t5_510_exists

    try:
        ok = _ensure_venv_t5_510_exists()
    except Exception as exc:  # noqa: BLE001 -- surface as the named sidecar failure
        logger.warning("video.ltx25_te: sidecar check failed: %s", exc)
        ok = False
    if not ok:
        raise RuntimeError(sidecar_missing_message(_VENV_T5_510_DIR))
    return _VENV_T5_510_DIR


def _server_reserved_mib(device: Optional[str]) -> Optional[int]:
    """What this (server) process holds on ``device`` right now, for the worker's budget check."""
    kind = str(device or "").split(":", 1)[0]
    if kind not in ("cuda", "xpu"):
        return None
    try:
        import torch
        return int(getattr(torch, kind).memory_reserved(torch.device(str(device)))) >> 20
    except Exception:  # noqa: BLE001 -- unknown: the worker judges by free memory alone
        return None


class TextEncoderWorkerError(RuntimeError):
    pass


class Ltx25TextEncoderClient:
    """One spawned encoder worker for one loaded LTX-2.5 pipeline.

    ``start`` spawns it (the Gemma 4 load overlaps the DiT load); ``wait_ready`` blocks until it has the model;
    ``encode`` sends prompts and returns ``(prompt_embeds, attention_mask)`` on the host. A dead worker is restarted on
    the next encode. ``close`` stops it (unload / failed load)."""

    # Two recent encodes kept on the host: a re-render with the same prompt (seed sweeps) skips the worker.
    _CACHE_ENTRIES = 2

    def __init__(
        self,
        *,
        repo: str,
        hf_token: Optional[str],
        cache_dir: Optional[str],
        local_files_only: bool,
        device: str,
        dtype: str,
        resident: bool,
        venv_dir: str,
        cancel_event: Optional[threading.Event] = None,
        single_file: Optional[str] = None,
        budget_mib: Optional[int] = None,
        te_quant: Optional[str] = None,
    ) -> None:
        self.config = {
            "repo": repo,
            # The single-file release's encoder file (config and tokenizer embedded) instead of pack subfolders.
            "single_file": single_file,
            "key_mapping": dict(LTX25_TEXT_ENCODER_KEY_MAPPING) if single_file else None,
            "hf_token": hf_token,
            "cache_dir": cache_dir,
            "local_files_only": bool(local_files_only),
            "device": device,
            "dtype": dtype,
            "resident": bool(resident),
            "venv_dir": venv_dir,
            # The load's memory budget: a non-resident encoder onloads only into what the server leaves of it.
            "budget_mib": budget_mib,
            "te_quant": te_quant,
        }
        self.cancel_event = cancel_event
        self._proc: Any = None
        self._cmd: Any = None
        self._resp: Any = None
        self._ready = False
        self._seq = 0
        self._lock = threading.Lock()
        self._cache: list[tuple[tuple, Any]] = []
        self.info: dict[str, Any] = {}
        self.last_encode: dict[str, Any] = {}

    # process management
    def start(self) -> None:
        import multiprocessing as mp

        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )

        from utils.process_lifetime import is_process_shutting_down

        # A quit during an LTX-2.5 load must not start an encoder worker the shutdown sweep has already passed.
        if is_process_shutting_down():
            raise TextEncoderWorkerError(
                "Unsloth is shutting down; not starting the LTX-2.5 text encoder worker."
            )
        ctx = mp.get_context("spawn")
        cache_env = get_hf_cache_paths().child_env({})
        with child_environment_for_spawn(cache_env), native_path_secret_removed_for_child_start():
            self._cmd = ctx.Queue()
            self._resp = ctx.Queue()
            proc = ctx.Process(
                target = run_without_native_path_secret,
                args = ("core.inference.video_ltx25_te_worker", "run_text_encoder_worker", cache_env),
                kwargs = {"cmd_queue": self._cmd, "resp_queue": self._resp, "config": self.config},
                daemon = True,
            )
            from utils.process_lifetime import spawn_on_lifetime_thread

            # The child arms PR_SET_PDEATHSIG, which follows the spawning THREAD: a load runs on a short-lived thread,
            # and a worker started there dies the moment the load returns.
            spawn_on_lifetime_thread(proc.start)
        self._proc = proc
        self._ready = False
        try:
            from utils.process_lifetime import adopt_pid
            adopt_pid(proc.pid)
        except Exception:  # noqa: BLE001 -- the daemon flag still ties it to this process
            pass
        # Recheck once the pid is recorded: the latch can be set between the gate above and the record.
        if is_process_shutting_down():
            logger.info(
                "video.ltx25_te: shutdown began during the spawn; reaping the encoder worker"
            )
            self._proc = None
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
            except Exception:  # noqa: BLE001 -- the reap is best-effort
                pass
            raise TextEncoderWorkerError(
                "Unsloth is shutting down; stopped the LTX-2.5 text encoder worker it had just started."
            )
        logger.info("video.ltx25_te: started the Gemma 4 encoder worker (pid %s)", proc.pid)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _get(
        self,
        cancel: Optional[threading.Event],
        what: str,
        timeout: Optional[float] = None,
        request_id: Optional[int] = None,
    ) -> dict:
        """The next worker reply. With ``request_id``, an error for another request (one a cancel abandoned) is skipped
        rather than raised, so it cannot fail the encode that follows."""
        began = time.monotonic()
        while True:
            if cancel is not None and cancel.is_set():
                from .video_families import VIDEO_CANCELLED_MSG
                raise RuntimeError(VIDEO_CANCELLED_MSG)
            try:
                msg = self._resp.get(timeout = 0.25)
            except _queue.Empty:
                if not self.alive():
                    code = getattr(self._proc, "exitcode", None)
                    self._proc = None
                    raise TextEncoderWorkerError(
                        f"The LTX-2.5 text encoder worker exited (code {code}) while {what}. Check the server log "
                        "for its error; the next request restarts it."
                    )
                if timeout is not None and time.monotonic() - began > timeout:
                    raise TextEncoderWorkerError(
                        f"The LTX-2.5 text encoder worker timed out while {what}."
                    )
                continue
            if msg.get("type") == "error":
                if (
                    request_id is not None
                    and msg.get("id") is not None
                    and msg.get("id") != request_id
                ):
                    continue
                raise TextEncoderWorkerError(str(msg.get("error") or "unknown error"))
            return msg

    def wait_ready(self, cancel: Optional[threading.Event] = None) -> dict:
        if self._ready:
            return self.info
        msg = self._get(cancel, "loading the Gemma 4 encoder")
        if msg.get("type") != "ready":
            raise TextEncoderWorkerError(
                f"Unexpected reply from the text encoder worker: {msg.get('type')}"
            )
        self._ready = True
        self.info = {k: v for k, v in msg.items() if k != "type"}
        logger.info("video.ltx25_te: encoder ready %s", self.info)
        return self.info

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self._ready = False
        self._cache.clear()
        if proc is None:
            return
        try:
            self._cmd.put({"type": "shutdown"})
            proc.join(timeout = 10)
        except Exception:  # noqa: BLE001 -- fall through to terminate
            pass
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout = 5)
        if proc.is_alive():
            proc.kill()
        logger.info("video.ltx25_te: encoder worker stopped")

    # encoding
    def encode(
        self,
        prompts: list[str],
        max_sequence_length: int,
        cancel: Optional[threading.Event] = None,
        *,
        copy: bool = True,
    ) -> tuple[Any, Any]:
        """``copy=False`` returns the cached host tensors themselves, for a caller that only reads them: a 385 MB host
        clone per call is ~40 ms on a B200 host."""
        key = (tuple(prompts), int(max_sequence_length))
        with self._lock:
            for cached_key, value in self._cache:
                if cached_key == key:
                    return (value[0].clone(), value[1].clone()) if copy else value
            if not self.alive():
                logger.warning("video.ltx25_te: encoder worker not running; restarting it")
                self.start()
            cancel = cancel if cancel is not None else self.cancel_event
            self.wait_ready(cancel)
            self._seq += 1
            seq = self._seq
            self._cmd.put(
                {
                    "type": "encode",
                    "id": seq,
                    "prompts": list(prompts),
                    "max_sequence_length": int(max_sequence_length),
                    "server_reserved_mib": _server_reserved_mib(self.config.get("device")),
                }
            )
            while True:
                msg = self._get(cancel, "encoding the prompt", request_id = seq)
                # A reply to a request a cancel abandoned arrives first; skip it.
                if msg.get("type") == "embeds" and msg.get("id") == seq:
                    break
            embeds, mask = msg["embeds"], msg["mask"]
            self.last_encode = {
                k: v for k, v in msg.items() if k not in ("type", "id", "embeds", "mask")
            }
            logger.info(
                "video.ltx25_te: encoded on %s %s", self.last_encode.get("device"), self.last_encode
            )
            self._cache.append((key, (embeds, mask)))
            del self._cache[: -self._CACHE_ENTRIES]
            return (embeds.clone(), mask.clone()) if copy else (embeds, mask)


def install_remote_encode_prompt(pipe: Any, client: Ltx25TextEncoderClient) -> None:
    """Replace ``pipe.encode_prompt`` with the worker round trip, keeping diffusers' signature and semantics (packed
    hidden states, left padding, CFG negative, ``num_videos_per_prompt`` repeat), so Studio's prompt cache wraps it
    like any other encoder."""

    def encode_prompt(
        prompt: Any = None,
        negative_prompt: Any = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Any = None,
        negative_prompt_embeds: Any = None,
        prompt_attention_mask: Any = None,
        negative_prompt_attention_mask: Any = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: Any = None,
        dtype: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        device = device or pipe._execution_device
        dtype = dtype or getattr(getattr(pipe, "transformer", None), "dtype", None)

        def _one(texts: list[str]) -> tuple[Any, Any]:
            # The client's own cached tensors (read only): every op below returns a new tensor.
            embeds, mask = client.encode(texts, max_sequence_length, copy = False)
            batch = embeds.shape[0]
            embeds = embeds.to(device = device, dtype = dtype)
            mask = mask.to(device = device)
            seq_len = embeds.shape[1]
            embeds = embeds.repeat(1, num_videos_per_prompt, 1).view(
                batch * num_videos_per_prompt, seq_len, -1
            )
            mask = mask.view(batch, -1).repeat(num_videos_per_prompt, 1)
            return embeds, mask

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = _one([p.strip() for p in prompt])
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` has batch size "
                    f"{batch_size}."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = _one(
                [p.strip() for p in negative_prompt]
            )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    pipe.encode_prompt = encode_prompt
    pipe._unsloth_ltx25_encoder = client


def close_encoder(pipe: Any) -> None:
    client = getattr(pipe, "_unsloth_ltx25_encoder", None) if pipe is not None else None
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 -- teardown is best-effort
            logger.warning("video.ltx25_te: could not stop the encoder worker: %s", exc)


def start_text_encoder(
    *,
    repo: str,
    revision: Optional[str] = None,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
    device: str,
    dtype: str,
    resident: bool,
    budget_mib: Optional[int] = None,
    te_quant: Optional[str] = None,
) -> Ltx25TextEncoderClient:
    """Check the sidecar (a named refusal before any weight is read), then spawn the encoder worker so the Gemma 4
    load overlaps the DiT load."""
    venv_dir = ensure_text_encoder_sidecar()
    client = Ltx25TextEncoderClient(
        repo = repo,
        hf_token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        device = device,
        dtype = dtype,
        resident = resident,
        venv_dir = venv_dir,
        single_file = LTX25_SINGLE_TEXT_ENCODER if is_ltx25_single_file_base(repo) else None,
        budget_mib = budget_mib,
        te_quant = te_quant,
    )
    from .diffusion_families import pinned_revision_kwargs

    # The worker is another process: the plan's commit travels in its config (absent when unpinned).
    client.config.update({"revision": revision} if revision else pinned_revision_kwargs(repo))
    client.start()
    return client


def attach_to_pipeline(
    pipe: Any,
    client: Ltx25TextEncoderClient,
    *,
    dev: bool,
    base_repo: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
) -> None:
    """Wire a freshly built pipe for LTX-2.5: the remote ``encode_prompt``, the variant's sampler and (distilled,
    two-stage) the latent upsampler, held beside the pipe so the memory plan leaves it alone."""
    install_remote_encode_prompt(pipe, client)
    pipe._unsloth_ltx25_dev = bool(dev)
    pipe._unsloth_ltx25_sampler = prepare_scheduler(pipe, dev = dev)
    pipe._unsloth_ltx25_upsampler = (
        load_latent_upsampler(
            base_repo,
            torch_dtype = torch_dtype,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        )
        if not dev and two_stage_enabled()
        else None
    )


# ── sampler ──────────────────────────────────────────────────────────────────


def make_ancestral_scheduler(base_scheduler: Any) -> Any:
    """``FlowMatchEulerDiscreteScheduler`` stepping euler-ancestral in its rectified-flow form (eta 1, s_noise 1):
    Euler down to ``sigma_down = sigma_next**2 / sigma``, then re-noise up to ``sigma_next``. The last step (to 0)
    is plain Euler. Tokens named by ``keep_tokens`` (the conditioned first latent frame of an image-to-video stage)
    take the deterministic Euler step, so the conditioning the pipeline blends in x0 space is not re-noised. Noise
    comes from ``noise_generator`` (the request's generator) for reproducible renders."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor

    class LTX25EulerAncestralScheduler(FlowMatchEulerDiscreteScheduler):
        noise_generator: Any = None
        # (keep, total): when a sample has ``total`` tokens, its first ``keep`` step deterministically.
        keep_tokens: Optional[tuple[int, int]] = None

        def __deepcopy__(self, memo: dict) -> Any:
            # The pipelines deep-copy this for the audio stream; the generator is shared, not cloned, so the audio
            # draws continue the request's stream instead of replaying the video's.
            import copy

            new = self.__class__.__new__(self.__class__)
            memo[id(self)] = new
            for key, value in self.__dict__.items():
                setattr(new, key, value if key == "noise_generator" else copy.deepcopy(value, memo))
            return new

        def step(self, model_output, timestep, sample, *args: Any, **kwargs: Any):  # type: ignore[override]
            return_dict = kwargs.get("return_dict", True)
            if self.step_index is None:
                self._init_step_index(timestep)
            index = self.step_index
            sigma = self.sigmas[index]
            sigma_next = self.sigmas[index + 1]
            work = sample.to(torch.float32)
            out = model_output.to(torch.float32)
            deterministic = work + (sigma_next - sigma) * out
            if float(sigma_next) <= 0.0 or float(sigma) <= 0.0:
                prev = deterministic
            else:
                denoised = work - sigma * out
                sigma_down = sigma_next * sigma_next / sigma
                ratio = sigma_down / sigma
                x_down = ratio * work + (1.0 - ratio) * denoised
                alpha_next = 1.0 - sigma_next
                alpha_down = 1.0 - sigma_down
                renoise = torch.sqrt(
                    torch.clamp(
                        sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, min = 0.0
                    )
                )
                noise = randn_tensor(
                    work.shape,
                    generator = self.noise_generator,
                    device = work.device,
                    dtype = torch.float32,
                )
                prev = (alpha_next / alpha_down) * x_down + renoise * noise
                keep = self.keep_tokens
                if keep is not None and prev.ndim == 3 and prev.shape[1] == keep[1] and keep[0] > 0:
                    prev[:, : keep[0]] = deterministic[:, : keep[0]]
            self._step_index += 1
            prev = prev.to(model_output.dtype)
            if not return_dict:
                return (prev,)
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteSchedulerOutput,
            )

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample = prev)

    return LTX25EulerAncestralScheduler.from_config(base_scheduler.config)


def prepare_scheduler(pipe: Any, *, dev: bool) -> str:
    """The family's sampler on the loaded pipe: dev = diffusers Euler with the README's dynamic shifting, distilled =
    euler-ancestral (unless ``UNSLOTH_LTX25_SAMPLER=euler``). Returns its name for the status."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = pipe.scheduler
    if dev:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, **LTX25_DEV_SCHEDULER_OVERRIDES
        )
        return "euler"
    if ancestral_enabled():
        pipe.scheduler = make_ancestral_scheduler(scheduler)
        return "euler_ancestral"
    return "euler"


# ── generation ───────────────────────────────────────────────────────────────


def condition_pipe(pipe: Any) -> Any:
    """``LTX2ConditionPipeline`` over the loaded pipe's modules (built once), sharing its video processor so Studio's
    frame conversion applies, and its remote ``encode_prompt``."""
    cached = getattr(pipe, "_unsloth_ltx25_condition_pipe", None)
    if cached is not None:
        return cached
    import inspect

    from .ltx2_import_compat import ensure_ltx2_pipelines_importable

    ensure_ltx2_pipelines_importable(logger)
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2ConditionPipeline

    params = inspect.signature(LTX2ConditionPipeline.__init__).parameters
    components = {k: v for k, v in pipe.components.items() if k in params}
    cond = LTX2ConditionPipeline(**components)
    cond.video_processor = pipe.video_processor
    cond.encode_prompt = pipe.encode_prompt
    try:
        cond.set_progress_bar_config(disable = True)
    except Exception:  # noqa: BLE001 -- cosmetic
        pass
    pipe._unsloth_ltx25_condition_pipe = cond
    return cond


def _guidance_off(call_params: Any) -> dict[str, float]:
    keys = {
        "stg_scale": 0.0,
        "modality_scale": 1.0,
        "guidance_rescale": 0.0,
        "audio_guidance_scale": 1.0,
        "audio_stg_scale": 0.0,
        "audio_modality_scale": 1.0,
        "audio_guidance_rescale": 0.0,
    }
    return {k: v for k, v in keys.items() if k in call_params}


def stage_steps(steps: int, *, dev: bool) -> tuple[int, int]:
    """(stage 1, stage 2) denoise steps a request runs."""
    if dev or not two_stage_enabled():
        return int(steps), 0
    return int(steps), len(LTX25_STAGE2_SIGMAS)


def render(
    pipe: Any,
    kwargs: dict[str, Any],
    *,
    dev: bool,
    image: Any = None,
    upsampler: Any = None,
    on_step: Optional[Callable[..., Any]] = None,
    cancel: Optional[threading.Event] = None,
    on_stage: Optional[Callable[[int, int], Any]] = None,
) -> Any:
    """Run one LTX-2.5 request. ``kwargs`` is the generic video call (prompt, size, frames, steps, guidance,
    frame_rate, generator, negative_prompt, callback); returns the final pipeline output (``.frames`` / ``.audio``).
    ``on_stage(width, height)`` runs before each denoise stage with the size that stage denoises at (live previews
    unpack the packed latents with it)."""
    failed = False
    try:
        return _render(
            pipe,
            kwargs,
            dev = dev,
            image = image,
            upsampler = upsampler,
            on_step = on_step,
            cancel = cancel,
            on_stage = on_stage,
        )
    except BaseException:
        failed = True
        raise
    finally:
        if image is not None or failed:
            # The condition pipeline shares pipe's modules but not its offload-hook list, so on every exit (dev,
            # single stage, two stage, cancel) pipe's own hooks are reset here, or its last onloaded module stays on
            # the card. A text render that stops between its stages (cancel, error) may hold the DiT stage 1
            # onloaded (diffusion_multistage.hold_denoiser_between_calls): reset it too.
            free = getattr(pipe, "maybe_free_model_hooks", None)
            if callable(free):
                try:
                    free()
                except Exception:  # noqa: BLE001 -- best-effort offload reset
                    pass


def _render(
    pipe: Any,
    kwargs: dict[str, Any],
    *,
    dev: bool,
    image: Any = None,
    upsampler: Any = None,
    on_step: Optional[Callable[..., Any]] = None,
    cancel: Optional[threading.Event] = None,
    on_stage: Optional[Callable[[int, int], Any]] = None,
) -> Any:
    import inspect

    import torch

    from .video_ltx2 import ltx23_verbatim_sigmas

    client = getattr(pipe, "_unsloth_ltx25_encoder", None)
    if client is not None:
        client.cancel_event = cancel
    runner = pipe
    if image is not None:
        runner = condition_pipe(pipe)
    call_params = inspect.signature(runner.__call__).parameters
    width, height = int(kwargs["width"]), int(kwargs["height"])
    steps = int(kwargs.get("num_inference_steps") or 0)
    generator = kwargs.get("generator")
    scheduler = getattr(pipe, "scheduler", None)
    if hasattr(scheduler, "noise_generator"):
        scheduler.noise_generator = generator
    if runner is not pipe:
        runner.scheduler = scheduler

    base: dict[str, Any] = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "num_inference_steps",
            "width",
            "height",
            "callback_on_step_end",
            "image",
            "last_image",
            "sigmas",
        )
        and k in call_params
    }

    def _callback(offset: int) -> Optional[Callable[..., Any]]:
        if on_step is None:
            return None

        def _cb(p: Any, step_index: int, timestep: Any, callback_kwargs: dict) -> dict:
            result = on_step(p, offset + step_index, timestep, callback_kwargs)
            if getattr(p, "_interrupt", False) and runner is not pipe:
                pipe._interrupt = True
            return result

        return _cb

    def _conditions(strength: float) -> dict[str, Any]:
        if image is None:
            return {}
        from .ltx2_import_compat import ensure_ltx2_pipelines_importable

        ensure_ltx2_pipelines_importable(logger)
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        return {
            "conditions": [
                LTX2VideoCondition(frames = image, index = 0, strength = strength, crf = LTX25_IMAGE_CRF)
            ]
        }

    def _keep(stage_width: int, stage_height: int, frames: int) -> None:
        if on_stage is not None:
            try:
                on_stage(stage_width, stage_height)
            except Exception:  # noqa: BLE001 -- a preview hint must never fail a render
                pass
        if not hasattr(scheduler, "keep_tokens"):
            return
        if image is None:
            scheduler.keep_tokens = None
            return
        spatial = int(getattr(runner, "vae_spatial_compression_ratio", 32) or 32)
        temporal = int(getattr(runner, "vae_temporal_compression_ratio", 8) or 8)
        per_frame = (stage_height // spatial) * (stage_width // spatial)
        latent_frames = (frames - 1) // temporal + 1
        scheduler.keep_tokens = (per_frame, per_frame * latent_frames)

    frames = int(kwargs.get("num_frames") or 121)
    if dev:
        call = dict(base, width = width, height = height, num_inference_steps = steps)
        call.update(_conditions(1.0))
        cb = _callback(0)
        if cb is not None:
            call["callback_on_step_end"] = cb
        _keep(width, height, frames)
        return runner(**call)

    distilled_sigmas = steps == len(LTX25_STAGE1_SIGMAS)
    guidance_off = _guidance_off(call_params)
    if not two_stage_enabled():
        call = dict(base, width = width, height = height, **guidance_off)
        if distilled_sigmas:
            call["sigmas"] = list(LTX25_STAGE1_SIGMAS)
        else:
            call["num_inference_steps"] = steps
        call.update(_conditions(1.0))
        cb = _callback(0)
        if cb is not None:
            call["callback_on_step_end"] = cb
        _keep(width, height, frames)
        with ltx23_verbatim_sigmas(pipe):
            return runner(**call)

    if upsampler is None:
        raise RuntimeError(
            "LTX-2.5's two-stage recipe needs the latent upsampler (latent_upsampler/ of the base pack), which "
            "is not loaded."
        )
    from .diffusion_multistage import encode_once, hold_denoiser_between_calls

    with encode_once(runner):
        half_w, half_h = width // 2, height // 2
        stage1 = dict(
            base,
            width = half_w,
            height = half_h,
            output_type = "latent",
            return_dict = False,
            **guidance_off,
        )
        if distilled_sigmas:
            stage1["sigmas"] = list(LTX25_STAGE1_SIGMAS)
        else:
            stage1["num_inference_steps"] = steps
        stage1.update(_conditions(LTX25_STAGE1_IMAGE_STRENGTH))
        cb = _callback(0)
        if cb is not None:
            stage1["callback_on_step_end"] = cb
        _keep(half_w, half_h, frames)
        # The connectors and the upsampler run before the DiT is needed again.
        companions = (getattr(pipe, "connectors", None), upsampler)
        with ltx23_verbatim_sigmas(pipe), hold_denoiser_between_calls(pipe, runner, companions):
            video_latents, audio_latents = runner(**stage1)
        if cancel is not None and cancel.is_set():
            from .video_families import VIDEO_CANCELLED_MSG
            raise RuntimeError(VIDEO_CANCELLED_MSG)

        with torch.inference_mode() if torch.is_inference_mode_enabled() else torch.no_grad():
            param = next(upsampler.parameters())
            # Saved before the move: Module.to() swaps the parameter's storage in place, so param.device follows it.
            home = param.device
            moved = home != video_latents.device
            if moved:
                upsampler.to(video_latents.device)
            try:
                upsampled = upsampler(video_latents.to(param.dtype))
            finally:
                if moved:
                    upsampler.to(home)
        if cancel is not None and cancel.is_set():
            from .video_families import VIDEO_CANCELLED_MSG
            raise RuntimeError(VIDEO_CANCELLED_MSG)

        stage2 = dict(
            base,
            width = half_w * 2,
            height = half_h * 2,
            latents = upsampled,
            audio_latents = audio_latents,
            sigmas = list(LTX25_STAGE2_SIGMAS),
            noise_scale = LTX25_STAGE2_SIGMAS[0],
            **guidance_off,
        )
        stage2.update(_conditions(LTX25_STAGE2_IMAGE_STRENGTH))
        cb = _callback(steps)
        if cb is not None:
            stage2["callback_on_step_end"] = cb
        _keep(half_w * 2, half_h * 2, frames)
        with ltx23_verbatim_sigmas(pipe):
            return runner(**stage2)
