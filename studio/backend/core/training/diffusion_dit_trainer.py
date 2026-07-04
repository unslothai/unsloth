# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flow-matching LoRA training for the DiT image families (FLUX.1-dev, Qwen-Image, Z-Image).

These are rectified-flow transformers, not the SDXL U-Net, so they share only the plumbing
in ``diffusion_train_common`` (config, dataset discovery, events, stop, publishing). The
training math here is flow matching: sample a sigma with the logit-normal density used by
the diffusers dreambooth scripts, form ``noisy = (1 - sigma) * latents + sigma * noise``,
predict the velocity, and regress it onto ``target = noise - latents``.

The per-family differences (latent normalisation + packing, the transformer forward
signature, and the LoRA save entrypoint) live in small ``_FamilySpec`` objects; the loop
itself is family-agnostic. Verified against diffusers 0.38.0.

Memory: the text encoder(s) are the largest module (T5-XXL ~9 GB for FLUX, Qwen2.5-VL ~7 GB
for Qwen-Image, Qwen3 for Z-Image), so captions are encoded ONCE up front and the encoders
are freed before the loop. The transformer trains as a QLoRA (nf4) adapter by default with
gradient checkpointing and 8-bit AdamW, so only the (small) LoRA params + optimizer state
and the frozen 4-bit base sit in VRAM during the loop.
"""

from __future__ import annotations

import gc
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.training.diffusion_train_common import (
    DEFAULT_LORA_FILENAME,
    DiffusionLoraConfig,
    EventCb,
    StopCb,
    _assert_trusted_base_model,
    _emit,
    _publish_to_lora_catalog,
    discover_image_caption_pairs,
)

# Per-family LoRA target modules (attention projections). FLUX / Qwen double-stream blocks
# also carry added-kv projections; Z-Image is single-stream. Kept here (not in the generic
# DEFAULT_LORA_TARGETS) because they are architecture-specific.
_FLUX_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)
_QWEN_TARGETS = _FLUX_TARGETS
_ZIMAGE_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def _select_lora_targets(
    cfg_targets: tuple[str, ...], spec_targets: tuple[str, ...]
) -> tuple[str, ...]:
    """Pick the LoRA target modules for a DiT run.

    ``normalized()`` leaves ``lora_target_modules`` empty when a caller does not set it, so
    an empty tuple means "unset" here: use the family's ``spec.lora_targets`` (which add the
    DiT-specific joint-attention projections). Any explicit tuple is a deliberate override
    and still wins."""
    if not tuple(cfg_targets):
        return tuple(spec_targets)
    return tuple(cfg_targets)


@dataclass
class _FamilySpec:
    """Everything the shared loop needs that differs by family."""

    family: str
    lora_targets: tuple[str, ...]
    # bf16 only (Z-Image overflows fp16 and its RoPE/embedder run in fp32).
    force_bf16: bool
    # Builds (pipe, transformer, vae) with the transformer loaded as a trainable nf4 QLoRA
    # when qlora=True. Returns the pipeline (for save_lora_weights + encode_prompt), the
    # transformer to attach LoRA to, and the VAE (kept resident for latent encoding).
    load: Callable[..., tuple[Any, Any, Any]]
    # Encode a list of captions -> a per-caption tuple of CPU tensors (the family's embeds).
    encode_prompts: Callable[..., list[tuple]]
    # Encode a pixel tensor [B,3,H,W] in [-1,1] -> latents (family-normalised, on device).
    encode_latents: Callable[..., Any]
    # One transformer forward: (transformer, noisy, timesteps, sigmas, embeds_batch, cfg,
    # device, weight_dtype) -> model_pred aligned with target = noise - latents.
    forward: Callable[..., Any]
    # Save the LoRA in diffusers format via the family pipeline's save_lora_weights.
    save: Callable[..., None]


# ── shared flow-matching helpers ──────────────────────────────────────────────
def _get_sigmas(scheduler, timesteps, device, dtype, n_dim):
    """Gather per-sample sigmas for ``timesteps`` and broadcast to ``n_dim`` (matches the
    diffusers dreambooth get_sigmas helper)."""
    import torch

    sigmas = scheduler.sigmas.to(device = device, dtype = dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _sample_timesteps(scheduler, batch_size, device):
    """Logit-normal density timestep sampling (weighting_scheme='logit_normal'), returning
    (timesteps, indices) into the scheduler's schedule."""
    import torch
    from diffusers.training_utils import compute_density_for_timestep_sampling

    u = compute_density_for_timestep_sampling(
        weighting_scheme = "logit_normal",
        batch_size = batch_size,
        logit_mean = 0.0,
        logit_std = 1.0,
        mode_scale = 1.29,
    )
    num_train = scheduler.config.num_train_timesteps
    indices = (u * num_train).long().clamp(0, num_train - 1)
    timesteps = scheduler.timesteps.to(device)[indices].to(device)
    return timesteps


def _encoders_to_device(pipe, device) -> None:
    """Move the pipeline's (non-quantized) text encoders to ``device`` before encoding.

    A QLoRA FLUX load places the nf4 transformer on GPU but leaves the text encoders on
    CPU, so encode_prompt would mix devices. Best-effort per encoder: a 4-bit encoder that
    is already placed raises on .to() and is left as-is."""
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, attr, None)
        if enc is None:
            continue
        try:
            enc.to(device)
        except (ValueError, RuntimeError, NotImplementedError):
            pass  # already-placed 4-bit encoder / non-movable module


def _bnb_4bit_config():
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    import torch
    return DiffusersBnb(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )


def _repo_is_prequantized(base_model: str) -> bool:
    """Heuristic: a repo whose name marks a bitsandbytes 4-bit build already ships a
    quantized transformer, so we load it as-is rather than re-quantizing on the fly. A
    dense (bf16) base instead gets on-the-fly nf4 quantization for QLoRA."""
    name = str(base_model or "").lower()
    return "bnb-4bit" in name or "-4bit" in name or "int4" in name or "nf4" in name


def _load_quantized_transformer(transformer_cls, cfg):
    """Load ``cfg.base_model``'s transformer subfolder as a trainable nf4 QLoRA module."""
    import torch
    return transformer_cls.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        quantization_config = _bnb_4bit_config(),
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    )


# ── FLUX.1-dev ────────────────────────────────────────────────────────────────
def _flux_load(cfg, device, weight_dtype, qlora):
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel

    if qlora:
        transformer = FluxTransformer2DModel.from_pretrained(
            cfg.base_model,
            subfolder = "transformer",
            quantization_config = _bnb_4bit_config(),
            torch_dtype = torch.bfloat16,
            token = cfg.hf_token,
        )
        pipe = FluxPipeline.from_pretrained(
            cfg.base_model,
            transformer = transformer,
            torch_dtype = torch.bfloat16,
            token = cfg.hf_token,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            cfg.base_model, torch_dtype = weight_dtype, token = cfg.hf_token
        )
        transformer = pipe.transformer
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, transformer, pipe.vae


def _flux_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            pe, pooled, text_ids = pipe.encode_prompt(
                prompt = cap,
                prompt_2 = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 512,
            )
            out.append((pe.cpu(), pooled.cpu(), text_ids.cpu()))
    return out


def _flux_encode_latents(vae, pixel_values):
    import torch

    with torch.no_grad():
        lat = vae.encode(pixel_values.to(torch.float32)).latent_dist.sample()
    lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
    return lat


def _flux_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    import torch
    from diffusers import FluxPipeline

    pe, pooled, text_ids = embeds_batch
    bsz, c, h, w = noisy.shape
    packed = FluxPipeline._pack_latents(noisy, bsz, c, h, w)
    # Position ids drive RoPE and are indices, not activations -- keep them float32 (the
    # dtype diffusers' own pipeline builds) regardless of the bf16 training dtype.
    img_ids = FluxPipeline._prepare_latent_image_ids(bsz, h // 2, w // 2, device, torch.float32)
    guidance = torch.full((bsz,), 1.0, device = device, dtype = torch.float32)
    model_pred = transformer(
        hidden_states = packed,
        timestep = timesteps / 1000,
        guidance = guidance,
        pooled_projections = pooled.to(weight_dtype),
        encoder_hidden_states = pe.to(weight_dtype),
        txt_ids = text_ids.to(torch.float32),
        img_ids = img_ids,
        return_dict = False,
    )[0]
    return FluxPipeline._unpack_latents(model_pred, h * 8, w * 8, 8)


def _flux_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import FluxPipeline
    FluxPipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


# ── Qwen-Image ────────────────────────────────────────────────────────────────
def _qwen_load(cfg, device, weight_dtype, qlora):
    import torch
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

    # The prequant default (unsloth/Qwen-Image-2512-unsloth-bnb-4bit) ships the transformer
    # 4-bit, so from_pretrained loads it trainable as-is. A dense (bf16) base -- the 20B
    # Qwen/Qwen-Image -- is quantized to nf4 on the fly so QLoRA still fits.
    kwargs = {"torch_dtype": torch.bfloat16, "token": cfg.hf_token}
    if qlora and not _repo_is_prequantized(cfg.base_model):
        kwargs["transformer"] = _load_quantized_transformer(QwenImageTransformer2DModel, cfg)
    pipe = QwenImagePipeline.from_pretrained(cfg.base_model, **kwargs)
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.transformer, pipe.vae


def _qwen_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            pe, mask = pipe.encode_prompt(
                prompt = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 1024,
            )
            out.append((pe.cpu(), mask.cpu() if mask is not None else None))
    return out


def _qwen_encode_latents(vae, pixel_values):
    import torch

    # AutoencoderKLQwenImage is a 3D (video) VAE: add a temporal dim, encode, drop it back
    # into a [B,16,1,H,W] latent normalised by the per-channel latents_mean / latents_std.
    px = pixel_values.to(torch.float32).unsqueeze(2)  # [B,3,1,H,W]
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.sample()  # [B,16,1,h,w]
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device = lat.device, dtype = lat.dtype)
    std = torch.tensor(vae.config.latents_std, device = lat.device, dtype = lat.dtype)
    mean = mean.view(1, z, 1, 1, 1)
    std = std.view(1, z, 1, 1, 1)
    return (lat - mean) / std


def _qwen_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    import torch
    from diffusers import QwenImagePipeline

    pe, mask = embeds_batch
    bsz, c, f, h, w = noisy.shape
    packed = QwenImagePipeline._pack_latents(noisy, bsz, c, h, w)
    # Each batch entry is a LIST of one (frame, h/2, w/2) tuple: the transformer indexes
    # sample[0] / sample[1:] per entry (transformer_qwenimage.py), so a flat list breaks it.
    img_shapes = [[(1, h // 2, w // 2)]] * bsz
    pred = transformer(
        hidden_states = packed,
        encoder_hidden_states = pe.to(weight_dtype),
        encoder_hidden_states_mask = mask.to(device) if mask is not None else None,
        timestep = timesteps / 1000,
        img_shapes = img_shapes,
        return_dict = False,
    )[0]
    return QwenImagePipeline._unpack_latents(pred, h * 8, w * 8, 8)


def _qwen_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import QwenImagePipeline
    QwenImagePipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


# ── Z-Image ───────────────────────────────────────────────────────────────────
def _zimage_load(cfg, device, weight_dtype, qlora):
    import torch
    from diffusers import ZImagePipeline, ZImageTransformer2DModel

    # Prequant default loads 4-bit as-is; the dense bf16 Tongyi-MAI base is quantized to nf4
    # on the fly. Z-Image is bf16 only (its RoPE/embedder run fp32; fp16 overflows).
    kwargs = {"torch_dtype": torch.bfloat16, "token": cfg.hf_token}
    if qlora and not _repo_is_prequantized(cfg.base_model):
        kwargs["transformer"] = _load_quantized_transformer(ZImageTransformer2DModel, cfg)
    pipe = ZImagePipeline.from_pretrained(cfg.base_model, **kwargs)
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.transformer, pipe.vae


def _zimage_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            pe, _neg = pipe.encode_prompt(
                prompt = cap,
                device = device,
                do_classifier_free_guidance = False,
                max_sequence_length = 512,
            )
            # pe is a list of one variable-length [seq, 2560] tensor per prompt.
            emb = pe[0] if isinstance(pe, (list, tuple)) else pe
            out.append((emb.cpu(),))
    return out


def _zimage_encode_latents(vae, pixel_values):
    import torch
    with torch.no_grad():
        lat = vae.encode(pixel_values.to(torch.float32)).latent_dist.mode()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor


def _zimage_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    import torch

    (emb,) = embeds_batch
    # List I/O: one [C,1,H,W] latent + one [seq,2560] caption per sample. The timestep
    # convention is REVERSED ((1000 - t) / 1000) and the prediction is NEGATED.
    x_list = list(noisy.unsqueeze(2).unbind(dim = 0))
    cap_list = [emb.to(device = device, dtype = weight_dtype)]
    t_norm = (1000 - timesteps) / 1000
    out = transformer(x_list, t_norm, cap_list, return_dict = False)[0]
    return -torch.stack(out, dim = 0).squeeze(2)


def _zimage_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import ZImagePipeline
    ZImagePipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


_SPECS: dict[str, _FamilySpec] = {
    "flux.1": _FamilySpec(
        family = "flux.1",
        lora_targets = _FLUX_TARGETS,
        force_bf16 = False,
        load = _flux_load,
        encode_prompts = _flux_encode_prompts,
        encode_latents = _flux_encode_latents,
        forward = _flux_forward,
        save = _flux_save,
    ),
    "qwen-image": _FamilySpec(
        family = "qwen-image",
        lora_targets = _QWEN_TARGETS,
        force_bf16 = True,
        load = _qwen_load,
        encode_prompts = _qwen_encode_prompts,
        encode_latents = _qwen_encode_latents,
        forward = _qwen_forward,
        save = _qwen_save,
    ),
    "z-image": _FamilySpec(
        family = "z-image",
        lora_targets = _ZIMAGE_TARGETS,
        force_bf16 = True,
        load = _zimage_load,
        encode_prompts = _zimage_encode_prompts,
        encode_latents = _zimage_encode_latents,
        forward = _zimage_forward,
        save = _zimage_save,
    ),
}


# HF repos that gate access behind a license acceptance: training needs a token whose
# account has accepted the license. Checked by name (no network) so a missing token fails
# fast with an actionable message instead of a confusing 401 mid-load.
_GATED_TRAIN_REPOS = frozenset({"black-forest-labs/flux.1-dev"})


def _assert_gated_access(base_model: str, hf_token: Optional[str]) -> None:
    """Raise a clear error before loading a gated base without a token."""
    name = str(base_model or "").strip().lower()
    if name in _GATED_TRAIN_REPOS and not (hf_token and str(hf_token).strip()):
        raise ValueError(
            f"'{base_model}' is a gated Hugging Face repo. Accept its license on the Hub "
            f"and add your HF token in Studio settings before training from it."
        )


def _load_pixel_tensor(path, resolution, center_crop, random_flip, rng):
    """Load an image -> a normalised [3,H,W] tensor in [-1,1]. Same geometry as the SDXL
    loader but without the SDXL time-ids (DiT families don't use them)."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w0, h0 = img.size
    scale = resolution / min(w0, h0)
    rw, rh = max(resolution, round(w0 * scale)), max(resolution, round(h0 * scale))
    img = img.resize((rw, rh), Image.LANCZOS)
    if center_crop:
        left, top = (rw - resolution) // 2, (rh - resolution) // 2
    else:
        left = rng.randint(0, max(0, rw - resolution))
        top = rng.randint(0, max(0, rh - resolution))
    img = img.crop((left, top, left + resolution, top + resolution))
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img, dtype = np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def run_dit_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """Train a flow-matching DiT LoRA (FLUX.1-dev / Qwen-Image / Z-Image) and export it."""
    import torch
    import torch.nn.functional as F
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    cfg = config.normalized()
    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        raise ValueError(f"No DiT trainer for family {cfg.resolved_family!r}")

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    save_on_stop = True

    def _check_stop() -> bool:
        nonlocal save_on_stop
        if should_stop is None:
            return False
        sig = should_stop()
        if not sig:
            return False
        if isinstance(sig, dict) and sig.get("save") is False:
            save_on_stop = False
        return True

    # DiT families train in bf16 (Z-Image/Qwen require it; FLUX prefers it). A caller that
    # explicitly asks for fp16 on a bf16-only family is refused rather than silently
    # upgraded, so the choice is never misrepresented.
    if cfg.mixed_precision == "fp16" and spec.force_bf16:
        raise ValueError(
            f"{spec.family} LoRA training requires bf16: fp16 overflows its fp32 RoPE / "
            f"embedder internals. Set mixed precision to bf16."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The flow-matching + 4-bit path is bf16 throughout (fp32 on a CPU-only box, which is
    # unsupported for real runs but keeps import/unit tests architecture-agnostic).
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    use_lora_targets = _select_lora_targets(cfg.lora_target_modules, spec.lora_targets)

    _assert_trusted_base_model(cfg.base_model)
    _assert_gated_access(cfg.base_model, cfg.hf_token)
    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    _emit(on_event, "model_load_started", num_images = len(pairs))
    if _check_stop():
        out_dir = Path(cfg.output_dir).expanduser()
        _emit(
            on_event, "complete", output_dir = str(out_dir), lora_path = None, stopped = True, steps_run = 0
        )
        return str(out_dir)

    # QLoRA by default for the big DiTs (nf4 transformer). The prequant Qwen/Z-Image repos
    # are already 4-bit; FLUX quantizes its transformer on the fly.
    pipe, transformer, vae = spec.load(cfg, device, weight_dtype, qlora = True)

    # Precompute all caption embeddings, then free the (large) text encoder(s): captions are
    # constant and the encoders are frozen, so this is exact and the biggest memory win.
    image_paths = [p for p, _ in pairs]
    captions = [c for _, c in pairs]
    uniq = sorted(set(captions))
    encoded = spec.encode_prompts(pipe, uniq, device)
    caption_embeds = {cap: emb for cap, emb in zip(uniq, encoded)}
    _free_text_encoders(pipe)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Freeze the base; attach the trainable LoRA to the transformer.
    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r = cfg.lora_rank,
            lora_alpha = cfg.lora_alpha,
            lora_dropout = cfg.lora_dropout,
            init_lora_weights = "gaussian",
            target_modules = list(use_lora_targets),
        )
    )
    if cfg.gradient_checkpointing:
        # Non-reentrant checkpointing: reentrant recompute of a bnb 4-bit LoRA linear can
        # trip an illegal memory access on the larger FLUX transformer, and non-reentrant
        # is the recommended mode anyway (it also handles a checkpointed segment whose
        # inputs do not require grad, which happens with a frozen 4-bit base).
        import functools
        import torch.utils.checkpoint as _ckpt
        transformer.enable_gradient_checkpointing(
            gradient_checkpointing_func = functools.partial(_ckpt.checkpoint, use_reentrant = False)
        )
    cast_training_params(transformer, dtype = torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    optimizer = _make_optimizer(lora_params, cfg.learning_rate)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model, subfolder = "scheduler", token = cfg.hf_token
    )

    _emit(on_event, "model_load_completed")

    transformer.train()
    stopped = False
    running_loss = 0.0
    peak_gb = 0.0
    t_start = time.time()
    done = 0
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none = True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            i = rng.randrange(len(image_paths))
            px = (
                _load_pixel_tensor(
                    image_paths[i], cfg.resolution, cfg.center_crop, cfg.random_flip, rng
                )
                .unsqueeze(0)
                .to(device)
            )
            latents = spec.encode_latents(vae, px).to(weight_dtype)

            noise = torch.randn_like(latents)
            timesteps = _sample_timesteps(scheduler, latents.shape[0], device)
            sigmas = _get_sigmas(scheduler, timesteps, device, weight_dtype, latents.ndim)
            noisy = (1.0 - sigmas) * latents + sigmas * noise

            emb = caption_embeds[captions[i]]
            emb_dev = tuple(
                t.to(device = device, dtype = weight_dtype)
                if (t is not None and t.is_floating_point())
                else (t.to(device) if t is not None else None)
                for t in emb
            )
            # bf16 autocast around the forward + loss, matching the diffusers dreambooth
            # scripts' accelerator.autocast: it reconciles the fp32 LoRA params with the
            # bnb 4-bit base matmuls in one compute dtype. Without it the 4-bit backward
            # on FLUX dies with an illegal-address / CUBLAS failure.
            autocast = (
                torch.autocast(device_type = "cuda", dtype = torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            )
            with autocast:
                model_pred = spec.forward(
                    transformer, noisy, timesteps, sigmas, emb_dev, cfg, device, weight_dtype
                )
                target = noise - latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps

        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm)
        optimizer.step()

        running_loss += step_loss
        done = opt_step + 1
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            if device == "cuda":
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            sps = round(
                (done * cfg.train_batch_size * cfg.gradient_accumulation_steps)
                / max(time.time() - t_start, 1e-6),
                3,
            )
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                learning_rate = cfg.learning_rate,
                samples_per_second = sps,
                peak_memory_gb = peak_gb or None,
            )
        if _check_stop():
            stopped = True
            break

    out_dir = Path(cfg.output_dir).expanduser()
    lora_path: Optional[str] = None
    catalog_path: Optional[str] = None
    if not (stopped and not save_on_stop):
        out_dir.mkdir(parents = True, exist_ok = True)
        layers = get_peft_model_state_dict(transformer)
        spec.save(pipe, str(out_dir), layers)
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        catalog_path = _publish_to_lora_catalog(lora_path, cfg)
    _emit(
        on_event,
        "complete",
        output_dir = str(out_dir),
        lora_path = lora_path,
        catalog_path = catalog_path,
        family = cfg.resolved_family,
        base_model = cfg.base_model,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
    )
    return str(out_dir)


def _make_optimizer(params, lr):
    """8-bit AdamW (bitsandbytes) when available -- half the optimizer state, no accuracy
    regression for LoRA -- else the torch AdamW fallback."""
    import torch
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr = lr)
    except Exception:  # noqa: BLE001 -- bnb missing / no CUDA: fall back to torch AdamW
        return torch.optim.AdamW(params, lr = lr)


def _free_text_encoders(pipe) -> None:
    """Drop every text-encoder / tokenizer the pipeline holds, so the (large) encoders do
    not sit in VRAM during training. The embeddings are already precomputed."""
    for attr in (
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
    ):
        if getattr(pipe, attr, None) is not None:
            try:
                setattr(pipe, attr, None)
            except Exception:  # noqa: BLE001
                pass
