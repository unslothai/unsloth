# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flow-matching LoRA training for the DiT image families (FLUX.1-dev, Qwen-Image, Z-Image).

These are rectified-flow transformers, not the SDXL U-Net, so they share only the plumbing
in ``diffusion_train_common`` (config, dataset discovery, events, stop, publishing). The
training math here is flow matching: sample a sigma with the logit-normal density used by
the diffusers dreambooth scripts, form ``noisy = (1 - sigma) * latents + sigma * noise``,
predict the velocity, and regress it onto ``target = noise - latents``.

The per-family differences (latent normalisation + packing, the transformer forward
signature, embedding collation, and the LoRA save entrypoint) live in small ``_FamilySpec``
objects; the loop itself is family-agnostic. Verified against diffusers 0.38.0.

Memory: the text encoder(s) are the largest module (T5-XXL ~9 GB for FLUX, Qwen2.5-VL ~7 GB
for Qwen-Image, Qwen3 for Z-Image), so captions are encoded ONCE up front and the encoders
are freed before the loop. VAE latents are likewise precomputed into a small CPU cache
(``cache_latents``) and the VAE freed: the cache stores the posterior's affine parameters
(mean/std folded through the family's latent normalisation), so every step still draws a
fresh VAE sample -- distribution-identical to encoding in the loop, without keeping the VAE
resident or paying a per-step encode. The transformer trains as a QLoRA (nf4) adapter by
default with gradient checkpointing and 8-bit AdamW, so only the (small) LoRA params +
optimizer state and the frozen 4-bit base sit in VRAM during the loop.
"""

from __future__ import annotations

import gc
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core.training.diffusion_train_common import (
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    EventCb,
    LATENT_CACHE_OVER_BUDGET,
    StopCb,
    _apply_perf_flags,
    _assert_trusted_base_model,
    _emit,
    _latent_cache_forced,
    _latent_cache_over_budget,
    _plan_cache_variants,
    _publish_to_lora_catalog,
    _restore_perf_flags,
    discover_image_caption_pairs,
    has_functional_torchao,
    native_bf16_supported,
    PermutationBatchSampler,
    repo_is_prequantized,
    resolve_train_steps,
)

# Per-family LoRA target modules (attention projections). FLUX / Qwen double-stream blocks
# also carry added-kv projections; Z-Image is single-stream. Architecture-specific, so kept
# here rather than in the generic DEFAULT_LORA_TARGETS.
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
# The Krea 2 authors' recommended default targets (their DreamBooth reference script):
# attention + SwiGLU + text-fusion projector + conditioning embedders. For long runs they
# suggest narrowing to the attention layers so prompt adherence doesn't drop.
_KREA2_TARGETS = (
    "img_in",
    "final_layer.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "ff.up",
    "ff.down",
    "text_fusion.projector",
    "txt_in.linear_1",
    "txt_in.linear_2",
    "time_embed.linear_1",
    "time_embed.linear_2",
    "time_mod_proj",
)


def _select_lora_targets(
    cfg_targets: tuple[str, ...], spec_targets: tuple[str, ...]
) -> tuple[str, ...]:
    """Pick the LoRA target modules for a DiT run.

    ``normalized()`` always fills ``lora_target_modules`` with the generic
    ``DEFAULT_LORA_TARGETS`` when a caller does not set it, so that value means "unset"
    here: prefer the family's ``spec.lora_targets`` (which add the DiT-specific
    projections). Any OTHER explicit tuple is a deliberate override and still wins."""
    if tuple(cfg_targets) == DEFAULT_LORA_TARGETS:
        return tuple(spec_targets)
    return tuple(cfg_targets)


@dataclass
class _FamilySpec:
    """Everything the shared loop needs that differs by family."""

    family: str
    lora_targets: tuple[str, ...]
    # bf16 only (Z-Image overflows fp16 and its RoPE/embedder run in fp32).
    force_bf16: bool
    # Approximate dense-bf16 transformer weight size, used by base_precision="auto" to
    # decide which mode fits the free VRAM (with headroom for activations + optimizer).
    dense_bf16_gb: float
    # Phased load so the multi-GB transformer never coexists with the text encoders + VAE:
    # ``load_conditioners`` builds the pipeline WITHOUT its transformer (encode_prompt + VAE
    # only) and returns (pipe, vae); ``load_transformer`` then loads the transformer alone (as
    # trainable nf4 QLoRA when qlora=True) once the conditioners are freed. Roughly halves peak
    # VRAM for the big DiTs.
    load_conditioners: Callable[..., tuple[Any, Any]]
    load_transformer: Callable[..., Any]
    # Encode a list of captions -> a per-caption tuple of CPU tensors (the family's embeds).
    encode_prompts: Callable[..., list[tuple]]
    # Encode a pixel tensor [B,3,H,W] in [-1,1] -> latents (family-normalised, on device).
    encode_latents: Callable[..., Any]
    # Encode a pixel tensor -> (A, B) affine posterior params so a per-step sample is
    # A + B * randn (family normalisation folded in). B is None for a deterministic
    # (mode-based) family. Used by the latent cache.
    encode_latent_stats: Callable[..., tuple]
    # Collate a list of per-caption embed tuples -> one batched tuple on device. ``pad_to``
    # pins a fixed text length for families with variable-length embeds (compile).
    collate: Callable[..., tuple]
    # One transformer forward: (transformer, noisy, timesteps, sigmas, embeds_batch, cfg,
    # device, weight_dtype) -> model_pred aligned with target = noise - latents.
    forward: Callable[..., Any]
    # Save the LoRA in diffusers format via the family pipeline's save_lora_weights.
    save: Callable[..., None]


# ── shared flow-matching helpers ──────────────────────────────────────────────
def _gather_sigmas(scheduler, indices, device, dtype, n_dim):
    """Gather per-sample sigmas for schedule ``indices`` and broadcast to ``n_dim``.
    Index-based (no per-item search): ``indices`` are the positions ``_sample_timesteps``
    drew from ``scheduler.timesteps``, and ``scheduler.sigmas`` is aligned with it, so this
    returns exactly what the diffusers ``get_sigmas`` timestep-matching helper would."""
    sigma = scheduler.sigmas[indices].to(device = device, dtype = dtype).flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _sample_timesteps(scheduler, batch_size, device):
    """Logit-normal density timestep sampling (weighting_scheme='logit_normal'), returning
    (timesteps, indices) into the scheduler's schedule."""
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
    return timesteps, indices


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


# Kept as a module name for existing callers/tests; the heuristic itself moved to
# diffusion_train_common so config validation can use it without importing this module.
_repo_is_prequantized = repo_is_prequantized


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


def _load_pipe_without_transformer(pipe_cls, cfg, device):
    """Load a pipeline for conditioning only: ``transformer = None`` skips the multi-GB
    denoiser entirely (the documented diffusers pattern), leaving just the text encoders +
    tokenizers + VAE + scheduler. The transformer loads later, after these are freed."""
    import torch

    pipe = pipe_cls.from_pretrained(
        cfg.base_model,
        transformer = None,
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    )
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.vae


def _load_dit_transformer(transformer_cls, cfg, device, base_precision):
    """Load the transformer alone in the resolved ``base_precision``:

    - nf4: a prequant (bnb-4bit) repo carries its quantization config and loads 4-bit
      as-is; a dense base is quantized to nf4 on the fly. The memory floor.
    - bf16 / fp8 / mxfp8: the dense transformer (fp8/mxfp8 convert its frozen linears to
      float8 training compute AFTER the LoRA attaches; storage stays bf16).
    - int8: the dense transformer quantized in place to torchao weight-only int8 (the
      PEFT-attachable scheme), roughly halving the bf16 weight footprint."""
    import torch

    if base_precision == "nf4":
        if not repo_is_prequantized(cfg.base_model):
            return _load_quantized_transformer(transformer_cls, cfg)
        transformer = transformer_cls.from_pretrained(
            cfg.base_model,
            subfolder = "transformer",
            torch_dtype = torch.bfloat16,
            token = cfg.hf_token,
        )
        # A prequant load is already device-placed by bitsandbytes.
        if not getattr(transformer, "is_loaded_in_4bit", False):
            transformer = transformer.to(device)
        return transformer

    # Dense load for bf16 / fp8 / mxfp8 / int8. int8 quantizes AFTER the LoRA attaches (see
    # _int8_quantize_base): quantizing first makes peft dispatch its TorchaoLoraLinear wrapper,
    # whose peft-0.18 constructor is incompatible with the torchao-0.16 config API (missing
    # get_apply_tensor_subclass).
    return transformer_cls.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    ).to(device)


def _int8_quantize_base(transformer) -> None:
    """torchao weight-only int8 on the big frozen linears, applied after add_adapter so
    the base_layer inside each LoRA wrapper quantizes while the adapters stay high
    precision. ``make_filter_fn`` (shared with the inference quant layer) keeps only
    Linears with >= 512 features -- which also naturally skips the rank-sized LoRA
    matrices -- and drops the M=1 modulation projections int8 kernels reject."""
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme, make_filter_fn
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    quantize_(
        transformer,
        Int8WeightOnlyConfig(),
        filter_fn = make_filter_fn(512, exclude_name_tokens = exclude_tokens_for_scheme("int8")),
    )


def _fp8_module_filter(mod, fqn: str) -> bool:
    """Which frozen linears get float8 training compute: skip anything LoRA-owned (the
    adapters must stay high precision -- PEFT has no float8 base support), the output
    projection, and shapes float8 kernels reject (dims not divisible by 16), matching the
    diffusers FLUX2 reference filter."""
    import torch.nn as nn

    if not isinstance(mod, nn.Linear):
        return False
    if "lora_" in fqn:
        return False
    if fqn.endswith("proj_out") or ".proj_out." in fqn:
        return False
    return mod.in_features % 16 == 0 and mod.out_features % 16 == 0


def _apply_fp8_training(transformer, on_event) -> bool:
    """Convert the frozen base linears to torchao float8 training compute (dynamic scaling;
    weights stay bf16 in memory). Applied AFTER add_adapter so the filter can exclude the
    LoRA modules. Never fatal: on any failure the run continues in bf16 with a warning."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        convert_to_float8_training(
            transformer,
            module_filter_fn = _fp8_module_filter,
            config = Float8LinearConfig(pad_inner_dim = True),
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- fp8 is an optimisation, never fatal
        _emit(on_event, "warning", message = f"fp8 training unavailable, using bf16 compute: {exc}")
        return False


def _mx_module_filter(mod, fqn: str) -> bool:
    """Which frozen linears get mxfp8 training compute: skip anything LoRA-owned (the
    adapters must stay high precision), the output projection (same guard as fp8), and
    shapes the 32-wide MX block scaling cannot tile (dims not divisible by 32)."""
    import torch.nn as nn

    if not isinstance(mod, nn.Linear):
        return False
    if "lora_" in fqn:
        return False
    if fqn.endswith("proj_out") or ".proj_out." in fqn:
        return False
    # Skip biased linears: the torchao 0.17 MX training path swaps the weight for a wrapper whose
    # linear override computes input @ weight_t and drops the bias entirely, so an mxfp8'd FROZEN
    # base linear loses its bias and changes the output the LoRA regresses against (verified on
    # Blackwell: bias fully dropped). Keep biased linears in bf16.
    if getattr(mod, "bias", None) is not None:
        return False
    return mod.in_features % 32 == 0 and mod.out_features % 32 == 0


def _mxfp8_training_config():
    """The torchao MX training config across the prototype API's revisions: torchao 0.16
    ships ``MXLinearConfig`` in ``prototype.mx_formats``; 0.17 removed it in favour of the
    ``MXFP8TrainingOpConfig`` recipe API shared with MoE training. Both feed ``quantize_``.
    Raises ImportError when neither API exists (mxfp8 then falls back to bf16)."""
    try:
        from torchao.prototype.mx_formats import MXLinearConfig
        return MXLinearConfig.from_recipe_name("mxfp8_cublas")
    except ImportError:
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        return MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_RCEIL)


def _apply_mxfp8_training(transformer, on_event) -> bool:
    """Swap the frozen base linears to torchao MX float8 training compute (mxfp8, the
    Blackwell-native block-scaled format; the swap is in place and the weights stay bf16
    in memory, so like fp8 this is a speed mode, not a memory mode). Applied AFTER
    add_adapter so the filter can exclude the LoRA modules. Only competitive under
    torch.compile and only ahead of compiled bf16 at large token counts (high resolution
    or batch), which is why it stays an explicit opt-in rather than an "auto" pick.
    Never fatal: on any failure the run continues in bf16 with a warning."""
    try:
        from torchao.quantization import quantize_
        quantize_(
            transformer,
            _mxfp8_training_config(),
            filter_fn = _mx_module_filter,
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- mxfp8 is an optimisation, never fatal
        _emit(on_event, "warning", message = f"mxfp8 training unavailable, using bf16 compute: {exc}")
        return False


def _pick_auto_precision(
    prequant,
    device,
    free_gb,
    dense_gb,
    capability,
    has_fp8,
    has_torchao = True,
) -> str:
    """Pure policy for base_precision="auto": nf4 for a prequant base or no CUDA; else the
    fastest dense mode whose weights + headroom (activations, optimizer, cache) fit the
    free VRAM at decision time. bf16 + regional compile is the measured speed winner
    (2.3-2.6x over nf4 on B200); fp8 stays an explicit opt-in because torchao float8's
    dynamic-scaling overhead made it SLOWER than compiled bf16 at LoRA-training shapes on
    the same hardware. int8 must still materialise the full bf16 transformer before
    ``quantize_`` shrinks it module-by-module, so its band requires the dense-load
    transient (1.15x dense) to fit -- what int8 buys in that band is steady-state
    headroom for activations and the latent cache, not load-time memory. int8 also needs
    torchao at runtime (``_int8_quantize_base`` has no fallback, unlike fp8), so auto only
    picks it when torchao is importable and drops to nf4 otherwise. ``capability``/``has_fp8``
    remain parameters so the policy can be revisited per GPU generation without changing
    callers."""
    _ = capability, has_fp8
    if prequant or device != "cuda" or not free_gb or not dense_gb:
        return "nf4"
    if free_gb > dense_gb * 1.5:
        return "bf16"
    if free_gb > dense_gb * 1.15:
        return "int8" if has_torchao else "nf4"
    return "nf4"


def _resolve_base_precision(cfg, spec, device) -> str:
    """Resolve "auto" against the live GPU (free VRAM measured BEFORE anything loads);
    explicit modes pass through (normalized() already validated them against the repo and
    compute dtype) but are re-checked against the live device here: the dense modes are
    CUDA-only, and /info never advertises them on a host without a GPU, so an explicit
    request from a stale or direct client fails fast instead of loading a full dense
    transformer onto the CPU."""
    mode = (cfg.base_precision or "nf4").strip().lower()
    if mode != "auto":
        if mode in ("bf16", "int8", "fp8", "mxfp8") and device != "cuda":
            raise ValueError(
                f"base_precision={mode!r} needs a CUDA GPU; this host has none. "
                f"Use base_precision='nf4' or 'auto'."
            )
        # int8 has no runtime fallback (_int8_quantize_base imports torchao unconditionally), so
        # an explicit int8 against a missing torchao or the Windows-ROCm stub would leave the
        # transformer dense with compile disabled -- memory saving gone, likely OOM. The auto pick
        # and /info already gate on a FUNCTIONAL torchao; apply the same gate to the explicit
        # request so it fails fast. fp8 keeps its own fallback (_apply_fp8_training), so int8-only.
        if mode == "int8" and not has_functional_torchao():
            raise ValueError(
                "base_precision='int8' needs a functional torchao install; this host's "
                "torchao is missing or the non-functional Windows-ROCm stub. Use "
                "base_precision='nf4', 'bf16', or 'auto'."
            )
        # mxfp8 needs Blackwell (sm100+): its MX GEMM has no kernel below sm100 and raises at the
        # first training step, AFTER a full dense load. /info advertises mxfp8 only on sm100+
        # (train_precision_modes), so re-check here to fail fast for a stale/direct client on an
        # older GPU instead of crashing mid-run.
        if mode == "mxfp8" and device == "cuda":
            try:
                import torch
                blackwell = torch.cuda.get_device_capability() >= (10, 0)
            except Exception:  # noqa: BLE001 -- probe failure -> treat as unsupported, fail fast
                blackwell = False
            if not blackwell:
                raise ValueError(
                    "base_precision='mxfp8' needs a Blackwell (sm100+) GPU; this GPU is older. "
                    "Use base_precision='bf16', 'int8', 'nf4', or 'auto'."
                )
        return mode
    # auto may only resolve to the dense modes when the run uses bf16 compute, mirroring
    # the normalized() rule for explicit dense modes; otherwise stay on the nf4 floor.
    if getattr(cfg, "mixed_precision", "bf16") != "bf16":
        return "nf4"
    prequant = repo_is_prequantized(cfg.base_model)
    free_gb = None
    capability = None
    has_fp8 = False
    # int8 has no runtime fallback, so gate the auto pick on a FUNCTIONAL torchao: a plain
    # find_spec("torchao") is satisfied by the Windows-ROCm import stub whose quantize_ is a
    # no-op that leaves the transformer dense with compile disabled. has_functional_torchao
    # imports the exact symbols _int8_quantize_base uses and rejects the stub.
    has_torchao = has_functional_torchao()
    if device == "cuda":
        try:
            import torch

            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            capability = torch.cuda.get_device_capability()
            has_fp8 = hasattr(torch, "float8_e4m3fn")
        except Exception:  # noqa: BLE001 -- probe failure -> the safe mode
            pass
    return _pick_auto_precision(
        prequant, device, free_gb, spec.dense_bf16_gb, capability, has_fp8, has_torchao
    )


# ── FLUX.1-dev ────────────────────────────────────────────────────────────────
def _flux_load_conditioners(cfg, device, weight_dtype):
    from diffusers import FluxPipeline
    return _load_pipe_without_transformer(FluxPipeline, cfg, device)


def _flux_load_transformer(cfg, device, weight_dtype, base_precision):
    from diffusers import FluxTransformer2DModel
    return _load_dit_transformer(FluxTransformer2DModel, cfg, device, base_precision)


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


def _flux_encode_latent_stats(vae, pixel_values):
    import torch

    with torch.no_grad():
        dist = vae.encode(pixel_values.to(torch.float32)).latent_dist
    scale = vae.config.scaling_factor
    return (dist.mean - vae.config.shift_factor) * scale, dist.std * scale


def _flux_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # FLUX embeds are fixed-length (encode_prompt pads to max_sequence_length), so a plain
    # cat batches them; text_ids are shared position ids, identical across prompts.
    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    pooled = torch.cat([e[1] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = entries[0][2].to(device = device, dtype = torch.float32)
    return (pe, pooled, text_ids)


# Per-run cache of the step-invariant FLUX conditioning tensors (RoPE image ids + guidance
# vector): shapes are fixed once resolution/batch are, so rebuilding them every step is pure
# allocator churn. Cleared at run start (subprocess-local anyway).
_FLUX_STATIC: dict[tuple, tuple] = {}


def _flux_static_inputs(bsz, h, w, device):
    import torch
    from diffusers import FluxPipeline

    key = (bsz, h, w, str(device))
    hit = _FLUX_STATIC.get(key)
    if hit is None:
        # Position ids drive RoPE and are indices, not activations -- keep them float32 (the
        # dtype diffusers' own pipeline builds) regardless of the bf16 training dtype.
        img_ids = FluxPipeline._prepare_latent_image_ids(bsz, h // 2, w // 2, device, torch.float32)
        guidance = torch.full((bsz,), 1.0, device = device, dtype = torch.float32)
        hit = _FLUX_STATIC[key] = (img_ids, guidance)
    return hit


def _flux_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import FluxPipeline

    pe, pooled, text_ids = embeds_batch
    bsz, c, h, w = noisy.shape
    packed = FluxPipeline._pack_latents(noisy, bsz, c, h, w)
    img_ids, guidance = _flux_static_inputs(bsz, h, w, device)
    model_pred = transformer(
        hidden_states = packed,
        timestep = timesteps / 1000,
        guidance = guidance,
        pooled_projections = pooled,
        encoder_hidden_states = pe,
        txt_ids = text_ids,
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
def _qwen_load_conditioners(cfg, device, weight_dtype):
    from diffusers import QwenImagePipeline
    return _load_pipe_without_transformer(QwenImagePipeline, cfg, device)


def _qwen_load_transformer(cfg, device, weight_dtype, base_precision):
    # The prequant default (unsloth/Qwen-Image-2512-unsloth-bnb-4bit) ships the transformer
    # 4-bit and loads trainable as-is under nf4; the dense modes need the 20B Qwen/Qwen-Image base.
    from diffusers import QwenImageTransformer2DModel
    return _load_dit_transformer(QwenImageTransformer2DModel, cfg, device, base_precision)


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


def _qwen_latent_affine(vae, ref):
    import torch

    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device = ref.device, dtype = ref.dtype)
    std = torch.tensor(vae.config.latents_std, device = ref.device, dtype = ref.dtype)
    return mean.view(1, z, 1, 1, 1), std.view(1, z, 1, 1, 1)


def _qwen_encode_latents(vae, pixel_values):
    import torch

    # AutoencoderKLQwenImage is a 3D (video) VAE: add a temporal dim, encode, drop it back
    # into a [B,16,1,H,W] latent normalised by the per-channel latents_mean / latents_std.
    px = pixel_values.to(torch.float32).unsqueeze(2)  # [B,3,1,H,W]
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.sample()  # [B,16,1,h,w]
    mean, std = _qwen_latent_affine(vae, lat)
    return (lat - mean) / std


def _qwen_encode_latent_stats(vae, pixel_values):
    import torch

    px = pixel_values.to(torch.float32).unsqueeze(2)
    with torch.no_grad():
        dist = vae.encode(px).latent_dist
    mean, std = _qwen_latent_affine(vae, dist.mean)
    return (dist.mean - mean) / std, dist.std / std


def _qwen_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch
    import torch.nn.functional as F

    # Qwen embeds are variable-length: pad to the batch max (or a pinned ``pad_to`` bucket
    # under compile so the graph shape stays fixed) and batch the validity mask with them.
    seqs = [e[0].shape[1] for e in entries]
    target = max(pad_to or 0, max(seqs))
    pes, masks = [], []
    for pe, mask in entries:
        s = pe.shape[1]
        if mask is None:
            mask = torch.ones((1, s), dtype = torch.int64)
        if s < target:
            pe = F.pad(pe, (0, 0, 0, target - s))
            mask = F.pad(mask, (0, target - s))
        pes.append(pe)
        masks.append(mask)
    pe_b = torch.cat(pes).to(device = device, dtype = weight_dtype)
    mask_b = torch.cat(masks).to(device)
    # A single unpadded sample keeps the legacy None mask (identical math; avoids any
    # behaviour delta for existing single-image runs whose pipeline returned None).
    if len(entries) == 1 and entries[0][1] is None and target == seqs[0]:
        mask_b = None
    return (pe_b, mask_b)


def _qwen_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import QwenImagePipeline

    pe, mask = embeds_batch
    bsz, c, f, h, w = noisy.shape
    packed = QwenImagePipeline._pack_latents(noisy, bsz, c, h, w)
    # Each batch entry is a LIST of one (frame, h/2, w/2) tuple: the transformer indexes
    # sample[0] / sample[1:] per entry (transformer_qwenimage.py), so a flat list breaks it.
    img_shapes = [[(1, h // 2, w // 2)]] * bsz
    pred = transformer(
        hidden_states = packed,
        encoder_hidden_states = pe,
        encoder_hidden_states_mask = mask,
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
def _zimage_load_conditioners(cfg, device, weight_dtype):
    from diffusers import ZImagePipeline
    return _load_pipe_without_transformer(ZImagePipeline, cfg, device)


def _zimage_load_transformer(cfg, device, weight_dtype, base_precision):
    # Prequant default loads 4-bit as-is under nf4; the dense modes use the bf16 Tongyi-MAI
    # base. Z-Image is bf16 only (its RoPE/embedder run fp32; fp16 overflows).
    from diffusers import ZImageTransformer2DModel
    return _load_dit_transformer(ZImageTransformer2DModel, cfg, device, base_precision)


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


def _zimage_encode_latent_stats(vae, pixel_values):
    # Z-Image trains from the posterior mode (deterministic), so the cached entry is the
    # final latent itself: B is None and the loop skips the per-step sampling draw.
    return _zimage_encode_latents(vae, pixel_values), None


def _zimage_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    caps = [e[0].to(device = device, dtype = weight_dtype) for e in entries]
    return (caps,)


def _zimage_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    import torch

    (caps,) = embeds_batch
    # List I/O: one [C,1,H,W] latent + one [seq,2560] caption per sample. The timestep
    # convention is REVERSED ((1000 - t) / 1000) and the prediction is NEGATED.
    x_list = list(noisy.unsqueeze(2).unbind(dim = 0))
    t_norm = (1000 - timesteps) / 1000
    out = transformer(x_list, t_norm, list(caps), return_dict = False)[0]
    return -torch.stack(out, dim = 0).squeeze(2)


def _zimage_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import ZImagePipeline
    ZImagePipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


# ── Krea 2 ────────────────────────────────────────────────────────────────────
def _krea2_load_conditioners(cfg, device, weight_dtype):
    # The krea repo ships transformers-5.x style configs the pinned 4.x line can't parse, so the
    # conditioning pipeline is assembled per-component (with tokenizer and rope compat) rather
    # than from_pretrained(transformer = None); see diffusion_krea2.py for the compat story.
    import torch
    from core.inference.diffusion_krea2 import load_krea2_pipeline

    pipe = load_krea2_pipeline(
        cfg.base_model, torch.bfloat16, hf_token = cfg.hf_token, with_transformer = False
    )
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.vae


def _krea2_load_transformer(cfg, device, weight_dtype, base_precision):
    # The transformer subfolder is diffusers-format (no transformers compat needed).
    # There is no prequant repo yet, so nf4 quantizes the 12B transformer on the fly.
    from diffusers import Krea2Transformer2DModel
    return _load_dit_transformer(Krea2Transformer2DModel, cfg, device, base_precision)


def _krea2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            # encode_prompt pads/truncates to the fixed max_sequence_length, so every embed is
            # [1, 512, num_text_layers, 2560] with a [1, 512] validity mask -- static shapes
            # (padding sits mid-template, BEFORE the assistant suffix, matching how the model was
            # sampled at training time).
            pe, mask = pipe.encode_prompt(
                prompt = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 512,
            )
            out.append((pe.cpu(), mask.cpu()))
    return out


# Krea 2 conditions on the Qwen-Image VAE (AutoencoderKLQwenImage) with the same
# per-channel latents_mean / latents_std normalisation, so latent encoding is shared.
_krea2_encode_latents = _qwen_encode_latents
_krea2_encode_latent_stats = _qwen_encode_latent_stats


def _krea2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # Fixed-length embeds (see _krea2_encode_prompts), so collation is a plain concat
    # with the mask riding along; ``pad_to`` is moot because the shapes are static.
    pe_b = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    mask_b = torch.cat([e[1] for e in entries]).to(device)
    return (pe_b, mask_b)


def _krea2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import Krea2Pipeline

    pe, mask = embeds_batch
    # [B,16,1,H,W] -> [B, (H/2)*(W/2), 64] 2x2 patches. Krea2Pipeline._pack_latents /
    # _unpack_latents are instance methods (they read self.patch_size), so the packing is inlined
    # here like the reference DreamBooth script (patch_size = 2).
    bsz, c, _f, h, w = noisy.shape
    packed = noisy.reshape(bsz, c, h // 2, 2, w // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5).reshape(bsz, (h // 2) * (w // 2), c * 4)
    # Text tokens sit at the rotary origin, so one shared position grid serves the batch.
    position_ids = Krea2Pipeline.prepare_position_ids(pe.shape[1], h // 2, w // 2, device)
    pred = transformer(
        hidden_states = packed,
        encoder_hidden_states = pe,
        timestep = timesteps / 1000,
        position_ids = position_ids,
        encoder_attention_mask = mask,
        return_dict = False,
    )[0]
    pred = pred.view(bsz, h // 2, w // 2, c, 2, 2)
    pred = pred.permute(0, 3, 1, 4, 2, 5)
    return pred.reshape(bsz, c, 1, h, w)


def _krea2_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import Krea2Pipeline
    Krea2Pipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


# ── FLUX.2 (dev + Klein) ──────────────────────────────────────────────────────
# Both variants share Flux2Transformer2DModel and the packing/forward conventions of the
# upstream DreamBooth references (train_dreambooth_lora_flux2[_klein].py); they differ only
# in the conditioning stack (dev: Mistral-3-Small via Flux2Pipeline; Klein: Qwen3 via
# Flux2KleinPipeline) and size. Latents train in the PATCHIFIED, batch-norm-normalised space
# the references use (space-to-depth then (x - bn_mean) / bn_std from the VAE's running
# stats), from the posterior MODE (deterministic, so the latent cache stores the final
# latent and skips the per-step draw).
_FLUX2_TARGETS = (
    # Double-stream blocks: separate q/k/v plus the ModuleList out proj.
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    # Single-stream blocks: the fused qkv+mlp input projection carries the bulk of the
    # capacity. Their out proj is a PLAIN Linear named to_out whose suffix would also match
    # the double-stream ModuleList container (which peft cannot wrap), and the upstream
    # scripts enumerate it per block index (a variant-dependent count), so it stays dense.
    "to_qkv_mlp_proj",
)
# The references train dev with its guidance-distillation vector at 3.5 (Klein applies it
# only when the variant's config carries guidance_embeds).
_FLUX2_TRAIN_GUIDANCE = 3.5


def _flux2_load_conditioners(cfg, device, weight_dtype):
    from diffusers import Flux2Pipeline
    return _load_pipe_without_transformer(Flux2Pipeline, cfg, device)


def _flux2_klein_load_conditioners(cfg, device, weight_dtype):
    from diffusers import Flux2KleinPipeline
    return _load_pipe_without_transformer(Flux2KleinPipeline, cfg, device)


def _flux2_load_transformer(cfg, device, weight_dtype, base_precision):
    from diffusers import Flux2Transformer2DModel
    return _load_dit_transformer(Flux2Transformer2DModel, cfg, device, base_precision)


def _flux2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            # encode_prompt pads to max_sequence_length (padding="max_length"), so the
            # embeds are fixed-length and text_ids are per-caption [1, txt_len, 4]. The
            # per-class text_encoder_out_layers defaults differ (dev (10,20,30) vs Klein
            # (9,18,27)) and are left to the pipeline.
            pe, text_ids = pipe.encode_prompt(
                prompt = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 512,
            )
            out.append((pe.cpu(), text_ids.cpu()))
    return out


def _flux2_encode_latents(vae, pixel_values):
    import torch
    from diffusers import Flux2Pipeline

    with torch.no_grad():
        lat = vae.encode(pixel_values.to(torch.float32)).latent_dist.mode()
    lat = Flux2Pipeline._patchify_latents(lat)
    # The FLUX.2 VAE normalises latents with its BatchNorm running stats, not
    # shift/scaling_factor (mirrors the upstream reference).
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat)
    return (lat - mean) / std


def _flux2_encode_latent_stats(vae, pixel_values):
    # FLUX.2 trains from the posterior mode (deterministic): the cached entry is the final
    # latent itself; B is None and the loop skips the per-step sampling draw.
    return _flux2_encode_latents(vae, pixel_values), None


def _flux2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # Fixed-length embeds (padding="max_length" in encode_prompt) -> plain concat;
    # text_ids are PER-SAMPLE [1, txt_len, 4] (unlike FLUX.1's shared grid), so they batch
    # by concat too. ``pad_to`` is moot because the shapes are static.
    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = torch.cat([e[1] for e in entries]).to(device = device, dtype = torch.float32)
    return (pe, text_ids)


# Step-invariant FLUX.2 position ids, keyed on (batch, patched h, patched w, device): the ids
# derive only from the latent shape, so rebuilding them every step is allocator churn.
_FLUX2_STATIC: dict[tuple, Any] = {}


def _flux2_static_img_ids(latents, device):
    from diffusers import Flux2Pipeline

    key = (latents.shape[0], latents.shape[-2], latents.shape[-1], str(device))
    hit = _FLUX2_STATIC.get(key)
    if hit is None:
        hit = _FLUX2_STATIC[key] = Flux2Pipeline._prepare_latent_ids(latents).to(device = device)
    return hit


def _flux2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    import torch
    from diffusers import Flux2Pipeline

    pe, text_ids = embeds_batch
    # ``noisy`` is already in the patchified normalised space (_flux2_encode_latents), so
    # packing is the [B,C,H,W] -> [B, H*W, C] flatten the pipeline uses.
    packed = Flux2Pipeline._pack_latents(noisy)
    img_ids = _flux2_static_img_ids(noisy, device)
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.full(
            (noisy.shape[0],), _FLUX2_TRAIN_GUIDANCE, device = device, dtype = torch.float32
        )
    pred = transformer(
        hidden_states = packed,
        timestep = timesteps / 1000,
        guidance = guidance,
        encoder_hidden_states = pe,
        txt_ids = text_ids,
        img_ids = img_ids,
        return_dict = False,
    )[0]
    pred = pred[:, : packed.size(1)]
    # _unpack_latents_with_ids scatters per sample; diffusers 0.39 stacks the per-sample
    # tensors itself (its list annotation is stale), older/newer builds may hand back the
    # list. Same-resolution training batches align either way with target = noise - latents.
    unpacked = Flux2Pipeline._unpack_latents_with_ids(pred, img_ids)
    if isinstance(unpacked, (list, tuple)):
        unpacked = torch.stack(unpacked)
    return unpacked


def _flux2_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import Flux2Pipeline
    Flux2Pipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


def _flux2_klein_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import Flux2KleinPipeline
    Flux2KleinPipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


_SPECS: dict[str, _FamilySpec] = {
    "flux.1": _FamilySpec(
        family = "flux.1",
        lora_targets = _FLUX_TARGETS,
        force_bf16 = False,
        dense_bf16_gb = 23.8,
        load_conditioners = _flux_load_conditioners,
        load_transformer = _flux_load_transformer,
        encode_prompts = _flux_encode_prompts,
        encode_latents = _flux_encode_latents,
        encode_latent_stats = _flux_encode_latent_stats,
        collate = _flux_collate,
        forward = _flux_forward,
        save = _flux_save,
    ),
    "qwen-image": _FamilySpec(
        family = "qwen-image",
        lora_targets = _QWEN_TARGETS,
        force_bf16 = True,
        dense_bf16_gb = 41.0,
        load_conditioners = _qwen_load_conditioners,
        load_transformer = _qwen_load_transformer,
        encode_prompts = _qwen_encode_prompts,
        encode_latents = _qwen_encode_latents,
        encode_latent_stats = _qwen_encode_latent_stats,
        collate = _qwen_collate,
        forward = _qwen_forward,
        save = _qwen_save,
    ),
    "z-image": _FamilySpec(
        family = "z-image",
        lora_targets = _ZIMAGE_TARGETS,
        force_bf16 = True,
        dense_bf16_gb = 12.3,
        load_conditioners = _zimage_load_conditioners,
        load_transformer = _zimage_load_transformer,
        encode_prompts = _zimage_encode_prompts,
        encode_latents = _zimage_encode_latents,
        encode_latent_stats = _zimage_encode_latent_stats,
        collate = _zimage_collate,
        forward = _zimage_forward,
        save = _zimage_save,
    ),
    "krea-2": _FamilySpec(
        family = "krea-2",
        lora_targets = _KREA2_TARGETS,
        force_bf16 = True,
        dense_bf16_gb = 26.3,
        load_conditioners = _krea2_load_conditioners,
        load_transformer = _krea2_load_transformer,
        encode_prompts = _krea2_encode_prompts,
        encode_latents = _krea2_encode_latents,
        encode_latent_stats = _krea2_encode_latent_stats,
        collate = _krea2_collate,
        forward = _krea2_forward,
        save = _krea2_save,
    ),
    "flux.2-klein": _FamilySpec(
        family = "flux.2-klein",
        lora_targets = _FLUX2_TARGETS,
        # The upstream references train in bf16; fp16 is unvalidated on the FLUX.2 stack.
        force_bf16 = True,
        dense_bf16_gb = 8.1,
        load_conditioners = _flux2_klein_load_conditioners,
        load_transformer = _flux2_load_transformer,
        encode_prompts = _flux2_encode_prompts,
        encode_latents = _flux2_encode_latents,
        encode_latent_stats = _flux2_encode_latent_stats,
        collate = _flux2_collate,
        forward = _flux2_forward,
        save = _flux2_klein_save,
    ),
    "flux.2-dev": _FamilySpec(
        family = "flux.2-dev",
        lora_targets = _FLUX2_TARGETS,
        force_bf16 = True,
        # 32B DiT; the Mistral conditioning stack (~46 GB bf16) is loaded, encoded, and
        # freed BEFORE this lands on the device (the shared phased load).
        dense_bf16_gb = 64.5,
        load_conditioners = _flux2_load_conditioners,
        load_transformer = _flux2_load_transformer,
        encode_prompts = _flux2_encode_prompts,
        encode_latents = _flux2_encode_latents,
        encode_latent_stats = _flux2_encode_latent_stats,
        collate = _flux2_collate,
        forward = _flux2_forward,
        save = _flux2_save,
    ),
}


# HF repos that gate access behind a license acceptance: training needs a token whose account
# accepted the license. Checked by name (no network) so a missing token fails fast with an
# actionable message instead of a confusing 401 mid-load.
_GATED_TRAIN_REPOS = frozenset(
    {"black-forest-labs/flux.1-dev", "black-forest-labs/flux.2-dev"}
)


def _assert_gated_access(base_model: str, hf_token: Optional[str]) -> None:
    """Raise a clear error before loading a gated base without a token."""
    name = str(base_model or "").strip().lower()
    if name in _GATED_TRAIN_REPOS and not (hf_token and str(hf_token).strip()):
        raise ValueError(
            f"'{base_model}' is a gated Hugging Face repo. Accept its license on the Hub "
            f"and add your HF token in Studio settings before training from it."
        )


def _open_resized(path, resolution):
    """Open + EXIF-orient + short-side resize to ``resolution`` (same geometry as the SDXL
    loader). Returns the resized PIL image and its (rw, rh)."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w0, h0 = img.size
    scale = resolution / min(w0, h0)
    rw, rh = max(resolution, round(w0 * scale)), max(resolution, round(h0 * scale))
    return img.resize((rw, rh), Image.LANCZOS), rw, rh


def _to_unit_tensor(img):
    import numpy as np
    import torch

    arr = np.asarray(img, dtype = np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def _load_pixel_tensor(path, resolution, center_crop, random_flip, rng):
    """Load an image -> a normalised [3,H,W] tensor in [-1,1]. Same geometry as the SDXL
    loader but without the SDXL time-ids (DiT families don't use them)."""
    from PIL import Image

    img, rw, rh = _open_resized(path, resolution)
    if center_crop:
        left, top = (rw - resolution) // 2, (rh - resolution) // 2
    else:
        left = rng.randint(0, max(0, rw - resolution))
        top = rng.randint(0, max(0, rh - resolution))
    img = img.crop((left, top, left + resolution, top + resolution))
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return _to_unit_tensor(img)


def _load_pixel_tensor_planned(path, resolution, center_crop, u_left, u_top, flip):
    """Deterministic variant of ``_load_pixel_tensor`` for the latent cache: the crop comes
    as unit fractions (mapped uniformly over the same inclusive integer range ``randint``
    draws from) and the flip as a bool. ``center_crop`` reproduces the exact legacy
    floor-div center so a cached center-crop run matches the uncached one bit-for-bit."""
    from PIL import Image

    img, rw, rh = _open_resized(path, resolution)
    if center_crop:
        left, top = (rw - resolution) // 2, (rh - resolution) // 2
    else:
        left = min(int(u_left * (rw - resolution + 1)), max(0, rw - resolution))
        top = min(int(u_top * (rh - resolution + 1)), max(0, rh - resolution))
    img = img.crop((left, top, left + resolution, top + resolution))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return _to_unit_tensor(img)


def _build_latent_cache(spec, vae, image_paths, cfg, device, weight_dtype, on_event, check_stop):
    """Precompute the per-image latent posterior cache: for each planned crop/flip variant,
    encode once and store the affine (A, B) pair on CPU (pinned when possible) in fp32. The
    stats stay fp32 so the per-step sample happens in fp32 and only the RESULT is cast to
    weight_dtype, matching the in-loop path (encode fp32 -> sample/normalise fp32 ->
    .to(weight_dtype)); fp32 doubles the cache RAM over bf16 but the cache is tiny (a handful
    of latents per image). Returns None if the build was interrupted by a stop request."""

    plan = _plan_cache_variants(
        len(image_paths), cfg.cache_variants, cfg.center_crop, cfg.random_flip, cfg.seed
    )

    def _hold(t):
        if t is None:
            return None
        import torch

        t = t.to(torch.float32).cpu()
        if device == "cuda":
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        return t

    cache: list[list[tuple]] = []
    total = len(image_paths)
    total_variants = sum(len(v) for v in plan)
    forced = _latent_cache_forced()
    gated = False
    for i, path in enumerate(image_paths):
        variants = []
        for u_left, u_top, flip in plan[i]:
            px = (
                _load_pixel_tensor_planned(
                    path, cfg.resolution, cfg.center_crop, u_left, u_top, flip
                )
                .unsqueeze(0)
                .to(device)
            )
            a, b = spec.encode_latent_stats(vae, px)
            a, b = _hold(a), _hold(b)
            if not forced and not gated:
                # Size-gate the automatic cache off the first REAL encoded variant, before
                # building the rest: packed 16-channel DiT latents x variants x images of two fp32
                # tensors can exhaust host/pinned RAM. Over budget we bail with the VAE resident so
                # the loop encodes per step. ``b`` is None for a deterministic-latent family, so
                # only ``a`` contributes bytes there.
                per_variant = a.numel() * a.element_size()
                if b is not None:
                    per_variant += b.numel() * b.element_size()
                if _latent_cache_over_budget(per_variant, total_variants):
                    _emit(
                        on_event,
                        "warning",
                        message = (
                            "Latent cache disabled: estimated "
                            f"{per_variant * total_variants / 1024 ** 3:.1f} GiB over the "
                            "budget; encoding latents per step instead. Set "
                            "UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE=1 to keep it."
                        ),
                    )
                    return LATENT_CACHE_OVER_BUDGET
                gated = True
            variants.append((a, b))
        cache.append(variants)
        if (i + 1) % 4 == 0 or i + 1 == total:
            _emit(on_event, "preparing", stage = "cache_latents", done = i + 1, total = total)
        if check_stop():
            return None
    return cache


def _sample_cached_latents(cache, idxs, variant_rng, device, weight_dtype):
    """Draw one latent per index from the cache: pick a variant, then sample the posterior
    (A + B * randn) when the family is stochastic. Fresh noise per step, exactly like an
    in-loop ``latent_dist.sample()``. The cached stats are fp32, so the sample is drawn in
    fp32 and only the RESULT is cast to weight_dtype (matching the in-loop path's
    ``encode_latents(...).to(weight_dtype)``)."""
    import torch

    parts_a, parts_b = [], []
    for i in idxs:
        variants = cache[i]
        a, b = variants[variant_rng.randrange(len(variants))] if len(variants) > 1 else variants[0]
        parts_a.append(a)
        parts_b.append(b)
    lat_a = torch.cat(parts_a).to(device, non_blocking = True)
    if parts_b[0] is None:
        return lat_a.to(dtype = weight_dtype)
    lat_b = torch.cat(parts_b).to(device, non_blocking = True)
    return (lat_a + lat_b * torch.randn_like(lat_a)).to(dtype = weight_dtype)


def _should_compile(
    cfg,
    base_is_bnb,
    device,
    base_precision = "nf4",
) -> bool:
    mode = (cfg.compile_transformer or "auto").strip().lower()
    if device != "cuda" or mode == "off":
        return False
    # torch.compile cannot trace the torchao int8 subclass in training (inductor rejects
    # the aliased subclass graph outputs), so int8 always runs eager.
    if base_precision == "int8":
        return False
    if mode == "on":
        return True
    # auto: regional compile is the whole point of the dense modes (measured 2.6x on Z-Image
    # bf16) but fragile over bitsandbytes 4-bit modules (graph breaks in the dequant path), so it
    # stays off for QLoRA. fp8/mxfp8 are only competitive compiled (eager, their per-matmul
    # dynamic casts run 4-5x slower than bf16).
    return base_precision in ("bf16", "fp8", "mxfp8")


def _maybe_compile_transformer(
    transformer,
    cfg,
    base_is_bnb,
    device,
    on_event,
    base_precision = "nf4",
) -> bool:
    """Regionally compile the transformer blocks (diffusers compile_repeated_blocks) after
    the LoRA is attached. Never fatal: a wrap failure falls back to eager with a warning
    event, and dynamo's suppress_errors keeps a frame that fails to COMPILE at the first
    step running eager instead of raising mid-run."""
    if not _should_compile(cfg, base_is_bnb, device, base_precision):
        if base_precision in ("fp8", "mxfp8"):
            _emit(
                on_event,
                "warning",
                message = (
                    f"{base_precision} training without torch.compile is slow; "
                    f"enable compile for the speedup."
                ),
            )
        return False
    import torch

    fn = getattr(transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        _emit(
            on_event, "warning", message = "torch.compile unavailable for this model; running eager."
        )
        return False
    try:
        dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_cfg is not None:
            # Heterogeneous-block DiTs (Z-Image: ~11 distinct block shapes) exceed dynamo's
            # default recompile limit of 8; bump it like the inference speed layer does
            # (diffusers' documented regional-compile fix).
            for attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo_cfg, attr):
                    setattr(dynamo_cfg, attr, max(getattr(dynamo_cfg, attr) or 0, 64))
            if hasattr(dynamo_cfg, "suppress_errors"):
                dynamo_cfg.suppress_errors = True
        # dynamic=True matches the inference speed layer's proven default: on torch 2.10 / B200
        # the dynamic=False specialisation fused a gemm_and_bias epilogue that failed with
        # CUBLAS_STATUS_EXECUTION_FAILED then an illegal memory access on the FLUX training graph.
        # fullgraph only on a dense base: bnb 4-bit layers graph-break by design.
        fn(fullgraph = not base_is_bnb, dynamic = True)
        return True
    except Exception as exc:  # noqa: BLE001 -- optimisation only, never fatal
        _emit(on_event, "warning", message = f"torch.compile disabled (eager fallback): {exc}")
        return False


def run_dit_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """Train a flow-matching DiT LoRA (FLUX.1 / FLUX.2 / Qwen-Image / Z-Image / Krea 2) and export it."""
    cfg = config.normalized()
    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        raise ValueError(f"No DiT trainer for family {cfg.resolved_family!r}")

    # DiT families train in bf16 (Z-Image/Qwen require it; FLUX prefers it). An explicit fp16
    # request on a bf16-only family is refused, not silently upgraded, so the choice is never
    # misrepresented. Validation runs before the heavy imports so a host without diffusers still
    # sees the real error.
    if cfg.mixed_precision == "fp16" and spec.force_bf16:
        raise ValueError(
            f"{spec.family} LoRA training requires bf16: fp16 overflows its fp32 RoPE / "
            f"embedder internals. Set mixed precision to bf16."
        )

    import torch

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    _FLUX_STATIC.clear()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The flow-matching + 4-bit path is bf16 throughout (fp32 on a CPU-only box: unsupported for
    # real runs but keeps import/unit tests architecture-agnostic). Fail fast on pre-Ampere CUDA
    # (T4/V100/RTX 20xx): bf16 is required and the run would otherwise die deep in model load with
    # an opaque dtype error. Gate on NATIVE bf16 (capability major >= 8) -- is_bf16_supported()
    # counts pre-Ampere emulation as supported, which this guard rejects; shared with /info modes
    # + start preflight.
    if device == "cuda" and not native_bf16_supported():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU (Ampere or newer); "
            "this CUDA device does not support bf16."
        )
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _assert_trusted_base_model(cfg.base_model)
    _assert_gated_access(cfg.base_model, cfg.hf_token)
    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    # Resolve num_epochs -> a concrete train_steps now the dataset size is known, and rebind cfg
    # so every downstream read (scheduler length, loop range, progress total_steps, steps_run)
    # sees the same value.
    cfg = replace(cfg, train_steps = resolve_train_steps(cfg, len(pairs)), num_epochs = 0)
    _emit(on_event, "model_load_started", num_images = len(pairs))
    if _check_stop():
        out_dir = Path(cfg.output_dir).expanduser()
        _emit(
            on_event, "complete", output_dir = str(out_dir), lora_path = None, stopped = True, steps_run = 0
        )
        return str(out_dir)

    # TF32 / cudnn.benchmark for the run, restored on the way out (the trainer subprocess is
    # disposable, but restoring keeps in-process callers -- tests, notebooks -- clean).
    perf_snap = _apply_perf_flags(cfg, device)
    try:
        return _train_dit(
            cfg,
            spec,
            pairs,
            rng,
            device,
            weight_dtype,
            on_event,
            _check_stop,
            lambda: save_on_stop,
        )
    finally:
        _restore_perf_flags(perf_snap)


def _train_dit(cfg, spec, pairs, rng, device, weight_dtype, on_event, _check_stop, _save_on_stop):
    """The body of ``run_dit_lora_training``, split out so the backend perf flags are
    snapshot/restored around it in exactly one place."""
    import torch
    import torch.nn.functional as F
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    use_lora_targets = _select_lora_targets(cfg.lora_target_modules, spec.lora_targets)
    out_dir = Path(cfg.output_dir).expanduser()

    # Phase 1: conditioning only. The pipeline loads WITHOUT its transformer, so the text
    # encoders + VAE never share VRAM with the multi-GB denoiser.
    pipe, vae = spec.load_conditioners(cfg, device, weight_dtype)

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

    # Phase 2: the VAE latent cache, then free the VAE too (see module docstring: the cache
    # keeps the posterior affine parameters, so per-step sampling noise is preserved).
    use_cache = cfg.cache_latents and os.environ.get(
        "UNSLOTH_DIFFUSION_NO_LATENT_CACHE", ""
    ) not in ("1", "true")
    latent_cache = None
    if use_cache:
        latent_cache = _build_latent_cache(
            spec, vae, image_paths, cfg, device, weight_dtype, on_event, _check_stop
        )
        if latent_cache is LATENT_CACHE_OVER_BUDGET:
            # The estimated cache exceeded the host-memory budget; keep the VAE resident and
            # fall through to the in-loop encode path (latent_cache stays None).
            latent_cache = None
        elif latent_cache is None:  # stopped during the cache build; nothing trained yet
            _emit(
                on_event,
                "complete",
                output_dir = str(out_dir),
                lora_path = None,
                stopped = True,
                steps_run = 0,
            )
            return str(out_dir)
        else:
            try:
                pipe.vae = None
            except Exception:  # noqa: BLE001 -- a pipeline without a settable vae keeps it
                pass
            del vae
            vae = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    # Variant picks use their own stream so the training loop's index/noise draws stay on
    # the same seed-deterministic sequence whether or not the cache is enabled.
    variant_rng = random.Random(cfg.seed + 1)

    # Phase 3: only now load the transformer, in the resolved base precision (nf4 QLoRA by
    # default; bf16 / int8 / fp8 / mxfp8 are the dense speed modes; "auto" picks from free VRAM
    # measured before the load).
    base_precision = _resolve_base_precision(cfg, spec, device)
    transformer = spec.load_transformer(cfg, device, weight_dtype, base_precision)
    base_is_bnb = base_precision == "nf4"

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
        # Non-reentrant checkpointing: reentrant recompute of a bnb 4-bit LoRA linear can trip
        # an illegal memory access on the larger FLUX transformer, and non-reentrant is the
        # recommended mode anyway (it also handles a checkpointed segment whose inputs don't
        # require grad, which happens with a frozen 4-bit base).
        import functools
        import torch.utils.checkpoint as _ckpt
        transformer.enable_gradient_checkpointing(
            gradient_checkpointing_func = functools.partial(_ckpt.checkpoint, use_reentrant = False)
        )
    cast_training_params(transformer, dtype = torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    # int8 / fp8 / mxfp8 convert the frozen base linears AFTER the LoRA attaches, so the
    # adapter modules are excluded and stay high precision.
    if base_precision == "int8":
        _int8_quantize_base(transformer)
    if base_precision == "fp8" and not _apply_fp8_training(transformer, on_event):
        base_precision = "bf16"
    if base_precision == "mxfp8" and not _apply_mxfp8_training(transformer, on_event):
        base_precision = "bf16"

    compiled = _maybe_compile_transformer(
        transformer, cfg, base_is_bnb, device, on_event, base_precision
    )
    # Compiled Qwen graphs need one fixed text length across steps: pin the pad bucket to
    # the dataset's longest caption (encode_prompt already caps it at 1024 tokens).
    qwen_pad_to = None
    if compiled and spec.family == "qwen-image":
        qwen_pad_to = max(e[0].shape[1] for e in caption_embeds.values())

    optimizer = _make_optimizer(lora_params, cfg.learning_rate)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model, subfolder = "scheduler", token = cfg.hf_token
    )
    # The LR schedule advances once per optimizer update, so warmup/decay are counted in
    # optimizer steps (matching the SDXL trainer; multiplying by the accumulation factor would
    # stretch warmup past the run and never reach the decay).
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )

    _emit(on_event, "model_load_completed", compiled = compiled, base_precision = base_precision)

    transformer.train()
    n_images = len(image_paths)
    batch_size = cfg.train_batch_size
    # Permutation-cycle index sampler (shared with the SDXL trainer): visits every image once per
    # cycle before repeating, so a short run covers the whole dataset instead of the old
    # with-replacement draw. Uses the loop's own rng to stay seed-deterministic.
    index_sampler = PermutationBatchSampler(n_images, rng)
    stopped = False
    running_loss = 0.0
    peak_gb = 0.0
    t_start = time.time()
    t_steady = None
    done = 0
    # bf16 autocast around the forward + loss, matching the diffusers dreambooth scripts'
    # accelerator.autocast: reconciles the fp32 LoRA params with the bnb 4-bit base matmuls in
    # one compute dtype. Without it the 4-bit backward on FLUX dies with an illegal-address /
    # CUBLAS failure.
    autocast = (
        torch.autocast(device_type = "cuda", dtype = torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none = True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            idxs = index_sampler.next_batch(batch_size)
            if latent_cache is not None:
                latents = _sample_cached_latents(
                    latent_cache, idxs, variant_rng, device, weight_dtype
                )
            else:
                px = torch.stack(
                    [
                        _load_pixel_tensor(
                            image_paths[i], cfg.resolution, cfg.center_crop, cfg.random_flip, rng
                        )
                        for i in idxs
                    ]
                ).to(device)
                latents = spec.encode_latents(vae, px).to(weight_dtype)

            noise = torch.randn_like(latents)
            timesteps, t_indices = _sample_timesteps(scheduler, latents.shape[0], device)
            sigmas = _gather_sigmas(scheduler, t_indices, device, weight_dtype, latents.ndim)
            noisy = (1.0 - sigmas) * latents + sigmas * noise

            embeds = spec.collate(
                [caption_embeds[captions[i]] for i in idxs],
                device,
                weight_dtype,
                pad_to = qwen_pad_to,
            )
            with autocast:
                model_pred = spec.forward(
                    transformer, noisy, timesteps, sigmas, embeds, cfg, device, weight_dtype
                )
                target = noise - latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps

        grad_norm = None
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            # clip_grad_norm_ returns the total PRE-clip norm: the health signal the UI
            # charts (an exploding norm shows up here even while the clip caps the update).
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
        optimizer.step()
        lr_sched.step()

        running_loss += step_loss
        done = opt_step + 1
        now = time.time()
        if done == 1:
            # Step 1 pays the one-time costs (cudnn autotune, torch.compile warmup), so the
            # reported rate starts after it and reflects the steady state.
            t_steady = now
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            if device == "cuda":
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            per_step = batch_size * cfg.gradient_accumulation_steps
            if t_steady is not None and done > 1:
                sps = round((done - 1) * per_step / max(now - t_steady, 1e-6), 3)
            else:
                sps = round(done * per_step / max(now - t_start, 1e-6), 3)
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                learning_rate = lr_sched.get_last_lr()[0],
                grad_norm = round(grad_norm, 5) if grad_norm is not None else None,
                samples_per_second = sps,
                peak_memory_gb = peak_gb or None,
            )
        if _check_stop():
            stopped = True
            break

    lora_path: Optional[str] = None
    catalog_path: Optional[str] = None
    if not (stopped and not _save_on_stop()):
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
        wall_seconds = round(time.time() - t_start, 1),
    )
    return str(out_dir)


def _make_optimizer(params, lr):
    """8-bit AdamW (bitsandbytes) when available -- half the optimizer state, no accuracy
    regression for LoRA -- else torch AdamW, fused on CUDA (with a fallback when this
    build/device lacks the fused kernel)."""
    import torch

    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr = lr)
    except Exception:  # noqa: BLE001 -- bnb missing / no CUDA: fall back to torch AdamW
        pass
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr = lr, fused = True)
        except Exception:  # noqa: BLE001 -- fused unsupported on this build/device
            pass
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
