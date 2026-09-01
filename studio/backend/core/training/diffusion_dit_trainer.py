# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flow-matching LoRA training for the rectified-flow DiT families.

Image: FLUX.1-dev, FLUX.2 (dev + Klein), Qwen-Image, Z-Image, Krea 2. Video: LTX-2 (from a
still-image dataset; see the ``_LTX2_TARGETS`` block for what a video family adds).

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
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core._torchao_stub import is_stubbed, torch_is_rocm
from core.training.diffusion_train_common import (
    AUTO_FLOW_SHIFT_FAMILIES,
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
    restore_resume_state,
    write_resume_checkpoint,
)
from core.training.diffusion_checkpoint import (
    clear_own_checkpoints,
    discard_preexisting_checkpoints,
    retire_own_checkpoints,
    resumed_into_this_dir,
    snapshot_checkpoints,
    identity_for_config,
    with_cache_mode,
    with_resolved_base_precision,
    with_resolved_revision,
    preflight_resume,
)
from core.training.diffusion_train_extras import (
    LoRAEMA,
    PersistentConditioningCache,
    save_ema_adapter,
)

# Per-family LoRA target modules (attention projections). FLUX / Qwen double-stream blocks also
# carry added-kv projections; Z-Image is single-stream.
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
# The Krea 2 authors' recommended defaults (their DreamBooth script): attention + SwiGLU + text-
# fusion projector + conditioning embedders. For long runs they suggest narrowing to attention.
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
    # Used by base_precision="auto" to pick a mode that fits free VRAM; a family covering more than one
    # size (flux.2-klein is 4B and 9B) must not be sized off this alone -- see _dense_bf16_gb.
    dense_bf16_gb: float
    # Phased load so the multi-GB transformer never coexists with the text encoders + VAE:
    # load_conditioners builds the pipeline WITHOUT it and returns (pipe, vae), then load_transformer
    # loads it alone once the conditioners are freed.
    load_conditioners: Callable[..., tuple[Any, Any]]
    load_transformer: Callable[..., Any]
    # Encode a list of captions -> a per-caption tuple of CPU tensors (the family's embeds).
    encode_prompts: Callable[..., list[tuple]]
    # Encode a pixel tensor [B,3,H,W] in [-1,1] -> latents (family-normalised, on device).
    encode_latents: Callable[..., Any]
    # Encode pixels to (A, B) affine posterior params so a per-step sample is A + B * randn. B is None
    # for a deterministic family.
    encode_latent_stats: Callable[..., tuple]
    # Collate per-caption embed tuples into one batched tuple on device. ``pad_to`` pins a fixed text
    # length for variable-length embeds (compile).
    collate: Callable[..., tuple]
    # One transformer forward: (transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype)
    # -> model_pred aligned with target = noise - latents.
    forward: Callable[..., Any]
    # Save the LoRA in diffusers format via the family pipeline's save_lora_weights.
    save: Callable[..., None]


def _gather_sigmas(sigma_table, indices, device, dtype, n_dim):
    """Gather per-sample sigmas for schedule ``indices`` and broadcast to ``n_dim``.
    Index-based (no per-item search): ``indices`` are the positions ``_sample_timesteps``
    drew from ``scheduler.timesteps``, and ``sigma_table`` (the scheduler's own sigmas, or
    the shifted copy from ``_training_sigma_table``) is aligned with it, so the identity
    table returns exactly what the diffusers ``get_sigmas`` helper would."""
    sigma = sigma_table[indices].to(device = device, dtype = dtype).flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _training_sigma_table(scheduler, flow_shift):
    """The sigma table training draws index into, per ``cfg.flow_shift``.

    ``1.0`` (every non-Qwen family's default) returns ``scheduler.sigmas`` unchanged: the
    historical behavior, correct for the families whose schedule already matches training
    convention. ``"auto"`` (the qwen-image default) reproduces the family's INFERENCE sigma
    distribution, which the scheduler never bakes into ``sigmas`` when
    ``use_dynamic_shifting`` is true (the static ``shift`` at init is skipped, so Qwen-Image
    otherwise trains on unshifted uniform sigmas): apply the scheduler's own ``time_shift``
    at mu = ``max_shift`` (Qwen pins base_shift = max_shift = log 3, so the inference mu is
    constant at every resolution) followed by its ``stretch_shift_to_terminal`` -- using the
    scheduler's methods keeps the transform faithful across diffusers versions. A numeric
    value applies the standard linear shift s*u/(1+(s-1)*u) (musubi/kohya style
    discrete_flow_shift)."""
    sigmas = scheduler.sigmas
    if flow_shift == "auto":
        sc = scheduler.config
        if not getattr(sc, "use_dynamic_shifting", False):
            return sigmas  # static-shift family: its init already baked the shift in
        mu = float(getattr(sc, "max_shift", None) or math.log(3.0))
        shifted = scheduler.time_shift(mu, 1.0, sigmas)
        if getattr(sc, "shift_terminal", None):
            # The stretch scales off the table's own final sigma, so it must see the full descending schedule,
            # never a batch.
            shifted = scheduler.stretch_shift_to_terminal(shifted)
        return shifted
    s = float(flow_shift)
    if s == 1.0:
        return sigmas
    return s * sigmas / (1.0 + (s - 1.0) * sigmas)


def _bell_loss_weights(num_train_timesteps):
    """bsmntw-style bell loss-weight table over the training schedule: a Gaussian bell
    centered mid-schedule, floored at 0 and normalized to mean 1 (so the expected loss
    scale is unchanged). Indexed by round(sigma * num_train_timesteps)."""
    import torch

    steps = num_train_timesteps
    t = torch.arange(steps, dtype = torch.float32)
    w = torch.exp(-2.0 * ((t - steps / 2) / steps) ** 2)
    w = w - w.min()
    return w * (steps / w.sum())


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
            pass


def _bnb_4bit_config():
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    import torch
    return DiffusersBnb(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
        # Double quantization compresses the per-block absmax scales with a second 8-bit pass: ~0.4 bits/param
        # off the frozen base at no fidelity cost, which matters most on the 20B+ DiTs.
        bnb_4bit_use_double_quant = True,
    )


# The heuristic moved to diffusion_train_common so config validation can use it without importing
# this module; the name is kept for existing callers.
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

    # int8 quantizes AFTER the LoRA attaches: quantizing first makes peft dispatch TorchaoLoraLinear,
    # whose peft-0.18 constructor rejects the torchao-0.16 config API.
    return transformer_cls.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    ).to(device)


def _int8_quantize_base(transformer, family: Optional[str] = None) -> None:
    """torchao weight-only int8 on the big frozen linears, applied after add_adapter so
    the base_layer inside each LoRA wrapper quantizes while the adapters stay high
    precision. ``make_filter_fn`` (shared with the inference quant layer) keeps only
    Linears with >= 512 features -- which also naturally skips the rank-sized LoRA
    matrices -- and drops the M=1 modulation projections int8 kernels reject.

    ``family`` selects the per-family small-M exclusions on top of those. Passing it is what
    keeps training and inference on the same list: without it LTX-2's one-token audio stream
    (and Qwen-Image's unpadded text stream) is quantized here and the first forward raises,
    after the whole base has been loaded.

    The family's PAD list is applied for the same reason, and the same way the inference path
    applies it: a small-M Linear is either excluded or padded, and a family that chose padding
    (MiniMax-H3's context_embedder and token_refiner blocks) has those names in neither the
    generic nor the family exclusions, so without this they are quantized bare and raise on the
    first forward exactly like an unexcluded one."""
    from core.inference.diffusion_transformer_quant import (
        apply_small_m_padding,
        exclude_tokens_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    quantize_(
        transformer,
        Int8WeightOnlyConfig(),
        filter_fn = make_filter_fn(
            512, exclude_name_tokens = exclude_tokens_for_scheme("int8", family)
        ),
    )
    # After quantize_, as the helper requires (it reparents the Linears it wraps). Not best-effort: a
    # raise means the base is quantized but not safely runnable.
    apply_small_m_padding(transformer, "int8", family)


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


def _fp8_training_config():
    """The float8 training config: per-ROW (rowwise) scaling when this torchao build ships
    the recipe, else the tensorwise default. The DiT families carry extreme activation
    outliers (Z-Image MLP activations peak near 6.6e4, the same range that forced per-row
    scaling in the inference quant layer): one tensor-wide dynamic scale pushes normal
    values (~1-30) below fp8 resolution, so the frozen base's forward -- the signal the
    LoRA regresses against -- degrades. Per-row scaling confines each outlier to its own
    token/channel. The tensorwise fallback keeps pad_inner_dim so a non-16-aligned inner
    dim never aborts the scaled_mm (the rowwise recipe manages its own padding rules and
    rejects the knob, hence the split)."""
    from torchao.float8 import Float8LinearConfig
    try:
        return Float8LinearConfig.from_recipe_name("rowwise")
    except Exception:  # noqa: BLE001 -- older torchao without the rowwise recipe
        return Float8LinearConfig(pad_inner_dim = True)


def _apply_fp8_training(transformer, on_event) -> bool:
    """Convert the frozen base linears to torchao float8 training compute (dynamic scaling;
    weights stay bf16 in memory). Applied AFTER add_adapter so the filter can exclude the
    LoRA modules. Never fatal: on any failure the run continues in bf16 with a warning."""
    try:
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(
            transformer,
            module_filter_fn = _fp8_module_filter,
            config = _fp8_training_config(),
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
    # Skip biased linears: the torchao 0.17 MX path swaps the weight for a wrapper that drops the bias,
    # changing the output the LoRA regresses against.
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


def _dense_bf16_gb(spec, base_model: str) -> float:
    """The dense-bf16 transformer size of the base this run actually trains from.

    ``spec.dense_bf16_gb`` is one number per FAMILY, and flux.2-klein is a family with two
    transformer sizes under it: the 4B default (8.1 GB) and the 9B / base-9B pair (18.2 GB).
    Sizing a 9B run off the family number understates it by 2.3x, so base_precision="auto"
    (the mode /info recommends, and therefore the Train tab's default) resolves to bf16 on a
    16-24 GB GPU and the dense load OOMs -- the run fails before the first step.

    The inference auto-policy already keeps per-base overrides for exactly this, so read them
    here instead of adding a second table that can drift. Reads the per-base OVERRIDES only,
    never the family table underneath them: this spec's own number is the family default, and
    the two are independently maintained, so falling through to the shared family entry would
    silently re-size every base that has no override (klein's 4B default from 8.1 to 7.8, which
    moves the bf16 band by half a GB). Never raises: a sizing lookup must not be able to fail a
    run that would otherwise train."""
    try:
        from core.inference.diffusion_auto_policy import base_repo_bf16_components_gb
        components = base_repo_bf16_components_gb(base_model)
        if components:
            return float(components[0])
    except Exception:  # noqa: BLE001 -- table miss / import failure -> the family number
        pass
    return float(spec.dense_bf16_gb)


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
        # Mirror the pre-eviction ROCm guard inside the trainer child.
        if mode in ("int8", "fp8", "mxfp8") and torch_is_rocm():
            raise ValueError(
                f"base_precision={mode!r} is a torchao NVIDIA tensor-core path (int8 sm_80+, fp8 "
                "sm_89+, mxfp8 sm_100+) and this is a ROCm/AMD GPU. Use base_precision='nf4', "
                "'bf16', or 'auto'."
            )
        # int8 has no runtime fallback, so gate on a FUNCTIONAL torchao here as the auto pick and /info do;
        # find_spec is satisfied by the Windows-ROCm stub.
        if mode == "int8" and not has_functional_torchao():
            raise ValueError(
                "base_precision='int8' needs a functional torchao install; this host's "
                "torchao is missing or the non-functional Windows-ROCm stub. Use "
                "base_precision='nf4', 'bf16', or 'auto'."
            )
        # The stub answers torchao.float8 / torchao.prototype.mx_formats with a no-op that reports success,
        # so the run would report fp8 while training bf16.
        # Keyed on the stub, not has_functional_torchao(): that probes int8's symbols, and a real-but-
        # partial torchao must still reach the arch checks below.
        if mode in ("fp8", "mxfp8") and is_stubbed("torchao"):
            raise ValueError(
                f"base_precision={mode!r} is not available on this host: torchao is the "
                "non-functional Windows-ROCm stub. Use base_precision='nf4', 'bf16', or 'auto'."
            )
        # mxfp8 needs Blackwell (sm100+): its MX GEMM raises at the first training step, after a full dense
        # load. Re-check here to fail fast for a stale client.
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
    # auto may only resolve to the dense modes when the run uses bf16 compute, mirroring the
    # normalized() rule for explicit dense modes; otherwise stay on the nf4 floor.
    if getattr(cfg, "mixed_precision", "bf16") != "bf16":
        return "nf4"
    prequant = repo_is_prequantized(cfg.base_model)
    free_gb = None
    capability = None
    has_fp8 = False
    # Auto must not select a torchao mode on ROCm.
    has_torchao = has_functional_torchao() and not torch_is_rocm()
    if device == "cuda":
        try:
            import torch

            # Windows ROCm over-reports free VRAM (#8403), which would pick a dense precision the card cannot
            # actually hold.
            from utils.hardware import trusted_mem_get_info

            free_gb = trusted_mem_get_info()[0] / 1e9
            capability = torch.cuda.get_device_capability()
            has_fp8 = hasattr(torch, "float8_e4m3fn")
        except Exception:  # noqa: BLE001 -- probe failure -> the safe mode
            pass
    return _pick_auto_precision(
        prequant,
        device,
        free_gb,
        _dense_bf16_gb(spec, cfg.base_model),
        capability,
        has_fp8,
        has_torchao,
    )


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

    # FLUX embeds are fixed-length, so a plain cat batches them; text_ids are shared position ids,
    # identical across prompts.
    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    pooled = torch.cat([e[1] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = entries[0][2].to(device = device, dtype = torch.float32)
    return (pe, pooled, text_ids)


# Per-run cache of the step-invariant FLUX conditioning tensors: their shapes are fixed once
# resolution and batch are. Cleared at run start.
_FLUX_STATIC: dict[tuple, tuple] = {}


def _flux_static_inputs(bsz, h, w, device):
    import torch
    from diffusers import FluxPipeline

    key = (bsz, h, w, str(device))
    hit = _FLUX_STATIC.get(key)
    if hit is None:
        # Position ids drive RoPE and are indices, not activations, so keep them float32 regardless of the
        # training dtype.
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


def _qwen_load_conditioners(cfg, device, weight_dtype):
    from diffusers import QwenImagePipeline
    return _load_pipe_without_transformer(QwenImagePipeline, cfg, device)


def _qwen_load_transformer(cfg, device, weight_dtype, base_precision):
    # The prequant default ships the transformer 4-bit and loads trainable as-is under nf4; the dense
    # modes need the 20B Qwen/Qwen-Image base.
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

    # AutoencoderKLQwenImage is a 3D (video) VAE: add a temporal dim, encode, and normalise by the per-
    # channel latents_mean / latents_std.
    px = pixel_values.to(torch.float32).unsqueeze(2)
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.sample()
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

    # Qwen embeds are variable-length: pad to the batch max (or a pinned ``pad_to`` bucket under
    # compile) and batch the validity mask with them.
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
    # A single unpadded sample keeps the legacy None mask (identical math; avoids a behaviour delta for
    # existing single-image runs).
    if len(entries) == 1 and entries[0][1] is None and target == seqs[0]:
        mask_b = None
    return (pe_b, mask_b)


def _qwen_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import QwenImagePipeline

    pe, mask = embeds_batch
    bsz, c, f, h, w = noisy.shape
    packed = QwenImagePipeline._pack_latents(noisy, bsz, c, h, w)
    # Each batch entry is a LIST of one (frame, h/2, w/2) tuple: the transformer indexes sample[0] /
    # sample[1:] per entry, so a flat list breaks it.
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


def _zimage_load_conditioners(cfg, device, weight_dtype):
    from diffusers import ZImagePipeline
    return _load_pipe_without_transformer(ZImagePipeline, cfg, device)


def _zimage_load_transformer(cfg, device, weight_dtype, base_precision):
    # Prequant default loads 4-bit as-is under nf4; the dense modes use the bf16 Tongyi-MAI base.
    # Z-Image is bf16 only (its RoPE/embedder run fp32; fp16 overflows).
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
    # Z-Image trains from the posterior mode (deterministic), so the cached entry is the final latent: B
    # is None and the loop skips the per-step draw.
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
    # List I/O: one [C,1,H,W] latent + one [seq,2560] caption per sample. The timestep convention is
    # REVERSED ((1000 - t) / 1000) and the prediction is NEGATED.
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


def _krea2_load_conditioners(cfg, device, weight_dtype):
    # The krea repo ships transformers-5.x style configs the pinned 4.x line cannot parse, so the
    # conditioning pipeline is assembled per-component (diffusion_krea2.py).
    import torch
    from core.inference.diffusion_krea2 import load_krea2_pipeline

    pipe = load_krea2_pipeline(
        cfg.base_model, torch.bfloat16, hf_token = cfg.hf_token, with_transformer = False
    )
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.vae


def _krea2_load_transformer(cfg, device, weight_dtype, base_precision):
    # The transformer subfolder is diffusers-format. No prequant repo yet, so nf4 quantizes the 12B
    # transformer on the fly.
    from diffusers import Krea2Transformer2DModel
    return _load_dit_transformer(Krea2Transformer2DModel, cfg, device, base_precision)


def _krea2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            # encode_prompt pads/truncates to max_sequence_length, so every embed is
            # [1, 512, num_text_layers, 2560] with a [1, 512] mask (static shapes, padding before the suffix).
            pe, mask = pipe.encode_prompt(
                prompt = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 512,
            )
            out.append((pe.cpu(), mask.cpu()))
    return out


# Krea 2 conditions on the Qwen-Image VAE with the same per-channel latents_mean / latents_std normalisation, so
# latent encoding is shared.
_krea2_encode_latents = _qwen_encode_latents
_krea2_encode_latent_stats = _qwen_encode_latent_stats


def _krea2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # Fixed-length embeds, so collation is a plain concat with the mask riding along; ``pad_to`` is moot.
    pe_b = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    mask_b = torch.cat([e[1] for e in entries]).to(device)
    return (pe_b, mask_b)


def _krea2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import Krea2Pipeline

    pe, mask = embeds_batch
    # Krea2Pipeline._pack_latents is an instance method (it reads self.patch_size), so the packing is
    # inlined like the reference script.
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


# Both variants share Flux2Transformer2DModel and the upstream packing/forward conventions,
# differing only in the conditioning stack (dev: Mistral-3-Small; Klein: Qwen3) and size. Latents
# train patchified and batch-norm-normalised, from the posterior MODE (deterministic).
_FLUX2_COMMON_TARGETS = (
    # Double-stream blocks: separate q/k/v plus the ModuleList out proj.
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    # Single-stream blocks use one fused qkv+mlp input projection.
    "to_qkv_mlp_proj",
)
# A bare "to_out" suffix also matches the double-stream ModuleList, which PEFT cannot wrap, so each
# single block is named explicitly; missing high indexes are harmless on the Klein-4B layout.
_FLUX2_KLEIN_TARGETS = _FLUX2_COMMON_TARGETS + tuple(
    f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)
)
_FLUX2_DEV_TARGETS = _FLUX2_COMMON_TARGETS + tuple(
    f"single_transformer_blocks.{i}.attn.to_out" for i in range(48)
)
# The references train dev with its guidance-distillation vector at 3.5 (Klein applies it only when
# the variant config carries guidance_embeds).
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
            # encode_prompt pads to max_sequence_length, so embeds are fixed-length and text_ids are per-caption
            # [1, txt_len, 4]. The per-class text_encoder_out_layers defaults are left to the pipeline.
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
    # The FLUX.2 VAE normalises latents with its BatchNorm running stats, not shift/scaling_factor.
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat)
    return (lat - mean) / std


def _flux2_encode_latent_stats(vae, pixel_values):
    # FLUX.2 trains from the posterior mode (deterministic): the cached entry is the final latent
    # itself; B is None and the loop skips the per-step sampling draw.
    return _flux2_encode_latents(vae, pixel_values), None


def _flux2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # Fixed-length embeds give a plain concat; text_ids are PER-SAMPLE (unlike FLUX.1's shared grid), so
    # they concat too.
    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = torch.cat([e[1] for e in entries]).to(device = device, dtype = torch.float32)
    return (pe, text_ids)


# The ids derive only from the latent shape, so rebuilding them every step is allocator churn.
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
    # ``noisy`` is already in the patchified normalised space, so packing is the [B,C,H,W] to [B, H*W, C]
    # flatten the pipeline uses.
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
    # _unpack_latents_with_ids scatters per sample; diffusers 0.39 stacks them itself (its list
    # annotation is stale), other builds hand back the list. Same-resolution batches align either way.
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


# Milestone one trains from STILL IMAGES: a 1-frame clip is valid LTX-2 input (its VAE compresses
# time by 8) and style LoRAs converge on 20-50 stills, so this spec reuses the image dataset layer.
# AUDIOVISUAL: forward REQUIRES audio arguments and diffusers has no video-only escape hatch, so
# feed a one-token audio stream with isolate_modalities=True and keep audio out of the loss and
# out of the LoRA targets. The reason _LTX2_TARGETS names its modules in full.
# Two-stage conditioning: the Gemma3-12B hidden states are ~370 MB per caption, so encode_prompts
# runs both stages and caches only the small connector output.
# Video-stream attention only, fully qualified: a bare "to_q" would also match
# audio_attn1/audio_attn2 and the cross-modality attentions.
_LTX2_TARGETS = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
)

# LTX-2's rotary temporal coordinate is in SECONDS, so a 1-frame clip at the native 24 fps lands
# where a generated clip's first latent frame lands; Lightricks instead pin images to fps=1.0.
_LTX2_TRAIN_FPS = 24.0


def _ltx2_load_conditioners(cfg, device, weight_dtype):
    from diffusers import LTX2Pipeline

    pipe, vae = _load_pipe_without_transformer(LTX2Pipeline, cfg, device)
    # The connectors are not a text_encoder attribute, so _encoders_to_device never reaches them.
    if getattr(pipe, "connectors", None) is not None:
        pipe.connectors.to(device)
    return pipe, vae


def _ltx2_load_transformer(cfg, device, weight_dtype, base_precision):
    from diffusers import LTX2VideoTransformer3DModel
    return _load_dit_transformer(LTX2VideoTransformer3DModel, cfg, device, base_precision)


def _ltx2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    # The Gemma3 hidden states are per-LAYER stacked ([1, 1024, 3840 * layers]); only the connector output
    # reaches the transformer, so that is what gets cached.
    out = []
    with torch.no_grad():
        for cap in captions:
            pe, mask, _neg, _neg_mask = pipe.encode_prompt(
                prompt = cap,
                do_classifier_free_guidance = False,
                num_videos_per_prompt = 1,
                max_sequence_length = 1024,
                device = device,
            )
            # Read AFTER encode_prompt, which sets padding_side="left" itself: the connectors build the valid-
            # token mask from this, so a stale "right" would mask every short caption on the wrong end.
            padding_side = getattr(getattr(pipe, "tokenizer", None), "padding_side", "left")
            video_emb, audio_emb, conn_mask = pipe.connectors(pe, mask, padding_side = padding_side)
            out.append((video_emb.cpu(), audio_emb.cpu(), conn_mask.cpu()))
    return out


def _ltx2_latent_affine(vae, ref):
    import torch

    mean = vae.latents_mean.to(device = ref.device, dtype = ref.dtype).view(1, -1, 1, 1, 1)
    std = vae.latents_std.to(device = ref.device, dtype = ref.dtype).view(1, -1, 1, 1, 1)
    # scaling_factor is 1.0 on the shipped checkpoint but is applied for fidelity to _normalize_latents.
    return mean, std / float(vae.config.scaling_factor or 1.0)


def _ltx2_encode_latents(vae, pixel_values):
    import torch

    # A still is a 1-frame clip: [B,3,H,W] -> [B,3,1,H,W]. The VAE compresses 32x spatially and 8x
    # temporally, so a 512px still becomes [B,128,1,16,16].
    px = pixel_values.to(torch.float32).unsqueeze(2)
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.sample()
    mean, std = _ltx2_latent_affine(vae, lat)
    return (lat - mean) / std


def _ltx2_encode_latent_stats(vae, pixel_values):
    import torch

    px = pixel_values.to(torch.float32).unsqueeze(2)
    with torch.no_grad():
        dist = vae.encode(px).latent_dist
    mean, std = _ltx2_latent_affine(vae, dist.mean)
    return (dist.mean - mean) / std, dist.std / std


def _ltx2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # encode_prompt pads to max_sequence_length, so every connector embed is [1, 1024, 3840]
    video = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    audio = torch.cat([e[1] for e in entries]).to(device = device, dtype = weight_dtype)
    mask = torch.cat([e[2] for e in entries]).to(device)
    return (video, audio, mask)


def _ltx2_audio_token_count(config, num_pixel_frames: int, fps: float) -> int:
    """Audio latent tokens accompanying ``num_pixel_frames`` at ``fps``.

    The pipeline derives this as ``round(duration_s * sampling_rate / hop_length /
    temporal_compression)``; every term is on the transformer config, so the trainer does not
    need the audio VAE resident (it never encodes audio -- see the spec comment). Floored at
    one token: the transformer indexes the audio stream unconditionally, so an empty one
    would trip its RoPE."""
    per_second = (
        float(config.audio_sampling_rate)
        / float(config.audio_hop_length)
        / float(config.audio_scale_factor)
    )
    return max(1, round((num_pixel_frames / float(fps)) * per_second))


def _ltx2_pack(latents, conf):
    """[B,C,F,H,W] -> [B, F*H*W, C] via the pipeline's own patchifier."""
    from diffusers import LTX2Pipeline
    return LTX2Pipeline._pack_latents(latents, conf.patch_size, conf.patch_size_t)


def _ltx2_unpack(pred, f, h, w, conf):
    """The inverse of ``_ltx2_pack``, back to the 5-D shape ``target = noise - latents`` has."""
    from diffusers import LTX2Pipeline
    return LTX2Pipeline._unpack_latents(pred, f, h, w, conf.patch_size, conf.patch_size_t)


def _ltx2_audio_state(sigmas, bsz, audio_len, channels, device, dtype):
    """The audio-stream input for a step at ``sigmas``.

    A still-image dataset carries no audio ground truth, so the placeholder stream rides the
    SAME flow-matching state a zero clean latent would produce: ``(1 - sigma) * 0 + sigma *
    noise``. Zero is the mean of the normalised audio latent distribution, so this keeps the
    stream at the right SCALE for every sigma instead of feeding unit noise at a timestep the
    model expects nearly-clean latents at. With ``isolate_modalities = True`` it cannot reach
    the video prediction at all; it exists only because ``forward`` requires the argument."""
    import torch

    noise = torch.randn((bsz, audio_len, channels), device = device, dtype = dtype)
    # sigmas arrives broadcast to the 5-D video latent; the audio stream is 3-D.
    return sigmas.reshape(bsz, 1, 1) * noise


def _ltx2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    video_emb, audio_emb, mask = embeds_batch
    bsz, _c, f, h, w = noisy.shape
    conf = transformer.config
    packed = _ltx2_pack(noisy, conf)

    # vae_scale_factors is (temporal, height, width), so [0] is the 8x temporal compression.
    num_pixel_frames = (f - 1) * int(conf.vae_scale_factors[0]) + 1
    audio_len = _ltx2_audio_token_count(conf, num_pixel_frames, _LTX2_TRAIN_FPS)
    audio_noisy = _ltx2_audio_state(
        sigmas, bsz, audio_len, conf.audio_in_channels, packed.device, packed.dtype
    )

    pred, _audio_pred = transformer(
        hidden_states = packed,
        audio_hidden_states = audio_noisy,
        encoder_hidden_states = video_emb,
        audio_encoder_hidden_states = audio_emb,
        # LTX-2 conditions on the UNSCALED timestep (its config carries timestep_scale_multiplier = 1000
        # and the pipeline passes scheduler timesteps through as-is), unlike FLUX / Qwen's timestep / 1000.
        timestep = timesteps,
        sigma = timesteps,
        encoder_attention_mask = mask,
        audio_encoder_attention_mask = mask,
        num_frames = f,
        height = h,
        width = w,
        fps = _LTX2_TRAIN_FPS,
        audio_num_frames = audio_len,
        # Disable the a2v / v2a cross attention so the placeholder audio stream cannot perturb the video
        # prediction the LoRA is regressing.
        isolate_modalities = True,
        return_dict = False,
    )
    return _ltx2_unpack(pred, f, h, w, conf)


def _ltx2_save(pipe_cls, out_dir, transformer_lora_layers):
    from diffusers import LTX2Pipeline
    LTX2Pipeline.save_lora_weights(
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
        lora_targets = _FLUX2_KLEIN_TARGETS,
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
        lora_targets = _FLUX2_DEV_TARGETS,
        force_bf16 = True,
        # 32B DiT; the Mistral conditioning stack (~46 GB bf16) is loaded, encoded and freed BEFORE this lands
        # on the device (the shared phased load).
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
    "ltx-2": _FamilySpec(
        family = "ltx-2",
        lora_targets = _LTX2_TARGETS,
        force_bf16 = True,
        # 19B audiovisual DiT (37.76 GB bf16); the Gemma3-12B conditioning stack is loaded, encoded and
        # freed BEFORE this lands on the device, via the shared phased load.
        dense_bf16_gb = 37.8,
        load_conditioners = _ltx2_load_conditioners,
        load_transformer = _ltx2_load_transformer,
        encode_prompts = _ltx2_encode_prompts,
        encode_latents = _ltx2_encode_latents,
        encode_latent_stats = _ltx2_encode_latent_stats,
        collate = _ltx2_collate,
        forward = _ltx2_forward,
        save = _ltx2_save,
    ),
}


# HF repos gating access behind a license acceptance: training needs a token whose account accepted
# it. Checked by name (no network) so a missing token fails fast with an actionable message.
_GATED_TRAIN_REPOS = frozenset({"black-forest-labs/flux.1-dev", "black-forest-labs/flux.2-dev"})


def _assert_gated_access(base_model: str, hf_token: Optional[str]) -> None:
    """Raise a clear error before loading a gated base without a token."""
    from core.inference.diffusion_families import _is_local_path

    name = str(base_model or "").strip().lower()
    # A local clone named like the vendor repo is weights on disk, not a Hub fetch: refusing it by name
    # alone is what made that documented layout untrainable.
    if _is_local_path(base_model):
        return
    if name in _GATED_TRAIN_REPOS and not (hf_token and str(hf_token).strip()):
        raise ValueError(
            f"'{base_model}' is a gated Hugging Face repo. Accept its license on the Hub "
            f"and add your HF token in Unsloth settings before training from it."
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


def _build_latent_cache(
    spec,
    vae,
    image_paths,
    cfg,
    device,
    weight_dtype,
    on_event,
    check_stop,
    pcache = None,
    plan = None,
):
    """Precompute the per-image latent posterior cache: for each planned crop/flip variant,
    encode once and store the affine (A, B) pair on CPU (pinned when possible) in fp32. The
    stats stay fp32 so the per-step sample happens in fp32 and only the RESULT is cast to
    weight_dtype, matching the in-loop path (encode fp32 -> sample/normalise fp32 ->
    .to(weight_dtype)); fp32 doubles the cache RAM over bf16 but the cache is tiny (a handful
    of latents per image). ``pcache`` (a PersistentConditioningCache) serves hits from disk
    and receives every fresh encode, so the next run of the same config starts warm.
    Returns None if the build was interrupted by a stop request."""

    if plan is None:
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
            key = pcache.latent_key(path, (u_left, u_top, flip)) if pcache is not None else None
            entry = pcache.get(key) if key is not None else None
            if entry is not None:
                a, b = entry
            else:
                px = (
                    _load_pixel_tensor_planned(
                        path, cfg.resolution, cfg.center_crop, u_left, u_top, flip
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                a, b = spec.encode_latent_stats(vae, px)
                if key is not None:
                    try:
                        pcache.put(key, (a, b))
                    except Exception:  # noqa: BLE001 -- disk-full etc.: the run still trains
                        pass
            a, b = _hold(a), _hold(b)
            if not forced and not gated:
                # Size-gate the automatic cache off the first REAL encoded variant: packed latents x variants x
                # images can exhaust pinned RAM.
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


def _encode_prompts_cached(spec, pipe, to_encode, device, pcache):
    """Encode captions, serving hits from the persistent cache and writing misses back.
    The returned list is aligned with ``to_encode``; without a cache this is exactly
    ``spec.encode_prompts``. The cached tuples are the family's own CPU embed tuples
    (dtype preserved by safetensors), so a hit is identical to a fresh encode."""
    if pcache is None:
        return spec.encode_prompts(pipe, to_encode, device)
    hits: dict = {}
    misses = []
    for cap in to_encode:
        entry = pcache.get(pcache.text_key(cap))
        if entry is None:
            misses.append(cap)
        else:
            hits[cap] = entry
    if misses:
        for cap, emb in zip(misses, spec.encode_prompts(pipe, misses, device)):
            hits[cap] = emb
            try:
                pcache.put(pcache.text_key(cap), emb)
            except Exception:  # noqa: BLE001 -- disk-full etc.: the run still trains
                pass
    return [hits[cap] for cap in to_encode]


def _load_warm_conditioning(pcache, image_paths, plan, to_encode, device):
    """Load the FULL conditioning set (caption embeds + latent posterior stats) from the
    persistent cache. Returns (caption_embeds, latent_cache) on a complete hit, else
    (None, None) so the caller takes the cold path -- any missing/corrupt entry, and also
    an in-memory holding that would blow the host budget (the cold path re-decides that
    with the VAE resident and can fall back to per-step encoding). On success the VAE and
    text encoders are never loaded."""
    embeds = []
    for cap in to_encode:
        entry = pcache.get(pcache.text_key(cap))
        if entry is None:
            return None, None
        embeds.append(entry)

    def _hold(t):
        if t is None:
            return None
        import torch  # noqa: F401 -- pin_memory needs torch initialised

        t = t.contiguous()
        if device == "cuda":
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        return t

    forced = _latent_cache_forced()
    total_variants = sum(len(v) for v in plan)
    gated = False
    cache: list[list[tuple]] = []
    for i, path in enumerate(image_paths):
        variants = []
        for variant in plan[i]:
            entry = pcache.get(pcache.latent_key(path, variant))
            if entry is None:
                return None, None
            a, b = entry
            if not forced and not gated:
                per_variant = a.numel() * a.element_size()
                if b is not None:
                    per_variant += b.numel() * b.element_size()
                if _latent_cache_over_budget(per_variant, total_variants):
                    return None, None
                gated = True
            variants.append((_hold(a), _hold(b)))
        cache.append(variants)
    return {cap: emb for cap, emb in zip(to_encode, embeds)}, cache


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
    # torch.compile cannot trace the torchao int8 subclass in training (inductor rejects the aliased
    # subclass graph outputs), so int8 always runs eager.
    if base_precision == "int8":
        return False
    if mode == "on":
        return True
    # Regional compile is the whole point of the dense modes (2.6x measured on Z-Image bf16) but is
    # fragile over bitsandbytes 4-bit modules, so it stays off for QLoRA.
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
            # Heterogeneous-block DiTs (Z-Image: ~11 distinct block shapes) exceed dynamo's default recompile
            # limit of 8; bump it like the inference speed layer does.
            for attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo_cfg, attr):
                    setattr(dynamo_cfg, attr, max(getattr(dynamo_cfg, attr) or 0, 64))
            if hasattr(dynamo_cfg, "suppress_errors"):
                dynamo_cfg.suppress_errors = True
        # dynamic=True matches the inference speed layer: on torch 2.10 / B200 the dynamic=False
        # specialisation failed with CUBLAS_STATUS_EXECUTION_FAILED.
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
    """Train a flow-matching DiT LoRA (FLUX.1 / FLUX.2 / Qwen-Image / Z-Image / Krea 2 / LTX-2) and export it.

    Resumable: ``cfg.resume_from_checkpoint`` restores the adapter, optimizer moments, LR
    position, EMA shadow, sampler cycle and RNG streams from a ``checkpoint-<N>`` bundle,
    and the loop runs steps N+1..train_steps (the TARGET TOTAL). A stop-and-save and every
    ``cfg.save_steps`` interval write such a bundle."""
    cfg = config.normalized()
    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        raise ValueError(f"No DiT trainer for family {cfg.resolved_family!r}")

    # DiT families train in bf16, so an explicit fp16 request is refused, not silently upgraded.
    # Validated before the heavy imports so a host without diffusers still sees the real error.
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
    # Fail fast on pre-Ampere CUDA, gating on NATIVE bf16 (capability major >= 8), since
    # is_bf16_supported() counts emulation.
    if device == "cuda" and not native_bf16_supported():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU (Ampere or newer); "
            "this CUDA device does not support bf16."
        )
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _assert_trusted_base_model(cfg.base_model)
    # Check the repo this run will FETCH: the canonical id would raise for a gated base already
    # redirected to its ungated mirror, after the route answered 200 and freed the residents.
    _assert_gated_access(cfg.fetch_base_model or cfg.base_model, cfg.hf_token)
    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    # Resolve num_epochs into a concrete train_steps now the dataset size is known, and rebind cfg so every
    # downstream read agrees.
    cfg = replace(cfg, train_steps = resolve_train_steps(cfg, len(pairs)), num_epochs = 0)
    # Validate a resume against this run's identity BEFORE the multi-GB phased load, using the RESOLVED
    # LoRA targets rather than the generic default the config carries.
    identity = identity_for_config(
        cfg,
        dataset_pairs = pairs,
        resolved_targets = _select_lora_targets(cfg.lora_target_modules, spec.lora_targets),
    )
    if cfg.resume_from_checkpoint:
        preflight_resume(
            cfg.resume_from_checkpoint, identity = identity, target_steps = cfg.train_steps
        )
    _emit(on_event, "model_load_started", num_images = len(pairs))
    if _check_stop():
        out_dir = Path(cfg.output_dir).expanduser()
        _emit(
            on_event,
            "complete",
            output_dir = str(out_dir),
            lora_path = None,
            stopped = True,
            steps_run = 0,
            # A stop with save=false is a DISCARD however early it lands; without it the resume fallback offers
            # the source bundle back as though the attempt were still live.
            discarded = not save_on_stop,
        )
        return str(out_dir)

    # TF32 / cudnn.benchmark for the run, restored on the way out (the trainer subprocess is disposable, but
    # restoring keeps in-process callers clean).
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
            identity,
        )
    finally:
        _restore_perf_flags(perf_snap)


def _train_dit(
    cfg, spec, pairs, rng, device, weight_dtype, on_event, _check_stop, _save_on_stop, identity
):
    """The body of ``run_dit_lora_training``, split out so the backend perf flags are
    snapshot/restored around it in exactly one place."""
    import torch
    import torch.nn.functional as F
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    # diffusers honours no env var and is only in sys.modules from here, so its pipeline-loading bars
    # can only be turned off now.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a training run
        pass

    use_lora_targets = _select_lora_targets(cfg.lora_target_modules, spec.lora_targets)
    out_dir = Path(cfg.output_dir).expanduser()
    # Load from the byte-identical public mirror selected during normalization, while keeping
    # cfg.base_model canonical for the adapter sidecar, completion event and resume identity.
    fetch_cfg = replace(cfg, base_model = cfg.fetch_base_model or cfg.base_model)

    # Phase 0: the persistent conditioning cache (opt-in via cond_cache_dir). When every planned latent variant
    # AND caption embedding is on disk, the run is "warm": the VAE and text encoders never load.
    image_paths = [p for p, _ in pairs]
    captions = [c for _, c in pairs]
    uniq = sorted(set(captions))
    # CFG dropout swaps a sample's conditioning for the empty prompt, so its embedding must be
    # precomputed alongside the captions (the encoders are freed right after).
    cfg_dropout = float(getattr(cfg, "cfg_dropout", 0.0) or 0.0)
    to_encode = uniq + ([""] if cfg_dropout > 0 and "" not in uniq else [])
    use_cache = cfg.cache_latents and os.environ.get(
        "UNSLOTH_DIFFUSION_NO_LATENT_CACHE", ""
    ) not in ("1", "true")
    pcache = None
    if getattr(cfg, "cond_cache_dir", None):
        try:
            # Namespace on the CHECKPOINT, not just the family: the keys carry only caption/image content, so
            # one cache dir reused for two checkpoints would train on the other model's embeddings.
            from .diffusion_train_extras import source_revision  # noqa: PLC0415
            namespace = f"{spec.family}_{cfg.base_model}_{source_revision(fetch_cfg.base_model)}"
            pcache = PersistentConditioningCache(cfg.cond_cache_dir, namespace, cfg.resolution)
        except Exception as exc:  # noqa: BLE001 -- the cache is an optimisation, never fatal
            _emit(on_event, "warning", message = f"conditioning cache disabled: {exc}")
    # The crop/flip variant plan is seed-deterministic, so persistent keys are stable across runs and
    # the warm check can run before anything loads.
    plan = _plan_cache_variants(
        len(image_paths), cfg.cache_variants, cfg.center_crop, cfg.random_flip, cfg.seed
    )

    pipe = None
    vae = None
    caption_embeds = None
    # The estimated cache exceeded the host-memory budget; keep the VAE resident and fall through to the in-loop
    # encode path (latent_cache stays None).
    latent_cache = None
    if pcache is not None and use_cache:
        caption_embeds, latent_cache = _load_warm_conditioning(
            pcache, image_paths, plan, to_encode, device
        )
        if caption_embeds is not None:
            _emit(
                on_event,
                "preparing",
                stage = "cache_latents",
                done = len(image_paths),
                total = len(image_paths),
            )
    if caption_embeds is None:
        latent_cache = None
        pipe, vae = spec.load_conditioners(fetch_cfg, device, weight_dtype)
        encoded = _encode_prompts_cached(spec, pipe, to_encode, device, pcache)
        caption_embeds = {cap: emb for cap, emb in zip(to_encode, encoded)}
        _free_text_encoders(pipe)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # The cache keeps the posterior affine parameters, so per-step sampling noise is preserved.
        if use_cache:
            latent_cache = _build_latent_cache(
                spec,
                vae,
                image_paths,
                cfg,
                device,
                weight_dtype,
                on_event,
                _check_stop,
                pcache = pcache,
                plan = plan,
            )
            if latent_cache is LATENT_CACHE_OVER_BUDGET:
                # The estimated cache exceeded the host-memory budget; keep the VAE resident and fall through to the
                # in-loop encode path (latent_cache stays None).
                latent_cache = None
            elif latent_cache is None:
                _emit(
                    on_event,
                    "complete",
                    output_dir = str(out_dir),
                    lora_path = None,
                    stopped = True,
                    steps_run = 0,
                    # A discard is a discard however early the stop lands.
                    discarded = not _save_on_stop(),
                )
                return str(out_dir)
    if latent_cache is not None and vae is not None:
        try:
            pipe.vae = None
        except Exception:  # noqa: BLE001 -- a pipeline without a settable vae keeps it
            pass
        del vae
        vae = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    # Variant picks use their own stream so the training loop index/noise draws stay seed-deterministic.
    variant_rng = random.Random(cfg.seed + 1)

    # Phase 3: only now load the transformer, in the resolved base precision (nf4 QLoRA by default; bf16 / int8
    # / fp8 / mxfp8 are the dense speed modes; "auto" picks from free VRAM).
    base_precision = _resolve_base_precision(cfg, spec, device)
    transformer = spec.load_transformer(fetch_cfg, device, weight_dtype, base_precision)
    base_is_bnb = base_precision == "nf4"

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
        # Non-reentrant checkpointing: reentrant recompute of a bnb 4-bit LoRA linear can trip an illegal
        # memory access on the larger FLUX transformer, and it is the recommended mode anyway.
        import functools
        import torch.utils.checkpoint as _ckpt
        transformer.enable_gradient_checkpointing(
            gradient_checkpointing_func = functools.partial(_ckpt.checkpoint, use_reentrant = False)
        )
    cast_training_params(transformer, dtype = torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    # int8 / fp8 / mxfp8 convert the frozen base linears AFTER the LoRA attaches, so the adapter modules
    # are excluded and stay high precision.
    if base_precision == "int8":
        _int8_quantize_base(transformer, cfg.resolved_family)
    if base_precision == "fp8" and not _apply_fp8_training(transformer, on_event):
        base_precision = "bf16"
    if base_precision == "mxfp8" and not _apply_mxfp8_training(transformer, on_event):
        base_precision = "bf16"

    # LoRA EMA (opt-in via cfg.ema_decay): shadows ONLY the trainable adapter params, initialised after
    # the precision conversions so they track the final fp32 objects.
    ema = LoRAEMA(transformer, decay = cfg.ema_decay) if getattr(cfg, "ema_decay", 0.0) else None

    compiled = _maybe_compile_transformer(
        transformer, cfg, base_is_bnb, device, on_event, base_precision
    )
    # Compiled Qwen graphs need one fixed text length across steps: pin the pad bucket to the dataset's longest caption.
    qwen_pad_to = None
    if compiled and spec.family == "qwen-image":
        qwen_pad_to = max(e[0].shape[1] for e in caption_embeds.values())

    optimizer = _make_optimizer(lora_params, cfg.learning_rate)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        fetch_cfg.base_model, subfolder = "scheduler", token = cfg.hf_token
    )
    # getattr defaults keep an un-normalized config on the historical behaviour.
    flow_shift = getattr(cfg, "flow_shift", None)
    if flow_shift is None:
        flow_shift = "auto" if spec.family in AUTO_FLOW_SHIFT_FAMILIES else 1.0
    sigma_table = _training_sigma_table(scheduler, flow_shift)
    shift_active = sigma_table is not scheduler.sigmas
    num_train_ts = scheduler.config.num_train_timesteps
    bell_weights = (
        _bell_loss_weights(num_train_ts).to(device)
        if str(getattr(cfg, "weighting_scheme", "none") or "none") == "bell"
        else None
    )
    # The LR schedule advances once per optimizer update, so warmup/decay count optimizer steps; scaling
    # by the accumulation factor would stretch warmup past the run.
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )

    _emit(on_event, "model_load_completed", compiled = compiled, base_precision = base_precision)
    # Read the commit now the base is on disk: an identity built before the load says "unresolved",
    # which is not comparable, so a later resume could not tell the repo had moved underneath it.
    # Record the repo actually fetched: the canonical base is not on disk at all when the mirror was
    # selected, and mismatch_reason only compares two revisions that came from the same repo.
    identity = with_resolved_revision(identity, fetch_cfg.base_model)
    # See the SDXL trainer: the cache path the loop actually took, not the one requested.
    identity = with_cache_mode(identity, latent_cache is not None)
    # base_precision is still the request here ("auto" resolves at load, and a failed fp8/mxfp8
    # conversion has fallen back to bf16), so without this a bundle records a base it never trained on.
    identity = with_resolved_base_precision(identity, base_precision)

    transformer.train()
    n_images = len(image_paths)
    batch_size = cfg.train_batch_size
    # Permutation-cycle index sampler (shared with the SDXL trainer): visits every image once per cycle,
    # so a short run covers the whole dataset.
    index_sampler = PermutationBatchSampler(n_images, rng)

    # Restore a previous run (adapter, optimizer moments, LR position, EMA shadow, sampler cycle, RNG streams)
    # before the loop, so it picks up at `resumed + 1`. None for a fresh run.
    rng_streams = {"loop": rng, "variant": variant_rng}
    restored = restore_resume_state(
        cfg,
        model = transformer,
        optimizer = optimizer,
        lr_scheduler = lr_sched,
        identity = identity,
        on_event = on_event,
        ema = ema,
        sampler = index_sampler,
        rng_streams = rng_streams,
    )
    resumed = restored.step if restored is not None else 0
    # Resuming from directory A into a reused output_dir B leaves B's existing bundles as foreign as
    # for a fresh run, so the first save must clear them or a higher-numbered one outranks this run.
    resumed_here = bool(resumed) and resumed_into_this_dir(cfg, out_dir)
    # The bundles already here when this run started, so a discard removes only what this run wrote.
    preexisting_checkpoints = snapshot_checkpoints(out_dir)
    if restored is not None:
        was = restored.progress.get("resolved_base_precision")
        if was and was != base_precision:
            # Not fatal, since only the frozen base's numerics change, but base_precision="auto" picks from free
            # VRAM, so this can happen silently.
            _emit(
                on_event,
                "warning",
                message = (
                    f"Resuming a checkpoint trained with a {was} base in {base_precision}; "
                    f"set base_precision explicitly to keep them identical."
                ),
            )

    stopped = False
    # Carried over so avg_loss stays an average over the whole run, not just since the resume.
    running_loss = restored.running_loss if restored is not None else 0.0
    peak_gb = 0.0
    t_start = time.time()
    t_steady = None
    # Starts at the resumed step so a no-op resume still reports the real step.
    done = resumed

    def _save_checkpoint(step: int) -> None:
        # write_resume_checkpoint reports its own checkpoint_saved / checkpoint_failed events, so a run
        # that crashes after a save is still known to be resumable and one whose save failed is still
        # known to be blocked.
        _written, _error = write_resume_checkpoint(
            cfg,
            step = step,
            model = transformer,
            optimizer = optimizer,
            lr_scheduler = lr_sched,
            identity = identity,
            on_event = on_event,
            ema = ema,
            sampler = index_sampler,
            rng_streams = rng_streams,
            # "auto" resolves from free VRAM at load time, so the identity, which records the REQUESTED mode,
            # cannot tell nf4 from bf16 across two "auto" runs.
            progress = {"running_loss": running_loss, "resolved_base_precision": base_precision},
            # NOT discard_existing: deleting the previous run's bundles at the FIRST periodic save spends them
            # before this run produced anything, so cancelling a retrain destroyed the thing being retrained.
            discard_existing = False,
            # A branched resume must not prune the higher-numbered checkpoints it did not write.
            preexisting = preexisting_checkpoints,
        )

    # bf16 autocast around the forward + loss, matching the diffusers dreambooth scripts: it reconciles
    # the fp32 LoRA params with the bnb 4-bit base matmuls.
    autocast = (
        torch.autocast(device_type = "cuda", dtype = torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    for opt_step in range(resumed, cfg.train_steps):
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
            sigmas = _gather_sigmas(sigma_table, t_indices, device, weight_dtype, latents.ndim)
            if shift_active:
                # The model's timestep conditioning must follow the shifted sigma (timestep = sigma *
                # num_train_timesteps). Gather in fp32 so bf16 rounding never skews it.
                timesteps = (
                    sigma_table[t_indices].to(device = device, dtype = torch.float32).flatten()
                    * num_train_ts
                )
            noisy = (1.0 - sigmas) * latents + sigmas * noise

            embeds = spec.collate(
                [
                    caption_embeds[""]
                    if cfg_dropout > 0 and rng.random() < cfg_dropout
                    else caption_embeds[captions[i]]
                    for i in idxs
                ],
                device,
                weight_dtype,
                pad_to = qwen_pad_to,
            )
            with autocast:
                model_pred = spec.forward(
                    transformer, noisy, timesteps, sigmas, embeds, cfg, device, weight_dtype
                )
                target = noise - latents
                if bell_weights is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")
                else:
                    per = F.mse_loss(model_pred.float(), target.float(), reduction = "none")
                    w_idx = (
                        (sigmas.flatten().float() * num_train_ts).long().clamp(0, num_train_ts - 1)
                    )
                    w = bell_weights[w_idx].view(-1, *([1] * (per.ndim - 1)))
                    loss = (per * w).mean()
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps

        grad_norm = None
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            # clip_grad_norm_ returns the total PRE-clip norm: the health signal the UI charts (an exploding
            # norm shows even while the clip caps the update).
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
        optimizer.step()
        lr_sched.step()
        if ema is not None:
            ema.update(transformer)

        running_loss += step_loss
        done = opt_step + 1
        now = time.time()
        # Rates count only the steps THIS process ran, so a resumed run does not divide by steps it never executed.
        ran_here = done - resumed
        if ran_here == 1:
            # Step 1 pays the one-time costs (cudnn autotune, compile warmup), so the reported rate starts after it.
            t_steady = now
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            if device == "cuda":
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            per_step = batch_size * cfg.gradient_accumulation_steps
            if t_steady is not None and ran_here > 1:
                sps = round((ran_here - 1) * per_step / max(now - t_steady, 1e-6), 3)
            else:
                sps = round(ran_here * per_step / max(now - t_start, 1e-6), 3)
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
        stop_now = _check_stop()
        # Skipped on the final step and when stopping, since the stop path writes one at the exact step.
        if (
            not stop_now
            and cfg.save_steps
            and done % cfg.save_steps == 0
            and done < cfg.train_steps
        ):
            _save_checkpoint(done)
        if stop_now:
            stopped = True
            break

    lora_path: Optional[str] = None
    catalog_path: Optional[str] = None
    ema_path: Optional[str] = None
    if not (stopped and not _save_on_stop()):
        out_dir.mkdir(parents = True, exist_ok = True)
        # Write the bundle BEFORE the adapter export: if that export fails, the run still comes back.
        if stopped and done > 0:
            _save_checkpoint(done)
        layers = get_peft_model_state_dict(transformer)
        spec.save(pipe, str(out_dir), layers)
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        # ``done`` (the step reached), not cfg.train_steps: a stop at 11/500 must not advertise 500.
        catalog_path = _publish_to_lora_catalog(lora_path, cfg, done)
        if ema is not None and ema.updates > 0:
            try:
                # Report the adapter FILE, like lora_path, so a caller can load it directly.
                ema_dir = save_ema_adapter(ema, transformer, spec.save, str(out_dir))
                ema_path = str(Path(ema_dir) / DEFAULT_LORA_FILENAME)
            except Exception as exc:  # noqa: BLE001 -- the primary adapter is already saved
                _emit(on_event, "warning", message = f"EMA adapter save failed: {exc}")
        if not stopped:
            # A completed run has nothing to resume and the final iteration writes no bundle, so save_steps
            # would leave this run's own checkpoint behind for a later resume to roll back to.
            retire_own_checkpoints(out_dir, preexisting_checkpoints, resumed_here = resumed_here)
        elif not resumed_here:
            # A stop-with-save on a fresh retrain is a LOWER step than the earlier run's leftovers, and resume-
            # by-directory picks the newest by step, so those would outrank the partial just saved.
            discard_preexisting_checkpoints(out_dir, preexisting_checkpoints)
    else:
        # save_steps writes bundles as the run goes, so without this a discard leaves up to
        # save_total_limit copies of the optimizer state in a directory the user got no artifact from:
        # invisible to every scanner, unresumable, with no delete path in the UI.
        clear_own_checkpoints(out_dir, preexisting_checkpoints)
        try:
            out_dir.rmdir()
        except OSError:
            # Not ours to remove if anything else is in it (a previous run's adapter).
            pass
    _emit(
        on_event,
        "complete",
        output_dir = str(out_dir),
        lora_path = lora_path,
        ema_path = ema_path,
        catalog_path = catalog_path,
        family = cfg.resolved_family,
        base_model = cfg.base_model,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
        wall_seconds = round(time.time() - t_start, 1),
        resumed_from_step = resumed or None,
        # "Stop without saving" discards the run, so its own periodic checkpoints must not keep
        discarded = bool(stopped and not _save_on_stop()),
    )
    return str(out_dir)


def _make_optimizer(params, lr):
    """8-bit AdamW (bitsandbytes) when available -- half the optimizer state, no accuracy
    regression for LoRA -- else torch AdamW, fused on CUDA (with a fallback when this
    build/device lacks the fused kernel).

    UNSLOTH_DIFFUSION_FP32_OPTIM forces plain (non-fused) AdamW, as it does for SDXL: the
    accuracy guard wants the reference optimizer, and a host where the override means one
    thing for one trainer and nothing for the other cannot answer "can this checkpoint be
    resumed here" before the run starts."""
    import torch

    if os.environ.get("UNSLOTH_DIFFUSION_FP32_OPTIM", "") in ("1", "true"):
        return torch.optim.AdamW(params, lr = lr)
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
    """Drop every conditioning module the pipeline holds once the embeddings are
    precomputed, so they do not sit in VRAM during training. ``connectors`` is LTX-2's
    second conditioning stage (~2.7 GB) and belongs here for the same reason as the text
    encoders; ``audio_vae`` / ``vocoder`` are LTX-2 decode-side modules the trainer never
    touches. Absent attributes are skipped, so this is a no-op for the image families."""
    for attr in (
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "connectors",
        "audio_vae",
        "vocoder",
    ):
        if getattr(pipe, attr, None) is not None:
            try:
                setattr(pipe, attr, None)
            except Exception:  # noqa: BLE001
                pass
