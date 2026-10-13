# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flow-matching LoRA training for the rectified-flow DiT families.

Image: FLUX.1-dev, FLUX.2 (dev + Klein), Qwen-Image, Z-Image, Krea 2. Video: LTX-2 (from a
still-image dataset; see the ``_LTX2_TARGETS`` block for what a video family adds).

These are rectified-flow transformers, not the SDXL U-Net, so they share only the plumbing in
``diffusion_train_common`` (config, dataset discovery, events, stop, publishing). The training math
here is flow matching: sample a sigma with the logit-normal density used by the diffusers dreambooth
scripts, form ``noisy = (1 - sigma) * latents + sigma * noise``, predict the velocity, and regress
it onto ``target = noise - latents``. The per-family differences (latent normalisation + packing,
the transformer forward signature, embedding collation, and the LoRA save entrypoint) live in small
``_FamilySpec`` objects; the loop itself is family-agnostic. Verified against diffusers 0.38.0.

Memory: the text encoder(s) are the largest module (T5-XXL ~9 GB for FLUX, Qwen2.5-VL ~7 GB for
Qwen-Image, Qwen3 for Z-Image), so captions are encoded ONCE up front and the encoders are freed
before the loop. VAE latents are likewise precomputed into a small CPU cache (``cache_latents``) and
the VAE freed: the cache stores the posterior's affine parameters (mean/std folded through the
family's latent normalisation), so every step still draws a fresh VAE sample --
distribution-identical to encoding in the loop, without keeping the VAE resident or paying a
per-step encode. The transformer trains as a QLoRA (nf4) adapter by default with gradient
checkpointing and fused AdamW, so only the (small) LoRA params + optimizer state and the frozen
4-bit base sit in VRAM during the loop.
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
    enable_diffusion_gradient_checkpointing,
    TrainingPeakMemory,
    resume_steps_left,
    make_lora_optimizer,
    recorded_optimizer_class,
    _apply_perf_flags,
    _assert_trusted_base_model,
    _emit,
    _latent_cache_forced,
    _latent_cache_over_budget,
    _plan_cache_variants,
    _publish_to_lora_catalog,
    _restore_perf_flags,
    bucket_groups,
    discover_image_caption_pairs,
    has_functional_torchao,
    flow_bf16_trainable,
    native_bf16_supported_xpu,
    PermutationBatchSampler,
    plan_image_canvases,
    repo_is_prequantized,
    resolve_bucketing,
    resolve_train_device,
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
from core.training.diffusion_samples import (
    discard_samples,
    euler_flow_sample,
    isolated_sampling,
    plan_samples,
    run_sample_round,
)
from core.training.diffusion_train_extras import (
    BucketBatchSampler,
    LoRAEMA,
    cover_resize_dims,
    PersistentConditioningCache,
    save_ema_adapter,
)

# FLUX / Qwen double-stream blocks also carry added-kv projections; Z-Image is single-stream.
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
# Attention + SwiGLU (DiffSynth-Studio's Z-Image recipe); old attention-only runs keep theirs on resume.
_ZIMAGE_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3")
# The Krea 2 authors' recommended defaults (their DreamBooth script): attention + SwiGLU + text-fusion projector +
# conditioning embedders. For long runs they suggest narrowing to attention.
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
    """The generic DEFAULT_LORA_TARGETS means unset, so the family's targets apply; other tuples win."""
    if tuple(cfg_targets) == DEFAULT_LORA_TARGETS:
        return tuple(spec_targets)
    return tuple(cfg_targets)


@dataclass
class _FamilySpec:
    """Everything the shared loop needs that differs by family."""

    family: str
    lora_targets: tuple[str, ...]
    # Z-Image overflows fp16.
    force_bf16: bool
    # A family covering several sizes (flux.2-klein 4B/9B) must use _dense_bf16_gb.
    dense_bf16_gb: float
    # Phased load: the transformer never coexists with the text encoders + VAE.
    load_conditioners: Callable[..., tuple[Any, Any]]
    load_transformer: Callable[..., Any]
    encode_prompts: Callable[..., list[tuple]]
    encode_latents: Callable[..., Any]
    # (A, B) posterior params so a per-step sample is A + B * randn; B is None if deterministic.
    encode_latent_stats: Callable[..., tuple]
    # pad_to pins a fixed text length for variable-length embeds (compile).
    collate: Callable[..., tuple]
    # Returns model_pred aligned with target = noise - latents.
    forward: Callable[..., Any]
    save: Callable[..., None]
    # (vae, h, w) -> (latent shape, packed seq len for the shift), and latents -> pixels. None = no sampling path.
    latent_shape: Optional[Callable[..., tuple]] = None
    decode_latents: Optional[Callable[..., Any]] = None
    sample_forward_kwargs: Optional[dict] = None


def _gather_sigmas(sigma_table, indices, device, dtype, n_dim):
    """Gathers by index, so sigma_table must stay aligned with the timesteps the indices were drawn from."""
    sigma = sigma_table[indices].to(device = device, dtype = dtype).flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _unshifted_sigmas(scheduler):
    """Uniform u before any init-time static shift, as ``FlowMatchEulerDiscreteScheduler.__init__`` builds it."""
    import numpy as np
    import torch

    sc = scheduler.config
    if getattr(sc, "use_dynamic_shifting", False) or float(getattr(sc, "shift", 1.0) or 1.0) == 1.0:
        return scheduler.sigmas
    n = int(sc.num_train_timesteps)
    u = torch.from_numpy(np.linspace(1, n, n, dtype = np.float32)[::-1].copy()) / n
    return u.to(scheduler.sigmas.dtype)


def _training_sigma_table(scheduler, flow_shift):
    """None: the scheduler's own sigmas. "auto": the inference distribution, which dynamic-shifting
    schedulers never bake into ``sigmas`` (Qwen pins mu = max_shift = log 3). A number: the effective
    shift s*u/(1+(s-1)*u) on unshifted u (musubi/kohya); applying it on top of a baked-in shift is WRONG."""
    sigmas = scheduler.sigmas
    if flow_shift is None:
        return sigmas
    if flow_shift == "auto":
        sc = scheduler.config
        if not getattr(sc, "use_dynamic_shifting", False):
            return sigmas  # static-shift family: its init already baked the shift in
        mu = float(getattr(sc, "max_shift", None) or math.log(3.0))
        shifted = scheduler.time_shift(mu, 1.0, sigmas)
        if getattr(sc, "shift_terminal", None):
            # Stretch must see the full descending table, never a batch.
            shifted = scheduler.stretch_shift_to_terminal(shifted)
        return shifted
    s = float(flow_shift)
    u = _unshifted_sigmas(scheduler)
    if s == 1.0:
        return u
    return s * u / (1.0 + (s - 1.0) * u)


def _bell_loss_weights(num_train_timesteps):
    """bsmntw-style bell loss-weight table over the training schedule: a Gaussian bell centered
    mid-schedule, floored at 0 and normalized to mean 1 (so the expected loss scale is
    unchanged). Indexed by round(sigma * num_train_timesteps)."""
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
    """A QLoRA load leaves text encoders on CPU, so move them; a 4-bit encoder already placed is skipped."""
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
        bnb_4bit_use_double_quant = True,
    )


# Alias kept for callers; the heuristic moved to diffusion_train_common for config validation.
_repo_is_prequantized = repo_is_prequantized


def _load_quantized_transformer(transformer_cls, cfg, device):
    """Load ``cfg.base_model``'s transformer subfolder as a trainable nf4 QLoRA module."""
    import torch
    return transformer_cls.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        quantization_config = _bnb_4bit_config(),
        device_map = {"": device},
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    )


def _load_pipe_without_transformer(pipe_cls, cfg, device):
    """Load a pipeline for conditioning only: ``transformer = None`` skips the multi-GB denoiser
    entirely (the documented diffusers pattern), leaving just the text encoders + tokenizers +
    VAE + scheduler. The transformer loads later, after these are freed."""
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
    """Load the transformer alone in the resolved ``base_precision``. nf4: a prequant (bnb-4bit)
    repo carries its quantization config and loads 4-bit as-is, while a dense base is quantized
    to nf4 on the fly; the memory floor. bf16 / fp8 / mxfp8: the dense transformer, converted AFTER the LoRA
    attaches (fp8 stores its frozen linears in e4m3, about half the bf16 weight memory; mxfp8 keeps bf16 storage).
    int8: the dense transformer quantized in place to torchao weight-only int8 (the
    PEFT-attachable scheme), roughly halving the bf16 weight footprint."""
    import torch

    if base_precision == "nf4":
        if not repo_is_prequantized(cfg.base_model):
            return _load_quantized_transformer(transformer_cls, cfg, device)
        transformer = transformer_cls.from_pretrained(
            cfg.base_model,
            subfolder = "transformer",
            torch_dtype = torch.bfloat16,
            token = cfg.hf_token,
        )
        if not getattr(transformer, "is_loaded_in_4bit", False):
            transformer = transformer.to(device)
        return transformer

    if base_precision == "fp8":
        transformer = _load_fp8_prequant_transformer(transformer_cls, cfg, device)
        if transformer is not None:
            return transformer

    # int8 quantizes AFTER the LoRA attaches: quantizing first makes peft dispatch TorchaoLoraLinear, whose peft-0.18
    # constructor rejects the torchao-0.16 config API (peft 0.19+ accepts it; the order works for both).
    return transformer_cls.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    ).to(device)


def _load_fp8_prequant_transformer(transformer_cls, cfg, device):
    """The family's hosted fp8 denoiser (the file Studio inference uses) as the fp8 base, skipping the bf16 transformer
    download; None (load bf16 and quantize, as before) when none is mapped for this exact base or it fails to load."""
    import logging

    from core.training.diffusion_fp8_prequant import load_fp8_transformer, resolve_fp8_checkpoint

    log = logging.getLogger(__name__)
    path = resolve_fp8_checkpoint(
        getattr(cfg, "resolved_family", None), cfg.base_model, cfg.hf_token, logger = log
    )
    if path is None:
        return None
    try:
        transformer = load_fp8_transformer(
            transformer_cls, cfg.base_model, path, device, hf_token = cfg.hf_token
        )
    except Exception as exc:  # noqa: BLE001 -- the dense path still works
        log.warning("fp8 prequant: %s did not load (%s); loading bf16", path, exc)
        gc.collect()
        return None
    transformer._studio_fp8_prequant_path = path
    log.info("fp8 base loaded from the pre-quantized checkpoint %s", path)
    return transformer


def _int8_quantize_base(transformer, family: Optional[str] = None) -> None:
    """Passing family keeps training and inference on one exclusion list, else the first forward raises."""
    from core.inference.diffusion_transformer_quant import (
        _quiet_config,
        apply_small_m_padding,
        exclude_tokens_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    # Int8WeightOnlyConfig defaults to set_inductor_config=True, process-wide.
    quantize_(
        transformer,
        _quiet_config(Int8WeightOnlyConfig),
        filter_fn = make_filter_fn(
            512, exclude_name_tokens = exclude_tokens_for_scheme("int8", family)
        ),
    )
    # After quantize_: the helper reparents the Linears it wraps. Not best-effort.
    apply_small_m_padding(transformer, "int8", family)


# Small-M linears (conditioning vectors, LTX-2 one-token audio) stay bf16: the float8 / MX casts cost more than the GEMM.
# final_layer is Krea 2's output projection.
_LOWP_EXTRA_EXCLUDE_TOKENS = ("final_layer",)


def _lowp_exclude_tokens(family: Optional[str] = None) -> tuple[str, ...]:
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme
    return tuple(exclude_tokens_for_scheme("int8", family)) + _LOWP_EXTRA_EXCLUDE_TOKENS


def _lowp_module_filter(
    mod,
    fqn: str,
    align: int,
    exclude_tokens: tuple[str, ...] = (),
) -> bool:
    import torch.nn as nn

    if not isinstance(mod, nn.Linear):
        return False
    if "lora_" in fqn:
        return False
    if fqn.endswith("proj_out") or ".proj_out." in fqn:
        return False
    name = fqn.lower()
    if any(tok in name for tok in exclude_tokens):
        return False
    return mod.in_features % align == 0 and mod.out_features % align == 0


def _fp8_module_filter(
    mod,
    fqn: str,
    exclude_tokens: tuple[str, ...] = (),
) -> bool:
    """Dims divisible by 16 (the diffusers FLUX2 reference filter)."""
    return _lowp_module_filter(mod, fqn, 16, exclude_tokens)


def _in_repeated_blocks(transformer, module_filter):
    """Restrict the filter to the regionally compiled ``_repeated_blocks`` (if any)."""
    names = set(getattr(transformer, "_repeated_blocks", None) or ())
    prefixes = tuple(
        f"{fqn}." for fqn, mod in transformer.named_modules() if mod.__class__.__name__ in names
    )
    if not prefixes:
        return module_filter
    return lambda mod, fqn: fqn.startswith(prefixes) and module_filter(mod, fqn)


def _lowp_targets(transformer, base_filter, family: Optional[str]) -> list:
    tokens = _lowp_exclude_tokens(family)
    keep = _in_repeated_blocks(transformer, lambda mod, fqn: base_filter(mod, fqn, tokens))
    return [
        (fqn, mod)
        for fqn, mod in transformer.named_modules()
        if not getattr(mod, "_studio_lowp", None) and keep(mod, fqn)
    ]


def _require_lowp_modules(transformer, mode: str) -> bool:
    """Raises when nothing converted (FLUX.1 / LTX-2 on torchao 0.17, all biased), so bf16 is recorded."""
    if not any(getattr(mod, "_studio_lowp", None) == mode for mod in transformer.modules()):
        raise RuntimeError("no eligible linear layers")
    return True


def _convert_fp8(transformer, family: Optional[str] = None) -> bool:
    """Frozen base linears to weight-cached fp8 (``diffusion_frozen_fp8``): rowwise float scales, as DiT activation
    outliers (Z-Image ~2.7e6) crush a tensorwise one."""
    from core.training.diffusion_frozen_fp8 import convert_frozen_fp8

    _, failed = convert_frozen_fp8(
        transformer, {fqn for fqn, _ in _lowp_targets(transformer, _fp8_module_filter, family)}
    )
    transformer._studio_fp8_failed = failed
    return _require_lowp_modules(transformer, "fp8")


def _probe_lowp(convert, transformer) -> None:
    """Tiny converted Linear fwd + bwd so an unverified arch falls back to bf16 at load, not mid-run.
    48 rows: not a multiple of 16 / 32."""
    import torch
    import torch.nn as nn

    try:
        device = next(transformer.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        return

    class _Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = nn.Linear(64, 64, bias = False)

        def forward(self, x):
            return self.ff(x)

    # Forked RNG: the probe's init + randn run after the seed and would shift the noise / timestep draws vs bf16.
    with torch.random.fork_rng(devices = [device]):
        probe = _Probe().to(device = device, dtype = torch.bfloat16).requires_grad_(False)
        convert(probe)
        x = torch.randn(1, 48, 64, device = device, requires_grad = True)
        with torch.autocast("cuda", dtype = torch.bfloat16):
            y = probe(x)
        y.float().sum().backward()
        torch.cuda.synchronize(device)
    if not (torch.isfinite(y).all() and torch.isfinite(x.grad).all()):
        raise RuntimeError("probe GEMM returned non-finite values")


def _apply_fp8_training(
    transformer,
    on_event,
    family: Optional[str] = None,
) -> bool:
    """fp8 compute + storage for the frozen base, after add_adapter. Never fatal: falls back to bf16."""
    from core.training.diffusion_fp8_prequant import dequantize_placeholders
    try:
        _probe_lowp(_convert_fp8, transformer)
        converted = _convert_fp8(transformer, family)
        # A prequant base's fp8 layers the conversion did not take stay bf16, as on the dense path.
        dequantize_placeholders(transformer)
        failed = getattr(transformer, "_studio_fp8_failed", 0)
        if failed:
            _emit(
                on_event,
                "warning",
                message = f"fp8: {failed} layer(s) could not be converted and stay bf16.",
            )
        import torch

        device = next(transformer.parameters()).device.type
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "xpu":
            torch.xpu.empty_cache()
        return converted
    except Exception as exc:  # noqa: BLE001 -- fp8 is an optimisation, never fatal
        dequantize_placeholders(transformer)
        _emit(on_event, "warning", message = f"fp8 training unavailable, using bf16 compute: {exc}")
        return False


def _mx_module_filter(
    mod,
    fqn: str,
    exclude_tokens: tuple[str, ...] = (),
) -> bool:
    """32-wide MX blocks and a bf16 weight (the MX casts assert it)."""
    import torch
    return _lowp_module_filter(mod, fqn, 32, exclude_tokens) and mod.weight.dtype == torch.bfloat16


def _mxfp8_training_config():
    """RCEIL MX config: torchao 0.14-0.16 ``MXLinearConfig``, else 0.17+ with hp wgrad: torchao computes grad_weight
    even for a frozen weight, and its dim1 cast asserts tokens % 32 == 0 (the gradient is discarded anyway)."""
    try:
        from torchao.prototype.mx_formats import MXLinearConfig
        return MXLinearConfig.from_recipe_name("mxfp8_cublas_rceil")
    except ImportError:
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        return MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP)


def _mx_linear_class():
    try:
        from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
        return MXFP8Linear
    except ImportError:
        return None


_ROW_PADDED_MX: dict = {}


def _row_padded_mx_linear(base_cls):
    """torchao <= 0.16 MXLinear has no hp wgrad and asserts tokens % 32 == 0: pad zero rows, slice off
    (MX scales are per block along the feature dim, so real rows are unchanged)."""
    cls = _ROW_PADDED_MX.get(base_cls)
    if cls is None:
        import torch.nn.functional as F
        def forward(self, x):
            shape = x.shape
            rows = x.reshape(-1, shape[-1])
            m = rows.shape[0]
            y = base_cls.forward(self, F.pad(rows, (0, 0, 0, (-m) % 32)))[:m]
            return y.reshape(*shape[:-1], y.shape[-1])

        cls = _ROW_PADDED_MX[base_cls] = type(
            f"RowPadded{base_cls.__name__}", (base_cls,), {"forward": forward}
        )
    return cls


def _autocast_mx_input(module, args):
    import torch

    x = args[0] if args else None
    if (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.float32
        and x.is_cuda
        and torch.is_autocast_enabled("cuda")
    ):
        return (x.to(torch.get_autocast_dtype("cuda")), *args[1:])
    return None


def _convert_mxfp8(transformer, family: Optional[str] = None) -> bool:
    import torch.nn as nn

    config = _mxfp8_training_config()
    targets = [
        (fqn, mod)
        for fqn, mod in _lowp_targets(transformer, _mx_module_filter, family)
        if type(mod) is nn.Linear
    ]
    if not hasattr(config, "wgrad_with_hp"):
        from torchao.quantization import quantize_

        names = {fqn for fqn, _ in targets}
        quantize_(transformer, config, filter_fn = lambda mod, fqn: fqn in names)
        for _, mod in targets:
            mod.__class__ = _row_padded_mx_linear(mod.__class__)
    else:
        linear_cls = _mx_linear_class()
        if linear_cls is not None:
            for _, mod in targets:
                mod.__class__ = linear_cls
                mod.kernel_preference = config.kernel_preference
                mod.scale_calculation_mode = config.scale_calculation_mode
                mod.wgrad_with_hp = config.wgrad_with_hp
        else:
            from torchao.quantization import quantize_

            targets = [(fqn, mod) for fqn, mod in targets if mod.bias is None]
            names = {fqn for fqn, _ in targets}
            quantize_(transformer, config, filter_fn = lambda mod, fqn: fqn in names)
        # The MX linear asserts bf16 input above autocast (fp32 LayerNorm output into FLUX.2 ff.linear_in raised).
        for _, mod in targets:
            mod.register_forward_pre_hook(_autocast_mx_input)
    for _, mod in targets:
        mod._studio_lowp = "mxfp8"
    return _require_lowp_modules(transformer, "mxfp8")


def _apply_mxfp8_training(
    transformer,
    on_event,
    family: Optional[str] = None,
) -> bool:
    """torchao mxfp8 compute for the frozen base (speed only), after add_adapter; never fatal.
    0.18+ swaps the class to MXFP8Linear, not ``quantize_``: its wrapper drops the bias (pytorch/ao#4846),
    so on torchao 0.17 biased linears stay bf16."""
    try:
        _probe_lowp(_convert_mxfp8, transformer)
        return _convert_mxfp8(transformer, family)
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
    """Pure policy for base_precision="auto": nf4 for a prequant base or no CUDA; else the fastest
    dense mode whose weights + headroom (activations, optimizer, cache) fit the free VRAM at
    decision time. bf16 + regional compile is the measured speed winner (2.3-2.6x over nf4 on
    B200); fp8 stays an explicit opt-in: it trades e4m3 rounding (~0.5% loss vs bf16) for ~half the base weight
    memory and only ~7% speed at LoRA-training shapes. int8 must still
    materialise the full bf16 transformer before ``quantize_`` shrinks it module-by-module, so
    its band requires the dense-load transient (1.15x dense) to fit -- what int8 buys in that
    band is steady-state headroom for activations and the latent cache, not load-time memory.
    int8 also needs torchao at runtime (``_int8_quantize_base`` has no fallback, unlike fp8), so
    auto only picks it when torchao is importable and drops to nf4 otherwise.
    ``capability``/``has_fp8`` remain parameters so the policy can be revisited per GPU
    generation without changing callers."""
    _ = capability, has_fp8
    if prequant or device != "cuda" or not free_gb or not dense_gb:
        return "nf4"
    if free_gb > dense_gb * 1.5:
        return "bf16"
    if free_gb > dense_gb * 1.15:
        return "int8" if has_torchao else "nf4"
    return "nf4"


def _dense_bf16_gb(spec, base_model: str) -> float:
    """Per-base overrides, not the family size: flux.2-klein's 9B transformer is 2.3x its 4B default."""
    try:
        from core.inference.diffusion_auto_policy import base_repo_bf16_components_gb
        components = base_repo_bf16_components_gb(base_model)
        if components:
            return float(components[0])
    except Exception:  # noqa: BLE001 -- table miss / import failure -> the family number
        pass
    return float(spec.dense_bf16_gb)


def _resolve_base_precision(cfg, spec, device) -> str:
    """Explicit dense modes are re-checked against the live device, since they are CUDA-only."""
    mode = (cfg.base_precision or "nf4").strip().lower()
    if mode != "auto":
        if mode in ("bf16", "int8", "fp8", "mxfp8") and device != "cuda":
            raise ValueError(
                f"base_precision={mode!r} needs a CUDA GPU; this host has none. "
                f"Use base_precision='nf4' or 'auto'."
            )
        if mode in ("int8", "fp8", "mxfp8") and torch_is_rocm():
            raise ValueError(
                f"base_precision={mode!r} is a torchao NVIDIA tensor-core path (int8 sm_80+, fp8 "
                "sm_89+, mxfp8 sm_100+) and this is a ROCm/AMD GPU. Use base_precision='nf4', "
                "'bf16', or 'auto'."
            )
        # int8 has no runtime fallback; find_spec is satisfied by the Windows-ROCm stub.
        if mode == "int8" and not has_functional_torchao():
            raise ValueError(
                "base_precision='int8' needs a functional torchao install; this host's "
                "torchao is missing or the non-functional Windows-ROCm stub. Use "
                "base_precision='nf4', 'bf16', or 'auto'."
            )
        # The stub fakes torchao.float8 / mx_formats success, so the run would silently train bf16.
        if mode in ("fp8", "mxfp8") and is_stubbed("torchao"):
            raise ValueError(
                f"base_precision={mode!r} is not available on this host: torchao is the "
                "non-functional Windows-ROCm stub. Use base_precision='nf4', 'bf16', or 'auto'."
            )
        # mxfp8 needs Blackwell (sm100+): its MX GEMM raises at the first training step, after a full dense load.
        # Re-check here to fail fast for a stale client.
        # fp8 needs sm89+ (Ada / Hopper / Blackwell): below it the first float8 _scaled_mm raises inside the loop.
        if mode == "fp8" and device == "cuda":
            try:
                import torch
                fp8_ok = torch.cuda.get_device_capability() >= (8, 9)
            except Exception:  # noqa: BLE001 -- probe failure -> treat as unsupported, fail fast
                fp8_ok = False
            if not fp8_ok:
                raise ValueError(
                    "base_precision='fp8' needs an Ada, Hopper or Blackwell (sm89+) GPU; this GPU is "
                    "older. Use base_precision='bf16', 'int8', 'nf4', or 'auto'."
                )
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
    # auto may resolve to dense modes only under bf16 compute; otherwise stay on nf4.
    if getattr(cfg, "mixed_precision", "bf16") != "bf16":
        return "nf4"
    prequant = repo_is_prequantized(cfg.base_model)
    free_gb = None
    capability = None
    has_fp8 = False
    has_torchao = has_functional_torchao() and not torch_is_rocm()
    if device == "cuda":
        try:
            import torch

            # Windows ROCm over-reports free VRAM (#8403).
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

    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    pooled = torch.cat([e[1] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = entries[0][2].to(device = device, dtype = torch.float32)
    return (pe, pooled, text_ids)


# Per-run cache of step-invariant FLUX conditioning tensors; cleared at run start.
_FLUX_STATIC: dict[tuple, tuple] = {}


def _flux_static_inputs(
    bsz,
    h,
    w,
    device,
    guidance_value = 1.0,
):
    import torch
    from diffusers import FluxPipeline

    key = (bsz, h, w, str(device), float(guidance_value))
    hit = _FLUX_STATIC.get(key)
    if hit is None:
        # Position ids drive RoPE and are indices, so keep them float32.
        img_ids = FluxPipeline._prepare_latent_image_ids(bsz, h // 2, w // 2, device, torch.float32)
        guidance = torch.full((bsz,), float(guidance_value), device = device, dtype = torch.float32)
        hit = _FLUX_STATIC[key] = (img_ids, guidance)
    return hit


def _flux_forward(
    transformer,
    noisy,
    timesteps,
    sigmas,
    embeds_batch,
    cfg,
    device,
    weight_dtype,
    guidance = 1.0,
):
    from diffusers import FluxPipeline

    pe, pooled, text_ids = embeds_batch
    bsz, c, h, w = noisy.shape
    packed = FluxPipeline._pack_latents(noisy, bsz, c, h, w)
    # Training feeds guidance 1.0 (diffusers dreambooth convention); sampling feeds the inference 3.5.
    img_ids, guidance = _flux_static_inputs(bsz, h, w, device, guidance)
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

    # AutoencoderKLQwenImage is a 3D (video) VAE: add a temporal dim.
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
    # A single unpadded sample keeps the legacy None mask.
    if len(entries) == 1 and entries[0][1] is None and target == seqs[0]:
        mask_b = None
    return (pe_b, mask_b)


def _qwen_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import QwenImagePipeline

    pe, mask = embeds_batch
    bsz, c, f, h, w = noisy.shape
    packed = QwenImagePipeline._pack_latents(noisy, bsz, c, h, w)
    # Each entry is a LIST of one tuple: the transformer indexes sample[0] / sample[1:].
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
            emb = pe[0] if isinstance(pe, (list, tuple)) else pe
            out.append((emb.cpu(),))
    return out


def _zimage_encode_latents(vae, pixel_values):
    import torch
    with torch.no_grad():
        lat = vae.encode(pixel_values.to(torch.float32)).latent_dist.mode()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor


def _zimage_encode_latent_stats(vae, pixel_values):
    # Z-Image trains from the posterior mode, so B is None and the loop skips the per-step draw.
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
    # Timestep convention is REVERSED ((1000 - t) / 1000) and the prediction is NEGATED.
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
    # The krea repo ships transformers-5.x configs the pinned 4.x cannot parse (diffusion_krea2.py).
    import torch
    from core.inference.diffusion_krea2 import load_krea2_pipeline

    pipe = load_krea2_pipeline(
        cfg.base_model, torch.bfloat16, hf_token = cfg.hf_token, with_transformer = False
    )
    pipe.vae.to(device, dtype = torch.float32)
    return pipe, pipe.vae


def _krea2_load_transformer(cfg, device, weight_dtype, base_precision):
    from diffusers import Krea2Transformer2DModel
    return _load_dit_transformer(Krea2Transformer2DModel, cfg, device, base_precision)


def _krea2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    out = []
    with torch.no_grad():
        for cap in captions:
            pe, mask = pipe.encode_prompt(
                prompt = cap,
                device = device,
                num_images_per_prompt = 1,
                max_sequence_length = 512,
            )
            out.append((pe.cpu(), mask.cpu()))
    return out


_krea2_encode_latents = _qwen_encode_latents
_krea2_encode_latent_stats = _qwen_encode_latent_stats


def _krea2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    pe_b = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    mask_b = torch.cat([e[1] for e in entries]).to(device)
    return (pe_b, mask_b)


def _krea2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    from diffusers import Krea2Pipeline

    pe, mask = embeds_batch
    # Krea2Pipeline._pack_latents is an instance method, so packing is inlined.
    bsz, c, _f, h, w = noisy.shape
    packed = noisy.reshape(bsz, c, h // 2, 2, w // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5).reshape(bsz, (h // 2) * (w // 2), c * 4)
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


# FLUX.2 latents train patchified and batch-norm-normalised, from the posterior MODE.
_FLUX2_COMMON_TARGETS = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
)
# A bare "to_out" also matches the double-stream ModuleList, which PEFT cannot wrap.
_FLUX2_KLEIN_TARGETS = _FLUX2_COMMON_TARGETS + tuple(
    f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)
)
_FLUX2_DEV_TARGETS = _FLUX2_COMMON_TARGETS + tuple(
    f"single_transformer_blocks.{i}.attn.to_out" for i in range(48)
)
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
    # The FLUX.2 VAE normalises with its BatchNorm running stats, not shift/scaling_factor.
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat)
    return (lat - mean) / std


def _flux2_encode_latent_stats(vae, pixel_values):
    return _flux2_encode_latents(vae, pixel_values), None


def _flux2_collate(
    entries,
    device,
    weight_dtype,
    pad_to = None,
):
    import torch

    # text_ids are PER-SAMPLE (unlike FLUX.1's shared grid).
    pe = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    text_ids = torch.cat([e[1] for e in entries]).to(device = device, dtype = torch.float32)
    return (pe, text_ids)


_FLUX2_STATIC: dict[tuple, Any] = {}


def _flux2_static_img_ids(latents, device):
    from diffusers import Flux2Pipeline

    key = (latents.shape[0], latents.shape[-2], latents.shape[-1], str(device))
    hit = _FLUX2_STATIC.get(key)
    if hit is None:
        hit = _FLUX2_STATIC[key] = Flux2Pipeline._prepare_latent_ids(latents).to(device = device)
    return hit


def _flux2_forward(
    transformer,
    noisy,
    timesteps,
    sigmas,
    embeds_batch,
    cfg,
    device,
    weight_dtype,
    guidance_value = None,
):
    import torch
    from diffusers import Flux2Pipeline

    pe, text_ids = embeds_batch
    packed = Flux2Pipeline._pack_latents(noisy)
    img_ids = _flux2_static_img_ids(noisy, device)
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.full(
            (noisy.shape[0],),
            _FLUX2_TRAIN_GUIDANCE if guidance_value is None else guidance_value,
            device = device,
            dtype = torch.float32,
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
    # diffusers 0.39 stacks the unpacked samples; other builds return a list.
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


# Targets are fully qualified: a bare to_q would also match audio and cross-modality attention.
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

# Rotary time is in SECONDS: at 24 fps a still lands on a generated clip's first frame
# (Lightricks pin images to fps=1.0 instead).
_LTX2_TRAIN_FPS = 24.0


def _ltx2_pipeline_cls():
    """``diffusers.LTX2Pipeline``, importable on the pinned transformers (see ltx2_import_compat)."""
    from core.inference.ltx2_import_compat import ensure_ltx2_pipelines_importable

    ensure_ltx2_pipelines_importable()
    from diffusers import LTX2Pipeline

    return LTX2Pipeline


def _ltx2_load_conditioners(cfg, device, weight_dtype):
    LTX2Pipeline = _ltx2_pipeline_cls()

    pipe, vae = _load_pipe_without_transformer(LTX2Pipeline, cfg, device)
    if getattr(pipe, "connectors", None) is not None:
        pipe.connectors.to(device)
    return pipe, vae


def _ltx2_load_transformer(cfg, device, weight_dtype, base_precision):
    from diffusers import LTX2VideoTransformer3DModel
    return _load_dit_transformer(LTX2VideoTransformer3DModel, cfg, device, base_precision)


def _ltx2_encode_prompts(pipe, captions, device):
    import torch

    _encoders_to_device(pipe, device)
    # Gemma3 hidden states are stacked per layer; only the connector output reaches the transformer.
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
            # Read AFTER encode_prompt, which sets padding_side="left"; the connector mask depends on it.
            padding_side = getattr(getattr(pipe, "tokenizer", None), "padding_side", "left")
            video_emb, audio_emb, conn_mask = pipe.connectors(pe, mask, padding_side = padding_side)
            out.append((video_emb.cpu(), audio_emb.cpu(), conn_mask.cpu()))
    return out


def _ltx2_latent_affine(vae, ref):
    import torch

    mean = vae.latents_mean.to(device = ref.device, dtype = ref.dtype).view(1, -1, 1, 1, 1)
    std = vae.latents_std.to(device = ref.device, dtype = ref.dtype).view(1, -1, 1, 1, 1)
    return mean, std / float(vae.config.scaling_factor or 1.0)


def _ltx2_encode_latents(vae, pixel_values):
    import torch

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

    video = torch.cat([e[0] for e in entries]).to(device = device, dtype = weight_dtype)
    audio = torch.cat([e[1] for e in entries]).to(device = device, dtype = weight_dtype)
    mask = torch.cat([e[2] for e in entries]).to(device)
    return (video, audio, mask)


def _ltx2_audio_token_count(config, num_pixel_frames: int, fps: float) -> int:
    """Floored at one token because the transformer indexes the audio stream unconditionally."""
    per_second = (
        float(config.audio_sampling_rate)
        / float(config.audio_hop_length)
        / float(config.audio_scale_factor)
    )
    return max(1, round((num_pixel_frames / float(fps)) * per_second))


def _ltx2_pack(latents, conf):
    """[B,C,F,H,W] -> [B, F*H*W, C] via the pipeline's own patchifier."""
    LTX2Pipeline = _ltx2_pipeline_cls()
    return LTX2Pipeline._pack_latents(latents, conf.patch_size, conf.patch_size_t)


def _ltx2_unpack(pred, f, h, w, conf):
    """The inverse of ``_ltx2_pack``, back to the 5-D shape ``target = noise - latents`` has."""
    LTX2Pipeline = _ltx2_pipeline_cls()
    return LTX2Pipeline._unpack_latents(pred, f, h, w, conf.patch_size, conf.patch_size_t)


def _ltx2_audio_state(sigmas, bsz, audio_len, channels, device, dtype):
    """Still images have no audio, so the placeholder follows the flow state of a zero clean latent."""
    import torch

    noise = torch.randn((bsz, audio_len, channels), device = device, dtype = dtype)
    # sigmas arrives broadcast to the 5-D video latent; the audio stream is 3-D.
    return sigmas.reshape(bsz, 1, 1) * noise


def _ltx2_forward(transformer, noisy, timesteps, sigmas, embeds_batch, cfg, device, weight_dtype):
    video_emb, audio_emb, mask = embeds_batch
    bsz, _c, f, h, w = noisy.shape
    conf = transformer.config
    packed = _ltx2_pack(noisy, conf)

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
        # LTX-2 conditions on the UNSCALED timestep, unlike FLUX / Qwen's timestep / 1000.
        timestep = timesteps,
        sigma = timesteps,
        encoder_attention_mask = mask,
        audio_encoder_attention_mask = mask,
        num_frames = f,
        height = h,
        width = w,
        fps = _LTX2_TRAIN_FPS,
        audio_num_frames = audio_len,
        # Keep the placeholder audio stream from perturbing the video prediction.
        isolate_modalities = True,
        return_dict = False,
    )
    return _ltx2_unpack(pred, f, h, w, conf)


def _ltx2_save(pipe_cls, out_dir, transformer_lora_layers):
    LTX2Pipeline = _ltx2_pipeline_cls()
    LTX2Pipeline.save_lora_weights(
        save_directory = out_dir,
        transformer_lora_layers = transformer_lora_layers,
        weight_name = DEFAULT_LORA_FILENAME,
    )


def _vae8_latent_shape(vae, height, width):
    return (1, vae.config.latent_channels, height // 8, width // 8), (height // 16) * (width // 16)


def _flux_decode_latents(vae, latents):
    lat = latents / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(lat.to(vae.dtype)).sample


def _qwen_latent_shape(vae, height, width):
    return (1, vae.config.z_dim, 1, height // 8, width // 8), (height // 16) * (width // 16)


def _qwen_decode_latents(vae, latents):
    mean, std = _qwen_latent_affine(vae, latents)
    return vae.decode((latents * std + mean).to(vae.dtype)).sample[:, :, 0]


def _flux2_latent_shape(vae, height, width):
    return (1, vae.config.latent_channels * 4, height // 16, width // 16), (height // 16) * (
        width // 16
    )


def _flux2_decode_latents(vae, latents):
    import torch
    from diffusers import Flux2Pipeline

    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents)
    lat = Flux2Pipeline._unpatchify_latents(latents * std + mean)
    return vae.decode(lat.to(vae.dtype)).sample


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
        latent_shape = _vae8_latent_shape,
        decode_latents = _flux_decode_latents,
        sample_forward_kwargs = {"guidance": 3.5},
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
        latent_shape = _qwen_latent_shape,
        decode_latents = _qwen_decode_latents,
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
        latent_shape = _vae8_latent_shape,
        decode_latents = _flux_decode_latents,
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
        latent_shape = _qwen_latent_shape,
        decode_latents = _qwen_decode_latents,
    ),
    "flux.2-klein": _FamilySpec(
        family = "flux.2-klein",
        lora_targets = _FLUX2_KLEIN_TARGETS,
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
        latent_shape = _flux2_latent_shape,
        decode_latents = _flux2_decode_latents,
    ),
    "flux.2-dev": _FamilySpec(
        family = "flux.2-dev",
        lora_targets = _FLUX2_DEV_TARGETS,
        force_bf16 = True,
        dense_bf16_gb = 64.5,
        load_conditioners = _flux2_load_conditioners,
        load_transformer = _flux2_load_transformer,
        encode_prompts = _flux2_encode_prompts,
        encode_latents = _flux2_encode_latents,
        encode_latent_stats = _flux2_encode_latent_stats,
        collate = _flux2_collate,
        forward = _flux2_forward,
        save = _flux2_save,
        latent_shape = _flux2_latent_shape,
        decode_latents = _flux2_decode_latents,
        # Flux2Pipeline's default guidance_scale, which Studio's FLUX.2-dev inference also uses.
        sample_forward_kwargs = {"guidance_value": 4.0},
    ),
    "ltx-2": _FamilySpec(
        family = "ltx-2",
        lora_targets = _LTX2_TARGETS,
        force_bf16 = True,
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


# Checked by name (no network) so a missing token fails fast.
_GATED_TRAIN_REPOS = frozenset({"black-forest-labs/flux.1-dev", "black-forest-labs/flux.2-dev"})


def _assert_gated_access(base_model: str, hf_token: Optional[str]) -> None:
    """Raise a clear error before loading a gated base without a token."""
    from core.inference.diffusion_families import _is_local_path

    name = str(base_model or "").strip().lower()
    # A local clone named like the vendor repo is weights on disk, not a Hub fetch.
    if _is_local_path(base_model):
        return
    if name in _GATED_TRAIN_REPOS and not (hf_token and str(hf_token).strip()):
        raise ValueError(
            f"'{base_model}' is a gated Hugging Face repo. Accept its license on the Hub "
            f"and add your HF token in Unsloth settings before training from it."
        )


def _canvas(resolution):
    """(w, h) of the training canvas: a bucket tuple, or the legacy square ``resolution``."""
    if isinstance(resolution, (tuple, list)):
        return int(resolution[0]), int(resolution[1])
    return int(resolution), int(resolution)


def _open_resized(path, resolution):
    """Open + EXIF-orient + cover-resize to the canvas (same geometry as the SDXL loader; a square
    canvas is the legacy short-side resize). Returns the resized PIL image and its (rw, rh)."""
    from PIL import Image, ImageOps

    from core.inference.mcp_images import flattened_rgb

    img = flattened_rgb(ImageOps.exif_transpose(Image.open(path)), background = (255, 255, 255))
    rw, rh = cover_resize_dims(*img.size, *_canvas(resolution))
    return img.resize((rw, rh), Image.LANCZOS), rw, rh


def _to_unit_tensor(img):
    import numpy as np
    import torch

    arr = np.asarray(img, dtype = np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def _load_pixel_tensor(path, resolution, center_crop, random_flip, rng):
    """Load an image -> a normalised [3,H,W] tensor in [-1,1]. Same geometry as the SDXL loader but
    without the SDXL time-ids (DiT families don't use them)."""
    from PIL import Image

    cw, ch = _canvas(resolution)
    img, rw, rh = _open_resized(path, resolution)
    if center_crop:
        left, top = (rw - cw) // 2, (rh - ch) // 2
    else:
        left = rng.randint(0, max(0, rw - cw))
        top = rng.randint(0, max(0, rh - ch))
    img = img.crop((left, top, left + cw, top + ch))
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return _to_unit_tensor(img)


def _load_pixel_tensor_planned(path, resolution, center_crop, u_left, u_top, flip):
    """Deterministic crop and flip for the latent cache; center_crop matches the uncached path exactly."""
    from PIL import Image

    cw, ch = _canvas(resolution)
    img, rw, rh = _open_resized(path, resolution)
    if center_crop:
        left, top = (rw - cw) // 2, (rh - ch) // 2
    else:
        left = min(int(u_left * (rw - cw + 1)), max(0, rw - cw))
        top = min(int(u_top * (rh - ch + 1)), max(0, rh - ch))
    img = img.crop((left, top, left + cw, top + ch))
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
    canvases = None,
):
    """Stats stay fp32 to match the in-loop path; only the sampled result is cast to weight_dtype."""

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
            canvas = cfg.resolution if canvases is None else canvases[i]
            key = (
                pcache.latent_key(path, (u_left, u_top, flip), None if canvases is None else canvas)
                if pcache is not None
                else None
            )
            entry = pcache.get(key) if key is not None else None
            if entry is not None:
                a, b = entry
            else:
                px = (
                    _load_pixel_tensor_planned(path, canvas, cfg.center_crop, u_left, u_top, flip)
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
                # Size-gate the cache off the first REAL variant: the cache can exhaust pinned RAM.
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
    """Cached entries are the family's own CPU embed tuples, so a hit is identical to a fresh encode."""
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


def _load_warm_conditioning(
    pcache,
    image_paths,
    plan,
    to_encode,
    device,
    canvases = None,
):
    """Load the FULL conditioning set (caption embeds + latent posterior stats) from the persistent
    cache. Returns (caption_embeds, latent_cache) on a complete hit, else (None, None) so the
    caller takes the cold path -- any missing/corrupt entry, and also an in-memory holding that
    would blow the host budget (the cold path re-decides that with the VAE resident and can fall
    back to per-step encoding). On success the VAE and text encoders are never loaded."""
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
            entry = pcache.get(
                pcache.latent_key(path, variant, None if canvases is None else canvases[i])
            )
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
    """Draws fresh noise per step in fp32, as an in-loop latent_dist.sample() does; cast after."""
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
    steps_left = None,
) -> bool:
    mode = (cfg.compile_transformer or "auto").strip().lower()
    if device != "cuda" or mode == "off":
        return False
    # torch.compile cannot trace the torchao int8 subclass in training, so int8 runs eager.
    if base_precision == "int8":
        return False
    if mode == "on":
        return True
    if base_precision == "nf4":
        steps = cfg.train_steps if steps_left is None else steps_left
        return steps >= NF4_COMPILE_MIN_STEPS and _nf4_compile_supported()
    return base_precision in ("bf16", "fp8", "mxfp8")


# Lowest stack where bnb 4-bit ops are torch custom ops (0 graph breaks, loss matches eager); ROCm unmeasured, stays eager.
NF4_COMPILE_MIN_TORCH = "2.10"
NF4_COMPILE_MIN_BNB = "0.46.1"
# The nf4 compile only pays back after a few hundred steps, so a short run stays eager.
NF4_COMPILE_MIN_STEPS = 200


def _nf4_compile_supported() -> bool:
    try:
        import importlib.metadata

        import torch
        from packaging.version import Version

        if torch_is_rocm():
            return False
        if Version(torch.__version__) < Version(NF4_COMPILE_MIN_TORCH):
            return False
        return Version(importlib.metadata.version("bitsandbytes")) >= Version(NF4_COMPILE_MIN_BNB)
    except Exception:  # noqa: BLE001 -- unknown versions keep the historical eager path
        return False


def _maybe_compile_transformer(
    transformer,
    cfg,
    base_is_bnb,
    device,
    on_event,
    base_precision = "nf4",
    steps_left = None,
) -> bool:
    """Regionally compile the transformer blocks (diffusers compile_repeated_blocks) after the LoRA
    is attached. Never fatal: a wrap failure falls back to eager with a warning event, and
    dynamo's suppress_errors keeps a frame that fails to COMPILE at the first step running eager
    instead of raising mid-run."""
    if not _should_compile(cfg, base_is_bnb, device, base_precision, steps_left):
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
            # Z-Image has ~11 distinct block shapes, over dynamo's default recompile limit of 8.
            for attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo_cfg, attr):
                    setattr(dynamo_cfg, attr, max(getattr(dynamo_cfg, attr) or 0, 64))
            if hasattr(dynamo_cfg, "suppress_errors"):
                dynamo_cfg.suppress_errors = True
        # dynamic=True matches the inference speed layer: on torch 2.10 / B200 the dynamic=False specialisation failed
        # with CUBLAS_STATUS_EXECUTION_FAILED.
        try:
            # torch 2.12 / 2.13 inductor CantSplit in the dynamic joint-stream cat backward (FLUX.1 fp8); probe-gated.
            from core.inference.diffusion_inductor_backports import install as _install_backports
            _install_backports()
        except Exception:  # noqa: BLE001 -- the backport is best-effort
            pass
        fn(fullgraph = not base_is_bnb, dynamic = True)
        return True
    except Exception as exc:  # noqa: BLE001 -- optimisation only, never fatal
        _emit(on_event, "warning", message = f"torch.compile disabled (eager fallback): {exc}")
        return False


def _dynamo_compile_counters() -> Optional[tuple[int, int, int]]:
    """(graph breaks, failed frames, unique graphs), or None."""
    try:
        from torch._dynamo.utils import counters
        frames = counters["frames"]
        return (
            sum(counters["graph_break"].values()),
            frames["total"] - frames["ok"],
            counters["stats"]["unique_graphs"],
        )
    except Exception:  # noqa: BLE001 -- diagnostics only
        return None


def _compile_engagement_warning(before, after) -> Optional[str]:
    """Why compiled blocks run eager since ``before`` (suppress_errors hides failures), else None."""
    if before is None or after is None:
        return None
    breaks, failed, graphs = (a - b for a, b in zip(after, before))
    if breaks > 0 or failed > 0:
        return (
            f"torch.compile did not fully engage ({breaks} graph break(s), {failed} frame(s) "
            f"fell back to eager): training continues, slower than compiled."
        )
    if graphs <= 0:
        return "torch.compile produced no graphs: the transformer blocks are running eager."
    return None


def run_dit_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """Train a flow-matching DiT LoRA (FLUX.1 / FLUX.2 / Qwen-Image / Z-Image / Krea 2 / LTX-2) and
    export it. Resumable: ``cfg.resume_from_checkpoint`` restores the adapter, optimizer moments,
    LR position, EMA shadow, sampler cycle and RNG streams from a ``checkpoint-<N>`` bundle, and
    the loop runs steps N+1..train_steps (the TARGET TOTAL). A stop-and-save and every
    ``cfg.save_steps`` interval write such a bundle."""
    cfg = resolve_bucketing(config.normalized())
    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        raise ValueError(f"No DiT trainer for family {cfg.resolved_family!r}")

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

    device = resolve_train_device()
    # is_bf16_supported() counts emulation on CUDA and XPU, so gate on NATIVE bf16.
    if device == "cuda" and not flow_bf16_trainable():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU (Ampere or newer); "
            "this CUDA device does not support bf16."
        )
    if device == "xpu" and not native_bf16_supported_xpu():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU; this XPU device does not "
            "support bf16 natively."
        )
    weight_dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32

    _assert_trusted_base_model(cfg.base_model)
    # Check the fetched repo: the canonical id would raise for a gated base redirected to its mirror.
    _assert_gated_access(cfg.fetch_base_model or cfg.base_model, cfg.hf_token)
    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    # Resolve num_epochs into a concrete train_steps now the dataset size is known, and rebind cfg so every downstream
    # read agrees.
    cfg = replace(
        cfg,
        train_steps = resolve_train_steps(cfg, len(pairs), [p for p, _ in pairs]),
        num_epochs = 0,
    )
    # Validate a resume against this run's identity BEFORE the multi-GB phased load, using the RESOLVED LoRA targets
    # rather than the generic default the config carries.
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
            # A stop with save=false is a DISCARD however early it lands.
            discarded = not save_on_stop,
        )
        return str(out_dir)

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

    # diffusers honours no env var for its bars and is only imported from here.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a training run
        pass

    use_lora_targets = _select_lora_targets(cfg.lora_target_modules, spec.lora_targets)
    out_dir = Path(cfg.output_dir).expanduser()
    # cfg.base_model stays canonical for the sidecar, completion event and resume identity.
    fetch_cfg = replace(cfg, base_model = cfg.fetch_base_model or cfg.base_model)

    # When every latent variant AND caption embedding is cached, the encoders never load.
    image_paths = [p for p, _ in pairs]
    captions = [c for _, c in pairs]
    uniq = sorted(set(captions))
    # CFG dropout needs the empty-prompt embedding precomputed before the encoders are freed.
    cfg_dropout = float(getattr(cfg, "cfg_dropout", 0.0) or 0.0)
    to_encode = uniq + ([""] if cfg_dropout > 0 and "" not in uniq else [])
    # Own dict: caption_embeds sizes the qwen compile pad bucket, which a long preview prompt must not grow.
    sample_plan = plan_samples(cfg, spec.family, captions) if spec.decode_latents else None
    sample_texts = sample_plan.encode_texts if sample_plan is not None else []
    sample_embeds: dict = {}
    # Kept past a failed round (which drops sample_plan) so a discard still removes what was written.
    sample_owner = sample_plan
    use_cache = cfg.cache_latents and os.environ.get(
        "UNSLOTH_DIFFUSION_NO_LATENT_CACHE", ""
    ) not in ("1", "true")
    pcache = None
    if getattr(cfg, "cond_cache_dir", None):
        try:
            # Namespace on the CHECKPOINT: keys carry only content, so caches would cross models.
            from .diffusion_train_extras import source_revision  # noqa: PLC0415
            namespace = f"{spec.family}_{cfg.base_model}_{source_revision(fetch_cfg.base_model)}"
            pcache = PersistentConditioningCache(cfg.cond_cache_dir, namespace, cfg.resolution)
        except Exception as exc:  # noqa: BLE001 -- the cache is an optimisation, never fatal
            _emit(on_event, "warning", message = f"conditioning cache disabled: {exc}")
    # The crop/flip variant plan is seed-deterministic, so persistent keys are stable across runs and the warm check
    # can run before anything loads.
    canvases, crop_room = plan_image_canvases(cfg, image_paths)
    plan = _plan_cache_variants(
        len(image_paths),
        cfg.cache_variants,
        cfg.center_crop,
        cfg.random_flip,
        cfg.seed,
        crop_room = crop_room,
    )

    pipe = None
    vae = None
    caption_embeds = None
    # Over the host-memory budget: keep the VAE and encode in-loop.
    latent_cache = None
    if pcache is not None and use_cache:
        caption_embeds, latent_cache = _load_warm_conditioning(
            pcache, image_paths, plan, to_encode, device, canvases
        )
        if caption_embeds is not None:
            _emit(
                on_event,
                "preparing",
                stage = "cache_latents",
                done = len(image_paths),
                total = len(image_paths),
            )
    if caption_embeds is not None and sample_texts:
        # Warm conditioning skips the VAE and encoders, but previews need the VAE and their own embeddings. RNG-isolated:
        # a run without previews never makes this load, so it must not move the training streams.
        with isolated_sampling(device, False):
            pipe, vae = spec.load_conditioners(fetch_cfg, device, weight_dtype)
            encoded = _encode_prompts_cached(spec, pipe, sample_texts, device, pcache)
        sample_embeds = dict(zip(sample_texts, encoded))
        _free_text_encoders(pipe)
    if caption_embeds is None:
        latent_cache = None
        pipe, vae = spec.load_conditioners(fetch_cfg, device, weight_dtype)
        extra = [t for t in sample_texts if t not in to_encode]
        encoded = _encode_prompts_cached(spec, pipe, to_encode + extra, device, pcache)
        caption_embeds = {cap: emb for cap, emb in zip(to_encode, encoded)}
        all_embeds = dict(zip(to_encode + extra, encoded))
        sample_embeds = {t: all_embeds[t] for t in sample_texts}
        _free_text_encoders(pipe)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "xpu":
            torch.xpu.empty_cache()

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
                canvases = canvases,
            )
            if latent_cache is LATENT_CACHE_OVER_BUDGET:
                # Over the host-memory budget: keep the VAE and encode in-loop.
                latent_cache = None
            elif latent_cache is None:
                _emit(
                    on_event,
                    "complete",
                    output_dir = str(out_dir),
                    lora_path = None,
                    stopped = True,
                    steps_run = 0,
                    discarded = not _save_on_stop(),
                )
                return str(out_dir)
    # VAE parked on the host between rounds when the latent cache would have freed it: training VRAM unchanged.
    sample_vae = vae if sample_plan is not None else None
    if sample_vae is not None and latent_cache is not None:
        sample_vae.to("cpu")
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
        elif device == "xpu":
            torch.xpu.empty_cache()
    # Own stream so the loop's index/noise draws stay seed-deterministic.
    variant_rng = random.Random(cfg.seed + 1)

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
        enable_diffusion_gradient_checkpointing(transformer)
    cast_training_params(transformer, dtype = torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    # Convert AFTER the LoRA attaches so the adapter modules stay high precision.
    if base_precision == "int8":
        _int8_quantize_base(transformer, cfg.resolved_family)
    if base_precision == "fp8" and not _apply_fp8_training(
        transformer, on_event, cfg.resolved_family
    ):
        base_precision = "bf16"
    if base_precision == "mxfp8" and not _apply_mxfp8_training(
        transformer, on_event, cfg.resolved_family
    ):
        base_precision = "bf16"

    # Initialised after the precision conversions so it tracks the final fp32 objects.
    ema = LoRAEMA(transformer, decay = cfg.ema_decay) if getattr(cfg, "ema_decay", 0.0) else None

    # A resume runs only the remaining steps, so the nf4 compile break-even is judged on those.
    compiled = _maybe_compile_transformer(
        transformer,
        cfg,
        base_is_bnb,
        device,
        on_event,
        base_precision,
        resume_steps_left(cfg, identity),
    )
    # Compiled Qwen graphs need one fixed text length across steps.
    qwen_pad_to = None
    if compiled and spec.family == "qwen-image":
        qwen_pad_to = max(e[0].shape[1] for e in caption_embeds.values())

    optimizer = _make_optimizer(
        lora_params, cfg.learning_rate, recorded_optimizer_class(cfg, identity)
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        fetch_cfg.base_model, subfolder = "scheduler", token = cfg.hf_token
    )
    flow_shift = getattr(cfg, "flow_shift", None)
    if flow_shift is None and spec.family in AUTO_FLOW_SHIFT_FAMILIES:
        flow_shift = "auto"
    sigma_table = _training_sigma_table(scheduler, flow_shift)
    shift_active = sigma_table is not scheduler.sigmas
    num_train_ts = scheduler.config.num_train_timesteps
    bell_weights = (
        _bell_loss_weights(num_train_ts).to(device)
        if str(getattr(cfg, "weighting_scheme", "none") or "none") == "bell"
        else None
    )
    # Warmup/decay count optimizer steps; scaling by accumulation would stretch warmup.
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )

    _emit(on_event, "model_load_completed", compiled = compiled, base_precision = base_precision)
    # Read the commit of the repo actually fetched, now that it is on disk.
    identity = with_resolved_revision(identity, fetch_cfg.base_model)
    identity = with_cache_mode(identity, latent_cache is not None)
    # base_precision is resolved now (auto, or fp8/mxfp8 fallback to bf16).
    identity = with_resolved_base_precision(identity, base_precision)

    transformer.train()
    n_images = len(image_paths)
    batch_size = cfg.train_batch_size
    # Permutation-cycle index sampler (shared with the SDXL trainer): visits every image once per cycle, so a short
    # run covers the whole dataset. Bucketed runs draw each batch from one bucket so its latents stack into one shape.
    index_sampler = (
        PermutationBatchSampler(n_images, rng)
        if canvases is None
        else BucketBatchSampler(bucket_groups(canvases), rng)
    )

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
    # Resuming from dir A into a reused dir B: B's bundles are foreign and must be cleared.
    resumed_here = bool(resumed) and resumed_into_this_dir(cfg, out_dir)
    preexisting_checkpoints = snapshot_checkpoints(out_dir)
    if restored is not None:
        was = restored.progress.get("resolved_base_precision")
        if was and was != base_precision:
            # Not fatal, but base_precision="auto" can make this happen silently.
            _emit(
                on_event,
                "warning",
                message = (
                    f"Resuming a checkpoint trained with a {was} base in {base_precision}; "
                    f"set base_precision explicitly to keep them identical."
                ),
            )

    stopped = False
    running_loss = restored.running_loss if restored is not None else 0.0
    peak_gb = 0.0
    t_start = time.time()
    t_steady = None
    t_rate = t_start
    # Starts at the resumed step so a no-op resume still reports the real step.
    done = resumed

    def _save_checkpoint(step: int) -> None:
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
            # The identity records the REQUESTED mode, which cannot tell nf4 from bf16 under "auto".
            progress = {"running_loss": running_loss, "resolved_base_precision": base_precision},
            # NOT discard_existing: a cancelled retrain would destroy the run being retrained.
            discard_existing = False,
            preexisting = preexisting_checkpoints,
        )

    # Reconciles the fp32 LoRA params with the bnb 4-bit base matmuls.
    autocast = (
        torch.autocast(device_type = device, dtype = torch.bfloat16)
        if device in ("cuda", "xpu")
        else nullcontext()
    )

    def _render_dit(index, prompt):
        shape, seq_len = spec.latent_shape(
            sample_vae, sample_plan.resolution, sample_plan.resolution
        )
        extra = spec.sample_forward_kwargs or {}

        def velocity(x, t, sig, embeds):
            with autocast:
                return spec.forward(
                    transformer, x, t, sig, embeds, cfg, device, weight_dtype, **extra
                )

        cond = spec.collate([sample_embeds[prompt]], device, weight_dtype)
        uncond = (
            spec.collate([sample_embeds[""]], device, weight_dtype)
            if sample_plan.uses_cfg
            else None
        )
        latents = euler_flow_sample(
            plan = sample_plan,
            index = index,
            latent_shape = shape,
            image_seq_len = seq_len,
            family = spec.family,
            scheduler_config = scheduler.config,
            velocity = velocity,
            cond = cond,
            uncond = uncond,
            device = device,
            weight_dtype = weight_dtype,
            stop_requested = _stop_during_sample,
        )
        return spec.decode_latents(sample_vae, latents.to(next(sample_vae.parameters()).device))

    # The stop poll drains the request, so one seen mid-round is latched here for the loop to act on.
    stop_latched = False

    def _stop_during_sample() -> bool:
        nonlocal stop_latched
        if not stop_latched and _check_stop():
            stop_latched = True
        return stop_latched

    def _sample(step: int) -> None:
        nonlocal sample_plan, t_rate, t_steady
        if sample_plan is None:
            return
        t_round = time.time()
        parked = next(sample_vae.parameters()).device.type == "cpu"
        try:
            with isolated_sampling(device, compiled):
                transformer.eval()
                if parked:
                    sample_vae.to(device)
                run_sample_round(
                    sample_plan, step, _render_dit, on_event, _emit, _stop_during_sample
                )
        except Exception as exc:  # noqa: BLE001 -- previews are diagnostics, never a reason to lose the run
            _emit(on_event, "warning", message = f"Sample images disabled after an error: {exc}")
            sample_plan = None
        finally:
            transformer.train()
            if parked:
                sample_vae.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "xpu":
                    torch.xpu.empty_cache()
            spent = time.time() - t_round
            t_rate += spent
            if t_steady is not None:
                t_steady += spent

    if sample_plan is not None and resumed < cfg.train_steps:
        _sample(resumed)

    train_peak = TrainingPeakMemory(device)
    compile_watch = _dynamo_compile_counters() if compiled else None
    for opt_step in range(resumed, cfg.train_steps):
        if stop_latched:
            stopped = True
            break
        optimizer.zero_grad(set_to_none = True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            idxs = index_sampler.next_batch(batch_size)
            if canvases is not None:
                idxs = idxs[1]
            if latent_cache is not None:
                latents = _sample_cached_latents(
                    latent_cache, idxs, variant_rng, device, weight_dtype
                )
            else:
                px = torch.stack(
                    [
                        _load_pixel_tensor(
                            image_paths[i],
                            cfg.resolution if canvases is None else canvases[i],
                            cfg.center_crop,
                            cfg.random_flip,
                            rng,
                        )
                        for i in idxs
                    ]
                ).to(device)
                latents = spec.encode_latents(vae, px).to(weight_dtype)

            noise = torch.randn_like(latents)
            timesteps, t_indices = _sample_timesteps(scheduler, latents.shape[0], device)
            sigmas = _gather_sigmas(sigma_table, t_indices, device, weight_dtype, latents.ndim)
            if shift_active:
                # timestep = sigma * num_train_timesteps, gathered in fp32 so bf16 rounding never skews it.
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
            # Pre-clip norm: the UI's health signal.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
        optimizer.step()
        lr_sched.step()
        if ema is not None:
            ema.update(transformer)

        running_loss += step_loss
        done = opt_step + 1
        now = time.time()
        ran_here = done - resumed
        if ran_here == 1:
            # Step 1 pays one-time costs (autotune, compile).
            t_steady = now
        if compile_watch is not None:
            reason = _compile_engagement_warning(compile_watch, _dynamo_compile_counters())
            if reason:
                _emit(on_event, "warning", message = reason)
                compile_watch = None
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            peak_gb = train_peak.gb()
            per_step = batch_size * cfg.gradient_accumulation_steps
            if t_steady is not None and ran_here > 1:
                sps = round((ran_here - 1) * per_step / max(now - t_steady, 1e-6), 3)
            else:
                sps = round(ran_here * per_step / max(now - t_rate, 1e-6), 3)
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
        if not stop_now and sample_plan is not None and sample_plan.due(done, cfg.train_steps):
            train_peak.pause()
            _sample(done)
            train_peak.resume()
            stop_now = stop_latched
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
        # Bundle BEFORE the adapter export, so a failed export is still resumable.
        if stopped and done > 0:
            _save_checkpoint(done)
        layers = get_peft_model_state_dict(transformer)
        spec.save(pipe, str(out_dir), layers)
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        catalog_path = _publish_to_lora_catalog(lora_path, cfg, done)
        if ema is not None and ema.updates > 0:
            try:
                ema_dir = save_ema_adapter(ema, transformer, spec.save, str(out_dir))
                ema_path = str(Path(ema_dir) / DEFAULT_LORA_FILENAME)
            except Exception as exc:  # noqa: BLE001 -- the primary adapter is already saved
                _emit(on_event, "warning", message = f"EMA adapter save failed: {exc}")
        if not stopped:
            # A completed run must not leave its own checkpoint for a later resume to roll back to.
            retire_own_checkpoints(out_dir, preexisting_checkpoints, resumed_here = resumed_here)
        elif not resumed_here:
            # A partial save on a fresh retrain is a LOWER step than leftovers, which would outrank it.
            discard_preexisting_checkpoints(out_dir, preexisting_checkpoints)
    else:
        # Otherwise a discard leaves unresumable optimizer bundles with no delete path in the UI.
        clear_own_checkpoints(out_dir, preexisting_checkpoints)
        discard_samples(sample_owner)
        try:
            out_dir.rmdir()
        except OSError:
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
        discarded = bool(stopped and not _save_on_stop()),
    )
    return str(out_dir)


def _make_optimizer(
    params,
    lr,
    resume_optimizer_class = None,
):
    return make_lora_optimizer(params, lr, resume_optimizer_class)


def _free_text_encoders(pipe) -> None:
    """Frees conditioning modules once embeddings are precomputed, so they do not sit in VRAM."""
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
