# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision casting of the diffusion pipeline's text encoder(s).

The transformer arrives quantised in the GGUF, but the companion text encoder loads dense
(bf16) and is often the largest resident component (Qwen3 / T5-XXL / Mistral run to many GB).
This shrinks it in place, with four backends:

  fp8         - diffusers layerwise casting: 8-bit (e4m3) storage, upcast per layer. ~2x
                smaller. Any fp8-capable CUDA card (cc >= 8.9).
  fp8_dynamic - torchao dynamic fp8 COMPUTE (per-row): keeps the matmul in fp8 on the tensor
                cores (torch._scaled_mm) instead of upcasting. ~2x smaller + speedup; cc >= 8.9.
  int8        - torchao dynamic int8 COMPUTE (per-token act + per-channel weight, _int_mm),
                with per-layer keep-bf16 selection. Degrades on large encoders unless the
                sensitive decoder blocks stay bf16, so applied only for families with a
                measured schedule (else falls back to fp8). ~2x smaller; cc >= 8.0.
  nvfp4       - torchao NVFP4 weight-only: 4-bit float, two-level microscaling, Blackwell
                sm_100+ FP4 cores. ~4x smaller (lowest VRAM) but a steeper quality cost.

All keep norms / embeddings full precision, are a memory-vs-quality tradeoff (off by default),
and pair well with streamed (group) offload where the text encoder stays resident. Quantify
the quality cost with scripts/diffusion_quality.py. torch / diffusers / torchao imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TE_QUANT_FP8 = "fp8"
TE_QUANT_NVFP4 = "nvfp4"
TE_QUANT_INT8 = "int8"
TE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
TE_QUANT_MODES = (TE_QUANT_FP8, TE_QUANT_NVFP4, TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC)

# Pipeline attributes that hold a text encoder, in order.
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")

# int8 degrades on large text encoders unless the quant-sensitive decoder blocks stay bf16.
# Per-family (skip_first, skip_last) blocks to keep dense, from measured hidden-state fidelity
# (per-token cosine vs bf16 at the consumed layer): keeping first blocks stops early-layer error
# seeding, last blocks protect the read layer. Families absent have no schedule clearing the bar,
# so int8 falls back to fp8.
#   qwen-image (Qwen2.5-VL-7B): first+last 6 -> ~0.997 cosine (both ends; outlier-bound).
#   flux.2-dev (Mistral-Small-24B): first 3 -> ~0.98 cosine (early-layer seeding).
_TE_INT8_SKIP: dict[str, tuple[int, int]] = {
    "qwen-image": (6, 6),
    "qwen-image-edit": (6, 6),
    "flux.2-dev": (3, 0),
}


def normalize_te_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested text-encoder quant; None / "" / "none" -> None.

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized == "none":
        return None
    if normalized not in TE_QUANT_MODES:
        raise ValueError(
            f"Unsupported text_encoder_quant '{value}'. Use one of: {', '.join(TE_QUANT_MODES)}."
        )
    return normalized


def te_quant_supported(target: Any, mode: str) -> bool:
    """Whether ``mode`` is usable for ``target``: a CUDA bf16 device plus the tensor-core class
    each backend needs -- fp8 dtype (fp8), fp8 GEMM sm_89+ (fp8_dynamic), int8 sm_80+ (int8),
    Blackwell sm_100+ (nvfp4)."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch

        if getattr(target, "dtype", None) is not torch.bfloat16:
            return False
        if mode == TE_QUANT_FP8:
            return hasattr(torch, "float8_e4m3fn")
        if mode == TE_QUANT_FP8_DYNAMIC:
            # fp8 GEMM needs Ada sm_89+ / Hopper / Blackwell.
            return hasattr(torch, "float8_e4m3fn") and torch.cuda.get_device_capability() >= (8, 9)
        if mode == TE_QUANT_INT8:
            return torch.cuda.get_device_capability()[0] >= 8  # int8 cores: Ampere sm_80+
        if mode == TE_QUANT_NVFP4:
            return torch.cuda.get_device_capability()[0] >= 10  # NVFP4 cores: Blackwell sm_100+
    except Exception:
        return False
    return False


def quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    offload_active: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """Quantise each present text encoder in place with ``mode``. Returns the mode applied, or
    None when disabled, unsupported, or nothing was cast. ``int8`` needs a per-family schedule
    (``_TE_INT8_SKIP``); without one it falls back to ``fp8``. Under ``offload_active`` the torchao
    modes are skipped (their subclasses reject ``Module.to()``); layerwise ``fp8`` still engages.
    Best-effort: any failure leaves the encoder dense."""
    mode = normalize_te_quant(mode)
    if mode is None:
        return None
    skip: Optional[tuple[int, int]] = None
    if mode == TE_QUANT_INT8:
        skip = _TE_INT8_SKIP.get((family or "").lower())
        if skip is None:
            _note(logger, f"int8 has no keep-bf16 schedule for family '{family}'; using fp8")
            mode = TE_QUANT_FP8
    # torchao modes produce subclasses that reject Module.to(), which an offload placement uses
    # (the DiT path skips torchao under offload for the same reason). Layerwise fp8 streams fine.
    if offload_active and mode in (TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC, TE_QUANT_NVFP4):
        _note(
            logger,
            f"text-encoder '{mode}' skipped under offload (torchao tensors reject Module.to()); "
            "pin a resident memory mode or use fp8",
        )
        return None
    if not te_quant_supported(target, mode):
        return None
    if mode == TE_QUANT_INT8:
        first, last = skip  # type: ignore[misc]

        def caster(enc: Any, tgt: Any) -> None:
            _cast_int8_selective(enc, tgt, first, last)
    elif mode == TE_QUANT_FP8_DYNAMIC:
        caster = _cast_fp8_dynamic
    elif mode == TE_QUANT_NVFP4:
        caster = _cast_nvfp4
    else:
        caster = _cast_fp8
    cast: list[str] = []
    for attr in _TEXT_ENCODER_ATTRS:
        encoder = getattr(pipe, attr, None)
        if encoder is None:
            continue
        try:
            caster(encoder, target)
            cast.append(attr)
        except Exception as exc:  # noqa: BLE001 — leave this encoder dense
            _warn(logger, f"{mode}:{attr}", exc)
    return mode if cast else None


def _te_exclude_tokens(encoder: Any) -> tuple[str, ...]:
    """fqn tokens whose Linears stay bf16 in a torchao TE quant: the VLM vision tower, the unused
    lm_head, and the encoder's own fp32-kept modules (T5 ``wo``, which explodes in low precision)."""
    tokens = ["visual", "vision_tower", "lm_head"]
    tokens += [str(m).lower() for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ())]
    return tuple(dict.fromkeys(tokens))


def _keep_bf16_block_fqns(encoder: Any, skip_first: int, skip_last: int) -> set[str]:
    """FQNs of decoder blocks to keep bf16: the first ``skip_first`` and last ``skip_last`` of
    each top-level ``nn.ModuleList`` stack. Structural, so no per-architecture table."""
    import torch

    keep: set[str] = set()
    for name, module in encoder.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        n = len(module)
        if n <= skip_first + skip_last:
            continue
        for i in list(range(skip_first)) + list(range(n - skip_last, n)):
            keep.add(f"{name}.{i}" if name else str(i))
    return keep


def _cast_int8_selective(encoder: Any, target: Any, skip_first: int, skip_last: int) -> None:
    # torchao dynamic int8 on the FLOP-heavy Linears, keeping the first/last decoder blocks (and
    # vision tower / lm_head / T5 wo) bf16. Reuses the transformer-quant factory so config never drifts.
    from torchao.quantization import quantize_
    from .diffusion_transformer_quant import (
        TQ_INT8,
        DEFAULT_MIN_LINEAR_FEATURES,
        _make_quant_config,
        make_filter_fn,
        exclude_tokens_for_scheme,
    )

    base = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES,
        exclude_tokens_for_scheme(TQ_INT8) + _te_exclude_tokens(encoder),
    )
    keep = _keep_bf16_block_fqns(encoder, skip_first, skip_last)

    def filter_fn(module: Any, fqn: str = "") -> bool:
        if not base(module, fqn):
            return False
        return not any(fqn == k or fqn.startswith(k + ".") for k in keep)

    quantize_(encoder, _make_quant_config(TQ_INT8), filter_fn = filter_fn)


def _weight_has_zero_output_row(module: Any) -> bool:
    """True when a Linear's weight has an all-zero OUTPUT row. torchao per-row fp8 derives a
    per-channel scale from that row's amax, so a dead row gives scale 0 -> 0/0 = NaN through the
    forward. Real checkpoints ship such rows: SDXL's text_encoder_2 (OpenCLIP ViT-bigG) has one in
    ``text_model.encoder.layers.2.self_attn.out_proj`` -- B200: every fp8_dynamic SDXL render came
    out black until this Linear is left dense. Cheap (one amax per Linear); False on any error."""
    try:
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return False
        return bool((weight.abs().amax(dim = -1) == 0).any().item())
    except Exception:  # noqa: BLE001 -- unreadable weight: let quantize_ decide
        return False


def _cast_fp8_dynamic(encoder: Any, target: Any) -> None:
    # torchao dynamic fp8 COMPUTE, per-row (torch._scaled_mm on the fp8 cores). Unlike layerwise
    # `fp8` this keeps the matmul in fp8 instead of upcasting. Robust across encoder sizes, so no
    # per-layer keep-bf16; only the vision tower / lm_head / T5 wo are excluded.
    from torchao.quantization import quantize_
    from .diffusion_transformer_quant import (
        TQ_FP8,
        DEFAULT_MIN_LINEAR_FEATURES,
        _make_quant_config,
        make_filter_fn,
    )

    # require_bf16: scaled_mm asserts a bf16 weight, so skip any stray non-bf16 Linear rather than
    # aborting the pass (belt-and-suspenders over the named T5 wo exclusion).
    base = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder), require_bf16 = True
    )

    # An all-zero output row NaNs under per-row scaling (scale 0 -> 0/0); keep those dense.
    def filter_fn(module: Any, fqn: str = "") -> bool:
        return base(module, fqn) and not _weight_has_zero_output_row(module)

    quantize_(encoder, _make_quant_config(TQ_FP8), filter_fn = filter_fn)


def _cast_fp8(encoder: Any, target: Any) -> None:
    import re
    import torch
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    # Idempotent: a pre-cast encoder (diffusion_te_prequant) arrives with the layerwise hooks
    # already installed, and re-registering the same hook name raises -- which would make
    # quantize_text_encoders report the (actually engaged) cast as failed. Keyed on the explicit
    # completion marker this function sets, NOT on hook presence alone: leftover hooks from a
    # cast that failed mid-pass must still fail closed, not read as "already cast".
    if getattr(encoder, "_unsloth_te_cast_complete", False) and _has_layerwise_hooks(encoder):
        return

    # Layerwise casting stores each leaf's weights in fp8 and upcasts per forward. Two things on a
    # transformers encoder push an fp8 weight/activation into an op that can't handle it, both
    # crashing only at generation, so skip the offending modules:
    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN)

    # (1) dtype-sensitive modules the encoder flags. T5 keeps "wo" in fp32: its gated FF reads
    # self.wo.weight.dtype and casts activations to match BEFORE calling wo (transformers#20287),
    # racing the upcast hook so F.linear sees fp8 input vs bf16 weight. Literal substrings.
    skip += tuple(re.escape(m) for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ()))

    # (2) an output projection tied to the input embedding. A CausalLM encoder (FLUX.2's Qwen3)
    # ties lm_head.weight to embed_tokens.weight; casting lm_head (an nn.Linear) to fp8 drags the
    # shared embedding down, which then emits fp8 activations that crash the first RMSNorm. Skip
    # the tied projection so the shared tensor stays dense (lm_head is unused for prompt encoding).
    get_out, get_in = (
        getattr(encoder, "get_output_embeddings", None),
        getattr(encoder, "get_input_embeddings", None),
    )
    out_emb = get_out() if callable(get_out) else None
    in_emb = get_in() if callable(get_in) else None
    if out_emb is not None and in_emb is not None and out_emb.weight is in_emb.weight:
        tied_name = next((n for n, m in encoder.named_modules() if m is out_emb), None)
        if tied_name:
            skip += (rf"^{re.escape(tied_name)}$",)

    apply_layerwise_casting(
        encoder,
        storage_dtype = torch.float8_e4m3fn,
        compute_dtype = target.dtype,
        skip_modules_pattern = skip,
        # Keep token-embedding tables full precision: the diffusers default only skips vision
        # pos/patch embeds, and fp8'ing nn.Embedding quantizes every prompt token to the coarse
        # fp8 grid, hurting fidelity.
        skip_modules_classes = (torch.nn.Embedding,),
    )

    # Module.dtype reports the first floating parameter, which is now fp8 STORAGE; pipelines
    # derive tensor dtypes from encoder.dtype (Flux2 casts prompt embeds to it and feeds the
    # result to randn_tensor, which has no fp8 kernel; VLM pipelines cast pixel_values to it,
    # racing the upcast hooks). The encoder computes in target.dtype, so report that -- via a
    # property shadowed on the ORIGINAL class reading a per-instance override. Swapping
    # __class__ to a dynamic subclass instead breaks transformers' kwargs-based output
    # recording (Qwen3VLModel returned hidden_states=None and krea-2 crashed at encode).
    compute_dtype = getattr(target, "dtype", None)
    try:
        if compute_dtype is not None:
            _install_dtype_override(type(encoder))
            encoder._unsloth_te_compute_dtype = compute_dtype
        # Marks the cast COMPLETE (hooks fully installed), enabling the idempotent early return
        # above. Best-effort like the dtype override: a non-Module double without settable
        # attributes still counts as cast, it just re-casts on a repeat call.
        encoder._unsloth_te_cast_complete = True
    except Exception:  # noqa: BLE001 — real HF encoders are heap-type nn.Modules; only doubles fail
        pass


def _install_dtype_override(cls: type) -> None:
    """Shadow ``cls.dtype`` with a property preferring the per-instance compute-dtype
    override ``_cast_fp8`` sets; instances without it keep the original behaviour. Class
    identity is untouched, applied once per class."""
    existing = cls.__dict__.get("dtype")
    if getattr(getattr(existing, "fget", None), "_unsloth_te_dtype_override", False):
        return
    # The property object itself when accessed through the class (property.__get__(None, cls)).
    original_fget = getattr(getattr(cls, "dtype", None), "fget", None)

    def _dtype(self):
        override = self.__dict__.get("_unsloth_te_compute_dtype")
        if override is not None:
            return override
        if original_fget is not None:
            return original_fget(self)
        raise AttributeError("dtype")

    _dtype._unsloth_te_dtype_override = True
    cls.dtype = property(_dtype)


def _has_layerwise_hooks(encoder: Any) -> bool:
    """True when any submodule already carries the diffusers layerwise-casting hook."""
    modules = getattr(encoder, "modules", None)
    if not callable(modules):
        return False
    for module in modules():
        registry = getattr(module, "_diffusers_hook", None)
        get_hook = getattr(registry, "get_hook", None)
        if callable(get_hook) and get_hook("layerwise_casting") is not None:
            return True
    return False


def _cast_nvfp4(encoder: Any, target: Any) -> None:
    # Weight-only NVFP4: linear weights become 4-bit NVFP4 on Blackwell FP4 cores; norms /
    # embeddings untouched. Exclude the VLM vision tower / lm_head / T5 wo and sub-512 projections
    # like the int8/fp8 TE modes (4-bit-ing a VLM image tower degrades the edit conditioning);
    # require_bf16 skips non-bf16 Linears so the cast engages instead of aborting.
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES, make_filter_fn

    filter_fn = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder), require_bf16 = True
    )
    quantize_(encoder, NVFP4WeightOnlyConfig(), filter_fn = filter_fn)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: text-encoder quant (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.precision: %s", msg)
