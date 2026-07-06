# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision casting of the diffusion pipeline's text encoder(s).

The transformer arrives quantised in the GGUF, but the companion text encoder loads
dense (bf16) from the base repo and is often the largest resident component (a Qwen3
/ T5-XXL / Mistral encoder runs to many GB). This shrinks it in place, with four
backends:

  fp8         - diffusers layerwise casting: 8-bit (e4m3) storage, upcast per layer to
                the compute dtype. ~2x smaller. Works on any fp8-capable CUDA card (cc >= 8.9).
  fp8_dynamic - torchao dynamic fp8 COMPUTE (per-row): keeps the matmul in fp8 on the
                fp8 tensor cores (torch._scaled_mm) instead of upcasting each forward.
                ~2x smaller plus a tensor-core speedup; needs fp8-GEMM silicon (cc >= 8.9).
  int8        - torchao dynamic int8 COMPUTE (per-token act + per-channel weight ->
                torch._int_mm), with per-layer keep-bf16 selection. int8 degrades on large
                encoders unless the most quant-sensitive decoder blocks stay bf16, so it is
                applied only for families with a measured keep-bf16 schedule (else it falls
                back to fp8). ~2x smaller; needs int8 tensor cores (cc >= 8.0).
  nvfp4       - torchao NVFP4 weight-only: 4-bit float with two-level microscaling, run on
                Blackwell's (sm_100+) FP4 tensor cores. ~4x smaller and the lowest-VRAM
                option, but a steeper quality cost than fp8.

All keep normalisations / embeddings full precision and are a memory-vs-quality tradeoff,
not free, so all are off by default. They pair especially well with streamed (group)
offload, where the text encoder stays resident -- this is where the companion footprint
dominates. Quantify the quality cost per model with the quality harness
(scripts/diffusion_quality.py). torch / diffusers / torchao are imported lazily so the
module stays importable in a no-torch runtime.
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

# int8 (torch._int_mm) degrades on large text encoders unless the most quant-sensitive decoder
# blocks stay bf16. Per-family (skip_first, skip_last) decoder blocks to keep dense, from measured
# hidden-state fidelity (mean per-token cosine vs the bf16 reference, at the layer each pipeline
# consumes): keeping the first blocks stops early-layer error seeding, keeping the last blocks
# protects the read layer. Families absent here have no int8 schedule that clears the bar, so an
# int8 request for them falls back to fp8.
#   qwen-image (Qwen2.5-VL-7B): first+last 6 -> ~0.997 cosine (both ends needed; outlier-bound).
#   flux.2-dev (Mistral-Small-24B): first 3 -> ~0.98 cosine (pure early-layer seeding).
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
    """Whether ``mode`` is usable for ``target``: a CUDA device with a bf16 compute dtype, plus
    the tensor-core class each backend needs -- fp8 dtype (fp8 layerwise), fp8 GEMM sm_89+
    (fp8_dynamic), int8 tensor cores sm_80+ (int8), or Blackwell sm_100+ (nvfp4)."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch

        if getattr(target, "dtype", None) is not torch.bfloat16:
            return False
        if mode == TE_QUANT_FP8:
            return hasattr(torch, "float8_e4m3fn")
        if mode == TE_QUANT_FP8_DYNAMIC:
            # Compute fp8 (torch._scaled_mm) needs fp8-GEMM silicon: Ada sm_89+ / Hopper / Blackwell.
            return hasattr(torch, "float8_e4m3fn") and torch.cuda.get_device_capability() >= (8, 9)
        if mode == TE_QUANT_INT8:
            # int8 tensor cores (torch._int_mm) need Ampere sm_80+.
            return torch.cuda.get_device_capability()[0] >= 8
        if mode == TE_QUANT_NVFP4:
            # NVFP4 tensor cores need Blackwell (compute capability major >= 10).
            return torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False
    return False


def quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Quantise each present text encoder in place with ``mode`` (fp8 / fp8_dynamic / int8 / nvfp4).
    Returns the mode actually applied, or None when disabled, unsupported, or no encoder was cast.
    ``int8`` needs a per-family keep-bf16 schedule (``_TE_INT8_SKIP``); a family without one falls
    back to ``fp8``. Best-effort: any failure leaves the encoder dense."""
    mode = normalize_te_quant(mode)
    if mode is None:
        return None
    skip: Optional[tuple[int, int]] = None
    if mode == TE_QUANT_INT8:
        skip = _TE_INT8_SKIP.get((family or "").lower())
        if skip is None:
            _note(logger, f"int8 has no keep-bf16 schedule for family '{family}'; using fp8")
            mode = TE_QUANT_FP8
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
    """fqn tokens whose Linears stay bf16 in a torchao text-encoder quant: the VLM vision tower
    and the unused lm_head (not used for prompt encoding), plus the encoder's own fp32-kept
    modules (T5 ``wo``, which the gated feed-forward reads the dtype of and which explodes in
    low precision)."""
    tokens = ["visual", "vision_tower", "lm_head"]
    tokens += [str(m).lower() for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ())]
    return tuple(dict.fromkeys(tokens))


def _keep_bf16_block_fqns(encoder: Any, skip_first: int, skip_last: int) -> set[str]:
    """FQNs of the decoder blocks to keep bf16: the first ``skip_first`` and last ``skip_last`` of
    each top-level ``nn.ModuleList`` stack (a T5 ``encoder.block`` / a decoder ``...layers``).
    Structural, so it needs no per-architecture table."""
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
    # torchao dynamic int8 (per-token act + per-channel weight -> torch._int_mm) on the FLOP-heavy
    # Linears, but keeping the first/last decoder blocks (and the vision tower / lm_head / T5 wo)
    # in bf16. Reuses the committed transformer-quant factory so the config never drifts.
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

    quantize_(encoder, _make_quant_config(TQ_INT8), filter_fn=filter_fn)


def _cast_fp8_dynamic(encoder: Any, target: Any) -> None:
    # torchao dynamic fp8 COMPUTE, per-row (per-token activation + per-output-channel weight ->
    # torch._scaled_mm on the fp8 tensor cores). Unlike the layerwise `fp8` backend this keeps the
    # matmul in fp8 instead of upcasting each forward. fp8 is robust across encoder sizes, so no
    # per-layer keep-bf16 is needed; only the vision tower / lm_head / T5 wo are excluded.
    from torchao.quantization import quantize_
    from .diffusion_transformer_quant import (
        TQ_FP8,
        DEFAULT_MIN_LINEAR_FEATURES,
        _make_quant_config,
        make_filter_fn,
    )

    filter_fn = make_filter_fn(DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder))
    quantize_(encoder, _make_quant_config(TQ_FP8), filter_fn=filter_fn)


def _cast_fp8(encoder: Any, target: Any) -> None:
    import re
    import torch
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    # diffusers' layerwise casting stores each supported leaf module's weights in fp8 and
    # upcasts them per forward. Two things on a transformers text encoder can push an fp8
    # weight or activation into an op that can't handle it, and both crash only at
    # generation (the load-time guard can't see them), so skip the offending modules:
    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN)

    # (1) dtype-sensitive modules the encoder itself flags. T5 keeps "wo" in fp32: its
    # gated feed-forward reads self.wo.weight.dtype and casts the activations to match
    # BEFORE calling wo (transformers#20287), racing the forward-time upcast hook so
    # F.linear sees an fp8 input against a bf16 weight. Names are literal substrings.
    skip += tuple(re.escape(m) for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ()))

    # (2) an output projection tied to the input embedding. A CausalLM encoder (FLUX.2's
    # Qwen3) ties lm_head.weight to embed_tokens.weight; lm_head is an nn.Linear so it
    # gets cast to fp8 and, sharing one tensor, drags the embedding to fp8 with it. The
    # embedding then emits fp8 activations that crash the first RMSNorm. Skip the tied
    # projection so the shared tensor stays dense (lm_head is unused for prompt encoding).
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
        # Keep token-embedding tables (T5 "shared", Qwen "embed_tokens", etc.) full
        # precision: the diffusers default pattern only skips vision pos/patch
        # embeds, not nn.Embedding lookups, and fp8'ing those quantizes every prompt
        # token straight to the coarse fp8 grid, hurting prompt fidelity.
        skip_modules_classes = (torch.nn.Embedding,),
    )


def _cast_nvfp4(encoder: Any, target: Any) -> None:
    # Weight-only NVFP4: linear weights become 4-bit (packed) NVFP4 tensors and run
    # on Blackwell FP4 tensor cores; norms / embeddings (not nn.Linear) are untouched.
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    quantize_(encoder, NVFP4WeightOnlyConfig())


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: text-encoder quant (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.precision: %s", msg)
