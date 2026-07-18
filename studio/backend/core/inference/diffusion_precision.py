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

All keep normalisations / embeddings full precision and are a memory-vs-quality tradeoff.
``auto`` (the loader default) walks ``select_te_quant_scheme``'s per-GPU ladder for the best
accurate scheme (fp8_dynamic / int8 / layerwise fp8), falling back to dense; ``none``/``off``
stays dense, an explicit scheme forces it. They pair well with streamed (group) offload, where
the resident text encoder dominates the companion footprint. Quantify the quality cost per model
with scripts/diffusion_quality.py and scripts/diffusion_quant_builder.py. torch / diffusers /
torchao imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TE_QUANT_FP8 = "fp8"
TE_QUANT_NVFP4 = "nvfp4"
TE_QUANT_INT8 = "int8"
TE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
TE_QUANT_AUTO = "auto"
# Concrete schemes (excludes "auto") the casters dispatch on.
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
    """Lower/strip a requested text-encoder quant; None / "" / "none" / "off" -> None, "auto" ->
    "auto" (resolved later by select_te_quant_scheme). Raises ValueError for an unsupported value."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized == TE_QUANT_AUTO:
        return TE_QUANT_AUTO
    if normalized not in TE_QUANT_MODES:
        raise ValueError(
            f"Unsupported text_encoder_quant '{value}'. Use one of: "
            f"{', '.join((TE_QUANT_AUTO,) + TE_QUANT_MODES)}, none/off."
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


# Per-arch preference order for text-encoder ``auto`` (best-first), mirroring the transformer's
# ``_AUTO_LADDER``. fp8_dynamic (compute fp8) leads on fp8-GEMM silicon; int8 sits second but only
# engages for a family with a measured keep-bf16 schedule; layerwise ``fp8`` (storage cast) is the
# universal fallback and the only scheme that survives group offload. nvfp4 stays explicit-only.
_TE_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    ((8, 9), (TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8)),  # Ada sm_89 / Hopper / Blackwell
    (
        (8, 0),
        (TE_QUANT_INT8, TE_QUANT_FP8),
    ),  # Ampere sm_80/86: no fp8 GEMM -> int8 or layerwise fp8
)

# Text encoders whose activation ranges break a scheme at the MODEL level (measured hidden-state
# cosine vs bf16). A denied scheme is skipped by ``auto`` and refused when requested explicitly.
# int8 already gates on a per-family keep-bf16 schedule (``_TE_INT8_SKIP``), so this covers the
# rarer case where a scheme breaks the encoder outright.
#   ltx-2 / fp8_dynamic: compute-fp8 on the Gemma3-27B encoder BLACK-FRAMES the whole clip (B200,
#   pairwise vs the dense encoder: mean luma 137.9 -> 0.0, LPIPS 0.78). Layerwise fp8 on the same
#   encoder is near-lossless (LPIPS 0.0043) at the same ~2x shrink, so auto falls through to it.
_TE_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "ltx-2": frozenset({TE_QUANT_FP8_DYNAMIC}),
}

# Families whose AUTO text-encoder quant resolves dense: measured out-of-bar trajectory drift for
# zero speed win. Unlike the deny table this only steers the AUTO default; an explicit scheme is
# still honored.
#   wan2.2-t2v-a14b: TE fp8_dynamic ALONE moves the clip to pairwise LPIPS 0.1195 vs the dense-TE
#   stack (B200, 1280x720/33f/50 steps) for 146.7 -> 142.7 s e2e -- the UMT5 encoder runs once per
#   generation, so the 1.03x is noise next to being the dominant accuracy cost. The MoE trajectory
#   amplifies the perturbation ~3x harder than on wan2.2-ti2v-5b (0.0396, kept quantized there for
#   a real 1.09x on its faster DiT).
_TE_AUTO_DENSE_FAMILIES: frozenset[str] = frozenset(
    # wan2.2-i2v-a14b inherits the T2V entry: same UMT5 encoder and MoE trajectory.
    {"hunyuanvideo-1.5", "hunyuanvideo-1.5-720p", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b"}
)

# Map a TE torchao scheme to the transformer smoke-probe scheme (same GEMM), so ``auto`` degrades
# gracefully when a build lacks a kernel. Layerwise fp8 has no torchao GEMM to probe.
_TE_SMOKE_SCHEME = {TE_QUANT_FP8_DYNAMIC: "fp8", TE_QUANT_INT8: "int8", TE_QUANT_NVFP4: "nvfp4"}


def _te_family_denied(family: Optional[str], scheme: str) -> bool:
    return scheme in _TE_FAMILY_SCHEME_DENY.get((family or "").strip().lower(), frozenset())


# nvfp4 TE casts WEIGHT-ONLY (see _cast_nvfp4), a different kernel from the transformer's
# dynamic-activation NVFP4 GEMM, so it gets its own cached smoke probe.
_TE_NVFP4_PROBE_CACHE: dict[str, bool] = {}


def _te_nvfp4_weightonly_probe(device: str) -> bool:
    """True iff weight-only NVFP4 (the ``_cast_nvfp4`` config) runs one forward on this build. The
    transformer ``_smoke_probe`` tests the DYNAMIC-activation NVFP4 GEMM, a different kernel: a
    Blackwell build can carry the weight-only path without the dynamic one, so an explicit TE
    ``nvfp4`` needs this dedicated probe. Cached per device."""
    if device in _TE_NVFP4_PROBE_CACHE:
        return _TE_NVFP4_PROBE_CACHE[device]
    ok = False
    try:
        import torch
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        from torchao.quantization import quantize_

        from .diffusion_transformer_quant import make_filter_fn

        lin = torch.nn.Linear(512, 512, bias = False).to(device = device, dtype = torch.bfloat16)
        quantize_(lin, NVFP4WeightOnlyConfig(), filter_fn = make_filter_fn(0))
        x = torch.randn(32, 512, device = device, dtype = torch.bfloat16)
        with torch.no_grad():
            lin(x)
        torch.cuda.synchronize()
        ok = True
    except Exception:  # noqa: BLE001 -- an unavailable kernel just means stay dense
        ok = False
    _TE_NVFP4_PROBE_CACHE[device] = ok
    return ok


def _te_scheme_probe(scheme: str, device: str) -> bool:
    """True iff ``scheme`` runs on this build. Layerwise fp8 (no torchao GEMM) always passes; nvfp4
    probes its weight-only kernel; the other torchao modes reuse the transformer's cached smoke
    test."""
    tq = _TE_SMOKE_SCHEME.get(scheme)
    if tq is None:
        return True
    # nvfp4 casts weight-only, so probe that kernel rather than the transformer's dynamic GEMM.
    if scheme == TE_QUANT_NVFP4:
        return _te_nvfp4_weightonly_probe(device)
    try:
        from .diffusion_transformer_quant import _smoke_probe
        return _smoke_probe(tq, device)
    except Exception:
        return False


def select_te_quant_scheme(
    target: Any,
    requested: Optional[str],
    *,
    family: Optional[str] = None,
    offload_active: bool = False,
) -> Optional[str]:
    """Resolve the concrete text-encoder scheme to apply, or None to stay dense bf16.

    An explicit scheme is returned as-is. ``auto`` walks ``_TE_AUTO_LADDER`` for this GPU and
    returns the first scheme that survives offload (torchao needs pinned residency -> only fp8),
    has a keep-bf16 schedule if int8, is not family-denied, is hardware-supported, and passes a
    kernel smoke test. Returns None when nothing qualifies."""
    requested = normalize_te_quant(requested)
    if requested is None or requested != TE_QUANT_AUTO:
        return requested
    # AUTO resolves dense for these families regardless of hardware. TE quant perturbs the
    # CONDITIONING and a multi-step video trajectory amplifies that chaotically: on
    # HunyuanVideo-1.5-720p (B200, 720p/33f/30 steps) TE fp8_dynamic ALONE moves the clip to LPIPS
    # 0.236 vs the bit-exact reference while the rest of the stack sits at 0.052-0.053, for ZERO
    # speed win (35.48 vs 35.36 s e2e). The ~6.7 GB saved isn't worth being the dominant accuracy
    # cost. (VAE fp8 stays in auto: 0.053, decode-only, no trajectory to amplify.)
    if (family or "").strip().lower() in _TE_AUTO_DENSE_FAMILIES:
        return None
    from .diffusion_transformer_quant import _capability, _is_consumer_gpu

    cap = _capability()
    if cap is None:
        return None
    device = str(getattr(target, "device", "cuda"))
    for floor, schemes in _TE_AUTO_LADDER:
        if cap >= floor:
            # Consumer GDDR parts run int8 full-rate but halve fp8 FP32-accumulate: prefer int8.
            ordered = (
                (TE_QUANT_INT8,) + tuple(s for s in schemes if s != TE_QUANT_INT8)
                if TE_QUANT_INT8 in schemes
                and schemes[0] != TE_QUANT_INT8
                and _is_consumer_gpu(device)
                else schemes
            )
            for scheme in ordered:
                # torchao tensors reject Module.to(); only layerwise fp8 streams, so skip the
                # torchao modes under offload.
                if offload_active and scheme in (
                    TE_QUANT_INT8,
                    TE_QUANT_FP8_DYNAMIC,
                    TE_QUANT_NVFP4,
                ):
                    continue
                # int8 only clears the bar on a family with a measured keep-bf16 schedule.
                if scheme == TE_QUANT_INT8 and (family or "").lower() not in _TE_INT8_SKIP:
                    continue
                if _te_family_denied(family, scheme):
                    continue
                if not te_quant_supported(target, scheme):
                    continue
                if not _te_scheme_probe(scheme, device):
                    continue
                return scheme
            return None
    return None


def quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    offload_active: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """Quantise each present text encoder in place with ``mode`` (auto / fp8 / fp8_dynamic / int8 /
    nvfp4). Returns the applied mode, or None when disabled, unsupported, or nothing was cast.
    ``auto`` resolves via ``select_te_quant_scheme``. ``int8`` needs a per-family keep-bf16 schedule
    (``_TE_INT8_SKIP``); a family without one falls back to ``fp8``. Under offload the torchao modes
    are skipped (their tensors reject Module.to()); layerwise ``fp8`` still engages. Best-effort:
    any failure leaves the encoder dense."""
    mode = normalize_te_quant(mode)
    if mode is None:
        return None
    if mode == TE_QUANT_AUTO:
        mode = select_te_quant_scheme(
            target, TE_QUANT_AUTO, family = family, offload_active = offload_active
        )
        if mode is None:
            return None
    skip: Optional[tuple[int, int]] = None
    if mode == TE_QUANT_INT8:
        skip = _TE_INT8_SKIP.get((family or "").lower())
        if skip is None:
            _note(logger, f"int8 has no keep-bf16 schedule for family '{family}'; using fp8")
            mode = TE_QUANT_FP8
    # A denied scheme is refused even when requested explicitly. Gate the FINAL concrete mode so
    # an int8 -> fp8 fallback is re-checked too (auto already filtered in select_te_quant_scheme).
    if _te_family_denied(family, mode):
        _note(
            logger,
            f"text-encoder '{mode}' denied for family '{family}' (out-of-bar; staying dense)",
        )
        return None
    # The torchao modes (int8, fp8_dynamic, nvfp4) produce tensors that reject Module.to(), which
    # an offload placement crashes on; layerwise fp8 streams fine. Skip the torchao modes here.
    if offload_active and mode in (TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC, TE_QUANT_NVFP4):
        _note(
            logger,
            f"text-encoder '{mode}' skipped under offload (torchao tensors reject Module.to()); "
            "pin a resident memory mode or use fp8",
        )
        return None
    if not te_quant_supported(target, mode):
        return None
    # An EXPLICIT torchao mode can clear the capability gate yet fail the real GEMM on a build
    # where quantize_ wraps the encoder but the kernel is broken (the caster's try/except catches
    # the cast, not the first forward). Run the auto ladder's kernel smoke test so a failing kernel
    # falls back to dense here instead of crashing at generation. No-op for layerwise fp8.
    device = str(getattr(target, "device", "cuda"))
    if not _te_scheme_probe(mode, device):
        _note(logger, f"text-encoder '{mode}' failed the kernel smoke test; staying dense")
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
            # A mid-pass caster failure may have left the encoder PARTIALLY quantized (can't
            # run as dense), so fail the load for that; a clean miss stays best-effort dense.
            # raise_if_partially_quantized only recognises torchao parameter subclasses, so it
            # cannot see a partial layerwise fp8 mutation (diffusers apply_layerwise_casting installs
            # upcast hooks + fp8 storage in place, leaving no torchao params). Detect a leftover
            # layerwise hook directly and fail closed there too; a clean failure stays dense.
            from .diffusion_transformer_quant import raise_if_partially_quantized

            if mode == TE_QUANT_FP8 and _has_layerwise_casting(encoder):
                raise RuntimeError(
                    f"text_encoder_quant fp8:{attr} failed after partially installing layerwise "
                    "casting (leftover fp8 hooks); reload the model instead of a dense fallback "
                    f"(original error: {exc})"
                ) from exc
            raise_if_partially_quantized(encoder, what = f"text_encoder_quant {mode}:{attr}", exc = exc)
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
    # racing the upcast hooks). The encoder computes in target.dtype, so report that.
    compute_dtype = getattr(target, "dtype", None)
    try:
        if compute_dtype is not None and not getattr(encoder, "_unsloth_te_dtype_override", False):
            cls = type(encoder)
            encoder.__class__ = type(
                cls.__name__,
                (cls,),
                {
                    "dtype": property(lambda self, _d = compute_dtype: _d),
                    "_unsloth_te_dtype_override": True,
                },
            )
        # Marks the cast COMPLETE (hooks fully installed), enabling the idempotent early return
        # above. Best-effort like the dtype override: a non-Module double without settable
        # attributes still counts as cast, it just re-casts on a repeat call.
        encoder._unsloth_te_cast_complete = True
    except Exception:  # noqa: BLE001 — real HF encoders are heap-type nn.Modules; only doubles fail
        pass


def _has_layerwise_hooks(encoder: Any) -> bool:
    """True when any submodule already carries the diffusers layerwise-casting hook."""
    return _has_layerwise_casting(encoder)


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


def _has_layerwise_casting(module: Any) -> bool:
    """True when any submodule still carries a diffusers layerwise-casting hook -- i.e. an
    ``apply_layerwise_casting`` pass installed an fp8-storage upcast hook before failing. torchao's
    partial-quant detector cannot see these, so a mid-pass layerwise failure would otherwise report
    a dense fallback over a half-cast encoder. Best-effort: a module without ``.modules()`` or a
    moved diffusers internal returns False (defer to the torchao check)."""
    try:
        hook_name = "layerwise_casting"
        try:
            from diffusers.hooks.layerwise_casting import _LAYERWISE_CASTING_HOOK
            hook_name = _LAYERWISE_CASTING_HOOK
        except Exception:  # noqa: BLE001 -- const moved: fall back to the stable literal
            pass
        for sub in module.modules():
            registry = getattr(sub, "_diffusers_hook", None)
            get_hook = getattr(registry, "get_hook", None)
            if callable(get_hook) and get_hook(hook_name) is not None:
                return True
    except Exception:  # noqa: BLE001 -- unqueryable module: defer to the torchao check
        return False
    return False


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: text-encoder quant (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.precision: %s", msg)
