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

All keep normalisations / embeddings full precision and are a memory-vs-quality tradeoff.
``auto`` (the loader default) walks ``select_te_quant_scheme``'s per-GPU ladder and picks the
best accurate scheme (fp8_dynamic / int8 / layerwise fp8), falling back to dense when nothing
qualifies; ``none``/``off`` keeps the encoder dense, and an explicit scheme forces it. They pair
especially well with streamed (group) offload, where the text encoder stays resident -- this is
where the companion footprint dominates. Quantify the quality cost per model with the quality
harness (scripts/diffusion_quality.py) and the hidden-state gate (scripts/diffusion_quant_builder.py).
torch / diffusers / torchao are imported lazily so the module stays importable in a no-torch runtime.
"""

from __future__ import annotations

from typing import Any, Optional

TE_QUANT_FP8 = "fp8"
TE_QUANT_NVFP4 = "nvfp4"
TE_QUANT_INT8 = "int8"
TE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
TE_QUANT_AUTO = "auto"
# Concrete schemes (excludes "auto"): "auto" resolves to one of these via select_te_quant_scheme,
# and these are the values the casters dispatch on.
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
    """Lower/strip a requested text-encoder quant; None / "" / "none" / "off" -> None,
    "auto" -> "auto" (resolved later by select_te_quant_scheme).

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
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


# Per-arch preference order for text-encoder ``auto`` (best-first), mirroring the transformer's
# ``_AUTO_LADDER``. fp8_dynamic (compute fp8 on the tensor cores) leads on fp8-GEMM silicon; int8
# (torch._int_mm) sits second but only engages for a family with a measured keep-bf16 schedule;
# layerwise ``fp8`` (storage cast, no torchao) is the universal fallback -- it needs only the fp8
# dtype and is the sole scheme that survives group offload. nvfp4 stays explicit-only (weight-only
# 4-bit is a steeper quality cost), never an auto pick, consistent with the DiT ladder.
_TE_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    ((8, 9), (TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8)),  # Ada sm_89 / Hopper / Blackwell
    (
        (8, 0),
        (TE_QUANT_INT8, TE_QUANT_FP8),
    ),  # Ampere sm_80/86: no fp8 GEMM -> int8 or layerwise fp8
)

# Text encoders whose activation ranges break a scheme at the MODEL level (measured hidden-state
# cosine vs bf16, via scripts/diffusion_quant_builder.py). Populated from the accuracy sweep; a
# denied scheme is skipped by ``auto`` and refused when requested explicitly. int8 already gates
# on a per-family keep-bf16 schedule (``_TE_INT8_SKIP``), so this is for the rarer case where a
# scheme breaks the encoder outright.
#   ltx-2 / fp8_dynamic: torchao per-row compute-fp8 on the Gemma3-27B encoder BLACK-FRAMES the
#   whole clip (B200, measured pairwise vs the dense encoder at identical seed/settings: mean
#   luma 137.9 -> 0.0, LPIPS 0.78; reproduced at 1216x704/33f/40 steps compiled and 384x256/9f/10
#   steps eager). Layerwise fp8 on the same encoder is near-lossless (pairwise LPIPS 0.0043) at
#   the same ~2x shrink, so auto falls through to it -- the deny costs nothing.
_TE_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "ltx-2": frozenset({TE_QUANT_FP8_DYNAMIC}),
}

# Families whose AUTO text-encoder quant resolves dense (see select_te_quant_scheme):
# measured out-of-bar trajectory drift for zero speed win on the video families below.
# Unlike the deny table this only steers the AUTO default; an explicit scheme request
# (text_encoder_quant="fp8_dynamic") is still honored verbatim.
#   wan2.2-t2v-a14b: TE fp8_dynamic ALONE moves the clip to pairwise LPIPS 0.1195 vs
#   the dense-TE stack (B200, 1280x720/33f/50 steps, identical seed) for 146.7 ->
#   142.7 s e2e -- the UMT5 encoder runs once per generation, so the 1.03x is noise
#   next to being the dominant accuracy cost. The dual-expert MoE trajectory amplifies
#   the conditioning perturbation ~3x harder than the same encoder on wan2.2-ti2v-5b
#   (0.0396 pairwise, kept quantized there for a real 1.09x on its much faster DiT).
_TE_AUTO_DENSE_FAMILIES: frozenset[str] = frozenset(
    {"hunyuanvideo-1.5", "hunyuanvideo-1.5-720p", "wan2.2-t2v-a14b"}
)

# Map a TE torchao scheme to the transformer smoke-probe scheme (same torchao GEMM), so ``auto``
# degrades gracefully when a build lacks a kernel. Layerwise fp8 has no torchao GEMM to probe.
_TE_SMOKE_SCHEME = {TE_QUANT_FP8_DYNAMIC: "fp8", TE_QUANT_INT8: "int8", TE_QUANT_NVFP4: "nvfp4"}


def _te_family_denied(family: Optional[str], scheme: str) -> bool:
    return scheme in _TE_FAMILY_SCHEME_DENY.get((family or "").strip().lower(), frozenset())


# nvfp4 TE casts WEIGHT-ONLY (see _cast_nvfp4), a different torchao kernel from the transformer's
# dynamic-activation NVFP4 GEMM, so it gets its own cached smoke probe.
_TE_NVFP4_PROBE_CACHE: dict[str, bool] = {}


def _te_nvfp4_weightonly_probe(device: str) -> bool:
    """True iff weight-only NVFP4 (the config ``_cast_nvfp4`` applies) runs one forward on this
    build. The transformer ``_smoke_probe`` tests the DYNAMIC-activation NVFP4 GEMM instead, a
    different kernel: a Blackwell build can carry the weight-only FP4 path without the dynamic
    one, so an explicit TE ``nvfp4`` request needs this dedicated probe to avoid falsely staying
    dense when the caster would in fact run. Cached per device."""
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
    """True iff ``scheme`` actually runs on this build. Layerwise fp8 (no torchao GEMM) always
    passes; nvfp4 probes its weight-only kernel (``_cast_nvfp4``); the other torchao TE modes
    reuse the transformer module's cached quantise+matmul smoke test (same dynamic config)."""
    tq = _TE_SMOKE_SCHEME.get(scheme)
    if tq is None:
        return True
    # nvfp4 TE casts weight-only, whereas the transformer smoke probe tests the dynamic-activation
    # NVFP4 GEMM: probe the kernel that will actually run so a weight-only-only build is not skipped.
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

    An explicit scheme is returned as-is (``quantize_text_encoders`` re-gates it). ``auto`` walks
    ``_TE_AUTO_LADDER`` for this GPU's capability and returns the first scheme that: survives the
    active offload policy (torchao modes need pinned residency -> only layerwise fp8 under offload),
    has a keep-bf16 schedule if int8, is not family-denied, is hardware-supported, and passes a
    real kernel smoke test. Returns None when nothing qualifies (e.g. no CUDA / pre-Ampere)."""
    requested = normalize_te_quant(requested)
    if requested is None or requested != TE_QUANT_AUTO:
        return requested
    # AUTO resolves dense for these families regardless of hardware. TE quant perturbs
    # the CONDITIONING and a multi-step video trajectory amplifies that chaotically: on
    # HunyuanVideo-1.5-720p (B200, 720p/33f/30 steps) TE fp8_dynamic ALONE moves the
    # clip to LPIPS 0.236 vs the bit-exact reference while the rest of the shipped
    # stack sits at 0.052-0.053, for ZERO speed win (35.48 vs 35.36 s e2e -- the TE
    # runs once per generation). The ~6.7 GB of weight savings is not worth being the
    # single dominant accuracy cost of the default stack. (VAE fp8 stays in auto:
    # measured 0.053, at the compile floor -- decode-only, no trajectory to amplify.)
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
                # torchao tensor subclasses reject the Module.to() an offload hook uses; only the
                # layerwise fp8 cast streams, so under offload skip the torchao modes.
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
    nvfp4). Returns the mode actually applied, or None when disabled, unsupported, or no encoder was
    cast. ``auto`` resolves to the best scheme for the GPU + family via ``select_te_quant_scheme``.
    ``int8`` needs a per-family keep-bf16 schedule (``_TE_INT8_SKIP``); a family without one falls
    back to ``fp8``. When ``offload_active`` the torchao modes are skipped (their tensor subclasses
    reject the ``Module.to()`` an offload hook uses); layerwise ``fp8`` still engages. Best-effort:
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
    # The deny table's contract (_TE_FAMILY_SCHEME_DENY) is that a denied scheme is refused even
    # when requested explicitly, mirroring the VAE module. Gate the FINAL concrete mode so an
    # int8 -> fp8 fallback is re-checked too (auto already filtered in select_te_quant_scheme).
    if _te_family_denied(family, mode):
        _note(
            logger,
            f"text-encoder '{mode}' denied for family '{family}' (out-of-bar; staying dense)",
        )
        return None
    # The torchao modes (int8 with a schedule, fp8_dynamic, nvfp4) produce tensor subclasses that
    # reject Module.to(); an offload placement moves the encoder that way and hard-crashes -- the
    # DiT path skips torchao quant under offload for exactly this reason. Layerwise fp8 is not
    # torchao and streams fine, so it still engages. Skip the torchao modes under offload.
    if offload_active and mode in (TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC, TE_QUANT_NVFP4):
        _note(
            logger,
            f"text-encoder '{mode}' skipped under offload (torchao tensors reject Module.to()); "
            "pin a resident memory mode or use fp8",
        )
        return None
    if not te_quant_supported(target, mode):
        return None
    # Mirror select_te_quant_scheme's auto path: an EXPLICIT torchao TE mode (int8 / fp8_dynamic /
    # nvfp4) can clear the capability gate above yet fail the real GEMM on a torchao/torch build
    # where quantize_ wraps the encoder but the kernel is broken -- and the caster's try/except only
    # catches the cast, not the first prompt-encoder forward. Run the same kernel smoke test the auto
    # ladder uses so a failing kernel falls back to dense here instead of crashing at generation.
    # Layerwise fp8 (no torchao GEMM) has no smoke scheme, so the probe is a no-op (True) for it.
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
            # The torchao casters mutate the encoder in place module-by-module, so a
            # mid-pass failure may have left it PARTIALLY quantized -- a state that
            # cannot run as the dense encoder this fallback would report (and that
            # offload's Module.to() hard-crashes on). Fail the load for that; a clean
            # miss (nothing swapped, e.g. layerwise fp8) stays best-effort dense.
            from .diffusion_transformer_quant import raise_if_partially_quantized

            raise_if_partially_quantized(encoder, what = f"text_encoder_quant {mode}:{attr}", exc = exc)
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

    quantize_(encoder, _make_quant_config(TQ_INT8), filter_fn = filter_fn)


def _weight_has_zero_output_row(module: Any) -> bool:
    """True when a Linear's weight contains an all-zero OUTPUT row. torchao's per-row
    fp8 scheme derives a per-output-channel scale from that row's amax, so a dead row
    yields scale 0 -> 0/0 = NaN through the whole forward. Real checkpoints ship such
    rows: SDXL's text_encoder_2 (OpenCLIP ViT-bigG) has one in
    ``text_model.encoder.layers.2.self_attn.out_proj`` -- measured on B200: every
    fp8_dynamic SDXL render came out black (NaN embeddings) until this Linear is left
    dense. Cheap (one amax per Linear, once per load); False on any error so the
    caster's own failure handling stays in charge."""
    try:
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return False
        return bool((weight.abs().amax(dim = -1) == 0).any().item())
    except Exception:  # noqa: BLE001 -- unreadable weight: let quantize_ decide
        return False


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

    # require_bf16: scaled_mm asserts a bf16 weight, so skip any stray non-bf16 Linear the encoder
    # keeps (belt-and-suspenders over the named T5 wo exclusion) rather than aborting the pass.
    base = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder), require_bf16 = True
    )

    # A Linear with an all-zero output row NaNs under per-row scaling (scale 0 -> 0/0);
    # keep exactly those Linears dense so one dead row cannot black out every render.
    def filter_fn(module: Any, fqn: str = "") -> bool:
        return base(module, fqn) and not _weight_has_zero_output_row(module)

    quantize_(encoder, _make_quant_config(TQ_FP8), filter_fn = filter_fn)


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
    # Exclude the VLM vision tower / lm_head / T5 wo and the sub-512 projections, exactly like
    # the int8 / fp8 torchao TE modes -- 4-bit-ing a VLM encoder's image tower (qwen-image /
    # qwen-image-edit's Qwen2.5-VL) degrades the image/edit conditioning the sibling schemes
    # deliberately protect, and require_bf16 skips any non-bf16 Linear the encoder keeps so the
    # NVFP4 (scaled_mm-family) cast engages on the bf16 linears instead of aborting the pass.
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
