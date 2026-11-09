# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SAM 3 pre-quantized FP8/INT8/ConvRot detector weights sharing diffusion quant code; quantized
kernels CUDA only (CPU/MPS dequantize a lone quantized file); other exclusions refused at load"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

# ── precision choice ───────────────────────────────────────────────────────────────────────────────────────────────
PRECISION_AUTO = "auto"
PRECISION_BF16 = "bf16"
PRECISION_FP8 = "fp8"
PRECISION_INT8 = "int8"
PRECISIONS = (PRECISION_AUTO, PRECISION_BF16, PRECISION_FP8, PRECISION_INT8)
QUANT_PRECISIONS = (PRECISION_FP8, PRECISION_INT8)
ENV_PRECISION = "UNSLOTH_SAM3_PRECISION"
# int8 loads the ConvRot artifact unless this is 0 (then the plain INT8 one).
ENV_INT8_CONVROT = "UNSLOTH_SAM3_INT8_CONVROT"

# What ``auto`` may pick on a CUDA card short of memory, best first (needs GATE_PASSED); empty: always bf16. With room
# auto runs bf16: at this size the 8-bit files save ~0.5 GB but are ~2x slower per Find (activation quant cost).
AUTO_ORDER: tuple[str, ...] = (PRECISION_INT8,)

# Minimum compute capability per scheme: fp8 scaled_mm needs Ada / Hopper (sm_89+), int8 _int_mm Ampere (sm_80+).
_MIN_CAPABILITY = {PRECISION_FP8: (8, 9), PRECISION_INT8: (8, 0)}

# ── what is quantized ──────────────────────────────────────────────────────────────────────────────────────────────
SEGMENTER_QUANT_FAMILY = "sam3"
# The diffusion runtime's DEFAULT_MIN_LINEAR_FEATURES. Every DETR encoder / decoder, geometry, mask-decoder and
# scoring Linear in SAM 3 is 256 wide, so this leaves all of them bf16; the ViT (1024 / 4736) and the CLIP text
# encoder (1024 / 4096) are what gets quantized.
SEGMENTER_MIN_FEATURES = 512
SEGMENTER_CONVROT_GROUP = 256


def _first_vision_blocks(count: int) -> tuple[str, ...]:
    return tuple(f"backbone.layers.{index}." for index in range(count))


def artifact_tag(scheme: str, rotated: bool = False) -> str:
    """ "fp8", "int8" or "int8-convrot": one per published file."""
    return scheme + ("-convrot" if rotated else "")


# Name tokens left in bf16 on top of the size rule, per file. The sensitivity scan (one group quantized at a time,
# 53 prompts) put the error in the first ViT blocks for every scheme (the text encoder, its projection and the late
# ViT blocks cost a fraction of it; the 256-wide DETR / mask / scoring Linears, already bf16 by the size rule, cost as
# much as a whole ViT half for 4% of the bytes). Each file keeps the smallest set of early blocks that brought it
# inside the gate tolerance (or, for fp8, closest to it).
SEGMENTER_EXCLUDE_TOKENS: dict[str, tuple[str, ...]] = {
    "fp8": _first_vision_blocks(8),
    "int8": _first_vision_blocks(4),
    "int8-convrot": _first_vision_blocks(8),
}
# Each file's verdict on the accuracy gate (unsloth/sam3.1 model card, Checks: the built file through the segment
# path, 53 prompts, every metric within 2x of what bf16 itself gives up against the fp32 weights, no kept-count,
# false-positive or absent-concept change). ``auto`` loads only a file that passed; an explicit precision loads any.
GATE_PASSED: dict[str, bool] = {"fp8": False, "int8": True, "int8-convrot": True}


def exclude_tokens(scheme: str, rotated: bool = False) -> tuple[str, ...]:
    return SEGMENTER_EXCLUDE_TOKENS.get(artifact_tag(scheme, rotated), ())


def auto_may_load(tag: str) -> bool:
    return bool(GATE_PASSED.get(tag))


def segmenter_filter_fn(
    scheme: str,
    *,
    exclude: Optional[tuple[str, ...]] = None,
    select: Any = None,
    min_features: int = SEGMENTER_MIN_FEATURES,
) -> Any:
    """The torchao ``quantize_`` filter for ``scheme``: the diffusion runtime's filter with SAM's exclusions.
    ``select(fqn)`` narrows it further (the sensitivity scan quantizes one group at a time)."""
    from .diffusion_transformer_quant import (
        _REQUIRE_BF16_SCHEMES,
        divisible_for_scheme,
        make_filter_fn,
    )

    base = make_filter_fn(
        min_features,
        exclude_name_tokens = exclude_tokens(scheme)
        if exclude is None
        else tuple(exclude),  # plain-file set
        require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
        require_divisible = divisible_for_scheme(scheme),
    )
    if select is None:
        return base
    return lambda module, fqn = "": bool(select(fqn)) and base(module, fqn)


def quantize_segmenter_(
    model: Any,
    scheme: str,
    *,
    convrot_group: int = 0,
    exclude: Optional[tuple[str, ...]] = None,
    select: Any = None,
    min_features: int = SEGMENTER_MIN_FEATURES,
) -> dict:
    """Quantize ``model`` (a dense bf16 Sam3Model) in place with the Studio runtime config; with ``convrot_group``,
    rotate every selected Linear whose input width the group divides first. Returns the metadata fragment to record
    (exclusions, filter facts, and the rotation contract when rotated). ``select`` / ``min_features`` exist for the
    sensitivity scan; artifacts use the defaults, which the loader checks."""
    from torchao.quantization import quantize_

    from .diffusion_transformer_quant import _REQUIRE_BF16_SCHEMES, _make_quant_config
    from .diffusion_transformer_quant import divisible_for_scheme

    if scheme not in QUANT_PRECISIONS:
        raise ValueError(f"unknown segmenter quant scheme {scheme!r}")
    if convrot_group and scheme != PRECISION_INT8:
        raise ValueError("ConvRot is an int8 rotation")
    if exclude is None:
        exclude = exclude_tokens(scheme, bool(convrot_group))
    filter_fn = segmenter_filter_fn(
        scheme, exclude = exclude, select = select, min_features = min_features
    )
    meta: dict = {
        "scheme": scheme,
        "min_features": int(min_features),
        "exclude_name_tokens": list(exclude),
        "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
        "require_divisible": divisible_for_scheme(scheme),
    }
    if convrot_group:
        from .diffusion_convrot import rotatable_fqns, rotate_linears_, rotation_metadata

        rotatable, plain = rotatable_fqns(model, filter_fn, int(convrot_group))
        if not rotatable:
            raise ValueError(f"ConvRot group {convrot_group} divides no quantized Linear")
        rotate_linears_(model, rotatable, int(convrot_group))
        meta.update(rotation_metadata(int(convrot_group), rotatable))
        meta["convrot_plain_linears"] = len(plain)
    quantize_(model, _make_quant_config(scheme), filter_fn = filter_fn)
    meta["quantized_linears"] = sum(
        1 for _n, m in model.named_modules() if _is_quantized_weight(getattr(m, "weight", None))
    )
    return meta


def _is_quantized_weight(weight: Any) -> bool:
    if weight is None:
        return False
    data = getattr(weight, "data", weight)
    import torch

    return type(data) not in (torch.Tensor, torch.nn.Parameter)


# ── artifacts ──────────────────────────────────────────────────────────────────────────────────────────────────────
def write_prequant_file(
    path: str,
    model: Any,
    quant_meta: dict,
    *,
    base_model_id: str,
    source_revision: Optional[str] = None,
) -> dict:
    """Save a quantized Sam3Model (``quantize_segmenter_``'s output and fragment) as a segmenter prequant artifact:
    the diffusion prequant safetensors container, v2 when a rotation is baked, with the canonical fingerprint.
    Returns the metadata written."""
    import safetensors
    import torch
    import torchao
    import transformers

    from .diffusion_prequant import canonical_weight_fingerprint, prequant_format_for
    from .diffusion_transformer_quant import FP8_GRANULARITY, _resolve_fast_accum
    from .prequant_safetensors import save_prequant_safetensors

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in model.state_dict().items()
    }
    scheme = quant_meta["scheme"]
    metadata = {
        "base_model_id": base_model_id,
        "source_revision": source_revision,
        "family": SEGMENTER_QUANT_FAMILY,
        "model_class": type(model).__name__,
        "component": "detector",
        "torch_dtype": "bfloat16",
        "quant_backend": "torchao",
        "fast_accum": _resolve_fast_accum(None) if scheme == PRECISION_FP8 else None,
        "torch_version": torch.__version__,
        "torchao_version": getattr(torchao, "__version__", "?"),
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        **quant_meta,
    }
    if scheme == PRECISION_FP8:
        metadata["fp8_granularity"] = FP8_GRANULARITY
    metadata["fingerprint"] = canonical_weight_fingerprint(state_dict)
    fmt = prequant_format_for(metadata)
    save_prequant_safetensors(str(path), fmt = fmt, state_dict = state_dict, metadata = metadata)
    return {"format": fmt, **metadata}


def artifact_filename(
    prefix: str,
    scheme: str,
    *,
    rotated: bool = False,
) -> str:
    """``<prefix>-FP8.safetensors`` / ``-INT8.safetensors`` / ``-INT8-ConvRot.safetensors`` (the diffusion prequant
    naming)."""
    tag = scheme.upper() + ("-ConvRot" if rotated else "")
    return f"{prefix}-{tag}.safetensors"


def int8_convrot_enabled() -> bool:
    return (os.environ.get(ENV_INT8_CONVROT) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def checkpoint_problem(
    fmt: Any,
    meta: Any,
    *,
    scheme: str,
    rotated: bool,
    base_model_id: Optional[str] = None,
    convrot_group: int = SEGMENTER_CONVROT_GROUP,
) -> Optional[str]:
    """Why a segmenter prequant header must not be loaded, or None. Torch free, header only. A rotated file must use
    ``convrot_group``: the accuracy gate and the exclusions were measured at that group, not at any power of four."""
    from .diffusion_convrot import ROTATION_GROUP_KEY, declares_rotation, rotation_metadata_error
    from .diffusion_prequant import PREQUANT_FORMAT, PREQUANT_FORMAT_ROTATED

    if not isinstance(meta, dict):
        return "no metadata"
    if meta.get("family") != SEGMENTER_QUANT_FAMILY:
        return f"family {meta.get('family')!r} is not {SEGMENTER_QUANT_FAMILY!r}"
    if meta.get("scheme") != scheme:
        return f"scheme {meta.get('scheme')!r}, expected {scheme!r}"
    if base_model_id and str(meta.get("base_model_id") or "").lower() != base_model_id.lower():
        return f"built from {meta.get('base_model_id')!r}, not {base_model_id!r}"
    if meta.get("min_features") != SEGMENTER_MIN_FEATURES:
        return f"min_features {meta.get('min_features')!r} != {SEGMENTER_MIN_FEATURES}"
    declared = declares_rotation(meta)
    if declared != (fmt == PREQUANT_FORMAT_ROTATED):
        return f"format {fmt!r} and the declared rotation ({declared}) disagree"
    if fmt not in (PREQUANT_FORMAT, PREQUANT_FORMAT_ROTATED):
        return f"unknown format {fmt!r}"
    if declared != rotated:
        return "rotation expected" if rotated else "unexpected rotation"
    if rotated and meta.get(ROTATION_GROUP_KEY) != convrot_group:
        return f"ConvRot group {meta.get(ROTATION_GROUP_KEY)!r}, expected {convrot_group}"
    if sorted(meta.get("exclude_name_tokens") or ()) != sorted(exclude_tokens(scheme, rotated)):
        return (
            f"excludes {meta.get('exclude_name_tokens')!r} differ from this build's "
            f"{list(exclude_tokens(scheme, rotated))!r}"
        )
    return rotation_metadata_error(meta)


@contextmanager
def _default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _build_skeleton(model_dir: str) -> Any:
    """Sam3Model with meta parameters (the file supplies every one) and real buffers. The buffers are built in
    float32, which is what ``from_pretrained(dtype = bfloat16)`` leaves them in: built under a bf16 default the
    global-attention RoPE tables of the ViT come out up to 0.04 off and every mask moves."""
    import torch
    from accelerate import init_empty_weights
    from transformers import Sam3Config, Sam3Model

    config = Sam3Config.from_pretrained(model_dir)
    with _default_dtype(torch.float32), init_empty_weights(include_buffers = False):
        return Sam3Model(config)


def _dequantize_state_dict(state_dict: dict, dtype: Any) -> dict:
    """Every quantized weight as a dense ``dtype`` tensor (rotated weights stay rotated)."""
    out = {}
    for key, value in state_dict.items():
        if _is_quantized_weight(value):
            value = value.dequantize().to(dtype)
        elif getattr(value, "is_floating_point", lambda: False)():
            value = value.to(dtype)
        out[key] = value
    return out


def load_prequantized_segmenter(
    model_dir: str,
    weights_path: str,
    *,
    scheme: str,
    rotated: bool,
    device: str,
    dtype: Any,
    base_model_id: Optional[str] = None,
    dequantize: bool = False,
    convrot_group: int = SEGMENTER_CONVROT_GROUP,
) -> tuple[Any, dict]:
    """``(Sam3Model, info)`` from a segmenter prequant artifact. On CUDA the weights stay quantized unless
    ``dequantize``; elsewhere they are dequantized to ``dtype``. Raises on anything wrong with the file (the caller
    falls back to bf16)."""
    from .diffusion_convrot import apply_activation_rotation
    from .diffusion_prequant import _pin_kernel_preference, _verify_packed_fingerprint
    from .prequant_safetensors import load_prequant_safetensors

    started = time.perf_counter()
    quantized = device.startswith("cuda") and not dequantize
    ckpt = load_prequant_safetensors(weights_path, device = device if quantized else "cpu")
    fmt, meta, state_dict = ckpt["format"], ckpt["metadata"], ckpt["state_dict"]
    problem = checkpoint_problem(
        fmt,
        meta,
        scheme = scheme,
        rotated = rotated,
        base_model_id = base_model_id,
        convrot_group = convrot_group,
    )
    if problem:
        raise ValueError(f"segmenter prequant refused: {problem}")
    if not _verify_packed_fingerprint(state_dict, meta, logger = logger, path = weights_path):
        raise ValueError("segmenter prequant fingerprint mismatch")
    if quantized:
        _pin_kernel_preference(state_dict)
    else:
        state_dict = _dequantize_state_dict(state_dict, dtype)
    model = _build_skeleton(model_dir)
    model.load_state_dict(state_dict, strict = True, assign = True)
    rotated_fqns = apply_activation_rotation(model, meta, logger = logger)
    model.to(device)
    model.eval()
    stray = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    stray += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if stray:
        raise ValueError(f"segmenter prequant left {len(stray)} tensors unset (e.g. {stray[0]})")
    info = {
        "precision": scheme if quantized else "bf16",
        "scheme": scheme,
        "rotated": bool(rotated_fqns),
        "rotated_linears": len(rotated_fqns),
        "quantized_linears": sum(
            1 for _n, m in model.named_modules() if _is_quantized_weight(getattr(m, "weight", None))
        ),
        "kernels": "quantized" if quantized else "dequantized",
        "file": os.path.basename(weights_path),
        "load_s": round(time.perf_counter() - started, 2),
    }
    logger.info(
        "images.segment: pre-quantized %s file engaged (%s weights, %d quantized linears, %d rotated, on %s): %s",
        scheme,
        info["kernels"],
        info["quantized_linears"],
        info["rotated_linears"],
        device,
        info["file"],
    )
    return model, info


# ── which precision runs ───────────────────────────────────────────────────────────────────────────────────────────
def requested_precision() -> str:
    """UNSLOTH_SAM3_PRECISION (auto / bf16 / fp8 / int8; "bfloat16" reads as bf16). Unknown values warn, read auto."""
    raw = (os.environ.get(ENV_PRECISION) or "").strip().lower()
    if raw in ("", PRECISION_AUTO):
        return PRECISION_AUTO
    if raw in ("bfloat16", "bf16", "dense", "none", "off"):
        return PRECISION_BF16
    if raw in QUANT_PRECISIONS:
        return raw
    logger.warning(
        "images.segment: unknown %s=%r (choices: %s), using auto",
        ENV_PRECISION,
        raw,
        ", ".join(PRECISIONS),
    )
    return PRECISION_AUTO


def _cuda_capability(device: str) -> Optional[tuple[int, int]]:
    """The card's compute capability; reads properties only, so no CUDA context is created."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        ordinal = int(device.split(":", 1)[1]) if ":" in device else None
        major, minor = torch.cuda.get_device_capability(ordinal)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


def unsupported_reason(
    precision: str,
    device: str,
    dtype: Any = None,
) -> Optional[str]:
    """Why ``precision`` cannot run quantized on ``device``, or None."""
    if not device.startswith("cuda"):
        return "quantized kernels run on CUDA only"
    from core._torchao_stub import is_stubbed, torch_is_rocm

    # ROCm reports gfx versions through the CUDA capability API and these dynamic INT8 / FP8 paths are NVIDIA-only
    # (diffusion_transformer_quant.dense_transformer_supported refuses ROCm for the same reason).
    if torch_is_rocm():
        return "quantized kernels run on NVIDIA CUDA only (ROCm build)"
    if is_stubbed("torchao"):
        return "torchao is a stub on this install"
    if dtype is not None:
        import torch
        if dtype != torch.bfloat16:
            return "the card has no bf16"
    capability = _cuda_capability(device)
    need = _MIN_CAPABILITY[precision]
    if capability is None or capability < need:
        return f"{precision} needs compute capability {need[0]}.{need[1]}+"
    from .diffusion_transformer_quant import torchao_unavailable_reason
    from .prequant_safetensors import safetensors_prequant_supported

    why = torchao_unavailable_reason()
    if why:
        return why
    if not safetensors_prequant_supported():
        return "this torchao cannot read safetensors pre-quant files"
    return None


def resolve_precision(
    device: str,
    dtype: Any = None,
    *,
    available: tuple[str, ...] = QUANT_PRECISIONS,
    memory_tight: bool = False,
) -> tuple[str, str]:
    """``(precision, why)`` for a run on ``device``. ``available`` is the quantized arms the variant ships (and, at
    load, has on disk). bf16 when asked for, when nothing quantized can run here, when auto has no gated arm, or for
    auto on CUDA unless ``memory_tight``. An explicit fp8 / int8 is honoured either way."""
    requested = requested_precision()
    if requested == PRECISION_BF16:
        return PRECISION_BF16, f"{ENV_PRECISION}=bf16"
    if requested == PRECISION_AUTO and not memory_tight and device.startswith("cuda"):
        return PRECISION_BF16, "auto: bf16 fits and is faster"
    candidates = AUTO_ORDER if requested == PRECISION_AUTO else (requested,)
    reasons = []
    for precision in candidates:
        if precision not in available:
            reasons.append(f"{precision}: not available")
            continue
        why = unsupported_reason(precision, device, dtype)
        if why is None:
            return (
                precision,
                "auto" if requested == PRECISION_AUTO else f"{ENV_PRECISION}={precision}",
            )
        reasons.append(f"{precision}: {why}")
    if requested == PRECISION_AUTO and not AUTO_ORDER:
        return PRECISION_BF16, "auto: no quantized arm passed the accuracy gate"
    return PRECISION_BF16, "; ".join(reasons) or "bf16"


__all__ = [
    "AUTO_ORDER",
    "ENV_INT8_CONVROT",
    "ENV_PRECISION",
    "PRECISIONS",
    "PRECISION_AUTO",
    "PRECISION_BF16",
    "PRECISION_FP8",
    "PRECISION_INT8",
    "SEGMENTER_CONVROT_GROUP",
    "SEGMENTER_EXCLUDE_TOKENS",
    "SEGMENTER_MIN_FEATURES",
    "GATE_PASSED",
    "artifact_filename",
    "artifact_tag",
    "auto_may_load",
    "write_prequant_file",
    "checkpoint_problem",
    "load_prequantized_segmenter",
    "quantize_segmenter_",
    "requested_precision",
    "resolve_precision",
    "segmenter_filter_fn",
]
