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
  nvfp4       - torchao NVFP4 weight-only: 4-bit float storage, two-level microscaling, each
                weight dequantised to bf16 per forward (no FP4 GEMM). ~3.5x smaller (lowest
                VRAM) but a steeper quality cost and a slower prompt encode; cc >= 8.0.

All keep norms / embeddings full precision, are a memory-vs-quality tradeoff (off by default),
and pair well with streamed (group) offload where the text encoder stays resident. Quantify
the quality cost with scripts/diffusion_quality.py. torch / diffusers / torchao imported lazily.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from .diffusion_auto_policy import (
    RESOLVED_APPLIED,
    RESOLVED_FELL_BACK,
    RESOLVED_UNSUPPORTED,
)

# stdlib-only module (no torch), so this stays inside the "imported lazily" promise above.
from functools import lru_cache

from core._torchao_stub import is_stubbed, torch_is_rocm

from .diffusion_nvfp4_flag import nvfp4_blocked, nvfp4_disabled_message
from .family_profile import profile_table as _profile_table

TE_QUANT_FP8 = "fp8"
TE_QUANT_NVFP4 = "nvfp4"
TE_QUANT_INT8 = "int8"
TE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
TE_QUANT_MODES = (TE_QUANT_FP8, TE_QUANT_NVFP4, TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC)
# The modes that go through torchao; plain fp8 is a layerwise torch cast and needs none.
_TE_TORCHAO_MODES = frozenset({TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC, TE_QUANT_NVFP4})

# Pipeline attributes that hold a text encoder, in order.
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")

# int8 degrades on large text encoders unless the quant-sensitive decoder blocks stay bf16. Per-family (skip_first,
# skip_last) blocks to keep dense (``FamilyProfile.te_int8_skip``, measured hidden-state fidelity); absent families
# have no schedule clearing the bar, so int8 falls back to fp8.
_TE_INT8_SKIP: dict[str, tuple[int, int]] = _profile_table("te_int8_skip")


def normalize_te_quant(value: Optional[str]) -> Optional[str]:
    """none, off and auto collapse to None; read the raw request first where they differ; raises ValueError for an unsupported value."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off", "auto"):
        return None
    if normalized not in TE_QUANT_MODES:
        raise ValueError(
            f"Unsupported text_encoder_quant '{value}'. Use one of: {', '.join(TE_QUANT_MODES)}."
        )
    if nvfp4_blocked(normalized):
        raise ValueError(nvfp4_disabled_message("text_encoder_quant"))
    return normalized


def te_quant_is_auto(value: Optional[str]) -> bool:
    """Unset, empty or auto means choose for me; none and off pin bf16 and need the raw value."""
    if value is None:
        return True
    normalized = str(value).strip().lower().replace("-", "_")
    return not normalized or normalized == "auto"


class AutoTeQuant(str):
    """A scheme ``resolve_te_quant_request`` picked from the family default, not one the caller typed. Equal to and
    hashed as the plain string, so every consumer that only needs the scheme is unchanged; the type lets the planners
    that run outside a load (download plan, memory budget) tell an automatic pick from an explicit one."""

    __slots__ = ()


def te_quant_picked_automatically(mode: Any) -> bool:
    """Whether ``mode`` is an automatic pick: an ``AutoTeQuant``, or any mode inside a load whose raw request was
    unset / "auto" (``diffusion_te_prequant.auto_te_request``)."""
    if isinstance(mode, AutoTeQuant):
        return True
    try:
        from .diffusion_te_prequant import te_request_is_auto
        return te_request_is_auto()
    except Exception:  # noqa: BLE001
        return False


def resolve_te_quant_request(
    value: Optional[str], auto_scheme: Optional[str]
) -> tuple[Optional[str], bool]:
    """Unset or auto takes the family's auto scheme, but a miss falls back to dense instead of refusing."""
    if not te_quant_is_auto(value):
        return normalize_te_quant(value), False
    if auto_scheme is None or nvfp4_blocked(auto_scheme):
        return None, False
    # Validate the family's own field rather than trusting it: a typo here would otherwise reach
    # quantize_text_encoders as an unknown mode on every default load of that family.
    scheme = normalize_te_quant(auto_scheme)
    return (AutoTeQuant(scheme) if scheme is not None else None), True


def effective_te_quant(mode: Optional[str], family: Optional[str]) -> Optional[str]:
    """An explicit int8 on a family with no keep-bf16 schedule becomes fp8 before support is checked."""
    normalized = normalize_te_quant(mode)
    if normalized == TE_QUANT_INT8 and (
        _TE_INT8_SKIP.get((family or "").lower()) is None or _family_hosts_te_int8(family)
    ):
        # A family hosting an int8 ConvRot file loads it as plain tensors (no torchao), with its fp8 file behind it.
        return TE_QUANT_FP8
    return normalized


def preflight_te_quant(
    mode: Optional[str],
    fam: Any,
    base: Optional[str],
    target: Any,
    *,
    supplied: Any = (),
    local_files_only: Optional[bool] = None,
) -> Optional[str]:
    """The text-encoder mode a pre-load gate must check for an EXPLICIT request: None when an int8 request will
    load the hosted int8 ConvRot file (plain tensors, no torchao, offload-safe, runs on MPS), else the mode the
    runtime will attempt without it -- torchao int8 on a family with a keep-bf16 schedule and torchao present (with
    its device and offload limits), fp8 otherwise -- so the gate refuses what the load would refuse."""
    normalized = normalize_te_quant(mode)
    family = getattr(fam, "name", None)
    if normalized != TE_QUANT_INT8:
        return effective_te_quant(normalized, family)
    from .diffusion_te_prequant import (
        TE_INT8_CONVROT_FILES,
        explicit_int8_hosted_components,
        explicit_int8_takes_hosted_file,
    )

    if explicit_int8_takes_hosted_file(
        fam, base, target, supplied = supplied or (), local_files_only = local_files_only
    ):
        return None
    takes = explicit_int8_hosted_components(
        fam, base, target, supplied = supplied or (), local_files_only = local_files_only
    )
    missing = [c for c, ok in takes.items() if not ok]
    supplied_hosted = [c for c in missing if c in set(supplied or ())]
    if supplied_hosted and len(missing) < len(takes):
        # A supplied dense encoder in one hosted slot beside a hosted int8 file in another: mixed precisions.
        raise RuntimeError(
            f"Text encoder precision int8 would load the hosted int8 ConvRot file for some {family} text encoders, "
            f"but the supplied {', '.join(supplied_hosted)} is not int8, so the encoders would load at mixed "
            "precisions. Choose fp8 or Dense (bf16), or supply every text encoder."
        )
    if missing and len(missing) < len(takes):
        # Some encoders would load their hosted int8 file and the rest fall back (HiDream: T5 int8 beside an fp8
        # Llama), a mixed build quantize_text_encoders refuses only after the download and the eviction. Refuse here.
        files = TE_INT8_CONVROT_FILES.get((family or "").strip().lower(), {})
        named = ", ".join(
            f"{c} ({files[c][1]})" if c in files and len(files[c]) > 1 else c for c in missing
        )
        raise RuntimeError(
            f"Text encoder precision int8 needs the hosted int8 ConvRot file for every {family} text encoder, but "
            f"{named} is not available here while the others are, so the encoders would load at mixed precisions. "
            "Choose fp8 for all of them, or Dense (bf16)."
        )
    if _family_hosts_te_int8(family):
        if _TE_INT8_SKIP.get((family or "").lower()) is not None and torchao_quantize_importable():
            return TE_QUANT_INT8
        return TE_QUANT_FP8
    return effective_te_quant(normalized, family)


def te_quant_needs_resident_weights(mode: Optional[str]) -> bool:
    """torchao casts use subclasses that Module.to() cannot move, so they need resident weights."""
    return mode in _TE_TORCHAO_MODES


@lru_cache(maxsize = 1)
def torchao_quantize_importable() -> bool:
    """Whether torchao quantize_ is real: rejects the Windows ROCm stub (no-op), checked before download; cached since it cannot change in-process."""
    try:
        from torchao.quantization import quantize_  # noqa: F401
    except Exception:  # noqa: BLE001 -- absent, broken build, missing native symbol
        return False
    return not is_stubbed("torchao")


_TE_QUANT_REQUIREMENTS = {
    TE_QUANT_FP8: "an NVIDIA or AMD GPU that runs bf16 (NVIDIA Ampere sm_80 or newer)",
    TE_QUANT_FP8_DYNAMIC: "an NVIDIA GPU with fp8 tensor cores (Ada sm_89 or newer)",
    TE_QUANT_INT8: "an NVIDIA GPU that runs bf16 (Ampere sm_80 or newer)",
    TE_QUANT_NVFP4: (
        "an NVIDIA GPU that runs bf16 (Ampere sm_80 or newer) and torchao 0.15 or newer on "
        "torch 2.8 or newer"
    ),
}


def te_quant_unsupported_reason(mode: str) -> str:
    return f"text-encoder '{mode}' needs {_TE_QUANT_REQUIREMENTS[mode]}, which this host does not provide"


@lru_cache(maxsize = 1)
def nvfp4_weight_only_importable() -> bool:
    """requires torchao 0.15+; older Studio installs kept 0.14 with torch <= 2.8."""
    try:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        from .diffusion_transformer_quant import _quiet_config
        _quiet_config(NVFP4WeightOnlyConfig)
    except Exception:  # noqa: BLE001
        return False
    return True


def te_quant_supported(target: Any, mode: str) -> bool:
    """CUDA bf16: e4m3 for fp8/nvfp4, sm_89+ for dynamic fp8, sm_80+ for int8; no FP4 cores."""
    if getattr(target, "device", None) != "cuda":
        return False
    if nvfp4_blocked(mode):
        return False
    # Torchao modes fail on the Windows stub or ROCm non-SM capabilities; plain fp8 is only a dtype cast.
    if mode in _TE_TORCHAO_MODES and (is_stubbed("torchao") or torch_is_rocm()):
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
            # weight-only: dequantised to bf16 for a plain gemm, so only the e4m3 block-scale dtype is needed
            return hasattr(torch, "float8_e4m3fn") and nvfp4_weight_only_importable()
    except Exception:
        return False
    return False


class TEQuantOutcome(NamedTuple):
    """What the text-encoder pass actually did, so status can report it instead of guessing.

    ``mode`` is the quantisation APPLIED (None = the encoders stayed dense bf16), ``reason`` is
    the short human-readable why when that differs from the request, and ``status`` is one of the
    ``RESOLVED_*`` constants. Every early return below used to be a bare ``None``: an int8 request
    silently became fp8, an offloaded load silently kept a dense encoder, and an unsupported GPU
    returned without so much as a log line.

    ``partial`` is True when SOME encoder took the cast and another did not. The mode did engage,
    so ``mode`` is not None, but a pipeline conditioning off a mixture of quantised and dense
    encoders is not the build that was asked for and the loaders refuse it like any other declined
    explicit precision."""

    mode: Optional[str]
    reason: str = ""
    status: str = RESOLVED_APPLIED
    partial: bool = False


# Set on an encoder cast in place outside this pass (HiDream's text_encoder_4), so the outcome can report it.
TE_RUNTIME_SCHEME_ATTR = "_unsloth_te_runtime_scheme"


def quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    offload_active: bool = False,
    logger: Any = None,
    encoders: Optional[tuple[str, ...]] = None,
) -> TEQuantOutcome:
    """See ``_quantize_text_encoders``; also folds in HiDream's ``text_encoder_4``, which its family loader quantizes
    (``hidream_te4_kwargs``), so a TE4 that ended up on another scheme is reported rather than hidden."""
    outcome = _quantize_text_encoders(
        pipe,
        target,
        mode = mode,
        family = family,
        offload_active = offload_active,
        logger = logger,
        encoders = encoders,
    )
    te4 = getattr(pipe, "text_encoder_4", None)
    if te4 is None or outcome.mode is None:
        return outcome
    te4_scheme = _hosted_te_scheme(te4) or getattr(te4, TE_RUNTIME_SCHEME_ATTR, None)
    if te4_scheme == outcome.mode:
        return outcome
    held = f"text_encoder_4 is {te4_scheme}" if te4_scheme else "text_encoder_4 stayed dense bf16"
    reason = f"{outcome.reason}; {held}" if outcome.reason else held
    # The fourth conditioner is not on the scheme the rest took: a mixed build, which an explicit request refuses.
    return outcome._replace(reason = reason, status = RESOLVED_FELL_BACK, partial = True)


def _quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    offload_active: bool = False,
    logger: Any = None,
    encoders: Optional[tuple[str, ...]] = None,
) -> TEQuantOutcome:
    """Quantise each present text encoder in place with ``mode``. Returns a ``TEQuantOutcome``
    carrying the mode applied (None when disabled, unsupported, or nothing was cast) plus WHY it
    differs from the request. ``int8`` needs a per-family schedule (``_TE_INT8_SKIP``); without one
    it falls back to ``fp8``. Under ``offload_active`` the torchao modes are skipped (their
    subclasses reject ``Module.to()``); layerwise ``fp8`` still engages. Best-effort: any failure
    leaves the encoder dense. ``encoders`` limits the cast to those pipeline attributes (the rest stay as loaded),
    e.g. HunyuanVideo-1.5's automatic fp8 targets its Qwen tower and leaves the ByT5 glyph encoder dense."""
    picked_automatically = te_quant_picked_automatically(mode)
    mode = normalize_te_quant(mode)
    attrs = tuple(a for a in _TEXT_ENCODER_ATTRS if encoders is None or a in encoders)
    if mode is None:
        present = [a for a in _TEXT_ENCODER_ATTRS if getattr(pipe, a, None) is not None]
        stored_int8 = [a for a in present if _hosted_te_scheme(getattr(pipe, a)) == TE_QUANT_INT8]
        if stored_int8:
            return TEQuantOutcome(
                TE_QUANT_INT8,
                "int8 ConvRot weight-only text encoder, as stored in the supplied file"
                + ("" if len(stored_int8) == len(present) else f" ({', '.join(stored_int8)} only)"),
                RESOLVED_APPLIED,
            )
        return TEQuantOutcome(None)
    # A hosted int8 ConvRot encoder is already quantized (plain tensors, offload-safe): report it, never re-cast it.
    present = [a for a in attrs if getattr(pipe, a, None) is not None]
    hosted_int8 = [a for a in present if _hosted_te_scheme(getattr(pipe, a)) == TE_QUANT_INT8]
    if mode == TE_QUANT_INT8 and hosted_int8 and len(hosted_int8) == len(present):
        return TEQuantOutcome(
            TE_QUANT_INT8,
            "hosted int8 ConvRot weight-only text encoder",
            RESOLVED_APPLIED,
        )
    if (
        mode == TE_QUANT_INT8
        and hosted_int8
        and _hosts_te_int8_for_all_but_small(family, hosted_int8, present)
    ):
        # FLUX.1's CLIP-L, HunyuanImage's ByT5: the hosted file covers the big encoder, and casting a sub-GB pooled /
        # glyph encoder in place would be a runtime-only conversion the file does not carry. Leave them bf16.
        dense = [a for a in present if a not in hosted_int8]
        return TEQuantOutcome(
            TE_QUANT_INT8,
            f"hosted int8 ConvRot weight-only {', '.join(hosted_int8)}; {', '.join(dense)} kept bf16",
            RESOLVED_APPLIED,
        )
    if picked_automatically and not any(
        _hosted_te_scheme(getattr(pipe, a)) == mode for a in present
    ):
        # The automatic default is the measured hosted file and nothing else: a runtime cast (fp8, torchao int8) has a
        # dense bf16 peak and is not what was measured, so without the file the released bf16 encoder stays.
        label = "int8 ConvRot" if mode == TE_QUANT_INT8 else mode
        why = "is not available (not published yet, offline, or refused, see the server log)"
        if mode == TE_QUANT_INT8 and _family_hosts_te_int8(family):
            from .diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV, family_trims_lm_head
            if not family_trims_lm_head(family):
                why = f"carries no lm_head, which {KEEP_LM_HEAD_ENV} keeps"
        _note(logger, f"automatic {label} text encoder not used; keeping the dense bf16 encoder")
        return TEQuantOutcome(
            None,
            f"the hosted {label} text encoder {why}, so the automatic default kept the dense bf16 encoder",
            RESOLVED_FELL_BACK,
        )
    if picked_automatically:
        # The automatic default is the hosted file and nothing more: a sibling it does not replace (Kontext's CLIP-L)
        # stays bf16 rather than taking a runtime cast nobody measured. The hosted encoders arrive with their hooks.
        hosted = [a for a in present if _hosted_te_scheme(getattr(pipe, a)) == mode]
        dense = [a for a in present if a not in hosted]
        return TEQuantOutcome(
            mode,
            f"hosted {mode} {', '.join(hosted)}"
            + (f"; {', '.join(dense)} kept bf16" if dense else ""),
            RESOLVED_APPLIED,
        )
    downgrade_reason = ""
    skip: Optional[tuple[int, int]] = None
    if mode == TE_QUANT_INT8:
        skip = _TE_INT8_SKIP.get((family or "").lower())
        pre_cast = [a for a in present if _hosted_te_scheme(getattr(pipe, a)) is not None]
        if (
            skip is not None
            and _family_hosts_te_int8(family)
            and (pre_cast or not torchao_quantize_importable())
        ):
            # The hosted int8 file did not engage. The on-the-fly torchao int8 (the path before the file existed) is
            # the fallback only over a dense encoder and with torchao present; otherwise fp8.
            skip = None
        if skip is None:
            _note(
                logger,
                f"the hosted int8 ConvRot text encoder for '{family}' did not load; using fp8"
                if _family_hosts_te_int8(family)
                else f"int8 has no keep-bf16 schedule for family '{family}'; using fp8",
            )
            mode = TE_QUANT_FP8
            downgrade_reason = (
                f"int8 has no measured keep-bf16 schedule for family '{family}' "
                "(it degrades large encoders without one), so fp8 was used instead"
            )
            if _family_hosts_te_int8(family):
                from .diffusion_te_prequant import _TE_INT8_HEADLESS_FAMILIES
                from .diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV, family_trims_lm_head
                downgrade_reason = (
                    f"{KEEP_LM_HEAD_ENV} keeps lm_head, which the int8 ConvRot text encoder does not carry, "
                    "so fp8 was used instead"
                    if not family_trims_lm_head(family)
                    and (family or "").strip().lower() not in _TE_INT8_HEADLESS_FAMILIES
                    else "the hosted int8 ConvRot text encoder is not available (not published yet, offline, "
                    "or refused, see the server log), so fp8 was used instead"
                )
    # torchao subclasses reject Module.to(), which offload placement uses.
    if offload_active and mode in (TE_QUANT_INT8, TE_QUANT_FP8_DYNAMIC, TE_QUANT_NVFP4):
        _note(
            logger,
            f"text-encoder '{mode}' skipped under offload (torchao tensors reject Module.to()); "
            "pin a resident memory mode or use fp8",
        )
        return TEQuantOutcome(
            None,
            f"text-encoder '{mode}' cannot run under offload (torchao tensors reject "
            "Module.to()); pin a resident memory mode, or use fp8",
            RESOLVED_UNSUPPORTED,
        )
    if not te_quant_supported(target, mode):
        # Previously a silent return: the only decline site in the loader with no log at all.
        _note(logger, f"text-encoder '{mode}' is not supported on this device; left dense")
        return TEQuantOutcome(
            None,
            f"{te_quant_unsupported_reason(mode)}, so the dense bf16 encoder was kept",
            RESOLVED_UNSUPPORTED,
        )
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
    failed: list[str] = []
    for attr in attrs:
        encoder = getattr(pipe, attr, None)
        if encoder is None:
            continue
        if attr in hosted_int8:
            continue
        try:
            caster(encoder, target)
            cast.append(attr)
        except Exception as exc:  # noqa: BLE001 - leave this encoder dense
            failed.append(attr)
            _warn(logger, f"{mode}:{attr}", exc)
    if not cast:
        return TEQuantOutcome(
            None,
            f"no text encoder on this pipeline could be cast to '{mode}' (see the server log)",
            RESOLVED_FELL_BACK,
        )
    if failed:
        # Mode engaged on some encoders, the uncast ones stay dense bf16; do not report it as fully applied.
        return TEQuantOutcome(
            mode,
            f"'{mode}' engaged on {', '.join(cast)} but {', '.join(failed)} could not be cast and "
            "stayed dense bf16 (see the server log), so conditioning is a mixture",
            RESOLVED_FELL_BACK,
            True,
        )
    if downgrade_reason:
        return TEQuantOutcome(mode, downgrade_reason, RESOLVED_FELL_BACK)
    return TEQuantOutcome(mode, "dense text encoder(s) quantised in place", RESOLVED_APPLIED)


def _hosted_te_scheme(encoder: Any) -> Optional[str]:
    """The scheme of the hosted artifact ``encoder`` was loaded from ("fp8" / "int8"), or None for a dense load."""
    from .diffusion_te_prequant import TE_PREQUANT_SCHEME_ATTR

    value = getattr(encoder, TE_PREQUANT_SCHEME_ATTR, None)
    return value if isinstance(value, str) else None


def _hosts_te_int8_for_all_but_small(family: Optional[str], hosted: list, present: list) -> bool:
    """Whether every encoder the family hosts an int8 ConvRot file for loaded from it, so the rest are the small
    companions the family deliberately leaves dense."""
    try:
        from .diffusion_te_prequant import TE_INT8_CONVROT_FILES

        # HiDream's text_encoder_4 is assembled outside the attributes this pass walks.
        listed = set(TE_INT8_CONVROT_FILES.get((family or "").strip().lower(), {})) & set(
            _TEXT_ENCODER_ATTRS
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(listed) and listed <= set(hosted) and set(hosted) < set(present)


def _family_hosts_te_int8(family: Optional[str]) -> bool:
    try:
        from .diffusion_te_prequant import TE_INT8_CONVROT_FILES
        return (family or "").strip().lower() in TE_INT8_CONVROT_FILES
    except Exception:  # noqa: BLE001
        return False


def _te_exclude_tokens(encoder: Any) -> tuple[str, ...]:
    """fqn tokens whose Linears stay bf16 in a torchao TE quant: the VLM vision tower, the unused
    lm_head, and the encoder's own fp32-kept modules (T5 ``wo``, which explodes in low precision)."""
    tokens = ["visual", "vision_tower", "lm_head"]
    tokens += [str(m).lower() for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ())]
    return tuple(dict.fromkeys(tokens))


def _keep_bf16_block_fqns(encoder: Any, skip_first: int, skip_last: int) -> set[str]:
    """FQNs of decoder blocks to keep bf16: the first ``skip_first`` and last ``skip_last`` of
    each top-level ``nn.ModuleList`` stack. Structural, so no per-architecture table.
    Counts live slots only (truncated dead tails are passthroughs)."""
    import torch

    from .diffusion_te_truncate import live_slots

    keep: set[str] = set()
    for name, module in encoder.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        live = live_slots(module)
        n = len(live)
        if n <= skip_first + skip_last:
            continue
        for i in live[:skip_first] + live[n - skip_last :]:
            keep.add(f"{name}.{i}" if name else str(i))
    return keep


def _cast_int8_selective(encoder: Any, target: Any, skip_first: int, skip_last: int) -> None:
    # Torchao int8 on FLOP-heavy Linears, first/last blocks bf16; config shared with transformer quant.
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
    """A zero output row gives fp8 scale 0 and NaN; only that Linear in SDXL text_encoder_2 stays dense."""
    try:
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return False
        return bool((weight.abs().amax(dim = -1) == 0).any().item())
    except Exception:  # noqa: BLE001 -- unreadable weight: let quantize_ decide
        return False


def _cast_fp8_dynamic(encoder: Any, target: Any) -> None:
    # Torchao per-row fp8 compute via torch._scaled_mm; unlike layerwise fp8, the matmul stays in fp8.
    from torchao.quantization import quantize_
    from .diffusion_transformer_quant import (
        TQ_FP8,
        DEFAULT_MIN_LINEAR_FEATURES,
        _make_quant_config,
        make_filter_fn,
    )

    # require_bf16: scaled_mm asserts a bf16 weight, so skip a stray non-bf16 Linear rather than aborting the pass.
    base = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder), require_bf16 = True
    )

    def filter_fn(module: Any, fqn: str = "") -> bool:
        return base(module, fqn) and not _weight_has_zero_output_row(module)

    quantize_(encoder, _make_quant_config(TQ_FP8), filter_fn = filter_fn)


def _cast_fp8(encoder: Any, target: Any) -> None:
    import re
    import torch
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    # Idempotent via the completion marker, NOT hook presence: a pre-cast encoder already has hooks and re-registering one raises.
    if getattr(encoder, "_unsloth_te_cast_complete", False) and _has_layerwise_hooks(encoder):
        return

    # Layerwise fp8: skip two module kinds that crash at generation:
    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN)

    # (1) _keep_in_fp32_modules, e.g. T5 wo: its gated FF casts activations to the weight dtype, racing the upcast hook. Literal substrings.
    skip += tuple(re.escape(m) for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ()))

    # (2) lm_head tied to embed_tokens (FLUX.2 Qwen3): casting it drags the embedding to fp8 and the first RMSNorm crashes.
    get_out, get_in = (
        getattr(encoder, "get_output_embeddings", None),
        getattr(encoder, "get_input_embeddings", None),
    )
    out_emb = get_out() if callable(get_out) else None
    in_emb = get_in() if callable(get_in) else None
    # getattr: a dropped head (diffusion_text_encoder_trim.NoLogitsHead) has no weight.
    if (
        out_emb is not None
        and in_emb is not None
        and getattr(out_emb, "weight", None) is getattr(in_emb, "weight", object())
    ):
        tied_name = next((n for n, m in encoder.named_modules() if m is out_emb), None)
        if tied_name:
            skip += (rf"^{re.escape(tied_name)}$",)

    # (3) Qwen3.5 / Qwen3-Next gated-delta layers read ``linear_attn.conv1d.weight`` directly into the causal-conv
    # kernel (causal-conv1d / fla installed) and never call the module, so the upcast hook would not run. Tiny tensor.
    skip += (r"(^|\.)linear_attn\.conv1d$",)

    apply_layerwise_casting(
        encoder,
        storage_dtype = torch.float8_e4m3fn,
        compute_dtype = target.dtype,
        skip_modules_pattern = skip,
        # Keep nn.Embedding full precision: fp8 puts every prompt token on the coarse grid.
        skip_modules_classes = (torch.nn.Embedding,),
    )

    # Module.dtype would report fp8 storage, which randn_tensor cannot take; shadow it on the ORIGINAL class, since a __class__ swap breaks output recording.
    compute_dtype = getattr(target, "dtype", None)
    try:
        if compute_dtype is not None:
            _install_dtype_override(type(encoder))
            encoder._unsloth_te_compute_dtype = compute_dtype
        # Flags the cast complete for the idempotent early return; non-Module doubles just re-cast.
        encoder._unsloth_te_cast_complete = True
    except Exception:  # noqa: BLE001 - real HF encoders are heap-type nn.Modules; only doubles fail
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
    # weight-only nvfp4: weights stored 4-bit, dequantised to bf16 per forward; same exclusions as the int8/fp8 modes
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from .diffusion_transformer_quant import (
        DEFAULT_MIN_LINEAR_FEATURES,
        _quiet_config,
        make_filter_fn,
    )

    filter_fn = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES, _te_exclude_tokens(encoder), require_bf16 = True
    )
    # No-op today (the prototype config has no set_inductor_config knob), but no torchao config is built bare.
    quantize_(encoder, _quiet_config(NVFP4WeightOnlyConfig), filter_fn = filter_fn)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: text-encoder quant (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.precision: %s", msg)
