# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in dynamic-activation torchao DiT quant; needs torch.compile (eager ~30x slower); torch and
torchao imported lazily; probes best-effort, GGUF fallback"""

from __future__ import annotations

import contextlib as _contextlib
import contextvars as _contextvars
import inspect as _inspect
import os as _os
import re as _re
import sys as _sys
import threading as _threading
from typing import Any, Callable, Optional

# stdlib-only module (no torch), so this stays inside the "imported lazily" promise above.
from core._torchao_stub import is_stubbed, torch_is_rocm
from .diffusion_native_quant import int8_act_disabled, int8_act_requested, is_native_linear
from .diffusion_nvfp4_flag import (
    nvfp4_blocked,
    nvfp4_diffusion_enabled,
    nvfp4_disabled_message,
    without_nvfp4,
)
from .diffusion_torchao_patches import install_torchao_int_mm_patch
from .family_profile import AutoPrefer as _AutoPrefer
from .family_profile import profile_table as _profile_table

# Also runs in the spawned smoke-probe child, which imports this module and nothing else of the backend.
install_torchao_int_mm_patch()

TQ_INT8 = "int8"
TQ_FP8 = "fp8"
TQ_NVFP4 = "nvfp4"
TQ_MXFP8 = "mxfp8"
TQ_AUTO = "auto"
TQ_SCHEMES = (TQ_INT8, TQ_FP8, TQ_NVFP4, TQ_MXFP8)
TQ_MODES = (TQ_AUTO,) + TQ_SCHEMES

# Schemes whose torchao path asserts bf16 weights: skip non-bf16 Linears.
_REQUIRE_BF16_SCHEMES = (TQ_FP8, TQ_MXFP8)

# Per-ROW fp8 is REQUIRED on outlier-heavy DiTs; stamped into prequant metadata.
FP8_GRANULARITY = "per_row"

# torchao ``activation_value_lb``: keeps an all-zero activation row off scale 0 (NaN, black frames); not calibrated.
FP8_ACTIVATION_VALUE_LB = 1e-12

# Skip linears below this feature size: a small FLOP share, so leaving them bf16 costs ~nothing.
DEFAULT_MIN_LINEAR_FEATURES = 512

# int8-only exclusions: _int_mm needs M > 16; AdaLN/timestep/pooled embedders run at M = 1.
_INT8_EXCLUDE_NAME_TOKENS = (
    "norm",  # AdaLN modulation .linear
    "_mod",  # Qwen img_mod / txt_mod
    "modulation",  # Flux.2 double/single_stream_modulation
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",  # pooled-text + timestep; NOT context_embedder
    "pooled",
    # Krea 2 time_embed.linear_2 (M = batch); time_mod_proj is caught by "_mod", and the rest fall under min_features.
    "time_embed",
)


# int8 PER-FAMILY exclusions, on top of _INT8_EXCLUDE_NAME_TOKENS: ``FamilyProfile.int8_exclude_tokens`` on the
# registry rows (small-M text / audio streams, measured quality levers; the reasons sit next to the data). Baked into
# published int8 artifacts and validated at load, so tests/test_diffusion_family_data_freeze.py pins every value.
_INT8_FAMILY_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = _profile_table(
    "int8_exclude_tokens", quant = True
)
_QWENIMAGE_INT8_EXCLUDES = _INT8_FAMILY_EXCLUDE_NAME_TOKENS["qwen-image"]

# Applied ONLY inside ``int8_text_stream_scope(True)`` (streamed builds); hosted artifacts still validate since their
# exclusions are a superset. The scope is part of the prequant-cache key.
_INT8_FAMILY_RUNTIME_QUANT_TOKENS: dict[str, tuple[str, ...]] = _profile_table(
    "int8_runtime_quant_tokens", quant = True
)
INT8_TEXT_STREAM_ENV = "UNSLOTH_DIFFUSION_INT8_TEXT_STREAM"
_TEXT_STREAM_SCOPE: "_contextvars.ContextVar[bool]" = _contextvars.ContextVar(
    "unsloth_int8_text_stream", default = False
)


@_contextlib.contextmanager
def int8_text_stream_scope(streamed: bool):
    token = _TEXT_STREAM_SCOPE.set(bool(streamed))
    try:
        yield
    finally:
        _TEXT_STREAM_SCOPE.reset(token)


def int8_text_stream_enabled() -> bool:
    return (_os.environ.get(INT8_TEXT_STREAM_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def runtime_quant_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    """Baked int8 exclusions quantised anyway; () outside ``int8_text_stream_scope(True)``."""
    if scheme != TQ_INT8 or not _TEXT_STREAM_SCOPE.get() or not int8_text_stream_enabled():
        return ()
    return _INT8_FAMILY_RUNTIME_QUANT_TOKENS.get(str(family or "").strip().lower(), ())


# ALL-scheme PER-FAMILY exclusions (``FamilyProfile.quant_exclude_tokens``): unlike the int8 table above, these also
# apply to fp8 / mxfp8 / nvfp4, whose scaled_mm paths otherwise quantise every eligible Linear.
_FAMILY_QUANT_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = _profile_table(
    "quant_exclude_tokens", quant = True
)


# Int8 PER-FAMILY row PADDING (``FamilyProfile.int8_pad_tokens``), the alternative to excluding a small-M Linear.
# ``_int_mm``'s floor is a GEMM shape constraint, not a numerical one: padding the activation up to 32 rows, running
# the matmul and slicing the result back returns the caller's rows BITWISE unchanged (torchao's int8 dynamic path
# scales activations per row, so the pad rows cannot reach them -- see diffusion_quant_pad for why that is asserted
# rather than assumed). A family with a PUBLISHED int8 checkpoint moves from exclude to pad only with a rebuild and
# republish, since _validate_checkpoint compares the baked exclusion set against exclude_tokens_for_scheme.
_INT8_FAMILY_PAD_NAME_TOKENS: dict[str, tuple[str, ...]] = _profile_table(
    "int8_pad_tokens", quant = True
)


def pad_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    """int8 only, disjoint from the exclude tokens; includes a streamed build's runtime-quantised text stream."""
    if scheme != TQ_INT8:
        return ()
    key = str(family or "").strip().lower()
    tokens = _INT8_FAMILY_PAD_NAME_TOKENS.get(key, ())
    runtime = tuple(t for t in runtime_quant_tokens_for_scheme(scheme, key) if t not in tokens)
    return tokens + runtime


def apply_small_m_padding(
    transformer: Any,
    scheme: str,
    family: Optional[str] = None,
    *,
    logger: Any = None,
) -> tuple[str, ...]:
    """Call after quantising in place, since padding reparents Linears; raises on any unsafe one."""
    tokens = pad_tokens_for_scheme(scheme, family)
    if not tokens:
        return ()
    from .diffusion_quant_pad import matching_linear_fqns, wrap_small_m_linears

    wrapped = wrap_small_m_linears(transformer, matching_linear_fqns(transformer, tokens))
    if wrapped and logger is not None:
        logger.info(
            "diffusion.transformer_quant: padded %d small-M %s linears on %s",
            len(wrapped),
            scheme,
            family,
        )
    return wrapped


# Per-family int8 ConvRot (``FamilyProfile.int8_convrot``): (group, rotated fqn suffixes); not an exclusion, so plain
# artifacts still validate.
_INT8_FAMILY_CONVROT_SPECS = _profile_table("int8_convrot", quant = True)
# A spec with group 0 (ltx-2.5) only names a HOSTED rotated build: no runtime rotation, so no entry here.
_INT8_FAMILY_CONVROT: dict[str, tuple[int, tuple[str, ...]]] = {
    name: (spec.group, spec.suffixes)
    for name, spec in _INT8_FAMILY_CONVROT_SPECS.items()
    if spec.group
}
_INT8_FAMILY_CONVROT_FILENAME: dict[str, str] = {
    name: spec.filename for name, spec in _INT8_FAMILY_CONVROT_SPECS.items() if spec.filename
}
# Only this repo gets the rotated name prepended: a same-named file elsewhere is not this build.
_INT8_FAMILY_CONVROT_REPO: dict[str, str] = {
    name: spec.repo for name, spec in _INT8_FAMILY_CONVROT_SPECS.items() if spec.repo
}
# Variant repos that also host a rotated build (``<Model>-INT8-ConvRot.safetensors``), listed for the same reason.
_INT8_FAMILY_CONVROT_VARIANT_REPOS: dict[str, tuple[str, ...]] = {
    name: spec.variant_repos
    for name, spec in _INT8_FAMILY_CONVROT_SPECS.items()
    if spec.variant_repos
}
# Families whose int8 runs ConvRot unless the env turns it off; the rest stay opt-in (``=1``).
_INT8_FAMILY_CONVROT_DEFAULT_ON: frozenset[str] = frozenset(
    name for name, spec in _INT8_FAMILY_CONVROT_SPECS.items() if spec.default_on
)

INT8_CONVROT_ENV = "UNSLOTH_DIFFUSION_INT8_CONVROT"


def int8_convrot_enabled(family: Optional[str] = None) -> bool:
    """``UNSLOTH_DIFFUSION_INT8_CONVROT``: ``1`` turns ConvRot on for every family with a table entry, ``0`` turns it
    off everywhere (the kill switch); unset follows the family default."""
    raw = (_os.environ.get(INT8_CONVROT_ENV) or "").strip().lower()
    if raw in ("1", "on", "true", "yes"):
        return True
    if raw in ("0", "off", "false", "no"):
        return False
    return str(family or "").strip().lower() in _INT8_FAMILY_CONVROT_DEFAULT_ON


def convrot_spec_for_scheme(
    scheme: str, family: Optional[str] = None
) -> tuple[int, tuple[str, ...]]:
    """The family's int8 ConvRot table entry, regardless of the opt-in flag."""
    if scheme != TQ_INT8:
        return 0, ()
    return _INT8_FAMILY_CONVROT.get(str(family or "").strip().lower(), (0, ()))


def convrot_prequant_filename(scheme: str, family: Optional[str] = None) -> Optional[str]:
    if scheme != TQ_INT8:
        return None
    return _INT8_FAMILY_CONVROT_FILENAME.get(str(family or "").strip().lower())


def convrot_prequant_repo(scheme: str, family: Optional[str] = None) -> Optional[str]:
    """The repo the family's rotated int8 artifact is published to, or None."""
    if scheme != TQ_INT8:
        return None
    return _INT8_FAMILY_CONVROT_REPO.get(str(family or "").strip().lower())


def convrot_prequant_variant_repo(
    scheme: str, family: Optional[str], repo_id: Optional[str]
) -> bool:
    """Whether ``repo_id`` is a variant repo that hosts the family's rotated int8 build under its own name."""
    if scheme != TQ_INT8 or not repo_id:
        return False
    key = str(repo_id).strip().lower()
    repos = _INT8_FAMILY_CONVROT_VARIANT_REPOS.get(str(family or "").strip().lower(), ())
    return any(key == r.lower() for r in repos)


def convrot_fqns(
    transformer: Any, filter_fn: Any, group: int, suffixes: tuple[str, ...]
) -> tuple[str, ...]:
    """Quantized Linears matching ``suffixes`` with width divisible by ``group`` (LoRA ``base_layer`` stays exact)."""
    from .diffusion_convrot import rotatable_fqns

    rotatable, _ = rotatable_fqns(transformer, filter_fn, group)
    names = tuple(n for s in suffixes for n in (s, s + ".base_layer"))
    return tuple(f for f in rotatable if any(f == n or f.endswith("." + n) for n in names))


def apply_runtime_convrot(
    transformer: Any,
    scheme: str,
    family: Optional[str],
    filter_fn: Any,
    *,
    target: Any = None,
    logger: Any = None,
) -> tuple[str, ...]:
    """Rotate BEFORE quantize_; a later failure leaves an exact dense model, so fallback stays correct."""
    group, suffixes = convrot_spec_for_scheme(scheme, family)
    if not group or not int8_convrot_enabled(family):
        return ()
    from .diffusion_convrot import CONVROT_ATTR, CONVROT_KIND, rotate_linears_, warm_rotation_cache

    rotated = rotate_linears_(
        transformer, convrot_fqns(transformer, filter_fn, group, suffixes), group
    )
    if rotated:
        # forward device, not the weights' (CPU-first quantize)
        weight = transformer.get_submodule(rotated[0]).weight
        device = getattr(target, "torch_device", None) or getattr(target, "device", None)
        dtype = getattr(target, "dtype", None)
        try:
            warm_rotation_cache(
                transformer,
                device or weight.device,
                dtype if dtype is not None and not isinstance(dtype, str) else weight.dtype,
            )
        except Exception:  # noqa: BLE001 - only saves one recompile
            pass
    try:
        setattr(
            transformer,
            CONVROT_ATTR,
            {"kind": CONVROT_KIND, "group": group, "linears": len(rotated)},
        )
    except Exception:  # noqa: BLE001 - diagnostic marker only
        pass
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: ConvRot group %d on %d %s linears (%s)",
            group,
            len(rotated),
            scheme,
            family,
        )
    return rotated


# nvfp4 raises on an empty activation (``FamilyProfile.nvfp4_zero_row_tokens``); HunyuanVideo-1.5's attention trim
# hits it every t2v render.
_NVFP4_FAMILY_ZERO_ROW_NAME_TOKENS: dict[str, tuple[str, ...]] = _profile_table(
    "nvfp4_zero_row_tokens", quant = True
)


def zero_row_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    if scheme != TQ_NVFP4:
        return ()
    return _NVFP4_FAMILY_ZERO_ROW_NAME_TOKENS.get(str(family or "").strip().lower(), ())


# Weight subclasses whose activation quantiser asserts a contiguous input (torchao ``to_mx`` "unsupported",
# ``nvfp4_quantize`` "Only support contiguous data for now"), and whose dispatch has no op for a stride-0 view.
_CONTIGUOUS_INPUT_WEIGHTS = frozenset({"MXTensor", "NVFP4Tensor"})


def _contiguous_input_pre_hook(module: Any, args: tuple) -> Optional[tuple]:
    if args and hasattr(args[0], "is_contiguous") and not args[0].is_contiguous():
        return (args[0].contiguous(), *args[1:])
    return None


def guard_contiguous_activations(module: Any, *, logger: Any = None) -> int:
    """Hand MX / NVFP4 Linears a contiguous activation. A broadcast input such as ``temb.expand(B, S, D)`` (FLUX.3's
    per-token modulation, any adaLN fed an expanded embedding) is a stride-0 view; torchao's MX and NVFP4 activation
    quantisers assert contiguity and crash at the first denoise step. ``contiguous()`` is a no-op for the contiguous
    inputs every other call passes, and dynamo traces the pre-hook into the compiled graph. Idempotent; returns how
    many Linears it hooked."""
    named = getattr(module, "named_modules", None)
    if not callable(named):
        return 0
    hooked = 0
    for _name, sub in named():
        weight = getattr(sub, "weight", None)
        if type(weight).__name__ not in _CONTIGUOUS_INPUT_WEIGHTS or getattr(
            sub, "_unsloth_contig_hook", False
        ):
            continue
        try:
            sub.register_forward_pre_hook(_contiguous_input_pre_hook)
            sub._unsloth_contig_hook = True
            hooked += 1
        except Exception:  # noqa: BLE001 - a module refusing hooks keeps its old behaviour
            continue
    if hooked and logger is not None:
        logger.info(
            "diffusion.transformer_quant: contiguous-input guard on %d MX/NVFP4 linears", hooked
        )
    return hooked


def apply_zero_row_guard(
    transformer: Any,
    scheme: str,
    family: Optional[str] = None,
    *,
    logger: Any = None,
) -> tuple[str, ...]:
    """Guard zero-row-reachable quantized Linears; call AFTER quantization (else the first t2v render crashes). Every
    post-quantisation site calls this, so it also installs the contiguous-input guard for MX / NVFP4 weights."""
    guard_contiguous_activations(transformer, logger = logger)
    tokens = zero_row_tokens_for_scheme(scheme, family)
    if not tokens:
        return ()
    from .diffusion_quant_pad import matching_linear_fqns, wrap_zero_row_linears

    wrapped = wrap_zero_row_linears(transformer, matching_linear_fqns(transformer, tokens))
    if wrapped and logger is not None:
        logger.info(
            "diffusion.transformer_quant: zero-row guarded %d %s linears on %s",
            len(wrapped),
            scheme,
            family,
        )
    return wrapped


def exclude_tokens_for_scheme(
    scheme: str,
    family: Optional[str] = None,
    *,
    baked: bool = False,
) -> tuple[str, ...]:
    """Name tokens to exclude from quantisation for ``scheme`` (optionally family-specific). int8
    (M>16) skips the M=1 modulation / conditioning-embedder projections
    (_INT8_EXCLUDE_NAME_TOKENS) plus per-family small-M text streams
    (_INT8_FAMILY_EXCLUDE_NAME_TOKENS); other schemes (scaled_mm) exclude nothing.
    ``family=None`` preserves the historical behaviour. Shared by the runtime path and the
    offline prequant builder so they never drift (else an int8 checkpoint bakes the small-M
    projections and crashes at the first denoise on Flux / Qwen). ``FamilyProfile.quant_exclude_tokens`` applies
    to every scheme. Streamed builds drop runtime tokens; ``baked`` keeps the published-artifact list."""
    key = str(family or "").strip().lower()
    every_scheme = _FAMILY_QUANT_EXCLUDE_NAME_TOKENS.get(key, ())
    if scheme == TQ_INT8:
        family_tokens = _INT8_FAMILY_EXCLUDE_NAME_TOKENS.get(key, ())
        runtime = () if baked else runtime_quant_tokens_for_scheme(scheme, key)
        if runtime:
            family_tokens = tuple(t for t in family_tokens if t not in runtime)
        return _INT8_EXCLUDE_NAME_TOKENS + family_tokens + every_scheme
    return every_scheme


_SCHEME_DIVISIBLE: dict[str, int] = {TQ_FP8: 16, TQ_NVFP4: 16, TQ_MXFP8: 32}


def divisible_for_scheme(scheme: str) -> int:
    """The in/out feature alignment ``scheme``'s GEMM requires, 0 when it has none."""
    return _SCHEME_DIVISIBLE.get(scheme, 0)


# int8 leads: full-rate on consumer cards (which halve fp8 FP32 accumulate).
# nvfp4 reaches auto only via ``_FAMILY_AUTO_PREFER``.
_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    (
        (10, 0),
        (TQ_INT8, TQ_FP8, TQ_MXFP8),
    ),  # Blackwell sm_100+
    ((8, 9), (TQ_INT8, TQ_FP8)),  # Ada sm_89 / Hopper sm_90
    ((8, 0), (TQ_INT8,)),  # Ampere sm_80 / sm_86
)

# Families whose activation ranges break specific schemes at the MODEL level (``FamilyProfile.scheme_deny``; the smoke
# probe only proves the GEMM runs). The deny also applies to an EXPLICIT request; a passing NVFP4 gate record lifts an
# nvfp4 deny per base. Measurements next to the data (qwen-image).
_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = _profile_table("scheme_deny", quant = True)


# ``FamilyProfile.auto_prefer``: a family's head of the ``auto`` order, ahead of ``_AUTO_LADDER``.
_FAMILY_AUTO_PREFER: dict[str, _AutoPrefer] = _profile_table("auto_prefer", quant = True)

_FAMILY_AUTO_SKIP: dict[str, frozenset[str]] = _profile_table("auto_scheme_skip", quant = True)


# AUTO keeps bf16 while it fits for these (``FamilyProfile.auto_bf16_when_resident``: measured with both sides compiled,
# INT8 was at most ~5% faster and failed the default-on LPIPS bar); when bf16 would offload, INT8 is still picked.
_FAMILY_AUTO_BF16_WHEN_RESIDENT: dict[str, str] = _profile_table(
    "auto_bf16_when_resident", quant = True
)


def auto_bf16_when_resident_reason(family: Optional[str]) -> Optional[str]:
    return _FAMILY_AUTO_BF16_WHEN_RESIDENT.get(str(family or "").strip().lower())


# Schemes denied for TRAINING on top of the inference table (``FamilyProfile.train_scheme_deny``): the inference
# evidence is rendering evidence, not evidence that a LoRA converges on those frozen linears.
_FAMILY_TRAIN_SCHEME_DENY: dict[str, frozenset[str]] = _profile_table(
    "train_scheme_deny", quant = True
)


def _nvfp4_gate_passed(family, base_repo) -> bool:
    """Lazy import: the stdlib-only smoke-probe child imports this module whole."""
    if not nvfp4_diffusion_enabled():
        return False
    try:
        from .diffusion_nvfp4_gate import nvfp4_gate_passed
        return bool(nvfp4_gate_passed(family, base_repo))
    except Exception:  # noqa: BLE001 -- see the docstring: an unanswerable gate keeps the deny
        return False


def _nvfp4_gate_backend_ok(family, base_repo, device: Any) -> bool:
    """Whether a passing gate record was measured on the backend this device selects; failure keeps the deny."""
    if not nvfp4_diffusion_enabled():
        return False
    try:
        from .diffusion_nvfp4_gate import nvfp4_gate_backends, nvfp4_gate_flashinfer_mismatch
        from .diffusion_nvfp4_ops import select_nvfp4_backend

        backend = str(select_nvfp4_backend(device)).strip().lower()
        if backend not in nvfp4_gate_backends(family, base_repo):
            return False
        # Auto takes a gated head only on a flashinfer whose kernels the record covers; an unreadable version is not
        # covered. Silent: auto then runs another scheme, and the warning belongs to a load that runs those kernels.
        return (
            backend != "flashinfer"
            or nvfp4_gate_flashinfer_mismatch(family, base_repo, backend_active = True) is None
        )
    except Exception:  # noqa: BLE001 -- see the docstring: an unanswerable probe keeps the deny
        return False


def _nvfp4_backend_is(device: Any, name: str) -> bool:
    """Whether ``select_nvfp4_backend`` picks ``name`` for ``device``; failure drops the head."""
    if not nvfp4_diffusion_enabled():
        return False
    try:
        from .diffusion_nvfp4_ops import select_nvfp4_backend
        return str(select_nvfp4_backend(device)) == str(name)
    except Exception:  # noqa: BLE001 -- see the docstring: an unanswerable probe drops the head
        return False


def _family_denied(
    family,
    scheme: str,
    base_repo: Optional[str] = None,
) -> bool:
    """Whether the deny table rules ``scheme`` out; a gate record lifts nvfp4 per BASE, not per family."""
    if scheme not in _FAMILY_SCHEME_DENY.get(str(family or "").strip().lower(), ()):
        return False
    if scheme == TQ_NVFP4 and _nvfp4_gate_passed(family, base_repo):
        return False
    return True


def _nvfp4_gate_failure(family) -> Optional[str]:
    """The measured gate failure behind a family's nvfp4 deny, or None; lazy, the video registry imports late."""
    try:
        from .video_families import nvfp4_gate_failure
        return nvfp4_gate_failure(family)
    except Exception:  # noqa: BLE001 -- no reason on file falls back to the generic deny text
        return None


def _family_train_denied(
    family,
    scheme: str,
    base_repo: Optional[str] = None,
) -> bool:
    """``_family_denied`` plus training-only denies (always a superset); nvfp4 is denied for every family."""
    if scheme == TQ_NVFP4:
        return True
    key = str(family or "").strip().lower()
    return _family_denied(family, scheme, base_repo) or scheme in _FAMILY_TRAIN_SCHEME_DENY.get(
        key, ()
    )


def nvfp4_gated_base(family: Optional[str]) -> Optional[str]:
    """A base of ``family`` whose passing NVFP4 gate record lifts the family deny, or None. For a load whose base is
    not known yet (a GGUF pick resolves it from its card tag at load time), so a network-free check can judge the
    scheme the way the loader will if the pick turns out to be that base."""
    if not family:
        return None
    try:
        from .diffusion_nvfp4_gate import load_gate_records
        records = load_gate_records()
    except Exception:  # noqa: BLE001 -- no readable record means nothing is lifted
        return None
    name = str(family).strip().lower()
    for record in records:
        base = record.get("base_repo")
        if (
            str(record.get("family", "")).strip().lower() == name
            and base
            and _nvfp4_gate_passed(name, base)
        ):
            return str(base)
    return None


def family_denies_scheme(
    family: Optional[str],
    scheme: str,
    base_repo: Optional[str] = None,
) -> bool:
    """Whether the deny list rules ``scheme`` out for ``family``, so a refusal can tell it from a missing kernel."""
    return _family_denied(family, scheme, base_repo)


def explain_unusable_scheme(
    family: Optional[str],
    scheme: str,
    base_repo: Optional[str] = None,
    *,
    prequant_missing: bool = False,
) -> str:
    """Why an EXPLICIT ``scheme`` got None: family deny, unusable torchao, or a GPU without the kernels."""
    if nvfp4_blocked(scheme):
        return nvfp4_disabled_message()
    if family_denies_scheme(family, scheme, base_repo):
        if scheme == TQ_NVFP4:
            failure = _nvfp4_gate_failure(family)
            if failure is not None:
                return (
                    f"'{scheme}' is ruled out for family '{family}'"
                    + (f" on base '{base_repo}'" if base_repo else "")
                    + f": it was measured and failed the NVFP4 accuracy gate ({failure})"
                )
            return (
                f"'{scheme}' is ruled out for family '{family}'"
                + (f" on base '{base_repo}'" if base_repo else "")
                + ": no reviewed NVFP4 accuracy-gate record (core/inference/"
                "nvfp4_gate_record.json) covers this base at the policy this build resolves, so "
                "nothing has measured what those per-layer precisions do to the render"
            )
        return (
            f"'{scheme}' is ruled out for family '{family}' by the measured accuracy gate (it "
            "renders black frames or fails the quality bar on this DiT), whatever the GPU"
        )
    if prequant_missing:
        return (
            f"'{scheme}' needs a pre-quantized checkpoint for family '{family}' and none is "
            "available for this model (no hosted repo for this base and no transformer_prequant_path)"
        )
    unavailable = torchao_unavailable_reason()
    if unavailable is not None:
        return (
            f"the torchao in this install cannot run any dense quant scheme ({unavailable}); this "
            "is a package/version problem, not a limit of the GPU"
        )
    return f"'{scheme}' is not usable for family '{family}' on this GPU"


# The reason torchao cannot be used AT ALL, resolved once. None means it imports.
_TORCHAO_UNAVAILABLE: Optional[tuple[Optional[str]]] = None


def torchao_unavailable_reason() -> Optional[str]:
    """A torchao that fails to import must not read as a GPU lacking kernels; the smoke probe
    swallows it."""
    global _TORCHAO_UNAVAILABLE
    if _TORCHAO_UNAVAILABLE is not None:
        return _TORCHAO_UNAVAILABLE[0]
    reason: Optional[str] = None
    if is_stubbed("torchao"):
        # the Windows-ROCm stub imports fine and quantises nothing, so the import test below passes
        reason = "this platform ships a torchao stub whose quantize_ is a no-op"
    else:
        try:
            from torchao.quantization import quantize_  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - any import failure means the same thing here
            import logging

            # Full text to server log; client gets paths stripped (it is returned as the 409 detail).
            logging.getLogger(__name__).warning(
                "torchao is unusable: %s: %s", type(exc).__name__, exc
            )
            reason = _strip_paths(f"{type(exc).__name__}: {exc}")
    _TORCHAO_UNAVAILABLE = (reason,)
    return reason


# Requires a directory separator so dotted module names survive.
_ABS_PATH_RE = _re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s'\"()<>|]*[\\/])+[^\s'\"()<>|]*")


def _strip_paths(text: str) -> str:
    """``text`` with server filesystem paths replaced by ``<path>``. Refusal reasons reach an
    authenticated caller as a 409 body: the reason a load was refused is the caller's business,
    where this machine keeps its site-packages is not."""
    return _ABS_PATH_RE.sub("<path>", text)


# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}


def _smoke_cache_device_key(device: str, ordinal: Optional[int] = None) -> str:
    """``device`` qualified with the current CUDA index, or ``ordinal`` when given."""
    if device != "cuda":
        return device
    if ordinal is not None:
        return f"cuda:{ordinal}"
    try:
        import torch
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:  # noqa: BLE001 -- an unreadable index falls back to the un-indexed key
        return device


# Data-center GPU tokens, matched as whole tokens so "A4000" is not "A40".
_DATACENTER_GPU_TOKENS = frozenset(
    {
        "B200",
        "B100",
        "B300",
        "GB200",
        "GB300",
        "GB10",
        "H200",
        "H100",
        "H800",
        "H20",
        "GH200",  # Grace-Hopper superchip
        "A100",
        "A800",
        "A30",
        "A40",
        "A16",
        "A10",
        "A2",
        "L40",
        "L40S",
        "L4",
        "L20",
        "L2",
        "V100",
        "P100",
        "P40",
        "T4",  # legacy data center
    }
)


# professional parts the backend treats as datacenter-class, matched as phrases since the marker spans tokens
_PROFESSIONAL_GPU_MARKERS = ("RTX PRO 6000", "RTX 6000 ADA")


def _is_consumer_gpu(device: Any = None) -> bool:
    """Heuristic on the device name; defaults to consumer, which costs nothing on data-center parts."""
    try:
        import re

        import torch
        name = torch.cuda.get_device_name(device).upper()
    except Exception:  # noqa: BLE001 - no torch / no device -> assume consumer
        return True
    if "GEFORCE" in name or "TITAN" in name:
        return True
    if any(marker in name for marker in _PROFESSIONAL_GPU_MARKERS):
        return False
    tokens = set(re.split(r"[^A-Z0-9]+", name))
    return not (tokens & _DATACENTER_GPU_TOKENS)


def normalize_transformer_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested transformer quant; None / "" / "none" / "off" -> None. Raises
    ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized not in TQ_MODES:
        raise ValueError(
            f"Unsupported transformer_quant '{value}'. Use one of: {', '.join(TQ_MODES)}."
        )
    if nvfp4_blocked(normalized):
        raise ValueError(nvfp4_disabled_message("transformer_quant"))
    return normalized


# GGUF substitutes dense base weights; pipeline rewrites in place; single-file keeps stored precision.
DENSE_QUANT_KINDS: tuple[str, ...] = ("gguf", "pipeline")


def dense_quant_supported_kind(model_kind: Optional[str]) -> bool:
    """Whether ``model_kind`` can reach the dense transformer-quant path at all."""
    return (model_kind or "").strip().lower() in DENSE_QUANT_KINDS


def dense_quant_unsupported_kind_reason(model_kind: Optional[str]) -> str:
    return (
        f"the dense transformer-quant path applies to GGUF and pipeline picks, and this is a "
        f"'{model_kind}' load, which runs the precision its checkpoint carries"
    )


# Both Ideogram branches must use the same precision.
DENOISER_ATTRS: tuple[str, ...] = ("transformer", "unconditional_transformer")

# Some dense DiTs retain norms or embedders in fp32. Other parameter dtypes are not dense sources.
_DENSE_PARAM_DTYPE_NAMES = ("bfloat16", "float32")


class DenoiserView:
    """Expose ``pipe.<attr>`` as ``pipe.transformer`` for the quantisation helpers."""

    def __init__(self, pipe: Any, attr: str) -> None:
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_attr", attr)

    @property
    def transformer(self) -> Any:
        return getattr(
            object.__getattribute__(self, "_pipe"), object.__getattribute__(self, "_attr")
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_pipe"), name)


def denoiser_modules(pipe: Any) -> tuple[tuple[str, Any], ...]:
    """The ``(attribute, module)`` denoisers present on ``pipe``, in conversion order."""
    found = []
    for attr in DENOISER_ATTRS:
        module = getattr(pipe, attr, None)
        if module is not None:
            found.append((attr, module))
    return tuple(found)


# Records a quantised source that a loader widened to a dense dtype.
SOURCE_PRECISION_ATTR = "_unsloth_source_precision"


def mark_source_precision(module: Any, precision: str) -> Any:
    """Record the precision from which a loader dequantised ``module``."""
    try:
        setattr(module, SOURCE_PRECISION_ATTR, precision)
    except Exception:  # noqa: BLE001 -- a marker must never fail the load it describes
        pass
    return module


# safetensors header dtype strings (read before anything is instantiated).
_NARROW_STORED_DTYPES: dict[str, str] = {
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
    "F8_E8M0": "fp8",
    "I8": "int8",
    "U8": "uint8",
    "I4": "int4",
    "U4": "uint4",
    "F4": "fp4",
}


def _cached_snapshot_dir(repo_id: str) -> Any:
    """``repo_id``'s already-cached snapshot, Studio's live cache first (the loaders pass it as ``cache_dir``)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        roots = (active_hf_hub_cache(), None)
    except Exception:  # noqa: BLE001 -- the smoke-probe child has no Studio settings
        roots = (None,)
    for cache_dir in roots:
        try:
            return Path(snapshot_download(repo_id, cache_dir = cache_dir, local_files_only = True))
        except Exception:  # noqa: BLE001 -- not cached there
            continue
    return None


def stored_denoiser_precision(local_dir: Optional[str]) -> Optional[str]:
    """Reads shard headers under a local snapshot only, since from_pretrained widens fp8 to bf16 on load."""
    if not local_dir:
        return None
    try:
        from pathlib import Path

        import safetensors

        root = Path(local_dir).expanduser()
        if not root.is_dir():
            root = _cached_snapshot_dir(str(local_dir))
            if root is None:
                return None
        for attr in DENOISER_ATTRS:
            sub = root / attr
            if not sub.is_dir():
                continue
            # EVERY shard: shards are cut by size, so a later one may hold the fp8 linears.
            for shard in sorted(sub.glob("*.safetensors")):
                with safetensors.safe_open(str(shard), "pt") as handle:
                    for key in handle.keys():
                        dtype = str(handle.get_slice(key).get_dtype()).upper()
                        narrow = _NARROW_STORED_DTYPES.get(dtype)
                        if narrow is not None:
                            return narrow
    except Exception:  # noqa: BLE001 -- an unreadable header is not evidence of anything
        return None
    return None


def _module_quantised_marker(module: Any) -> Optional[str]:
    """Explain why ``module`` is not a dense bf16 source, or return None."""
    if getattr(module, "_unsloth_runtime_quant", None):
        return "it is already quantised by this loader"
    source = getattr(module, SOURCE_PRECISION_ATTR, None)
    if source:
        return (
            f"its published weights are {source} and were widened to bf16 on load, so quantising "
            "them again would compound that loss"
        )
    for flag, what in (("is_loaded_in_4bit", "4-bit"), ("is_loaded_in_8bit", "8-bit")):
        if getattr(module, flag, False):
            return f"it was loaded {what} (bitsandbytes)"
    config = getattr(module, "config", None)
    if getattr(config, "quantization_config", None) is not None:
        return "its checkpoint declares a quantization_config"
    try:
        for param in module.parameters(recurse = True):
            name = str(getattr(param, "dtype", "")).rsplit(".", 1)[-1]
            if name not in _DENSE_PARAM_DTYPE_NAMES:
                return f"its weights are stored as {name}, not dense bfloat16"
    except Exception:  # noqa: BLE001 -- an unwalkable module is not evidence of anything
        return None
    return None


def dense_quant_blocker(pipe: Any) -> Optional[str]:
    """Explain why a pipeline cannot be quantised in place, or return None."""
    denoisers = denoiser_modules(pipe)
    if not denoisers:
        return (
            "this pipeline has no transformer to quantise (its denoiser is a UNet, which the "
            "dense torchao schemes do not cover)"
        )
    for attr, module in denoisers:
        marker = _module_quantised_marker(module)
        if marker is not None:
            where = "its transformer" if attr == "transformer" else f"its {attr}"
            return f"{where} is not a dense bf16 source: {marker}"
    return None


def transformer_is_quantised(module: Any) -> bool:
    """Whether any Linear weight has been replaced by a torchao tensor subclass, or any Linear by a
    torchao-free weight-only twin (``diffusion_native_quant``)."""
    try:
        import torch
        for sub in module.modules():
            if is_native_linear(sub):
                return True
            if not isinstance(sub, torch.nn.Linear):
                continue
            weight = getattr(sub, "weight", None)
            data = getattr(weight, "data", None)
            if data is not None and type(data) is not torch.Tensor:
                return True
            if weight is not None and type(weight) not in (torch.nn.Parameter, torch.Tensor):
                return True
    except Exception:  # noqa: BLE001 -- a probe must never replace the failure it is describing
        return False
    return False


def dense_transformer_supported(target: Any) -> bool:
    """Whether the dense-source quant path is usable for ``target``: a CUDA device with bf16 dtype
    (the only config any torchao dynamic scheme accelerates). Cheap loader pre-check."""
    if getattr(target, "device", None) != "cuda":
        return False
    # The Windows-ROCm torchao stub's quantize_ is a no-op: the probe would pass falsely.
    if is_stubbed("torchao"):
        return False
    # ROCm reports gfx versions via CUDA capability APIs; keep ROCm on GGUF.
    if torch_is_rocm():
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def dense_transformer_unsupported_reason(target: Any) -> str:
    """Why ``dense_transformer_supported`` rejected the target."""
    if getattr(target, "device", None) == "cuda" and torch_is_rocm():
        return (
            "the dense torchao quant schemes are NVIDIA tensor-core paths (int8 sm_80+, fp8 "
            "sm_89+, fp4/mx sm_100+) and this is a ROCm/AMD GPU, so the checkpoint runs at the "
            "precision it ships with"
        )
    if is_stubbed("torchao"):
        return "this platform ships a torchao stub whose quantize_ is a no-op"
    return "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)"


# ``auto`` never lands here: weight-only measured no speed win on gfx1151, only memory.
NATIVE_QUANT_SCHEMES = (TQ_INT8, TQ_FP8)


def _mps_bf16(target: Any) -> bool:
    if getattr(target, "device", None) != "mps":
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def native_quant_host(target: Any) -> bool:
    """Whether ``target`` is a bf16 GPU whose torchao path is closed for a platform reason: a ROCm torch
    (the NVIDIA-shaped capability ladder misclassifies it, and gfx1151 refuses fp8 ``_scaled_mm``), the
    Windows-ROCm torchao stub, or Apple Silicon MPS (no ``_int_mm`` and no fp8; int8 weight-only runs on the
    Metal ``_weight_int8pack_mm`` kernel). False on every NVIDIA, XPU and CPU host, which keep the torchao path."""
    if _mps_bf16(target):
        return True
    if getattr(target, "device", None) != "cuda":
        return False
    if not (torch_is_rocm() or is_stubbed("torchao")):
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


NATIVE_OFFLOAD_SCHEMES = (TQ_INT8,)


def native_offload_host(target: Any) -> bool:
    if getattr(target, "device", None) != "cuda":
        return False
    if torch_is_rocm() or is_stubbed("torchao"):
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def native_quant_scheme(
    target: Any,
    requested: Optional[str],
    family: Optional[str] = None,
    *,
    offload: bool = False,
) -> Optional[str]:
    """Not ``select_transformer_quant_scheme``: that also feeds hosted prequant planners (torchao checkpoints)."""
    scheme = normalize_transformer_quant(requested)
    if scheme not in NATIVE_QUANT_SCHEMES:
        return None
    if not native_quant_host(target) and not (
        offload and scheme in NATIVE_OFFLOAD_SCHEMES and native_offload_host(target)
    ):
        return None
    # MPS cannot cast to fp8 before torch 2.15; checkpoint fp8 codes take ``comfy_fp8_storage`` instead.
    if _mps_bf16(target) and scheme != TQ_INT8:
        return None
    if _family_denied(family, scheme):
        return None
    return scheme


def native_int8_act(target: Any) -> bool:
    # MPS has no ``_int_mm``: W8A8 would only add the activation quant.
    if _mps_bf16(target):
        return False
    if native_quant_host(target):
        return int8_act_requested()
    return not int8_act_disabled()


def select_transformer_quant_scheme(
    target: Any,
    requested: Optional[str],
    family: Optional[str] = None,
    *,
    unproven_ok: bool = False,
    base_repo: Optional[str] = None,
    require_prequant: frozenset[str] = frozenset({TQ_NVFP4}),
    has_prequant: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Auto walks the ladder past failing smoke tests; an explicit scheme is honored or None, not
    swapped."""
    requested = normalize_transformer_quant(requested)
    if requested is None or not dense_transformer_supported(target):
        return None
    device = str(getattr(target, "device", "cuda"))
    if requested != TQ_AUTO:
        if _family_denied(family, requested, base_repo):
            return None
        if not _scheme_supported(requested, device, unproven_ok = unproven_ok):
            return None
        return requested
    cap = _capability()
    if cap is None:
        return None
    for scheme in _auto_scheme_order(family, device, cap, base_repo):
        if not _auto_offers_scheme(scheme, family, base_repo, require_prequant, has_prequant):
            continue
        if _scheme_supported(scheme, device):
            return scheme
    return None


def explicit_scheme_cached_ok(
    target: Any,
    requested: Optional[str],
    family: Optional[str] = None,
) -> bool:
    """``select_transformer_quant_scheme(...) == requested`` for an explicit scheme, read from ``_SMOKE_CACHE`` only:
    never spawns the smoke probe, so a caller that must not touch the GPU (training active) can ask. Unprobed counts
    as supported, like ``unproven_ok``; a probed failure does not."""
    requested = normalize_transformer_quant(requested)
    if requested in (None, TQ_AUTO) or not dense_transformer_supported(target):
        return False
    if nvfp4_blocked(requested) or _family_denied(family, requested, None):
        return False
    card = _smoke_cache_device_key(str(getattr(target, "device", "cuda")))
    return _SMOKE_CACHE.get((requested, card), True)


def dense_quant_host_capable(target: Any) -> bool:
    """Non-allocating: reads only the smoke cache, so an unprobed scheme counts as usable."""
    if not dense_transformer_supported(target):
        return False
    # No inductor means no fast path to advertise (host-level only).
    from .diffusion_speed import compile_eligible

    if not compile_eligible(target, is_gguf = False, family = None, probe = False):
        return False
    # Not redundant: a torchao import failure (ABI skew) rejects every scheme.
    if torchao_unavailable_reason() is not None:
        return False
    ordinal = getattr(target, "ordinal", None)
    cap = _capability(ordinal)
    if cap is None:
        return False
    card = _smoke_cache_device_key(str(getattr(target, "device", "cuda")), ordinal)
    for floor, schemes in _AUTO_LADDER:
        if cap >= floor:
            return any(_SMOKE_CACHE.get((scheme, card), True) for scheme in without_nvfp4(schemes))
    return False


def _auto_offers_scheme(
    scheme: str,
    family: Optional[str],
    base_repo: Optional[str],
    require_prequant: frozenset[str],
    has_prequant: Optional[Callable[[str], bool]],
) -> bool:
    if _family_denied(family, scheme, base_repo):
        return False
    if scheme in (require_prequant or ()):
        if has_prequant is None:
            return False
        try:
            return bool(has_prequant(scheme))
        except Exception:  # noqa: BLE001 -- see the docstring: unanswerable means not offered
            return False
    return True


def auto_scheme_candidates(
    target: Any,
    family: Optional[str] = None,
    *,
    base_repo: Optional[str] = None,
    require_prequant: frozenset[str] = frozenset({TQ_NVFP4}),
    has_prequant: Optional[Callable[[str], bool]] = None,
) -> tuple[str, ...]:
    """Every scheme auto would accept, best first, so a caller can reach one with a hosted checkpoint."""
    if not dense_transformer_supported(target):
        return ()
    device = str(getattr(target, "device", "cuda"))
    cap = _capability()
    if cap is None:
        return ()
    return tuple(
        scheme
        for scheme in _auto_scheme_order(family, device, cap, base_repo)
        if _auto_offers_scheme(scheme, family, base_repo, require_prequant, has_prequant)
        and _scheme_supported(scheme, device)
    )


def _auto_scheme_order(
    family: Optional[str],
    device: Any,
    cap: tuple[int, int],
    base_repo: Optional[str] = None,
) -> tuple[str, ...]:
    """Schemes ``auto`` would try here, best first, before deny list and smoke probe; empty below every tier."""
    tier: tuple[str, ...] = ()
    for floor, schemes in _AUTO_LADDER:
        if cap >= floor:
            tier = schemes
            break
    if not tier:
        return ()
    prefer = _FAMILY_AUTO_PREFER.get(str(family or "").strip().lower())
    if prefer is not None and (prefer.gated or prefer.backend) and not nvfp4_diffusion_enabled():
        # Switch off: drop gated/backend rows before their gate record or backend probe (imports flashinfer).
        prefer = None
    head: tuple[str, ...] = ()
    if prefer is not None and cap >= prefer.floor:
        if prefer.consumer_ok or not _is_consumer_gpu(device):
            if not prefer.gated or (
                _nvfp4_gate_passed(family, base_repo)
                and _nvfp4_gate_backend_ok(family, base_repo, device)
            ):
                if prefer.backend is None or _nvfp4_backend_is(device, prefer.backend):
                    head = prefer.schemes
    order: list[str] = []
    skip = _FAMILY_AUTO_SKIP.get(str(family or "").strip().lower(), frozenset())
    # Filtered here so the NVFP4 switch reaches every reader of the order.
    for scheme in without_nvfp4(head + tier):
        if scheme not in order and scheme not in skip:
            order.append(scheme)
    return tuple(order)


def auto_scheme_candidates_cached(target: Any, family: Optional[str] = None) -> tuple[str, ...]:
    """Answers from _SMOKE_CACHE alone, with no probe, since probing can spawn a child or allocate VRAM."""
    if not dense_transformer_supported(target):
        return ()
    ordinal = getattr(target, "ordinal", None)
    cap = _capability(ordinal)
    if cap is None:
        return ()
    device = str(getattr(target, "device", "cuda"))
    card = _smoke_cache_device_key(device, ordinal)
    return tuple(
        scheme
        for scheme in _auto_scheme_order(family, device, cap)
        if _auto_offers_scheme(scheme, family, None, frozenset({TQ_NVFP4}), None)
        and _SMOKE_CACHE.get((scheme, card), True)
    )


def _capability(ordinal: Optional[int] = None) -> Optional[tuple[int, int]]:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(ordinal)
        return (int(major), int(minor))
    except Exception:
        return None


def _scheme_supported(
    scheme: str,
    device: str,
    *,
    unproven_ok: bool = False,
) -> bool:
    """CUDA + (for fp8) the fp8 dtype + a cached quantise+matmul smoke test for ``scheme``."""
    if nvfp4_blocked(scheme):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if scheme == TQ_FP8 and not hasattr(torch, "float8_e4m3fn"):
            return False
    except Exception:
        return False
    # Resolve in a throwaway child: the first CUDA allocation costs ~700 MiB never returned.
    # Key is card-qualified; a spawn starts on ordinal 0, so pass the card explicitly.
    card = _smoke_cache_device_key(device)
    if (scheme, card) not in _SMOKE_CACHE:
        with _CHILD_PROBE_LOCK:
            # Re-checked under the lock: concurrent plans must not each spawn a child.
            if (scheme, card) not in _SMOKE_CACHE:
                table = _persisted_probe_table(card)
                if table is None or scheme not in table:
                    table = _child_probe_table(card)
                if table is not None:
                    for name, child_verdict in table.items():
                        # None is the child's out-of-memory: not a verdict, so not cached.
                        if child_verdict is not None:
                            _SMOKE_CACHE[(name, card)] = child_verdict
                    if scheme in table:
                        if table[scheme] is None:
                            return unproven_ok
                        # The child ran the same probe. Missing entries still fall through below.
                        return table[scheme]
    return _smoke_probe(scheme, device, unproven_ok = unproven_ok)


# Bounded so a wedged child cannot hang the route; a timeout is no verdict (fall back in-process).
_CHILD_PROBE_TIMEOUT = 180.0

# Latched once spawn is shown not to work here, so the fallback is paid once.
_CHILD_PROBE_UNAVAILABLE = False

# Consecutive spawn OSErrors (transient, retried); reset by the first spawn that works.
_CHILD_PROBE_SPAWN_ERROR_LIMIT = 3
_CHILD_PROBE_SPAWN_ERRORS = 0

_CHILD_PROBE_LOCK = _threading.Lock()


def _child_probe_table(device: str) -> Optional[dict[str, Optional[bool]]]:
    """One child probes every scheme so its CUDA context dies with it; None if no child ran."""
    global _CHILD_PROBE_UNAVAILABLE, _CHILD_PROBE_SPAWN_ERRORS
    if _CHILD_PROBE_UNAVAILABLE:
        return None
    import time

    proc = None
    queue = None
    try:
        import multiprocessing as mp

        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )

        # spawn, never fork: the backend is threaded, and Windows has no fork.
        ctx = mp.get_context("spawn")
        with native_path_secret_removed_for_child_start():
            # Inside the scrub: the resource tracker outlives children and would keep the lease secret.
            queue = ctx.Queue()
            # Same entrypoint as the inference worker: scrubs the lease secret and binds lifetime.
            proc = ctx.Process(
                target = run_without_native_path_secret,
                args = (
                    __name__,
                    "_child_probe_entry",
                    {},
                    device,
                    without_nvfp4(TQ_SCHEMES),
                    queue,
                ),
                daemon = True,
            )
            proc.start()
        # PDEATHSIG is Linux-only and the Windows job object can fail; this record is the backup.
        _adopt_probe_pid(proc.pid)
        _CHILD_PROBE_SPAWN_ERRORS = 0
    except Exception as exc:  # noqa: BLE001 - no child here: probe in-process instead
        import logging

        # OSError is transient: latching would pin the ~800 MiB in-process probe until restart.
        transient = isinstance(exc, OSError)
        if transient:
            _CHILD_PROBE_SPAWN_ERRORS += 1
        if not transient or _CHILD_PROBE_SPAWN_ERRORS >= _CHILD_PROBE_SPAWN_ERROR_LIMIT:
            _CHILD_PROBE_UNAVAILABLE = True
        logging.getLogger(__name__).info(
            "diffusion.transformer_quant: out-of-process probe unavailable (%s: %s); "
            "probing in-process%s",
            type(exc).__name__,
            exc,
            "" if _CHILD_PROBE_UNAVAILABLE else " and retrying the child on the next miss",
        )
        _close_probe_child(proc, queue)
        return None

    table: Optional[dict[str, Optional[bool]]] = None
    deadline = time.monotonic() + _CHILD_PROBE_TIMEOUT
    try:
        while True:
            try:
                table = queue.get(timeout = 0.5)
                break
            except Exception:  # noqa: BLE001 - Empty, or a queue torn down under us
                pass
            if not proc.is_alive():
                # A put lands via a feeder thread: one blocking look before calling it a loss.
                try:
                    table = queue.get(timeout = 1.0)
                except Exception:  # noqa: BLE001 - the child really did die empty
                    table = None
                break
            if time.monotonic() >= deadline:
                break
    finally:
        _close_probe_child(proc, queue)
    if isinstance(table, dict):
        try:
            from . import diffusion_probe_cache
            diffusion_probe_cache.store(device, table)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass
        return table
    return _crashed_child_verdict(proc, device)


def prewarm_probe_table(device: str = "cuda") -> bool:
    """Resolves the card's probe table at boot; UNSLOTH_DIFFUSION_PROBE_PREWARM=0 disables it."""
    if (_os.environ.get("UNSLOTH_DIFFUSION_PROBE_PREWARM") or "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        from . import diffusion_probe_cache

        if not diffusion_probe_cache.enabled():
            return False
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:  # noqa: BLE001
        return False
    card = _smoke_cache_device_key(device)
    schemes = without_nvfp4(TQ_SCHEMES)
    if any((scheme, card) in _SMOKE_CACHE for scheme in schemes):
        return False
    if _persisted_probe_table(card) is not None:
        return False
    with _CHILD_PROBE_LOCK:
        if any((scheme, card) in _SMOKE_CACHE for scheme in schemes):
            return False
        table = _child_probe_table(card)
        if table is None:
            return False
        for name, verdict in table.items():
            if verdict is not None:
                _SMOKE_CACHE[(name, card)] = verdict
    return True


def _persisted_probe_table(card: str) -> Optional[dict[str, bool]]:
    """A previous process's clean child table for this card and stack, or None."""
    try:
        from . import diffusion_probe_cache
        return diffusion_probe_cache.load(card)
    except Exception:  # noqa: BLE001 - a miss, never an error
        return None


# SIGKILL may be OOM and SIGTERM is timeout cleanup, so neither is a scheme verdict.
_PROBE_CRASH_SIGNALS = frozenset({4, 6, 7, 8, 11})  # ILL, ABRT, BUS, FPE, SEGV

# Windows returns the raw NTSTATUS, not -SIGSEGV; only TERMINATE is negative.
_NTSTATUS_ERROR_FLOOR = 0xC0000000


def _crashed_child_verdict(proc: Any, device: str) -> Optional[dict[str, Optional[bool]]]:
    """Return an all-false table after a fatal probe crash, otherwise None. The child may die before
    identifying the scheme, and every scheme shares ``quantize_``, so disable the table rather
    than repeating a proven-fatal probe in the backend."""
    try:
        exitcode = proc.exitcode if proc is not None else None
    except Exception:  # noqa: BLE001 -- no handle left: nothing proven, fall back as before
        return None
    if exitcode is None:
        return None
    if exitcode < 0:
        if -exitcode not in _PROBE_CRASH_SIGNALS:
            return None
        cause = f"signal {-exitcode}"
    elif _sys.platform == "win32" and exitcode >= _NTSTATUS_ERROR_FLOOR:
        cause = f"status 0x{exitcode:08X}"
    else:
        return None
    import logging

    logging.getLogger(__name__).warning(
        "diffusion.transformer_quant: probe child died on %s quantising on %s; treating "
        "every dense quant scheme as unusable here rather than re-running the same probe in this "
        "process, which would take the server down the same way",
        cause,
        device,
    )
    return {scheme: False for scheme in TQ_SCHEMES}


def _adopt_probe_pid(pid: Optional[int]) -> None:
    """Record the probe child against this process's lifetime, so the shutdown sweep and the next
    startup can both reach it. Best-effort: a probe must not fail over its bookkeeping."""
    try:
        from utils.process_lifetime import adopt_pid
        adopt_pid(pid)
    except Exception:  # noqa: BLE001
        pass


def _forget_probe_pid(pid: Optional[int]) -> None:
    try:
        from utils.process_lifetime import forget_pid
        forget_pid(pid)
    except Exception:  # noqa: BLE001
        pass


def _probe_child_alive(proc: Any) -> bool:
    """Whether the child is KNOWN to be running: a missing or never-started handle raises here, and
    that is not a survivor."""
    try:
        return bool(proc is not None and proc.is_alive())
    except Exception:  # noqa: BLE001
        return False


def _close_probe_child(proc: Any, queue: Any) -> bool:
    """Keeps the lifetime record unless death is confirmed, as it is the only handle on held VRAM."""
    try:
        pid = proc.pid if proc is not None else None
    except Exception:  # noqa: BLE001
        pid = None

    for stop, wait in ((lambda: proc.terminate(), 5.0), (lambda: proc.kill(), 3.0)):
        if not _probe_child_alive(proc):
            break
        for step in (stop, lambda: proc.join(timeout = wait)):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass
    for step in (
        lambda: queue.close() if queue is not None else None,
        lambda: queue.join_thread() if queue is not None else None,
    ):
        try:
            step()
        except Exception:  # noqa: BLE001
            pass

    if _probe_child_alive(proc):
        import logging
        logging.getLogger(__name__).error(
            "diffusion.transformer_quant: probe child pid=%s survived terminate and kill; "
            "keeping its lifetime record so the shutdown sweep can still reach it",
            pid,
        )
        return False
    _forget_probe_pid(pid)
    return True


def _select_probe_card(device: str) -> None:
    """Makes the ordinal current so argument-less CUDA calls act on the card the tensors live on."""
    ordinal = device.partition(":")[2]
    if not ordinal.isdigit():
        return
    try:
        import torch
        torch.cuda.set_device(int(ordinal))
    except Exception:  # noqa: BLE001 -- an unselectable index still probes, on the default card
        pass


def _child_probe_entry(device: str, schemes: tuple[str, ...], out: Any) -> None:
    """Probes every scheme even after one fails: verdicts are independent, and CUDA setup is paid once."""
    _select_probe_card(device)
    table: dict[str, Optional[bool]] = {}
    for scheme in schemes:
        try:
            table[scheme] = _run_smoke_probe(scheme, device)
        except Exception:  # noqa: BLE001 - _run_smoke_probe already swallows; keep the table whole
            table[scheme] = False
    try:
        out.put(table)
    except Exception:  # noqa: BLE001 - parent gave up; nothing left to say
        pass


def _smoke_probe(
    scheme: str,
    device: str,
    *,
    unproven_ok: bool = False,
) -> bool:
    """Half the input is zero rows: fp8 scale 0 gives NaN there unless activation_value_lb floors it."""
    # Keyed by CARD: on a heterogeneous host one GPU's verdict must not stand for another.
    key = (scheme, _smoke_cache_device_key(device))
    if key in _SMOKE_CACHE:
        return _SMOKE_CACHE[key]
    ok = _run_smoke_probe(scheme, device)
    if ok is None:
        # OOM is NOT cached nor a verdict: the probe runs before the arbiter evicts the chat model.
        return unproven_ok
    _SMOKE_CACHE[key] = ok
    return ok


def _run_smoke_probe(scheme: str, device: str) -> Optional[bool]:
    """The probe body, uncached: True, False, or None for an allocator failure, which is not a
    verdict on the scheme (see ``_smoke_probe``). Split out so the out-of-process child runs the
    exact same code the in-process fallback does and the two verdicts cannot drift."""
    try:
        import torch
        from torchao.quantization import quantize_

        lin = torch.nn.Linear(512, 512, bias = False).to(device = device, dtype = torch.bfloat16)
        quantize_(lin, _make_quant_config(scheme), filter_fn = make_filter_fn(0))
        # M stays 32 (scaled_mm wants 16-aligned dims); the zero rows go inside it, not after it.
        x = torch.randn(32, 512, device = device, dtype = torch.bfloat16)
        x[16:] = 0
        with torch.no_grad():
            out = lin(x)
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all().item())
    except Exception as exc:  # noqa: BLE001
        return None if _is_out_of_memory(exc) else False


def _is_out_of_memory(exc: BaseException) -> bool:
    """torch.OutOfMemoryError is a RuntimeError, not MemoryError; the message text is the backstop."""
    try:
        import torch
        named = tuple(
            t
            for t in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        if named and isinstance(exc, named):
            return True
    except Exception:  # noqa: BLE001 - no torch: the message test below still applies
        pass
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def _resolve_fast_accum(fast_accum: Optional[bool]) -> bool:
    """Auto (None) is fast accumulate: a large win on RTX 6000 Ada, free on B200."""
    return True if fast_accum is None else bool(fast_accum)


# Disable torchao's process-wide inductor config: it changes results, not just speed.
# UNSLOTH_TORCHAO_INDUCTOR_CONFIG=1 restores it for A/B benchmarking.
_TORCHAO_INDUCTOR_CONFIG_ENV = "UNSLOTH_TORCHAO_INDUCTOR_CONFIG"


def _torchao_may_set_inductor_config() -> bool:
    return (_os.environ.get(_TORCHAO_INDUCTOR_CONFIG_ENV) or "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _quiet_config(cls: Any, **kwargs: Any) -> Any:
    """Build a torchao config with ``set_inductor_config = False`` when the class accepts it. Guarded by
    signature, not version: the mx_formats configs lack the kwarg and never call the setter anyway."""
    if not _torchao_may_set_inductor_config():
        try:
            if "set_inductor_config" in _inspect.signature(cls).parameters:
                kwargs = {**kwargs, "set_inductor_config": False}
        except (
            TypeError,
            ValueError,
        ):  # C-implemented or otherwise unintrospectable: leave it alone
            pass
    return cls(**kwargs)


def _install_fp8_unit_scale() -> None:
    try:
        import logging

        from .diffusion_fp8_unit_scale import install
        install(logger = logging.getLogger(__name__))
    except Exception:  # noqa: BLE001 - never block a load on a speed path
        pass


def _make_quant_config(scheme: str, fast_accum: Optional[bool] = None) -> Any:
    """The torchao dynamic-activation config for ``scheme`` (lazy imports per branch).
    ``fast_accum`` applies to fp8 only: None auto-detects by GPU class, True/False force it."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    if scheme == TQ_INT8:
        return _quiet_config(Int8DynamicActivationInt8WeightConfig)
    if scheme == TQ_FP8:
        # Per-ROW granularity is REQUIRED: per-tensor scale collapses on outliers (Z-Image).
        # activation_value_lb floors the scale: an all-zero row otherwise gives NaN.
        from torchao.quantization import PerRow

        _install_fp8_unit_scale()
        fp8_kwargs: dict = {"granularity": PerRow()}
        config_params = _inspect.signature(Float8DynamicActivationFloat8WeightConfig).parameters
        if "activation_value_lb" in config_params:
            fp8_kwargs["activation_value_lb"] = FP8_ACTIVATION_VALUE_LB
        # Pin the plain-torch quantize kernel: MSLK changes fp8 rounding, breaking prequant identity.
        if "kernel_preference" in config_params:
            try:
                from torchao.quantization.quantize_.common.kernel_preference import (
                    KernelPreference,
                )
                fp8_kwargs["kernel_preference"] = KernelPreference.TORCH
            except Exception:  # noqa: BLE001 - enum moved: keep the library default
                pass
        try:
            from torchao.float8 import Float8MMConfig
            return _quiet_config(
                Float8DynamicActivationFloat8WeightConfig,
                mm_config = Float8MMConfig(use_fast_accum = _resolve_fast_accum(fast_accum)),
                **fp8_kwargs,
            )
        except Exception:  # noqa: BLE001 - older torchao without the explicit mm knob
            return _quiet_config(Float8DynamicActivationFloat8WeightConfig, **fp8_kwargs)
    if scheme == TQ_NVFP4:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        from .diffusion_nvfp4_act_quant import install_nvfp4_act_quant_patch

        # Before quantize_: the load-time weight pass stays on torchao (eager); compiled activations use the kernel.
        install_nvfp4_act_quant_patch()

        # Select the CUTLASS FP4 path, not the default Triton kernel (which needs MSLK): on a Blackwell box with
        # CUTLASS FP4 but no MSLK the default fails the smoke probe and falls back to GGUF.
        try:
            return _quiet_config(NVFP4DynamicActivationNVFP4WeightConfig, use_triton_kernel = False)
        except TypeError:  # older torchao without the knob
            return _quiet_config(NVFP4DynamicActivationNVFP4WeightConfig)
    if scheme == TQ_MXFP8:
        import torch
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        try:
            return _quiet_config(
                MXDynamicActivationMXWeightConfig,
                activation_dtype = torch.float8_e4m3fn,
                weight_dtype = torch.float8_e4m3fn,
            )
        except (TypeError, AttributeError):
            # TypeError: older torchao; AttributeError: torch without float8_e4m3fn.
            return _quiet_config(MXDynamicActivationMXWeightConfig)
    raise ValueError(f"unknown transformer quant scheme '{scheme}'")


def make_filter_fn(
    min_features: int,
    exclude_name_tokens: tuple[str, ...] = (),
    *,
    require_bf16: bool = False,
    require_divisible: int = 0,
):
    """A non-bf16 Linear aborts the whole quantise pass; an unaligned one crashes the first matmul later."""

    def filter_fn(module: Any, fqn: str = "") -> bool:
        try:
            import torch
            if not isinstance(module, torch.nn.Linear):
                return False
        except Exception:
            return False
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            return False
        if in_features < min_features or out_features < min_features:
            return False
        if require_divisible and (
            in_features % require_divisible or out_features % require_divisible
        ):
            return False
        if exclude_name_tokens:
            name = fqn.lower() if fqn else ""
            if any(tok in name for tok in exclude_name_tokens):
                return False
        if require_bf16:
            import torch
            weight = getattr(module, "weight", None)
            if weight is None or weight.dtype != torch.bfloat16:
                return False
        return True

    return filter_fn


def quantize_transformer(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    base_repo: Optional[str] = None,
    min_features: int = DEFAULT_MIN_LINEAR_FEATURES,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
    offload: bool = False,
    act_int8: Optional[bool] = None,
) -> Optional[str]:
    """Quantise ``pipe.transformer`` in place, returning the scheme or None (GGUF); ``base_repo`` picks the NVFP4 policy."""
    native = native_quant_scheme(target, mode, family = family, offload = offload)
    if native is not None:
        mps = _mps_bf16(target)
        if act_int8 is None or mps:
            act_int8 = native_int8_act(target)
        return _quantize_native(
            pipe,
            native,
            family = family,
            min_features = min_features,
            act_int8 = bool(act_int8) and native == TQ_INT8,
            logger = logger,
            scale_dtype = getattr(target, "dtype", None) if mps else None,
        )
    scheme = select_transformer_quant_scheme(target, mode, family = family, base_repo = base_repo)
    if scheme is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    try:
        from torchao.quantization import quantize_

        policy = None
        if scheme == TQ_NVFP4:
            from .diffusion_nvfp4_policy import quantize_with_policy, resolve_policy
            policy = resolve_policy(family, base_repo)
        if policy is not None:
            quantize_with_policy(
                transformer,
                policy,
                min_features = min_features,
                fast_accum = fast_accum,
                logger = logger,
            )
            apply_small_m_padding(transformer, scheme, family, logger = logger)
            apply_zero_row_guard(transformer, scheme, family, logger = logger)
            for attr, value in (
                ("_unsloth_runtime_quant", scheme),
                ("_unsloth_nvfp4_policy", policy.policy_id),
            ):
                try:
                    setattr(transformer, attr, value)
                except Exception:  # noqa: BLE001 - markers are best-effort
                    pass
            return scheme

        # Runtime only: NOT part of exclude_tokens_for_scheme (baked into prequant metadata).
        exclude = exclude_tokens_for_scheme(scheme, family) + ("lora_",)
        divisible = divisible_for_scheme(scheme)
        filter_fn = make_filter_fn(
            min_features,
            exclude_name_tokens = exclude,
            require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
            require_divisible = divisible,
        )
        rotated = apply_runtime_convrot(
            transformer, scheme, family, filter_fn, target = target, logger = logger
        )
        quantize_(
            transformer,
            _make_quant_config(scheme, fast_accum = fast_accum),
            filter_fn = filter_fn,
        )
        # Not best-effort: a raise means quantized but not compilable, so the caller loads GGUF.
        apply_small_m_padding(transformer, scheme, family, logger = logger)
        apply_zero_row_guard(transformer, scheme, family, logger = logger)
        try:
            transformer._unsloth_runtime_quant = scheme
            # What diffusion_prequant_cache needs to persist this exact build.
            transformer._unsloth_runtime_quant_info = {
                "scheme": scheme,
                "family": family,
                "min_features": int(min_features),
                "exclude_name_tokens": list(exclude),
                "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
                "require_divisible": divisible,
                "fast_accum": fast_accum,
                "convrot_group": convrot_spec_for_scheme(scheme, family)[0] if rotated else None,
                "convrot_fqns": list(rotated or ()),
                "dtype": str(getattr(target, "dtype", None) or _first_weight_dtype(transformer)),
            }
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        return scheme
    except Exception as exc:  # noqa: BLE001 - leave the transformer dense -> GGUF fallback
        _warn(logger, scheme, exc)
        return None


def _quantize_native(
    pipe: Any,
    scheme: str,
    *,
    family: Optional[str],
    min_features: int,
    act_int8: bool,
    logger: Any,
    scale_dtype: Any = None,
) -> Optional[str]:
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    try:
        from .diffusion_native_quant import apply_native_weight_quant

        # require_bf16 keeps deliberate fp32 linears (Wan, Hunyuan). Baked list: W8A8's small-M branch would recompile.
        filter_fn = make_filter_fn(
            min_features,
            exclude_name_tokens = exclude_tokens_for_scheme(scheme, family, baked = True) + ("lora_",),
            require_bf16 = True,
        )
        if not apply_native_weight_quant(
            transformer,
            scheme,
            filter_fn = filter_fn,
            act_int8 = act_int8,
            logger = logger,
            scale_dtype = scale_dtype,
        ):
            return None
        if scale_dtype is not None:
            from .diffusion_native_quant import probe_mps_int8pack

            # Timed on this chip, not assumed from the device type.
            probe_mps_int8pack(transformer, scale_dtype, logger = logger)
        try:
            transformer._unsloth_runtime_quant = scheme
            transformer._unsloth_runtime_quant_info = {"scheme": scheme, "native": True}
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        return scheme
    except Exception as exc:  # noqa: BLE001 - a partial swap is reported by transformer_is_quantised
        _warn(logger, f"{scheme} (weight-only)", exc)
        return None


def quantize_runtime_leftovers(
    transformer: Any,
    scheme: str,
    family: Optional[str],
    baked_excludes: Any,
    *,
    min_features: int = DEFAULT_MIN_LINEAR_FEATURES,
    device: Any = None,
    logger: Any = None,
) -> int:
    """Quantise the dense Linears an artifact baked under ``baked_excludes`` that this runtime quantises
    (``runtime_quant_tokens_for_scheme``); host-placed Linears are quantised on ``device`` and moved back. Returns the
    count."""
    runtime = runtime_quant_tokens_for_scheme(scheme, family)
    if not runtime or not isinstance(baked_excludes, (list, tuple)):
        return 0
    tokens = tuple(t for t in runtime if t in baked_excludes)
    if not tokens:
        return 0
    import time

    import torch
    from torchao.quantization import quantize_

    from .diffusion_quant_pad import is_quantized_linear

    keep = make_filter_fn(
        min_features,
        exclude_name_tokens = exclude_tokens_for_scheme(scheme, family) + ("lora_",),
        require_bf16 = True,
    )
    targets = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(tok in name.lower() for tok in tokens)
        and not is_quantized_linear(module)
        and keep(module, name)
    ]
    if not targets:
        return 0
    work = None
    try:
        work = torch.device(device) if device is not None else None
    except Exception:  # noqa: BLE001 - an unparsable device quantises in place
        work = None
    if work is not None and work.type != "cuda":
        work = None
    config = _make_quant_config(scheme)
    t0 = time.perf_counter()
    for _name, module in targets:
        home = module.weight.device
        hop = work is not None and home.type == "cpu"
        if hop:
            module.to(work)
        quantize_(module, config)
        if hop:
            module.to(home)
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: quantised %d %s Linear(s) the artifact kept dense (%s) in %.1f s",
            len(targets),
            scheme,
            ", ".join(tokens),
            time.perf_counter() - t0,
        )
    return len(targets)


def _first_weight_dtype(module: Any) -> Any:
    """The dtype the dense model was loaded in: a parameter left high precision by the quant filter (norms, embeds)."""
    try:
        import torch
        for param in module.parameters():
            if (
                type(param.data) in (torch.Tensor, torch.nn.Parameter)
                and param.dtype.is_floating_point
            ):
                return param.dtype
    except Exception:  # noqa: BLE001
        pass
    return None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.transformer_quant: %s failed: %s", what, exc)
