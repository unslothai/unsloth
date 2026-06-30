# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in speed optimisations for the local diffusion backend.

Off by default, so the default render path stays bit-identical to a plain run (the
property the regression harness checks). When the operator opts in, this applies the
lossless-to-near-lossless speedups in the order the diffusers guides recommend
(channels_last -> regional compile, with TF32 / fused-QKV under "max"):

  off     - nothing (default).
  default - lossless: channels_last VAE memory format + regional torch.compile of
            the denoiser's repeated block WHERE eligible (bf16, CUDA, and a
            compile-friendly family).
  max     - default plus near-lossless TF32 matmul and fused QKV projections.

Regional compile is gated only by family compatibility (Z-Image's block is flagged
off) and a bf16 / CUDA target. A GGUF transformer compiles cleanly on diffusers'
native dequant path -- verified empirically: fullgraph, zero graph breaks, ~1.35-1.44x
faster with lower peak VRAM (the per-op dequant is pure tensor ops inductor fuses) --
so it is NOT excluded. torch is imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

SPEED_OFF = "off"
SPEED_DEFAULT = "default"
SPEED_MAX = "max"
SPEED_MODES = (SPEED_OFF, SPEED_DEFAULT, SPEED_MAX)


def normalize_speed_mode(value: Optional[str]) -> str:
    """Lower/strip a requested speed mode (dashes ok); None / "" -> off."""
    if value is None:
        return SPEED_OFF
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return SPEED_OFF
    if normalized not in SPEED_MODES:
        raise ValueError(
            f"Unsupported diffusion speed_mode '{value}'. Use one of: {', '.join(SPEED_MODES)}."
        )
    return normalized


def compile_eligible(target: Any, *, family: Any) -> bool:
    """Whether the denoiser's repeated block should be regionally compiled.

    Only on CUDA (incl. ROCm via supports_default_torch_compile), for a bf16
    transformer, on a compile-friendly family. A GGUF transformer IS eligible -- it
    compiles cleanly on diffusers' native dequant path (verified ~1.4x faster)."""
    if not bool(getattr(target, "supports_default_torch_compile", False)):
        return False
    if not bool(getattr(family, "supports_torch_compile", True)):
        return False
    return _is_bfloat16(getattr(target, "dtype", None))


def _is_bfloat16(dtype: Any) -> bool:
    try:
        import torch
        return dtype is torch.bfloat16
    except Exception:
        return str(dtype).endswith("bfloat16")


def apply_speed_optims(
    pipe: Any,
    target: Any,
    *,
    family: Any,
    speed_mode: str = SPEED_OFF,
    logger: Any = None,
) -> dict[str, bool]:
    """Apply the opt-in speed optimisations for ``speed_mode`` to a built pipeline,
    BEFORE placement / offload. Returns which optimisations actually engaged. Every
    step is best-effort: a pipeline that doesn't support one is simply skipped."""
    applied = {"channels_last": False, "tf32": False, "fused_qkv": False, "compiled": False}
    mode = normalize_speed_mode(speed_mode)
    # TF32 is the one PROCESS-GLOBAL flag we flip (on max). Restore it whenever this
    # load isn't max, so a later default/off diffusion load -- or chat inference in the
    # same long-lived process -- doesn't silently inherit a prior max load's TF32 and
    # lose the bit-identical default the regression harness checks.
    if mode != SPEED_MAX:
        _restore_tf32(logger)
    if mode == SPEED_OFF:
        return applied

    # Lossless: a channels-last VAE speeds up its convolutions with no numeric change.
    applied["channels_last"] = _vae_channels_last(pipe, logger)

    # Lossless-ish: regional compile of the repeated denoiser block, where eligible.
    if compile_eligible(target, family = family):
        applied["compiled"] = _compile_repeated_blocks(pipe, logger)

    if mode == SPEED_MAX:
        # Near-lossless: TF32 matmul (CUDA only) trades a few mantissa bits for speed.
        if getattr(target, "device", None) == "cuda":
            applied["tf32"] = _enable_tf32(logger)
        applied["fused_qkv"] = _fuse_qkv(pipe, logger)

    return applied


def _vae_channels_last(pipe: Any, logger: Any) -> bool:
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "to"):
        return False
    try:
        import torch
        vae.to(memory_format = torch.channels_last)
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "channels_last", exc)
        return False


def _compile_repeated_blocks(pipe: Any, logger: Any) -> bool:
    transformer = getattr(pipe, "transformer", None)
    fn = getattr(transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        return False
    try:
        fn(fullgraph = True, dynamic = True)
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "compile_repeated_blocks", exc)
        return False


# The TF32 flag values from before the first max load flipped them, so a later
# non-max load / unload can put the process back exactly as it found it (rather than
# forcing a hardcoded default that might clobber another component's choice).
_tf32_prev: Optional[tuple[bool, bool]] = None


def _enable_tf32(logger: Any) -> bool:
    global _tf32_prev
    try:
        import torch

        if _tf32_prev is None:
            _tf32_prev = (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "tf32", exc)
        return False


def restore_tf32(logger: Any = None) -> None:
    """Put the process-global TF32 flags back to their pre-max-load values. No-op if
    a max load never set them. Called on a non-max load and on unload."""
    _restore_tf32(logger)


def _restore_tf32(logger: Any) -> None:
    global _tf32_prev
    if _tf32_prev is None:
        return
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = _tf32_prev[0]
        torch.backends.cudnn.allow_tf32 = _tf32_prev[1]
    except Exception as exc:  # noqa: BLE001 — best-effort restore
        _warn(logger, "tf32_restore", exc)
    finally:
        _tf32_prev = None


def _fuse_qkv(pipe: Any, logger: Any) -> bool:
    for owner in (pipe, getattr(pipe, "transformer", None)):
        fn = getattr(owner, "fuse_qkv_projections", None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception as exc:  # noqa: BLE001 — optimisation only
                _warn(logger, "fuse_qkv_projections", exc)
                return False
    return False


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.speed: %s failed: %s", what, exc)
