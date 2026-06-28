# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in speed optimisations for the local diffusion backend.

Off by default, so the default render path stays bit-identical to a plain run (the
property the regression harness checks). When the operator opts in, this applies the
near-lossless speedups in the order the diffusers guides recommend
(channels_last + cudnn.benchmark -> regional compile, with TF32 / fused-QKV under
"max"):

  off     - nothing (default; bit-identical reference).
  default - near-lossless: channels_last VAE memory format + cudnn.benchmark conv
            autotune + regional torch.compile of the denoiser's repeated block WHERE
            eligible (bf16, CUDA, a compile-friendly family). Compile is the big win
            (~2.3x denoise on the GGUF Z-Image transformer, PSNR ~36 dB vs eager,
            well above the Q4 quantisation noise floor, so it does not meaningfully
            move output quality).
  max     - default plus near-lossless TF32 matmul and fused QKV projections.

Regional compile used to be gated off for the GGUF transformer, but it compiles and
runs faster on the current diffusers/torch (measured; the GGUF dequant ops stay
eager and the rest of the repeated block compiles), so the GGUF gate is removed; the
per-family ``supports_torch_compile`` flag and the bf16/CUDA checks still apply.

The backend flags this layer flips (TF32, cudnn.benchmark) are PROCESS-WIDE, so
``snapshot_backend_flags`` / ``restore_backend_flags`` let the caller capture the
prior values at load and restore them at unload, keeping a later ``off`` load
bit-identical instead of inheriting a previous ``max`` run's globals. torch is
imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

SPEED_OFF = "off"
SPEED_DEFAULT = "default"
SPEED_MAX = "max"
SPEED_MODES = (SPEED_OFF, SPEED_DEFAULT, SPEED_MAX)


def snapshot_backend_flags() -> Optional[dict]:
    """Capture the process-wide torch backend flags this layer may mutate, so the
    caller can restore them on unload. None if torch is unavailable. Each flag is read
    defensively so a build/platform missing one (e.g. no cuda.matmul on CPU/MPS) still
    captures the rest -- otherwise a single missing attribute would skip the whole
    snapshot and a real mutated flag would leak."""
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> nothing to snapshot/restore
        return None
    state: dict[str, bool] = {}
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        state["matmul_tf32"] = bool(matmul.allow_tf32)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "allow_tf32"):
            state["cudnn_tf32"] = bool(cudnn.allow_tf32)
        if hasattr(cudnn, "benchmark"):
            state["cudnn_benchmark"] = bool(cudnn.benchmark)
    return state


def restore_backend_flags(state: Optional[dict]) -> None:
    """Restore the flags captured by ``snapshot_backend_flags``. No-op on None. Each
    flag is restored independently so one failure can't leave the others leaked."""
    if not state:
        return
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> nothing to restore
        return

    def _set(obj: Any, attr: str, key: str) -> None:
        if obj is not None and key in state and hasattr(obj, attr):
            try:
                setattr(obj, attr, state[key])
            except Exception:  # noqa: BLE001 — best-effort per-flag restore
                pass

    _set(getattr(getattr(torch.backends, "cuda", None), "matmul", None), "allow_tf32", "matmul_tf32")
    cudnn = getattr(torch.backends, "cudnn", None)
    _set(cudnn, "allow_tf32", "cudnn_tf32")
    _set(cudnn, "benchmark", "cudnn_benchmark")


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


def resolve_speed_mode(value: Optional[str], *, is_gguf: bool) -> str:
    """The effective speed mode when the caller leaves it UNSET (``None``).

    A GGUF model defaults to ``default``: regional compile is ~2.2x faster and its
    numeric perturbation sits well below the quantisation noise floor (measured
    PSNR ~37 dB compile-vs-eager versus ~21 dB Q4-vs-bf16), so it does not reduce
    output quality relative to the dense reference. A dense (non-GGUF) model stays
    ``off`` / bit-identical, since there compile would be the only source of drift.
    An explicit value -- including ``"off"`` -- is always honored verbatim."""
    if value is None:
        return SPEED_DEFAULT if is_gguf else SPEED_OFF
    return normalize_speed_mode(value)


def compile_eligible(target: Any, *, is_gguf: bool, family: Any) -> bool:
    """Whether the denoiser's repeated block should be regionally compiled.

    Only on CUDA (incl. ROCm via supports_default_torch_compile), for a bf16
    transformer, on a compile-friendly family. ``is_gguf`` no longer disqualifies:
    ``compile_repeated_blocks`` runs fine on the GGUF transformer (the per-op
    dequant stays eager, the rest of the block compiles) and is ~2.3x faster, so it
    is kept only for signature/logging compatibility."""
    del is_gguf  # GGUF is compile-eligible now; param kept for call-site compat.
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
    is_gguf: bool,
    family: Any,
    speed_mode: str = SPEED_OFF,
    logger: Any = None,
) -> dict[str, bool]:
    """Apply the opt-in speed optimisations for ``speed_mode`` to a built pipeline,
    BEFORE placement / offload. Returns which optimisations actually engaged. Every
    step is best-effort: a pipeline that doesn't support one is simply skipped."""
    applied = {
        "channels_last": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fused_qkv": False,
        "compiled": False,
    }
    mode = normalize_speed_mode(speed_mode)
    if mode == SPEED_OFF:
        return applied

    # Lossless: a channels-last VAE speeds up its convolutions with no numeric change.
    applied["channels_last"] = _vae_channels_last(pipe, logger)

    # Near-lossless: let cuDNN autotune the fixed-shape VAE convs (CUDA only). It may
    # pick a different conv algorithm, so it is a "default"-tier (not bit-identical) win.
    if getattr(target, "device", None) == "cuda":
        applied["cudnn_benchmark"] = _enable_cudnn_benchmark(logger)

    # Near-lossless and the largest win: regional compile of the repeated denoiser
    # block, where eligible (now incl. the GGUF transformer). `max` opts into
    # max-autotune (longer compile, autotuned kernels).
    if compile_eligible(target, is_gguf = is_gguf, family = family):
        applied["compiled"] = _compile_repeated_blocks(pipe, logger, max_autotune = mode == SPEED_MAX)

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


def _compile_repeated_blocks(
    pipe: Any,
    logger: Any,
    *,
    max_autotune: bool = False,
) -> bool:
    transformer = getattr(pipe, "transformer", None)
    fn = getattr(transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        return False
    # default: mode="default" + dynamic=True -- fast cold start, robust to resolution
    # changes (no recompile). max: mode="max-autotune-no-cudagraphs" + dynamic=False --
    # Triton autotuning for a few % more on GEMM/conv-heavy models, at a much longer
    # compile and a recompile per new resolution. The CUDA-graph modes (reduce-overhead
    # / max-autotune) are deliberately NOT used: they crash on the regionally-compiled
    # block because its static output buffer is overwritten across denoise steps.
    kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": not max_autotune}
    if max_autotune:
        kwargs["mode"] = "max-autotune-no-cudagraphs"
    try:
        fn(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "compile_repeated_blocks", exc)
        return False


def _enable_cudnn_benchmark(logger: Any) -> bool:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "cudnn_benchmark", exc)
        return False


def _enable_tf32(logger: Any) -> bool:
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "tf32", exc)
        return False


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
