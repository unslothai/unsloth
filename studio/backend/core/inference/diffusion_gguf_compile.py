# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF dequant accelerator for the light-compile (``default``) diffusion path.

Profiling (outputs/profile_eager/) showed that ~70-80% of EAGER GGUF denoise CUDA
time is the per-forward weight dequant: every ``GGUFLinear.forward`` calls
``diffusers.quantizers.gguf.utils.dequantize_gguf_tensor`` -> ``dequantize_blocks_Q4_K``,
a ~20-op pure-PyTorch chain (nibble shifts + masks + block-scale mul + zero-point sub +
assembly), once per linear per step.

COMPILED DEQUANT (``install_compiled_dequant``): swap ``dequantize_gguf_tensor`` for
``torch.compile(orig, dynamic=True)``. Inductor fuses the op chain into a few kernels.
Measured 1.24-1.64x warm with a small one-time compile (~7.5-10.4s) and ZERO extra VRAM --
the weights stay quantized. ``dynamic=True`` is key: the dequant inputs are the WEIGHT
tensors (fixed shapes, independent of image resolution / batch), so it compiles once and
never recompiles on a resolution change. ``GGUFLinear.forward_native`` resolves the
function as a module global, so replacing the module attribute reroutes every linear.

It is the ``default`` tier's lever (the transformer block stays eager). It is deliberately
NOT used under ``max`` (full regional block compile): there the block is compiled as one
graph which fuses the dequant inline, and a separately-compiled dequant would be traced
into that graph and break it -- so ``max`` runs the stock dequant and lets the block
compile fuse it.

The swap goes through the shared, fingerprint-checked, reversible patch backend
(``diffusion_patch_backend``); ``uninstall_*`` restores the exact original so a later
bit-identical ``off`` load runs the stock dequant. Kill-switch:
``UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT=0``. torch / diffusers are imported lazily.

(A global weight-buffer accelerator lived here too but was removed: it measured neutral
end-to-end -- the CUDA caching allocator already serves the per-forward cast allocation
from its pool with zero ``cudaMalloc`` churn, so reusing one buffer saved nothing on a
DiT's compute-bound forward. See outputs/arch_patch/SUMMARY.md.)
"""

from __future__ import annotations

import os
from typing import Any

from .diffusion_patch_backend import apply_patch, revert_patch

# --- kill-switch -------------------------------------------------------------------

_ENV_COMPILE_DEQUANT = "UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT"
_DISABLED = {"0", "off", "false", "no"}


def _enabled(env_name: str) -> bool:
    """Enabled unless explicitly disabled (default ON)."""
    return str(os.environ.get(env_name, "1")).strip().lower() not in _DISABLED


def _gguf_utils():
    """The diffusers GGUF utils module, or None if this diffusers build lacks it."""
    try:
        from diffusers.quantizers.gguf import utils as gguf_utils  # noqa: PLC0415
        return gguf_utils
    except Exception:  # noqa: BLE001 — old/!GGUF diffusers -> accelerator is a no-op
        return None


# --- compiled dequant --------------------------------------------------------------

# True while our compiled wrapper is installed (the shared patch backend stashes the
# original on the module for an exact restore).
_compiled_dequant_installed = False
_DEQUANT_ATTR = "dequantize_gguf_tensor"


def is_compiled_dequant_installed() -> bool:
    return _compiled_dequant_installed


def install_compiled_dequant(logger: Any = None) -> bool:
    """Replace ``dequantize_gguf_tensor`` with ``torch.compile(orig, dynamic=True)`` via the
    shared patch backend (original stashed for restore).

    Idempotent (a second call is a no-op while installed). Returns True if the compiled
    dequant is in place afterwards, False if disabled / unavailable / it failed."""
    global _compiled_dequant_installed
    if not _enabled(_ENV_COMPILE_DEQUANT):
        return False
    if _compiled_dequant_installed:
        return True
    gguf_utils = _gguf_utils()
    if gguf_utils is None or not hasattr(gguf_utils, _DEQUANT_ATTR):
        return False
    try:
        import torch  # noqa: PLC0415

        compiled = torch.compile(gguf_utils.dequantize_gguf_tensor, dynamic = True)
        # force=True: the new callable is the SAME function compiled, so its fingerprint
        # differs from the original and can_safely_patch would (correctly) reject it.
        if apply_patch(gguf_utils, _DEQUANT_ATTR, compiled, force = True):
            _compiled_dequant_installed = True
            return True
        return False
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "install_compiled_dequant", exc)
        _compiled_dequant_installed = False
        return False


def uninstall_compiled_dequant() -> None:
    """Restore the original ``dequantize_gguf_tensor``. Idempotent."""
    global _compiled_dequant_installed
    if not _compiled_dequant_installed:
        return
    gguf_utils = _gguf_utils()
    if gguf_utils is not None:
        revert_patch(gguf_utils, _DEQUANT_ATTR)
    _compiled_dequant_installed = False


# --- convenience -------------------------------------------------------------------


def uninstall_all() -> None:
    """Uninstall the GGUF accelerator. Idempotent; safe to call on every unload."""
    uninstall_compiled_dequant()


def is_installed() -> bool:
    return is_compiled_dequant_installed()


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.gguf_compile: %s failed: %s", what, exc)
