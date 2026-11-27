# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11xx: store denoiser 16-bit ``nn.Linear`` weights as a view of an [N, K + P] buffer so hipBLASLt avoids
power-of-two strides that alias on Strix Halo memory channels (gfx1151 1.1-1.3x block GEMMs, +P / K memory). Weights
whose (N, K) has a table row keep lda = K and the tuned row; padded shapes gave byte-identical outputs on the measured
gfx1151 shapes and renders (not proven for every shape / kernel choice).

TunableOp rows are keyed on the leading dimension, so a padded weight misses the shipped table (0.90-0.94x). Env unset:
weights whose (N, K) has a loaded table row stay contiguous (byte-identical to main). ``UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH``:
``0`` off; ``<P>`` (multiple of 8) forces pitch P on any gfx11xx, keeping table rows only for ``TABLE_FAMILIES`` (or
while tuning / with a local tuned cache / user-managed TunableOp). User-enabled TunableOp (``PYTORCH_TUNABLEOP_ENABLED=1``)
guards its loaded rows the same way; enabled but unreadable keeps every weight contiguous."""

from __future__ import annotations

import os
from typing import Any, Optional

PITCH_ENV = "UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH"
DEFAULT_PITCH = 64
ARCH_PREFIX = "gfx11"
DEFAULT_ON_ARCHES: frozenset = frozenset({"gfx1151"})
MIN_IN_FEATURES = 1024
TABLE_FAMILIES = frozenset({"qwen-image-2.1", "flux.3-klein", "flux.3-klein-base", "flux.2-klein"})
TABLE_STEM_TOKENS = {
    "qwenimage21": ("qwen-image-2.1",),
    "flux3klein": ("flux.3-klein", "flux.3-klein-base"),
    "flux2klein4b": ("flux.2-klein",),
}
STATS = {"padded": 0, "bytes": 0, "skipped_table": 0, "skipped_other": 0, "calls": 0, "oom": 0}
_HOOK_ATTR = "_unsloth_weight_pitch_hook"
_DEVICE_TYPES = ("cuda",)


def _env() -> str:
    return (os.environ.get(PITCH_ENV) or "").strip().lower()


def pitch(arch: Optional[str]) -> int:
    """Pitch in elements for a gfx11xx ``arch`` (from ``_arch``); 0 = off."""
    if not arch or not str(arch).startswith(ARCH_PREFIX):
        return 0
    raw = _env()
    if raw in ("0", "off", "false", "no"):
        return 0
    if raw:
        if not raw.isdigit():
            return 0
        p = int(raw)
        return p if p > 0 and p % 8 == 0 else 0
    return DEFAULT_PITCH if arch in DEFAULT_ON_ARCHES else 0


def _arch(device: Any) -> Optional[str]:
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch

        if not is_rocm_torch(torch):
            return None
        name = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", "") or "")
        name = name.split(":")[0]
        return name if name.startswith(ARCH_PREFIX) else None
    except Exception:  # noqa: BLE001
        return None


def parse_row(param_sig: str) -> Optional[tuple]:
    """``tn_N_M_K_ld_LDA_LDB_LDC`` -> (N, K, M) for a weight-on-the-left TN GEMM with an unpadded weight."""
    parts = str(param_sig).split("_")
    if len(parts) < 8 or parts[0] != "tn" or parts[4] != "ld":
        return None
    try:
        n, m, k, lda = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[5])
    except ValueError:
        return None
    return (n, k, m) if lda == k else None


def _user_tunableop(torch: Any) -> bool:
    """TunableOp enabled outside Studio (``PYTORCH_TUNABLEOP_ENABLED=1``, a user call): its rows are the user's."""
    try:
        return bool(torch.cuda.tunable.is_enabled())
    except Exception:  # noqa: BLE001 - no TunableOp in this build: nothing to guard
        return False


def table_shapes() -> Optional[set]:
    """(N, K) of every TN row TunableOp currently holds, Studio- or user-enabled; None when TunableOp is off."""
    try:
        import torch

        from . import diffusion_rocm_tunableop as tunableop

        if not tunableop.is_active() and not _user_tunableop(torch):
            return None
        rows = list(torch.cuda.tunable.get_results() or ())
    except Exception:  # noqa: BLE001 - enabled but unreadable: treat every weight as covered
        return {"*"}
    out = set()
    for row in rows:
        try:
            got = parse_row(row[1])
        except Exception:  # noqa: BLE001
            got = None
        if got is not None:
            out.add(got[:2])
    return out


def _table_family(family: Optional[str], arch: Optional[str]) -> bool:
    if str(family or "").strip().lower() in TABLE_FAMILIES:
        return True
    try:
        import torch

        from . import diffusion_rocm_tunableop as tunableop

        if tunableop.tuning_requested():
            return True
        if not tunableop.is_active() and _user_tunableop(torch):
            # User-managed rows are tuned for lda = K whatever the family: never pad over them.
            return True
        return bool(arch) and tunableop.results_file(arch).exists()
    except Exception:  # noqa: BLE001
        return True


def install(
    module: Any,
    logger: Any = None,
    family: Optional[str] = None,
) -> int:
    """Pads on the first forward (weights are placed by then, not at speed-optim time); 1 when armed."""
    if module is None or _env() in ("0", "off", "false", "no"):
        return 0
    if not callable(getattr(module, "register_forward_pre_hook", None)):
        return 0
    if module.__dict__.get(_HOOK_ATTR) is not None:
        return 0
    if pitch(_arch(0)) <= 0:
        return 0

    def hook(mod, args):
        try:
            import torch
            if _must_defer(torch):
                return  # stay armed: pad on the next eager, non-captured forward
        except Exception:  # noqa: BLE001 - fall through to a single pad attempt
            pass
        handle = mod.__dict__.get(_HOOK_ATTR)
        if handle:
            handle.remove()
        mod.__dict__[_HOOK_ATTR] = False
        try:
            pad_weights(mod, logger, family = family)
        except Exception as exc:  # noqa: BLE001 - every weight is still a valid (padded or contiguous) tensor
            if logger is not None:
                logger.warning("diffusion.rocm_weight_pitch: skipped: %s", exc)

    module.__dict__[_HOOK_ATTR] = module.register_forward_pre_hook(hook)
    return 1


def _must_defer(torch: Any) -> bool:
    """Re-storing weights is illegal while tracing (compile) or capturing a CUDA / HIP graph."""
    if torch.compiler.is_compiling():
        return True
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _release_old_storage(torch: Any) -> None:
    """Free the replaced weights, else reserved memory holds a second denoiser copy (Wan 5B +5.2 GiB)."""
    try:
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a later allocation reuses the blocks anyway
        pass


def pad_weights(
    module: Any,
    logger: Any = None,
    family: Optional[str] = None,
) -> int:
    if module is None:
        return 0
    import torch

    if _must_defer(torch):
        return 0
    STATS["calls"] += 1
    shapes = table_shapes()
    guard: Optional[bool] = None
    n = skipped_table = skipped_other = 0
    arch_by_dev: dict = {}
    p = 0
    oom: Optional[BaseException] = None
    # Not inference tensors: later LoRA / training must save these weights for backward.
    try:
        with torch.inference_mode(False), torch.no_grad():
            for m in module.modules():
                if type(m) is not torch.nn.Linear:
                    continue
                w = m.weight
                if (
                    not isinstance(w, torch.nn.Parameter)
                    or type(w.data) is not torch.Tensor
                    or w.dim() != 2
                    or w.device.type not in _DEVICE_TYPES
                    or w.dtype not in (torch.bfloat16, torch.float16)
                    or not w.is_contiguous()
                    or w.shape[1] < MIN_IN_FEATURES
                ):
                    skipped_other += 1
                    continue
                idx = w.device.index or 0
                if idx not in arch_by_dev:
                    arch = _arch(w.device)
                    arch_by_dev[idx] = (arch, pitch(arch))
                arch, p = arch_by_dev[idx]
                if p <= 0:
                    skipped_other += 1
                    continue
                if guard is None:
                    guard = shapes is not None and (not _env() or _table_family(family, arch))
                N, K = w.shape
                if guard and ("*" in shapes or (int(N), int(K)) in shapes):
                    skipped_table += 1
                    continue
                # The weight is swapped only after the full copy, so a failure here leaves it untouched.
                try:
                    buf = torch.empty((N, K + p), dtype = w.dtype, device = w.device)
                    buf[:, :K].copy_(w.data)
                except torch.cuda.OutOfMemoryError as exc:  # torch.OutOfMemoryError on 2.5+
                    oom = exc
                    break
                m.weight.data = buf[:, :K]
                n += 1
                STATS["bytes"] += N * p * w.element_size()
    finally:
        # Also on error / OOM: replaced blocks stay reserved otherwise (system RAM on an APU).
        if n or oom is not None:
            _release_old_storage(torch)
        STATS["padded"] += n
        STATS["skipped_table"] += skipped_table
        STATS["skipped_other"] += skipped_other
    if oom is not None:
        # Padding is optional: stop, keep every weight valid (padded or original) and let the render go on.
        STATS["oom"] += 1
        if logger is not None:
            logger.warning(
                "diffusion.rocm_weight_pitch: out of memory after re-storing %d Linear weights; the rest stay "
                "contiguous: %s",
                n,
                oom,
            )
    if logger is not None and (n or skipped_table):
        logger.info(
            "diffusion.rocm_weight_pitch: %d Linear weights re-stored with row pitch K + %d; %d kept for the "
            "TunableOp table (family %s)",
            n,
            p,
            skipped_table,
            family,
        )
    return n
