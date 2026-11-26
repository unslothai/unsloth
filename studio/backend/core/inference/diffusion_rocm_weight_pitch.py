# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11xx: store the denoiser's 16-bit ``nn.Linear`` weights with a padded row pitch (K + P elements, a view
of an [N, K + P] buffer) so hipBLASLt reads them off the power-of-two strides that alias on Strix Halo's memory
channels. Same GEMM with another leading dimension: torch.equal on every measured GEMM. AMD CI Linux gfx1151,
untuned hipBLASLt, P = 64: block linear sums 1.09x Wan 5B, 1.26x Krea-2, 1.31x Qwen-Image-2.1, 1.30x LTX-2.5, 1.10x
FLUX.3 klein; real Krea-2 block 1.21x; Wan 2.2 5B 832x480x49 denoise 42.2 -> 38.8 s. Weight memory +P / K (0.4-2%).

A TunableOp row is keyed on the leading dimension, so a padded weight misses the shipped table (0.90-0.94x on Qwen /
klein GEMMs). With a table active, a family the table was tuned for (``TABLE_FAMILIES``, or any family while the user
tunes or has a local tuned cache) keeps every weight whose (N, K) has a row in the loaded results contiguous; other
families' token counts never reach those rows, so they are padded as without a table.

``UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH``: unset = on (P = 64) for ``DEFAULT_ON_ARCHES``; ``0`` off; ``<P>`` (a
multiple of 8) forces it with pitch P on any gfx11xx card. Counts land in ``STATS``."""

from __future__ import annotations

import os
from typing import Any, Optional

PITCH_ENV = "UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH"
DEFAULT_PITCH = 64
ARCH_PREFIX = "gfx11"
# Measured default-on archs; other gfx11xx cards take it only through the env var.
DEFAULT_ON_ARCHES: frozenset = frozenset({"gfx1151"})
MIN_IN_FEATURES = 1024
# Families whose denoiser GEMMs the shipped rocm_gemm_tables/*.csv were tuned on (the table name lists them).
TABLE_FAMILIES = frozenset({"qwen-image-2.1", "flux.3-klein", "flux.3-klein-base", "flux.2-klein"})
TABLE_STEM_TOKENS = {
    "qwenimage21": ("qwen-image-2.1",),
    "flux3klein": ("flux.3-klein", "flux.3-klein-base"),
    "flux2klein4b": ("flux.2-klein",),
}
STATS = {"padded": 0, "bytes": 0, "skipped_table": 0, "skipped_other": 0, "calls": 0}
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


def table_shapes() -> Optional[set]:
    """(N, K) of every TN row TunableOp currently holds; None when no Studio table is active."""
    try:
        from . import diffusion_rocm_tunableop as tunableop

        if not tunableop.is_active():
            return None
        import torch

        rows = list(torch.cuda.tunable.get_results() or ())
    except Exception:  # noqa: BLE001 - active but unreadable: treat every weight as covered
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
    """A family whose GEMMs a loaded table may hold: a shipped-table family, or any while tuned rows come from the
    user's own renders (tune mode, or a local cache from an earlier tune)."""
    if str(family or "").strip().lower() in TABLE_FAMILIES:
        return True
    try:
        from . import diffusion_rocm_tunableop as tunableop
        if tunableop.tuning_requested():
            return True
        return bool(arch) and tunableop.results_file(arch).exists()
    except Exception:  # noqa: BLE001
        return True


def install(
    module: Any,
    logger: Any = None,
    family: Optional[str] = None,
) -> int:
    """Speed optims run before placement, so the re-store waits for the module's first forward (weights on the card
    by then); returns 1 when the hook is armed."""
    if module is None or _env() in ("0", "off", "false", "no"):
        return 0
    if not callable(getattr(module, "register_forward_pre_hook", None)):
        return 0
    if module.__dict__.get(_HOOK_ATTR) is not None:
        return 0
    if pitch(_arch(0)) <= 0:
        return 0

    def hook(mod, args):
        handle = mod.__dict__.get(_HOOK_ATTR)
        if handle:
            handle.remove()
        mod.__dict__[_HOOK_ATTR] = False
        try:
            pad_weights(mod, logger, family = family)
        except Exception as exc:  # noqa: BLE001 - contiguous weights stay
            if logger is not None:
                logger.warning("diffusion.rocm_weight_pitch: skipped: %s", exc)

    module.__dict__[_HOOK_ATTR] = module.register_forward_pre_hook(hook)
    return 1


def pad_weights(
    module: Any,
    logger: Any = None,
    family: Optional[str] = None,
) -> int:
    """Re-store every eligible plain ``nn.Linear`` weight of ``module`` with the padded pitch. Returns the count."""
    if module is None:
        return 0
    import torch

    if torch.compiler.is_compiling():
        return 0
    STATS["calls"] += 1
    shapes = table_shapes()
    guard: Optional[bool] = None
    n = skipped_table = skipped_other = 0
    arch_by_dev: dict = {}
    p = 0
    with torch.no_grad():
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
                guard = shapes is not None and _table_family(family, arch)
            N, K = w.shape
            if guard and ("*" in shapes or (int(N), int(K)) in shapes):
                skipped_table += 1
                continue
            buf = torch.empty((N, K + p), dtype = w.dtype, device = w.device)
            buf[:, :K].copy_(w.data)
            m.weight.data = buf[:, :K]
            n += 1
            STATS["bytes"] += N * p * w.element_size()
    STATS["padded"] += n
    STATS["skipped_table"] += skipped_table
    STATS["skipped_other"] += skipped_other
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
