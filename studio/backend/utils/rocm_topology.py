# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ROCm unified-memory verdict, as a leaf module with no package baggage.

It lives here rather than in ``core.training.worker`` because the chat backend
needs it and the training worker is not reachable from a GGUF-only install.
``worker`` imports ``utils.wheel_utils``, ``core.training.dataset_bounds`` and
several other training-stack modules at import time; a ``--no-torch`` install has
none of them. ``LlamaCppBackend._torch_unified_memory_classification_known``
imported the classifier from there INSIDE its ``try``, and the whole method is
wrapped in ``except Exception: return False``, so on such an install every ROCm
device silently read "unclassifiable", the device was treated as shared, and the
host-pinned VRAM discount was never applied at all.

MEASURED on the AMD CI Windows runners, which install with ``--no-torch``: eleven
``test_a_known_discrete_rocm_arch_is_proved_discrete`` cases failed there while
passing on the Linux runner, and blocking this one import locally reproduces it
exactly (15 passed becomes 12 failed). The direction was safe -- an unclassified
device is charged the old, larger budget -- but the optimisation was inert in
precisely the configuration llama.cpp users run.

Same reasoning as ``utils/gguf_archs.py``: one definition, importable without
dragging a package tree behind it. ``worker`` re-exports it so existing callers
and the training-side tests are unaffected.
"""

from __future__ import annotations

from typing import Any


def _rocm_classify_unified_memory(props: Any) -> tuple[str, bool]:
    """Classify a ROCm device as unified-memory (APU) or discrete.

    Returns ``(gcn_arch, is_unified)``:
    - ``gcn_arch``: canonical arch string (e.g. ``"gfx1151"``) when a known
      attribute is present, else ``""``.
    - ``is_unified``: ``True`` for AMD APUs with a shared GPU/system-RAM pool
      (gfx1150 Strix Point, gfx1151 Strix Halo, gfx1152 Krackan Point) — these
      need a lower ``set_per_process_memory_fraction`` cap to leave OS headroom.

    Classification priority:
    1. ``props.is_integrated`` truthy (hipDeviceProp_t.integrated -- the
       driver's own unified-memory answer; covers APUs beyond the hardcoded
       arch set, e.g. gfx1103 Phoenix iGPUs). Only ever upgrades to unified.
    2. ``gcnArchName`` / variant spellings (stable, naming-independent).
    3. Device-name substring match (last resort when all arch attrs absent;
       AMD SDK / Radeon wheels may not populate them):
         - gfx1150 Strix Point: ``Radeon 890M``, ``Radeon 880M``
         - gfx1151 Strix Halo / Gorgon Halo:  ``Radeon 8065S`` (Ryzen AI
                                Max+ 495), ``Radeon 8060S`` (Ryzen AI MAX+
                                395), ``Radeon 8050S`` (cut-down SKU)
         - gfx1152 Krackan Point: ``Radeon 860M``, ``Radeon 840M``
    """
    gcn_arch = ""
    for _attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        _v = (getattr(props, _attr, "") or "").split(":")[0].strip()
        if _v:
            gcn_arch = _v
            break

    # Driver's own answer first: hipDeviceProp_t.integrated (props.is_integrated, the same
    # gate PR #5988's UMA safetensors fast-load uses). Strictly additive -- only a truthy
    # value upgrades to unified, so a wheel that omits the field can't downgrade the known
    # APU set. Covers unified APUs outside the hardcoded arches (gfx1103 Phoenix, future).
    if getattr(props, "is_integrated", 0):
        return gcn_arch, True

    if gcn_arch:
        # gfx1152 is Krackan Point: same shared GPU/system-RAM pool as gfx1150/gfx1151.
        # Case-folded: the attribute is lowercase in practice but is not guaranteed.
        return gcn_arch, gcn_arch.lower() in {"gfx1150", "gfx1151", "gfx1152"}

    # Arch attrs absent -- fall back to device-name matching. Only reached under _hw.IS_ROCM,
    # so the NVIDIA GeForce 840M cannot collide with the Krackan markers.
    dev_lower = (getattr(props, "name", "") or "").lower()
    is_unified = (
        "890m" in dev_lower
        or "880m" in dev_lower
        or "8065s" in dev_lower
        or "8060s" in dev_lower
        or "8050s" in dev_lower
        or "860m" in dev_lower
        or "840m" in dev_lower
    )
    return gcn_arch, is_unified
