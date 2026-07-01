# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One vetted path for every diffusion monkey-patch.

Thin wrappers over ``unsloth_zoo.temporary_patches.utils`` ``patch_function`` /
``restore_original`` so all of the backend's runtime patching (eager fusions, GGUF
accelerators, per-arch block rewrites) goes through the SAME fingerprint-checked,
reversible mechanism:

* ``patch_function`` stores the live original under a unique attribute and, unless
  ``force=True``, runs ``can_safely_patch`` (a parameter-name/kind/required fingerprint;
  ``match_level="relaxed"`` ignores type-annotation drift but still rejects a real
  signature change) -- so a future diffusers/transformers that renamed or reordered a
  forward's parameters is simply left unpatched instead of silently miscompiled.
* ``restore_original`` puts the stored original back -- exact, idempotent uninstall.

``unsloth_zoo`` is imported LAZILY inside each call, never at module import: it runs GPU
detection at import time and raises without an accelerator (set ``UNSLOTH_ALLOW_CPU=1`` to
bypass, as the test conftest does), and the diffusion backend must stay importable on a
CPU-only host. This mirrors the backend's existing deferred unsloth_zoo imports (see
``core/training/trainer.py``, ``core/export/export.py``). If the import fails (unsloth_zoo
absent, or a no-GPU host without the bypass), patching is a best-effort no-op: the stock
forward runs, correctness is preserved, only the optimisation is skipped.
"""

from __future__ import annotations

from typing import Any


def apply_patch(
    target: Any,
    attr: str,
    new_fn: Any,
    *,
    match_level: str = "relaxed",
    force: bool = False,
) -> bool:
    """Patch ``target.attr -> new_fn`` via ``unsloth_zoo`` ``patch_function`` (the original is
    stashed for ``revert_patch``). Returns True iff the swap was applied. Returns False
    (never raises) if unsloth_zoo is unavailable or ``can_safely_patch`` rejects the swap.

    ``force=True`` skips the safety check -- use it only when the new callable is the SAME
    function transformed (e.g. its ``torch.compile`` wrapper), where a fingerprint mismatch
    is expected and benign."""
    try:
        from unsloth_zoo.temporary_patches.utils import patch_function
    except Exception:  # noqa: BLE001 — no unsloth_zoo / no-GPU host -> optimisation skipped
        return False
    try:
        return bool(patch_function(target, attr, new_fn, match_level = match_level, force = force))
    except Exception:  # noqa: BLE001 — best-effort; leave the original in place
        return False


def revert_patch(target: Any, attr: str) -> bool:
    """Restore ``target.attr`` from the original stashed by ``apply_patch``. Idempotent and
    best-effort: returns False (never raises) if there is nothing stored or unsloth_zoo is
    unavailable."""
    try:
        from unsloth_zoo.temporary_patches.utils import restore_original
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(restore_original(target, attr))
    except Exception:  # noqa: BLE001
        return False
