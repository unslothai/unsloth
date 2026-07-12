# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One vetted path for every diffusion monkey-patch.

Thin wrappers over ``unsloth_zoo.temporary_patches.utils`` ``patch_function`` /
``restore_original`` so all runtime patching (eager fusions, GGUF accelerators, per-arch rewrites)
goes through the SAME fingerprint-checked, reversible mechanism:

* ``patch_function`` stashes the live original and, unless ``force=True``, runs
  ``can_safely_patch`` (a param-name/kind/required fingerprint; ``relaxed`` ignores annotation
  drift but rejects a real signature change) -- so a changed forward is left unpatched, not
  miscompiled.
* ``restore_original`` restores the original -- exact, idempotent uninstall.

``unsloth_zoo`` is imported LAZILY per call: it runs GPU detection at import and raises without an
accelerator (``UNSLOTH_ALLOW_CPU=1`` bypasses), and the backend must stay importable on CPU-only
hosts. If the import fails, patching is a best-effort no-op (stock forward runs, correctness kept).
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
    """Patch ``target.attr -> new_fn`` via ``patch_function`` (original stashed for
    ``revert_patch``). Returns True iff applied; False (never raises) if unsloth_zoo is
    unavailable or ``can_safely_patch`` rejects it.

    ``force=True`` skips the check -- only when new_fn is the SAME function transformed (e.g. a
    ``torch.compile`` wrapper), where a fingerprint mismatch is expected."""
    try:
        from unsloth_zoo.temporary_patches.utils import patch_function
    except Exception:  # noqa: BLE001 — no unsloth_zoo / no-GPU host -> optimisation skipped
        return False
    try:
        return bool(patch_function(target, attr, new_fn, match_level = match_level, force = force))
    except Exception:  # noqa: BLE001 — best-effort; leave the original in place
        return False


def revert_patch(target: Any, attr: str) -> bool:
    """Restore ``target.attr`` from the original stashed by ``apply_patch``. Idempotent; returns
    False (never raises) if nothing is stored or unsloth_zoo is unavailable."""
    try:
        from unsloth_zoo.temporary_patches.utils import restore_original
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(restore_original(target, attr))
    except Exception:  # noqa: BLE001
        return False
