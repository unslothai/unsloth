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

from typing import Any, Callable, Optional

# Memoised: resolution can import unsloth, too heavy per call.
_HELPERS: Optional[dict] = None


def _retry_could_help(exc: BaseException) -> bool:
    """Whether importing unsloth could fix exc; the import is costly, so only try it where it can
    succeed."""
    import importlib.util
    import os
    import sys

    if not isinstance(exc, ImportError) or "unsloth" in sys.modules:
        return False
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    if os.environ.get("UNSLOTH_ALLOW_CPU", "").strip().lower() not in ("1", "true", "yes", "on"):
        try:
            xpu = getattr(torch, "xpu", None)
            if not (torch.cuda.is_available() or (xpu is not None and xpu.is_available())):
                return False
        except Exception:  # noqa: BLE001 - an unprobeable device is not one unsloth can use
            return False
    try:
        return importlib.util.find_spec("unsloth") is not None
    except Exception:  # noqa: BLE001 - an unimportable package cannot set the sentinel either
        return False


def _helpers() -> Optional[dict]:
    """Retries the unsloth import once, only with torch loaded and a supported accelerator present."""
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS or None

    def _load() -> dict:
        from unsloth_zoo.temporary_patches.utils import patch_function, restore_original
        return {"patch": patch_function, "restore": restore_original}

    for attempt in (0, 1):
        try:
            _HELPERS = _load()
            return _HELPERS
        except Exception as exc:  # noqa: BLE001 - no unsloth_zoo / no-GPU host -> skip the patch
            if attempt or not _retry_could_help(exc):
                break
            try:
                import unsloth  # noqa: F401 - sets UNSLOTH_IS_PRESENT for the retry
            except Exception:  # noqa: BLE001 - not installed / no accelerator: give up quietly
                break
    _HELPERS = {}
    return None


def _helper(name: str) -> Optional[Callable]:
    helpers = _helpers()
    return helpers.get(name) if helpers else None


def apply_patch(
    target: Any,
    attr: str,
    new_fn: Any,
    *,
    match_level: str = "relaxed",
    force: bool = False,
) -> bool:
    """Returns False instead of raising when unsloth_zoo is missing; force skips the fingerprint check."""
    patch_function = _helper("patch")
    if patch_function is None:
        return False
    try:
        return bool(patch_function(target, attr, new_fn, match_level = match_level, force = force))
    except Exception:  # noqa: BLE001 - best-effort; leave the original in place
        return False


def revert_patch(target: Any, attr: str) -> bool:
    """Restore ``target.attr`` from the original stashed by ``apply_patch``. Idempotent; returns
    False (never raises) if nothing is stored or unsloth_zoo is unavailable."""
    restore_original = _helper("restore")
    if restore_original is None:
        return False
    try:
        return bool(restore_original(target, attr))
    except Exception:  # noqa: BLE001
        return False
