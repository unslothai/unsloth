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

# Resolved ``unsloth_zoo.temporary_patches.utils`` helpers, memoised per process (None = tried and unavailable): resolution can import ``unsloth``, far too heavy to repeat per call.
_HELPERS: Optional[dict] = None


def _retry_could_help(exc: BaseException) -> bool:
    """Whether importing ``unsloth`` could turn ``exc`` into a successful retry.

    See ``_helpers`` for why each condition is here; the short version is that the import is
    expensive and must not run where it cannot succeed."""
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
        except Exception:  # noqa: BLE001 — an unprobeable device is not one unsloth can use
            return False
    try:
        return importlib.util.find_spec("unsloth") is not None
    except Exception:  # noqa: BLE001 — an unimportable package cannot set the sentinel either
        return False


def _helpers() -> Optional[dict]:
    """``{"patch": patch_function, "restore": restore_original}``, or None when unavailable.

    ``unsloth_zoo.__init__`` refuses to import unless ``UNSLOTH_IS_PRESENT`` is in the environment,
    and that is set by ``unsloth`` itself. The Studio server imports ``unsloth`` at boot so this
    always resolved there, but ANY process that reaches the patch backend first -- the test suite,
    a worker subprocess -- got an ImportError and silently ran unpatched (every install returning
    False). So on failure, import ``unsloth`` and retry once, which is also the import order Unsloth
    documents.

    The retry is gated, because it is not free: importing ``unsloth`` costs ~940 MB of RSS measured
    in a process that had not already loaded torch, and on a host unsloth does not support it pays
    that and fails anyway. Ungated it took two cross-platform CI runners down -- a Linux job that had
    generated fine at ~900 s died 19 s in with SIGTERM and every ``if: always()`` step skipped (the
    runner torn down, not a step failing), and a 7 GB macOS runner lost the server 26 s into a load.
    So it runs only when it can actually succeed:

      * ``torch`` is already imported -- true of the server and of anything patching a real module,
        and the condition that stops the retry from being what loads torch,
      * an accelerator ``unsloth`` supports is present (CUDA/ROCm or XPU; ``UNSLOTH_ALLOW_CPU``
        overrides). MPS and plain CPU are not, and there the import raises after paying,
      * ``unsloth`` is installed but not yet imported (if it were, the sentinel would be set and the
        first attempt would have worked),
      * and the first failure was the ImportError that guard raises."""
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
        except Exception as exc:  # noqa: BLE001 — no unsloth_zoo / no-GPU host -> skip the patch
            if attempt or not _retry_could_help(exc):
                break
            try:
                import unsloth  # noqa: F401 — sets UNSLOTH_IS_PRESENT for the retry
            except Exception:  # noqa: BLE001 — not installed / no accelerator: give up quietly
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
    """Patch ``target.attr -> new_fn`` via ``patch_function`` (original stashed for
    ``revert_patch``). Returns True iff applied; False (never raises) if unsloth_zoo is
    unavailable or ``can_safely_patch`` rejects it.

    ``force=True`` skips the check -- only when new_fn is the SAME function transformed (e.g. a
    ``torch.compile`` wrapper), where a fingerprint mismatch is expected."""
    patch_function = _helper("patch")
    if patch_function is None:
        return False
    try:
        return bool(patch_function(target, attr, new_fn, match_level = match_level, force = force))
    except Exception:  # noqa: BLE001 — best-effort; leave the original in place
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
