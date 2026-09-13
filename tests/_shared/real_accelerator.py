# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Does this machine have an accelerator, asked so that the CUDA spoof cannot answer.

`torch.cuda.is_available()` is not a safe thing for a test to gate hardware on in
this repo, because two harnesses deliberately make it lie:

- ``tests/conftest.py`` patches it to True for the window in which it pre-loads
  ``unsloth_zoo.device_type``, so ``get_device_type()``'s ``@cache`` captures
  "cuda" on a GPU-less runner. That window is closed in a ``finally``.
- ``tests/_zoo_aggressive_cuda_spoof.py`` patches it to True and leaves it that
  way. It is applied at module import by every file under ``tests/version_compat``
  and ``tests/vllm_compat``, and it is not undoable by design: the whole point is
  that unsloth's import chain sees a card for the rest of the process.

The second one crosses module boundaries. pytest imports every selected test
module before running anything, and a ``@pytest.mark.skipif`` is evaluated at
import, so one spoofing module in the session flips the guard for every other
module collected after it. On a CPU-only box the guarded tests then un-skip and
die inside torch with

    RuntimeError: Cannot access accelerator device when none is available.

which reads like a torch or a driver problem and is neither. Observed on
``pytest tests/utils/test_packing.py tests/version_compat/...`` in one session:
2 failed, and the same two pass when either file is run alone.

CI does not hit this today only because the three jobs that touch these
directories each ``--ignore`` the others. That is an accident of the current
ignore lists, not a property anything checks.

So record the answer once, before anything has had the chance to spoof, and hand
that out afterwards. ``tests/conftest.py`` primes the cache at its own import,
which is ahead of every test module in the session.
"""

from __future__ import annotations

_REAL_ACCELERATOR: bool | None = None


def _probe() -> bool:
    try:
        import torch
    except Exception:
        return False
    for probe in (
        lambda: hasattr(torch, "cuda") and torch.cuda.is_available(),
        lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
        lambda: hasattr(torch, "accelerator") and torch.accelerator.is_available(),
    ):
        try:
            if probe():
                return True
        except Exception:
            # A probe that raises is not an accelerator. torch.xpu on a build without
            # XPU support, and torch.accelerator on torch < 2.6, both do this.
            pass
    return False


def has_real_accelerator() -> bool:
    """True if this machine really has a CUDA / XPU / torch.accelerator device.

    Cached on the first call. ``tests/conftest.py`` makes that first call before
    any spoof runs, so every later caller gets the pre-spoof answer no matter what
    has patched ``torch.cuda`` since.
    """
    global _REAL_ACCELERATOR
    if _REAL_ACCELERATOR is None:
        _REAL_ACCELERATOR = _probe()
    return _REAL_ACCELERATOR
