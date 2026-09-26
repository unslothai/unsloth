# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect when a diffusion pipeline leaves its denoise loop and enters the decoder."""

from __future__ import annotations

import contextlib
from typing import Any

# In the order MiniMax-H3's modular workflow runs them.
DECODE_ATTRS = ("vae", "audio_vae")


@contextlib.contextmanager
def decode_phase(
    pipe: Any,
    on_decode: Any,
    attrs: "tuple[str, ...]" = DECODE_ATTRS,
):
    """Call ``on_decode`` (at most once, must not raise) when a decoder is first entered.

    The decode runs inside ``pipe()`` after the last step callback, so nothing outside can see it.
    This is a HOST position: denoise kernels may still be queued, so a caller moving the step count
    must treat it as a boundary mark (``_CompletedStepTicker.mark_boundary``). Wrappers are removed
    on every exit, restoring any compiled decode the speed layer put in the instance ``__dict__``.
    """
    fired = {"done": False}
    restore: list = []

    def _wrap(original: Any):
        def _decode(*args: Any, **kwargs: Any) -> Any:
            if not fired["done"]:
                fired["done"] = True
                on_decode()
            return original(*args, **kwargs)

        return _decode

    for name in attrs:
        owner = getattr(pipe, name, None)
        original = getattr(owner, "decode", None) if owner is not None else None
        if not callable(original):
            continue
        had_own = "decode" in getattr(owner, "__dict__", {})
        try:
            owner.decode = _wrap(original)
        except Exception:  # noqa: BLE001 -- a decoder that refuses assignment goes unreported
            continue
        restore.append((owner, original, had_own))
    try:
        yield
    finally:
        for owner, original, had_own in restore:
            try:
                if had_own:
                    owner.decode = original
                else:
                    # A parked bound method would be a reference cycle back to the module.
                    del owner.decode
            except Exception:  # noqa: BLE001 -- cleanup is best-effort
                pass
