# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect when a diffusion pipeline enters its denoise loop, and when it leaves it for the decoder."""

from __future__ import annotations

import contextlib
from typing import Any

DECODE_ATTRS = ("vae", "audio_vae")


@contextlib.contextmanager
def decode_phase(
    pipe: Any,
    on_decode: Any,
    attrs: "tuple[str, ...]" = DECODE_ATTRS,
):
    """Fires on_decode once at decoder entry, a host position that may trail queued denoise kernels."""
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


@contextlib.contextmanager
def denoise_phase(pipe: Any, on_denoise: Any):
    """Call ``on_denoise`` once when the loop is entered: every diffusers pipeline opens
    ``self.progress_bar`` right after the prompt encode and latent setup. A HOST position."""
    original = getattr(pipe, "progress_bar", None)
    if not callable(original):
        yield
        return
    fired = {"done": False}
    had_own = "progress_bar" in getattr(pipe, "__dict__", {})

    def _progress_bar(*args: Any, **kwargs: Any) -> Any:
        if not fired["done"]:
            fired["done"] = True
            try:
                on_denoise()
            except Exception:  # noqa: BLE001 -- a phase label must never fail a render
                pass
        return original(*args, **kwargs)

    try:
        pipe.progress_bar = _progress_bar
    except Exception:  # noqa: BLE001 -- a pipe that refuses assignment keeps the step-callback fallback
        yield
        return
    try:
        yield
    finally:
        try:
            if had_own:
                pipe.progress_bar = original
            else:
                del pipe.progress_bar
        except Exception:  # noqa: BLE001 -- cleanup is best-effort
            pass
