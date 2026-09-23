# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Knowing when a diffusion pipeline has left its denoise loop and entered the decoder.

Shared by video.py and diffusion.py: both run the decode INSIDE the pipeline call, so neither can
report it from the outside.
"""

from __future__ import annotations

import contextlib
from typing import Any

# The decoders a pipeline may run after its denoise loop, in the order the modular MiniMax-H3
# workflow runs them. Wrapping the bound method is the one hook every family shares.
DECODE_ATTRS = ("vae", "audio_vae")


@contextlib.contextmanager
def decode_phase(pipe: Any, on_decode: Any, attrs: "tuple[str, ...]" = DECODE_ATTRS):
    """Flip the reported phase to "decode" the instant the decoder is entered.

    HunyuanVideo-1.5, Wan and MiniMax-H3's modular workflow all run the decode INSIDE the pipeline
    call with nothing between the denoise loop and it, so a phase set only after ``pipe()`` returns
    reports the whole decode as the last denoise step. On H3 that decode plus its post-processing
    is ~3.9 s of the render, and the VAE decode is the memory peak. The image path is the same
    shape: the bar reaches steps/steps and then sits through a decode nothing reports.

    Note what this hook is and is not: it is the one HOST position that knows the denoise loop is
    over, and nothing more. On a family where the host runs ahead of the device, the denoise
    kernels can still be queued when it fires, so a caller that also moves the STEP count must
    treat it as "mark the boundary", not as "the denoise finished" -- see
    ``_CompletedStepTicker.mark_boundary``. A caller that only changes the label has no such
    problem, because the count is already at total by then.

    ``on_decode`` fires at most once per context and must not raise. Every wrapper installed here
    is removed again, including when the decode raises -- and restored to whatever was there,
    since the speed layer may have already put a compiled decode in the instance ``__dict__``.
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
                    # Nothing was shadowing the class method, so leave nothing behind -- a bound
                    # method parked in a module's __dict__ is a reference cycle back to the module.
                    del owner.decode
            except Exception:  # noqa: BLE001 -- cleanup is best-effort
                pass
