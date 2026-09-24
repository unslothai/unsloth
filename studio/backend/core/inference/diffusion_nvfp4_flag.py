# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one switch for NVFP4 in Studio's image and video paths.

Off by default until the hosted ``unsloth/*-NVFP4`` checkpoints are public. While it is off NVFP4
is unavailable in general, not only the hosted checkpoints: ``auto`` never picks it for the
transformer, an explicit ``transformer_quant`` or ``text_encoder_quant`` of ``nvfp4`` is refused,
no ``*-NVFP4`` repo is resolved, planned, sized or downloaded, no flashinfer install or preflight
runs for it, and the capability payloads do not advertise it. ``UNSLOTH_NVFP4_DIFFUSION=1``
restores the full NVFP4 behaviour.

Torch-free and read at CALL time (never cached), so a test can flip it with ``monkeypatch.setenv``
and a restarted backend picks up a new value without an import-order dependency.

The offline builder scripts (``scripts/build_prequant_checkpoint.py`` and the gate scripts) do not
consult this: they are developer tools for producing the checkpoints in the first place.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

NVFP4_DIFFUSION_ENV = "UNSLOTH_NVFP4_DIFFUSION"

# The default when the variable is unset. Flipping NVFP4 on for everyone, once the hosted repos are
# public, is this one line.
NVFP4_DIFFUSION_DEFAULT = False

NVFP4_SCHEME = "nvfp4"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def nvfp4_diffusion_enabled() -> bool:
    """Whether NVFP4 may be used, planned or advertised for image and video generation.

    ``1`` / ``true`` / ``yes`` / ``on`` enable it and ``0`` / ``false`` / ``no`` / ``off`` disable
    it. Unset, empty or unrecognised values take ``NVFP4_DIFFUSION_DEFAULT``, which is off."""
    raw = os.environ.get(NVFP4_DIFFUSION_ENV)
    if raw is None:
        return NVFP4_DIFFUSION_DEFAULT
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return NVFP4_DIFFUSION_DEFAULT


def is_nvfp4(value: Optional[str]) -> bool:
    """Whether ``value`` names the NVFP4 scheme, in any case or spacing."""
    if value is None:
        return False
    return str(value).strip().lower().replace("-", "_") == NVFP4_SCHEME


def nvfp4_blocked(scheme: Optional[str]) -> bool:
    """True when ``scheme`` is NVFP4 and the switch is off: the one test every gate asks."""
    return is_nvfp4(scheme) and not nvfp4_diffusion_enabled()


def nvfp4_repo_blocked(repo_id: Optional[str]) -> bool:
    """True for a hosted ``*-NVFP4`` repo while the switch is off. A backstop for the helpers that
    talk to the Hub, behind the scheme gates that normally stop the lookup long before."""
    if nvfp4_diffusion_enabled() or not repo_id:
        return False
    return str(repo_id).strip().rstrip("/").lower().endswith("-nvfp4")


def without_nvfp4(schemes: Iterable[str]) -> tuple[str, ...]:
    """``schemes`` in order, minus NVFP4 while the switch is off."""
    if nvfp4_diffusion_enabled():
        return tuple(schemes)
    return tuple(scheme for scheme in schemes if not is_nvfp4(scheme))


def nvfp4_disabled_message(control: str = "transformer_quant") -> str:
    """The refusal a request naming NVFP4 gets while the switch is off."""
    return (
        f"{control}='nvfp4' could not be used: NVFP4 is disabled in this build. Pick another "
        f"precision, or set {NVFP4_DIFFUSION_ENV}=1 on the Studio backend to enable it."
    )


def refuse_disabled_nvfp4(
    *,
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` (the routes' 400) when a request names NVFP4 while the switch is off.

    Runs ahead of every other precision check, including the opt-in silent fallback, so a disabled
    scheme is refused outright rather than quietly swapped for another one."""
    if nvfp4_diffusion_enabled():
        return
    if is_nvfp4(transformer_quant):
        raise ValueError(nvfp4_disabled_message("transformer_quant"))
    if is_nvfp4(text_encoder_quant):
        raise ValueError(nvfp4_disabled_message("text_encoder_quant"))
