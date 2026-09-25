# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one NVFP4 switch for Studio image/video; read at call time, never cached, so tests can monkeypatch it."""

from __future__ import annotations

import os
from typing import Iterable, Optional

NVFP4_DIFFUSION_ENV = "UNSLOTH_NVFP4_DIFFUSION"

# Flip to True once the hosted *-NVFP4 repos are public.
NVFP4_DIFFUSION_DEFAULT = False

NVFP4_SCHEME = "nvfp4"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def nvfp4_diffusion_enabled() -> bool:
    """Whether NVFP4 may be used; unrecognised values take ``NVFP4_DIFFUSION_DEFAULT``."""
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
    """True for a hosted ``*-NVFP4`` repo while the switch is off (Hub-helper backstop)."""
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
    *, transformer_quant: Optional[str] = None, text_encoder_quant: Optional[str] = None
) -> None:
    """Raise ``ValueError`` (400) for NVFP4 while off, before any silent fallback could swap it."""
    if nvfp4_diffusion_enabled():
        return
    if is_nvfp4(transformer_quant):
        raise ValueError(nvfp4_disabled_message("transformer_quant"))
    if is_nvfp4(text_encoder_quant):
        raise ValueError(nvfp4_disabled_message("text_encoder_quant"))


def nvfp4_checkpoint_disabled_message() -> str:
    """The refusal a load or plan of an NVFP4 checkpoint gets while the switch is off."""
    return (
        "model_path could not be used: it is an NVFP4 checkpoint and NVFP4 is disabled in this "
        f"build. Pick another model, or set {NVFP4_DIFFUSION_ENV}=1 on the Studio backend to "
        "enable it."
    )


def refuse_disabled_nvfp4_checkpoint(*model_paths: Optional[str]) -> None:
    """Raise ``ValueError`` (400) for an NVFP4 checkpoint model path while off; it bypasses the scheme gates."""
    if nvfp4_diffusion_enabled():
        return
    from .diffusion_prequant import declares_nvfp4_checkpoint

    for path in model_paths:
        if path and declares_nvfp4_checkpoint(path):
            raise ValueError(nvfp4_checkpoint_disabled_message())
