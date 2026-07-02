# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selects the diffusion engine (diffusers vs native sd.cpp) for the live route.

The image routes drive one engine at a time. On a CUDA/ROCm/XPU GPU that engine is
the diffusers ``DiffusionBackend`` (the default, and the only path with the torchao
fast-quant / compile stack). With no usable GPU (CPU, or Apple MPS when explicitly
enabled) it is the native ``SdCppDiffusionBackend``, which is faster and far lighter
on RAM there. The choice is made once at load time and remembered, so ``generate`` /
``unload`` / ``status`` / progress all act on the same engine the load committed to.

Selection is centralised here and built on the existing pure ``select_diffusion_engine``
decision; this module adds the policy around it (env opt-out, MPS gating, per-family
native-asset support, lazy binary availability) and records why a fallback happened.

Env knobs (one canonical interpretation each):
  UNSLOTH_DIFFUSION_ENGINE=auto|diffusers|sd_cpp   force an engine (auto = decide)
  UNSLOTH_DIFFUSION_SD_CPP=auto|0|1                 enable/disable the native route
  UNSLOTH_DIFFUSION_SD_CPP_MPS=0|1                  allow native on Apple MPS (default off)
  UNSLOTH_DIFFUSION_SD_CPP_INSTALL=auto|0|1         allow lazy binary install (in sd_cpp_backend)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import DiffusionFamily, family_sd_cpp_supported
from core.inference.sd_cpp_backend import (
    _install_allowed,
    _server_binary_runnable,
    ensure_sd_cpp_binary,
    ensure_sd_server_binary,
)
from core.inference.sd_cpp_engine import (
    ENGINE_DIFFUSERS,
    ENGINE_SD_CPP,
    SdCppEngine,
    select_diffusion_engine,
)
from loggers import get_logger

logger = get_logger(__name__)

_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
_ENABLE_TOKENS = frozenset({"1", "on", "true", "yes"})

# Resolved device backend -> the prebuilt sd-cli accelerator to install. Only used
# when a *force-native* load on a GPU host has to install the binary: without this the
# installer defaults to "cpu" and downloads the plain build, so an sd_cpp generation
# forced on a ROCm/Intel box would silently run on CPU. Unknown/GPU-less backends fall
# back to "auto" (the CPU/Metal plain build), matching the installer's own default.
# (install_sd_cpp_prebuilt has no CUDA-Linux asset, so "cuda" only differs on Windows.)
_INSTALL_ACCELERATOR = {"rocm": "rocm", "cuda": "cuda", "xpu": "vulkan"}


def _install_accelerator_for(backend: str) -> str:
    return _INSTALL_ACCELERATOR.get(backend, "auto")


# The engine the current (or most recent) load committed to, and why a non-native
# choice was made. Mutated only under _lock during selection.
_lock = threading.Lock()
_active_engine_name: str = ENGINE_DIFFUSERS
_fallback_reason: Optional[str] = None


def _engine_config() -> tuple[str, str, bool]:
    forced = os.environ.get("UNSLOTH_DIFFUSION_ENGINE", "auto").strip().lower()
    sd_cpp = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP", "auto").strip().lower()
    mps = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_MPS", "0").strip().lower() in _ENABLE_TOKENS
    return forced, sd_cpp, mps


def get_active_diffusion_engine() -> Any:
    """The engine object the active selection points at (defaults to diffusers)."""
    if _active_engine_name == ENGINE_SD_CPP:
        from core.inference.sd_cpp_backend import get_sd_cpp_backend
        return get_sd_cpp_backend()
    from core.inference.diffusion import get_diffusion_backend

    return get_diffusion_backend()


def active_engine_name() -> str:
    return _active_engine_name


def _activate(name: str, reason: Optional[str]) -> Any:
    global _active_engine_name, _fallback_reason
    # Switching engines: unload the one being deactivated first, or its model
    # stays resident but unreachable (the arbiter evictor only targets the active
    # engine), leaking 10+ GB and defeating the chat<->diffusion handoff. The unload
    # itself (freeing 10+ GB / syncing CUDA) is slow, so resolve the engine under the
    # lock but run unload() OUTSIDE it -- holding _lock across a slow unload would
    # block every other selection caller.
    engine_to_unload = None
    old_name = None
    with _lock:
        if name != _active_engine_name:
            engine_to_unload = get_active_diffusion_engine()
            old_name = _active_engine_name
        _active_engine_name = name
        _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
    if engine_to_unload is not None:
        try:
            engine_to_unload.unload()
        except Exception as exc:  # noqa: BLE001 -- best-effort; never block the switch
            logger.warning("failed to unload previous engine %s: %s", old_name, exc)
    if name == ENGINE_SD_CPP:
        logger.info("diffusion engine: sd_cpp")
    else:
        logger.info("diffusion engine: diffusers (%s)", reason or "selected")
    return get_active_diffusion_engine()


def select_and_activate_engine(fam: DiffusionFamily, *, hf_token: Optional[str] = None) -> Any:
    """Pick + activate the engine for loading ``fam`` on this host; return the engine.

    Falls back to diffusers (recording a reason) whenever the native route is
    disabled, the device has a usable GPU, MPS is not enabled, the family has no
    native asset mapping, or the sd-cli binary is unavailable -- always BEFORE the
    slow load begins, so a fallback never strands a half-native load.
    """
    forced, sd_cpp_pref, mps_enabled = _engine_config()

    if forced == ENGINE_DIFFUSERS:
        return _activate(ENGINE_DIFFUSERS, "forced (UNSLOTH_DIFFUSION_ENGINE=diffusers)")

    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return _activate(ENGINE_DIFFUSERS, "native engine disabled (UNSLOTH_DIFFUSION_SD_CPP=0)")

    target = resolve_diffusion_device_target()
    backend = target.backend
    # Policy: CPU is always native-eligible; MPS only when explicitly enabled; a GPU
    # backend (cuda/rocm/xpu) never is, unless the user force-selects sd_cpp.
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    fam_ok = family_sd_cpp_supported(fam)

    binary = None
    server_binary = None
    if policy_eligible and fam_ok:
        # Probe the resident sd-server FIRST: the backend PREFERS it, and an sd-server-only
        # install (no sd-cli) must still route to native rather than silently falling back to
        # diffusers. Checking it before the sd-cli install also means a server-only host does
        # not pay an avoidable sd-cli download. Install the accelerator-matched build (ROCm /
        # Vulkan / CUDA) so a forced-native GPU load gets the GPU server, not the CPU one.
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed(),
            accelerator = _install_accelerator_for(backend),
        )
        if server_binary and not _server_binary_runnable(server_binary):
            logger.warning(
                "sd-server at %s is present but not runnable; not using it", server_binary
            )
            server_binary = None
        # sd-cli is the one-shot fallback. Always LOCATE an existing binary, but only
        # auto-INSTALL it when there is no usable server, so a server-only install is not
        # forced to also download a CLI it will never use. Probe runnability before
        # committing native: a present but non-runnable binary (wrong arch, missing shared
        # libs, no execute bit) would otherwise pass as available and only fail inside the
        # background load, instead of falling back to diffusers now.
        binary = ensure_sd_cpp_binary(
            allow_install = _install_allowed() and server_binary is None,
            accelerator = _install_accelerator_for(backend),
        )
        if binary and SdCppEngine(binary = binary).version() is None:
            logger.warning("sd-cli at %s is present but not runnable; not using it", binary)
            binary = None

    native_available = bool(binary or server_binary) and policy_eligible and fam_ok
    choice = select_diffusion_engine(
        backend, native_available = native_available, prefer_native = prefer_native
    )
    if choice == ENGINE_SD_CPP:
        return _activate(ENGINE_SD_CPP, None)

    # Explain the diffusers choice for status/telemetry.
    if not policy_eligible:
        reason = f"GPU backend '{backend}' uses diffusers"
    elif not fam_ok:
        reason = f"family '{fam.name}' has no native sd.cpp asset mapping"
    elif not (binary or server_binary):
        reason = "native sd.cpp binary unavailable"
    else:
        reason = "diffusers selected"
    return _activate(ENGINE_DIFFUSERS, reason)


def annotate_status(status: dict[str, Any]) -> dict[str, Any]:
    """Tag a backend status dict with the active engine + any fallback reason."""
    out = dict(status)
    out["engine"] = _active_engine_name
    out["fallback_reason"] = _fallback_reason
    return out


def active_status() -> dict[str, Any]:
    """The active engine's status, annotated with which engine + any fallback reason."""
    return annotate_status(get_active_diffusion_engine().status())
