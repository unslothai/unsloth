# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selects the diffusion engine (diffusers vs native sd.cpp) for the live route.

One engine at a time. On a CUDA/ROCm/XPU GPU it is the diffusers ``DiffusionBackend`` (the default,
the only path with the torchao fast-quant / compile stack); with no usable GPU (CPU, or MPS when
enabled) the native ``SdCppDiffusionBackend`` (faster, lighter on RAM there). Chosen once at load
and remembered, so ``generate`` / ``unload`` / ``status`` / progress all act on the same engine.

Built on the pure ``select_diffusion_engine`` decision; this module adds the policy (env opt-out,
MPS gating, per-family native-asset support, lazy binary availability) and records fallbacks.

Env knobs:
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

# Resolved device backend -> the prebuilt sd-cli accelerator to install. Used only for a
# force-native load on a GPU host: without it the installer defaults to "cpu" and a forced
# ROCm/Intel sd_cpp generation would silently run on CPU. Unknown backends -> "auto" (CPU/Metal),
# matching the installer default. (No CUDA-Linux asset, so "cuda" only differs on Windows.)
_INSTALL_ACCELERATOR = {"rocm": "rocm", "cuda": "cuda", "xpu": "vulkan"}


def _install_accelerator_for(backend: str) -> str:
    return _INSTALL_ACCELERATOR.get(backend, "auto")


# The engine the current load committed to, and why a non-native choice was made. Mutated only
# under _lock during selection.
_lock = threading.Lock()
# Serializes a WHOLE engine switch (check -> unload -> publish). _lock alone is released during the
# slow unload(), so two overlapping selections could interleave: one observes the not-yet-updated
# active engine, returns it, and loads onto the very engine the other is concurrently unloading.
_transition_lock = threading.Lock()
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
    # Serialize the ENTIRE check -> unload -> publish transition. _lock is released during the slow
    # unload() below (holding it across the unload would block every status/selection reader), which
    # opens a window where a second _activate could read the still-old active engine, take the "no
    # change" branch, and return that engine -- then load onto it while this call is unloading it.
    # _transition_lock closes the window without holding _lock across the unload; the final
    # get_active_diffusion_engine() now reflects the committed state.
    with _transition_lock:
        # Switching engines: unload the deactivated one first, else its model stays resident but
        # unreachable (the evictor only targets the active engine), leaking 10+ GB. The unload is
        # slow, so resolve the engine under _lock but run unload() OUTSIDE it.
        engine_to_unload = None
        old_name = None
        with _lock:
            if name != _active_engine_name:
                engine_to_unload = get_active_diffusion_engine()
                old_name = _active_engine_name
            else:
                # No engine change: publish the (possibly refreshed) fallback reason now.
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if engine_to_unload is not None:
            # Publish the new engine only AFTER the old one unloads. The evictor unloads
            # get_active_diffusion_engine(), so flipping the name first would let a concurrent
            # acquire_for evict the new (empty) engine and take the GPU while the old model is still
            # freeing VRAM -- two large models briefly resident. Keeping the OLD engine as the evict
            # target serializes a concurrent evict on its unload(), granting the GPU only once freed.
            try:
                engine_to_unload.unload()
            except Exception as exc:  # noqa: BLE001 -- best-effort; never block the switch
                logger.warning("failed to unload previous engine %s: %s", old_name, exc)
            with _lock:
                _active_engine_name = name
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if name == ENGINE_SD_CPP:
            logger.info("diffusion engine: sd_cpp")
        else:
            logger.info("diffusion engine: diffusers (%s)", reason or "selected")
        return get_active_diffusion_engine()


def select_and_activate_engine(
    fam: DiffusionFamily,
    *,
    hf_token: Optional[str] = None,
    model_kind: Optional[str] = None,
) -> Any:
    """Pick + activate the engine for loading ``fam`` on this host; return the engine.

    Falls back to diffusers (recording a reason) when the native route is disabled, the device has
    a usable GPU, MPS is not enabled, the family has no native asset, or the binary is unavailable
    -- always BEFORE the slow load, so a fallback never strands a half-native load.
    """
    # Non-GGUF loads run on diffusers only (the native engine consumes single-file GGUF only).
    if model_kind and model_kind != "gguf":
        return _activate(ENGINE_DIFFUSERS, f"non-GGUF load ({model_kind}) requires diffusers")

    forced, sd_cpp_pref, mps_enabled = _engine_config()

    if forced == ENGINE_DIFFUSERS:
        return _activate(ENGINE_DIFFUSERS, "forced (UNSLOTH_DIFFUSION_ENGINE=diffusers)")

    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return _activate(ENGINE_DIFFUSERS, "native engine disabled (UNSLOTH_DIFFUSION_SD_CPP=0)")

    target = resolve_diffusion_device_target()
    backend = target.backend
    # Policy: CPU always native-eligible; MPS only when enabled; a GPU backend never, unless forced.
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    fam_ok = family_sd_cpp_supported(fam)

    binary = None
    server_binary = None
    if policy_eligible and fam_ok:
        # Probe the resident sd-server FIRST (the backend prefers it): a server-only install must
        # still route to native, and a server-only host shouldn't pay an sd-cli download. Install
        # the accelerator-matched build so a forced-native GPU load gets the GPU server.
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed(),
            accelerator = _install_accelerator_for(backend),
        )
        if server_binary and not _server_binary_runnable(server_binary):
            logger.warning(
                "sd-server at %s is present but not runnable; not using it", server_binary
            )
            server_binary = None
        # sd-cli is the one-shot fallback. Always LOCATE an existing binary, but auto-INSTALL only
        # when there is no usable server. Probe runnability before committing native: a present but
        # non-runnable binary would otherwise pass as available and fail inside the background load.
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
