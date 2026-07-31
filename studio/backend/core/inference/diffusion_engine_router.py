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
from typing import Any, Callable, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import (
    DiffusionFamily,
    family_pipeline_available,
    family_sd_cpp_supported,
)
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

# Resolved device backend -> the prebuilt sd-cli accelerator to install, used only for a force-native load on a GPU host:
# without it the installer defaults to "cpu" and a forced ROCm/Intel generation silently runs on CPU. Unknown -> "auto".
_INSTALL_ACCELERATOR = {"rocm": "rocm", "cuda": "cuda", "xpu": "vulkan"}


def _install_accelerator_for(backend: str) -> str:
    return _INSTALL_ACCELERATOR.get(backend, "auto")


# The engine the current load committed to, and why a non-native choice was made. Mutated only under _lock.
_lock = threading.Lock()
# Serializes a whole engine switch (check -> unload -> publish); _lock alone is released during the slow unload().
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
    # Serialize check -> unload -> publish without holding _lock across the slow unload(), closing the window where a second
    # _activate reads the still-old active engine and loads onto the engine this call is unloading.
    with _transition_lock:
        # Switching engines: unload the deactivated one first, else its model stays resident but unreachable (the evictor only
        # targets the active engine), leaking 10+ GB. The unload is slow, so resolve under _lock but run unload() OUTSIDE it.
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
            # Publish the new engine only AFTER the old one unloads: the evictor unloads get_active_diffusion_engine(), so flipping
            # the name first would let a concurrent acquire_for evict the new (empty) engine while the old model frees VRAM.
            try:
                engine_to_unload.unload()
            except Exception as exc:
                # Do NOT publish the new engine after a failed teardown. The old model (or the resident sd-server) still holds its memory, and flipping the
                # name would hide it from get_active_diffusion_engine(), which the evictor, /images/unload and the next load all resolve through, so the leak
                # would be permanent. Leaving the old engine active keeps it reclaimable and lets the caller retry.
                logger.error("failed to unload previous engine %s: %s", old_name, exc)
                raise RuntimeError(
                    f"Could not switch the diffusion engine to {name}: unloading the current "
                    f"{old_name} model failed ({exc}). The current model is still loaded; "
                    "unload it and try again."
                ) from exc
            with _lock:
                _active_engine_name = name
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if name == ENGINE_SD_CPP:
            logger.info("diffusion engine: sd_cpp")
        else:
            logger.info("diffusion engine: diffusers (%s)", reason or "selected")
        return get_active_diffusion_engine()


def begin_load_on(expected_engine: Any, start: Callable[[], Any]) -> Any:
    """Run ``start`` under the transition lock, refusing if the engine changed since selection.

    A load route selects its engine, then yields (device probe, arbiter acquire) before it
    registers the load. A second /images/load picking the OTHER engine can transition in that
    gap and unload the still-idle engine this request captured, which would then load a model
    nothing can reach: generate / status / unload and the arbiter's evictor all resolve through
    get_active_diffusion_engine(). Re-checking under the same lock the switch takes makes
    selection and registration one operation.
    """
    with _transition_lock:
        if expected_engine is not get_active_diffusion_engine():
            raise RuntimeError(
                "The diffusion engine changed while this load was starting. Retry the load."
            )
        return start()


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
        # Probe the resident sd-server FIRST (the backend prefers it): a server-only install must still route to native and should
        # not pay an sd-cli download. Install the accelerator-matched build so a forced-native GPU load gets the GPU server.
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed(),
            accelerator = _install_accelerator_for(backend),
        )
        if server_binary and not _server_binary_runnable(server_binary):
            logger.warning(
                "sd-server at %s is present but not runnable; not using it", server_binary
            )
            server_binary = None
        # sd-cli is the one-shot fallback: always LOCATE an existing binary, but auto-INSTALL only when there is no usable server.
        # Probe runnability first, else a present but non-runnable binary passes as available and fails inside the background load.
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


def predict_engine(fam: DiffusionFamily, *, model_kind: Optional[str] = None) -> str:
    """The engine a load of ``fam`` would select on this host, WITHOUT any side effect.

    Same policy as ``select_and_activate_engine`` -- and it has to be, because the download plan
    is built from it: the two engines need different files (sharded diffusers components vs
    sd-cli's single-file VAE + text encoders), so a plan built for the wrong one stages GB the
    load never opens and then fetches the right files inline, outside the manager.

    It differs from selection in exactly two ways, both deliberate. Nothing is activated (staging
    a download must not unload the resident model), and the binary is only LOCATED, never
    installed -- but an absent binary on a host where install is allowed still counts as
    available, because that is what the load will do. Getting that wrong the other way (planning
    diffusers for the very first native load, when no binary is on disk yet) would mispredict the
    common case: a fresh CPU host.
    """
    if model_kind and model_kind != "gguf":
        return ENGINE_DIFFUSERS

    forced, sd_cpp_pref, mps_enabled = _engine_config()
    if forced == ENGINE_DIFFUSERS:
        return ENGINE_DIFFUSERS
    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return ENGINE_DIFFUSERS

    backend = resolve_diffusion_device_target().backend
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    if not (policy_eligible and family_sd_cpp_supported(fam)):
        return ENGINE_DIFFUSERS

    server_binary = ensure_sd_server_binary(allow_install = False)
    if server_binary and not _server_binary_runnable(server_binary):
        server_binary = None
    binary = ensure_sd_cpp_binary(allow_install = False)
    if binary and SdCppEngine(binary = binary).version() is None:
        binary = None
    native_available = bool(binary or server_binary) or _install_allowed()
    return select_diffusion_engine(
        backend, native_available = native_available, prefer_native = prefer_native
    )


def family_buildable_here(fam: Optional[DiffusionFamily], *, model_kind: Optional[str]) -> bool:
    """True when THIS host can actually build ``fam`` for a ``model_kind`` load, by either engine.

    The two engines need different things. diffusers instantiates ``fam.pipeline_class``, which the
    newer families ship only in a newer diffusers -- and packaging still allows an older one on
    Python 3.9, whose ceiling predates them. The native sd.cpp engine assembles the same GGUF from
    single-file assets and imports no pipeline class at all, so a supported family loads there on a
    diffusers that has never heard of it.

    A family is therefore unbuildable only when NEITHER engine can do it, and both one-sided answers
    are wrong: gating on the diffusers class alone refuses (and hides) a GGUF the native engine
    serves fine, on the very hosts the native engine exists for; not gating at all advertises a pick
    that can only fail. The listing routes and ``validate_load_request`` share this one predicate so
    the picker and the loader cannot disagree.

    Cheap on an ordinary host: the engine prediction is reached only when the class is missing.
    ``predict_engine`` activates nothing and installs nothing."""
    if fam is None:
        return False
    if family_pipeline_available(fam):
        return True
    # Only a GGUF can go native, and only for a family with the single-file assets sd.cpp needs.
    if model_kind != "gguf" or not family_sd_cpp_supported(fam):
        return False
    try:
        return predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    except Exception:  # noqa: BLE001 -- a probe failure must not hide/refuse a usable model
        return False


def annotate_status(status: dict[str, Any]) -> dict[str, Any]:
    """Tag a backend status dict with the active engine + any fallback reason."""
    out = dict(status)
    out["engine"] = _active_engine_name
    out["fallback_reason"] = _fallback_reason
    return out


def active_status() -> dict[str, Any]:
    """The active engine's status, annotated with which engine + any fallback reason."""
    return annotate_status(get_active_diffusion_engine().status())
