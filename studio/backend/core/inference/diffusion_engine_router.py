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
import sys
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
    _managed_tree_in_use,
    _server_binary_runnable,
    ensure_sd_cpp_binary,
    ensure_sd_server_binary,
    note_unlaunchable_accelerator_build,
    preferred_accelerator,
    usable_or_recorded_failure,
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

# Resolved device backend -> the prebuilt sd-cli accelerator to install, used only for a force-native load on a GPU
# host: without it the installer defaults to "cpu" and a forced ROCm/Intel generation silently runs on CPU. Unknown ->
# "auto".
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


def engine_for(name: str) -> Any:
    """The engine object a name refers to, WITHOUT activating it: activating unloads the resident
    model, so /images/load's gated-repo preflight needs the pending engine before the switch."""
    if name == ENGINE_SD_CPP:
        from core.inference.sd_cpp_backend import get_sd_cpp_backend
        return get_sd_cpp_backend()
    from core.inference.diffusion import get_diffusion_backend

    return get_diffusion_backend()


def get_active_diffusion_engine() -> Any:
    """The engine object the active selection points at (defaults to diffusers)."""
    return engine_for(_active_engine_name)


def cancel_generation_for_account(account_id: str) -> bool:
    """Stop an in-flight image generation owned by ``account_id``; True when one was signalled.

    Engines come from ``sys.modules`` (no import, no construction); both are checked because a
    deselected engine can still be draining."""
    cancelled = False
    for module_name, attribute in (
        ("core.inference.diffusion", "_diffusion_backend"),
        ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ):
        module = sys.modules.get(module_name)
        engine = getattr(module, attribute, None) if module is not None else None
        if engine is None or engine._active_generate_account != account_id:
            continue
        # cancel_generate rechecks the owner under its lock.
        if engine.cancel_generate(expected_account = account_id):
            cancelled = True
    return cancelled


def retire_load_for_account(account_id: str) -> bool:
    """Tear down an in-flight image load ``account_id`` started; True when one was found."""
    from hub.services.models.account_access import retire_media_load

    retired = False
    for module_name, attribute in (
        ("core.inference.diffusion", "_diffusion_backend"),
        ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ):
        module = sys.modules.get(module_name)
        engine = getattr(module, attribute, None) if module is not None else None
        if retire_media_load("diffusion", account_id, engine):
            retired = True
    return retired


def active_engine_name() -> str:
    return _active_engine_name


def _activate(name: str, reason: Optional[str]) -> Any:
    global _active_engine_name, _fallback_reason
    with _transition_lock:
        # Switching engines: unload the deactivated one first, else its model stays resident but unreachable (the
        # evictor only targets the active engine), leaking 10+ GB. The unload is slow, so resolve under _lock but run
        # unload() OUTSIDE it.
        engine_to_unload = None
        old_name = None
        with _lock:
            if name != _active_engine_name:
                engine_to_unload = get_active_diffusion_engine()
                old_name = _active_engine_name
            else:
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if engine_to_unload is not None:
            # Publish the new engine only AFTER the old one unloads: the evictor unloads
            # get_active_diffusion_engine(), so flipping the name first would let a concurrent acquire_for evict the
            # new (empty) engine while the old model frees VRAM.
            try:
                engine_to_unload.unload()
            except Exception as exc:
                # Do NOT publish the new engine after a failed teardown. The old model (or the resident sd-server)
                # still holds its memory, and flipping the name would hide it from get_active_diffusion_engine(),
                # which the evictor, /images/unload and the next load all resolve through, so the leak would be
                # permanent. Leaving the old engine active keeps it reclaimable and lets the caller retry.
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


def _selected_card(gpu_ordinal) -> Optional[str]:
    """The card this request picked, or ``None``, meaning every record applies. A recorded failure
    is about a CARD: one ROCm bundle can carry one host card's gfx target and not another's.

    Takes the ordinal the caller already RESOLVED, never the id list. Ranking by free VRAM is what
    turns several ids into one ordinal, and free VRAM moves the moment a checkpoint lands, so
    re-deriving it here could name a different card from the one the load then runs on."""
    if gpu_ordinal is None:
        return None
    try:
        from core.inference.sd_cpp_backend import selected_card_identity
        return selected_card_identity(gpu_ordinal)
    except Exception:  # noqa: BLE001
        return None


def select_and_activate_engine(
    fam: DiffusionFamily,
    *,
    hf_token: Optional[str] = None,
    model_kind: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
) -> Any:
    """Pick + activate the engine for loading ``fam`` on this host; return the engine.

    Falls back to diffusers (recording a reason) when the native route is disabled, the device has
    a usable GPU, MPS is not enabled, the family has no native asset, or the binary is unavailable
    -- always BEFORE the slow load, so a fallback never strands a half-native load.
    """
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
    # CPU always native-eligible; MPS only when enabled; a GPU backend never, unless forced
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    fam_ok = family_sd_cpp_supported(fam)

    binary = None
    server_binary = None
    if policy_eligible and fam_ok:
        # Once, per card, so server and CLI cannot disagree: a card whose ROCm build will not start (#9278, #8814) must not divert the others.
        selected_card = _selected_card(gpu_ordinal)
        install_accelerator = preferred_accelerator(
            _install_accelerator_for(backend), selected_card
        )
        # Probe the resident sd-server FIRST (the backend prefers it): a server-only install must still route to
        # native and should not pay an sd-cli download. Install the accelerator-matched build so a forced-native GPU
        # load gets the GPU server.
        # An ensure does not promise the accelerator it was given: offline it hands back the
        # condemned ROCm build, which passes the probes below and dies mid-render. Only a SUBSTITUTE
        # is refused -- with the fallback off, ROCm is the request again. A merely DEFERRED upgrade
        # keeps the native selection for the teardown to land, else the reload goes to diffusers.
        upgrade_is_deferred = _managed_tree_in_use() and _install_allowed()

        def _accept(candidate):
            if candidate and upgrade_is_deferred:
                return candidate
            return usable_or_recorded_failure(candidate, install_accelerator, selected_card)

        server_binary = _accept(
            ensure_sd_server_binary(
                allow_install = _install_allowed(),
                accelerator = install_accelerator,
            )
        )
        unlaunchable_server: Optional[str] = None
        if server_binary and not _server_binary_runnable(server_binary):
            logger.warning(
                "sd-server at %s is present but not runnable; not using it", server_binary
            )
            # HELD, not recorded. See the single recorder below.
            unlaunchable_server = server_binary
            server_binary = None
        # sd-cli is the one-shot fallback: always LOCATE an existing binary, but auto-INSTALL only when there is no
        # usable server. Probe runnability first, else a present but non-runnable binary passes as available and fails
        # inside the background load.
        binary = _accept(
            ensure_sd_cpp_binary(
                allow_install = _install_allowed() and server_binary is None,
                accelerator = install_accelerator,
            )
        )
        unlaunchable_cli: Optional[str] = None
        if binary and SdCppEngine(binary = binary).version() is None:
            logger.warning("sd-cli at %s is present but not runnable; not using it", binary)
            unlaunchable_cli = binary
            binary = None
        if binary is None and server_binary is None and (unlaunchable_cli or unlaunchable_server):
            # ONE recorder, and only once NEITHER executable can run. sd-server and sd-cli are two
            # files out of a single install, so either one failing alone says nothing about the
            # accelerator: the other may run it perfectly, the load succeeds, and a strike for it is
            # a strike on a working host. Two of those reach the diversion bar and replace a healthy
            # ROCm bundle with Vulkan for good. Recording each separately had the same effect from
            # the other direction, spending both strikes on one install event.
            #
            # Counted here rather than in the load because the router runs first, so a build it
            # rejects never reaches the recorders there. Against the CARD being selected, since a
            # card-less note is read as host-wide and would move every other card to Vulkan too.
            note_unlaunchable_accelerator_build(
                unlaunchable_cli or unlaunchable_server, card = selected_card
            )

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


def native_binary_installed(*, gpu_ordinal: Optional[int] = None) -> bool:
    """Whether a RUNNABLE sd.cpp binary is already on disk, installing nothing to find out.

    Separated from the prediction because the two answers differ where it matters: prediction
    counts an absent binary as available whenever installing one is allowed, and a caller that
    must know whether selection could still fall back to diffusers needs the unassumed answer.

    Must filter exactly as selection does: this prediction picks which planner stages the download,
    so disagreeing leaves an offline load on diffusers with none of its assets staged. That includes
    the CARD, because the failure records are card-scoped: with no ordinal a per-card record reads as
    host-wide here while selection, which is given the ordinal, correctly clears the working card, and
    the plan then stages the diffusers files the load never opens. ``gpu_ordinal`` None keeps the
    host-wide reading, which is right for a caller that has no card in hand.
    """
    selected_card = _selected_card(gpu_ordinal)
    install_accelerator = preferred_accelerator(
        _install_accelerator_for(resolve_diffusion_device_target().backend), selected_card
    )
    server_binary = usable_or_recorded_failure(
        ensure_sd_server_binary(allow_install = False, accelerator = install_accelerator),
        install_accelerator,
        selected_card,
    )
    if server_binary and _server_binary_runnable(server_binary):
        return True
    binary = usable_or_recorded_failure(
        ensure_sd_cpp_binary(allow_install = False, accelerator = install_accelerator),
        install_accelerator,
        selected_card,
    )
    return bool(binary and SdCppEngine(binary = binary).version() is not None)


def predict_engine(
    fam: DiffusionFamily,
    *,
    model_kind: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
) -> str:
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

    native_available = native_binary_installed(gpu_ordinal = gpu_ordinal) or _install_allowed()
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
    # only a GGUF can go native, and only for a family with the single-file assets sd.cpp needs
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
