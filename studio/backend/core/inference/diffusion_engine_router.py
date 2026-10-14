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
  UNSLOTH_DIFFUSION_SD_CPP_DEVICE=nvidia[:N]        run native on a card torch cannot see
  UNSLOTH_DIFFUSION_MLX=0|1|auto                    opt in to the Apple Silicon MLX engine (default 0; auto
                                                    behaves like 0 until a real-Mac A/B against diffusers MPS
                                                    and sd.cpp Metal settles the default)

The MLX engine (``mlx_diffusion_backend``) is considered only when UNSLOTH_DIFFUSION_MLX=1, on macOS 14+ on
Apple Silicon, for a GGUF of a family that declares ``mlx_model_config`` in a quant the converter handles
(Q4_0 / Q4_1 / Q8_0 / BF16 / F16), with ``mlx`` and ``mflux`` installed. Every refusal is recorded
(``mlx_fallback_reason`` in the status) and falls through to the choice this module makes without it.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import platform
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import (
    DiffusionFamily,
    family_mlx_supported,
    family_pipeline_available,
    family_sd_cpp_supported,
    pipeline_available_family_names,
)
from core.inference.sd_cpp_backend import (
    _card_lookup_inventory,
    _install_allowed,
    _managed_tree_in_use,
    _pin_moved,
    _server_binary_runnable,
    ensure_sd_cpp_binary,
    ensure_sd_server_binary,
    note_unlaunchable_accelerator_build,
    off_torch_build_mismatch,
    preferred_accelerator,
    sd_cpp_binary_runs_family,
    sd_cpp_vulkan_fallback_enabled,
    usable_or_recorded_failure,
)
from core.inference.sd_cpp_engine import (
    ENGINE_DIFFUSERS,
    ENGINE_MLX,
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


_OFF_TORCH_VENDOR_ACCELERATOR = {"nvidia": "cuda"}
_off_torch_warned: set[str] = set()


@dataclass(frozen = True)
class OffTorchDevice:
    """A card torch cannot see. ``index`` is nvidia-smi's row, never a torch ordinal."""

    vendor: str
    index: int
    accelerator: str

    @property
    def label(self) -> str:
        return f"{self.vendor}:{self.index}"

    def child_env(self) -> dict[str, str]:
        # nvidia-smi numbers cards in PCI order; CUDA's default is fastest-first.
        return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(self.index)}


def _warn_off_torch_once(raw: str, message: str, *args: Any) -> None:
    if raw in _off_torch_warned:
        return
    _off_torch_warned.add(raw)
    logger.warning(message, *args)


def _physical_inventory() -> dict:
    return _card_lookup_inventory() or {}


def off_torch_sd_cpp_device(backend: Optional[str] = None) -> Optional[OffTorchDevice]:
    """The card ``UNSLOTH_DIFFUSION_SD_CPP_DEVICE`` asks native image generation to run on, when
    torch cannot drive it (an NVIDIA card beside a ROCm torch). None keeps today's routing: unset,
    unparseable, a vendor torch already serves, or an inventory that answered without that card."""
    raw = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "").strip().lower()
    if not raw:
        return None
    vendor, _, index_text = raw.partition(":")
    vendor = vendor.strip()
    accelerator = _OFF_TORCH_VENDOR_ACCELERATOR.get(vendor)
    try:
        index = int(index_text) if index_text.strip() else 0
    except ValueError:
        index = -1
    if accelerator is None or index < 0:
        _warn_off_torch_once(
            raw,
            "UNSLOTH_DIFFUSION_SD_CPP_DEVICE=%r is not understood (expected nvidia or "
            "nvidia:<index>); ignoring it",
            raw,
        )
        return None
    if backend is None:
        backend = resolve_diffusion_device_target().backend
    if _INSTALL_ACCELERATOR.get(backend) == accelerator:
        return None
    inventory = _physical_inventory()
    unanswered = set(inventory.get("unanswered") or ())
    # Per vendor: an unanswered AMD probe says nothing about NVIDIA. Unknown naming nobody = cold cache.
    if vendor not in unanswered and not (inventory.get("unknown") and not unanswered):
        present = any(
            isinstance(device, dict)
            and device.get("vendor") == vendor
            and device.get("index") == index
            for device in inventory.get("devices") or ()
        )
        if not present:
            _warn_off_torch_once(
                raw,
                "UNSLOTH_DIFFUSION_SD_CPP_DEVICE=%s: this host lists no such card; ignoring it",
                raw,
            )
            return None
    return OffTorchDevice(vendor = vendor, index = index, accelerator = accelerator)


# Hosts the pinned release ships a Vulkan sd.cpp bundle for.
_VULKAN_SD_CPP_PLATFORMS = frozenset({("linux", "x86_64"), ("win32", "amd64")})


@functools.lru_cache(maxsize = 1)
def _vulkan_loader_loads() -> bool:
    """Whether the Vulkan loader library opens here. The Vulkan build links it, so without it
    neither sd-cli nor sd-server would start."""
    import ctypes

    try:
        if sys.platform == "win32":
            ctypes.WinDLL("vulkan-1.dll")
        else:
            ctypes.CDLL("libvulkan.so.1")
        return True
    except Exception:  # noqa: BLE001
        return False


def _vulkan_gpu_for_cpu_torch() -> bool:
    """True when torch is CPU-only but the host has a GPU the Vulkan sd.cpp build can render on:
    an AMD card (Vega, RDNA1 and Polaris get no ROCm wheel) or an Intel Arc. Never beside NVIDIA,
    which Vulkan could enumerate first, nor under a mask Vulkan does not read (#13666)."""
    if not sd_cpp_vulkan_fallback_enabled():
        return False
    if (sys.platform, platform.machine().lower()) not in _VULKAN_SD_CPP_PLATFORMS:
        return False
    if os.environ.get("ROCR_VISIBLE_DEVICES") is not None or os.environ.get(
        "GGML_VK_VISIBLE_DEVICES"
    ) is not None:
        return False
    inventory = _physical_inventory()
    if inventory.get("unknown") or "nvidia" in (inventory.get("unanswered") or ()):
        return False
    devices = [d for d in inventory.get("devices") or () if isinstance(d, dict)]
    if any(d.get("vendor") == "nvidia" for d in devices):
        return False

    def _renders(device: dict) -> bool:
        if device.get("vendor") == "amd":
            return True
        if device.get("vendor") != "intel":
            return False
        # Arc only: a UHD iGPU renders slower than the CPU build. Linux carries the PCI class,
        # the Windows registry only the name.
        if "xpu_class" in device:
            return device.get("xpu_class") is True
        return "arc" in str(device.get("name") or "").lower().replace("(tm)", " ").split()

    if not any(_renders(d) for d in devices):
        return False
    try:
        from utils.hardware.amd import the_vulkan_loader_has_no_usable_driver

        if the_vulkan_loader_has_no_usable_driver():
            return False
    except Exception:  # noqa: BLE001
        pass
    return _vulkan_loader_loads()


def image_install_accelerator(backend: str) -> str:
    """The sd.cpp build the IMAGE engine wants: the off-torch card's when one is set, else torch's,
    else Vulkan for a CPU torch beside an AMD or Arc GPU. Video keeps ``_install_accelerator_for``,
    since torch places its loads."""
    off_torch = off_torch_sd_cpp_device(backend)
    if off_torch is not None:
        return off_torch.accelerator
    if backend == "cpu" and _vulkan_gpu_for_cpu_torch():
        return "vulkan"
    return _install_accelerator_for(backend)


# The engine the current load committed to, and why a non-native choice was made. Mutated only under _lock.
_lock = threading.Lock()
# Serializes a whole engine switch (check -> unload -> publish); _lock alone is released during the slow unload().
_transition_lock = threading.Lock()
_active_engine_name: str = ENGINE_DIFFUSERS
_fallback_reason: Optional[str] = None
_mlx_fallback_reason: Optional[str] = None

_MLX_MIN_MACOS = (14, 0)


def _mlx_requested() -> bool:
    """``auto`` means off for now."""
    return os.environ.get("UNSLOTH_DIFFUSION_MLX", "0").strip().lower() in _ENABLE_TOKENS


def _mlx_host_refusal() -> Optional[str]:
    if sys.platform != "darwin":
        return "the MLX engine runs on macOS only"
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return "the MLX engine needs Apple Silicon"
    release = platform.mac_ver()[0]
    try:
        version = tuple(int(part) for part in release.split(".")[:2])
    except ValueError:
        version = ()
    if not version or version < _MLX_MIN_MACOS:
        return f"the MLX engine needs macOS 14 or newer (found {release or 'unknown'})"
    return None


# mflux's own darwin mlx range plus the light deps install_python_stack.py installs beside it (mflux goes in --no-deps).
_MLX_RUNTIME_REQUIREMENTS: tuple[str, ...] = (
    "mlx>=0.32.0,<0.33.0",
    "piexif>=1.1.3,<2",
    "platformdirs>=4,<5",
    "toml>=0.10.2,<1",
)


def _installed_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(name)
    except (PackageNotFoundError, ValueError):
        return None


def _mlx_runtime_refusal() -> Optional[str]:
    """find_spec and distribution metadata only: never import mlx / mflux here."""
    missing = [name for name in ("mlx", "mflux") if importlib.util.find_spec(name) is None]
    if missing:
        return f"{' and '.join(missing)} not installed"
    from core.inference.mlx_diffusion_convert import MFLUX_VERSION

    try:
        from packaging.requirements import Requirement
        for spec in (f"mflux=={MFLUX_VERSION}", *_MLX_RUNTIME_REQUIREMENTS):
            requirement = Requirement(spec)
            installed = _installed_version(requirement.name)
            if installed is None or not requirement.specifier.contains(installed, prereleases = True):
                return f"the MLX engine needs {spec} (found {installed or 'none'})"
    except Exception as exc:  # noqa: BLE001 -- no packaging, or a version it cannot parse
        return f"could not verify the MLX runtime versions ({exc})"
    return None


def mlx_selection(
    fam: Optional[DiffusionFamily],
    *,
    model_kind: Optional[str],
    gguf_filename: Optional[str],
    backend: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """``(use_mlx, refusal)``. Shared by selection and ``predict_engine`` so the download plan and the load
    cannot disagree about MLX."""
    if not _mlx_requested():
        return False, None
    forced = os.environ.get("UNSLOTH_DIFFUSION_ENGINE", "auto").strip().lower()
    if forced in (ENGINE_DIFFUSERS, ENGINE_SD_CPP):
        return False, f"UNSLOTH_DIFFUSION_ENGINE={forced} overrides UNSLOTH_DIFFUSION_MLX"
    if model_kind and model_kind != "gguf":
        return False, f"the MLX engine loads GGUF picks only (got {model_kind})"
    if fam is None or not family_mlx_supported(fam):
        name = getattr(fam, "name", None) or "unknown"
        return False, f"family '{name}' has no MLX implementation"
    from core.inference.mlx_diffusion_convert import quant_convertible

    convertible, why = quant_convertible(gguf_filename)
    if not convertible:
        return False, why or "GGUF quant not convertible to MLX"
    host = _mlx_host_refusal()
    if host:
        return False, host
    if backend is None:
        backend = resolve_diffusion_device_target().backend
    if backend not in ("mps", "cpu"):
        return False, f"device backend '{backend}' is not Apple MPS"
    runtime = _mlx_runtime_refusal()
    if runtime:
        return False, runtime
    return True, None


def _note_mlx_refusal(reason: Optional[str]) -> None:
    global _mlx_fallback_reason
    with _lock:
        _mlx_fallback_reason = reason
    if reason:
        logger.warning("UNSLOTH_DIFFUSION_MLX=1 but the MLX engine is not used: %s", reason)


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
    if name == ENGINE_MLX:
        from core.inference.mlx_diffusion_backend import get_mlx_diffusion_backend
        return get_mlx_diffusion_backend()
    from core.inference.diffusion import get_diffusion_backend

    return get_diffusion_backend()


def get_active_diffusion_engine() -> Any:
    """The engine object the active selection points at (defaults to diffusers)."""
    return engine_for(_active_engine_name)


# (module, singleton attribute) of every image engine, read from sys.modules only.
_ENGINE_SINGLETONS = (
    ("core.inference.diffusion", "_diffusion_backend"),
    ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ("core.inference.mlx_diffusion_backend", "_mlx_diffusion_backend"),
)


def cancel_generation_for_account(account_id: str) -> bool:
    """Stop an in-flight image generation owned by ``account_id``; True when one was signalled.

    Engines come from ``sys.modules`` (no import, no construction); all are checked because a
    deselected engine can still be draining."""
    cancelled = False
    for module_name, attribute in _ENGINE_SINGLETONS:
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
    for module_name, attribute in _ENGINE_SINGLETONS:
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
        elif name == ENGINE_MLX:
            logger.info("diffusion engine: mlx")
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
    """The card at an already RESOLVED ordinal, or ``None``, meaning every record applies. Never
    re-derived from the id list: free-VRAM ranking can name a different card the second time."""
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
    before_fallback: Optional[Callable[[], None]] = None,
    gguf_filename: Optional[str] = None,
) -> Any:
    """Pick + activate the engine for loading ``fam`` on this host; return the engine.
    ``before_fallback`` may raise to refuse a diffusers fallback before it unloads the resident model.

    Falls back to diffusers (recording a reason) when the native route is disabled, the device has
    a usable GPU, MPS is not enabled, the family has no native asset, or the binary is unavailable
    -- always BEFORE the slow load, so a fallback never strands a half-native load.

    With UNSLOTH_DIFFUSION_MLX=1 the MLX engine is tried first (``mlx_selection``, which needs
    ``gguf_filename`` to judge the quant); a refusal is recorded and the choice below runs unchanged.
    """
    use_mlx, mlx_refusal = mlx_selection(fam, model_kind = model_kind, gguf_filename = gguf_filename)
    _note_mlx_refusal(mlx_refusal)
    if use_mlx:
        return _activate(ENGINE_MLX, None)
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
    off_torch = off_torch_sd_cpp_device(backend)
    if off_torch is not None:
        # A card torch cannot see is only reachable natively, and torch's ordinals name other cards.
        prefer_native = True
        gpu_ordinal = None
    # CPU always native-eligible; MPS only when enabled; a GPU backend never, unless forced
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    fam_ok = family_sd_cpp_supported(fam)

    binary = None
    server_binary = None
    incapable_build = False
    if policy_eligible and fam_ok:
        # Once, so server and CLI cannot disagree.
        selected_card = _selected_card(gpu_ordinal)
        install_accelerator = preferred_accelerator(
            image_install_accelerator(backend), selected_card
        )
        if off_torch is not None:
            install_accelerator = preferred_accelerator(off_torch.accelerator, selected_card)
        # Probe the resident sd-server FIRST (the backend prefers it): a server-only install must still route to
        # native and should not pay an sd-cli download. Install the accelerator-matched build so a forced-native GPU
        # load gets the GPU server.
        # Offline an ensure hands back the condemned ROCm build, so a substitute is refused; a
        # DEFERRED upgrade keeps native for the teardown to land.
        upgrade_is_deferred = _managed_tree_in_use() and _install_allowed()

        def _accept(candidate):
            if candidate and upgrade_is_deferred:
                return candidate
            wrong_build = off_torch_build_mismatch(off_torch, candidate)
            if wrong_build:
                logger.warning(
                    "%s is the %s sd.cpp build; %s needs %s",
                    candidate,
                    wrong_build,
                    off_torch.label,
                    off_torch.accelerator,
                )
                return None
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
            # Held for the single recorder below.
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
            # One strike per bundle, only when NEITHER executable runs: one failing alone says
            # nothing about the accelerator, and two strikes from one install event would divert.
            # Here, not in the load, because a build the router rejects never reaches the load.
            note_unlaunchable_accelerator_build(
                unlaunchable_cli or unlaunchable_server, card = selected_card
            )
        # Runnable is not the same as capable. A build installed before this family's architecture
        # existed upstream is reused untouched -- nothing upgrades a runnable build of the right
        # accelerator -- and sd-cli only discovers it cannot read the model deep inside the load,
        # which for the server means a failed load and for the one-shot path a backend that
        # reported ready and dies on the first generation. Diffusers can run it, so drop the
        # incapable binary here and let the fallback below say why.
        for name, candidate in (("sd-server", server_binary), ("sd-cli", binary)):
            if candidate and not sd_cpp_binary_runs_family(candidate, fam):
                incapable_build = True
                logger.warning(
                    "%s at %s predates %s support; using diffusers. Reinstall the native engine "
                    "(delete the managed stable-diffusion.cpp install, or point SD_CLI_PATH at a "
                    "build from after upstream added it) to use it here",
                    name,
                    candidate,
                    fam.name,
                )
                if name == "sd-server":
                    server_binary = None
                else:
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
    elif incapable_build:
        reason = f"the installed sd.cpp build predates '{fam.name}' support"
    elif not (binary or server_binary):
        reason = "native sd.cpp binary unavailable"
    else:
        reason = "diffusers selected"
    if off_torch is not None:
        reason = f"{reason}; UNSLOTH_DIFFUSION_SD_CPP_DEVICE={off_torch.label} not honoured"
        logger.warning("image load stays on torch's device: %s", reason)
    if before_fallback is not None:
        before_fallback()
    return _activate(ENGINE_DIFFUSERS, reason)


def native_binary_installed(
    *, gpu_ordinal: Optional[int] = None, fam: Optional[DiffusionFamily] = None
) -> bool:
    """Whether a RUNNABLE sd.cpp binary is already on disk, installing nothing to find out.

    Separated from the prediction because the two answers differ where it matters: prediction
    counts an absent binary as available whenever installing one is allowed, and a caller that
    must know whether selection could still fall back to diffusers needs the unassumed answer.

    Filters exactly as selection does, card included, or the plan stages the wrong engine's files.
    Given ``fam``, that includes selection's architecture gate: a build that cannot run this family
    is not a native route for it, and a prediction that says otherwise stages sd-cli's companion
    VAE and text encoder for a load that goes to diffusers. Without ``fam`` the question is the
    older one, "is there a binary at all".
    """
    backend = resolve_diffusion_device_target().backend
    off_torch = off_torch_sd_cpp_device(backend)
    if off_torch is not None:
        gpu_ordinal = None
    selected_card = _selected_card(gpu_ordinal)
    install_accelerator = preferred_accelerator(image_install_accelerator(backend), selected_card)

    def _usable(candidate):
        if off_torch_build_mismatch(off_torch, candidate):
            return None
        return usable_or_recorded_failure(candidate, install_accelerator, selected_card)

    server_binary = _usable(
        ensure_sd_server_binary(allow_install = False, accelerator = install_accelerator)
    )
    if (
        server_binary
        and _server_binary_runnable(server_binary)
        and (fam is None or sd_cpp_binary_runs_family(server_binary, fam))
    ):
        return True
    binary = _usable(ensure_sd_cpp_binary(allow_install = False, accelerator = install_accelerator))
    if fam is not None and binary and not sd_cpp_binary_runs_family(binary, fam):
        return False
    return bool(binary and SdCppEngine(binary = binary).version() is not None)


def _resident_build_upgrades_on_load(gpu_ordinal: Optional[int]) -> bool:
    backend = resolve_diffusion_device_target().backend
    if off_torch_sd_cpp_device(backend) is not None:
        gpu_ordinal = None
    accelerator = preferred_accelerator(
        image_install_accelerator(backend), _selected_card(gpu_ordinal)
    )
    resident = ensure_sd_server_binary(
        allow_install = False, accelerator = accelerator
    ) or ensure_sd_cpp_binary(allow_install = False, accelerator = accelerator)
    return bool(resident) and _pin_moved(resident, accelerator)


def predict_engine(
    fam: DiffusionFamily,
    *,
    model_kind: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
    gguf_filename: Optional[str] = None,
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
    if mlx_selection(fam, model_kind = model_kind, gguf_filename = gguf_filename)[0]:
        return ENGINE_MLX
    if model_kind and model_kind != "gguf":
        return ENGINE_DIFFUSERS

    forced, sd_cpp_pref, mps_enabled = _engine_config()
    if forced == ENGINE_DIFFUSERS:
        return ENGINE_DIFFUSERS
    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return ENGINE_DIFFUSERS

    backend = resolve_diffusion_device_target().backend
    if off_torch_sd_cpp_device(backend) is not None:
        prefer_native = True
        gpu_ordinal = None
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    if not (policy_eligible and family_sd_cpp_supported(fam)):
        return ENGINE_DIFFUSERS

    # A permitted install counts as available only where selection would actually perform one.
    # ``ensure_*_binary`` keeps a runnable build of the right accelerator untouched, so a resident
    # build that cannot run this family is never upgraded away, and predicting native for it would
    # stage sd-cli's companions for a load the router sends to diffusers. With nothing resident (or
    # a moved pin) the install lands on the pinned prebuilt, which is current by definition.
    native_available = native_binary_installed(gpu_ordinal = gpu_ordinal, fam = fam) or (
        _install_allowed()
        and (
            not native_binary_installed(gpu_ordinal = gpu_ordinal)
            or _resident_build_upgrades_on_load(gpu_ordinal)
        )
    )
    return select_diffusion_engine(
        backend, native_available = native_available, prefer_native = prefer_native
    )


def family_buildable_here(
    fam: Optional[DiffusionFamily],
    *,
    model_kind: Optional[str],
    gguf_filename: Optional[str] = None,
) -> bool:
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
    if model_kind == "gguf" and gguf_filename and family_mlx_supported(fam):
        try:
            if predict_engine(fam, model_kind = "gguf", gguf_filename = gguf_filename) == ENGINE_MLX:
                return True
        except Exception:  # noqa: BLE001 -- a probe failure must not hide/refuse a usable model
            pass
    # only a GGUF can go native, and only for a family with the single-file assets sd.cpp needs
    if model_kind != "gguf" or not family_sd_cpp_supported(fam):
        return False
    try:
        return predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    except Exception:  # noqa: BLE001 -- a probe failure must not hide/refuse a usable model
        return False


@functools.cache
def _supported_family_capabilities() -> tuple[str, ...]:
    return tuple(pipeline_available_family_names())


def annotate_status(status: dict[str, Any]) -> dict[str, Any]:
    """Tag a backend status dict with the active engine + any fallback reason."""
    out = dict(status)
    out["engine"] = _active_engine_name
    out["fallback_reason"] = _fallback_reason
    out["mlx_fallback_reason"] = _mlx_fallback_reason
    out["supported_families"] = list(_supported_family_capabilities())
    return out


def active_status() -> dict[str, Any]:
    """The active engine's status, annotated with which engine + any fallback reason."""
    return annotate_status(get_active_diffusion_engine().status())
