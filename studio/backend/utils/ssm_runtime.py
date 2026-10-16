# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-install the SSM/Mamba kernels a hybrid model lazy-imports during ``from_pretrained``;
the inference-path counterpart of the training worker's install.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional

from loggers import get_logger
from utils.kernel_install import (
    CAUSAL_CONV1D,
    MAMBA_SSM,
    PinnedKernel,
    hipcc_gcc_install_dir,
    install_prebuilt,
    source_build_command,
    source_build_run_kwargs,
    uninstall_command,
)
from utils.wheel_utils import (
    direct_wheel_url,
    install_wheel,
    is_windows_wheel_env,
    probe_torch_wheel_env,
    url_exists,
)

logger = get_logger(__name__)

StatusCb = Optional[Callable[[str], None]]

# Lowercased-id substring matches, shared with the training worker. mamba-ssm models are a subset of the causal-conv1d set.
# The Mamba-2 hybrids whose modeling (or remote) code cannot load without mamba_ssm: a failed install is fatal for them.
REQUIRED_SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
)
SSM_MODEL_SUBSTRINGS = REQUIRED_SSM_MODEL_SUBSTRINGS + (
    "mamba",
    "jamba",
    "zamba",
    "bamba",
)
# config.json model types whose modeling files run mamba_ssm (and so causal_conv1d). Matched on the
# resolved config, so a renamed checkpoint or an id without the substrings above
# (NVIDIA-Nemotron-Nano-9B-v2, granite-4.0-tiny) still counts. The Mamba-1 family and the other
# hybrids have a working torch path in transformers, so mamba_ssm is only an acceleration there.
REQUIRED_SSM_MODEL_TYPES = frozenset({"nemotron_h", "falcon_h1", "granitemoehybrid"})
SSM_MODEL_TYPES = REQUIRED_SSM_MODEL_TYPES | frozenset(
    {"bamba", "jamba", "zamba", "zamba2", "mamba", "mamba2", "falcon_mamba"}
)
CAUSAL_CONV1D_MODEL_SUBSTRINGS = (
    "qwen3.5",
    "qwen3_5",
    "qwen3.6",
    "qwen3_6",
    "qwen3-next",
    "qwen3_next",
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
    "lfm2",
    "mamba",
    "jamba",
    "zamba",
    "bamba",
)
_TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE: dict[tuple[str, str], bool | None] = {}


class SsmKernels(NamedTuple):
    """What one model needs: ``causal_conv1d`` and ``mamba_ssm`` (best effort), and whether a
    missing ``mamba_ssm`` must stop the load. ``probe`` is the identifier the decision was made for
    (an adapter's base when there is one); ``model_types`` the config's model types, if read."""

    causal_conv1d: bool
    mamba_ssm: bool
    mamba_ssm_required: bool
    probe: str
    model_types: frozenset[str] = frozenset()
    # (dynamic-module namespace, implementation module) pairs from the config's auto_map, as
    # transformers names them under transformers_modules (see remote_code_modules).
    remote_code: frozenset = frozenset()


def _base_model_type(model_type: str) -> str:
    """A config's model type with its `_text` / `_vision` sub-config suffix removed."""
    candidate = model_type.strip().lower().replace("-", "_")
    for suffix in ("_text", "_vision"):
        candidate = candidate.removesuffix(suffix)
    return candidate


def model_is_ssm(model_name: str) -> bool:
    """Whether *model_name* is a Mamba/SSM model whose fast path is ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in SSM_MODEL_SUBSTRINGS)


def model_requires_mamba_ssm(model_name: str) -> bool:
    """Whether *model_name* cannot load at all without ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in REQUIRED_SSM_MODEL_SUBSTRINGS)


def model_wants_causal_conv1d(model_name: str) -> bool:
    """Whether *model_name* needs ``causal_conv1d`` (the SSM set plus linear-attention hybrids like Qwen3.5 / 3.6 / Qwen3-Next / LFM2 whose modeling files lazy-import it)."""
    name = (model_name or "").lower()
    return any(sub in name for sub in CAUSAL_CONV1D_MODEL_SUBSTRINGS)


def _normalized_model_identifier(value: str) -> str:
    return "".join(
        character for character in value.lower() if character.isascii() and character.isalnum()
    )


def _transformers_models_dir() -> Path | None:
    """The active transformers' ``models`` folder, found without importing transformers: the
    inference worker asks before its transformers import, which snapshots the kernel gates."""
    try:
        spec = importlib.util.find_spec("transformers")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or ())
    if not locations and spec.origin:
        locations = [str(Path(spec.origin).parent)]
    for location in locations:
        models_dir = Path(location) / "models"
        if models_dir.is_dir():
            return models_dir
    return None


def _transformers_model_type_uses_causal_conv1d(model_type: str) -> bool | None:
    candidate = model_type.strip().lower().replace("-", "_")
    if not candidate or any(
        not (character.isascii() and (character.isalnum() or character == "_"))
        for character in candidate
    ):
        return None
    models_dir = _transformers_models_dir()
    cache_key = (str(models_dir), candidate)
    if cache_key in _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE:
        return _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[cache_key]

    result: bool | None = None
    try:
        model_dir = models_dir / candidate if models_dir is not None else None
        if model_dir is not None and model_dir.is_dir():
            for modeling_file in model_dir.glob("modeling_*.py"):
                try:
                    source = modeling_file.read_text(encoding = "utf-8", errors = "ignore")
                except OSError:
                    continue
                result = False
                if "causal_conv1d" in source:
                    result = True
                    break
    except Exception as exc:
        logger.debug("causal-conv1d model-type inspection skipped: %s", exc)

    _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[cache_key] = result
    return result


def _config_identifiers(model_config: dict) -> tuple[set[str], set[str]]:
    """Every model_type and architecture in *model_config*, nested sub-configs included."""
    model_types: set[str] = set()
    architectures: set[str] = set()
    pending: list[Any] = [model_config]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            model_type = value.get("model_type")
            if isinstance(model_type, str):
                model_types.add(model_type)
            model_architectures = value.get("architectures")
            if isinstance(model_architectures, (list, tuple)):
                architectures.update(
                    architecture
                    for architecture in model_architectures
                    if isinstance(architecture, str)
                )
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return model_types, architectures


def model_config_is_ssm(model_config: dict) -> bool | None:
    """True for a Mamba-family model type, False for another known type, None without one."""
    model_types, _ = _config_identifiers(model_config)
    if not model_types:
        return None
    return any(_base_model_type(model_type) in SSM_MODEL_TYPES for model_type in model_types)


def model_config_requires_mamba_ssm(model_config: dict) -> bool | None:
    """True when the config's model cannot load without ``mamba_ssm``; None without a model type."""
    model_types, _ = _config_identifiers(model_config)
    if not model_types:
        return None
    return any(
        _base_model_type(model_type) in REQUIRED_SSM_MODEL_TYPES for model_type in model_types
    )


def model_config_wants_causal_conv1d(model_config: dict) -> bool | None:
    model_types, architectures = _config_identifiers(model_config)
    if model_config_is_ssm(model_config):
        return True
    source_requirements = {
        _transformers_model_type_uses_causal_conv1d(model_type) for model_type in model_types
    }
    if True in source_requirements:
        return True
    config_identifiers = model_types | architectures
    normalized_needles = {
        _normalized_model_identifier(value) for value in CAUSAL_CONV1D_MODEL_SUBSTRINGS
    }
    if any(
        needle in _normalized_model_identifier(identifier)
        for identifier in config_identifiers
        for needle in normalized_needles
    ):
        return True
    if False in source_requirements:
        return False
    return None


def _local_adapter_base(model_load_target: str) -> str | None:
    """A local PEFT adapter folder's base model (``adapter_config.json`` without a ``config.json``)."""
    try:
        folder = Path(model_load_target)
        if not folder.is_dir() or (folder / "config.json").is_file():
            return None
        adapter_config = folder / "adapter_config.json"
        if not adapter_config.is_file():
            return None
        import json

        base = json.loads(adapter_config.read_text(encoding = "utf-8-sig")).get(
            "base_model_name_or_path"
        )
    except Exception as exc:
        logger.debug("Could not read the adapter base of %s: %s", model_load_target, exc)
        return None
    return base if isinstance(base, str) and base.strip() else None


def _dependency_target(model_load_target: str, adapter_base: str | None) -> str:
    """What decides the kernels: an adapter's (pinned, else recorded) base, else the target itself.
    The adapter stays the load target; only the dependency decision moves to its base."""
    if adapter_base:
        return adapter_base
    return _local_adapter_base(model_load_target) or model_load_target


def _resolved_model_config(model_load_target: str, hf_token: str | None) -> dict | None:
    try:
        from utils.transformers_version import _load_config_json
        model_config = _load_config_json(model_load_target, hf_token)
    except Exception as exc:
        logger.debug("Could not inspect model config for SSM kernels: %s", exc)
        return None
    return model_config if isinstance(model_config, dict) else None


def resolve_ssm_kernels(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
) -> SsmKernels:
    """The kernels the model at *model_load_target* needs, from its config (an adapter's base
    config when it is an adapter); the name lists only when there is no config to read."""
    target = _dependency_target(model_load_target, adapter_base)
    probe = target if target != model_load_target else model_name
    model_config = _resolved_model_config(target, hf_token)
    model_types: frozenset[str] = frozenset()
    is_ssm = required = wants = None
    remote_code: frozenset = frozenset()
    if model_config is not None:
        remote_code = remote_code_modules(target, model_config)
        model_types = frozenset(
            _base_model_type(model_type) for model_type in _config_identifiers(model_config)[0]
        )
        is_ssm = model_config_is_ssm(model_config)
        required = model_config_requires_mamba_ssm(model_config)
        wants = model_config_wants_causal_conv1d(model_config)
        logger.info(
            "SSM kernels resolved from model architecture: causal-conv1d=%s mamba-ssm=%s",
            wants,
            is_ssm,
        )
    if is_ssm is None:
        is_ssm = model_is_ssm(probe)
    if required is None:
        required = model_requires_mamba_ssm(probe)
    if wants is None:
        wants = model_wants_causal_conv1d(probe)
    # Every mamba-ssm model also runs causal-conv1d.
    return SsmKernels(
        causal_conv1d = bool(wants or is_ssm),
        mamba_ssm = bool(is_ssm),
        mamba_ssm_required = bool(is_ssm and required),
        probe = probe,
        model_types = model_types,
        remote_code = remote_code,
    )


def resolved_model_wants_causal_conv1d(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
) -> bool:
    return resolve_ssm_kernels(
        model_name, model_load_target, hf_token, adapter_base = adapter_base
    ).causal_conv1d


def resolved_model_is_ssm(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
) -> bool:
    """Whether the model at *model_load_target* runs ``mamba_ssm``, from its (adapter's base) config;
    the name list only when there is no config to read."""
    return resolve_ssm_kernels(
        model_name, model_load_target, hf_token, adapter_base = adapter_base
    ).mamba_ssm


def refresh_transformers_kernel_gates() -> None:
    """After an in-process install, let an already-imported transformers see the new kernels: its
    availability gates are lru_cached and the hub-kernel map can hold a cached None. Never raises.
    This cannot rebind functions a modeling module already captured; see
    ``modeling_modules_needing_restart``."""
    if "transformers" not in sys.modules:
        return
    try:
        import_utils = sys.modules.get("transformers.utils.import_utils")
        for gate_name in (
            "is_causal_conv1d_available",
            "is_mamba_ssm_available",
            "is_mamba_2_ssm_available",
        ):
            cache_clear = getattr(getattr(import_utils, gate_name, None), "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    except Exception as exc:
        logger.debug("Could not clear transformers' kernel gates: %s", exc)
    try:
        hub_kernels = sys.modules.get("transformers.integrations.hub_kernels")
        mapping = getattr(hub_kernels, "_KERNEL_MODULE_MAPPING", None)
        if isinstance(mapping, dict):
            for key in [key for key, value in mapping.items() if value is None]:
                if any(
                    kernel in str(key) for kernel in ("causal-conv1d", "causal_conv1d", "mamba")
                ):
                    mapping.pop(key, None)
    except Exception as exc:
        logger.debug("Could not refresh transformers' hub kernel map: %s", exc)


def _sanitized_module_name(name: str) -> str:
    # transformers.dynamic_module_utils._sanitize_module_name (4.57+, 5.x).
    name = name.replace(".", "_dot_").replace("-", "_hyphen_")
    return f"_{name}" if name and name[0].isdigit() else name


def _dynamic_module_namespaces(checkpoint: str) -> set[tuple[str, ...]]:
    """The transformers_modules package(s) transformers puts *checkpoint*'s remote code under: a
    Hub repo by its id (org, name), a local folder by its name. Sanitized as 4.57+ and 5.x spell
    them, and raw as older 4.x did."""
    checkpoint = str(checkpoint).strip()
    if not checkpoint:
        return set()
    if os.path.isdir(checkpoint):
        parts = [os.path.basename(os.path.normpath(checkpoint))]
    else:
        parts = checkpoint.split("/")
    if not all(parts):
        return set()
    return {tuple(parts), tuple(_sanitized_module_name(part) for part in parts)}


def remote_code_modules(checkpoint: str, model_config: dict) -> frozenset:
    """The (namespace, module) pairs of the model classes a config's ``auto_map`` points at: an
    entry ``"model.CustomForCausalLM"`` is module ``model`` of *checkpoint*, and
    ``"org/repo--model.Custom"`` module ``model`` of ``org/repo``. Any module name counts, not just
    ``modeling_*``; AutoConfig / AutoTokenizer / processor entries capture no kernels."""
    auto_map = model_config.get("auto_map") if isinstance(model_config, dict) else None
    if not isinstance(auto_map, dict):
        return frozenset()
    pairs = set()
    for key, value in auto_map.items():
        if not str(key).startswith("AutoModel"):
            continue
        for reference in value if isinstance(value, (list, tuple)) else (value,):
            if not isinstance(reference, str) or "." not in reference:
                continue
            source, _, reference = reference.rpartition("--")
            module = tuple(reference.rsplit(".", 1)[0].split("."))
            for namespace in _dynamic_module_namespaces(source or checkpoint):
                pairs.add((namespace, module))
    return frozenset(pairs)


def _is_remote_code_module(name: str, remote_code) -> bool:
    # transformers_modules.<namespace>[.<commit or source hash>].<module>, matched exactly.
    parts = tuple(name.split("."))
    if parts[0] != "transformers_modules":
        return False
    for namespace, module in remote_code:
        if parts[1 : 1 + len(namespace)] != namespace:
            continue
        rest = parts[1 + len(namespace) :]
        if rest == module or (len(rest) == len(module) + 1 and rest[1:] == module):
            return True
    return False


def modeling_modules_needing_restart(model_types, remote_code = ()) -> list[str]:
    """The already-imported implementation modules of *model_types* (``transformers.models.<type>.
    modeling_*``) and of the selected checkpoint's remote code (*remote_code*, from
    :func:`remote_code_modules`). Each captured its kernel functions (module globals, hub-kernel
    decorators) when it was imported, so a kernel installed after that is only picked up by a fresh
    process. A configuration or tokenizer module captures none, so importing one needs no restart."""
    model_types = {model_type for model_type in model_types if model_type}
    stale = []
    for name in list(sys.modules):
        parts = name.split(".")
        if parts[0] == "transformers":
            if (
                len(parts) == 4
                and parts[1] == "models"
                and parts[2] in model_types
                and parts[3].startswith("modeling_")
            ):
                stale.append(name)
        elif remote_code and _is_remote_code_module(name, remote_code):
            stale.append(name)
    return sorted(stale)


def ssm_probe_identifier(model_name: str, base: str | None = None) -> str:
    """The identifier whose architecture decides the SSM kernels. The substring match needs a real model id: a LoRA adapter id or a local checkpoint's parent folders are unrelated to its architecture (a Llama LoRA at ``user/falcon-h1-lora`` is not SSM). Prefer *base*; for a bare local checkpoint use its basename."""
    probe = base or model_name
    if probe == model_name:
        try:
            from utils.paths import is_local_path
            if is_local_path(model_name):
                probe = os.path.basename((model_name or "").rstrip("/\\")) or model_name
        except Exception:
            pass
    return probe


def _is_importable(import_name: str) -> bool:
    # Invalidate finder caches so a kernel installed earlier in this process is seen.
    importlib.invalidate_caches()
    try:
        __import__(import_name)
        return True
    except Exception as exc:
        # An ABI-incompatible kernel (undefined symbol after a torch/CUDA upgrade) raises OSError/RuntimeError, not ImportError; treat any failure as "not importable" so the caller reinstalls or source-builds instead of hard-failing on a merely broken kernel.
        logger.debug("%s is not importable (%s: %s)", import_name, type(exc).__name__, exc)
        return False


def _emit(status_cb: StatusCb, message: str) -> None:
    logger.info(message)
    if status_cb is None:
        return
    try:
        status_cb(message)
    except Exception:  # status is best-effort; never fail a load over a UI message
        logger.debug("ssm_runtime status callback raised", exc_info = True)


_hipcc_gcc_install_dir = hipcc_gcc_install_dir


# Keep quiet downloads and builds inside the orchestrator's inactivity deadline.
_HEARTBEAT_SECONDS = 60.0


@contextmanager
def _heartbeat(status_cb: StatusCb, message: str) -> Iterator[None]:
    """Emit *message* on a timer while the wrapped work runs. The inference orchestrator treats silence as a dead load, and status messages reset its inactivity deadline; prebuilt wheel installs and source builds can both stay quiet for minutes on aarch64 or slow links, so both paths use this."""
    done = threading.Event()

    def _beat() -> None:
        while not done.wait(_HEARTBEAT_SECONDS):
            _emit(status_cb, message)

    thread = threading.Thread(target = _beat, daemon = True, name = "ssm-install-heartbeat")
    thread.start()
    try:
        yield
    finally:
        done.set()
        # Wait out a tick that already left done.wait().
        thread.join(timeout = 1)


def _run_with_heartbeat(run, cmd, status_cb, display_name, **kwargs):
    """Run *cmd* via *run*, emitting a status every 60s so the parent's inactivity timeout is not tripped by a long (e.g. ROCm) source build."""
    with _heartbeat(
        status_cb,
        f"Still building {display_name} (this can take several minutes)...",
    ):
        return run(cmd, **kwargs)


def _install_kernel(
    *,
    import_name: str,
    display_name: str,
    pypi_name: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    status_cb: StatusCb,
    run: Callable[..., Any],
    newly_installed: list[str] | None = None,
) -> bool:
    """Install one kernel wheel-first, then a HIP-aware PyPI source build (never on Windows). Returns True iff importable afterwards; idempotent (no-op when already installed). An install this call made is appended to *newly_installed*."""
    if _is_importable(import_name):
        logger.info("%s already installed", display_name)
        return True

    from utils.utils import package_installs_offline

    if package_installs_offline():
        # No URL probe, no source build, and no uv-to-pip fallback that would reach the network.
        logger.info("Skipping %s installation while offline", display_name)
        return False

    env = probe_torch_wheel_env(timeout = 30, include_windows = True)
    wheel_only = sys.platform == "win32" or is_windows_wheel_env(env)
    wheel_url = direct_wheel_url(
        filename_prefix = import_name,
        package_version = package_version,
        release_tag = release_tag,
        release_base_url = release_base_url,
        env = env,
    )
    wheel_available = url_exists(wheel_url) if wheel_url else False
    if wheel_available:
        _emit(status_cb, f"Installing {display_name} (prebuilt kernel) for this model...")
        # Keep quiet downloads and unpacks within the inactivity deadline (#9398).
        with _heartbeat(
            status_cb,
            f"Still installing {display_name} (prebuilt kernel)...",
        ):
            # A cold first import can also stay quiet for tens of seconds.
            outcome = install_prebuilt(
                wheel_url,
                install = install_wheel,
                verify = lambda: _is_importable(import_name),
                on_failed = lambda installer, result: logger.warning(
                    "%s could not install %s wheel:\n%s",
                    installer,
                    display_name,
                    getattr(result, "stdout", ""),
                ),
                use_uv = bool(shutil.which("uv")),
                run = run,
            )
            if outcome == "installed":
                logger.info("Installed prebuilt %s wheel", display_name)
                if newly_installed is not None:
                    newly_installed.append(import_name)
                return True
            if outcome == "rejected":
                if wheel_only:
                    removed = (
                        run(
                            uninstall_command(pypi_name, use_uv = bool(shutil.which("uv"))),
                            stdout = subprocess.DEVNULL,
                            stderr = subprocess.DEVNULL,
                        ).returncode
                        == 0
                    )
                    logger.warning(
                        "%s wheel installed but not importable; %s",
                        display_name,
                        "removed it" if removed else f"uninstall {pypi_name} manually",
                    )
                    return False
                logger.warning(
                    "%s wheel installed but not importable; building from source",
                    display_name,
                )
    elif wheel_only:
        logger.info(
            "No prebuilt %s wheel for this Windows environment (%s); using the torch fallback",
            display_name,
            wheel_url,
        )
    elif wheel_available is None:
        _emit(
            status_cb,
            f"Could not check the {display_name} prebuilt wheel; building it from source.",
        )
    else:
        logger.info(
            "No prebuilt %s wheel for this environment (%s); building from source",
            display_name,
            wheel_url,
        )

    if wheel_only:
        return False

    # Source build (slow). ROCm has no prebuilt wheel and needs hipcc + a gcc-install-dir shim.
    spec = f"{pypi_name}=={package_version}"
    is_hip = bool((env or {}).get("hip_version"))
    if is_hip and not shutil.which("hipcc"):
        _emit(status_cb, f"{display_name}: hipcc not found; install the ROCm HIP SDK to build it.")
        return False
    _emit(
        status_cb,
        f"Building {display_name} from source for this model (this can take several minutes)...",
    )
    # Reinstall so the source build replaces a broken wheel instead of no-opping as "already satisfied".
    cmd = source_build_command(spec, use_uv = bool(shutil.which("uv")), is_hip = is_hip, reinstall = True)
    run_kwargs, _ = source_build_run_kwargs(is_hip = is_hip, gcc_install_dir = _hipcc_gcc_install_dir)
    try:
        result = _run_with_heartbeat(run, cmd, status_cb, display_name, **run_kwargs)
    except subprocess.TimeoutExpired:
        logger.error("%s source build timed out", display_name)
        _emit(status_cb, f"{display_name} source build timed out.")
        return False
    if getattr(result, "returncode", 1) != 0:
        logger.warning("%s source install failed:\n%s", display_name, getattr(result, "stdout", ""))
    importable = _is_importable(import_name)
    if importable and newly_installed is not None:
        newly_installed.append(import_name)
    return importable


def _pinned_kwargs(kernel: PinnedKernel) -> dict[str, str]:
    """_install_kernel arguments for a pinned kernel release."""
    return {
        "import_name": kernel.import_name,
        "display_name": kernel.display_name,
        "pypi_name": kernel.pypi_name,
        "package_version": kernel.package_version,
        "release_tag": kernel.release_tag,
        "release_base_url": kernel.release_base_url,
    }


def ensure_ssm_runtime(
    model_name: str,
    *,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
    wants_causal_conv1d: bool | None = None,
    is_ssm: bool | None = None,
    mamba_ssm_required: bool | None = None,
) -> list[str]:
    """Install the SSM kernels *model_name* needs before load, wheel-first; a no-op for non-SSM models and idempotent. Returns the kernels newly installed by this call. Only a model that cannot load without ``mamba_ssm`` (Nemotron-H, Falcon-H1, Granite-4.0-H) fails on it (``RuntimeError`` instead of a cryptic mid-load failure); for the rest, and for ``causal_conv1d``, a missing kernel leaves transformers' torch path."""
    # The caller's config-resolved answers win; the name match is the default.
    if is_ssm is None:
        is_ssm = model_is_ssm(model_name)
    if mamba_ssm_required is None:
        mamba_ssm_required = model_requires_mamba_ssm(model_name)
    mamba_ssm_required = bool(is_ssm and mamba_ssm_required)
    if wants_causal_conv1d is None:
        wants_causal_conv1d = model_wants_causal_conv1d(model_name)
    # Every mamba-ssm model also runs causal-conv1d.
    wants_causal_conv1d = wants_causal_conv1d or is_ssm
    installed: list[str] = []
    if not (wants_causal_conv1d or is_ssm):
        return installed

    def _install(kernel: PinnedKernel) -> bool:
        return _install_kernel(
            **_pinned_kwargs(kernel),
            status_cb = status_cb,
            run = run,
            newly_installed = installed,
        )

    # causal-conv1d first: SSM modeling files lazy-import it, and mamba-ssm's fast path uses it.
    if wants_causal_conv1d and not _install(CAUSAL_CONV1D):
        logger.warning("causal-conv1d unavailable; continuing on the model's torch fallback")

    if not is_ssm:
        return installed
    offline_hint = (
        "If this machine is offline, run `unsloth install-kernels` once while online, "
        "then reload the model."
    )
    if sys.platform == "win32" and importlib.util.find_spec("triton") is None:
        # mamba_ssm imports triton at load and the wheel goes in --no-deps.
        if mamba_ssm_required:
            raise RuntimeError(
                "mamba-ssm, required by this Mamba model, needs Triton on Windows "
                "(the triton-windows package); install it and reload the model."
            )
        _emit(
            status_cb, "mamba-ssm needs Triton on Windows (triton-windows); using the torch path."
        )
        return installed
    if not _install(MAMBA_SSM):
        if mamba_ssm_required:
            raise RuntimeError(
                f"Could not install mamba-ssm, required by this Mamba model. {offline_hint}"
            )
        _emit(status_cb, "mamba-ssm is not available; continuing on the model's torch path.")
    return installed


def provision_ssm_kernels(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
) -> tuple[SsmKernels, list[str]]:
    """Resolve and install the kernels a load needs (training, inference, export): returns what was
    resolved and what this call newly installed. Raises ``RuntimeError`` only for a model that
    cannot load without ``mamba_ssm``."""
    kernels = resolve_ssm_kernels(
        model_name, model_load_target, hf_token, adapter_base = adapter_base
    )
    installed = ensure_ssm_runtime(
        kernels.probe,
        status_cb = status_cb,
        run = run,
        wants_causal_conv1d = kernels.causal_conv1d,
        is_ssm = kernels.mamba_ssm,
        mamba_ssm_required = kernels.mamba_ssm_required,
    )
    if installed:
        refresh_transformers_kernel_gates()
    return kernels, installed
