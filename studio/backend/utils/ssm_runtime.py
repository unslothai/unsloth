# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-install the SSM/Mamba kernels a hybrid model lazy-imports during ``from_pretrained``;
the inference-path counterpart of the training worker's install.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from loggers import get_logger
from utils.kernel_install import (
    CAUSAL_CONV1D,
    MAMBA_SSM,
    PinnedKernel,
    hipcc_gcc_install_dir,
    install_prebuilt,
    source_build_command,
    source_build_run_kwargs,
)
from utils.wheel_utils import (
    direct_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)

logger = get_logger(__name__)

StatusCb = Optional[Callable[[str], None]]

# Lowercased-id substring matches, shared with the training worker. mamba-ssm models are a subset of the causal-conv1d set.
SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
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
_TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE: dict[str, bool | None] = {}


def model_is_ssm(model_name: str) -> bool:
    """Whether *model_name* is a Mamba/SSM hybrid that needs ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in SSM_MODEL_SUBSTRINGS)


def model_wants_causal_conv1d(model_name: str) -> bool:
    """Whether *model_name* needs ``causal_conv1d`` (the SSM set plus linear-attention hybrids like Qwen3-Next / LFM2 whose modeling files lazy-import it)."""
    name = (model_name or "").lower()
    return any(sub in name for sub in CAUSAL_CONV1D_MODEL_SUBSTRINGS)


def _normalized_model_identifier(value: str) -> str:
    return "".join(
        character for character in value.lower() if character.isascii() and character.isalnum()
    )


def _transformers_model_type_uses_causal_conv1d(model_type: str) -> bool | None:
    candidate = model_type.strip().lower().replace("-", "_")
    if not candidate or any(
        not (character.isascii() and (character.isalnum() or character == "_"))
        for character in candidate
    ):
        return None
    if candidate in _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE:
        return _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[candidate]

    result: bool | None = None
    try:
        import transformers
        model_dir = Path(transformers.__file__).parent / "models" / candidate
        if model_dir.is_dir():
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

    _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[candidate] = result
    return result


def model_config_wants_causal_conv1d(model_config: dict) -> bool | None:
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


def resolved_model_wants_causal_conv1d(
    model_name: str, model_load_target: str, hf_token: str | None
) -> bool:
    try:
        from utils.transformers_version import _load_config_json
        model_config = _load_config_json(model_load_target, hf_token)
    except Exception as exc:
        logger.debug("Could not inspect model config for causal-conv1d: %s", exc)
        model_config = None

    if isinstance(model_config, dict):
        requirement = model_config_wants_causal_conv1d(model_config)
        if requirement is not None:
            logger.info(
                "causal-conv1d requirement resolved from model architecture: %s",
                requirement,
            )
            return requirement
    return model_wants_causal_conv1d(model_name)


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
    importlib.invalidate_caches()
    try:
        __import__(import_name)
        return True
    except Exception as exc:
        # ABI-broken kernels raise OSError/RuntimeError, not ImportError; treat any failure as missing.
        logger.debug("%s is not importable (%s: %s)", import_name, type(exc).__name__, exc)
        return False


def _emit(status_cb: StatusCb, message: str) -> None:
    logger.info(message)
    if status_cb is None:
        return
    try:
        status_cb(message)
    except Exception:  # best-effort; never fail a load over a UI message
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
) -> bool:
    """Install one kernel wheel-first, then a HIP-aware PyPI source build. Returns True iff importable afterwards; idempotent (no-op when already installed)."""
    if _is_importable(import_name):
        logger.info("%s already installed", display_name)
        return True

    from utils.utils import hf_env_offline

    if hf_env_offline():
        logger.info("Skipping %s installation while offline", display_name)
        return False

    env = probe_torch_wheel_env(timeout = 30)
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
                return True
            if outcome == "rejected":
                logger.warning(
                    "%s wheel installed but not importable; building from source",
                    display_name,
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

    # ROCm has no prebuilt wheel and needs hipcc + a gcc-install-dir shim.
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
    return _is_importable(import_name)


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
) -> None:
    """Install the SSM kernels *model_name* needs before load, wheel-first; a no-op for non-SSM models and idempotent. Only a true SSM hybrid's ``mamba_ssm`` is fatal (raises ``RuntimeError`` instead of a cryptic mid-load failure); ``causal_conv1d`` is best-effort, with Qwen3-Next/LFM2 falling back to torch."""
    wants_causal_conv1d = model_wants_causal_conv1d(model_name)
    is_ssm = model_is_ssm(model_name)
    if not (wants_causal_conv1d or is_ssm):
        return

    # No prebuilt Windows wheel: skip causal-conv1d (mirrors training) instead of a long build.
    if wants_causal_conv1d and sys.platform == "win32":
        logger.info(
            "Skipping causal-conv1d on Windows (no prebuilt wheel); using the torch fallback"
        )
        wants_causal_conv1d = False

    # causal-conv1d first: SSM modeling files lazy-import it and mamba-ssm's fast path uses it.
    if wants_causal_conv1d and not _install_kernel(
        **_pinned_kwargs(CAUSAL_CONV1D), status_cb = status_cb, run = run
    ):
        logger.warning("causal-conv1d unavailable; continuing on the model's torch fallback")

    if is_ssm and not _install_kernel(**_pinned_kwargs(MAMBA_SSM), status_cb = status_cb, run = run):
        raise RuntimeError("Could not install mamba-ssm, required by this Mamba model.")
