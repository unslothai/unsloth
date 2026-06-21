# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-install the SSM/Mamba runtime kernels a hybrid model needs before it loads.

Mamba/SSM hybrids (Nemotron-H / Nemotron-3-Nano, Falcon-H1, Granite-4.0-H,
GraniteMoEHybrid, ...) lazily ``import mamba_ssm`` (and ``causal_conv1d``) inside
their ``modeling_*.py`` during ``from_pretrained``. When those packages are absent
the load dies with::

    mamba-ssm is required by the Mamba model but cannot be imported

The training worker already guards against this: ``core/training/worker.py`` calls
``_ensure_causal_conv1d_fast_path`` + ``_ensure_mamba_ssm`` (wheel-first) before a
fine-tune. The inference worker had no equivalent, so loading the same model for
chat failed even though training "auto-installs it". This module is the shared,
callback-based implementation the inference load path calls so both surfaces behave
the same.

Detection and pinned versions mirror the training worker; ``tests/test_ssm_runtime.py``
fails if the two copies drift.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

from loggers import get_logger
from utils.wheel_utils import (
    direct_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)

logger = get_logger(__name__)

StatusCb = Optional[Callable[[str], None]]

# Pinned wheel versions/tags -- kept in lockstep with core/training/worker.py
# (_CAUSAL_CONV1D_* / _MAMBA_SSM_*) by tests/test_ssm_runtime.py.
CAUSAL_CONV1D_PACKAGE_VERSION = "1.6.1"
CAUSAL_CONV1D_RELEASE_TAG = "v1.6.1.post4"
CAUSAL_CONV1D_RELEASE_BASE_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download"
MAMBA_SSM_PACKAGE_VERSION = "2.3.1"
MAMBA_SSM_RELEASE_TAG = "v2.3.1"
MAMBA_SSM_RELEASE_BASE_URL = "https://github.com/state-spaces/mamba/releases/download"

# Substring matches on the lowercased model id. Mirror of the training worker's
# _SSM_MODEL_SUBSTRINGS (needs mamba-ssm) and _model_wants_causal_conv1d (needs
# causal-conv1d). mamba-ssm models are a subset of the causal-conv1d set.
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
)


def model_is_ssm(model_name: str) -> bool:
    """Whether *model_name* is a Mamba/SSM hybrid that needs ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in SSM_MODEL_SUBSTRINGS)


def model_wants_causal_conv1d(model_name: str) -> bool:
    """Whether *model_name* needs ``causal_conv1d`` (the SSM set plus linear-attention
    hybrids like Qwen3-Next / LFM2 whose modeling files lazy-import it)."""
    name = (model_name or "").lower()
    return any(sub in name for sub in CAUSAL_CONV1D_MODEL_SUBSTRINGS)


def _is_importable(import_name: str) -> bool:
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _emit(status_cb: StatusCb, message: str) -> None:
    logger.info(message)
    if status_cb is None:
        return
    try:
        status_cb(message)
    except Exception:  # status is best-effort; never fail a load over a UI message
        logger.debug("ssm_runtime status callback raised", exc_info = True)


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
    """Install one kernel package, wheel-first then PyPI source build. Returns True
    iff it is importable afterwards. Idempotent: a no-op when already installed.

    The wheel-first path uses the same ``utils.wheel_utils`` primitives the training
    worker's ``_install_package_wheel_first`` uses, so a CUDA/ROCm user gets the
    identical prebuilt wheel. The source-build fallback is the standard
    ``--no-build-isolation --no-deps`` install (no ROCm gcc shimming -- that lives in
    the training worker for the fine-tune path).
    """
    if _is_importable(import_name):
        logger.info("%s already installed", display_name)
        return True

    env = probe_torch_wheel_env(timeout = 30)
    wheel_url = direct_wheel_url(
        filename_prefix = import_name,
        package_version = package_version,
        release_tag = release_tag,
        release_base_url = release_base_url,
        env = env,
    )
    if wheel_url and url_exists(wheel_url):
        _emit(status_cb, f"Installing {display_name} (prebuilt kernel) for this model...")
        for installer, result in install_wheel(
            wheel_url,
            python_executable = sys.executable,
            use_uv = bool(shutil.which("uv")),
            run = run,
        ):
            if getattr(result, "returncode", 1) == 0:
                logger.info("Installed prebuilt %s wheel", display_name)
                return True
            logger.warning(
                "%s could not install %s wheel:\n%s",
                installer,
                display_name,
                getattr(result, "stdout", ""),
            )
    else:
        logger.info(
            "No prebuilt %s wheel for this environment (%s); building from source",
            display_name,
            wheel_url,
        )

    # Source build (slow, minutes). --no-build-isolation/--no-deps mirrors the
    # training worker's non-HIP fallback.
    spec = f"{pypi_name}=={package_version}"
    _emit(
        status_cb,
        f"Building {display_name} from source for this model (this can take several minutes)...",
    )
    if shutil.which("uv"):
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-build-isolation",
            "--no-deps",
            spec,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            spec,
        ]
    result = run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    if getattr(result, "returncode", 1) != 0:
        logger.warning("%s source install failed:\n%s", display_name, getattr(result, "stdout", ""))
    return _is_importable(import_name)


def ensure_ssm_runtime(
    model_name: str,
    *,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    """Ensure the SSM kernel libraries *model_name* needs are importable before load.

    Installs ``causal_conv1d`` (and ``mamba_ssm`` for true SSM hybrids), wheel-first.
    A no-op for non-SSM models and idempotent when the kernels are already present.
    Raises ``RuntimeError`` if a required kernel cannot be installed, so the caller
    can surface a clear "couldn't prepare this model" error instead of a cryptic
    ``import mamba_ssm`` failure mid-load.
    """
    wants_causal_conv1d = model_wants_causal_conv1d(model_name)
    is_ssm = model_is_ssm(model_name)
    if not (wants_causal_conv1d or is_ssm):
        return

    # causal-conv1d first: SSM modeling files lazy-import it during from_pretrained,
    # and mamba-ssm's fast path uses it.
    if wants_causal_conv1d and not _install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = CAUSAL_CONV1D_PACKAGE_VERSION,
        release_tag = CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = CAUSAL_CONV1D_RELEASE_BASE_URL,
        status_cb = status_cb,
        run = run,
    ):
        raise RuntimeError(
            "Could not install causal-conv1d, required by this model's Mamba layers."
        )

    if is_ssm and not _install_kernel(
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        package_version = MAMBA_SSM_PACKAGE_VERSION,
        release_tag = MAMBA_SSM_RELEASE_TAG,
        release_base_url = MAMBA_SSM_RELEASE_BASE_URL,
        status_cb = status_cb,
        run = run,
    ):
        raise RuntimeError("Could not install mamba-ssm, required by this Mamba model.")
