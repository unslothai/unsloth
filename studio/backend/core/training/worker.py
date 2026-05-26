# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each training job runs in a fresh subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

Pattern follows core/data_recipe/jobs/worker.py.
"""

from __future__ import annotations

import structlog
from loggers import get_logger
import math
import os
import shutil
import sys
import time
import traceback
import subprocess as _sp
from pathlib import Path
from typing import Any, Callable

logger = get_logger(__name__)
from utils.hardware import apply_gpu_ids
from utils.wheel_utils import (
    direct_wheel_url,
    flash_attn_wheel_url,
    has_blackwell_gpu,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)


def _output_dir_from_resume_checkpoint(
    resume_from_checkpoint: str | None,
) -> str | None:
    if not resume_from_checkpoint:
        return None
    path = Path(resume_from_checkpoint)
    return str(path.parent if path.name.startswith("checkpoint-") else path)


_CAUSAL_CONV1D_RELEASE_TAG = "v1.6.1.post4"
_CAUSAL_CONV1D_PACKAGE_VERSION = "1.6.1"
_MAMBA_SSM_RELEASE_TAG = "v2.3.1"
_MAMBA_SSM_PACKAGE_VERSION = "2.3.1"
_FLASH_ATTN_RUNTIME_MIN_SEQ_LEN = 32768
_FLASH_ATTN_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL"
# apache-tvm-ffi 0.1.10/0.1.11 crash Triton with "CUDA: misaligned address" on sm_100.
_TILELANG_PACKAGE_VERSION = "0.1.8"
_APACHE_TVM_FFI_PACKAGE_VERSION = "0.1.9"
_TILELANG_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_TILELANG_INSTALL"
# Pin both so plain pip cannot silently upgrade torch under the worker (fla-core needs torch>=2.7).
_FLA_PACKAGE_VERSION = "0.5.0"
_FLA_CORE_PACKAGE_VERSION = "0.5.0"
_FLA_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FLA_INSTALL"
# `--no-deps` saves torch but loses fla-core's transitive deps; `packaging` is also undeclared upstream.
_FLA_RUNTIME_DEPS = ("einops", "packaging", "triton")
_FLA_MIN_TORCH = (2, 7)
_FLA_MIN_PYTHON = (3, 10)
# tilelang 0.1.8 ships wheels only for these Linux arches and macOS arm64; never fall back to its 93MB sdist.
_TILELANG_SUPPORTED_LINUX_MACHINES = frozenset(("x86_64", "amd64", "aarch64", "arm64"))
_TILELANG_INSTALL_TIMEOUT_S = 600
_TVM_FFI_BROKEN_VERSIONS = ("0.1.10", "0.1.11")
_FAST_PATH_HOOKS_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS"


def _set_offline_env(keys: tuple[str, ...], enabled: bool) -> dict[str, str | None]:
    if not enabled:
        return {}
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _model_local_files_only(config: dict) -> bool:
    return bool(config.get("model_known_cached") or config.get("model_local_path"))


def _dataset_local_files_only(config: dict) -> bool:
    return bool(config.get("dataset_known_cached") or config.get("dataset_local_path"))


def _untrainable_model_format_error(config: dict) -> str | None:
    model_format = str(config.get("model_format") or "").strip().lower()
    if model_format == "gguf":
        return "GGUF models are inference-only and cannot be trained."
    if model_format == "adapter":
        return "Adapter models are inference-only and cannot be trained as base models."
    return None


def _cached_dataset_training_files_for_config(config: dict, split: str) -> list[str]:
    hf_dataset = (config.get("hf_dataset") or "").strip()
    local_path = config.get("dataset_local_path")
    if not hf_dataset or not local_path:
        return []
    from utils.datasets.cache_paths import cached_dataset_training_files

    return cached_dataset_training_files(
        hf_dataset,
        local_path,
        subset = config.get("subset") or None,
        train_split = split or "train",
    )


def _resolve_cached_model_load_name(config: dict) -> str:
    model_name = config["model_name"]
    if not _model_local_files_only(config):
        return model_name
    local_path = config.get("model_local_path")
    if local_path:
        try:
            from utils import hf_cache_scan

            snapshot = hf_cache_scan.latest_snapshot_from_cache_path(
                local_path,
                "model",
                model_name,
                ("config.json", "adapter_config.json"),
            )
            if snapshot:
                return snapshot
        except Exception:
            pass
    try:
        from utils.models.model_config import _cached_transformers_snapshot_path

        return _cached_transformers_snapshot_path(model_name) or model_name
    except Exception:
        return model_name


def _model_wants_causal_conv1d(model_name: str) -> bool:
    name = model_name.lower()
    return any(
        key in name
        for key in (
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
    )


def _hipcc_gcc_install_dir() -> str | None:
    """Return the highest-numbered ``/usr/lib/gcc/x86_64-linux-gnu/<N>`` that has
    BOTH the gcc runtime dir AND the corresponding ``/usr/include/c++/<N>`` C++
    headers, or ``None`` if no match (or non-Linux / non-x86_64).

    Ubuntu 24.04 ships ``/usr/lib/gcc/x86_64-linux-gnu/14/`` (gcc-14 runtime
    objects) but does NOT ship ``/usr/include/c++/14`` in its default apt set;
    libstdc++ headers come from ``libstdc++-13-dev``. ROCm clang-20 picks the
    highest-numbered runtime dir by default, finds no ``<cstdlib>``, and the
    HIP source build fails with::

        /opt/rocm-X.Y/lib/llvm/lib/clang/20/include/__clang_hip_runtime_wrapper.h:112:10:
          fatal error: 'cstdlib' file not found

    Returning a path lets the caller pass ``--gcc-install-dir=<path>`` to clang
    via ``HIPCC_COMPILE_FLAGS_APPEND``. Mirrors the same loop ``bbf004c`` added
    to ``studio/setup.sh`` for the llama.cpp HIP build branch (PR #5301).
    """
    if not sys.platform.startswith("linux"):
        return None
    import platform as _platform

    if _platform.machine().lower() != "x86_64":
        return None
    for _ver in (14, 13, 12, 11):
        _runtime = f"/usr/lib/gcc/x86_64-linux-gnu/{_ver}/include"
        _headers = f"/usr/include/c++/{_ver}"
        if os.path.isdir(_runtime) and os.path.isdir(_headers):
            return f"/usr/lib/gcc/x86_64-linux-gnu/{_ver}"
    return None


def _install_package_wheel_first(
    *,
    event_queue: Any,
    import_name: str,
    display_name: str,
    pypi_name: str,
    pypi_version: str | None = None,
    filename_prefix: str | None = None,
    release_tag: str | None = None,
    release_base_url: str | None = None,
    wheel_url_builder: Callable[[dict[str, str] | None], str | None] | None = None,
    pypi_spec: str | None = None,
    pypi_status_message: str | None = None,
) -> bool:
    try:
        __import__(import_name)
        logger.info("%s already installed", display_name)
        return True
    except ImportError:
        pass

    env = probe_torch_wheel_env(timeout = 30)
    if wheel_url_builder is not None:
        wheel_url = wheel_url_builder(env)
    else:
        wheel_url = direct_wheel_url(
            filename_prefix = filename_prefix,
            package_version = pypi_version,
            release_tag = release_tag,
            release_base_url = release_base_url,
            env = env,
        )

    if wheel_url is None:
        logger.info("No compatible %s wheel candidate", display_name)
    elif url_exists(wheel_url):
        _send_status(event_queue, f"Installing {display_name} for faster training...")
        for installer, result in install_wheel(
            wheel_url,
            python_executable = sys.executable,
            use_uv = bool(shutil.which("uv")),
            run = _sp.run,
        ):
            if result.returncode == 0:
                logger.info("Installed prebuilt %s wheel successfully", display_name)
                return True
            logger.warning(
                "%s failed to install %s wheel:\n%s",
                installer,
                display_name,
                result.stdout,
            )
    else:
        logger.info("No published %s wheel found: %s", display_name, wheel_url)

    is_hip = env and env.get("hip_version")
    if is_hip and not shutil.which("hipcc"):
        logger.error(
            "%s requires hipcc for source compilation on ROCm. "
            "Install the ROCm HIP SDK: https://rocm.docs.amd.com",
            display_name,
        )
        _send_status(
            event_queue,
            f"{display_name}: hipcc not found (ROCm HIP SDK required)",
        )
        return False

    if pypi_spec is None:
        pypi_spec = f"{pypi_name}=={pypi_version}"

    if pypi_status_message is None:
        if is_hip:
            pypi_status_message = (
                f"Compiling {display_name} from source for ROCm "
                "(this may take several minutes)..."
            )
        else:
            pypi_status_message = (
                f"Installing {display_name} from PyPI for faster training..."
            )

    _send_status(event_queue, pypi_status_message)

    # Prefer uv for faster dependency resolution when available
    plain_pypi_install = pypi_version is None
    if plain_pypi_install:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                pypi_spec,
            ]
        else:
            pypi_cmd = [sys.executable, "-m", "pip", "install", pypi_spec]
    else:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-build-isolation",
                "--no-deps",
            ]
            # Avoid stale cache artifacts from partial HIP source builds
            if is_hip:
                pypi_cmd.append("--no-cache")
            pypi_cmd.append(pypi_spec)
        else:
            pypi_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--no-cache-dir",
                pypi_spec,
            ]

    # Source compilation on ROCm can take 10-30 minutes; use a generous
    # timeout. Non-HIP installs preserve the pre-existing "no timeout"
    # behaviour so unrelated slow installs (e.g. causal-conv1d source
    # build on Linux aarch64 or unsupported torch/CUDA combinations)
    # are not aborted at 5 minutes by this PR.
    _run_kwargs: dict[str, Any] = {
        "stdout": _sp.PIPE,
        "stderr": _sp.STDOUT,
        "text": True,
    }
    if is_hip:
        _run_kwargs["timeout"] = 1800
        # On Ubuntu 24.04 + ROCm clang-20, the HIP source build (causal-conv1d,
        # mamba-ssm source fallback, flash-attn source fallback) defaults to
        # /usr/lib/gcc/x86_64-linux-gnu/14/ which has the runtime dir but no
        # /usr/include/c++/14 headers, and dies at:
        #   __clang_hip_runtime_wrapper.h:112:10:
        #     fatal error: 'cstdlib' file not found
        # Inject --gcc-install-dir for a gcc whose C++ headers actually exist.
        # Respect any pre-existing --gcc-install-dir in HIPCC_COMPILE_FLAGS_APPEND
        # (user knows best); otherwise append. Mirrors the same fix bbf004c
        # added to studio/setup.sh for the llama.cpp HIP build (PR #5301).
        _existing_flags = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "")
        if "--gcc-install-dir" not in _existing_flags:
            _gcc_dir = _hipcc_gcc_install_dir()
            if _gcc_dir is not None:
                _appended = (f"{_existing_flags} --gcc-install-dir={_gcc_dir}").strip()
                _env = _run_kwargs.get("env", os.environ).copy()
                _env["HIPCC_COMPILE_FLAGS_APPEND"] = _appended
                _run_kwargs["env"] = _env
                logger.info(
                    "HIP source build for %s: appended "
                    "--gcc-install-dir=%s to HIPCC_COMPILE_FLAGS_APPEND",
                    display_name,
                    _gcc_dir,
                )

    try:
        result = _sp.run(pypi_cmd, **_run_kwargs)
    except _sp.TimeoutExpired:
        logger.error(
            "%s installation timed out after %ds",
            display_name,
            _run_kwargs.get("timeout"),
        )
        _send_status(
            event_queue,
            f"{display_name} installation timed out after "
            f"{_run_kwargs.get('timeout')}s",
        )
        return False

    if result.returncode != 0:
        if is_hip:
            # Surface a clear error for ROCm source build failures
            error_lines = (result.stdout or "").strip().splitlines()
            snippet = "\n".join(error_lines[-5:]) if error_lines else "(no output)"
            logger.error(
                "Failed to compile %s for ROCm:\n%s",
                display_name,
                result.stdout,
            )
            _send_status(
                event_queue,
                f"Failed to compile {display_name} for ROCm. "
                "Check that hipcc and ROCm development headers are installed.\n"
                f"{snippet}",
            )
        else:
            logger.error(
                "Failed to install %s from PyPI:\n%s",
                display_name,
                result.stdout,
            )
        return False

    if is_hip:
        logger.info("Compiled and installed %s from source for ROCm", display_name)
    else:
        logger.info("Installed %s from PyPI", display_name)
    return True


def _ensure_causal_conv1d_fast_path(event_queue: Any, model_name: str) -> None:
    if not _model_wants_causal_conv1d(model_name):
        return

    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


def _installed_torch_version_tuple() -> tuple[int, int] | None:
    """Return ``(major, minor)`` of the installed torch, else None."""
    try:
        from importlib.metadata import version as _pkg_version

        raw = _pkg_version("torch").split("+", 1)[0]
        parts = raw.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def _flash_linear_attention_importable() -> bool:
    """Catch any exception (not just ImportError) so a broken native lib doesn't abort the worker."""
    try:
        import fla.modules  # noqa: F401
        import fla.ops.gated_delta_rule  # noqa: F401

        return True
    except Exception as exc:
        logger.warning(
            "flash-linear-attention is not importable; continuing with install/fallback: %s",
            exc,
        )
        return False


def _flash_linear_attention_current(already_importable: bool | None = None) -> bool:
    """True iff FLA imports AND is at the pinned version (older FLA lacks gated_delta_rule kernels)."""
    if already_importable is None:
        already_importable = _flash_linear_attention_importable()
    if not already_importable:
        return False
    try:
        from importlib.metadata import version as _pkg_version
        from packaging.version import Version

        fla_v = Version(_pkg_version("flash-linear-attention"))
        core_v = Version(_pkg_version("fla-core"))
        return fla_v >= Version(_FLA_PACKAGE_VERSION) and core_v >= Version(
            _FLA_CORE_PACKAGE_VERSION
        )
    except Exception as exc:
        logger.warning(
            "flash-linear-attention importable but version check failed; treating as stale: %s",
            exc,
        )
        return False


def _ensure_flash_linear_attention_unconditional(event_queue: Any) -> bool:
    """Install pinned FLA + fla-core with --no-deps. Returns True iff importable post-call."""
    if os.getenv(_FLA_SKIP_ENV) == "1":
        return False
    if sys.version_info < _FLA_MIN_PYTHON:
        logger.info(
            "Skipping flash-linear-attention install: requires Python >= %d.%d, have %s",
            _FLA_MIN_PYTHON[0],
            _FLA_MIN_PYTHON[1],
            sys.version.split()[0],
        )
        return False
    torch_ver = _installed_torch_version_tuple()
    if torch_ver is not None and torch_ver < _FLA_MIN_TORCH:
        _send_status(
            event_queue,
            (
                f"Skipping flash-linear-attention install: fla-core requires "
                f"torch>={_FLA_MIN_TORCH[0]}.{_FLA_MIN_TORCH[1]}, have "
                f"{torch_ver[0]}.{torch_ver[1]}"
            ),
        )
        return False

    # Probe once; reuse result so the --force-reinstall decision and the short-circuit
    # share the same call count (stable for tests).
    already_importable = _flash_linear_attention_importable()
    if already_importable and _flash_linear_attention_current(already_importable = True):
        logger.info("flash-linear-attention already importable at the pinned version")
        return True

    _send_status(
        event_queue,
        f"Installing flash-linear-attention=={_FLA_PACKAGE_VERSION} for faster training...",
    )

    # `--no-deps` blocks the silent torch upgrade; we bring the non-torch runtime deps in by hand.
    specs = [
        *_FLA_RUNTIME_DEPS,
        f"fla-core=={_FLA_CORE_PACKAGE_VERSION}",
        f"flash-linear-attention=={_FLA_PACKAGE_VERSION}",
    ]
    extra_args = ["--no-deps"]
    if already_importable:
        # Older FLA already imported; pip skips reinstall without this flag.
        extra_args.append("--force-reinstall")

    if shutil.which("uv"):
        pypi_cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            *extra_args,
            *specs,
        ]
    else:
        pypi_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *extra_args,
            *specs,
        ]

    try:
        result = _sp.run(
            pypi_cmd,
            stdout = _sp.PIPE,
            stderr = _sp.STDOUT,
            text = True,
            timeout = _TILELANG_INSTALL_TIMEOUT_S,
        )
    except _sp.TimeoutExpired:
        logger.warning("flash-linear-attention install timed out; continuing")
        _send_status(
            event_queue, "flash-linear-attention install timed out; continuing"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "flash-linear-attention install failed (continuing on torch fallback):\n%s",
            result.stdout,
        )
        _send_status(
            event_queue,
            "flash-linear-attention install failed; continuing without it",
        )
        return False

    # pip can exit 0 with a missing transitive runtime dep; verify the import.
    if not _flash_linear_attention_importable():
        _send_status(
            event_queue,
            "flash-linear-attention installed but is not importable; continuing without it",
        )
        return False

    logger.info("Installed flash-linear-attention for the FLA fast path")
    return True


def _ensure_flash_linear_attention(event_queue: Any, model_name: str) -> None:
    """Legacy model-name-gated FLA install, used when UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1."""
    if not _model_wants_tilelang(model_name):
        return
    _ensure_flash_linear_attention_unconditional(event_queue)


_SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
)


def _ensure_mamba_ssm(event_queue: Any, model_name: str) -> None:
    if not any(sub in model_name.lower() for sub in _SSM_MODEL_SUBSTRINGS):
        return

    logger.info("SSM model detected; setting up mamba-ssm after causal-conv1d")
    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = _MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = _MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


# Auto-derived from installed transformers: model_types whose modeling_*.py imports `from fla.*`.
# Cached per process. Empty when transformers can't be inspected -> we skip tilelang pre-install
# (the FLA Triton path still runs via the runtime hook).
_TRANSFORMERS_FLA_MODEL_TYPES_CACHE: frozenset[str] | None = None
_MODEL_NAME_SEP_CHARS = ("-", ".", "/", " ")


def _discover_fla_model_types() -> frozenset[str]:
    """Model_types in the installed transformers whose modeling file imports `from fla.*`."""
    global _TRANSFORMERS_FLA_MODEL_TYPES_CACHE
    if _TRANSFORMERS_FLA_MODEL_TYPES_CACHE is not None:
        return _TRANSFORMERS_FLA_MODEL_TYPES_CACHE
    found: set[str] = set()
    try:
        import transformers

        models_root = Path(transformers.__file__).parent / "models"
        for modeling in models_root.glob("*/modeling_*.py"):
            try:
                src = modeling.read_text(encoding = "utf-8", errors = "ignore")
            except OSError:
                continue
            if "from fla." in src:
                found.add(modeling.parent.name)
    except Exception as exc:
        logger.debug("FLA model-type discovery skipped: %s", exc)
    _TRANSFORMERS_FLA_MODEL_TYPES_CACHE = frozenset(found)
    return _TRANSFORMERS_FLA_MODEL_TYPES_CACHE


def _model_wants_tilelang(model_name: str) -> bool:
    """True iff model_name normalizes to contain a discovered FLA model_type."""
    types = _discover_fla_model_types()
    if not types:
        return False
    name = model_name.lower()
    for sep in _MODEL_NAME_SEP_CHARS:
        name = name.replace(sep, "_")
    return any(t in name for t in types)


def _installed_tvm_ffi_version() -> str | None:
    """Installed apache-tvm-ffi version, or None if missing/unimportable."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("apache-tvm-ffi")
    except Exception:
        return None


def _tilelang_importable() -> bool:
    """Catch any exception (not just ImportError) so a broken native lib doesn't abort the worker."""
    try:
        import tilelang  # noqa: F401
        import tvm_ffi  # noqa: F401

        return True
    except Exception as exc:
        logger.warning(
            "tilelang/tvm_ffi is not importable; continuing with install/fallback: %s",
            exc,
        )
        return False


def _torch_has_hip() -> bool:
    """True iff torch is a ROCm build; `torch.version.hip` is the only reliable signal on x86_64 ROCm."""
    try:
        import torch as _torch

        return getattr(_torch.version, "hip", None) is not None
    except Exception:
        return False


def _tilelang_platform_supported() -> bool:
    """True iff a tilelang 0.1.8 wheel will load: Linux x86_64/aarch64, non-HIP torch.

    HIP excluded because tilelang 0.1.8 has no HIP GEMM instruction and crashes mid-backward.
    """
    import platform as _platform

    if not sys.platform.startswith("linux"):
        return False
    if _platform.machine().lower() not in _TILELANG_SUPPORTED_LINUX_MACHINES:
        return False
    if _torch_has_hip():
        return False
    return True


def _pip_install_cmd(*args: str) -> list[str]:
    """`uv pip install` if uv is on PATH, else `python -m pip install`."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", *args]


def _run_pip(cmd: list[str], event_queue: Any, label: str) -> bool:
    """Run a pip install and surface success/failure via status events."""
    try:
        result = _sp.run(
            cmd,
            stdout = _sp.PIPE,
            stderr = _sp.STDOUT,
            text = True,
            timeout = _TILELANG_INSTALL_TIMEOUT_S,
        )
    except _sp.TimeoutExpired:
        logger.warning("%s install timed out; continuing", label)
        _send_status(event_queue, f"{label} install timed out; continuing")
        return False
    if result.returncode != 0:
        logger.warning(
            "%s install failed (continuing without it):\n%s", label, result.stdout
        )
        _send_status(event_queue, f"{label} install failed; continuing")
        return False
    return True


def _ensure_tilelang_backend_unconditional(event_queue: Any) -> bool:
    """Install pinned tilelang + apache-tvm-ffi; two-step repair if a broken tvm-ffi is present.

    Returns True iff both import post-call. Step 1 surgically downgrades a broken tvm-ffi
    with --force-reinstall --no-deps so torch / CUDA stay untouched; step 2 is a regular
    install for missing transitive deps. Bypass via UNSLOTH_STUDIO_SKIP_TILELANG_INSTALL=1.
    """
    if os.getenv(_TILELANG_SKIP_ENV) == "1":
        return False
    if sys.version_info < _FLA_MIN_PYTHON:
        logger.info(
            "Skipping tilelang install: requires Python >= %d.%d, have %s",
            _FLA_MIN_PYTHON[0],
            _FLA_MIN_PYTHON[1],
            sys.version.split()[0],
        )
        return False
    if not _tilelang_platform_supported():
        import platform as _platform

        logger.info(
            "Skipping tilelang install: no prebuilt wheel for %s/%s",
            sys.platform,
            _platform.machine(),
        )
        return False

    existing_tvm_ffi = _installed_tvm_ffi_version()
    needs_repair = existing_tvm_ffi in _TVM_FFI_BROKEN_VERSIONS

    if not needs_repair and _tilelang_importable():
        logger.info("tilelang + apache-tvm-ffi already installed")
        return True

    # Step 1: --no-deps keeps --force-reinstall from touching torch/CUDA via the dep graph.
    if needs_repair:
        logger.info(
            "Forcing apache-tvm-ffi downgrade: %s is on the broken list",
            existing_tvm_ffi,
        )
        _send_status(
            event_queue,
            (
                f"Downgrading apache-tvm-ffi {existing_tvm_ffi} -> "
                f"{_APACHE_TVM_FFI_PACKAGE_VERSION} (broken-versions list)"
            ),
        )
        repair_cmd = _pip_install_cmd(
            "--only-binary=:all:",
            "--force-reinstall",
            "--no-deps",
            f"apache-tvm-ffi=={_APACHE_TVM_FFI_PACKAGE_VERSION}",
        )
        if not _run_pip(repair_cmd, event_queue, "TileLang backend repair"):
            return False

    # Step 2: regular install pulls in transitive deps (z3-solver, ml-dtypes) without touching torch.
    _send_status(
        event_queue,
        f"Installing TileLang=={_TILELANG_PACKAGE_VERSION} for faster training...",
    )
    install_cmd = _pip_install_cmd(
        "--only-binary=:all:",
        f"apache-tvm-ffi=={_APACHE_TVM_FFI_PACKAGE_VERSION}",
        f"tilelang=={_TILELANG_PACKAGE_VERSION}",
    )
    if not _run_pip(install_cmd, event_queue, "TileLang backend"):
        return False

    # pip can exit 0 while a native lib (libz3.so) is missing; verify the import.
    if not _tilelang_importable():
        _send_status(
            event_queue,
            "TileLang backend installed but is not importable; continuing on the FLA Triton path",
        )
        return False

    logger.info("Installed TileLang backend for FLA fast path")
    return True


def _ensure_tilelang_backend(event_queue: Any, model_name: str) -> None:
    """Legacy substring-gated tilelang installer (opt-out path)."""
    if not _model_wants_tilelang(model_name):
        return
    _ensure_tilelang_backend_unconditional(event_queue)


# ── Fast-path hooks ──
# Wrap transformers' is_{flash_linear_attention,causal_conv1d}_available so the first call
# (at modeling import time) drives the install. Any model that queries the gate gets the
# install; models that never query it (Llama, Gemma, dense Qwen) pay nothing.
# UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the legacy substring path.


def _rebind_in_already_imported_modules(
    *, attr_name: str, old_obj: Any, new_obj: Any
) -> int:
    """Rebind `attr_name -> new_obj` in every module that already imported `old_obj`.

    `from X import Y` creates a local binding that reassigning X.Y won't reach.
    Uses `__dict__.get` (not `getattr`) to skip lazy `__getattr__` aliases.
    """
    count = 0
    missing = object()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        module_dict = getattr(mod, "__dict__", None)
        if not isinstance(module_dict, dict):
            continue
        existing = module_dict.get(attr_name, missing)
        if existing is old_obj:
            try:
                setattr(mod, attr_name, new_obj)
                count += 1
            except Exception as exc:
                logger.debug("Could not rebind %s in %s: %s", attr_name, mod_name, exc)
    return count


def _install_fast_path_hooks(event_queue: Any, model_name: str) -> None:
    """Hook transformers' is_*_available gates so the first call drives the install.

    Idempotent. UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring gate.
    """
    if os.getenv(_FAST_PATH_HOOKS_SKIP_ENV) == "1":
        logger.info("Fast-path hooks disabled via env; using substring fallback")
        return

    # On HIP torch, even already-installed tilelang crashes FLA's TileLang dispatch.
    # User can override with FLA_TILELANG=1.
    if _torch_has_hip() and os.environ.get("FLA_TILELANG") is None:
        os.environ["FLA_TILELANG"] = "0"
        logger.info(
            "HIP/ROCm torch detected; setting FLA_TILELANG=0 (no HIP GEMM in tilelang 0.1.8)"
        )

    try:
        from transformers.utils import import_utils as _iu
    except Exception as exc:
        logger.warning(
            "transformers.utils.import_utils not importable; skipping fast-path hooks: %s",
            exc,
        )
        return

    def _make_wrapper(
        original: Callable[[], bool],
        install_fn: Callable[[Any], bool],
        gate_name: str,
        post_available_fn: Callable[[Any], None] | None = None,
    ) -> Callable[[], bool]:
        state = {"installed": False}

        def wrapper() -> bool:
            if state["installed"]:
                return original()
            try:
                original.cache_clear()  # defensive; worker subprocess is fresh
            except AttributeError:
                pass
            ok = original()
            ran_install = False
            if not ok:
                ran_install = True
                logger.info("Hook fired for %s; triggering install", gate_name)
                try:
                    ok = bool(install_fn(event_queue))
                except Exception as exc:
                    logger.warning(
                        "%s install raised: %s; falling back to torch", gate_name, exc
                    )
                    ok = False
                logger.info("%s hook done; available=%s", gate_name, ok)
            # post_available_fn handles "gate already True but ancillary kernel broken" (e.g. tilelang
            # missing while FLA imports fine); skip when install_fn already chained the follow-up.
            if ok and not ran_install and post_available_fn is not None:
                try:
                    post_available_fn(event_queue)
                except Exception as exc:
                    logger.warning(
                        "%s post-available step raised: %s; continuing", gate_name, exc
                    )
            state["installed"] = True
            return ok

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        wrapper.cache_clear = getattr(original, "cache_clear", lambda: None)  # type: ignore[attr-defined]
        return wrapper

    def _fla_install(eq: Any) -> bool:
        # FLA alone ~2.35x; +tilelang adds ~26%. tilelang is GDN-only (Qwen3.5 family).
        if not _ensure_flash_linear_attention_unconditional(eq):
            logger.info(
                "FLA install did not produce an importable runtime; skipping TileLang"
            )
            return False
        if _model_wants_tilelang(model_name):
            _ensure_tilelang_backend_unconditional(eq)
        else:
            logger.info(
                "Model %r outside TileLang allowlist; FLA Triton path is sufficient",
                model_name,
            )
        return True

    def _fla_post_available(eq: Any) -> None:
        # FLA already imports; repair tilelang if missing or on the broken tvm-ffi list.
        if not _model_wants_tilelang(model_name):
            return
        if (
            _installed_tvm_ffi_version() not in _TVM_FFI_BROKEN_VERSIONS
            and _tilelang_importable()
        ):
            return
        _ensure_tilelang_backend_unconditional(eq)

    def _causal_conv1d_install(eq: Any) -> bool:
        ok = _install_package_wheel_first(
            event_queue = eq,
            import_name = "causal_conv1d",
            display_name = "causal-conv1d",
            pypi_name = "causal-conv1d",
            pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
            filename_prefix = "causal_conv1d",
            release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
            release_base_url = (
                "https://github.com/Dao-AILab/causal-conv1d/releases/download"
            ),
        )
        return bool(ok)

    for gate_name, install_fn, post_fn in (
        ("is_flash_linear_attention_available", _fla_install, _fla_post_available),
        ("is_causal_conv1d_available", _causal_conv1d_install, None),
    ):
        original = getattr(_iu, gate_name, None)
        if original is None:
            logger.info(
                "%s missing on transformers.utils.import_utils; skipping hook",
                gate_name,
            )
            continue
        wrapped = _make_wrapper(original, install_fn, gate_name, post_fn)
        setattr(_iu, gate_name, wrapped)
        rebound = _rebind_in_already_imported_modules(
            attr_name = gate_name, old_obj = original, new_obj = wrapped
        )
        logger.info(
            "Installed fast-path hook on %s (rebound %d modules)", gate_name, rebound
        )


def _should_try_runtime_flash_attn_install(max_seq_length: int) -> bool:
    if os.getenv(_FLASH_ATTN_SKIP_ENV) == "1":
        return False
    if max_seq_length < _FLASH_ATTN_RUNTIME_MIN_SEQ_LEN:
        return False
    return sys.platform.startswith("linux")


def _ensure_flash_attn_for_long_context(event_queue: Any, max_seq_length: int) -> None:
    if not _should_try_runtime_flash_attn_install(max_seq_length):
        return
    if has_blackwell_gpu():
        _send_status(
            event_queue,
            "Skipping flash-attn install: Blackwell GPU detected (sm_100+); no compatible prebuilt wheel",
        )
        return

    installed = _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "flash_attn",
        display_name = "flash-attn",
        pypi_name = "flash-attn",
        wheel_url_builder = flash_attn_wheel_url,
        pypi_spec = "flash-attn",
        pypi_status_message = "Installing flash-attn from PyPI for long-context training...",
    )
    if not installed:
        _send_status(event_queue, "Continuing without flash-attn")


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name)


def _adapt_for_mlx_vlm(items):
    """Adapt GPU-path VLM dataset output for mlx-vlm consumption.

    The GPU path embeds PIL images inside messages content as
    {"type": "image", "image": PIL_Image}. mlx-vlm's prepare_inputs
    needs images at top-level to produce pixel_values — regardless of
    model type. Extract them and leave bare {"type": "image"} placeholders.
    """
    adapted = []
    for item in items:
        images = []
        messages = []
        for msg in item.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part.get("image")
                        if img is not None:
                            images.append(img)
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(part)
                messages.append({"role": msg["role"], "content": new_content})
            else:
                messages.append(msg)
        out = {"messages": messages}
        if images:
            out["image"] = images[0] if len(images) == 1 else images
        elif "image" in item:
            out["image"] = item["image"]
        elif "images" in item:
            out["images"] = item["images"]
        adapted.append(out)
    return adapted


_MLX_STUDIO_OPTIM_MAP = {
    "adamw_8bit": "adamw",
    "paged_adamw_8bit": "adamw",
    "adamw_bnb_8bit": "adamw",
    "paged_adamw_32bit": "adamw",
    "adamw_torch": "adamw",
    "adamw_torch_fused": "adamw",
    "adamw": "adamw",
    "adafactor": "adafactor",
    "sgd": "sgd",
    "adam": "adam",
    "muon": "muon",
    "lion": "lion",
}
_MLX_STUDIO_LR_SCHEDULERS = {"linear", "cosine", "constant"}


def _normalize_mlx_studio_optimizer(value):
    raw = str(value or "adamw_8bit").strip().lower()
    try:
        return _MLX_STUDIO_OPTIM_MAP[raw]
    except KeyError:
        supported = ", ".join(sorted(_MLX_STUDIO_OPTIM_MAP))
        raise ValueError(
            f"Unsupported optimizer for MLX training: {value!r}. "
            f"Supported values: {supported}."
        )


def _normalize_mlx_studio_scheduler(value):
    raw = str(value or "linear").strip().lower()
    if raw not in _MLX_STUDIO_LR_SCHEDULERS:
        supported = ", ".join(sorted(_MLX_STUDIO_LR_SCHEDULERS))
        raise ValueError(
            f"Unsupported LR scheduler for MLX training: {value!r}. "
            f"Supported values: {supported}."
        )
    return raw


def _resolve_mlx_local_dataset_files(file_paths: list) -> list[str]:
    """Resolve Studio local dataset uploads without importing the GPU trainer."""
    from utils.paths import resolve_dataset_path

    all_files: list[str] = []
    for dataset_file in file_paths or []:
        file_path = (
            dataset_file
            if os.path.isabs(dataset_file)
            else str(resolve_dataset_path(dataset_file))
        )
        file_path_obj = Path(file_path)

        if file_path_obj.is_dir():
            parquet_dir = (
                file_path_obj / "parquet-files"
                if (file_path_obj / "parquet-files").exists()
                else file_path_obj
            )
            parquet_files = sorted(parquet_dir.glob("*.parquet"))
            if parquet_files:
                all_files.extend(str(p) for p in parquet_files)
                continue

            candidates: list[Path] = []
            for ext in (".json", ".jsonl", ".csv", ".parquet"):
                candidates.extend(sorted(file_path_obj.glob(f"*{ext}")))
            if candidates:
                all_files.extend(str(c) for c in candidates)
                continue

            raise ValueError(f"No supported data files in directory: {file_path_obj}")

        all_files.append(str(file_path_obj))

    return all_files


def _mlx_local_dataset_loader_for_files(files: list[str]) -> str:
    first_ext = Path(files[0]).suffix.lower()
    if first_ext in (".json", ".jsonl"):
        return "json"
    if first_ext == ".csv":
        return "csv"
    if first_ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported dataset format: {files[0]}")


def _run_mlx_training(event_queue, stop_queue, config):
    """Self-contained MLX training path for Apple Silicon.

    Uses MLXTrainer from unsloth_zoo directly -- no torch/SFTTrainer needed.
    Mirrors the event_queue protocol so the parent process pump works unchanged.
    """
    import time
    import gc
    import math
    import threading
    import queue as _queue
    from pathlib import Path

    def _send(event_type, **kwargs):
        if event_type == "status" and "message" not in kwargs:
            sm = kwargs.get("status_message")
            if sm is not None:
                kwargs["message"] = sm
        event_queue.put({"type": event_type, "ts": time.time(), **kwargs})

    _send("status", status_message = "Loading MLX libraries...")

    import mlx.core as mx

    try:
        from unsloth_zoo.mlx.loader import FastMLXModel
        from unsloth_zoo.mlx.trainer import (
            MLXTrainer,
            MLXTrainingConfig,
            train_on_responses_only,
        )
    except ImportError as e:
        raise ImportError(
            "Unsloth: MLX training requires unsloth-zoo with the MLX modules "
            "(unsloth_zoo.mlx.loader / unsloth_zoo.mlx.trainer). Reinstall via "
            "install.sh on Apple Silicon."
        ) from e
    from datasets import load_dataset

    if mx.metal.is_available():
        info = mx.device_info()
        rec_bytes = info.get("max_recommended_working_set_size", 0) or 0
        if rec_bytes > 0:
            memory_cap = int(rec_bytes * 0.85)
            wired_cap = min(int(rec_bytes), memory_cap)
            mx.set_memory_limit(memory_cap)
            mx.set_wired_limit(wired_cap)

    model_name = config["model_name"]
    model_load_name = _resolve_cached_model_load_name(config)
    model_local_only = _model_local_files_only(config)
    dataset_local_only = _dataset_local_files_only(config)
    hf_token = config.get("hf_token") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    if config.get("use_loftq"):
        message = "LoftQ is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)

    optim_name = _normalize_mlx_studio_optimizer(config.get("optim", "adamw_8bit"))
    lr_scheduler_type = _normalize_mlx_studio_scheduler(
        config.get("lr_scheduler_type", "linear")
    )

    # ── 1. Load model ──
    # Force text-only if the dataset is not an image dataset, even if the model
    # has vision capabilities (e.g. Qwen3.5-VL trained on plain alpaca text).
    _send("status", status_message = f"Loading {model_name}...")
    is_dataset_image = bool(config.get("is_dataset_image", False))
    training_type = config.get("training_type", "LoRA/QLoRA")
    use_lora = training_type == "LoRA/QLoRA"
    previous_model_env = _set_offline_env(
        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"),
        model_local_only,
    )
    try:
        model, tokenizer = FastMLXModel.from_pretrained(
            model_load_name,
            load_in_4bit = config.get("load_in_4bit", True),
            full_finetuning = not use_lora,
            text_only = None if is_dataset_image else True,
            token = hf_token,
            trust_remote_code = bool(config.get("trust_remote_code", False)),
            random_state = config.get("random_seed", 3407),
        )
    finally:
        _restore_env(previous_model_env)

    is_vlm = bool(is_dataset_image and getattr(model, "_is_vlm_model", False))
    model._is_vlm_model = is_vlm

    # ── 2. Apply LoRA / full FT ──
    # Pass gradient_checkpointing as string ("mlx"/"unsloth"/"none"/etc.)
    # get_peft_model and MLXTrainer both accept strings and handle them.
    gc_setting = config.get("gradient_checkpointing", "mlx")
    if isinstance(gc_setting, str):
        use_grad_checkpoint = (
            gc_setting if gc_setting.lower() not in ("false", "") else False
        )
    else:
        use_grad_checkpoint = gc_setting

    if use_lora:
        _send("status", status_message = "Configuring LoRA adapters...")
        peft_kwargs = dict(
            r = config.get("lora_r", 16),
            lora_alpha = config.get("lora_alpha", 16),
            lora_dropout = config.get("lora_dropout", 0.0),
            use_rslora = config.get("use_rslora", False),
            init_lora_weights = config.get("init_lora_weights", True),
            random_state = config.get("random_seed", 3407),
            target_modules = config.get("target_modules")
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing = use_grad_checkpoint,
        )
        finetune_language = config.get("finetune_language_layers", True)
        finetune_attention = config.get("finetune_attention_modules", True)
        finetune_mlp = config.get("finetune_mlp_modules", True)
        finetune_vision = (
            config.get("finetune_vision_layers", False) if is_vlm else False
        )

        if (
            (finetune_attention or finetune_mlp)
            and not finetune_language
            and not finetune_vision
        ):
            finetune_language = True

        peft_kwargs["finetune_language_layers"] = finetune_language
        peft_kwargs["finetune_attention_modules"] = finetune_attention
        peft_kwargs["finetune_mlp_modules"] = finetune_mlp
        if is_vlm:
            peft_kwargs["finetune_vision_layers"] = finetune_vision
        model = FastMLXModel.get_peft_model(model, **peft_kwargs)

    # ── 3. Load dataset ──
    _send("status", status_message = "Loading dataset...")
    hf_dataset = config.get("hf_dataset", "")
    subset = config.get("subset")
    train_split = config.get("train_split", "train") or "train"
    eval_split = config.get("eval_split")
    slice_start = config.get("dataset_slice_start")
    slice_end = config.get("dataset_slice_end")

    def _slice(ds):
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(ds) - 1
            if end < start:
                return ds.select([])
            ds = ds.select(range(start, min(end + 1, len(ds))))
        return ds

    def _load_local(file_paths):
        from datasets import load_from_disk

        if len(file_paths) == 1:
            p = Path(file_paths[0])
            if p.is_dir() and (
                (p / "dataset_info.json").exists() or (p / "state.json").exists()
            ):
                return load_from_disk(str(p))
        all_files = _resolve_mlx_local_dataset_files(file_paths)
        if not all_files:
            raise ValueError("No local dataset files found")
        loader = _mlx_local_dataset_loader_for_files(all_files)
        return load_dataset(loader, data_files = all_files, split = "train")

    def _load_hf_dataset(*args, **kwargs):
        previous_dataset_env = _set_offline_env(
            ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"),
            dataset_local_only,
        )
        try:
            return load_dataset(*args, **kwargs)
        finally:
            _restore_env(previous_dataset_env)

    if hf_dataset:
        cached_files = _cached_dataset_training_files_for_config(config, train_split)
        if cached_files:
            loader = _mlx_local_dataset_loader_for_files(cached_files)
            dataset = load_dataset(loader, data_files = cached_files, split = "train")
        else:
            load_kwargs = {"split": train_split, "token": hf_token}
            if subset:
                load_kwargs["name"] = subset
            dataset = _load_hf_dataset(hf_dataset, **load_kwargs)
        dataset = _slice(dataset)
    elif config.get("local_datasets"):
        dataset = _load_local(config["local_datasets"])
        dataset = _slice(dataset)
    else:
        raise ValueError("No dataset specified")

    # Eval dataset (separate split or local file)
    eval_dataset = None
    if eval_split and hf_dataset:
        cached_eval_files = _cached_dataset_training_files_for_config(config, eval_split)
        try:
            if cached_eval_files:
                eval_loader = _mlx_local_dataset_loader_for_files(cached_eval_files)
                eval_dataset = load_dataset(
                    eval_loader,
                    data_files = cached_eval_files,
                    split = "train",
                )
            else:
                eval_kwargs = {"split": eval_split, "token": hf_token}
                if subset:
                    eval_kwargs["name"] = subset
                eval_dataset = _load_hf_dataset(hf_dataset, **eval_kwargs)
        except Exception as e:
            _send("status", status_message = f"Eval split load failed: {e}")
            eval_dataset = None
    elif config.get("local_eval_datasets"):
        eval_dataset = _load_local(config["local_eval_datasets"])

    # ── 3b. Format dataset (VLM or text) ──
    # Reuse the GPU path's format pipeline for both VLM (auto-detects OCR/caption/
    # llava/sharegpt+images) and text (alpaca/sharegpt/chatml → "text" column).
    format_type = config.get("format_type", "")
    try:
        from utils.datasets import format_and_template_dataset

        def _fmt_progress(status_message = "", **_kw):
            _send("status", status_message = status_message)

        if is_vlm:
            _send("status", status_message = "Formatting VLM dataset...")
            vlm_info = format_and_template_dataset(
                dataset,
                model_name = model_name,
                tokenizer = tokenizer,
                is_vlm = True,
                dataset_name = hf_dataset or "local",
                progress_callback = _fmt_progress,
            )
            if vlm_info.get("success"):
                dataset = _adapt_for_mlx_vlm(vlm_info["dataset"])
            else:
                errors = vlm_info.get("errors", [])
                raise ValueError(
                    f"VLM dataset format conversion failed: {'; '.join(errors)}"
                )
            if eval_dataset is not None:
                ev_info = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = True,
                    dataset_name = hf_dataset or "local",
                )
                if ev_info.get("success"):
                    eval_dataset = _adapt_for_mlx_vlm(ev_info["dataset"])

        elif format_type:
            _send("status", status_message = f"Formatting dataset ({format_type})...")
            info = format_and_template_dataset(
                dataset,
                model_name = model_name,
                tokenizer = tokenizer,
                is_vlm = False,
                format_type = format_type,
                dataset_name = hf_dataset or "local",
            )
            if info.get("success", True):
                dataset = info.get("dataset", dataset)
            if eval_dataset is not None:
                ev = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = False,
                    format_type = format_type,
                    dataset_name = hf_dataset or "local",
                )
                if ev.get("success", True):
                    eval_dataset = ev.get("dataset", eval_dataset)
    except ImportError:
        _send("status", status_message = "Format helper unavailable, using raw dataset")

    # ── 4. Resolve training steps ──
    max_steps = config.get("max_steps", 0) or 0
    num_epochs = config.get("num_epochs", 3)
    max_seq_length = config.get("max_seq_length", 2048)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("gradient_accumulation_steps", 4)

    if max_steps <= 0:
        max_steps = max(
            1,
            math.ceil(len(dataset) / batch_size / grad_accum) * num_epochs,
        )

    lr_value = float(config.get("learning_rate", "2e-4"))

    # Warmup: prefer warmup_steps; fall back to warmup_ratio
    warmup_steps = config.get("warmup_steps")
    warmup_ratio = config.get("warmup_ratio")
    if warmup_steps is None and warmup_ratio is not None:
        warmup_steps = int(round(warmup_ratio * max_steps))
    if warmup_steps is None:
        warmup_steps = 5

    # ── 5. Build output dir ──
    output_dir = config.get("output_dir", "")
    if not output_dir:
        output_dir = f"{model_name.replace('/', '_')}_{int(time.time())}"
    # Resolve to ~/.unsloth/studio/outputs/ so the export page can find it
    from utils.paths import resolve_output_dir, ensure_dir

    output_dir = str(resolve_output_dir(output_dir))
    ensure_dir(Path(output_dir))

    # ── 6. Create trainer ──
    eval_steps_val = config.get("eval_steps", 0) or 0
    if isinstance(eval_steps_val, float) and 0 < eval_steps_val < 1:
        # Studio sometimes sends fraction-of-total-steps
        eval_steps_val = max(1, int(eval_steps_val * max_steps))
    else:
        eval_steps_val = int(eval_steps_val)

    # MLX: per-element clip to [-1, 1]; norm clip disabled (it needs a
    # global reduction that breaks MLX's eager pipeline). 1.0 (not 5.0):
    # |g_i| > 5 rarely fires, so the historical 5.0 was effectively no-op.
    max_grad_norm = 0.0
    max_grad_value = 1.0  # TODO: expose MLX grad-clip in Studio UI for power users

    trainer = MLXTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        args = MLXTrainingConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            max_steps = max_steps,
            learning_rate = lr_value,
            warmup_steps = warmup_steps,
            lr_scheduler_type = lr_scheduler_type,
            optim = optim_name,
            weight_decay = float(config.get("weight_decay", 0.001) or 0.001),
            max_grad_norm = max_grad_norm,
            max_grad_value = max_grad_value,
            logging_steps = 1,
            max_seq_length = max_seq_length,
            seed = config.get("random_seed", 3407),
            use_cce = True,
            compile = True,
            gradient_checkpointing = use_grad_checkpoint,
            streaming = is_vlm,
            packing = bool(config.get("packing", False)),
            output_dir = output_dir,
            save_steps = int(config.get("save_steps", 0) or 0),
            eval_steps = eval_steps_val,
        ),
    )

    # Tell the parent that eval is configured so the frontend shows the eval chart
    if eval_dataset is not None and eval_steps_val > 0:
        _send("eval_configured")

    # ── 7. Apply train_on_responses_only if requested ──
    if config.get("train_on_completions", False):
        _send("status", status_message = "Configuring response-only training...")
        try:
            from utils.datasets import (
                MODEL_TO_TEMPLATE_MAPPER,
                TEMPLATE_TO_RESPONSES_MAPPER,
            )

            template_name = MODEL_TO_TEMPLATE_MAPPER.get(model_name.lower())
            markers = (
                TEMPLATE_TO_RESPONSES_MAPPER.get(template_name)
                if template_name
                else None
            )
            if markers:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = markers["instruction"],
                    response_part = markers["response"],
                )
            else:
                _send(
                    "status",
                    status_message = f"train_on_completions skipped (no template for {model_name})",
                )
        except Exception as e:
            _send("status", status_message = f"train_on_completions failed: {e}")

    # ── 8. Setup wandb / tensorboard ──
    wandb_run = None
    tb_writer = None
    if config.get("enable_wandb", False):
        try:
            import wandb as _wandb

            wandb_token = config.get("wandb_token")
            if wandb_token:
                os.environ["WANDB_API_KEY"] = wandb_token
            _wandb_sensitive = {"hf_token", "wandb_token"}
            wandb_run = _wandb.init(
                project = config.get("wandb_project") or "unsloth-mlx",
                config = {k: v for k, v in config.items() if k not in _wandb_sensitive},
                reinit = True,
            )
        except Exception as e:
            _send("status", status_message = f"wandb init failed: {e}")
    if config.get("enable_tensorboard", False):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                SummaryWriter = None
        if SummaryWriter is not None:
            try:
                tb_dir = config.get("tensorboard_dir") or f"{output_dir}/runs"
                tb_writer = SummaryWriter(log_dir = tb_dir)
            except Exception as e:
                _send("status", status_message = f"tensorboard init failed: {e}")
        else:
            _send(
                "status",
                status_message = "tensorboard unavailable (install tensorboardX)",
            )

    # ── 9. Real-time progress callback ──
    _send("status", status_message = f"Training {model_name}...")

    def _on_step(
        step,
        total,
        loss,
        lr,
        tok_s,
        peak_gb,
        elapsed,
        num_tokens,
        grad_norm = None,
    ):
        eta = (elapsed / step * (total - step)) if step > 0 else 0
        _send(
            "progress",
            step = step,
            epoch = round(step / total * num_epochs, 2) if total > 0 else 0,
            loss = loss,
            learning_rate = lr,
            total_steps = total,
            elapsed_seconds = elapsed,
            eta_seconds = max(0, eta),
            grad_norm = grad_norm,
            num_tokens = num_tokens,
            eval_loss = None,
            status_message = None,
            peak_memory_gb = peak_gb,
        )
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/peak_gb": peak_gb,
                        "train/num_tokens": num_tokens,
                        **(
                            {"train/grad_norm": grad_norm}
                            if grad_norm is not None
                            else {}
                        ),
                    },
                    step = step,
                )
            except Exception:
                pass
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("train/loss", loss, step)
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/tokens_per_sec", tok_s, step)
                tb_writer.add_scalar("train/peak_gb", peak_gb, step)
                if grad_norm is not None:
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            except Exception:
                pass

    trainer.add_step_callback(_on_step)

    def _on_eval(step, eval_loss, perplexity):
        _send("progress", step = step, eval_loss = eval_loss)
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {"eval/loss": eval_loss, "eval/perplexity": perplexity}, step = step
                )
            except Exception:
                pass
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("eval/loss", eval_loss, step)
                tb_writer.add_scalar("eval/perplexity", perplexity, step)
            except Exception:
                pass

    trainer.add_eval_callback(_on_eval)

    # ── 10. Stop signal polling ──
    _stop_save = [True]  # mutable so thread can update; [save_flag]

    def _poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    _stop_save[0] = msg.get("save", True)
                    trainer.stop_requested = True
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                # why safe: pipe permanently broken, no further messages can arrive
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 11. Run training ──
    gc.collect()
    mx.synchronize()
    trainer.train()

    # ── 12. Save and finalize ──
    if trainer.stop_requested and not _stop_save[0]:
        # User clicked "Cancel" (save=False) — skip saving
        _send("complete", output_dir = None, status_message = "Training cancelled")
    else:
        _send("status", status_message = "Saving model...")
        mx.synchronize()
        trainer.save_model(output_dir)
        _send("complete", output_dir = output_dir, status_message = "Training completed")

    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:
            pass
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


def run_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for sending progress/status/error events to parent.
        stop_queue: mp.Queue for receiving stop commands from parent.
        config: Training configuration dict with all parameters.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    # Offline auto-detect: skip ~25s of HF retries per call when DNS is
    # dead. Scoped to this subprocess (orchestrator spawns a fresh one).
    if "HF_HUB_OFFLINE" not in os.environ:
        import socket as _socket
        import threading as _threading

        # Daemon thread so we don't mutate process-wide setdefaulttimeout.
        _result: list = [None]

        def _probe() -> None:
            try:
                _socket.gethostbyname("huggingface.co")
                _result[0] = False
            except Exception:
                _result[0] = True

        _t = _threading.Thread(target = _probe, daemon = True)
        _t.start()
        _t.join(2.0)
        if _result[0] is None or _result[0] is True:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            # logger isn't configured yet; print to stderr instead.
            print(
                "huggingface.co unreachable; HF_HUB_OFFLINE=1 set for this worker.",
                file = sys.stderr,
                flush = True,
            )

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-training-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    apply_gpu_ids(config.get("resolved_gpu_ids"))

    model_name = config["model_name"]
    format_error = _untrainable_model_format_error(config)
    if format_error:
        event_queue.put(
            {
                "type": "error",
                "error": format_error,
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    # ── 0. MLX FAST-PATH (must run before any torch/transformers imports) ──
    # Apple Silicon uses MLXTrainer directly -- skip transformers version
    # activation, causal-conv1d install, and torch imports entirely.
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if _hw.DEVICE == _hw.DeviceType.MLX:
        if config.get("is_dataset_audio"):
            event_queue.put(
                {
                    "type": "error",
                    "error": "Audio dataset training is not yet supported on Apple Silicon.",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return
        # Activate correct transformers version (Gemma-4 needs 5.5.0, etc.)
        # Must happen before any transformers/mlx-lm imports in _run_mlx_training.
        try:
            _activate_transformers_version(model_name)
        except Exception:
            pass  # Non-fatal: fall through with whatever version is installed
        try:
            _run_mlx_training(event_queue, stop_queue, config)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        return

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1a. Auto-enable trust_remote_code for NemotronH/Nano models ──
    # NemotronH has config parsing bugs in transformers that require
    # trust_remote_code=True as a workaround. Other transformers 5.x models
    # (Qwen3.5, Gemma 4, etc.) are native and do NOT need it — enabling it
    # bypasses the compiler (disabling fused CE).
    # NOTE: Must NOT match Llama-Nemotron (standard Llama architecture).
    _NEMOTRON_TRUST_SUBSTRINGS = ("nemotron_h", "nemotron-h", "nemotron-3-nano")
    _lowered = model_name.lower()
    if (
        any(sub in _lowered for sub in _NEMOTRON_TRUST_SUBSTRINGS)
        and (_lowered.startswith("unsloth/") or _lowered.startswith("nvidia/"))
        and not config.get("trust_remote_code", False)
    ):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for Nemotron model: %s",
            model_name,
        )

    # ── 1b. Install fast-path kernel libraries for the chosen model.
    #
    # 1) causal-conv1d ALWAYS runs eagerly via the substring path.
    #    Some SSM modeling files (nemotron_h, falcon_h1, granitemoehybrid)
    #    use `lazy_load_kernel("causal-conv1d")` directly and never call
    #    transformers' `is_causal_conv1d_available()`, so the runtime
    #    hook on that gate would not fire for them.
    # 2) FLA + tilelang: primary gate is the runtime hook on transformers'
    #    `is_flash_linear_attention_available`. Models whose architecture
    #    queries that gate auto-trigger the install; others never pay.
    #    `_install_fast_path_hooks` also wraps `is_causal_conv1d_available`
    #    as a defence in depth for newer modeling files that do use it.
    # 3) mamba-ssm + flash-attn keep their existing substring / size gates.
    # 4) `UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1` falls back to the
    #    substring path for FLA / tilelang.
    try:
        _ensure_causal_conv1d_fast_path(event_queue, model_name)
        if os.getenv(_FAST_PATH_HOOKS_SKIP_ENV) == "1":
            _ensure_flash_linear_attention(event_queue, model_name)
            _ensure_tilelang_backend(event_queue, model_name)
        else:
            _install_fast_path_hooks(event_queue, model_name)
        _ensure_mamba_ssm(event_queue, model_name)
        _ensure_flash_attn_for_long_context(
            event_queue,
            int(config.get("max_seq_length", 2048)),
        )
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    f"Please choose another model to train, since "
                    f"a fast-path kernel library "
                    f"(causal-conv1d / flash-linear-attention / "
                    f"mamba-ssm / tilelang) failed to install "
                    f"with error: {exc}"
                ),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1c. Set fork start method so dataset.map() can multiprocess ──
    # The parent launched us via spawn (clean process), but the compiled
    # SFTTrainer checks get_start_method() and disables num_proc if not "fork".
    # Linux only: fork is the default start method and is safe here (no CUDA
    # context exists yet). macOS defaults to spawn since Python 3.8 because
    # fork is unsafe with macOS frameworks (Metal/MPS, CoreFoundation) --
    # do NOT override on macOS. Windows has no fork at all.
    if sys.platform == "linux":
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force = True)
        except RuntimeError:
            pass  # Already set

    # ── 1c. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer, TrainingProgress
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
        )

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 2b. EMBEDDING MODEL FAST-PATH ──
    # Embedding models use a completely different pipeline (FastSentenceTransformer
    # + SentenceTransformerTrainer + MultipleNegativesRankingLoss) so we branch
    # early and handle the entire flow in a self-contained function.
    if config.get("is_embedding", False):
        try:
            _run_embedding_training(event_queue, stop_queue, config)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        return

    # ── 3. Create a fresh trainer instance ──
    trainer = UnslothTrainer()

    # Wire up progress callback → event_queue
    def _on_progress(progress: TrainingProgress):
        has_train_loss = progress.step > 0 and progress.loss is not None
        has_eval_loss = progress.eval_loss is not None
        if has_train_loss or has_eval_loss:
            event_queue.put(
                {
                    "type": "progress",
                    "step": progress.step,
                    "epoch": progress.epoch,
                    "loss": progress.loss,
                    "learning_rate": progress.learning_rate,
                    "total_steps": progress.total_steps,
                    "elapsed_seconds": progress.elapsed_seconds,
                    "eta_seconds": progress.eta_seconds,
                    "grad_norm": progress.grad_norm,
                    "num_tokens": progress.num_tokens,
                    "eval_loss": progress.eval_loss,
                    "status_message": progress.status_message,
                    "ts": time.time(),
                }
            )
        if progress.status_message:
            _send_status(event_queue, progress.status_message)

    trainer.add_progress_callback(_on_progress)

    # Wire up stop_queue polling to trainer.should_stop
    import threading
    import queue as _queue

    def _poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    save = msg.get("save", True)
                    trainer.should_stop = True
                    trainer.save_on_stop = save
                    logger.info("Stop signal received (save=%s)", save)
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 4. Execute the training pipeline ──
    # Order: detect → dataset → model → prepare → train
    # Dataset processing (including LLM-assisted detection) runs BEFORE model
    # loading so both never occupy VRAM at the same time.
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        model_load_name = _resolve_cached_model_load_name(config)
        model_local_only = _model_local_files_only(config)
        dataset_local_only = _dataset_local_files_only(config)

        # ── 4a. Lightweight detection + tokenizer (no VRAM) ──
        _send_status(event_queue, "Detecting model type...")
        trainer.pre_detect_and_load_tokenizer(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
            model_load_name = model_load_name,
            local_files_only = model_local_only,
        )
        if trainer.should_stop:
            event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            return

        # ── 4b. Load and format dataset (LLM helper may use VRAM briefly) ──
        _send_status(event_queue, "Loading and formatting dataset...")
        hf_dataset = config.get("hf_dataset", "")
        training_type = config.get("training_type", "LoRA/QLoRA")
        _is_cpt_for_dataset = training_type == "Continued Pretraining"
        dataset_result = trainer.load_and_format_dataset(
            dataset_source = hf_dataset if hf_dataset and hf_dataset.strip() else None,
            format_type = config.get("format_type", ""),
            local_datasets = config.get("local_datasets") or None,
            local_eval_datasets = config.get("local_eval_datasets") or None,
            custom_format_mapping = config.get("custom_format_mapping"),
            subset = config.get("subset"),
            train_split = config.get("train_split", "train"),
            eval_split = config.get("eval_split"),
            eval_steps = config.get("eval_steps", 0.00),
            dataset_slice_start = config.get("dataset_slice_start"),
            dataset_slice_end = config.get("dataset_slice_end"),
            is_cpt = _is_cpt_for_dataset,
            dataset_local_files_only = dataset_local_only,
            dataset_local_path = config.get("dataset_local_path"),
        )

        if isinstance(dataset_result, tuple):
            dataset, eval_dataset = dataset_result
        else:
            dataset = dataset_result
            eval_dataset = None

        # [DEBUG] Print first sample before model is loaded
        # dataset is a dict {"dataset": <Dataset>, "detected_format": ..., ...}
        # or a raw Dataset for audio paths
        # try:
        #     ds = dataset["dataset"] if isinstance(dataset, dict) else dataset
        #     print(
        #         f"\n[DEBUG] Dataset loaded BEFORE model. type={type(ds).__name__}, len={len(ds)}",
        #         flush = True,
        #     )
        #     print(f"[DEBUG] Columns: {ds.column_names}", flush = True)
        #     sample = ds[0]
        #     preview = {k: str(v)[:300] for k, v in sample.items()}
        #     print(f"[DEBUG] First sample: {preview}\n", flush = True)
        # except Exception as e:
        #     print(
        #         f"[DEBUG] Could not preview first sample: {type(e).__name__}: {e}",
        #         flush = True,
        #     )

        # Disable eval if eval_steps <= 0
        eval_steps = config.get("eval_steps", 0.00)
        if eval_steps is not None and float(eval_steps) <= 0:
            eval_dataset = None

        # Tell the parent process that eval is configured so the frontend
        # shows "Waiting for first evaluation step..." instead of "not configured"
        if eval_dataset is not None:
            event_queue.put(
                {
                    "type": "eval_configured",
                    "ts": time.time(),
                }
            )

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to load dataset",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── Start tqdm monitor early so it captures download + tokenization bars ──
        import threading as _th

        _tqdm_stop = _th.Event()

        def _monitor_tqdm():
            from tqdm.auto import tqdm as _tqdm_cls

            while not _tqdm_stop.is_set():
                for bar in list(getattr(_tqdm_cls, "_instances", set())):
                    try:
                        n, total = bar.n or 0, bar.total or 0
                        desc = getattr(bar, "desc", "") or ""
                        if total > 0 and n > 0 and desc:
                            pct = min(int(n * 100 / total), 100)
                            _send_status(
                                event_queue, f"{desc.strip()} {pct}% ({n:,}/{total:,})"
                            )
                    except (AttributeError, ReferenceError):
                        pass
                _tqdm_stop.wait(3)

        _tqdm_thread = _th.Thread(target = _monitor_tqdm, daemon = True)
        _tqdm_thread.start()

        training_type = config.get("training_type", "LoRA/QLoRA")
        is_cpt = training_type == "Continued Pretraining"
        use_lora = training_type in ("LoRA/QLoRA", "Continued Pretraining")
        cpt_trains_embeddings = False

        # ── 4c. Load training model (uses VRAM — dataset already formatted) ──
        _send_status(event_queue, "Loading model...")
        success = trainer.load_model(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            load_in_4bit = config["load_in_4bit"],
            full_finetuning = not use_lora,
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
            gpu_ids = config.get("resolved_gpu_ids"),
            model_load_name = model_load_name,
            local_files_only = model_local_only,
        )
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                error_msg = trainer.training_progress.error or "Failed to load model"
                event_queue.put(
                    {
                        "type": "error",
                        "error": error_msg,
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── 4d. Prepare model (LoRA, full finetuning, or CPT) ──
        if is_cpt:
            _send_status(event_queue, "Configuring LoRA for continued pretraining...")
            # embed_tokens (if the user included it) goes to modules_to_save —
            # trained full-precision at embedding_learning_rate. lm_head stays as
            # a LoRA target for merge compatibility (see unsloth PR #4106).
            _user_modules = config.get("target_modules") or []
            wants_embed = "embed_tokens" in _user_modules
            cpt_trains_embeddings = wants_embed
            cpt_target_modules = [m for m in _user_modules if m != "embed_tokens"]
            if not cpt_target_modules:
                cpt_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ]
            success = trainer.prepare_model_for_training(
                use_lora = True,
                target_modules = cpt_target_modules,
                modules_to_save = ["embed_tokens"] if wants_embed else None,
                lora_r = config.get("lora_r", 128),
                lora_alpha = config.get("lora_alpha", 32),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get(
                    "gradient_checkpointing", "unsloth"
                ),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        elif use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora = True,
                finetune_vision_layers = config.get("finetune_vision_layers", True),
                finetune_language_layers = config.get("finetune_language_layers", True),
                finetune_attention_modules = config.get(
                    "finetune_attention_modules", True
                ),
                finetune_mlp_modules = config.get("finetune_mlp_modules", True),
                target_modules = config.get("target_modules"),
                lora_r = config.get("lora_r", 16),
                lora_alpha = config.get("lora_alpha", 16),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get(
                    "gradient_checkpointing", "unsloth"
                ),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora = False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to prepare model",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        lr_default = "5e-5" if is_cpt else "2e-4"
        try:
            lr_value = float(config.get("learning_rate", lr_default))
        except ValueError:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Invalid learning rate: {config.get('learning_rate')}",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # embedding_learning_rate is validated by the Pydantic model (Optional[float],
        # gt=0, lt=1.0); if present it is already a finite float in range.
        embedding_lr_value = config.get("embedding_learning_rate")
        if is_cpt:
            if cpt_trains_embeddings:
                if embedding_lr_value is None:
                    # Default embedding_learning_rate = lr/10 per Unsloth's CPT notebook.
                    embedding_lr_value = lr_value / 10.0
                    logger.info(
                        f"CPT: using default embedding_learning_rate={embedding_lr_value:.1e} "
                        f"(lr/10). Set explicitly to override.\n"
                    )
            elif embedding_lr_value is not None:
                logger.warning(
                    "CPT: embedding_learning_rate was provided but embed_tokens is "
                    "not being trained; ignoring the override.\n"
                )
                embedding_lr_value = None

        # Generate output dir
        resume_from_checkpoint = config.get("resume_from_checkpoint")
        output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
            resume_from_checkpoint
        )
        if not output_dir:
            output_dir = f"{model_name.replace('/', '_')}_{int(time.time())}"
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))

        tensorboard_dir = config.get("tensorboard_dir")
        if config.get("enable_tensorboard", False):
            tensorboard_dir = str(resolve_tensorboard_dir(tensorboard_dir))
            ensure_dir(Path(tensorboard_dir))

        # Start training (directly — no inner thread, we ARE the subprocess)
        dataset_display = (
            config.get("hf_dataset", "") or config.get("uploaded_file", "") or ""
        )
        _send_status(
            event_queue,
            f'Training "{model_name}"'
            + (f"\nDataset = {dataset_display}" if dataset_display else ""),
        )
        max_steps = config.get("max_steps", 0)
        save_steps = config.get("save_steps", 0)

        trainer._train_worker(
            dataset,
            output_dir = output_dir,
            num_epochs = config.get("num_epochs", 3),
            learning_rate = lr_value,
            embedding_learning_rate = embedding_lr_value,
            batch_size = config.get("batch_size", 2),
            gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4),
            warmup_steps = config.get("warmup_steps"),
            warmup_ratio = config.get("warmup_ratio"),
            max_steps = max_steps if max_steps and max_steps > 0 else 0,
            save_steps = save_steps if save_steps and save_steps > 0 else 0,
            weight_decay = config.get("weight_decay", 0.001),
            random_seed = config.get("random_seed", 3407),
            packing = config.get("packing", False),
            train_on_completions = False
            if is_cpt
            else config.get("train_on_completions", False),
            enable_wandb = config.get("enable_wandb", False),
            wandb_project = config.get("wandb_project", "unsloth-training"),
            wandb_token = config.get("wandb_token"),
            enable_tensorboard = config.get("enable_tensorboard", False),
            tensorboard_dir = tensorboard_dir,
            eval_dataset = eval_dataset,
            eval_steps = eval_steps,
            max_seq_length = config.get("max_seq_length", 2048),
            optim = config.get("optim", "adamw_8bit"),
            lr_scheduler_type = config.get("lr_scheduler_type", "linear"),
            is_cpt = is_cpt,
            resume_from_checkpoint = resume_from_checkpoint,
        )

        _tqdm_stop.set()

        # Check final state
        progress = trainer.get_training_progress()
        if progress.error:
            event_queue.put(
                {
                    "type": "error",
                    "error": progress.error,
                    "stack": "",
                    "ts": time.time(),
                }
            )
        else:
            saved_output_dir = (
                None if trainer.should_stop and not trainer.save_on_stop else output_dir
            )
            event_queue.put(
                {
                    "type": "complete",
                    "output_dir": saved_output_dir,
                    "status_message": progress.status_message or "Training completed",
                    "ts": time.time(),
                }
            )

    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )


def _send_status(event_queue: Any, message: str) -> None:
    """Send a status update to the parent process."""
    event_queue.put(
        {
            "type": "status",
            "message": message,
            "ts": time.time(),
        }
    )


def _run_embedding_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Self-contained embedding model training pipeline.

    Uses FastSentenceTransformer + SentenceTransformerTrainer +
    MultipleNegativesRankingLoss — completely separate from the
    LLM/VLM/audio paths in UnslothTrainer.

    Mirrors the pattern from the reference embedding notebooks:
      All_MiniLM_L6_v2.py, BGE_M3.py, EmbeddingGemma_300M.py,
      ModernBert.py, Qwen3_Embedding_0_6B.py
    """
    import math
    import queue as _queue
    import threading

    model_name = config["model_name"]
    model_load_name = _resolve_cached_model_load_name(config)
    model_local_only = _model_local_files_only(config)
    dataset_local_only = _dataset_local_files_only(config)
    training_start_time = time.time()

    # ── 1. Import embedding-specific libraries ──
    _send_status(event_queue, "Importing embedding libraries...")
    try:
        from unsloth import FastSentenceTransformer, is_bfloat16_supported
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from datasets import load_dataset, Dataset
        from transformers import TrainerCallback
        from utils.paths import datasets_root, resolve_output_dir
    except ImportError as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import embedding libraries: {e}. "
                "Ensure 'sentence_transformers' and 'unsloth' are installed.",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── Stop signal handling ──
    _should_stop = False
    _save_on_stop = True

    def _poll_stop():
        nonlocal _should_stop, _save_on_stop
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    _save_on_stop = msg.get("save", True)
                    _should_stop = True
                    logger.info(
                        "Embedding training: stop signal received (save=%s)",
                        _save_on_stop,
                    )
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 2. Load model ──
    _send_status(event_queue, "Loading embedding model...")
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        max_seq_length = config.get("max_seq_length", 512)
        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        previous_model_env = _set_offline_env(
            ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"),
            model_local_only,
        )
        try:
            model = FastSentenceTransformer.from_pretrained(
                model_name = model_load_name,
                max_seq_length = max_seq_length,
                full_finetuning = not use_lora,
                token = hf_token,
            )
        finally:
            _restore_env(previous_model_env)
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load embedding model '{model_name}': {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 3. Apply LoRA ──
    if use_lora:
        _send_status(event_queue, "Configuring LoRA adapters (FEATURE_EXTRACTION)...")
        try:
            gradient_checkpointing = config.get("gradient_checkpointing", False)
            # Normalize: "none" or empty → False
            if gradient_checkpointing in ("none", "", None):
                gradient_checkpointing = False

            model = FastSentenceTransformer.get_peft_model(
                model,
                r = config.get("lora_r", 32),
                target_modules = config.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = config.get("lora_alpha", 64),
                lora_dropout = config.get("lora_dropout", 0.0),
                bias = "none",
                use_gradient_checkpointing = gradient_checkpointing,
                random_state = config.get("random_seed", 3407),
                use_rslora = config.get("use_rslora", False),
                loftq_config = {"loftq_bits": 4, "loftq_iter": 1}
                if config.get("use_loftq")
                else None,
                task_type = "FEATURE_EXTRACTION",
            )
        except Exception as e:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Failed to configure LoRA for embedding model: {e}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
            return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 4. Load dataset ──
    _send_status(event_queue, "Loading dataset...")
    try:
        hf_dataset = config.get("hf_dataset", "")
        local_datasets = config.get("local_datasets") or []
        subset = config.get("subset") or None
        train_split = config.get("train_split", "train") or "train"

        if hf_dataset and hf_dataset.strip():
            hf_token = config.get("hf_token", "")
            hf_token = hf_token if hf_token and hf_token.strip() else None
            cached_files = _cached_dataset_training_files_for_config(config, train_split)
            if cached_files:
                loader = _mlx_local_dataset_loader_for_files(cached_files)
                dataset = load_dataset(loader, data_files = cached_files, split = "train")
            else:
                previous_dataset_env = _set_offline_env(
                    ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"),
                    dataset_local_only,
                )
                try:
                    dataset = load_dataset(
                        hf_dataset.strip(),
                        subset,
                        split = train_split,
                        token = hf_token,
                    )
                finally:
                    _restore_env(previous_dataset_env)
        elif local_datasets:
            # Load from local file(s) — mirrors the non-embedding pipeline's
            # directory handling so recipe outputs (parquet-files/) work.
            all_files: list[str] = []
            for dataset_file in local_datasets:
                file_path = (
                    dataset_file
                    if os.path.isabs(dataset_file)
                    else os.path.join(
                        str(datasets_root()),
                        dataset_file,
                    )
                )
                if os.path.isdir(file_path):
                    file_path_obj = Path(file_path)
                    parquet_dir = (
                        file_path_obj / "parquet-files"
                        if (file_path_obj / "parquet-files").exists()
                        else file_path_obj
                    )
                    parquet_files = sorted(parquet_dir.glob("*.parquet"))
                    if parquet_files:
                        all_files.extend(str(p) for p in parquet_files)
                        continue
                    candidates: list[Path] = []
                    for ext in (".json", ".jsonl", ".csv", ".parquet"):
                        candidates.extend(sorted(file_path_obj.glob(f"*{ext}")))
                    if candidates:
                        all_files.extend(str(c) for c in candidates)
                        continue
                    raise ValueError(
                        f"No supported data files in directory: {file_path_obj}"
                    )
                else:
                    all_files.append(file_path)

            if all_files:
                first_ext = Path(all_files[0]).suffix.lower()
                if first_ext in (".json", ".jsonl"):
                    loader = "json"
                elif first_ext == ".csv":
                    loader = "csv"
                elif first_ext == ".parquet":
                    loader = "parquet"
                else:
                    raise ValueError(
                        f"Unsupported local dataset format: {all_files[0]}"
                    )
                dataset = load_dataset(loader, data_files = all_files, split = "train")
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": "No dataset specified for embedding training.",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Apply dataset slicing if specified
        slice_start = config.get("dataset_slice_start")
        slice_end = config.get("dataset_slice_end")
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(dataset)
            dataset = dataset.select(range(start, min(end + 1, len(dataset))))

        logger.info(f"Embedding dataset loaded: {len(dataset)} samples")
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load dataset: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 5. Create loss function ──
    loss = MultipleNegativesRankingLoss(model)

    # ── 6. Build training arguments ──
    _send_status(event_queue, "Configuring training...")
    try:
        lr_value = float(config.get("learning_rate", "2e-4"))
    except ValueError:
        event_queue.put(
            {
                "type": "error",
                "error": f"Invalid learning rate: {config.get('learning_rate')}",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
        resume_from_checkpoint
    )
    if not output_dir:
        output_dir = str(
            resolve_output_dir(f"{model_name.replace('/', '_')}_{int(time.time())}")
        )
    output_dir = str(resolve_output_dir(output_dir))

    num_epochs = config.get("num_epochs", 2)
    batch_size = config.get("batch_size", 256)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_steps_val = config.get("max_steps", 0)
    save_steps_val = config.get("save_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    warmup_steps_val = config.get("warmup_steps")
    log_frequency = config.get("log_frequency", 50)

    # Build args dict
    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": lr_value,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 1,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": config.get("weight_decay", 0.001),
        "seed": config.get("random_seed", 3407),
    }

    # max_steps vs epochs
    if max_steps_val and max_steps_val > 0:
        training_args_kwargs["max_steps"] = max_steps_val
    else:
        training_args_kwargs["num_train_epochs"] = num_epochs if num_epochs > 0 else 2

    # warmup: prefer warmup_ratio (standard for embedding scripts), fallback to steps
    if warmup_ratio is not None and warmup_ratio > 0:
        training_args_kwargs["warmup_ratio"] = warmup_ratio
    elif warmup_steps_val is not None and warmup_steps_val > 0:
        training_args_kwargs["warmup_steps"] = warmup_steps_val

    # save_steps
    if save_steps_val and save_steps_val > 0:
        training_args_kwargs["save_steps"] = save_steps_val
        training_args_kwargs["save_strategy"] = "steps"

    args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # ── 7. Calculate total steps for progress tracking ──
    if max_steps_val and max_steps_val > 0:
        total_steps = max_steps_val
    else:
        effective_epochs = num_epochs if num_epochs > 0 else 2
        len_dataloader = math.ceil(len(dataset) / batch_size)
        steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * effective_epochs

    # ── 8. Create progress callback ──
    class _EmbeddingProgressCallback(TrainerCallback):
        """Sends training progress events to the parent process via event_queue."""

        def on_log(self, args, state, control, logs = None, **kwargs):
            if not logs:
                return
            loss_value = logs.get("loss", logs.get("train_loss", None))
            current_step = state.global_step

            elapsed = time.time() - training_start_time
            eta = None
            if current_step > 0 and total_steps > 0:
                remaining = total_steps - current_step
                if remaining > 0:
                    eta = (elapsed / current_step) * remaining

            event_queue.put(
                {
                    "type": "progress",
                    "step": current_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "loss": loss_value,
                    "learning_rate": logs.get("learning_rate", None),
                    "total_steps": total_steps,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "grad_norm": logs.get("grad_norm"),
                    "num_tokens": getattr(state, "num_input_tokens_seen", None),
                    "eval_loss": logs.get("eval_loss"),
                    "status_message": "",
                    "ts": time.time(),
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if _should_stop:
                logger.info("Embedding training: stop at step %d", state.global_step)
                control.should_training_stop = True
                return control

    # ── 9. Create trainer and train ──
    _send_status(event_queue, "Starting embedding training...")
    try:
        trainer = SentenceTransformerTrainer(
            model = model,
            train_dataset = dataset,
            loss = loss,
            args = args,
            callbacks = [_EmbeddingProgressCallback()],
        )

        trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Embedding training failed: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 10. Save model ──
    if _should_stop and not _save_on_stop:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training cancelled",
                "ts": time.time(),
            }
        )
        return

    _send_status(event_queue, "Saving model...")
    try:
        if _should_stop and _save_on_stop:
            trainer._save_checkpoint(trainer.model, trial = None)
        model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        logger.info("Embedding model saved to %s", output_dir)
    except Exception as e:
        logger.error("Failed to save embedding model: %s", e)
        event_queue.put(
            {
                "type": "error",
                "error": f"Training completed but failed to save: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 11. Done ──
    event_queue.put(
        {
            "type": "complete",
            "output_dir": output_dir,
            "status_message": "Embedding training completed",
            "ts": time.time(),
        }
    )
