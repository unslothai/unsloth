# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each job runs in a fresh subprocess (mp.get_context("spawn")): a clean
interpreter with no stale module state, which solves transformers
version-switching. Pattern follows core/data_recipe/jobs/worker.py.
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
import gc
import re
import types
import subprocess as _sp
from pathlib import Path
from typing import Any, Callable

# ── WSL AMD Strix Halo (gfx1151): enable ROCDXG before any torch import ──────
# Mirrors main.py. In WSL the AMD GPU is reached via the ROCDXG bridge
# (librocdxg.so over /dev/dxg), which HSA loads only when HSA_ENABLE_DXG_
# DETECTION=1 is set before torch touches the GPU. A worker spawned outside a
# login shell misses the installer's persisted env and falls back to CPU.
# Gated to no-op unless BOTH /dev/dxg and librocdxg.so exist, so native Linux
# ROCm, NVIDIA, macOS and Windows are unaffected.
if sys.platform.startswith("linux") and "HSA_ENABLE_DXG_DETECTION" not in os.environ:
    try:
        if os.path.exists("/dev/dxg") and any(
            os.path.exists(_p + "/librocdxg.so") for _p in ("/opt/rocm/lib", "/opt/rocm/lib64")
        ):
            os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"
    except Exception:
        pass

logger = get_logger(__name__)
from utils.hardware import apply_gpu_ids
from utils.training_runs import build_default_output_dir_name
from utils.wheel_utils import (
    direct_wheel_url,
    flash_attn_wheel_url,
    has_blackwell_gpu,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)


def _output_dir_from_resume_checkpoint(resume_from_checkpoint: str | None) -> str | None:
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
# Pin both so plain pip can't silently upgrade torch under the worker (fla-core needs torch>=2.7).
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

# Module-level handle so the torch.library.Library registration survives past
# run_training_process() and isn't GC'd mid-run.
_WINDOWS_ROCM_GROUPED_MM_LIB = None


def _install_grouped_mm_cpu_fallback(torch_mod, logger, label):
    """Register a Python mm/bmm fallback for torch._grouped_mm and return the Library.

    RDNA4 (gfx1200/gfx1201) ships a null HIP _grouped_mm kernel on ROCm <= 7.12
    (fixed in 7.13; ROCm/TheRock #5284). JitDecomp dispatches _grouped_mm to the
    null kernel and crashes; overriding the CUDA dispatch key bypasses it. Shared
    by the Windows and Linux ROCm guards. Keep the returned Library referenced so
    the registration outlives the caller.
    """
    import warnings as _warnings

    _gm_lib = torch_mod.library.Library("aten", "IMPL")

    def _grouped_mm_safe_impl(
        self,
        mat2,
        offs = None,
        bias = None,
        out_dtype = None,
    ):
        """Python mm/bmm fallback for _grouped_mm on gfx120X (null HIP kernel, ROCm <= 7.12)."""
        _t = torch_mod
        if offs is None:
            # No offsets: 2-D -> mm, 3-D batched -> bmm (unconditional mm broke 3-D MoE).
            if self.dim() == 3 and mat2.dim() == 3:
                result = _t.bmm(self.contiguous(), mat2.contiguous())
            elif self.dim() == 3 and mat2.dim() == 2:
                result = _t.matmul(self.contiguous(), mat2.contiguous())
            elif self.dim() == 2 and mat2.dim() == 3:
                result = _t.matmul(self.contiguous(), mat2.contiguous())
            else:
                result = _t.mm(self.contiguous(), mat2.contiguous())
        else:
            # Grouped: offs[i] is the exclusive end-row of group i.
            offs_list = offs.tolist()
            pieces = []
            prev = 0
            for idx, end in enumerate(offs_list):
                end = int(end)
                a_part = self[prev:end].contiguous()
                b_part = mat2[idx].contiguous() if mat2.dim() == 3 else mat2.contiguous()
                pieces.append(_t.mm(a_part, b_part))
                prev = end
            # Include trailing rows not covered by offs.
            if prev < self.shape[0]:
                a_tail = self[prev:].contiguous()
                b_tail = mat2[-1].contiguous() if mat2.dim() == 3 else mat2.contiguous()
                pieces.append(_t.mm(a_tail, b_tail))
            result = (
                _t.cat(pieces, dim = 0)
                if pieces
                else _t.zeros(0, mat2.shape[-1], device = self.device, dtype = self.dtype)
            )
        if bias is not None:
            result = result + bias
        if out_dtype is not None:
            result = result.to(out_dtype)
        elif result.dtype != self.dtype:
            result = result.to(self.dtype)
        return result

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        _gm_lib.impl("_grouped_mm", _grouped_mm_safe_impl, "CUDA")
    logger.info(
        "%s: patched _grouped_mm CUDA dispatch (null HIP kernel on gfx120X, "
        "ROCm <= 7.12 -- bypassed with Python mm fallback)",
        label,
    )
    return _gm_lib


# Subprocesses don't inherit os.add_dll_directory registrations. Replicate
# main.py's Windows ROCm DLL setup so the first `import torch` finds
# amdhip64.dll. Handles retained at module scope so they aren't GC'd.
_ROCM_DLL_HANDLES: list = []
if sys.platform == "win32":

    def _add_rocm_dll_dirs_worker() -> None:
        _candidates: list[str] = []
        for _var in ("HIP_PATH", "ROCM_PATH"):
            _val = os.environ.get(_var)
            if _val:
                _candidates.append(os.path.join(_val, "bin"))
        _default_root = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm"
        )

        def _ver_key(name: str) -> tuple:
            # Numeric tuple key so "10.0" sorts after "7.0".
            parts = []
            for chunk in name.split("."):
                try:
                    parts.append((0, int(chunk)))
                except ValueError:
                    parts.append((1, chunk))
            return tuple(parts)

        try:
            if os.path.isdir(_default_root):
                for _ver in sorted(os.listdir(_default_root), key = _ver_key, reverse = True):
                    _bin = os.path.join(_default_root, _ver, "bin")
                    if os.path.isdir(_bin):
                        _candidates.append(_bin)
        except OSError:
            pass
        for _d in _candidates:
            if os.path.isdir(_d):
                try:
                    _ROCM_DLL_HANDLES.append(os.add_dll_directory(_d))
                except (OSError, AttributeError):
                    pass

    _add_rocm_dll_dirs_worker()
    del _add_rocm_dll_dirs_worker


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
    """Highest-numbered ``/usr/lib/gcc/x86_64-linux-gnu/<N>`` that has BOTH the
    gcc runtime dir AND ``/usr/include/c++/<N>`` headers, or None.

    Ubuntu 24.04 ships gcc-14 runtime but not ``/usr/include/c++/14``; ROCm
    clang-20 picks the highest runtime dir, finds no ``<cstdlib>``, and the HIP
    build fails. The returned path is passed to clang via
    ``--gcc-install-dir``. Mirrors bbf004c in studio/setup.sh (PR #5301).
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
            pypi_status_message = f"Installing {display_name} from PyPI for faster training..."

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

    # ROCm source compilation can take 10-30 min; use a generous timeout.
    # Non-HIP installs keep the pre-existing "no timeout" behaviour so unrelated
    # slow installs (e.g. causal-conv1d source build on Linux aarch64, or
    # unsupported torch/CUDA combos) aren't aborted at 5 minutes.
    _run_kwargs: dict[str, Any] = {
        "stdout": _sp.PIPE,
        "stderr": _sp.STDOUT,
        "text": True,
    }
    if is_hip:
        _run_kwargs["timeout"] = 1800
        # On Ubuntu 24.04 + ROCm clang-20 the HIP source build dies on a missing
        # <cstdlib> (gcc-14 runtime dir lacks C++ headers). Inject
        # --gcc-install-dir for a gcc whose headers exist, respecting any
        # pre-existing one. Mirrors bbf004c in studio/setup.sh (PR #5301).
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
            f"{display_name} installation timed out after " f"{_run_kwargs.get('timeout')}s",
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
            if sys.platform == "win32":
                # No prebuilt wheel and no source toolchain on Windows --
                # expected for packages like causal-conv1d. Log at info so
                # users aren't alarmed by what looks like an error.
                logger.info(
                    "%s is not available on Windows (no prebuilt wheel); skipping",
                    display_name,
                )
                logger.debug("Install output:\n%s", result.stdout)
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
    if sys.platform == "win32":
        logger.info("causal-conv1d: no prebuilt wheel for Windows; skipping")
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
    if sys.platform == "win32":
        logger.info("Skipping flash-linear-attention install: no prebuilt wheel for Windows")
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

    # Probe once; reuse so the --force-reinstall decision and the short-circuit
    # share the same call count (stable for tests).
    already_importable = _flash_linear_attention_importable()
    if already_importable and _flash_linear_attention_current(already_importable = True):
        logger.info("flash-linear-attention already importable at the pinned version")
        return True

    _send_status(
        event_queue,
        f"Installing flash-linear-attention=={_FLA_PACKAGE_VERSION} for faster training...",
    )

    # `--no-deps` blocks the silent torch upgrade; bring non-torch runtime deps in by hand.
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
        _send_status(event_queue, "flash-linear-attention install timed out; continuing")
        return False

    if result.returncode != 0:
        if sys.platform == "win32":
            logger.info(
                "flash-linear-attention not available on Windows (no prebuilt wheel); "
                "continuing on torch fallback"
            )
            logger.debug("Install output:\n%s", result.stdout)
        else:
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
    """Installed-transformers model_types whose modeling file imports `from fla.*`."""
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
    """True iff torch is a ROCm build.

    `torch.version.hip` covers official PyTorch ROCm wheels; AMD SDK / Radeon
    wheels can leave it unset but still encode "rocm" in `torch.__version__`.
    """
    try:
        import torch as _torch
        return bool(
            getattr(_torch.version, "hip", None)
            or "rocm" in getattr(_torch, "__version__", "").lower()
        )
    except Exception:
        return False


def _rocm_classify_unified_memory(props: Any) -> tuple[str, bool]:
    """Classify a ROCm device as unified-memory (APU) or discrete.

    Returns ``(gcn_arch, is_unified)``:
    - ``gcn_arch``: canonical arch string (e.g. ``"gfx1151"``) when a known
      attribute is present, else ``""``.
    - ``is_unified``: ``True`` for AMD APUs with a shared GPU/system-RAM pool
      (gfx1150 Strix Point, gfx1151 Strix Halo) — these need a lower
      ``set_per_process_memory_fraction`` cap to leave OS headroom.

    Classification priority:
    1. ``props.is_integrated`` truthy (hipDeviceProp_t.integrated -- the
       driver's own unified-memory answer; covers APUs beyond the hardcoded
       arch set, e.g. gfx1103 Phoenix iGPUs). Only ever upgrades to unified.
    2. ``gcnArchName`` / variant spellings (stable, naming-independent).
    3. Device-name substring match (last resort when all arch attrs absent;
       AMD SDK / Radeon wheels may not populate them):
         - gfx1150 Strix Point: ``Radeon 890M``, ``Radeon 880M``
         - gfx1151 Strix Halo:  ``Radeon 8060S`` (Ryzen AI MAX+ 395),
                                ``Radeon 8050S`` (cut-down SKU)
    """
    gcn_arch = ""
    for _attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        _v = (getattr(props, _attr, "") or "").split(":")[0].strip()
        if _v:
            gcn_arch = _v
            break

    # Driver's own answer first: hipDeviceProp_t.integrated (exposed as
    # props.is_integrated; same gate PR #5988's UMA safetensors fast-load
    # uses). Strictly additive -- only a truthy value upgrades to unified;
    # 0/absent falls through to the arch/name logic below, so a wheel that
    # omits or zeroes the field can never downgrade the known APU set. This
    # covers unified APUs outside the hardcoded arches (gfx1103 Phoenix
    # iGPUs, future parts) with one universal signal.
    if getattr(props, "is_integrated", 0):
        return gcn_arch, True

    if gcn_arch:
        return gcn_arch, gcn_arch in {"gfx1150", "gfx1151"}

    # Arch attrs absent — fall back to device-name matching.
    dev_lower = (getattr(props, "name", "") or "").lower()
    is_unified = (
        "890m" in dev_lower or "880m" in dev_lower or "8060s" in dev_lower or "8050s" in dev_lower
    )
    return gcn_arch, is_unified


def _tilelang_platform_supported() -> bool:
    """True iff a tilelang 0.1.8 wheel will load: Linux x86_64/aarch64, non-HIP torch.

    HIP excluded: tilelang 0.1.8 has no HIP GEMM and crashes mid-backward.
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
        logger.warning("%s install failed (continuing without it):\n%s", label, result.stdout)
        _send_status(event_queue, f"{label} install failed; continuing")
        return False
    return True


def _ensure_tilelang_backend_unconditional(event_queue: Any) -> bool:
    """Install pinned tilelang + apache-tvm-ffi; two-step repair if a broken tvm-ffi is present.

    Returns True iff both import post-call. Step 1 downgrades a broken tvm-ffi
    with --force-reinstall --no-deps so torch / CUDA stay untouched; step 2 is a
    regular install for missing transitive deps. Bypass via
    UNSLOTH_STUDIO_SKIP_TILELANG_INSTALL=1.
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

    # Step 1: --no-deps keeps --force-reinstall off torch/CUDA via the dep graph.
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

    # Step 2: regular install pulls transitive deps (z3-solver, ml-dtypes) without touching torch.
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
# Wrap transformers' is_{flash_linear_attention,causal_conv1d}_available so the
# first call (at modeling import) drives the install. Models that never query
# the gate (Llama, Gemma, dense Qwen) pay nothing.
# UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring path.


def _rebind_in_already_imported_modules(*, attr_name: str, old_obj: Any, new_obj: Any) -> int:
    """Rebind `attr_name -> new_obj` in every module that imported `old_obj`.

    `from X import Y` creates a local binding that reassigning X.Y won't reach.
    Uses `__dict__.get` to skip lazy `__getattr__` aliases.
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

    # On HIP torch, even installed tilelang crashes FLA's TileLang dispatch.
    # Override with FLA_TILELANG=1.
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
                    logger.warning("%s install raised: %s; falling back to torch", gate_name, exc)
                    ok = False
                logger.info("%s hook done; available=%s", gate_name, ok)
            # post_available_fn handles "gate already True but ancillary kernel broken"
            # (e.g. tilelang missing while FLA imports); skip when install_fn already chained it.
            if ok and not ran_install and post_available_fn is not None:
                try:
                    post_available_fn(event_queue)
                except Exception as exc:
                    logger.warning("%s post-available step raised: %s; continuing", gate_name, exc)
            state["installed"] = True
            return ok

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        wrapper.cache_clear = getattr(original, "cache_clear", lambda: None)  # type: ignore[attr-defined]
        return wrapper

    def _fla_install(eq: Any) -> bool:
        # FLA alone ~2.35x; +tilelang adds ~26%. tilelang is GDN-only (Qwen3.5 family).
        if not _ensure_flash_linear_attention_unconditional(eq):
            logger.info("FLA install did not produce an importable runtime; skipping TileLang")
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
        # FLA imports; repair tilelang if missing or on the broken tvm-ffi list.
        if not _model_wants_tilelang(model_name):
            return
        if _installed_tvm_ffi_version() not in _TVM_FFI_BROKEN_VERSIONS and _tilelang_importable():
            return
        _ensure_tilelang_backend_unconditional(eq)

    def _causal_conv1d_install(eq: Any) -> bool:
        if sys.platform == "win32":
            logger.info("causal-conv1d: no prebuilt wheel for Windows; skipping")
            return False
        ok = _install_package_wheel_first(
            event_queue = eq,
            import_name = "causal_conv1d",
            display_name = "causal-conv1d",
            pypi_name = "causal-conv1d",
            pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
            filename_prefix = "causal_conv1d",
            release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
            release_base_url = ("https://github.com/Dao-AILab/causal-conv1d/releases/download"),
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
        logger.info("Installed fast-path hook on %s (rebound %d modules)", gate_name, rebound)


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


def _activate_transformers_version(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name, hf_token)


def _activate_transformers_version_or_warn(model_name: str, hf_token: str | None = None) -> None:
    """Activate the required transformers version for the MLX fast-path.

    Unlike the non-MLX path (which treats activation failure as fatal and
    reports it via the event queue), the MLX path is intentionally non-fatal:
    it falls through with whatever transformers version is installed. The
    failure used to be swallowed by a bare ``except: pass``, leaving no trace
    and only a confusing downstream crash. Log a warning instead so the cause
    is visible, while keeping the fall-through behaviour.
    """
    try:
        _activate_transformers_version(model_name, hf_token)
    except Exception as exc:
        logger.warning(
            "Failed to activate transformers version for '%s' (MLX); "
            "training may fail if this model requires a specific version. Error: %s",
            model_name,
            exc,
        )


def _mlx_vlm_max_resized_size(width: int, height: int, target: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or target <= 0:
        return width, height
    largest_side = max(width, height)
    if largest_side <= target:
        return width, height
    # Integer formula matches unsloth_zoo's collator (Python round() differs by
    # 1px on half-pixel cases). max(1, _) avoids a zero-side degenerate output.
    new_w = max(1, (width * target + largest_side // 2) // largest_side)
    new_h = max(1, (height * target + largest_side // 2) // largest_side)
    return new_w, new_h


_MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE = {}


def _mlx_vlm_resized_image_layout(processor = None) -> str | None:
    """Return the numpy image layout expected after Unsloth-side VLM resizing."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None
    cls = image_processor.__class__
    key = (getattr(cls, "__module__", ""), getattr(cls, "__qualname__", cls.__name__))
    if key in _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE:
        return _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE[key]
    copied_image_processor = _copy_mlx_vlm_image_processor(image_processor)
    layout = (
        _probe_mlx_vlm_numpy_image_layout(copied_image_processor)
        if copied_image_processor is not None
        else None
    )
    _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE[key] = layout
    return layout


def _copy_mlx_vlm_image_processor(image_processor):
    import copy
    try:
        return copy.deepcopy(image_processor)
    except Exception:
        try:
            return copy.copy(image_processor)
        except Exception:
            return None


def _probe_mlx_vlm_numpy_image_layout(image_processor) -> str | None:
    try:
        import numpy as np
    except ImportError:
        return None

    def _accepts(candidate) -> bool:
        try:
            image_processor(images = [candidate])
            return True
        except TypeError:
            try:
                image_processor([candidate])
                return True
            except Exception:
                return False
        except Exception:
            return False

    # Use an asymmetric image so CHW-vs-HWC mistakes are visible to processors
    # that skip conversion for 3D numpy arrays.
    hwc = np.zeros((64, 96, 3), dtype = np.uint8)
    chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))
    if _accepts(hwc):
        return None
    if _accepts(chw):
        return "chw"
    return None


def _resize_mlx_vlm_image(
    image,
    resize,
    image_layout = None,
):
    if resize is None:
        return image
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return image
    if not isinstance(image, Image.Image):
        return image
    image = image.convert("RGB")
    new_size = _mlx_vlm_max_resized_size(*image.size, int(resize))
    if new_size != image.size:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(new_size, resampling)
    # On resize, hand mlx-vlm a writable RGB ndarray so its PIL-path
    # square-resize is skipped and HF processors don't warn on non-writable
    # views. resize=None above keeps the original PIL.
    array = np.array(image, copy = True)
    if image_layout == "chw":
        return np.ascontiguousarray(array.transpose(2, 0, 1))
    return array


def _resize_mlx_vlm_images(
    value,
    resize,
    image_layout = None,
):
    if isinstance(value, list):
        return [_resize_mlx_vlm_image(image, resize, image_layout = image_layout) for image in value]
    return _resize_mlx_vlm_image(value, resize, image_layout = image_layout)


def _adapt_for_mlx_vlm(
    items,
    resize = None,
    image_layout = None,
):
    """Adapt GPU-path VLM dataset output for mlx-vlm.

    The GPU path embeds PIL images in message content as
    {"type": "image", "image": PIL_Image}, but mlx-vlm's prepare_inputs needs
    images at top-level to produce pixel_values (any model type). Extract them
    and leave bare {"type": "image"} placeholders.
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
                            images.append(
                                _resize_mlx_vlm_image(
                                    img,
                                    resize,
                                    image_layout = image_layout,
                                )
                            )
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
            out["image"] = _resize_mlx_vlm_images(
                item["image"],
                resize,
                image_layout = image_layout,
            )
        elif "images" in item:
            out["images"] = _resize_mlx_vlm_images(
                item["images"],
                resize,
                image_layout = image_layout,
            )
        adapted.append(out)
    return adapted


_MLX_STUDIO_LR_SCHEDULERS = {"linear", "cosine", "constant"}


# Fallback alias map mirroring unsloth_zoo._normalize_mlx_optimizer_name, used
# only when mlx (Apple Silicon) is not importable so Unsloth config validation
# still works on non-MLX hosts. The zoo function stays the source of truth.
_MLX_STUDIO_ADAMW_ALIASES = frozenset(
    (
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_bnb_8bit",
        "paged_adamw_32bit",
        "adamw_torch",
        "adamw_torch_fused",
        "paged_adamw",
        "adamw_32bit",
        "adamw_hf",
        "adamw_anyprecision",
        "adamw_apex_fused",
    )
)
_MLX_STUDIO_NATIVE_OPTIMIZERS = ("adafactor", "adamw", "adam", "sgd", "muon", "lion")


def _normalize_mlx_studio_optimizer(value):
    try:
        from unsloth_zoo.mlx.trainer import _normalize_mlx_optimizer_name
        return _normalize_mlx_optimizer_name(value or "adamw_8bit")
    except (ImportError, ValueError):
        # Missing mlx, or an older unsloth-zoo whose normalizer lacks CUDA/TRL
        # aliases: map common adamw_* names locally so notebook defaults work.
        opt = str(getattr(value, "value", value) or "adamw_8bit").strip().lower()
        opt = opt.rsplit(".", 1)[-1].replace("-", "_")
        if opt in _MLX_STUDIO_ADAMW_ALIASES:
            opt = "adamw"
        if opt not in _MLX_STUDIO_NATIVE_OPTIMIZERS:
            supported = ", ".join(_MLX_STUDIO_NATIVE_OPTIMIZERS)
            raise ValueError(
                f"Unsupported optimizer for MLX training: {value!r}. "
                f"Supported optimizers: {supported}."
            )
        return opt


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
    """Resolve CLI paths and Unsloth local dataset uploads without importing the GPU trainer."""
    from utils.paths import resolve_dataset_path

    all_files: list[str] = []
    for dataset_file in file_paths or []:
        dataset_path = Path(os.path.expanduser(str(dataset_file)))
        if dataset_path.is_absolute():
            file_path = str(dataset_path)
        elif dataset_path.exists():
            file_path = str(dataset_path.resolve())
        else:
            file_path = str(resolve_dataset_path(str(dataset_file)))
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


_MLX_WORKER_COMPLETE = "_mlx_worker_complete"


def _start_mlx_stop_poller(stop_queue):
    import queue as _queue
    import threading

    stop_save = [True]
    stop_requested = [False]
    trainer_ref = [None]

    def is_stop_requested():
        return stop_requested[0]

    def poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout = 0.25)
                if msg and msg.get("type") == _MLX_WORKER_COMPLETE:
                    return
                if msg and msg.get("type") == "stop":
                    stop_save[0] = msg.get("save", True)
                    stop_requested[0] = True
                    trainer = trainer_ref[0]
                    if trainer is not None:
                        trainer.stop_requested = True
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = poll_stop, daemon = True)
    stop_thread.start()
    return stop_save, stop_requested, trainer_ref, is_stop_requested, stop_thread


def _resolve_mlx_output_dir(config, model_name):
    from utils.paths import resolve_output_dir, default_run_dir_name

    output_dir = config.get("output_dir", "")
    if not output_dir:
        output_dir = f"{default_run_dir_name(model_name)}_{int(time.time())}"
        return str(resolve_output_dir(output_dir))
    if config.get("allow_external_output_dir"):
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return str(output_path.resolve())
    return str(resolve_output_dir(output_dir))


def _run_mlx_training(event_queue, stop_queue, config):
    """Self-contained MLX training path for Apple Silicon.

    Uses unsloth_zoo's MLXTrainer directly (no torch/SFTTrainer). Mirrors the
    event_queue protocol so the parent process pump works unchanged.
    """
    import time
    import math
    from pathlib import Path

    def _send(event_type, **kwargs):
        if event_type == "status" and "message" not in kwargs:
            sm = kwargs.get("status_message")
            if sm is not None:
                kwargs["message"] = sm
        event_queue.put({"type": event_type, "ts": time.time(), **kwargs})

    _stop_save, _stop_requested, _trainer_ref, _is_stop_requested, _stop_thread = (
        _start_mlx_stop_poller(stop_queue)
    )

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
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    if mx.metal.is_available():
        info = mx.device_info()
        rec_bytes = info.get("max_recommended_working_set_size", 0) or 0
        if rec_bytes > 0:
            memory_cap = int(rec_bytes * 0.85)
            wired_cap = min(int(rec_bytes), memory_cap)
            mx.set_memory_limit(memory_cap)
            mx.set_wired_limit(wired_cap)

    model_name = config["model_name"]
    hf_token = config.get("hf_token") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    if config.get("use_loftq"):
        message = "LoftQ is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)
    if config.get("is_embedding"):
        message = "Embedding model training is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)
    if config.get("training_type") == "Continued Pretraining":
        message = "Continued Pretraining is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)

    optim_name = _normalize_mlx_studio_optimizer(config.get("optim", "adamw_8bit"))
    lr_scheduler_type = _normalize_mlx_studio_scheduler(config.get("lr_scheduler_type", "linear"))

    # ── 1. Load model ──
    # Force text-only for non-image datasets even on vision-capable models
    # (e.g. Qwen3.5-VL trained on plain alpaca text).
    _send("status", status_message = f"Loading {model_name}...")
    # Pull through resume_from_checkpoint so MLXTrainer.train() can restore
    # optimizer + step state and continue cleanly. Was previously dropped on
    # the floor for the MLX path, so the Resume UI button silently restarted
    # from step 0 (the CUDA path at lines 2729 / 3108 has been forwarding
    # this all along).
    resume_from_checkpoint = config.get("resume_from_checkpoint") or None
    is_dataset_image = bool(config.get("is_dataset_image", False))
    training_type = config.get("training_type", "LoRA/QLoRA")
    use_lora = training_type == "LoRA/QLoRA"
    # Normalize seed; explicit None must not reach the seed chain.
    _raw_seed = config.get("random_seed", 3407)
    random_seed = 3407 if _raw_seed is None else int(_raw_seed)
    # `config.get(k, d)` only fills d when key is missing; handle explicit None too.
    _model_seed = config.get("model_random_state")
    model_random_state = random_seed if _model_seed is None else int(_model_seed)
    _lora_seed = config.get("lora_random_state")
    lora_random_state = random_seed if _lora_seed is None else int(_lora_seed)

    # Malware gate (MLX): a poisoned pickle deserializes on load even with
    # trust_remote_code False, so check HF's security scan (metadata-only) first.
    # For a LoRA, gate the base whose weights deserialize.
    from utils.security import evaluate_file_security

    malware_targets = [model_name]
    try:
        from utils.models.model_config import get_base_model_from_lora_identifier

        # Resolve a LOCAL or REMOTE adapter's base so a remote LoRA base is gated too.
        _base = get_base_model_from_lora_identifier(model_name, config.get("hf_token") or None)
        if _base:
            malware_targets.append(_base)
    except Exception as exc:
        logger.debug("Could not resolve LoRA base for malware scan: %s", exc)
    from utils.security import security_load_subdirs

    for target in dict.fromkeys(malware_targets):
        _fs = evaluate_file_security(
            target, hf_token = hf_token, load_subdirs = security_load_subdirs(target, hf_token)
        )
        if _fs.blocked:
            _send(
                "error",
                error = _fs.reason,
                error_kind = "malware_blocked",
                security = _fs.response_payload(),
            )
            return

    # Consent gate (MLX): the CUDA path gates in run_training_process, but MLX returns
    # before that, so scan auto_map code here before FastMLXModel runs it. Block
    # CRITICAL/HIGH unless pinned-approved; for a LoRA, gate the base whose code runs.
    if config.get("trust_remote_code", False):
        from utils.security import evaluate_remote_code_consent_for_targets

        consent_targets = [model_name]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so a remote LoRA base is gated too.
            base_model = get_base_model_from_lora_identifier(
                model_name, config.get("hf_token") or None
            )
            if base_model:
                consent_targets.append(base_model)
        except Exception as exc:
            logger.debug("Could not resolve LoRA base for consent scan: %s", exc)
        # Scan adapter + base as one combined unit, pinned by a single fingerprint.
        _rc = evaluate_remote_code_consent_for_targets(
            consent_targets,
            hf_token = hf_token,
            trust_remote_code = True,
            approved_fingerprint = config.get("approved_remote_code_fingerprint"),
            subject = config.get("subject"),
        )
        if _rc.blocked:
            _send(
                "error",
                error = (
                    f"Model '{_rc.model_name}' ships custom code flagged as "
                    f"{_rc.max_severity} by the security scan. Review it and "
                    f"re-run with approval to proceed.\n\n{_rc.findings_summary}"
                ),
                error_kind = "remote_code_blocked",
                remote_code = _rc.response_payload(),
            )
            return

    model, tokenizer = FastMLXModel.from_pretrained(
        model_name,
        load_in_4bit = config.get("load_in_4bit", True),
        full_finetuning = not use_lora,
        text_only = None if is_dataset_image else True,
        token = hf_token,
        trust_remote_code = bool(config.get("trust_remote_code", False)),
        random_state = model_random_state,
    )

    is_vlm = bool(is_dataset_image and getattr(model, "_is_vlm_model", False))
    model._is_vlm_model = is_vlm
    vision_image_size = config.get("vision_image_size")
    # DeepSeek OCR uses a coupled preset tuple; skip resize like the Torch path.
    _model_name_lower = str(config.get("model_name", "")).lower()
    _is_deepseek_ocr = "deepseek" in _model_name_lower and "ocr" in _model_name_lower
    if is_vlm and vision_image_size is not None and _is_deepseek_ocr:
        _send(
            "status",
            status_message = (
                "MLX vision image resize ignored for DeepSeek OCR (uses fixed Gundam preset)."
            ),
        )
        vision_image_size = None
    elif is_vlm and vision_image_size is not None:
        vision_image_size = int(vision_image_size)
        _send(
            "status",
            status_message = f"MLX vision image resize: {vision_image_size} (max dimension)",
        )
    # ── 2. Apply LoRA / full FT ──
    # gradient_checkpointing stays a string ("mlx"/"unsloth"/"none"/etc.);
    # get_peft_model and MLXTrainer both accept and handle strings.
    gc_setting = config.get("gradient_checkpointing", "mlx")
    if isinstance(gc_setting, str):
        use_grad_checkpoint = (
            gc_setting if gc_setting.lower() not in ("false", "none", "") else False
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
            random_state = lora_random_state,
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
        finetune_vision = config.get("finetune_vision_layers", False) if is_vlm else False

        if (finetune_attention or finetune_mlp) and not finetune_language and not finetune_vision:
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
            if p.is_dir() and ((p / "dataset_info.json").exists() or (p / "state.json").exists()):
                return load_from_disk(str(p))
        all_files = _resolve_mlx_local_dataset_files(file_paths)
        if not all_files:
            raise ValueError("No local dataset files found")
        loader = _mlx_local_dataset_loader_for_files(all_files)
        return load_dataset(loader, data_files = all_files, split = "train")

    if hf_dataset:
        load_kwargs = {"split": train_split, "token": hf_token}
        if subset:
            load_kwargs["name"] = subset
        dataset = load_dataset(hf_dataset, **load_kwargs)
        dataset = _slice(dataset)
    elif config.get("local_datasets"):
        dataset = _load_local(config["local_datasets"])
        dataset = _slice(dataset)
    elif config.get("s3_config"):
        from core.training.s3_dataset import (
            S3DownloadCancelled,
            prepare_s3_dataset_download,
        )

        _send("status", status_message = "Downloading dataset from S3...")
        try:
            s3_download = prepare_s3_dataset_download(
                config["s3_config"],
                cancel_callback = _is_stop_requested,
            )
            try:
                dataset = _load_local(s3_download.files)
            finally:
                s3_download.cleanup()
        except S3DownloadCancelled:
            _send("complete", output_dir = None, status_message = "Training cancelled")
            return
        dataset = _slice(dataset)
    else:
        raise ValueError("No dataset specified")

    # Eval dataset (separate split or local file)
    eval_dataset = None
    if eval_split and hf_dataset:
        eval_kwargs = {"split": eval_split, "token": hf_token}
        if subset:
            eval_kwargs["name"] = subset
        try:
            eval_dataset = load_dataset(hf_dataset, **eval_kwargs)
        except Exception as e:
            _send("status", status_message = f"Eval split load failed: {e}")
            eval_dataset = None
    elif config.get("local_eval_datasets"):
        eval_dataset = _load_local(config["local_eval_datasets"])

    # ── 3b. Format dataset (VLM or text) ──
    # Reuse the GPU format pipeline for VLM (auto-detects OCR/caption/llava/
    # sharegpt+images) and text (alpaca/sharegpt/chatml → "text" column).
    format_type = config.get("format_type", "")
    custom_format_mapping = config.get("custom_format_mapping")
    dataset_final_format = ""
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
                custom_format_mapping = custom_format_mapping,
                progress_callback = _fmt_progress,
            )
            if vlm_info.get("success"):
                vision_image_layout = (
                    _mlx_vlm_resized_image_layout(tokenizer)
                    if vision_image_size is not None
                    else None
                )
                dataset = _adapt_for_mlx_vlm(
                    vlm_info["dataset"],
                    resize = vision_image_size,
                    image_layout = vision_image_layout,
                )
            else:
                errors = vlm_info.get("errors", [])
                raise ValueError(f"VLM dataset format conversion failed: {'; '.join(errors)}")
            if eval_dataset is not None:
                ev_info = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = True,
                    dataset_name = hf_dataset or "local",
                    custom_format_mapping = custom_format_mapping,
                )
                if ev_info.get("success"):
                    vision_image_layout = (
                        _mlx_vlm_resized_image_layout(tokenizer)
                        if vision_image_size is not None
                        else None
                    )
                    eval_dataset = _adapt_for_mlx_vlm(
                        ev_info["dataset"],
                        resize = vision_image_size,
                        image_layout = vision_image_layout,
                    )

        elif format_type:
            _send("status", status_message = f"Formatting dataset ({format_type})...")
            info = format_and_template_dataset(
                dataset,
                model_name = model_name,
                tokenizer = tokenizer,
                is_vlm = False,
                format_type = format_type,
                dataset_name = hf_dataset or "local",
                custom_format_mapping = custom_format_mapping,
                progress_callback = _fmt_progress,
            )
            if info.get("success", True):
                dataset = info.get("dataset", dataset)
            dataset_final_format = str(info.get("final_format", "") or "").lower()
            if eval_dataset is not None:
                ev = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = False,
                    format_type = format_type,
                    dataset_name = hf_dataset or "local",
                    custom_format_mapping = custom_format_mapping,
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
    # Resolve to ~/.unsloth/studio/outputs/ so the export page finds it
    from utils.paths import ensure_dir

    # Resume must land in the original run dir even when config lacks output_dir.
    resume_dir = config.get("output_dir", "") or _output_dir_from_resume_checkpoint(
        resume_from_checkpoint
    )
    output_dir = _resolve_mlx_output_dir(
        {**config, "output_dir": resume_dir} if resume_dir else config, model_name
    )
    ensure_dir(Path(output_dir))
    _emit_output_dir(event_queue, output_dir)

    # ── 6. Create trainer ──
    eval_steps_val = config.get("eval_steps", 0) or 0
    if isinstance(eval_steps_val, float) and 0 < eval_steps_val < 1:
        eval_steps_val = max(1, int(eval_steps_val * max_steps))
    else:
        eval_steps_val = int(eval_steps_val)

    # Per-element clipping only; trainer owns the None default. Re-validate
    # for direct worker callers (training.py normalizes the main path).
    max_grad_norm = 0.0
    max_grad_value = config.get("max_grad_value")
    if max_grad_value is not None:
        max_grad_value = float(max_grad_value)
        if max_grad_value < 0:
            raise ValueError(
                f"Unsloth MLX: max_grad_value={max_grad_value} must be >= 0 "
                "(0 or None disables elementwise clipping)."
            )
    max_grad_leaf_norm = config.get("max_grad_leaf_norm")
    if max_grad_leaf_norm is not None:
        max_grad_leaf_norm = float(max_grad_leaf_norm)
        if max_grad_leaf_norm < 0:
            raise ValueError(
                f"Unsloth MLX: max_grad_leaf_norm={max_grad_leaf_norm} must be >= 0 "
                "(0 or None disables proportional leaf-norm clipping)."
            )
    weight_decay = config.get("weight_decay", 0.001)
    weight_decay = 0.001 if weight_decay is None else float(weight_decay)

    mlx_config_kwargs = dict(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        max_steps = max_steps,
        learning_rate = lr_value,
        warmup_steps = warmup_steps,
        lr_scheduler_type = lr_scheduler_type,
        optim = optim_name,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        max_grad_value = max_grad_value,
        logging_steps = 1,
        max_seq_length = max_seq_length,
        seed = random_seed,
        use_cce = True,
        compile = True,
        gradient_checkpointing = use_grad_checkpoint,
        streaming = is_vlm,
        packing = bool(config.get("packing", False)),
        output_dir = output_dir,
        save_steps = int(config.get("save_steps", 0) or 0),
        eval_steps = eval_steps_val,
    )

    # Also gates the masking skip below, so defined outside the feature-detect block.
    raw_text_mode = training_type == "Continued Pretraining" or format_type == "raw"

    # Feature-detect optional fields so this PR works without the paired zoo bump.
    _supported_fields = getattr(MLXTrainingConfig, "__dataclass_fields__", {})
    if "cast_norm_output_to_input_dtype" in _supported_fields:
        # Explicit None falls back to True (default).
        _raw_cast = config.get("cast_norm_output_to_input_dtype", True)
        mlx_config_kwargs["cast_norm_output_to_input_dtype"] = (
            True if _raw_cast is None else bool(_raw_cast)
        )
    if "dataset_order" in _supported_fields:
        mlx_config_kwargs["dataset_order"] = "torch_randperm"
    if "max_grad_leaf_norm" in _supported_fields:
        mlx_config_kwargs["max_grad_leaf_norm"] = max_grad_leaf_norm
    if "append_eos" in _supported_fields:
        # Unsloth SFT formatting owns rendered examples; raw/CPT text still
        # needs MLX to append EOS like the CUDA raw-text path.
        mlx_config_kwargs["append_eos"] = bool(raw_text_mode)

    trainer = MLXTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        args = MLXTrainingConfig(**mlx_config_kwargs),
    )
    _trainer_ref[0] = trainer
    if _stop_requested[0]:
        trainer.stop_requested = True

    # Tell the parent eval is configured so the frontend shows the eval chart
    if eval_dataset is not None and eval_steps_val > 0:
        _send("eval_configured")

    # ── 7. Apply train_on_responses_only if requested ──
    # Auto-detect markers from the chat template first, manual table as
    # fallback. Mirror the CUDA skips: raw/CPT text has no chat turns and
    # Alpaca-rendered text lacks the chat markers. Also check the resolved
    # format, since format_type="auto" can land on alpaca or raw text.
    if (
        config.get("train_on_completions", False)
        and not raw_text_mode
        and format_type != "alpaca"
        and dataset_final_format not in ("alpaca", "raw_text")
    ):
        _send("status", status_message = "Configuring response-only training...")
        # No catch: the helper handles detection failures and double misses, so
        # an exception here is a real masking failure that must fail the run,
        # not silently train on full sequences.
        from utils.datasets.completion_masking import apply_completion_masking
        trainer, _masking_applied = apply_completion_masking(
            trainer,
            model_name,
            train_on_responses_only,
            notify = lambda level, message: _send("status", status_message = message),
        )

    # ── 8. Setup wandb / tensorboard ──
    wandb_run = None
    tb_writer = None
    if config.get("enable_wandb", False):
        try:
            import wandb as _wandb

            wandb_token = config.get("wandb_token")
            if wandb_token:
                os.environ["WANDB_API_KEY"] = wandb_token
            # Keep the authenticated subject out of W&B run config (mirrors _sanitize_db_config).
            _wandb_sensitive = {"hf_token", "wandb_token", "s3_config", "subject"}
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
                        **({"train/grad_norm": grad_norm} if grad_norm is not None else {}),
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
                wandb_run.log({"eval/loss": eval_loss, "eval/perplexity": perplexity}, step = step)
            except Exception:
                pass
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("eval/loss", eval_loss, step)
                tb_writer.add_scalar("eval/perplexity", perplexity, step)
            except Exception:
                pass

    trainer.add_eval_callback(_on_eval)

    _opt_ref = [None]
    _orig_build_optimizer = getattr(trainer, "_build_optimizer", None)

    if callable(_orig_build_optimizer):

        def _capture_optimizer(total_steps):
            _opt_ref[0] = _orig_build_optimizer(total_steps)
            return _opt_ref[0]

        trainer._build_optimizer = _capture_optimizer

    # ── 11. Run training ──
    gc.collect()
    mx.synchronize()
    _save_model = trainer.save_model

    def _skip_internal_final_save(*args, **kwargs):
        raise ValueError("worker owns final save")

    trainer.save_model = _skip_internal_final_save
    try:
        trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    finally:
        trainer.save_model = _save_model

    # ── 12. Save and finalize ──
    def _finish_tracking() -> None:
        # Runs on every save/finalize exit so TB/W&B never leak on early return.
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

    def _stop_checkpoint_ok() -> bool:
        if _write_mlx_stop_checkpoint(trainer, _opt_ref[0], output_dir):
            return True
        _send(
            "error",
            error = (
                "Failed to save a resumable checkpoint after stop. "
                "Model files were saved, but this run cannot be resumed."
            ),
            # A user stop finalizes as 'stopped'; keep this failure's error status so history explains it.
            keep_error_status = True,
            # Older checkpoints are stale; resuming would roll back past this stop.
            resume_blocked = True,
        )
        return False

    try:
        if trainer.stop_requested:
            if not _stop_save[0]:
                # Cancel (save=False): skip saving.
                _send("complete", output_dir = None, status_message = "Training cancelled")
            else:
                _send("status", status_message = "Saving stopped model...")
                mx.synchronize()
                trainer.save_model(output_dir)
                # Stop-and-save promises a resumable checkpoint, not just model files.
                if not _stop_checkpoint_ok():
                    return
                _send("complete", output_dir = output_dir, status_message = "Training stopped")
        else:
            _send("status", status_message = "Saving model...")
            mx.synchronize()
            trainer.save_model(output_dir)
            # A save-stop can race the natural final save; it made the same promise.
            if trainer.stop_requested and _stop_save[0] and not _stop_checkpoint_ok():
                return
            _send("complete", output_dir = output_dir, status_message = "Training completed")
    finally:
        _finish_tracking()


def _is_current_process_apple_silicon() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def run_mlx_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
    transformers_activated: bool = False,
) -> None:
    """MLX worker entrypoint shared by Unsloth subprocesses and the CLI adapter."""
    model_name = config["model_name"]

    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.hf_xet_fallback import child_should_disable_xet

    if child_should_disable_xet(config):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    if not transformers_activated:
        # Must precede detect_hardware(): its MLX stack check imports mlx_lm, hence transformers.
        _activate_transformers_version_or_warn(model_name, config.get("hf_token") or None)

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if _hw.DEVICE != _hw.DeviceType.MLX:
        event_queue.put(
            {
                "type": "error",
                "error": "MLX training requires Apple Silicon with the MLX backend available.",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

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

    try:
        try:
            _run_mlx_training(event_queue, stop_queue, config)
        finally:
            try:
                stop_queue.put({"type": _MLX_WORKER_COMPLETE})
            except (EOFError, OSError, ValueError):
                pass
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )


def run_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for progress/status/error events to the parent.
        stop_queue: mp.Queue for stop commands from the parent.
        config: Training config dict with all parameters.
    """
    # Off on Linux (forked datasets map() workers deadlock otherwise); on spawn
    # platforms map() is in-process, so keep tokenizer threads on for faster prep.
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "true" if sys.platform in ("win32", "darwin") else "false"
    )
    os.environ["PYTHONWARNINGS"] = "ignore"  # before imports

    # HTTP-fallback respawn: disable Xet before any huggingface_hub import (the
    # var is read at import time). Mirrors core/inference/worker.py.
    from utils.hf_xet_fallback import child_should_disable_xet

    if child_should_disable_xet(config):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(
            "Xet transport disabled for this training worker (HF_HUB_DISABLE_XET=1).",
            file = sys.stderr,
            flush = True,
        )

    # Offline auto-detect: skip ~25s of HF retries per call when DNS is dead.
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

    # ── 0. MLX FAST-PATH (must run before any torch/transformers imports) ──
    # Apple Silicon uses MLXTrainer directly -- skip torch imports / installs.
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from .training import is_apple_silicon_training_platform, should_use_mlx_training_backend

    mlx_backend_requested = is_apple_silicon_training_platform()

    mlx_transformers_activated = False
    if mlx_backend_requested and _is_current_process_apple_silicon():
        # Must precede detect_hardware(): its MLX stack check imports mlx_lm, hence transformers.
        _activate_transformers_version_or_warn(model_name, config.get("hf_token") or None)
        mlx_transformers_activated = True

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if mlx_backend_requested or should_use_mlx_training_backend(device = _hw.DEVICE):
        run_mlx_training_process(
            event_queue = event_queue,
            stop_queue = stop_queue,
            config = config,
            transformers_activated = mlx_transformers_activated,
        )
        return

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name, config.get("hf_token") or None)
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
    # NemotronH needs trust_remote_code=True to work around config-parsing bugs.
    # Other 5.x models are native and don't need it (it bypasses the compiler,
    # disabling fused CE). Must NOT match Llama-Nemotron (standard Llama arch).
    from utils.security.trusted_org import is_trusted_org_repo

    _NEMOTRON_TRUST_SUBSTRINGS = ("nemotron_h", "nemotron-h", "nemotron-3-nano")
    _lowered = model_name.lower()
    if (
        any(sub in _lowered for sub in _NEMOTRON_TRUST_SUBSTRINGS)
        and (_lowered.startswith("unsloth/") or _lowered.startswith("nvidia/"))
        # Confirm a genuine first-party Hub repo (not a local/spoofed name starting
        # with "unsloth/"); authenticated so private first-party repos resolve.
        and is_trusted_org_repo(model_name, hf_token = config.get("hf_token") or None)
        and not config.get("trust_remote_code", False)
    ):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for Nemotron model: %s",
            model_name,
        )

    # 1a. Malware gate: a poisoned pickle deserializes on load even with
    # trust_remote_code False, so check HF's security scan (metadata-only) first.
    # For a LoRA, gate the base whose weights deserialize.
    from utils.security import evaluate_file_security

    malware_targets = [model_name]
    try:
        from utils.models.model_config import get_base_model_from_lora_identifier

        # Resolve a LOCAL or REMOTE adapter's base so a remote LoRA base is gated too.
        _base = get_base_model_from_lora_identifier(model_name, config.get("hf_token") or None)
        if _base:
            malware_targets.append(_base)
    except Exception as exc:
        logger.debug("Could not resolve LoRA base for malware scan: %s", exc)
    from utils.security import security_load_subdirs

    _ls_hf = config.get("hf_token") or None
    for target in dict.fromkeys(malware_targets):
        _fs = evaluate_file_security(
            target, hf_token = _ls_hf, load_subdirs = security_load_subdirs(target, _ls_hf)
        )
        if _fs.blocked:
            event_queue.put(
                {
                    "type": "error",
                    "error": _fs.reason,
                    "error_kind": "malware_blocked",
                    "security": _fs.response_payload(),
                    "ts": time.time(),
                }
            )
            return

    # 1a'. Consent gate: scan auto_map Python before it runs; refuse CRITICAL/HIGH
    # unless pinned-approved.
    if config.get("trust_remote_code", False):
        from utils.security import evaluate_remote_code_consent_for_targets

        # A LoRA adapter's base is where custom code runs, so gate it too.
        consent_targets = [model_name]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so a remote LoRA base is gated too.
            base_model = get_base_model_from_lora_identifier(
                model_name, config.get("hf_token") or None
            )
            if base_model:
                consent_targets.append(base_model)
        except Exception as exc:
            logger.debug("Could not resolve LoRA base for consent scan: %s", exc)
        # Scan adapter + base as one combined unit, pinned by a single fingerprint.
        _rc = evaluate_remote_code_consent_for_targets(
            consent_targets,
            hf_token = config.get("hf_token") or None,
            trust_remote_code = True,
            approved_fingerprint = config.get("approved_remote_code_fingerprint"),
            subject = config.get("subject"),
        )
        if _rc.blocked:
            event_queue.put(
                {
                    "type": "error",
                    "error": (
                        f"Model '{_rc.model_name}' ships custom code flagged as "
                        f"{_rc.max_severity} by the security scan. Review it and "
                        f"re-run with approval to proceed.\n\n{_rc.findings_summary}"
                    ),
                    "error_kind": "remote_code_blocked",
                    "remote_code": _rc.response_payload(),
                    "ts": time.time(),
                }
            )
            return

    # ── 1b. Install fast-path kernel libraries for the chosen model.
    # 1) causal-conv1d ALWAYS runs eagerly via the substring path: some SSM
    #    modeling files lazy_load it without calling is_causal_conv1d_available.
    # 2) FLA + tilelang: gated by the runtime hook on
    #    is_flash_linear_attention_available (hooks also wrap causal-conv1d).
    # 3) mamba-ssm + flash-attn keep their substring / size gates.
    # 4) UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring path.
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
    # The compiled SFTTrainer disables num_proc if start method isn't "fork".
    # Linux only and safe here (no CUDA context yet); macOS/Windows excluded.
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

    # ── 1d. Stub torchao on Windows ROCm ──
    # See core/_torchao_stub.py for the rationale (no RCCL backend on Windows
    # ROCm). No-op elsewhere. Must run before importing transformers/unsloth_zoo.
    from core._torchao_stub import install_torchao_windows_rocm_stub

    install_torchao_windows_rocm_stub()

    # ── 1e. Ensure torch.distributed helper attrs are present ──
    # Single-GPU never inits the process group, but transformers/trl import
    # these unconditionally.
    _td_stubs = {
        "is_initialized": lambda: False,
        "is_available": lambda: False,
        "is_torchelastic_launched": lambda: False,
        "get_rank": lambda: 0,
        "get_world_size": lambda: 1,
        "barrier": lambda: None,
    }

    try:
        import torch.distributed as _td
        for _name, _stub in _td_stubs.items():
            if not hasattr(_td, _name):
                setattr(_td, _name, _stub)
    except Exception:
        _td_mock = types.ModuleType("torch.distributed")
        for _name, _stub in _td_stubs.items():
            setattr(_td_mock, _name, _stub)
        sys.modules["torch.distributed"] = _td_mock
        try:
            import torch as _torch
            _torch.distributed = _td_mock
        except Exception:
            pass

    # ── 1f. Windows ROCm runtime patches ──
    # torch._grouped_mm has a null HIP kernel on gfx1200 (ROCm ≤ 7.12 Windows),
    # causing 0xC0000005 during training. Root cause: JitDecomp (not
    # torch.compile) dispatches _grouped_mm → null crash; TORCHDYNAMO_DISABLE
    # doesn't cover JitDecomp, so we also override the CUDA dispatch key with a
    # Python fallback. Fixed in torch==2.11.0+rocm7.13.0, so gate on HIP < 7.13.
    # Schema: _grouped_mm(self, mat2, offs=None, bias=None, out_dtype=None);
    #   offs: optional group-split offsets (MoE-style variable-size batches).
    # _WINDOWS_ROCM_GROUPED_MM_LIB keeps the registration alive past return/GC.
    global _WINDOWS_ROCM_GROUPED_MM_LIB
    if sys.platform == "win32":
        _torch_for_rocm = sys.modules.get("torch")
        # Broad check (torch.version.hip OR "rocm" in __version__): AMD SDK /
        # Radeon wheels don't always set torch.version.hip, and without it the
        # BNB pin, dynamo-disable, and _grouped_mm fallback would silently skip.
        _build_version_for_rocm = (
            getattr(_torch_for_rocm, "__version__", "").lower()
            if _torch_for_rocm is not None
            else ""
        )
        _is_win_rocm_torch = bool(
            _torch_for_rocm is not None
            and (
                getattr(getattr(_torch_for_rocm, "version", None), "hip", None)
                or "rocm" in _build_version_for_rocm
            )
        )
        if _is_win_rocm_torch:
            # Disable dynamo (belt-and-suspenders; the JitDecomp patch is the
            # real fix, but this avoids other compile paths).
            if "TORCHDYNAMO_DISABLE" not in os.environ:
                os.environ["TORCHDYNAMO_DISABLE"] = "1"
                logger.info("Windows ROCm: torch.compile (dynamo) disabled")

            # bitsandbytes' import-time get_rocm_gpu_arch() probe runs
            # `hipinfo.exe` from PATH; the AMD torch wheel ships it in the venv
            # Scripts dir, which is on PATH only for activated venvs. Prepend
            # it so the probe succeeds instead of logging a scary (harmless)
            # "Could not detect ROCm GPU architecture" ERROR on every import.
            # Normally inherited from main.py's env, but workers can also be
            # spawned standalone (tests, CLI) -- keep the guard here too.
            _scripts_dir = os.path.dirname(sys.executable)
            if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")):
                import shutil as _shutil
                if not _shutil.which("hipinfo.exe"):
                    os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")

            # BNB picks a rocm DLL from torch.version.hip, but AMD's Windows BNB
            # wheel may ship a DLL whose suffix doesn't match. Detect the actual
            # DLL name and override. Values seeded by the installer are
            # redetectable defaults, while caller overrides remain authoritative.
            if (
                "BNB_ROCM_VERSION" not in os.environ
                or os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize"
            ):
                _bnb_rocm_ver = None
                _found_rocm_bnb = False
                try:
                    import glob as _glob
                    import importlib.util as _ilu
                    import re as _re

                    _bnb_spec = _ilu.find_spec("bitsandbytes")
                    if _bnb_spec and _bnb_spec.submodule_search_locations:
                        _all_vers: list[str] = []
                        for _pkg_dir in _bnb_spec.submodule_search_locations:
                            for _dll in _glob.glob(
                                os.path.join(_pkg_dir, "libbitsandbytes_rocm*.dll")
                            ):
                                _found_rocm_bnb = True
                                _m = _re.search(
                                    r"libbitsandbytes_rocm(\d+)\.dll",
                                    os.path.basename(_dll),
                                )
                                if _m:
                                    _all_vers.append(_m.group(1))
                        # Highest numeric suffix wins (glob order isn't sorted).
                        if _all_vers:
                            _bnb_rocm_ver = max(_all_vers, key = lambda v: int(v))
                except Exception:
                    pass
                # Only when a ROCm bnb DLL actually exists (mirrors main.py):
                # without one the seeded value and its marker stay untouched,
                # so later import fixes can still redetect or opt out. DLL
                # with unparsable name -> seeded value or "72".
                if _found_rocm_bnb:
                    _bnb_rocm_ver = _bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"
                    os.environ["BNB_ROCM_VERSION"] = _bnb_rocm_ver
                    os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
                    logger.info(
                        "Windows ROCm: set BNB_ROCM_VERSION=%s "
                        "(detected from installed BNB wheel; "
                        "overrides torch.version.hip auto-detection)",
                        _bnb_rocm_ver,
                    )

            # Setting BNB_ROCM_VERSION makes bitsandbytes log a benign override
            # notice on import; drop only that record so real errors and mismatch
            # warnings still show.
            if os.environ.get("BNB_ROCM_VERSION"):
                import logging as _logging
                _logging.getLogger("bitsandbytes.cextension").addFilter(
                    lambda _r: "environment variable detected" not in _r.getMessage()
                )

            # Parse HIP version for the kernel-fix gate below, falling back to
            # the rocm version embedded in torch.__version__ when version.hip is
            # unset (AMD SDK / Radeon wheels).
            def _hip_ver_at_least(major: int, minor: int) -> bool:
                _hip_str = getattr(getattr(_torch_for_rocm, "version", None), "hip", None)
                if not _hip_str:
                    # Try the standard "+rocmX.Y.Z" embedded version first.
                    _ver_match = re.search(r"rocm(\d+)\.(\d+)", _build_version_for_rocm)
                    if _ver_match:
                        return (
                            int(_ver_match.group(1)),
                            int(_ver_match.group(2)),
                        ) >= (major, minor)
                    # "+rocmsdk<date>" wheels postdate the gfx120X null-kernel
                    # fix (ROCm 7.13), so treat them as >= 7.13 (no workaround).
                    if "rocmsdk" in _build_version_for_rocm:
                        logger.debug(
                            "Windows ROCm: AMD SDK wheel detected (%r); "
                            "assuming HIP >= %d.%d (rocmsdk wheels post-date "
                            "the gfx120X null-kernel fix)",
                            _build_version_for_rocm,
                            major,
                            minor,
                        )
                        return True
                    return False
                try:
                    _parts = [int(x) for x in str(_hip_str).split(".")[:2]]
                    if len(_parts) < 2:
                        logger.warning(
                            "Windows ROCm: torch.version.hip %r has fewer than "
                            "two components; cannot compare against %d.%d",
                            _hip_str,
                            major,
                            minor,
                        )
                        return False
                    return (_parts[0], _parts[1]) >= (major, minor)
                except ValueError:
                    logger.warning(
                        "Windows ROCm: could not parse torch.version.hip %r as "
                        "a version number; assuming HIP < %d.%d",
                        _hip_str,
                        major,
                        minor,
                    )
                    return False

            # Install the Python fallback only on affected versions (ROCm ≤ 7.12)
            # so 7.13+ uses the real GPU kernel.
            if not _hip_ver_at_least(7, 13):
                try:
                    _WINDOWS_ROCM_GROUPED_MM_LIB = _install_grouped_mm_cpu_fallback(
                        _torch_for_rocm, logger, "Windows ROCm"
                    )
                except Exception as _patch_exc:
                    logger.warning(
                        "Windows ROCm: could not patch _grouped_mm — "
                        "training may crash with 0xC0000005: %s",
                        _patch_exc,
                    )
            else:
                logger.info(
                    "Windows ROCm: HIP >= 7.13 — _grouped_mm kernel is functional, "
                    "skipping Python fallback (AMD fixed gfx1200 null kernel in ROCm 7.13)"
                )

    # ── 1f-linux. Linux ROCm RDNA4 _grouped_mm null kernel ──
    # The guard above is win32-only; RDNA4 (gfx1200/gfx1201) hits the same null
    # HIP _grouped_mm kernel on Linux at ROCm <= 7.12 (fixed in 7.13;
    # ROCm/TheRock #5284). Gate on arch + HIP < 7.13 so NVIDIA/CUDA and non-RDNA4
    # AMD are never touched; no-op on fixed runtimes.
    if sys.platform.startswith("linux") and _hw.IS_ROCM:
        try:
            _torch_lin = sys.modules.get("torch")
            if _torch_lin is not None and _torch_lin.cuda.is_available():
                # Prefer torch.version.hip, else the rocmX.Y embedded in
                # torch.__version__ (AMD SDK / Radeon wheels leave version.hip
                # unset). Unknown version on a recognized gfx120X ROCm build ->
                # treat as affected unless it is a post-fix rocmsdk wheel.
                _hip_str = str(getattr(getattr(_torch_lin, "version", None), "hip", "") or "")
                _ver = getattr(_torch_lin, "__version__", "").lower()
                _m = re.match(r"(\d+)\.(\d+)", _hip_str) or re.search(r"rocm(\d+)\.(\d+)", _ver)
                if _m:
                    _hip_lt_713 = (int(_m.group(1)), int(_m.group(2))) < (7, 13)
                else:
                    _hip_lt_713 = "rocmsdk" not in _ver
                # Scan every visible GPU: device_map="balanced" can place layers
                # on a later RDNA4 card, so device 0 alone is not enough. Match
                # gfx120X by arch, or by RX 9000 / R9700 name when the wheel omits
                # gcnArchName (name check only when arch is unknown).
                _rdna4 = False
                for _i in range(_torch_lin.cuda.device_count()):
                    _props = _torch_lin.cuda.get_device_properties(_i)
                    _lin_arch, _ = _rocm_classify_unified_memory(_props)
                    _lin_name = (getattr(_props, "name", "") or "").lower()
                    if _lin_arch.lower() in ("gfx1200", "gfx1201") or (
                        not _lin_arch and re.search(r"rx\s*90[0-9]0|r9700", _lin_name)
                    ):
                        _rdna4 = True
                        break
                if _rdna4 and _hip_lt_713:
                    _WINDOWS_ROCM_GROUPED_MM_LIB = _install_grouped_mm_cpu_fallback(
                        _torch_lin, logger, "Linux ROCm gfx120X"
                    )
        except Exception as _gm_lin_exc:
            logger.warning("Linux ROCm gfx120X: could not patch _grouped_mm: %s", _gm_lin_exc)

    # ── 1g. ROCm OOM guard ──
    # On ROCm, exhausting VRAM can hang the HIP driver instead of raising.
    # set_per_process_memory_fraction caps the allocator so PyTorch raises
    # OutOfMemoryError first (NVIDIA already has a graceful OOM path).
    # Unified-memory APUs (gfx1150/gfx1151) share GPU+system RAM, so use 0.80
    # vs 0.90 for discrete. Classify via gcnArchName, else device-name markers.
    # Non-fatal: skipped if torch is not importable.
    if _hw.IS_ROCM:
        try:
            import torch as _torch_mem
            if _torch_mem.cuda.is_available():
                # Classify unified vs discrete via _rocm_classify_unified_memory
                # (see its docstring for classification priority).
                _props = _torch_mem.cuda.get_device_properties(0)
                _dev_name = _props.name
                _gcn_arch, _is_unified = _rocm_classify_unified_memory(_props)
                if _is_unified and not _gcn_arch:
                    logger.debug(
                        "ROCm OOM guard: gcnArchName absent -- inferred "
                        "unified memory from device name %r; applying unified cap",
                        _dev_name,
                    )
                # Unified hosts on native Windows: mem_get_info's total is the
                # WDDM budget the driver grants HIP (BIOS carve + ~half of the
                # remaining RAM) -- the OS share is already outside it, so the
                # Linux 0.80 starve-protection double-taxes (48.49 GiB budget →
                # 38.79 allowed) and blocks loads that fit in free memory.
                # 1.0 removes the double-tax. Current AMD Windows wheels only
                # enforce sub-1.0 fractions (measured on gfx1151: 0.5 caps,
                # 1.0 still allocates past the budget via WDDM overcommit), so
                # 1.0 behaves like torch's uncapped default, with WDDM
                # arbitrating residency; on wheels that do enforce it, it caps
                # at exactly the driver-granted budget. On Linux the total
                # spans nearly all RAM, so keep the 0.80 OS headroom there.
                if _is_unified:
                    _mem_fraction = 1.0 if sys.platform == "win32" else 0.80
                else:
                    _mem_fraction = 0.90
                _torch_mem.cuda.set_per_process_memory_fraction(_mem_fraction)
                logger.info(
                    "ROCm OOM guard: set_per_process_memory_fraction(%.2f) — "
                    "%s memory host (%s, %s)",
                    _mem_fraction,
                    "unified" if _is_unified else "discrete",
                    _dev_name,
                    _gcn_arch or "unknown arch",
                )
                # Unified Windows APUs: the WDDM budget is user-raisable, but
                # nothing on the box says so -- users see "48 GB VRAM" on a
                # 96 GB machine and assume an Unsloth bug. Say where the limit
                # comes from and how to raise it.
                if _is_unified and sys.platform == "win32":
                    try:
                        import psutil as _psutil

                        _phys = _psutil.virtual_memory().total
                        _granted = _torch_mem.cuda.mem_get_info(0)[1]
                        if _granted < 0.75 * _phys:
                            logger.info(
                                "Windows grants the GPU %.1f GiB of %.1f GiB "
                                "system RAM (driver/WDDM budget). To raise it: "
                                "increase the BIOS UMA frame buffer size, or "
                                "AMD Software > Performance > Tuning > "
                                "Variable Graphics Memory.",
                                _granted / 1024**3,
                                _phys / 1024**3,
                            )
                    except Exception:
                        pass
        except Exception as _oom_guard_err:
            logger.debug("Could not set GPU memory fraction: %s", _oom_guard_err)

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.training import TrainingProgress
        from core.training.trainer import UnslothTrainer
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
            default_run_dir_name,
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
    # Embedding models use a different pipeline (FastSentenceTransformer +
    # SentenceTransformerTrainer + MultipleNegativesRankingLoss), so branch early
    # and handle the whole flow in a self-contained function.
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
        if (progress.step == 0 and progress.total_steps > 0) or has_train_loss or has_eval_loss:
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
    # Order: detect → dataset → model → prepare → train. Dataset processing runs
    # BEFORE model loading so both never occupy VRAM at once.
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None

        # ── 4a. Lightweight detection + tokenizer (no VRAM) ──
        _send_status(event_queue, "Detecting model type...")
        trainer.pre_detect_and_load_tokenizer(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
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
            dataset_streaming = config.get("dataset_streaming", False),
            eval_steps = config.get("eval_steps", 0.00),
            dataset_slice_start = config.get("dataset_slice_start"),
            dataset_slice_end = config.get("dataset_slice_end"),
            is_cpt = _is_cpt_for_dataset,
            s3_config = config.get("s3_config"),
        )

        if isinstance(dataset_result, tuple):
            dataset, eval_dataset = dataset_result
        else:
            dataset = dataset_result
            eval_dataset = None

        # Disable eval if eval_steps <= 0
        eval_steps = config.get("eval_steps", 0.00)
        if eval_steps is not None and float(eval_steps) <= 0:
            eval_dataset = None

        # Tell the parent eval is configured so the frontend shows
        # "Waiting for first evaluation step..." instead of "not configured".
        if eval_dataset is not None:
            event_queue.put(
                {
                    "type": "eval_configured",
                    "ts": time.time(),
                }
            )

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error or "Failed to load dataset",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── Start tqdm monitor early to capture download + tokenization bars ──
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
                            _send_status(event_queue, f"{desc.strip()} {pct}% ({n:,}/{total:,})")
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
        # Watchdog lets the parent recover a stalled Xet download via respawn.
        _send_status(event_queue, "Loading model...")
        from utils.hf_xet_fallback import start_watchdog

        event_queue.put({"type": "model_load_started", "ts": time.time()})
        _load_watchdog_stop = start_watchdog(
            repo_ids = [model_name],
            on_stall = lambda msg: event_queue.put(
                {"type": "stall", "message": msg, "ts": time.time()}
            ),
            xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1",
        )
        # Latest-sidecar models load 16-bit here too: bnb 4-bit feeds quantized
        # expert weights into unvalidated paths (same flip as the chat worker).
        _train_load_in_4bit = config["load_in_4bit"]
        if _train_load_in_4bit:
            from utils.transformers_version import latest_tier_active_for
            if latest_tier_active_for(model_name, hf_token):
                _train_load_in_4bit = False
                logger.info(
                    "Latest-transformers sidecar active for %s - forcing a 16-bit "
                    "training load (4-bit is disabled for brand-new architectures)",
                    model_name,
                )

        try:
            success = trainer.load_model(
                model_name = model_name,
                max_seq_length = config["max_seq_length"],
                load_in_4bit = _train_load_in_4bit,
                full_finetuning = not use_lora,
                hf_token = hf_token,
                is_dataset_image = config.get("is_dataset_image", False),
                is_dataset_audio = config.get("is_dataset_audio", False),
                trust_remote_code = config.get("trust_remote_code", False),
                gpu_ids = config.get("resolved_gpu_ids"),
            )
        finally:
            _load_watchdog_stop.set()
            event_queue.put({"type": "model_load_completed", "ts": time.time()})
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
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
            # embed_tokens (if included) goes to modules_to_save — trained
            # full-precision at embedding_learning_rate. lm_head stays a LoRA
            # target for merge compatibility (see unsloth PR #4106).
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
                use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        elif use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora = True,
                finetune_vision_layers = config.get("finetune_vision_layers", True),
                finetune_language_layers = config.get("finetune_language_layers", True),
                finetune_attention_modules = config.get("finetune_attention_modules", True),
                finetune_mlp_modules = config.get("finetune_mlp_modules", True),
                target_modules = config.get("target_modules"),
                lora_r = config.get("lora_r", 16),
                lora_alpha = config.get("lora_alpha", 16),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora = False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error or "Failed to prepare model",
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

        # embedding_learning_rate is validated by Pydantic (Optional[float],
        # gt=0, lt=1.0); if present it's already a finite float in range.
        embedding_lr_value = config.get("embedding_learning_rate")
        if is_cpt:
            if cpt_trains_embeddings:
                if embedding_lr_value is None:
                    # Default embedding_learning_rate = lr/10 (Unsloth CPT notebook).
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
            output_dir = build_default_output_dir_name(
                model_name,
                config.get("project_name"),
            )
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))
        _emit_output_dir(event_queue, output_dir)

        tensorboard_dir = config.get("tensorboard_dir")
        if config.get("enable_tensorboard", False):
            tensorboard_dir = str(resolve_tensorboard_dir(tensorboard_dir))
            ensure_dir(Path(tensorboard_dir))

        # Start training directly — no inner thread, we ARE the subprocess.
        dataset_display = config.get("hf_dataset", "") or config.get("uploaded_file", "") or ""
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
            train_on_completions = False if is_cpt else config.get("train_on_completions", False),
            enable_wandb = config.get("enable_wandb", False),
            wandb_project = config.get("wandb_project", "unsloth-training"),
            wandb_token = config.get("wandb_token"),
            enable_tensorboard = config.get("enable_tensorboard", False),
            tensorboard_dir = tensorboard_dir,
            eval_dataset = eval_dataset,
            eval_steps = eval_steps,
            max_seq_length = config.get("max_seq_length", 2048),
            vision_image_size = config.get("vision_image_size"),
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
        _exc_str = str(exc).lower()
        _is_oom = (
            "out of memory" in _exc_str
            or "hip out of memory" in _exc_str
            or "cuda out of memory" in _exc_str
            or type(exc).__name__ == "OutOfMemoryError"
        )
        if _is_oom:
            _oom_msg = (
                "GPU ran out of VRAM during training.\n"
                "To fix: reduce max_seq_length (e.g. 2048–4096), enable "
                "gradient_checkpointing=True, lower per_device_train_batch_size, "
                "or use a smaller model / higher quantization."
            )
            logger.error("Training stopped: GPU OOM — %s", exc)
            event_queue.put(
                {
                    "type": "error",
                    "error": _oom_msg,
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        else:
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


def _emit_output_dir(event_queue: Any, output_dir: str) -> None:
    try:
        event_queue.put({"type": "output_dir", "output_dir": output_dir, "ts": time.time()})
    except Exception:
        pass


def _mlx_has_checkpoint_at_step(output_dir, step: int) -> bool:
    if step <= 0:
        return False
    from core.training.resume import is_resume_checkpoint_valid
    return is_resume_checkpoint_valid(
        Path(output_dir) / f"checkpoint-{step}", expected_step = step, backend = "mlx"
    )


def _write_mlx_stop_checkpoint(trainer, optimizer, output_dir) -> bool:
    """Write a full resume checkpoint for a stopped MLX run.

    Returns True when a checkpoint for the current training step exists.
    """
    step = int(getattr(trainer, "_global_step", 0) or 0)
    # A periodic save or a resumed run may already cover the current step.
    if _mlx_has_checkpoint_at_step(output_dir, step):
        return True
    if step <= 0 or optimizer is None:
        return False
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    if ckpt_dir.is_symlink():
        # Refuse a symlinked dir: it could redirect writes outside output_dir.
        logger.error("Refusing to write MLX stop checkpoint through symlink: %s", ckpt_dir)
        return False
    try:
        ckpt_dir.mkdir(parents = True, exist_ok = True)
        from unsloth_zoo.mlx.utils import (
            save_optimizer_state,
            save_trainable_adapters,
            save_trainer_state,
        )

        save_trainable_adapters(trainer.model, str(ckpt_dir))
        save_optimizer_state(optimizer, str(ckpt_dir))
        save_trainer_state(
            {
                "global_step": step,
                "train_loss_history": list(getattr(trainer, "_train_loss_history", [])),
            },
            str(ckpt_dir),
        )
        logger.info("Saved stop checkpoint to %s", ckpt_dir)
    except Exception:
        logger.exception("Failed to write stop checkpoint under %s", output_dir)
    return _mlx_has_checkpoint_at_step(output_dir, step)


def _run_embedding_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Self-contained embedding model training pipeline.

    Uses FastSentenceTransformer + SentenceTransformerTrainer +
    MultipleNegativesRankingLoss — separate from UnslothTrainer's LLM/VLM/audio
    paths. Mirrors the reference embedding notebooks:
      All_MiniLM_L6_v2.py, BGE_M3.py, EmbeddingGemma_300M.py,
      ModernBert.py, Qwen3_Embedding_0_6B.py
    """
    import math
    import queue as _queue
    import threading

    model_name = config["model_name"]
    training_start_time = time.time()

    # ── 1. Import embedding-specific libraries ──
    _send_status(event_queue, "Importing embedding libraries...")
    try:
        # Recover from a namespace-package shadow (embedding imports unsloth directly).
        from core.import_guards import ensure_real_packages

        ensure_real_packages("unsloth_zoo", "unsloth")
        from unsloth import FastSentenceTransformer, is_bfloat16_supported
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from datasets import Dataset
        from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset
        from transformers import TrainerCallback
        from utils.paths import datasets_root, resolve_output_dir, default_run_dir_name
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

        # Malware gate (embedding): a poisoned pickle deserializes on load even with
        # trust_remote_code False, so check HF's security scan (metadata-only) first.
        # For a LoRA, gate the base whose weights deserialize.
        from utils.security import evaluate_file_security

        malware_targets = [model_name]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier
            _base = get_base_model_from_lora_identifier(model_name, hf_token)
            if _base:
                malware_targets.append(_base)
        except Exception as exc:
            logger.debug("Could not resolve LoRA base for malware scan: %s", exc)
        from utils.security import security_load_subdirs

        for target in dict.fromkeys(malware_targets):
            _fs = evaluate_file_security(
                target, hf_token = hf_token, load_subdirs = security_load_subdirs(target, hf_token)
            )
            if _fs.blocked:
                event_queue.put(
                    {
                        "type": "error",
                        "error": _fs.reason,
                        "error_kind": "malware_blocked",
                        "security": _fs.response_payload(),
                        "ts": time.time(),
                    }
                )
                return

        # Consent gate (embedding): scan any auto_map code before it runs; block
        # CRITICAL/HIGH unless pinned-approved. A no-op without auto_map.
        if config.get("trust_remote_code", False):
            from utils.security import evaluate_remote_code_consent_for_targets

            consent_targets = [model_name]
            try:
                from utils.models.model_config import get_base_model_from_lora_identifier
                _cbase = get_base_model_from_lora_identifier(model_name, hf_token)
                if _cbase:
                    consent_targets.append(_cbase)
            except Exception as exc:
                logger.debug("Could not resolve LoRA base for consent scan: %s", exc)
            # Scan adapter + base as one combined unit, pinned by a single fingerprint.
            _rc = evaluate_remote_code_consent_for_targets(
                consent_targets,
                hf_token = hf_token,
                trust_remote_code = True,
                approved_fingerprint = config.get("approved_remote_code_fingerprint"),
                subject = config.get("subject"),
            )
            if _rc.blocked:
                event_queue.put(
                    {
                        "type": "error",
                        "error": (
                            f"Model '{_rc.model_name}' ships custom code flagged as "
                            f"{_rc.max_severity} by the security scan. Review it and "
                            f"re-run with approval to proceed.\n\n{_rc.findings_summary}"
                        ),
                        "error_kind": "remote_code_blocked",
                        "remote_code": _rc.response_payload(),
                        "ts": time.time(),
                    }
                )
                return

        model = FastSentenceTransformer.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            full_finetuning = not use_lora,
            token = hf_token,
        )
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
            # Normalize "none"/empty → False.
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

        def _load_local_embedding_dataset(dataset_paths: list[str]):
            all_files: list[str] = []
            for dataset_file in dataset_paths:
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
                    raise ValueError(f"No supported data files in directory: {file_path_obj}")
                else:
                    all_files.append(file_path)

            if not all_files:
                raise ValueError("No local dataset files found")

            first_ext = Path(all_files[0]).suffix.lower()
            if first_ext in (".json", ".jsonl"):
                loader = "json"
            elif first_ext == ".csv":
                loader = "csv"
            elif first_ext == ".parquet":
                loader = "parquet"
            else:
                raise ValueError(f"Unsupported local dataset format: {all_files[0]}")
            return load_dataset(loader, data_files = all_files, split = "train")

        if hf_dataset and hf_dataset.strip():
            hf_token = config.get("hf_token", "")
            hf_token = hf_token if hf_token and hf_token.strip() else None
            dataset = load_dataset(
                hf_dataset.strip(),
                subset,
                split = train_split,
                token = hf_token,
            )
        elif local_datasets:
            dataset = _load_local_embedding_dataset(local_datasets)
        elif config.get("s3_config"):
            from core.training.s3_dataset import (
                S3DownloadCancelled,
                prepare_s3_dataset_download,
            )

            _send_status(event_queue, "Downloading dataset from S3...")
            s3_download = None
            try:
                s3_download = prepare_s3_dataset_download(
                    config["s3_config"],
                    cancel_callback = lambda: _should_stop,
                )
                dataset = _load_local_embedding_dataset(s3_download.files)
            except S3DownloadCancelled:
                event_queue.put(
                    {
                        "type": "complete",
                        "output_dir": None,
                        "status_message": "Training cancelled",
                        "ts": time.time(),
                    }
                )
                return
            finally:
                if s3_download is not None:
                    s3_download.cleanup()
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
        output_dir = build_default_output_dir_name(
            model_name,
            config.get("project_name"),
        )
    output_dir = str(resolve_output_dir(output_dir))
    _emit_output_dir(event_queue, output_dir)

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

    # warmup: prefer warmup_ratio (standard for embedding scripts), else steps
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
        """Send training progress events to the parent via event_queue."""

        def on_log(
            self,
            args,
            state,
            control,
            logs = None,
            **kwargs,
        ):
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
