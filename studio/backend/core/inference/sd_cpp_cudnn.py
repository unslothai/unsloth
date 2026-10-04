# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cuDNN fused attention for the Unsloth stable-diffusion.cpp fork.

A fork build made with ``GGML_CUDA_CUDNN=ON`` opens ``libcudnn.so.9`` with ``dlopen`` the first time an attention op
runs and falls back to the ggml kernels when it cannot. It only gets faster with a cuDNN 9 built for the SAME CUDA major
as the binary: torch cu130's own cuDNN (CUDA 13) loads, builds no plan inside a CUDA 12 binary, and falls back.

So for an NVIDIA sm80+ card on Linux, and only when the binary carries the cuDNN build, this module provides a
matching cuDNN from PyPI in a Studio-managed directory (``pip install --target``, never the venv: the CUDA 12 and
CUDA 13 wheels both write ``nvidia/cudnn/lib/libcudnn.so.9``, so installing one beside torch would replace torch's own)
and names it to the sd.cpp child through ``GGML_CUDA_CUDNN_LIB``. Nothing here loads cuDNN into this process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# auto (default) installs on demand; 0 / false / off never installs and never sets the env.
CUDNN_INSTALL_ENV = "UNSLOTH_SD_CPP_CUDNN"
# Read by the fork (cudnn-attn-graph.cpp): the library to dlopen. A user-exported value always wins.
GGML_CUDNN_LIB_ENV = "GGML_CUDA_CUDNN_LIB"
# The fork's own kill switch; 0 restores the ggml kernels everywhere.
GGML_CUDNN_ATTN_ENV = "GGML_CUDA_CUDNN_ATTN"
# The fork's eligibility floor (cc >= 80); below it the library is never opened.
CUDNN_MIN_CC = (8, 0)

# Present in every binary built with GGML_CUDA_CUDNN=ON (the getenv name), absent otherwise.
_CUDNN_BUILD_MARKER = b"GGML_CUDA_CUDNN_LIB"
_CUDART_MARKERS = {12: b"libcudart.so.12\x00", 13: b"libcudart.so.13\x00"}


@dataclass(frozen = True)
class CudnnRuntime:
    """The exact wheels tested for one CUDA major. Exact pins, never a floor: a different cuDNN release can pick
    other SDPA engines, and only this one was measured."""

    cuda_major: int
    cudnn_package: str
    cudnn_version: str
    nvrtc_package: str
    nvrtc_version: str

    @property
    def cudnn_numeric(self) -> int:
        # cudnnGetVersion(): major * 10000 + minor * 100 + patch
        major, minor, patch = (int(x) for x in self.cudnn_version.split(".")[:3])
        return major * 10000 + minor * 100 + patch

    @property
    def dirname(self) -> str:
        return f"cu{self.cuda_major}-cudnn{self.cudnn_version}-nvrtc{self.nvrtc_version}"

    def requirements(self) -> list[str]:
        return [
            f"{self.cudnn_package}=={self.cudnn_version}",
            f"{self.nvrtc_package}=={self.nvrtc_version}",
        ]


# The prebuilt is a CUDA 12.8 build. NVRTC is what cuDNN's runtime-compiled fused attention engines dlopen; the cuDNN
# libraries find it through their RUNPATH ($ORIGIN/../../cuda_nvrtc/lib), so the two wheels share one --target dir.
# cuBLAS (the other wheel dependency) is not installed: the binary already links the copy the prebuilt bundles.
CUDNN_RUNTIMES: dict[int, CudnnRuntime] = {
    12: CudnnRuntime(12, "nvidia-cudnn-cu12", "9.27.0.42", "nvidia-cuda-nvrtc-cu12", "12.8.93"),
}

_LIB_RELPATH = Path("nvidia") / "cudnn" / "lib" / "libcudnn.so.9"
_NVRTC_RELPATH = Path("nvidia") / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
_MARKER_NAME = "unsloth-cudnn.json"
# Two ~0.85 GB wheels unpack to ~1.4 GB; the staging copy and the download cache both sit on this volume.
_MIN_FREE_BYTES = 4 << 30
_INSTALL_TIMEOUT_S = 1800
_VERIFY_TIMEOUT_S = 120
_LOCK_TIMEOUT_S = _INSTALL_TIMEOUT_S + 60
_PYPI_PROBE_URL = "https://pypi.org/simple/nvidia-cudnn-cu12/"
_INDEX_ENVS = (
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
)

_SCAN_MEMO: dict[tuple[str, int, int], tuple[bool, Optional[int]]] = {}
_SCAN_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
# One failed install per process and runtime: a load must not retry a 0.85 GB download that just failed.
_FAILED: dict[int, str] = {}

StatusCb = Optional[Callable[[str], None]]


def _off(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("0", "false", "no", "off")


def install_enabled(environ: Optional[dict] = None) -> bool:
    environ = os.environ if environ is None else environ
    return not _off(environ.get(CUDNN_INSTALL_ENV))


def _scan_file(path: Path) -> tuple[bool, Optional[int]]:
    """(carries the cuDNN build marker, CUDA runtime major it links) for one ELF, by a streamed byte scan."""
    try:
        st = path.stat()
    except OSError:
        return False, None
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    with _SCAN_LOCK:
        if key in _SCAN_MEMO:
            return _SCAN_MEMO[key]
    needles = [_CUDNN_BUILD_MARKER, *_CUDART_MARKERS.values()]
    overlap = max(len(n) for n in needles)
    found: set[bytes] = set()
    try:
        with path.open("rb") as fh:
            tail = b""
            while True:
                chunk = fh.read(8 << 20)
                if not chunk:
                    break
                window = tail + chunk
                for needle in needles:
                    if needle not in found and needle in window:
                        found.add(needle)
                tail = window[-overlap:]
    except OSError:
        return False, None
    major = next((m for m, n in _CUDART_MARKERS.items() if n in found), None)
    result = (_CUDNN_BUILD_MARKER in found, major)
    with _SCAN_LOCK:
        _SCAN_MEMO[key] = result
    return result


def binary_cudnn_build(binary: Optional[str]) -> tuple[bool, Optional[int]]:
    """Whether ``binary`` (sd-cli or sd-server) was built with the fork's cuDNN attention, and the CUDA runtime major
    it links. ggml may be linked in statically (local builds) or ship as ``libggml*.so`` beside the binary."""
    if not binary:
        return False, None
    path = Path(binary)
    carries, major = _scan_file(path)
    if carries and major is not None:
        return carries, major
    try:
        siblings = sorted(path.resolve().parent.glob("libggml*.so*"))
    except OSError:
        siblings = []
    for lib in siblings:
        lib_carries, lib_major = _scan_file(lib)
        carries = carries or lib_carries
        major = major or lib_major
    if major is None:
        # A dynamically linked binary names libcudart in its own NEEDED list; a bundle may also carry it beside.
        for m in _CUDART_MARKERS:
            if (path.resolve().parent / f"libcudart.so.{m}").exists():
                major = m
                break
    return carries, major


def managed_root() -> Path:
    from utils.paths.storage_roots import studio_bin_root

    return studio_bin_root() / "sd-cpp-cudnn"


def _verified_library(directory: Path, runtime: CudnnRuntime) -> Optional[str]:
    try:
        marker = json.loads((directory / _MARKER_NAME).read_text())
    except (OSError, ValueError):
        return None
    if marker.get("requirements") != runtime.requirements():
        return None
    lib = directory / _LIB_RELPATH
    if not lib.is_file() or not (directory / _NVRTC_RELPATH).is_file():
        return None
    return str(lib)


def installed_library(runtime: CudnnRuntime, root: Optional[Path] = None) -> Optional[str]:
    """The verified managed library for ``runtime``, or None."""
    return _verified_library((root or managed_root()) / runtime.dirname, runtime)


def _uv_executable() -> Optional[str]:
    try:
        from utils.mlx_repair import _uv_executable as find_uv

        return find_uv()
    except Exception:  # noqa: BLE001
        return shutil.which("uv")


def _child_env() -> dict[str, str]:
    try:
        from utils.child_stdio import utf8_child_env
        from utils.native_path_leases import child_env_without_native_path_secret

        return utf8_child_env(child_env_without_native_path_secret())
    except Exception:  # noqa: BLE001
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        return env


def _reachable(url: str) -> bool:
    try:
        from .diffusion_nvfp4_install import _reachable as probe

        return probe(url)
    except Exception:  # noqa: BLE001 - an unanswerable probe means no network
        return False


def install_command(
    runtime: CudnnRuntime, target: Path, uv: Optional[str]
) -> list[str]:
    """``--target`` a Studio-managed dir, ``--no-deps``, wheels only: nothing in the venv can change."""
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--target", str(target)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(target),
        ]
    return [*cmd, "--no-deps", "--only-binary", ":all:", *runtime.requirements()]


def _verify_command(lib: Path) -> list[str]:
    # A fresh interpreter: a CUDA 12 cuDNN must never be mapped into this process beside torch's own.
    code = (
        "import ctypes,sys;"
        "h=ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_LOCAL);"
        "f=h.cudnnGetVersion;f.restype=ctypes.c_size_t;print(f())"
    )
    return [sys.executable, "-c", code, str(lib)]


def _venv_snapshot() -> dict[str, str]:
    """torch and every nvidia-* distribution in this environment: the install must leave them all as they were."""
    try:
        import importlib.metadata as md
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for dist in md.distributions():
        name = str(dist.metadata.get("Name") or "").lower()
        if name == "torch" or name.startswith("nvidia-"):
            out[name] = str(dist.version)
    return out


def _run(cmd: list[str], timeout: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            env = _child_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return result.returncode == 0, (result.stdout or "").strip()[-2000:]


def _file_lock(path: Path):
    try:
        from filelock import FileLock
    except ImportError:
        return None
    return FileLock(str(path))


def ensure_library(
    runtime: CudnnRuntime,
    *,
    allow_install: bool = True,
    status_cb: StatusCb = None,
    root: Optional[Path] = None,
) -> tuple[Optional[str], Optional[str]]:
    """``(library path, None)`` once a verified copy exists, else ``(None, why not)``. Installs at most once per process
    and runtime; every failure leaves the sd.cpp run on its ggml kernels."""
    root = root or managed_root()
    dest = root / runtime.dirname
    found = _verified_library(dest, runtime)
    if found:
        return found, None
    if not allow_install:
        return None, "the CUDA 12 cuDNN is not installed and this load does not download"
    if not install_enabled():
        return None, f"{CUDNN_INSTALL_ENV}=0"
    if runtime.cuda_major in _FAILED:
        return None, _FAILED[runtime.cuda_major]
    with _INSTALL_LOCK:
        found = _verified_library(dest, runtime)
        if found:
            return found, None
        if runtime.cuda_major in _FAILED:
            return None, _FAILED[runtime.cuda_major]
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError as exc:
            return None, f"cannot create {root.name}: {exc}"
        lock = _file_lock(root / ".install.lock")
        try:
            if lock is not None:
                lock.acquire(timeout = _LOCK_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - another process holds it past an install's worth of time
            return None, "another Studio process is installing cuDNN"
        try:
            # Another process may have finished while this one waited.
            found = _verified_library(dest, runtime)
            if found:
                return found, None
            reason = _install_locked(runtime, root, dest, status_cb)
            found = _verified_library(dest, runtime)
            if found:
                return found, None
            reason = reason or "the installed cuDNN failed verification"
            _FAILED[runtime.cuda_major] = reason
            return None, reason
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:  # noqa: BLE001
                    pass


def _install_locked(
    runtime: CudnnRuntime, root: Path, dest: Path, status_cb: StatusCb
) -> Optional[str]:
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        free = None
    if free is not None and free < _MIN_FREE_BYTES:
        return f"needs {_MIN_FREE_BYTES >> 30} GiB free beside the Studio home, {free / 2**30:.1f} GiB free"
    if not any(os.environ.get(v) for v in _INDEX_ENVS) and not _reachable(_PYPI_PROBE_URL):
        # A mirror configured in a pip / uv config file is not probed here; the installer reports its own failure.
        return "pypi.org is not reachable"
    uv = _uv_executable()
    staging = Path(tempfile.mkdtemp(prefix = ".staging-", dir = str(root)))
    try:
        before = _venv_snapshot()
        msg = f"Installing {', '.join(runtime.requirements())} for sd.cpp cuDNN attention"
        logger.info("sd_cpp.cudnn: %s into %s", msg, dest)
        if status_cb is not None:
            try:
                status_cb(msg)
            except Exception:  # noqa: BLE001
                pass
        ok, output = _run(install_command(runtime, staging, uv), _INSTALL_TIMEOUT_S)
        after = _venv_snapshot()
        if before != after:
            # Cannot happen with --target; if it ever does, say so loudly rather than run on a moved torch stack.
            drift = {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}
            logger.error("sd_cpp.cudnn: the install changed this environment: %s", drift)
            return f"the install changed this environment ({sorted(drift)})"
        if not ok:
            return f"install failed: {output[-400:]}"
        lib = staging / _LIB_RELPATH
        if not lib.is_file() or not (staging / _NVRTC_RELPATH).is_file():
            return "install finished without libcudnn.so.9 and libnvrtc.so.12"
        ok, output = _run(_verify_command(lib), _VERIFY_TIMEOUT_S)
        version = output.strip().splitlines()[-1] if ok and output.strip() else ""
        if not ok or not version.isdigit() or int(version) != runtime.cudnn_numeric:
            return f"verification failed: libcudnn.so.9 reports {version or output[-200:]!r}, expected {runtime.cudnn_numeric}"
        (staging / _MARKER_NAME).write_text(
            json.dumps(
                {"requirements": runtime.requirements(), "cudnn_version": int(version)}, indent = 1
            )
        )
        if dest.exists():
            shutil.rmtree(dest, ignore_errors = True)
        os.replace(staging, dest)
        logger.info("sd_cpp.cudnn: cuDNN %s ready at %s", version, dest)
        return None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors = True)


# ---------------------------------------------------------------------------------------------------------------
# Status: whether the sd.cpp child actually ran cuDNN, read off the fork's own INFO lines (printed once per process /
# per shape, no GGML_CUDA_CUDNN_ATTN_LOG needed).

_LOADED_RE = re.compile(r"cuDNN (\d+) loaded from \S+ for attention")
_PLAN_RE = re.compile(r"cuDNN SDPA plan b=\d+ hq=\d+ hk=\d+ sq=(\d+) skv=(\d+) d=(\d+)")
_NO_PLAN_RE = re.compile(r"no cuDNN SDPA plan for")
_EXEC_FAIL_RE = re.compile(r"cuDNN SDPA execute failed")
_TOO_OLD_RE = re.compile(r"reports cuDNN \d+, need 9\.0 or newer")

STATE_READY = "ready"  # library named to the child, no render seen yet
STATE_ENGAGED = "engaged"  # at least one attention shape runs on cuDNN
STATE_FALLBACK = "fallback"  # the child kept the ggml kernels
STATE_UNAVAILABLE = "unavailable"  # eligible, but no CUDA 12 cuDNN could be provided
STATE_OFF = "off"  # switched off by the user


@dataclass
class CudnnAttention:
    """One load's cuDNN attention state. ``env`` goes to every sd.cpp spawn of the load; ``feed`` reads its output."""

    state: Optional[str] = None
    reason: Optional[str] = None
    env: tuple[tuple[str, str], ...] = ()
    cudnn_version: Optional[int] = None
    shapes: set = field(default_factory = set)
    _render: dict = field(default_factory = dict)
    _lock: threading.Lock = field(default_factory = threading.Lock, repr = False)

    def begin_render(self) -> None:
        with self._lock:
            self._render = {"loaded": False, "plans": 0, "fallbacks": 0}

    def feed(self, line: str) -> None:
        if self.state is None or self.state in (STATE_UNAVAILABLE, STATE_OFF) or "cuDNN" not in line:
            return
        with self._lock:
            m = _LOADED_RE.search(line)
            if m:
                self._render["loaded"] = True
                self.cudnn_version = int(m.group(1))
                return
            m = _PLAN_RE.search(line)
            if m:
                self._render["plans"] = self._render.get("plans", 0) + 1
                self.shapes.add(tuple(int(g) for g in m.groups()))
                self.state, self.reason = STATE_ENGAGED, None
                return
            if _NO_PLAN_RE.search(line) or _EXEC_FAIL_RE.search(line) or _TOO_OLD_RE.search(line):
                self._render["fallbacks"] = self._render.get("fallbacks", 0) + 1

    def end_render(self, ok: bool = True) -> None:
        """Settle a render. A reused sd-server prints nothing again, so only a still-``ready`` state moves to
        fallback here."""
        with self._lock:
            seen = self._render
            self._render = {}
            if not ok or self.state != STATE_READY:
                return
            if seen.get("plans"):
                self.state, self.reason = STATE_ENGAGED, None
            elif seen.get("fallbacks"):
                self.state, self.reason = STATE_FALLBACK, "cuDNN built no attention plan; the ggml kernels ran"
            elif not seen.get("loaded"):
                self.state, self.reason = STATE_FALLBACK, "sd.cpp did not load the cuDNN library"

    def status_fields(self) -> dict[str, Any]:
        return {"sd_cpp_cudnn_attention": self.state, "sd_cpp_cudnn_reason": self.reason}


def plan_cudnn_attention(
    binary: Optional[str],
    cuda_cc: "Optional[tuple[int, int]]",
    *,
    allow_install: bool = True,
    status_cb: StatusCb = None,
    platform: Optional[str] = None,
    environ: Optional[dict] = None,
    root: Optional[Path] = None,
) -> CudnnAttention:
    """What one sd.cpp load should do about cuDNN attention. Inert (state None, no env) unless every condition holds:
    Linux, a binary built with the fork's cuDNN attention, a known NVIDIA compute capability of 8.0 or newer, and a
    tested runtime for the binary's CUDA major. Windows, macOS, AMD, Vulkan, CPU, Turing and older, and every
    prebuilt without the cuDNN build come back untouched."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if not platform.startswith("linux") or not binary or cuda_cc is None:
        return CudnnAttention()
    if tuple(cuda_cc) < CUDNN_MIN_CC:
        return CudnnAttention()
    carries, major = binary_cudnn_build(binary)
    if not carries:
        return CudnnAttention()
    if _off(environ.get(GGML_CUDNN_ATTN_ENV)):
        return CudnnAttention(STATE_OFF, f"{GGML_CUDNN_ATTN_ENV}=0")
    user_lib = str(environ.get(GGML_CUDNN_LIB_ENV) or "").strip()
    if user_lib:
        # Inherited by the child unchanged; the log decides whether it engaged.
        return CudnnAttention(STATE_READY, f"{GGML_CUDNN_LIB_ENV} set by the user")
    runtime = CUDNN_RUNTIMES.get(major) if major is not None else None
    if runtime is None:
        return CudnnAttention(
            STATE_UNAVAILABLE,
            f"no tested cuDNN runtime for a CUDA {major if major is not None else 'unknown'} build",
        )
    if not install_enabled(environ):
        return CudnnAttention(STATE_OFF, f"{CUDNN_INSTALL_ENV}=0")
    lib, reason = ensure_library(runtime, allow_install = allow_install, status_cb = status_cb, root = root)
    if lib is None:
        logger.info("sd_cpp.cudnn: attention stays on the ggml kernels: %s", reason)
        return CudnnAttention(STATE_UNAVAILABLE, reason)
    return CudnnAttention(STATE_READY, None, env = ((GGML_CUDNN_LIB_ENV, lib),))
