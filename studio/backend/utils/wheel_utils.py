# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import functools
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable

from utils.native_path_leases import child_env_without_native_path_secret
from utils.child_stdio import utf8_child_env
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_logger = logging.getLogger(__name__)

FLASH_ATTN_RELEASE_BASE_URL = "https://github.com/Dao-AILab/flash-attention/releases/download"


@functools.lru_cache(maxsize = 1)
def has_blackwell_gpu() -> bool:
    """Return True if any visible NVIDIA GPU has compute capability >= 10.0 (Blackwell).

    Cached for the process lifetime; tests mocking nvidia-smi must call
    ``has_blackwell_gpu.cache_clear()`` first.
    """
    # Detection disabled for now: Dao-AILab ships Blackwell (sm_100+) flash-attn
    # wheels and url_exists() already gates resolution, so we no longer skip
    # flash-attn on Blackwell. The nvidia-smi probe below is kept for possible
    # future arch-based gating; drop this early return to re-enable it.
    return False
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            env = child_env_without_native_path_secret(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        cap = line.strip()
        if not cap:
            continue
        major_part = cap.split(".", 1)[0]
        try:
            major = int(major_part)
        except ValueError:
            continue
        if major >= 10:
            return True
    return False


def wheel_platform_tag() -> str | None:
    """pip platform tag for this host, or None where nothing we resolve is published.

    Windows is included because download.pytorch.org publishes CUDA-matched
    ``win_amd64`` xFormers wheels (see ``xformers_wheel_url``). It is NOT included for
    flash-attn / causal-conv1d / mamba-ssm, whose upstreams publish Linux assets only --
    ``probe_torch_wheel_env`` keeps that gate, not this function.
    """
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    elif sys.platform == "win32":
        if machine in {"x86_64", "amd64"}:
            return "win_amd64"
        # Windows on ARM: no CUDA, and no win_arm64 wheel on any index.
    # No prebuilt wheels published for macOS
    return None


def probe_torch_wheel_env(
    *, timeout: int | None = None, include_windows: bool = False
) -> dict[str, str] | None:
    """Describe the resident torch build for wheel-URL resolution, or None.

    Windows is opt-in via ``include_windows``: every existing caller resolves a
    flash-attn / causal-conv1d / mamba-ssm asset, and those projects publish no
    win_amd64 wheels at all, so returning an env there would only build 404s.
    """
    platform_tag = wheel_platform_tag()
    if platform_tag is None:
        return None
    if platform_tag == "win_amd64" and not include_windows:
        return None

    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys, re, torch; "
                    "parts = torch.__version__.split('+', 1)[0].split('.')[:2]; "
                    "minor = re.sub(r'[^0-9].*', '', parts[1]) if len(parts) > 1 else '0'; "
                    "torch_mm = parts[0] + '.' + minor; "
                    "print(json.dumps({"
                    "'python_tag': f'cp{sys.version_info.major}{sys.version_info.minor}', "
                    "'torch_mm': torch_mm, "
                    # Full release + full CUDA version: xFormers publishes one wheel per
                    # exact torch PATCH and per CUDA MINOR (cu126 and cu128 are different
                    # builds of the same version string), so 'torch_mm' / 'cuda_major'
                    # cannot pick between them.
                    "'torch_version': str(torch.__version__), "
                    "'cuda_version': str(torch.version.cuda) if torch.version.cuda else '', "
                    "'cuda_major': str(int(str(torch.version.cuda).split('.', 1)[0])) if torch.version.cuda else '', "
                    "'hip_version': str(torch.version.hip) if getattr(torch.version, 'hip', None) else '', "
                    "'cxx11abi': str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()"
                    "}))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            env = utf8_child_env(child_env_without_native_path_secret()),
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return None

    if probe.returncode != 0:
        return None

    try:
        env = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        return None
    env["platform_tag"] = platform_tag
    return env


# torch 2.11 and 2.12 ship no native prebuilt wheels for flash-attn /
# causal-conv1d / mamba-ssm, but the torch2.10 CUDA wheels load and pass each
# project's own suite on both (B200, py3.12, torch 2.12.1+cu130: causal-conv1d
# 9412 passed / 3888 skipped / 0 failed, mamba tests/ops 20 passed, flash-attn
# splitkv+qkvpacked 848 passed; pass/fail/skip counts and the failing test-ID
# sets are identical to a torch 2.10 control). Reuse them so a 2.11 / 2.12
# install still gets prebuilt accelerators instead of building from source.
#
# The window is bounded, not open ended: torch broke extension ABI between 2.9
# and 2.10, and the torch2.9 flash-attn .so raises "undefined symbol" on torch
# 2.10 and on 2.12 alike. A wheel cannot skip a torch minor backwards, so every
# new key here must be measured against the real wheels before it is added.
_PREBUILT_WHEEL_TORCH_MM = {"2.11": "2.10", "2.12": "2.10"}


def prebuilt_wheel_torch_mm(torch_mm: str) -> str:
    """Map a torch major.minor to the one whose prebuilt accelerator wheels to use."""
    return _PREBUILT_WHEEL_TORCH_MM.get(torch_mm, torch_mm)


def direct_wheel_url(
    *,
    filename_prefix: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    env: dict[str, str] | None,
) -> str | None:
    if env is None or not env.get("cuda_major"):
        return None

    filename = (
        f"{filename_prefix}-{package_version}"
        f"+cu{env['cuda_major']}torch{prebuilt_wheel_torch_mm(env['torch_mm'])}"
        f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}"
        f"-{env['platform_tag']}.whl"
    )
    return f"{release_base_url}/{release_tag}/{filename}"


# ── xFormers ──────────────────────────────────────────────────────────────────
# xformers/_C.pyd (_C.so on Linux) is linked against ONE exact (torch, CUDA) pair.
# Loaded beside any other pair torch.ops.load_library raises, and xformers/_cpp_lib.py
# turns that into a log warning rather than an error -- so the import "succeeds" and
# memory-efficient attention, SwiGLU and the sparse ops are silently gone. PyPI
# publishes only the CUDA-12.8 flavour, which is why `pip install xformers` next to a
# cu130 torch reports "xFormers was built for PyTorch 2.10.0+cu128 with CUDA 1208 (you
# have 2.10.0+cu130)".
#
# download.pytorch.org publishes one wheel per (CUDA family, torch patch), so resolve
# the exact URL instead. torch release -> {CUDA family: xFormers version}. Every row was
# HEAD-verified live and its xformers/cpp_lib.json read back, e.g.
# cu130/xformers-0.0.34-cp39-abi3-win_amd64.whl reports {"torch": "2.10.0+cu130"}.
#
# Rows are exact, never interpolated: xFormers' extension ABI does not survive a torch
# minor bump (unlike the flash-attn window above, which was measured), no cu130 build
# exists for torch <= 2.8, and no cu118 / cu121 / cu124 win_amd64 build exists at all.
# An unlisted pair means "install nothing", which is the safe answer.
#
# Keep in step with $script:XformersWheelVersions in install.ps1 and the matrix in
# tests/python/test_windows_xformers_wheel_match.py.
PYTORCH_WHEEL_INDEX_BASE_URL = "https://download.pytorch.org/whl"

_XFORMERS_WHEEL_VERSIONS: dict[str, dict[str, str]] = {
    "2.7.0": {"cu126": "0.0.30", "cu128": "0.0.30"},
    "2.7.1": {"cu126": "0.0.31.post1", "cu128": "0.0.31.post1"},
    "2.8.0": {"cu126": "0.0.32.post2", "cu128": "0.0.32.post2", "cu129": "0.0.32.post2"},
    "2.9.0": {"cu126": "0.0.33.post1", "cu128": "0.0.33.post1", "cu130": "0.0.33.post1"},
    "2.9.1": {"cu126": "0.0.33.post2", "cu128": "0.0.33.post2", "cu130": "0.0.33.post2"},
    "2.10.0": {"cu126": "0.0.34", "cu128": "0.0.34", "cu130": "0.0.34"},
}

# xFormers switched to a single stable-ABI wheel at 0.0.31; 0.0.30 and earlier publish
# one wheel per interpreter (cp310-cp310, ... up to cp312 only).
_XFORMERS_ABI3_MIN_VERSION = (0, 0, 31)

# platform_tag from wheel_platform_tag() -> the leaf in the wheel filename. aarch64 and
# macOS are absent because download.pytorch.org publishes no xFormers wheel for them.
_XFORMERS_PLATFORM_LEAVES = {
    "linux_x86_64": "manylinux_2_28_x86_64",
    "win_amd64": "win_amd64",
}


def _xformers_version_tuple(version: str) -> tuple[int, ...]:
    """'0.0.33.post1' -> (0, 0, 33). Stops at the first non-numeric component."""
    parts: list[int] = []
    for chunk in str(version).split("."):
        digits = re.sub(r"[^0-9].*", "", chunk)
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def xformers_cuda_family(cuda_version: str | None) -> str | None:
    """torch.version.cuda -> the download.pytorch.org index leaf ('12.8' -> 'cu128').

    None for a ROCm / CPU / XPU torch, which has no xFormers wheel anywhere.
    """
    if not cuda_version:
        return None
    parts = str(cuda_version).split(".")
    try:
        major = int(re.sub(r"[^0-9].*", "", parts[0]))
        minor = int(re.sub(r"[^0-9].*", "", parts[1])) if len(parts) > 1 else 0
    except (IndexError, ValueError):
        return None
    return f"cu{major}{minor}"


def xformers_wheel_version(torch_version: str | None, cuda_family: str | None) -> str | None:
    """The xFormers release built for exactly this (torch, CUDA family), else None."""
    if not torch_version or not cuda_family:
        return None
    # '2.10.0+cu130' -> '2.10.0'. A dev/rc torch has no wheel and must miss the table.
    release = str(torch_version).split("+", 1)[0].strip()
    return _XFORMERS_WHEEL_VERSIONS.get(release, {}).get(cuda_family)


def xformers_wheel_url(env: dict[str, str] | None) -> str | None:
    """Direct URL of the xFormers wheel matching ``env``'s torch build, else None.

    None means "no matched wheel exists" and callers must install nothing rather than
    fall back to an unpinned resolve -- an unpinned install is what produces the
    mismatched extension in the first place.
    """
    if env is None:
        return None
    platform_leaf = _XFORMERS_PLATFORM_LEAVES.get(str(env.get("platform_tag") or ""))
    if platform_leaf is None:
        return None
    family = xformers_cuda_family(env.get("cuda_version"))
    version = xformers_wheel_version(env.get("torch_version"), family)
    if version is None:
        return None
    if _xformers_version_tuple(version) >= _XFORMERS_ABI3_MIN_VERSION:
        python_tag = "cp39-abi3"
    else:
        interpreter = env.get("python_tag")
        if not interpreter:
            return None
        python_tag = f"{interpreter}-{interpreter}"
    return (
        f"{PYTORCH_WHEEL_INDEX_BASE_URL}/{family}"
        f"/xformers-{version}-{python_tag}-{platform_leaf}.whl"
    )


def flash_attn_package_version(torch_mm: str) -> str | None:
    if torch_mm == "2.10":
        # Newest flash-attn release still carrying the full torch2.10 asset
        # matrix (cu12 + cu13, cp312 + cp313, x86_64 + aarch64). Do not bump
        # this to "the latest release": v2.8.3 publishes only cu13/cp312 for
        # torch2.10 and v2.8.3.post1 dropped every torch2.10 asset, so both
        # 404 most users back to a source build, and post1's newest tag is
        # torch2.9, which will not load here at all.
        return "2.8.1"
    try:
        major, minor = (int(part) for part in torch_mm.split(".", 1))
    except ValueError:
        return None
    if major == 2 and 4 <= minor <= 9:
        return "2.8.3"
    return None


def flash_attn_wheel_url(env: dict[str, str] | None) -> str | None:
    if env is None:
        return None
    package_version = flash_attn_package_version(prebuilt_wheel_torch_mm(env["torch_mm"]))
    if package_version is None:
        return None
    return direct_wheel_url(
        filename_prefix = "flash_attn",
        package_version = package_version,
        release_tag = f"v{package_version}",
        release_base_url = FLASH_ATTN_RELEASE_BASE_URL,
        env = env,
    )


def install_wheel(
    wheel_url: str,
    *,
    python_executable: str,
    use_uv: bool,
    uv_needs_system: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, subprocess.CompletedProcess[str]]]:
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []

    # Try uv first if available, then fall back to pip
    if use_uv and shutil.which("uv"):
        uv_cmd = ["uv", "pip", "install"]
        if uv_needs_system:
            uv_cmd.append("--system")
        uv_cmd.extend(["--python", python_executable, "--no-deps", wheel_url])
        result = run(
            uv_cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = child_env_without_native_path_secret(),
        )
        attempts.append(("uv", result))
        if result.returncode == 0:
            return attempts

    pip_cmd = [python_executable, "-m", "pip", "install", "--no-deps", wheel_url]
    result = run(
        pip_cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        # Make the Python child emit the UTF-8 we decode above.
        env = utf8_child_env(child_env_without_native_path_secret()),
    )
    attempts.append(("pip", result))
    return attempts


def url_exists(url: str) -> bool:
    try:
        request = urllib.request.Request(url, method = "HEAD")
        with urllib.request.urlopen(request, timeout = 10):
            return True
    except urllib.error.HTTPError as exc:
        _logger.debug("url_exists(%s): HTTP %s", url, exc.code)
    except (urllib.error.URLError, TimeoutError) as exc:
        _logger.debug("url_exists(%s): %s", url, exc)
    return False
