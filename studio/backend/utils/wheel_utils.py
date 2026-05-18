# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import functools
import json
import logging
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable

from utils.native_path_leases import child_env_without_native_path_secret

_logger = logging.getLogger(__name__)

FLASH_ATTN_RELEASE_BASE_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download"
)


@functools.lru_cache(maxsize = 1)
def has_blackwell_gpu() -> bool:
    """Return True if any visible NVIDIA GPU has compute capability >= 10.0
    (Blackwell: sm_100, sm_120, sm_121, ...).

    Dao-AILab does not publish prebuilt flash-attention wheels for these
    architectures, and the older-arch wheels fail to load on Blackwell, so
    callers use this gate to skip the flash-attn install/upgrade path.

    Result is cached for the process lifetime since GPU hardware does not
    change. Tests that mock subprocess/nvidia-smi must call
    ``has_blackwell_gpu.cache_clear()`` before each invocation.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
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


def linux_wheel_platform_tag() -> str | None:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    # No prebuilt wheels published for macOS or Windows
    return None


def probe_torch_wheel_env(*, timeout: int | None = None) -> dict[str, str] | None:
    platform_tag = linux_wheel_platform_tag()
    if platform_tag is None:
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
                    "'cuda_major': str(int(str(torch.version.cuda).split('.', 1)[0])) if torch.version.cuda else '', "
                    "'hip_version': str(torch.version.hip) if getattr(torch.version, 'hip', None) else '', "
                    "'cxx11abi': str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()"
                    "}))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            timeout = timeout,
            env = child_env_without_native_path_secret(),
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
        f"+cu{env['cuda_major']}torch{env['torch_mm']}"
        f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}"
        f"-{env['platform_tag']}.whl"
    )
    return f"{release_base_url}/{release_tag}/{filename}"


def flash_attn_package_version(torch_mm: str) -> str | None:
    if torch_mm == "2.10":
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
    package_version = flash_attn_package_version(env["torch_mm"])
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
        env = child_env_without_native_path_secret(),
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
