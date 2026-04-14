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
from dataclasses import dataclass
from typing import Callable

from utils.native_path_leases import child_env_without_native_path_secret

_logger = logging.getLogger(__name__)

FLASH_ATTN_RELEASE_BASE_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download"
)
CAUSAL_CONV1D_RELEASE_BASE_URL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download"
)
MAMBA_SSM_RELEASE_BASE_URL = "https://github.com/state-spaces/mamba/releases/download"

@dataclass(frozen = True)
class KernelPackageSpec:
    import_name: str
    display_name: str
    pypi_spec: str
    wheel_url_builder: Callable[[dict[str, str] | None], str | None] | None = None
    filename_prefix: str | None = None
    package_version: str | None = None
    release_tag: str | None = None
    release_base_url: str | None = None
    pypi_status_message: str | None = None

    def build_wheel_url(self, env: dict[str, str] | None) -> str | None:
        if self.wheel_url_builder is not None:
            return self.wheel_url_builder(env)
        if (
            self.filename_prefix is None
            or self.package_version is None
            or self.release_tag is None
            or self.release_base_url is None
        ):
            return None
        return direct_wheel_url(
            filename_prefix = self.filename_prefix,
            package_version = self.package_version,
            release_tag = self.release_tag,
            release_base_url = self.release_base_url,
            env = env,
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


CAUSAL_CONV1D_SPEC = KernelPackageSpec(
    import_name = "causal_conv1d",
    display_name = "causal-conv1d",
    pypi_spec = "causal-conv1d==1.6.1",
    filename_prefix = "causal_conv1d",
    package_version = "1.6.1",
    release_tag = "v1.6.1.post4",
    release_base_url = CAUSAL_CONV1D_RELEASE_BASE_URL,
)

MAMBA_SSM_SPEC = KernelPackageSpec(
    import_name = "mamba_ssm",
    display_name = "mamba-ssm",
    pypi_spec = "mamba-ssm==2.3.1",
    filename_prefix = "mamba_ssm",
    package_version = "2.3.1",
    release_tag = "v2.3.1",
    release_base_url = MAMBA_SSM_RELEASE_BASE_URL,
)

FLASH_ATTN_SPEC = KernelPackageSpec(
    import_name = "flash_attn",
    display_name = "flash-attn",
    pypi_spec = "flash-attn",
    wheel_url_builder = flash_attn_wheel_url,
    pypi_status_message = "Installing flash-attn from PyPI for long-context training...",
)

FLASH_LINEAR_ATTN_SPEC = KernelPackageSpec(
    import_name = "fla",
    display_name = "flash-linear-attention",
    pypi_spec = "flash-linear-attention==0.5.0",
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


def _package_is_importable(import_name: str) -> bool:
    try:
        __import__(import_name)
    except ImportError:
        return False
    return True


def _default_pypi_status_message(
    spec: KernelPackageSpec,
    *,
    is_hip: bool,
) -> str:
    if spec.pypi_status_message is not None:
        return spec.pypi_status_message
    if is_hip:
        return (
            f"Compiling {spec.display_name} from source for ROCm "
            "(this may take several minutes)..."
        )
    return f"Installing {spec.display_name} from PyPI..."


def _pypi_install_command(
    spec: KernelPackageSpec,
    *,
    python_executable: str,
    use_uv: bool,
    uv_needs_system: bool,
    is_hip: bool,
) -> list[str]:
    has_uv = use_uv and shutil.which("uv")
    plain_pypi_install = spec.package_version is None

    if plain_pypi_install:
        if has_uv:
            cmd = ["uv", "pip", "install"]
            if uv_needs_system:
                cmd.append("--system")
            cmd.extend(["--python", python_executable, spec.pypi_spec])
            return cmd
        return [python_executable, "-m", "pip", "install", spec.pypi_spec]

    if has_uv:
        cmd = ["uv", "pip", "install"]
        if uv_needs_system:
            cmd.append("--system")
        cmd.extend(
            [
                "--python",
                python_executable,
                "--no-build-isolation",
                "--no-deps",
            ]
        )
        if is_hip:
            cmd.append("--no-cache")
        cmd.append(spec.pypi_spec)
        return cmd

    cmd = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--no-cache-dir",
        spec.pypi_spec,
    ]
    return cmd


def install_optional_kernel(
    spec: KernelPackageSpec,
    *,
    python_executable: str,
    use_uv: bool,
    uv_needs_system: bool = False,
    allow_pypi_fallback: bool,
    status: Callable[[str], None] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    if _package_is_importable(spec.import_name):
        _logger.info("%s already installed", spec.display_name)
        return True

    env = probe_torch_wheel_env(timeout = 30)
    wheel_url = spec.build_wheel_url(env)

    if wheel_url is None:
        _logger.info("No compatible %s wheel candidate", spec.display_name)
        if status is not None and not allow_pypi_fallback:
            status(f"No compatible {spec.display_name} prebuilt wheel found")
    elif url_exists(wheel_url):
        if status is not None:
            status(f"Installing prebuilt {spec.display_name} wheel...")
        for installer, result in install_wheel(
            wheel_url,
            python_executable = python_executable,
            use_uv = use_uv,
            uv_needs_system = uv_needs_system,
            run = run,
        ):
            if result.returncode == 0:
                _logger.info(
                    "Installed prebuilt %s wheel successfully",
                    spec.display_name,
                )
                return True
            _logger.warning(
                "%s failed to install %s wheel:\n%s",
                installer,
                spec.display_name,
                result.stdout,
            )
            if status is not None and not allow_pypi_fallback:
                status(
                    f"Installing {spec.display_name} prebuilt wheel with "
                    f"{installer} failed (exit code {result.returncode})"
                )
    else:
        _logger.info("No published %s wheel found: %s", spec.display_name, wheel_url)
        if status is not None and not allow_pypi_fallback:
            status(f"No published {spec.display_name} prebuilt wheel found")

    if not allow_pypi_fallback:
        return False

    is_hip = bool(env and env.get("hip_version"))
    if is_hip and not shutil.which("hipcc"):
        _logger.error(
            "%s requires hipcc for source compilation on ROCm. "
            "Install the ROCm HIP SDK: https://rocm.docs.amd.com",
            spec.display_name,
        )
        if status is not None:
            status(f"{spec.display_name}: hipcc not found (ROCm HIP SDK required)")
        return False

    if status is not None:
        status(_default_pypi_status_message(spec, is_hip = is_hip))

    pypi_cmd = _pypi_install_command(
        spec,
        python_executable = python_executable,
        use_uv = use_uv,
        uv_needs_system = uv_needs_system,
        is_hip = is_hip,
    )

    run_kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": child_env_without_native_path_secret(),
    }
    if is_hip:
        run_kwargs["timeout"] = 1800

    try:
        result = run(pypi_cmd, **run_kwargs)
    except subprocess.TimeoutExpired:
        _logger.error(
            "%s installation timed out after %ds",
            spec.display_name,
            run_kwargs.get("timeout"),
        )
        if status is not None:
            status(
                f"{spec.display_name} installation timed out after "
                f"{run_kwargs.get('timeout')}s"
            )
        return False

    if result.returncode != 0:
        if is_hip:
            error_lines = (result.stdout or "").strip().splitlines()
            snippet = "\n".join(error_lines[-5:]) if error_lines else "(no output)"
            _logger.error(
                "Failed to compile %s for ROCm:\n%s",
                spec.display_name,
                result.stdout,
            )
            if status is not None:
                status(
                    f"Failed to compile {spec.display_name} for ROCm. "
                    "Check that hipcc and ROCm development headers are installed.\n"
                    f"{snippet}"
                )
        else:
            _logger.error(
                "Failed to install %s from PyPI:\n%s",
                spec.display_name,
                result.stdout,
            )
        return False

    if is_hip:
        _logger.info("Compiled and installed %s from source for ROCm", spec.display_name)
    else:
        _logger.info("Installed %s from PyPI", spec.display_name)
    return True


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
