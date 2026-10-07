# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import http.client
import json
import logging
import os
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
# Pinned wheels; utils.kernel_install wraps them for the training worker, SSM runtime and CLI.
CAUSAL_CONV1D_PACKAGE_VERSION = "1.6.1"
CAUSAL_CONV1D_RELEASE_TAG = "v1.6.1.post4"
CAUSAL_CONV1D_RELEASE_BASE_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download"
MAMBA_SSM_PACKAGE_VERSION = "2.3.1"
MAMBA_SSM_RELEASE_TAG = "v2.3.1"
MAMBA_SSM_RELEASE_BASE_URL = "https://github.com/state-spaces/mamba/releases/download"


# No arch gate on purpose: it goes stale as upstream ships wheels; the import check catches bad ones.
def wheel_platform_tag() -> str | None:
    """pip platform tag for this host, or None where nothing we resolve is published. Windows is included because download.pytorch.org publishes CUDA-matched ``win_amd64`` xFormers wheels (see ``xformers_wheel_url``). It is NOT included for flash-attn / causal-conv1d / mamba-ssm, whose upstreams publish Linux assets only; ``probe_torch_wheel_env`` keeps that gate, not this function."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    elif sys.platform == "win32":
        if machine in {"x86_64", "amd64"}:
            return "win_amd64"
        # Windows on ARM: no CUDA and no win_arm64 wheel on any index.
    # No prebuilt wheels for macOS.
    return None


def probe_torch_wheel_env(
    *, timeout: int | None = None, include_windows: bool = False
) -> dict[str, str] | None:
    """Describe the resident torch build for wheel-URL resolution, or None. Windows is opt-in via ``include_windows``: every existing caller resolves a flash-attn / causal-conv1d / mamba-ssm asset, and those projects publish no win_amd64 wheels at all, so returning an env there would only build 404s."""
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
                    # xFormers ships one wheel per exact torch patch and CUDA minor, so record the full versions.
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


# torch 2.11/2.12 have no native wheels but the torch2.10 ones load and pass upstream suites.
# Add a key only after measuring real wheels: torch broke extension ABI between 2.9 and 2.10.
_PREBUILT_WHEEL_TORCH_MM = {"2.11": "2.10", "2.12": "2.10"}


def prebuilt_wheel_torch_mm(torch_mm: str) -> str:
    """Map a torch major.minor to the one whose prebuilt accelerator wheels to use."""
    return _PREBUILT_WHEEL_TORCH_MM.get(torch_mm, torch_mm)


# torch 2.13+ broke the extension ABI again, so we build these wheels ourselves
# (.github/workflows/prebuilt-cuda-wheels.yml) and resolve to our own release.
UNSLOTH_PREBUILT_RELEASE_BASE_URL = "https://github.com/unslothai/unsloth/releases/download"
UNSLOTH_PREBUILT_RELEASE_TAG = "prebuilt-wheels-cu13"

# Exact torch minors, not a floor: a row is added only after the workflow builds and smoke-tests it.
_UNSLOTH_PREBUILT_TORCH_MM = frozenset({"2.13", "2.14"})

# Newer than upstream versions: cut from revisions that compile against torch 2.13.
_UNSLOTH_PREBUILT_VERSIONS = {
    "flash_attn": "2.8.4",
    "causal_conv1d": "1.7.0",
    "mamba_ssm": "2.3.2.post1",
}


def unsloth_prebuilt_wheel_url(*, filename_prefix: str, env: dict[str, str] | None) -> str | None:
    """Our own prebuilt wheel for this environment, or None to leave resolution unchanged.

    Every gate here is narrower than it strictly has to be, because the cost of the two answers is not symmetric: returning None costs a source build the user was already facing, and returning a URL for a combination we did not publish costs a 404 and the same source build with a misleading log line in front of it. So it answers only for the exact cells the workflow builds, and the torch minor, CUDA major, ABI, interpreter and platform must all match. Linux x86_64 only: nothing else is built, and Windows, macOS and linux_aarch64 keep whatever behaviour they have today.
    """
    if env is None:
        return None
    if env.get("torch_mm") not in _UNSLOTH_PREBUILT_TORCH_MM:
        return None
    if env.get("cuda_major") != "13":
        return None
    if env.get("platform_tag") != "linux_x86_64":
        return None
    # torch pip wheels from 2.7 use _GLIBCXX_USE_CXX11_ABI=1; no abiFALSE variant is built.
    if env.get("cxx11abi") != "TRUE":
        return None
    python_tag = env.get("python_tag")
    if not python_tag:
        return None
    package_version = _UNSLOTH_PREBUILT_VERSIONS.get(filename_prefix)
    if package_version is None:
        return None

    filename = (
        f"{filename_prefix}-{package_version}"
        f"+cu{env['cuda_major']}torch{env['torch_mm']}"
        f"cxx11abi{env['cxx11abi']}-{python_tag}-{python_tag}"
        f"-{env['platform_tag']}.whl"
    )
    return f"{UNSLOTH_PREBUILT_RELEASE_BASE_URL}/{UNSLOTH_PREBUILT_RELEASE_TAG}/{filename}"


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

    # Checked before the upstream filename: for torch 2.13+ that asset never existed.
    ours = unsloth_prebuilt_wheel_url(filename_prefix = filename_prefix, env = env)
    if ours is not None:
        return ours

    filename = (
        f"{filename_prefix}-{package_version}"
        f"+cu{env['cuda_major']}torch{prebuilt_wheel_torch_mm(env['torch_mm'])}"
        f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}"
        f"-{env['platform_tag']}.whl"
    )
    return f"{release_base_url}/{release_tag}/{filename}"


# xformers/_C links one exact (torch, CUDA) pair and a mismatch silently drops attention kernels.
# Keyed on cpp_lib.json `torch`; keep in step with install.ps1 and test_windows_xformers_wheel_match.py.
PYTORCH_WHEEL_INDEX_BASE_URL = "https://download.pytorch.org/whl"


def pytorch_wheel_index_base_url() -> str:
    """Where torch-family wheels are fetched from: ``UNSLOTH_PYTORCH_MIRROR`` when set. Read per call rather than frozen at import: this module is imported early, and the mirror is the one setting an air-gapped deployment has. The whole installer stack already honours it (``install_python_stack._PYTORCH_WHL_BASE``, install.sh, setup.ps1), so a direct-URL install that hard-coded download.pytorch.org was the one path that could not reach a mirror-only host, failing the explicit xFormers request and dropping the user back to native attention."""
    return (os.environ.get("UNSLOTH_PYTORCH_MIRROR") or PYTORCH_WHEEL_INDEX_BASE_URL).rstrip("/")


_XFORMERS_WHEEL_VERSIONS: dict[str, dict[str, str]] = {
    # torch 2.7.0 omitted: pre-stable-ABI wheels stop at cp312, below the default Python 3.13.
    "2.7.1": {"cu126": "0.0.31.post1", "cu128": "0.0.31.post1"},
    "2.8.0": {"cu126": "0.0.32.post2", "cu128": "0.0.32.post2", "cu129": "0.0.32.post2"},
    "2.9.0": {"cu126": "0.0.33.post1", "cu128": "0.0.33.post1", "cu130": "0.0.33.post1"},
    "2.9.1": {"cu126": "0.0.33.post2", "cu128": "0.0.33.post2", "cu130": "0.0.33.post2"},
    "2.10.0": {"cu126": "0.0.34", "cu128": "0.0.34", "cu130": "0.0.34"},
    "2.11.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
    "2.12.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
    "2.13.0": {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"},
}

# Every torch strictly above this floor maps to the stable-ABI release; exact rows still win.
_XFORMERS_STABLE_ABI_FLOOR: tuple[int, ...] = (2, 10, 0)
_XFORMERS_STABLE_ABI_VERSIONS = {"cu126": "0.0.35", "cu128": "0.0.35", "cu130": "0.0.35"}

# Wheel filename interpreter tag changed per release range; unknown releases resolve to nothing.
_XFORMERS_FILENAME_PYTHON_TAGS: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...] = (
    ((0, 0, 31), (0, 0, 34), "cp39-abi3"),
    ((0, 0, 35), (0, 0, 35), "py39-none"),
)

# No xFormers wheels for aarch64 or macOS on download.pytorch.org.
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


def xformers_filename_python_tag(version: str) -> str | None:
    """The interpreter tag in an xFormers wheel filename, or None for an unknown release."""
    parsed = _xformers_version_tuple(version)
    if not parsed:
        return None
    for low, high, tag in _XFORMERS_FILENAME_PYTHON_TAGS:
        if low <= parsed <= high:
            return tag
    return None


def xformers_cuda_family(cuda_version: str | None) -> str | None:
    """torch.version.cuda -> the download.pytorch.org index leaf ('12.8' -> 'cu128'). None for a ROCm / CPU / XPU torch, which has no xFormers wheel anywhere."""
    if not cuda_version:
        return None
    parts = str(cuda_version).strip().split(".")
    try:
        major = int(re.sub(r"[^0-9].*", "", parts[0]))
        minor = int(re.sub(r"[^0-9].*", "", parts[1])) if len(parts) > 1 else 0
    except (IndexError, ValueError):
        return None
    return f"cu{major}{minor}"


def xformers_wheel_version(torch_version: str | None, cuda_family: str | None) -> str | None:
    """The xFormers release for this (torch, CUDA family), else None. An exact row wins; failing that, any release above the stable-ABI floor resolves to the wheel that serves that whole era, since the exact table cannot list patch releases published after this code ships and refusing them left supported builds (2.11.1, 2.12.1) with no xFormers at all."""
    if not torch_version or not cuda_family:
        return None
    release = str(torch_version).split("+", 1)[0].strip()
    exact = _XFORMERS_WHEEL_VERSIONS.get(release, {}).get(cuda_family)
    if exact is not None:
        return exact
    # dev/rc torch is not a release and must miss the table.
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", release):
        return None
    if _xformers_version_tuple(release) > _XFORMERS_STABLE_ABI_FLOOR:
        return _XFORMERS_STABLE_ABI_VERSIONS.get(cuda_family)
    return None


def xformers_wheel_url(env: dict[str, str] | None) -> str | None:
    """Direct URL of the xFormers wheel matching ``env``'s torch build, else None. None means "no matched wheel exists" and callers must install nothing rather than fall back to an unpinned resolve, since an unpinned install is what produces the mismatched extension in the first place."""
    if env is None:
        return None
    platform_leaf = _XFORMERS_PLATFORM_LEAVES.get(str(env.get("platform_tag") or ""))
    if platform_leaf is None:
        return None
    family = xformers_cuda_family(env.get("cuda_version"))
    version = xformers_wheel_version(env.get("torch_version"), family)
    if version is None:
        return None
    python_tag = xformers_filename_python_tag(version)
    if python_tag is None:
        return None
    return join_wheel_url(
        pytorch_wheel_index_base_url(),
        f"{family}/xformers-{version}-{python_tag}-{platform_leaf}.whl",
    )


def xformers_torch_requirement_unmet() -> tuple[str, str, str] | None:
    """(xformers version, unmet torch specifier, torch version) from metadata, or None if unmet cannot be shown."""
    try:
        from importlib.metadata import requires, version

        from packaging.requirements import Requirement
        from packaging.version import Version

        xformers_version = version("xformers")
        torch_version = version("torch")
        installed = Version(torch_version)
        declared = requires("xformers") or []
    except Exception:  # noqa: BLE001 -- either package absent, or no packaging
        return None
    for line in declared:
        try:
            requirement = Requirement(line)
            if requirement.name.lower() != "torch":
                continue
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
        except Exception:  # noqa: BLE001 -- a line packaging cannot parse is not a verdict
            continue
        # Local tags ignored as pip does: "torch==2.6.0" accepts 2.6.0+cu124.
        if not requirement.specifier.contains(installed, prereleases = True):
            return xformers_version, str(requirement.specifier), torch_version
    return None


def join_wheel_url(base: str, path: str) -> str:
    """``base`` + ``path``, with any ?query / #fragment kept at the end. UNSLOTH_PYTORCH_MIRROR is allowed to authenticate by query string (``https://mirror/whl?token=abc``), and appending after the query put the wheel path INSIDE the token value, leaving the request path at /whl and the token unusable. The tokenized private mirror this setting exists for was the one shape that could not resolve a wheel."""
    cut = min([i for i in (base.find("?"), base.find("#")) if i >= 0], default = -1)
    if cut < 0:
        return f"{base.rstrip('/')}/{path}"
    return f"{base[:cut].rstrip('/')}/{path}{base[cut:]}"


def redact_url_credentials(url: str) -> str:
    """A URL safe to log: no userinfo, no query, no fragment. UNSLOTH_PYTORCH_MIRROR is allowed to be a private index, and people put credentials in it (``https://user:token@mirror/whl`` or ``...?token=``). The wheel URL built from it is handed to pip AND printed, so without this the secret lands in the backend log the first time Unsloth installs (or fails to install) xFormers. Same rule as the installer's Remove-IndexUrlCredentials, so both sides redact identically."""
    separator = url.find("://")
    if separator < 0:
        return url
    scheme, rest = url[:separator], url[separator + 3 :]
    cut = min([i for i in (rest.find("?"), rest.find("#")) if i >= 0], default = -1)
    if cut >= 0:
        rest = rest[:cut]
    slash = rest.find("/")
    authority, path = (rest[:slash], rest[slash:]) if slash >= 0 else (rest, "")
    at = authority.rfind("@")
    if at >= 0:
        authority = authority[at + 1 :]
    return f"{scheme}://{authority}{path}"


def flash_attn_package_version(torch_mm: str) -> str | None:
    if torch_mm == "2.10":
        # Newest release with the full torch2.10 asset matrix; v2.8.3+ dropped most of it. Do not bump.
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
    # flash-attn never reaches direct_wheel_url on torch 2.13+, so the override is asked here too.
    ours = unsloth_prebuilt_wheel_url(filename_prefix = "flash_attn", env = env)
    if ours is not None:
        return ours
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
    reinstall: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, subprocess.CompletedProcess[str]]]:
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []

    if use_uv and shutil.which("uv"):
        uv_cmd = ["uv", "pip", "install"]
        if uv_needs_system:
            uv_cmd.append("--system")
        uv_cmd.extend(["--python", python_executable, "--no-deps"])
        # Without it a same-version build (other CUDA, or broken) is kept.
        if reinstall:
            uv_cmd.append("--reinstall")
        uv_cmd.append(wheel_url)
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

    pip_cmd = [python_executable, "-m", "pip", "install", "--no-deps"]
    if reinstall:
        pip_cmd.append("--force-reinstall")
    pip_cmd.append(wheel_url)
    result = run(
        pip_cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = utf8_child_env(child_env_without_native_path_secret()),
    )
    attempts.append(("pip", result))
    return attempts


def url_exists(url: str) -> bool | None:
    """True if reachable, False on a 404, None when it cannot be checked: a refusal is no proof the wheel is unpublished."""
    try:
        request = urllib.request.Request(url, method = "HEAD")
        with urllib.request.urlopen(request, timeout = 10):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        reason = f"HTTP {exc.code}"
    except (OSError, http.client.HTTPException) as exc:
        reason = str(exc)
    shown = redact_url_credentials(url)
    if shown != url:
        # The error text can echo userinfo (urllib reads `user:token@host` as a port).
        reason = reason if reason.startswith("HTTP ") else "unreachable"
    _logger.warning(
        "url_exists(%s): %s; could not check prebuilt wheel availability", shown, reason
    )
    return None
