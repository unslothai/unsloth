#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by setup.sh (Linux/WSL) and setup.ps1 (Windows) after the venv is
activated. Expects `pip` and `python` on PATH to point at the venv.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import urllib.request
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(1, str(_BACKEND_DIR))

from backend.utils.wheel_utils import (
    flash_attn_package_version,
    flash_attn_wheel_url,
    has_blackwell_gpu,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)
from backend.utils.uv_path_safety import uv_safe_path as _uv_safe_path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_MAC_INTEL = IS_MACOS and platform.machine() == "x86_64"
IS_MAC_ARM = IS_MACOS and platform.machine() == "arm64"
IS_LINUX = sys.platform.startswith("linux")

# amd-smi auto-elevates on Windows (UAC/DiskPart prompt mid-install). This installer
# only spawns probes and pip/uv (no elevation), so set __COMPAT_LAYER=RunAsInvoker
# process-wide; amd-smi then runs un-elevated. setup.ps1 keeps per-call guards (it
# also spawns winget installers that need elevation).
if IS_WINDOWS:
    os.environ.setdefault("__COMPAT_LAYER", "RunAsInvoker")
# torchcodec ships wheels only for manylinux_2_28_x86_64, macosx_12_0_arm64,
# and win_amd64. On other hosts the audio extras must be filtered out (the
# extras-no-deps step would otherwise fail), regardless of NO_TORCH.
PLATFORM_LACKS_TORCHCODEC_WHEEL = (
    (IS_LINUX and platform.machine() in {"aarch64", "arm64"})
    or (IS_WINDOWS and platform.machine().lower() in {"arm64", "aarch64"})
    or IS_MAC_INTEL
)

# ── ROCm / AMD GPU support ─────────────────────────────────────────────────────
# Detected ROCm (major, minor) -> best PyTorch wheel tag on
# download.pytorch.org. Checked newest-first (>=).
_ROCM_TORCH_INDEX: dict[tuple[int, int], str] = {
    (7, 2): "rocm7.2",  # torch 2.11.0
    (7, 1): "rocm7.1",  # torch 2.10.0
    (7, 0): "rocm7.0",
    (6, 4): "rocm6.4",
    (6, 3): "rocm6.3",
    (6, 2): "rocm6.2",
    (6, 1): "rocm6.1",
    (6, 0): "rocm6.0",
}


def _generic_pytorch_rocm_tag(ver: tuple[int, int]) -> str | None:
    """Newest download.pytorch.org rocmX.Y tag for a host ROCm version."""
    return next(
        (t for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True) if ver >= (maj, mn)),
        None,
    )


_ROCM_ARCH_INDEX_FLOOR = (7, 13)  # AMD per-arch index ships torch 2.11+rocm7.13


def _strix_needs_amd_arch_index(ver: tuple[int, int]) -> bool:
    """True when Strix's generic pytorch.org index sits below the AMD arch floor
    (7.13), so gfx1150/1151 must use repo.amd.com's per-arch wheels. Mirrors
    install.sh _rocm_leaf_below: reroute any generic rocm index (6.x/7.0/7.2 and a
    future 7.3+), never one at/above the floor."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is not None and key < _ROCM_ARCH_INDEX_FLOOR


# AMD per-arch leaves needing the torch 2.11 floor (the _grouped_mm <2.11 bug).
# Mirrors *FloorMap in install.ps1 / setup.ps1; other arches ship <2.11 and stay bare.
_ROCM_GFX_TORCH211_LEAVES: frozenset[str] = frozenset({"gfx120x-all", "gfx1151", "gfx1150"})

# pytorch.org rocmX.Y indexes KNOWN to ship torch 2.11 (rocm7.2 only today); don't
# floor an unknown newer rocm speculatively. Match install.sh / setup.ps1 / install.ps1.
_ROCM_KNOWN_TORCH211_VERSIONS: frozenset[tuple[int, int]] = frozenset({(7, 2)})

# Per-tag pip specs; rocm7.2 ships torch 2.11.0 (older tags cap at 2.10.x).
_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
    "rocm7.2": (
        "torch>=2.11.0,<2.12.0",
        "torchvision>=0.26.0,<0.27.0",
        "torchaudio>=2.11.0,<2.12.0",
    ),
    # rocm7.1 and earlier: torch 2.x below 2.11
    "_default": (
        "torch>=2.4,<2.11.0",
        "torchvision>=0.19,<0.26.0",
        "torchaudio>=2.4,<2.11.0",
    ),
}
# Windows AMD per-arch companion pins for the repo.amd.com index (mirrors the install.ps1 /
# setup.ps1 floor maps): pinning stops the per-arch index (each published independently) from
# resolving an ABI-mismatched companion. Unlisted arches have no floor, so stay bare.
_WINDOWS_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
    "gfx1201": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1200": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1151": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1150": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
}
_PYTORCH_WHL_BASE = (
    os.environ.get("UNSLOTH_PYTORCH_MIRROR") or "https://download.pytorch.org/whl"
).rstrip("/")


def _strip_index_url_credentials(url: str) -> str:
    """Strip userinfo (user:password@) AND query/fragment from a wheel index URL.

    An authenticated pin must not leak credentials in printed output; query/fragment
    may hold tokens and aren't part of the PEP 503 index identity. Host/path stay
    exact. MUST match install.sh / setup.ps1 / install.ps1.
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    rest = rest.split("?", 1)[0].split("#", 1)[0]  # drop query / fragment
    authority, slash, tail = rest.partition("/")
    host = authority.rpartition("@")[2]  # drop user:pass@ userinfo
    return f"{scheme}://{host}{slash}{tail}"


_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s`]+@")
_URL_QUERY_VALUE_RE = re.compile(r"([?&][^=\s&`]+)=[^&#\s`]+")
# URL-anchored so a bare "#..." (a shell comment in tool output) is never touched.
_URL_FRAGMENT_RE = re.compile(r"(https?://[^\s`#]+)#[^\s`]+")


def _redact_install_output(output: "bytes | str") -> str:
    """Redact index-URL credentials (userinfo + query values + fragments) from captured
    installer output before printing. uv/pip failure text embeds the failing --index-url
    verbatim, which can carry a user:token@, ?token= or #token= secret. MUST match
    install.sh / setup.ps1 / install.ps1's output sanitizers."""
    text = output.decode(errors = "replace") if isinstance(output, bytes) else output
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _URL_QUERY_VALUE_RE.sub(r"\1=<redacted>", text)
    return _URL_FRAGMENT_RE.sub(r"\1#<redacted>", text)


def _trim_index_path_slashes(url: str) -> str:
    """Trim trailing slashes from the URL PATH only, preserving ?query / #fragment. A
    whole-URL rstrip("/") corrupts a token that ends in "/" (e.g. base64 ...abc/) and a
    single-slash strip leaves .../cu128// classifying as an empty leaf. MUST match
    install.sh / setup.ps1 / install.ps1."""
    value = url.strip()
    match = re.fullmatch(r"([^?#]*)([?#].*)?", value)
    if match is None:
        return value.rstrip("/")
    return match.group(1).rstrip("/") + (match.group(2) or "")


def _torch_index_leaf(url: str) -> str:
    """Final URL path segment, lowercased, query/fragment removed first.

    So a token-authenticated pin (.../cu128?token=x) classifies as cu128 (a raw leaf
    keeps the query, never equals the +cu128 tag, and force-reinstalls every update).
    CLASSIFICATION only; the install keeps the full URL. MUST match install.sh /
    setup.ps1 / install.ps1.
    """
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.rstrip("/").rsplit("/", 1)[-1].lower()


# CUDA torch repair specs (see _ensure_cuda_torch). torch 2.11 is allowed (torchao
# 0.17 cpp loads cleanly, and the flash-attn/causal-conv1d/mamba wheels pass on 2.11).
# torchvision/torchaudio are pinned (not bare) so the exclusive --index-url can't
# resolve one built against a different torch major -> ABI mismatch.
_CUDA_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

# CPU torch repair specs (see _ensure_cpu_torch). Same bounds/reasoning as CUDA: the
# /cpu index also serves newer torch, so a bare trio could resolve out of range or ABI-
# mismatched.
_CPU_TORCH_PKG_SPEC: tuple[str, str, str] = _CUDA_TORCH_PKG_SPEC

# torchao's cpp extensions are pinned to ONE torch release AND CUDA major. A torch
# mismatch just skips the cpp kernels (slow Python fallback); a CUDA mismatch fails
# to import ("libcudart.so.12: cannot open shared object file"). The torch pin is a
# range, so match torchao to the installed torch (table: pytorch/ao#2919):
#   2.9.x            -> 0.14.0
#   2.10.x, CUDA<=12 -> 0.16.0 (cpp built for 2.10, loads via the CUDA-12 wheel)
#   2.10.x, CUDA>=13 -> 0.17.0 (cu130: 0.16.0's CUDA-12 cpp crashes on load; 0.17.0
#                       targets torch 2.11 so its cpp is cleanly skipped, not crashed)
#   2.11.x           -> 0.17.0 (reachable via CUDA or ROCm rocm7.2)
# Unknown/older torch keeps the conservative default.
_TORCHAO_DEFAULT_SPEC = "torchao==0.14.0"
_TORCHAO_TORCH_210_SPEC = "torchao==0.16.0"
_TORCHAO_TORCH_210_CUDA13_SPEC = "torchao==0.17.0"
_TORCHAO_TORCH_211_PLUS_SPEC = "torchao==0.17.0"
# torch 2.10 built against CUDA >= this major can't load 0.16.0's CUDA-12 cpp.
_TORCHAO_CUDA13_MIN_MAJOR = 13


def _cuda_major_from_torch_version(torch_version: str) -> int | None:
    """Extract the CUDA major from a torch local version tag, e.g. '2.10.0+cu130'
    -> 13, '2.10.0+cu128' -> 12. Returns None for rocm/cpu/tagless builds."""
    local = str(torch_version).split("+", 1)
    if len(local) < 2 or not local[1].startswith("cu"):
        return None
    digits = re.sub(r"[^0-9].*", "", local[1][2:])  # 'cu130' -> '130'
    if not digits:
        return None
    return int(digits) // 10  # '130' -> 13, '128' -> 12, '118' -> 11


def _select_torchao_spec(torch_version: str | None) -> str:
    """Map an installed torch version string (e.g. '2.10.0+cu130') to the torchao
    pip spec whose cpp extensions match it. Falls back to _TORCHAO_DEFAULT_SPEC for
    torch <=2.9, a non-2.x major, or an unparseable/missing version. Pure function.
    """
    if not torch_version:
        return _TORCHAO_DEFAULT_SPEC
    release = str(torch_version).split("+", 1)[0]  # drop +cu130/+rocm6.4/+cpu
    parts = release.split(".")
    try:
        # Strip any pre-release/dev suffix from the minor (e.g. '10rc1' -> '10'),
        # matching wheel_utils.probe_torch_wheel_env.
        minor_str = re.sub(r"[^0-9].*", "", parts[1]) if len(parts) > 1 else ""
        major, minor = int(parts[0]), int(minor_str)
    except (IndexError, ValueError):
        return _TORCHAO_DEFAULT_SPEC
    if major != 2:
        return _TORCHAO_DEFAULT_SPEC
    if minor >= 11:
        return _TORCHAO_TORCH_211_PLUS_SPEC  # newest known build; covers 2.11+
    if minor == 10:
        # cu130+ can't load 0.16.0's CUDA-12 cpp; use 0.17.0 (cpp skipped, not crashed).
        cuda_major = _cuda_major_from_torch_version(str(torch_version))
        if cuda_major is not None and cuda_major >= _TORCHAO_CUDA13_MIN_MAJOR:
            return _TORCHAO_TORCH_210_CUDA13_SPEC
        return _TORCHAO_TORCH_210_SPEC
    return _TORCHAO_DEFAULT_SPEC


def _probe_installed_torch_version() -> str | None:
    """Return torch.__version__ from the target venv (sys.executable), or None if
    torch is absent/unimportable. Cross-platform (unlike probe_torch_wheel_env,
    which is Linux-only); mirrors the subprocess probe in _ensure_cuda_torch.
    """
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, sys; sys.stdout.write(getattr(torch, '__version__', ''))",
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 90,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    lines = [line.strip() for line in (probe.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _installed_torch_is_windows_rocm() -> bool:
    """Return True when the target venv currently has a Windows ROCm torch build.

    This is a belt-and-suspenders guard for the torchao override step: if the
    earlier ROCm install path failed to set _rocm_windows_torch_installed but the
    venv already contains a ROCm torch wheel, still skip torchao because it
    crashes on import on Windows ROCm.
    """
    if not IS_WINDOWS:
        return False
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, torch; "
                    "hip = getattr(getattr(torch, 'version', None), 'hip', None) or ''; "
                    "ver = getattr(torch, '__version__', '').lower(); "
                    "sys.stdout.write('yes' if (hip or 'rocm' in ver or 'rocmsdk' in ver) else '')"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 90,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    lines = [line.strip() for line in (probe.stdout or "").splitlines() if line.strip()]
    return probe.returncode == 0 and bool(lines and lines[-1] == "yes")


# constraints.txt caps new anyio resolutions at <4.14 (#6483), but an install
# from before the cap existed can already be stuck at 4.14+, which later
# constrained installs won't touch since it already satisfies mcp/fastmcp.
_ANYIO_BAD_FLOOR = (4, 14)


def _installed_anyio_version() -> tuple[int, int] | None:
    try:
        from importlib.metadata import version as _pkg_version
        raw = _pkg_version("anyio")
    except Exception:
        return None
    parts = raw.split(".")
    try:
        major = int(parts[0])
        minor = int(re.sub(r"[^0-9].*", "", parts[1])) if len(parts) > 1 else 0
    except (IndexError, ValueError):
        return None
    return (major, minor)


def _repair_bad_anyio() -> None:
    installed = _installed_anyio_version()
    if installed is None or installed < _ANYIO_BAD_FLOOR:
        return
    _safe_print(_dim(f"   anyio {installed[0]}.{installed[1]} found -- reinstalling anyio<4.14..."))
    pip_install(
        "Repairing anyio version",
        "--no-cache-dir",
        "--force-reinstall",
        "anyio<4.14.0",
        constrain = False,
    )


# AMD Windows ROCm wheels (repo.amd.com/rocm/whl/{arch_family}/).
# Override with UNSLOTH_ROCM_WINDOWS_MIRROR for air-gapped/mirror installs.
_ROCM_WINDOWS_INDEX_BASE = (
    os.environ.get("UNSLOTH_ROCM_WINDOWS_MIRROR") or "https://repo.amd.com/rocm/whl"
).rstrip("/")

# gfx arch → AMD index arch-family suffix; each family is a separate
# pip index on repo.amd.com.
_GFX_TO_AMD_INDEX_ARCH: dict[str, str] = {
    "gfx1201": "gfx120X-all",
    "gfx1200": "gfx120X-all",  # RDNA 4
    "gfx1151": "gfx1151",
    "gfx1150": "gfx1150",  # RDNA 3.5 (Strix Halo/Point)
    "gfx1103": "gfx110X-all",
    "gfx1102": "gfx110X-all",  # RDNA 3
    "gfx1101": "gfx110X-all",
    "gfx1100": "gfx110X-all",
    "gfx1036": "gfx103X-all",
    "gfx1035": "gfx103X-all",  # RDNA 2 (RX 6000)
    "gfx1034": "gfx103X-all",
    "gfx1033": "gfx103X-all",
    "gfx1032": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1030": "gfx103X-all",
    "gfx90a": "gfx90a",
    "gfx908": "gfx908",  # MI200/MI100
}

# bitsandbytes continuous-release_main wheels with the ROCm 4-bit GEMV fix
# (bnb PR #1887, post-0.49.2). bnb <= 0.49.2 NaNs at decode shape on every
# AMD GPU. Drop the pin once bnb 0.50+ ships on PyPI.
_BNB_ROCM_PRERELEASE_URLS: dict[str, str] = {
    "x86_64": (
        "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/"
        "download/continuous-release_main/"
        "bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
    ),
    "aarch64": (
        "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/"
        "download/continuous-release_main/"
        "bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl"
    ),
    # Windows ROCm wheel ships libbitsandbytes_rocm{VER}.dll. BNB's HIP
    # auto-detect may mismatch the DLL suffix, so we scan the wheel and set
    # BNB_ROCM_VERSION in _install_bnb_windows_rocm() and worker.py.
    "win_amd64": (
        "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/"
        "download/continuous-release_main/"
        "bitsandbytes-1.33.7.preview-py3-none-win_amd64.whl"
    ),
}
_BNB_ROCM_PYPI_FALLBACK = "bitsandbytes>=0.49.1"


def _bnb_rocm_prerelease_url() -> str | None:
    """Return the continuous-release_main bnb wheel URL for the current arch,
    or None when no pre-release wheel is available.
    """
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return _BNB_ROCM_PRERELEASE_URLS.get(arch)


def _amd_smi_env() -> dict[str, str] | None:
    """On Windows, env with __COMPAT_LAYER=RunAsInvoker; None elsewhere.
    NB: RunAsInvoker doesn't stop amd-smi's runtime elevation (its manifest is
    asInvoker -- it elevates a child via ShellExecute). The real guard is
    _amd_smi_allowed() below; this is harmless belt-and-suspenders."""
    if platform.system() != "Windows":
        return None
    return {**os.environ, "__COMPAT_LAYER": "RunAsInvoker"}


def _path_inside_venv(path: str) -> bool:
    """True if ``path`` is inside the active venv (sys.prefix).

    The venv hipInfo.exe (AMD wheel, put on PATH by the bnb fix) is NOT a HIP SDK
    (_amd_smi_allowed)."""
    try:
        # realpath (not abspath): resolve symlinks/8.3 names so an aliased venv matches.
        _root = os.path.normcase(os.path.realpath(sys.prefix))
        # Guard a root-dir prefix (C:\ or /): commonpath would match every path on
        # it. A venv is never at root, so treat that as outside.
        if os.path.dirname(_root) == _root:
            return False
        return os.path.normcase(os.path.commonpath([os.path.realpath(path), _root])) == _root
    except (ValueError, OSError):
        # Different drive / unresolvable -> treat as outside the venv.
        return False


def _external_hipinfo_on_path() -> bool:
    """True if a hipinfo OUTSIDE the venv is on PATH.

    shutil.which returns only the first hit, so the venv hipInfo could shadow a
    real HIP SDK's; scan every PATH entry and skip the venv copy."""
    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        _dir = _dir.strip('"')  # PATH entries can be quoted on Windows
        if not _dir:
            continue
        _candidate = os.path.join(_dir, "hipinfo.exe")
        if os.path.isfile(_candidate) and not _path_inside_venv(_candidate):
            return True
    return False


def _amd_smi_allowed() -> bool:
    """Whether it is safe to spawn amd-smi here.

    On Windows w/o a working HIP runtime, amd-smi elevates a child and pops a
    UAC/DiskPart prompt RunAsInvoker can't suppress. Only call it on Windows with
    a HIP SDK (hipinfo present) or UNSLOTH_ENABLE_AMD_SMI=1; Linux/macOS always.
    """
    if platform.system() != "Windows":
        return True
    flag = os.environ.get("UNSLOTH_ENABLE_AMD_SMI", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    # A real HIP SDK lets amd-smi run un-elevated; hipinfo-on-PATH is the proxy.
    # Ignore the venv hipInfo.exe (AMD wheel via bnb fix): not a HIP SDK, doesn't
    # stop amd-smi's DiskPart UAC.
    if _external_hipinfo_on_path():
        return True
    for _var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        _root = os.environ.get(_var)
        if not _root:
            continue
        _candidate = os.path.join(_root, "bin", "hipinfo.exe")
        if os.path.isfile(_candidate) and not _path_inside_venv(_candidate):
            return True
    return False


def _detect_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm stack, or None."""
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            # Explicit length guard: don't rely on the broad except below to
            # swallow IndexError on a single-component version (e.g. "6\n").
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # Try amd-smi version (outputs "... | ROCm version: X.Y.Z").
    # Gated off on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt);
    # hipconfig below covers that case.
    amd_smi = shutil.which("amd-smi") if _amd_smi_allowed() else None
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
                env = _amd_smi_env(),
            )
            if result.returncode == 0:
                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    # Try hipconfig --version (outputs bare version like "6.3.21234.2")
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run(
                [hipconfig, "--version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                timeout = 5,
            )
            if result.returncode == 0:
                raw = result.stdout.decode().strip().split("\n")[0]
                parts = raw.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].split("-")[0].isdigit():
                    return int(parts[0]), int(parts[1].split("-")[0])
        except Exception:
            pass

    # Distro package-manager fallbacks: package-managed ROCm can expose GPUs via
    # rocminfo/amd-smi but lack /opt/rocm/.info/version and hipconfig, so probe
    # dpkg (Debian/Ubuntu) and rpm (RHEL/Fedora/SUSE) for the rocm-core version.
    # Matches install.sh::get_torch_index_url so `studio update` == fresh install.
    for cmd in (
        ["dpkg-query", "-W", "-f=${Version}\n", "rocm-core"],
        ["rpm", "-q", "--qf", "%{VERSION}\n", "rocm-core"],
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue
        raw = result.stdout.strip()
        # dpkg can prepend an epoch ("1:6.3.0-1"); strip it before parsing.
        raw = re.sub(r"^\d+:", "", raw)
        m = re.match(r"(\d+)[.-](\d+)", raw)
        if m:
            return int(m.group(1)), int(m.group(2))

    return None


def _pick_visible_index(num_tokens: int) -> int:
    """Resolve HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES to an index into a
    list of length num_tokens. Returns 0 (first GPU) for unset, empty, '-1',
    UUID-style, or out-of-range values."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        _val = os.environ.get(_env)
        if _val is None:
            continue
        _val = _val.strip()
        if _val == "" or _val == "-1":
            return 0
        _first = _val.split(",")[0].strip()
        try:
            _idx = int(_first)
            if 0 <= _idx < num_tokens:
                return _idx
        except ValueError:
            pass
        return 0
    return 0


def _detect_windows_gfx_arch() -> str | None:
    """Return the gcnArchName on Windows (e.g. 'gfx1200'), or None.

    Probe order matches the PowerShell installer: env-var override, then
    hipinfo (PATH or HIP_PATH/ROCM_PATH bin), then amd-smi. Without the
    amd-smi fallback, runtime-only AMD installs lacking hipinfo on PATH
    return early and `studio update` cannot repair a CPU-only venv.

    On multi-GPU hosts, detected gfx tokens are deduplicated (preserving
    enumeration order) and HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES picks
    which to install for. The first GPU is used when no env var is set.
    """
    # 1. Explicit override (matches PowerShell installer's env-var path).
    _override = os.environ.get("UNSLOTH_ROCM_GFX_ARCH")
    if _override and _override.strip():
        return _override.strip().lower()

    def _dedup_pick(tokens: list[str]) -> "str | None":
        if not tokens:
            return None
        # Index into the full ordered list so HIP_VISIBLE_DEVICES addresses
        # GPU N on mixed-arch hosts, then return that arch.
        return tokens[_pick_visible_index(len(tokens))]

    # 2. hipinfo via PATH, then HIP_PATH\bin / ROCM_PATH\bin.
    hipinfo = shutil.which("hipinfo")
    if not hipinfo:
        for _env_var in ("HIP_PATH", "ROCM_PATH"):
            _root = os.environ.get(_env_var)
            if _root:
                _candidate = os.path.join(_root, "bin", "hipinfo.exe")
                if os.path.isfile(_candidate):
                    hipinfo = _candidate
                    break
    if not hipinfo:
        # 2b. AMD torch wheels ship hipInfo.exe into the venv Scripts dir
        # (next to python.exe); resolvable even on driver-only hosts with no
        # SDK install at all. Lets `studio update` re-detect the arch on a
        # venv that already has the AMD wheel.
        _venv_hipinfo = os.path.join(os.path.dirname(sys.executable), "hipInfo.exe")
        if os.path.isfile(_venv_hipinfo):
            hipinfo = _venv_hipinfo
    if hipinfo:
        try:
            result = subprocess.run(
                [hipinfo],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                timeout = 10,
            )
            # Accept partial output even when hipinfo crashes (e.g. 0xC0000005 /
            # STATUS_ACCESS_VIOLATION on some RDNA 4 hosts): a gcnArchName in stdout
            # means the device was enumerated pre-crash, so the arch is trustworthy.
            # Ignoring it causes a silent CPU PyTorch fallback (issue #6043).
            text = result.stdout.decode(errors = "replace")
            # findall gets every gcnArchName line so multi-GPU hosts are
            # enumerable and HIP_VISIBLE_DEVICES selects correctly.
            _tokens = [
                t.strip().lower() for t in re.findall(r"(?im)^\s*gcnArchName\s*:\s*(\S+)", text)
            ]
            _pick = _dedup_pick(_tokens)
            if _pick:
                return _pick
        except Exception:
            pass

    # 3. amd-smi fallback -- runtime-only Radeon installs ship amd-smi but no hipinfo.
    # Gated off on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt); the arch
    # arrives via --rocm-gfx / name inference there, so this is only needed when safe.
    amd_smi = shutil.which("amd-smi") if _amd_smi_allowed() else None
    if amd_smi:
        for _args in (("static", "--asic"), ("list",)):
            try:
                result = subprocess.run(
                    [amd_smi, *_args],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    timeout = 10,
                    env = _amd_smi_env(),
                )
                if result.returncode != 0:
                    continue
                text = result.stdout.decode(errors = "replace")
                # Prefer labelled gfx lines; fall back to bare tokens.
                _labelled = re.findall(
                    r"(?im)^\s*(?:target_graphics_version|gfx|arch|asic)\b[^:\r\n]*:\s*(gfx[1-9][0-9a-z]{2,3})\b",
                    text,
                )
                _tokens = [t.lower() for t in _labelled]
                if not _tokens:
                    _tokens = re.findall(r"\bgfx[1-9][0-9a-z]{2,3}\b", text.lower())
                _pick = _dedup_pick(_tokens)
                if _pick:
                    return _pick
            except Exception:
                continue

    # 4. Last resort: GPU marketing name via WMI → arch table. Driver-only
    #    hosts (Adrenalin, no HIP SDK) have neither hipinfo nor amd-smi
    #    (amd-smi does not exist on Windows at all), but the display driver
    #    always knows the GPU name. Mirrors setup.ps1's $nameArchTable so a
    #    standalone `studio update` can repair a CPU-only venv on such hosts.
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 30,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            _tokens = []
            for _name in result.stdout.decode(errors = "replace").splitlines():
                _arch = _gfx_arch_from_gpu_name(_name.strip())
                if _arch:
                    _tokens.append(_arch)
            _pick = _dedup_pick(_tokens)
            if _pick:
                print(f"   gfx arch inferred from GPU name (WMI): {_pick}")
                return _pick
    except Exception:
        pass
    return None


# GPU marketing-name → gfx arch table, mirroring setup.ps1's $nameArchTable.
# Most-specific first; first match wins. Covers only arches the ROCm
# prebuilts / AMD Windows torch indexes support; unknown names return None
# (callers then fall back cleanly to CPU).
_WIN_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    (r"9070 XT|9080", "gfx1201"),  # RDNA 4 (Radeon RX 9070 XT / 9080)
    (r"9070|9060", "gfx1200"),  # RDNA 4 (Radeon RX 9070 / 9060)
    # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
    (r"8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max", "gfx1151"),
    # RDNA 3.5 (Strix/Krackan Point: Radeon 890M/880M iGPU, Ryzen AI 9 HX 370/375)
    (
        r"890M|880M|860M|840M|Strix Point|Krackan|HX 37[05]|AI 9 HX|AI 9 36[05]"
        r"|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33",
        "gfx1150",
    ),
    # RDNA 3 desktop / workstation (Navi 31)
    (r"RX 7900|RX 7800|RX 7700(?!S)|PRO W7900|PRO W7800|PRO W7700", "gfx1100"),
    (r"RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500|PRO V710", "gfx1102"),  # Navi 33
    # RDNA 3 iGPU (Phoenix / Hawk Point)
    (r"780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme", "gfx1103"),
    (r"RX 6900|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900", "gfx1030"),  # Navi 21
    (r"RX 6650|RX 6600|PRO W6600|PRO W6650", "gfx1032"),  # Navi 23
    (r"RX 6500|RX 6400|RX 6300|PRO W6400|PRO W6500", "gfx1034"),  # Navi 24
]


def _gfx_arch_from_gpu_name(name: str) -> "str | None":
    """Map a GPU marketing name to its gfx arch via _WIN_GPU_NAME_ARCH_TABLE."""
    if not name:
        return None
    for _pat, _arch in _WIN_GPU_NAME_ARCH_TABLE:
        if re.search(_pat, name, re.IGNORECASE):
            return _arch
    return None


def _linux_amd_gfx_from_cpuinfo() -> "str | None":
    """Infer gfx arch from /proc/cpuinfo on integrated AMD APUs (Strix Halo/Point)."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return None
    if re.search(r"Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo", text, re.IGNORECASE):
        return "gfx1151"
    if re.search(
        r"890M|880M|860M|840M|Strix Point|Krackan|HX 37[05]|AI 9 HX|AI 9 36[05]"
        r"|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33",
        text,
        re.IGNORECASE,
    ):
        return "gfx1150"
    return None


def _linux_amd_gfx_from_lspci() -> "str | None":
    """First AMD display-class lspci line mapping to a known gfx arch. A non-AMD
    controller can enumerate first (Intel/ASPEED before an AMD dGPU), so scan
    them all. The vendor guard is case-SENSITIVE: a -i "ATI" would match
    "CorporATIon" on every Intel/NVIDIA line. Whole-line matching also survives
    the 0000: PCI domain prefix."""
    lspci = shutil.which("lspci")
    if not lspci:
        return None
    try:
        result = subprocess.run(
            [lspci, "-nn"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if not re.search(r"VGA compatible controller|3D controller|Display controller", line, re.I):
            continue
        if not re.search(r"AMD|ATI", line):
            continue
        arch = _gfx_arch_from_gpu_name(line)
        if arch:
            return arch
    return None


def _is_wsl() -> bool:
    """True on WSL, where the AMD GPU is reached via /dev/dxg (not /dev/kfd)."""
    if os.path.exists("/dev/dxg"):
        return True
    try:
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_rocm_runtime_present() -> bool:
    """librocdxg (the WSL ROCDXG bridge that lets HIP reach the GPU over /dev/dxg)
    under a ROCm lib dir. Its absence marks a WSL box whose ROCm was never set up."""
    dirs = ["/opt/rocm/lib", "/opt/rocm/lib64"]
    dirs += glob.glob("/opt/rocm-*/lib") + glob.glob("/opt/rocm-*/lib64")
    return any(
        os.path.exists(os.path.join(d, so))
        for d in dirs
        for so in ("librocdxg.so", "librocdxg.so.1")
    )


def _linux_amd_display_device_present() -> bool:
    """Any AMD (vendor 0x1002) PCI display-class (0x03*) device in sysfs.
    /proc/cpuinfo leaks the HOST CPU model into VMs/containers that received no
    AMD GPU, so the CPU-model text alone is not GPU evidence; this is the
    device-level check (mirrors install.sh _amd_gpu_present_via_pci)."""
    try:
        for dev in Path("/sys/bus/pci/devices").iterdir():
            try:
                if (dev / "vendor").read_text().strip() != "0x1002":
                    continue
                if (dev / "class").read_text().strip().startswith("0x03"):
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def _infer_linux_amd_gfx_arch() -> "str | None":
    """Infer gfx when ROCm runtime is absent but the host is a known AMD arch (unslothai#7301)."""
    override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
    if override:
        return override
    if _is_wsl():
        # cpuinfo/lspci see the host APU even on a WSL box whose ROCDXG runtime
        # was never bootstrapped; inferring there would install per-arch ROCm
        # wheels into an env that still can't expose the GPU. Skip unless that
        # runtime is present -- WSL enumerates no PCI display device, so
        # /dev/dxg + librocdxg IS the GPU evidence there.
        if not _wsl_rocm_runtime_present():
            return None
    elif not _linux_amd_display_device_present():
        # Native Linux: a VM/container on a Strix host still shows the host CPU
        # model in /proc/cpuinfo while receiving no AMD GPU, so require an AMD
        # display device before trusting the CPU-model inference. The lspci
        # fallback reads the same PCI space and would find nothing here either.
        return None
    cpu_gfx = _linux_amd_gfx_from_cpuinfo()
    if cpu_gfx:
        return cpu_gfx
    return _linux_amd_gfx_from_lspci()


def _amd_arch_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD per-arch pip index URL for a gfx arch (Linux + Windows).

    Windows honors UNSLOTH_ROCM_WINDOWS_MIRROR (via _windows_rocm_index_url);
    Linux honors UNSLOTH_AMD_ROCM_MIRROR -- the same var install.sh uses -- so a
    mirrored/air-gapped Linux repair reaches the index install.sh chose rather
    than falling back to repo.amd.com. Both default to repo.amd.com when unset.
    """
    if IS_WINDOWS:
        return _windows_rocm_index_url(gfx_arch)
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    base = (os.environ.get("UNSLOTH_AMD_ROCM_MIRROR") or "https://repo.amd.com/rocm/whl").rstrip(
        "/"
    )
    return f"{base}/{arch_family}/"


def _windows_rocm_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD pip index URL for the given GPU arch, or None if unsupported."""
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    return f"{_ROCM_WINDOWS_INDEX_BASE}/{arch_family}/"


def _detect_bnb_rocm_dll_ver() -> str | None:
    """Scan the installed bitsandbytes package for libbitsandbytes_rocm{VER}.dll.

    Returns the version suffix (e.g. ``"72"``, ``"713"``) or ``None`` if
    bitsandbytes is not installed or no ROCm DLL is found. Does NOT import
    bitsandbytes — uses importlib.util.find_spec, so it is safe to call
    before BNB is imported.
    """
    import importlib.util

    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None or not spec.submodule_search_locations:
        return None
    all_vers: list[str] = []
    for pkg_dir in spec.submodule_search_locations:
        for dll in glob.glob(os.path.join(pkg_dir, "libbitsandbytes_rocm*.dll")):
            m = re.search(r"libbitsandbytes_rocm(\d+)\.dll", os.path.basename(dll))
            if m:
                all_vers.append(m.group(1))
    # Highest numeric suffix wins (e.g. "713" over "72"); glob order is not
    # guaranteed, so sort rather than take the first match.
    return max(all_vers, key = lambda v: int(v)) if all_vers else None


_BNB_ROCM_SITECUSTOMIZE_BEGIN = "# BEGIN Unsloth BNB_ROCM_VERSION"
_BNB_ROCM_SITECUSTOMIZE_END = "# END Unsloth BNB_ROCM_VERSION"
_BNB_ROCM_VERSION_SOURCE_ENV = "UNSLOTH_BNB_ROCM_VERSION_SOURCE"
_BNB_ROCM_VERSION_SOURCE_SITECUSTOMIZE = "sitecustomize"
_BNB_ROCM_VERSION_SOURCE_DETECTED = "detected"


def _persist_bnb_rocm_version(version: str) -> bool:
    """Persist BNB_ROCM_VERSION for future Python processes in this venv."""
    version = str(version).strip()
    if not version:
        return False

    site_packages = sysconfig.get_path("purelib")
    if not site_packages:
        return False

    sitecustomize_path = Path(site_packages) / "sitecustomize.py"
    block = (
        f"{_BNB_ROCM_SITECUSTOMIZE_BEGIN}\n"
        "import os as _unsloth_os\n"
        "_unsloth_existing_bnb_rocm = _unsloth_os.environ.get('BNB_ROCM_VERSION')\n"
        f"_unsloth_os.environ.setdefault('BNB_ROCM_VERSION', {version!r})\n"
        "if _unsloth_existing_bnb_rocm is None and "
        f"_unsloth_os.environ.get('BNB_ROCM_VERSION') == {version!r}:\n"
        "    _unsloth_os.environ.setdefault("
        f"{_BNB_ROCM_VERSION_SOURCE_ENV!r}, "
        f"{_BNB_ROCM_VERSION_SOURCE_SITECUSTOMIZE!r})\n"
        "del _unsloth_existing_bnb_rocm\n"
        f"{_BNB_ROCM_SITECUSTOMIZE_END}\n"
    )

    try:
        sitecustomize_path.parent.mkdir(parents = True, exist_ok = True)
        existing = (
            sitecustomize_path.read_text(encoding = "utf-8") if sitecustomize_path.exists() else ""
        )
        # Strip all managed regions, including one whose END marker was lost to
        # an interrupted write, then append exactly one fresh block.
        pattern = re.compile(
            rf"{re.escape(_BNB_ROCM_SITECUSTOMIZE_BEGIN)}.*?"
            rf"(?:{re.escape(_BNB_ROCM_SITECUSTOMIZE_END)}\n?|\Z)",
            re.DOTALL,
        )
        remainder = pattern.sub("", existing)
        separator = "" if not remainder or remainder.endswith("\n") else "\n"
        updated = f"{remainder}{separator}{block}"
        tmp_path = sitecustomize_path.with_name(
            f"{sitecustomize_path.name}.unsloth-tmp{os.getpid()}"
        )
        try:
            tmp_path.write_text(updated, encoding = "utf-8")
            if sitecustomize_path.exists():
                shutil.copymode(sitecustomize_path, tmp_path)
            os.replace(tmp_path, sitecustomize_path)
        finally:
            tmp_path.unlink(missing_ok = True)
    except (OSError, UnicodeDecodeError) as exc:
        print(
            f"   Warning: could not persist BNB_ROCM_VERSION={version} "
            f"to {sitecustomize_path}: {exc}"
        )
        return False

    return True


def _has_rocm_gpu() -> bool:
    """Return True only if an actual AMD GPU is visible (not just ROCm tools installed).

    Always returns False when an NVIDIA GPU is present -- NVIDIA takes
    priority on mixed hosts and prevents every detection path below
    (rocminfo, amd-smi, KFD sysfs) from producing a false positive even
    if ROCm tools are installed alongside the NVIDIA driver.
    """
    if _has_usable_nvidia_gpu():
        return False
    for cmd, check_fn in (
        # rocminfo: look for a real gfx GPU id (3-4 chars, nonzero first digit).
        # gfx000 is the CPU agent; ROCm 6.1+ also emits generic ISA lines like
        # "gfx11-generic"/"gfx9-4-generic" with only 1-2 digits before the dash,
        # which must not be treated as a real GPU.
        (
            ["rocminfo"],
            lambda out: bool(re.search(r"gfx[1-9][0-9a-z]{2,3}", out.lower())),
        ),
        # amd-smi list: require "GPU: <number>" data rows, not just a header
        (
            ["amd-smi", "list"],
            lambda out: bool(re.search(r"(?im)^gpu\s*[:\[]\s*\d", out)),
        ),
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        # Skip amd-smi on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt);
        # rely on rocminfo / the sysfs fallback there.
        if cmd[0] == "amd-smi" and not _amd_smi_allowed():
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
                env = _amd_smi_env() if cmd[0] == "amd-smi" else None,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            if check_fn(result.stdout):
                return True
    # sysfs KFD topology fallback (Linux only) -- matches install.sh's runtime-only
    # detection. On minimal package-managed installs (no rocminfo / amd-smi), the
    # kernel exposes AMD GPUs via /sys/class/kfd so `studio update` can still repair.
    #
    # Guard: reject any KFD node whose properties file reports a non-AMD vendor. The
    # NVIDIA open kernel module (driver 560+) registers KFD nodes with a non-zero
    # gpu_id and vendor_id 4318 (0x10DE), not the AMD 4098 (0x1002); without this
    # check the fallback returns True on NVIDIA-only hosts, installing ROCm wheels.
    if sys.platform != "win32":
        try:
            kfd_nodes = "/sys/class/kfd/kfd/topology/nodes"
            if os.path.isdir(kfd_nodes):
                for entry in os.listdir(kfd_nodes):
                    gpu_id_path = os.path.join(kfd_nodes, entry, "gpu_id")
                    try:
                        with open(gpu_id_path) as fh:
                            gpu_id = fh.read().strip()
                    except OSError:
                        continue
                    if not gpu_id or gpu_id == "0":  # gpu_id 0 = CPU node
                        continue
                    # Require AMD vendor_id 4098 (0x1002). KFD properties files exist
                    # on every kernel exposing /sys/class/kfd, so a missing file means
                    # AMD ownership is unconfirmed -- skip the node rather than risk a
                    # false positive (e.g. NVIDIA open-driver KFD nodes lacking it).
                    props_path = os.path.join(kfd_nodes, entry, "properties")
                    try:
                        with open(props_path) as fh:
                            props = fh.read()
                    except OSError:
                        continue  # can't confirm vendor -- skip
                    if not re.search(r"\bvendor_id\s+4098\b", props):
                        continue
                    return True
        except OSError:
            pass
    return False


def _has_usable_nvidia_gpu() -> bool:
    """Return True when an NVIDIA GPU is present and usable.

    Primary probe: nvidia-smi -L (subprocess).
    Fallback: /proc/driver/nvidia/gpus/ sysfs (Linux only) -- handles the
    case where nvidia-smi is present but the subprocess fails (PATH gap,
    timeout, driver initialisation race). If either probe confirms an
    NVIDIA GPU the function returns True so _has_rocm_gpu() is blocked.

    CUDA_VISIBLE_DEVICES set to "" or "-1" hides every NVIDIA device (mixed
    AMD+NVIDIA hosts steering work to the AMD card); neither probe honours
    that env var, so check it first and report the GPU as not usable. Unset
    means all devices visible.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return False
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            result = subprocess.run(
                [exe, "-L"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
            if result.returncode == 0 and "GPU " in result.stdout:
                return True
        except Exception:
            pass
    # Fallback: the NVIDIA driver exposes one subdirectory per GPU under
    # /proc/driver/nvidia/gpus/ on Linux regardless of nvidia-smi state.
    if sys.platform != "win32":
        try:
            gpu_dir = "/proc/driver/nvidia/gpus"
            if os.path.isdir(gpu_dir) and os.listdir(gpu_dir):
                return True
        except OSError:
            pass
    return False


def _detect_amd_gfx_codes() -> list[str]:
    """Return the AMD gfx ISA strings visible to ROCm (e.g. ['gfx1151']).

    Probes rocminfo, then falls back to ``amd-smi list`` and ``amd-smi
    static --asic`` for runtime-only Radeon hosts that ship amd-smi but no
    rocminfo. Returns an empty list when no probe yields a gfx target.
    """

    def _extract(text: str) -> list[str]:
        codes = re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())
        return list(dict.fromkeys(f"gfx{c}" for c in codes))

    probes: list[list[str]] = []
    if shutil.which("rocminfo"):
        probes.append(["rocminfo"])
    # Gate amd-smi off on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt).
    if shutil.which("amd-smi") and _amd_smi_allowed():
        probes.append(["amd-smi", "list"])
        probes.append(["amd-smi", "static", "--asic"])
    for cmd in probes:
        try:
            result = subprocess.run(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 15,
                env = _amd_smi_env() if cmd[0] == "amd-smi" else None,
            )
        except Exception:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue
        codes = _extract(result.stdout)
        if codes:
            return codes
    return []


# Set by _ensure_rocm_torch() on success; suppresses the post-install AMD warning.
_rocm_windows_torch_installed: bool = False


def _install_bnb_windows_rocm() -> bool:
    """Install the AMD Windows BNB prerelease wheel. Returns True on success.

    The continuous-release wheel is intentionally mismatched: the filename
    encodes 1.33.7.preview (parsed as 1.33.7rc0 by PEP 440) while the wheel
    metadata reports 0.50.0.dev0. uv rejects this filename/metadata mismatch,
    and bypassing it with UV_SKIP_WHEEL_FILENAME_CHECK still leaves uv mangling
    the bitsandbytes install. Per the AMD install guide
    (https://unsloth.ai/docs/get-started/install/amd/amd-hackathon) the wheel
    must be installed with plain pip, not uv, so we force pip (force_pip=True);
    plain pip performs no wheel filename/metadata check.
    """
    _bnb_win_url = _BNB_ROCM_PRERELEASE_URLS.get("win_amd64")
    if _bnb_win_url is None:
        return False
    _ok = pip_install_try(
        "bitsandbytes (AMD Windows, pre-release main)",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-deps",
        _bnb_win_url,
        constrain = False,
        force_pip = True,
    )
    if not _ok:
        return False
    # Detect the actual ROCm DLL suffix in the wheel and set BNB_ROCM_VERSION so bnb
    # loads the right DLL regardless of torch.version.hip (the wheel may ship "72"
    # while torch reports 7.13). The worker subprocess inherits it; fall back to "72"
    # if detection fails (e.g. a no-op / dry-run install).
    _env_ver = os.environ.get("BNB_ROCM_VERSION")
    _env_is_persisted_default = (
        os.environ.get(_BNB_ROCM_VERSION_SOURCE_ENV) == _BNB_ROCM_VERSION_SOURCE_SITECUSTOMIZE
    )
    _persist_detected_version = False
    if _env_ver and not _env_is_persisted_default:
        _ver = _env_ver
    else:
        _ver = _detect_bnb_rocm_dll_ver() or "72"
        os.environ["BNB_ROCM_VERSION"] = _ver
        os.environ[_BNB_ROCM_VERSION_SOURCE_ENV] = _BNB_ROCM_VERSION_SOURCE_DETECTED
        _persist_detected_version = True
    if _persist_detected_version:
        _persist_bnb_rocm_version(_ver)
    # Make hipInfo.exe (shipped into venv Scripts by the AMD torch wheel) resolvable
    # via PATH for this process and every child python (import checks, precompile):
    # bitsandbytes runs hipinfo.exe at import to detect the GPU arch and logs a scary
    # (harmless) ERROR + WARNING when it is missing. Scripts is on PATH only for an
    # activated venv, which neither Unsloth nor the installer's children ever do.
    _scripts_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")) and not shutil.which(
        "hipinfo.exe"
    ):
        os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")
    return True


def _detect_cuda_torch_index_url() -> str:
    """Return the pytorch.org CUDA wheel index URL for the host's NVIDIA driver.

    Mirrors install.sh::get_torch_index_url's CUDA ladder so `studio update` repairs
    to the same wheel family a fresh install would pick. Honours the explicit
    overrides first (UNSLOTH_TORCH_INDEX_URL / _FAMILY) so a headless / CI install
    never lets the host GPU decide. Otherwise probes nvidia-smi (parsing both "CUDA
    Version:" and "CUDA UMD Version:"), defaulting to cu126 when unreadable.
    """
    _override_url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if _override_url:
        return _trim_index_path_slashes(_override_url)
    _override_family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    if _override_family:
        return f"{_PYTORCH_WHL_BASE}/{_override_family.strip('/')}"
    exe = shutil.which("nvidia-smi")
    if not exe and os.path.isfile("/usr/bin/nvidia-smi"):
        exe = "/usr/bin/nvidia-smi"
    tag = "cu126"  # default when the driver CUDA version cannot be read
    if exe:
        try:
            result = subprocess.run(
                [exe],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
            if result.returncode == 0:
                m = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    major, minor = int(m.group(1)), int(m.group(2))
                    if major >= 13:
                        tag = "cu130"
                    elif major == 12 and minor >= 8:
                        tag = "cu128"
                    elif major == 12 and minor >= 6:
                        tag = "cu126"
                    elif major >= 12:
                        tag = "cu124"
                    elif major >= 11:
                        tag = "cu118"
                    else:
                        tag = "cpu"  # ancient driver: no usable CUDA wheels
        except Exception:
            pass
    return f"{_PYTORCH_WHL_BASE}/{tag}"


def _explicit_torch_index_url() -> "str | None":
    """The wheel index URL pinned via UNSLOTH_TORCH_INDEX_URL / _FAMILY, else None.

    Lets the CUDA/ROCm repair helpers honour the exact pinned family/URL instead
    of re-probing the GPU. Mirrors install.sh::get_torch_index_url's override.
    """
    url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if url:
        return _trim_index_path_slashes(url)
    family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    if family:
        return f"{_PYTORCH_WHL_BASE}/{family.strip('/')}"
    return None


def _is_pip_rocm_family_leaf(leaf: str) -> bool:
    """True when a lowercased leaf names a pip --index-url ROCm family: an EXACT
    rocm<digits>[.<digits>] leaf or a gfx leaf. A suffixed leaf (rocm-rel-7.2.1,
    rocm7.2-private) starts with "rocm" but is a custom pin the verbatim path owns, so
    match EXACTLY. Mirrors install.sh / setup.ps1.
    """
    # gfx must be followed by a digit (gfx90a, gfx1151, gfx120X-all): a gfx-prefixed
    # custom leaf (gfx-private) is a verbatim pin, like rocm7.2-private.
    return bool(re.fullmatch(r"rocm\d+(?:\.\d+)?", leaf)) or bool(re.match(r"gfx\d", leaf))


def _explicit_rocm_torch_index_url() -> "str | None":
    """The pinned wheel index URL when it names a pip ROCm family (rocm<d>/gfx*), else None."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _is_pip_rocm_family_leaf(_torch_index_leaf(url)) else None


def _rocm_pin_family_mismatch(pin_url: str, installed_ver: str) -> bool:
    """True when an explicit ROCm pin names a different ROCm family than the installed
    ROCm torch, so the pin needs a reinstall. Mirrors setup.ps1's stale-venv comparison;
    same three pin-leaf cases as _ensure_rocm_torch. A same-family pin is NOT a mismatch.
    """
    leaf = _torch_index_leaf(pin_url)
    # Pinned ROCm version. The family classifier accepts a major-only rocm<d> leaf too,
    # so parse the minor as optional; a major-only pin compares on the major alone.
    _pin_rocm = re.match(r"^rocm(\d+)(?:\.(\d+))?", leaf)
    _pin_major = int(_pin_rocm.group(1)) if _pin_rocm else None
    _pin_ver = (
        (int(_pin_rocm.group(1)), int(_pin_rocm.group(2)))
        if _pin_rocm and _pin_rocm.group(2) is not None
        else None
    )
    # Installed +rocmX.Y version; a THREE-part +rocmA.B.C tag is the AMD per-arch
    # (repo.amd.com/gfx*) signature vs a two-part pytorch.org wheel.
    _inst_rocm = re.search(r"\+rocm(\d+)\.(\d+)", installed_ver)
    _inst_ver = (int(_inst_rocm.group(1)), int(_inst_rocm.group(2))) if _inst_rocm else None
    _inst_is_perarch = re.search(r"\+rocm\d+\.\d+\.\d+", installed_ver) is not None
    # A ROCm build MUST carry a +rocm tag; an untagged wheel never satisfies a ROCm pin.
    _inst_has_rocm = re.search(r"\+rocm", installed_ver) is not None
    # Installed torch RELEASE (before "+") is 2.11+.
    _inst_rel = re.match(r"^(\d+)\.(\d+)", installed_ver)
    _inst_is_211 = (
        (int(_inst_rel.group(1)), int(_inst_rel.group(2))) >= (2, 11) if _inst_rel else False
    )

    if leaf.startswith("gfx"):
        # 2.11-allowlist arches expect the AMD per-arch wheel (three-part +rocmA.B.C,
        # torch 2.11+); a generic or pre-2.11 build is a mismatch.
        if leaf in _ROCM_GFX_TORCH211_LEAVES:
            return not (_inst_is_211 and _inst_is_perarch)
        # Non-2.11 gfx leaf (<2.11 specs): mismatch on an untagged wheel or torch 2.11+.
        return (not _inst_has_rocm) or _inst_is_211

    # Major-only rocm pin (rocm7): compare majors only -- a +rocm6.4 wheel under a rocm7
    # pin is a mismatch, any +rocm7.x wheel satisfies it (there is no pinned minor to
    # compare, and the 2.11-line fallback below would invert both verdicts).
    if _pin_major is not None and _pin_ver is None:
        if _inst_ver is not None:
            return _inst_ver[0] != _pin_major
        # Untagged wheel never satisfies a ROCm pin; a +rocm tag with an unreadable
        # version is accepted (matches the lenient unreadable fallback below).
        return not _inst_has_rocm

    # rocmX.Y pin. Only KNOWN-2.11 rocm is the 2.11 line (no speculative floor).
    _pin_is_211 = _pin_ver in _ROCM_KNOWN_TORCH211_VERSIONS if _pin_ver is not None else False
    if _pin_ver is not None and _inst_ver is not None:
        # Both readable: exact (major, minor) compare (rocm7.2 pin over +rocm7.13.x ->
        # mismatch, reinstall the pinned wheel).
        if _pin_ver != _inst_ver:
            return True
        # Same family: a KNOWN-2.11 pin whose release drifted off 2.11 (2.12+rocm7.2)
        # violates the spec -> reinstall to floor (exact compare, not >=2.11).
        if _pin_is_211 and _inst_rel is not None:
            if (int(_inst_rel.group(1)), int(_inst_rel.group(2))) != (2, 11):
                return True
        return False
    # rocm pin, unreadable installed version: compare on the 2.11 line, but an untagged
    # wheel never satisfies a rocmX.Y pin -> mismatch.
    if not _inst_has_rocm:
        return True
    return _pin_is_211 != _inst_is_211


def _explicit_cpu_torch_index_url() -> "str | None":
    """The pinned wheel index URL when it names the CPU family (leaf == cpu), else None.

    An explicit CPU pin (UNSLOTH_TORCH_INDEX_FAMILY=cpu or a URL ending in /cpu)
    is authoritative -- see _ensure_cpu_torch.
    """
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _torch_index_leaf(url) == "cpu" else None


def _is_cuda_family_leaf(leaf: str) -> bool:
    """True only for a real CUDA wheel-family leaf: "cu" + digits (cu118, cu128, ...).

    A bare startswith("cu") would match "custom"/"current". The match is EXACT so
    "cu128-private" is NOT a family leaf and routes to the verbatim path instead.
    """
    return re.fullmatch(r"cu[0-9]+", leaf) is not None


def _explicit_cuda_torch_index_url() -> "str | None":
    """The pinned wheel index URL when it names a CUDA family (leaf cuXXX), else None.

    Mirrors _explicit_rocm/cpu_torch_index_url so _ensure_cuda_torch only treats a
    *CUDA* pin as authority to override the NVIDIA-presence gate (an arbitrary mirror
    or a ROCm/CPU pin must not force a CUDA reinstall on a non-NVIDIA host).
    """
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _is_cuda_family_leaf(_torch_index_leaf(url)) else None


def _explicit_unknown_family_torch_index_url() -> "str | None":
    """The pinned index URL when its leaf names NO known torch family, else None.

    Known = rocm* / gfx* / cpu / cuXXX. Anything else (a private mirror /simple,
    /current) is UNKNOWN: version-tag heuristics can't judge it, so the family
    repair helpers must leave it alone (the install applied it verbatim).
    Matches install.sh / setup.ps1 / install.ps1.
    """
    url = _explicit_torch_index_url()
    if url is None:
        return None
    leaf = _torch_index_leaf(url)
    if _is_pip_rocm_family_leaf(leaf) or leaf == "cpu" or _is_cuda_family_leaf(leaf):
        return None
    return url


def _ensure_cuda_torch() -> None:
    """Repair a venv whose torch is a ROCm build on an NVIDIA host.

    Counterpart to _ensure_rocm_torch. A venv poisoned by the pre-fix KFD
    gpu_id false positive (ROCm torch installed on an NVIDIA-only machine)
    keeps that broken torch on `studio update`, because a torch+rocm wheel
    satisfies the version constraint and nothing force-reinstalls it. This
    detects that exact case and reinstalls CUDA torch.

    Only repairs when torch actually links against HIP/ROCm. Healthy CUDA
    torch and deliberate CPU-only torch are left untouched.
    """
    # Respect install.sh's backend: only "" (standalone update) or "cuda" force CUDA
    # wheels; "rocm"/"cpu"/unrecognised are deliberate.
    if _TORCH_BACKEND not in ("", "cuda"):
        return
    # An explicit unknown-family pin was applied VERBATIM at install time; leave it alone.
    if _explicit_unknown_family_torch_index_url() is not None:
        return
    # No CUDA torch on macOS; Windows torch is owned by install.ps1 (KFD bug is Linux-only).
    if IS_MACOS or IS_WINDOWS or NO_TORCH:
        return
    # Never undo a deliberate ROCm install (setup.ps1 sets this marker).
    if os.environ.get("UNSLOTH_ROCM_TORCH_INSTALLED") == "1":
        return
    # An explicit CUDA pin (headless / CI cross-install) commits to CUDA wheels and skips ALL
    # GPU probing, so it clears both the CUDA_VISIBLE_DEVICES hide gate and the NVIDIA gate below.
    _cuda_pinned = _explicit_cuda_torch_index_url() is not None
    # CUDA_VISIBLE_DEVICES="" / "-1" deliberately hides the NVIDIA GPU; never force CUDA
    # wheels over that unless a CUDA index is pinned.
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not _cuda_pinned and _cvd is not None and _cvd.strip() in ("", "-1"):
        return
    # Only NVIDIA hosts carry CUDA torch (the CUDA pin overrides this gate too).
    if not _cuda_pinned and not _has_usable_nvidia_gpu():
        return

    # Classify the installed torch: "hip" (ROCm poisoning signature), "cuda" (healthy),
    # or "cpu". A non-zero exit means torch is missing/un-importable: without a pin the
    # base install owns it, but a pinned CUDA index reinstalls it below.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch, re; "
                    "hip = getattr(torch.version, 'hip', '') or ''; "
                    "cuda = getattr(torch.version, 'cuda', '') or ''; "
                    "ver = getattr(torch, '__version__', '').lower(); "
                    "m = re.search(r'\\+(cu\\d+)', ver); "
                    "marker = 'hip' if (hip or 'rocm' in ver) else ('cuda' if cuda else 'cpu'); "
                    "print(marker + '|' + (m.group(1) if m else ''))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if probe.returncode != 0:
        # torch present but can't import. Without a pin the base install owns it; but an
        # explicit CUDA pin forces this pass (failed probe) and the base update won't
        # reinstall an already-installed torch, so reinstall from the pin (self-resolving).
        if not _cuda_pinned:
            return
        index_url = _detect_cuda_torch_index_url()
        _torch_pkg, _vision_pkg, _audio_pkg = _CUDA_TORCH_PKG_SPEC
        print(
            f"   torch cannot import but an explicit CUDA index is pinned -- reinstalling "
            f"CUDA torch from {_strip_index_url_credentials(index_url)}"
        )
        pip_install(
            "CUDA torch repair",
            "--force-reinstall",
            "--no-cache-dir",
            _torch_pkg,
            _vision_pkg,
            _audio_pkg,
            "--index-url",
            index_url,
            constrain = False,
        )
        return
    # Last non-empty line: stray sitecustomize/import-hook output must not mask the marker.
    _marker_lines = [
        line.strip() for line in probe.stdout.decode(errors = "replace").splitlines() if line.strip()
    ]
    if not _marker_lines:
        return
    _marker, _, _installed_cu = _marker_lines[-1].partition("|")
    # Reinstall CUDA torch on a ROCm build on an NVIDIA host (poisoning signature), or when a
    # CUDA index is pinned but the venv has the wrong family (CPU or a different cuXXX). A
    # healthy match, or a CPU wheel with no CUDA pin, is left alone.
    _pin = _explicit_torch_index_url()
    _pin_leaf = _torch_index_leaf(_pin) if _pin else ""
    _pinned_cuda = _is_cuda_family_leaf(_pin_leaf)
    if _marker == "hip":
        _why = "torch is a ROCm build on an NVIDIA host"
    elif _marker == "cpu" and _pinned_cuda:
        _why = "torch is a CPU build but an explicit CUDA index is pinned"
    elif _marker == "cuda" and _pinned_cuda and _installed_cu != _pin_leaf:
        # Installed cuXXX differs from the pin. An untagged build (empty) counts too:
        # the family can't be confirmed, so reinstall to enforce it (idempotent).
        _installed_desc = _installed_cu if _installed_cu else "an untagged CUDA build"
        _why = f"torch is {_installed_desc} but the pinned CUDA index is {_pin_leaf}"
    else:
        return  # healthy CUDA torch matching the pin, or a deliberate CPU wheel

    index_url = _detect_cuda_torch_index_url()
    _torch_pkg, _vision_pkg, _audio_pkg = _CUDA_TORCH_PKG_SPEC
    print(
        f"   {_why} -- reinstalling CUDA torch from {_strip_index_url_credentials(index_url)}\n"
        f"   (set UNSLOTH_TORCH_BACKEND=rocm or cpu to keep a deliberate "
        f"non-CUDA torch)"
    )
    pip_install(
        "CUDA torch repair",
        "--force-reinstall",
        "--no-cache-dir",
        _torch_pkg,
        _vision_pkg,
        _audio_pkg,
        "--index-url",
        index_url,
        constrain = False,
    )


def _ensure_cpu_torch() -> None:
    """Reinstall CPU torch when an explicit CPU pin is set but the venv has a GPU build.

    Counterpart to _ensure_cuda/rocm_torch for the explicit-CPU case (those treat a CPU
    backend as a skip, so a standalone `studio update` would ignore the authoritative CPU
    pin). Only fires for an EXPLICIT pin.
    """
    if NO_TORCH:
        return
    pin = _explicit_cpu_torch_index_url()
    if pin is None:
        return

    # Classify the installed torch family. A non-zero exit means torch is missing or
    # un-importable: the explicit CPU pin reinstalls it below.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch, re; "
                    "hip = getattr(torch.version, 'hip', '') or ''; "
                    "cuda = getattr(torch.version, 'cuda', '') or ''; "
                    "ver = getattr(torch, '__version__', '').lower(); "
                    "gpu = bool(hip) or 'rocm' in ver or bool(cuda) or bool(re.search(r'\\+cu\\d+', ver)); "
                    "print('gpu' if gpu else 'cpu')"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if probe.returncode != 0:
        # torch present but can't import. The explicit CPU pin forces this pass (failed
        # probe) and the base update won't reinstall an already-installed torch, so
        # reinstall from the pin (self-resolving, no loop).
        _torch_pkg, _vision_pkg, _audio_pkg = _CPU_TORCH_PKG_SPEC
        print(
            f"   torch cannot import but an explicit CPU index is pinned -- reinstalling "
            f"CPU torch from {_strip_index_url_credentials(pin)}"
        )
        pip_install(
            "CPU torch repair",
            "--force-reinstall",
            "--no-cache-dir",
            _torch_pkg,
            _vision_pkg,
            _audio_pkg,
            "--index-url",
            pin,
            constrain = False,
        )
        return
    _lines = [
        line.strip() for line in probe.stdout.decode(errors = "replace").splitlines() if line.strip()
    ]
    if not _lines:
        return  # unreadable -- the base install step handles a missing torch
    if _lines[-1] != "gpu":
        return  # already a CPU build

    print(
        "   torch is a GPU build but an explicit CPU index is pinned -- reinstalling "
        f"CPU torch from {_strip_index_url_credentials(pin)}"
    )
    # Pin the supported torch<2.11 family (the /cpu index now serves 2.11+, so a bare
    # trio could resolve out of range or ABI-mismatched).
    _torch_pkg, _vision_pkg, _audio_pkg = _CPU_TORCH_PKG_SPEC
    pip_install(
        "CPU torch repair",
        "--force-reinstall",
        "--no-cache-dir",
        _torch_pkg,
        _vision_pkg,
        _audio_pkg,
        "--index-url",
        pin,
        constrain = False,
    )


def _ensure_rocm_torch() -> None:
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch.

    On Linux x86_64: uses pytorch.org ROCm wheel index tags.
    On Windows: uses AMD's repo.amd.com arch-specific pip index.
    No-op on macOS, non-x86_64 Linux, NVIDIA-primary hosts, or when torch
    already links against HIP.
    Uses pip_install() to respect uv, constraints, and --python targeting.
    """
    global _rocm_windows_torch_installed
    # install.sh's resolved backend is authoritative: skip ROCm when it already chose a
    # non-ROCm family (avoids re-detecting in a subprocess that may see a different env).
    if _TORCH_BACKEND in ("cuda", "cpu"):
        return
    # An explicit unknown-family pin was applied VERBATIM at install time; leave it alone.
    if _explicit_unknown_family_torch_index_url() is not None:
        return
    # setup.ps1 sets this after installing AMD wheels; skip only when torch is actually
    # importable as ROCm (a wiped venv leaves a stale env-var that must not suppress it).
    if os.environ.get("UNSLOTH_ROCM_TORCH_INSTALLED") == "1":
        _torch_ok = False
        try:
            _probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import torch; "
                        "hip=getattr(torch.version,'hip','') or ''; "
                        "import sys; "
                        "sys.exit(0 if (hip or 'rocm' in torch.__version__.lower()) else 1)"
                    ),
                ],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 90,
            )
            _torch_ok = _probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
        if _torch_ok:
            _rocm_windows_torch_installed = True
            # ROCm torch is already installed, but the AMD Windows BNB wheel is still
            # needed (the PyPI bitsandbytes ships only CUDA DLLs, fails on ROCm).
            _install_bnb_windows_rocm()
            return
        # torch was wiped between runs; fall through to the full install path
    if IS_MACOS:
        return

    if IS_WINDOWS:
        # An explicit ROCm-family pin commits to ROCm wheels regardless of the visible
        # GPU and overrides the public per-arch index (mirrors the Linux pin handling
        # below): after a pinned setup.ps1 install fails to CPU, this repair must retry
        # the PINNED index, not repo.amd.com.
        _win_rocm_pin = _explicit_rocm_torch_index_url()
        if _win_rocm_pin is None and _has_usable_nvidia_gpu():
            return
        gfx_arch = _detect_windows_gfx_arch()
        if not gfx_arch and _win_rocm_pin is None:
            return  # no AMD GPU visible via hipinfo
        # Probe whether torch already links against HIP.
        _torch_already_rocm = False
        try:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import torch; "
                        "hip=getattr(torch.version,'hip','') or ''; "
                        "ver=torch.__version__; "
                        "print('yes' if hip or 'rocm' in ver.lower() else '')"
                    ),
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                timeout = 90,
            )
            if probe.returncode == 0 and probe.stdout.decode().strip() == "yes":
                _torch_already_rocm = True
        except (OSError, subprocess.TimeoutExpired):
            pass
        if not _torch_already_rocm:
            index_url = _win_rocm_pin or _windows_rocm_index_url(gfx_arch)
            if index_url is None:
                print(f"   No AMD Windows torch index for GPU arch {gfx_arch} -- skipping")
                return
            print(
                f"   {gfx_arch or 'pinned ROCm index'} (Windows) -- installing torch from "
                f"{_strip_index_url_credentials(index_url)}"
            )
            # Pin companions for the arches install.ps1/setup.ps1 pin (gfx120X / Strix)
            # so the per-arch index resolves an ABI-consistent trio; other arches stay bare.
            _torch_pkg, _vision_pkg, _audio_pkg = _WINDOWS_ROCM_TORCH_PKG_SPECS.get(
                gfx_arch, ("torch", "torchvision", "torchaudio")
            )
            # Nonfatal: a transient AMD-index failure must not abort the install.
            # --force-reinstall resolves before uninstalling, so a failed index keeps the
            # existing build intact; let the user retry.
            if not pip_install_try(
                f"ROCm torch (Windows, {gfx_arch or 'pinned'})",
                "--force-reinstall",
                "--index-url",
                index_url,
                _torch_pkg,
                _vision_pkg,
                _audio_pkg,
                constrain = False,
            ):
                print(
                    f"   Warning: AMD Windows ROCm torch install failed for {gfx_arch or 'the pinned index'}; "
                    "keeping the existing torch build. Re-run 'unsloth studio update' "
                    "later to retry ROCm."
                )
                return
        # ROCm torch is installed (or already was); flag it so later phases
        # do not overwrite it with the generic CPU torch wheel. BNB is a
        # separate dependency -- a BNB install failure must NOT roll back the
        # torch ROCm install.
        _rocm_windows_torch_installed = True
        # Always install AMD Windows bitsandbytes -- the PyPI wheel ships only
        # CUDA DLLs and fails on ROCm. Install even when torch was already a
        # ROCm build so `studio update` repairs a broken bnb.
        if not _install_bnb_windows_rocm():
            print(
                "   Warning: AMD Windows bitsandbytes install failed; "
                "ROCm torch is installed but bitsandbytes may need manual install"
            )
        return

    # ── Linux x86_64 only: PyTorch ROCm wheels are not published for aarch64 ──
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return
    # An explicit ROCm pin commits to ROCm wheels regardless of the visible GPU (headless / CI).
    # Mirror _ensure_cuda_torch: skip the NVIDIA/no-AMD/unreadable gates.
    _rocm_pin = _explicit_rocm_torch_index_url()
    _inferred_linux_gfx = (
        _infer_linux_amd_gfx_arch() if (_rocm_pin is None and not IS_WINDOWS) else None
    )
    if _rocm_pin is None:
        # NVIDIA takes precedence on mixed hosts (only if a GPU is usable).
        if _has_usable_nvidia_gpu():
            return
        # _has_rocm_gpu() (rocminfo / amd-smi rows) is the authoritative AMD-host signal;
        # the old /opt/rocm-or-hipcc gate broke runtime-only ROCm installs.
        if not _has_rocm_gpu() and not _inferred_linux_gfx:
            return  # no AMD GPU visible

    ver = _detect_rocm_version()
    if ver is None:
        if _rocm_pin is None and not _inferred_linux_gfx:
            print("   ROCm detected but version unreadable -- skipping torch reinstall")
            return
        # Explicit pin or inferred gfx: the index drives the install.
        ver = (0, 0)

    # Probe whether torch links against HIP, capturing the installed ROCm tag for pin-mismatch
    # detection. Emit ONE "<hip_marker>|<version>" line: marker (HIP version, "rocm" sentinel,
    # or empty for CPU/CUDA) before "|", wheel version after.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "hip=getattr(torch.version,'hip','') or ''; "
                    "ver=getattr(torch,'__version__','').lower(); "
                    # HIP version if present, else a "rocm" sentinel when only the
                    # version string flags ROCm; empty marker = CPU/CUDA torch.
                    "marker=hip if hip else ('rocm' if 'rocm' in ver else ''); "
                    "print(marker + '|' + ver)"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        probe = None
    # Last non-empty line, split on the FIRST "|" so the empty HIP field is preserved.
    _marker_lines = (
        [ln.strip() for ln in probe.stdout.decode(errors = "replace").splitlines() if ln.strip()]
        if (probe is not None and probe.returncode == 0)
        else []
    )
    _hip_marker, _sep, _installed_torch_ver = (
        _marker_lines[-1].partition("|") if _marker_lines else ("", "", "")
    )
    # A "|"-delimited line is required; without it treat HIP as absent -> reinstall.
    has_hip_torch = bool(_sep) and _hip_marker != ""

    # An explicit ROCm pin whose family differs from the installed torch must reinstall, else a
    # rocm7.2/gfx* pin over an older +rocm6.4/7.1 build never applies. Version-tag heuristic
    # only: a same-tag per-arch switch (gfx1151 -> gfx120X-all, both +rocm7.13.0) isn't detectable.
    _rocm_pin_mismatch = (
        _rocm_pin_family_mismatch(_rocm_pin, _installed_torch_ver)
        if (has_hip_torch and _rocm_pin is not None)
        else False
    )

    rocm_torch_ready = has_hip_torch and not _rocm_pin_mismatch

    # Inferred-gfx path: ROCm runtime missing but install.sh would route to AMD wheels.
    # Gated on the runtime NOT enumerating a GPU: when it can, the runtime-visible
    # arch (Strix override / generic below) decides, not cpuinfo -- a mixed Strix
    # APU + dGPU box with HIP_VISIBLE_DEVICES on the dGPU must not get APU wheels.
    # An explicit UNSLOTH_ROCM_GFX_ARCH is exempt from that runtime gate (mirrors
    # install.sh): a visible GPU with an unreadable/unsupported ROCm version must
    # not silently discard the user's named arch and leave CPU torch in place.
    _gfx_override_env = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
    if (
        _inferred_linux_gfx
        and not has_hip_torch
        and _rocm_pin is None
        and (_gfx_override_env or not _has_rocm_gpu())
    ):
        index_url = _amd_arch_index_url(_inferred_linux_gfx)
        if index_url is not None:
            _torch_pkg, _vision_pkg, _audio_pkg = _WINDOWS_ROCM_TORCH_PKG_SPECS.get(
                _inferred_linux_gfx, ("torch", "torchvision", "torchaudio")
            )
            print(
                f"\n   {_inferred_linux_gfx} inferred (ROCm runtime not visible) -- "
                f"installing torch from {_strip_index_url_credentials(index_url)}\n"
                f"   AMD wheels bundle their own ROCm runtime; install the kernel stack "
                f"for native GPU compute.\n"
            )
            pip_install(
                f"ROCm torch (inferred {_inferred_linux_gfx})",
                "--force-reinstall",
                "--no-cache-dir",
                _torch_pkg,
                _vision_pkg,
                _audio_pkg,
                "--index-url",
                index_url,
                constrain = False,
            )
            rocm_torch_ready = True

    # Strix Halo / Point (gfx1151 / gfx1150) need torch from AMD's per-gfx index
    # (2.11+rocm7.13); any generic pytorch.org rocm index lacks the fixes (ROCm 7.1
    # segfaults in _grouped_mm). See _strix_needs_amd_arch_index for the floor gate.
    _strix_override_url: "str | None" = None
    _strix_override_pkgs: "tuple[str, str, str] | None" = None
    # An explicit ROCm pin is authoritative: never auto-reroute it.
    if _strix_needs_amd_arch_index(ver) and _explicit_rocm_torch_index_url() is None:
        gfx_codes = _detect_amd_gfx_codes()
        _strix_gfx = {"gfx1151", "gfx1150"}
        _detected_strix = _strix_gfx.intersection(gfx_codes)
        if _detected_strix:
            # Runtime-visible GPU (HIP_VISIBLE_DEVICES index into gfx_codes, else first);
            # skip the override unless it's Strix.
            _runtime_gfx = gfx_codes[_pick_visible_index(len(gfx_codes))] if gfx_codes else None
            if _runtime_gfx in _strix_gfx:
                _selected_gfx = _runtime_gfx
                _amd_mirror = (
                    os.environ.get("UNSLOTH_AMD_ROCM_MIRROR") or "https://repo.amd.com/rocm/whl"
                ).rstrip("/")
                _strix_override_url = f"{_amd_mirror}/{_selected_gfx}/"
                _strix_override_pkgs = (
                    "torch>=2.11.0,<2.12.0",
                    # Pin companions to the 2.11.x range: the exclusive --index-url could
                    # otherwise resolve a build for a different torch major (ABI mismatch).
                    "torchvision>=0.26.0,<0.27.0",
                    "torchaudio>=2.11.0,<2.12.0",
                )
                print(
                    f"\n   {_selected_gfx} (AMD Strix) is the runtime target with ROCm "
                    f"{ver[0]}.{ver[1]}.\n"
                    f"   Routing torch install to AMD's arch-specific index\n"
                    f"   ({_strix_override_url}) which serves torch 2.11.0+rocm7.13.0\n"
                    f"   with AMD's gfx1150/gfx1151 fixes (more reliable than the generic\n"
                    f"   pytorch.org rocm7.2 index on ROCm 7.3+ hosts).\n"
                )
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                print(
                    f"\n   Strix GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects a non-Strix runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

    # The Strix override must fire even when has_hip_torch is True: an existing
    # torch.version.hip == "7.1" is exactly the broken combo it repairs.
    if _strix_override_url is not None and _strix_override_pkgs is not None:
        index_url = _strix_override_url
        _torch_pkg, _vision_pkg, _audio_pkg = _strix_override_pkgs
        print(
            f"   Strix arch-specific override -- installing torch from "
            f"{_strip_index_url_credentials(index_url)}"
        )
        pip_install(
            "ROCm torch (Strix arch-specific)",
            "--force-reinstall",
            "--no-cache-dir",
            _torch_pkg,
            _vision_pkg,
            _audio_pkg,
            "--index-url",
            index_url,
            constrain = False,
        )
        rocm_torch_ready = True
    elif not rocm_torch_ready:
        # Reinstall when torch is not ROCm yet, OR a ROCm build's family differs from a pin.
        # Gate on rocm_torch_ready (not has_hip_torch alone) so a successful inferred-gfx
        # install above is not overwritten by the generic pytorch.org/rocmX.Y path -- that
        # would undo the fresh-ROCm/no-/dev/kfd repair this path exists for (Codex P1 #7305).
        # Honour a ROCm pin verbatim; else pick the newest wheel tag <= host.
        _override_idx = _explicit_rocm_torch_index_url()
        if _override_idx is not None:
            index_url = _override_idx
            tag = _torch_index_leaf(index_url)
        else:
            tag = next(
                (
                    t
                    for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                    if ver >= (maj, mn)
                ),
                None,
            )
        if tag is None:
            print(f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- skipping torch reinstall")
        else:
            if _override_idx is None:
                index_url = f"{_PYTORCH_WHL_BASE}/{tag}"
            print(f"   ROCm torch -- installing from {_strip_index_url_credentials(index_url)}")
            # Only the _grouped_mm-bug gfx arches need the 2.11 spec; other gfx indexes ship
            # <2.11 and stay on the default range (matches install.ps1 / setup.ps1).
            if tag in _ROCM_GFX_TORCH211_LEAVES:
                _torch_pkg, _vision_pkg, _audio_pkg = _ROCM_TORCH_PKG_SPECS["rocm7.2"]
            elif tag.startswith("gfx"):
                _torch_pkg, _vision_pkg, _audio_pkg = _ROCM_TORCH_PKG_SPECS["_default"]
            else:
                _torch_pkg, _vision_pkg, _audio_pkg = _ROCM_TORCH_PKG_SPECS.get(
                    tag, _ROCM_TORCH_PKG_SPECS["_default"]
                )
            pip_install(
                f"ROCm torch ({tag})",
                "--force-reinstall",
                "--no-cache-dir",
                _torch_pkg,
                _vision_pkg,
                _audio_pkg,
                "--index-url",
                index_url,
                constrain = False,
            )
            rocm_torch_ready = True

    # Install bitsandbytes only when torch links against ROCm. Prefers the
    # continuous-release_main wheel (bnb PR #1887 4-bit GEMV fix), falling back
    # to PyPI when the pre-release wheel won't install. Use pip for the
    # pre-release wheel because uv rejects its filename/metadata version mismatch.
    if rocm_torch_ready:
        _bnb_url = _bnb_rocm_prerelease_url()
        _bnb_installed = False
        if _bnb_url is not None:
            _bnb_installed = pip_install_try(
                "bitsandbytes (AMD, pre-release main)",
                "--force-reinstall",
                "--no-cache-dir",
                "--no-deps",
                _bnb_url,
                constrain = False,
                force_pip = True,
            )
            if not _bnb_installed:
                print(
                    _red(
                        "   bnb pre-release install failed; falling back to PyPI "
                        "(4-bit decode will be broken on ROCm)"
                    )
                )
        if not _bnb_installed:
            pip_install(
                "bitsandbytes (AMD)",
                "--force-reinstall",
                "--no-cache-dir",
                "--no-deps",
                _BNB_ROCM_PYPI_FALLBACK,
                constrain = False,
            )


# _uv_safe_path is imported from backend.utils.uv_path_safety (shared with mlx_repair).


def _windows_hidden_subprocess_kwargs() -> dict[str, object]:
    """Return Windows-only subprocess kwargs that suppress console windows."""
    if not IS_WINDOWS:
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_factory is not None and startf_use_showwindow:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


def _infer_no_torch() -> bool:
    """Determine whether to run in no-torch (GGUF-only) mode.

    Checks UNSLOTH_NO_TORCH first. When unset, falls back to platform
    detection so Intel Macs use GGUF-only mode even when invoked from
    ``unsloth studio update`` (which does not inject the env var).
    """
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return IS_MAC_INTEL


NO_TORCH = _infer_no_torch()

# UNSLOTH_TORCH_BACKEND is set by install.sh after get_torch_index_url() ("cuda", "rocm",
# "cpu"; empty = standalone `studio update`, where we re-detect).
_TORCH_BACKEND: str = os.environ.get("UNSLOTH_TORCH_BACKEND", "").lower()
# Standalone update with an explicit pin: derive the backend from the override (classify on
# the final URL/family segment, mirroring install.sh) instead of re-probing the GPU.
if not _TORCH_BACKEND:
    _idx_override = (
        os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
        or os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    )
    _idx_leaf = _torch_index_leaf(_idx_override)
    if _idx_leaf.startswith(("rocm", "gfx")):
        _TORCH_BACKEND = "rocm"
    elif _idx_leaf == "cpu":
        _TORCH_BACKEND = "cpu"
    elif _is_cuda_family_leaf(_idx_leaf):
        # Require a digit after "cu" so /current or /custom is NOT branded CUDA (a wrong backend
        # makes _ensure_rocm_torch return early on AMD hosts). An unknown leaf keeps "" so the
        # helpers probe the GPU.
        _TORCH_BACKEND = "cuda"


def _torch_step_label(suffix: str) -> str:
    """Return a progress label like 'torch check (cuda)' using the known backend.

    Falls back to GPU detection when UNSLOTH_TORCH_BACKEND is not set (e.g.
    standalone `unsloth studio update` runs that bypass install.sh).
    """
    backend = _TORCH_BACKEND
    if not backend:
        if _has_usable_nvidia_gpu():
            backend = "cuda"
        elif _has_rocm_gpu():
            backend = "rocm"
        else:
            backend = "cpu"
    return f"torch {suffix} ({backend})"


# -- Verbosity control ----------------------------------------------------------
# By default the installer shows a minimal in-place one-line progress bar.
# Set UNSLOTH_VERBOSE=1 to restore full per-step output:
#   CLI:        unsloth studio setup --verbose
#   Linux/Mac:  UNSLOTH_VERBOSE=1 ./studio/setup.sh
#   Windows:    $env:UNSLOTH_VERBOSE="1" ; .\studio\setup.ps1
VERBOSE: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

# Progress bar state -- updated by _progress() per install step.
# Update _TOTAL if you add/remove steps in install_python_stack().
_STEP: int = 0
_TOTAL: int = 0  # set at runtime in install_python_stack() based on platform
_PROGRESS_LINE_ACTIVE: bool = False

# -- Paths --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)
LOCAL_DD_GITHUB_PLUGIN = SCRIPT_DIR / "backend" / "plugins" / "data-designer-github-repo-seed"

# mlx-lm 0.31.3 broke gemma4 / qwen3_5 loading (strict load_weights rejects the
# QK-norm q_norm/k_norm tensors); exclude just that release. See mlx-lm #1242.
MLX_LM_BAD_VERSION_EXCLUSION = "!=0.31.3"

# Apple Silicon: override mlx-vlm/mlx-lm's transformers pin (see overrides).
# _uv_safe_path: uv truncates UV_OVERRIDE at the first space too (issue #6503).
_MLX_OVERRIDES = SINGLE_ENV / "overrides-darwin-arm64.txt"
if IS_MAC_ARM and _MLX_OVERRIDES.is_file() and "UV_OVERRIDE" not in os.environ:
    os.environ["UV_OVERRIDE"] = _uv_safe_path(_MLX_OVERRIDES)

# -- Unicode-safe printing ---------------------------------------------
# On Windows the console encoding may be a legacy code page (e.g. CP1252)
# that cannot represent glyphs like ✅ or ❌. _safe_print() degrades to ASCII
# equivalents so the installer never crashes over a status glyph.

_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2705": "[OK]",  # ✅
    "\u274c": "[FAIL]",  # ❌
    "\u26a0\ufe0f": "[!]",  # ⚠️  (warning + variation selector)
    "\u26a0": "[!]",  # ⚠  (warning without variation selector)
}


def _safe_print(*args: object, **kwargs: object) -> None:
    """Drop-in print() replacement that survives non-UTF-8 consoles and detached stdout."""
    try:
        print(*args, **kwargs)
    except OSError:
        return
    except UnicodeEncodeError:
        # Stringify, then swap emoji for ASCII equivalents.
        text = " ".join(str(a) for a in args)
        for uni, ascii_alt in _UNICODE_TO_ASCII.items():
            text = text.replace(uni, ascii_alt)
        # Final fallback: replace any remaining unencodable chars.
        print(
            text.encode(sys.stdout.encoding or "ascii", errors = "replace").decode(
                sys.stdout.encoding or "ascii", errors = "replace"
            ),
            **kwargs,
        )


# ── Color support ──────────────────────────────────────────────────────
# Same logic as startup_banner: NO_COLOR disables, FORCE_COLOR or TTY enables.


def _stdout_supports_color() -> bool:
    """True if we should emit ANSI colors (matches startup_banner)."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, OSError, ValueError):
        return False
    if IS_WINDOWS:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (ImportError, AttributeError, OSError):
            return False
    return True


_HAS_COLOR = _stdout_supports_color()


# Column layout — matches setup.sh step() helper:
#   2-space indent, 15-char label (dim), then value.
_LABEL = "deps"
_COL = 15
_INDENT = 2


def _green(msg: str) -> str:
    return f"\033[38;5;108m{msg}\033[0m" if _HAS_COLOR else msg


def _cyan(msg: str) -> str:
    return f"\033[96m{msg}\033[0m" if _HAS_COLOR else msg


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m" if _HAS_COLOR else msg


def _dim(msg: str) -> str:
    return f"\033[38;5;245m{msg}\033[0m" if _HAS_COLOR else msg


def _title(msg: str) -> str:
    return f"\033[38;5;150m{msg}\033[0m" if _HAS_COLOR else msg


_RULE = "\u2500" * 52


def _step(
    label: str,
    value: str,
    color_fn = None,
) -> None:
    """Print a single step line in the column format."""
    global _PROGRESS_LINE_ACTIVE
    if color_fn is None:
        color_fn = _green
    padded = label[:_COL]
    plain_prefix_width = _INDENT + _COL
    prefix = f"{' ' * _INDENT}{_dim(padded)}{' ' * (_COL - len(padded))}"
    wrap_width = max(
        24,
        shutil.get_terminal_size((100, 20)).columns - plain_prefix_width,
    )
    lines = textwrap.wrap(
        value,
        width = wrap_width,
        break_long_words = False,
        break_on_hyphens = False,
    ) or [""]
    if _PROGRESS_LINE_ACTIVE and not VERBOSE:
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except OSError:
            pass
        _PROGRESS_LINE_ACTIVE = False
    _safe_print(f"{prefix}{color_fn(lines[0])}")
    continuation_prefix = " " * plain_prefix_width
    for line in lines[1:]:
        _safe_print(f"{continuation_prefix}{color_fn(line)}")


def _progress(label: str) -> None:
    """Print an in-place progress bar aligned to the step column layout."""
    global _STEP, _PROGRESS_LINE_ACTIVE
    _STEP += 1
    if VERBOSE:
        return
    width = 20
    filled = int(width * _STEP / _TOTAL)
    bar = "=" * filled + "-" * (width - filled)
    pad = " " * (_COL - len(_LABEL))
    end = "\n" if _STEP >= _TOTAL else ""
    try:
        sys.stdout.write(f"\r  {_dim(_LABEL)}{pad}[{bar}] {_STEP:2}/{_TOTAL}  {label:<20}{end}")
        sys.stdout.flush()
        _PROGRESS_LINE_ACTIVE = end == ""
    except OSError:
        pass


def run(
    label: str,
    cmd: list[str],
    *,
    quiet: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit."""
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
        env = _install_env_for_cmd(cmd),
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        _step("error", f"{label} failed (exit code {result.returncode})", _red)
        if result.stdout:
            # Redact before printing: the failing pip command may carry a pinned --index-url
            # with userinfo/?token= creds, so raw pip error text would leak them.
            print(_redact_install_output(result.stdout))
        sys.exit(result.returncode)
    return result


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"triton_kernels"}

# Packages to skip when torch is unavailable (Intel Mac GGUF-only mode). These
# either *are* torch extensions or have unconditional ``Requires-Dist: torch``, so
# installing them pulls torch back in. ``librosa`` is here despite not requiring
# torch: upstream ``llvmlite`` dropped its macOS x86_64 wheel (0.46.0+ ships only
# macosx_arm64 / manylinux / win_amd64), so on Intel Mac the librosa -> numba ->
# llvmlite chain triggers a from-source build that fails without LLVM 14/15 headers.
# Tracked in unslothai/unsloth#5046.
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "librosa",
}


def _select_flash_attn_version(torch_mm: str) -> str | None:
    return flash_attn_package_version(torch_mm)


def _build_flash_attn_wheel_url(env: dict[str, str]) -> str | None:
    return flash_attn_wheel_url(env)


def _print_optional_install_failure(label: str, result: subprocess.CompletedProcess[str]) -> None:
    _step("warning", f"{label} failed (exit code {result.returncode})", _cyan)
    if result.stdout:
        # Redact any pinned --index-url credentials before printing captured output.
        print(_redact_install_output(result.stdout).strip())


def _flash_attn_install_disabled() -> bool:
    return os.getenv("UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL") == "1"


def _ensure_flash_attn() -> None:
    if _flash_attn_install_disabled():
        return
    if NO_TORCH:
        return
    if has_blackwell_gpu():
        _step(
            "warning",
            "Skipping flash-attn: Blackwell GPU detected (sm_100+); no compatible prebuilt wheel",
            _cyan,
        )
        return
    if IS_WINDOWS or IS_MACOS:
        return
    if (
        subprocess.run(
            [sys.executable, "-c", "import flash_attn"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        ).returncode
        == 0
    ):
        return

    env = probe_torch_wheel_env()
    wheel_url = _build_flash_attn_wheel_url(env) if env else None
    if wheel_url and url_exists(wheel_url):
        for installer, wheel_result in install_wheel(
            wheel_url,
            python_executable = sys.executable,
            use_uv = USE_UV,
            uv_needs_system = UV_NEEDS_SYSTEM,
        ):
            if wheel_result.returncode == 0:
                return
            _print_optional_install_failure(
                f"Installing flash-attn prebuilt wheel with {installer}",
                wheel_result,
            )
        _step("warning", "Continuing without flash-attn", _cyan)
        return

    if wheel_url is None:
        _step("warning", "No compatible flash-attn prebuilt wheel found", _cyan)
    else:
        _step("warning", "No published flash-attn prebuilt wheel found", _cyan)


# -- uv bootstrap ------------------------------------------------------

USE_UV = False  # Set by _bootstrap_uv() at the start of install_python_stack()
UV_NEEDS_SYSTEM = False  # Set by _bootstrap_uv() via probe


def _bootstrap_uv() -> bool:
    """Check if uv is available and probe whether --system is needed."""
    global UV_NEEDS_SYSTEM
    if not shutil.which("uv"):
        return False
    # Probe: try a dry-run install targeting the current Python explicitly.
    # Without --python, uv can ignore the activated venv on some platforms.
    probe = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", sys.executable, "pip"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        **_windows_hidden_subprocess_kwargs(),
    )
    if probe.returncode != 0:
        # Retry with --system (some envs need it when uv can't find a venv)
        probe_sys = subprocess.run(
            ["uv", "pip", "install", "--dry-run", "--system", "pip"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            **_windows_hidden_subprocess_kwargs(),
        )
        if probe_sys.returncode != 0:
            return False  # uv is broken, fall back to pip
        UV_NEEDS_SYSTEM = True
    return True


def _filter_requirements(req: Path, skip: set[str]) -> Path:
    """Return a temp copy of a requirements file with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line for line in lines if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode = "w",
        suffix = ".txt",
        delete = False,
        encoding = "utf-8",
    )
    tmp.writelines(filtered)
    tmp.close()
    return Path(tmp.name)


def _translate_pip_args_for_uv(args: tuple[str, ...]) -> list[str]:
    """Translate pip flags to their uv equivalents."""
    translated: list[str] = []
    for arg in args:
        if arg == "--no-cache-dir":
            continue  # uv cache is fast; drop this flag
        elif arg == "--force-reinstall":
            translated.append("--reinstall")
        else:
            translated.append(arg)
    return translated


def _build_pip_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a standard pip install command.

    Strips uv-only flags like --upgrade-package that pip doesn't understand.
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--upgrade-package":
            skip_next = True  # skip the flag and its value
            continue
        cmd.append(arg)
    return cmd


def _build_uv_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a uv pip install command with translated flags."""
    cmd = ["uv", "pip", "install"]
    if UV_NEEDS_SYSTEM:
        cmd.append("--system")
    # Always pass --python so uv targets the right environment. Without it, uv
    # can ignore an activated venv and install into the system Python (seen on
    # Colab and similar).
    cmd.extend(["--python", sys.executable])
    cmd.extend(_translate_pip_args_for_uv(args))
    # Torch is pre-installed, so don't add --torch-backend by default (solver dead-ends on
    # CPU-only machines); callers can set UV_TORCH_BACKEND. Never add it to a pinned-index
    # command: uv's torch backend redirects torch to its own per-backend index, defeating the pin.
    _tb = os.environ.get("UV_TORCH_BACKEND", "")
    if _tb and not _is_pinned_index_cmd(cmd):
        cmd.append(f"--torch-backend={_tb}")
    return cmd


# uv resolves --index-url / --default-index at LOWEST priority, so an inherited UV_INDEX /
# UV_EXTRA_INDEX_URL mirror wins and a pinned torch repair silently ignores the pin.
# Neutralise these for pinned installs (as install.sh #6898 / install.ps1 / setup.ps1 do).
# UV_TORCH_BACKEND redirects torch; PIP_* matter for the pip FALLBACK; UV_CONFIG_FILE is
# stripped + UV_NO_CONFIG=1 (a discovered uv.toml outranks the CLI pin, uv 0.10).
_UV_INDEX_ENV_VARS = (
    "UV_CONFIG_FILE",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_TORCH_BACKEND",
    "UV_FIND_LINKS",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    # PIP_NO_INDEX=1 makes the pip fallback ignore ALL indexes (defeating --index-url);
    # PIP_INDEX_URL is dropped too so a stale mirror env can't outrank the pin.
    "PIP_NO_INDEX",
    "PIP_INDEX_URL",
)


def _is_pinned_index_cmd(cmd: "list[str] | tuple[str, ...]") -> bool:
    """True when the command pins an index via --index-url / --default-index."""
    return any(arg in ("--index-url", "--default-index") for arg in cmd)


def _install_env_for_cmd(cmd: "list[str]") -> "dict[str, str] | None":
    """Return an env with the uv index vars stripped for a pinned-index install.

    None (inherit env) when the command does NOT pin an index, so ordinary installs honour
    the user's mirror. For pinned commands, the uv index/backend vars are removed,
    UV_NO_CONFIG=1 set (a discovered uv.toml outranks the CLI pin), and PIP_CONFIG_FILE
    pointed at os.devnull for the pip fallback. Mirrors install.sh's gate (#6898).
    """
    if not _is_pinned_index_cmd(cmd):
        return None
    env = os.environ.copy()
    for name in _UV_INDEX_ENV_VARS:
        env.pop(name, None)
    env["UV_NO_CONFIG"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def pip_install_try(
    label: str,
    *args: str,
    constrain: bool = True,
    force_pip: bool = False,
) -> bool:
    """Like pip_install but returns False on failure instead of exiting.
    For optional installs that have a follow-up fallback.
    """
    constraint_args_pip: list[str] = []
    constraint_args_uv: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args_pip = ["-c", str(CONSTRAINTS)]
        constraint_args_uv = ["-c", _uv_safe_path(CONSTRAINTS)]

    if USE_UV and not force_pip:
        cmd = _build_uv_cmd(args) + constraint_args_uv
    else:
        cmd = _build_pip_cmd(args) + constraint_args_pip

    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        env = _install_env_for_cmd(cmd),
    )
    if result.returncode == 0:
        return True
    if VERBOSE and result.stdout:
        # pip/uv echo index URLs (credentials included) in failure output.
        print(_redact_install_output(result.stdout))
    return False


def pip_install(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    """Build and run a pip install command (uses uv when available, falls back to pip)."""
    constraint_args_pip: list[str] = []
    constraint_args_uv: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args_pip = ["-c", str(CONSTRAINTS)]
        constraint_args_uv = ["-c", _uv_safe_path(CONSTRAINTS)]

    actual_req = req
    temp_reqs: list[Path] = []
    if req is not None and IS_WINDOWS and WINDOWS_SKIP_PACKAGES:
        actual_req = _filter_requirements(req, WINDOWS_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    if actual_req is not None and NO_TORCH and NO_TORCH_SKIP_PACKAGES:
        actual_req = _filter_requirements(actual_req, NO_TORCH_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    if actual_req is not None and PLATFORM_LACKS_TORCHCODEC_WHEEL:
        # Linux aarch64 / Windows ARM64 / Intel Mac have no torchcodec
        # wheel. `unsloth studio update --local` does not pass
        # --no-torch, so the NO_TORCH filter above does not fire; do
        # the targeted skip independently so the audio extras step
        # does not take down the whole update.
        actual_req = _filter_requirements(actual_req, {"torchcodec"})
        temp_reqs.append(actual_req)
    req_args_pip: list[str] = []
    req_args_uv: list[str] = []
    if actual_req is not None:
        req_args_pip = ["-r", str(actual_req)]
        req_args_uv = ["-r", _uv_safe_path(actual_req)]

    try:
        if USE_UV:
            uv_cmd = _build_uv_cmd(args) + constraint_args_uv + req_args_uv
            if VERBOSE:
                print(f"   {label}...")
            result = subprocess.run(
                uv_cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                env = _install_env_for_cmd(uv_cmd),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                return
            print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                print(_redact_install_output(result.stdout))

        pip_cmd = _build_pip_cmd(args) + constraint_args_pip + req_args_pip
        run(f"{label} (pip)" if USE_UV else label, pip_cmd)
    finally:
        for temp_req in temp_reqs:
            temp_req.unlink(missing_ok = True)


def download_file(url: str, dest: Path) -> None:
    """Download a file using urllib (no curl dependency)."""
    urllib.request.urlretrieve(url, dest)


def patch_package_file(package_name: str, relative_path: str, url: str) -> None:
    """Download a file from url and overwrite a file inside an installed package."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output = True,
        text = True,
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        _step(_LABEL, f"package {package_name} not found, skipping patch", _red)
        return

    location = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break

    if not location:
        _step(_LABEL, f"could not locate {package_name}", _red)
        return

    dest = Path(location) / relative_path
    _step(_LABEL, f"patching {dest.name} in {package_name}...", _dim)
    download_file(url, dest)


# -- Main install sequence ---------------------------------------------


def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL
    _STEP = 0

    # install.sh sets SKIP_STUDIO_BASE=1 to avoid reinstalling base packages;
    # `studio update` does NOT, so unsloth + unsloth-zoo are reinstalled to pick
    # up new versions.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # --package installs a different package name (for testing).
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # --local overlays a local repo checkout after updating deps.
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    base_total = 11 if IS_WINDOWS else 12  # +1 for the anyio repair check (step 8b)
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    if not IS_MACOS and not NO_TORCH:
        base_total += 1  # ROCm torch check (step 2b), non-macOS
        if not IS_WINDOWS:
            base_total += 2  # flash-attn + torch final repair (step 13), Linux
    _TOTAL = (base_total - 1) if skip_base else base_total

    # 1. Try uv for faster installs (before pip upgrade -- uv venvs don't
    #    include pip by default).
    USE_UV = _bootstrap_uv()

    # 2. Ensure pip is available (uv venvs from install.sh omit pip).
    _progress("pip bootstrap")
    if USE_UV:
        run(
            "Bootstrapping pip via uv",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "pip",
            ],
        )
    else:
        # pip may not exist yet (uv-created venvs omit it). Try ensurepip,
        # then upgrade. Direct upgrade only when pip is already present.
        _has_pip = (
            subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                **_windows_hidden_subprocess_kwargs(),
            ).returncode
            == 0
        )

        if not _has_pip:
            run(
                "Bootstrapping pip via ensurepip",
                [sys.executable, "-m", "ensurepip", "--upgrade"],
            )
        else:
            run(
                "Upgrading pip",
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            )

    # macOS arm64: install MLX stack at latest (UV_OVERRIDE relaxes the
    # mlx-vlm / mlx-lm transformers pin -- set at module load).
    # Exclude mlx-lm 0.31.3 (see MLX_LM_BAD_VERSION_EXCLUSION); it broke
    # gemma4 / qwen3_5 QK-norm loading. mlx-lm #1242.
    if IS_MAC_ARM and not skip_base:
        _progress("MLX stack (Apple Silicon)")
        pip_install(
            "Installing MLX stack (mlx + mlx-lm + mlx-vlm)",
            "--no-cache-dir",
            "--upgrade",
            "mlx",
            "mlx-metal",
            f"mlx-lm{MLX_LM_BAD_VERSION_EXCLUSION}",
            "mlx-vlm",
        )

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        pass
    elif NO_TORCH:
        # No-torch update path: install unsloth + unsloth-zoo, then runtime deps,
        # both with --no-deps (PyPI metadata declares torch a hard dep; avoid it).
        _progress("base packages (no torch)")
        pip_install(
            f"Updating {package_name} + unsloth-zoo (no-torch mode)",
            "--no-cache-dir",
            "--no-deps",
            "--upgrade-package",
            package_name,
            "--upgrade-package",
            "unsloth-zoo",
            package_name,
            "unsloth-zoo",
        )
        # Resolve pydantic WITH deps so pip pins pydantic-core to the exact version
        # its metadata declares (under --no-deps pip picks the latest of each and
        # trips pydantic's _ensure_pydantic_core_version check). Deps are torch-free.
        pip_install(
            "Installing pydantic (with deps for compatible core)",
            "--no-cache-dir",
            "pydantic",
        )
        pip_install(
            "Installing no-torch runtime deps",
            "--no-cache-dir",
            "--no-deps",
            req = REQ_ROOT / "no-torch-runtime.txt",
        )
        if local_repo:
            _step(_LABEL, f"overlaying local repo (editable): {local_repo}")
            pip_install(
                "Overlaying local repo (editable)",
                "--no-cache-dir",
                "--no-deps",
                "-e",
                local_repo,
                constrain = False,
            )
            _step(_LABEL, "overlaying unsloth-zoo from git main")
            pip_install(
                "Overlaying unsloth-zoo from git main",
                "--no-cache-dir",
                "--no-deps",
                "--force-reinstall",
                "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo",
                constrain = False,
            )
    elif local_repo:
        # Local dev install: update deps from base.txt, then overlay the local
        # checkout as an editable install (--no-deps so torch is not re-resolved).
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )
        _step(_LABEL, f"overlaying local repo (editable): {local_repo}")
        pip_install(
            "Overlaying local repo (editable)",
            "--no-cache-dir",
            "--no-deps",
            "-e",
            local_repo,
            constrain = False,
        )
        _step(_LABEL, "overlaying unsloth-zoo from git main")
        pip_install(
            "Overlaying unsloth-zoo from git main",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo",
            constrain = False,
        )
    elif package_name != "unsloth":
        # Custom package name (for testing): install directly.
        _progress("base packages")
        pip_install(
            f"Installing {package_name}",
            "--no-cache-dir",
            package_name,
        )
    else:
        # Update path: upgrade only unsloth + unsloth-zoo, preserving existing
        # torch/CUDA installs. Torch is pre-installed by install.sh/setup.ps1;
        # --upgrade-package targets only base pkgs.
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )

    # 2b. AMD ROCm: reinstall torch with HIP wheels if the host has ROCm but the
    #     venv got CPU-only torch (common when pip resolves torch from PyPI).
    #     Must follow base packages so torch is present for inspection.
    if not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("check"))
        _ensure_cuda_torch()
        _ensure_rocm_torch()
        _ensure_cpu_torch()

    # Windows + AMD GPU: warn if ROCm torch was not installed (wrong Python
    # version or unknown ROCm version).
    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
        # Validate actual AMD GPU presence (not just tool existence).
        import re as _re_win

        def _win_amd_smi_has_gpu(stdout: str) -> bool:
            return bool(_re_win.search(r"(?im)^gpu\s*[:\[]\s*\d", stdout))

        _win_amd_gpu = False
        for _wcmd, _check_fn in (
            (["hipinfo"], lambda out: "gcnarchname" in out.lower()),
            (["amd-smi", "list"], _win_amd_smi_has_gpu),
        ):
            _wexe = shutil.which(_wcmd[0])
            if not _wexe:
                continue
            # Skip amd-smi on Windows w/o a HIP SDK (avoids the UAC/DiskPart
            # prompt), as _has_rocm_gpu()/_detect_amd_gfx_codes do. The only loss
            # is the best-effort "AMD GPU detected" note; ROCm-torch state below
            # comes from the install itself.
            if _wcmd[0] == "amd-smi" and not _amd_smi_allowed():
                continue
            try:
                _wr = subprocess.run(
                    [_wexe, *_wcmd[1:]],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    text = True,
                    timeout = 10,
                    env = _amd_smi_env() if _wcmd[0] == "amd-smi" else None,
                )
            except Exception:
                continue
            if _wr.returncode == 0 and _check_fn(_wr.stdout):
                _win_amd_gpu = True
                break
        if _win_amd_gpu and not _rocm_windows_torch_installed:
            _safe_print(
                _dim("  Note:"),
                "AMD GPU detected but ROCm PyTorch could not be auto-installed.",
            )
            _safe_print(
                " " * 8,
                "Manual install may be required. See: https://docs.unsloth.ai/get-started/install-and-update/amd",
            )

    # 3. Extra dependencies
    _progress("unsloth extras")
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "extras.txt",
    )

    # 3b. Extra dependencies (no-deps) -- audio model support etc.
    _progress("extra codecs")
    pip_install(
        "Installing extras (no-deps)",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "extras-no-deps.txt",
    )

    # 4. Overrides (torchao) -- force-reinstall to a version matching the venv's
    #    torch so its C++ extensions load (see _select_torchao_spec). Skipped when
    #    torch is unavailable (Intel Mac GGUF-only) and on Windows ROCm (no working
    #    build; see below).
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    elif _rocm_windows_torch_installed or _installed_torch_is_windows_rocm():
        # No working Windows ROCm torchao build: it imports an absent c10d backend
        # and crashes transformers.quantizers. Unsloth stubs it at runtime, so
        # installing it only ships a package that crashes on import -- skip it.
        _progress("dependency overrides (skipped, Windows ROCm)")
        _safe_print("   Windows ROCm -- skipping torchao (no working build; stubbed at runtime)")
    else:
        _progress("dependency overrides")
        _torch_ver = _probe_installed_torch_version()
        _torchao_spec = _select_torchao_spec(_torch_ver)
        _safe_print(f"   torch {_torch_ver or 'unknown'} detected -- installing {_torchao_spec}")
        pip_install(
            "Installing dependency overrides",
            "--force-reinstall",
            "--no-cache-dir",
            _torchao_spec,
        )

    # 5. Triton kernels (no-deps, from source). Skip on Windows and macOS
    #    (no support).
    if not IS_WINDOWS and not IS_MACOS:
        _progress("triton kernels")
        pip_install(
            "Installing triton kernels",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "triton-kernels.txt",
            constrain = False,
        )

    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress("flash-attn")
        _ensure_flash_attn()

    # # 6. Patch: override llama_cpp.py with fix from unsloth-zoo  feature/llama-cpp-windows-support branch
    # patch_package_file(
    #     "unsloth-zoo",
    #     os.path.join("unsloth_zoo", "llama_cpp.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py",
    # )

    # # 7a. Patch: override vision.py with fix from unsloth PR #4091
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "models", "vision.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/80e0108a684c882965a02a8ed851e3473c1145ab/unsloth/models/vision.py",
    # )

    # # 7b. Patch : override save.py with fix from feature/llama-cpp-windows-support
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "save.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/unsloth/save.py",
    # )

    # 8. Unsloth dependencies
    _progress("studio deps")
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

    # 8b. anyio repair (#6483)
    _progress("anyio check")
    _repair_bad_anyio()

    # 9. Data-designer dependencies
    _progress("data designer deps")
    pip_install(
        "Installing data-designer base dependencies",
        "--no-cache-dir",
        req = SINGLE_ENV / "data-designer-deps.txt",
    )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    _progress("data designer")
    pip_install(
        "Installing data-designer",
        "--no-cache-dir",
        "--no-deps",
        req = SINGLE_ENV / "data-designer.txt",
    )

    # 11. Local Data Designer seed plugins
    local_dd_plugins = [
        ("unstructured", LOCAL_DD_UNSTRUCTURED_PLUGIN),
        ("github", LOCAL_DD_GITHUB_PLUGIN),
    ]
    for _plugin_name, plugin_dir in local_dd_plugins:
        if not plugin_dir.is_dir():
            _safe_print(
                _red(
                    f"❌ Missing local plugin directory: {plugin_dir}",
                ),
            )
            return 1
    _progress("local plugin")
    for plugin_name, plugin_dir in local_dd_plugins:
        pip_install(
            f"Installing local data-designer {plugin_name} plugin",
            "--no-cache-dir",
            "--no-deps",
            str(plugin_dir),
            constrain = False,
        )

    # 12. Patch metadata for single-env compatibility
    _progress("finalizing")
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )

    # 13. Final torch repair. Steps above can pull CUDA torch from PyPI, so repair last.
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("final"))
        _ensure_cuda_torch()
        _ensure_rocm_torch()
        _ensure_cpu_torch()

    # 14. Final check (silent; third-party conflicts are expected)
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        **_windows_hidden_subprocess_kwargs(),
    )

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
