#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by setup.sh (Linux/WSL) and setup.ps1 (Windows) after the venv is
activated. Expects `pip` and `python` on PATH to point at the venv.
"""

from __future__ import annotations

import glob
import importlib.util
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

# setup.sh/setup.ps1 invoke this by path, so its directory is sys.path[0].
import install_manifest  # noqa: E402

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


# MI50 / Radeon VII (gfx906, Vega 20): rocm6.4+/7.x wheels bundle ROCm libraries
# whose Tensile kernels dropped gfx906 (rocBLAS "TensileLibrary.dat ... not read
# for gfx906", ROCm/TheRock#1844), failing at the first BLAS call. The rocm6.3
# index is the last one whose wheels run on gfx906 (torch 2.7.0 verified on MI50
# 32GB; up to 2.9 in community use). Uses the _default (<2.11) pkg specs -- the
# rocm7.2 floor of 2.11 cannot be satisfied there. Mirrors install.sh.
_GFX906_LEGACY_TAG = "rocm6.3"


def _gfx906_needs_legacy_index(ver: tuple[int, int]) -> bool:
    """True when the generic tag picked for the host ROCm version is newer than
    rocm6.3, i.e. its wheels lack gfx906 kernels and must be rerouted."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is not None and key > (6, 3)


def _runtime_target_is_gfx906() -> bool:
    """True when the runtime GPU target is gfx906 (MI50 / Radeon VII).

    An explicit UNSLOTH_ROCM_GFX_ARCH wins (mirrors _infer_linux_amd_gfx_arch /
    the display path), so a host whose rocminfo/amd-smi emit no gfx token can
    still opt in. Otherwise report gfx906 only when it is the SOLE distinct arch:
    _detect_amd_gfx_codes() de-duplicates arches, which loses per-device ordinals
    on a mixed host, so a non-gfx906 selection is never mis-identified as gfx906
    (and downgraded to rocm6.3). Mixed gfx906+dGPU hosts opt in with the env var.
    """
    # Normalize a copied HIP gcnArchName (gfx906:sramecc-:xnack- -> gfx906) so the
    # feature-flag suffix does not defeat the exact comparison (mirrors device_type.py).
    override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":")[0]
    if override:
        return override == "gfx906"
    return set(_detect_amd_gfx_codes()) == {"gfx906"}


# AMD per-arch leaves needing the torch 2.11 floor (the _grouped_mm <2.11 bug).
# Mirrors *FloorMap in install.ps1 / setup.ps1; other arches ship <2.11 and stay bare.
_ROCM_GFX_TORCH211_LEAVES: frozenset[str] = frozenset(
    {"gfx120x-all", "gfx1151", "gfx1150", "gfx1152"}
)

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
    "gfx1152": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
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
    _note(f"anyio {installed[0]}.{installed[1]} found -- reinstalling anyio<4.14...")
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
    "gfx1152": "gfx1152",  # RDNA 3.5 (Krackan Point)
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
# (bnb #1887, post-0.49.2). bnb <= 0.49.2 NaNs at decode shape on every AMD GPU;
# PyPI 0.50.0 is the first release with the fix, so the fallback below is safe.
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
# Keep in step with the amd extra in pyproject.toml and the install.sh fallback.
_BNB_ROCM_PYPI_FALLBACK = "bitsandbytes>=0.50.0"


def _bnb_rocm_prerelease_url() -> str | None:
    """Return the continuous-release_main bnb wheel URL for the current arch,
    or None when no pre-release wheel is available.
    """
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return _BNB_ROCM_PRERELEASE_URLS.get(arch)


def _bnb_rocm_arch_has_binary() -> bool:
    """False on aarch64: bitsandbytes ships no ROCm kernels there at any version.
    The PyPI 0.50.0 and continuous-release_main aarch64 wheels both carry only
    libbitsandbytes_cpu.so plus CUDA variants, so neither install path gives
    aarch64 a 4-bit backend and neither message may claim one.
    """
    arch = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch) != "aarch64"


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
            with open(path, encoding = "utf-8") as fh:
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


# APU gfx arches whose board commonly also carries a discrete Radeon. HIP often
# enumerates the APU first, so an index-0 pick installs wheels for the iGPU and the
# dGPU is never used (#7776: gfx1036 Raphael shadowing a gfx1200 RX 9060 XT). Excludes
# the Strix arches (gfx1150/1151/1152): first-class training targets, left untouched.
_SHADOWING_INTEGRATED_GFX: "frozenset[str]" = frozenset(
    {
        "gfx90c",  # Renoir / Cezanne
        "gfx1013",  # Cyan Skillfish
        "gfx1033",  # Van Gogh
        "gfx1035",  # Rembrandt
        "gfx1036",  # Raphael / Mendocino
        "gfx1103",  # Phoenix / Hawk Point
        "gfx1153",  # Krackan Point 2
    }
)


def _visible_devices_pinned() -> bool:
    """True when the user selected devices via HIP_VISIBLE_DEVICES /
    ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES.

    First-set-wins, and ANY value counts -- including "" and "-1", which select
    *no* GPU rather than meaning "unset". The ROCm runtime stores an explicitly
    empty var as " " (clr `flags.cpp`) and then picks the HIP mask whenever its
    first byte is not NUL (`paldevice.cpp` / `rocdevice.cpp`), so an empty
    HIP_VISIBLE_DEVICES shadows CUDA_VISIBLE_DEVICES instead of deferring to it;
    `parseRequestedDeviceList` then surfaces zero devices for " " and "-1", which
    ROCR states outright. Whatever the user selected is honoured verbatim, so the
    iGPU-shadowing preference below never overrides a deliberate choice. Same
    precedence as `_pick_rocm_gfx_target` in install_llama_prebuilt.py."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(_env) is not None:
            return True
    return False


def _pick_visible_index(num_tokens: int, warn: bool = True) -> int:
    """Resolve HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
    to an index into a list of length num_tokens. Returns 0 (first GPU) for
    unset, empty, '-1', UUID-style, or out-of-range values.

    First-set-wins, matching `_visible_devices_pinned()` and
    `_pick_rocm_gfx_target` in install_llama_prebuilt.py. Falling through to the
    next var on "" / "-1" would contradict the runtime: an empty HIP mask
    shadows CUDA_VISIBLE_DEVICES rather than deferring to it, and selects no GPU
    at all."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
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
            # Say so rather than silently installing for GPU 0, which on a mixed host
            # is the iGPU the user tried to mask off (setup.ps1's Resolve-VisibleGpuIndex
            # warns the same). Callers with a deduplicated list pass warn=False: there the
            # index space is arches, not devices, so out of range is normal.
            if warn:
                _safe_print(
                    f"   [WARN] {_env}={_first} is out of range ({num_tokens} GPU(s) "
                    f"detected); defaulting to GPU 0 for arch selection."
                )
        except ValueError:
            if warn:
                _safe_print(
                    f"   [WARN] {_env}={_val} is not a device index; defaulting to "
                    f"GPU 0 for arch selection. Use UNSLOTH_ROCM_GFX_ARCH to choose "
                    f"the arch directly."
                )
        return 0
    return 0


def _detect_windows_gfx_arch() -> str | None:
    """Return the gcnArchName on Windows (e.g. 'gfx1200'), or None.

    Probe order matches the PowerShell installer: env-var override, then
    hipinfo (PATH or HIP_PATH/ROCM_PATH bin), then amd-smi. Without the
    amd-smi fallback, runtime-only AMD installs lacking hipinfo on PATH
    return early and `studio update` cannot repair a CPU-only venv.

    On multi-GPU hosts, detected gfx tokens are deduplicated (preserving
    enumeration order) and HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES /
    CUDA_VISIBLE_DEVICES picks which to install for. Without a mask, the
    first GPU is used -- except when it is a shadowing iGPU leading the
    enumeration, in which case the discrete GPU is preferred (issue #7776).
    """
    # 1. Explicit override (matches PowerShell installer's env-var path).
    _override = os.environ.get("UNSLOTH_ROCM_GFX_ARCH")
    if _override and _override.strip():
        return _override.strip().lower()

    def _dedup_pick(
        tokens: list[str],
        mask_resolved: bool = False,
        warn: bool = True,
    ) -> "str | None":
        if not tokens:
            return None
        # Index into the full ordered list so the mask addresses GPU N on mixed-arch
        # hosts. mask_resolved probes already did that: hipinfo is itself a HIP
        # application, so under a mask it enumerates only the VISIBLE devices,
        # renumbered from 0. Indexing that again applies the mask twice, so
        # CUDA_VISIBLE_DEVICES=1,0 would read token 1 (the iGPU) on a host whose mask
        # put the dGPU first. amd-smi and WMI list every GPU, so those keep the index.
        _pick = tokens[0 if mask_resolved else _pick_visible_index(len(tokens), warn = warn)]
        _distinct = list(dict.fromkeys(tokens))
        if len(_distinct) < 2 or _visible_devices_pinned():
            # A pin is honoured verbatim, but say so when it selected a card with no AMD
            # Windows wheels while another enumerated GPU has them: torch silently drops
            # to CPU and the mask is the reason.
            if (
                len(_distinct) >= 2
                and _windows_rocm_index_url(_pick) is None
                and any(_windows_rocm_index_url(t) for t in _distinct)
            ):
                _usable = [t for t in _distinct if _windows_rocm_index_url(t)]
                _safe_print(
                    f"   [WARN] the pinned GPU is {_pick}, which has no AMD Windows "
                    f"wheels, so torch will be CPU-only. {', '.join(_usable)} on this "
                    f"host does have wheels -- clear the visible-device mask or point "
                    f"it at that GPU to use it."
                )
            return _pick
        # Unpinned mixed-arch host: skip a leading shadowing iGPU so the discrete card
        # decides the wheel family (#7776), and say so, since only enumeration order
        # put the APU first and the user may still want a different device.
        if _pick in _SHADOWING_INTEGRATED_GFX:
            _others = [t for t in tokens if t not in _SHADOWING_INTEGRATED_GFX]
            # Deposing the pick for a card with no Windows wheels (gfx1036 + an older
            # gfx1010) resolves to no index and drops the host to CPU, so prefer a
            # wheel-backed candidate; fall back only when the pick has no wheels either.
            _withWheels = [t for t in _others if _windows_rocm_index_url(t) is not None]
            _candidates = _withWheels or (
                [] if _windows_rocm_index_url(_pick) is not None else _others
            )
            if _candidates:
                _other = _candidates[0]
                # Not always device 1: on gfx1036,gfx1010,gfx1200 it is device 2, and
                # saying "mask 1" would expose the gfx1010 the wheels do not target.
                _other_idx = tokens.index(_other)
                _safe_print(
                    f"   multiple AMD GPUs detected ({', '.join(_distinct)}); "
                    f"installing for {_other} instead of the integrated {_pick}."
                )
                _safe_print(
                    f"   Run 'setx HIP_VISIBLE_DEVICES {_other_idx}' and reopen your "
                    f"terminal so Unsloth uses {_other} at runtime too, not just at "
                    f"install time."
                )
                return _other
        _safe_print(
            f"   multiple AMD GPUs detected ({', '.join(_distinct)}); "
            f"installing for {_pick}. Set HIP_VISIBLE_DEVICES to the GPU index "
            f"you want (then rerun) to install for a different device."
        )
        return _pick

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
            # Split on ':' like setup.ps1: "gfx90a:sramecc+:xnack-" matches neither
            # the wheel table nor the shadowing set.
            _tokens = [
                t.split(":")[0].strip().lower()
                for t in re.findall(r"(?im)^\s*gcnArchName\s*:\s*(\S+)", text)
            ]
            # hipinfo already applied the mask, so do not apply it again.
            _pick = _dedup_pick(_tokens, mask_resolved = True)
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
    #
    #    ConfigManagerErrorCode 0 is "working properly". WMI keeps listing adapters that
    #    are disabled or on a driver error, and _dedup_pick()'s shadowing skip would let
    #    one depose the working iGPU. Same filter setup.ps1 applies to $wmiGpus, incl. the
    #    AMD name match: the masks index AMD devices, so an Intel or NVIDIA adapter would
    #    shift every index away from what HIP_VISIBLE_DEVICES names.
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | Where-Object { "
                "$_.Name -match 'AMD|Radeon' } | ForEach-Object { "
                '"$($_.Name)|$($_.ConfigManagerErrorCode)" }',
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 30,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            # Lines are "<name>|<ConfigManagerErrorCode>"; a bare name (older probe
            # output, and the shape most tests use) counts as healthy.
            _all_names, _healthy = [], []
            for _line in result.stdout.decode(errors = "replace").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                _nm, _sep, _code = _line.rpartition("|")
                if not _sep:
                    _nm, _code = _line, "0"
                _nm = _nm.strip()
                # Re-apply the vendor filter rather than trusting the command string:
                # everything below treats these as AMD devices, and a stray NVIDIA or
                # Intel adapter would shift the mask index and warn at a non-AMD user.
                if not _nm or not re.search(r"AMD|Radeon", _nm, re.IGNORECASE):
                    continue
                _all_names.append(_nm)
                if _code.strip() in ("", "0"):
                    _healthy.append(_nm)
            # Drop adapters Windows flags as not working so one cannot depose a live card.
            # If that leaves NOTHING, the filter alone made the host look GPU-less and
            # returning None hands it CPU torch: code 45 ("not connected") is routine on
            # muxless laptops with a parked dGPU. Fall back to the full list -- with no
            # healthy peer there is nothing to depose.
            _names = _healthy or _all_names
            _tokens = [_a for _a in map(_gfx_arch_from_gpu_name, _names) if _a]
            # Resolve the mask over the ADAPTER list (setup.ps1's $nameIdx): a name the
            # table does not know drops out of _tokens, and indexing that shortened list
            # would name a different card.
            _sel = _pick_visible_index(len(_names)) if _names else 0
            _named = _gfx_arch_from_gpu_name(_names[_sel]) if _names else None
            # Borrow another adapter's arch only when unpinned: under a mask,
            # substituting installs for a GPU the user masked away.
            if not _named and not _visible_devices_pinned() and _tokens:
                _named = _tokens[0]
            # Repick only when every AMD adapter mapped: an unknown name may BE the
            # discrete card, so skipping the iGPU could pick the wrong one and the index
            # would count arches, not devices. warn=False since the mask was already
            # resolved and reported against the adapter list above.
            _pick = _dedup_pick(_tokens, warn = False) if len(_tokens) == len(_names) else _named
            if _pick:
                _safe_print(f"   gfx arch inferred from GPU name (WMI): {_pick}")
                return _pick
            if _names and not _pick:
                # No arch means CPU-only torch; name the adapter instead of failing silently.
                _safe_print(
                    f"   [WARN] could not map '{_names[_sel]}' to a gfx arch, so torch "
                    f"will be CPU-only. Set UNSLOTH_ROCM_GFX_ARCH to your GPU's arch "
                    f"(e.g. gfx1200) to install AMD wheels."
                )
    except Exception:
        pass
    return None


# GPU marketing-name → gfx arch table, mirroring setup.ps1's $nameArchTable.
# Most-specific first; first match wins. Covers only arches the ROCm
# prebuilts / AMD Windows torch indexes support; unknown names return None
# (callers then fall back cleanly to CPU).
_WIN_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    (r"9070|9080", "gfx1201"),  # RDNA 4 (Navi 48: Radeon RX 9070 XT / 9070 GRE / 9070 / 9080)
    (r"9060", "gfx1200"),  # RDNA 4 (Navi 44: Radeon RX 9060 XT / 9060)
    # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
    (r"8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max", "gfx1151"),
    # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
    (r"890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]", "gfx1150"),
    # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
    (r"860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33", "gfx1152"),
    # RDNA 3 desktop / workstation (Navi 31)
    (r"RX 7900|PRO W7900|PRO W7800", "gfx1100"),
    (r"RX 7800|RX 7700(?!S)|PRO W7700|PRO V710", "gfx1101"),  # Navi 32
    (r"RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500", "gfx1102"),  # Navi 33
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
    except (OSError, UnicodeDecodeError):
        return None
    if re.search(r"Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo", text, re.IGNORECASE):
        return "gfx1151"
    if re.search(r"890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]", text, re.IGNORECASE):
        return "gfx1150"
    if re.search(
        r"860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33", text, re.IGNORECASE
    ):
        return "gfx1152"
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
    except (OSError, UnicodeDecodeError):
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
                if (dev / "vendor").read_text(encoding = "utf-8").strip() != "0x1002":
                    continue
                if (dev / "class").read_text(encoding = "utf-8").strip().startswith("0x03"):
                    return True
            except (OSError, UnicodeDecodeError):
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


def _rocm_family_token(text: str) -> "str | None":
    """Family out of a 'rocm-sdk-libraries-<family>' name or requirement string."""
    _m = re.search(r"rocm[-_]sdk[-_]libraries[-_]([A-Za-z0-9][A-Za-z0-9._-]*)", text, re.IGNORECASE)
    if not _m:
        return None
    # Requirement strings carry a specifier and marker: "...-gfx120X-all==7.13.0; extra".
    return re.split(r"[=<>!~;,\[\]()\s]", _m.group(1))[0].strip().lower().replace("_", "-")


def _installed_rocm_wheel_family() -> str | None:
    """The AMD per-arch family the installed torch actually runs on, normalized (e.g.
    'gfx120x-all'), or None when nothing on disk identifies it unambiguously.

    torch.version.hip only says "some ROCm build", so it cannot tell a gfx103X wheel
    from a gfx120X one. AMD's torch requires rocm[libraries], and that extra resolves
    to the arch-specific rocm-sdk-libraries-<family> runtime, so the installed `rocm`
    meta-package names the active family. Read it there rather than by scanning for a
    rocm-sdk-libraries-* distribution: `rocm` is upgraded in place across a family
    switch, but the previous arch-specific runtime keeps its own distribution name and
    so is never uninstalled, and mistaking that orphan for the active family would
    reinstall the multi-GB stack on every update.

    None means "unknowable" -- an older wheel predating the split runtime, a pinned
    index, or two runtimes with no `rocm` to arbitrate. Callers must leave the install
    alone rather than guess.
    """
    try:
        from importlib import metadata
        for _req in metadata.requires("rocm") or []:
            _fam = _rocm_family_token(_req)
            if _fam:
                return _fam
    except Exception:
        pass
    # No `rocm` meta-package: fall back to the runtimes on disk, but only when exactly
    # one is present, since with several there is nothing left to say which is active.
    try:
        from importlib import metadata

        _found = set()
        for _dist in metadata.distributions():
            _fam = _rocm_family_token((_dist.metadata["Name"] or "").strip())
            if _fam:
                _found.add(_fam)
        if len(_found) == 1:
            return _found.pop()
    except Exception:
        return None
    return None


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


# Set right before the base unsloth install (which resolves its unconditional
# bitsandbytes dependency); read by _ensure_rocm_torch to drop a freshly pulled
# generic wheel on gfx906 while leaving a pre-existing source build untouched.
_GFX906_BNB_ABSENT_BEFORE_BASE = False


def _bitsandbytes_installed() -> bool:
    """True if bitsandbytes is importable in the target venv. Runs a fresh
    subprocess so a package installed earlier this run is seen; only checks the
    spec (does NOT import bitsandbytes)."""
    try:
        return (
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import importlib.util, sys; "
                    "sys.exit(0 if importlib.util.find_spec('bitsandbytes') else 1)",
                ],
                capture_output = True,
                timeout = 60,
            ).returncode
            == 0
        )
    except Exception:
        return False


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
        _safe_print(
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
                        with open(gpu_id_path, encoding = "utf-8") as fh:
                            gpu_id = fh.read().strip()
                    except (OSError, UnicodeDecodeError):
                        continue
                    if not gpu_id or gpu_id == "0":  # gpu_id 0 = CPU node
                        continue
                    # Require AMD vendor_id 4098 (0x1002). KFD properties files exist
                    # on every kernel exposing /sys/class/kfd, so a missing file means
                    # AMD ownership is unconfirmed -- skip the node rather than risk a
                    # false positive (e.g. NVIDIA open-driver KFD nodes lacking it).
                    props_path = os.path.join(kfd_nodes, entry, "properties")
                    try:
                        with open(props_path, encoding = "utf-8") as fh:
                            props = fh.read()
                    except (OSError, UnicodeDecodeError):
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


# Which probe answered the last _detect_amd_gfx_codes() call: only rocminfo is subject
# to a visible-device mask, so the Strix reroute needs to know. None when stubbed, which
# keeps the plain indexing behaviour.
_LAST_AMD_GFX_PROBE: "str | None" = None


def _detect_amd_gfx_codes(dedup: bool = True) -> list[str]:
    """Return the AMD gfx ISA strings visible to ROCm (e.g. ['gfx1151']).

    Probes rocminfo, then falls back to ``amd-smi list`` and ``amd-smi
    static --asic`` for runtime-only Radeon hosts that ship amd-smi but no
    rocminfo. Returns an empty list when no probe yields a gfx target.

    dedup=False keeps one entry per DEVICE instead of one per arch, which a
    caller resolving HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES needs: those
    mask values are device ordinals, so indexing a deduplicated list reads the
    wrong card whenever the host has two GPUs of the same arch. rocminfo prints
    the same token several times per agent (Name, ISA, marketing name), so split
    on agent headers first, exactly as _list_rocm_gfx_targets() does, or one GPU
    contributes several entries and every ordinal after it is wrong.

    Records the answering probe in _LAST_AMD_GFX_PROBE, since only rocminfo is
    filtered by a visible-device mask (and only by ROCR_VISIBLE_DEVICES).
    """
    global _LAST_AMD_GFX_PROBE
    _LAST_AMD_GFX_PROBE = None

    def _extract(text: str) -> list[str]:
        if dedup:
            codes = [f"gfx{c}" for c in re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())]
            return list(dict.fromkeys(codes))
        # One entry per agent section; fall back to dedup for flat output.
        _sections = re.split(
            r"(?mi)^\s*\*+\s*$\s*agent\s+\d+\s*$|\bagent\s+\d+\b|\bdevice\s*#\s*\d+\b",
            text,
        )
        if len(_sections) > 1:
            _per_device = []
            for _sec in _sections[1:]:
                _m = re.search(r"gfx[1-9][0-9a-z]{2,3}", _sec.lower())
                if _m:
                    _per_device.append(_m.group(0))
            if _per_device:
                return _per_device
        _raw = [f"gfx{c}" for c in re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())]
        return list(dict.fromkeys(_raw))

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
            _LAST_AMD_GFX_PROBE = cmd[0]
            return codes
    return []


def _first_set_visible_mask() -> "str | None":
    """Name of the visible-device variable in force, first-set-wins, or None."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(_env) is not None:
            return _env
    return None


# Set by _ensure_rocm_torch() on success; suppresses the post-install AMD warning.
_rocm_windows_torch_installed: bool = False


def _install_bnb_windows_rocm() -> bool:
    """Install AMD Windows BNB, pre-release wheel first. Returns True on success.

    The wheel's filename version (1.33.7.preview, PEP 440 1.33.7rc0) does not
    match its metadata (0.50.x.dev0). uv rejects the mismatch and still mangles
    the install under UV_SKIP_WHEEL_FILENAME_CHECK, so force plain pip, which
    performs no such check. Per the AMD install guide
    (https://unsloth.ai/docs/get-started/install/amd/amd-hackathon).

    When that URL is blocked, fall back to PyPI. Its win_amd64 wheel ships
    libbitsandbytes_rocm{714,72}.dll from 0.50.0 on, so the fallback is a real
    ROCm build; before 0.50.0 it was CUDA-only, which is why there was none.
    """
    _bnb_win_url = _BNB_ROCM_PRERELEASE_URLS.get("win_amd64")
    _ok = False
    if _bnb_win_url is not None:
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
            _safe_print(
                _red(
                    "   bnb pre-release install failed; falling back to PyPI "
                    f"{_BNB_ROCM_PYPI_FALLBACK}, which carries the ROCm 4-bit fix"
                )
            )
    if not _ok:
        _ok = pip_install_try(
            "bitsandbytes (AMD Windows)",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            _BNB_ROCM_PYPI_FALLBACK,
            constrain = False,
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


def _nvidia_smi_path() -> "str | None":
    """nvidia-smi from PATH, falling back to the canonical Linux install path a
    stripped-down PATH (systemd units, cron) can miss."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.path.isfile("/usr/bin/nvidia-smi"):
        exe = "/usr/bin/nvidia-smi"
    return exe


def _nvidia_compute_sms(exe: str) -> "list[int] | None":
    """Every GPU's sm_NN as nvidia-smi reports it, or None when the inventory is
    unreadable. One unparseable row (an "N/A" capability on a vGPU, a driver too
    old for --query-gpu=compute_cap) poisons the whole answer, so a partial
    reading can never drive a wheel decision."""
    try:
        result = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    sms: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        match = re.fullmatch(r"(\d+)\.(\d+)", value)
        if match is None:
            return None
        sms.append((int(match.group(1)) * 10) + int(match.group(2)))
    return sms or None


# PyTorch 2.11's cu126 spans sm_50-90 (Maxwell to Hopper) with no PTX above that. It is
# the fallback family, so a Kepler or Blackwell card in the mix leaves the host uncovered.
_CU126_SM_RANGE = (50, 90)


def _cuda_family_sm_range(family: str, torch_release: str = "") -> "tuple[int, int] | None":
    """Return the supported SM span for a CUDA wheel family.

    cu128 and cu129 include sm_70 only for torch 2.8 through 2.10.
    An empty release models a fresh torch 2.11 installation.
    """
    if not _is_cuda_family_leaf(family):
        return None
    number = int(family[len("cu") :])
    if number < 124:
        return (37, 90)
    if number < 128:
        return _CU126_SM_RANGE
    if number < 130:
        release = re.match(r"(\d+)\.(\d+)", torch_release)
        if release and (2, 8) <= (int(release.group(1)), int(release.group(2))) < (2, 11):
            return (70, 120)
    return (75, 120)


def _span_covers(span: "tuple[int, int]", sms: "list[int]") -> bool:
    """Whether a wheel family's sm span holds every GPU on the host."""
    return all(span[0] <= sm <= span[1] for sm in sms)


def _cap_cuda_family_for_pre_turing(family: str, exe: "str | None") -> str:
    """Use cu126 when it covers every physical GPU missed by the selected family.

    CUDA_VISIBLE_DEVICES is intentionally ignored. Non-x86_64 hosts retain the
    driver-derived family because their wheel matrices differ.
    """
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return family
    span = _cuda_family_sm_range(family)
    if span is None or exe is None:
        return family
    if span[0] <= _CU126_SM_RANGE[0]:
        return family  # nothing lower to fall back to
    floor = span[0]
    sms = _nvidia_compute_sms(exe)
    if not sms or all(sm >= floor for sm in sms):
        return family  # no GPU here sits under the family's floor
    if not _span_covers(_CU126_SM_RANGE, sms):
        _safe_print(
            f"   NVIDIA GPUs below sm_{floor} are present, but no PyTorch 2.11 CUDA "
            f"family covers this mix -- keeping {family}, which cannot use "
            + ",".join(f"sm_{sm}" for sm in sorted(set(sms)) if sm < floor)
            + ". Set UNSLOTH_TORCH_INDEX_FAMILY=cu126 to choose the other way"
        )
        return family
    _safe_print(
        f"   NVIDIA GPUs below sm_{floor} are present -- selecting cu126, because "
        f"PyTorch 2.11's {family} wheels ship no kernels for them"
    )
    return "cu126"


def _detect_cuda_torch_index_url() -> str:
    """Return the pytorch.org CUDA wheel index URL for the host's NVIDIA driver.

    Mirrors install.sh::get_torch_index_url's CUDA ladder so `studio update` repairs
    to the same wheel family a fresh install would pick. Honours the explicit
    overrides first (UNSLOTH_TORCH_INDEX_URL / _FAMILY) so a headless / CI install
    never lets the host GPU decide. Otherwise probes nvidia-smi (parsing both "CUDA
    Version:" and "CUDA UMD Version:"), defaulting to cu126 when unreadable. The
    driver version is only an upper bound, so the GPU architectures can cap the
    result at cu126 (see _cap_cuda_family_for_pre_turing).
    """
    _override_url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if _override_url:
        return _trim_index_path_slashes(_override_url)
    _override_family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    if _override_family:
        return f"{_PYTORCH_WHL_BASE}/{_override_family.strip('/')}"
    exe = _nvidia_smi_path()
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
        tag = _cap_cuda_family_for_pre_turing(tag, exe)
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


# Intel XPU wheels. Own range, not the CUDA spec above: the xpu index serves past our tested
# ceiling, and the floor is 2.6 because unsloth/models/_utils.py raises at import for an XPU
# device below it. Kept in step with install.sh by tests/sh/test_xpu_torch_spec_parity.sh.
_XPU_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.6,<2.11.0",
    "torchvision>=0.21,<0.26.0",
    "torchaudio>=2.6,<2.11.0",
)


def _explicit_xpu_torch_index_url() -> "str | None":
    """The pinned wheel index URL when it names the XPU family (leaf == xpu), else None.

    Intel support is a pin, never autodetection, so the pin is the only signal there is.
    """
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _torch_index_leaf(url) == "xpu" else None


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

    Also repairs a CUDA torch whose wheel family ships no kernels for the host's
    GPUs (a pre-Turing box that the driver-only ladder sent to cu128/cu130).
    Healthy CUDA torch and deliberate CPU-only torch are left untouched.
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
                    "print('|'.join((marker, m.group(1) if m else '', ver.split('+', 1)[0], "
                    "('cu' + cuda.replace('.', '')) if cuda else '')))"
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
        _safe_print(
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
    # marker | +cuXXX local tag | release | family from torch.version.cuda. The last is the
    # only CUDA clue an untagged wheel gives: PyPI forbids the local +cuXXX version.
    _marker, _installed_cu, _installed_release, _runtime_cu = (
        _marker_lines[-1].split("|") + ["", "", ""]
    )[:4]
    # Reinstall on a ROCm build on an NVIDIA host (poisoning signature), when a CUDA index
    # is pinned but the venv has the wrong family (CPU or a different cuXXX), or when the
    # installed family ships no kernels for this host's GPUs. A healthy match, or a CPU
    # wheel with no CUDA pin, is left alone.
    _pin = _explicit_torch_index_url()
    _pin_leaf = _torch_index_leaf(_pin) if _pin else ""
    _pinned_cuda = _is_cuda_family_leaf(_pin_leaf)
    index_url: "str | None" = None
    if _marker == "hip":
        _why = "torch is a ROCm build on an NVIDIA host"
    elif _marker == "cpu" and _pinned_cuda:
        _why = "torch is a CPU build but an explicit CUDA index is pinned"
    elif _marker == "cuda" and _pinned_cuda and _installed_cu != _pin_leaf:
        # Installed cuXXX differs from the pin. An untagged build (empty) counts too:
        # the family can't be confirmed, so reinstall to enforce it (idempotent).
        _installed_desc = _installed_cu if _installed_cu else "an untagged CUDA build"
        _why = f"torch is {_installed_desc} but the pinned CUDA index is {_pin_leaf}"
    elif _marker == "cuda" and not _pinned_cuda:
        # x86_64 only, like the cap: the spans below are the x86_64 build matrix.
        if platform.machine().lower() not in ("x86_64", "amd64"):
            return
        _family = _installed_cu or _runtime_cu
        _span = _cuda_family_sm_range(_family, _installed_release)
        if _span is None:
            return  # untagged or unrecognised build: not this check's business
        _smi = _nvidia_smi_path()
        _sms = _nvidia_compute_sms(_smi) if _smi else None
        if not _sms or _span_covers(_span, _sms):
            return  # healthy CUDA torch this host can use
        # Never trade one partial family for another, or reinstall the same one forever.
        index_url = _detect_cuda_torch_index_url()
        _target = _torch_index_leaf(index_url)
        _target_span = _cuda_family_sm_range(_target)
        if _target_span is None or not _span_covers(_target_span, _sms):
            return
        _why = (
            f"torch is {_family} but this host has GPUs outside its "
            f"sm_{_span[0]}-{_span[1]} range"
        )
    else:
        return  # healthy CUDA torch matching the pin, or a deliberate CPU wheel

    if index_url is None:
        index_url = _detect_cuda_torch_index_url()
    _torch_pkg, _vision_pkg, _audio_pkg = _CUDA_TORCH_PKG_SPEC
    _safe_print(
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


def _ensure_xpu_torch() -> None:
    """Install XPU torch when an explicit XPU pin is set but the venv has another build.

    Counterpart to _ensure_cpu_torch for Intel. `unsloth studio update` runs setup.sh, never
    install.sh, so its XPU install path is unreachable there; and an xpu leaf names no family
    the cuda/rocm helpers know, so they skip it and the CPU wheel survives the pin forever.

    Windows is excluded on purpose: setup.ps1 owns torch there and already installs the XPU
    trio itself, so acting here would fight it. macOS has no XPU at all.
    """
    if NO_TORCH or IS_MACOS or IS_WINDOWS:
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None:
        return

    # A non-zero exit means torch is missing or un-importable; the pin installs it below
    # either way. Bounded like every other probe here.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    # Flavour AND range: a migrated 2.5+xpu venv is broken, not correct, so the
                    # tag alone is not enough. Range matches _XPU_TORCH_PKG_SPEC.
                    "import torch; "
                    "ver = getattr(torch, '__version__', '').lower(); "
                    "rel = ver.split('+')[0].split('.'); "
                    "n = tuple(int(x) for x in rel[:2] if x.isdigit()); "
                    "ok = '+xpu' in ver and len(n) == 2 and (2, 6) <= n < (2, 11); "
                    "print('ok' if ok else 'repair')"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Inconclusive, so ask the disk, which answers without loading SYCL. An unsupported or
        # missing wheel does need the reinstall (the resolver keeps a too-old +xpu wheel because
        # it satisfies the base range). A supported one means the Intel DRIVER is stalled, and
        # reinstalling never fixes a driver -- it would just re-download the trio twice per
        # update, since this helper runs at two repair points.
        if _xpu_wheel_supported_on_disk():
            _safe_print(
                _red(
                    "   torch did not respond in time; the installed XPU build is supported, "
                    "so this is the Intel GPU compute driver -- update it and re-run"
                )
            )
            return
        probe = None
    _lines = (
        [
            line.strip()
            for line in probe.stdout.decode(errors = "replace").splitlines()
            if line.strip()
        ]
        if probe is not None
        else []
    )
    if probe is None:
        _why = "torch could not be probed"
    elif probe.returncode == 0:
        if not _lines:
            return  # unreadable -- the base install step handles a missing torch
        if _lines[-1] == "ok":
            return  # already the pinned family, in the supported range
        _why = "torch is not a supported XPU build"
    else:
        _why = "torch cannot import"

    _safe_print(
        f"   {_why} but an explicit XPU index is pinned -- reinstalling XPU torch from "
        f"{_strip_index_url_credentials(pin)}"
    )
    _torch_pkg, _vision_pkg, _audio_pkg = _XPU_TORCH_PKG_SPEC
    pip_install(
        "XPU torch repair",
        "--force-reinstall",
        "--no-cache-dir",
        _torch_pkg,
        _vision_pkg,
        _audio_pkg,
        "--index-url",
        pin,
        constrain = False,
    )


def _installed_torch_version_label() -> str:
    """torch's full version string, read OFF DISK without importing torch.

    Neither obvious route works here: importlib.metadata drops the local label (it reports
    2.9.1 for a 2.9.1+xpu wheel, so the flavour is gone), and `import torch` loads the SYCL
    runtime, which can block indefinitely on a wedged Intel driver. find_spec locates the
    package without executing it. Empty when torch is absent or unreadable.
    """
    try:
        # torch may have been installed earlier in THIS run, after the path finders cached
        # site-packages' directory listing.
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin:
        return ""
    try:
        text = (
            Path(spec.origin).with_name("version.py").read_text(encoding = "utf-8", errors = "replace")
        )
    except OSError:
        return ""
    match = re.search(r"""^__version__\s*=\s*['"]([^'"]*)['"]""", text, re.MULTILINE)
    return match.group(1) if match else ""


def _xpu_wheel_supported_on_disk() -> bool:
    """True when torch ON DISK is a +xpu wheel inside the supported release range.

    The same flavour-and-range test the interpreter probe runs, but off version.py, so it
    still answers when `import torch` cannot. Floor 2.6 because unsloth/models/_utils.py
    raises at import for an XPU device below it; ceiling from _XPU_TORCH_PKG_SPEC.
    """
    label = _installed_torch_version_label().lower()
    if "+xpu" not in label:
        return False
    nums = tuple(int(p) for p in label.split("+")[0].split(".")[:2] if p.isdigit())
    return len(nums) == 2 and (2, 6) <= nums < (2, 11)


def _ensure_venv_pip() -> bool:
    """Make `python -m pip` work in the target venv, bootstrapping it if needed.

    `uv venv` is created without --seed, so a fresh venv has no pip at all. Mirrors the
    bootstrap install.sh already does before its pre-release bitsandbytes wheel.
    """

    def _has_pip() -> bool:
        try:
            return (
                subprocess.run(
                    [sys.executable, "-m", "pip", "--version"],
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                    timeout = 90,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

    if _has_pip():
        return True
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 300,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if _has_pip():
        return True
    pip_install_try("pip (bootstrap)", "pip", constrain = False)
    return _has_pip()


def _ensure_xpu_triton() -> None:
    """Replace generic Triton with the XPU build torch asks for.

    Generic `triton` and torch's `pytorch-triton-xpu` / `triton-xpu` both own the top-level
    `triton` package, and resolving unsloth against a pinned +xpu torch pulls BOTH (uv reports
    pytorch-triton-xpu 3.5.0 alongside triton 3.7.1), so the CUDA-oriented build can land last
    and torch.compile then loads the wrong library on an Intel GPU.

    Lives here, not in install.sh, because install.sh runs setup.sh which runs this file: one
    copy covers the fresh install AND `unsloth studio update`, which never touches install.sh.
    Windows is excluded -- studio/setup.ps1 owns the same swap there.
    """
    if NO_TORCH or IS_MACOS or IS_WINDOWS:
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None:
        # A one-shot pin (UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh) is gone by the next plain
        # update, but its +xpu wheel is not and a dependency pass can pull generic triton back
        # in, so the INSTALLED wheel is the pin. setup.sh keys its bnb floor on the same signal.
        if "+xpu" not in _installed_torch_version_label().lower():
            return
        pin = f"{_PYTORCH_WHL_BASE}/xpu"

    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.metadata as m\n"
                    "try:\n"
                    "    reqs = m.requires('torch') or []\n"
                    "except Exception:\n"
                    "    reqs = []\n"
                    "print('SPEC=' + next((r.split(';')[0].strip() "
                    "for r in reqs if 'triton' in r.lower()), ''))\n"
                    "print('GENERIC=' + next((d.version for d in m.distributions() "
                    "if (d.metadata['Name'] or '').lower().replace('_','-') == 'triton'), ''))\n"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if probe.returncode != 0:
        return
    out = probe.stdout.decode(errors = "replace")
    spec = next((ln[5:].strip() for ln in out.splitlines() if ln.startswith("SPEC=")), "")
    generic = next((ln[8:].strip() for ln in out.splitlines() if ln.startswith("GENERIC=")), "")
    # Act only when generic triton is present AND torch asks for an XPU triton; anything else
    # means torch is not the +xpu wheel this assumes.
    if not generic or "xpu" not in spec.lower():
        return

    _safe_print(f"   replacing triton {generic} with {spec} (Intel XPU)")
    if not _ensure_venv_pip():
        _safe_print(
            _red(
                f"   no pip in the venv to fetch {spec}; generic triton {generic} left in "
                "place -- it shadows torch XPU triton, so torch.compile will not use the XPU"
            )
        )
        return

    # Fetch, THEN uninstall, THEN install from the file. The uninstall cannot go last: the shared
    # paths live in generic triton's OWN record, so removing it afterwards deletes what the XPU
    # build just wrote. Pre-fetching stops a dead mirror stranding the venv between the two
    # steps. uv has no `pip download`, hence pip here.
    tmp = tempfile.mkdtemp(prefix = "unsloth_triton_xpu_")
    try:
        _dl_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "-d",
            tmp,
            spec,
            "--index-url",
            pin,
        ]
        try:
            dl = subprocess.run(
                _dl_cmd,
                # Same scrub every other pinned install gets: PIP_NO_INDEX would ignore
                # --index-url outright, and PIP_EXTRA_INDEX_URL / PIP_FIND_LINKS are consulted in
                # addition to it, so an inherited environment could serve the wheel from
                # somewhere the pin never named.
                env = _install_env_for_cmd(_dl_cmd),
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                timeout = 900,
            )
        except (OSError, subprocess.TimeoutExpired):
            dl = None
        wheels = glob.glob(os.path.join(tmp, "*.whl"))
        # The exit code alone is not enough: no wheel on disk means nothing to install from.
        if dl is None or dl.returncode != 0 or not wheels:
            _safe_print(
                _red(
                    f"   could not fetch {spec}; generic triton {generic} left in place -- "
                    "it shadows torch XPU triton, so torch.compile will not use the XPU"
                )
            )
            return
        removed = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "triton"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
        if removed.returncode != 0:
            # A read-only or locked venv leaves generic triton REGISTERED; installing over it
            # would let a later upgrade or uninstall delete the shared files again. Change
            # nothing.
            _safe_print(
                _red(
                    f"   could not remove generic triton {generic}; leaving it in place -- it "
                    "shadows torch XPU triton, so torch.compile will not use the XPU"
                )
            )
            return
        # Past this point the venv has NO triton: the uninstall took the shared top-level files
        # with it. pip_install, not pip_install_try -- a warning would let the caller write a
        # completion manifest over a venv whose torch.compile is broken, which the next update
        # then fast-paths past (no generic distribution is left to trigger on).
        pip_install(
            "triton (Intel XPU)",
            "--force-reinstall",
            "--no-deps",
            wheels[0],
            constrain = False,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def _installed_torch_label_on_disk() -> str:
    """torch.__version__ read from torch/version.py, launching no interpreter.

    `import torch` loads the SYCL runtime and can block indefinitely on a wedged Intel driver
    -- which is the host an explicit pin is meant to rescue, so the classifier cannot depend
    on the import succeeding. find_spec locates the package without importing it.
    """
    try:
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            return ""
        text = (Path(spec.origin).parent / "version.py").read_text(
            encoding = "utf-8", errors = "replace"
        )
    except Exception:
        return ""
    m = re.search(r"^__version__ = '([^']*)'", text, re.M)
    return m.group(1).lower() if m else ""


def _is_gpu_torch_label(label: str) -> bool:
    """GPU build by local label alone. Weaker than the probe (which also reads
    torch.version.hip/cuda), so it is only used when the probe could not run."""
    return "+xpu" in label or "+rocm" in label or bool(re.search(r"\+cu\d+", label))


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
                    # '+xpu' too: an XPU wheel sets neither torch.version.cuda nor .hip, so
                    # without it a working Intel build reads as "cpu" and a CPU pin over it does
                    # nothing. Local label, since torch.version.xpu is None on some builds.
                    "gpu = bool(hip) or 'rocm' in ver or bool(cuda) or bool(re.search(r'\\+cu\\d+', ver)) or '+xpu' in ver; "
                    "print('gpu' if gpu else 'cpu')"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        # A hung import is the wedged-driver case this pin exists to rescue, so returning here
        # made the pin a no-op on exactly that host. Classify off disk instead, and only go on
        # for a GPU label: a merely slow CPU-only box must not reinstall torch every update.
        if not _is_gpu_torch_label(_installed_torch_label_on_disk()):
            return
        probe = None
    if probe is None or probe.returncode != 0:
        # torch present but can't import. The explicit CPU pin forces this pass (failed
        # probe) and the base update won't reinstall an already-installed torch, so
        # reinstall from the pin (self-resolving, no loop).
        _torch_pkg, _vision_pkg, _audio_pkg = _CPU_TORCH_PKG_SPEC
        _safe_print(
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

    _safe_print(
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
    if _TORCH_BACKEND in ("cuda", "cpu", "xpu"):
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
            # ROCm torch is already installed, but bnb still needs the ROCm build
            # (pre-release wheel, else PyPI >=0.50.0).
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
        # "Is ROCm" is not "is the RIGHT ROCm": wheels are per-family, so a host whose arch
        # now resolves elsewhere (dGPU added, or the #7776 repick) would keep the old family
        # forever. setup.ps1 force-reinstalls every run, so this only bites standalone
        # `studio update`. Act only on a family read back positively, never on a guess.
        if _torch_already_rocm and _win_rocm_pin is None:
            _want = (_GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "") or "").lower()
            _have = _installed_rocm_wheel_family()
            if _want and _have and _have != _want:
                _safe_print(
                    f"   installed ROCm torch is the {_have} build but {gfx_arch} needs "
                    f"{_want} -- reinstalling for this GPU"
                )
                _torch_already_rocm = False
        if not _torch_already_rocm:
            index_url = _win_rocm_pin or _windows_rocm_index_url(gfx_arch)
            if index_url is None:
                _safe_print(f"   No AMD Windows torch index for GPU arch {gfx_arch} -- skipping")
                return
            _safe_print(
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
                _safe_print(
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
        # Always install AMD Windows bitsandbytes, even when torch was already a
        # ROCm build, so `studio update` repairs a broken bnb.
        if not _install_bnb_windows_rocm():
            _safe_print(
                "   Warning: AMD Windows bitsandbytes install failed "
                "(pre-release and PyPI); "
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
            _safe_print("   ROCm detected but version unreadable -- skipping torch reinstall")
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
            _safe_print(
                f"   {_inferred_linux_gfx} inferred (ROCm runtime not visible) -- "
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

    # An explicit UNSLOTH_ROCM_GFX_ARCH=gfx906 pins the runtime target to the
    # MI50 / Radeon VII path; it must win over the Strix probe-order detection
    # below (a mixed Strix + MI50 host could otherwise route to gfx1151), so the
    # Strix override is skipped when it is set.
    _gfx906_arch_override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(
        ":"
    )[0] == "gfx906"

    # Strix Halo / Point (gfx1151 / gfx1150) need torch from AMD's per-gfx index
    # (2.11+rocm7.13); any generic pytorch.org rocm index lacks the fixes (ROCm 7.1
    # segfaults in _grouped_mm). See _strix_needs_amd_arch_index for the floor gate.
    _strix_override_url: "str | None" = None
    _strix_override_pkgs: "tuple[str, str, str] | None" = None
    # An explicit ROCm pin is authoritative: never auto-reroute it.
    if (
        _strix_needs_amd_arch_index(ver)
        and _explicit_rocm_torch_index_url() is None
        and not _gfx906_arch_override
    ):
        # One entry per DEVICE: the mask names a device ordinal, so a deduplicated list
        # picks the wrong card when two GPUs share an arch (gfx1100,gfx1100,gfx1151
        # would read index 1 as the Strix).
        gfx_devices = _detect_amd_gfx_codes(dedup = False)
        gfx_codes = list(dict.fromkeys(gfx_devices))
        _strix_gfx = {"gfx1151", "gfx1150", "gfx1152"}
        _detected_strix = _strix_gfx.intersection(gfx_codes)
        if _detected_strix:
            # rocminfo links HSA/ROCr directly and never loads HIP, so only
            # ROCR_VISIBLE_DEVICES filters and renumbers its output; HIP_VISIBLE_DEVICES
            # and CUDA_VISIBLE_DEVICES (a HIP-layer alias) do not touch it. Re-indexing an
            # already-ROCR-filtered list applies the mask twice, while skipping the index
            # for the other two would ignore the pin. amd-smi reads the driver and is
            # filtered by none of them, so it always indexes.
            _rocr_applied = (
                _LAST_AMD_GFX_PROBE == "rocminfo"
                and _first_set_visible_mask() == "ROCR_VISIBLE_DEVICES"
            )
            _runtime_gfx = (
                gfx_devices[0 if _rocr_applied else _pick_visible_index(len(gfx_devices))]
                if gfx_devices
                else None
            )
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
                _safe_print(
                    f"   {_selected_gfx} (AMD Strix) is the runtime target with ROCm "
                    f"{ver[0]}.{ver[1]}.\n"
                    f"   Routing torch install to AMD's arch-specific index\n"
                    f"   ({_strix_override_url}) which serves torch 2.11.0+rocm7.13.0\n"
                    f"   with AMD's gfx1150/gfx1151 fixes (more reliable than the generic\n"
                    f"   pytorch.org rocm7.2 index on ROCm 7.3+ hosts).\n"
                )
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                _safe_print(
                    f"   Strix GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects a non-Strix runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

    # gfx906 (MI50 / Radeon VII): is this the runtime GPU target? Used below to skip
    # the generic bitsandbytes wheel (no gfx906 kernels). This must hold even under
    # an explicit torch-index pin: a gfx906 host that pins rocm6.3 (without also
    # setting UNSLOTH_ROCM_GFX_ARCH) would otherwise reinstall the prebuilt bnb wheel
    # over the user's source-built gfx906 bnb. So a pin suppresses only the torch
    # reroute (_gfx906_override below), NOT the gfx906 detection for the bnb skip.
    _runtime_is_gfx906 = _gfx906_arch_override or _runtime_target_is_gfx906()
    # Reroute torch to the last gfx906-capable wheel family (rocm6.3) only when the
    # host ROCm version would otherwise pick a newer, kernel-less index -- and never
    # over an explicit pin or an active Strix reroute (the pin/Strix path installs
    # its own index; only the bnb skip must still apply on those paths).
    _gfx906_override = (
        _runtime_is_gfx906
        and _gfx906_needs_legacy_index(ver)
        and _explicit_rocm_torch_index_url() is None
        and _strix_override_url is None
    )
    if _gfx906_override:
        _safe_print(
            f"   gfx906 (MI50 / Radeon VII / Vega 20) is the runtime target with ROCm "
            f"{ver[0]}.{ver[1]}.\n"
            f"   Routing torch install to the {_GFX906_LEGACY_TAG} index: the last wheel\n"
            f"   family that runs on gfx906 (newer rocm wheels ship without gfx906 BLAS\n"
            f"   kernels and fail at first use). gfx906 is a community-maintained legacy\n"
            f"   path: 16-bit LoRA and full finetuning work; bitsandbytes 4-bit QLoRA\n"
            f"   requires a source build of bitsandbytes for gfx906 (see docs.unsloth.ai/amd).\n"
        )

    # The Strix override must fire even when has_hip_torch is True: an existing
    # torch.version.hip == "7.1" is exactly the broken combo it repairs.
    if _strix_override_url is not None and _strix_override_pkgs is not None:
        index_url = _strix_override_url
        _torch_pkg, _vision_pkg, _audio_pkg = _strix_override_pkgs
        _safe_print(
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
    # gfx906 fires even when has_hip_torch is True: a +rocm7.x build IS the broken
    # combo it repairs. A torch already on rocm6.3 wheels is left alone (the tag
    # check below is False, and rocm_torch_ready is already True from has_hip_torch,
    # so the generic fallback is skipped).
    elif _gfx906_override and _GFX906_LEGACY_TAG not in _installed_torch_ver:
        index_url = f"{_PYTORCH_WHL_BASE}/{_GFX906_LEGACY_TAG}"
        _torch_pkg, _vision_pkg, _audio_pkg = _ROCM_TORCH_PKG_SPECS["_default"]
        _safe_print(
            f"   gfx906 legacy override -- installing torch from "
            f"{_strip_index_url_credentials(index_url)}"
        )
        pip_install(
            f"ROCm torch (gfx906, {_GFX906_LEGACY_TAG})",
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
            _safe_print(
                f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- skipping torch reinstall"
            )
        else:
            if _override_idx is None:
                index_url = f"{_PYTORCH_WHL_BASE}/{tag}"
            _safe_print(
                f"   ROCm torch -- installing from {_strip_index_url_credentials(index_url)}"
            )
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

    # gfx906 has no prebuilt bitsandbytes: the continuous-release/PyPI wheels ship
    # no gfx906 kernels, and force-reinstalling them would clobber a user's
    # source-built bnb (the only 4-bit path on this arch) on every `studio update`.
    # Skip the auto-install and leave whatever bnb is present.
    if rocm_torch_ready and _runtime_is_gfx906:
        _safe_print(
            _dim(
                "   gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels). "
                "Build bitsandbytes from source for 4-bit QLoRA -- "
                "see docs.unsloth.ai/get-started/install-and-update/amd."
            )
        )
        # The base install resolves unsloth's unconditional bitsandbytes dep to a
        # generic CUDA wheel with no gfx906 kernels ("invalid device function" at
        # 4-bit use). Drop it if this run pulled it in; a pre-existing source build
        # (present before the base install) is left untouched.
        if _GFX906_BNB_ABSENT_BEFORE_BASE and _bitsandbytes_installed():
            _safe_print(_dim("   gfx906: removing generic bitsandbytes pulled in as a dependency"))
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"],
                capture_output = True,
            )
    # Install bitsandbytes only when torch links against ROCm. Prefers the
    # continuous-release_main wheel (bnb PR #1887 4-bit GEMV fix), falling back
    # to PyPI when the pre-release wheel won't install. Use pip for the
    # pre-release wheel because uv rejects its filename/metadata version mismatch.
    elif rocm_torch_ready:
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
                _fallback_note = (
                    ", which carries the ROCm 4-bit fix" if _bnb_rocm_arch_has_binary() else ""
                )
                _safe_print(
                    _red(
                        "   bnb pre-release install failed; falling back to PyPI "
                        f"{_BNB_ROCM_PYPI_FALLBACK}{_fallback_note}"
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
        if not _bnb_rocm_arch_has_binary():
            _safe_print(
                _red(
                    "   aarch64: bitsandbytes ships no ROCm kernels on this arch; "
                    "4-bit QLoRA needs a source build -- "
                    "https://docs.unsloth.ai/get-started/install-and-update/amd"
                )
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

    Precedence: UNSLOTH_NO_TORCH (install.sh / install.ps1 export it, "false"
    included, so an explicit value always wins) -> the mode recorded in this
    venv's install manifest -> platform detection, so Intel Macs use GGUF-only
    mode even when invoked from ``unsloth studio update``.

    The manifest tier is what keeps ``unsloth studio update`` in no-torch mode:
    it injects no env var, so without it every update reinstalls torch into a
    GGUF-only venv. Note setup.ps1 resolves the mode itself and re-exports
    UNSLOTH_NO_TORCH, because it drops the manifest before invoking this script.

    An empty value counts as unset: PowerShell cannot represent a set-but-empty
    variable (assigning "" deletes it), so the two must mean the same thing here.

    Evaluated at import, which is before install_python_stack() drops the
    manifest. Do not defer this call into main().
    """
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None and env.strip():
        return env.strip().lower() in install_manifest.NO_TORCH_TRUTHY
    recorded = install_manifest.recorded_no_torch()
    if recorded is not None:
        return recorded
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
    elif _idx_leaf == "xpu":
        # Without this the leaf falls through as unknown and the standalone update never acts
        # on an authoritative XPU pin -- see _ensure_xpu_torch.
        _TORCH_BACKEND = "xpu"
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
    """Drop-in print() replacement that survives non-UTF-8 consoles and detached stdout.

    Closes an open progress bar line first: _progress() leaves the cursor mid-line,
    so centralising it here (nothing calls print() directly -- see
    test_no_bare_print_calls) keeps every message off the bar.
    """
    _end_progress_line()
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


def _end_progress_line() -> None:
    """Close an in-place progress bar line so the next print starts on its own line."""
    global _PROGRESS_LINE_ACTIVE
    if not _PROGRESS_LINE_ACTIVE or VERBOSE:
        return
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    # Every _safe_print() lands here: a detached (None) or closed stdout must not
    # take down a message bound for stderr.
    except (AttributeError, OSError, ValueError):
        pass
    _PROGRESS_LINE_ACTIVE = False


def _note(message: str, color_fn = None) -> None:
    """Print a detail line under the current step, aligned to the value column."""
    if color_fn is None:
        color_fn = _dim
    # Verbose prints no bar and no step line, so there is no value column to align to.
    prefix = "   " if VERBOSE else " " * (_INDENT + _COL)
    wrap_width = max(24, shutil.get_terminal_size((100, 20)).columns - len(prefix))
    lines = textwrap.wrap(
        message,
        width = wrap_width,
        break_long_words = False,
        break_on_hyphens = False,
    ) or [""]
    for line in lines:
        _safe_print(f"{prefix}{color_fn(line)}")


def _step(
    label: str,
    value: str,
    color_fn = None,
) -> None:
    """Print a single step line in the column format."""
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
            _safe_print(_redact_install_output(result.stdout))
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
        _safe_print(_redact_install_output(result.stdout).strip())


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
        # As pip_install below: `nobuild` only catches a build that reaches the log.
        if VERBOSE and result.stdout:
            _safe_print(_redact_install_output(result.stdout))
        return True
    if VERBOSE and result.stdout:
        # pip/uv echo index URLs (credentials included) in failure output.
        _safe_print(_redact_install_output(result.stdout))
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
                _safe_print(f"   {label}...")
            result = subprocess.run(
                uv_cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                env = _install_env_for_cmd(uv_cmd),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                # Echo success under UNSLOTH_VERBOSE, as install.sh's run_install_cmd
                # does. Without it the dependency phase never reached the log that
                # clean-machine-assert.sh's `nobuild` greps for uv's
                # "Building <pkg>==<ver>", so a source build here -- the studio.txt
                # install, where sdist-only dependencies show up -- reported
                # "built: none" and stayed green. Redacted: uv echoes credentialed URLs.
                if VERBOSE and result.stdout:
                    _safe_print(_redact_install_output(result.stdout))
                return
            _safe_print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                _safe_print(_redact_install_output(result.stdout))

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


def _has_working_git() -> bool:
    """Match install.sh's _has_working_git: on PATH *and* actually runnable.

    A present-but-broken git (a bare xcrun shim) counts as missing there too. Testing
    only shutil.which disagreed, so the installer promised to skip the git+https triton
    requirement and then tried to fetch it anyway.
    """
    exe = shutil.which("git")
    if exe is None:
        return False
    try:
        return (
            subprocess.run(
                [exe, "--version"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 30,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL, _PROGRESS_LINE_ACTIVE
    _STEP = 0
    # An aborted earlier run leaves it set, and every _safe_print() consumes it --
    # the first message would get a stray newline.
    _PROGRESS_LINE_ACTIVE = False

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

    # Drop it up front: a missing manifest is what tells the CLI, setup.sh and
    # the preflight that an interrupted run left the venv half-built. Stop if it
    # survives rather than mutate the venv behind a marker that still verifies.
    if not install_manifest.remove_manifest():
        _safe_print(
            f"error: could not remove the stale {install_manifest.MANIFEST_NAME} in "
            f"{install_manifest.venv_root()}; refusing to install behind a marker "
            "that would still report this venv as complete",
            file = sys.stderr,
        )
        return 1

    # The manifest just went away, so record the mode in a marker that survives a
    # pass killed part-way. Otherwise the next update sees neither, reads the
    # absent torch as a stale venv, and tries to delete the running environment.
    install_manifest.set_no_torch_marker(NO_TORCH)

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

    # gfx906: the base install below resolves unsloth's unconditional bitsandbytes
    # dep to a generic CUDA wheel (no gfx906 kernels). Record bnb's presence now so
    # _ensure_rocm_torch can drop a freshly pulled wheel while keeping a source build.
    global _GFX906_BNB_ABSENT_BEFORE_BASE
    if not skip_base:
        _GFX906_BNB_ABSENT_BEFORE_BASE = not _bitsandbytes_installed()

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
        _ensure_xpu_torch()
        _ensure_cpu_torch()
        # Last, after every torch migration: the swap keys off the installed +xpu label, so a
        # CPU pin over an XPU venv would leave XPU triton under a CPU torch.
        _ensure_xpu_triton()

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
            _note(
                "AMD GPU detected but ROCm PyTorch could not be auto-installed. "
                "Manual install may be required. See: "
                "https://docs.unsloth.ai/get-started/install-and-update/amd"
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
        _note("Windows ROCm -- skipping torchao (no working build; stubbed at runtime)")
    else:
        _progress("dependency overrides")
        _torch_ver = _probe_installed_torch_version()
        _torchao_spec = _select_torchao_spec(_torch_ver)
        _note(f"torch {_torch_ver or 'unknown'} detected -- installing {_torchao_spec}")
        pip_install(
            "Installing dependency overrides",
            "--force-reinstall",
            "--no-cache-dir",
            _torchao_spec,
        )

    # 5. Triton kernels (no-deps, from source). Skipped on Windows/macOS (no support)
    #    and without git (the requirement is a git+https URL); a training speedup
    #    only, so warn rather than fail the install.
    if not IS_WINDOWS and not IS_MACOS:
        if not _has_working_git():
            _progress("triton kernels (skipped, no git)")
            _note("no working git -- skipping triton kernels (training speedup only)")
        else:
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
            _note(f"❌ Missing local plugin directory: {plugin_dir}", _red)
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
        _ensure_xpu_torch()
        _ensure_cpu_torch()
        # Last, after every torch migration: the swap keys off the installed +xpu label, so a
        # CPU pin over an XPU venv would leave XPU triton under a CPU torch.
        _ensure_xpu_triton()

    # 14. Final check (silent; third-party conflicts are expected)
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        **_windows_hidden_subprocess_kwargs(),
    )

    # 15. Record success. Written last so an earlier kill leaves none. Exiting 0
    # without it reports a finished install every later check calls unfinished.
    if (
        install_manifest.write_manifest(
            req_root = REQ_ROOT,
            steps_total = _TOTAL,
            package_name = package_name,
            no_torch = NO_TORCH,
        )
        is None
    ):
        _safe_print(
            f"error: could not write {install_manifest.MANIFEST_NAME} to "
            f"{install_manifest.venv_root()}",
            file = sys.stderr,
        )
        return 1

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
