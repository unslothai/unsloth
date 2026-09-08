#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer, run by setup.sh / setup.ps1 inside the activated venv."""

from __future__ import annotations

import ast
import functools
import glob
import importlib
import importlib.util
import json
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

_STUDIO_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _STUDIO_DIR / "backend"
for _dir in (_BACKEND_DIR, _STUDIO_DIR):
    # -P / PYTHONSAFEPATH drops the script directory, so do not rely on sys.path[0].
    if str(_dir) not in sys.path:
        sys.path.insert(1, str(_dir))

# setup.sh/setup.ps1 invoke this by path, so its directory is sys.path[0].
import install_manifest  # noqa: E402

from backend.utils.wheel_utils import (
    flash_attn_package_version,
    flash_attn_wheel_url,
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

# amd-smi auto-elevates on Windows (UAC/DiskPart); RunAsInvoker keeps probes un-elevated.
if IS_WINDOWS:
    os.environ.setdefault("__COMPAT_LAYER", "RunAsInvoker")
# Platforms that HAVE a torchcodec wheel (manylinux_2_28 x86_64/aarch64, macosx arm64,
# win_amd64); elsewhere the audio extras are filtered out or extras-no-deps fails. Stated
# as the allowlist, like the pyproject markers this mirrors: the denylist spelling missed
# every Linux arch past Arm. aarch64 arrived at 0.11.0, which is the row torch 2.11 selects,
# so omitting it would deny audio to the hosts that row serves. Whether a platform ever
# published, not when -- that is _torchcodec_platform_floor's job.
_PLATFORM_HAS_TORCHCODEC_WHEEL = (
    (IS_LINUX and platform.machine() in {"x86_64", "AMD64", "aarch64", "arm64"})
    or (IS_WINDOWS and platform.machine().lower() in {"amd64", "x86_64"})
    or IS_MAC_ARM
)
PLATFORM_LACKS_TORCHCODEC_WHEEL = not _PLATFORM_HAS_TORCHCODEC_WHEEL


def _is_windows_arm64() -> bool:
    """Windows on ARM by MACHINE arch: platform.machine() says AMD64 under emulated x64 Python, and PROCESSOR_ARCHITEW6432 is ARM64 only then. Mirrors Get-HostMachineArch in install.ps1 / setup.ps1."""
    if not IS_WINDOWS:
        return False
    return any(
        (value or "").strip().lower() in {"arm64", "aarch64"}
        for value in (
            os.environ.get("PROCESSOR_ARCHITEW6432"),
            os.environ.get("PROCESSOR_ARCHITECTURE"),
            platform.machine(),
        )
    )


# ── ROCm / AMD GPU support ─────────────────────────────────────────────────────
_ROCM_TORCH_INDEX: dict[tuple[int, int], str] = {
    (7, 2): "rocm7.2",
    (7, 1): "rocm7.1",
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
    """True when the host ROCm's generic index sits below the AMD per-arch floor (7.13), so gfx1150/1151 need repo.amd.com wheels; an unreadable version resolves no generic index at all. Mirrors install.sh _rocm_leaf_below."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is None or key < _ROCM_ARCH_INDEX_FLOOR


# MI50 / Radeon VII (gfx906): rocm6.4+/7.x wheels dropped gfx906 Tensile kernels (ROCm/TheRock#1844), so the first BLAS call fails; rocm6.3 is the last working index, on the _default (<2.11) specs.
_GFX906_LEGACY_TAG = "rocm6.3"


def _gfx906_needs_legacy_index(ver: tuple[int, int]) -> bool:
    """True when the generic tag for this ROCm version is newer than rocm6.3, whose wheels are the last carrying gfx906 kernels."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is not None and key > (6, 3)


def _runtime_target_is_gfx906() -> bool:
    """True when the runtime GPU target is gfx906. UNSLOTH_ROCM_GFX_ARCH wins; otherwise only when gfx906 is the SOLE arch, since _detect_amd_gfx_codes() de-duplicates and would mis-downgrade a mixed host."""
    # A mask hiding every GPU must not buy a force-reinstall onto an OLDER tag for the card the user hid; no probe below is mask-filtered.
    if _visible_masks_select_no_gpu():
        return False
    # Normalize a copied gcnArchName (gfx906:sramecc-:xnack- -> gfx906) so the suffix does not defeat the exact match.
    override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":")[0]
    if override:
        return override == "gfx906"
    # Unmasked, the sole-arch question is about the machine: ROCR_VISIBLE_DEVICES filters rocminfo first, so a mask naming the MI50 would unlock the downgrade this rule withholds.
    return set(_detect_amd_gfx_codes(ignore_visible_masks = True)) == {"gfx906"}


def _torch_below_211(installed_ver: str) -> bool:
    """True when an installed torch version reads as below 2.11; unreadable reads as NOT below, since this gates a multi-GB --force-reinstall."""
    _m = re.match(r"\s*(\d+)\.(\d+)", installed_ver or "")
    return bool(_m) and (int(_m.group(1)), int(_m.group(2))) < (2, 11)


# AMD per-arch leaves needing the torch 2.11 floor (the _grouped_mm <2.11 bug). Mirrors *FloorMap in install.ps1 / setup.ps1.
_ROCM_GFX_TORCH211_LEAVES: frozenset[str] = frozenset(
    {"gfx120x-all", "gfx1151", "gfx1150", "gfx1152"}
)

# rocmX.Y indexes KNOWN to ship torch 2.11; never floor an unknown newer rocm.
_ROCM_KNOWN_TORCH211_VERSIONS: frozenset[tuple[int, int]] = frozenset({(7, 2)})

# Per-tag repair specs; must land on the same wheels a fresh install.sh run does.
_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
    # Floored at 2.11 (the _grouped_mm bug), matching install.sh's rocm7.2|gfx* case.
    "rocm7.2": (
        "torch>=2.11.0,<2.12.0",
        "torchvision>=0.26.0,<0.27.0",
        "torchaudio>=2.11.0,<2.12.0",
    ),
    # rocm7.1 also serves 2.11, so a <2.11 cap would force-reinstall 2.10 over it.
    "rocm7.1": (
        "torch>=2.4,<2.12.0",
        "torchvision>=0.19,<0.27.0",
        "torchaudio>=2.4,<2.12.0",
    ),
    # rocm7.0 and earlier top out below 2.11, so the ceiling stays literal.
    "_default": (
        "torch>=2.4,<2.11.0",
        "torchvision>=0.19,<0.26.0",
        "torchaudio>=2.4,<2.11.0",
    ),
}
# Windows AMD per-arch pins for repo.amd.com, stopping an ABI-mismatched companion.
_WINDOWS_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
    "gfx1201": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1200": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1151": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1150": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1152": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
}
# Bound companion versions for ABI compatibility while retaining older per-arch mirror builds.
_ROCM_ARCH_INDEX_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

_PYTORCH_WHL_BASE = (
    os.environ.get("UNSLOTH_PYTORCH_MIRROR") or "https://download.pytorch.org/whl"
).rstrip("/")


def _strip_index_url_credentials(url: str) -> str:
    """Strip userinfo and query/fragment from a wheel index URL so printed output cannot leak credentials. MUST match install.sh / setup.ps1 / install.ps1."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    rest = rest.split("?", 1)[0].split("#", 1)[0]
    authority, slash, tail = rest.partition("/")
    host = authority.rpartition("@")[2]
    return f"{scheme}://{host}{slash}{tail}"


_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s`]+@")
_URL_QUERY_VALUE_RE = re.compile(r"([?&][^=\s&`]+)=[^&#\s`]+")
# URL-anchored so a bare "#..." (a shell comment in tool output) is never touched.
_URL_FRAGMENT_RE = re.compile(r"(https?://[^\s`#]+)#[^\s`]+")


def _redact_install_output(output: "bytes | str") -> str:
    """Redact index-URL credentials from captured output: uv/pip failure text embeds the failing --index-url verbatim. MUST match install.sh / setup.ps1 / install.ps1."""
    text = output.decode(errors = "replace") if isinstance(output, bytes) else output
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _URL_QUERY_VALUE_RE.sub(r"\1=<redacted>", text)
    return _URL_FRAGMENT_RE.sub(r"\1#<redacted>", text)


def _trim_index_path_slashes(url: str) -> str:
    """Trim trailing slashes from the URL PATH only: a whole-URL rstrip corrupts a token ending in "/", a single-slash strip leaves .../cu128// as an empty leaf. MUST match install.sh / setup.ps1 / install.ps1."""
    value = url.strip()
    match = re.fullmatch(r"([^?#]*)([?#].*)?", value)
    if match is None:
        return value.rstrip("/")
    return match.group(1).rstrip("/") + (match.group(2) or "")


def _torch_index_leaf(url: str) -> str:
    """Final URL path segment, lowercased, query/fragment stripped first, so .../cu128?token=x still classifies as cu128. Classification only. MUST match install.sh / setup.ps1 / install.ps1."""
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.rstrip("/").rsplit("/", 1)[-1].lower()


# CUDA repair specs (see _ensure_cuda_torch); companions pinned against an ABI mismatch.
_CUDA_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

# CPU repair specs (see _ensure_cpu_torch); the /cpu index also serves newer torch.
_CPU_TORCH_PKG_SPEC: tuple[str, str, str] = _CUDA_TORCH_PKG_SPEC

# Byte-identical to install.ps1's non-XPU $_fix*Spec scalars, NOT _CUDA_TORCH_PKG_SPEC: `studio update` must repair to the same wheels install.ps1 does.
_TORCH_FLAVOR_REPAIR_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

# torchao's cpp is built for ONE torch release AND CUDA major. Either mismatch costs the
# kernels, never the import: torchao/__init__.py has caught the dlopen failure since 0.12 and
# import_fixes.py filters that warning. Match torchao to the installed torch (pytorch/ao#2919):
#   2.9.x            -> 0.14.0
#   2.10.x, CUDA<=12 -> 0.16.0 (cpp built for 2.10, loads via the CUDA-12 wheel)
#   2.10.x, CUDA>=13 -> 0.17.0 (cu130: 0.16.0's CUDA-12 cpp crashes on load; 0.17.0
#                       targets torch 2.11 so its cpp is cleanly skipped, not crashed)
#   2.11.x           -> 0.17.0 (cpp built for 2.11)
#   2.12.x and up    -> 0.18.0 (dropped torch <2.11; release CI pinned to 2.13)
# Unknown/older torch keeps the conservative default.
#
# The version alone is not enough: torchao ships per accelerator under /whl/<tag>, and PyPI's
# single default tracks whatever major PyTorch currently ships (13 as of 0.18.0), so it cannot
# be treated as a fixed fallback major. The caller pins the index to the resident torch.
_TORCHAO_DEFAULT_SPEC = "torchao==0.14.0"
_TORCHAO_TORCH_210_SPEC = "torchao==0.16.0"
_TORCHAO_TORCH_210_CUDA13_SPEC = "torchao==0.17.0"
_TORCHAO_TORCH_211_SPEC = "torchao==0.17.0"
_TORCHAO_TORCH_212_PLUS_SPEC = "torchao==0.18.0"
# torch 2.10 built against CUDA >= this major can't load 0.16.0's CUDA-12 cpp.
_TORCHAO_CUDA13_MIN_MAJOR = 13


def _cuda_major_from_torch_version(torch_version: str) -> int | None:
    """CUDA major from a torch local version tag ('2.10.0+cu130' -> 13); None for rocm/cpu/tagless builds."""
    local = str(torch_version).split("+", 1)
    if len(local) < 2 or not local[1].startswith("cu"):
        return None
    digits = re.sub(r"[^0-9].*", "", local[1][2:])
    if not digits:
        return None
    return int(digits) // 10


def _select_torchao_spec(torch_version: str | None) -> str:
    """Map an installed torch version to the torchao spec whose cpp matches; _TORCHAO_DEFAULT_SPEC when unknown. Pure function."""
    if not torch_version:
        return _TORCHAO_DEFAULT_SPEC
    release = str(torch_version).split("+", 1)[0]
    parts = release.split(".")
    try:
        minor_str = re.sub(r"[^0-9].*", "", parts[1]) if len(parts) > 1 else ""
        major, minor = int(parts[0]), int(minor_str)
    except (IndexError, ValueError):
        return _TORCHAO_DEFAULT_SPEC
    if major != 2:
        return _TORCHAO_DEFAULT_SPEC
    if minor >= 12:
        return _TORCHAO_TORCH_212_PLUS_SPEC  # newest known build; covers 2.12+
    if minor == 11:
        return _TORCHAO_TORCH_211_SPEC  # 0.17.0's cpp is built for exactly this minor
    if minor == 10:
        # cu130+ can't load 0.16.0's CUDA-12 cpp; use 0.17.0 (cpp skipped, not crashed).
        cuda_major = _cuda_major_from_torch_version(str(torch_version))
        if cuda_major is not None and cuda_major >= _TORCHAO_CUDA13_MIN_MAJOR:
            return _TORCHAO_TORCH_210_CUDA13_SPEC
        return _TORCHAO_TORCH_210_SPEC
    return _TORCHAO_DEFAULT_SPEC


# torchcodec up to 0.11 is built against one torch minor and declares no
# `Requires-Dist: torch`, so pip cannot catch a mismatch. 0.12+ is ABI-stable against torch
# >=2.11, hence the open floor. Mirrors pyproject's audio-torch2xx and import_fixes.
_TORCHCODEC_DEFAULT_SPEC = "torchcodec>=0.10.0,<0.11.0"
_TORCHCODEC_ABI_STABLE_SPEC = "torchcodec>=0.12.0"
_TORCHCODEC_TORCH_SPECS: dict[int, str] = {
    12: _TORCHCODEC_ABI_STABLE_SPEC,
    11: "torchcodec>=0.11.0,<0.12.0",
    10: "torchcodec>=0.10.0,<0.11.0",
    9: "torchcodec>=0.8.0,<0.10.0",
    8: "torchcodec>=0.6.0,<0.8.0",
    7: "torchcodec>=0.3.0,<0.6.0",
    6: "torchcodec>=0.2.0,<0.3.0",
    5: "torchcodec>=0.1.0,<0.2.0",
}
_TORCHCODEC_MAX_KNOWN_MINOR = max(_TORCHCODEC_TORCH_SPECS)

# Not every platform was published from 0.1. Read off the live PyPI index:
#
#   win_amd64            first at 0.7.0   (0.1 .. 0.6 are Linux/macOS only)
#   manylinux aarch64    first at 0.11.0
#   manylinux x86_64     from the start
#   macosx arm64         from the start, min macOS moves 11.0 -> 14.0 at 0.12.0
#
# A window whose releases have no wheel here aborts the install rather than skipping audio,
# and it is reachable: cu118 tops out at torch 2.7, so an older-driver Windows box selects
# `>=0.3.0,<0.6.0`, and no release in that window ships win_amd64.
_TORCHCODEC_MIN_WHEEL_VERSION = (0, 1, 0)
_TORCHCODEC_MIN_WHEEL_WINDOWS = (0, 7, 0)
_TORCHCODEC_MIN_WHEEL_LINUX_AARCH64 = (0, 11, 0)
# 0.12.0 raised its macOS floor; a Mac below this cannot use the ABI-stable line at all.
_TORCHCODEC_MACOS_14_ONLY_FROM = (0, 12, 0)


def _torchcodec_platform_floor() -> "tuple[int, int, int] | None":
    """Earliest torchcodec release with a wheel for THIS host, or None when there is none."""
    machine = platform.machine().lower()
    if IS_WINDOWS:
        return _TORCHCODEC_MIN_WHEEL_WINDOWS if machine in {"amd64", "x86_64"} else None
    if IS_LINUX:
        if machine in {"x86_64", "amd64"}:
            return _TORCHCODEC_MIN_WHEEL_VERSION
        if machine in {"aarch64", "arm64"}:
            return _TORCHCODEC_MIN_WHEEL_LINUX_AARCH64
        return None  # ppc64le, s390x, riscv64: no wheel at any version
    if IS_MAC_ARM:
        return _TORCHCODEC_MIN_WHEEL_VERSION
    return None  # Intel Mac


def _macos_release_major() -> "int | None":
    """Major macOS version, or None off macOS / when it cannot be read."""
    if not IS_MACOS:
        return None
    try:
        release = platform.mac_ver()[0]
        return int(release.split(".", 1)[0]) if release else None
    except (ValueError, IndexError):
        return None


# The pinned MLX versions publish macosx_14_0_arm64 wheels, no sdist and no cp39, so
# macOS 13 and Python 3.9 have nothing to resolve to (`uv pip install --python-platform
# aarch64-apple-darwin mlx==0.32.1`). Asked before the install, like
# _torchcodec_spec_is_installable: pip_install exits on failure, so trying would end an
# install that today just comes up chat-only.
_MLX_MIN_PYTHON = (3, 10)
_MLX_MIN_MACOS_MAJOR = 14


def _mlx_pins_are_installable() -> bool:
    """Wheel for the pinned MLX versions here? An unreadable macOS reads as too old."""
    if sys.version_info < _MLX_MIN_PYTHON:
        return False
    return (_macos_release_major() or 0) >= _MLX_MIN_MACOS_MAJOR


# The supported Python range moves three times across the lines we select from.
# Transcribed from upstream's published table (README / PyPI):
#
#   0.1        >=3.9,  <=3.12
#   0.2 .. 0.7 >=3.9,  <=3.13
#   0.8        >=3.10, <=3.13
#   0.9 +      >=3.10, <=3.14
#
# Entries are (first release of the run, min python, max python); a run ends where the next
# begins. A separate axis from the platform floor: a host can have a wheel for its
# architecture and none for its interpreter. Reachable at torch 2.5 on Python 3.13, whose
# only line is 0.1, which stops at 3.12.
_TORCHCODEC_PYTHON_WINDOWS: "tuple[tuple[tuple[int, int, int], tuple[int, int], tuple[int, int]], ...]" = (
    ((0, 1, 0), (3, 9), (3, 12)),
    ((0, 2, 0), (3, 9), (3, 13)),
    ((0, 8, 0), (3, 10), (3, 13)),
    ((0, 9, 0), (3, 10), (3, 14)),
)


def _torchcodec_python_is_supported(
    floor: "tuple[int, ...]", ceiling: "tuple[int, ...] | None"
) -> bool:
    """Does any release in [floor, ceiling) ship a wheel for the running interpreter?"""
    running = sys.version_info[:2]
    for index, (start, py_min, py_max) in enumerate(_TORCHCODEC_PYTHON_WINDOWS):
        end = (
            _TORCHCODEC_PYTHON_WINDOWS[index + 1][0]
            if index + 1 < len(_TORCHCODEC_PYTHON_WINDOWS)
            else None
        )
        if ceiling is not None and start >= ceiling:
            continue  # run begins above the window
        if end is not None and end <= floor:
            continue  # run ends below the window
        if py_min <= running <= py_max:
            return True
    return False


# download.pytorch.org carries torchcodec only from 0.3 up; 0.1 and 0.2 were published to
# PyPI alone. So the two oldest rows (torch 2.5 -> 0.1, 2.6 -> 0.2) cannot be pinned at all,
# and pinning them would turn a working-or-not install into a guaranteed skip on exactly the
# oldest venvs. They keep today's unpinned behavior.
_TORCHCODEC_MIN_ON_TORCH_INDEX = (0, 3, 0)

# torchcodec has no xpu build: the xpu leaf republishes cpu wheels, Linux x86_64 only, from
# 0.13 up, so xpu takes cpu. Unpinned is not an option -- PyPI's default is the CUDA build.
_TORCHCODEC_INDEX_TAGS = {"xpu": "cpu"}


def _cuda_major_for_npp(torch_version: "str | None", index_url: str) -> str:
    """`"12"`, `"13"`, or `""` when this codec install needs no NPP runtime. The resident torch's LOCAL TAG first, the index URL only as a fallback: matching `/cuNNN$` on the URL failed for a supported UNSLOTH_TORCH_INDEX_URL ending in `/simple?token=...`, so a `+cu128` host skipped NPP and the codec then failed to import without a system CUDA toolkit. The tag is also the better source, since _torchcodec_index_url only returns an index once it has seen a `cpu` or `cuNNN` tag."""
    local = str(torch_version or "").partition("+")[2].strip().lower()
    match = re.fullmatch(r"cu(\d+)", local)
    if match:
        return match.group(1)[:2]
    # No usable local tag: a torch from PyPI carries none. Fall back to the leaf, which
    # still answers for the public download.pytorch.org form.
    match = re.search(r"/cu(\d+)/?$", index_url or "")
    return match.group(1)[:2] if match else ""


# Any sign of the CUDA runtime, versioned or not: nvcudart_hybrid64.dll is the Windows cu130
# spelling and carries no major. Absent entirely from a cpu build, which is what makes "" safe.
_CUDA_RUNTIME_MARKER_RE = re.compile(
    rb"nvcuda\.dll|torch_cuda|nvcudart|libcudart|cudart64|libcuda\.so"
)


def _pytorch_whl_leaf_url(leaf: str) -> "str | None":
    """_PYTORCH_WHL_BASE plus an accelerator leaf, or None when it cannot be expressed.

    No URL shape pins a query-auth mirror -- pip joins the project name as text, so the token
    swallows either the leaf or the name (see _warn_query_index_unusable). Constructing one
    anyway is worse than declining: --index-url makes _install_env_for_cmd strip pip.conf and
    ~/.netrc, the only channel that can carry that credential.
    """
    if "?" in _PYTORCH_WHL_BASE or "#" in _PYTORCH_WHL_BASE:
        _warn_query_index_unusable(_PYTORCH_WHL_BASE)
        return None
    return f"{_PYTORCH_WHL_BASE}/{leaf}"


def _torchcodec_distribution_for_probe():
    """The installed torchcodec distribution, or None when it cannot be read."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        return distribution("torchcodec")
    except (PackageNotFoundError, ValueError, OSError):
        return None


def _installed_torchcodec_cuda_major() -> "str | None":
    """The CUDA major the INSTALLED torchcodec links, "" when it links none, None when unknown.

    Only the unpinned fallback needs this: the default index carries whichever major PyTorch
    currently ships, not the resident torch's (PyPI's 0.16.0 links libcudart.so.13 while a
    +cu129 tag says 12), so it is read off the wheel rather than inferred.

    The three answers must stay distinct because "" SKIPS the NPP install. win_amd64 cu130
    names no major anywhere -- only nvcudart_hybrid64.dll -- so it is None, not "".
    """
    dist = _torchcodec_distribution_for_probe()
    if dist is None:
        return None
    files = list(getattr(dist, "files", None) or [])
    majors, saw_cuda, inspected = set(), False, False
    for entry in files:
        name = str(entry).lower()
        if not (name.endswith((".so", ".dll", ".pyd")) or ".so." in name):
            continue
        try:
            blob = Path(entry.locate()).read_bytes()
        except OSError:
            continue
        inspected = True
        majors |= {m.decode()[-2:] for m in re.findall(rb"libcudart\.so\.\d+", blob)}
        majors |= {m.decode()[9:11] for m in re.findall(rb"cudart64_\d+\.dll", blob)}
        saw_cuda = saw_cuda or bool(_CUDA_RUNTIME_MARKER_RE.search(blob))
    if not inspected:
        return None  # nothing readable: say so rather than claim there is no CUDA
    if majors:
        return sorted(majors)[-1]
    # CUDA is clearly there but unversioned in the names, so the tag-derived answer stands.
    return None if saw_cuda else ""


# Local tags a companion wheel is served under. xpu and rocmX.Y are here because those leaves
# really do carry companion builds; which a given package may use is that package's call.
_TORCH_ACCELERATOR_TAG_RE = re.compile(r"cpu|cu\d+|xpu|rocm\d+(\.\d+)?")


def _torch_accelerator_index_url(
    torch_version: "str | None", substitutions: "dict[str, str] | None" = None
) -> "str | None":
    """The download.pytorch.org leaf serving the resident torch's build, or None.

    Companion wheels (torchcodec, torchao) are published per accelerator exactly the way
    torch is: PyPI carries one default flavor and the rest live under /whl/<tag>. The right
    version from the wrong index is a wheel whose cpp cannot dlopen against the resident
    CUDA or HIP runtime.

    Only an EXPLICIT local tag pins. An untagged torch is PyPI's own build, whose counterpart
    is PyPI's default wheel -- already the right pairing. That is the opposite reading from
    _torch_flavor_tag, which maps untagged to "cpu" for the Windows repair path; here an
    untagged Linux torch is a CUDA build, so pinning cpu would install the wrong one.
    """
    if not torch_version:
        return None
    substitutions = substitutions or {}
    # Verbatim, and BEFORE the tag check: only download.pytorch.org stamps +cuNNN, so a
    # private mirror rebuilding torch ships it bare, and requiring a tag first sent exactly
    # that host to PyPI. Rewriting a leaf inside the URL would be a guess.
    url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if url:
        return _trim_index_path_slashes(url)
    # Substitution still applies (FAMILY=xpu must not send torchcodec to a leaf with no codec
    # below 0.13). Before the tag check for the same reason as the URL, and because
    # _explicit_unknown_family_torch_index_url leaves such a torch alone: a None is final.
    family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip().strip("/")
    if family:
        return _pytorch_whl_leaf_url(substitutions.get(family.lower(), family))
    local = str(torch_version).partition("+")[2].strip().lower()
    if not _TORCH_ACCELERATOR_TAG_RE.fullmatch(local):
        return None
    return _pytorch_whl_leaf_url(substitutions.get(local, local))


def _torchcodec_index_url(torch_version: "str | None", spec: str = "") -> "str | None":
    """The torchcodec index serving the resident torch's build, or None to stay unpinned.

    Upstream's install docs say to pass --index-url and "make sure to install the
    corresponding PyTorch version as well", so a cu126 or cu128 venv that takes PyPI's
    default gets a codec built against a different CUDA and libtorchcodec cannot dlopen.
    docker/Dockerfile already pins cu128 by hand for this reason.

    rocm is the one accelerator excluded: every rocm leaf answers 404/403 for torchcodec,
    so a pin there is a guaranteed wasted resolve. xpu is redirected to cpu, for the reason
    beside _TORCHCODEC_INDEX_TAGS. Where a pinned index turns out not to serve the selected
    window (cu129 has no 0.8 or 0.9), the caller's unpinned retry recovers; that is
    deliberate, because no local table of index CONTENTS stays true.
    """
    local = str(torch_version or "").partition("+")[2].strip().lower()
    if local.startswith("rocm"):
        return None
    if spec:
        _, ceiling = _torchcodec_spec_bounds(spec)
        if ceiling is not None and ceiling <= _TORCHCODEC_MIN_ON_TORCH_INDEX:
            return None  # window sits entirely below what any torch index publishes
    return _torch_accelerator_index_url(torch_version, _TORCHCODEC_INDEX_TAGS)


def _torch_index_tag(
    torch_version: "str | None", substitutions: "dict[str, str] | None" = None
) -> "str | None":
    """The local tag of the build the index pin will fetch, or None when unknowable.

    NOT always the resident torch's own tag: a FAMILY override names a leaf of its own, and
    a per-package substitution can redirect one (xpu takes the cpu codec). Provenance checks
    compare against this, so it has to be derived the same way the URL is, or a pin to one
    leaf gets validated against another leaf's tag and never fires.

    None means an explicit UNSLOTH_TORCH_INDEX_URL is in play. That URL is opaque -- it can
    be an accelerator-specific private mirror -- so nothing here can say which build it
    serves. Callers must treat that as "cannot prove the installed wheel came from there"
    and replace, not as "no tag required": an untagged wheel already satisfies the version,
    so pip would fetch nothing and the mirror would never be reached.
    """
    if os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip():
        return None
    substitutions = substitutions or {}
    family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip().strip("/").lower()
    local = family or str(torch_version or "").partition("+")[2].strip().lower()
    return substitutions.get(local, local)


def _torchcodec_index_tag(torch_version: "str | None") -> "str | None":
    return _torch_index_tag(torch_version, _TORCHCODEC_INDEX_TAGS)


def _torchcodec_spec_bounds(spec: str) -> "tuple[tuple[int, ...], tuple[int, ...] | None]":
    """`torchcodec>=0.6.0,<0.8.0` -> ((0,6,0), (0,8,0)); an open floor gives (floor, None)."""

    def _v(text: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", text)[:3])

    body = spec.split("torchcodec", 1)[1]
    floor_match = re.search(r">=\s*([0-9.]+)", body)
    ceiling_match = re.search(r"<\s*([0-9.]+)", body)
    floor = _v(floor_match.group(1)) if floor_match else (0,)
    ceiling = _v(ceiling_match.group(1)) if ceiling_match else None
    return floor, ceiling


def _torchcodec_spec_is_installable(spec: str) -> bool:
    """Does this host have a wheel for any release the spec admits? Asked before the install rather than discovered by it, because the install step exits on failure; answering no means the audio extra is skipped, which is what a host with no wheel got before this step existed."""
    host_floor = _torchcodec_platform_floor()
    if host_floor is None:
        return False
    floor, ceiling = _torchcodec_spec_bounds(spec)
    # The window has to reach the first release this platform actually published.
    if ceiling is not None and host_floor >= ceiling:
        return False
    if not _torchcodec_python_is_supported(max(floor, host_floor), ceiling):
        return False
    if IS_MAC_ARM and (_macos_release_major() or 0) < 14:
        # 0.12+ is macosx_14_0 only. Reachable when the window starts at or above it.
        effective_floor = max(floor, host_floor)
        if effective_floor >= _TORCHCODEC_MACOS_14_ONLY_FROM:
            return False
    return True


def _select_torchcodec_spec(torch_version: "str | None") -> str:
    """Map an installed torch version (e.g. '2.11.0+cu128') to the torchcodec spec built against it. Falls back to _TORCHCODEC_DEFAULT_SPEC for torch <=2.4, a non-2.x major, or an unparseable/missing version. Pure function."""
    if not torch_version:
        return _TORCHCODEC_DEFAULT_SPEC
    release = str(torch_version).split("+", 1)[0]  # drop +cu128/+rocm7.2/+cpu
    parts = release.split(".")
    try:
        # '11rc1' -> '11', matching _select_torchao_spec.
        minor_str = re.sub(r"[^0-9].*", "", parts[1]) if len(parts) > 1 else ""
        major, minor = int(parts[0]), int(minor_str)
    except (IndexError, ValueError):
        return _TORCHCODEC_DEFAULT_SPEC
    if major != 2:
        return _TORCHCODEC_DEFAULT_SPEC
    # Clamp to the ABI-stable floor, never the 0.11 row: 0.11 is locked to torch 2.11 exactly.
    minor = min(minor, _TORCHCODEC_MAX_KNOWN_MINOR)
    return _TORCHCODEC_TORCH_SPECS.get(minor, _TORCHCODEC_DEFAULT_SPEC)


# Memoized `import torch` classification of the target venv; pip_install() / pip_install_try() reset it. None means not probed yet.
_TORCH_RUNTIME_PROBE: "tuple[bool, bool, str | None, str, str] | None" = None
# Beside the tuple, not in it: thirteen call sites unpack that, and only the GPU-build verdict needs this.
_TORCH_RUNTIME_XPU: str = ""

# Prefix on the probe's own stdout line: import chatter and atexit/CUDA teardown notices land on either side, so "the last non-empty line" is not reliably ours.
_TORCH_PROBE_MARKER = "UNSLOTH_TORCH_PROBE|"

# Prefix on the --amd-torch-needs-dependency-pass decision line: five states share exit 1, so a caller (CI) must read WHICH input decided.
_AMD_FASTPATH_DECISION_MARKER = "UNSLOTH_AMD_FASTPATH|"


def _invalidate_torch_runtime_probe() -> None:
    """Forget the memoized torch classification after a pip operation."""
    global _TORCH_RUNTIME_PROBE
    _TORCH_RUNTIME_PROBE = None


def _probe_torch_runtime() -> "tuple[bool, bool, str | None, str, str]":
    """Classify the venv's torch with ONE `import torch` per install run, bounding a stalled driver at a single 90s wait. Returns (ran, importable, version, hip, cuda); ran=False is the wedged-driver case (fall back to on-disk classifiers) and version None means no answer, so the venv must be left alone, unlike ""."""
    global _TORCH_RUNTIME_PROBE
    if _TORCH_RUNTIME_PROBE is not None:
        return _TORCH_RUNTIME_PROBE
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "v = getattr(torch, '__version__', '') or ''; "
                    # torch.version is not guaranteed to exist, and reaching through it unguarded reads as "torch cannot import" and force-reinstalls a working venv.
                    "_v = getattr(torch, 'version', None); "
                    "h = getattr(_v, 'hip', '') or ''; "
                    "c = getattr(_v, 'cuda', '') or ''; "
                    # XPU too: such a wheel carries its runtime here and nowhere else.
                    "x = getattr(_v, 'xpu', '') or ''; "
                    f"print('{_TORCH_PROBE_MARKER}' + '|'.join((v, h, c, x)))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            # errors="replace": text=True decodes strictly, and a UnicodeDecodeError in torch's import chatter would take the installer down instead of falling back to the on-disk classifier.
            errors = "replace",
            timeout = 90,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        _TORCH_RUNTIME_PROBE = (False, False, None, "", "")
        return _TORCH_RUNTIME_PROBE
    # Our own marked line, last one wins: chatter can land on either side of it.
    _marked = [
        line.strip()
        for line in (probe.stdout or "").splitlines()
        if line.strip().startswith(_TORCH_PROBE_MARKER)
    ]
    version: "str | None" = None
    hip = cuda = ""
    global _TORCH_RUNTIME_XPU
    if _marked:
        _fields = _marked[-1][len(_TORCH_PROBE_MARKER) :].split("|")
        version, hip, cuda = (_fields + ["", "", ""])[:3]
        _TORCH_RUNTIME_XPU = (_fields + ["", "", "", ""])[3]
    _TORCH_RUNTIME_PROBE = (True, probe.returncode == 0, version, hip, cuda)
    return _TORCH_RUNTIME_PROBE


def _probe_installed_torch_version() -> str | None:
    """torch.__version__ from the target venv, or None. Cross-platform, unlike the Linux-only probe_torch_wheel_env."""
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran or not _importable:
        return None
    return _version or None


def _installed_distribution_version(name: str) -> str | None:
    """Return installed distribution metadata without importing the package."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(name)
    except (PackageNotFoundError, ValueError):
        return None


def _exact_distribution_spec_is_installed(spec: str) -> bool:
    """Whether a simple ``name==version`` pin already matches this venv."""
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)", spec)
    if match is None:
        return False
    installed = _installed_distribution_version(match.group(1))
    return installed is not None and installed == match.group(2)


def _pin_needs_reinstall(spec: str, want_tag: "str | None" = "") -> bool:
    """Whether a ``name==version`` pin has to be forced over what is already installed.

    Two reasons, and the second only exists once an index is pinned. The version can be
    wrong, which _exact_distribution_spec_is_installed already answers. Or the version can
    be RIGHT while the build is wrong: an accelerator index stamps its tag into the local
    version (``0.18.0+cu130``) and PyPI forbids one, so a wheel already inside the pin
    satisfies pip, nothing is fetched, and the wrong-accelerator build this pin exists to
    replace stays put.

    want_tag is the tag the pin will FETCH (_torch_index_tag), not the resident torch's,
    which a FAMILY override can differ from. "" is the unpinned case: the default index stamps
    no tag, so a tagged wheel there came from elsewhere and is replaced once, then settles.
    None is an opaque mirror, where provenance is unprovable, so it is replaced every time.
    """
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)", spec)
    if match is None:
        return False
    installed = _installed_distribution_version(match.group(1))
    if installed is None:
        return True  # absent; kept True so the flag matches what this step passed before
    if installed.partition("+")[0] != match.group(2).partition("+")[0]:
        return True
    return installed.partition("+")[2].strip().lower() != want_tag


def _installed_torch_is_windows_rocm() -> bool:
    """True when the venv holds a Windows ROCm torch build: belt-and-suspenders for the torchao skip, since torchao crashes on import there."""
    if not IS_WINDOWS:
        return False
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran or not _importable:
        return False
    _ver = (_version or "").lower()
    return bool(_hip) or "rocm" in _ver or "rocmsdk" in _ver


# constraints.txt caps anyio <4.14 (#6483), but a pre-cap install stuck at 4.14+ is untouched.
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


# AMD Windows ROCm wheels (repo.amd.com/rocm/whl/{arch_family}/); UNSLOTH_ROCM_WINDOWS_MIRROR overrides. Kept verbatim: "/" is in the base64 alphabet, so a whole-URL trim eats a slash belonging to a query token.
_ROCM_WINDOWS_INDEX_BASE = (
    os.environ.get("UNSLOTH_ROCM_WINDOWS_MIRROR") or "https://repo.amd.com/rocm/whl"
)

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

# Archs that install ROCm wheels fine and then compute WRONG answers, so they stay on CPU torch. See install.sh's matching gate.
_ROCM_MISCOMPUTING_GFX: "frozenset[str]" = frozenset({"gfx1033"})  # Van Gogh (Steam Deck)

# bitsandbytes continuous-release_main wheels with the ROCm 4-bit GEMV fix (bnb #1887): <=0.49.2 NaNs at decode shape on every AMD GPU, PyPI 0.50.0 is the first release with it.
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
    # The Windows ROCm wheel ships libbitsandbytes_rocm{VER}.dll; BNB_ROCM_VERSION must match.
    "win_amd64": (
        "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/"
        "download/continuous-release_main/"
        "bitsandbytes-1.33.7.preview-py3-none-win_amd64.whl"
    ),
}
# Keep in step with the amd extra in pyproject.toml and the install.sh fallback.
_BNB_ROCM_PYPI_FALLBACK = "bitsandbytes>=0.50.0"


def _bnb_rocm_prerelease_url() -> str | None:
    """continuous-release_main bnb wheel URL for the current arch, or None when there is none."""
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return _BNB_ROCM_PRERELEASE_URLS.get(arch)


def _bnb_rocm_arch_has_binary() -> bool:
    """False on aarch64: no bitsandbytes wheel there ships ROCm kernels at any version, so no install path gives it a 4-bit backend."""
    arch = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch) != "aarch64"


def _amd_smi_env() -> dict[str, str] | None:
    """On Windows, env with __COMPAT_LAYER=RunAsInvoker; None elsewhere. It does not stop amd-smi elevating a child; _amd_smi_allowed() is the real guard."""
    if platform.system() != "Windows":
        return None
    return {**os.environ, "__COMPAT_LAYER": "RunAsInvoker"}


def _path_inside_venv(path: str) -> bool:
    """True if ``path`` is inside the active venv. The venv hipInfo.exe (AMD wheel) is NOT a HIP SDK."""
    try:
        # realpath (not abspath): resolve symlinks/8.3 names so an aliased venv matches.
        _root = os.path.normcase(os.path.realpath(sys.prefix))
        # A root prefix (C:\ or /) would commonpath-match everything; a venv is never at root.
        if os.path.dirname(_root) == _root:
            return False
        return os.path.normcase(os.path.commonpath([os.path.realpath(path), _root])) == _root
    except (ValueError, OSError):
        return False


def _external_hipinfo_on_path() -> bool:
    """True if a hipinfo OUTSIDE the venv is on PATH; scan every entry, since the venv hipInfo can shadow a real HIP SDK's."""
    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        _dir = _dir.strip('"')  # PATH entries can be quoted on Windows
        if not _dir:
            continue
        _candidate = os.path.join(_dir, "hipinfo.exe")
        if os.path.isfile(_candidate) and not _path_inside_venv(_candidate):
            return True
    return False


def _amd_smi_allowed() -> bool:
    """Whether it is safe to spawn amd-smi: on Windows without a HIP runtime it pops a UAC/DiskPart prompt RunAsInvoker cannot suppress, so require hipinfo or UNSLOTH_ENABLE_AMD_SMI=1; Linux/macOS always."""
    if platform.system() != "Windows":
        return True
    flag = os.environ.get("UNSLOTH_ENABLE_AMD_SMI", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    # hipinfo-on-PATH proxies a real HIP SDK; the venv hipInfo.exe is not one.
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


# Memoized: _ensure_rocm_torch() runs twice on Linux; the disagreement warning prints once.
_ROCM_VERSION_PROBE: "tuple[int, int] | None" = None
_ROCM_VERSION_PROBED: bool = False


def _invalidate_rocm_version_probe() -> None:
    """Forget the memoized host ROCm version."""
    global _ROCM_VERSION_PROBE, _ROCM_VERSION_PROBED
    _ROCM_VERSION_PROBE = None
    _ROCM_VERSION_PROBED = False


def _detect_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm stack, or None. Memoized per run."""
    global _ROCM_VERSION_PROBE, _ROCM_VERSION_PROBED
    if not _ROCM_VERSION_PROBED:
        _ROCM_VERSION_PROBE = _detect_rocm_version_uncached()
        _ROCM_VERSION_PROBED = True
    return _ROCM_VERSION_PROBE


def _detect_rocm_version_uncached() -> tuple[int, int] | None:
    """Probe every ROCm version source and return the highest reading, or None."""
    readings: list[tuple[str, tuple[int, int]]] = []

    def _record(source: str, major: int, minor: int) -> None:
        readings.append((source, (major, minor)))

    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path, encoding = "utf-8") as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            if len(parts) >= 2:
                _record("ROCm version file", int(parts[0]), int(parts[1]))
                break
        except Exception:
            pass

    # amd-smi version ("ROCm version: X.Y.Z"); off on Windows without a HIP SDK (UAC prompt).
    amd_smi = shutil.which("amd-smi") if _amd_smi_allowed() else None
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
                env = _amd_smi_env(),
            )
            if result.returncode == 0:
                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    _record(
                        "amd-smi",
                        int(m.group(1)),
                        int(m.group(2)),
                    )
        except Exception:
            pass

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
                    _record(
                        "hipconfig",
                        int(parts[0]),
                        int(parts[1].split("-")[0]),
                    )
        except Exception:
            pass

    # Only "installed" counts: dpkg-query still reports removed-but-not-purged packages. rocm-core wins outright over the distro's libhsa-runtime64-1, which can be older.
    dpkg = shutil.which("dpkg-query")
    if dpkg:
        try:
            result = subprocess.run(
                [
                    dpkg,
                    "-W",
                    "-f=${Package} ${Status} ${Version}\n",
                    "rocm-core",
                    "libhsa-runtime64-1",
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
            )
            # dpkg-query exits nonzero when either package is absent but still prints the other's line, so parse stdout regardless of the return code.
            _dpkg_readings: "dict[str, list[tuple[int, int]]]" = {"rocm-core": [], "hsa": []}
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 5 or fields[3] != "installed":
                    continue
                package, raw = fields[0], fields[4]
                if package not in ("rocm-core", "libhsa-runtime64-1"):
                    continue
                # dpkg can prepend an epoch ("1:6.3.0-1"); strip it before parsing.
                raw = re.sub(r"^\d+:", "", raw)
                m = re.match(r"(\d+)[.-](\d+)", raw)
                if m:
                    _key = "rocm-core" if package == "rocm-core" else "hsa"
                    _dpkg_readings[_key].append((int(m.group(1)), int(m.group(2))))
            if _dpkg_readings["rocm-core"]:
                for _major, _minor in _dpkg_readings["rocm-core"]:
                    _record("dpkg rocm-core", _major, _minor)
            else:
                for _major, _minor in _dpkg_readings["hsa"]:
                    _record("dpkg HSA runtime", _major, _minor)
        except Exception:
            pass

    rpm = shutil.which("rpm")
    if rpm:
        try:
            result = subprocess.run(
                [
                    rpm,
                    "-q",
                    "--qf",
                    "%{VERSION}\n",
                    "rocm-core",
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = result.stdout.strip()
                m = re.match(r"(\d+)[.-](\d+)", raw)
                if m:
                    _record(
                        "rpm rocm-core",
                        int(m.group(1)),
                        int(m.group(2)),
                    )
        except Exception:
            pass

    if not readings:
        return None

    best = max(version for _, version in readings)
    distinct = {version for _, version in readings}

    if len(distinct) > 1:
        details = ", ".join(
            f"{source}=rocm{version[0]}.{version[1]}" for source, version in readings
        )
        _safe_print(
            f"WARNING: ROCm version sources disagree ({details}) -- "
            f"using the highest, rocm{best[0]}.{best[1]}.",
            file = sys.stderr,
        )

    return best


# APU arches whose board commonly also carries a discrete Radeon: HIP often enumerates the APU first, so an index-0 pick installs iGPU wheels (#7776). Strix (gfx1150/1151/1152) excluded.
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
    """True when the user selected devices via HIP_/ROCR_/CUDA_VISIBLE_DEVICES. First-set-wins, and ANY value counts, including "" and "-1", which select NO GPU: the runtime stores an empty var as " " (clr flags.cpp) and picks the HIP mask whenever its first byte is not NUL, so an empty HIP mask shadows CUDA_VISIBLE_DEVICES."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(_env) is not None:
            return True
    return False


def _pick_visible_index(
    num_tokens: int,
    warn: bool = True,
    masks: "tuple[str, ...] | None" = None,
) -> int:
    """Resolve HIP_/ROCR_/CUDA_VISIBLE_DEVICES to an index into a list of num_tokens; 0 for unset, empty, '-1', UUID-style or out-of-range. First-set-wins, because an empty HIP mask shadows CUDA_VISIBLE_DEVICES in the runtime. ``masks`` narrows the layers consulted, for callers that already applied the ROCr one."""
    for _env in masks if masks is not None else _VISIBLE_DEVICE_MASKS:
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
            # Say so rather than silently installing for GPU 0, the iGPU the user masked off. Callers with a deduplicated list pass warn=False: there the index space is arches, not devices.
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
    """gcnArchName on Windows, or None. Probe order matches the PowerShell installer: env override, hipinfo, then amd-smi, without which runtime-only AMD installs cannot repair a CPU-only venv. Tokens are de-duplicated in enumeration order and the masks pick one; unmasked, a shadowing iGPU yields to the discrete GPU (#7776)."""
    # 1. Explicit override (matches the PowerShell installer's env-var path).
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
        # Index into the full ordered list: hipinfo is itself a HIP application, so under a mask it enumerates only visible devices renumbered from 0 and indexing again applies the mask twice. amd-smi and WMI list every GPU.
        _pick = tokens[0 if mask_resolved else _pick_visible_index(len(tokens), warn = warn)]
        _distinct = list(dict.fromkeys(tokens))
        if len(_distinct) < 2 or _visible_devices_pinned():
            # A pin is honoured verbatim, but say so when it selected a card with no AMD Windows wheels while another enumerated GPU has them: torch silently drops to CPU.
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
        # Unpinned mixed-arch host: skip a leading shadowing iGPU so the discrete card decides the wheel family (#7776), and say so, since only enumeration order put the APU first.
        if _pick in _SHADOWING_INTEGRATED_GFX:
            _others = [t for t in tokens if t not in _SHADOWING_INTEGRATED_GFX]
            # Prefer a wheel-backed candidate: deposing the pick for a card with no Windows wheels resolves to no index and drops the host to CPU.
            _withWheels = [t for t in _others if _windows_rocm_index_url(t) is not None]
            _candidates = _withWheels or (
                [] if _windows_rocm_index_url(_pick) is not None else _others
            )
            if _candidates:
                _other = _candidates[0]
                # Not always device 1: on gfx1036,gfx1010,gfx1200 it is device 2, and "mask 1" would expose the gfx1010 the wheels do not target.
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
        # 2b. AMD torch wheels drop hipInfo.exe into venv Scripts, so driver-only hosts re-detect.
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
            # Accept partial output when hipinfo crashes (0xC0000005 on some RDNA 4, #6043).
            text = result.stdout.decode(errors = "replace")
            # findall gets every gcnArchName line so multi-GPU hosts are enumerable. Split on ':' like setup.ps1: "gfx90a:sramecc+:xnack-" matches neither the wheel table nor the shadowing set.
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

    # 4. Last resort: GPU marketing name via WMI -> arch table, for driver-only hosts with neither hipinfo nor amd-smi. Mirrors setup.ps1's $nameArchTable and $wmiGpus filter: ConfigManagerErrorCode 0 and AMD names only, since the masks index AMD devices and a stray adapter shifts every index.
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
            # Lines are "<name>|<ConfigManagerErrorCode>"; a bare name (older probe output) counts as healthy.
            _all_names, _healthy = [], []
            for _line in result.stdout.decode(errors = "replace").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                _nm, _sep, _code = _line.rpartition("|")
                if not _sep:
                    _nm, _code = _line, "0"
                _nm = _nm.strip()
                # Re-apply the vendor filter rather than trusting the command string: a stray NVIDIA or Intel adapter would shift the mask index and warn at a non-AMD user.
                if not _nm or not re.search(r"AMD|Radeon", _nm, re.IGNORECASE):
                    continue
                _all_names.append(_nm)
                if _code.strip() in ("", "0"):
                    _healthy.append(_nm)
            # Drop adapters Windows flags as not working, but fall back to the full list when that leaves NOTHING: code 45 is routine on muxless laptops with a parked dGPU, and the filter alone hands the host CPU torch.
            _names = _healthy or _all_names
            _tokens = [_a for _a in map(_gfx_arch_from_gpu_name, _names) if _a]
            # Resolve the mask over the ADAPTER list (setup.ps1's $nameIdx): a name the table does not know drops out of _tokens, and indexing that shortened list would name a different card.
            _sel = _pick_visible_index(len(_names)) if _names else 0
            _named = _gfx_arch_from_gpu_name(_names[_sel]) if _names else None
            # Borrow another adapter's arch only when unpinned: under a mask, substituting installs for a GPU the user masked away.
            if not _named and not _visible_devices_pinned() and _tokens:
                _named = _tokens[0]
            # Repick only when every AMD adapter mapped: an unknown name may BE the discrete card, so the index would count arches, not devices.
            _pick = _dedup_pick(_tokens, warn = False) if len(_tokens) == len(_names) else _named
            if _pick:
                _safe_print(f"   gfx arch inferred from GPU name (WMI): {_pick}")
                return _pick
            if _names and not _pick:
                # No arch means CPU-only torch; name the adapter. RDNA 1 / Polaris is not an unknown card: an override there sends the user after a fix that does not exist (#8529, #8458).
                _unsupported = _unsupported_gfx_arch_from_gpu_name(_names[_sel])
                if _unsupported:
                    # The CPU-only half is false under an explicit index pin, which is honoured for any arch.
                    _pinned = bool(
                        (os.environ.get("UNSLOTH_TORCH_INDEX_URL") or "").strip()
                        or (os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY") or "").strip()
                    )
                    _tail = (
                        "so the torch index you pinned is used as given."
                        if _pinned
                        else (
                            "so torch will be CPU-only. No HIP SDK install and "
                            "no UNSLOTH_ROCM_GFX_ARCH value changes that on this GPU."
                        )
                    )
                    _safe_print(
                        f"   [WARN] '{_names[_sel]}' is {_unsupported}, which Unsloth's ROCm "
                        f"PyTorch wheels do not cover, {_tail}"
                    )
                    # Torch ends here, llama.cpp does not: Vulkan drives these cards (#8458). PowerShell syntax since this branch is Windows-only, and not on ARM64, where setup.ps1 THROWS on that variable.
                    if _is_windows_arm64():
                        _safe_print(
                            "   [INFO] GGUF chat would need Vulkan on this GPU, and no "
                            "Windows ARM64 Vulkan bundle is published: build llama.cpp "
                            "from source, or run this on x64."
                        )
                    else:
                        _safe_print(
                            "   [INFO] GGUF chat can still run on this GPU through Vulkan: set "
                            '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan" and re-run the installer. '
                            "It selects the llama.cpp bundle at install time, so setting it "
                            "afterwards has no effect until you install or update again."
                        )
                else:
                    _safe_print(
                        f"   [WARN] could not map '{_names[_sel]}' to a gfx arch, so torch "
                        f"will be CPU-only. Set UNSLOTH_ROCM_GFX_ARCH to your GPU's arch "
                        f"(e.g. gfx1200) to install AMD wheels."
                    )
    except Exception:
        pass
    return None


# GPU marketing-name -> gfx arch (mirrors setup.ps1's $nameArchTable), most-specific first.
_WIN_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    # RDNA 4 (Navi 48). R9700 is listed separately: its name holds neither 9070 nor 9080, so it fell through to CPU torch (#7624, #7307).
    (r"9070|9080|R9700", "gfx1201"),
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


# GPU name -> gfx arch for AMD generations our ROCm wheels do NOT cover (RDNA 1, Polaris 10/20/30; #8529, #8458). Deliberately SEPARATE from _WIN_GPU_NAME_ARCH_TABLE: nothing here may ever route to a wheel index. Every (?!0) guard stops "RX 570" swallowing "RX 5700", so each row is order-independent. Names from LLVM's AMDGPU tables plus libdrm amdgpu.ids; nothing is guessed, so Polaris 11/12 is left out.
_UNSUPPORTED_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    (r"Radeon Pro V520|Radeon Pro 5600M", "gfx1011"),  # RDNA 1
    (
        r"RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700",
        "gfx1010",
    ),  # RDNA 1 (Navi 10)
    (r"RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300", "gfx1012"),  # RDNA 1 (Navi 14)
    (
        r"RX 4[78]0(?!0)|RX 5[789]0(?!0)|Radeon Pro WX 7100|Radeon Pro WX 5100",
        "gfx803",
    ),  # Polaris 10/20/30
]


def _unsupported_gfx_arch_from_gpu_name(name: str) -> "str | None":
    """Name the gfx arch of a GPU whose generation has no ROCm wheels. Messaging only: never feed the result into index selection."""
    if not name:
        return None
    for _pat, _arch in _UNSUPPORTED_GPU_NAME_ARCH_TABLE:
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
    """First AMD display-class lspci line mapping to a known gfx arch; scan them all, since a non-AMD controller can enumerate first. The vendor guard is case-SENSITIVE, or -i "ATI" matches "CorporATIon"."""
    lspci = shutil.which("lspci")
    if not lspci:
        return None
    try:
        result = subprocess.run(
            [lspci, "-nn"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
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
    """librocdxg, the WSL ROCDXG bridge, under a ROCm lib dir; its absence marks a WSL box whose ROCm was never set up."""
    dirs = ["/opt/rocm/lib", "/opt/rocm/lib64"]
    dirs += glob.glob("/opt/rocm-*/lib") + glob.glob("/opt/rocm-*/lib64")
    return any(
        os.path.exists(os.path.join(d, so))
        for d in dirs
        for so in ("librocdxg.so", "librocdxg.so.1")
    )


def _linux_amd_display_device_present() -> bool:
    """Any AMD (0x1002) PCI display-class device in sysfs. /proc/cpuinfo leaks the HOST CPU model into VMs with no GPU, so CPU-model text is not GPU evidence (mirrors install.sh _amd_gpu_present_via_pci)."""
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
        # cpuinfo/lspci see the host APU even on a WSL box with no ROCDXG runtime; WSL enumerates no PCI display device, so /dev/dxg + librocdxg IS the GPU evidence there.
        if not _wsl_rocm_runtime_present():
            return None
    elif not _linux_amd_display_device_present():
        # Native Linux: a VM on a Strix host shows the host CPU model while receiving no AMD GPU, so require an AMD display device before trusting the inference.
        return None
    cpu_gfx = _linux_amd_gfx_from_cpuinfo()
    if cpu_gfx:
        return cpu_gfx
    return _linux_amd_gfx_from_lspci()


# Mirrors already named, so a repeated repair does not repeat the notice.
_WARNED_QUERY_INDEX_BASES: "set[str]" = set()


def _warn_query_index_unusable(base: str) -> None:
    """Say so when a mirror carries its credential in the query or fragment: pip joins project URLs as text, so ".../gfx1151/?token=x" asks the index ROOT and a fragment never reaches the server. Reportable, not repairable."""
    if ("?" not in base and "#" not in base) or base in _WARNED_QUERY_INDEX_BASES:
        return
    _WARNED_QUERY_INDEX_BASES.add(base)
    _safe_print(
        "   The ROCm mirror carries its credential in the URL query or fragment. pip\n"
        "   appends the package name to the index URL as text, so the name lands inside\n"
        "   the credential and no package resolves. Put the credential in the URL itself\n"
        "   (https://user:token@host/path/), or in ~/.netrc, instead.\n"
    )


def _index_url_join(base: str, leaf: str) -> str:
    """Append a path segment to a wheel index URL, keeping any query / fragment, splitting on the FIRST "?" or "#"; rstrip + concat would bury the leaf inside a token. The lesser of two corruptions, not a working index."""
    _warn_query_index_unusable(base)
    _cuts = [base.index(_c) for _c in "?#" if _c in base]
    _head, _sep, _tail = (
        (base[: min(_cuts)], base[min(_cuts)], base[min(_cuts) + 1 :]) if _cuts else (base, "", "")
    )
    return f"{_head.rstrip('/')}/{leaf}/{_sep}{_tail}"


def _amd_arch_index_url(gfx_arch: str | None) -> str | None:
    """AMD per-arch pip index URL for a gfx arch. Windows honors UNSLOTH_ROCM_WINDOWS_MIRROR, Linux UNSLOTH_AMD_ROCM_MIRROR (install.sh's var); both default to repo.amd.com."""
    if IS_WINDOWS:
        return _windows_rocm_index_url(gfx_arch)
    # gfx1033 (Van Gogh) miscomputes under ROCm (studio/ROCM_RDNA2_APU.md), so the inferred-gfx repair below must not force-reinstall the wheels install.sh's gate avoids. Linux only, matching where it was measured.
    if (gfx_arch or "").lower() in _ROCM_MISCOMPUTING_GFX:
        return None
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    base = os.environ.get("UNSLOTH_AMD_ROCM_MIRROR") or "https://repo.amd.com/rocm/whl"
    return _index_url_join(base, arch_family)


def _physical_amd_gfx_archs() -> "list[str]":
    """The AMD arches this Linux host physically has, from sources an override cannot move. Strongest first: ROCm userland probes (override and masks stripped), KFD sysfs, product-name inference, then the declared UNSLOTH_ROCM_GFX_ARCH -- last, because a stale gfx1030 on a real Van Gogh otherwise hides the arch."""
    _archs = [
        _code.strip().lower().split(":")[0]
        for _code in _detect_amd_gfx_codes(ignore_hsa_override = True, ignore_visible_masks = True)
    ]
    if not _archs:
        _archs = [_code.strip().lower().split(":")[0] for _code in _kfd_gfx_targets()]
    if not _archs:
        _inferred = (_infer_linux_amd_gfx_arch() or "").strip().lower().split(":")[0]
        _archs = [_inferred] if _inferred else []
    if not _archs:
        _env_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":")[0]
        _archs = [_env_gfx] if _env_gfx else []
    return _archs


def _miscomputing_arch_host() -> bool:
    """True when EVERY AMD arch this host has computes incorrectly under ROCm. Every, not any: gfx1033 can lead the enumeration on a box whose real accelerator is a discrete Radeon (#7776)."""
    if IS_WINDOWS or IS_MACOS:
        return False
    _archs = _physical_amd_gfx_archs()
    return bool(_archs) and all(_arch in _ROCM_MISCOMPUTING_GFX for _arch in _archs)


def _rocm_miscomputing_host() -> bool:
    """True when every AMD GPU here miscomputes under ROCm, ROCm torch is already installed, and no index pin overrides it: declining to GIVE such a host ROCm wheels never demoted a venv that already HOLDS them, so treat the arch as CPU authority and let _ensure_cpu_torch() demote. KFD sysfs is consulted before inference and the declared arch, since a Van Gogh host reaches here with neither answering."""
    if IS_WINDOWS or IS_MACOS:
        return False
    if _explicit_torch_index_url() is not None:
        return False
    if "+rocm" not in _installed_torch_label_on_disk():
        return False
    # Declared arch LAST: this asks what silicon is PRESENT, and a stale UNSLOTH_ROCM_GFX_ARCH=gfx1030 on a real Van Gogh answered with a healthy arch. install.sh's "physical" mode agrees.
    return _miscomputing_arch_host()


def _windows_rocm_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD pip index URL for the given GPU arch, or None if unsupported."""
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    return _index_url_join(_ROCM_WINDOWS_INDEX_BASE, arch_family)


def _rocm_family_token(text: str) -> "str | None":
    """Family out of a 'rocm-sdk-libraries-<family>' name or requirement string."""
    _m = re.search(r"rocm[-_]sdk[-_]libraries[-_]([A-Za-z0-9][A-Za-z0-9._-]*)", text, re.IGNORECASE)
    if not _m:
        return None
    # Requirement strings carry a specifier and marker: "...-gfx120X-all==7.13.0; extra".
    return re.split(r"[=<>!~;,\[\]()\s]", _m.group(1))[0].strip().lower().replace("_", "-")


def _installed_rocm_wheel_family() -> str | None:
    """The AMD per-arch family the installed torch runs on, or None when nothing on disk says unambiguously. torch.version.hip only says "some ROCm build"; AMD's torch requires rocm[libraries], so the installed `rocm` meta-package names the family. Do NOT scan for rocm-sdk-libraries-* instead: the previous arch's runtime is never uninstalled, and mistaking that orphan for the active family reinstalls the stack every update."""
    try:
        from importlib import metadata
        for _req in metadata.requires("rocm") or []:
            _fam = _rocm_family_token(_req)
            if _fam:
                return _fam
    except Exception:
        pass
    # No `rocm` meta-package: fall back to the runtimes on disk, but only when exactly one is present, since with several nothing says which is active.
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


def _torch_requires_rocm_sdk() -> bool:
    """Whether the INSTALLED torch is an AMD per-arch build. pip leaves an orphan `rocm` behind when a generic ROCm torch is force-reinstalled over a per-arch one, and torch.version.hip is set on both, so a caller that SKIPS work on a family match must ask this too."""
    try:
        from importlib import metadata
        for _req in metadata.requires("torch") or []:
            # The distribution named exactly `rocm`, anchored so rocm-sdk-core and triton-rocm do not match. Case-insensitive: Requires-Dist keeps the author's spelling, and reading "ROCm[libraries]" as absent would call a per-arch build generic.
            if re.match(r"\s*rocm\s*(?:\[|[=<>!~;,]|$)", _req, re.IGNORECASE):
                return True
    except Exception:
        pass
    return False


def _detect_bnb_rocm_dll_ver() -> str | None:
    """Version suffix of the installed bitsandbytes libbitsandbytes_rocm{VER}.dll, or None. Uses find_spec, so it never imports bitsandbytes."""
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
    # Highest numeric suffix wins ("713" over "72"); glob order is not guaranteed.
    return max(all_vers, key = lambda v: int(v)) if all_vers else None


# Set right before the base unsloth install; read by _ensure_rocm_torch to drop a freshly pulled generic wheel on gfx906 while leaving a pre-existing source build alone.
_GFX906_BNB_ABSENT_BEFORE_BASE = False


def _bitsandbytes_installed() -> bool:
    """True if bitsandbytes is importable in the target venv. Fresh subprocess, so a package installed earlier this run is seen; checks the spec only."""
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
        # Strip all managed regions (even END-less, from an interrupted write), append one block.
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
    """True only if an actual AMD GPU is visible. Always False when an NVIDIA GPU is present: NVIDIA takes priority on mixed hosts, and every probe below can false-positive on ROCm tools installed beside its driver."""
    if _has_usable_nvidia_gpu():
        return False
    for cmd, check_fn in (
        # rocminfo: real gfx GPU ids only (gfx000 = CPU agent, "gfx11-generic" = ISA line).
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
        if cmd[0] == "amd-smi" and not _amd_smi_allowed():
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = _amd_smi_env() if cmd[0] == "amd-smi" else None,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            if check_fn(result.stdout):
                return True
    # sysfs KFD fallback for hosts without rocminfo/amd-smi. Non-AMD vendors are rejected: the NVIDIA open kernel module also registers KFD nodes.
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
                    # Require AMD vendor_id 4098 (0x1002); a missing properties file stays unconfirmed.
                    props_path = os.path.join(kfd_nodes, entry, "properties")
                    try:
                        with open(props_path, encoding = "utf-8") as fh:
                            props = fh.read()
                    except (OSError, UnicodeDecodeError):
                        continue
                    if not re.search(r"\bvendor_id\s+4098\b", props):
                        continue
                    return True
        except OSError:
            pass
    return False


def _has_usable_nvidia_gpu() -> bool:
    """True when an NVIDIA GPU is present and usable: nvidia-smi -L, falling back to /proc/driver/nvidia/gpus/ on Linux and to install.ps1's fixed driver locations on Windows, where nvidia-smi.exe is often off PATH. CUDA_VISIBLE_DEVICES "" or "-1" hides every device and neither probe honours it, so check it first."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return False

    def _lists_a_gpu(exe: str) -> bool:
        try:
            result = subprocess.run(
                [exe, "-L"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
            )
        except Exception:
            return False
        return result.returncode == 0 and "GPU " in result.stdout

    # A stale nvidia-smi on PATH exits non-zero listing nothing, so try every candidate: install.ps1 / setup.ps1 also gate the fixed-location fallback on the GPU check failing.
    candidates = []
    _path_exe = shutil.which("nvidia-smi")
    if _path_exe:
        candidates.append(_path_exe)
    if IS_WINDOWS:
        candidates.extend(
            (
                os.path.join(
                    os.environ.get("ProgramFiles", r"C:\Program Files"),
                    "NVIDIA Corporation",
                    "NVSMI",
                    "nvidia-smi.exe",
                ),
                os.path.join(
                    os.environ.get("SystemRoot", r"C:\Windows"),
                    "System32",
                    "nvidia-smi.exe",
                ),
            )
        )
    for _candidate in candidates:
        if _candidate != _path_exe and not os.path.isfile(_candidate):
            continue
        if _lists_a_gpu(_candidate):
            return True
    # Fallback: /proc/driver/nvidia/gpus/ has one subdir per GPU whatever nvidia-smi does.
    if sys.platform != "win32":
        try:
            gpu_dir = "/proc/driver/nvidia/gpus"
            if os.path.isdir(gpu_dir) and os.listdir(gpu_dir):
                return True
        except OSError:
            pass
    return False


# Which probe answered the last _detect_amd_gfx_codes() call: only rocminfo is subject to a visible-device mask, so the Strix reroute needs to know. None when stubbed.
_LAST_AMD_GFX_PROBE: "str | None" = None


def _detect_amd_gfx_codes(
    dedup: bool = True,
    ignore_hsa_override: bool = False,
    ignore_visible_masks: bool = False,
) -> list[str]:
    """The AMD gfx ISA strings visible to ROCm: rocminfo, then amd-smi for runtime-only Radeon hosts. dedup=False keeps one entry per DEVICE, because mask values are device ordinals and rocminfo repeats a token per agent, so split on agent headers first. Records the answering probe in _LAST_AMD_GFX_PROBE, since only rocminfo is mask-filtered. ignore_hsa_override strips HSA_OVERRIDE_GFX_VERSION, which ROCr applies in userland so rocminfo reports the SPOOFED ISA (#7331); ignore_visible_masks also strips the masks so the re-probe sees the whole machine."""
    global _LAST_AMD_GFX_PROBE
    _LAST_AMD_GFX_PROBE = None

    def _extract(text: str) -> list[str]:
        if dedup:
            codes = [f"gfx{c}" for c in re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())]
            return list(dict.fromkeys(codes))
        # One entry per agent / GPU section; fall back to dedup for flat output. amd-smi heads each device with a line-leading "GPU: N", and without that header two cards of one arch collapse and later ordinals read wrong.
        _sections = re.split(
            r"(?mi)^\s*\*+\s*$\s*agent\s+\d+\s*$|\bagent\s+\d+\b|\bdevice\s*#\s*\d+\b"
            r"|^[ \t]*gpu\s*[:\[]\s*\d+",
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
        _env = _amd_smi_env() if cmd[0] == "amd-smi" else None
        _strip = set()
        if ignore_hsa_override:
            _strip.add("HSA_OVERRIDE_GFX_VERSION")
        if ignore_visible_masks:
            _strip.update(("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"))
        if _strip & set(os.environ):
            # env=None means "inherit", so drop the variables from an explicit copy.
            _env = {
                k: v
                for k, v in (_env if _env is not None else os.environ).items()
                if k not in _strip
            }
        try:
            result = subprocess.run(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = _env,
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


# Arches the product name can establish whose wheels differ from the arch people spoof them to: RDNA 3.5 APUs ROCm did not support natively, so HSA_OVERRIDE_GFX_VERSION=11.0.0 became the circulated workaround.
_HSA_SPOOFABLE_PHYSICAL_GFX: frozenset[str] = frozenset({"gfx1151", "gfx1150", "gfx1152"})


# Arches the generic pytorch.org ROCm wheels carry, identical across every generic wheel the pins resolve. The rocm6.3 wheel lacks gfx1150/gfx1151 (the Strix reroute covers it), and torch 2.13.0+rocm7.1 adds gfx1103, so raising the <2.11 cap means rechecking this set.
_GENERIC_ROCM_WHEEL_GFX: frozenset[str] = frozenset(
    {
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)


# Arches whose only wheel route is unreachable once a second GPU is present, so they must never depose an integrated card: gfx906's rocm6.3 route needs it to be the sole detected arch.
_MIXED_HOST_UNROUTABLE: "frozenset[str]" = frozenset({"gfx906"})


def _gfx_route_on_host(gfx: "str | None", host_codes: "list[str] | None" = None) -> bool:
    """Whether an index can serve ``gfx`` ON THIS HOST: gfx906 answers yes in the abstract while its only route opens solely when it is the one arch present. ``host_codes`` is the machine BEFORE the ROCr layer, or a masked gfx906 would demote and then be refused the tag by a function seeing both cards."""
    return _gfx_has_a_wheel_route(gfx) and not (
        gfx in _MIXED_HOST_UNROUTABLE and len(set(host_codes or ())) > 1
    )


def _gfx_has_a_wheel_route(gfx: "str | None") -> bool:
    """Whether ANY index this installer can pick carries kernels for ``gfx``. One in neither (gfx1010 / RDNA 1) cannot be fixed by another index, so it must never depose a card that can."""
    return bool(gfx) and (gfx in _GENERIC_ROCM_WHEEL_GFX or gfx in _GFX_TO_AMD_INDEX_ARCH)


# Measured on the tags the pins resolve (rocm7.0+), but _ROCM_TORCH_INDEX also maps ROCm 6.0-6.4, whose wheels predate some arches: a host can resolve an old tag while the KERNEL names a new card. gfx1150/gfx1151 need an entry despite the Strix reroute, which is inactive on a bundled-runtime host reading 0.0.
_GENERIC_WHEEL_GFX_MIN_ROCM: "dict[str, tuple[int, int]]" = {
    # gfx950 has no _GFX_TO_AMD_INDEX_ARCH entry, so the reroute answers no for it; this entry is read by the tag choice, which can pick a generic tag that does carry it.
    "gfx950": (7, 0),
    "gfx1150": (7, 0),
    "gfx1151": (7, 0),
    "gfx1200": (6, 4),
    "gfx1201": (6, 4),
}


def _generic_rocm_wheel_lacks_kernels(
    gfx: "str | None", ver: "tuple[int, int] | None" = None
) -> bool:
    """Whether only an AMD per-arch index carries kernels for ``gfx``. Passing the host ROCm ``ver`` lets an arch the OLD tags predate be rerouted rather than installed without kernels."""
    if not gfx or gfx not in _GFX_TO_AMD_INDEX_ARCH:
        return False
    if gfx not in _GENERIC_ROCM_WHEEL_GFX:
        return True
    return ver is not None and _generic_tag_lacks_kernels(gfx, ver)


def _generic_tag_lacks_kernels(gfx: "str | None", ver: "tuple[int, int]") -> bool:
    """Whether the generic wheel ``ver`` resolves to predates ``gfx``: the tag question alone, so it also answers for arches _generic_rocm_wheel_lacks_kernels declines, which have a second answer -- take a newer tag."""
    _min = _GENERIC_WHEEL_GFX_MIN_ROCM.get(gfx or "")
    if _min is None:
        return False
    # The tag the version selects, not the version: a 6.3.9 host takes the rocm6.3 wheel. Below the oldest known index nothing resolves, and reading that as "support unknown" preserves exactly the build that cannot run.
    _tag_key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return _tag_key is None or _tag_key < _min


def _generic_only_target_below_floor(gfx: "str | None", ver: "tuple[int, int] | None") -> bool:
    """Whether a target with NO per-arch index sits on a generic tag that predates it: the repair question for gfx950 and friends, which _generic_rocm_wheel_lacks_kernels answers False for by design. An unreadable version answers True."""
    if not gfx or gfx in _GFX_TO_AMD_INDEX_ARCH or gfx not in _GENERIC_WHEEL_GFX_MIN_ROCM:
        return False
    return ver is None or _generic_tag_lacks_kernels(gfx, ver)


def _runtime_gfx_target(
    inferred_linux_gfx: "str | None",
) -> "tuple[str | None, list[str], str | None, list[str]]":
    """The selected gfx target, detected arches, corrected physical arch, and the machine as the probes saw it before ROCr filtering. Strongest first: ROCm userland probes, KFD sysfs, then the explicit / inferred arch, which matter because a runtime-only ROCm install ships neither rocminfo nor amd-smi."""
    # An empty (or "-1") mask selects NO GPU, deliberately. Decided before any probe runs: only ROCR_VISIBLE_DEVICES reaches rocminfo, so _pick_visible_index would otherwise map the no-GPU mask onto index 0 and reinstall a multi-GB stack for the hidden card.
    if _visible_masks_select_no_gpu():
        return None, [], None, []
    # An explicit arch outranks the probes, not merely the cases where they say nothing; the masks stay above it. Split on ":" as the probe normalization does, or a copied "gfx1151:sramecc-:xnack-" keys no routing table.
    _explicit_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":", 1)[0]
    if _explicit_gfx:
        # Returning early skips _hsa_spoofed_physical_gfx, so answer the spoof here: leaving HSA_OVERRIDE_GFX_VERSION set while installing per-arch wheels is #7331 exactly. Only when the override names a DIFFERENT arch.
        _spoofed = _explicit_gfx if _hsa_spoof_contradicts(_explicit_gfx) else None
        return _explicit_gfx, [_explicit_gfx], _spoofed, [_explicit_gfx]
    gfx_devices = _detect_amd_gfx_codes(dedup = False)
    # Keyed to the userland probe: ROCr spoofs that reading and no other.
    physical_gfx = _hsa_spoofed_physical_gfx(inferred_linux_gfx, gfx_devices)
    if physical_gfx is not None:
        gfx_devices = [physical_gfx]
    if not gfx_devices:
        # The kernel's own topology: one entry per GPU node, in node order.
        gfx_devices = _kfd_gfx_targets()
        # With no userland reading to distrust, the spoof check declined, but the runtime is still spoofed (#7331). amdkfd writes gfx_target_version and ROCr never touches it, so a single-arch kernel reading contradicting the override IS the corroborated spoof.
        if physical_gfx is None and len(set(gfx_devices)) == 1:
            _override_arch = _hsa_override_gfx_arch(os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
            if _override_arch is not None and _override_arch != gfx_devices[0]:
                physical_gfx = gfx_devices[0]
    if not gfx_devices and inferred_linux_gfx:
        # Nothing enumerated a device, so one guess is not a device list: _pick_visible_index's out-of-range rule would answer with the guess and commit a per-arch reinstall to it. Decline unless the arch was named outright.
        if not (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip() and _pick_visible_index(
            _INDEX_PROBE_LEN, warn = False, masks = _HIP_LAYER_MASKS
        ):
            _safe_print(
                f"   {_first_set_visible_mask()} selects a GPU past the only architecture\n"
                f"   this host could name ({inferred_linux_gfx}, inferred from the product\n"
                f"   name with no ROCm runtime to enumerate devices); which GPU it means\n"
                f"   cannot be read here, so the AMD per-gfx index is left alone.\n"
                f"   Set UNSLOTH_ROCM_GFX_ARCH to the arch you want wheels for.\n"
            )
            return None, [], None, []
        gfx_devices = [inferred_linux_gfx]
    # The machine as the probes saw it, before ROCr reduces it to a lone survivor: rocminfo runs on the ROCm user-mode stack, so a mask-selected MI50 beside a dGPU would read as single-architecture and be granted the rocm6.3 tag, downgrading a shared install on one session's mask.
    _probe_source = _LAST_AMD_GFX_PROBE
    if _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ:
        try:
            _unmasked = _detect_amd_gfx_codes(dedup = False, ignore_visible_masks = True)
        except Exception:
            _unmasked = []
        host_codes = list(dict.fromkeys(_unmasked or gfx_devices))
    else:
        host_codes = list(dict.fromkeys(gfx_devices))
    # The two mask layers COMPOSE: ROCr filters first and HIP indexes the survivors. Only rocminfo is renumbered by ROCr, so for amd-smi and KFD sysfs the ROCr layer is applied here first.
    rocr_applied = _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ
    _unlike_adapters = len(set(gfx_devices)) > 1
    if not rocr_applied:
        # amd-smi enumerates in DISCOVERY order while the masks index HIP/ROCr order from the KFD node id, and the two disagree on real hardware (MI350X SPX/NPS1). setup.sh translates through `amd-smi list -e`'s HIP_ID map; no such map is read here, so on unlike adapters any ordinal, set or unset, can name another card.
        _discovery_ordered = _probe_source == "amd-smi"
        if _discovery_ordered and _unlike_adapters:
            # Discovery order is unusable, the kernel's topology is not: KFD nodes ARE the order HIP and ROCr index (#9396). Only on an equal device count, since another length is a different view of the machine.
            _kfd_ordered = _kfd_gfx_targets()
            if len(_kfd_ordered) == len(gfx_devices):
                gfx_devices = _kfd_ordered
                _unlike_adapters = len(set(gfx_devices)) > 1
                _discovery_ordered = False
        gfx_devices, _rocr_unresolved = _rocr_visible_subset(gfx_devices)
        # A UUID names a device this cannot place. Judged against the list BEFORE the mask: dropping the tokens that did resolve can leave one arch standing and hide the ambiguity.
        if (_rocr_unresolved or _discovery_ordered) and _unlike_adapters:
            # The message below offers UNSLOTH_ROCM_GFX_ARCH as the way through, so honour it. Read from the environment, not inferred_linux_gfx, which also carries the weaker product guess.
            _named_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
            if _named_gfx:
                # The named arch leads the returned set: the caller reads that set to decide which arch lacks kernels, so a target missing from it is selected and then never repaired.
                return (
                    _named_gfx,
                    list(dict.fromkeys([_named_gfx, *gfx_devices])),
                    physical_gfx,
                    host_codes,
                )
            _why = (
                "ROCR_VISIBLE_DEVICES names a GPU by UUID"
                if _rocr_unresolved
                else "amd-smi reports GPUs in discovery order, which is not the order the\n"
                "   visible-device masks index,"
            )
            _safe_print(
                f"   {_why} and this host has more than one\n"
                f"   architecture ({', '.join(dict.fromkeys(gfx_devices))}); which one is\n"
                f"   selected cannot be read here, so the AMD per-gfx index is left alone.\n"
                f"   Set UNSLOTH_ROCM_GFX_ARCH to the arch you want wheels for.\n"
            )
            return None, [], None, host_codes
    runtime_gfx = (
        gfx_devices[_pick_visible_index(len(gfx_devices), masks = _HIP_LAYER_MASKS)]
        if gfx_devices
        else None
    )
    if runtime_gfx in _SHADOWING_INTEGRATED_GFX and not _visible_devices_pinned():
        # Unpinned mixed host: the wheel family is picked for ONE arch, so letting a leading APU decide strands the discrete card (#7776). gfx906 is excluded as a candidate, since its only route needs it to be the sole arch, so naming it strands BOTH cards.
        _others = [
            g
            for g in gfx_devices
            if g not in _SHADOWING_INTEGRATED_GFX and g not in _MIXED_HOST_UNROUTABLE
        ]
        # Prefer a discrete card the installer can serve: deposing a routable APU for one no index carries (an RDNA 1 dGPU beside a gfx1103) trades a repairable GPU for nothing.
        _routable = [g for g in _others if _gfx_has_a_wheel_route(g)]
        _candidates = _routable or ([] if _gfx_has_a_wheel_route(runtime_gfx) else _others)
        _discrete = _candidates[0] if _candidates else None
        if _discrete is not None:
            _safe_print(
                f"   multiple AMD GPUs detected "
                f"({', '.join(dict.fromkeys(gfx_devices))}); installing for {_discrete}\n"
                f"   instead of the integrated {runtime_gfx}. Set HIP_VISIBLE_DEVICES to the\n"
                f"   GPU index you want (then rerun) to install for a different device.\n"
            )
            runtime_gfx = _discrete
    return runtime_gfx, list(dict.fromkeys(gfx_devices)), physical_gfx, host_codes


def _hsa_override_gfx_arch(value: "str | None") -> "str | None":
    """gfx arch named by an HSA_OVERRIDE_GFX_VERSION value, or None. ROCr reads a major.minor.stepping triple and builds gfx<major><minor><stepping in hex>: 11.5.1 -> gfx1151, 9.0.10 -> gfx90a."""
    if not value:
        return None
    # [0-9] rather than str.isdigit()/\\d, both of which accept non-ASCII digits ("١١.0.0" would read as 11.0.0 here and be rejected by install.sh's awk).
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value.strip()):
        return None
    major, minor, step = (int(p) for p in value.strip().split("."))
    # Steppings are a single hex nibble; anything wider is not a real target.
    if not (0 <= step <= 15) or major <= 0 or minor > 9:
        return None
    return f"gfx{major}{minor}{step:x}"


def _kfd_gfx_targets() -> list[str]:
    """gfx arches of the AMD GPUs the KERNEL sees, one per GPU node in node order. gfx_target_version is written by amdkfd, so it is immune to HSA_OVERRIDE_GFX_VERSION and is the ground truth for #7331; encoding is major*10000 + minor*100 + stepping in hex. The vendor_id 4098 guard keeps NVIDIA's open-driver KFD nodes out."""
    if sys.platform == "win32":
        return []
    nodes_dir = "/sys/class/kfd/kfd/topology/nodes"
    targets: list[str] = []
    try:
        entries = sorted(os.listdir(nodes_dir), key = lambda e: (len(e), e))
    except OSError:
        return []
    for entry in entries:
        try:
            with open(os.path.join(nodes_dir, entry, "properties"), encoding = "utf-8") as fh:
                props = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not re.search(r"\bvendor_id\s+4098\b", props):
            continue
        _m = re.search(r"\bgfx_target_version\s+(\d+)\b", props)
        if not _m:
            continue
        raw = int(_m.group(1))
        if raw <= 0:
            continue
        major, minor, step = (raw // 10000) % 100, (raw // 100) % 100, raw % 100
        if major <= 0 or minor > 9 or step > 15:
            continue  # not a shape the gfx name concatenation can represent
        targets.append(f"gfx{major}{minor}{step:x}")
    return targets


def _hsa_spoofed_physical_gfx(
    inferred_gfx: "str | None", gfx_devices: "list[str] | None" = None
) -> "str | None":
    """Physical arch when the ISA probe is an HSA_OVERRIDE_GFX_VERSION spoof (#7331). None ("believe the probe") unless all of: the override is set; the product name inferred a spoofable arch and the probe reports a DIFFERENT one; the probe saw exactly one arch; the variable names EXACTLY the reported arch; and a source the override cannot reach corroborates it (KFD sysfs, then rocminfo re-probed without it). Corroboration is REQUIRED: a truthful gfx1100 dGPU in a Ryzen AI Max chassis looks identical, and rerouting it is worse than #7331."""
    global _LAST_AMD_GFX_PROBE

    raw = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if not raw or not inferred_gfx or inferred_gfx not in _HSA_SPOOFABLE_PHYSICAL_GFX:
        return None
    if gfx_devices is None:
        gfx_devices = _detect_amd_gfx_codes(dedup = False)
    if len(set(gfx_devices)) != 1:
        return None
    probed = gfx_devices[0]
    if probed == inferred_gfx:
        return None
    # Only the arch the variable names can be a spoof of that variable's doing.
    if _hsa_override_gfx_arch(raw) != probed:
        return None

    _safe_print(
        f"   HSA_OVERRIDE_GFX_VERSION={raw} is set; ROCm reports {probed} but this host's\n"
        f"   product name is {inferred_gfx}. Checking whether the ISA is being spoofed.\n"
    )

    def _confirm(physical: "list[str]", source: str) -> "str | None":
        """Decisive only when the source names the product arch and nothing else: a second arch means the single-arch premise was wrong, so decline. install.sh compares the same two strings."""
        if physical == [inferred_gfx]:
            _safe_print(
                f"   {source} reports {inferred_gfx} -- {probed} is a spoof of the "
                f"physical arch.\n"
            )
            return inferred_gfx
        # Say so rather than leaving "Checking whether..." hanging: on a real gfx1100 card in a Ryzen AI Max chassis this is the CORRECT outcome.
        _safe_print(
            f"   {source} does not corroborate a spoof "
            f"({physical or 'no answer'}); keeping {probed}.\n"
        )
        return None

    # 1. The kernel, which the override cannot reach. Decisive either way: if it answers at all, no weaker source overrules it.
    kfd = _kfd_gfx_targets()
    if kfd:
        return _confirm(kfd, "KFD topology sysfs")

    # 2. The runtime, asked again without the override and without the visible masks, so a mask cannot hide the second GPU that would veto the correction.
    _saved_probe = _LAST_AMD_GFX_PROBE
    try:
        reprobed = _detect_amd_gfx_codes(
            dedup = False, ignore_hsa_override = True, ignore_visible_masks = True
        )
    except Exception:
        reprobed = []
    finally:
        _LAST_AMD_GFX_PROBE = _saved_probe
    # A re-probe that still answers `probed` is evidence FOR the probe, which is why a genuine gfx1100 dGPU in a Ryzen AI Max chassis keeps its own wheels.
    return _confirm(list(dict.fromkeys(reprobed)), "rocminfo with HSA_OVERRIDE_GFX_VERSION unset")


def _hsa_spoof_contradicts(gfx: "str | None") -> bool:
    """True when HSA_OVERRIDE_GFX_VERSION names an arch other than ``gfx``: per-arch wheels carry one arch's code objects and ROCr builds the agent's ISA name from this variable."""
    _override_arch = _hsa_override_gfx_arch(os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
    return bool(gfx) and _override_arch is not None and _override_arch != gfx


def _clear_confirmed_hsa_spoof(physical_gfx: str) -> None:
    """Drop a CONFIRMED HSA_OVERRIDE_GFX_VERSION spoof from this process's env: ROCr reads it afresh in every LATER process, so leaving it set hands the new wheel a device matching none of its code objects (#7331). A shell profile that exports it will set it again next login, so name the variable and say to remove it."""
    if os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None) is None:
        return
    _safe_print(
        f"   Clearing HSA_OVERRIDE_GFX_VERSION for the rest of this install: the\n"
        f"   {physical_gfx} wheels carry {physical_gfx} kernels, so the runtime has to\n"
        f"   report the real arch. Remove the export from your shell profile\n"
        f"   (~/.bashrc, ~/.profile) as well, or the next terminal restores it.\n"
    )


# First-set-wins order, as _pick_visible_index documents below.
_VISIBLE_DEVICE_MASKS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# The HIP layer alone: CUDA_VISIBLE_DEVICES is HIP's alias on AMD, while ROCR_VISIBLE_DEVICES is the layer BENEATH, applied by _rocr_visible_subset.
_HIP_LAYER_MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# A device count no host reaches, for asking what index a mask NAMES rather than which entry of a real list it selects: any ordinal below it resolves to itself, not to the 0 fallback.
_INDEX_PROBE_LEN = 1 << 20


def _rocr_visible_subset(gfx_devices: "list[str]") -> "tuple[list[str], bool]":
    """Apply the ROCr layer to a device list no probe filtered; returns the survivors and whether any token went unresolved. ROCr filters below HIP and neither amd-smi nor KFD sysfs honours it, so a HIP index over their whole-machine list names a GPU the runtime does not expose. The mask may MIX indices and UUIDs, and a UUID resolves to no position rather than being guessed at."""
    _raw = (os.environ.get("ROCR_VISIBLE_DEVICES") or "").strip()
    if not _raw or not gfx_devices:
        return gfx_devices, False
    _kept: "list[str]" = []
    _unresolved = False
    for _tok in _raw.split(","):
        _tok = _tok.strip()
        try:
            _idx = int(_tok)
        except ValueError:
            _unresolved = True  # a UUID: this names a device, but not a position
            continue
        if 0 <= _idx < len(gfx_devices):
            _kept.append(gfx_devices[_idx])
    # An out-of-range index keeps the whole list, deliberately: _pick_visible_index warns and falls back to GPU 0 (as setup.ps1 does), and reading a typo as "no GPU" withdraws the repair from the hosts this exists for.
    return (_kept or gfx_devices), _unresolved


def _visible_masks_select_no_gpu() -> bool:
    """True when a set visible-device mask exposes NO GPU at either layer: ROCr filters beneath HIP, so an empty or -1 mask on either leaves nothing to target. CUDA_VISIBLE_DEVICES is read only when HIP itself is unset."""
    _hip = "HIP_VISIBLE_DEVICES" if "HIP_VISIBLE_DEVICES" in os.environ else "CUDA_VISIBLE_DEVICES"
    return any(
        (os.environ.get(_mask) or "").strip() in ("", "-1")
        for _mask in ("ROCR_VISIBLE_DEVICES", _hip)
        if _mask in os.environ
    )


def _first_set_visible_mask() -> "str | None":
    """Name of the visible-device variable in force, first-set-wins, or None."""
    for _env in _VISIBLE_DEVICE_MASKS:
        if os.environ.get(_env) is not None:
            return _env
    return None


# Set by _ensure_rocm_torch() on success; suppresses the post-install AMD warning.
_rocm_windows_torch_installed: bool = False


def _install_bnb_windows_rocm() -> bool:
    """Install AMD Windows BNB, pre-release wheel first. The wheel's filename version does not match its metadata and uv mangles the install even under UV_SKIP_WHEEL_FILENAME_CHECK, so force plain pip. If that URL is blocked, PyPI's win_amd64 wheel ships libbitsandbytes_rocm{714,72}.dll from 0.50.0 on."""
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
    # BNB_ROCM_VERSION from the DLL suffix (the wheel may ship "72" while torch reports 7.13).
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
    # venv Scripts (hipInfo.exe from the AMD torch wheel) on PATH, else bnb logs a stray ERROR.
    _scripts_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")) and not shutil.which(
        "hipinfo.exe"
    ):
        os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")
    return True


def _nvidia_smi_path() -> "str | None":
    """nvidia-smi from PATH, falling back to the canonical Linux install path a stripped-down PATH (systemd units, cron) can miss."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.path.isfile("/usr/bin/nvidia-smi"):
        exe = "/usr/bin/nvidia-smi"
    return exe


def _nvidia_compute_sms(exe: str) -> "list[int] | None":
    """Every GPU's sm_NN per nvidia-smi, or None when the inventory is unreadable: one unparseable row poisons the whole answer, so a partial reading can never drive a wheel decision."""
    try:
        result = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
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


# PyTorch 2.11's cu126 spans sm_50-90 (Maxwell to Hopper) with no PTX above that, and it is the fallback family, so a Kepler or Blackwell card leaves the host uncovered.
_CU126_SM_RANGE = (50, 90)


def _cuda_family_sm_range(family: str, torch_release: str = "") -> "tuple[int, int] | None":
    """Supported SM span for a CUDA wheel family; cu128/cu129 include sm_70 only for torch 2.8-2.10, and an empty release models a fresh torch 2.11."""
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
    """Use cu126 when it covers every physical GPU missed by the selected family. CUDA_VISIBLE_DEVICES is intentionally ignored, and non-x86_64 hosts keep the driver-derived family."""
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
    """The pytorch.org CUDA wheel index URL for the host's NVIDIA driver, mirroring install.sh::get_torch_index_url so `studio update` repairs to a fresh install's family. Explicit overrides first, so a headless/CI install never lets the host GPU decide; the driver version is only an upper bound, so the GPU architectures can cap the result at cu126."""
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
                encoding = "utf-8",
                errors = "replace",
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
    """The wheel index URL pinned via UNSLOTH_TORCH_INDEX_URL / _FAMILY, so repair helpers honour the pin instead of re-probing the GPU. Mirrors install.sh::get_torch_index_url."""
    url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if url:
        return _trim_index_path_slashes(url)
    family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    if family:
        return f"{_PYTORCH_WHL_BASE}/{family.strip('/')}"
    return None


def _is_pip_rocm_family_leaf(leaf: str) -> bool:
    """True when a leaf names a pip ROCm family: an EXACT rocm<digits>[.<digits>] or gfx leaf. A suffixed leaf (rocm-rel-7.2.1) is a custom pin the verbatim path owns. Mirrors install.sh / setup.ps1."""
    # gfx must be followed by a digit; a gfx-private custom leaf is a verbatim pin.
    return bool(re.fullmatch(r"rocm\d+(?:\.\d+)?", leaf)) or bool(re.match(r"gfx\d", leaf))


def _explicit_rocm_torch_index_url() -> "str | None":
    """The pinned wheel index URL when it names a pip ROCm family (rocm<d>/gfx*), else None."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _is_pip_rocm_family_leaf(_torch_index_leaf(url)) else None


def _rocm_pin_family_mismatch(pin_url: str, installed_ver: str) -> bool:
    """True when an explicit ROCm pin names a different family than the installed ROCm torch. Mirrors setup.ps1's stale-venv comparison; same pin-leaf cases as _ensure_rocm_torch."""
    leaf = _torch_index_leaf(pin_url)
    # Pinned ROCm version. The family classifier accepts a major-only rocm<d> leaf too, so the minor is optional and a major-only pin compares on the major alone.
    _pin_rocm = re.match(r"^rocm(\d+)(?:\.(\d+))?", leaf)
    _pin_major = int(_pin_rocm.group(1)) if _pin_rocm else None
    _pin_ver = (
        (int(_pin_rocm.group(1)), int(_pin_rocm.group(2)))
        if _pin_rocm and _pin_rocm.group(2) is not None
        else None
    )
    # Installed +rocmX.Y version; a THREE-part +rocmA.B.C tag is the AMD per-arch (repo.amd.com/gfx*) signature, versus a two-part pytorch.org wheel.
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
        # A repin BETWEEN per-arch leaves keeps every shape the version string carries, so the pin the user just edited would read as satisfied. The installed `rocm` meta-package names the family; None is unknowable, so this can only ADD a mismatch.
        _family = _installed_rocm_wheel_family()
        if _family is not None and _family != leaf:
            return True
        # 2.11-allowlist arches expect the AMD per-arch wheel. Asked before the family can settle it, because these leaves have a FLOOR: a matching family on a 2.10 build is the _grouped_mm bug.
        if leaf in _ROCM_GFX_TORCH211_LEAVES:
            return not (_inst_is_211 and _inst_is_perarch)
        # Decisive the other way on leaves with no floor: the heuristic below reads any 2.11 build as a mismatch, so without this a correctly pinned gfx110X host force-reinstalls on every update.
        if _family is not None and _inst_is_perarch:
            return False
        # Non-2.11 gfx leaf (<2.11 specs): mismatch on an untagged wheel or torch 2.11+.
        return (not _inst_has_rocm) or _inst_is_211

    # Major-only rocm pin (rocm7): compare majors only. There is no pinned minor to compare, and the 2.11-line fallback below would invert both verdicts.
    if _pin_major is not None and _pin_ver is None:
        if _inst_ver is not None:
            return _inst_ver[0] != _pin_major
        # Untagged wheel never satisfies a ROCm pin; a +rocm tag with an unreadable version is accepted (matching the lenient fallback below).
        return not _inst_has_rocm

    # rocmX.Y pin. Only KNOWN-2.11 rocm is the 2.11 line (no speculative floor).
    _pin_is_211 = _pin_ver in _ROCM_KNOWN_TORCH211_VERSIONS if _pin_ver is not None else False
    if _pin_ver is not None and _inst_ver is not None:
        # Both readable: exact (major, minor) compare, so a rocm7.2 pin over +rocm7.13.x reinstalls the pinned wheel.
        if _pin_ver != _inst_ver:
            return True
        # Same family: a KNOWN-2.11 pin whose release drifted off 2.11 (2.12+rocm7.2) violates the spec, so reinstall to floor (exact compare, not >=2.11).
        if _pin_is_211 and _inst_rel is not None:
            if (int(_inst_rel.group(1)), int(_inst_rel.group(2))) != (2, 11):
                return True
        return False
    # rocm pin, unreadable installed version: compare on the 2.11 line, but an untagged wheel never satisfies a rocmX.Y pin.
    if not _inst_has_rocm:
        return True
    return _pin_is_211 != _inst_is_211


# Intel XPU wheels, own range: the xpu index serves past our tested ceiling, and the floor is 2.6 because unsloth/models/_utils.py raises at import below it. Kept in step with install.sh by tests/sh/test_xpu_torch_spec_parity.sh.
_XPU_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.6,<2.11.0",
    "torchvision>=0.21,<0.26.0",
    "torchaudio>=2.6,<2.11.0",
)


def _explicit_xpu_torch_index_url() -> "str | None":
    """The pinned index URL when it names the XPU family, else None. Intel support is a pin, never autodetection."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _torch_index_leaf(url) == "xpu" else None


def _explicit_cpu_torch_index_url() -> "str | None":
    """The pinned index URL when it names the CPU family, else None; an explicit CPU pin is authoritative (see _ensure_cpu_torch)."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _torch_index_leaf(url) == "cpu" else None


def _is_cuda_family_leaf(leaf: str) -> bool:
    """True only for a real CUDA family leaf ("cu" + digits): a bare startswith("cu") would match "custom", and the EXACT match keeps "cu128-private" on the verbatim path."""
    return re.fullmatch(r"cu[0-9]+", leaf) is not None


def _explicit_cuda_torch_index_url() -> "str | None":
    """The pinned index URL when it names a CUDA family, so _ensure_cuda_torch treats only a CUDA pin as authority to override the NVIDIA-presence gate."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    return url if _is_cuda_family_leaf(_torch_index_leaf(url)) else None


def _explicit_unknown_family_torch_index_url() -> "str | None":
    """The pinned index URL when its leaf names NO known family: version-tag heuristics cannot judge a private mirror, so the repair helpers must leave it alone. Matches install.sh / setup.ps1 / install.ps1."""
    url = _explicit_torch_index_url()
    if url is None:
        return None
    leaf = _torch_index_leaf(url)
    if _is_pip_rocm_family_leaf(leaf) or leaf == "cpu" or _is_cuda_family_leaf(leaf):
        return None
    return url


def _ensure_cuda_torch() -> None:
    """Repair a venv whose torch is a ROCm build on an NVIDIA host: a torch+rocm wheel satisfies the version constraint, so nothing else force-reinstalls it. Also repairs a CUDA torch whose family ships no kernels for the host's GPUs; healthy CUDA and deliberate CPU torch are untouched."""
    # Respect install.sh's backend: only "" (standalone update) or "cuda" force CUDA wheels.
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
    # An explicit CUDA pin commits to CUDA wheels and skips ALL GPU gates below.
    _cuda_pinned = _explicit_cuda_torch_index_url() is not None
    # CUDA_VISIBLE_DEVICES="" / "-1" hides the GPU; honour it unless a CUDA index is pinned.
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not _cuda_pinned and _cvd is not None and _cvd.strip() in ("", "-1"):
        return
    # Only NVIDIA hosts carry CUDA torch (the CUDA pin overrides this gate too).
    if not _cuda_pinned and not _has_usable_nvidia_gpu():
        return

    # Classify the installed torch: "hip" (poisoning signature), "cuda" or "cpu". Un-importable means missing or broken, which the base install owns unless a CUDA index is pinned.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        return
    if not _importable:
        # torch present but cannot import: an explicit CUDA pin forces this pass and the base update will not reinstall an already-installed torch, so reinstall from the pin.
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
    if _version is None:
        # Nothing readable came back, so classify nothing: this is where the per-path probe returned on empty stdout.
        return
    # marker | +cuXXX local tag | release | family from torch.version.cuda. The last is the only CUDA clue an untagged wheel gives: PyPI forbids the local +cuXXX version.
    _ver = _version.lower()
    _cu_match = re.search(r"\+(cu\d+)", _ver)
    _marker = "hip" if (_hip or "rocm" in _ver) else ("cuda" if _cuda else "cpu")
    _installed_cu = _cu_match.group(1) if _cu_match else ""
    _installed_release = _ver.split("+", 1)[0]
    _runtime_cu = ("cu" + _cuda.replace(".", "")) if _cuda else ""
    # Reinstall on a ROCm build on an NVIDIA host, on a pinned CUDA index with the wrong family, or when the installed family ships no kernels for these GPUs. A healthy match, or a CPU wheel with no CUDA pin, is left alone.
    _pin = _explicit_torch_index_url()
    _pin_leaf = _torch_index_leaf(_pin) if _pin else ""
    _pinned_cuda = _is_cuda_family_leaf(_pin_leaf)
    index_url: "str | None" = None
    if _marker == "hip":
        _why = "torch is a ROCm build on an NVIDIA host"
    elif _marker == "cpu" and _pinned_cuda:
        _why = "torch is a CPU build but an explicit CUDA index is pinned"
    elif _marker == "cuda" and _pinned_cuda and _installed_cu != _pin_leaf:
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
    """Install XPU torch when an explicit XPU pin is set but the venv has another build: `studio update` runs setup.sh, never install.sh, and an xpu leaf names no family the cuda/rocm helpers know, so the CPU wheel would survive the pin forever. Windows is excluded (setup.ps1 owns torch there); macOS has no XPU."""
    if NO_TORCH or IS_MACOS or IS_WINDOWS:
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None:
        return

    # Un-importable either way installs from the pin below. One shared probe bounds it.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        # Inconclusive, so ask the disk, which answers without loading SYCL. An unsupported or missing wheel does need the reinstall; a supported one means the Intel DRIVER is stalled, which no reinstall fixes.
        if _xpu_wheel_supported_on_disk():
            _safe_print(
                _red(
                    "   torch did not respond in time; the installed XPU build is supported, "
                    "so this is the Intel GPU compute driver -- update it and re-run"
                )
            )
            return
        _why = "torch could not be probed"
    elif _importable:
        if _version is None:
            return  # unreadable -- the base install step handles a missing torch
        # Flavour AND range: a migrated 2.5+xpu venv is broken, not correct, so the tag alone is not enough. Range matches _XPU_TORCH_PKG_SPEC.
        _ver = _version.lower()
        _rel = _ver.split("+")[0].split(".")
        _n = tuple(int(x) for x in _rel[:2] if x.isdigit())
        if "+xpu" in _ver and len(_n) == 2 and (2, 6) <= _n < (2, 11):
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
    """torch's full version string read OFF DISK: importlib.metadata drops the local label, and `import torch` loads the SYCL runtime, which can block indefinitely on a wedged Intel driver. find_spec locates the package without executing it."""
    try:
        # torch may have been installed earlier in THIS run, after the path finders cached site-packages' listing.
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
    """True when torch ON DISK is a +xpu wheel in the supported range: the probe's flavour-and-range test off version.py, so it answers when `import torch` cannot. Floor 2.6, because _utils.py raises at import below it."""
    label = _installed_torch_version_label().lower()
    if "+xpu" not in label:
        return False
    nums = tuple(int(p) for p in label.split("+")[0].split(".")[:2] if p.isdigit())
    return len(nums) == 2 and (2, 6) <= nums < (2, 11)


def _ensure_venv_pip() -> bool:
    """Make `python -m pip` work in the target venv: `uv venv` is created without --seed, so a fresh venv has no pip. Mirrors install.sh's bootstrap."""

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
    """Replace generic Triton with the XPU build torch asks for: both own the top-level `triton` package and a pinned +xpu torch pulls BOTH, so the CUDA-oriented build can land last and torch.compile loads the wrong library. Lives here so one copy covers install and `studio update`; on Windows setup.ps1 performs the same swap afterwards."""
    if NO_TORCH or IS_MACOS:
        return
    if IS_WINDOWS and os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG", "").strip():
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None:
        # A one-shot pin is gone by the next plain update but its +xpu wheel is not, and a dependency pass can pull generic triton back in, so the INSTALLED wheel is the pin. setup.sh keys its bnb floor on the same signal.
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
    # Act only when generic triton is present AND torch asks for an XPU triton; anything else means torch is not the +xpu wheel this assumes.
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

    # Fetch, THEN uninstall, THEN install from the file: the shared paths live in generic triton's OWN record, so uninstalling last deletes what the XPU build just wrote. Pre-fetching stops a dead mirror stranding the venv; uv has no `pip download`.
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
                # Same scrub every pinned install gets: PIP_NO_INDEX would ignore --index-url, and PIP_EXTRA_INDEX_URL / PIP_FIND_LINKS are consulted in addition to it, so an inherited environment could serve the wheel from somewhere the pin never named.
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
            # A read-only or locked venv leaves generic triton REGISTERED, and installing over it would let a later uninstall delete the shared files again. Change nothing.
            _safe_print(
                _red(
                    f"   could not remove generic triton {generic}; leaving it in place -- it "
                    "shadows torch XPU triton, so torch.compile will not use the XPU"
                )
            )
            return
        # Past this point the venv has NO triton: the uninstall took the shared top-level files. pip_install, not pip_install_try -- a warning would let the caller write a completion manifest over a broken torch.compile, which the next update fast-paths past.
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
    """torch.__version__ from torch/version.py, launching no interpreter: `import torch` can block indefinitely on a wedged Intel driver, which is the host an explicit pin exists to rescue."""
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
    """GPU build by local label alone. Weaker than the probe (which also reads torch.version.hip/cuda), so used only when the probe could not run."""
    return "+xpu" in label or "+rocm" in label or bool(re.search(r"\+cu\d+", label))


def _ensure_cpu_torch() -> None:
    """Reinstall CPU torch when CPU is authoritative but the venv has a GPU build; _ensure_cuda/rocm_torch treat a CPU backend as a skip. Authority is an EXPLICIT pin, or an AMD arch measured to miscompute under ROCm (see _rocm_miscomputing_host)."""
    if NO_TORCH:
        return
    pin = _explicit_cpu_torch_index_url()
    _reason = "an explicit CPU index is pinned"
    if pin is None and _rocm_miscomputing_host():
        pin = f"{_PYTORCH_WHL_BASE}/cpu"
        _reason = "this AMD arch computes incorrectly under ROCm (studio/ROCM_RDNA2_APU.md)"
    if pin is None:
        return

    # Classify the torch family. Un-importable means missing or broken, and the explicit CPU pin reinstalls it below.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        # A hung import is the wedged-driver case this pin exists to rescue, so returning here made the pin a no-op there. Classify off disk, and only go on for a GPU label: a merely slow CPU-only box must not reinstall torch every update.
        if not _is_gpu_torch_label(_installed_torch_label_on_disk()):
            return
    if not _ran or not _importable:
        # torch present but cannot import. The explicit CPU pin forces this pass and the base update will not reinstall an already-installed torch, so reinstall from the pin (self-resolving, no loop).
        _torch_pkg, _vision_pkg, _audio_pkg = _CPU_TORCH_PKG_SPEC
        _safe_print(
            f"   torch cannot import and {_reason} -- reinstalling "
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
    if _version is None:
        return  # unreadable -- the base install step handles a missing torch
    # '+xpu' too: an XPU wheel sets neither torch.version.cuda nor .hip, so without it a working Intel build reads as "cpu" and the CPU pin does nothing. torch.version.xpu beside it for untagged builds, since _installed_flavor_tag_now reads that marker and the two must agree.
    _ver = _version.lower()
    _is_gpu_build = (
        bool(_hip)
        or "rocm" in _ver
        or bool(_cuda)
        or bool(re.search(r"\+cu\d+", _ver))
        or "+xpu" in _ver
        or bool(_TORCH_RUNTIME_XPU)
    )
    if not _is_gpu_build:
        return  # already a CPU build

    _safe_print(
        f"   torch is a GPU build but {_reason} -- reinstalling "
        f"CPU torch from {_strip_index_url_credentials(pin)}"
    )
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


def _torch_flavor_tag(version: str) -> str:
    """Classify torch.__version__ into the installers' flavor vocabulary (+cuNNN, rocm, xpu, cpu). MUST match install.ps1's ConvertTo-TorchFlavorTag and setup.ps1's stale-venv probe, which produce the tags this is compared against; "" means classification failed, not "cpu". Untagged reads as "cpu" because PyPI forbids the local +cuNNN label, so an untagged wheel is the PyPI build."""
    value = str(version).strip().lower()
    if not value:
        return ""
    match = re.search(r"\+(cu\d+)", value)
    if match:
        return match.group(1)
    if "+rocm" in value:
        return "rocm"
    if "+xpu" in value:
        return "xpu"
    return "cpu"


def _gpu_family_from_runtime_markers(hip: str, cuda: str) -> str:
    """Which GPU family an untagged wheel's runtime markers name. torch.version.xpu alongside .hip and .cuda: omitting it let an explicit /cpu pin over such a wheel compare equal and record a PINNED cpu flavor for an XPU venv."""
    if hip:
        return "rocm"
    if cuda:
        return "cuda"
    return "xpu"


def _torch_build_is_gpu() -> bool:
    """Whether the installed torch can use a GPU at all, on the evidence available. Weaker and more forgiving than _torch_flavor_tag, and used only for the FAIL verdict in _ensure_expected_torch_flavor: a wrong family is worth a reinstall, but only a build with no GPU support whatsoever is worth failing the update over. An answer that never arrived (a wedged driver hanging `import torch`) falls back to the on-disk label and then reads as a GPU build, since ambiguity must not fail an update."""
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran and _importable and _version:
        return (
            _is_gpu_torch_label(_version.lower())
            or bool(_hip)
            or bool(_cuda)
            # torch.version.xpu, for the same reason .cuda and .hip are here: an untagged source, conda or private-index XPU build carries its runtime there and nowhere else.
            or bool(_TORCH_RUNTIME_XPU)
        )
    label = _installed_torch_label_on_disk()
    return (not label) or _is_gpu_torch_label(label)


def _expected_torch_flavor_tag() -> str:
    """The torch flavor this venv is SUPPOSED to hold, or "" when nothing can say. Most authoritative first: (1) UNSLOTH_EXPECTED_TORCH_TAG, the setup script's own answer, exported immediately before it hands over; (2) an explicit index pin whose family this vocabulary can name, since a pin is the instruction for THIS run and resolving the manifest first let a cu128 pin lose to a cu124 manifest and repair from the PUBLIC cu124 index; (3) the flavor the last completed install recorded in the manifest, read at import because install_python_stack() drops the manifest before the dependency pass; (4) a live probe, for a hand-run script, where only an NVIDIA host or a pin may expect a GPU build."""
    env = os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG", "").strip().lower()
    if env:
        return env
    pin = _explicit_torch_index_url()
    if pin is not None:
        leaf = _torch_index_leaf(pin)
        # "rocm" names every AMD leaf (rocm6.4, gfx1151); an unreadable one falls through.
        if _is_pip_rocm_family_leaf(leaf):
            return "rocm"
        if _is_cuda_family_leaf(leaf) or leaf in ("xpu", "cpu"):
            return leaf
    # A resolved backend is a stated choice, not a probe result: an AMD host taking setup.sh's documented UNSLOTH_TORCH_BACKEND=cpu recorded nothing here, because the NVIDIA probe answers "" for it, and the next launch called the deliberate install broken. Only when the family agrees with the wheel actually installed, and ahead of the manifest, which describes the PREVIOUS install.
    if _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu"):
        _installed = _torch_flavor_tag(_installed_torch_version_label())
        # "cuda" names a family, not a leaf, so the wheel's own cu tag is the flavor. Without this arm a REMOVED CPU pin lived on: install.sh installs a CUDA wheel, the manifest still says cpu, and a later update that swaps in a CPU wheel reads as working as asked.
        if _TORCH_BACKEND == "cuda":
            if _is_cuda_family_leaf(_installed):
                return _installed
        elif _installed == _TORCH_BACKEND:
            return _TORCH_BACKEND
    # An unknown-family pin (a corporate /simple mirror, /current) was applied verbatim and nothing below it can name a family for this venv. The setup handover above still outranks this, because it describes the index this run actually installed from.
    if _explicit_unknown_family_torch_index_url() is not None:
        return ""
    if _RECORDED_TORCH_TAG:
        return _RECORDED_TORCH_TAG
    # An absent NVIDIA GPU with no pin means no CUDA expectation exists to enforce.
    if _explicit_torch_index_url() is None and not _has_usable_nvidia_gpu():
        return ""
    return _torch_index_leaf(_detect_cuda_torch_index_url())


def _expected_torch_flavor_is_explicit() -> bool:
    """Whether the expectation came from someone SAYING so rather than from a probe: true for the setup handover, an explicit pin, and the last completed install's record. False when only the live hardware probe can answer, which is the one case a visibility mask has any business overruling."""
    if os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG", "").strip():
        return True
    if _explicit_torch_index_url() is not None:
        return True
    return bool(_RECORDED_TORCH_TAG)


def _recordable_torch_flavor_tag(resolved: str) -> str:
    """The flavor worth writing to the manifest, or "" when nothing is: normally the flavor this run resolved, falling back to the previous install's. An explicit pin whose leaf names no family breaks that fallback, since the old record describes a venv that no longer exists and carrying it forward would hand a later unpinned run a flavor to "repair" the mirror's build back to."""
    if resolved:
        return resolved
    if _explicit_unknown_family_torch_index_url() is not None:
        return ""
    return _RECORDED_TORCH_TAG or ""


def _index_leaf_flavor_family(leaf: str) -> str:
    """The flavor family a pip index leaf names: cpu, xpu, rocm, cuda, or "" for none."""
    leaf = (leaf or "").strip().lower()
    if leaf in ("cpu", "xpu"):
        return leaf
    if _is_pip_rocm_family_leaf(leaf):
        return "rocm"
    if _is_cuda_family_leaf(leaf):
        return "cuda"
    return ""


def _flavor_tag_family(tag: str) -> str:
    """The family a flavor tag belongs to. cu124 and cu128 are both "cuda"."""
    tag = (tag or "").strip().lower()
    return "cuda" if tag.startswith("cu") else tag


def _expected_torch_flavor_was_pinned(flavor: str = "") -> bool:
    """Whether ``flavor`` was NAMED by whoever ran this install. Distinct from _expected_torch_flavor_is_explicit(), which counts setup.ps1's handover variable: setup.ps1 publishes that for an AUTOMATIC /cpu choice on a GPU-less host exactly as for a pinned one. An index pin, an index family and UNSLOTH_TORCH_BACKEND each answer, and each only for the family it NAMES: setup.ps1 falls back to the CPU index when a pinned ROCm/XPU install fails while the GPU pin is still set, and counting that pin would record a failed install as a deliberate CPU one and suppress the repair guidance for good."""

    def _names_it(family: str) -> bool:
        return True if not flavor else family == _flavor_tag_family(flavor)

    # _explicit_torch_index_url() already covers both variables WITH install.sh's precedence: the URL wins outright and the family is read only when no URL was supplied. Reading the family separately undid that, so an authoritative corporate /simple URL fell through to a stale ..._FAMILY=cpu and recorded a GPU-less host's CPU wheel as deliberately pinned.
    pin = _explicit_torch_index_url()
    if pin is not None and _names_it(_index_leaf_flavor_family(_torch_index_leaf(pin))):
        return True
    # install.sh derives UNSLOTH_TORCH_BACKEND from the index it RESOLVED -- "cpu" on any GPU-less machine, asked for or not -- and marks it derived. Only an unmarked value is a preference.
    if (
        _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu")
        and os.environ.get("UNSLOTH_TORCH_BACKEND_SOURCE", "").strip().lower() != "resolved"
        and _names_it(_TORCH_BACKEND)
    ):
        return True
    # A record can only speak for a run that said nothing to contradict it: a GPU family asked for HERE that settled on CPU because the GPU install failed is the same failed-pin case the arms above refuse to call deliberate. Only families this run NAMED count.
    if flavor and any(
        family and family != _flavor_tag_family(flavor)
        for family in _flavor_families_this_run_named()
    ):
        return False
    return bool(_RECORDED_TORCH_TAG_PINNED) and _names_it(
        _flavor_tag_family(_RECORDED_TORCH_TAG or "")
    )


def _flavor_families_this_run_named() -> tuple[str, ...]:
    """The torch families the CURRENT run was told to install, in no particular order."""
    named: list[str] = []
    pin = _explicit_torch_index_url()
    if pin is not None:
        named.append(_index_leaf_flavor_family(_torch_index_leaf(pin)))
    if (
        _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu")
        and os.environ.get("UNSLOTH_TORCH_BACKEND_SOURCE", "").strip().lower() != "resolved"
    ):
        named.append(_TORCH_BACKEND)
    return tuple(named)


def _expected_torch_index_url(tag: str) -> str:
    """The wheel index to repair `tag` from. Prefers the exact URL the setup script installed from (UNSLOTH_TORCH_INSTALL_INDEX_URL), then the explicit pin, because an authenticated mirror can only be repaired from the credentialed URL. Both are used only when their leaf IS this tag: setup.ps1 sends the /cpu index alongside a "rocm" tag on the AMD Windows path, and repairing a cu* mismatch from that URL would install the very CPU wheel this exists to remove. Otherwise rebuild it as setup.ps1 does: <mirror>/<tag>."""
    url = os.environ.get("UNSLOTH_TORCH_INSTALL_INDEX_URL", "").strip()
    if url:
        url = _trim_index_path_slashes(url)
        if _torch_index_leaf(url) == tag:
            return url
    pin = _explicit_torch_index_url()
    if pin is not None and _torch_index_leaf(pin) == tag:
        return pin
    return f"{_PYTORCH_WHL_BASE}/{tag}"


def _explicit_cpu_torch_index_pin() -> bool:
    """Whether this run was pinned to a CPU wheel index, by URL or by family. Only a pin counts: setup.ps1's published tag also reads "cpu" for a host whose nvidia-smi probe returned nothing, and treating that as an instruction would let a wedged driver downgrade a healthy CUDA venv."""
    pin = _explicit_torch_index_url()
    return pin is not None and _torch_index_leaf(pin) == "cpu"


# Not a flavor tag, so no `== expected` comparison can accept it, and truthy, so no `if not _now` branch can swallow it.
_TORCH_TAG_UNIMPORTABLE = "unimportable"


def _installed_flavor_tag_now(expected: str = "") -> str:
    """The venv's CURRENT flavor tag, re-probed, or "" when nothing could be read. ``expected`` only matters for "cpu": _torch_flavor_tag reads every untagged version as cpu, so a private index serving an untagged CUDA or ROCm wheel would satisfy a CPU expectation here exactly as it did in the pre-repair comparison, and the two comparisons must agree. "" rather than "cpu" for an unreadable venv, so ambiguity stays distinguishable from a positive CPU reading and never fails an update on its own."""
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran and _importable and _version:
        tag = _torch_flavor_tag(_version)
        if expected == "cpu" and tag == "cpu" and (_hip or _cuda or _TORCH_RUNTIME_XPU):
            return _gpu_family_from_runtime_markers(_hip, _cuda)
        return tag
    if _ran:
        # The probe ANSWERED, and its answer is that this torch does not import, not the ambiguity the disk fallback exists for. version.py would report the requested tag from a half-written or DLL-less wheel and the update would write a completion manifest over a torch nothing can import.
        return _TORCH_TAG_UNIMPORTABLE
    label = _installed_torch_label_on_disk()
    return _torch_flavor_tag(label) if label else ""


def _warn_repair_left_torch_unimportable(expected: str) -> bool:
    """Report a repair whose wheel cannot be imported, and fail. Always returns False. Distinct from _warn_wrong_flavor / _warn_still_cpu: the family on disk may well be the requested one, so naming it would send the reader after a wheel that is already correct. What is wrong is that it does not load."""
    _safe_print("")
    _safe_print(
        f"   [WARN] PyTorch was reinstalled for {expected} but the result cannot be imported."
    )
    _safe_print("   [WARN] The venv is not usable in this state.")
    _safe_print("   [WARN] Re-run this installer, or reinstall the build for your GPU manually.")
    _safe_print("   [WARN]     irm https://unsloth.ai/install.ps1 | iex")
    return False


def _warn_wrong_flavor(expected: str, installed: str) -> bool:
    """Report a repair that installed the wrong family, and fail. Always returns False. Distinct from _warn_still_cpu because "PyTorch is CPU-only" would be false here: the venv holds a GPU build, just not the one this host asked for."""
    _safe_print("")
    _safe_print(
        f"   [WARN] PyTorch is a {installed} build but {expected} was expected for this machine."
    )
    _safe_print("   [WARN] The repair did not install the requested build.")
    _safe_print("   [WARN] Re-run this installer, or reinstall the build for your GPU manually.")
    _safe_print("   [WARN]     irm https://unsloth.ai/install.ps1 | iex")
    return False


def _warn_still_cpu(expected: str) -> bool:
    """Report a repair that did not take, and fail the install. Always returns False. install.ps1 warns and exits 0; failing instead is the point, since a CPU-only torch on a host expecting a GPU build is exactly the state an update used to report as "dependencies up to date"."""
    _safe_print("")
    _safe_print(
        f"   [WARN] PyTorch is CPU-only but a {expected} GPU build was expected for this machine."
    )
    _safe_print("   [WARN] Training and GPU inference will run on CPU until this is fixed.")
    _safe_print(
        "   [WARN] Re-run this installer, or reinstall the GPU build manually for your GPU."
    )
    _safe_print("   [WARN]     irm https://unsloth.ai/install.ps1 | iex")
    return False


def _uninstall_distribution(name: str) -> bool:
    """Remove one distribution from the venv this script targets. True iff it is gone. --python sys.executable so a uv that also needs --system cannot remove from the system Python, with a pip fallback for the same interpreter. Output is swallowed; the caller reports."""
    if USE_UV and shutil.which("uv"):
        cmd = ["uv", "pip", "uninstall"]
        if UV_NEEDS_SYSTEM:
            cmd.append("--system")
        cmd.extend(["--python", sys.executable, name])
    else:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", name]
    removed = subprocess.run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    return removed.returncode == 0


def _resident_xformers_build_torch() -> "str | None":
    """The torch build the installed xFormers extension was compiled against, from ``xformers/cpp_lib.json`` (the file install.ps1's resident probe reads). Never raises and never imports xformers, whose mismatched _C.pyd would log its own warning into the installer's output."""
    try:
        spec = importlib.util.find_spec("xformers")
    except Exception:
        return None
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    if not locations:
        return None
    try:
        with open(os.path.join(locations[0], "cpp_lib.json"), encoding = "utf-8") as fh:
            recorded = json.load(fh).get("version", {}).get("torch")
    except (OSError, ValueError, AttributeError):
        return None
    return recorded.strip() if isinstance(recorded, str) and recorded.strip() else None


def _install_torchao_for_torch(torch_version: "str | None") -> None:
    """Select the torchao matching torch_version and install it from its own index.

    Called twice: as step 4, and again after the Linux torch repair, which can move torch
    across families and releases underneath the first call.
    """
    spec = _select_torchao_spec(torch_version)
    # See _TORCHAO_DEFAULT_SPEC. rocm is included here, unlike torchcodec: the rocm leaves
    # really do publish torchao.
    index = _torch_accelerator_index_url(torch_version)
    # --no-deps skips nothing today (no torchao release declares a runtime torch dependency)
    # and guards the second caller, which runs right after the torch repair.
    args = ["--no-deps", "--no-cache-dir"]
    if _pin_needs_reinstall(spec, _torch_index_tag(torch_version) if index else ""):
        args.insert(0, "--force-reinstall")
    _note(
        f"torch {torch_version or 'unknown'} detected -- installing {spec}"
        # Redacted for display only; the installer below still gets the exact URL.
        + (f" from {_strip_index_url_credentials(index)}" if index else "")
    )
    if not index:
        pip_install("Installing dependency overrides", *args, spec)
        return
    if pip_install_try("Installing dependency overrides", *args, "--index-url", index, spec):
        return
    # A leaf can lack this release outright (cu129 stops at 0.17.0 while serving torch 2.13),
    # and the wrong build only costs the kernels, so retry unpinned -- still fatally.
    _note(
        f"{_strip_index_url_credentials(index)} did not serve {spec} "
        "-- retrying from the default index; its kernels may be skipped"
    )
    pip_install("Installing dependency overrides", *args, spec)


def _resync_torch_coupled_packages(label_before: str) -> bool:
    """Re-settle the packages whose compiled extensions are tied to the torch build; False when this pass left the venv in a state the caller must re-verify. torchao's cpp is tied to the torch release AND its CUDA major, xFormers to the exact (torch, CUDA) pair, and beside a pair it was not built for its ops vanish behind a log line rather than an error. --no-deps is the whole safety of the torchao call: torchao depends on torch, so resolving would pull PyPI's CPU wheel back in. Never fatal, since both are secondary to the flavor repair that just succeeded."""
    _label_after = str(_probe_installed_torch_version() or "")
    if not _label_after or _label_after == label_before:
        return True
    _touched_torch = False
    # The whole local tag, not just the CUDA major: cpu to xpu moves no major at all (both
    # read None) yet still changes the build.
    _release_moved = _label_after.split("+", 1)[0] != str(label_before).split("+", 1)[0]
    _family_moved = _label_after.partition("+")[2].strip().lower() != (
        str(label_before).partition("+")[2].strip().lower()
    )
    _cuda_moved = _cuda_major_from_torch_version(_label_after) != (
        _cuda_major_from_torch_version(str(label_before))
    )
    if _release_moved or _family_moved or _cuda_moved:
        try:
            _spec = _select_torchao_spec(_label_after)
            # The same pin step 4 uses, or this reinstalls PyPI's build over it.
            # _pin_needs_reinstall, not an exact compare: ==0.18.0 never equals 0.18.0+cu130.
            _ao_index = _torch_accelerator_index_url(_label_after)
            _ao_args = ["--force-reinstall", "--no-deps", "--no-cache-dir"]
            if _pin_needs_reinstall(_spec, _torch_index_tag(_label_after) if _ao_index else ""):
                _note(f"torch {_label_after} after repair -- reinstalling {_spec}")
                _touched_torch = True
                _ao_ok = pip_install_try(
                    "Re-matching torchao to the repaired torch",
                    *_ao_args,
                    *(("--index-url", _ao_index) if _ao_index else ()),
                    _spec,
                )
                if not _ao_ok and _ao_index:
                    # Same starved-leaf fallback as step 4; see the note there.
                    _ao_ok = pip_install_try(
                        "Re-matching torchao to the repaired torch", *_ao_args, _spec
                    )
                if not _ao_ok:
                    # Across a CUDA-major move the resident build is not merely slower:
                    # _select_torchao_spec exists because a torchao compiled for CUDA 12
                    # cannot load its cpp under cu130. Remove it, the way the xFormers arm
                    # below removes a build made for a torch that is gone. A release-only
                    # move keeps the old warning: that one really is just the slow path.
                    if _cuda_moved and _uninstall_distribution("torchao"):
                        _note(
                            f"removed the torchao built for a different CUDA major; "
                            f"install {_spec} by hand to restore its kernels"
                        )
                    else:
                        _safe_print(
                            f"   [WARN] could not install {_spec} for the repaired torch; the "
                            f"torchao kernels will fall back to the slow path."
                        )
        except Exception as e:
            _safe_print(f"   [WARN] could not re-match torchao after the repair: {e}")
    try:
        _built_for = _resident_xformers_build_torch()
        if _built_for and _built_for != _label_after:
            _note(
                f"xFormers was built for torch {_built_for}, which is no longer "
                f"installed -- removing it so attention falls back to torch SDPA"
            )
            if not _uninstall_distribution("xformers"):
                _safe_print(
                    "   [WARN] could not remove the mismatched xFormers; its compiled "
                    "operations will stay unavailable until it is uninstalled by hand."
                )
    except Exception as e:
        _safe_print(f"   [WARN] could not re-check xFormers after the repair: {e}")
    return not _touched_torch


def _ensure_expected_torch_flavor(expected: "str | None" = None) -> bool:
    """Enforce that the venv still holds the torch flavor the install selected. `unsloth studio update` runs setup.ps1 and this script, never install.ps1, which held the only flavor repair, and the dependency steps above resolve torch from PyPI, whose Windows wheel is 2.11.0+cpu. Returns False when the flavor is wrong and could not be repaired, failing the install: that state used to be reported as success. ROCm is delegated to _ensure_rocm_torch, since AMD's Windows wheels live on a per-architecture repo.amd.com index a generic "rocm" tag cannot reconstruct."""
    if NO_TORCH:
        return True
    # rocm/xpu/cpu fall THROUGH: an explicit GPU pin sets _TORCH_BACKEND, and rejecting it here would skip the invariant on the hosts that asked for that family.
    if _TORCH_BACKEND not in ("", "cuda", "rocm", "xpu", "cpu"):
        return True
    if expected is None:
        expected = _expected_torch_flavor_tag()
    # The PIN, not the handover: setup.ps1 also publishes "cpu" when its nvidia-smi probe comes back empty, and that host must not be downgraded.
    _cpu_pinned = expected == "cpu" and _explicit_cpu_torch_index_pin()
    if not (_is_cuda_family_leaf(expected) or expected in ("xpu", "rocm") or _cpu_pinned):
        return True
    if _TORCH_BACKEND in ("rocm", "xpu", "cpu") and _TORCH_BACKEND != expected:
        return True
    # A pin whose leaf names no flavor was applied verbatim at install time, so acting on a manifest that predates it overrides an administrator's mirror. Compared rather than vetoed: the helper's known set predates XPU.
    _unknown_pin = _explicit_unknown_family_torch_index_url()
    if _unknown_pin is not None and _torch_index_leaf(_unknown_pin) != expected:
        return True
    # CUDA only, and only for an expectation INFERRED from hardware: an emptied mask is a reason not to conclude cu124 from a probe, not to ignore a stated one.
    if _is_cuda_family_leaf(expected) and not _expected_torch_flavor_is_explicit():
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if _cvd is not None and _cvd.strip() in ("", "-1"):
            return True

    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran and _importable and _version:
        installed_version = _version
    elif not _ran:
        # A wedged driver hangs `import torch`; version.py names the wheel without one.
        installed_version = _installed_torch_label_on_disk()
    else:
        # Missing or unimportable is the base install's job, a louder failure than this.
        return True
    if not installed_version:
        return True

    installed = _torch_flavor_tag(installed_version)
    # _torch_flavor_tag reads untagged as "cpu", so a private index's untagged CUDA build compares equal under a /cpu pin.
    if expected == "cpu" and installed == "cpu" and (_hip or _cuda or _TORCH_RUNTIME_XPU):
        installed = _gpu_family_from_runtime_markers(_hip, _cuda)
    if installed == expected:
        return True

    # install.ps1's line, word for word, so a support log from either path reads the same.
    _safe_print(
        f"   PyTorch flavor mismatch (installed {installed}, need {expected}) -- "
        f"reinstalling correct build..."
    )
    if expected == "rocm":
        # AMD's Windows wheels live on a per-architecture repo.amd.com index no "rocm" tag can name, and the handed-over URL still points at /cpu there.
        _ensure_rocm_torch()
        # The FAMILY: a transient repo.amd.com failure is non-fatal in there, and the cu124 wheel it leaves passes _torch_build_is_gpu.
        _now = _installed_flavor_tag_now(expected)
        if _now == _TORCH_TAG_UNIMPORTABLE:
            return _warn_repair_left_torch_unimportable(expected)
        if _now == expected:
            return True
        if not _now:
            # Ambiguity must not fail an update by itself.
            return True if _torch_build_is_gpu() else _warn_still_cpu(expected)
        if not _torch_build_is_gpu():
            return _warn_still_cpu(expected)
        return _warn_wrong_flavor(expected, _now)

    index_url = _expected_torch_index_url(expected)
    # XPU floor is 2.6, not 2.4: unsloth/models/_utils.py raises at import below it.
    _torch_pkg, _vision_pkg, _audio_pkg = (
        _XPU_TORCH_PKG_SPEC if expected == "xpu" else _TORCH_FLAVOR_REPAIR_PKG_SPEC
    )
    # No win_arm64 torchaudio wheel exists on any index ($WinArm64NoAudio in setup.ps1).
    _trio = [_torch_pkg, _vision_pkg, _audio_pkg]
    if _is_windows_arm64():
        _trio = [_torch_pkg, _vision_pkg]
    _label_before = str(installed_version)
    # --force-reinstall, not install.ps1's uv-only --reinstall-package: pip_install falls back to pip, which has no word for it. constrain=False: constraints.txt resolves against PyPI's torch, which is what put this venv here.
    pip_install(
        "PyTorch flavor repair",
        "--force-reinstall",
        "--no-cache-dir",
        *_trio,
        "--index-url",
        index_url,
        constrain = False,
    )

    # The family: a mirror can answer /cu128 with a cached cu124 wheel, and _torch_build_is_gpu is family-blind.
    _now = _installed_flavor_tag_now(expected)
    if _now == _TORCH_TAG_UNIMPORTABLE:
        return _warn_repair_left_torch_unimportable(expected)
    if _now == expected:
        if _resync_torch_coupled_packages(_label_before):
            return True
        # The resync installs --no-deps, but this function's verification is behind us.
        _after = _installed_flavor_tag_now(expected)
        if _after == _TORCH_TAG_UNIMPORTABLE:
            return _warn_repair_left_torch_unimportable(expected)
        if _after in (expected, ""):
            return True
        _safe_print("   [WARN] the post-repair package resync changed the torch build.")
        return _warn_wrong_flavor(expected, _after)
    if not _now:
        # Ambiguity must not fail an update by itself.
        if expected == "cpu":
            return True
        return True if _torch_build_is_gpu() else _warn_still_cpu(expected)
    if expected != "cpu" and not _torch_build_is_gpu():
        return _warn_still_cpu(expected)
    return _warn_wrong_flavor(expected, _now)


def _amd_torch_needs_dependency_pass() -> bool:
    """True when setup must run the dependency pass to repair non-ROCm torch. Scope is the wheel family, not the ROCm family: any ROCm marker keeps the fast path even when the repair would reroute it. Fails closed on an uncertain host or torch, and never installs."""
    if NO_TORCH or not IS_LINUX:
        return False
    # ROCm wheels are published for Linux x86_64 only.
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    # install.sh's resolved backend is authoritative, exactly as it is for the repair.
    if _TORCH_BACKEND in ("cuda", "cpu", "xpu"):
        return False
    # A ROCm pin bypasses hardware detection; any other pin owns its repair path.
    if _explicit_rocm_torch_index_url() is None:
        if _explicit_torch_index_url() is not None:
            return False
        if _has_usable_nvidia_gpu():
            return False
        # A hidden layer either side leaves no target to classify. Same reading the routing guard uses, so the two cannot drift.
        if _visible_masks_select_no_gpu():
            return False
        # Match the repair's two host signals: a visible GPU or an inferred/named arch.
        _inferred_gfx = _infer_linux_amd_gfx_arch()
        _rocm_visible = _has_rocm_gpu()
        if not _rocm_visible and not _inferred_gfx:
            return False
        # A mixed-arch host used to end it here, because probe and mask order could disagree with nothing to resolve it. _runtime_gfx_target now composes both mask layers and returns no target ONLY when the selection is genuinely unreadable, so gate on its answer, not the shape of the host.
        _selected_gfx, _, _selected_spoof, _selected_host = _runtime_gfx_target(_inferred_gfx)
        if _selected_gfx is None:
            return False
        # Every arm below is about the wheel. A corroborated spoof is not: the family can be perfect while ROCr presents an ISA those wheels have no code for, and only _ensure_rocm_torch clears the variable (#7331).
        if _selected_spoof is not None:
            return True
        _pre_ran, _pre_imp, _pre_ver, _pre_hip, _pre_cuda = _probe_torch_runtime()
        _pre_torch = (_pre_ver or "").lower() if (_pre_ran and _pre_imp) else ""
        # The compatibility reroutes have no version floor either, so a host whose ROCm version will not read must not be turned away before they are asked. Only with a torch that reads back: an unreadable one is not evidence its wheels are wrong.
        if _pre_torch and _rocm_compat_reroute_pending(
            _selected_gfx, _detect_rocm_version() or (0, 0), _pre_torch
        ):
            return True
        # The inferred per-arch repair can run without a readable ROCm version.
        _inferred_arm = (
            bool(_inferred_gfx)
            and bool((os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip() or not _rocm_visible)
            and _amd_arch_index_url(_inferred_gfx) is not None
        )
        # Both per-arch repairs are version-independent, and the host with no readable version is the one they exist for: a bundled-runtime install has no system ROCm to read. Requiring a version here refuses them at the door.
        _family_repair_arm = _rocm_torch_family_needs_repair(
            _selected_gfx, _detect_rocm_version(), _selected_host
        )
        if not _inferred_arm and not _family_repair_arm:
            # Other repair arms need a readable version with a published wheel family.
            _ver = _detect_rocm_version()
            if _ver is None or _generic_pytorch_rocm_tag(_ver) is None:
                return False

    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    # An unreadable torch is not evidence that its wheel family is wrong.
    if not (_ran and _importable) or not _version:
        return False
    # The +rocm tag covers builds that omit torch.version.hip.
    if not (_hip or "rocm" in _version.lower()):
        return True
    # Torch IS a ROCm build, which used to end it. The per-arch repairs act on exactly those hosts, so keeping the fast path makes them unreachable from `studio update`: a swapped card, or the reported gfx1103 on a generic wheel, would be repaired on a fresh install and never again. The pass costs a dependency resolution, not a reinstall, since _ensure_rocm_torch still keeps a family that matches. So ask only what it asks, only where it would act, and not under a pin, which commits to an index regardless.
    _pin = _explicit_rocm_torch_index_url()
    if _pin is not None:
        # A pin skips the hardware questions, but not its own: _ensure_rocm_torch reinstalls when the installed local tag names a different family than the pin, and returning here unconditionally meant a changed UNSLOTH_TORCH_INDEX_URL never reached it.
        return _rocm_pin_family_mismatch(_pin, _version.lower())
    _tail_gfx, _, _tail_spoof, _tail_host = _runtime_gfx_target(_infer_linux_amd_gfx_arch())
    _tail_ver = _detect_rocm_version() or (0, 0)
    # A corroborated HSA spoof is cleared by _ensure_rocm_torch and only by it: the family can match perfectly while ROCr presents an ISA the wheels have no code for (#7331), so the wheel question alone keeps the fast path and the variable stays set.
    if _tail_spoof is not None:
        return True
    # The two compatibility reroutes are repairs like any other, and asking only the missing-kernel question skipped both on update: Strix on a generic wheel below the AMD floor still needs the 7.13 fixes, and a sole gfx906 above rocm6.3 has no BLAS kernels.
    if _rocm_compat_reroute_pending(_tail_gfx, _tail_ver, _version.lower()):
        return True
    return _rocm_torch_family_needs_repair(_tail_gfx, _detect_rocm_version(), _tail_host)


def _already_on_amd_arch_leaf(leaf: "str | None", installed_ver: str) -> bool:
    """True when the installed torch already IS the AMD per-arch build for ``leaf``. The family is the direct reading. A family that will not read back is not evidence of the wrong wheels, so the local tag is accepted too: only the AMD index ships a rocm tag at or above the arch floor, and re-downloading a multi-GB stack each update to re-establish what the tag already says is the cost this guard avoids."""
    if _torch_below_211(installed_ver):
        return False
    _family = _installed_rocm_wheel_family() if _torch_requires_rocm_sdk() else None
    if _family is not None:
        # A family that reads back is the answer, whichever way it points: an AMD build for ANOTHER arch sits at the same floor tag and carries none of this one's kernels.
        return _family == (leaf or "").lower()
    _tag = re.search(r"\+rocm(\d+)\.(\d+)", installed_ver or "")
    return bool(_tag) and (int(_tag.group(1)), int(_tag.group(2))) >= _ROCM_ARCH_INDEX_FLOOR


def _rocm_compat_reroute_pending(
    runtime_gfx: "str | None", ver: "tuple[int, int]", installed_ver: str
) -> bool:
    """Whether a compatibility reroute _ensure_rocm_torch performs has not been applied yet. Neither reroute is about missing kernels, so neither is visible to the wheel-family question: Strix wants AMD's 7.13 build over any generic one below the floor, and gfx906 wants the last tag whose BLAS still carries it. Both compare against what is installed, so a host already on the right wheels keeps the fast path."""
    if not runtime_gfx:
        return False
    if runtime_gfx in _HSA_SPOOFABLE_PHYSICAL_GFX and _strix_needs_amd_arch_index(ver):
        return not _already_on_amd_arch_leaf(_GFX_TO_AMD_INDEX_ARCH.get(runtime_gfx), installed_ver)
    if _runtime_target_is_gfx906() and _gfx906_needs_legacy_index(ver):
        return _GFX906_LEGACY_TAG not in installed_ver
    return False


def _installed_generic_rocm_tag() -> "tuple[int, int] | None":
    """(major, minor) of the ROCm tag the INSTALLED torch names, or None if it names none. Generic pytorch.org wheels carry it in the local version ("2.9.1+rocm6.3"); AMD per-arch builds are read by _installed_rocm_wheel_family instead, so their tag is not wanted here."""
    _ran, _importable, _ver, _hip, _cuda = _probe_torch_runtime()
    if not (_ran and _importable):
        return None
    _m = re.search(r"\+rocm(\d+)\.(\d+)", (_ver or "").lower())
    return (int(_m.group(1)), int(_m.group(2))) if _m else None


def _rocm_torch_family_needs_repair(
    runtime_gfx: "str | None",
    ver: "tuple[int, int] | None" = None,
    host_codes: "list[str] | None" = None,
) -> bool:
    """Whether the installed ROCm torch carries no kernels for ``runtime_gfx``. Reads the same two signals _ensure_rocm_torch's arms read, in the same order, so the preflight cannot promise a repair the repair declines. A per-arch install names its family, and any family other than this target's is stale. A generic build names none (and _torch_requires_rocm_sdk rejects a stale `rocm` orphan beside one), so it is judged on whether the generic wheels carry kernels at all. An unknowable family answers False: leave the install alone rather than guess."""
    _owns_sdk = _torch_requires_rocm_sdk()
    _family = _installed_rocm_wheel_family() if _owns_sdk else None
    if _family is not None:
        _leaf = (_GFX_TO_AMD_INDEX_ARCH.get(runtime_gfx or "") or "").lower()
        if _family != _leaf:
            # Only when some index can serve the target ON THIS HOST: a gfx1010, or a mask-selected gfx906 beside another card, has a real mismatch and nowhere to go, and promising a repair _ensure_rocm_torch refuses buys a dependency pass on EVERY update and never a working torch.
            return _gfx_route_on_host(runtime_gfx, host_codes)
        # The right SHAPE, not necessarily a working build: below the 2.11 floor these leaves carry the _grouped_mm bug, and answering False on a 2.10 build would keep the fast path and leave it in place forever.
        _ran, _importable, _ver, _hip, _cuda = _probe_torch_runtime()
        return _leaf in _ROCM_GFX_TORCH211_LEAVES and _torch_below_211(
            (_ver or "").lower() if (_ran and _importable) else ""
        )
    if _owns_sdk:
        # A per-arch install whose family will not read back: _ensure_rocm_torch cannot skip on a family it never read, so it would reinstall the stack on EVERY update. Leave it alone.
        return False
    # Which arches a generic wheel carries belongs to THAT wheel, so read the tag off the installed torch when it states one; ``ver`` is the HOST's, and the two part company (pin rocm6.3 once, and a gfx1200 box on 2.9.1+rocm6.3 looks healthy forever). Both readings of "no code for this card" are needed: the reroute question, and the tag floor for a target with no index to be rerouted TO.
    _installed_tag = _installed_generic_rocm_tag() or ver
    return _generic_rocm_wheel_lacks_kernels(
        runtime_gfx, _installed_tag
    ) or _generic_only_target_below_floor(runtime_gfx, _installed_tag)


def _ensure_rocm_torch() -> None:
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch: pytorch.org ROCm wheel index tags on Linux x86_64, AMD's repo.amd.com arch-specific index on Windows. No-op on macOS, non-x86_64 Linux, NVIDIA-primary hosts, or when torch already links against HIP."""
    global _rocm_windows_torch_installed
    # install.sh's resolved backend is authoritative: skip ROCm when it already chose a non-ROCm family, rather than re-detecting in a subprocess that may see a different env.
    if _TORCH_BACKEND in ("cuda", "cpu", "xpu"):
        return
    # An explicit unknown-family pin was applied VERBATIM at install time; leave it alone.
    if _explicit_unknown_family_torch_index_url() is not None:
        return
    # setup.ps1's marker; trust it only when torch imports as ROCm, since a wiped venv leaves it stale.
    if os.environ.get("UNSLOTH_ROCM_TORCH_INSTALLED") == "1":
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_ok = _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
        if _torch_ok:
            _rocm_windows_torch_installed = True
            # ROCm torch is already installed, but bnb still needs the ROCm build (pre-release wheel, else PyPI >=0.50.0).
            _install_bnb_windows_rocm()
            return
        # torch was wiped between runs; fall through to the full install path
    if IS_MACOS:
        return

    if IS_WINDOWS:
        # An explicit ROCm pin overrides the per-arch index: retry the PINNED one, not repo.amd.com.
        _win_rocm_pin = _explicit_rocm_torch_index_url()
        if _win_rocm_pin is None and _has_usable_nvidia_gpu():
            return
        gfx_arch = _detect_windows_gfx_arch()
        if not gfx_arch and _win_rocm_pin is None:
            return  # no AMD GPU visible via hipinfo
        # Whether torch already links against HIP.
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_already_rocm = (
            _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
        )
        # "Is ROCm" is not "is the RIGHT ROCm": wheels are per-family, so a host whose arch now resolves elsewhere (dGPU added, or the #7776 repick) would keep the old family forever. setup.ps1 force-reinstalls every run, so this only bites standalone `studio update`. Act only on a family read back positively, never on a guess.
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
            _torch_pkg, _vision_pkg, _audio_pkg = _WINDOWS_ROCM_TORCH_PKG_SPECS.get(
                gfx_arch, ("torch", "torchvision", "torchaudio")
            )
            # Same win_arm64 exception setup.ps1 applies: no torchaudio wheel exists there, so asking for one makes the trio unresolvable.
            _rocm_trio = [_torch_pkg, _vision_pkg, _audio_pkg]
            if _is_windows_arm64():
                _rocm_trio = [_torch_pkg, _vision_pkg]
            # Nonfatal: a transient AMD-index failure must not abort the install. --force-reinstall resolves before uninstalling, so a failed index keeps the existing build intact.
            if not pip_install_try(
                f"ROCm torch (Windows, {gfx_arch or 'pinned'})",
                "--force-reinstall",
                "--index-url",
                index_url,
                *_rocm_trio,
                constrain = False,
            ):
                _safe_print(
                    f"   Warning: AMD Windows ROCm torch install failed for {gfx_arch or 'the pinned index'}; "
                    "keeping the existing torch build. Re-run 'unsloth studio update' "
                    "later to retry ROCm."
                )
                return
        # Flag ROCm torch installed so later phases keep it; a BNB failure must not roll it back.
        _rocm_windows_torch_installed = True
        # Always install AMD Windows bitsandbytes, even when torch was already a ROCm build, so `studio update` repairs a broken bnb.
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
    # An explicit ROCm pin commits to ROCm wheels whatever the visible GPU (headless / CI).
    _rocm_pin = _explicit_rocm_torch_index_url()
    # Before ANY install path, including the inferred-arch one below: that takes a declared UNSLOTH_ROCM_GFX_ARCH first, so a stale gfx1030 on a real Van Gogh force-installed the gfx103X-all stack and _ensure_cpu_torch() then undid it, a ROCm-to-CPU cycle on every update. An explicit index pin still wins.
    if _rocm_pin is None and not IS_WINDOWS and _miscomputing_arch_host():
        _safe_print(
            "   This host has an AMD arch measured to compute incorrectly under ROCm "
            "(studio/ROCM_RDNA2_APU.md) -- keeping CPU torch.\n"
        )
        return
    _inferred_linux_gfx = (
        _infer_linux_amd_gfx_arch() if (_rocm_pin is None and not IS_WINDOWS) else None
    )
    if _rocm_pin is None:
        # NVIDIA takes precedence on mixed hosts (only if a GPU is usable).
        if _has_usable_nvidia_gpu():
            return
        # _has_rocm_gpu() (rocminfo / amd-smi rows) is the authoritative AMD-host signal; the old /opt/rocm-or-hipcc gate broke runtime-only ROCm installs.
        if not _has_rocm_gpu() and not _inferred_linux_gfx:
            return  # no AMD GPU visible

    ver = _detect_rocm_version()
    if ver is None:
        # A host running the wheels' own bundled ROCm has no system version to read, and is the same host the missing-kernel route exists for. That route has no version floor, so returning here on an arch the generic wheel cannot serve refuses the repair at the door.
        _unknown_ver_gfx, _, _, _unknown_ver_host = _runtime_gfx_target(None)
        _uv_ran, _uv_imp, _uv_ver, _uv_hip, _uv_cuda = _probe_torch_runtime()
        _unknown_ver_torch = (_uv_ver or "").lower() if (_uv_ran and _uv_imp) else ""
        # A per-arch install can also outlive the GPU it was made for: swap a gfx1200 card into a box on gfx110X-all wheels and the generic index would serve it while those wheels carry no gfx1200 kernels. A matching family below its 2.11 floor is the same story. This is the reading the setup preflight uses, so anything narrower lets the pass run and then declines it.
        if (
            _rocm_pin is None
            and not _inferred_linux_gfx
            and not _generic_rocm_wheel_lacks_kernels(_unknown_ver_gfx)
            and not _rocm_torch_family_needs_repair(_unknown_ver_gfx, None, _unknown_ver_host)
            # The Strix reroute has no version floor to fail: with no tag resolving, the per-arch index is the only route these arches have, and exiting here left a visible Strix host on whatever non-ROCm torch it had.
            and not _rocm_compat_reroute_pending(_unknown_ver_gfx, (0, 0), _unknown_ver_torch)
        ):
            _safe_print("   ROCm detected but version unreadable -- skipping torch reinstall")
            return
        # Explicit pin or inferred gfx: the index drives the install.
        ver = (0, 0)

    # Whether torch links against HIP, capturing the installed ROCm tag for pin-mismatch detection. Marker is the HIP version, else a "rocm" sentinel when only the version string flags ROCm; empty = CPU/CUDA torch, or un-probeable, which reinstalls.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    _installed_torch_ver = (_version or "").lower() if (_ran and _importable) else ""
    _hip_marker = ""
    if _ran and _importable:
        _hip_marker = _hip if _hip else ("rocm" if "rocm" in _installed_torch_ver else "")
    has_hip_torch = _hip_marker != ""

    # A ROCm pin of another family reinstalls; a same-tag per-arch switch is undetectable.
    _rocm_pin_mismatch = (
        _rocm_pin_family_mismatch(_rocm_pin, _installed_torch_ver)
        if (has_hip_torch and _rocm_pin is not None)
        else False
    )

    rocm_torch_ready = has_hip_torch and not _rocm_pin_mismatch

    # Inferred-gfx path: ROCm runtime missing but install.sh would route to AMD wheels. Gated on the runtime NOT enumerating a GPU, so a mixed Strix APU + dGPU box with HIP_VISIBLE_DEVICES on the dGPU does not get APU wheels. An explicit UNSLOTH_ROCM_GFX_ARCH is exempt from that gate (mirrors install.sh): a visible GPU with an unreadable ROCm version must not silently discard the user's named arch.
    _gfx_override_env = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
    _inferred_arch_installed = False
    if (
        _inferred_linux_gfx
        and not has_hip_torch
        and _rocm_pin is None
        and (_gfx_override_env or not _has_rocm_gpu())
        # This branch installs for the inferred arch without asking the mask layers, so an ordinal the guess cannot account for reaches the wheel the preflight already declines. An explicit arch is exempt above.
        and (_gfx_override_env or _runtime_gfx_target(_inferred_linux_gfx)[0] is not None)
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
            _inferred_arch_installed = True
            # The same reconciliation the reroutes below make: these wheels carry _inferred_linux_gfx code objects alone, and a spoof naming another arch has ROCr hand them a device none of it matches (#7331). The install is already committed to that arch, so declining here only guarantees it is unusable.
            if _hsa_spoof_contradicts(_inferred_linux_gfx):
                _clear_confirmed_hsa_spoof(_inferred_linux_gfx)

    # An explicit UNSLOTH_ROCM_GFX_ARCH=gfx906 pins the runtime target to the MI50 / Radeon VII path and must win over the Strix probe-order detection below (a mixed Strix + MI50 host could otherwise route to gfx1151).
    _gfx906_arch_override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(
        ":"
    )[0] == "gfx906"

    # Where the two AMD per-gfx reroutes below deposit their choice. Strix Halo / Point (gfx1151 / gfx1150) need AMD's per-gfx index (2.11+rocm7.13) because every generic pytorch.org index lacks the fixes (ROCm 7.1 segfaults in _grouped_mm); see _strix_needs_amd_arch_index. The second reroute has no floor and fires for an arch the generic wheel carries no kernels for at all.
    _arch_index_url: "str | None" = None
    _arch_index_pkgs: "tuple[str, str, str] | None" = None
    # An explicit ROCm pin wins; otherwise both reroutes share one hardware probe. Skipped once the inferred-arch install above has run: it resolves the same index, so re-deriving it force-reinstalls what was just downloaded.
    if (
        _explicit_rocm_torch_index_url() is None
        and not _gfx906_arch_override
        and not _inferred_arch_installed
    ):
        _runtime_gfx, gfx_codes, _physical_gfx, _host_codes = _runtime_gfx_target(
            _inferred_linux_gfx
        )
        # A miscomputing target has no ROCm route, so the rest of this function is skipped: _amd_arch_index_url returns None for such an arch and the reroute below would either raise or fall through to generic wheels. Keyed on the SELECTED target, which install.sh's presence gate cannot see.
        if _runtime_gfx in _ROCM_MISCOMPUTING_GFX:
            # Declines to INSTALL ROCm for this target only; removing an existing ROCm build is _ensure_cpu_torch's call, and that asks about the whole host with masks stripped.
            _safe_print(
                f"   {_runtime_gfx} computes incorrect results under ROCm "
                f"(studio/ROCM_RDNA2_APU.md) -- not installing ROCm torch for it.\n"
            )
            return
        _strix_gfx = {"gfx1151", "gfx1150", "gfx1152"}
        # Only the Strix reroute has a ROCm-version floor.
        _detected_strix = (
            _strix_gfx.intersection(gfx_codes) if _strix_needs_amd_arch_index(ver) else set()
        )
        if _detected_strix:
            if _runtime_gfx in _strix_gfx and _already_on_amd_arch_leaf(
                _GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx), _installed_torch_ver
            ):
                # Already the build this branch would fetch. It force-reinstalls a multi-GB stack and _ensure_rocm_torch runs twice per install, so acting on the arch alone re-downloads it on every install and update.
                _safe_print(
                    f"   torch already runs on the AMD {_runtime_gfx} wheels; keeping it.\n"
                )
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_runtime_gfx)
            elif _runtime_gfx in _strix_gfx:
                _selected_gfx = _runtime_gfx
                # One owner for the mirror env var and the arch-to-leaf map, shared with the inferred-arch install above and the missing-kernel route below.
                _arch_index_url = _amd_arch_index_url(_selected_gfx)
                _arch_index_pkgs = (
                    "torch>=2.11.0,<2.12.0",
                    "torchvision>=0.26.0,<0.27.0",
                    "torchaudio>=2.11.0,<2.12.0",
                )
                _safe_print(
                    f"   {_selected_gfx} (AMD Strix) is the runtime target with ROCm "
                    f"{ver[0]}.{ver[1]}.\n"
                    f"   Routing torch install to AMD's arch-specific index\n"
                    f"   ({_strip_index_url_credentials(_arch_index_url)}) which serves torch\n"
                    f"   2.11.0+rocm7.13.0 with AMD's gfx1150/gfx1151 fixes (more reliable than\n"
                    f"   the generic pytorch.org rocm7.2 index on ROCm 7.3+ hosts).\n"
                )
                # Only on this branch: these wheels carry _selected_gfx kernels, so the runtime must stop reporting the spoofed arch or they have no code for the device (#7331). Never on the paths that keep generic wheels, where the override is the only source of usable kernels.
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_selected_gfx)
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                _safe_print(
                    f"   Strix GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects a non-Strix runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

        # If the generic wheel lacks the runtime target, replace it from AMD's per-arch index. No ROCm-version floor here: whichever generic index a host's version picks, its wheel carries no kernels for these targets.
        if _arch_index_url is None:
            # Judged by the installed wheel when it is a generic build naming its own tag, since ``ver`` is the HOST's. ``ver`` still chooses the tag for a reinstall below, which IS a question about the host.
            _kernel_ver = (
                has_hip_torch and not _torch_requires_rocm_sdk() and _installed_generic_rocm_tag()
            ) or ver
            _missing_kernels = {
                g for g in gfx_codes if _generic_rocm_wheel_lacks_kernels(g, _kernel_ver)
            }
            # A sub-2.11 build of a floor leaf is broken wherever it came from, and several of those leaves serve GPUs the generic wheel DOES list, so gating the floor on missing generic kernels never reaches them and the preflight forces a pass this block declines. The floor belongs to the family, not to the visit.
            _runtime_leaf = _GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx or "")
            _below_floor_on_leaf = (
                _runtime_leaf is not None
                and _runtime_leaf.lower() in _ROCM_GFX_TORCH211_LEAVES
                and has_hip_torch
                and _torch_requires_rocm_sdk()
                and _installed_rocm_wheel_family() == _runtime_leaf.lower()
                and _torch_below_211(_installed_torch_ver)
            )
            if _missing_kernels or _below_floor_on_leaf:
                _leaf = (
                    _runtime_leaf
                    if (
                        _generic_rocm_wheel_lacks_kernels(_runtime_gfx, _kernel_ver)
                        or _below_floor_on_leaf
                    )
                    else None
                )
                # Already running on this family's wheels. The reroute below is a --force-reinstall --no-cache-dir of the multi-GB stack and _ensure_rocm_torch runs twice per install, so a leaf derived from hardware alone re-downloads it twice per install and again on every update: act only on a family read back positively. The right SHAPE is not a working build either, since on these leaves a sub-2.11 wheel carries the _grouped_mm bug and an unreadable ROCm version routes gfx1152 here rather than through the Strix branch, so this is its only floor.
                _already_on_leaf = (
                    _leaf is not None
                    and has_hip_torch
                    and _torch_requires_rocm_sdk()
                    and _installed_rocm_wheel_family() == _leaf.lower()
                    and not (
                        _leaf.lower() in _ROCM_GFX_TORCH211_LEAVES
                        and _torch_below_211(_installed_torch_ver)
                    )
                )
                if _already_on_leaf:
                    _safe_print(
                        f"   torch already runs on the {_leaf} wheels {_runtime_gfx} needs; "
                        f"keeping it.\n"
                    )
                    # Keeping the wheels is not keeping the status quo: they carry the PHYSICAL arch alone, so a spoof left set has the runtime keep reporting the one arch they have no code for (#7331).
                    if _physical_gfx is not None:
                        _clear_confirmed_hsa_spoof(_runtime_gfx)
                elif _leaf is not None:
                    _arch_index_url = _amd_arch_index_url(_runtime_gfx)
                    # Keep older per-arch builds valid while bounding companion versions, except on the leaves whose sub-2.11 builds carry the _grouped_mm bug. gfx1152 reaches this branch when an unreadable ROCm version reads as 0.0.
                    _arch_index_pkgs = (
                        _ROCM_TORCH_PKG_SPECS["rocm7.2"]
                        if _leaf.lower() in _ROCM_GFX_TORCH211_LEAVES
                        else _ROCM_ARCH_INDEX_TORCH_PKG_SPEC
                    )
                    _safe_print(
                        f"   {_runtime_gfx} is the runtime target, and no pytorch.org ROCm wheel\n"
                        f"   carries kernels for it -- torch would load but fault on its first\n"
                        f"   GPU operation. Routing the torch install to AMD's arch-specific\n"
                        f"   index, which does:\n"
                        f"   {_strip_index_url_credentials(_arch_index_url)}\n"
                    )
                    # Let the runtime report the native target carried by these wheels.
                    if _physical_gfx is not None:
                        _clear_confirmed_hsa_spoof(_runtime_gfx)
                else:
                    _gfx_str = ", ".join(sorted(_missing_kernels))
                    _safe_print(
                        f"   GPU without generic-wheel kernels ({_gfx_str}) present, but the\n"
                        f"   selected runtime target is {_runtime_gfx}; keeping the generic index.\n"
                    )
            elif (
                _physical_gfx is not None
                and _runtime_leaf is not None
                and has_hip_torch
                and _torch_requires_rocm_sdk()
                and _installed_rocm_wheel_family() == _runtime_leaf.lower()
            ):
                # Nothing to reroute: the installed per-arch wheels are the target's own family, which is why neither condition above fires. The spoof is still exported and these wheels carry the PHYSICAL arch alone (#7331), and every arm that KEEPS matching wheels clears it, while this one reached no arm at all.
                _clear_confirmed_hsa_spoof(_runtime_gfx)

        # A per-arch install can outlive the GPU it was made for: add a dGPU, or point HIP_VISIBLE_DEVICES at one, and the target moves to an arch those wheels carry no kernels for, while torch.version.hip still reads ROCm so rocm_torch_ready would stop the generic fallback. Same #7776 repick, on a family read back positively.
        if _arch_index_url is None and rocm_torch_ready and _runtime_gfx is not None:
            _have = _installed_rocm_wheel_family() if _torch_requires_rocm_sdk() else None
            _want = (_GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx) or "").lower()
            # A target no index can serve (gfx1010 / RDNA 1) has an empty _want, which reads as "every family is wrong" and spends a multi-GB reinstall on a wheel that cannot carry kernels for it either. Demote only when there is somewhere to go, and ask that of THIS host (_MIXED_HOST_UNROUTABLE).
            if (
                _have is not None
                and _have != _want
                and _gfx_route_on_host(_runtime_gfx, _host_codes)
            ):
                _safe_print(
                    f"   installed ROCm torch is the {_have} build, which carries no\n"
                    f"   {_runtime_gfx} kernels -- reinstalling for this GPU.\n"
                )
                rocm_torch_ready = False
                # Demoting alone hands the job to the generic fallback, which resolves NO tag when the ROCm version is unreadable, as on a bundled-runtime host: it would announce the reinstall, install nothing, and keep the family it just called incompatible. The AMD index needs no host version.
                _tag = _generic_pytorch_rocm_tag(ver)
                if _want:
                    if _tag is None:
                        _arch_index_url = _amd_arch_index_url(_runtime_gfx)
                        if _arch_index_url is not None:
                            _arch_index_pkgs = (
                                _ROCM_TORCH_PKG_SPECS["rocm7.2"]
                                if _want in _ROCM_GFX_TORCH211_LEAVES
                                else _ROCM_ARCH_INDEX_TORCH_PKG_SPEC
                            )
                elif _runtime_gfx in _GENERIC_ROCM_WHEEL_GFX and (
                    _tag is None or _generic_tag_lacks_kernels(_runtime_gfx, ver)
                ):
                    # The replacement GPU has no AMD per-arch index (gfx942, gfx950 and the datacentre parts live only on the generic one), and a resolving tag can be as useless as none, so both take the newest generic index known.
                    ver = max(_ROCM_TORCH_INDEX)
            elif _have is None and _generic_only_target_below_floor(
                _runtime_gfx, _installed_generic_rocm_tag() or ver
            ):
                # A GENERIC build whose own tag predates the target: no family to compare and no per-arch index to move to, so both arms decline while torch.version.hip blocks the fallback and a gfx950 pinned to rocm6.3 stays kernel-less forever. Demoting is the whole repair.
                _safe_print(
                    f"   installed ROCm torch is a generic build whose own tag predates\n"
                    f"   {_runtime_gfx} -- reinstalling from an index that carries it.\n"
                )
                rocm_torch_ready = False

        # The floor above is reached only by a demoted host; a fresh install walks past it to the generic fallback, where a stale /opt/rocm puts gfx950 on rocm6.3 the same way. Which tag carries an arch is a fact about the arch.
        if (
            _arch_index_url is None
            and not rocm_torch_ready
            and _runtime_gfx in _GENERIC_ROCM_WHEEL_GFX
            and _generic_tag_lacks_kernels(_runtime_gfx, ver)
        ):
            ver = max(_ROCM_TORCH_INDEX)

    # gfx906 (MI50 / Radeon VII) as the runtime GPU target: used below to skip the generic bitsandbytes wheel, which has no gfx906 kernels. It must hold even under a torch-index pin, or a gfx906 host pinning rocm6.3 would clobber the user's source-built bnb, so a pin suppresses only the torch reroute (_gfx906_override).
    _runtime_is_gfx906 = _runtime_target_is_gfx906()
    # Reroute torch to the last gfx906-capable family (rocm6.3) only when the host ROCm version would otherwise pick a newer, kernel-less index, and never over an explicit pin or an active Strix reroute.
    _gfx906_override = (
        _runtime_is_gfx906
        and _gfx906_needs_legacy_index(ver)
        and _explicit_rocm_torch_index_url() is None
        and _arch_index_url is None
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

    # Either reroute must fire even when has_hip_torch is True: an existing torch.version.hip == "7.1" is exactly the broken combo they repair. The missing-kernel route declines above when torch already runs on its leaf.
    if _arch_index_url is not None and _arch_index_pkgs is not None:
        index_url = _arch_index_url
        _torch_pkg, _vision_pkg, _audio_pkg = _arch_index_pkgs
        _safe_print(
            f"   AMD per-gfx index override -- installing torch from "
            f"{_strip_index_url_credentials(index_url)}"
        )
        pip_install(
            f"ROCm torch (AMD per-gfx index, {_torch_index_leaf(index_url)})",
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
    # gfx906 fires even when has_hip_torch is True: a +rocm7.x build IS the broken combo it repairs. A torch already on rocm6.3 wheels is left alone, since the tag check below is False and rocm_torch_ready is already True.
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
        # Reinstall when torch is not ROCm yet, OR a ROCm build's family differs from a pin. Gate on rocm_torch_ready, not has_hip_torch alone, so the generic path does not overwrite a successful inferred-gfx install and undo the fresh-ROCm/no-/dev/kfd repair (#7305).
        _override_idx = _explicit_rocm_torch_index_url()
        if _override_idx is not None:
            index_url = _override_idx
            tag = _torch_index_leaf(index_url)
        else:
            tag = _generic_pytorch_rocm_tag(ver)
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
            # Only the _grouped_mm-bug gfx arches need the 2.11 spec; other gfx indexes ship <2.11 and stay on the default range (matches install.ps1 / setup.ps1).
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

    # gfx906 has no prebuilt bitsandbytes, and force-reinstalling the generic wheels would clobber a user's source-built bnb, the only 4-bit path on this arch.
    if rocm_torch_ready and _runtime_is_gfx906:
        _safe_print(
            _dim(
                "   gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels). "
                "Build bitsandbytes from source for 4-bit QLoRA -- "
                "see docs.unsloth.ai/get-started/install-and-update/amd."
            )
        )
        # The base install resolves unsloth's unconditional bitsandbytes dep to a generic CUDA wheel with no gfx906 kernels ("invalid device function"). Drop it only if this run pulled it in.
        if _GFX906_BNB_ABSENT_BEFORE_BASE and _bitsandbytes_installed():
            _safe_print(_dim("   gfx906: removing generic bitsandbytes pulled in as a dependency"))
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"],
                capture_output = True,
            )
    # bitsandbytes only when torch links ROCm; the pre-release wheel (bnb #1887) needs pip, not uv.
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
    """Whether to run in no-torch (GGUF-only) mode. Precedence: UNSLOTH_NO_TORCH ("false" included, so an explicit value wins) -> the venv's install manifest -> platform detection. The manifest tier is what keeps `unsloth studio update` in no-torch mode, since it injects no env var; an empty value counts as unset, because PowerShell cannot represent a set-but-empty variable. Evaluated at import, before install_python_stack() drops the manifest: do NOT defer it into main()."""
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None and env.strip():
        return env.strip().lower() in install_manifest.NO_TORCH_TRUTHY
    recorded = install_manifest.recorded_no_torch()
    if recorded is not None:
        return recorded
    return IS_MAC_INTEL


NO_TORCH = _infer_no_torch()

# Read at import: install_python_stack() drops the manifest before its dependency pass.
_RECORDED_TORCH_TAG = install_manifest.recorded_torch_flavor()
# Whether that record came from someone NAMING a flavor: setup.ps1 publishes an automatic /cpu choice the same way it publishes a pinned one.
_RECORDED_TORCH_TAG_PINNED = install_manifest.recorded_torch_flavor_was_pinned()

# UNSLOTH_TORCH_BACKEND is set by install.sh after get_torch_index_url(); empty = standalone `studio update`, where we re-detect.
_TORCH_BACKEND: str = os.environ.get("UNSLOTH_TORCH_BACKEND", "").lower()
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
        # Without this the leaf falls through as unknown and the standalone update never acts on an authoritative XPU pin; see _ensure_xpu_torch.
        _TORCH_BACKEND = "xpu"
    elif _is_cuda_family_leaf(_idx_leaf):
        # Require a digit after "cu" so /current or /custom is not branded CUDA.
        _TORCH_BACKEND = "cuda"


# Quoted values only, so the `hip: Optional[str] = None` a non-ROCm build writes is negative.
_TORCH_VERSION_PY_HIP_RE = re.compile(r"""^hip\s*(?::[^=]*)?=\s*['"]([^'"]*)['"]""", re.MULTILINE)


def _torch_hip_version_on_disk() -> str:
    """torch.version.hip read from torch/version.py, launching no interpreter."""
    try:
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
    match = _TORCH_VERSION_PY_HIP_RE.search(text)
    return match.group(1) if match else ""


def _installed_torch_is_windows_rocm_cheap() -> bool:
    """_installed_torch_is_windows_rocm's verdict without ever running `import torch`.

    _torch_step_label runs with the probe memo cold, so the probing form spent up to its
    90s timeout before _progress() emitted anything, to format a string.
    """
    if not IS_WINDOWS:
        return False
    if _TORCH_RUNTIME_PROBE is not None:
        return _installed_torch_is_windows_rocm()
    if _torch_hip_version_on_disk():
        return True
    return "rocm" in _installed_torch_version_label().lower()


def _torch_step_label(suffix: str) -> str:
    """Progress label like 'torch check (cuda)', falling back to GPU detection when UNSLOTH_TORCH_BACKEND is unset."""
    backend = _TORCH_BACKEND
    if not backend:
        if _has_usable_nvidia_gpu():
            backend = "cuda"
        # rocminfo and amd-smi ship with the HIP SDK, not with AMD's bundled-runtime
        # wheels, so a Windows ROCm host reads as CPU without the second operand.
        elif _has_rocm_gpu() or _installed_torch_is_windows_rocm_cheap():
            backend = "rocm"
        else:
            backend = "cpu"
    return f"torch {suffix} ({backend})"


VERBOSE: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

# Progress bar state; update _TOTAL when adding/removing steps in install_python_stack().
_STEP: int = 0
_TOTAL: int = 0  # set at runtime in install_python_stack() based on platform
_PROGRESS_LINE_ACTIVE: bool = False

SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)
LOCAL_DD_GITHUB_PLUGIN = SCRIPT_DIR / "backend" / "plugins" / "data-designer-github-repo-seed"

# Apple Silicon: override mlx-vlm/mlx-lm's transformers pin. _uv_safe_path because uv truncates UV_OVERRIDE at the first space (#6503).
_MLX_OVERRIDES = SINGLE_ENV / "overrides-darwin-arm64.txt"
if IS_MAC_ARM and _MLX_OVERRIDES.is_file() and "UV_OVERRIDE" not in os.environ:
    os.environ["UV_OVERRIDE"] = _uv_safe_path(_MLX_OVERRIDES)

# Windows consoles may be a legacy code page (CP1252); _safe_print() degrades glyphs to ASCII.

_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2705": "[OK]",  # ✅
    "\u274c": "[FAIL]",  # ❌
    "\u26a0\ufe0f": "[!]",  # ⚠️  (warning + variation selector)
    "\u26a0": "[!]",  # ⚠  (warning without variation selector)
}


def _safe_print(*args: object, **kwargs: object) -> None:
    """print() replacement surviving non-UTF-8 consoles and detached stdout. Closes an open progress-bar line first, since _progress() leaves the cursor mid-line (nothing calls print() directly, per test_no_bare_print_calls)."""
    _end_progress_line()
    try:
        print(*args, **kwargs)
    except OSError:
        return
    except UnicodeEncodeError:
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


# Column layout matching setup.sh's step(): 2-space indent, 15-char label (dim), then value.
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
    # Every _safe_print() lands here: a detached (None) or closed stdout must not take down a message bound for stderr.
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
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit, unless ``check`` is False."""
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
        if not check:
            # The caller inspects the failure itself, and is responsible for reporting and exiting if it cannot recover.
            return result
        _report_failed_command(label, result)
    return result


def _report_failed_command(label: str, result: subprocess.CompletedProcess[bytes]) -> None:
    """Print a failed command's redacted output and exit with its code."""
    _step("error", f"{label} failed (exit code {result.returncode})", _red)
    if result.stdout:
        # Redact before printing: the failing pip command may carry a pinned --index-url with userinfo/?token= creds.
        _safe_print(_redact_install_output(result.stdout))
    sys.exit(result.returncode)


# pip will not replace a distribution whose .dist-info carries no RECORD, and an interrupted install leaves exactly that. The venv is REUSED across runs, so once written every later install of that package dies on it with "The package's contents are unknown".
_NO_RECORD_MARKER = "no RECORD file was found"
_CANNOT_UNINSTALL_RE = re.compile(r"Cannot uninstall ([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _purge_recordless_distributions(output: "bytes | str | None") -> list[str]:
    """Delete the .dist-info directories pip named where they carry no RECORD. pip's own hint, --ignore-installed, would apply to every package in the command and silently skip real upgrades."""
    if not output:
        return []
    text = output.decode("utf-8", "replace") if isinstance(output, bytes) else output
    # Both halves are required: the marker alone can appear in unrelated advice, and a "Cannot uninstall" without it is a different problem that deleting will not fix.
    if _NO_RECORD_MARKER not in text:
        return []
    blocked = {_canonical_dist_name(n) for n in _CANNOT_UNINSTALL_RE.findall(text)}
    if not blocked:
        return []
    site_packages = sysconfig.get_path("purelib")
    if not site_packages:
        return []
    cleared: list[str] = []
    for dist_info in sorted(Path(site_packages).glob("*.dist-info")):
        if _canonical_dist_name(dist_info.name.split("-", 1)[0]) not in blocked:
            continue
        if (dist_info / "RECORD").is_file():
            continue  # complete install; whatever failed, it was not this
        try:
            shutil.rmtree(dist_info)
        except OSError:
            continue
        cleared.append(dist_info.name)
    return cleared


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"triton_kernels"}

# Skipped without torch (Intel Mac GGUF-only), plus librosa, whose numba chain fails (#5046).
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "librosa",
}

# Requirements with NO wheel on PyPI at any version (antlr4-python3-runtime arrives transitively via omegaconf==2.3.1). A user-level `no-build`/`only-binary = :all:` makes them unresolvable and fails the extras step (#8530), so a PACKAGE-SCOPED --no-binary overrides that policy for these names only. Keep in sync with .github/scripts/clean-machine-assert.sh and assert-nobuild.ps1.
SDIST_ONLY_PACKAGES = (
    "openai-whisper",
    "argbind",
    "randomname",
    "antlr4-python3-runtime",
)


def _sdist_only_build_args(*names: str) -> list[str]:
    """``--no-binary`` for each named wheel-less requirement, for uv and pip alike; naming a package the resolution never reaches is harmless."""
    args: list[str] = []
    for name in names:
        args += ["--no-binary", name]
    return args


def _extras_sdist_only_packages() -> tuple[str, ...]:
    """SDIST_ONLY_PACKAGES plus any this interpreter alone resolves to an sdist."""
    names = list(SDIST_ONLY_PACKAGES)
    # extras.txt pins MeCab==0.996.5 on macOS cp314+, the last release carrying an sdist. Conditional because everywhere else 0.996.13 resolves to a wheel and exempting it would force a compiler-dependent build.
    if IS_MACOS and sys.version_info >= (3, 14):
        names.append("MeCab")
    return tuple(names)


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


# Matches worker._is_importable_isolated: the same untrusted import, bounded the same way.
_FLASH_ATTN_IMPORT_PROBE_TIMEOUT = 300


def _flash_attn_importable() -> bool:
    """Whether flash_attn imports, checked out of process: a wrong-arch/ABI wheel installs fine and raises on import, and initialisation can hang rather than fail, so the probe is a bounded child."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import flash_attn"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = _FLASH_ATTN_IMPORT_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _remove_rejected_flash_attn() -> bool:
    """Uninstall a flash-attn that installed but will not import. Targets ``sys.executable``: --system ALONE would remove from the system Python, leaving the rejected wheel in the venv while setup reported it gone."""
    if USE_UV and shutil.which("uv"):
        cmd = ["uv", "pip", "uninstall"]
        if UV_NEEDS_SYSTEM:
            cmd.append("--system")
        cmd.extend(["--python", sys.executable, "flash-attn"])
    else:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn"]
    removed = subprocess.run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    return removed.returncode == 0


def _ensure_flash_attn() -> None:
    if _flash_attn_install_disabled():
        return
    if NO_TORCH:
        return
    if IS_WINDOWS or IS_MACOS:
        return
    if _flash_attn_importable():
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
                # Verify rather than trust the exit code, so setup reports what happened.
                if _flash_attn_importable():
                    return
                # Remove it before giving up: left installed, unsloth/models/_utils.py finds it by metadata and imports the native module in process, so a wheel that killed the probe would kill training too.
                if _remove_rejected_flash_attn():
                    _step(
                        "warning",
                        "flash-attn wheel installed but is not importable on this GPU; removed it",
                        _cyan,
                    )
                else:
                    # Say so plainly: it is still importable in process, so this is not the same state as never having installed it.
                    _step(
                        "warning",
                        "flash-attn wheel is not importable on this GPU and could not be "
                        "removed; uninstall flash-attn manually before training",
                        _cyan,
                    )
                break
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



USE_UV = False  # Set by _bootstrap_uv() at the start of install_python_stack()
UV_NEEDS_SYSTEM = False  # Set by _bootstrap_uv() via probe


def _bootstrap_uv() -> bool:
    """Check if uv is available and probe whether --system is needed."""
    global UV_NEEDS_SYSTEM
    if not shutil.which("uv"):
        return False
    # Explicit --python: uv can ignore the activated venv on some platforms.
    probe = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", sys.executable, "pip"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        **_windows_hidden_subprocess_kwargs(),
    )
    if probe.returncode != 0:
        # Retry with --system (some envs need it when uv cannot find a venv).
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
    """Return a temp copy, adjacent when writable, with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line for line in lines if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    # Beside the source so relative -r/-c includes resolve; a read-only tree (root-owned install, non-root user) falls back rather than aborting.
    kwargs = dict(
        mode = "w",
        prefix = f".{req.stem}-filtered-",
        suffix = ".txt",
        delete = False,
        encoding = "utf-8",
    )
    try:
        tmp = tempfile.NamedTemporaryFile(dir = req.parent, **kwargs)
    except OSError:
        tmp = tempfile.NamedTemporaryFile(**kwargs)
    tmp.writelines(filtered)
    tmp.close()
    return Path(tmp.name)


def _shared_base_requirements() -> Path | None:
    """The shared torch-bound requirements file, or None when it has no work."""
    if NO_TORCH:
        return None
    req = REQ_ROOT / "base.txt"
    try:
        # utf-8-sig: a BOM would otherwise read as content, scheduling an empty step.
        text = req.read_text(encoding = "utf-8-sig")
    except OSError:
        return None  # missing or unreadable: nothing to apply
    for line in text.splitlines():
        if line.split("#", 1)[0].strip():
            return req
    return None


_UNSLOTH_ZOO_GIT_URL = "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"


def _unsloth_zoo_ref() -> str:
    """The unsloth-zoo git ref the --local overlay installs (UNSLOTH_ZOO_REF, what install.sh reads into _ZOO_REF). Unset means main."""
    return os.environ.get("UNSLOTH_ZOO_REF", "").strip() or "main"


def _unsloth_zoo_git_spec() -> str:
    """The pip requirement for the unsloth-zoo overlay. An unset UNSLOTH_ZOO_REF leaves the URL bare rather than appending @main, so the default install is byte for byte the one every caller expects."""
    ref = os.environ.get("UNSLOTH_ZOO_REF", "").strip()
    return _UNSLOTH_ZOO_GIT_URL + ("@" + ref if ref else "")


def _overlay_local_core_package(
    name: str,
    local_repo: str,
    *,
    strict: bool = True,
) -> bool:
    """Install one core package from the source selected by --local. strict=False reports a failed install instead of exiting, which the metadata repair needs: by then it has already removed the records it is replacing."""
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    if canonical == "unsloth":
        step_label = f"overlaying local repo (editable): {local_repo}"
        install_label = "Overlaying local repo (editable)"
        args = ("-e", local_repo)
    elif canonical == "unsloth-zoo":
        zoo_ref = _unsloth_zoo_ref()
        step_label = f"overlaying unsloth-zoo from git {zoo_ref}"
        install_label = f"Overlaying unsloth-zoo from git {zoo_ref}"
        args = ("--force-reinstall", _unsloth_zoo_git_spec())
    else:
        return False
    _step(_LABEL, step_label)
    if not strict:
        return pip_install_try(install_label, "--no-cache-dir", "--no-deps", *args, constrain = False)
    pip_install(install_label, "--no-cache-dir", "--no-deps", *args, constrain = False)
    return True


def _overlay_local_core_packages(local_repo: str) -> None:
    for name in ("unsloth", "unsloth-zoo"):
        _overlay_local_core_package(name, local_repo)


def _run_ok(label: str, cmd: list) -> bool:
    """run() without the exit: the metadata repair has to unwind, not die."""
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        env = _install_env_for_cmd(cmd),
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0 and result.stdout:
        _safe_print(_redact_install_output(result.stdout))
    return result.returncode == 0


def _is_overlayable_core_package(name: str) -> bool:
    """Whether _overlay_local_core_package knows a source for this name."""
    return re.sub(r"[-_.]+", "-", name).lower() in ("unsloth", "unsloth-zoo")


def _overlay_source_spec(name: str, local_repo: str) -> str:
    """What pip would be asked to build for this overlay. unsloth-zoo comes from git, so an overlay is a network fetch and has to be staged before anything is uninstalled."""
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    if canonical == "unsloth":
        return local_repo
    if canonical == "unsloth-zoo":
        return _unsloth_zoo_git_spec()
    return ""


def _rewrite_minimal_metadata(path: str, name: str) -> bool:
    """Replace an unparseable METADATA with the least pip needs to uninstall by RECORD. False when there is no RECORD, the one case that must fail closed: nothing then knows which files belong to the package, and a replacement laid over them leaves whatever the new release dropped behind, still importable."""
    # invalid_metadata_paths() hands back Path objects, so normalise before the string work below.
    path = os.fspath(path)
    if not os.path.isfile(os.path.join(path, "RECORD")):
        return False
    stem = os.path.basename(path.rstrip(os.sep)).removesuffix(".dist-info")
    _package, separator, version = stem.rpartition("-")
    if not separator or not version:
        return False
    try:
        with open(os.path.join(path, "METADATA"), "w", encoding = "utf-8") as handle:
            handle.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    except OSError:
        return False
    return True


class _QuarantinedMetadata:
    """Invalid metadata directories moved aside, restorable until committed. A non-UTF-8 METADATA makes pip list, show and uninstall raise for the whole environment, so it must be out of the way before pip runs; deleting it outright would leave files and no install at all if staging then failed."""

    def __init__(self) -> None:
        self._holding = ""
        self._moved: list = []
        self._copied: list = []

    def _holding_dir(self) -> str:
        if not self._holding:
            self._holding = tempfile.mkdtemp(prefix = "unsloth_metadata_quarantine_")
        return self._holding

    def back_up(self, path) -> bool:
        """Keep a copy of a file about to be rewritten in place: the rewrite precedes staging, which can still fail, and without the copy the next run sees one readable synthetic record and never attempts the payload repair still owed. An absent METADATA is recorded as absent, so restore() deletes the synthetic file rather than inventing one."""
        path = os.fspath(path)
        if not os.path.exists(path):
            self._copied.append((path, None))
            return True
        target = os.path.join(self._holding_dir(), f"copy_{len(self._copied)}")
        try:
            shutil.copy2(path, target)
        except OSError:
            return False
        self._copied.append((path, target))
        return True

    def take(self, paths) -> bool:
        for path in paths:
            target = os.path.join(
                self._holding_dir(), f"{len(self._moved)}_{os.path.basename(path)}"
            )
            try:
                shutil.move(os.fspath(path), target)
            except OSError:
                return False
            self._moved.append((os.fspath(path), target))
        return True

    def forget_copies(self) -> None:
        """Drop the backed-up METADATA copies, keeping the moved directories: after a staged reinstall the wheel's own metadata is authoritative, so copying the original over it would re-break the package the rollback just repaired."""
        self._copied.clear()

    def restore(self) -> None:
        while self._moved:
            original, target = self._moved.pop()
            try:
                shutil.move(target, original)
            except OSError:
                pass
        while self._copied:
            original, target = self._copied.pop()
            try:
                if target is None:
                    os.remove(original)
                else:
                    shutil.copy2(target, original)
            except OSError:
                pass
        self.discard()

    def discard(self) -> None:
        if self._holding:
            shutil.rmtree(self._holding, ignore_errors = True)
            self._holding = ""
        self._moved.clear()
        self._copied.clear()


def _restore_from_staged(
    name: str,
    staged: str,
    removed_any: bool,
    quarantine: "_QuarantinedMetadata | None" = None,
) -> None:
    """Put the payload back when the uninstall loop stops part way: an earlier successful uninstall already deleted the package tree, so returning without this leaves a dist-info claiming an installed package whose files are gone."""
    if not (removed_any and staged):
        return
    if pip_install_try(
        f"Restoring {name} after an incomplete metadata repair",
        "--no-cache-dir",
        "--no-deps",
        "--force-reinstall",
        "--no-index",
        "--find-links",
        staged,
        name,
        # pip, not uv: the wheel is already built and sitting in staged, and uv would reject the unpinned name under UV_REQUIRE_HASHES with the package records already gone. Routing through pip also earns the PIP_REQUIRE_HASHES relaxation.
        force_pip = True,
    ):
        # The wheel just wrote its own valid metadata at the same path the rewritten record occupied, so the unwinding below must not put the original back over it. See _QuarantinedMetadata.forget_copies.
        if quarantine is not None:
            quarantine.forget_copies()
        _safe_print(_red(f"   restored {name} from the staged replacement"), file = sys.stderr)
    else:
        _safe_print(
            _red(f"   {name} is no longer installed. Re-run the installer to restore it."),
            file = sys.stderr,
        )


def _requirement_args(requirement: str, staging: str) -> "list[str]":
    """Hand pip the requirement as a file when it carries hashes: pip accepts --hash only from a requirements file, and the hashes stop it taking a different artifact of the same version from a source uv never considered."""
    if "--hash=" not in requirement:
        return [requirement]
    path = os.path.join(staging, "requirement.txt")
    with open(path, "w", encoding = "utf-8") as handle:
        handle.write(requirement + "\n")
    return ["-r", path]


def _stage_replacement(name: str):
    """Build the wheel that will replace a package before it is removed; None aborts the repair with the install intact. pip wheel, not pip download, because the install that follows runs --no-index and could not fetch setuptools for an sdist build; pip and not uv because uv has no `wheel` subcommand, so its index vars and upload cutoff are handed across explicitly."""
    requirement, overrides, build_options = name, {}, []
    offline_local = USE_UV and _uv_is_offline() and _is_local_source(name)
    if offline_local:
        # The checkout needs no network, but pip's isolated build fetches the build backend, which UV_OFFLINE does not reach, so it fails at "installing build dependencies". Build against the interpreter's own backend and forbid the index.
        build_options = ["--no-build-isolation"]
        overrides = {"PIP_NO_INDEX": "1"}
    if USE_UV and _uv_is_offline() and not _is_local_source(name):
        # A checkout on disk needs no network, so offline has nothing to say about it.
        _safe_print(
            _red(
                "   UV_OFFLINE is set and pip has no offline mode, so repairing "
                f"{name} would have to reach the network; leaving the install alone."
            ),
            file = sys.stderr,
        )
        return None
    if USE_UV and not _is_direct_reference(name):
        # A direct reference is its own provenance, so it is staged as written.
        plan = _uv_staging_plan(name)
        if plan is None:
            _safe_print(
                _red(
                    f"   uv could not resolve a replacement for {name}, so its source "
                    "cannot be preserved; leaving the install alone."
                ),
                file = sys.stderr,
            )
            return None
        requirement, overrides, build_options = plan
    cutoff_args = _uv_upload_cutoff_args()
    if cutoff_args is None:
        _safe_print(
            _red(
                "   UV_EXCLUDE_NEWER is set but this pip is too old to honour it "
                "(needs 25.3 for --uploaded-prior-to); leaving the install alone."
            ),
            file = sys.stderr,
        )
        return None
    staging = tempfile.mkdtemp(prefix = "unsloth_metadata_repair_")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        *cutoff_args,
        *build_options,
        "--wheel-dir",
        staging,
        *_requirement_args(requirement, staging),
    ]
    env = _install_env_for_cmd(cmd)
    if overrides:
        env = dict(env if env is not None else os.environ)
        env.update(overrides)
        # Written into the staging directory, so it is removed with it.
        env["PIP_CONFIG_FILE"] = _pip_config_without_sources(staging)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        env = env,
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode == 0 and glob.glob(os.path.join(staging, "*.whl")):
        return staging
    if VERBOSE and result.stdout:
        _safe_print(_redact_install_output(result.stdout))
    shutil.rmtree(staging, ignore_errors = True)
    return None


def _core_package_names(package_name: str) -> "tuple[str, ...]":
    """The distributions a run is responsible for. unsloth-zoo only for the default install: `--package X` installs X alone, so demanding the companion would fail the update on an unreachable zoo index."""
    default = re.sub(r"[-_.]+", "-", package_name).lower() == "unsloth"
    return (package_name, "unsloth-zoo") if default else (package_name,)


def _repair_damaged_core_payload(
    package_names: "tuple[str, ...]",
    *,
    local_repo: str = "",
    require_present: bool = False,
) -> bool:
    """Reinstall managed core packages whose recorded files are gone or truncated. An upgrade of a distribution already at the wanted version installs nothing (both uv and pip read metadata a payload quarantine leaves intact), so it has to be named for reinstall; skipped for a local checkout. Judged on the tree, not pip's exit code. `require_present` also refuses a distribution not installed at all, which has no RECORD and so reads as undamaged: off before the core phase, on after it."""
    if local_repo:
        return True
    if require_present:
        for name in package_names:
            try:
                present = any(install_manifest.installed_versions(name))
            except Exception:
                continue
            if not present:
                _safe_print(
                    _red(
                        f"   {name} is not installed, so this environment cannot run. "
                        "Rerun the installer with the package index available."
                    ),
                    file = sys.stderr,
                )
                return False
    damaged: list[str] = []
    seen: set[str] = set()
    for name in package_names:
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            if install_manifest.damaged_payload_files(name, limit = 1, budget_seconds = 0.0):
                damaged.append(name)
        except Exception:
            continue
    still_damaged: list[str] = []
    for name in damaged:
        _step(_LABEL, f"{name} is missing installed files; reinstalling it", _dim)
        # --no-deps: resolving the graph would drag the torch build along. Both flags: pip has no per-package form, uv's --reinstall is environment-wide.
        pip_install_try(
            f"Reinstalling {name}",
            "--no-cache-dir",
            "--no-deps",
            "--reinstall-package",
            name,
            "--force-reinstall",
            name,
        )
        importlib.invalidate_caches()
        # Presence first: --force-reinstall uninstalls before it installs, and what it leaves behind has no RECORD for the scan below to call damaged. Unconditional, unlike require_present: we only reinstall what the scan already found.
        try:
            remaining = (
                []
                if any(install_manifest.installed_versions(name))
                else [f"{name} is no longer installed"]
            )
        except Exception:
            remaining = []
        try:
            remaining = remaining or install_manifest.damaged_payload_files(
                name, budget_seconds = 0.0
            )
        except Exception:
            pass
        if remaining:
            still_damaged.append(name)
            _safe_print(
                _red(
                    f"   {name} is still missing installed files after a reinstall "
                    f"({remaining[0]}). An unreachable or offline index cannot "
                    "replace them; rerun with the index available."
                ),
                file = sys.stderr,
            )
    return not still_damaged


def _repair_duplicate_core_metadata(
    package_names: "tuple[str, ...]",
    *,
    local_repo: str = "",
    ci_source_overlay: str = "",
) -> bool:
    """Reinstall managed core packages whose metadata has more than one record: remove invalid records directly, since pip cannot parse them, then repeat a dependency-free uninstall until no valid record remains, because force-reinstall only uninstalls the one record pip's finder selects. The replacement is fetched BEFORE anything is removed, or an index unreachable at that moment leaves the venv with no unsloth and no way back."""
    duplicates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name in package_names:
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        versions = install_manifest.installed_versions(name)
        record_count = len(versions)
        # A sole `~` backup counts as one readable version, so metadata_conflict() cannot see it, yet pip and importlib both skip the directory. Repair it like any other duplicate.
        if install_manifest.metadata_conflict(
            versions
        ) or install_manifest.pip_backup_metadata_paths(name):
            duplicates.append((name, record_count))

    repaired: list[str] = []
    staging_dirs: list[str] = []
    # One quarantine per package, discarded as soon as that package is back: sharing one would, on a later failure, restore the first package's stale record over the install that replaced it.
    quarantine = _QuarantinedMetadata()
    succeeded = False
    try:
        for name, record_count in duplicates:
            quarantine = _QuarantinedMetadata()
            _step(_LABEL, f"duplicate metadata for {name} detected; reinstalling it", _dim)
            invalid_paths = install_manifest.invalid_metadata_paths(name)
            # Give pip a parseable METADATA beside every intact RECORD so it uninstalls exactly the files they list. Quarantining one instead drops its RECORD, leaving a module owned only by the older release on disk and importable while the repair reports success.
            unrewritable = [
                path
                for path in invalid_paths
                if not (
                    quarantine.back_up(os.path.join(path, "METADATA"))
                    and _rewrite_minimal_metadata(path, name)
                )
            ]
            # Every record must end up uninstallable by pip, or nothing is touched: quarantining one only hides it, so whatever that release owned alone stays importable while the repair reports success and deletes the evidence. Waiting for record_count to reach zero missed every case where another record survives.
            if unrewritable:
                _safe_print(
                    _red(
                        f"   the metadata for {name} at "
                        + ", ".join(os.path.basename(os.fspath(p)) for p in unrewritable)
                        + " cannot be read or rewritten, so the files that release owned "
                        "cannot be identified. Recreate the environment to repair it."
                    ),
                    file = sys.stderr,
                )
                return False
            # pip skips its own abandoned backup ("Ignoring invalid distribution ~nsloth") while its METADATA still names the project, so the record counts but no uninstall by name can consume it, and the loop below cannot converge. Quarantined, not deleted: staging can still fail.
            backups = install_manifest.pip_backup_metadata_paths(name)
            if backups and not quarantine.take(backups):
                _safe_print(
                    _red(f"   could not move pip's leftover backup for {name} aside"),
                    file = sys.stderr,
                )
                return False
            if invalid_paths or backups:
                importlib.invalidate_caches()
                record_count = len(install_manifest.installed_versions(name))
            # A backup names a payload pip has already renamed away, so nothing remains for a replacement to be laid over: a fresh install is right, and refusing would abort on the one state this can trivially fix.
            if invalid_paths and not record_count:
                # Nothing is left for pip to uninstall, so the replacement would be laid over a payload no record describes.
                _safe_print(
                    _red(
                        f"   no usable metadata record is left for {name}, so its "
                        "files cannot be removed safely. Recreate the environment "
                        "to repair it."
                    ),
                    file = sys.stderr,
                )
                return False

            canonical = re.sub(r"[-_.]+", "-", name).lower()
            source_repo = local_repo or (ci_source_overlay if canonical == "unsloth" else "")
            # A local or git source installs from a path or URL, so there is nothing to stage; anything else comes off an index, which has to be proven reachable while the current install is still intact.
            overlaid = bool(source_repo) and _is_overlayable_core_package(name)
            # Stage whichever source will be installed, overlay included: an overlay is a git fetch or a build, either of which can fail after the uninstall loop has removed every record.
            staged = _stage_replacement(
                _overlay_source_spec(name, source_repo) if overlaid else name
            )
            if staged is None:
                _safe_print(
                    _red(
                        f"   could not fetch a replacement for {name}; leaving "
                        "the existing install in place"
                    ),
                    file = sys.stderr,
                )
                return False
            staging_dirs.append(staged)

            removed_any = False
            while record_count:
                if not _run_ok(
                    f"Removing an installed metadata record for {name}",
                    [sys.executable, "-m", "pip", "uninstall", "-y", name],
                ):
                    _safe_print(
                        _red(f"   could not uninstall a metadata record for {name}"),
                        file = sys.stderr,
                    )
                    _restore_from_staged(name, staged, removed_any, quarantine)
                    return False
                importlib.invalidate_caches()
                remaining = len(install_manifest.installed_versions(name))
                if remaining >= record_count:
                    _safe_print(
                        _red(f"   could not remove every metadata record for {name}"),
                        file = sys.stderr,
                    )
                    _restore_from_staged(name, staged, removed_any, quarantine)
                    return False
                removed_any = True
                record_count = remaining

            # Installer handoffs may already have applied a local or CI source; restore that provenance now that no ambiguous record remains.
            restored = overlaid and _overlay_local_core_package(name, source_repo, strict = False)
            if not restored:
                # The overlay install is preferred because it keeps the editable or git provenance, but the staged wheel was built from that same source, so falling back to it never substitutes a release.
                restored = pip_install_try(
                    f"Repairing duplicate metadata for {name}",
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    "--no-index",
                    "--find-links",
                    staged,
                    name,
                    # As _restore_from_staged: pip, so a uv hash policy cannot reject the already-built wheel once every record has been removed.
                    force_pip = True,
                )
            if not restored:
                _safe_print(
                    _red(
                        f"   could not reinstall {name} after removing its duplicate "
                        "metadata; it is no longer installed. Re-run the installer "
                        "to restore it."
                    ),
                    file = sys.stderr,
                )
                return False
            repaired.append(name)
            # This package is back in place, so its old records must never return.
            quarantine.discard()

        importlib.invalidate_caches()
        unresolved = [
            name for name in repaired if not install_manifest.installed_version_probe(name)[0]
        ]
        if unresolved:
            _safe_print(
                _red(
                    "   package metadata is inconsistent after reinstall: " + ", ".join(unresolved)
                ),
                file = sys.stderr,
            )
            return False
        succeeded = True
        return True
    finally:
        for staging in staging_dirs:
            shutil.rmtree(staging, ignore_errors = True)
        # Anything short of a completed repair puts the quarantined records back, so a failure leaves the environment as it was found.
        if succeeded:
            quarantine.discard()
        else:
            quarantine.restore()


def _translate_pip_args_for_uv(args: tuple[str, ...]) -> list[str]:
    """Translate pip flags to their uv equivalents."""
    translated: list[str] = []
    for arg in args:
        if arg == "--no-cache-dir":
            continue  # uv cache is fast; drop this flag
        elif arg == "--force-reinstall" and "--reinstall-package" in args:
            # Already targeted by name; uv's --reinstall would rebuild torch.
            continue
        elif arg == "--force-reinstall":
            translated.append("--reinstall")
        else:
            translated.append(arg)
    return translated


def _build_pip_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a standard pip install command. pip has no --upgrade-package, so uv's flag is translated rather than dropped: dropping it made this fallback a no-op on the update path, where pip installed nothing and still reported success. --upgrade-strategy=only-if-needed is the load-bearing part, upgrading the named packages without dragging the existing torch build along."""
    cmd = [sys.executable, "-m", "pip", "install"]
    upgrade: list[str] = []
    drop_next = ""
    for arg in args:
        if drop_next:
            # The previous flag's value: never a bare positional of its own.
            if drop_next == "--upgrade-package":
                upgrade.append(arg)
            drop_next = ""
            continue
        if arg == "--upgrade-package":
            drop_next = arg  # the flag; its value is the package to upgrade
            continue
        if arg == "--reinstall-package":
            # uv-only. Its caller's --force-reinstall is environment-wide for pip, safe only because that call also passes --no-deps.
            drop_next = arg
            continue
        cmd.append(arg)
    if upgrade:
        cmd += ["--upgrade", "--upgrade-strategy", "only-if-needed"]
        # Every current caller also names these as positionals or via -r, but a future one might not, and pip would then upgrade nothing.
        cmd += [name for name in upgrade if name not in cmd]
    return cmd


def _build_uv_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a uv pip install command with translated flags."""
    cmd = ["uv", "pip", "install"]
    if UV_NEEDS_SYSTEM:
        cmd.append("--system")
    cmd.extend(["--python", sys.executable])
    cmd.extend(_translate_pip_args_for_uv(args))
    # No --torch-backend by default, and never on a pinned index: it would defeat the pin.
    _tb = os.environ.get("UV_TORCH_BACKEND", "")
    if _tb and not _is_pinned_index_cmd(cmd):
        cmd.append(f"--torch-backend={_tb}")
    return cmd


# uv ranks --index-url LOWEST, so inherited index vars defeat a pinned repair; neutralise them.
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
    # PIP_NO_INDEX=1 makes the pip fallback ignore ALL indexes, defeating --index-url; PIP_INDEX_URL is dropped too so a stale mirror env cannot outrank the pin.
    "PIP_NO_INDEX",
    "PIP_INDEX_URL",
)


def _is_pinned_index_cmd(cmd: "list[str] | tuple[str, ...]") -> bool:
    """True when the command pins an index via --index-url / --default-index."""
    return any(arg in ("--index-url", "--default-index") for arg in cmd)


# Restrictive policy a pinned install must not inherit from the ENVIRONMENT: the pinned branch neutralises the config FILES, but an env var outranks a config file, so a hardened shell could still fail a pinned torch repair (#8530).
_PM_POLICY_ENV_VARS = (
    "UV_NO_BUILD",
    "UV_NO_BUILD_PACKAGE",
    "UV_NO_BINARY",
    "UV_NO_BINARY_PACKAGE",
    "UV_REQUIRE_HASHES",
    "UV_EXCLUDE_NEWER",
    "PIP_ONLY_BINARY",
    "PIP_NO_BINARY",
    "PIP_REQUIRE_HASHES",
)


def _relaxed_pip_policy_env(cmd: "list[str]") -> "dict[str, str]":
    """Overrides that stop a hardened user pip config failing the installer's own pip. Empty for anything that is not a `pip install` / `download` / `wheel` this module drives, so non-pinned installs inherit the caller env unchanged; `wheel` is included because the duplicate-metadata repair stages with it. `require-hashes = true` rejects every requirements file we ship and took the pip FALLBACK down in #8530, and pip applies env vars AFTER config files, so PIP_REQUIRE_HASHES=0 overrides it while index-url, trusted-host, cert and proxy stay in force."""
    if cmd[:1] == ["uv"] or not any(arg in ("install", "download", "wheel") for arg in cmd):
        return {}
    return {"PIP_REQUIRE_HASHES": "0"}


def _uv_is_offline() -> bool:
    """True when uv has been told not to touch the network."""
    return os.environ.get("UV_OFFLINE", "").strip().lower() not in ("", "0", "false")


def _uv_staging_plan(name: str) -> "tuple[str, dict[str, str]] | None":
    """Ask uv which release and index it would use and reproduce that with pip; None when uv could not resolve it. Staging must run pip (uv has no `wheel`), and translating uv's index configuration out of the environment cannot be made correct, so a miss would uninstall a private build and reinstall the public package of the same name. `uv pip compile --emit-index-annotation` names the exact index per package instead."""
    cmd = [
        "uv",
        "pip",
        "compile",
        "--no-deps",
        "--python",
        sys.executable,
        "--emit-index-url",
        "--emit-find-links",
        "--emit-index-annotation",
        "--emit-build-options",
        "--generate-hashes",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            input = name.encode(),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            **_windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        return None
    if result.returncode != 0:
        if VERBOSE and result.stderr:
            _safe_print(_redact_install_output(result.stderr))
        return None
    requirement, origin = "", ""
    hashes: list[str] = []
    emitted: list[str] = []
    find_links: list[str] = []
    build_options: list[str] = []
    canonical = _canonical_package_name(name)
    for raw in (result.stdout or b"").decode("utf-8", "replace").splitlines():
        line = raw.strip()
        if line.startswith(("--index-url ", "--extra-index-url ")):
            emitted.append(line.split(" ", 1)[1].strip())
        elif line.startswith("--find-links "):
            find_links.append(line.split(" ", 1)[1].strip())
        elif line.startswith("--hash="):
            hashes.append(line.rstrip("\\").strip())
        elif line.startswith(("--no-binary ", "--only-binary ")):
            # uv's artifact policy, which pip reads none of. Without it the repair can download a wheel under a no-binary rule or build an sdist under an only-binary one, changing the artifact type mid-repair.
            option, _, value = line.partition(" ")
            build_options.extend((option, value.strip()))
        elif line.startswith("# from "):
            # Names the index this package came from, but is not usable as written: measured on uv 0.10.7 the annotation drops userinfo while the emitted index lines keep it, so it is matched back to the emitted URL. A private index would answer 401 otherwise.
            origin = line[len("# from ") :].strip() or origin
        elif line and not line.startswith(("#", "-")):
            # uv continues a hashed pin onto the following lines with a backslash.
            pinned = line.split(";", 1)[0].rstrip("\\").strip()
            if _canonical_package_name(_requirement_name(pinned)) == canonical:
                requirement = pinned
    if not requirement:
        return None
    # Replaying uv's answer replaces pip's candidate sources rather than adding to them: an inherited PIP_NO_INDEX would block the index uv picked, and an extra index or find-links could satisfy the version from a source uv never looked at. Empty rather than deleted, since on pip 26.2 an empty value reads as unset.
    overrides = {
        "PIP_EXTRA_INDEX_URL": "",
        "PIP_NO_INDEX": "",
        "PIP_FIND_LINKS": " ".join(find_links),
    }
    index_url = _credentialed_index(origin, emitted)
    if index_url:
        overrides["PIP_INDEX_URL"] = index_url
    elif find_links:
        # uv resolved this from a flat source with no index in play. Leaving PIP_NO_INDEX cleared would hand pip the default PyPI and let it stage the same name and version from a source uv was told to exclude.
        overrides["PIP_NO_INDEX"] = "1"
    # Measured on uv 0.10.7: --emit-build-options surfaces uv.toml's policy but not its environment-variable spelling, so that half is translated by hand where pip has no native setting. UV_KEYRING_PROVIDER likewise: uv reaches an authenticated index through the keyring CLI, and the URL alone leaves pip unable to fetch what uv resolved.
    for uv_name, pip_name in (
        ("UV_NO_BINARY", "PIP_NO_BINARY"),
        ("UV_ONLY_BINARY", "PIP_ONLY_BINARY"),
        ("UV_KEYRING_PROVIDER", "PIP_KEYRING_PROVIDER"),
    ):
        value = os.environ.get(uv_name, "").strip()
        if value and not os.environ.get(pip_name):
            overrides[pip_name] = value
    if hashes:
        # The hashes are what make this safe: pip verifies them even with PIP_REQUIRE_HASHES=0, and neither PIP_CONFIG_FILE nor --isolated suppresses a site pip.conf, so pip may still consult another source but can no longer accept a different artifact from it.
        requirement = " \\\n    ".join([requirement, *hashes])
    return requirement, overrides, build_options


_PIP_SOURCE_CONFIG_KEYS = ("index-url", "extra-index-url", "find-links", "no-index")


def _pip_config_without_sources(directory: str) -> str:
    """Write pip's own configuration back minus the candidate sources: on pip 26.2 an empty PIP_EXTRA_INDEX_URL does NOT suppress pip.conf's. Dropping the config wholesale would take proxy, cert, client-cert and trusted-host with it, which is how a private index is reached, so only the four source keys are removed. `pip config list` merges global, user and site in pip's own order; `:env:` entries are handled above."""
    path = os.path.join(directory, "pip.conf")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            **_windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        result = None
    sections: dict[str, list[tuple[str, str]]] = {}
    if result is not None and result.returncode == 0:
        for line in (result.stdout or b"").decode("utf-8", "replace").splitlines():
            name, separator, raw = line.partition("=")
            if not separator or name.startswith(":env:"):
                continue
            section, _, option = name.strip().rpartition(".")
            if not section or option in _PIP_SOURCE_CONFIG_KEYS:
                continue
            try:
                value = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                continue
            # pip renders a multi-value setting as one newline separated string; an indented continuation is how it is spelled back into a config file.
            sections.setdefault(section, []).append((option, str(value).replace("\n", "\n    ")))
    with open(path, "w", encoding = "utf-8") as handle:
        for section, options in sections.items():
            handle.write(f"[{section}]\n")
            for option, value in options:
                handle.write(
                    f"{option} ={value}\n" if value.startswith("\n") else f"{option} = {value}\n"
                )
    return path


def _requirement_name(requirement: str) -> str:
    """The distribution name from a pin or PEP 508 direct reference: an override can redirect a package to a path or URL, and uv then emits `name @ reference`, which read whole left the requirement empty and aborted every repair."""
    head = requirement.split("==", 1)[0]
    return head.split("@", 1)[0].strip()


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _strip_userinfo(url: str) -> str:
    """The URL without any `user:password@`, which is how uv writes an annotation."""
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    authority, slash, tail = rest.partition("/")
    _credentials, at_sign, host = authority.rpartition("@")
    return f"{scheme}://{host}{slash}{tail}" if at_sign else url


def _credentialed_index(origin: str, emitted: "list[str]") -> str:
    """The emitted index matching the annotated origin, credentials intact: uv emits every index with credentials but strips userinfo from the `# from` annotation, so the annotation alone hands pip an unauthenticated URL for a private index (401). Matched on the credential-free form."""
    if origin:
        target = _strip_userinfo(origin).rstrip("/")
        matches = [url for url in emitted if _strip_userinfo(url).rstrip("/") == target]
        # One index can be emitted both with and without credentials; the credentialed form is the point here, so it wins over a bare match on the same URL.
        for url in matches:
            if _strip_userinfo(url) != url:
                return url
        if matches:
            return matches[0]
    # The origin is not an emitted index, so it is a find-links source, which belongs in PIP_FIND_LINKS and must not displace the real index: an sdist from a flat directory still needs the index for its build backend.
    return emitted[0] if emitted else ""


def _is_local_source(requirement: str) -> bool:
    """True when the replacement is already on disk, so no network is needed."""
    return os.path.exists(requirement)


def _is_direct_reference(requirement: str) -> bool:
    """True when the requirement already names the source to build from: an overlay's git URL or checkout carries its own provenance, and asking uv would compare a bare spec against output that appends the resolved commit."""
    return "://" in requirement or _is_local_source(requirement)


def _uv_upload_cutoff_args() -> "list[str] | None":
    """pip arguments carrying UV_EXCLUDE_NEWER, or None when it cannot be honoured: pip's equivalent --uploaded-prior-to exists only from pip 25.3, and refusing to stage aborts with the installation intact rather than installing a wheel the cutoff forbids."""
    cutoff = os.environ.get("UV_EXCLUDE_NEWER", "").strip()
    if not cutoff:
        return []
    if not _pip_supports_upload_cutoff():
        return None
    return ["--uploaded-prior-to", cutoff]


@functools.lru_cache(maxsize = 1)
def _pip_supports_upload_cutoff() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--help"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            **_windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        return False
    return b"--uploaded-prior-to" in (result.stdout or b"")


def _install_env_for_cmd(cmd: "list[str]") -> "dict[str, str] | None":
    """An env with the uv index vars stripped for a pinned-index install; None (inherit) when the command pins no index, so ordinary installs honour the user's mirror. Pinned commands also set UV_NO_CONFIG=1, since a discovered uv.toml outranks the CLI pin, and point PIP_CONFIG_FILE at os.devnull. Mirrors install.sh's gate (#6898)."""
    if not _is_pinned_index_cmd(cmd):
        relaxed = _relaxed_pip_policy_env(cmd)
        if not relaxed:
            return None
        env = os.environ.copy()
        env.update(relaxed)
        return env
    env = os.environ.copy()
    for name in _UV_INDEX_ENV_VARS:
        env.pop(name, None)
    for name in _PM_POLICY_ENV_VARS:
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
    """Like pip_install but returns False on failure instead of exiting, for optional installs with a follow-up fallback."""
    # Same reason as pip_install: this installs torch too (the Windows AMD ROCm trio), so the memoized classification must not survive it.
    _invalidate_torch_runtime_probe()
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
    # Any pip operation can change which torch is installed, so the memoized classification must not outlive it.
    _invalidate_torch_runtime_probe()
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
        # Linux aarch64 / Windows ARM64 / Intel Mac have no torchcodec wheel, and `unsloth studio update --local` passes no --no-torch, so skip it independently or the audio extras step takes down the whole update. Nothing feeds torchcodec now; this stays for any file that reintroduces it.
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
                # Echo success under UNSLOTH_VERBOSE, as install.sh's run_install_cmd does: without it clean-machine-assert.sh's `nobuild` never saw uv's "Building <pkg>==<ver>" and a source build reported "built: none". Redacted, since uv echoes credentialed URLs.
                if VERBOSE and result.stdout:
                    _safe_print(_redact_install_output(result.stdout))
                return
            _safe_print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                _safe_print(_redact_install_output(result.stdout))

        pip_cmd = _build_pip_cmd(args) + constraint_args_pip + req_args_pip
        pip_label = f"{label} (pip)" if USE_UV else label
        result = run(pip_label, pip_cmd, check = False)
        if result.returncode != 0:
            # Retry once, and only after clearing something pip named as unremovable: a blind retry of a failing install just doubles the wait.
            cleared = _purge_recordless_distributions(result.stdout)
            if not cleared:
                _report_failed_command(pip_label, result)
            _step(_LABEL, f"cleared half-written {', '.join(cleared)}, retrying...", _dim)
            run(pip_label, pip_cmd)
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
        encoding = "utf-8",
        errors = "replace",
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




def _has_working_git() -> bool:
    """Match install.sh's _has_working_git: on PATH *and* runnable, since a bare xcrun shim counts as missing there. Testing only shutil.which had the installer promise to skip the git+https triton requirement and fetch it anyway."""
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


_MLX_HEALTH_PROBE = (
    "import json, sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "from utils.mlx_repair import mlx_stack_blockers;"
    "print(json.dumps(mlx_stack_blockers()))"
)


def _report_mlx_stack_health() -> None:
    """Name what would keep Train off on this Apple Silicon host. Advisory only, but it must not be silent, which is the whole of the reported "Train is blacked out after an update". Out of process, because a half-installed mlx / mlx_lm / mlx_vlm can abort rather than raise."""
    backend = str(SCRIPT_DIR / "backend")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _MLX_HEALTH_PROBE, backend],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 180,
            **_windows_hidden_subprocess_kwargs(),
        )
        blockers = json.loads(probe.stdout.strip() or "null")
    except Exception as exc:  # noqa: BLE001 - advisory, never fail the install
        _step("mlx", f"could not verify the MLX stack ({exc})", _dim)
        return
    if blockers is None:
        _step("mlx", "could not verify the MLX stack", _dim)
        return
    if not blockers:
        _step("mlx", "training stack ready")
        return
    _step("mlx", "Train and Export will stay off until this is resolved:", _cyan)
    for blocker in blockers:
        _step("", blocker, _cyan)


def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL, _PROGRESS_LINE_ACTIVE
    _STEP = 0
    # An aborted earlier run leaves it set, and every _safe_print() consumes it, so the first message would get a stray newline.
    _PROGRESS_LINE_ACTIVE = False

    # install.sh sets SKIP_STUDIO_BASE=1 to avoid reinstalling the core packages; `studio update` does NOT, so unsloth + unsloth-zoo are reinstalled to pick up new versions.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # --package installs a different package name (for testing).
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # --local overlays a local repo checkout after updating deps.
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    # Read where the overlay runs, so UNSLOTH_ZOO_REF reaches the metadata-repair reinstall path too. Clean-machine CI overlays only unsloth, not the full local source pair.
    ci_source_overlay = os.environ.get("UNSLOTH_CI_SOURCE_OVERLAY", "")
    # Three lettered steps on top of the numbered ones: anyio repair (8b), diffusers pin (11b), torchcodec (13b, which reports progress on every branch including its skips).
    base_total = 13 if IS_WINDOWS else 14
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    if not IS_MACOS and not NO_TORCH:
        base_total += 1  # ROCm torch check (step 2b), non-macOS
        if not IS_WINDOWS:
            base_total += 2  # flash-attn + torch final repair (step 13), Linux
        else:
            base_total += 1  # torch flavor invariant (step 13w), Windows
    if IS_MAC_ARM and not NO_TORCH:
        base_total += 1  # MLX stack, same gate as the step itself
    base_requirements = _shared_base_requirements() if skip_base else None
    # Core packages and shared base requirements occupy one progress slot. A shell-installer handoff skips that slot only while base.txt has no work.
    _TOTAL = base_total - int(skip_base and base_requirements is None)

    # Drop it up front: a missing manifest is what tells the CLI, setup.sh and the preflight that an interrupted run left the venv half-built. Stop if it survives rather than mutate the venv behind a marker that still verifies.
    if not install_manifest.remove_manifest():
        _safe_print(
            f"error: could not remove the stale {install_manifest.MANIFEST_NAME} in "
            f"{install_manifest.venv_root()}; refusing to install behind a marker "
            "that would still report this venv as complete",
            file = sys.stderr,
        )
        return 1

    # The manifest just went away, so record the mode in a marker that survives a pass killed part-way. Otherwise the next update sees neither, reads the absent torch as a stale venv, and tries to delete the running environment.
    install_manifest.set_no_torch_marker(NO_TORCH)

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
        # uv-created venvs omit pip: ensurepip, else direct upgrade.
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

    # A superseded dist-info makes version() and every RECORD consumer choose an arbitrary package version. Repair it before any fast package operation, including installer handoffs that set skip_base after their own upgrade.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

    # Intact metadata over a missing payload: the core phase would audit it as satisfied. Unbudgeted, since this run is already doing a full install.
    if not _repair_damaged_core_payload(_core_package_names(package_name), local_repo = local_repo):
        return 1

    # macOS arm64: not keyed off skip_base, because a fresh install skips core packages but still needs MLX; not on --no-torch, which declined the training stack. Pins stay aligned with utils/mlx_repair.py and unsloth-zoo; UV_OVERRIDE (set at module load) relaxes the mlx-vlm / mlx-lm transformers pin.
    if IS_MAC_ARM and not NO_TORCH:
        # Both branches spend the slot, so the denominator does not depend on the host.
        if _mlx_pins_are_installable():
            _progress("MLX stack (Apple Silicon)")
            pip_install(
                "Installing MLX stack (mlx + mlx-lm + mlx-vlm)",
                "--no-cache-dir",
                "--upgrade",
                "mlx==0.32.1",
                "mlx-metal==0.32.1",
                "mlx-lm==0.31.3",
                "mlx-vlm>=0.4.4,<0.7.0",
            )
        else:
            _progress("MLX stack (skipped, no wheel for this macOS or Python)")
            _note(
                f"macOS {_macos_release_major() or 'unknown'} on Python "
                f"{sys.version_info.major}.{sys.version_info.minor} publishes no wheel for the "
                f"supported MLX versions (needs macOS {_MLX_MIN_MACOS_MAJOR}+ and Python "
                f"{_MLX_MIN_PYTHON[0]}.{_MLX_MIN_PYTHON[1]}+) -- leaving Train/Export disabled "
                "rather than failing the install"
            )

    # gfx906: the base install below resolves unsloth's unconditional bitsandbytes dep to a generic CUDA wheel with no gfx906 kernels. Record bnb's presence now so _ensure_rocm_torch can drop a freshly pulled wheel while keeping a source build.
    global _GFX906_BNB_ABSENT_BEFORE_BASE
    if not skip_base:
        _GFX906_BNB_ABSENT_BEFORE_BASE = not _bitsandbytes_installed()

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        # install.sh / install.ps1 already installed both core distributions.
        pass
    elif NO_TORCH:
        # No-torch update path: --no-deps throughout (PyPI metadata makes torch a hard dep).
        _progress("base packages (no torch)")
        desktop_min_ver = os.environ.get("UNSLOTH_DESKTOP_BACKEND_VERSION", "").strip()
        unsloth_spec = (
            f"{package_name}>={desktop_min_ver}"
            if (desktop_min_ver and package_name == "unsloth")
            else package_name
        )
        pip_install(
            f"Updating {package_name} + unsloth-zoo (no-torch mode)",
            "--no-cache-dir",
            "--no-deps",
            "--upgrade-package",
            package_name,
            "--upgrade-package",
            "unsloth-zoo",
            unsloth_spec,
            "unsloth-zoo",
        )
        # pydantic WITH deps (all torch-free) so pip pins a matching pydantic-core.
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
            _overlay_local_core_packages(local_repo)
    elif local_repo:
        # Local dev install: update the released core packages, then overlay the checkout as an editable install (--no-deps so torch is not re-resolved).
        _progress("base packages")
        pip_install(
            "Updating core packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            "unsloth",
            "unsloth-zoo",
        )
        _overlay_local_core_packages(local_repo)
    elif package_name != "unsloth":
        _progress("base packages")
        pip_install(
            f"Installing {package_name}",
            "--no-cache-dir",
            package_name,
        )
    else:
        _progress("base packages")
        desktop_min_ver = os.environ.get("UNSLOTH_DESKTOP_BACKEND_VERSION", "").strip()
        unsloth_spec = (
            f"{package_name}>={desktop_min_ver}"
            if (desktop_min_ver and package_name == "unsloth")
            else package_name
        )
        pip_install(
            "Updating core packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            unsloth_spec,
            "unsloth-zoo",
        )

    if not skip_base:
        base_requirements = _shared_base_requirements()

    # Independent of the core phase: the shell installers skip that after installing the two distributions inline, but still apply this file.
    if base_requirements is not None:
        if skip_base:
            _progress("base requirements")
        else:
            _step(_LABEL, "applying shared base requirements")
        pip_install(
            "Applying shared base requirements",
            "--no-cache-dir",
            req = base_requirements,
        )

    # 2b. Torch repair (wrong-family / CPU-only); must follow base packages so torch is present.
    if not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("check"))
        _ensure_cuda_torch()
        _ensure_rocm_torch()
        _ensure_xpu_torch()
        _ensure_cpu_torch()
        # Last, after every torch migration: the swap keys off the installed +xpu label, so a CPU pin over an XPU venv would leave XPU triton under a CPU torch.
        _ensure_xpu_triton()

    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
        # Validate actual AMD GPU presence, not just tool existence.
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
            # Skip amd-smi without a HIP SDK (UAC prompt); only a best-effort note is lost.
            if _wcmd[0] == "amd-smi" and not _amd_smi_allowed():
                continue
            try:
                _wr = subprocess.run(
                    [_wexe, *_wcmd[1:]],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
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

    _progress("unsloth extras")
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        # extras.txt is where the wheel-less requirements live, so a user-level no-build/only-binary policy fails this step first (#8530).
        *_sdist_only_build_args(*_extras_sdist_only_packages()),
        req = REQ_ROOT / "extras.txt",
    )

    _progress("extra codecs")
    pip_install(
        "Installing extras (no-deps)",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "extras-no-deps.txt",
    )

    # 4. Reinstall the torch-matched torchao override only when the pin changes, since Windows can remove shared files during replacement. Skipped without torch, or on Windows ROCm, which has no working build.
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    elif _rocm_windows_torch_installed or _installed_torch_is_windows_rocm():
        # No working Windows ROCm torchao build (crashes on import; stubbed at runtime).
        _progress("dependency overrides (skipped, Windows ROCm)")
        _note("Windows ROCm -- skipping torchao (no working build; stubbed at runtime)")
    else:
        _progress("dependency overrides")
        _install_torchao_for_torch(_probe_installed_torch_version())

    # 5. Triton kernels (no-deps, from source). Skipped on Windows/macOS and without git, since the requirement is a git+https URL; a training speedup only, so warn rather than fail the install.
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




    _progress("studio deps")
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

    _progress("anyio check")
    _repair_bad_anyio()

    _progress("data designer deps")
    pip_install(
        "Installing data-designer base dependencies",
        "--no-cache-dir",
        req = SINGLE_ENV / "data-designer-deps.txt",
    )

    _progress("data designer")
    pip_install(
        "Installing data-designer",
        "--no-cache-dir",
        "--no-deps",
        req = SINGLE_ENV / "data-designer.txt",
    )

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

    # 11b. The pinned Diffusers release, after every other requirements file so nothing re-resolves it back to a release, and outside every skip_base / NO_TORCH branch so it reaches every path. constrain stays on so a future constraints.txt entry wins.
    _progress("diffusers pin")
    pip_install(
        "Installing the pinned Diffusers release",
        "--no-cache-dir",
        req = REQ_ROOT / "diffusers-pin.txt",
    )

    _progress("finalizing")
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )

    # 13. Final torch repair. Steps above can pull CUDA torch from PyPI, so repair last.
    torch_flavor_tag = ""
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("final"))
        _torch_before_repair = str(_probe_installed_torch_version() or "")
        _ensure_cuda_torch()
        _ensure_rocm_torch()
        _ensure_xpu_torch()
        _ensure_cpu_torch()
        # Last, after every torch migration: the swap keys off the installed +xpu label, so a CPU pin over an XPU venv would leave XPU triton under a CPU torch.
        _ensure_xpu_triton()
        # Step 4 chose torchao from the torch these repairs then moved. The XPU pin is the
        # sharp case: torch>=2.6,<2.11.0 lands below torchao 0.18.0's 2.11 floor. Only the
        # Windows repair reaches _resync_torch_coupled_packages, so on Linux nothing
        # re-selected it (#10493).
        _torch_after_repair = str(_probe_installed_torch_version() or "")
        if _torch_after_repair and _torch_after_repair != _torch_before_repair:
            _note(
                f"torch moved from {_torch_before_repair or 'unknown'} to "
                f"{_torch_after_repair} during the repair -- re-selecting torchao"
            )
            _install_torchao_for_torch(_torch_after_repair)

    # 13w. Windows torch flavor invariant, separate from step 13's Linux-shaped repair set but in the same position: last, after the with-deps steps re-resolved torch.
    if IS_WINDOWS and not NO_TORCH:
        _progress(_torch_step_label("flavor"))
        torch_flavor_tag = _expected_torch_flavor_tag()
        if not _ensure_expected_torch_flavor(torch_flavor_tag):
            return 1
        # A direct run has no setup.ps1 postlude to swap triton back. After the invariant, because the swap keys off the installed +xpu label.
        _ensure_xpu_triton()
    elif not NO_TORCH:
        # Resolve it on the other platforms too, for the RECORD only: without this a Linux GPU box installed with a transient explicit CPU pin looks, on the next launch, like a CPU wheel beside a physical GPU.
        torch_flavor_tag = _expected_torch_flavor_tag()

    # 13b. torchcodec, pinned to the venv's torch minor (_select_torchcodec_spec), which
    #      extras-no-deps.txt cannot do because markers cannot see torch. Must run after the
    #      repair above: that can move torch onto another minor, staling an earlier choice.
    #      The runtime probe reports nothing on a timeout (the wedged-driver host it exists to
    #      tolerate), so read the installed metadata before giving up: guessing here means
    #      downgrading a matching codec onto the default and recreating the mismatch.
    _codec_torch_ver = None
    if not NO_TORCH and not PLATFORM_LACKS_TORCHCODEC_WHEEL:
        _codec_torch_ver = _probe_installed_torch_version() or _installed_distribution_version(
            "torch"
        )
    if NO_TORCH:
        _progress("torchcodec (skipped, no torch)")
    elif PLATFORM_LACKS_TORCHCODEC_WHEEL:
        _progress("torchcodec (skipped, no wheel for this platform)")
    elif not _codec_torch_ver:
        _progress("torchcodec (skipped, torch version unknown)")
        _note("could not read the installed torch version -- leaving torchcodec alone")
    elif not _torchcodec_spec_is_installable(_select_torchcodec_spec(_codec_torch_ver)):
        # This platform published no wheel in the window this torch selects. Skipping is what
        # such a host got before this step existed; attempting it would end the install.
        _progress("torchcodec (skipped, no wheel for this torch on this platform)")
        _note(
            f"torch {_codec_torch_ver} wants {_select_torchcodec_spec(_codec_torch_ver)}, "
            "which publishes no wheel here -- leaving audio decoding disabled"
        )
    else:
        _progress("torchcodec")
        _codec_spec = _select_torchcodec_spec(_codec_torch_ver)
        # Pin the index to the resident torch's build. The version alone is not enough:
        # torchcodec ships a separate wheel per accelerator, and the right version from the
        # wrong index is a codec that cannot load.
        _codec_index = _torchcodec_index_url(_codec_torch_ver, _codec_spec)
        _codec_args = ("--no-deps", "--no-cache-dir")
        _codec_rebuild = False
        if _codec_index:
            _codec_args += ("--index-url", _codec_index)
            # A codec already inside the window satisfies the requirement, so pip and uv
            # skip it and the pin never fetches anything -- leaving in place exactly the
            # wrong-accelerator wheel this pin exists to replace. Provenance is readable
            # from the version: the torch indexes carry a +cuNNN / +cpu local tag and PyPI
            # forbids one, so a local tag that is missing or different means another build.
            _codec_have = _installed_distribution_version("torchcodec") or ""
            # The tag the PIN will fetch, not the resident torch's: an xpu torch is served
            # the cpu wheel, so comparing against "xpu" would force-reinstall every run.
            _codec_want = _torchcodec_index_tag(_codec_torch_ver)
            if _codec_have and (
                _codec_want is None or _codec_have.partition("+")[2].strip().lower() != _codec_want
            ):
                _codec_args += ("--force-reinstall",)
                _codec_rebuild = True
        _safe_print(
            f"   torch {_codec_torch_ver} detected -- installing {_codec_spec}"
            # Redacted for display only; the installer below still gets the exact URL.
            # An authenticated mirror puts its credentials in the userinfo or a query
            # token, and this line is printed straight to the terminal and CI log rather
            # than through _redact_install_output, which only covers captured pip output.
            + (f" from {_strip_index_url_credentials(_codec_index)}" if _codec_index else "")
            + (" (replacing a build from another index)" if _codec_rebuild else "")
        )
        # pip_install_try, not pip_install: audio is an optional extra, and pip_install's
        # failure path is run(check=True), i.e. exit. Letting an audio wheel end a Studio
        # install inverts the rule the extras-no-deps filter above exists to enforce, and
        # the index can refuse for reasons no local table predicts -- a yanked release, an
        # offline mirror, a platform tag added or dropped upstream after this shipped.
        _codec_ok = pip_install_try("Installing torchcodec", *_codec_args, _codec_spec)
        _codec_fellback = False
        if not _codec_ok and _codec_index:
            # The leaf may not carry this window at all: cu129 serves torch 2.8-2.13 but
            # publishes no 0.8 or 0.9. Retry unpinned rather than table what each index
            # holds, which is what goes stale. _torchcodec_provenance_hint explains at
            # import if that lands another accelerator's build.
            _note(
                f"{_strip_index_url_credentials(_codec_index)} did not serve {_codec_spec} "
                "-- retrying from the default index"
            )
            # Drop only the pin: --force-reinstall must survive, or pip calls the requirement
            # satisfied and leaves the wrong-accelerator wheel in place.
            _codec_retry_args = [
                a
                for i, a in enumerate(_codec_args)
                if a != "--index-url" and _codec_args[i - 1] != "--index-url"
            ]
            # _codec_index stays set: PyPI's wheel is still a CUDA build on Linux, so the
            # NPP step below still applies.
            _codec_ok = pip_install_try("Installing torchcodec", *_codec_retry_args, _codec_spec)
            _codec_fellback = _codec_ok
        if not _codec_ok:
            _note(
                f"could not install {_codec_spec} -- audio decoding stays disabled, "
                "the rest of the install is unaffected"
            )
        elif _codec_index:
            # torchcodec's CUDA build dlopens libnppicc and libnppc, and NPP is NOT in
            # torch's own dependency set, so a --no-deps install from a cuNNN index reports
            # success and then fails to import, disabling audio for a reason nothing here
            # would otherwise name. docker/Dockerfile installs nvidia-npp-cu12 beside the
            # same wheel for exactly this. cu13x wheels want nvidia-npp-cu13.
            _npp_major = _cuda_major_for_npp(_codec_torch_ver, _codec_index)
            if _codec_fellback:
                # The pin is gone, so the tag no longer describes this wheel. Probe EVERY
                # fallback, including tags implying no CUDA: an xpu host takes the cpu leaf
                # and still lands PyPI's CUDA build, needing NPP its tag never mentioned.
                _npp_probed = _installed_torchcodec_cuda_major()
                if _npp_probed is not None and _npp_probed != _npp_major:
                    _note(
                        f"the unpinned torchcodec links CUDA {_npp_probed or 'nothing'} "
                        f"rather than {'CUDA ' + _npp_major if _npp_major else 'nothing'}, "
                        "which its torch tag implies -- matching NPP to the wheel"
                    )
                    _npp_major = _npp_probed
            if _npp_major and not pip_install_try(
                "Installing torchcodec CUDA runtime (NPP)",
                "--no-cache-dir",
                f"nvidia-npp-cu{_npp_major}",
            ):
                _note(
                    f"could not install nvidia-npp-cu{_npp_major} -- torchcodec may fail to "
                    "import on a host without the CUDA toolkit, leaving audio disabled"
                )

    # 14. Final check (silent; third-party conflicts are expected).
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        **_windows_hidden_subprocess_kwargs(),
    )

    # 14b. Repair again before the manifest: the pass above runs before the core packages are installed, so an upgrade leaving a superseded record behind would survive it and write_manifest would record a null version.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

    # 14c. write_manifest reads the installed version, so a core package absent or quarantined now is recorded as a finished install. The skip_base handoff never reaches the core phase, so nothing else would see it.
    if not _repair_damaged_core_payload(
        _core_package_names(package_name),
        local_repo = local_repo,
        require_present = True,
    ):
        return 1

    # 15. Record success. Written last so an earlier kill leaves none: exiting 0 without it reports a finished install every later check calls unfinished.
    if (
        install_manifest.write_manifest(
            req_root = REQ_ROOT,
            steps_total = _TOTAL,
            package_name = package_name,
            no_torch = NO_TORCH,
            # A platform that never resolves a flavor carries the old record forward. An unknown-family pin is the exception: the previous record describes a venv that no longer exists.
            expected_torch_tag = _recordable_torch_flavor_tag(torch_flavor_tag),
            expected_torch_tag_pinned = bool(_recordable_torch_flavor_tag(torch_flavor_tag))
            and _expected_torch_flavor_was_pinned(_recordable_torch_flavor_tag(torch_flavor_tag)),
        )
        is None
    ):
        _safe_print(
            f"error: could not write {install_manifest.MANIFEST_NAME} to "
            f"{install_manifest.venv_root()}",
            file = sys.stderr,
        )
        return 1

    # 16. Apple Silicon: say so when the MLX stack just laid down is not one Train can use, or the app silently comes up chat-only. AFTER the manifest, because the probe is advisory and its imports are the ones that hang, so a kill during that wait would leave every step done and no record of it.
    if IS_MAC_ARM and not NO_TORCH:
        _report_mlx_stack_health()

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--amd-torch-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        _needs_pass = _amd_torch_needs_dependency_pass()
        # Exit 1 covers five states (no-torch venv, resolved non-ROCm backend, non-ROCm pin, absent or masked AMD host, unreadable torch), so a CI failure would otherwise report a bare `assert 1 == 0` with empty streams.
        _safe_print(
            f"{_AMD_FASTPATH_DECISION_MARKER}needs_pass={_needs_pass} no_torch={NO_TORCH} "
            f"is_linux={IS_LINUX} machine={platform.machine()!r} backend={_TORCH_BACKEND!r} "
            f"probe={_TORCH_RUNTIME_PROBE!r}"
        )
        sys.exit(0 if _needs_pass else 1)
    if any(_arg.startswith("-") for _arg in sys.argv[1:]):
        # Never let a malformed probe call fall through into a multi-gigabyte install.
        _safe_print(f"Unknown argument: {' '.join(sys.argv[1:])}")
        sys.exit(2)
    sys.exit(install_python_stack())
