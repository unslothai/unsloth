#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by setup.sh (Linux/WSL) and setup.ps1 (Windows) after the venv is
activated. Expects `pip` and `python` on PATH to point at the venv.
"""

from __future__ import annotations

import ast
import functools
import glob
import importlib
import importlib.util
import json
import locale
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

import install_manifest  # noqa: E402

try:
    import nvidia_probe as _nvidia_probe  # noqa: E402
except Exception:  # an older checkout beside a newer setup script
    _nvidia_probe = None


def _nvidia_library_inventory():
    """The NVML / CUDA driver inventory, or None. A seam, so tests can hide the real host."""
    return _nvidia_probe.probe() if _nvidia_probe is not None else None


from backend.utils.kernel_install import install_prebuilt, uninstall_command
from backend.utils.wheel_utils import (
    flash_attn_package_version,
    flash_attn_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
    xformers_torch_requirement_unmet,
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
# Allowlist of platforms that publish a torchcodec wheel (aarch64 from 0.11.0, the torch 2.11 row);
# elsewhere audio extras are filtered out. The first release per platform is _torchcodec_platform_floor's.
_PLATFORM_HAS_TORCHCODEC_WHEEL = (
    (IS_LINUX and platform.machine() in {"x86_64", "AMD64", "aarch64", "arm64"})
    or (IS_WINDOWS and platform.machine().lower() in {"amd64", "x86_64"})
    or IS_MAC_ARM
)
PLATFORM_LACKS_TORCHCODEC_WHEEL = not _PLATFORM_HAS_TORCHCODEC_WHEEL


def _machine_arch_from_registry() -> str:
    """The machine-scope PROCESSOR_ARCHITECTURE, which an emulated process cannot
    misreport. Every per-process signal follows the process: under x64 emulation on
    ARM64 Windows the process copy says AMD64, PROCESSOR_ARCHITEW6432 is unset (it is a
    WOW64-only variable), and platform.machine() says AMD64 too. Empty when unreadable
    or off Windows."""
    if not IS_WINDOWS:
        return ""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as key:
            return str(winreg.QueryValueEx(key, "PROCESSOR_ARCHITECTURE")[0] or "")
    except Exception:
        return ""


def _is_windows_arm64() -> bool:
    """Windows on ARM, machine arch rather than process arch. The registry value leads
    because the per-process signals all say AMD64 under an emulated x64 Python; they stay
    as fallbacks for a native interpreter. Mirrors Get-HostMachineArch in install.ps1 /
    setup.ps1. Wheel availability is an interpreter question, not a machine one: see
    ``_is_win_arm64_interpreter``."""
    if not IS_WINDOWS:
        return False
    return any(
        (value or "").strip().lower() in {"arm64", "aarch64"}
        for value in (
            _machine_arch_from_registry(),
            os.environ.get("PROCESSOR_ARCHITEW6432"),
            os.environ.get("PROCESSOR_ARCHITECTURE"),
            platform.machine(),
        )
    )


@functools.lru_cache(maxsize = None)
def _is_win_arm64_interpreter() -> bool:
    """Windows on ARM, the arch of THIS INTERPRETER rather than of the machine.

    The distinction decides which wheels exist. ``_is_windows_arm64`` above answers
    for the machine, and is true even under an emulated x64 Python -- which is what
    every install predating native ARM64 support is running, because install.ps1
    deliberately fetched an x64 interpreter there. Such a venv wants the x64 wheels
    and gets them: ``platform_machine == "ARM64"`` in a requirement marker is the
    INTERPRETER's arch, so the marker rows and this predicate have to agree or the
    same machine is served two different answers.

    ``sysconfig.get_platform()`` is what pip and uv tag wheels with, so it is the
    same authority; ``platform.machine()`` is the fallback and reports AMD64 under
    emulation, which is the answer we want there.
    """
    if not IS_WINDOWS:
        return False
    try:
        tag = (sysconfig.get_platform() or "").strip().lower()
        if tag:
            return tag == "win-arm64"
    except Exception:
        pass
    return (platform.machine() or "").strip().lower() in {"arm64", "aarch64"}


def _windows_arm64_has_torchaudio() -> bool:
    """Does the CUDA index this install used publish a win_arm64 torchaudio?

    NVIDIA's GA out-of-tree channel does (2.11.0+cu134); its nightly channel and
    download.pytorch.org do not. install.ps1 answers this in UNSLOTH_WOA_HAS_TORCHAUDIO.
    Unset means "assume not": asking for a wheel that does not exist makes the whole trio
    unresolvable, while skipping one that does costs only audio support.
    """
    return (os.environ.get("UNSLOTH_WOA_HAS_TORCHAUDIO") or "").strip() == "1"


_ROCM_TORCH_INDEX: dict[tuple[int, int], str] = {
    (7, 2): "rocm7.2",  # torch 2.11.0
    (7, 1): "rocm7.1",  # torch 2.11.0
    (7, 0): "rocm7.0",  # torch 2.10.0
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
    future 7.3+), never one at/above the floor.

    A version below every tag (what an unreadable one reads as, on a bundled-runtime host)
    resolves no generic index at all, so the per-arch index is the only route left."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is None or key < _ROCM_ARCH_INDEX_FLOOR


# RDNA 4 below 7.13 reroutes to AMD's per-arch index (TheRock #5284); gfx120X-all is cp310+ only.
_AMD_ARCH_INDEX_FLOOR_GFX: frozenset[str] = frozenset(
    {"gfx1151", "gfx1150", "gfx1152"}
    | ({"gfx1200", "gfx1201"} if sys.version_info >= (3, 10) else set())
)


# gfx906 (MI50 / Radeon VII): rocm6.4+ wheels dropped its Tensile kernels and fail at the first
# BLAS call; rocm6.3 is the last index that runs. Uses the _default (<2.11) specs. Mirrors install.sh.
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
    # A mask hiding every GPU must not force-reinstall onto an older tag; no probe below honours
    # HIP masks. Checked before the override, as in _runtime_gfx_target.
    if _visible_masks_select_no_gpu():
        return False
    # Strip the feature suffix (gfx906:sramecc-:xnack- -> gfx906), as device_type.py does.
    override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":")[0]
    if override:
        return override == "gfx906"
    # Unmasked on purpose: a mask naming the MI50 on a mixed host would make it look like the sole arch.
    return set(_detect_amd_gfx_codes(ignore_visible_masks = True)) == {"gfx906"}


def _torch_below_211(installed_ver: str) -> bool:
    """True when an installed torch version string is readable and below 2.11.

    Unreadable reads as NOT below: this gates a --force-reinstall of a multi-GB stack, and
    a version string no regex can parse is not evidence the build is the broken one.
    """
    _m = re.match(r"\s*(\d+)\.(\d+)", installed_ver or "")
    return bool(_m) and (int(_m.group(1)), int(_m.group(2))) < (2, 11)


# Per-arch leaves needing the torch 2.11 floor (_grouped_mm <2.11 bug); mirrors *FloorMap in
# install.ps1 / setup.ps1. gfx908 / gfx90a stay bare: no Windows wheels, Linux floors via rocm7.2.
_ROCM_GFX_TORCH211_LEAVES: frozenset[str] = frozenset(
    {"gfx120x-all", "gfx1151", "gfx1150", "gfx1152", "gfx103x-all", "gfx110x-all"}
)

# rocmX.Y indexes KNOWN to ship torch 2.11; never floor an unknown newer rocm.
_ROCM_KNOWN_TORCH211_VERSIONS: frozenset[tuple[int, int]] = frozenset({(7, 2)})

# Per-tag repair specs; must land on the same wheels a fresh install.sh run does.
_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
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
    "gfx1030": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1031": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1032": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1033": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1034": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1035": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1036": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1100": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1101": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1102": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
    "gfx1103": _ROCM_TORCH_PKG_SPECS["rocm7.2"],
}
# Windows RDNA uses AMD's multi-arch index, card picked by torch[device-gfxNNNN], pinned to one tag
# <2.12.0. Not rocm7.14.1: its AOTriton runtime/kernel mismatch breaks fused SDPA (TheRock#7992).
_ROCM_WINDOWS_MULTIARCH_INDEX_BASE = (
    os.environ.get("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR")
    or "https://repo.amd.com/rocm/whl-multi-arch"
)
_ROCM_MULTIARCH_TAG = "rocm7.14.0"
_ROCM_MULTIARCH_BROKEN_TAGS = frozenset({"rocm7.14.1"})
_ROCM_MULTIARCH_TORCH_VERSION = "2.11.0"
_ROCM_MULTIARCH_TORCHVISION_VERSION = "0.26.0"
_ROCM_MULTIARCH_TORCHAUDIO_VERSION = "2.11.0"
# gfx1033 excluded: in _ROCM_MISCOMPUTING_GFX, keeps its family route. CDNA stays on the family map.
_WINDOWS_MULTIARCH_GFX: "frozenset[str]" = frozenset(
    {
        "gfx1010",
        "gfx1011",
        "gfx1012",  # RDNA 1
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1034",
        "gfx1035",
        "gfx1036",  # RDNA 2
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",  # RDNA 3
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",  # RDNA 3.5
        "gfx1200",
        "gfx1201",  # RDNA 4
    }
)
_ROCM_WINDOWS_FAMILY_INDEX_DEFAULT = "https://repo.amd.com/rocm/whl"


def _bare_gfx(gfx_arch: "str | None") -> str:
    """'GFX1010:xnack-' -> 'gfx1010': hipinfo prints feature suffixes and users type any case."""
    return (gfx_arch or "").strip().lower().split(":")[0]


def _is_windows_multiarch_gfx(gfx_arch: "str | None") -> bool:
    return _bare_gfx(gfx_arch) in _WINDOWS_MULTIARCH_GFX


def _windows_family_mirror_pinned() -> bool:
    """A host that mirrors the per-family layout and not the multi-arch one keeps the family
    route for arches that have a family; a multi-arch mirror, or no mirror, routes there."""
    if os.environ.get("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR"):
        return False
    if os.environ.get("UNSLOTH_ROCM_WINDOWS_MIRROR"):
        return True
    return _ROCM_WINDOWS_INDEX_BASE.rstrip("/") != _ROCM_WINDOWS_FAMILY_INDEX_DEFAULT


def _windows_routes_multiarch(gfx_arch: "str | None") -> bool:
    """Whether the Windows install of `gfx_arch` goes to the multi-arch index: it has a
    device pack there, and no family-only mirror claims an arch that has a family."""
    if not _is_windows_multiarch_gfx(gfx_arch):
        return False
    if _bare_gfx(gfx_arch) in _GFX_TO_AMD_INDEX_ARCH and _windows_family_mirror_pinned():
        return False
    return True


def _multiarch_device_pack_installed(gfx_arch: "str | None") -> bool:
    """Whether the venv carries AMD's torch and torchvision kernel packs for this card."""
    try:
        from importlib import metadata
        names = {
            (d.metadata["Name"] or "").strip().lower().replace("_", "-")
            for d in metadata.distributions()
        }
    except Exception:
        return False
    gfx = _bare_gfx(gfx_arch)
    return {f"amd-torch-device-{gfx}", f"amd-torchvision-device-{gfx}"} <= names


def _windows_multiarch_torch_pkg_specs(gfx_arch: str) -> tuple[str, str, str]:
    gfx = _bare_gfx(gfx_arch)
    return (
        f"torch[device-{gfx}]=={_ROCM_MULTIARCH_TORCH_VERSION}+{_ROCM_MULTIARCH_TAG}",
        f"torchvision[device-{gfx}]=={_ROCM_MULTIARCH_TORCHVISION_VERSION}+{_ROCM_MULTIARCH_TAG}",
        f"torchaudio=={_ROCM_MULTIARCH_TORCHAUDIO_VERSION}+{_ROCM_MULTIARCH_TAG}",
    )


def _index_is_multiarch(index_url: "str | None") -> bool:
    """The multi-arch index, by identity with the configured base or by its leaf. An explicit
    UNSLOTH_TORCH_INDEX_URL pin never gets here: unknown-family pins install verbatim."""
    if not index_url:
        return False

    def _path(u: str) -> str:
        return re.split(r"[?#]", u, maxsplit = 1)[0].rstrip("/")

    _p = _path(index_url)
    return _p == _path(_ROCM_WINDOWS_MULTIARCH_INDEX_BASE) or _p.endswith("/whl-multi-arch")


def _windows_rocm_torch_pkg_specs_for(
    index_url: "str | None", gfx_arch: "str | None"
) -> tuple[str, str, str]:
    """The trio for the index that was actually chosen: the pinned multi-arch trio on that
    index, the per-arch ABI pin where one exists, bare names otherwise. Derived from the URL
    so the specs can never name a build the index does not serve."""
    if _index_is_multiarch(index_url):
        return _windows_multiarch_torch_pkg_specs(gfx_arch)
    return _WINDOWS_ROCM_TORCH_PKG_SPECS.get(
        _bare_gfx(gfx_arch), ("torch", "torchvision", "torchaudio")
    )


def _windows_rocm_torch_pkg_specs(gfx_arch: "str | None") -> tuple[str, str, str]:
    return _windows_rocm_torch_pkg_specs_for(_windows_rocm_index_url(gfx_arch), gfx_arch)


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


_CUDA_TORCH_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

# CPU repair specs (see _ensure_cpu_torch); the /cpu index also serves newer torch.
_CPU_TORCH_PKG_SPEC: tuple[str, str, str] = _CUDA_TORCH_PKG_SPEC

# Byte-identical to the non-XPU arm of install.ps1's $_fix*Spec scalars, NOT _CUDA_TORCH_PKG_SPEC.
_TORCH_FLAVOR_REPAIR_PKG_SPEC: tuple[str, str, str] = (
    "torch>=2.4,<2.12.0",
    "torchvision>=0.19,<0.27.0",
    "torchaudio>=2.4,<2.12.0",
)

# The install.sh _cu130_torch213_route: a repair keeps a resident 2.9-2.14 release.
_CU130_PRESERVE_TORCH_CEILING_MINOR = 15
_CU130_FIRST_TORCH_MINOR = 9  # download.pytorch.org/whl/cu130 starts at torch 2.9.0


def _is_cu130_torch213_route(index_url: str | None) -> bool:
    return (
        bool(index_url)
        and _torch_index_leaf(index_url) == "cu130"
        and sys.platform.startswith("linux")
        and platform.machine().lower() in ("x86_64", "amd64")
        and sys.version_info[:2] == (3, 13)
    )


def _resident_torch_release() -> str | None:
    """The installed torch's plain X.Y.Z release from its metadata, never importing it."""
    try:
        from importlib.metadata import version as _dist_version
        release = _dist_version("torch").split("+", 1)[0]
    except Exception:
        return None
    return release if re.fullmatch(r"2\.\d+\.\d+", release) else None


def _cuda_repair_torch_specs(
    index_url: str | None, default: tuple[str, str, str]
) -> tuple[str, str, str]:
    """``default``, except that the cu130 torch 2.13 route keeps a resident release it serves.

    Never installs 2.13 itself: whether a release admits it is install.sh's PyPI decision."""
    if not _is_cu130_torch213_route(index_url):
        return default
    release = _resident_torch_release()
    if release is not None:
        minor = int(release.split(".")[1])
        if _CU130_FIRST_TORCH_MINOR <= minor < _CU130_PRESERVE_TORCH_CEILING_MINOR:
            # torchaudio 2.11 is the last release (stable ABI), so newer minors pair with it.
            audio_minor = min(minor, 11)
            return (
                f"torch=={release}",
                f"torchvision==0.{minor + 15}.*",
                f"torchaudio==2.{audio_minor}.*",
            )
    return default


def _resident_torch_trio_pins() -> list[str]:
    """``name==version`` for the installed torch, torchvision and torchaudio (local tag kept)."""
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    pins = []
    for name in ("torch", "torchvision", "torchaudio"):
        try:
            pins.append(f"{name}=={_dist_version(name)}")
        except PackageNotFoundError:
            pass
    return pins


_OVERRIDE_INCLUDE = re.compile(r"^(\s*(?:-r|-c|--requirement|--constraint)(?:\s+|=))(\S+)(.*)$")
_TORCH_TRIO_LINE = re.compile(r"^\s*torch(vision|audio)?([\s<>=!~;@\[]|$)", re.IGNORECASE)
# True while _FreezeNewTorchForCoreUpdate's UV_OVERRIDE is the only thing keeping the trio.
_TORCH_FREEZE_ACTIVE = False


class _FreezeNewTorchForCoreUpdate:
    """Pin the resident torch trio via UV_OVERRIDE while unsloth / unsloth-zoo re-resolve.

    A released unsloth declares a torch ceiling (2026.9.11: <2.13.0), and a with-deps upgrade
    honours it: on a torch 2.13 install `studio update` swapped torch for PyPI's 2.12.1 and lost
    the matching prebuilt kernels. install.sh freezes the trio the same way for every with-deps
    unsloth install (_build_unsloth_torch_overrides). Scoped to Linux with torch >= 2.13, the
    releases past that ceiling, so every other install resolves exactly as before.
    """

    def __enter__(self):
        self._path = None
        self._previous = os.environ.get("UV_OVERRIDE")
        release = _resident_torch_release()
        if not (
            sys.platform.startswith("linux")
            and release is not None
            and int(release.split(".")[1]) >= 13
        ):
            return self
        pins = _resident_torch_trio_pins()
        if not pins:
            return self
        # uv applies every override for a package: fold inherited files in without their trio entries.
        lines = list(pins)
        for inherited in (self._previous or "").split():
            try:
                text = Path(inherited).read_text(encoding = "utf-8")
            except OSError:
                continue
            base = Path(inherited).parent
            for line in text.splitlines():
                if _TORCH_TRIO_LINE.match(line):
                    continue
                # A relative -r / -c resolves against its own file, which is no longer this one.
                include = _OVERRIDE_INCLUDE.match(line)
                if include and "://" not in include[2] and not os.path.isabs(include[2]):
                    line = f"{include[1]}{(base / include[2]).resolve()}{include[3]}"
                lines.append(line)
        fd, name = tempfile.mkstemp(prefix = "unsloth-torch-overrides-", suffix = ".txt")
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._path = Path(name)
        os.environ["UV_OVERRIDE"] = _uv_safe_path(self._path)
        global _TORCH_FREEZE_ACTIVE
        _TORCH_FREEZE_ACTIVE = True
        return self

    def __exit__(self, *exc):
        global _TORCH_FREEZE_ACTIVE
        _TORCH_FREEZE_ACTIVE = False
        if self._path is not None:
            if self._previous is None:
                os.environ.pop("UV_OVERRIDE", None)
            else:
                os.environ["UV_OVERRIDE"] = self._previous
            self._path.unlink(missing_ok = True)
        return False


# torchao's cpp is built for one torch release AND CUDA major; a mismatch loses kernels, not the
# import. Map: 2.9->0.14.0, 2.10 cu<=12->0.16.0, 2.10 cu>=13->0.17.0, 2.11->0.17.0, 2.12+->0.18.0.
# torchao ships per accelerator, so the caller pins the index to the resident torch.
_TORCHAO_DEFAULT_SPEC = "torchao==0.14.0"
_TORCHAO_TORCH_210_SPEC = "torchao==0.16.0"
_TORCHAO_TORCH_210_CUDA13_SPEC = "torchao==0.17.0"
_TORCHAO_TORCH_211_SPEC = "torchao==0.17.0"
_TORCHAO_TORCH_212_PLUS_SPEC = "torchao==0.18.0"
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
        cuda_major = _cuda_major_from_torch_version(str(torch_version))
        if cuda_major is not None and cuda_major >= _TORCHAO_CUDA13_MIN_MAJOR:
            return _TORCHAO_TORCH_210_CUDA13_SPEC
        return _TORCHAO_TORCH_210_SPEC
    return _TORCHAO_DEFAULT_SPEC


# torchcodec <=0.11 is built against one torch minor and declares no torch requirement, so pip
# cannot catch a mismatch; 0.12+ is ABI-stable for torch >=2.11. Mirrors pyproject audio-torch2xx.
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
_TORCHCODEC_MIN_KNOWN_MINOR = min(_TORCHCODEC_TORCH_SPECS)
_TORCHCODEC_MAX_KNOWN_MINOR = max(_TORCHCODEC_TORCH_SPECS)

# First release per platform (PyPI): win_amd64 0.7.0, manylinux aarch64 0.11.0, x86_64 and macOS
# arm64 from the start. A window with no wheel here aborts the install instead of skipping audio.
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


# Pinned MLX publishes only macosx_14_0_arm64 cp310+ wheels; checked up front because pip_install
# exits on failure, which would end an install that otherwise comes up chat-only.
_MLX_MIN_PYTHON = (3, 10)
_MLX_MIN_MACOS_MAJOR = 14


def _mlx_pins_are_installable() -> bool:
    """Wheel for the pinned MLX versions here? An unreadable macOS reads as too old."""
    if sys.version_info < _MLX_MIN_PYTHON:
        return False
    return (_macos_release_major() or 0) >= _MLX_MIN_MACOS_MAJOR


# Supported Python per torchcodec run (first release, min, max) from upstream's table:
# 0.1 3.9-3.12, 0.2-0.7 3.9-3.13, 0.8 3.10-3.13, 0.9+ 3.10-3.14. Separate from the platform floor.
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


# download.pytorch.org carries torchcodec only from 0.3, so torch 2.5/2.6 rows stay unpinned.
_TORCHCODEC_MIN_ON_TORCH_INDEX = (0, 3, 0)

# torchcodec has no xpu build (the xpu leaf republishes cpu wheels from 0.13), so xpu takes cpu;
# unpinned would get PyPI's CUDA build.
_TORCHCODEC_INDEX_TAGS = {"xpu": "cpu"}


def _cuda_major_for_npp(torch_version: "str | None", index_url: str) -> str:
    """`"12"`, `"13"`, or `""` when this codec install needs no NPP runtime.

    The resident torch's LOCAL TAG first, the index URL only as a fallback. Matching
    `/cuNNN$` on the URL failed for a supported UNSLOTH_TORCH_INDEX_URL ending in
    `/simple?token=...`, so a `+cu128` host skipped NPP and the codec then failed to import
    without a system CUDA toolkit. The tag is also the better source: _torchcodec_index_url
    only returns an index once it has seen a `cpu` or `cuNNN` tag, so the tag is always there.
    """
    local = str(torch_version or "").partition("+")[2].strip().lower()
    match = re.fullmatch(r"cu(\d+)", local)
    if match:
        return match.group(1)[:2]
    # A PyPI torch carries no local tag; fall back to the index leaf.
    match = re.search(r"/cu(\d+)/?$", index_url or "")
    return match.group(1)[:2] if match else ""


# nvidia-npp-cu13 is a wheel-less stub and plain nvidia-npp is the 13.x runtime. Do not generalise:
# nvidia-cudnn / nvidia-nccl kept their suffix and the unsuffixed names are fake packages.
_NPP_SUFFIXED_THROUGH_CUDA_MAJOR = 12


def _npp_requirement(cuda_major: str) -> str:
    """The NPP runtime for this CUDA major, spelled the way its publisher spells it.

    Bounded because `nvidia-npp` carries the same 0.0.0a0 junk the stub is made of, and a cu14
    host must not take a 13 runtime. `[0-9]+` not `isdigit()`, as in _hsa_override_gfx_arch:
    isdigit() also takes the superscripts, where `int()` below raises, and the non-ASCII digits,
    which reach a str `\\d` and emit an unparseable `>=١٣`.
    """
    if not re.fullmatch(r"[0-9]+", cuda_major):
        return f"nvidia-npp-cu{cuda_major}"
    if int(cuda_major) <= _NPP_SUFFIXED_THROUGH_CUDA_MAJOR:
        return f"nvidia-npp-cu{cuda_major}"
    return f"nvidia-npp>={cuda_major},<{int(cuda_major) + 1}"


# nvcudart_hybrid64.dll (Windows cu130) carries no major; absent from cpu builds, so "" is safe.
_CUDA_RUNTIME_MARKER_RE = re.compile(
    rb"nvcuda\.dll|torch_cuda|nvcudart|libcudart|cudart64|libcuda\.so"
)


def _pytorch_whl_leaf_url(leaf: str) -> "str | None":
    """_PYTORCH_WHL_BASE plus an accelerator leaf, or None when it cannot be expressed.

    No URL shape pins a query-auth mirror -- pip joins the project name as text, so the token
    swallows either the leaf or the name (see _warn_query_index_unusable). Constructing one
    anyway is worse than declining: --index-url makes _install_env_for_cmd strip pip.conf's
    index keys and ~/.netrc, the only channel that can carry that credential.
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
    # Verbatim and before the tag check: private mirrors rebuild torch without the +cuNNN tag.
    url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if url:
        return _trim_index_path_slashes(url)
    # Before the tag check too: FAMILY=xpu must still substitute, and a None here is final.
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
    """Does this host have a wheel for any release the spec admits?

    Asked before the install rather than discovered by it, because the install step exits on
    failure. Answering no means the audio extra is skipped, which is what a host with no wheel
    got before this step existed.
    """
    host_floor = _torchcodec_platform_floor()
    if host_floor is None:
        return False
    floor, ceiling = _torchcodec_spec_bounds(spec)
    if ceiling is not None and host_floor >= ceiling:
        return False
    if not _torchcodec_python_is_supported(max(floor, host_floor), ceiling):
        return False
    if IS_MAC_ARM and (_macos_release_major() or 0) < 14:
        effective_floor = max(floor, host_floor)
        if effective_floor >= _TORCHCODEC_MACOS_14_ONLY_FROM:
            return False
    return True


def _codec_spec_is_satisfied(spec: str, installed: str) -> bool:
    """Whether the resident torchcodec already sits inside the pin's window.

    The local tag is deliberately ignored here -- the caller has already decided the
    provenance question through _codec_rebuild, and a `+cu130` suffix makes the version
    fall outside a PyPI-shaped specifier that it in fact satisfies.
    """
    if not installed:
        return False
    base = installed.partition("+")[0]
    try:
        from packaging.requirements import Requirement
        return Requirement(spec).specifier.contains(base, prereleases = True)
    except Exception:  # noqa: BLE001 - no packaging, or a version it cannot parse
        return False


def _select_torchcodec_spec(torch_version: "str | None") -> "str | None":
    """Map an installed torch version (e.g. '2.11.0+cu128') to the torchcodec spec built
    against it, or None below the oldest known torch minor. Falls back to
    _TORCHCODEC_DEFAULT_SPEC for a non-2.x major or an unparseable/missing version."""
    if not torch_version:
        return _TORCHCODEC_DEFAULT_SPEC
    release = str(torch_version).split("+", 1)[0]  # drop +cu128/+rocm7.2/+cpu
    parts = release.split(".")
    try:
        minor_str = re.sub(r"[^0-9].*", "", parts[1]) if len(parts) > 1 else ""
        major, minor = int(parts[0]), int(minor_str)
    except (IndexError, ValueError):
        return _TORCHCODEC_DEFAULT_SPEC
    if major != 2:
        return _TORCHCODEC_DEFAULT_SPEC
    if minor < _TORCHCODEC_MIN_KNOWN_MINOR:
        return None
    # Clamp to the ABI-stable floor, never the 0.11 row: 0.11 is locked to torch 2.11 exactly.
    minor = min(minor, _TORCHCODEC_MAX_KNOWN_MINOR)
    return _TORCHCODEC_TORCH_SPECS.get(minor, _TORCHCODEC_DEFAULT_SPEC)


# Memoized torch classification of the target venv; reset by pip_install() / pip_install_try().
_TORCH_RUNTIME_PROBE: "tuple[bool, bool, str | None, str, str] | None" = None
# Kept out of the tuple above because thirteen call sites unpack it.
_TORCH_RUNTIME_XPU: str = ""

# Import chatter or teardown notices can surround the answer, so match the last line with this prefix.
_TORCH_PROBE_MARKER = "UNSLOTH_TORCH_PROBE|"

# Five states share exit 1, so callers (CI) read this line to see which input decided.
_AMD_FASTPATH_DECISION_MARKER = "UNSLOTH_AMD_FASTPATH|"


def _invalidate_torch_runtime_probe() -> None:
    """Forget the memoized torch classification after a pip operation."""
    global _TORCH_RUNTIME_PROBE
    _TORCH_RUNTIME_PROBE = None


def _probe_torch_runtime() -> "tuple[bool, bool, str | None, str, str]":
    """Classify the venv's torch with ONE `import torch` subprocess per install run.

    Returns ``(ran, importable, version, hip, cuda)``:
      ran        -- the subprocess finished; False on OSError/timeout, which is the
                    wedged-GPU-driver case where callers fall back to their on-disk
                    classifiers rather than trusting an absent answer
      importable -- ...and it exited 0, so `import torch` actually works. A False here
                    with `ran` True is the "installed but broken" signal the repair
                    paths use to force a reinstall.
      version    -- torch.__version__ verbatim, or None when no line of ours came back.
                    None is NOT "": an empty __version__ is a broken torch the pins
                    repair, while a missing line means we learned nothing and must leave
                    the venv alone, which is what the per-path probes did on empty stdout.
      hip        -- torch.version.hip  ("" when absent)
      cuda       -- torch.version.cuda ("" when absent)

    The four repair paths run back to back at both repair points, and each used to spawn
    its own `import torch` for these same facts: up to nine interpreter starts per Linux
    update. The real cost was the timeout, not the seconds and hundreds of MB -- each
    probe was bounded at 90s INDEPENDENTLY, so a stalled GPU driver (exactly the host
    these paths exist to rescue) could hang an update for many minutes before the first
    on-disk fallback ran. One probe per repair point bounds that at a single 90s wait.
    """
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
                    # torch.version may be missing; reaching through it would read as a broken torch and reinstall.
                    "_v = getattr(torch, 'version', None); "
                    "h = getattr(_v, 'hip', '') or ''; "
                    "c = getattr(_v, 'cuda', '') or ''; "
                    "x = getattr(_v, 'xpu', '') or ''; "
                    f"print('{_TORCH_PROBE_MARKER}' + '|'.join((v, h, c, x)))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            # text=True decodes strictly and UnicodeDecodeError would escape the except below; replace instead.
            errors = "replace",
            timeout = 90,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        _TORCH_RUNTIME_PROBE = (False, False, None, "", "")
        return _TORCH_RUNTIME_PROBE
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
    """Return torch.__version__ from the target venv (sys.executable), or None if
    torch is absent/unimportable. Cross-platform (unlike probe_torch_wheel_env,
    which is Linux-only); shares the one probe with the torch repair paths.
    """
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
    """Return True when the target venv currently has a Windows ROCm torch build.

    This is a belt-and-suspenders guard for the torchao override step: if the
    earlier ROCm install path failed to set _rocm_windows_torch_installed but the
    venv already contains a ROCm torch wheel, torchao still comes from PyPI.
    """
    if not IS_WINDOWS:
        return False
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran or not _importable:
        return False
    _ver = (_version or "").lower()
    return bool(_hip) or "rocm" in _ver or "rocmsdk" in _ver


# constraints.txt caps anyio <4.14 (#6483), but a pre-cap install stuck at 4.14+ is untouched.
_ANYIO_BAD_FLOOR = (4, 14)


def _installed_version(package: str) -> tuple[int, int] | None:
    try:
        from importlib.metadata import version as _pkg_version
        raw = _pkg_version(package)
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
    installed = _installed_version("anyio")
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


# install.ps1 resolves accelerate without -c and then sets SKIP_STUDIO_BASE=1, so the constraints
# cap never reaches a fresh install.
_ACCELERATE_BAD_FLOOR = (1, 15)


def _repair_bad_accelerate() -> None:
    if not IS_WINDOWS:
        return
    installed = _installed_version("accelerate")
    if installed is None or installed < _ACCELERATE_BAD_FLOOR:
        return
    _note(f"accelerate {installed[0]}.{installed[1]} found -- reinstalling accelerate<1.15...")
    # --no-deps: a with-deps reinstall replaces the ROCm torch. _try because pip_install exits.
    if not pip_install_try(
        "Repairing accelerate version",
        "--no-cache-dir",
        "--no-deps",
        "accelerate<1.15.0",
        constrain = False,
    ):
        _note(
            "could not install accelerate<1.15 -- training on a Windows AMD GPU will fail "
            "at trainer start until it is downgraded (huggingface/accelerate#4249)",
            _red,
        )


# Override with UNSLOTH_ROCM_WINDOWS_MIRROR. Kept verbatim: trimming the whole URL could eat a
# trailing "/" of a base64 query token; _index_url_join trims only the path head.
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

# These arches install ROCm wheels but compute wrong answers, so they stay on CPU torch (as install.sh).
_ROCM_MISCOMPUTING_GFX: "frozenset[str]" = frozenset({"gfx1033"})  # Van Gogh (Steam Deck)

# bnb <= 0.49.2 NaNs at decode shape on every AMD GPU; these prerelease wheels carry the fix and
# PyPI 0.50.0 is the first release with it, so the fallback is safe.
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
    """Return the continuous-release_main bnb wheel URL for the current arch,
    or None when no pre-release wheel is available.
    """
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return _BNB_ROCM_PRERELEASE_URLS.get(arch)


# Records WHAT bnb this pass installed: the second _ensure_rocm_torch() must skip the download yet
# repair a bnb that later steps re-resolved.
_BNB_ROCM_PASS_PROVENANCE: "str | None" = None
# Asset identity seen this pass: the URL alone cannot identify the build.
_BNB_ROCM_PASS_ASSET: "str | None" = None


def _installed_direct_url(dist_name: str) -> "str | None":
    """The URL pip recorded for a distribution installed from one, or None.

    PEP 610: an install from a wheel URL writes direct_url.json beside the metadata,
    and it is the only durable record of WHERE a wheel came from -- the version cannot
    say, because the same version is published to PyPI and to a release page.
    """
    try:
        from importlib.metadata import distribution
        recorded = distribution(dist_name).read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absent metadata is a reason to install
        return None
    if not recorded:
        return None
    try:
        payload = json.loads(recorded)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    url = payload.get("url")
    return str(url) if isinstance(url, str) and url else None


def _bnb_asset_identity(url: str) -> "str | None":
    """What is published at *url* right now, as an opaque string, or None if unreachable.

    The preferred ROCm wheel comes from bitsandbytes' continuous-release_main release, whose
    asset PATH never changes while its bytes do, so the URL pip recorded says where the wheel
    came from, not which one. One HEAD follows the redirect and reads the ETag and size of the
    current bytes; the pass records that beside the provenance and the next one compares.
    """
    import urllib.request

    try:
        request = urllib.request.Request(
            url, method = "HEAD", headers = {"User-Agent": "unsloth-studio-installer"}
        )
        with urllib.request.urlopen(request, timeout = 15) as response:
            etag = (response.headers.get("ETag") or "").strip()
            size = (response.headers.get("Content-Length") or "").strip()
    except Exception:  # noqa: BLE001 - unreachable is "cannot say"; the caller decides
        return None
    if not etag and not size:
        return None
    return f"{etag}|{size}"


def _bnb_wheel_version(url: str) -> "str | None":
    """The version in a bnb wheel filename, e.g. 1.33.7.preview.

    Compared textually against the installed version only as a fallback, and normalised
    the way PEP 440 does for this one wheel: its filename says 1.33.7.preview and its
    metadata says 1.33.7rc0, which is the same release spelled two ways.
    """
    name = url.rsplit("/", 1)[-1]
    parts = name.split("-")
    return parts[1] if len(parts) > 2 and parts[0] and parts[1] else None


def _installed_bnb_provenance() -> "str | None":
    """What the resident bitsandbytes IS, or None when there is nothing to keep.

    An opaque string compared only against another produced the same way: the URL pip
    recorded, or the version when nothing did. It says nothing about whether that is the RIGHT
    build, which is the caller's question; keeping the two apart lets a recorded provenance be
    checked against both what is on disk and what this run would install.
    """
    installed = _installed_distribution_version("bitsandbytes")
    if not installed:
        return None
    # Metadata survives a quarantined payload; the skipped reinstall is what would restore it.
    try:
        if install_manifest.damaged_payload_files("bitsandbytes", limit = 1):
            return None
    except Exception:  # noqa: BLE001 - an environment nobody can scan is not evidence
        return None
    recorded_url = _installed_direct_url("bitsandbytes")
    return f"url:{recorded_url}" if recorded_url else f"version:{installed}"


def _bnb_provenance_matches_request(provenance: str, url: "str | None") -> bool:
    """Whether *provenance* describes a build one of the two install paths would land."""
    kind, _, value = provenance.partition(":")
    if kind == "url":
        return url is not None and value == url
    # No direct_url.json: an older installer (filename version) or the PyPI fallback (the pin).
    if url is not None:
        wheel_version = _bnb_wheel_version(url)
        if wheel_version is not None and _versions_are_same_release(value, wheel_version):
            return True
    return _spec_is_satisfied(_BNB_ROCM_PYPI_FALLBACK, value)


def _refuse_bnb(reason: str) -> bool:
    """Say why the resident ROCm bitsandbytes is fetched again, then answer False."""
    if VERBOSE:
        _note(f"bitsandbytes (ROCm): {reason} -- reinstalling")
    return False


def _bnb_rocm_install_is_current(url: "str | None") -> bool:
    """Whether this pass may leave bitsandbytes exactly as it found it.

    --force-reinstall from a release URL downloads the wheel before it can notice it
    already has it, and this decision is reached twice per dependency pass, so an AMD
    host paid the whole wheel twice on every no-op `studio update`.
    """
    global _BNB_ROCM_PASS_ASSET
    provenance = _installed_bnb_provenance()
    if provenance is None:
        return _refuse_bnb("no usable bitsandbytes on disk")
    if _BNB_ROCM_PASS_PROVENANCE is not None:
        # Second call this pass: skip only while disk still holds what the first left.
        return provenance == _BNB_ROCM_PASS_PROVENANCE
    if not _may_skip_on_evidence():
        return _refuse_bnb("no dependency pass evidence")
    # A generic wheel another step pulled in reads like the PyPI fallback, so require a recorded landing.
    recorded = (_PASS_EVIDENCE or {}).get("bnb_rocm")
    if not isinstance(recorded, str) or recorded != provenance:
        return _refuse_bnb(f"last run recorded {recorded!r}, on disk {provenance!r}")
    if not _bnb_provenance_matches_request(provenance, url):
        return _refuse_bnb(f"{provenance!r} is not the build this run installs ({url})")
    if provenance.startswith("url:") and url is not None:
        # Same URL is not the same wheel: continuous-release_main republishes under a fixed path.
        recorded_asset = (_PASS_EVIDENCE or {}).get("bnb_rocm_asset")
        if not isinstance(recorded_asset, str) or not recorded_asset:
            return _refuse_bnb("the last run did not record which bytes the release URL served")
        published = _bnb_asset_identity(url)
        if published is None:
            # Offline: the reinstall could not fetch either, and the build on disk was deliberate.
            if VERBOSE:
                _note("bitsandbytes (ROCm): release page unreachable -- keeping the recorded build")
            _BNB_ROCM_PASS_ASSET = recorded_asset
            return True
        if published != recorded_asset:
            return _refuse_bnb(
                f"the wheel at {url} was republished ({recorded_asset} -> {published})"
            )
        _BNB_ROCM_PASS_ASSET = published
    return True


def _record_bnb_rocm_provenance() -> None:
    """What this pass leaves installed, for its own second call and for the manifest."""
    global _BNB_ROCM_PASS_PROVENANCE, _BNB_ROCM_PASS_ASSET
    _BNB_ROCM_PASS_PROVENANCE = _installed_bnb_provenance()
    if _BNB_ROCM_PASS_ASSET is not None:
        return
    url = _installed_direct_url("bitsandbytes")
    _BNB_ROCM_PASS_ASSET = _bnb_asset_identity(url) if url else None


def _versions_are_same_release(installed: str, wheel: str) -> bool:
    """Whether two spellings of a version name the same release.

    bnb's continuous-release wheel is 1.33.7.preview in its filename and 1.33.7rc0 in
    its metadata, so a string compare says no and a PEP 440 parse says yes.
    """
    if installed == wheel:
        return True
    try:
        from packaging.version import Version
        return Version(installed) == Version(wheel)
    except Exception:  # noqa: BLE001 - unparseable is not a match
        return False


def _spec_is_satisfied(spec: str, installed: str) -> bool:
    """Whether *installed* satisfies a requirement string such as `name>=0.50.0`."""
    try:
        from packaging.requirements import Requirement
        return Requirement(spec).specifier.contains(installed, prereleases = True)
    except Exception:  # noqa: BLE001 - no packaging, or a version it cannot parse
        return False


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
        # A root prefix (C:\ or /) would commonpath-match everything; a venv is never at root.
        if os.path.dirname(_root) == _root:
            return False
        return os.path.normcase(os.path.commonpath([os.path.realpath(path), _root])) == _root
    except (ValueError, OSError):
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

    # Only "installed" counts (dpkg-query lists removed-not-purged). rocm-core beats libhsa-runtime64-1,
    # which comes from the distro archive and can be older.
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
            # dpkg-query exits nonzero when either package is absent but still prints the other's line.
            _dpkg_readings: "dict[str, list[tuple[int, int]]]" = {"rocm-core": [], "hsa": []}
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 5 or fields[3] != "installed":
                    continue
                package, raw = fields[0], fields[4]
                if package not in ("rocm-core", "libhsa-runtime64-1"):
                    continue
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


# APUs that often sit beside a discrete Radeon; HIP may enumerate the APU first, so an index-0 pick
# would target the iGPU. Strix arches (gfx1150/1151/1152) are real targets and excluded.
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


def _pick_visible_index(
    num_tokens: int,
    warn: bool = True,
    masks: "tuple[str, ...] | None" = None,
) -> int:
    """Resolve HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
    to an index into a list of length num_tokens. Returns 0 (first GPU) for
    unset, empty, '-1', UUID-style, or out-of-range values.

    First-set-wins, matching `_visible_devices_pinned()` and
    `_pick_rocm_gfx_target` in install_llama_prebuilt.py. Falling through to the
    next var on "" / "-1" would contradict the runtime: an empty HIP mask
    shadows CUDA_VISIBLE_DEVICES rather than deferring to it, and selects no GPU
    at all.

    ``masks`` narrows which layers are consulted. Callers that have already applied
    the ROCr layer themselves pass _HIP_LAYER_MASKS so it is not applied twice."""
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
            # Warn instead of silently picking GPU 0. warn=False callers index arches, not devices.
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
        # mask_resolved probes (hipinfo is a HIP app) already list only visible devices renumbered from 0;
        # indexing again would apply the mask twice. amd-smi and WMI list every GPU, so they keep the index.
        _pick = tokens[0 if mask_resolved else _pick_visible_index(len(tokens), warn = warn)]
        _distinct = list(dict.fromkeys(tokens))
        if len(_distinct) < 2 or _visible_devices_pinned():
            # A pin is honoured verbatim, but warn when it picks a card with no Windows wheels while another has them.
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
        # Unpinned mixed-arch host: skip a leading shadowing iGPU so the discrete card decides the family.
        if _pick in _SHADOWING_INTEGRATED_GFX:
            _others = [t for t in tokens if t not in _SHADOWING_INTEGRATED_GFX]
            # Prefer a wheel-backed replacement; deposing for a wheel-less card would drop the host to CPU.
            _withWheels = [t for t in _others if _windows_rocm_index_url(t) is not None]
            _candidates = _withWheels or (
                [] if _windows_rocm_index_url(_pick) is not None else _others
            )
            if _candidates:
                _other = _candidates[0]
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
        # AMD torch wheels drop hipInfo.exe into venv Scripts, so driver-only hosts re-detect.
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
            # Accept partial output when hipinfo crashes (0xC0000005 on some RDNA 4).
            text = result.stdout.decode(errors = "replace")
            # Every gcnArchName line, split on ':' like setup.ps1 so feature suffixes do not defeat lookups.
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

    # Last resort: GPU name via WMI -> arch table (mirrors setup.ps1's $nameArchTable) for driver-only
    # hosts. Only AMD adapters with ConfigManagerErrorCode 0, as setup.ps1 filters $wmiGpus.
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
            # Lines are "<name>|<ConfigManagerErrorCode>"; a bare name counts as healthy.
            _all_names, _healthy = [], []
            for _line in result.stdout.decode(errors = "replace").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                _nm, _sep, _code = _line.rpartition("|")
                if not _sep:
                    _nm, _code = _line, "0"
                _nm = _nm.strip()
                # Re-apply the vendor filter: a non-AMD adapter would shift the mask index.
                if not _nm or not re.search(r"AMD|Radeon", _nm, re.IGNORECASE):
                    continue
                _all_names.append(_nm)
                if _code.strip() in ("", "0"):
                    _healthy.append(_nm)
            # Drop non-working adapters, but fall back to all when none are healthy: code 45 is routine on
            # muxless laptops with a parked dGPU.
            _names = _healthy or _all_names
            _tokens = [_a for _a in map(_gfx_arch_from_gpu_name, _names) if _a]
            # Resolve the mask over the adapter list (setup.ps1's $nameIdx), not the shortened token list.
            _sel = _pick_visible_index(len(_names)) if _names else 0
            _named = _gfx_arch_from_gpu_name(_names[_sel]) if _names else None
            # Borrow another adapter's arch only when unpinned.
            if not _named and not _visible_devices_pinned() and _tokens:
                _named = _tokens[0]
            # Repick only when every adapter mapped: an unknown name may be the discrete card.
            _pick = _dedup_pick(_tokens, warn = False) if len(_tokens) == len(_names) else _named
            if _pick:
                _safe_print(f"   gfx arch inferred from GPU name (WMI): {_pick}")
                return _pick
            if _names and not _pick:
                # No arch means CPU-only torch; name the adapter. Polaris gets no override hint since no fix exists.
                _unsupported = _unsupported_gfx_arch_from_gpu_name(_names[_sel])
                if _unsupported:
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
                    # llama.cpp still runs these cards via Vulkan. PowerShell syntax since this branch is Windows-only;
                    # skipped on ARM64 because setup.ps1 throws on that variable there.
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
    # RDNA 4 (Navi 48). R9700 is listed separately: its name has neither 9070 nor 9080.
    (r"9070|9080|R9700", "gfx1201"),
    (r"9060", "gfx1200"),  # RDNA 4 (Navi 44: Radeon RX 9060 XT / 9060)
    (r"8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max", "gfx1151"),
    (r"890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]", "gfx1150"),
    (r"860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33", "gfx1152"),
    (r"RX 7900|PRO W7900|PRO W7800", "gfx1100"),
    (r"RX 7800|RX 7700(?!S)|PRO W7700|PRO V710", "gfx1101"),  # Navi 32
    (r"RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500", "gfx1102"),  # Navi 33
    (r"780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme", "gfx1103"),
    # RDNA 2: every row resolves to gfx103X-all (6850M XT is Navi 22, filed here like 6750 / 6700).
    (r"RX 6950|RX 6900|RX 6850|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900", "gfx1030"),  # Navi 21
    (r"RX 6650|RX 6600|PRO W6600|PRO W6650", "gfx1032"),  # Navi 23
    (
        r"RX 6550|RX 6500|RX 6450|RX 6400|RX 6300|PRO W6400|PRO W6500|PRO W6300",
        "gfx1034",
    ),  # Navi 24
    # RDNA 1 (Navi 10 / 14), routed via _WINDOWS_MULTIARCH_GFX.
    (r"Radeon Pro V520|Radeon Pro 5600M", "gfx1011"),
    (r"RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700", "gfx1010"),
    (r"RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300", "gfx1012"),
]


def _gfx_arch_from_gpu_name(name: str) -> "str | None":
    """Map a GPU marketing name to its gfx arch via _WIN_GPU_NAME_ARCH_TABLE."""
    if not name:
        return None
    for _pat, _arch in _WIN_GPU_NAME_ARCH_TABLE:
        if re.search(_pat, name, re.IGNORECASE):
            return _arch
    return None


# Polaris arches no ROCm wheel covers. Kept separate from _WIN_GPU_NAME_ARCH_TABLE so nothing here
# routes to a wheel index; (?!0) stops "RX 570" matching "RX 5700".
_UNSUPPORTED_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    (
        r"RX 4[78]0(?!0)|RX 5[789]0(?!0)|Radeon Pro WX 7100|Radeon Pro WX 5100",
        "gfx803",
    ),  # Polaris 10/20/30
]


def _unsupported_gfx_arch_from_gpu_name(name: str) -> "str | None":
    """Name the gfx arch of a GPU whose generation Unsloth has no ROCm wheels for.

    Messaging only. Callers must not feed the result into index selection.
    """
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
        # WSL lists no PCI display device; /dev/dxg + librocdxg is the GPU evidence. Without that runtime,
        # per-arch wheels would land in an env that cannot expose the GPU.
        if not _wsl_rocm_runtime_present():
            return None
    elif not _linux_amd_display_device_present():
        # A VM/container on a Strix host shows the host CPU model but gets no GPU, so require a display device.
        return None
    cpu_gfx = _linux_amd_gfx_from_cpuinfo()
    if cpu_gfx:
        return cpu_gfx
    return _linux_amd_gfx_from_lspci()


# Mirrors already named, so a repeated repair does not repeat the notice.
_WARNED_QUERY_INDEX_BASES: "set[str]" = set()


def _warn_query_index_unusable(base: str) -> None:
    """Say so when a mirror carries its credential in the query or fragment.

    pip joins each project URL as text (``posixpath.join(index_url, name)``), so
    ".../gfx1151/?token=x" asks the index ROOT with the name buried in the token, and a
    fragment never reaches the server. No URL shape makes pip send both a path leaf and a
    query, so the join below cannot repair this -- but a mirror that resolves nothing
    should say why rather than 404 per package.
    """
    if ("?" not in base and "#" not in base) or base in _WARNED_QUERY_INDEX_BASES:
        return
    _WARNED_QUERY_INDEX_BASES.add(base)
    _safe_print(
        "   The wheel index mirror carries its credential in the URL query or fragment. pip\n"
        "   appends the package name to the index URL as text, so the name lands inside\n"
        "   the credential and no package resolves. Put the credential in the URL itself\n"
        "   (https://user:token@host/path/), or in ~/.netrc, instead.\n"
    )


def _index_url_join(base: str, leaf: str) -> str:
    """Append a path segment to a wheel index URL, keeping any query / fragment.

    rstrip + concat would bury the leaf in a token instead: "https://m/whl?token=x" +
    "gfx110X-all" asks for /whl with the arch inside the token. Splits on the FIRST of "?"
    or "#", so a URL carrying both keeps them in order. The lesser of two corruptions, not
    a working index: see _warn_query_index_unusable.
    """
    _warn_query_index_unusable(base)
    _cuts = [base.index(_c) for _c in "?#" if _c in base]
    _head, _sep, _tail = (
        (base[: min(_cuts)], base[min(_cuts)], base[min(_cuts) + 1 :]) if _cuts else (base, "", "")
    )
    return f"{_head.rstrip('/')}/{leaf}/{_sep}{_tail}"


def _amd_arch_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD per-arch pip index URL for a gfx arch (Linux + Windows).

    Windows honors UNSLOTH_ROCM_WINDOWS_MIRROR (via _windows_rocm_index_url);
    Linux honors UNSLOTH_AMD_ROCM_MIRROR -- the same var install.sh uses -- so a
    mirrored/air-gapped Linux repair reaches the index install.sh chose rather
    than falling back to repo.amd.com. Both default to repo.amd.com when unset.
    """
    # Users copy rocminfo's gfx1100:sramecc-:xnack- into UNSLOTH_ROCM_GFX_ARCH; tables key on the bare arch.
    gfx_arch = (gfx_arch or "").strip().split(":")[0] or None
    if IS_WINDOWS:
        return _windows_rocm_index_url(gfx_arch)
    # gfx1033 miscomputes under ROCm (studio/ROCM_RDNA2_APU.md); do not reinstall what install.sh avoids.
    if (gfx_arch or "").lower() in _ROCM_MISCOMPUTING_GFX:
        return None
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    base = os.environ.get("UNSLOTH_AMD_ROCM_MIRROR") or "https://repo.amd.com/rocm/whl"
    return _index_url_join(base, arch_family)


def _physical_amd_gfx_archs() -> "list[str]":
    """The AMD arches on this Linux host, read from sources an override cannot move.

    Strongest first: ROCm userland probes with HSA_OVERRIDE_GFX_VERSION and the visible-device
    masks stripped, then KFD topology sysfs, then product-name inference, then the declared
    UNSLOTH_ROCM_GFX_ARCH. Declared is LAST because it is a routing hint for a host whose
    probes cannot answer, not a statement about silicon, and taking it first let a stale
    gfx1030 on a real Van Gogh hide the arch. KFD precedes the inference for the same reason:
    _infer_linux_amd_gfx_arch() returns the declared value first, so behind it the kernel
    never answers.
    """
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
    """True when EVERY AMD arch this host physically has computes incorrectly under ROCm.

    Every, not any: gfx1033 is one of the integrated parts _SHADOWING_INTEGRATED_GFX lists, so
    it can lead the enumeration on a box whose real accelerator is a discrete Radeon (#7776),
    and declining ROCm on presence alone would strand that card. install.sh's gate is a
    presence test only because it cannot resolve which device the runtime picks; here the arch
    list IS the host.

    Shared with _rocm_miscomputing_host(), which adds only "and ROCm torch is already
    installed": same question about the hardware, withholding wheels rather than replacing them.
    """
    if IS_WINDOWS or IS_MACOS:
        return False
    _archs = _physical_amd_gfx_archs()
    return bool(_archs) and all(_arch in _ROCM_MISCOMPUTING_GFX for _arch in _archs)


def _rocm_miscomputing_host() -> bool:
    """True when every AMD GPU on this Linux host is an arch measured to compute
    incorrectly under ROCm, ROCm torch is already installed, and no explicit index pin
    overrides that finding.

    Returning None from _amd_arch_index_url() only stops such a host from being GIVEN ROCm
    wheels. A venv that already HOLDS them was never demoted: install.sh resolves
    UNSLOTH_TORCH_BACKEND=cpu, so _ensure_rocm_torch() returns at its first line,
    _ensure_cpu_torch() fires only for an EXPLICIT pin, and the base update does not reinstall
    an already-satisfied torch. Upgrading therefore left exactly the build the gate exists to
    remove. Treat the arch itself as CPU authority and let _ensure_cpu_torch() demote.

    EVERY arch, not any: a healthy dGPU beside a miscomputing APU is still served by ROCm. The
    disk label is read first so the ROCm probes cost nothing on the vast majority of hosts. An
    explicit UNSLOTH_TORCH_INDEX_URL / _FAMILY stays the escape hatch and wins.

    KFD topology sysfs comes after the runtime probes but BEFORE the product-name inference and
    the declared arch: a Van Gogh host can reach here with neither answering, since
    _detect_amd_gfx_codes() needs rocminfo or amd-smi (absent once ROCm is uninstalled, or with
    the user outside the render group) and _infer_linux_amd_gfx_arch() maps no Van Gogh product
    name. _archs then came back empty and the host kept the NaN-producing wheels. amdkfd is in
    the kernel driver, and is the source _hsa_probe_correction() already trusts over the runtime.
    """
    if IS_WINDOWS or IS_MACOS:
        return False
    # An unusable pin installed nothing: only a ROCm one may still overrule the demotion.
    if _explicit_torch_index_url() is not None or _is_pip_rocm_family_leaf(
        _explicit_torch_index_family() or ""
    ):
        return False
    if "+rocm" not in _installed_torch_label_on_disk():
        return False
    # Declared arch last: this asks what silicon is present, so a stale UNSLOTH_ROCM_GFX_ARCH must not win.
    return _miscomputing_arch_host()


def _windows_rocm_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD pip index URL for the given GPU arch, or None if unsupported.

    Every RDNA arch resolves to AMD's multi-arch index (one URL for every device on it; the
    device is selected by the `torch[device-gfxNNNN]` extra, not by the path), unless a
    family-only mirror is pinned and the arch has a family; CDNA and gfx1033 go to their
    repo.amd.com family."""
    if _windows_routes_multiarch(gfx_arch):
        # Slash on the path, not after a ?token= query (as _index_url_join splits).
        _base = _ROCM_WINDOWS_MULTIARCH_INDEX_BASE
        _cut = min([_base.index(_c) for _c in "?#" if _c in _base] or [len(_base)])
        return f"{_base[:_cut].rstrip('/')}/{_base[_cut:]}"
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(_bare_gfx(gfx_arch))
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
    # No `rocm` meta-package: use the runtimes on disk only when exactly one is present.
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
    """Whether the INSTALLED torch is an AMD per-arch build, i.e. torch itself pulls in the
    `rocm` SDK meta-package that names the family.

    _installed_rocm_wheel_family() reads that family off `rocm`, but pip leaves an orphan
    `rocm` behind when a generic ROCm torch is force-reinstalled over a per-arch one
    (measured 2026-08-27: 2.11.0+rocm7.13.0 -> 2.10.0+rocm7.1 dropped `rocm[libraries]`
    while `rocm` kept naming rocm-sdk-libraries-gfx110X-all). torch.version.hip is set on
    both, so a caller that SKIPS work on a family match must ask this too or the orphan
    hides the repair. One that reinstalls on a mismatch is fail-safe.
    """
    try:
        from importlib import metadata
        for _req in metadata.requires("torch") or []:
            # Exactly the `rocm` distribution (not rocm-sdk-core / triton-rocm), case-insensitive like pip.
            if re.match(r"\s*rocm\s*(?:\[|[=<>!~;,]|$)", _req, re.IGNORECASE):
                return True
    except Exception:
        pass
    return False


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
    # Highest numeric suffix wins ("713" over "72"); glob order is not guaranteed.
    return max(all_vers, key = lambda v: int(v)) if all_vers else None


# Set before the base unsloth install; lets _ensure_rocm_torch drop a freshly pulled generic bnb on
# gfx906 while keeping a pre-existing source build.
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


def _rocm_torch_explicitly_requested() -> bool:
    """Whether UNSLOTH_FORCE_ROCM_TORCH asked for ROCm torch whatever else the host has.

    Auto-detection stops probing AMD once CUDA is usable, leaving a mixed NVIDIA+AMD host no
    route to its AMD card but an index pin, which names a wheel family rather than a
    preference (#10450). Mirrors UNSLOTH_FORCE_VULKAN; a pin still outranks it. One torch
    install serves one vendor, so this SWAPS the stack: NVIDIA is unavailable while it is set.
    """
    return (os.environ.get("UNSLOTH_FORCE_ROCM_TORCH") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _has_rocm_gpu() -> bool:
    """Return True only if an actual AMD GPU is visible (not just ROCm tools installed).

    Returns False when an NVIDIA GPU is present -- NVIDIA takes priority on mixed
    hosts and prevents every detection path below (rocminfo, amd-smi, KFD sysfs) from
    producing a false positive even if ROCm tools are installed alongside the NVIDIA
    driver -- unless this run explicitly asked for ROCm, which is the one case where
    the AMD card is the point.
    """
    if _has_usable_nvidia_gpu() and not _rocm_torch_explicitly_requested():
        return False
    for cmd, check_fn in (
        # rocminfo: real gfx GPU ids only (gfx000 = CPU agent, "gfx11-generic" = ISA line).
        (
            ["rocminfo"],
            lambda out: bool(re.search(r"gfx[1-9][0-9a-z]{2,3}", out.lower())),
        ),
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
    # sysfs KFD fallback. Reject non-AMD vendors: the NVIDIA open kernel module also registers KFD nodes.
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

    On Windows nvidia-smi.exe is often off PATH, so also probe the fixed driver
    locations install.ps1 / setup.ps1 use, else NVIDIA+AMD hosts get ROCm wheels.

    CUDA_VISIBLE_DEVICES set to "" or "-1" hides every NVIDIA device (mixed
    AMD+NVIDIA hosts steering work to the AMD card); neither probe honours
    that env var, so check it first and report the GPU as not usable. Unset
    means all devices visible.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return False

    # A stale nvidia-smi on PATH lists nothing, so try every candidate (as install.ps1 / setup.ps1 do).
    _path_exe = shutil.which("nvidia-smi")
    for _candidate in _nvidia_smi_candidates():
        if _candidate != _path_exe and not os.path.isfile(_candidate):
            continue
        if _nvidia_smi_lists_a_gpu(_candidate):
            return True
    if sys.platform != "win32":
        try:
            gpu_dir = "/proc/driver/nvidia/gpus"
            if os.path.isdir(gpu_dir) and os.listdir(gpu_dir):
                return True
        except OSError:
            pass
    # Last: the driver libraries, which ship without nvidia-smi.
    inventory = _nvidia_library_inventory()
    if inventory is None or not inventory.devices:
        return False
    if inventory.source != "nvml" or cvd is None:
        return True  # the CUDA driver rows already honour the mask
    # NVML rows are physical: an index or GPU- UUID mask must name one; a MIG UUID is left usable.
    tokens = [t.strip().lower() for t in cvd.split(",") if t.strip()]
    if not all(t.isdigit() or t.startswith("gpu-") for t in tokens):
        return True
    return any(row["index"] in tokens or row["uuid"].lower() in tokens for row in inventory.devices)


# Which probe answered last: only rocminfo honours a visible-device mask. None when stubbed.
_LAST_AMD_GFX_PROBE: "str | None" = None

# Why the last _runtime_gfx_target call resolved no target; read only by replace-the-stack callers.
_LAST_HIP_MASK_RESOLVED = True
# _rocr_visible_subset keeps the whole list when the first ordinal names nothing.
_LAST_ROCR_MASK_RESOLVED = True
# Masks resolved but the arch is still ambiguous, so "no target" is not a detection miss.
_LAST_GFX_TARGET_AMBIGUOUS = False


def _detect_amd_gfx_codes(
    dedup: bool = True,
    ignore_hsa_override: bool = False,
    ignore_visible_masks: bool = False,
) -> list[str]:
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

    ignore_hsa_override=True strips HSA_OVERRIDE_GFX_VERSION from the probe's
    environment. ROCr applies it in userland, so rocminfo reports the SPOOFED ISA
    while it is set (unslothai#7331); re-probing without it is the one way to see
    the physical arch. amd-smi reads the driver, so stripping it there is a no-op.

    ignore_visible_masks=True additionally strips ROCR_VISIBLE_DEVICES and
    HIP_VISIBLE_DEVICES so the re-probe sees the WHOLE machine: a mask would
    otherwise hide the very second GPU whose presence is the reason to decline the
    correction. install.sh's re-probe unsets all three together.
    """
    global _LAST_AMD_GFX_PROBE
    _LAST_AMD_GFX_PROBE = None

    def _extract(text: str) -> list[str]:
        if dedup:
            codes = [f"gfx{c}" for c in re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())]
            return list(dict.fromkeys(codes))
        # One entry per agent / "GPU: N" section, else two cards of one arch collapse and ordinals shift.
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


# RDNA 3.5 APUs commonly spoofed with HSA_OVERRIDE_GFX_VERSION=11.0.0; the correction only fires here.
_HSA_SPOOFABLE_PHYSICAL_GFX: frozenset[str] = frozenset({"gfx1151", "gfx1150", "gfx1152"})


# From torch.cuda.get_arch_list() on the generic wheels the pins resolve (2.10/2.11, rocm7.0-7.2).
# torch 2.13+rocm7.1 adds gfx1103, so raising the <2.11 _default cap means rechecking this.
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


# gfx906's rocm6.3 route requires it to be the sole arch, so it must never depose an iGPU.
_MIXED_HOST_UNROUTABLE: "frozenset[str]" = frozenset({"gfx906"})


def _gfx_route_on_host(gfx: "str | None", host_codes: "list[str] | None" = None) -> bool:
    """Whether an index can serve ``gfx`` ON THIS HOST, not on some host.

    _gfx_has_a_wheel_route asks it in the abstract, and gfx906 answers yes on the generic
    union while its only usable route -- the rocm6.3 legacy tag -- opens solely when gfx906
    is the one arch on the machine. ``host_codes`` is that machine BEFORE the ROCr layer, to
    match _runtime_target_is_gfx906's unfiltered probe: judging by the survivors would call
    a masked gfx906 alone, demote, then be refused the tag by a function seeing both cards.
    """
    return _gfx_has_a_wheel_route(gfx) and not (
        gfx in _MIXED_HOST_UNROUTABLE and len(set(host_codes or ())) > 1
    )


def _amd_hardware_is_corroborated() -> bool:
    """AMD silicon this host can point at, with no declared arch anywhere in the chain.

    _infer_linux_amd_gfx_arch() returns UNSLOTH_ROCM_GFX_ARCH before it looks at hardware, so
    it cannot answer "is there a card". Under UNSLOTH_FORCE_ROCM_TORCH that matters: the
    request skips the NVIDIA precedence return, so a stale arch would otherwise force AMD
    wheels over a working CUDA stack on a host with no AMD GPU. Sources are the ones
    _infer_linux_amd_gfx_arch uses on its own non-declared path.
    """
    if IS_WINDOWS or IS_MACOS:
        return False
    if _has_rocm_gpu() or _kfd_gfx_targets():
        return True
    if _is_wsl():
        # /dev/dxg and leftover librocdxg name no vendor, so beside a usable NVIDIA card require an agent.
        return _wsl_rocm_runtime_present() and not _has_usable_nvidia_gpu()
    return _linux_amd_display_device_present()


def _forced_rocm_route_is_viable() -> bool:
    """Whether the request has something to swap TO on this host.

    The bar is _gfx_has_a_wheel_route's: an arch no index can serve must never depose a card
    that can. Presence is not it (gfx1010 is present with no route). Runtime visibility is
    not it either: a runtime-less but inferable AMD card is deliberately served per-arch
    wheels on a pure-AMD host, so requiring rocminfo would answer differently for the same
    silicon by whether an NVIDIA card sits beside it.
    """
    # ROCm wheels are x86_64 only; elsewhere standing CUDA repair down would leave torch broken.
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    if not _amd_hardware_is_corroborated():
        return False
    if _miscomputing_arch_host():
        return False
    # The card the runtime will hand torch, not any sibling. _runtime_gfx_target returns the
    # whole machine beside the target, which _gfx_route_on_host's gfx906 rule needs.
    _inferred = (_infer_linux_amd_gfx_arch() or "").strip().lower().split(":")[0] or None
    _target, _, _, _host_codes = _runtime_gfx_target(_inferred)
    # A declared arch replaces the inventory and can hide a Van Gogh sibling; probe-resolved is exempt.
    if (
        (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip()
        and _target is not None
        and any(_gfx in _ROCM_MISCOMPUTING_GFX for _gfx in _physical_amd_gfx_archs())
    ):
        return False
    # A mask HIP cannot resolve exposes no device; checked before the target test on purpose.
    if not _LAST_HIP_MASK_RESOLVED or not _LAST_ROCR_MASK_RESOLVED or _LAST_GFX_TARGET_AMBIGUOUS:
        return False
    if _target is not None:
        # Keyed on the SELECTED target: _miscomputing_arch_host needs every arch bad, a mask can pick one.
        if _target in _ROCM_MISCOMPUTING_GFX:
            return False
        # Below ROCm 6.0 no generic tag resolves; ask _ensure_rocm_torch's version-independent arms.
        _raw_ver = _detect_rocm_version()
        _ver = _raw_ver or (0, 0)
        _declared = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip()
        if _raw_ver is None:
            # An unreadable version is not (0, 0): the None branch answers differently for gfx1102/1200/1201.
            _torch_ran, _torch_imp, _torch_ver, _, _ = _probe_torch_runtime()
            _installed_ver = (_torch_ver or "").lower() if (_torch_ran and _torch_imp) else ""
            # Already on ROCm: "nothing to install" after a swap must not let _ensure_cuda_torch overwrite it.
            if "rocm" in _installed_ver or "hip" in _installed_ver:
                return _gfx_route_on_host(_target, _host_codes or [_target])
            if (
                _explicit_rocm_torch_index_url() is None
                and not _inferred
                and not _generic_rocm_wheel_lacks_kernels(_target)
                and not _rocm_torch_family_needs_repair(_target, None, _host_codes or [_target])
                and not _rocm_compat_reroute_pending(_target, (0, 0), _installed_ver)
            ):
                return False
            return _gfx_route_on_host(_target, _host_codes or [_target])
        if (
            _explicit_rocm_torch_index_url() is None
            and _generic_pytorch_rocm_tag(_ver) is None
            and not _generic_rocm_wheel_lacks_kernels(_target, _ver)
            and not (
                _inferred
                and (_declared or not _has_rocm_gpu())
                and _amd_arch_index_url(_inferred) is not None
            )
        ):
            return False
        return _gfx_route_on_host(_target, _host_codes or [_target])
    # No target resolved: a mask exposing no GPU is deliberate, anything else is a detection
    # miss where the inventory is still the best evidence.
    if _visible_masks_select_no_gpu():
        return False
    _archs = _physical_amd_gfx_archs()
    return any(_gfx_route_on_host(_gfx, _archs) for _gfx in _archs)


def _gfx_has_a_wheel_route(gfx: "str | None") -> bool:
    """Whether ANY index this installer can pick carries kernels for ``gfx``.

    Either the generic pytorch.org wheel serves it or AMD publishes a per-arch index. An arch
    in neither (gfx1010 / RDNA 1) cannot be fixed by picking a different index, so it must
    never depose a card that can.
    """
    return bool(gfx) and (gfx in _GENERIC_ROCM_WHEEL_GFX or gfx in _GFX_TO_AMD_INDEX_ARCH)


# Minimum ROCm tag whose generic wheels carry each arch: _ROCM_TORCH_INDEX also maps 6.0-6.4, and a
# new kernel beside a stale /opt/rocm can pick a tag with no kernels for the card.
_GENERIC_WHEEL_GFX_MIN_ROCM: "dict[str, tuple[int, int]]" = {
    # gfx950 has no _GFX_TO_AMD_INDEX_ARCH entry; this only steers the generic tag choice.
    "gfx950": (7, 0),
    "gfx1150": (7, 0),
    "gfx1151": (7, 0),
    # rocm6.3 is the first family whose rocBLAS / hipBLASLt carry gfx1102 (RX 7600).
    "gfx1102": (6, 3),
    "gfx1200": (6, 4),
    "gfx1201": (6, 4),
}


def _generic_rocm_wheel_lacks_kernels(
    gfx: "str | None", ver: "tuple[int, int] | None" = None
) -> bool:
    """Whether only an available AMD per-arch index carries kernels for ``gfx``.

    ``ver`` is the host ROCm version, when the caller has read one. Support belongs to the
    wheel a version resolves to, not to the generic index as a whole, so passing it lets an
    arch the OLD tags predate be rerouted rather than installed without kernels. Omitting it
    keeps the union reading, for callers with no version to key on.
    """
    if not gfx or gfx not in _GFX_TO_AMD_INDEX_ARCH:
        return False
    if gfx not in _GENERIC_ROCM_WHEEL_GFX:
        return True
    return ver is not None and _generic_tag_lacks_kernels(gfx, ver)


def _generic_tag_lacks_kernels(gfx: "str | None", ver: "tuple[int, int]") -> bool:
    """Whether the generic wheel ``ver`` resolves to predates ``gfx``.

    The tag question alone, with no reroute attached, so it also answers for the arches
    _generic_rocm_wheel_lacks_kernels declines: that one chooses between indexes and stays
    silent with no AMD leaf to move to, while a TAG has a second answer -- take a newer one.
    """
    _min = _GENERIC_WHEEL_GFX_MIN_ROCM.get(gfx or "")
    if _min is None:
        return False
    # Keyed on the selected tag (6.3.9 -> rocm6.3). Below every known tag, treat as unsupported.
    _tag_key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return _tag_key is None or _tag_key < _min


def _generic_only_target_below_floor(gfx: "str | None", ver: "tuple[int, int] | None") -> bool:
    """Whether a target with NO per-arch index is on a generic tag that predates it.

    The repair question for gfx950 and the other parts with no per-arch leaf.
    _generic_rocm_wheel_lacks_kernels answers False for them by design (it decides between
    INDEXES, and there is no second index), so callers asking only that read them as healthy
    on a wheel with no code for them. Their repair is a newer generic tag instead.

    An unreadable version answers True: that is the absence of a reading, not a reading that
    the tag clears the floor, and these parts run on hosts whose ROCm version does not read.
    """
    if not gfx or gfx in _GFX_TO_AMD_INDEX_ARCH or gfx not in _GENERIC_WHEEL_GFX_MIN_ROCM:
        return False
    return ver is None or _generic_tag_lacks_kernels(gfx, ver)


def _runtime_gfx_target(
    inferred_linux_gfx: "str | None",
) -> "tuple[str | None, list[str], str | None, list[str]]":
    """Return the selected gfx target, detected arches, corrected physical arch, and the
    machine as the probes saw it before any ROCr filtering.

    Sources, strongest first: the ROCm userland probes, then KFD topology sysfs, then the
    explicit UNSLOTH_ROCM_GFX_ARCH / inferred product arch. Only the first can be renumbered
    by a visible-device mask, so it leads and the rest answer only when it says nothing. They
    matter because a runtime-only ROCm install ships neither rocminfo nor amd-smi, and with
    no target the callers keep a wheel with no kernels for this GPU.
    """
    # Reset on entry so a caller never reads a previous host shape's answer.
    global _LAST_HIP_MASK_RESOLVED, _LAST_ROCR_MASK_RESOLVED, _LAST_GFX_TARGET_AMBIGUOUS
    _LAST_HIP_MASK_RESOLVED = True
    _LAST_ROCR_MASK_RESOLVED = True
    _LAST_GFX_TARGET_AMBIGUOUS = False
    # An empty or "-1" mask selects no GPU; decided before probing because amd-smi and KFD ignore
    # masks, and index 0 would reinstall the stack for a card the user hid.
    if _visible_masks_select_no_gpu():
        return None, [], None, []
    # An explicit arch outranks the probes (as in install.sh); the masks stay above it. Split on ":" so
    # a copied gcnArchName still keys the routing tables.
    _explicit_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":", 1)[0]
    if _explicit_gfx:
        # The early return skips _hsa_spoofed_physical_gfx, so check the spoof here: leaving
        # HSA_OVERRIDE_GFX_VERSION set with per-arch wheels hands torch an agent they have no code for.
        _spoofed = _explicit_gfx if _hsa_spoof_contradicts(_explicit_gfx) else None
        # Only explicit ROCm requests with a mask pay for these probes (up to 15s each).
        if _rocm_torch_explicitly_requested() and _visible_devices_pinned():
            # KFD node order is HIP/ROCr order; ignore masks since the ordinals index the unmasked list.
            _mask_devices = _kfd_gfx_targets() or _detect_amd_gfx_codes(
                dedup = False, ignore_visible_masks = True
            )
            if _mask_devices:
                _LAST_ROCR_MASK_RESOLVED = _rocr_layer_mask_names_a_device(len(_mask_devices))
                _LAST_HIP_MASK_RESOLVED = _hip_layer_mask_names_a_device(
                    len(_rocr_visible_subset(_mask_devices)[0])
                )
        return _explicit_gfx, [_explicit_gfx], _spoofed, [_explicit_gfx]
    gfx_devices = _detect_amd_gfx_codes(dedup = False)
    # Keyed to the userland probe: ROCr spoofs that reading and no other.
    physical_gfx = _hsa_spoofed_physical_gfx(inferred_linux_gfx, gfx_devices)
    if physical_gfx is not None:
        gfx_devices = [physical_gfx]
    if not gfx_devices:
        gfx_devices = _kfd_gfx_targets()
        # amdkfd's gfx_target_version is untouched by ROCr, so a single-arch kernel reading that contradicts
        # the override corroborates the spoof.
        if physical_gfx is None and len(set(gfx_devices)) == 1:
            _override_arch = _hsa_override_gfx_arch(os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
            if _override_arch is not None and _override_arch != gfx_devices[0]:
                physical_gfx = gfx_devices[0]
    if not gfx_devices and inferred_linux_gfx:
        # Nothing enumerated a device, so one product-name guess must not absorb an out-of-range ordinal.
        # Decline unless the arch was named outright.
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
            # A rejection, not a detection miss: callers must not stand down CUDA for a refused swap.
            _LAST_HIP_MASK_RESOLVED = False
            return None, [], None, []
        gfx_devices = [inferred_linux_gfx]
    # rocminfo is already ROCr-filtered, so a mask-selected MI50 would look single-arch and be granted
    # the rocm6.3 downgrade; ask the whole machine. Provenance is read before the re-probe rewrites it.
    _probe_source = _LAST_AMD_GFX_PROBE
    if _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ:
        try:
            _unmasked = _detect_amd_gfx_codes(dedup = False, ignore_visible_masks = True)
        except Exception:
            _unmasked = []
        host_codes = list(dict.fromkeys(_unmasked or gfx_devices))
    else:
        host_codes = list(dict.fromkeys(gfx_devices))
    # Masks compose: ROCr filters first, HIP indexes the survivors. Only rocminfo is ROCr-filtered, so
    # apply the ROCr layer here for amd-smi and KFD sysfs.
    rocr_applied = _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ
    _unlike_adapters = len(set(gfx_devices)) > 1
    if not rocr_applied:
        # amd-smi lists in discovery order, not HIP/ROCr order (they differ on MI350X), and no HIP_ID map is
        # read here, so on unlike adapters an ordinal can name another card even with no mask set.
        _discovery_ordered = _probe_source == "amd-smi"
        if _discovery_ordered and _unlike_adapters:
            # KFD node order is what HIP/ROCr index, so use it when the device count matches.
            _kfd_ordered = _kfd_gfx_targets()
            if len(_kfd_ordered) == len(gfx_devices):
                gfx_devices = _kfd_ordered
                _unlike_adapters = len(set(gfx_devices)) > 1
                _discovery_ordered = False
        # Against the list BEFORE the subset, which is what the ROCr ordinals index.
        _LAST_ROCR_MASK_RESOLVED = _rocr_layer_mask_names_a_device(len(gfx_devices))
        gfx_devices, _rocr_unresolved = _rocr_visible_subset(gfx_devices)
        # A UUID names a device this cannot place; judge against the unmasked list so ambiguity stays visible.
        if (_rocr_unresolved or _discovery_ordered) and _unlike_adapters:
            # Honour UNSLOTH_ROCM_GFX_ARCH (offered by the message below); read env, not the product guess.
            _named_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
            if _named_gfx:
                # The named arch leads the set, or a target missing from it is selected but never repaired.
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
            _LAST_GFX_TARGET_AMBIGUOUS = True
            return None, [], None, host_codes
    if gfx_devices:
        _LAST_HIP_MASK_RESOLVED = _hip_layer_mask_names_a_device(len(gfx_devices))
    runtime_gfx = (
        gfx_devices[_pick_visible_index(len(gfx_devices), masks = _HIP_LAYER_MASKS)]
        if gfx_devices
        else None
    )
    if runtime_gfx in _SHADOWING_INTEGRATED_GFX and not _visible_devices_pinned():
        # Unpinned mixed host: let the discrete card decide the family, as _detect_windows_gfx_arch does.
        # gfx906 is excluded: its route needs a sole-arch host, so it would strand both cards.
        _others = [
            g
            for g in gfx_devices
            if g not in _SHADOWING_INTEGRATED_GFX and g not in _MIXED_HOST_UNROUTABLE
        ]
        # Only depose a routable APU for a discrete card the installer can serve.
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
    """gfx arch named by an HSA_OVERRIDE_GFX_VERSION value, or None if unreadable.

    ROCr reads the variable as a major.minor.stepping triple and builds the target
    name as gfx<major><minor><stepping in hex>, which is why 9.0.10 is gfx90a:
    11.0.0 -> gfx1100, 11.5.1 -> gfx1151, 10.3.0 -> gfx1030.
    """
    if not value:
        return None
    # [0-9], not \d / isdigit(), which accept non-ASCII digits install.sh's awk rejects.
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value.strip()):
        return None
    major, minor, step = (int(p) for p in value.strip().split("."))
    # Steppings are a single hex nibble; anything wider is not a real target.
    if not (0 <= step <= 15) or major <= 0 or minor > 9:
        return None
    return f"gfx{major}{minor}{step:x}"


def _kfd_gfx_targets() -> list[str]:
    """gfx arches of the AMD GPUs the KERNEL sees, from KFD topology sysfs.

    /sys/class/kfd/kfd/topology/nodes/<n>/properties carries gfx_target_version,
    written by amdkfd itself, so it is immune to HSA_OVERRIDE_GFX_VERSION (which
    ROCr applies in userland) and is the ground truth for #7331. Encoding is
    major * 10000 + minor * 100 + stepping, the stepping in hex: 110000 -> gfx1100,
    110501 -> gfx1151, 90010 -> gfx90a.

    CPU nodes carry no gfx_target_version (or 0) and drop out; the vendor_id 4098
    (0x1002) guard mirrors _has_rocm_gpu() and keeps NVIDIA's open-driver KFD nodes
    out. Returns one entry per GPU node, in node order.
    """
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
    """Physical arch when the ISA probe is an HSA_OVERRIDE_GFX_VERSION spoof (#7331).

    Returns None -- "believe the probe", today's behaviour -- unless all of:

      * HSA_OVERRIDE_GFX_VERSION is set. Without it there is nothing to doubt, so
        the deliberate #7305 precedence (a mixed Strix APU + dGPU host with the
        dGPU selected must not get APU wheels) is untouched.
      * The product name inferred an arch that people spoof and the probe reports
        a DIFFERENT one. An override naming the arch the hardware already is masks
        nothing.
      * The probe saw exactly one arch. A pre-filter, not the safety property:
        install.sh can only count DISTINCT tokens (its probe greps rocminfo, which
        repeats the token per agent), so counting arches here keeps the two
        implementations at the same verdict.
      * The variable names EXACTLY the reported arch. ROCr can only spoof to the
        target the variable names, so any other reading is real silicon.
      * A source the override cannot reach corroborates it, strongest first:

        1. KFD topology sysfs (_kfd_gfx_targets). amdkfd writes gfx_target_version
           from the kernel's own IP-version table and ROCr, which applies the
           override in userland, never touches it. If the kernel names the
           inferred arch, the matter is settled.
        2. Re-probing rocminfo with HSA_OVERRIDE_GFX_VERSION stripped (and the
           visible masks with it, so the re-probe sees the whole machine): ROCr
           getenv()s the override while building agent names, so without it the
           runtime itself retracts the spoofed name.

    Corroboration is REQUIRED, with deliberately no "the variable names the
    reported arch, so assume a spoof" fallback: that shape is indistinguishable
    from a truthful host (a real gfx1100 dGPU in a Ryzen AI Max chassis whose owner
    set the override for unrelated reasons), and rerouting a working machine to the
    wrong wheels is worse than #7331 itself. Two independent readings have to agree
    against the one spoofed reading before anything is overridden.
    """
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
        """Decisive only when the source names the product arch and nothing else: a
        second arch means the single-arch premise was wrong (a mixed host whose
        second GPU the spoofed probe collapsed away), so decline. install.sh
        compares the same two strings, hence the verbatim KFD list and deduplicated
        re-probe."""
        if physical == [inferred_gfx]:
            _safe_print(
                f"   {source} reports {inferred_gfx} -- {probed} is a spoof of the physical arch.\n"
            )
            return inferred_gfx
        # On a real gfx1100 card in a Ryzen AI Max chassis this is the correct outcome.
        _safe_print(
            f"   {source} does not corroborate a spoof "
            f"({physical or 'no answer'}); keeping {probed}.\n"
        )
        return None

    # 1. The kernel, which the override cannot reach; decisive whenever it answers.
    kfd = _kfd_gfx_targets()
    if kfd:
        return _confirm(kfd, "KFD topology sysfs")

    # 2. The runtime, re-asked without the override and masks so a hidden second GPU can veto.
    _saved_probe = _LAST_AMD_GFX_PROBE
    try:
        reprobed = _detect_amd_gfx_codes(
            dedup = False, ignore_hsa_override = True, ignore_visible_masks = True
        )
    except Exception:
        reprobed = []
    finally:
        _LAST_AMD_GFX_PROBE = _saved_probe
    # Still answering `probed` without the override means real silicon, not a spoof.
    return _confirm(list(dict.fromkeys(reprobed)), "rocminfo with HSA_OVERRIDE_GFX_VERSION unset")


def _hsa_spoof_contradicts(gfx: "str | None") -> bool:
    """True when HSA_OVERRIDE_GFX_VERSION names an arch other than ``gfx``.

    Every branch installing per-arch wheels has to ask: those wheels carry one arch's code
    objects and ROCr builds the agent's ISA name from this variable. An override naming the
    SAME arch is not a spoof; a different one is left alone only where ``gfx`` is untrusted.
    """
    _override_arch = _hsa_override_gfx_arch(os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
    return bool(gfx) and _override_arch is not None and _override_arch != gfx


def _clear_confirmed_hsa_spoof(physical_gfx: str) -> None:
    """Drop a CONFIRMED HSA_OVERRIDE_GFX_VERSION spoof from this process's env.

    Routing the wheels is only half of #7331. ROCr reads the variable afresh in
    every LATER process -- libhsakmt writes props->EngineId straight from it while
    building the agent, so the agent's ISA name becomes the spoofed arch -- while
    AMD's per-gfx index ships code objects for the physical arch alone. Leave it set
    and the new wheel is handed a device whose name matches none of its code, so the
    first allocation fails exactly as before.

    Only ever called after corroboration and only on the branch installing native
    wheels for ``physical_gfx``, so the variable is provably lying about this host's
    only GPU and nothing on this path still needs it. A shell profile that exports
    it will set it again next login, which no installer can undo from here, so name
    the variable and say to remove it.
    """
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

# CUDA_VISIBLE_DEVICES aliases HIP's layer on AMD; ROCR_VISIBLE_DEVICES is beneath it, see
# _rocr_visible_subset.
_HIP_LAYER_MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# Larger than any device count, so an ordinal resolves to itself instead of the 0 fallback.
_INDEX_PROBE_LEN = 1 << 20


def _hip_layer_mask_names_a_device(device_count: int) -> bool:
    """Whether the HIP-layer mask, if set, names a device index this host has.

    _pick_visible_index answers 0 for a mask it cannot resolve, because for arch SELECTION a
    first-GPU guess beats no answer. Deciding whether to REPLACE a working CUDA stack is the
    opposite question: HIP exposes no device, so the ROCm wheels would land where the runtime
    hands torch nothing. install.sh fails closed on the same input.

    First-set-wins between the two spellings, as _pick_visible_index documents, against the
    list the HIP layer indexes (the ROCr survivors, which the caller has applied). True when
    no HIP-layer mask is set, which is not the same as failing to resolve one.
    """
    for _env in _HIP_LAYER_MASKS:
        _val = os.environ.get(_env)
        if _val is None:
            continue
        _val = _val.strip()
        if _val == "" or _val == "-1":
            return False
        _first = _val.split(",")[0].strip()
        try:
            return 0 <= int(_first) < device_count
        except ValueError:
            # A UUID or junk; ROCr UUID forms are resolved by _rocr_visible_subset.
            return False
    return True


def _rocr_layer_mask_names_a_device(device_count: int) -> bool:
    """Whether ROCR_VISIBLE_DEVICES, if set, leaves the runtime at least one device.

    ROCr's filter (ROCR-Runtime, core/inc/amd_filter_device.h) keeps the tokens that are
    "Legal and NOT Terminating", and an index terminates when it "lies outside the interval
    [0 - (numGpuDevices - 1)]" or "maps to a device that has been previously selected". So
    the survivors are a PREFIX: ROCR_VISIBLE_DEVICES=7 on a two-GPU box surfaces nothing.

    _rocr_visible_subset keeps the whole list there instead, deliberately, so arch SELECTION
    still guesses GPU 0. This is the other question -- whether to REPLACE a working CUDA
    stack -- and it fails closed, as _hip_layer_mask_names_a_device does one layer up. A UUID
    resolves to no position here. install.sh composes the same rule in _amd_mask_survivors.
    """
    _raw = (os.environ.get("ROCR_VISIBLE_DEVICES") or "").strip()
    if not _raw:
        return True
    _selected: "set[int]" = set()
    for _tok in _raw.split(","):
        _tok = _tok.strip()
        try:
            _idx = int(_tok)
        except ValueError:
            break
        if not (0 <= _idx < device_count) or _idx in _selected:
            break
        _selected.add(_idx)
    return bool(_selected)


def _rocr_visible_subset(gfx_devices: "list[str]") -> "tuple[list[str], bool]":
    """Apply the ROCr layer to a device list no probe filtered.

    Returns the surviving devices and whether any token could NOT be resolved to one.

    ROCR_VISIBLE_DEVICES is processed below HIP, deciding which devices exist at all before
    any HIP index resolves. Neither amd-smi (driver) nor KFD sysfs (kernel) is filtered by it,
    so a HIP index resolved against their whole-machine list names a GPU the runtime does not
    expose. The mask takes indices or UUIDs and may MIX them ("0,GPU-DEADBEEFDEADBEEF"); a
    probe reporting arches names no UUID, so those tokens resolve to no position and are
    reported unresolved rather than guessed at. An empty or "-1" mask never reaches here:
    _visible_masks_select_no_gpu already refused the host.
    """
    _raw = (os.environ.get("ROCR_VISIBLE_DEVICES") or "").strip()
    if not _raw or not gfx_devices:
        return gfx_devices, False
    _kept: "list[str]" = []
    _seen: "set[int]" = set()
    _unresolved = False
    for _tok in _raw.split(","):
        _tok = _tok.strip()
        try:
            _idx = int(_tok)
        except ValueError:
            _unresolved = True  # a UUID: this names a device, but not a position
            continue
        # Survivors are a prefix: an out-of-range or repeated index ends the mask.
        if _idx in _seen or not (0 <= _idx < len(gfx_devices)):
            break
        _seen.add(_idx)
        _kept.append(gfx_devices[_idx])
    # First index naming nothing keeps the whole list, matching _pick_visible_index; fail-closed via
    # _LAST_ROCR_MASK_RESOLVED.
    return (_kept or gfx_devices), _unresolved


def _visible_masks_select_no_gpu() -> bool:
    """True when a set visible-device mask exposes NO GPU, at either layer.

    ROCr filters BENEATH HIP, so an empty or -1 mask on either one leaves nothing to target
    however the other reads: HIP_VISIBLE_DEVICES=0 over ROCR_VISIBLE_DEVICES=-1 still exposes
    no device. CUDA_VISIBLE_DEVICES is the HIP alias and is read only when HIP itself is unset.
    """
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
    # --force-reinstall re-downloads 39 MB before noticing the wheel is installed; BNB_ROCM_VERSION below
    # still runs.
    if _bnb_rocm_install_is_current(_bnb_win_url):
        _safe_print(_dim("   bitsandbytes (AMD Windows) is already this build -- keeping it"))
        # Record on skip too, or the next update refetches.
        _record_bnb_rocm_provenance()
        _ok = True
    else:
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
        # Only after an install landed, so a failure never pairs the new asset identity with the old wheel.
        if _ok:
            _record_bnb_rocm_provenance()
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


def _nvidia_smi_candidates() -> "list[str]":
    """Every place nvidia-smi is worth looking for, PATH first.

    One list, because two callers that disagree about where nvidia-smi lives disagree
    about the host: _has_usable_nvidia_gpu would find the driver through the Windows
    fixed locations while _detect_cuda_torch_index_url, reading PATH only, fell back to
    its cu126 default and recorded a family the driver never reported.
    """
    candidates = []
    exe = shutil.which("nvidia-smi")
    if exe:
        candidates.append(exe)
    if IS_WINDOWS:
        # The locations install.ps1 / setup.ps1 use; nvidia-smi.exe is routinely off PATH.
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
    else:
        # The canonical Linux path a stripped-down PATH (systemd units, cron) can miss.
        candidates.append("/usr/bin/nvidia-smi")
    return candidates


def _nvidia_smi_lists_a_gpu(exe: str) -> bool:
    """Whether this nvidia-smi actually enumerates a GPU.

    The predicate both probes have to agree on. A stale copy can exit 0 and print a
    perfectly parseable "CUDA Version:" banner while `-L` lists nothing, which is the
    state setup.ps1's Test-NvidiaSmiHasGpu rejects.
    """
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


def _nvidia_smi_usable_candidates() -> "list[str]":
    """Candidates worth actually running: the PATH result plus every file that exists.

    `shutil.which` already proved its result runnable, so it is kept without an
    os.path.isfile check -- that would reject a bare "nvidia-smi" relative to a CWD it
    does not live in, which is both what a stubbed test double looks like and what a
    GPU-free runner would turn into a silent cu126 default.
    """
    path_exe = shutil.which("nvidia-smi")
    return [
        candidate
        for candidate in _nvidia_smi_candidates()
        if candidate == path_exe or os.path.isfile(candidate)
    ]


def _nvidia_smi_path() -> "str | None":
    """The first nvidia-smi worth running, or None."""
    candidates = _nvidia_smi_usable_candidates()
    return candidates[0] if candidates else None


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


# torch 2.11 cu126 spans sm_50-90 with no PTX above, so Kepler or Blackwell leaves the host uncovered.
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


def _cap_cuda_family_for_pre_turing(
    family: str,
    exe: "str | None",
    sms: "list[int] | None" = None,
) -> str:
    """Use cu126 when it covers every physical GPU missed by the selected family.

    CUDA_VISIBLE_DEVICES is intentionally ignored. Non-x86_64 hosts retain the
    driver-derived family because their wheel matrices differ. `sms` is the inventory
    already in hand (the library probe); otherwise it is read from `exe`.
    """
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return family
    span = _cuda_family_sm_range(family)
    if span is None or (exe is None and sms is None):
        return family
    if span[0] <= _CU126_SM_RANGE[0]:
        return family  # nothing lower to fall back to
    floor = span[0]
    if sms is None:
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


def _torch_family_for_cuda_version(major: int, minor: int) -> str:
    """install.sh::get_torch_index_url's CUDA ladder, from the driver's CUDA version."""
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major >= 12:
        return "cu124"
    if major >= 11:
        return "cu118"
    return "cpu"  # ancient driver: no usable CUDA wheels


def _detect_cuda_torch_index_family(*, known_only: bool = False) -> str | None:
    """Return the CUDA wheel family (index leaf) for the host's NVIDIA driver.

    Mirrors install.sh::get_torch_index_url's CUDA ladder so `studio update` repairs
    to the same wheel family a fresh install would pick. Honours the explicit
    overrides first (UNSLOTH_TORCH_INDEX_URL / _FAMILY) so a headless / CI install
    never lets the host GPU decide. Otherwise probes nvidia-smi (parsing both "CUDA
    Version:" and "CUDA UMD Version:"), then the driver library, defaulting to cu126
    when neither answers, or to None with known_only. The driver version is only an
    upper bound, so the GPU architectures can cap the result at cu126 (see
    _cap_cuda_family_for_pre_turing).
    """
    _override_url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if _override_url:
        return _torch_index_leaf(_trim_index_path_slashes(_override_url))
    _override_family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
    if _override_family:
        return _override_family.strip("/")
    # Try every candidate until one answers: a stale nvidia-smi on PATH would default to cu126, which
    # has no Blackwell kernels.
    for exe in _nvidia_smi_usable_candidates():
        # Same predicate as the presence probe (and setup.ps1's Test-NvidiaSmiHasGpu): skip copies listing
        # no GPU.
        if not _nvidia_smi_lists_a_gpu(exe):
            continue
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
        except Exception:
            continue
        if result.returncode != 0:
            continue
        m = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout)
        if m is None:
            continue
        tag = _torch_family_for_cuda_version(int(m.group(1)), int(m.group(2)))
        return _cap_cuda_family_for_pre_turing(tag, exe)
    # No nvidia-smi: the driver libraries give version and SMs; without SMs the caller default stays.
    inventory = _nvidia_library_inventory()
    if inventory is not None and inventory.cuda_driver_version:
        sms = _inventory_compute_sms(inventory)
        if sms:
            family = _torch_family_for_cuda_version(*inventory.cuda_driver_version)
            return _cap_cuda_family_for_pre_turing(family, None, sms)
    return None if known_only else "cu126"


def _detect_cuda_torch_index_url(*, known_only: bool = False) -> str | None:
    """_detect_cuda_torch_index_family as a URL; None also for a query-auth mirror."""
    _override_url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if _override_url:
        return _trim_index_path_slashes(_override_url)
    family = _detect_cuda_torch_index_family(known_only = known_only)
    return None if family is None else _pytorch_whl_leaf_url(family)


def _inventory_compute_sms(inventory) -> "list[int]":
    """Every sm_NN the driver library lists; empty when one row is unreadable, like
    _nvidia_compute_sms."""
    sms: list[int] = []
    for row in inventory.devices:
        m = re.fullmatch(r"(\d+)\.(\d+)", row.get("compute_cap", ""))
        if m is None:
            return []
        sms.append(int(m.group(1)) * 10 + int(m.group(2)))
    return sms


def _host_compute_sms() -> "list[int] | None":
    """The host's sm_NN list from nvidia-smi, else from the driver library; None when
    neither can say."""
    smi = _nvidia_smi_path()
    sms = _nvidia_compute_sms(smi) if smi else None
    if sms:
        return sms
    inventory = _nvidia_library_inventory()
    return (_inventory_compute_sms(inventory) or None) if inventory is not None else None


def _driver_cuda_torch_flavor_tag() -> str:
    """The CUDA wheel family this driver can actually run, or "" if it can run none.

    _detect_cuda_torch_index_family mirrors setup.ps1::Get-PytorchCudaTag, ancient-driver "cpu"
    and pre-Turing cap included, so its leaf asks the same question the handover did.
    """
    leaf = _torch_index_leaf(_detect_cuda_torch_index_family() or "")
    return leaf if _is_cuda_family_leaf(leaf) else ""


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
        return _pytorch_whl_leaf_url(family.strip("/"))
    return None


def _explicit_torch_index_family() -> "str | None":
    """The pin's family leaf even when no URL can express it; read this, not the URL, to ask
    whether a pin EXISTS, or a query-auth FAMILY pin is re-probed off the GPU."""
    url = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if url:
        return _torch_index_leaf(_trim_index_path_slashes(url))
    # Leaf, not the whole value: install.sh accepts a multi-segment FAMILY (nightly/cu128).
    family = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip().strip("/")
    return _torch_index_leaf(family) if family else None


def _explicit_torch_index_is_unusable() -> bool:
    """A pin exists, but its FAMILY leaf cannot become a pip URL on this mirror."""
    return _explicit_torch_index_family() is not None and _explicit_torch_index_url() is None


def _is_pip_rocm_family_leaf(leaf: str) -> bool:
    """True when a lowercased leaf names a pip --index-url ROCm family: an EXACT
    rocm<digits>[.<digits>] leaf or a gfx leaf. A suffixed leaf (rocm-rel-7.2.1,
    rocm7.2-private) starts with "rocm" but is a custom pin the verbatim path owns, so
    match EXACTLY. Mirrors install.sh / setup.ps1.
    """
    # gfx must be followed by a digit; a gfx-private custom leaf is a verbatim pin.
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
    # A major-only rocm<d> leaf is valid, so the minor is optional.
    _pin_rocm = re.match(r"^rocm(\d+)(?:\.(\d+))?", leaf)
    _pin_major = int(_pin_rocm.group(1)) if _pin_rocm else None
    _pin_ver = (
        (int(_pin_rocm.group(1)), int(_pin_rocm.group(2)))
        if _pin_rocm and _pin_rocm.group(2) is not None
        else None
    )
    # A three-part +rocmA.B.C tag marks AMD's per-arch wheel; pytorch.org uses two parts.
    _inst_rocm = re.search(r"\+rocm(\d+)\.(\d+)", installed_ver)
    _inst_ver = (int(_inst_rocm.group(1)), int(_inst_rocm.group(2))) if _inst_rocm else None
    _inst_is_perarch = re.search(r"\+rocm\d+\.\d+\.\d+", installed_ver) is not None
    _inst_has_rocm = re.search(r"\+rocm", installed_ver) is not None
    _inst_rel = re.match(r"^(\d+)\.(\d+)", installed_ver)
    _inst_is_211 = (
        (int(_inst_rel.group(1)), int(_inst_rel.group(2))) >= (2, 11) if _inst_rel else False
    )

    if leaf.startswith("gfx"):
        # Two per-arch leaves look identical by version (2.11, three-part tag); the installed `rocm`
        # meta-package names the family. None is unknowable, so this can only add a mismatch.
        _family = _installed_rocm_wheel_family()
        if _family is not None and _family != leaf:
            return True
        # Floor leaves: a matching family on a 2.10 build is still the _grouped_mm bug, so check first.
        if leaf in _ROCM_GFX_TORCH211_LEAVES:
            return not (_inst_is_211 and _inst_is_perarch)
        # On leaves without a floor the family is decisive, or a correctly pinned gfx90a reinstalls every update.
        if _family is not None and _inst_is_perarch:
            return False
        return (not _inst_has_rocm) or _inst_is_211

    # Major-only pin (rocm7): compare majors only.
    if _pin_major is not None and _pin_ver is None:
        if _inst_ver is not None:
            return _inst_ver[0] != _pin_major
        return not _inst_has_rocm

    # rocmX.Y pin. Only KNOWN-2.11 rocm is the 2.11 line (no speculative floor).
    _pin_is_211 = _pin_ver in _ROCM_KNOWN_TORCH211_VERSIONS if _pin_ver is not None else False
    if _pin_ver is not None and _inst_ver is not None:
        if _pin_ver != _inst_ver:
            return True
        # A known-2.11 pin whose release drifted off 2.11 (2.12+rocm7.2) violates the spec.
        if _pin_is_211 and _inst_rel is not None:
            if (int(_inst_rel.group(1)), int(_inst_rel.group(2))) != (2, 11):
                return True
        return False
    if not _inst_has_rocm:
        return True
    return _pin_is_211 != _inst_is_211


# Own XPU range: the xpu index serves past our ceiling, and unsloth raises below 2.6 on XPU.
# Kept in step with install.sh by tests/sh/test_xpu_torch_spec_parity.sh.
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
    leaf = _explicit_torch_index_family()
    if leaf is None:
        return None
    if _is_pip_rocm_family_leaf(leaf) or leaf == "cpu" or _is_cuda_family_leaf(leaf):
        return None
    # "" = exists but inexpressible: it installed nothing, so it is no provenance.
    return _explicit_torch_index_url() or ""


def _deliberate_cpu_torch() -> bool:
    """Someone chose CPU torch: an explicit CPU index pin, or a manifest that recorded cpu
    as NAMED rather than selected and nothing in this run names a GPU family instead.
    An unproven cpu record is not a choice."""
    return _explicit_cpu_torch_index_pin() or _expected_torch_flavor_was_pinned("cpu")


def _ensure_cuda_torch(*, probe_only: bool = False) -> "bool | None":
    """Repair a venv whose torch is a ROCm build on an NVIDIA host.

    Counterpart to _ensure_rocm_torch. A venv poisoned by the pre-fix KFD
    gpu_id false positive (ROCm torch installed on an NVIDIA-only machine)
    keeps that broken torch on `studio update`, because a torch+rocm wheel
    satisfies the version constraint and nothing force-reinstalls it. This
    detects that exact case and reinstalls CUDA torch.

    Also repairs a CUDA torch whose wheel family ships no kernels for the host's
    GPUs (a pre-Turing box that the driver-only ladder sent to cu128/cu130).
    Healthy CUDA torch and deliberate CPU-only torch are left untouched.

    probe_only returns True where the repair would install and installs nothing, so the
    fast-path escape that calls it cannot drift from what the repair actually does.
    False: a repair is required but the mirror cannot express its index; the caller aborts.
    """
    # Respect install.sh's backend: only "" (standalone update) or "cuda" force CUDA wheels.
    if _TORCH_BACKEND not in ("", "cuda"):
        return
    # An explicit unknown-family pin was applied VERBATIM at install time; leave it alone.
    if _explicit_unknown_family_torch_index_url() is not None:
        return
    _pin_family = _explicit_torch_index_family() or ""
    _pin_unusable = _explicit_torch_index_is_unusable()
    if _pin_unusable and not _is_cuda_family_leaf(_pin_family):
        return
    # No CUDA torch on macOS; Windows torch is owned by install.ps1 (KFD bug is Linux-only).
    if IS_MACOS or IS_WINDOWS or NO_TORCH:
        return
    # Never undo a deliberate ROCm install (setup.ps1 sets this marker).
    if os.environ.get("UNSLOTH_ROCM_TORCH_INSTALLED") == "1":
        return
    # Nor one this run asked for (standalone update leaves _TORCH_BACKEND empty); same route predicate
    # as _ensure_rocm_torch so the two cannot fight.
    if (
        _rocm_torch_explicitly_requested()
        and not _is_cuda_family_leaf(_pin_family)
        and _forced_rocm_route_is_viable()
    ):
        return
    _cuda_pinned = _explicit_cuda_torch_index_url() is not None or _pin_unusable
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not _cuda_pinned and _cvd is not None and _cvd.strip() in ("", "-1"):
        return
    if not _cuda_pinned and not _has_usable_nvidia_gpu():
        return

    # "hip" on an NVIDIA host is the poisoning signature.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        return
    if not _importable:
        # Unimportable: the base install owns it unless a CUDA pin is set, since the base update will not
        # reinstall an already-installed torch.
        if not _cuda_pinned:
            return
        if probe_only:
            return True
        index_url = _detect_cuda_torch_index_url()
        if index_url is None:
            return False
        _torch_pkg, _vision_pkg, _audio_pkg = _cuda_repair_torch_specs(
            index_url, _CUDA_TORCH_PKG_SPEC
        )
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
        return
    # torch.version.cuda is the only CUDA clue on an untagged wheel (PyPI forbids +cuXXX).
    _ver = _version.lower()
    _cu_match = re.search(r"\+(cu\d+)", _ver)
    _marker = "hip" if (_hip or "rocm" in _ver) else ("cuda" if _cuda else "cpu")
    _installed_cu = _cu_match.group(1) if _cu_match else ""
    _installed_release = _ver.split("+", 1)[0]
    _runtime_cu = ("cu" + _cuda.replace(".", "")) if _cuda else ""
    # Reinstall on ROCm-on-NVIDIA, a pinned family mismatch, or a family with no kernels for these GPUs.
    _pin_leaf = _pin_family
    _pinned_cuda = _is_cuda_family_leaf(_pin_leaf)
    if _marker == "hip":
        _why = "torch is a ROCm build on an NVIDIA host"
    elif _marker == "cpu" and _pinned_cuda:
        _why = "torch is a CPU build but an explicit CUDA index is pinned"
    elif _marker == "cuda" and _pinned_cuda and _installed_cu != _pin_leaf:
        _installed_desc = _installed_cu if _installed_cu else "an untagged CUDA build"
        _why = f"torch is {_installed_desc} but the pinned CUDA index is {_pin_leaf}"
    elif _marker == "cuda" and not _pinned_cuda:
        # x86_64 only: the spans below are the x86_64 build matrix.
        if platform.machine().lower() not in ("x86_64", "amd64"):
            return
        _family = _installed_cu or _runtime_cu
        _span = _cuda_family_sm_range(_family, _installed_release)
        if _span is None:
            return  # untagged or unrecognised build: not this check's business
        _sms = _host_compute_sms()
        if not _sms or _span_covers(_span, _sms):
            return  # healthy CUDA torch this host can use
        # Never trade one partial family for another, or reinstall the same one forever.
        _target = _torch_index_leaf(_detect_cuda_torch_index_family() or "")
        _target_span = _cuda_family_sm_range(_target)
        if _target_span is None or not _span_covers(_target_span, _sms):
            return
        _why = (
            f"torch is {_family} but this host has GPUs outside its sm_{_span[0]}-{_span[1]} range"
        )
    elif (
        _marker == "cpu"
        and not _deliberate_cpu_torch()
        and _is_cuda_family_leaf(
            _torch_index_leaf(_detect_cuda_torch_index_family(known_only = True) or "")
        )
    ):
        # An unrequested CPU wheel on an NVIDIA host whose driver is known to run CUDA (the cu126 default
        # for an unreadable driver is not evidence).
        _recorded = _RECORDED_TORCH_TAG or ""
        _why = "torch is a CPU build on an NVIDIA host" + (
            f" although this install recorded {_recorded}"
            if _is_cuda_family_leaf(_recorded)
            else ""
        )
    else:
        return  # healthy CUDA torch matching the pin, or a deliberate CPU wheel

    if probe_only:
        return True
    index_url = _detect_cuda_torch_index_url()
    if index_url is None:
        return False
    _torch_pkg, _vision_pkg, _audio_pkg = _cuda_repair_torch_specs(index_url, _CUDA_TORCH_PKG_SPEC)
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


def _ensure_xpu_torch() -> "bool | None":
    """Install XPU torch when an explicit XPU pin is set but the venv has another build.

    Counterpart to _ensure_cpu_torch for Intel. `unsloth studio update` runs setup.sh, never
    install.sh, so its XPU install path is unreachable there; and an xpu leaf names no family
    the cuda/rocm helpers know, so they skip it and the CPU wheel survives the pin forever.

    Windows is excluded on purpose: setup.ps1 owns torch there and already installs the XPU
    trio itself, so acting here would fight it. macOS has no XPU at all. False: a repair is
    required but the mirror cannot express the XPU index; the caller aborts.
    """
    if NO_TORCH or IS_MACOS or IS_WINDOWS:
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None and _explicit_torch_index_family() != "xpu":
        return

    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        # Inconclusive probe: ask the disk. A supported wheel means a stalled Intel driver, which reinstalling
        # never fixes.
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
        # Flavour AND range: a migrated 2.5+xpu venv is broken. Range matches _XPU_TORCH_PKG_SPEC.
        _ver = _version.lower()
        _rel = _ver.split("+")[0].split(".")
        _n = tuple(int(x) for x in _rel[:2] if x.isdigit())
        if "+xpu" in _ver and len(_n) == 2 and (2, 6) <= _n < (2, 11):
            return  # already the pinned family, in the supported range
        _why = "torch is not a supported XPU build"
    else:
        _why = "torch cannot import"

    if pin is None:
        return False
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
        # torch may have been installed earlier in this run, after finders cached site-packages.
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

    On Windows setup.ps1 performs the same swap after this script exits, so its handover
    variable means "someone else will"; a direct run has no such postlude.
    """
    if NO_TORCH or IS_MACOS:
        return
    if IS_WINDOWS and _handover_torch_flavor_tag():
        return
    pin = _explicit_xpu_torch_index_url()
    if pin is None:
        # A one-shot xpu pin is gone by the next update but its +xpu wheel remains, so the installed wheel
        # is the pin. setup.sh keys its bnb floor on the same signal.
        if "+xpu" not in _installed_torch_version_label().lower():
            return
        pin = _pytorch_whl_leaf_url("xpu")
        if pin is None:
            return

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

    # Fetch, then uninstall, then install: the shared paths live in generic triton's record, so
    # uninstalling last would delete the XPU files; pre-fetching keeps a dead mirror from stranding
    # the venv between the two. uv has no `pip download`.
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
                # Scrub PIP_NO_INDEX / PIP_EXTRA_INDEX_URL / PIP_FIND_LINKS so only the pinned index serves it.
                env = _install_env_for_cmd(_dl_cmd),
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                timeout = 900,
            )
        except (OSError, subprocess.TimeoutExpired):
            dl = None
        wheels = glob.glob(os.path.join(tmp, "*.whl"))
        if dl is None or dl.returncode != 0 or not wheels:
            _safe_print(
                _red(
                    f"   could not fetch {spec}; generic triton {generic} left in place -- "
                    "it shadows torch XPU triton, so torch.compile will not use the XPU"
                )
            )
            return
        _count_install_action()
        removed = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "triton"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
        if removed.returncode != 0:
            # Locked venv: generic triton stays registered, and installing over it would get the files deleted later.
            _safe_print(
                _red(
                    f"   could not remove generic triton {generic}; leaving it in place -- it "
                    "shadows torch XPU triton, so torch.compile will not use the XPU"
                )
            )
            return
        # The venv now has NO triton. pip_install (exits) so no completion manifest lands over a broken
        # torch.compile.
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


def _ensure_cpu_torch() -> "bool | None":
    """Reinstall CPU torch when CPU is authoritative but the venv has a GPU build.

    Counterpart to _ensure_cuda/rocm_torch for the CPU case: those treat a CPU backend as a
    skip, so a standalone `studio update` would ignore the authoritative CPU choice. Authority
    is an EXPLICIT pin, or an AMD arch measured to compute incorrectly under ROCm (see
    _rocm_miscomputing_host for why that one must demote rather than just decline).
    False: the demotion is required but the mirror cannot express the CPU index; the caller
    aborts rather than certify a build known to be wrong.
    """
    if NO_TORCH:
        return
    pin = _explicit_cpu_torch_index_url()
    _reason = "an explicit CPU index is pinned"
    if pin is None:
        if _rocm_miscomputing_host():
            _reason = "this AMD arch computes incorrectly under ROCm (studio/ROCM_RDNA2_APU.md)"
        elif _explicit_torch_index_family() != "cpu":
            return
        pin = _pytorch_whl_leaf_url("cpu")

    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        # A hung import is the wedged-driver case this pin rescues; continue only for a GPU label on disk.
        if not _is_gpu_torch_label(_installed_torch_label_on_disk()):
            return
    if not _ran or not _importable:
        # Unimportable: the base update will not reinstall an installed torch, so reinstall from the pin.
        if pin is None:
            return False
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
    # '+xpu' and torch.version.xpu too: XPU sets neither .cuda nor .hip, and _installed_flavor_tag_now
    # reads the same marker, so the two must agree.
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

    if pin is None:
        return False
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
    """Classify a torch.__version__ into the installers' flavor vocabulary.

    +cuNNN -> "cuNNN", +rocm -> "rocm", +xpu -> "xpu", +cpu or untagged -> "cpu".
    MUST match install.ps1's ConvertTo-TorchFlavorTag and setup.ps1's stale-venv probe,
    because the tag this returns is compared against one those produced. "" only for an
    empty version, which means the classification failed rather than "cpu".

    Untagged reads as "cpu" deliberately: PyPI forbids the local +cuNNN label, so an
    untagged wheel is the PyPI build, which on Windows is CPU-only. A repair off that
    verdict is self-resolving -- the replacement carries a +cuNNN tag and matches on the
    next pass -- and _torch_build_is_gpu, not this function, decides whether to FAIL.
    """
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
    """Which GPU family an untagged wheel's runtime markers name.

    torch.version.xpu alongside .hip and .cuda, for the reason _torch_build_is_gpu already
    reads all three: an untagged source, conda or private-index XPU build carries its
    runtime only there. Omitting it let an explicit /cpu pin over such a wheel compare
    equal, return success without replacing anything, and then record a PINNED cpu flavor
    for an environment that still holds an XPU build.
    """
    if hip:
        return "rocm"
    if cuda:
        return "cuda"
    return "xpu"


def _resident_torch_flavor_tag() -> str:
    """The flavor installed now, after dependency resolution, or "" when unreadable."""
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran:
        if not _importable or not _version:
            return ""
        tag = _torch_flavor_tag(_version)
        # Untagged is CPU on Windows but may be a private/source GPU build elsewhere.
        if tag == "cpu" and (_hip or _cuda or _TORCH_RUNTIME_XPU):
            return _gpu_family_from_runtime_markers(_hip, _cuda)
        return tag
    label = _installed_torch_label_on_disk()
    return _torch_flavor_tag(label) if label else ""


def _torch_build_is_gpu() -> bool:
    """Whether the installed torch can use a GPU at all, on the evidence available.

    Weaker and more forgiving than _torch_flavor_tag, and used only for the FAIL verdict
    in _ensure_expected_torch_flavor: a wrong family is worth a reinstall, but only a
    build with no GPU support whatsoever is worth failing the update over.

    torch.version.cuda / .hip count alongside the local label, so an untagged wheel that
    does carry a CUDA runtime is not called CPU-only. An answer that never arrived (a
    wedged driver hanging `import torch` -- the host these repairs exist for) falls back
    to the on-disk label and, failing that, reads as a GPU build: ambiguity must not fail
    an update by itself.
    """
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran and _importable and _version:
        return (
            _is_gpu_torch_label(_version.lower())
            or bool(_hip)
            or bool(_cuda)
            or bool(_TORCH_RUNTIME_XPU)
        )
    label = _installed_torch_label_on_disk()
    return (not label) or _is_gpu_torch_label(label)


def _handover_torch_flavor_tag() -> str:
    """The flavor setup.sh / setup.ps1 published for this run, lowercased, or "".

    Named rather than read inline: the invariant has to tell a handover apart from the other
    sources of the same string.
    """
    return os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG", "").strip().lower()


def _expected_torch_flavor_tag() -> str:
    """The torch flavor this venv is SUPPOSED to hold, or "" when nothing can say.

    Resolution order, most authoritative first:
      1. UNSLOTH_EXPECTED_TORCH_TAG -- the setup script's own answer, exported by
         setup.ps1 immediately before it hands over, so it describes the index the torch
         install arm actually used (pin, preserved venv, GPU probe and all).
      2. An explicit index pin, when its family is one this vocabulary can name. The
         manifest records what a PREVIOUS run installed; a pin is the instruction for
         THIS one, so a freshly selected family has to outrank a stale record. Resolving
         the manifest first let a cu128 pin lose to a cu124 manifest, and
         _expected_torch_index_url then rejected the cu128 pin as a family mismatch and
         repaired from the PUBLIC cu124 index -- undoing both the family and the source
         the user had just chosen.
      3. The flavor the last completed install recorded in the manifest. Read at import
         (_RECORDED_TORCH_TAG), because install_python_stack() drops the manifest before
         the dependency pass.
      4. A live probe, for a run nothing set up: `python install_python_stack.py` by hand.
         Only an NVIDIA host, or an explicit pin, can expect a GPU build -- otherwise
         return "" rather than invent an expectation from an absent GPU.
    """
    env = _handover_torch_flavor_tag()
    if env:
        # Unpinned "cpu" from the setup scripts can mean the GPU probe came back empty, so it does not
        # outrank a CUDA manifest while the GPU is present. Resolved here so the enforced tag is recorded.
        if env == "cpu" and not _explicit_cpu_torch_index_pin():
            recorded = (_RECORDED_TORCH_TAG or "").strip().lower()
            if _is_cuda_family_leaf(recorded) and _has_usable_nvidia_gpu():
                # The driver's own family (same probe as setup.ps1), not the recorded one, so an old driver keeps cpu.
                # An explicit CUDA pin outranks the probe.
                pinned = _explicit_torch_index_family() or ""
                driver = pinned if _is_cuda_family_leaf(pinned) else _driver_cuda_torch_flavor_tag()
                if not driver:
                    return env
                _safe_print(
                    f"   [WARN] the installer handed over a CPU torch expectation, but this venv "
                    f"was recorded as {recorded} and an NVIDIA GPU is still present; "
                    f"enforcing {driver}."
                )
                return driver
        return env
    leaf = _explicit_torch_index_family()
    if leaf is not None:
        # "rocm" names every AMD leaf (rocm6.4, gfx1151); an unreadable one falls through.
        if _is_pip_rocm_family_leaf(leaf):
            return "rocm"
        if _is_cuda_family_leaf(leaf) or leaf in ("xpu", "cpu"):
            return leaf
    # A resolved backend is a stated choice; honoured when it agrees with the installed wheel, and ahead of
    # the manifest, which describes the previous install.
    if _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu"):
        _installed = _torch_flavor_tag(_installed_torch_version_label())
        # "cuda" names a family, so the wheel's cu tag is the flavor; otherwise a removed CPU pin lives on.
        if _TORCH_BACKEND == "cuda":
            if _is_cuda_family_leaf(_installed):
                return _installed
        elif _installed == _TORCH_BACKEND:
            return _TORCH_BACKEND
    # An unknown-family pin was applied verbatim and nothing can name its family.
    unknown_pin = _explicit_unknown_family_torch_index_url()
    if unknown_pin is not None:
        return "" if unknown_pin else (_RECORDED_TORCH_TAG or "")
    if _RECORDED_TORCH_TAG:
        return _RECORDED_TORCH_TAG
    if _explicit_torch_index_family() is None and not _has_usable_nvidia_gpu():
        return ""
    return _torch_index_leaf(_detect_cuda_torch_index_family() or "")


def _expected_torch_flavor_is_explicit() -> bool:
    """Whether the expectation came from someone SAYING so, rather than from a probe.

    True for the setup script's handover, an explicit index pin, and the flavor the last
    completed install recorded: the first three steps of _expected_torch_flavor_tag.
    False when only the live hardware probe can answer, which is the one case a
    visibility mask has any business overruling.
    """
    if _handover_torch_flavor_tag():
        return True
    if _explicit_torch_index_family() is not None:
        return True
    return bool(_RECORDED_TORCH_TAG)


def _recordable_torch_flavor_tag(resolved: str) -> str:
    """The flavor worth writing to the manifest, or "" when nothing is.

    Normally the flavor this run resolved, falling back to the previous install's so an
    update does not erase a record it simply had no occasion to recompute. An explicit
    pin whose leaf names no family (a corporate /simple mirror, /current) breaks that
    fallback: the wheel now in the venv came from that mirror, the old record describes
    a venv that no longer exists, and carrying it forward would hand a later unpinned run
    a flavor to "repair" the mirror's build back to.
    """
    # Dependency steps may still have moved torch: record what is resident, not the request.
    if _explicit_torch_index_is_unusable():
        return _resident_torch_flavor_tag()
    if resolved:
        return resolved
    if _explicit_unknown_family_torch_index_url():
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
    """Whether ``flavor`` was NAMED by whoever ran this install.

    Distinct from _expected_torch_flavor_is_explicit(), which counts setup.ps1's
    handover variable. setup.ps1 publishes that variable for an AUTOMATIC /cpu choice on
    a GPU-less host exactly as it does for a pinned one, so the handover cannot answer
    this question. An index pin, an index family, and UNSLOTH_TORCH_BACKEND all can:
    each of them is someone saying which build they want. Carried forward from the
    previous manifest, so an update that names nothing does not erase the fact.

    Each of them only answers for the family it NAMES. setup.ps1 falls back to the CPU
    index when a pinned ROCm or XPU install fails, and publishes the resolved cpu tag
    while the original GPU pin is still in the environment: counting that pin would
    record a pinned CPU flavor, and _expected_cpu_flavor_was_chosen() would then read a
    failed install as a deliberate one and suppress the repair guidance for good.
    ``flavor`` empty means nobody asked about a specific one, and every pin counts.
    """

    def _names_it(family: str) -> bool:
        return True if not flavor else family == _flavor_tag_family(flavor)

    if _explicit_torch_index_is_unusable():
        resident = _resident_torch_flavor_tag()
        pin_family = _explicit_torch_index_family() or ""
        if resident and (
            pin_family == resident or (_is_pip_rocm_family_leaf(pin_family) and resident == "rocm")
        ):
            return True  # nothing to install: the requested family is already resident
        # A dependency move to another leaf must not inherit the old leaf's pin bit.
        return (
            bool(_RECORDED_TORCH_TAG_PINNED)
            and bool(resident)
            and resident == _RECORDED_TORCH_TAG
            and flavor == resident
        )

    # Uses install.sh precedence (URL wins, family only without URL); reading the family separately
    # let a stale ..._FAMILY=cpu mark a GPU-less host's wheel as pinned.
    pin_family = _explicit_torch_index_family()
    if pin_family is not None and _names_it(_index_leaf_flavor_family(pin_family)):
        return True
    # install.sh marks a backend derived from the index it resolved; only an unmarked one is a preference.
    if (
        _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu")
        and os.environ.get("UNSLOTH_TORCH_BACKEND_SOURCE", "").strip().lower() != "resolved"
        and _names_it(_TORCH_BACKEND)
    ):
        return True
    # A record only speaks for a run that named nothing to contradict it; derived backends and
    # family-less leaves do not count.
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
    pin_family = _explicit_torch_index_family()
    if pin_family is not None:
        named.append(_index_leaf_flavor_family(pin_family))
    if (
        _TORCH_BACKEND in ("cpu", "cuda", "rocm", "xpu")
        and os.environ.get("UNSLOTH_TORCH_BACKEND_SOURCE", "").strip().lower() != "resolved"
    ):
        named.append(_TORCH_BACKEND)
    return tuple(named)


def _expected_torch_index_url(tag: str) -> "str | None":
    """The wheel index to repair `tag` from, or None when the mirror cannot express it.

    Prefers the exact URL the setup script installed from (UNSLOTH_TORCH_INSTALL_INDEX_URL)
    and then the explicit pin, because an authenticated mirror can only be repaired from
    the credentialed URL -- neither is reconstructible from a family leaf. Both are only
    used when their leaf IS this tag: setup.ps1 sends the /cpu index alongside a "rocm"
    tag on the AMD Windows path, and repairing a cu* mismatch from that URL would install
    the very CPU wheel this exists to remove. Otherwise rebuild it the way setup.ps1 does
    when nothing is pinned: <mirror>/<tag>.
    """
    url = os.environ.get("UNSLOTH_TORCH_INSTALL_INDEX_URL", "").strip()
    if url:
        url = _trim_index_path_slashes(url)
        if _torch_index_leaf(url) == tag:
            return url
    pin = _explicit_torch_index_url()
    if pin is not None and _torch_index_leaf(pin) == tag:
        return pin
    return _pytorch_whl_leaf_url(tag)


def _explicit_cpu_torch_index_pin() -> bool:
    """Whether this run was pinned to a CPU wheel index, by URL or by family.

    Only a pin counts. setup.ps1's published tag also reads "cpu" for a host whose
    nvidia-smi probe simply returned nothing, and treating that as an instruction would
    let a wedged driver downgrade a healthy CUDA venv.
    """
    return _explicit_torch_index_family() == "cpu"


# Not a flavor tag (never == expected) and truthy (no `if not _now` branch swallows it).
_TORCH_TAG_UNIMPORTABLE = "unimportable"


def _installed_flavor_tag_now(expected: str = "") -> str:
    """The venv's CURRENT flavor tag, re-probed, or "" when nothing could be read.

    ``expected`` only matters for "cpu": _torch_flavor_tag reads every untagged version
    as cpu, so a private index serving an untagged CUDA or ROCm wheel would satisfy a
    CPU expectation here exactly as it did in the pre-repair comparison. The same
    runtime markers settle it, and the two comparisons have to agree or the repair is
    accepted on a rule the check that triggered it rejected.

    _torch_build_is_gpu answers a weaker question ("can this torch use a GPU at all")
    and is deliberately family-blind, so it cannot tell a repair that installed the
    requested family from one that left a different GPU wheel in place. "" is returned
    for an unreadable venv rather than "cpu", so ambiguity stays distinguishable from
    a positive CPU reading and never fails an update on its own.
    """
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if _ran and _importable and _version:
        tag = _torch_flavor_tag(_version)
        if expected == "cpu" and tag == "cpu" and (_hip or _cuda or _TORCH_RUNTIME_XPU):
            return _gpu_family_from_runtime_markers(_hip, _cuda)
        return tag
    if _ran:
        # The probe answered "does not import": version.py would still report the requested tag from a
        # half-written wheel, and the update would write a manifest over a torch nothing can import.
        return _TORCH_TAG_UNIMPORTABLE
    label = _installed_torch_label_on_disk()
    return _torch_flavor_tag(label) if label else ""


def _warn_repair_left_torch_unimportable(expected: str) -> bool:
    """Report a repair whose wheel cannot be imported, and fail. Always returns False.

    Distinct from _warn_wrong_flavor and _warn_still_cpu: the family on disk may well be
    the requested one, so naming it would send the reader after a wheel that is already
    correct. What is wrong is that it does not load.
    """
    _safe_print("")
    _safe_print(
        f"   [WARN] PyTorch was reinstalled for {expected} but the result cannot be imported."
    )
    _safe_print("   [WARN] The venv is not usable in this state.")
    _safe_print("   [WARN] Re-run this installer, or reinstall the build for your GPU manually.")
    _safe_print("   [WARN]     irm https://unsloth.ai/install.ps1 | iex")
    return False


def _warn_wrong_flavor(expected: str, installed: str) -> bool:
    """Report a repair that installed the wrong family, and fail. Always returns False.

    Distinct from _warn_still_cpu because "PyTorch is CPU-only" would be false here and
    would send the reader after the wrong problem: the venv holds a GPU build, just not
    the one this host asked for.
    """
    _safe_print("")
    _safe_print(
        f"   [WARN] PyTorch is a {installed} build but {expected} was expected for this machine."
    )
    _safe_print("   [WARN] The repair did not install the requested build.")
    _safe_print("   [WARN] Re-run this installer, or reinstall the build for your GPU manually.")
    _safe_print("   [WARN]     irm https://unsloth.ai/install.ps1 | iex")
    return False


def _warn_still_cpu(expected: str) -> bool:
    """Report a repair that did not take, and fail the install. Always returns False.

    install.ps1 warns here and exits 0. Failing instead is the entire point: a CPU-only
    torch on a host that expects a GPU build is precisely the state an update used to
    report as "dependencies up to date" while the app ran everything on CPU.
    """
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
    """Remove one distribution from the venv this script targets. True iff it is gone.
    Output is swallowed; the caller reports."""
    cmd = uninstall_command(
        name, use_uv = USE_UV and bool(shutil.which("uv")), uv_needs_system = UV_NEEDS_SYSTEM
    )
    _count_install_action()
    removed = subprocess.run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    return removed.returncode == 0


def _resident_xformers_build_torch() -> "str | None":
    """The torch build the installed xFormers extension was compiled against.

    Read from ``xformers/cpp_lib.json``, the same file install.ps1's resident probe
    reads. None when xFormers is absent or carries no build metadata. Never raises, and
    never imports xformers -- a mismatched _C.pyd logs its own warning on import, which
    would land in the middle of the installer's output.
    """
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


def _install_torchao_for_torch(torch_version: "str | None", default_index: bool = False) -> None:
    """Select the torchao matching torch_version and install it from its own index (PyPI's with
    default_index: download.pytorch.org's rocm leaves serve Linux only, so not Windows ROCm).

    Called twice: as step 4, and again after the Linux torch repair, which can move torch
    across families and releases underneath the first call.
    """
    spec = _select_torchao_spec(torch_version)
    # See _TORCHAO_DEFAULT_SPEC. Unlike torchcodec, rocm leaves do publish torchao.
    index = None if default_index else _torch_accelerator_index_url(torch_version)
    # --no-deps guards the second caller, which runs right after the torch repair.
    args = ["--no-deps", "--no-cache-dir"]
    needs_reinstall = _pin_needs_reinstall(spec, _torch_index_tag(torch_version) if index else "")
    if _may_skip_on_evidence() and not needs_reinstall:
        _note(f"torch {torch_version or 'unknown'} detected -- {spec} is already installed")
        _record_step("torchao", "skipped")
        return
    # Keyed off the pin alone: a matching build without evidence would otherwise re-download.
    if needs_reinstall:
        args.insert(0, "--force-reinstall")
    _record_step("torchao", "ran")
    _note(
        f"torch {torch_version or 'unknown'} detected -- installing {spec}"
        # Redacted for display only; the installer below still gets the exact URL.
        + (f" from {_strip_index_url_credentials(index)}" if index else "")
    )
    if default_index:
        # Optional on Windows ROCm: only export uses it.
        if not pip_install_try("Installing dependency overrides", *args, spec):
            _note(f"could not install {spec}; torchao export stays unavailable")
        return
    if not index:
        pip_install("Installing dependency overrides", *args, spec)
        return
    if pip_install_try("Installing dependency overrides", *args, "--index-url", index, spec):
        return
    # A leaf can lack this release (cu129 stops at 0.17.0), and the wrong build only costs kernels, so
    # retry unpinned, still fatally.
    _note(
        f"{_strip_index_url_credentials(index)} did not serve {spec} "
        "-- retrying from the default index; its kernels may be skipped"
    )
    pip_install("Installing dependency overrides", *args, spec)


def _resync_torch_coupled_packages(label_before: str) -> bool:
    """Re-settle the packages whose compiled extensions are tied to the torch build.

    Returns False when this pass left the venv in a state the caller must re-verify.

    torchao's cpp extensions are tied to the torch release AND its CUDA major, both of
    which _select_torchao_spec branches on; xFormers is tied to the exact (torch, CUDA)
    pair, and beside a pair it was not built for its ops vanish behind a log line rather
    than an error. --no-deps is the whole safety of the torchao call: torchao depends on
    torch, so resolving dependencies would pull PyPI's CPU wheel back in. Never fatal --
    both are secondary to the flavor repair that has just succeeded.
    """
    _label_after = str(_probe_installed_torch_version() or "")
    if not _label_after or _label_after == label_before:
        return True
    _touched_torch = False
    # The whole local tag: cpu -> xpu changes the build without moving any CUDA major.
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
            # Same pin as step 4, or PyPI's build lands over it; ==0.18.0 never equals 0.18.0+cu130.
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
                    _ao_ok = pip_install_try(
                        "Re-matching torchao to the repaired torch", *_ao_args, _spec
                    )
                if not _ao_ok:
                    # Across a CUDA-major move a torchao built for CUDA 12 cannot load under cu130, so remove it;
                    # a release-only move just keeps the slow-path warning.
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
    """Enforce that the venv still holds the torch flavor the install selected.

    `unsloth studio update` runs setup.ps1 and this script, never install.ps1, which held
    the only flavor repair; the dependency steps above resolve torch from PyPI, whose
    Windows wheel is 2.11.0+cpu. Returns False when the flavor is wrong and could not be
    repaired, which fails the install: that state used to be reported as success.

    ROCm is delegated to _ensure_rocm_torch -- AMD's Windows wheels live on a
    per-architecture repo.amd.com index a generic "rocm" tag cannot reconstruct.
    """
    if NO_TORCH:
        return True
    if _is_win_arm64_interpreter() and _explicit_torch_index_url() is None:
        _installed = _probe_installed_torch_version()
        if _installed and _is_cuda_family_leaf(_torch_flavor_tag(_installed)):
            return True
    # rocm/xpu/cpu fall through: an explicit GPU pin sets _TORCH_BACKEND.
    if _TORCH_BACKEND not in ("", "cuda", "rocm", "xpu", "cpu"):
        return True
    if expected is None:
        expected = _expected_torch_flavor_tag()
    # The PIN, not the handover: setup.ps1 also publishes "cpu" when nvidia-smi comes back empty.
    _cpu_pinned = expected == "cpu" and (
        _explicit_cpu_torch_index_pin()
        or (
            _explicit_torch_index_is_unusable()
            and _RECORDED_TORCH_TAG == "cpu"
            and bool(_RECORDED_TORCH_TAG_PINNED)
        )
    )
    if not (_is_cuda_family_leaf(expected) or expected in ("xpu", "rocm") or _cpu_pinned):
        return True
    if _TORCH_BACKEND in ("rocm", "xpu", "cpu") and _TORCH_BACKEND != expected:
        return True
    # A flavorless pin was applied verbatim; do not override an administrator's mirror.
    _unknown_pin = _explicit_unknown_family_torch_index_url()
    if _unknown_pin and _torch_index_leaf(_unknown_pin) != expected:
        return True
    # Only for a CUDA expectation inferred from hardware; an emptied mask does not cancel a stated one.
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
        return True
    if not installed_version:
        return True

    installed = _torch_flavor_tag(installed_version)
    # Untagged reads as "cpu", so a private untagged CUDA build compares equal under a /cpu pin.
    if expected == "cpu" and installed == "cpu" and (_hip or _cuda or _TORCH_RUNTIME_XPU):
        installed = _gpu_family_from_runtime_markers(_hip, _cuda)
    if installed == expected:
        return True

    # install.ps1's line, word for word.
    _safe_print(
        f"   PyTorch flavor mismatch (installed {installed}, need {expected}) -- "
        f"reinstalling correct build..."
    )
    if expected == "rocm":
        # AMD Windows wheels live on a per-arch repo.amd.com index the handed-over URL cannot name.
        _ensure_rocm_torch()
        # Check the family: a failed repo.amd.com fetch is non-fatal and leaves a cu124 wheel that looks GPU.
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
    if index_url is None:
        return _warn_wrong_flavor(expected, installed)
    _torch_pkg, _vision_pkg, _audio_pkg = (
        _XPU_TORCH_PKG_SPEC
        if expected == "xpu"
        else _cuda_repair_torch_specs(index_url, _TORCH_FLAVOR_REPAIR_PKG_SPEC)
    )
    # Keyed on the INTERPRETER: an emulated x64 venv installs win_amd64 wheels.
    _trio = [_torch_pkg, _vision_pkg, _audio_pkg]
    if _is_win_arm64_interpreter() and not _windows_arm64_has_torchaudio():
        _trio = [_torch_pkg, _vision_pkg]
    _label_before = str(installed_version)
    # --force-reinstall because pip_install may fall back to pip. constrain=False: constraints.txt
    # resolves against PyPI's torch, which is what put this venv here.
    pip_install(
        "PyTorch flavor repair",
        "--force-reinstall",
        "--no-cache-dir",
        *_trio,
        "--index-url",
        index_url,
        constrain = False,
    )

    # Check the family: a mirror can answer /cu128 with a cached cu124 wheel.
    _now = _installed_flavor_tag_now(expected)
    if _now == _TORCH_TAG_UNIMPORTABLE:
        return _warn_repair_left_torch_unimportable(expected)
    if _now == expected:
        if _resync_torch_coupled_packages(_label_before):
            return True
        _after = _installed_flavor_tag_now(expected)
        if _after == _TORCH_TAG_UNIMPORTABLE:
            return _warn_repair_left_torch_unimportable(expected)
        if _after in (expected, ""):
            return True
        _safe_print("   [WARN] the post-repair package resync changed the torch build.")
        return _warn_wrong_flavor(expected, _after)
    if not _now:
        if expected == "cpu":
            return True
        return True if _torch_build_is_gpu() else _warn_still_cpu(expected)
    if expected != "cpu" and not _torch_build_is_gpu():
        return _warn_still_cpu(expected)
    return _warn_wrong_flavor(expected, _now)


def _missing_torch_needs_dependency_pass() -> bool:
    """Missing torch that the dependency pass can actually reinstall, read from metadata.

    Gated on a live core requirement, so Apple Silicon (unsloth-zoo's torch marker is
    false there) does not run a useless pass on every update.
    """
    if NO_TORCH or _installed_distribution_version("torch") is not None:
        return False
    try:
        from importlib.metadata import PackageNotFoundError, requires
        from packaging.requirements import Requirement
    except ImportError:
        return False
    for package in _core_package_names(os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")):
        try:
            lines = requires(package) or []
        except PackageNotFoundError:
            continue
        for line in lines:
            try:
                req = Requirement(line)
            except Exception:  # noqa: BLE001 - ignore malformed requirements
                continue
            if req.name.lower() == "torch" and (
                req.marker is None or req.marker.evaluate({"extra": ""})
            ):
                return True
    return False


def _cuda_torch_needs_dependency_pass() -> bool:
    """Return True when only the dependency pass can put CUDA torch back on this host.

    The repair lives inside that pass, so an install whose GPU was hidden (or whose driver
    was broken) at install time keeps its CPU wheel on every "up to date" update until the
    package version happens to move. Answered by the repair in probe mode, so the two can
    never disagree; Windows heals this at setup.ps1's stale-venv check and macOS has no
    CUDA, and the repair already excludes both. Never installs, and fails closed.
    """
    try:
        return bool(_ensure_cuda_torch(probe_only = True))
    except Exception:  # noqa: BLE001 - a probe that cannot answer keeps the fast path
        return False


def _amd_torch_needs_dependency_pass() -> bool:
    """Return True when setup must run the dependency pass to repair non-ROCm torch.

    Scope is the wheel family, not the ROCm family: any ROCm marker keeps the fast path
    even when the repair would reroute it. Fails closed on an uncertain host or torch,
    and never installs.
    """
    if NO_TORCH or not IS_LINUX:
        return False
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    # install.sh's resolved backend is authoritative, exactly as it is for the repair.
    if _TORCH_BACKEND in ("cuda", "cpu", "xpu"):
        return False
    # A ROCm pin bypasses hardware detection; any other pin owns its repair path.
    if _explicit_rocm_torch_index_url() is None:
        if _explicit_torch_index_family() is not None:
            return False
        # Only outrank NVIDIA when there is a route, or setup.sh reruns the pass forever.
        if _has_usable_nvidia_gpu() and not (
            _rocm_torch_explicitly_requested() and _forced_rocm_route_is_viable()
        ):
            return False
        # Same reading as the routing guard, so the two cannot drift.
        if _visible_masks_select_no_gpu():
            return False
        _inferred_gfx = _infer_linux_amd_gfx_arch()
        _rocm_visible = _has_rocm_gpu()
        if not _rocm_visible and not _inferred_gfx:
            return False
        # _runtime_gfx_target composes both mask layers and returns no target only when the selection is
        # unreadable, so gate on it rather than on the host being mixed.
        _selected_gfx, _, _selected_spoof, _selected_host = _runtime_gfx_target(_inferred_gfx)
        if _selected_gfx is None:
            return False
        # A corroborated spoof needs _ensure_rocm_torch even when the wheel family is right.
        if _selected_spoof is not None:
            return True
        _pre_ran, _pre_imp, _pre_ver, _pre_hip, _pre_cuda = _probe_torch_runtime()
        _pre_torch = (_pre_ver or "").lower() if (_pre_ran and _pre_imp) else ""
        # The compatibility reroutes have no version floor; ask them before rejecting an unreadable version.
        if _pre_torch and _rocm_compat_reroute_pending(
            _selected_gfx, _detect_rocm_version() or (0, 0), _pre_torch
        ):
            return True
        _inferred_arm = (
            bool(_inferred_gfx)
            and bool((os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip() or not _rocm_visible)
            and _amd_arch_index_url(_inferred_gfx) is not None
        )
        # Per-arch repairs need no version: bundled-runtime hosts have no system ROCm to read.
        _family_repair_arm = _rocm_torch_family_needs_repair(
            _selected_gfx, _detect_rocm_version(), _selected_host
        )
        if not _inferred_arm and not _family_repair_arm:
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
    # Already ROCm is not the end: per-arch repairs (swapped card, gfx1103 on generic) must stay
    # reachable from `studio update`. Ask only what _ensure_rocm_torch asks, and not under a pin.
    _pin = _explicit_rocm_torch_index_url()
    if _pin is not None:
        # A pin still reinstalls when the installed family differs from it.
        return _rocm_pin_family_mismatch(_pin, _version.lower())
    _tail_gfx, _, _tail_spoof, _tail_host = _runtime_gfx_target(_infer_linux_amd_gfx_arch())
    _tail_ver = _detect_rocm_version() or (0, 0)
    # Only _ensure_rocm_torch clears a corroborated HSA spoof, even when the family matches.
    if _tail_spoof is not None:
        return True
    # Strix below the AMD floor and gfx906 above rocm6.3 also need the pass on update.
    if _rocm_compat_reroute_pending(_tail_gfx, _tail_ver, _version.lower()):
        return True
    return _rocm_torch_family_needs_repair(_tail_gfx, _detect_rocm_version(), _tail_host)


def _already_on_amd_arch_leaf(leaf: "str | None", installed_ver: str) -> bool:
    """True when the installed torch already IS the AMD per-arch build for ``leaf``.

    The family is the direct reading. A family that will not read back is not evidence of the
    wrong wheels, so the local tag is accepted too: only the AMD index ships a rocm tag at or
    above the arch floor, and re-downloading a multi-GB stack each update to re-establish what
    the tag already says is the cost this guard avoids.
    """
    if _torch_below_211(installed_ver):
        return False
    _family = _installed_rocm_wheel_family() if _torch_requires_rocm_sdk() else None
    if _family is not None:
        # A per-arch build for another arch sits at the same tag with none of this one's kernels.
        return _family == (leaf or "").lower()
    _tag = re.search(r"\+rocm(\d+)\.(\d+)", installed_ver or "")
    return bool(_tag) and (int(_tag.group(1)), int(_tag.group(2))) >= _ROCM_ARCH_INDEX_FLOOR


def _rocm_compat_reroute_pending(
    runtime_gfx: "str | None", ver: "tuple[int, int]", installed_ver: str
) -> bool:
    """Whether a compatibility reroute _ensure_rocm_torch performs has not been applied yet.

    Neither reroute is about missing kernels, so neither is visible to the wheel-family
    question: Strix / RDNA 4 want AMD's 7.13 build over any generic one below the floor, gfx906
    wants the last tag whose BLAS still carries it. Both compare against what is installed,
    so a host already on the right wheels keeps the fast path.
    """
    if not runtime_gfx:
        return False
    if runtime_gfx in _AMD_ARCH_INDEX_FLOOR_GFX and _strix_needs_amd_arch_index(ver):
        return not _already_on_amd_arch_leaf(_GFX_TO_AMD_INDEX_ARCH.get(runtime_gfx), installed_ver)
    if _runtime_target_is_gfx906() and _gfx906_needs_legacy_index(ver):
        return _GFX906_LEGACY_TAG not in installed_ver
    return False


def _installed_generic_rocm_tag() -> "tuple[int, int] | None":
    """(major, minor) of the ROCm tag the INSTALLED torch names, or None if it names none.

    Generic pytorch.org wheels carry it in the local version ("2.9.1+rocm6.3"). AMD per-arch
    builds are read by _installed_rocm_wheel_family instead, so their tag is not wanted here.
    """
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
    """Whether the installed ROCm torch carries no kernels for ``runtime_gfx``.

    Reads the same two signals _ensure_rocm_torch's arms read, in the same order, so the
    preflight cannot promise a repair the repair declines. A per-arch install names its
    family, and any family other than this target's is stale. A generic build names none (and
    _torch_requires_rocm_sdk rejects a stale `rocm` orphan beside one), so it is judged on
    whether the generic wheels carry kernels at all. An unknowable family answers False:
    leave the install alone rather than guess.
    """
    _owns_sdk = _torch_requires_rocm_sdk()
    _family = _installed_rocm_wheel_family() if _owns_sdk else None
    if _family is not None:
        _leaf = (_GFX_TO_AMD_INDEX_ARCH.get(runtime_gfx or "") or "").lower()
        if _family != _leaf:
            # Only when some index can serve the target on this host, matching _ensure_rocm_torch's decline.
            return _gfx_route_on_host(runtime_gfx, host_codes)
        # Below the 2.11 floor these leaves carry the _grouped_mm bug, so the shape alone is not enough.
        _ran, _importable, _ver, _hip, _cuda = _probe_torch_runtime()
        return _leaf in _ROCM_GFX_TORCH211_LEAVES and _torch_below_211(
            (_ver or "").lower() if (_ran and _importable) else ""
        )
    if _owns_sdk:
        # Family unreadable: _ensure_rocm_torch would reinstall every update, so leave it.
        return False
    # Use the installed wheel's own tag; ``ver`` is the host's and they can diverge. Check both the
    # reroute question and the tag floor (gfx950 on rocm6.3).
    _installed_tag = _installed_generic_rocm_tag() or ver
    return _generic_rocm_wheel_lacks_kernels(
        runtime_gfx, _installed_tag
    ) or _generic_only_target_below_floor(runtime_gfx, _installed_tag)


def _ensure_rocm_torch() -> "bool | None":
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch.

    On Linux x86_64: uses pytorch.org ROCm wheel index tags.
    On Windows: uses AMD's repo.amd.com arch-specific pip index.
    No-op on macOS, non-x86_64 Linux, NVIDIA-primary hosts, or when torch
    already links against HIP.
    Uses pip_install() to respect uv, constraints, and --python targeting.
    False: a Linux repair is required but the mirror cannot express its index; the caller
    aborts.
    """
    global _rocm_windows_torch_installed
    # install.sh's resolved backend is authoritative; avoids re-detecting in a different env.
    if _TORCH_BACKEND in ("cuda", "cpu", "xpu"):
        return
    # An explicit unknown-family pin was applied VERBATIM at install time; leave it alone.
    if _explicit_unknown_family_torch_index_url() is not None:
        return
    # An inexpressible pin stays authoritative: non-ROCm is not ours, ROCm is never re-probed.
    _pin_family = _explicit_torch_index_family() or ""
    _rocm_pin_unusable = _explicit_torch_index_is_unusable() and _is_pip_rocm_family_leaf(
        _pin_family
    )
    if _explicit_torch_index_is_unusable() and (IS_WINDOWS or not _rocm_pin_unusable):
        return
    # setup.ps1's marker; trust it only when torch imports as ROCm (a wiped venv leaves it stale).
    if os.environ.get("UNSLOTH_ROCM_TORCH_INSTALLED") == "1":
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_ok = _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
        if _torch_ok:
            _rocm_windows_torch_installed = True
            # bnb still needs the ROCm build (pre-release wheel, else PyPI >=0.50.0).
            _install_bnb_windows_rocm()
            return
    if IS_MACOS:
        return

    if IS_WINDOWS:
        # An explicit ROCm pin overrides the per-arch index: retry the PINNED one, not repo.amd.com.
        _win_rocm_pin = _explicit_rocm_torch_index_url()
        # UNSLOTH_FORCE_ROCM_TORCH is deliberately not read: install.ps1 / setup.ps1 would restore CUDA,
        # making this a multi-GB round trip.
        if _win_rocm_pin is None and _has_usable_nvidia_gpu():
            return
        gfx_arch = _detect_windows_gfx_arch()
        if not gfx_arch and _win_rocm_pin is None:
            return  # no AMD GPU visible via hipinfo
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_already_rocm = (
            _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
        )
        # Wheels are per family, so a host whose arch now resolves elsewhere must be repaired; act only on
        # a family read back positively. Multi-arch cards are decided by their device pack.
        if (
            _torch_already_rocm
            and _win_rocm_pin is None
            and _windows_routes_multiarch(gfx_arch)
            and not _multiarch_device_pack_installed(gfx_arch)
        ):
            _safe_print(
                f"   installed ROCm torch has no {gfx_arch} device pack -- reinstalling from "
                "AMD's multi-arch index"
            )
            _torch_already_rocm = False
        if (
            _torch_already_rocm
            and _win_rocm_pin is None
            and _windows_routes_multiarch(gfx_arch)
            and (_version or "").lower().rpartition("+")[2] in _ROCM_MULTIARCH_BROKEN_TAGS
        ):
            _safe_print(
                f"   installed ROCm torch {_version} cannot run fused attention -- reinstalling "
                f"{_ROCM_MULTIARCH_TORCH_VERSION}+{_ROCM_MULTIARCH_TAG}"
            )
            _torch_already_rocm = False
        # A multi-arch route is judged by its packs alone: a migrated venv keeps the orphaned family runtime.
        if (
            _torch_already_rocm
            and _win_rocm_pin is None
            and not _windows_routes_multiarch(gfx_arch)
        ):
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
            _torch_pkg, _vision_pkg, _audio_pkg = _windows_rocm_torch_pkg_specs_for(
                index_url, gfx_arch
            )
            _rocm_trio = [_torch_pkg, _vision_pkg, _audio_pkg]
            if _is_win_arm64_interpreter():
                _rocm_trio = [_torch_pkg, _vision_pkg]
            if _index_is_multiarch(index_url):
                # Not _bare_gfx(): a later local of that name shadows it (UnboundLocalError).
                _safe_print(
                    f"   {(gfx_arch or '').split(':')[0].lower()}: AMD's multi-arch index, pinned to "
                    f"{_ROCM_MULTIARCH_TORCH_VERSION}+{_ROCM_MULTIARCH_TAG} (torch, torchvision, torchaudio)"
                )
            # Nonfatal: --force-reinstall resolves before uninstalling, so a failed index keeps the build.
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
        # Always reinstall AMD Windows bnb so `studio update` repairs a broken one.
        if not _install_bnb_windows_rocm():
            _safe_print(
                "   Warning: AMD Windows bitsandbytes install failed "
                "(pre-release and PyPI); "
                "ROCm torch is installed but bitsandbytes may need manual install"
            )
        return

    # PyTorch ROCm wheels are not published for aarch64.
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return
    # An explicit ROCm pin commits to ROCm wheels whatever the visible GPU (headless / CI).
    _rocm_pin = _explicit_rocm_torch_index_url()
    _rocm_pinned = _rocm_pin is not None or _rocm_pin_unusable
    # Before any install path: a stale UNSLOTH_ROCM_GFX_ARCH on a real Van Gogh would otherwise cycle
    # ROCm -> CPU every update. An explicit index pin still wins.
    if not _rocm_pinned and not IS_WINDOWS and _miscomputing_arch_host():
        _safe_print(
            "   This host has an AMD arch measured to compute incorrectly under ROCm "
            "(studio/ROCM_RDNA2_APU.md) -- keeping CPU torch.\n"
        )
        return
    _inferred_linux_gfx = (
        _infer_linux_amd_gfx_arch() if (not _rocm_pinned and not IS_WINDOWS) else None
    )
    if not _rocm_pinned:
        # NVIDIA wins unless this run asked for ROCm; the presence test still applies.
        if _has_usable_nvidia_gpu() and (
            not _rocm_torch_explicitly_requested()
            or _explicit_cuda_torch_index_url() is not None
            or _explicit_cpu_torch_index_url() is not None
        ):
            return
        # rocminfo / amd-smi rows are authoritative; /opt/rocm or hipcc misses runtime-only installs.
        if not _has_rocm_gpu() and not _inferred_linux_gfx:
            return  # no AMD GPU visible
        # A stale UNSLOTH_ROCM_GFX_ARCH can pass the line above with no AMD card; require a viable route.
        if (
            _rocm_torch_explicitly_requested()
            and _has_usable_nvidia_gpu()
            and not _forced_rocm_route_is_viable()
        ):
            return

    ver = _detect_rocm_version()
    if ver is None:
        # Bundled-runtime hosts have no version to read but still need the missing-kernel route.
        _unknown_ver_gfx, _, _, _unknown_ver_host = _runtime_gfx_target(None)
        _uv_ran, _uv_imp, _uv_ver, _uv_hip, _uv_cuda = _probe_torch_runtime()
        _unknown_ver_torch = (_uv_ver or "").lower() if (_uv_ran and _uv_imp) else ""
        # A per-arch install can outlive its GPU; same reading as the setup preflight.
        if (
            not _rocm_pinned
            and not _inferred_linux_gfx
            and not _generic_rocm_wheel_lacks_kernels(_unknown_ver_gfx)
            and not _rocm_torch_family_needs_repair(_unknown_ver_gfx, None, _unknown_ver_host)
            # The Strix reroute has no version floor; with no tag the per-arch index is its only route.
            and not _rocm_compat_reroute_pending(_unknown_ver_gfx, (0, 0), _unknown_ver_torch)
        ):
            _safe_print("   ROCm detected but version unreadable -- skipping torch reinstall")
            return
        ver = (0, 0)

    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    _installed_torch_ver = (_version or "").lower() if (_ran and _importable) else ""
    _hip_marker = ""
    if _ran and _importable:
        _hip_marker = _hip if _hip else ("rocm" if "rocm" in _installed_torch_ver else "")
    has_hip_torch = _hip_marker != ""

    # A ROCm pin of another family reinstalls; a same-tag per-arch switch is undetectable.
    _rocm_pin_mismatch = (
        _rocm_pin_family_mismatch(_rocm_pin or _pin_family, _installed_torch_ver)
        if (has_hip_torch and _rocm_pinned)
        else False
    )

    rocm_torch_ready = has_hip_torch and not _rocm_pin_mismatch

    # Inferred-gfx path: only when the runtime enumerates no GPU, so a mixed Strix APU + dGPU box does
    # not get APU wheels. An explicit UNSLOTH_ROCM_GFX_ARCH is exempt (mirrors install.sh).
    _gfx_override_env = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
    _inferred_arch_installed = False
    if (
        _inferred_linux_gfx
        and not has_hip_torch
        and not _rocm_pinned
        and (_gfx_override_env or not _has_rocm_gpu())
        # Ask the mask layers too, or an ordinal the guess cannot explain reaches a declined wheel.
        and (_gfx_override_env or _runtime_gfx_target(_inferred_linux_gfx)[0] is not None)
    ):
        index_url = _amd_arch_index_url(_inferred_linux_gfx)
        if index_url is not None:
            # Use the bare arch: the suffixed form misses the specs table and _hsa_spoof_contradicts.
            _bare_gfx = (_inferred_linux_gfx or "").strip().lower().split(":")[0]
            # Same bounded companion tuple as the reroute path, so the two cannot drift.
            _torch_pkg, _vision_pkg, _audio_pkg = _WINDOWS_ROCM_TORCH_PKG_SPECS.get(
                _bare_gfx, _ROCM_ARCH_INDEX_TORCH_PKG_SPEC
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
            # The wheels carry only _inferred_linux_gfx code, so clear a spoof naming another arch.
            if _hsa_spoof_contradicts(_bare_gfx):
                _clear_confirmed_hsa_spoof(_bare_gfx)

    # UNSLOTH_ROCM_GFX_ARCH=gfx906 must beat the Strix probe-order detection below.
    _gfx906_arch_override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(
        ":"
    )[0] == "gfx906"

    # Strix (gfx1150/1151) needs AMD's per-gfx index (2.11+rocm7.13): generic wheels segfault in
    # _grouped_mm, see _strix_needs_amd_arch_index. The second reroute is for arches with no kernels.
    _arch_index_url: "str | None" = None
    _arch_index_pkgs: "tuple[str, str, str] | None" = None
    # Skipped after the inferred-arch install, which resolves the same index.
    if not _rocm_pinned and not _gfx906_arch_override and not _inferred_arch_installed:
        _runtime_gfx, gfx_codes, _physical_gfx, _host_codes = _runtime_gfx_target(
            _inferred_linux_gfx
        )
        # A miscomputing target has no ROCm route; keyed on the SELECTED target.
        if _runtime_gfx in _ROCM_MISCOMPUTING_GFX:
            # Only declines to install; removing existing ROCm is _ensure_cpu_torch's whole-host call.
            _safe_print(
                f"   {_runtime_gfx} computes incorrect results under ROCm "
                f"(studio/ROCM_RDNA2_APU.md) -- not installing ROCm torch for it.\n"
            )
            return
        _strix_gfx = _AMD_ARCH_INDEX_FLOOR_GFX
        _detected_strix = (
            _strix_gfx.intersection(gfx_codes) if _strix_needs_amd_arch_index(ver) else set()
        )
        if _detected_strix:
            if _runtime_gfx in _strix_gfx and _already_on_amd_arch_leaf(
                _GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx), _installed_torch_ver
            ):
                # Already this build: _ensure_rocm_torch runs twice per install, so do not re-download.
                _safe_print(
                    f"   torch already runs on the AMD {_runtime_gfx} wheels; keeping it.\n"
                )
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_runtime_gfx)
            elif _runtime_gfx in _strix_gfx:
                _selected_gfx = _runtime_gfx
                _arch_index_url = _amd_arch_index_url(_selected_gfx)
                _arch_index_pkgs = (
                    "torch>=2.11.0,<2.12.0",
                    "torchvision>=0.26.0,<0.27.0",
                    "torchaudio>=2.11.0,<2.12.0",
                )
                _safe_print(
                    f"   {_selected_gfx} is the runtime target with ROCm "
                    f"{ver[0]}.{ver[1]}.\n"
                    f"   Routing torch install to AMD's arch-specific index\n"
                    f"   ({_strip_index_url_credentials(_arch_index_url)}) which serves torch\n"
                    f"   2.11.0+rocm7.13.0 with AMD's fixes for this GPU (the generic pytorch.org\n"
                    f"   wheels below 7.13 lack them).\n"
                )
                # Only here: these wheels carry _selected_gfx kernels, so the spoof must be cleared. Generic paths
                # keep the override as their only source of usable kernels.
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_selected_gfx)
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                _safe_print(
                    f"   AMD per-gfx GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects another runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

        # No ROCm-version floor: no generic wheel carries kernels for these (see _GENERIC_ROCM_WHEEL_GFX).
        if _arch_index_url is None:
            # Judge by the installed generic wheel's own tag; ``ver`` (the host's) still picks a reinstall tag.
            _kernel_ver = (
                has_hip_torch and not _torch_requires_rocm_sdk() and _installed_generic_rocm_tag()
            ) or ver
            _missing_kernels = {
                g for g in gfx_codes if _generic_rocm_wheel_lacks_kernels(g, _kernel_ver)
            }
            # A sub-2.11 build of a floor leaf is broken even when the generic wheel lists the GPU.
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
                # Skip the multi-GB reinstall when already on this family (read back positively) at or above the
                # 2.11 floor. gfx1152 with an unreadable ROCm version only meets this floor here.
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
                    # Keeping the wheels still needs the spoof cleared: they carry only the physical arch.
                    if _physical_gfx is not None:
                        _clear_confirmed_hsa_spoof(_runtime_gfx)
                elif _leaf is not None:
                    _arch_index_url = _amd_arch_index_url(_runtime_gfx)
                    # Bound companions but keep older per-arch builds, except on _grouped_mm-bug leaves.
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
                # Already on the target's own family, yet a spoof may still be exported; clear it here too.
                _clear_confirmed_hsa_spoof(_runtime_gfx)

        # A per-arch install can outlive its GPU, and torch.version.hip would keep the fallback away.
        if _arch_index_url is None and rocm_torch_ready and _runtime_gfx is not None:
            _have = _installed_rocm_wheel_family() if _torch_requires_rocm_sdk() else None
            _want = (_GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx) or "").lower()
            # Demote only when this host has somewhere to go: gfx1010 has no index, and a mask-selected gfx906
            # beside another card has no route (_MIXED_HOST_UNROUTABLE).
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
                # With an unreadable ROCm version the generic fallback resolves no tag, so use the AMD index.
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
                    # Datacentre parts (gfx942, gfx950) live only on the generic index; with no tag or a too-old one,
                    # take the newest generic index known.
                    ver = max(_ROCM_TORCH_INDEX)
            elif _have is None and _generic_only_target_below_floor(
                _runtime_gfx, _installed_generic_rocm_tag() or ver
            ):
                # A generic build whose tag predates the target: demoting is the whole repair.
                _safe_print(
                    f"   installed ROCm torch is a generic build whose own tag predates\n"
                    f"   {_runtime_gfx} -- reinstalling from an index that carries it.\n"
                )
                rocm_torch_ready = False

        # Fresh / CPU / CUDA installs also need the per-arch tag floor.
        if (
            _arch_index_url is None
            and not rocm_torch_ready
            and _runtime_gfx in _GENERIC_ROCM_WHEEL_GFX
            and _generic_tag_lacks_kernels(_runtime_gfx, ver)
        ):
            ver = max(_ROCM_TORCH_INDEX)

    # gfx906 detection for the bnb skip holds even under a torch-index pin, so a source-built bnb is not
    # overwritten; the pin only suppresses the torch reroute.
    _runtime_is_gfx906 = _runtime_target_is_gfx906()
    # Reroute to rocm6.3 only when the host would pick a newer index, never over a pin or Strix reroute.
    _gfx906_override = (
        _runtime_is_gfx906
        and _gfx906_needs_legacy_index(ver)
        and not _rocm_pinned
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

    # Fire even with HIP torch: torch.version.hip == "7.1" is the broken combo they repair.
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
    # gfx906 fires even with HIP torch: a +rocm7.x build is the broken combo.
    elif _gfx906_override and _GFX906_LEGACY_TAG not in _installed_torch_ver:
        index_url = _pytorch_whl_leaf_url(_GFX906_LEGACY_TAG)
        if index_url is None:
            return False
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
        # Gate on rocm_torch_ready so the generic path does not overwrite a fresh inferred-gfx install.
        # Honour a ROCm pin verbatim; else the newest tag <= host.
        _override_idx = _explicit_rocm_torch_index_url()
        if _override_idx is not None:
            index_url = _override_idx
            tag = _torch_index_leaf(index_url)
        elif _rocm_pin_unusable:
            tag = _pin_family
        else:
            tag = _generic_pytorch_rocm_tag(ver)
        if tag is None:
            _safe_print(
                f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- skipping torch reinstall"
            )
        else:
            if _override_idx is None:
                index_url = _pytorch_whl_leaf_url(tag)
                if index_url is None:
                    return False
            _safe_print(
                f"   ROCm torch -- installing from {_strip_index_url_credentials(index_url)}"
            )
            # Only _grouped_mm-bug arches take the 2.11 spec (matches install.ps1 / setup.ps1).
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

    # gfx906 has no prebuilt bnb kernels; reinstalling would clobber a source-built bnb every update.
    if rocm_torch_ready and _runtime_is_gfx906:
        _safe_print(
            _dim(
                "   gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels). "
                "Build bitsandbytes from source for 4-bit QLoRA -- "
                "see docs.unsloth.ai/get-started/install-and-update/amd."
            )
        )
        # Drop a generic bnb the base install just pulled in; keep a pre-existing source build.
        if _GFX906_BNB_ABSENT_BEFORE_BASE and _bitsandbytes_installed():
            _safe_print(_dim("   gfx906: removing generic bitsandbytes pulled in as a dependency"))
            _count_install_action()
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"],
                capture_output = True,
            )
    # The pre-release bnb wheel needs pip, not uv.
    elif rocm_torch_ready:
        _bnb_url = _bnb_rocm_prerelease_url()
        # --force-reinstall from a URL fetches 43 MB before deciding, twice per pass.
        if _bnb_rocm_install_is_current(_bnb_url):
            _safe_print(_dim("   bitsandbytes (AMD) is already this build -- keeping it"))
            # Record on skip too, or the next update refetches.
            _record_bnb_rocm_provenance()
        else:
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
            _record_bnb_rocm_provenance()
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

# Read at import: install_python_stack() drops the manifest before its dependency pass.
_RECORDED_TORCH_TAG = install_manifest.recorded_torch_flavor()
# setup.ps1 publishes an automatic /cpu the same way as a pinned one.
_RECORDED_TORCH_TAG_PINNED = install_manifest.recorded_torch_flavor_was_pinned()

# Set by install.sh; empty means standalone `studio update`, which re-detects.
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
        # Without this an authoritative XPU pin is never acted on, see _ensure_xpu_torch.
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
    """Return a progress label like 'torch check (cuda)' using the known backend.

    Falls back to GPU detection when UNSLOTH_TORCH_BACKEND is not set (e.g.
    standalone `unsloth studio update` runs that bypass install.sh).
    """
    backend = _TORCH_BACKEND
    if not backend:
        if _has_usable_nvidia_gpu():
            backend = "cuda"
        # rocminfo / amd-smi ship with the HIP SDK, not AMD's bundled-runtime wheels.
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

# Apple Silicon override of mlx-vlm/mlx-lm's transformers pin; uv truncates UV_OVERRIDE at a space.
_MLX_OVERRIDES = SINGLE_ENV / "overrides-darwin-arm64.txt"
if IS_MAC_ARM and _MLX_OVERRIDES.is_file() and "UV_OVERRIDE" not in os.environ:
    os.environ["UV_OVERRIDE"] = _uv_safe_path(_MLX_OVERRIDES)

# Legacy code-page consoles (CP1252): _safe_print() degrades glyphs to ASCII.

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
        text = " ".join(str(a) for a in args)
        for uni, ascii_alt in _UNICODE_TO_ASCII.items():
            text = text.replace(uni, ascii_alt)
        print(
            text.encode(sys.stdout.encoding or "ascii", errors = "replace").decode(
                sys.stdout.encoding or "ascii", errors = "replace"
            ),
            **kwargs,
        )


# Same as startup_banner: NO_COLOR disables, FORCE_COLOR or TTY enables.


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


# Matches setup.sh step(): 2-space indent, 15-char dim label, then value.
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
    # A detached or closed stdout must not take down a message bound for stderr.
    except (AttributeError, OSError, ValueError):
        pass
    _PROGRESS_LINE_ACTIVE = False


def _note(message: str, color_fn = None) -> None:
    """Print a detail line under the current step, aligned to the value column."""
    if color_fn is None:
        color_fn = _dim
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


_ENV_FOR_CMD = object()


def run(
    label: str,
    cmd: list[str],
    *,
    quiet: bool = True,
    check: bool = True,
    env: "dict[str, str] | None | object" = _ENV_FOR_CMD,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit, unless ``check`` is False.

    ``env`` is for a caller that already derived argv from the environment: see
    _pinned_cmd_and_env.
    """
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
        env = _install_env_for_cmd(cmd) if env is _ENV_FOR_CMD else env,
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        if not check:
            return result
        _report_failed_command(label, result)
    return result


# First line of the overrides file install.ps1 generates; setup.ps1's merge copies it first.
WOA_OVERRIDES_HEADER = "# Generated by install.ps1 for Windows on ARM"


def _woa_overrides_are_load_bearing() -> bool:
    """Is this a native win_arm64 resolve whose correctness depends on UV_OVERRIDE?

    install.ps1 writes those overrides to lift the released torch cap -- no win_arm64 CUDA
    wheel satisfies it -- and to drop the packages with no win_arm64 build. pip has no
    override mechanism at all: constraints only narrow a requirement, they cannot replace
    one, so there is nothing to translate them into. Falling back to pip on this stack does
    not recover, it silently resolves the wrong thing.

    Judged by the generated file, not by the variable: a caller's own override on a run that
    never configured the stack (--no-torch, a direct run) keeps the fallback every host has.
    """
    if not _is_win_arm64_interpreter():
        return False
    for path in os.environ.get("UV_OVERRIDE", "").split():
        try:
            with open(path, encoding = "utf-8", errors = "replace") as handle:
                first = handle.readline()
        except OSError:
            continue
        if first.lstrip("\ufeff").startswith(WOA_OVERRIDES_HEADER):
            return True
    return False


def _report_failed_command(label: str, result: subprocess.CompletedProcess[bytes]) -> None:
    """Print a failed command's redacted output and exit with its code."""
    _step("error", f"{label} failed (exit code {result.returncode})", _red)
    if result.stdout:
        # Redact: a pinned --index-url may carry credentials.
        _safe_print(_redact_install_output(result.stdout))
    sys.exit(result.returncode)


# pip will not replace a dist-info without RECORD (left by an interrupted or uv-written install),
# and the reused venv would fail every later install of it.
_NO_RECORD_MARKER = "no RECORD file was found"
_CANNOT_UNINSTALL_RE = re.compile(r"Cannot uninstall ([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _purge_recordless_distributions(output: "bytes | str | None") -> list[str]:
    """Delete the .dist-info directories pip named, where they carry no RECORD.

    pip's own hint here is ``--ignore-installed``, which would apply to every
    package in the command and so silently skip real upgrades too. Dropping just
    the metadata that blocked it keeps the rest of the environment visible to the
    retry, and removes nothing that pip itself considers a tracked install.
    """
    if not output:
        return []
    text = output.decode("utf-8", "replace") if isinstance(output, bytes) else output
    # Require both halves: the marker alone appears in unrelated advice.
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
        # Counted here, not at entry, so the no-purge case keeps the cached constraint answer.
        _count_install_action()
        cleared.append(dist_info.name)
    return cleared


WINDOWS_SKIP_PACKAGES = {"triton_kernels"}

# No win_arm64 wheel or MSVC-free sdist; all optional. Lowercase only: _filter_requirements compares
# lowercased lines.
WINDOWS_ARM64_SKIP_PACKAGES = {
    "mecab",
    "sqlite-vec",
    "tiktoken",
    "tensorboard",
    "librosa",
    "openai-whisper",
    "torch-c-dlpack-ext",
    "pytorch_tokenizers",
    "hf_transfer",
    "xformers",
}


def _wheel_matches_interpreter(filename: str) -> bool:
    """Can THIS interpreter install the wheel named ``filename``?

    A wheelhouse is not built for one interpreter: install.ps1 stages every win_arm64
    wheel it finds, cp311 through cp314, as the wheelhouse published them. So a filename
    is not proof on its own -- a cp311 tiktoken is invisible to a cp313 resolver, and
    counting it as available drops the skip and sends the resolve to an sdist that cannot
    build here. PEP 425: a wheel is installable when one of its (python, abi, platform)
    triples is one the interpreter supports, and each filename field may be a
    "."-separated set expanded as their cartesian product. Unparseable means not
    installable, which only leaves the conservative skip in place.
    """
    stem = filename[:-4] if filename.lower().endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 5:
        return False
    py_tags, abi_tags, plat_tags = (set(field.split(".")) for field in parts[-3:])
    this_platform = (sysconfig.get_platform() or "").replace("-", "_").replace(".", "_").lower()
    if "any" not in plat_tags and this_platform not in plat_tags:
        return False
    major, minor = sys.version_info[:2]
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    this_cpython = f"cp{major}{minor}"
    this_abi = f"{this_cpython}t" if free_threaded else this_cpython
    for py_tag in py_tags:
        for abi_tag in abi_tags:
            # abi3 excluded HERE too: this branch shadows the one below (CPython #111506).
            if py_tag == this_cpython and (
                abi_tag in ("none", this_abi) or (abi_tag == "abi3" and not free_threaded)
            ):
                return True
            if abi_tag == "none":
                pure = re.fullmatch(r"py(\d)(\d*)", py_tag)
                if pure and int(pure.group(1)) == major and int(pure.group(2) or 0) <= minor:
                    return True
            elif abi_tag == "abi3" and not free_threaded:
                stable = re.fullmatch(r"cp(\d)(\d+)", py_tag)
                if stable and int(stable.group(1)) == major and 2 <= int(stable.group(2)) <= minor:
                    return True
    return False


@functools.lru_cache(maxsize = 1)
def _find_links_wheel_versions() -> "dict[str, frozenset[str]]":
    """Canonical name -> versions of the wheels in the configured find-links directories.

    install.ps1 points UV_FIND_LINKS / PIP_FIND_LINKS at a local wheelhouse on Windows on
    ARM, holding the wheels PyPI does not publish for win_arm64. Anything served there is
    installable, so it must not also be filtered out as unavailable -- but only the wheels
    tagged for this interpreter are, which is what the resolver will agree to.

    The VERSIONS come back too, not just the names: every requirement these gate is
    ``==``-pinned, and a wheelhouse holding tiktoken 0.12.0 against a ``tiktoken==0.13.0``
    line satisfies nothing -- the resolver goes to PyPI, finds no win_arm64 wheel for the
    pinned version, and falls to an sdist that cannot build here. That is the exact
    failure the skip list exists to prevent, so the caller checks the pin.
    """
    versions: "dict[str, set[str]]" = {}
    # UV_FIND_LINKS ONLY: uv does not consume PIP_FIND_LINKS. Comma-split, the way uv reads it.
    for value, separator in ((os.environ.get("UV_FIND_LINKS"), ","),):
        for entry in re.split(separator, value or ""):
            entry = entry.strip().strip('"')
            if not entry or "://" in entry:
                continue  # a URL index cannot be listed cheaply; treat it as unknown
            try:
                for wheel in Path(entry).glob("*.whl"):
                    if not _wheel_matches_interpreter(wheel.name):
                        continue
                    fields = wheel.name[:-4].split("-")
                    if len(fields) < 5:
                        continue  # not a wheel filename; _wheel_matches_interpreter agrees
                    name = _canonical_dist_name(fields[0])
                    versions.setdefault(name, set()).add(fields[1])
            except OSError:
                continue
    return {name: frozenset(vers) for name, vers in versions.items()}


def _find_links_wheel_names() -> frozenset[str]:
    """Just the names from :func:`_find_links_wheel_versions`."""
    return frozenset(_find_links_wheel_versions())


_find_links_wheel_names.cache_clear = _find_links_wheel_versions.cache_clear  # type: ignore[attr-defined]


def _parse_release(version: str) -> "tuple[int, ...] | None":
    """The numeric release segment of a PEP 440 version, or None when it has none."""
    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)", version or "")
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_satisfies(version: str, specifier: str) -> "bool | None":
    """Does ``version`` satisfy the PEP 440 specifier set ``specifier``?

    None means "cannot tell" -- an epoch, an arbitrary-equality clause, anything this
    deliberately small comparison does not model. The caller treats that as it treated
    every version before this existed, so an exotic pin is no worse off than it was.
    """
    specifier = (specifier or "").strip()
    if not specifier:
        return True
    for module_name in ("packaging.specifiers", "pip._vendor.packaging.specifiers"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        try:
            spec_set = module.SpecifierSet(specifier)
            # Prereleases off unless the specifier names one; packaging's default admits 0.13.0rc1.
            return bool(spec_set.contains(version, prereleases = bool(spec_set.prereleases)))
        except Exception:
            break
    if not re.fullmatch(r"\s*v?\d+(?:\.\d+)*\s*", version or ""):
        return False
    got = _parse_release(version)
    # "!" that is not part of "!=" is a PEP 440 epoch, which _parse_release does not model.
    if got is None or "!" in (version or "") or "!" in specifier.replace("!=", ""):
        return None
    for clause in specifier.split(","):
        clause = clause.strip()
        if not clause:
            continue
        match = re.fullmatch(r"(==|!=|>=|<=|~=|>|<)\s*([^\s]+)", clause)
        if not match:
            return None
        op, want_raw = match.group(1), match.group(2).rstrip(".*")
        wildcard = match.group(2).endswith(".*")
        want = _parse_release(want_raw)
        if want is None:
            return None
        width = max(len(got), len(want))
        lhs = got + (0,) * (width - len(got))
        rhs = want + (0,) * (width - len(want))
        if wildcard:
            # ==1.2.* / !=1.2.*: only the prefix is compared.
            prefix = got[: len(want)] + (0,) * max(0, len(want) - len(got))
            ok = prefix == want
            if op == "==":
                pass
            elif op == "!=":
                ok = not ok
            else:
                return None
        elif op == "==":
            ok = lhs == rhs
        elif op == "!=":
            ok = lhs != rhs
        elif op == ">=":
            ok = lhs >= rhs
        elif op == "<=":
            ok = lhs <= rhs
        elif op == ">":
            ok = lhs > rhs
        elif op == "<":
            ok = lhs < rhs
        else:  # ~=X.Y[.Z] is ">=X.Y[.Z], ==X.Y.*" one level up
            if len(want) < 2:
                return None
            ok = lhs >= rhs and got[: len(want) - 1] == want[: len(want) - 1]
        if not ok:
            return False
    return True


def _marker_is_active(marker: str) -> "bool | None":
    """Does ``marker`` hold for THIS interpreter? None when it cannot be decided.

    packaging is what pip and uv evaluate markers with, so borrowing it keeps the answer
    identical to the resolver's; pip vendors a copy, which is the fallback for a venv
    where packaging is not installed in its own right. Neither available means the
    caller keeps every clause instead of picking one.
    """
    if not marker:
        return True
    for module_name in ("packaging.markers", "pip._vendor.packaging.markers"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        try:
            return bool(module.Marker(marker).evaluate())
        except Exception:
            return None
    return None


def _requirement_pins(req: "Path | None") -> "dict[str, list[str]]":
    """Canonical name -> the version specifiers a requirements file states for it.

    A name can appear more than once, split by marker -- extras.txt carries
    ``MeCab==0.996.13`` and ``MeCab==0.996.5`` on complementary markers -- so keeping one
    specifier per name let the row for another platform overwrite the row that actually
    applies. Markers are evaluated for this interpreter and inactive rows dropped;
    when they cannot be evaluated every clause is kept, and the caller takes any of them
    as satisfied, which is no stricter than the name-only check that came before.
    """
    pins: "dict[str, list[str]]" = {}
    if req is None:
        return pins
    try:
        text = req.read_text(encoding = "utf-8-sig")
    except OSError:
        return pins
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "@" in line:
            continue
        line, _, marker = line.partition(";")
        if _marker_is_active(marker.strip()) is False:
            continue
        line = line.strip()
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$", line)
        if not match:
            continue
        pins.setdefault(_canonical_dist_name(match.group(1)), []).append(match.group(2).strip())
    return pins


# Skipped for their dependencies: whisper needs tiktoken unconditionally, so filtering the direct line is not enough.
WINDOWS_ARM64_SKIP_UNBLOCKED_BY = {
    "tensorboard": ("grpcio",),
    # soxr too: librosa 0.11.0 requires it and soxr has no win_arm64 wheel.
    "librosa": ("llvmlite", "numba", "soxr"),
    "openai_whisper": ("llvmlite", "numba", "tiktoken"),
}


# Blocker floors from the packages' own metadata: {blocker: (specifier, read from, its pinned version)}.
WINDOWS_ARM64_BLOCKER_FLOORS: "dict[str, tuple[str, str, str]]" = {
    "grpcio": (">=1.74.0", "tensorboard", "2.21.0"),
    "numba": (">=0.51.0", "librosa", "0.11.0"),
    "soxr": (">=0.3.2", "librosa", "0.11.0"),
}


# Excluded on win_arm64 by marker; floors given because --no-deps takes whatever is hosted.
WINDOWS_ARM64_WHEELHOUSE_OPTIONALS = {
    "hf-transfer": "",
    "xformers": ">=0.0.22.post7",
    "sqlite-vec": "",
}


def _wheelhouse_hosts(name: str) -> bool:
    """Does the resolver's own find-links carry a wheel for this distribution?"""
    return bool(_find_links_wheel_versions().get(_canonical_dist_name(name)))


def _wheelhouse_best_version(name: str, floor: str) -> "str | None":
    """The newest hosted version that clears ``floor``, or None if none does.

    An unreadable comparison keeps the answer it would have had before this floor
    existed, matching _windows_arm64_skip_packages: an exotic version is no worse off.
    """
    hosted = _find_links_wheel_versions().get(_canonical_dist_name(name)) or ()
    usable = [
        version
        for version in hosted
        if not floor or _version_satisfies(version, floor) is not False
    ]
    if not usable:
        return None
    try:
        from packaging.version import Version
        return max(usable, key = Version)
    except Exception:
        return sorted(usable)[-1]


def _wheelhouse_torchcodec_version(torch_version: "str | None") -> "str | None":
    """The hosted torchcodec inside the window this torch selects, or None.

    win_arm64 only: elsewhere the platform's own wheel decides. An unknown torch has no
    window, so it gets no answer either.
    """
    if not _is_win_arm64_interpreter() or not torch_version:
        return None
    spec = _select_torchcodec_spec(torch_version)
    if spec is None:
        return None
    return _wheelhouse_best_version("torchcodec", spec.split("torchcodec", 1)[1])


def _install_wheelhouse_optionals() -> None:
    """Install the hosted optionals the metadata cannot ask for. Best effort.

    --no-deps: the graph is already resolved and installed by the time this runs, and
    xformers names torch, so resolving here could walk the win_arm64 CUDA build off to
    whatever PyPI offers. A failure leaves the feature off, which is where it was.

    Pinned to the selected version rather than installed by bare name, so the floor
    checked here is the version that actually lands.
    """
    if not _is_win_arm64_interpreter():
        return
    for name, floor in WINDOWS_ARM64_WHEELHOUSE_OPTIONALS.items():
        version = _wheelhouse_best_version(name, floor)
        if version is None:
            if _wheelhouse_hosts(name):
                _note(f"windows on arm: the wheelhouse {name} is below {floor}; leaving it off")
            if _canonical_dist_name(name) == "xformers":
                _evict_xformers_built_for_another_torch()
            continue
        installed = pip_install_try(
            f"Installing {name}=={version} from the Windows on ARM wheelhouse",
            "--no-deps",
            "--no-cache-dir",
            f"{name}=={version}",
            constrain = False,
        )
        if not installed:
            _note(f"windows on arm: could not install the wheelhouse {name}; feature stays off")
        # Checked even when the refresh failed: an earlier torch's copy is still resident.
        if _canonical_dist_name(name) == "xformers" and _evict_xformers_built_for_another_torch():
            continue
        if not installed:
            continue
        _note(f"windows on arm: installed {name}=={version} from the wheelhouse")


def _torch_build_family(label: str) -> str:
    """cuda<major> / rocm / xpu / cpu from a torch.__version__ local tag, "" when it names none."""
    tag = label.partition("+")[2].strip().lower()
    if tag.startswith("cu"):
        major = _cuda_major_from_torch_version(label)
        return f"cuda{major}" if major else ""
    for prefix in ("rocm", "xpu", "cpu"):
        if tag.startswith(prefix):
            return prefix
    return ""


def _evict_xformers_built_for_another_torch(
    scope: str = "windows on arm", family_only: bool = False
) -> bool:
    """Remove a resident xFormers whose extension was built against another torch. True iff removed.

    xFormers links its extension against ONE (torch, CUDA) pair; beside any other it is mute,
    and a package install never uninstalls what an earlier run left behind. family_only keeps one
    of the same family and CUDA major (stable ABI since 0.0.34; _C.so links libcudart.so.<major>).
    """
    built_for = _resident_xformers_build_torch()
    resident = str(_probe_installed_torch_version() or "")
    if not (built_for and resident and built_for != resident):
        return False
    if family_only:
        built_family, resident_family = (
            _torch_build_family(built_for),
            _torch_build_family(resident),
        )
        if not (built_family and resident_family and built_family != resident_family):
            return False
    if not _uninstall_distribution("xformers"):
        _safe_print(
            f"   [WARN] {scope}: xFormers was built for torch {built_for}, not {resident}, "
            "and could not be removed; its compiled operations stay unavailable."
        )
        return False
    _note(
        f"{scope}: xFormers was built for torch "
        f"{built_for}, not {resident} -- removed; attention uses torch SDPA"
    )
    return True


def _evict_xformers_requiring_another_torch() -> bool:
    """Remove an xFormers whose torch requirement is unmet, even if torch is unchanged (--overrides, #11545)."""
    mismatch = xformers_torch_requirement_unmet()
    if mismatch is None:
        return False
    xformers_version, requirement, torch_version = mismatch
    if not _uninstall_distribution("xformers"):
        _safe_print(
            f"   [WARN] xformers {xformers_version} requires torch{requirement}, not "
            f"{torch_version}, and could not be removed; diffusers cannot import it."
        )
        return False
    _note(
        f"xformers {xformers_version} requires torch{requirement}, not {torch_version} "
        "-- removed; attention uses torch SDPA"
    )
    return True


WINDOWS_ARM64_PUBLIC_INDEX_WHEELS: "dict[str, dict[str, str]]" = {
    "llvmlite": {"cp314": "0.49.0"},
    "numba": {"cp314": "0.67.0"},
}


def _uv_config_files() -> "list[tuple[Path, str]]":
    """The persistent configuration uv would discover, most specific first, as (path, table).

    uv reads uv.toml or pyproject.toml [tool.uv] from the current directory or the nearest
    parent (uv.toml wins over pyproject.toml in the same directory), then the user file
    (%APPDATA%\\uv\\uv.toml on Windows, $XDG_CONFIG_HOME/uv/uv.toml elsewhere), then the
    system file (%PROGRAMDATA%\\uv\\uv.toml, /etc/uv/uv.toml). UV_CONFIG_FILE names one file
    instead of discovering; UV_NO_CONFIG discovers nothing. `table` is the prefix the index
    keys sit under: "" for uv.toml, "tool.uv" for pyproject.toml.
    """
    if _uv_env_flag("UV_NO_CONFIG"):
        return []
    explicit = os.environ.get("UV_CONFIG_FILE", "").strip()
    if explicit:
        return [(Path(explicit), "")]
    found: "list[tuple[Path, str]]" = []
    here = Path.cwd()
    for d in (here, *here.parents):
        uv_toml = d / "uv.toml"
        if uv_toml.is_file():
            found.append((uv_toml, ""))
            break
        pyproject = d / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding = "utf-8")
            except OSError:
                text = ""
            if re.search(r"(?m)^\s*\[+tool\.uv(\.|\])", text):
                found.append((pyproject, "tool.uv"))
                break
    if IS_WINDOWS:
        user = os.environ.get("APPDATA", "")
        system = os.environ.get("PROGRAMDATA", "")
        if user:
            found.append((Path(user) / "uv" / "uv.toml", ""))
        if system:
            found.append((Path(system) / "uv" / "uv.toml", ""))
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "") or str(Path.home() / ".config")
        found.append((Path(xdg) / "uv" / "uv.toml", ""))
        found.append((Path("/etc/uv/uv.toml"), ""))
    return [(p, table) for p, table in found if p.is_file()]


def _uv_config_index_policy() -> "dict[str, object]":
    """{no_index, default_index, unreadable} from uv's discovered configuration.

    Project outranks user outranks system for a scalar, so the first file that sets a key
    decides it. Only the keys that decide where a resolve looks are read: no-index and
    default-index (index-url is the older spelling), at the top level and under [pip], and
    an [[index]] entry carrying default = true. A file this cannot parse is reported rather
    than guessed at.
    """
    policy: "dict[str, object]" = {
        "no_index": None,
        "default_index": None,
        "unreadable": False,
        "extra_indexes": [],
    }
    try:
        import tomllib
    except ImportError:  # 3.10: the native path is 3.11+, so only the x64 fallback lands here
        policy["unreadable"] = bool(_uv_config_files())
        return policy
    for path, table in _uv_config_files():
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError):
            policy["unreadable"] = True
            continue
        section = data
        for part in [p for p in table.split(".") if p]:
            section = section.get(part, {}) if isinstance(section, dict) else {}
        if not isinstance(section, dict):
            continue
        # uv pip (0.10.7): [pip] scalars beat top-level, and [[index]] default = true beats both.
        pip_scope = section.get("pip", {}) if isinstance(section.get("pip"), dict) else {}
        file_no_index = None
        for scope in (pip_scope, section):
            if file_no_index is None and isinstance(scope.get("no-index"), bool):
                file_no_index = scope["no-index"]
        file_default = None
        extras: list[str] = []
        indexes = section.get("index")
        if isinstance(indexes, list):
            for entry in indexes:
                if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
                    continue
                if entry.get("explicit") is True:
                    # uv: explicit serves only packages pinned via [tool.uv.sources]; with default = true it also removes PyPI as the default (not modelled: doubt).
                    if entry.get("default") is True:
                        policy["unreadable"] = True
                    continue
                if entry.get("default") is True:
                    if file_default is None:
                        file_default = entry["url"]
                else:
                    extras.append(entry["url"])
        for scope in (pip_scope, section):
            value = scope.get("extra-index-url")
            if isinstance(value, str):
                extras.append(value)
            elif isinstance(value, list):
                extras.extend(v for v in value if isinstance(v, str))
        policy["extra_indexes"] = list(policy["extra_indexes"]) + extras
        if file_default is None:
            for scope in (pip_scope, section):
                for key in ("default-index", "index-url"):
                    if file_default is None and isinstance(scope.get(key), str):
                        file_default = scope[key]
        if policy["no_index"] is None and file_no_index is not None:
            policy["no_index"] = file_no_index
        if policy["default_index"] is None and file_default is not None:
            policy["default_index"] = file_default
    return policy


def _url_is_public_pypi(url: str) -> bool:
    """The host, not a substring: "https://pypi.org.corp.example/simple" and
    ".../api/pypi/pypi.org/simple" both contain the name and neither is public PyPI."""
    try:
        from urllib.parse import urlsplit
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return False
    return host is not None and host.lower() == "pypi.org"


def _public_pypi_is_reachable() -> bool:
    """Can this resolution actually reach public PyPI?

    The table below records what PyPI publishes, which is only availability if PyPI is where
    the resolve will look. Offline, or pointed at an exclusive corporate index, those wheels
    are neither cached nor served: unblocking librosa there drops the skip and then fails the
    whole extras pass on an unavailable numba, which is exactly what the skip prevents.

    Judged for the resolver that runs the pass. uv reads UV_* and its configuration files and
    ignores PIP_*; pip reads PIP_* and ignores UV_*. Mixing the two reported PyPI reachable
    from a PIP_EXTRA_INDEX_URL that uv, the resolver in use, never consults. A default index
    REPLACES PyPI; an extra index adds to it, so PyPI named there is still consulted.
    Environment outranks uv's configuration files. Doubt resolves to False: that answer keeps
    the skip, the other fails the extras pass.
    """
    if USE_UV:
        return _uv_reaches_public_pypi()
    return _pip_reaches_public_pypi()


def _uv_env_flag(name: str) -> bool:
    """uv's own boolish set, for every UV_* switch read out of the caller's environment.

    Verified against uv 0.10.7, crates/uv-static/src/lib.rs
    parse_boolish_environment_variable, which restates clap's str_to_bool: true is
    y, yes, t, true, on, 1; false is n, no, f, false, off, 0; case-insensitive, and
    anything else aborts uv rather than being guessed at.

    `not in ("", "0", "false")`, which this used to be, read off, no, n and f as TRUE,
    the exact opposite of uv's answer for them.

    Stripped where uv is not: uv aborts on a padded value, so the resolve fails whatever
    this returns, and stripping keeps the answer identical to setup.sh's
    _uv_offline_requested and the two PowerShell Test-UvEnvFlag copies.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "t", "true", "y", "yes", "on")


def _pip_env_flag(name: str) -> bool:
    """pip's rule, kept separate on purpose.

    PIP_* are pip's variables and uv never reads them, so uv's parser has no authority
    over them. pip routes them through ConfigOptionParser._update_defaults -> strtobool
    (pip/_internal/utils/misc.py): true is y, yes, t, true, on, 1; false is n, no, f,
    false, off, 0; case-insensitive and unstripped, with anything else exiting pip on
    "is not a valid value". An empty value never reaches strtobool, because
    _get_ordered_configuration_items drops falsy values first.

    The literals coincide with uv's today. They are restated rather than shared anyway,
    so that the day either project changes its mind this is a one-function edit instead
    of a silent behaviour change in the other resolver.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "t", "true", "y", "yes", "on")


def _no_index_requested() -> bool:
    """True when the operator asked us for no registry index. OUR convention, not uv's.

    The distinction is not pedantic: for UV_NO_INDEX, uv 0.10.7 defines no such
    environment variable. `--no-index` exists only as a command-line flag, it is absent from
    `uv pip install --help`'s environment list beside UV_OFFLINE and UV_NO_CONFIG, and
    grepping the 0.10.7 tree for the name returns nothing. uv ignores it however it is
    spelled, so this is not a prediction about uv; it is us honouring a stated intent by
    shaping the arguments we pass.

    Read with uv's boolish set deliberately, not by inheritance: a caller sets this beside
    UV_OFFLINE and UV_NO_CONFIG, which uv really does read, and one spelling across all
    three is the point. It is a choice, and the test says so.

    Deliberately NOT turned into a `--no-index` argument. That would make our behaviour and
    uv's agree, which is the honest long-term answer, but it would also turn a resolve that
    works today into one with no index at all: a behaviour change for existing users, and
    its own change rather than part of a truthiness fix.
    """
    return _uv_env_flag("UV_NO_INDEX")


def _uv_reaches_public_pypi() -> bool:
    # UV_OFFLINE stops uv's network; UV_NO_INDEX is ours and uv ignores it.
    if _uv_is_offline() or _no_index_requested():
        return False
    extra_is_pypi = any(
        _url_is_public_pypi(u)
        for var in ("UV_INDEX", "UV_EXTRA_INDEX_URL")
        for u in os.environ.get(var, "").split()
    )
    for var in ("UV_DEFAULT_INDEX", "UV_INDEX_URL"):
        value = os.environ.get(var, "").strip()
        if value:
            return extra_is_pypi or _url_is_public_pypi(value)
    policy = _uv_config_index_policy()
    if policy["unreadable"] or policy["no_index"] is True:
        return False
    extra_is_pypi = extra_is_pypi or any(
        _url_is_public_pypi(u) for u in policy["extra_indexes"] if isinstance(u, str)
    )
    default = policy["default_index"]
    if isinstance(default, str) and not _url_is_public_pypi(default):
        return extra_is_pypi
    return True


def _pip_reaches_public_pypi() -> bool:
    """pip's policy: its environment first, then the configuration files it would read.

    PIP_* outranks every file. Below that, `pip config list` reports the effective values from
    the site, user and global files, an `[install]` key outranking its `[global]` twin for an
    install. A `no-index` or an exclusive `index-url` set there replaces PyPI just as the
    environment does. Doubt (a `pip config` that cannot be read) keeps the skip.
    """
    if _pip_env_flag("PIP_NO_INDEX"):
        return False
    extra_is_pypi = any(
        _url_is_public_pypi(u) for u in os.environ.get("PIP_EXTRA_INDEX_URL", "").split()
    )
    value = os.environ.get("PIP_INDEX_URL", "").strip()
    if value:
        return extra_is_pypi or _url_is_public_pypi(value)
    policy = _pip_config_index_policy()
    if policy["unreadable"] or policy["no_index"] is True:
        return False
    extra_is_pypi = extra_is_pypi or any(_url_is_public_pypi(u) for u in policy["extra_index_urls"])
    index_url = policy["index_url"]
    if isinstance(index_url, str) and not _url_is_public_pypi(index_url):
        return extra_is_pypi
    return True


def _pip_config_index_policy() -> "dict[str, object]":
    """The index keys pip's configuration files set, read from `pip config list`.

    Lines are `<section>.<key>='<value>'`; `:env:` entries mirror PIP_* variables the caller
    already read, so they are skipped. `install.<key>` outranks `global.<key>`, as it does for
    pip itself. A `pip config` that cannot run or be parsed is reported unreadable.
    """
    policy: "dict[str, object]" = {
        "no_index": None,
        "index_url": None,
        "extra_index_urls": [],
        "unreadable": False,
    }
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output = True,
            text = True,
            timeout = 60,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        policy["unreadable"] = True
        return policy
    if done.returncode != 0:
        policy["unreadable"] = True
        return policy
    found: "dict[str, dict[str, str]]" = {"global": {}, "install": {}}
    for line in done.stdout.splitlines():
        m = re.match(r"^(global|install)\.([a-z-]+)=(.*)$", line.strip())
        if not m:
            continue
        section, key, raw = m.groups()
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1]
        found[section][key] = raw
    for section in ("global", "install"):
        keys = found[section]
        if "no-index" in keys:
            policy["no_index"] = keys["no-index"].strip().lower() in ("1", "true", "yes", "on")
        if keys.get("index-url", "").strip():
            policy["index_url"] = keys["index-url"].strip()
        if "extra-index-url" in keys:
            # pip prints the repr, so a multi-line value arrives with a literal backslash-n.
            policy["extra_index_urls"] = [
                u for u in re.split(r"\s+|\\n", keys["extra-index-url"]) if u
            ]
    return policy


def _woa_pypi_provided_versions() -> "dict[str, set[str]]":
    """{canonical name: versions} install.ps1 found PyPI publishing for THIS interpreter.

    UNSLOTH_WOA_PYPI_PROVIDED carries space-separated `name==version` entries: the wheelhouse
    wheels install.ps1 discarded because PyPI serves the same version. The managed copy is
    gone, so find-links no longer answers for them; this is the only record that they resolve.
    """
    provided: "dict[str, set[str]]" = {}
    for entry in os.environ.get("UNSLOTH_WOA_PYPI_PROVIDED", "").split():
        name, sep, version = entry.partition("==")
        if sep and name and version:
            provided.setdefault(_canonical_dist_name(name), set()).add(version)
    return provided


def _public_index_win_arm64_versions(canonical: str) -> "set[str]":
    """Versions the public index publishes a usable win_arm64 wheel of, for THIS build.

    Empty off win_arm64, empty when the resolve cannot reach PyPI, and empty for an
    interpreter the recorded wheel is not tagged for. The tag is judged by
    _wheel_matches_interpreter rather than by comparing strings, so a free-threaded build does
    not claim a wheel built for the GIL one. install.ps1's own probe is added on top.
    """
    if not _is_win_arm64_interpreter() or not _public_pypi_is_reachable():
        return set()
    return {
        version
        for tag, version in WINDOWS_ARM64_PUBLIC_INDEX_WHEELS.get(canonical, {}).items()
        if _wheel_matches_interpreter(f"{canonical}-{version}-{tag}-{tag}-win_arm64.whl")
    } | _woa_pypi_provided_versions().get(_canonical_dist_name(canonical), set())


def _windows_arm64_skip_packages(req: "Path | None" = None) -> set[str]:
    """WINDOWS_ARM64_SKIP_PACKAGES minus whatever the wheelhouse already provides, so
    hosting a wheel is all it takes to re-enable one of these features here.

    ``req`` is the requirements file about to be installed, when there is one. Its pins
    decide whether a hosted wheel is actually usable: a name match is not enough, because
    the resolver has to honour ``tiktoken==0.13.0`` and a staged 0.12.0 wheel leaves it
    with the unbuildable sdist rather than the skip this list is here to keep.
    """
    available = _find_links_wheel_versions()
    if not available and not any(
        _public_index_win_arm64_versions(name)
        for name in set(WINDOWS_ARM64_PUBLIC_INDEX_WHEELS) | set(_woa_pypi_provided_versions())
    ):
        return set(WINDOWS_ARM64_SKIP_PACKAGES)
    pins = _requirement_pins(req)

    def hosted(name: str) -> bool:
        canonical = _canonical_dist_name(name)
        versions = set(available.get(canonical) or ()) | _public_index_win_arm64_versions(canonical)
        if not versions:
            return False
        clauses = [clause for clause in pins.get(canonical, []) if clause]
        if not clauses:
            floor = WINDOWS_ARM64_BLOCKER_FLOORS.get(canonical)
            if floor is None:
                return True
            clauses = [floor[0]]
        verdicts = [_version_satisfies(v, c) for v in versions for c in clauses]
        if any(verdict is True for verdict in verdicts):
            return True
        return any(verdict is None for verdict in verdicts)

    keep_skipping: set[str] = set()
    for package in WINDOWS_ARM64_SKIP_PACKAGES:
        canonical = _canonical_dist_name(package)
        blockers = WINDOWS_ARM64_SKIP_UNBLOCKED_BY.get(canonical)
        if blockers:
            if all(hosted(b) for b in blockers):
                continue
        elif hosted(package):
            continue
        keep_skipping.add(package)
    return keep_skipping


# Skipped without torch (Intel Mac GGUF-only), plus librosa, whose numba chain fails (#5046).
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "librosa",
}

# No wheel on PyPI at any version, so always built from source (antlr4 arrives via omegaconf).
# A package-scoped --no-binary keeps a user's binary-only policy everywhere else.
# Keep in sync with .github/scripts/clean-machine-assert.sh and .github/scripts/assert-nobuild.ps1.
SDIST_ONLY_PACKAGES = (
    "openai-whisper",
    "argbind",
    "randomname",
    "antlr4-python3-runtime",
)


def _sdist_only_build_args(*names: str) -> list[str]:
    """``--no-binary`` for each named wheel-less requirement, for uv and pip alike.

    Naming a package that the resolution never reaches is harmless (verified), so this
    is safe next to the NO_TORCH / Windows requirement filtering.
    """
    args: list[str] = []
    for name in names:
        args += ["--no-binary", name]
    return args


def _extras_sdist_only_packages() -> tuple[str, ...]:
    """SDIST_ONLY_PACKAGES plus any this interpreter alone resolves to an sdist."""
    names = list(SDIST_ONLY_PACKAGES)
    # MeCab==0.996.5 (macOS cp314+) is the last release with an sdist; elsewhere it resolves to a wheel.
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
        _safe_print(_redact_install_output(result.stdout).strip())


def _flash_attn_install_disabled() -> bool:
    return os.getenv("UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL") == "1"


# Matches worker._is_importable_isolated: the same untrusted import, bounded the same way.
_FLASH_ATTN_IMPORT_PROBE_TIMEOUT = 300


def _flash_attn_importable() -> bool:
    """Whether flash_attn imports, checked out of process.

    A wrong-arch/ABI wheel installs fine and raises on import, so a zero pip exit code is
    not proof the install is usable. In a child, so a half-loaded native extension cannot
    poison the installer, and bounded, since initialisation can hang rather than fail.
    """
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
    """Uninstall a flash-attn that installed but will not import. True iff it is gone.

    uv gets --python as well as --system: --system ALONE would remove from the system
    Python, leaving the rejected wheel in the venv while setup reported it gone.
    """
    return _uninstall_distribution("flash-attn")


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
    wheel_available = url_exists(wheel_url) if wheel_url else False
    if wheel_available:
        _count_install_action()
        outcome = install_prebuilt(
            wheel_url,
            install = install_wheel,
            verify = _flash_attn_importable,
            on_failed = lambda installer, wheel_result: _print_optional_install_failure(
                f"Installing flash-attn prebuilt wheel with {installer}",
                wheel_result,
            ),
            use_uv = USE_UV,
            uv_needs_system = UV_NEEDS_SYSTEM,
        )
        if outcome == "installed":
            return
        if outcome == "rejected":
            # Remove it before giving up. Left installed, unsloth/models/_utils.py finds
            # it by metadata (_package_available) and then imports the native module
            # in process, so a wheel that killed the probe would kill training too.
            if _remove_rejected_flash_attn():
                _step(
                    "warning",
                    "flash-attn wheel installed but is not importable on this GPU; removed it",
                    _cyan,
                )
            else:
                # Still importable in process, unlike never having installed it.
                _step(
                    "warning",
                    "flash-attn wheel is not importable on this GPU and could not be "
                    "removed; uninstall flash-attn manually before training",
                    _cyan,
                )
        _step("warning", "Continuing without flash-attn", _cyan)
        return

    if wheel_url is None:
        _step("warning", "No compatible flash-attn prebuilt wheel found", _cyan)
    elif wheel_available is None:
        _step(
            "warning",
            "Could not check the flash-attn prebuilt wheel; skipped it",
            _cyan,
        )
    else:
        _step("warning", "No published flash-attn prebuilt wheel found", _cyan)


USE_UV = False  # Set by _bootstrap_uv() at the start of install_python_stack()
UV_NEEDS_SYSTEM = False  # Set by _bootstrap_uv() via probe


def _bootstrap_uv() -> bool:
    """Check if uv is available and probe whether --system is needed.

    `pip freeze`, not `pip install --dry-run pip`: the question is only whether this uv can
    address this interpreter, and a dry-run answers it by RESOLVING pip against the index.
    That is a PyPI round-trip on every run, most of the 12 s the "pip bootstrap" step cost,
    and it fails offline where uv could have served from its cache. freeze reads the venv's
    own metadata: no resolution, no network, same answer.
    """
    global UV_NEEDS_SYSTEM
    if not shutil.which("uv"):
        return False
    # Explicit --python: uv can ignore the activated venv on some platforms.
    probe = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        **_windows_hidden_subprocess_kwargs(),
    )
    if probe.returncode != 0:
        probe_sys = subprocess.run(
            ["uv", "pip", "freeze", "--system"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            **_windows_hidden_subprocess_kwargs(),
        )
        if probe_sys.returncode != 0:
            return False  # uv is broken, fall back to pip
        UV_NEEDS_SYSTEM = True
    return True


def _venv_pip_is_usable() -> bool:
    """Whether this interpreter already has a pip new enough for the pass.

    Read from metadata rather than by spawning it: the bootstrap below reinstalls pip from the
    index on every run, a download and a resolve for a package nothing changed. 23.0 is the
    floor because the constraint and override handling the pass relies on predates it
    comfortably; a fresh uv venv has no pip at all and still bootstraps.
    """
    try:
        if importlib.util.find_spec("pip") is None:
            return False
    except (ImportError, ValueError):
        return False
    installed = _installed_distribution_version("pip")
    if not installed:
        return False
    try:
        major = int(re.match(r"\d+", installed).group(0))
    except (AttributeError, ValueError):
        return False
    if major < 23:
        return False
    try:
        probe = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output = True,
            timeout = 60,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _filter_requirements(req: Path, skip: set[str]) -> Path:
    """Return a temp copy, adjacent when writable, with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line for line in lines if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    # Beside the source so relative -r/-c resolve; a read-only tree falls back.
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
    """The unsloth-zoo git ref the --local overlay installs.

    UNSLOTH_ZOO_REF lets the Studio venv track the requested zoo instead of
    always main, which is what the Docker build pins against and what
    install.sh reads into _ZOO_REF. Unset means main.
    """
    return os.environ.get("UNSLOTH_ZOO_REF", "").strip() or "main"


def _unsloth_zoo_git_spec() -> str:
    """The pip requirement string for the unsloth-zoo overlay.

    An unset UNSLOTH_ZOO_REF leaves the URL bare rather than appending @main: a
    bare git URL already clones the default branch, so the default install is
    byte for byte the one every caller and the staging path already expect.
    """
    ref = os.environ.get("UNSLOTH_ZOO_REF", "").strip()
    return _UNSLOTH_ZOO_GIT_URL + ("@" + ref if ref else "")


def _overlay_local_core_package(
    name: str,
    local_repo: str,
    *,
    strict: bool = True,
) -> bool:
    """Install one core package from the source selected by --local.

    strict=False reports a failed install instead of exiting, which the metadata
    repair needs: by the time it installs, it has already removed the records it
    is replacing, so it has to say so rather than die mid-way.
    """
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
    """What pip would be asked to build for this overlay.

    unsloth-zoo comes from git, so an overlay is a network fetch just as much as
    an index install is: it has to be staged before anything is uninstalled.
    """
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    if canonical == "unsloth":
        return local_repo
    if canonical == "unsloth-zoo":
        return _unsloth_zoo_git_spec()
    return ""


def _rewrite_minimal_metadata(path: str, name: str) -> bool:
    """Replace an unparseable METADATA with the least pip needs to uninstall by RECORD.

    Returns False when there is no RECORD to uninstall from, the one case that has
    to fail closed: without it neither pip nor this installer knows which files
    belong to the package, and laying a replacement over them would leave whatever
    the new release no longer ships behind, still importable. The version is taken
    from the directory name, where importlib's own fallback reads it from when
    METADATA cannot be parsed.
    """
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
    """Invalid metadata directories moved aside, restorable until committed.

    pip cannot parse an unreadable record: a non-UTF-8 METADATA makes pip list,
    show and uninstall raise for the whole environment, so it has to be out of
    the way before pip runs at all. Deleting it outright is not an option
    either, because staging the replacement can still fail and a package whose
    only record was deleted is left with files and no install at all.
    """

    def __init__(self) -> None:
        self._holding = ""
        self._moved: list = []
        self._copied: list = []

    def _holding_dir(self) -> str:
        if not self._holding:
            self._holding = tempfile.mkdtemp(prefix = "unsloth_metadata_quarantine_")
        return self._holding

    def back_up(self, path) -> bool:
        """Keep a copy of a file that is about to be rewritten in place.

        The rewrite has to happen before staging, and staging can still fail. Without
        this the original is gone and what remains is a synthetic record that parses:
        the next run would see one readable record, decide nothing is wrong, and never
        attempt the payload repair that is still owed.

        An absent METADATA is nothing to back up rather than a failure. Recording it
        as absent still lets the rewrite proceed, so pip can uninstall that record by
        its RECORD; restore() then deletes the synthetic file instead of reinstating
        one that never existed.
        """
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
        """Drop the backed-up METADATA copies, keeping the moved directories.

        Called once a staged reinstall has put the package back: the wheel's own
        metadata is authoritative, so copying the original over it would re-break the
        package the rollback just repaired, and deleting it (where the original was
        absent) would strip a record pip just wrote. The moved entries stay, because a
        record pip cannot consume still has to go back as found.
        """
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
    """Put the payload back when the uninstall loop stops part way through.

    An earlier successful uninstall has already deleted the package tree, so
    returning here without this leaves a surviving dist-info claiming an
    installed core package whose files are gone.
    """
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
        # pip, not uv: uv would reject the unpinned name under UV_REQUIRE_HASHES with records already gone.
        force_pip = True,
    ):
        # The wheel wrote fresh metadata at the same path; do not restore the original over it.
        if quarantine is not None:
            quarantine.forget_copies()
        _safe_print(_red(f"   restored {name} from the staged replacement"), file = sys.stderr)
    else:
        _safe_print(
            _red(f"   {name} is no longer installed. Re-run the installer to restore it."),
            file = sys.stderr,
        )


def _requirement_args(requirement: str, staging: str) -> "list[str]":
    """How to hand pip the requirement, as a file when it carries hashes.

    pip only accepts --hash entries from a requirements file, and the hashes are what
    stop it accepting a different artifact of the same version from a source uv never
    considered. The file lives in the staging directory, so it is removed with it.
    """
    if "--hash=" not in requirement:
        return [requirement]
    path = os.path.join(staging, "requirement.txt")
    with open(path, "w", encoding = "utf-8") as handle:
        handle.write(requirement + "\n")
    return ["-r", path]


def _stage_replacement(name: str):
    """Build the wheel that will replace a package, before it is removed.

    Returns a directory to install from, or None when the package cannot be
    obtained, which must abort the repair while the existing install is still
    intact.

    pip wheel, not pip download: a source-only index leaves an sdist, and the
    install that follows runs --no-index, so its isolated build could not fetch
    setuptools and the package would stay uninstalled. Building here, while the
    index is still reachable, keeps that install offline-safe. pip and not uv
    because uv has no `wheel` subcommand, so uv's own index variables and upload
    cutoff have to be handed across explicitly to keep the provenance and the
    reproducibility policy the other installs run under.
    """
    requirement, overrides, build_options = name, {}, []
    offline_local = USE_UV and _uv_is_offline() and _is_local_source(name)
    if offline_local:
        # pip's isolated build fetches the build backend even offline; build against the interpreter's
        # own backend and forbid the index.
        build_options = ["--no-build-isolation"]
        overrides = {"PIP_NO_INDEX": "1"}
    if USE_UV and _uv_is_offline() and not _is_local_source(name):
        _safe_print(
            _red(
                "   UV_OFFLINE is set and pip has no offline mode, so repairing "
                f"{name} would have to reach the network; leaving the install alone."
            ),
            file = sys.stderr,
        )
        return None
    if USE_UV and not _is_direct_reference(name):
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
    """The distributions a run is responsible for.

    unsloth-zoo only for the default install: `--package X` installs X alone, so
    demanding the companion would force-reinstall an unrelated distribution and
    fail the update on an unreachable zoo index. Canonically, because
    verify_install reads `--package Unsloth` that way and the two disagreeing
    had the deep check scan zoo and force a pass no repair gate would act on.
    """
    default = re.sub(r"[-_.]+", "-", package_name).lower() == "unsloth"
    return (package_name, "unsloth-zoo") if default else (package_name,)


def _repair_damaged_core_payload(
    package_names: "tuple[str, ...]",
    *,
    local_repo: str = "",
    require_present: bool = False,
) -> bool:
    """Reinstall managed core packages whose recorded files are gone or truncated.

    An upgrade of a distribution already at the wanted version installs nothing:
    uv audits it, pip calls it satisfied, and both read metadata a quarantine of
    the payload leaves intact, so it has to be named for reinstall. Skipped for
    a local checkout, whose core packages are an editable overlay.

    False when the files are still missing afterwards and the caller aborts, or
    the pass that follows audits the intact metadata as satisfied and
    write_manifest records a success nothing rechecks. Judged on the tree, not
    pip's exit code, so a reinstall that restored the files while exiting
    non-zero still counts.

    `require_present` also refuses a distribution not installed at all, which
    has no RECORD and so reads as undamaged. Off before the core phase, where a
    fresh run has nothing yet; on after it, where absence means the phase was
    skipped (SKIP_STUDIO_BASE=1) or silently did nothing.
    """
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
        # --no-deps keeps torch out; pip has no per-package reinstall and uv's --reinstall is env-wide.
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
        # Presence first: --force-reinstall uninstalls before installing.
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
    """Reinstall managed core packages whose metadata has more than one record.

    Remove invalid records directly because pip cannot parse them, then repeat a
    dependency-free uninstall until no valid record remains, because pip's
    force-reinstall only uninstalls the one record its finder selects. The
    requested source can then be installed without asking a resolver to replace
    the existing torch build. The normal dependency pass still follows.

    The replacement is fetched BEFORE anything is removed: the uninstall loop
    deletes every record it finds, so an index unreachable at that moment
    (offline, a private package name, a mirror outage) would otherwise leave the
    venv with no unsloth at all and no way back.
    """
    duplicates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name in package_names:
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        versions = install_manifest.installed_versions(name)
        record_count = len(versions)
        # A sole `~` backup reads as one version to metadata_conflict(), yet pip and importlib skip it.
        if install_manifest.metadata_conflict(
            versions
        ) or install_manifest.pip_backup_metadata_paths(name):
            duplicates.append((name, record_count))

    repaired: list[str] = []
    staging_dirs: list[str] = []
    # One quarantine per package: a shared one would restore a stale record over a later reinstall.
    quarantine = _QuarantinedMetadata()
    succeeded = False
    try:
        for name, record_count in duplicates:
            quarantine = _QuarantinedMetadata()
            _step(_LABEL, f"duplicate metadata for {name} detected; reinstalling it", _dim)
            invalid_paths = install_manifest.invalid_metadata_paths(name)
            # Rewrite METADATA beside each intact RECORD so pip uninstalls exactly their files; quarantining
            # would leave modules only the older release owned.
            unrewritable = [
                path
                for path in invalid_paths
                if not (
                    quarantine.back_up(os.path.join(path, "METADATA"))
                    and _rewrite_minimal_metadata(path, name)
                )
            ]
            # Every record must be uninstallable by pip or nothing is touched, so a later run still sees the
            # conflict instead of a falsely successful repair.
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
            # pip ignores its own abandoned ~ backup yet the record counts here, so the loop could never converge.
            # Quarantined, not deleted: staging can still fail.
            backups = install_manifest.pip_backup_metadata_paths(name)
            if backups and not quarantine.take(backups):
                _safe_print(
                    _red(f"   could not move pip's leftover backup for {name} aside"),
                    file = sys.stderr,
                )
                return False
            if invalid_paths or backups:
                # Metadata changed outside the install helpers, so retire the constraint cache here.
                _count_install_action()
                importlib.invalidate_caches()
                record_count = len(install_manifest.installed_versions(name))
            # A backup names a payload pip already renamed away, so a fresh install is exactly right.
            if invalid_paths and not record_count:
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
            # Local / git sources need no staging; index sources must be proven reachable before uninstalling.
            overlaid = bool(source_repo) and _is_overlayable_core_package(name)
            # Stage overlays too: a git fetch or build can fail after every record is removed.
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
                # Counted before the command: a failed uninstall can still have removed files.
                _count_install_action()
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

            restored = overlaid and _overlay_local_core_package(name, source_repo, strict = False)
            if not restored:
                # The staged wheel was built from the same source, so falling back never substitutes a release.
                restored = pip_install_try(
                    f"Repairing duplicate metadata for {name}",
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    "--no-index",
                    "--find-links",
                    staged,
                    name,
                    # pip, as _restore_from_staged, so uv hash policy cannot reject the built wheel.
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
    """Build a standard pip install command.

    pip has no --upgrade-package, so uv's flag is translated rather than
    dropped. Dropping it made this fallback a no-op on the update path: pip saw
    the named distributions as already satisfied, installed nothing, and the
    update still reported success. Any uv failure reached that, not just the
    Windows in-use launcher.

    --upgrade-strategy is pinned to only-if-needed rather than left to pip's
    default, because that default is the load-bearing part: it upgrades the
    named packages without dragging the existing torch build along.
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    upgrade: list[str] = []
    drop_next = ""
    for arg in args:
        if drop_next:
            if drop_next == "--upgrade-package":
                upgrade.append(arg)
            drop_next = ""
            continue
        if arg == "--upgrade-package":
            drop_next = arg  # the flag; its value is the package to upgrade
            continue
        if arg == "--reinstall-package":
            # uv-only; pip's --force-reinstall is env-wide, safe only because the caller passes --no-deps.
            drop_next = arg
            continue
        cmd.append(arg)
    if upgrade:
        cmd += ["--upgrade", "--upgrade-strategy", "only-if-needed"]

        # By canonical project name: pip refuses `--upgrade-package mlx` beside `mlx==0.32.3` as a double
        # requirement where uv does not.
        def _project(requirement: str) -> str:
            # _requirement_name stops at "==" and "@"; a range needs the rest.
            return _canonical_package_name(
                re.split(r"[<>=!~;\[ ]", _requirement_name(requirement), maxsplit = 1)[0]
            )

        named = {_project(arg) for arg in cmd if arg and not arg.startswith("-")}
        cmd += [name for name in upgrade if _project(name) not in named]
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


def _pinned_cmd_and_env(cmd: "list[str]") -> "tuple[list[str], dict[str, str] | None]":
    """A command with its pinned binary-policy arguments, and the env it runs with, from ONE read.

    A failed read is not memoised, so a second one can succeed: the policy would then reach
    the environment but not the argv that carries it (uv) or exempts from it (both).
    """
    env = _install_env_for_cmd(cmd)
    return cmd + _pinned_binary_policy_args(cmd, env), env


# Every torch on AMD's per-arch indexes requires rocm[libraries], published only as an sdist;
# exempted on those pins alone.
_AMD_ARCH_INDEX_SDIST_ONLY_PACKAGES = ("rocm",)


def _pins_amd_arch_index(cmd: "list[str]") -> bool:
    """True when the index ``cmd`` pins has a gfx leaf, the shape of every AMD per-arch index."""
    for flag, value in zip(cmd, cmd[1:]):
        if flag in ("--index-url", "--default-index"):
            return bool(re.match(r"gfx\d", _torch_index_leaf(value)))
    return False


def _pinned_binary_policy_args(cmd: "list[str]", env: "dict[str, str] | None") -> "list[str]":
    """The re-asserted only-binary as argv for a pinned command, plus its package exemptions.

    uv reads neither pip.conf nor PIP_ONLY_BINARY, and a pinned command runs with
    UV_NO_CONFIG=1 since a discovered uv.toml outranks the CLI pin (#6898), so the
    environment alone leaves the policy unenforced on the leg that runs. Measured on uv
    0.10.7: a pinned install builds the sdist with PIP_ONLY_BINARY=:all: set and refuses it
    given --only-binary. Pinned only: others keep their config file and uv applies it.

    Only when a policy is in force, so an unconfigured host's argv is unchanged, and never for a
    package the operator named: a CLI --no-binary overrides their rule for it (pip 26.2).
    """
    if not _is_pinned_index_cmd(cmd):
        return []
    parts = [part.strip() for part in (env or {}).get("PIP_ONLY_BINARY", "").split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return []
    args: list[str] = []
    if cmd[:1] == ["uv"]:
        # Repeatable rather than comma joined, which is the spelling uv takes (pip takes both).
        for part in parts:
            args.extend(["--only-binary", part])
    if not _pins_amd_arch_index(cmd):
        return args
    named = {_canonical_package_name(part) for part in parts}
    exempt = [
        name
        for name in _AMD_ARCH_INDEX_SDIST_ONLY_PACKAGES
        if _canonical_package_name(name) not in named
    ]
    return args + _sdist_only_build_args(*exempt)


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
    # PIP_NO_INDEX=1 makes pip ignore --index-url; PIP_INDEX_URL goes too so a stale mirror cannot win.
    "PIP_NO_INDEX",
    "PIP_INDEX_URL",
)


def _is_pinned_index_cmd(cmd: "list[str] | tuple[str, ...]") -> bool:
    """True when the command pins an index via --index-url / --default-index."""
    return any(arg in ("--index-url", "--default-index") for arg in cmd)


# Our requirements files are unhashed, so require-hashes can only abort. Env beats config.
_PM_HASH_ENV_VARS = (
    "UV_REQUIRE_HASHES",
    "PIP_REQUIRE_HASHES",
)

# no-binary would force source builds of pinned wheels. UV_EXCLUDE_NEWER is cleared because only uv
# reads it, so the pip fallback could install past the cutoff.
_PM_FORCE_SOURCE_ENV_VARS = (
    "UV_NO_BINARY",
    "UV_NO_BINARY_PACKAGE",
    "PIP_NO_BINARY",
    "UV_EXCLUDE_NEWER",
)

# PIP_ONLY_BINARY stays: pinned indexes serve wheels (rocm aside). UV_NO_BUILD / UV_NO_BINARY /
# UV_ONLY_BINARY are not uv env vars (uv 0.10.7).


def _relaxed_pip_policy_env(cmd: "list[str]") -> "dict[str, str]":
    """Hash mode off for one pip install / download / wheel, and nothing else relaxed.

    pip applies env vars AFTER config files, so this wins while pip.conf's index-url, cert,
    proxy and only-binary stay in force; the wheel-less requirements go through the
    package-scoped --no-binary in _sdist_only_build_args(). `wheel` is in the set because
    the duplicate-metadata repair stages with it, and require-hashes rejects that too
    (#8530).
    """
    if not _is_pip_subcommand(cmd, ("install", "download", "wheel")):
        return {}
    return {"PIP_REQUIRE_HASHES": "0"}


def _executable_stem(path: str) -> str:
    """argv[0] as a bare program name, on either platform's spelling.

    Both separators, because os.path.basename does not split a backslash off-Windows and
    the same argv reaches here from a Windows host, a test, or WSL interop.
    """
    leaf = re.split(r"[\\/]", path)[-1]
    return leaf.split(".")[0].lower()


def _is_pip_subcommand(cmd: "list[str]", subcommands: "tuple[str, ...]") -> bool:
    """True when ``cmd`` is a pip invocation whose SUBCOMMAND is one of ``subcommands``.

    Structural, so the relaxation cannot ride along on a requirements path, a package
    named ``wheel``, or a uv command that merely contains the word.
    """
    args = list(cmd)
    if args[:1] == ["uv"]:
        return False
    if len(args) >= 3 and args[1] == "-m" and args[2] in ("pip", "pip3"):
        args = args[3:]
    elif args and _executable_stem(args[0]) in ("pip", "pip3"):
        args = args[1:]
    else:
        return False
    # The first token naming a subcommand: `pip --cache-dir /tmp/c install x` has a bare path first.
    for arg in args:
        if arg in _PIP_SUBCOMMANDS:
            return arg in subcommands
    return False


# pip 26.2 `pip --help`; only marks where the options stop, so a missing future one is free.
_PIP_SUBCOMMANDS = frozenset(
    (
        "install",
        "download",
        "uninstall",
        "freeze",
        "inspect",
        "list",
        "show",
        "check",
        "config",
        "search",
        "cache",
        "index",
        "wheel",
        "hash",
        "completion",
        "debug",
        "help",
        "lock",
    )
)


def _uv_is_offline() -> bool:
    """True when uv has been told not to touch the network.

    uv's own boolish set, as both setup scripts read it. `not in (0, false)` also read `off`
    and `no` as offline, declining repairs with a message saying the opposite.
    """
    return _uv_env_flag("UV_OFFLINE")


def _uv_staging_plan(name: str) -> "tuple[str, dict[str, str]] | None":
    """Ask uv which release and which index it would use, and reproduce that with pip.

    Returns (requirement, pip env overrides), or None when uv could not resolve it.

    Staging has to run pip, because uv has no `wheel` subcommand. Translating uv's index
    configuration out of the environment cannot be made correct: uv also discovers
    uv.toml, pyproject.toml [tool.uv] and a user config, honours UV_CONFIG_FILE, applies
    an implicit PyPI default, and resolves under an index-strategy pip has no equivalent
    for. Any of those missed means the repair can uninstall a private build and reinstall
    the public package of the same name.

    So uv is asked instead. `uv pip compile --emit-index-annotation` reports the exact
    index each package resolved from, under uv's own discovery, priority, strategy and
    upload cutoff, and pip is pointed at that one index with that one version. An
    unreachable higher-priority index fails the compile rather than falling through to a
    public fallback, which is the behaviour first-index exists to give.
    """
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
            # uv's artifact policy, which pip does not read; without it the artifact type could change mid-repair.
            option, _, value = line.partition(" ")
            build_options.extend((option, value.strip()))
        elif line.startswith("# from "):
            # uv 0.10.7 drops userinfo from this annotation but keeps it on index lines, so match back to the
            # emitted URL or a private index answers 401.
            origin = line[len("# from ") :].strip() or origin
        elif line and not line.startswith(("#", "-")):
            # uv continues a hashed pin onto the following lines with a backslash.
            pinned = line.split(";", 1)[0].rstrip("\\").strip()
            if _canonical_package_name(_requirement_name(pinned)) == canonical:
                requirement = pinned
    if not requirement:
        return None
    # Replace pip's candidate sources, not add to them, so nothing uv never looked at can serve it.
    # Empty rather than deleted: pip 26.2 reads an empty value as unset.
    overrides = {
        "PIP_EXTRA_INDEX_URL": "",
        "PIP_NO_INDEX": "",
        "PIP_FIND_LINKS": " ".join(find_links),
    }
    index_url = _credentialed_index(origin, emitted)
    if index_url:
        overrides["PIP_INDEX_URL"] = index_url
    elif find_links:
        # Flat source with no index means a configured no-index; keep pip off default PyPI.
        overrides["PIP_NO_INDEX"] = "1"
    # --emit-build-options surfaces uv.toml policy but not the env spelling (uv 0.10.7), so translate
    # that by hand, including UV_KEYRING_PROVIDER for authenticated indexes.
    for uv_name, pip_name in (
        ("UV_NO_BINARY", "PIP_NO_BINARY"),
        ("UV_ONLY_BINARY", "PIP_ONLY_BINARY"),
        ("UV_KEYRING_PROVIDER", "PIP_KEYRING_PROVIDER"),
    ):
        value = os.environ.get(uv_name, "").strip()
        if value and not os.environ.get(pip_name):
            overrides[pip_name] = value
    if hashes:
        # The hashes make this safe: pip verifies them even with PIP_REQUIRE_HASHES=0 and may still read a
        # site pip.conf.
        requirement = " \\\n    ".join([requirement, *hashes])
    return requirement, overrides, build_options


_PIP_SOURCE_CONFIG_KEYS = ("index-url", "extra-index-url", "find-links", "no-index")


def _pip_config_without_sources(directory: str) -> str:
    """Write pip's own configuration back minus the candidate sources.

    The environment overrides above cannot do this alone. Measured on pip 26.2: with
    `extra-index-url` in pip.conf, an empty PIP_EXTRA_INDEX_URL does NOT suppress it,
    so the config has to go for this one command, or uv's chosen index is only one
    candidate among the user's.

    Dropping it wholesale would take proxy, cert, client-cert and trusted-host with it,
    and those are how a private index is reached, so uv would resolve and pip would then
    fail to fetch. Everything except the four source keys is written back instead.
    `pip config list` is asked rather than the files located, so global, user and site
    are merged in pip's own order; `:env:` entries are skipped as they come from the
    environment, handled above.
    """
    path = os.path.join(directory, "pip.conf")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            # Dictate the child's encoding (see _decode_pip_output); a non-ASCII cert path must survive.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"},
            **_windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        result = None
    sections: dict[str, list[tuple[str, str]]] = {}
    if result is not None and result.returncode == 0:
        for line in _decode_pip_output(result.stdout or b"").splitlines():
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
            # pip renders a multi-value setting as one newline separated string; an
            # indented continuation is how it is spelled back into a config file.
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
    """The distribution name from a pin or a PEP 508 direct reference.

    An override can redirect a package to a path, a repository or a URL, and uv then
    emits `name @ reference` rather than `name==version`. Treating the whole line as
    the name left the requirement empty and aborted every repair under that policy.
    """
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
    """The emitted index matching the annotated origin, credentials intact.

    uv emits every index it was configured with, credentials and all, but strips
    userinfo from the `# from` annotation that says which one answered. Taking the
    annotation at face value hands pip an unauthenticated URL for a private index,
    which answers 401 and aborts the repair. Matching is on the credential-free form
    of each emitted URL, so an authenticated extra index is recovered too.
    """
    if origin:
        target = _strip_userinfo(origin).rstrip("/")
        matches = [url for url in emitted if _strip_userinfo(url).rstrip("/") == target]
        # The credentialed form wins over a bare match on the same URL.
        for url in matches:
            if _strip_userinfo(url) != url:
                return url
        if matches:
            return matches[0]
    # Not an emitted index, so a find-links source (already in PIP_FIND_LINKS); an sdist from it still
    # needs the real index for its build backend.
    return emitted[0] if emitted else ""


def _is_local_source(requirement: str) -> bool:
    """True when the replacement is already on disk, so no network is needed."""
    return os.path.exists(requirement)


def _is_direct_reference(requirement: str) -> bool:
    """True when the requirement already names the source to build from.

    The overlay paths hand staging a git URL or a local checkout rather than a bare
    name, and such a requirement carries its own provenance: no index was consulted
    to choose it, so there is nothing for uv to have decided and nothing to preserve.
    Asking uv to resolve it would also compare a bare spec against uv's output, which
    appends the resolved commit and so could never match.
    """
    return "://" in requirement or _is_local_source(requirement)


def _uv_upload_cutoff_args() -> "list[str] | None":
    """pip arguments carrying UV_EXCLUDE_NEWER, or None when it cannot be honoured.

    uv's --exclude-newer limits candidates by upload time, and staging runs pip, which
    ignores the variable and would stage a release the user's policy excludes. pip's
    --uploaded-prior-to is the same filter and takes the same date spellings, but it only
    exists from pip 25.3. Refusing to stage is the correct answer on an older pip: the
    repair then aborts with the installation still intact, rather than quietly installing
    a wheel the cutoff forbids.
    """
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
    """Return an env with the uv index vars stripped for a pinned-index install.

    None (inherit env) when the command does NOT pin an index, so ordinary installs honour
    the user's mirror. Pinned ones also get UV_NO_CONFIG=1, since a discovered uv.toml
    outranks the CLI pin (#6898), and no blanket amnesty from the operator's policy: only
    the hash and no-binary variables go, for the reasons at their definitions.

    PIP_CONFIG_FILE stays at os.devnull, the ONLY way to stop a site or global pip.conf
    contributing (measured on pip 26.2: naming a real file suppresses the per-user file
    alone). What that removes is put back key by key by _pinned_pip_config_overrides().
    """
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
    for name in _PM_HASH_ENV_VARS + _PM_FORCE_SOURCE_ENV_VARS:
        env.pop(name, None)
    # Both config files go for the pin, so re-assert what they carried that it does not
    # conflict with, from the section belonging to THIS command.
    for name, value in _pinned_pip_config_overrides(_pip_subcommand_of(cmd)).items():
        # Not setdefault: pip ignores an EMPTY env value and falls through to the (now devnull) config.
        if not env.get(name):
            env[name] = value
        elif name in _PINNED_PIP_ENV_ACCUMULATING:
            # pip applies env entries after the file's, so append rather than replace.
            env[name] = f"{value},{env[name]}"
    env["UV_NO_CONFIG"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


# Allowlist of config keys devnull must not drop (transport, only-binary). Source keys, no-binary and
# require-hashes are absent on purpose.
_PINNED_PIP_CONFIG_KEEP_KEYS = (
    "cert",
    "client-cert",
    "proxy",
    "trusted-host",
    "timeout",
    "retries",
    "keyring-provider",
    "only-binary",
)

# pip config is per subcommand while PIP_ vars are command-wide, so read [global] plus this command's
# section. uv commands default to install.
_PINNED_PIP_CONFIG_GLOBAL_SECTION = "global"
_PINNED_PIP_CONFIG_DEFAULT_SECTION = "install"

# Only list-valued keys: collapsing newlines elsewhere would corrupt paths with spaces.
_PINNED_PIP_CONFIG_SEPARATORS = {"trusted-host": " ", "only-binary": ","}

# Only only-binary accumulates across sections (pip 26.2); accumulating trusted-host would re-trust a
# host install dropped.
_PINNED_PIP_CONFIG_ACCUMULATING = frozenset({"only-binary"})
_PINNED_PIP_ENV_ACCUMULATING = frozenset(
    f"PIP_{option.upper().replace('-', '_')}" for option in _PINNED_PIP_CONFIG_ACCUMULATING
)

_PINNED_PIP_CONFIG_LISTING: "bytes | None" = None

# Failures are not memoised (transient misses, pip appearing later); only a hang is budgeted.
_PINNED_PIP_CONFIG_TIMEOUT = 30
_PINNED_PIP_CONFIG_ATTEMPTS = 2


def _pip_subcommand_of(cmd: "list[str]") -> str:
    """The pip subcommand ``cmd`` runs, or the default when it is not a pip command."""
    for name in ("install", "download", "wheel"):
        if _is_pip_subcommand(cmd, (name,)):
            return name
    return _PINNED_PIP_CONFIG_DEFAULT_SECTION


def _pinned_pip_config_overrides(
    subcommand: str = _PINNED_PIP_CONFIG_DEFAULT_SECTION,
) -> "dict[str, str]":
    """pip's configured transport and binary policy, as PIP_ environment variables.

    ONLY a successful read is memoised, or a transient miss would cost the operator their
    cert and proxy for the rest of the run, and a venv with no pip yet could never answer
    once it has one. A HANG is budgeted instead, so a wedged pip costs the run one timeout
    budget rather than one per pinned command. Empty when pip cannot answer, the normal
    case early in a fresh venv. `:env:` rows are skipped: the child inherits those.
    """
    global _PINNED_PIP_CONFIG_LISTING, _PINNED_PIP_CONFIG_ATTEMPTS
    if _PINNED_PIP_CONFIG_LISTING is not None:
        return _parse_pinned_pip_config(_PINNED_PIP_CONFIG_LISTING, subcommand)
    if _PINNED_PIP_CONFIG_ATTEMPTS <= 0:
        return {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            # Dictate the child's encoding (see _decode_pip_output) so a non-ASCII cert path survives.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout = _PINNED_PIP_CONFIG_TIMEOUT,
            **_windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        _PINNED_PIP_CONFIG_ATTEMPTS -= 1
        return {}
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    _PINNED_PIP_CONFIG_LISTING = result.stdout or b""
    return _parse_pinned_pip_config(_PINNED_PIP_CONFIG_LISTING, subcommand)


def _decode_pip_output(raw: bytes) -> str:
    r"""`pip config list` bytes as text.

    The child is told to write UTF-8 (see PYTHONIOENCODING above). The fallback covers a
    listing produced some other way, a pip old enough to ignore that variable among them,
    where the locale codec is the child's encoding since both share a locale. Sniffing
    replaces neither: cp1252 bytes can form valid UTF-8, so it is not recoverable after.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), "replace")


def _parse_pinned_pip_config(
    stdout: bytes, subcommand: str = _PINNED_PIP_CONFIG_DEFAULT_SECTION
) -> "dict[str, str]":
    """`pip config list` output, filtered to the allowlist, as PIP_ variables.

    Reads `[global]` plus `[<subcommand>]`, the two sections pip itself would apply to that
    command, with the command section winning. Resolved by position rather than by the
    order the listing prints in. An unparseable line is skipped, never fatal.
    """
    sections = (_PINNED_PIP_CONFIG_GLOBAL_SECTION, subcommand)
    found: dict[str, dict[str, str]] = {}
    for line in _decode_pip_output(stdout).splitlines():
        name, separator, raw = line.partition("=")
        if not separator or name.startswith(":env:"):
            continue
        section, _, option = name.strip().rpartition(".")
        if section not in sections:
            continue
        if option not in _PINNED_PIP_CONFIG_KEEP_KEYS:
            continue
        try:
            value = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            continue
        separator_for_key = _PINNED_PIP_CONFIG_SEPARATORS.get(option)
        if separator_for_key is None:
            text = str(value).strip()
        else:
            text = separator_for_key.join(str(value).split())
        if text:
            found.setdefault(option, {})[section] = text
    overrides: dict[str, str] = {}
    for option, by_section in found.items():
        separator_for_key = _PINNED_PIP_CONFIG_SEPARATORS.get(option)
        present = [by_section[name] for name in sections if name in by_section]
        if option not in _PINNED_PIP_CONFIG_ACCUMULATING:
            value = present[-1]  # the command's section overrides global
        else:
            # Keep order and duplicates: pip applies entries in order, so a re-add after `:none:` must stay last.
            parts = [part for chunk in present for part in chunk.split(separator_for_key) if part]
            value = separator_for_key.join(parts)
        overrides[f"PIP_{option.upper().replace('-', '_')}"] = value
    return overrides


def pip_install_try(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
    force_pip: bool = False,
) -> bool:
    """Like pip_install but returns False on failure instead of exiting.
    For optional installs that have a follow-up fallback.

    ``req`` goes through ``_effective_requirements`` exactly as in pip_install, so a file
    installed through either entry point is the same file the install-manifest gate audits.
    """
    # This installs torch too (Windows AMD ROCm trio), so drop the memoized classification.
    _invalidate_torch_runtime_probe()
    # Counted before the command: a failed install can still have moved metadata.
    _count_install_action()
    constraint_args_pip: list[str] = []
    constraint_args_uv: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args_pip = ["-c", str(CONSTRAINTS)]
        constraint_args_uv = ["-c", _uv_safe_path(CONSTRAINTS)]

    actual_req = req
    temp_reqs: list[Path] = []
    if req is not None:
        actual_req, temp_reqs = _effective_requirements(req)
    req_args_pip: list[str] = []
    req_args_uv: list[str] = []
    if actual_req is not None:
        req_args_pip = ["-r", str(actual_req)]
        req_args_uv = ["-r", _uv_safe_path(actual_req)]

    if USE_UV and not force_pip:
        cmd, env = _pinned_cmd_and_env(_build_uv_cmd(args) + constraint_args_uv + req_args_uv)
    else:
        cmd, env = _pinned_cmd_and_env(_build_pip_cmd(args) + constraint_args_pip + req_args_pip)

    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    try:
        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            env = env,
        )
    finally:
        for temp_req in temp_reqs:
            temp_req.unlink(missing_ok = True)
    if result.returncode == 0:
        # As pip_install below: `nobuild` only catches a build that reaches the log.
        if VERBOSE and result.stdout:
            _safe_print(_redact_install_output(result.stdout))
        return True
    if VERBOSE and result.stdout:
        # pip/uv echo index URLs (credentials included) in failure output.
        _safe_print(_redact_install_output(result.stdout))
    return bool(
        _mirror_retry(
            args,
            result.stdout or b"",
            lambda *retry: pip_install_try(
                label, *retry, req = req, constrain = constrain, force_pip = force_pip
            ),
        )
    )


_PYTORCH_DEFAULT_WHL = "https://download.pytorch.org/whl"
_MIRROR_TRANSPORT_ERROR = re.compile(
    r"error sending request|timed out|network timeout|connection (reset|refused|closed|aborted)|"
    r"broken pipe|dns error|failed to lookup address|name resolution|nodename nor servname|"
    r"network is unreachable|error decoding response body|end of file before message length|"
    r"unexpected eof|tls handshake|sslerror|"
    r"certificate verify failed|server error|service unavailable|bad gateway|gateway time-?out|"
    r"too many requests|max retries exceeded|remotedisconnected|incompleteread",
    re.IGNORECASE,
)
_MIRROR_HOST_NAMES = (
    ("torch", re.compile(r"download(-r2)?\.pytorch\.org")),
    ("pypi", re.compile(r"pypi\.org|pythonhosted\.org")),
)
_MIRROR_NAMES = {"torch": "download.pytorch.org", "pypi": "PyPI", "unsynced": "The PyPI mirror"}
_MIRROR_UNSYNCED = re.compile(
    r"only \S+ (.* )?(is|are) available|no versions? of|not found in the package registry|"
    r"could not find a version that satisfies|no matching distribution found",
    re.IGNORECASE,
)
_failed_install_output = b""


def _mirror_retry(args: "tuple[str, ...]", output: bytes, rerun) -> "bool | None":
    """Reruns a failed install once through the mirror of the host its output shows failing.

    The installer's probe exports ``_UNSLOTH_MIRROR_SPARE`` as ``host|VAR=URL|...`` entries for
    the hosts it left on their defaults; each gets one rerun, and later installs keep the mirror
    only when it worked. None when there is no such host.
    """
    global _PYTORCH_WHL_BASE
    if not os.environ.get("_UNSLOTH_MIRROR_SPARE", "").strip():
        return None
    text = output.decode("utf-8", "replace")
    torch = any(_PYTORCH_DEFAULT_WHL in arg for arg in args)
    if not _MIRROR_TRANSPORT_ERROR.search(text):
        if _is_pinned_index_cmd(args) or "--no-index" in args or not _MIRROR_UNSYNCED.search(text):
            return None
        host = "unsynced"
    else:
        pinned = _is_pinned_index_cmd(args) or any(
            arg in ("--find-links", "--no-index") or "://" in arg for arg in args
        )
        host = next((name for name, pattern in _MIRROR_HOST_NAMES if pattern.search(text)), None)
        if host is None and not re.search(r"https?://", text):
            host = "torch" if torch else "pypi"
        if host is None or (host == "torch" and not torch) or (host == "pypi" and pinned):
            return None
    spare = os.environ.get("_UNSLOTH_MIRROR_SPARE", "").split()
    entry = next((e for e in spare if e.split("|", 1)[0] == host), None)
    if entry is None:
        return None
    os.environ["_UNSLOTH_MIRROR_SPARE"] = " ".join(e for e in spare if e != entry)
    pairs = dict(pair.split("=", 1) for pair in entry.split("|")[1:])
    _step(
        "mirror",
        f"{_MIRROR_NAMES[host]} failed; retrying through {next(iter(pairs.values()))}",
        _cyan,
    )
    saved = ({name: os.environ.get(name) for name in pairs}, _PYTORCH_WHL_BASE)
    os.environ.update(pairs)
    if host == "torch" and _PYTORCH_WHL_BASE == _PYTORCH_DEFAULT_WHL:
        _PYTORCH_WHL_BASE = pairs["UNSLOTH_PYTORCH_MIRROR"].rstrip("/")
    ok = False
    try:
        ok = rerun(*(arg.replace(_PYTORCH_DEFAULT_WHL, _PYTORCH_WHL_BASE) for arg in args))
    finally:
        if not ok:
            for name, value in saved[0].items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _PYTORCH_WHL_BASE = saved[1]
    return ok


def pip_install(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    """Build and run a pip install command (uses uv when available, falls back to pip)."""
    try:
        _pip_install_once(label, *args, req = req, constrain = constrain)
    except SystemExit:
        rerun = (
            lambda *retry: _pip_install_once(label, *retry, req = req, constrain = constrain) or True
        )
        if not _mirror_retry(args, _failed_install_output, rerun):
            raise


def _pip_install_once(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    global _failed_install_output
    _failed_install_output = b""
    _invalidate_torch_runtime_probe()
    _count_install_action()
    constraint_args_pip: list[str] = []
    constraint_args_uv: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args_pip = ["-c", str(CONSTRAINTS)]
        constraint_args_uv = ["-c", _uv_safe_path(CONSTRAINTS)]

    # _effective_requirements drops torchcodec where no wheel exists; audit that file, not the original.
    actual_req = req
    temp_reqs: list[Path] = []
    if req is not None:
        actual_req, temp_reqs = _effective_requirements(req)
    req_args_pip: list[str] = []
    req_args_uv: list[str] = []
    if actual_req is not None:
        req_args_pip = ["-r", str(actual_req)]
        req_args_uv = ["-r", _uv_safe_path(actual_req)]

    try:
        if USE_UV:
            uv_cmd, uv_env = _pinned_cmd_and_env(
                _build_uv_cmd(args) + constraint_args_uv + req_args_uv
            )
            if VERBOSE:
                _safe_print(f"   {label}...")
            result = subprocess.run(
                uv_cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                env = uv_env,
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                # Echo under UNSLOTH_VERBOSE like install.sh: clean-machine-assert.sh's `nobuild` greps the log for
                # uv's "Building <pkg>==<ver>". Redacted: uv echoes credentialed URLs.
                if VERBOSE and result.stdout:
                    _safe_print(_redact_install_output(result.stdout))
                return
            _failed_install_output = result.stdout or b""
            if _woa_overrides_are_load_bearing():
                _step("error", f"{label} failed and pip cannot stand in for it", _red)
                _safe_print(
                    _red(
                        "   The Windows on ARM stack resolves through UV_OVERRIDE, which pip has no "
                        "equivalent for: overrides REPLACE a requirement, and pip constraints can "
                        "only narrow one."
                    )
                )
                _safe_print(
                    _red(
                        "   Falling back here would honour the released torch cap, which no "
                        "win_arm64 CUDA wheel satisfies, and pull back the packages that have no "
                        "win_arm64 build at all -- downgrading a working CUDA torch or failing "
                        "later, with nothing to say why."
                    )
                )
                _safe_print(_red("   Install uv and re-run, or re-run install.ps1."))
                _report_failed_command(label, result)
            if _TORCH_FREEZE_ACTIVE:
                _step("error", f"{label} failed and pip cannot stand in for it", _red)
                _safe_print(
                    _red(
                        "   torch is held on its installed release through UV_OVERRIDE, which pip "
                        "ignores: a pip fallback would downgrade it to the released cap."
                    )
                )
                _report_failed_command(label, result)
            _safe_print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                _safe_print(_redact_install_output(result.stdout))

        elif _TORCH_FREEZE_ACTIVE:
            _step("error", f"{label} needs uv to keep the installed torch", _red)
            _safe_print(
                _red(
                    "   Install uv and re-run, or set UNSLOTH_TORCH_UPGRADE=1 and re-run install.sh."
                )
            )
            sys.exit(1)
        elif _woa_overrides_are_load_bearing():
            _step("error", f"{label} needs uv on the Windows on ARM stack", _red)
            _safe_print(
                _red(
                    "   The native ARM64 resolve depends on UV_OVERRIDE, which pip cannot express. "
                    "Install uv and re-run, or re-run install.ps1."
                )
            )
            sys.exit(1)

        pip_cmd, pip_env = _pinned_cmd_and_env(
            _build_pip_cmd(args) + constraint_args_pip + req_args_pip
        )
        pip_label = f"{label} (pip)" if USE_UV else label
        result = run(pip_label, pip_cmd, check = False, env = pip_env)
        if result.returncode != 0:
            _failed_install_output += result.stdout or b""
            # Retry once, only after clearing something pip named as unremovable.
            cleared = _purge_recordless_distributions(result.stdout)
            if not cleared:
                _report_failed_command(pip_label, result)
            _step(_LABEL, f"cleared half-written {', '.join(cleared)}, retrying...", _dim)
            run(pip_label, pip_cmd, env = pip_env)
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


# Apple's Command Line Tools shim, which pops a GUI install dialog when run without a toolchain.
_CLT_GIT_SHIM = "/usr/bin/git"


def _apple_silicon_hardware() -> bool:
    """Whether the MACHINE is Apple Silicon, even when this Python runs under Rosetta.

    install.sh's _MAC_ROSETTA: an x86_64 shell on an arm64 Mac reports x86_64, while
    hw.optional.arm64 stays 1. Intel Macs keep probing /usr/bin/git by running it, as install.sh
    does, because a CI Intel image ships a working one there.
    """
    if not IS_MACOS:
        return False
    if platform.machine() == "arm64":
        return True
    try:
        answer = subprocess.run(
            ["sysctl", "-in", "hw.optional.arm64"],
            capture_output = True,
            text = True,
            timeout = 10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return answer.strip() == "1"


def _is_unarmed_clt_git_shim(exe: str) -> bool:
    """`exe` is Apple Silicon's /usr/bin/git shim and `xcode-select -p` names no toolchain."""
    if exe != _CLT_GIT_SHIM or not _apple_silicon_hardware():
        return False
    try:
        return (
            subprocess.run(
                ["xcode-select", "-p"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 30,
            ).returncode
            != 0
        )
    except (OSError, subprocess.SubprocessError):
        return True


def _has_working_git() -> bool:
    """Match install.sh's _has_working_git: on PATH *and* actually runnable.

    A present-but-broken git (a bare xcrun shim) counts as missing there too. Testing
    only shutil.which disagreed, so the installer promised to skip the git+https triton
    requirement and then tried to fetch it anyway.
    """
    exe = shutil.which("git")
    if exe is None:
        return False
    # Without the CLT, /usr/bin/git is Apple's shim and running it pops the install dialog; answer from
    # the path as install.sh does. Homebrew / Xcode.app gits are still run.
    if _is_unarmed_clt_git_shim(exe):
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


# The MLX stack, one place; test_mlx_install.py compares it with utils/mlx_repair.py's
# _MLX_INSTALL_SPECS.
_MLX_PINS: tuple[str, ...] = ("mlx==0.32.3", "mlx-metal==0.32.3", "mlx-lm==0.31.3")
_MLX_VLM_SPEC = "mlx-vlm>=0.4.4,<=0.7.4"
# Exact: llguidance.mlx / llguidance.hf are the API grammar_constraint.py binds to.
_LLGUIDANCE_PIN = "llguidance==1.8.0"
_MLX_NAMES: tuple[str, ...] = tuple(spec.partition("==")[0] for spec in _MLX_PINS) + ("mlx-vlm",)


def _mlx_vlm_spec_for_installed_zoo() -> str:
    """_MLX_VLM_SPEC, intersected with the range the INSTALLED unsloth-zoo declares.

    This step runs before the core phase, and on a fresh install (SKIP_STUDIO_BASE=1) the core
    phase is skipped entirely, so the zoo install.sh already put down is the one that stays.
    mlx-vlm 0.7.1 passes `cache` to gated_delta_update and a zoo predating that keyword does not
    accept it, so admitting 0.7.1 beside such a zoo raises TypeError at the first Qwen3.5 VLM
    training step, after mlx_stack_available() has cleared the chat-only gate. Reading what the
    installed zoo itself declares keeps the two in step without naming a zoo version here, and it
    widens on its own once a zoo declaring 0.7.1 is installed. Mirrors utils/mlx_repair.py's
    _install_packages, which does the same for the unattended self-heal.
    """
    try:
        from importlib.metadata import requires
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        return _MLX_VLM_SPEC
    try:
        declared = requires("unsloth_zoo") or ()
    except Exception:  # noqa: BLE001 - not installed, or unreadable metadata
        return _MLX_VLM_SPEC
    for raw in declared:
        try:
            requirement = Requirement(raw)
        except Exception:  # noqa: BLE001 - a requirement string packaging cannot parse
            continue
        if canonicalize_name(requirement.name) == "mlx-vlm" and str(requirement.specifier):
            return f"{_MLX_VLM_SPEC},{requirement.specifier}"
    return _MLX_VLM_SPEC


def _mlx_stack_is_current() -> bool:
    """Whether the three exact pins and mlx-vlm's range are all already satisfied.

    Without this the step ran `--upgrade` on every update and fetched ~60 MB of wheels
    it then discarded, because the pins were already exact and the upgrade only moved
    mlx-vlm inside a range it was already in.
    """
    if not all(_exact_distribution_spec_is_installed(spec) for spec in _MLX_PINS):
        return False
    installed = _installed_distribution_version("mlx-vlm")
    if not installed:
        return False
    try:
        from packaging.requirements import Requirement

        # The narrowed spec: an older zoo beside an installed 0.7.1 would read the static range as satisfied.
        if not Requirement(_mlx_vlm_spec_for_installed_zoo()).specifier.contains(
            installed,
            prereleases = True,
        ):
            return False
    except Exception:  # noqa: BLE001 - no packaging, or a version it cannot parse
        return False
    # Pins satisfied is not closure satisfied; this step installs with deps, so audit the closure.
    return not _mlx_closure_unmet()


def _overridden_project_names() -> "set[str]":
    """Canonical names the bundled macOS arm64 override file replaces every requirement on."""
    names: set[str] = set()
    try:
        text = _MLX_OVERRIDES.read_text(encoding = "utf-8-sig")
    except (OSError, ValueError, UnicodeDecodeError):
        return names
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        head = re.split(r"[<>=!~;\[ ]", line, maxsplit = 1)[0].strip()
        if head:
            names.add(install_manifest._canonical(head))
    return names


def _mlx_closure_unmet() -> bool:
    """Whether anything in the MLX pins' installed closure is missing or outside its pin."""
    handle = None
    try:
        import tempfile as _tempfile  # noqa: PLC0415

        # Close mkstemp's descriptor: it leaks per update and on Windows blocks the unlink.
        _fd, _name = _tempfile.mkstemp(prefix = "unsloth-mlx-", suffix = ".txt", text = True)
        os.close(_fd)
        handle = Path(_name)
        handle.write_text(
            "\n".join([*_MLX_PINS, _mlx_vlm_spec_for_installed_zoo()]) + "\n",
            encoding = "utf-8",
        )
        unmet = install_manifest.closure_unmet_requirements(handle, _installed_index())
        # An override replaces every requirement on its package, so compare against the override; an absent
        # overridden package still counts.
        overridden = _overridden_project_names()
        if overridden:
            unmet = [
                entry
                for entry in unmet
                if " " not in entry
                or install_manifest._canonical(entry.split(" ", 1)[0]) not in overridden
            ]
    except Exception:  # noqa: BLE001 - an audit that cannot run is a reason to run the step
        return True
    finally:
        if handle is not None:
            try:
                handle.unlink(missing_ok = True)
            except OSError:
                pass
    if unmet and VERBOSE:
        _note(f"MLX stack: {unmet[0]} is not satisfied -- running the step")
    return bool(unmet)


_MLX_HEALTH_PROBE = (
    "import json, sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "from utils.mlx_repair import mlx_stack_blockers;"
    "print(json.dumps(mlx_stack_blockers()))"
)


# Ranged deps the probe imports through mlx_lm / mlx_vlm; one can move with every step satisfied.
_MLX_IMPORTED_DEPENDENCIES = (
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "pillow",
    "protobuf",
    "sentencepiece",
)


def _mlx_health_fingerprint() -> dict:
    """What a recorded MLX verdict is only valid for."""
    return {
        # The narrowed spec, so installing a different zoo invalidates a recorded verdict.
        "pins": list(_MLX_PINS) + [_mlx_vlm_spec_for_installed_zoo()],
        "python": _installer_python_tag(),
        "mlx_vlm": _installed_distribution_version("mlx-vlm") or "",
        # Versions as installed; an older record without this key is probed once and rewritten.
        "imports": {
            name: _installed_distribution_version(name) or "" for name in _MLX_IMPORTED_DEPENDENCIES
        },
    }


def _mlx_payload_present() -> bool:
    """The MLX stack's payload is on disk as its RECORDs describe it, without importing it.

    The recorded verdict below is keyed on pins and interpreter, which a payload deleted or
    truncated after the pass leaves unchanged, and find_spec sees only the top-level
    directory. So every file each RECORD names is checked for presence and size (bytecode
    excepted, it is regenerated): a few hundred stats, well under the import the probe pays.
    """
    try:
        import importlib.metadata
        import importlib.util

        for name in ("mlx", "mlx_lm", "mlx_vlm", "transformers"):
            if importlib.util.find_spec(name) is None:
                return False
        for dist_name in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm"):
            # No RECORD is not "present".
            if _recorded_payload_damaged(dist_name) is not False:
                return False
        # A truncated tokenizers keeps its version and metadata; only the import notices.
        for dist_name in _MLX_IMPORTED_DEPENDENCIES:
            if not _installed_distribution_version(dist_name):
                continue
            if _recorded_payload_damaged(dist_name) is not False:
                return False
        return True
    except Exception:  # noqa: BLE001 - not finding it is the probe's job to explain
        return False


def _recorded_payload_damaged(dist_name: str) -> "bool | None":
    """Whether a file the distribution's RECORD names is gone or has another size.

    None when the distribution has no readable RECORD, which is not evidence of an intact
    payload. Read from RECORD rows, not Distribution.files: CPython 3.13 drops paths that no
    longer exist from `files`, so a deleted file is never found through it. Bytecode is left
    out (recompiled after install); rows without a size say nothing.
    """
    import csv
    import io

    dist = importlib.metadata.distribution(dist_name)
    record = dist.read_text("RECORD")
    if not record:
        return None
    for row in csv.reader(io.StringIO(record)):
        if len(row) < 3 or not row[0] or row[0].endswith(".pyc") or not row[2]:
            continue
        try:
            size = int(row[2])
        except ValueError:
            continue
        try:
            if os.stat(dist.locate_file(row[0])).st_size != size:
                return True
        except OSError:
            return True
    return False


def _report_mlx_stack_health(skipped: bool = False) -> None:
    """Name what would keep Train off on this Apple Silicon host, if anything.

    Advisory only: the install has succeeded, chat still works and the self-heal gets another
    go at startup. It just must not be silent, which is the whole of "Train is blacked out
    after an update".

    Run out of process, since a half-installed mlx can abort rather than raise. That costs a
    full torch and mlx import, so the recorded verdict stands when the MLX step was skipped,
    nothing else installed anything, and the last run recorded a healthy stack for these exact
    pins and this interpreter. Anything else runs the probe.
    """
    fingerprint = _mlx_health_fingerprint()
    recorded = (_PASS_EVIDENCE or {}).get("mlx_health")
    if (
        skipped
        # A later step moving transformers or tokenizers can break the import with pins unchanged.
        and _INSTALL_ACTIONS == 0
        and isinstance(recorded, dict)
        and recorded.get("ok") is True
        and recorded.get("pins") == fingerprint["pins"]
        and recorded.get("python") == fingerprint["python"]
        and recorded.get("mlx_vlm") == fingerprint["mlx_vlm"]
        and recorded.get("imports") == fingerprint["imports"]
        and _mlx_payload_present()
    ):
        _step("mlx", "training stack ready")
        # Written back so the record does not age out of the manifest.
        install_manifest.update_manifest(mlx_health = {**fingerprint, "ok": True})
        return
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
    # After the manifest write, so a kill during the 180 s probe cannot lose a finished install.
    install_manifest.update_manifest(mlx_health = {**fingerprint, "ok": not blockers})
    if not blockers:
        _step("mlx", "training stack ready")
        return
    _step("mlx", "Train and Export will stay off until this is resolved:", _cyan)
    for blocker in blockers:
        _step("", blocker, _cyan)


# Idempotent dependency pass: skip only when (a) the last run recorded this exact work, (b) inputs
# are byte-identical, and (c) a cheap on-disk check of the output passes. When in doubt, do the work.

_FULL_DEPS_ENV = "UNSLOTH_STUDIO_FULL_DEPS"

# Real install/uninstall commands this pass ran; `pip check` and the metadata patch are gated on it.
_INSTALL_ACTIONS = 0
_PASS_EVIDENCE: "dict | None" = None
# False when a peer is already inside a pass on this venv, which refuses the evidence below.
_PASS_UNCONTENDED = True
# What this pass did per step. Only "ran" and "skipped" let the NEXT run skip.
_STEP_RESULTS: "dict[str, str]" = {}
_AUDITED_STEPS: "dict[str, Path]" = {}
_CONSTRAINTS_CACHE: "tuple[int, list[str]] | None" = None


def _count_install_action() -> None:
    """Called by every command that can change what is installed."""
    global _INSTALL_ACTIONS
    _INSTALL_ACTIONS += 1


def _full_deps_requested() -> bool:
    """The escape hatch. A skip nobody can turn off is a bug nobody can work around."""
    return os.environ.get(_FULL_DEPS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _record_step(key: str, result: str) -> None:
    _STEP_RESULTS[key] = result


def _closure_record() -> "dict[str, list[str]]":
    """What each audited with-deps step leaves unmet in its closure, after the pass.

    Recorded for the next run's gate: a requirement still unmet once its step has run is one
    the step cannot satisfy (sqlfluff 3.x pins click<=8.3.0 while huggingface-hub 1.23+ needs
    >=8.4.2), and re-running the step for it resolved nothing and needed the index. A skipped
    step carries the record it was skipped on. Audit failures are not recorded, or an
    unreadable environment would skip the step next time.
    """
    record: dict[str, list[str]] = {}
    # Under PIP_NO_DEPS or a foreign UV_OVERRIDE the unmet dependencies are deliberate.
    if _foreign_resolver_inputs() or _foreign_uv_override_in_effect():
        return record
    # A hand-edited manifest can carry anything here, and this must not raise.
    previous = (_PASS_EVIDENCE or {}).get("known_unmet")
    if not isinstance(previous, dict):
        previous = {}
    for key, req in _AUDITED_STEPS.items():
        # Runs while building write_manifest's arguments, after every install: must not raise.
        temps: list[Path] = []
        try:
            effective, temps = _effective_requirements(req)
            unmet = install_manifest.closure_unmet_requirements(effective, _installed_index())
        except Exception:  # noqa: BLE001 - nothing recorded means nothing ignored next time
            unmet = ["<audit failed>"]
        finally:
            for temp in temps:
                # A leftover temp file is cheaper than ending the pass before the manifest write.
                try:
                    temp.unlink(missing_ok = True)
                except OSError:
                    pass
        audited = not any(entry.startswith("<") for entry in unmet)
        # Only version conflicts can be known conflicts; an absent distribution is work this step does.
        unmet = [entry for entry in unmet if entry.startswith("<") or " " in entry]
        if _STEP_RESULTS.get(key) == "skipped":
            # Narrowed to what is still unmet, so a later loss does not hide behind the record.
            carried = list(previous.get(key) or []) if isinstance(previous.get(key), list) else []
            carried = [entry for entry in carried if " " in entry]
            if audited:
                carried = [entry for entry in carried if entry in unmet]
            if carried:
                record[key] = carried
            continue
        unmet = [entry for entry in unmet if not entry.startswith("<")]
        if unmet:
            record[key] = unmet
    return record


def _may_skip_on_evidence() -> bool:
    """Whether a step is allowed to answer from what is already on disk.

    The requirements steps ask through _requirements_satisfied. The three pin-shaped steps
    (torchao, MLX, torchcodec) compare against the installed distribution rather than a
    recorded digest, so they ask for themselves: otherwise a satisfied pin outranks every
    reason _plan_pass had for refusing evidence, including UNSLOTH_STUDIO_FULL_DEPS, a failed
    deep verify, and a moved interpreter, platform or torch flavour.

    _full_deps_requested is asked again rather than left to _plan_pass: these steps are also
    reachable from _ensure_expected_torch_flavor and the post-repair torchao re-selection, and
    the one switch a user is told to set has to hold on every path.
    """
    return _PASS_EVIDENCE is not None and not _full_deps_requested()


_FOREIGN_RESOLVER_ENV = (
    "UV_CONSTRAINT",
    "UV_BUILD_CONSTRAINT",
    "PIP_CONSTRAINT",
    "PIP_NO_DEPS",
    "UV_NO_DEPS",
)


def _foreign_resolver_inputs() -> list:
    """Environment inputs that change what a step installs and that no digest covers."""
    return [name for name in _FOREIGN_RESOLVER_ENV if (os.environ.get(name) or "").strip()]


def _foreign_uv_override_in_effect() -> bool:
    """Whether UV_OVERRIDE names a file other than the bundled macOS arm64 overrides."""
    value = (os.environ.get("UV_OVERRIDE") or "").strip()
    if not value:
        return False

    def _canonical(path: str) -> str:
        try:
            return os.path.normcase(os.path.realpath(path))
        except (OSError, ValueError):
            return path

    bundled = {_canonical(str(_MLX_OVERRIDES))}
    try:
        bundled.add(_canonical(_uv_safe_path(_MLX_OVERRIDES)))
    except Exception:  # noqa: BLE001 - the short-path form is an optimisation, not evidence
        pass
    return _canonical(value) not in bundled


def _refuse_evidence(reason: str) -> None:
    """Say why last run's evidence is not usable, then answer "none".

    Named under UNSLOTH_VERBOSE for the same reason as the closure audit's note: a
    refusal here turns every update into a full dependency pass, which is correct, slow,
    and otherwise indistinguishable from a pass that had no evidence to begin with.
    """
    if VERBOSE:
        _note(f"dependency pass evidence not used: {reason}")
    return None


def _plan_pass(package_name: str, local_repo: str, ci_source_overlay: str) -> "dict | None":
    """Last run's evidence, or None when nothing may be skipped.

    Must run BEFORE remove_manifest: the manifest is the only copy, and it is dropped
    up front precisely so a run killed mid-pass cannot leave a valid one behind.
    """
    # Consume the parked copy before the refusals: a forced pass killed part-way must not leave it as
    # evidence.
    manifest, manifest_error = None, False
    try:
        manifest = install_manifest.read_manifest()
        # setup.ps1 drops the live manifest first; the parked copy is evidence only, re-verified on disk.
        if not manifest:
            manifest = install_manifest.read_previous_manifest()
            if manifest:
                install_manifest.consume_previous_manifest()
    except Exception:  # noqa: BLE001 - an unreadable manifest is a full pass, never a crash
        manifest_error = True
    if _full_deps_requested():
        return _refuse_evidence("UNSLOTH_STUDIO_FULL_DEPS requested")
    # A peer pass is replacing these packages, so run every step.
    if not _PASS_UNCONTENDED:
        return _refuse_evidence("another install is already running in this venv")
    if local_repo or ci_source_overlay or package_name != "unsloth":
        return _refuse_evidence("development install shape")
    # A caller's own UV_OVERRIDE file is an input no digest covers.
    if _foreign_uv_override_in_effect():
        return _refuse_evidence("a caller-supplied UV_OVERRIDE is in effect")
    # Constraint env vars change installs without touching digested files; PIP_NO_DEPS leaves deliberate gaps.
    foreign = _foreign_resolver_inputs()
    if foreign:
        return _refuse_evidence(f"caller-supplied resolver input in effect: {', '.join(foreign)}")
    if manifest_error:
        return _refuse_evidence("manifest unreadable")
    if not manifest or manifest.get("schema") != install_manifest.MANIFEST_SCHEMA:
        return _refuse_evidence("no manifest, or a schema this build does not read")
    inputs = manifest.get("pass_inputs")
    results = manifest.get("step_results")
    if not isinstance(inputs, dict) or not isinstance(results, dict):
        return _refuse_evidence("manifest written by a build that recorded no pass inputs")
    if manifest.get("python") != platform.python_version():
        return _refuse_evidence(
            f"python moved ({manifest.get('python')} -> {platform.python_version()})"
        )
    if manifest.get("platform") != f"{sys.platform}-{platform.machine()}":
        return _refuse_evidence(
            f"platform moved ({manifest.get('platform')} -> {sys.platform}-{platform.machine()})"
        )
    # The version cannot tell a GIL 3.14 from a free-threaded one; the tag can.
    if manifest.get("installer_python_tag") != _installer_python_tag():
        return _refuse_evidence(
            f"interpreter ABI moved ({manifest.get('installer_python_tag')} -> "
            f"{_installer_python_tag()})"
        )
    # Absent is unknown, not False.
    if manifest.get("no_torch") is not bool(NO_TORCH):
        return _refuse_evidence(
            f"no-torch mode moved ({manifest.get('no_torch')} -> {bool(NO_TORCH)})"
        )
    if not NO_TORCH:
        # The flavour tag, not the index URL: only the tag is safe to keep on disk.
        try:
            expected = _recordable_torch_flavor_tag(_expected_torch_flavor_tag())
        except Exception:  # noqa: BLE001 - a probe that did not answer buys a full pass
            return _refuse_evidence("torch flavour probe did not answer")
        if (manifest.get("expected_torch_tag") or "") != expected:
            return _refuse_evidence(
                f"torch flavour moved ({manifest.get('expected_torch_tag')!r} -> {expected!r})"
            )
    # Last, being the only check that walks the filesystem.
    try:
        verified = install_manifest.verify_install(deep = True, manifest = manifest)
    except Exception as exc:  # noqa: BLE001
        return _refuse_evidence(f"deep verify raised {exc!r}")
    if not verified.get("ok"):
        return _refuse_evidence(
            f"deep verify failed ({verified.get('reason')}; missing {verified.get('missing')})"
        )
    return {
        "pass_inputs": inputs,
        "step_results": results,
        "pip_check_ok": manifest.get("pip_check_ok"),
        "mlx_health": manifest.get("mlx_health"),
        "known_unmet": manifest.get("known_unmet"),
        "known_unmet_index": manifest.get("known_unmet_index"),
        "bnb_rocm": manifest.get("bnb_rocm"),
        "bnb_rocm_asset": manifest.get("bnb_rocm_asset"),
    }


def _effective_requirements(req: Path) -> "tuple[Path, list[Path]]":
    """The file pip_install will really install from, plus temp copies to unlink.

    Shared with pip_install so the skip gate audits exactly what the install would do.
    extras.txt names torchcodec and three platforms filter it out; auditing the raw file
    would report it missing on every one of them and never skip anything again.
    """
    temps: list[Path] = []
    actual = req
    if IS_WINDOWS and WINDOWS_SKIP_PACKAGES:
        actual = _filter_requirements(actual, WINDOWS_SKIP_PACKAGES)
        temps.append(actual)
    if _is_win_arm64_interpreter():
        # Judged against the original file: the filtered copy lost the pinned rows.
        arm64_skip = _windows_arm64_skip_packages(req)
        if arm64_skip:
            actual = _filter_requirements(actual, arm64_skip)
            temps.append(actual)
    if NO_TORCH and NO_TORCH_SKIP_PACKAGES:
        actual = _filter_requirements(actual, NO_TORCH_SKIP_PACKAGES)
        temps.append(actual)
    if PLATFORM_LACKS_TORCHCODEC_WHEEL and not _wheelhouse_hosts("torchcodec"):
        # No torchcodec wheel on these hosts; step 13b installs a wheelhouse copy itself.
        actual = _filter_requirements(actual, {"torchcodec"})
        temps.append(actual)
    return actual, temps


def _pass_input_key(path: Path) -> "str | None":
    """A requirements file as the manifest names it: relative to REQ_ROOT, posix."""
    try:
        return Path(path).resolve().relative_to(REQ_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _violated_constraints() -> "list[str]":
    """Constrained distributions currently outside their pin.

    Recomputed whenever something was installed: a constraint the last step satisfied
    must not keep the next one running, and one it broke must not be missed. Cached
    otherwise, because every gated step asks.
    """
    global _CONSTRAINTS_CACHE
    if _CONSTRAINTS_CACHE is None or _CONSTRAINTS_CACHE[0] != _INSTALL_ACTIONS:
        try:
            found = install_manifest.violated_constraints(CONSTRAINTS)
        except Exception:  # noqa: BLE001 - unreadable is a reason to install, not to crash
            found = ["<unreadable>"]
        _CONSTRAINTS_CACHE = (_INSTALL_ACTIONS, found)
    return _CONSTRAINTS_CACHE[1]


_CLOSURE_INDEX_CACHE: "tuple[int, dict | None] | None" = None


def _installed_index_digest() -> "str | None":
    """One sha256 over (name, version) of everything installed: what a recorded conflict
    is evidence about. None when the index cannot be read, which matches no record."""
    import hashlib

    index = _installed_index()
    if index is None:
        return None
    digest = hashlib.sha256()
    for name in sorted(index):
        digest.update(f"{name}=={index[name][0]}\n".encode("utf-8"))
    return digest.hexdigest()


def _installed_index() -> "dict | None":
    """This venv's installed distributions, read once per install action.

    Every with-deps step walks the same site-packages, and building the index costs one
    pass over every dist-info; `distribution(name)` inside the walk would re-read one per
    node. Invalidated the moment something is installed, like _violated_constraints.
    """
    global _CLOSURE_INDEX_CACHE
    if _CLOSURE_INDEX_CACHE is None or _CLOSURE_INDEX_CACHE[0] != _INSTALL_ACTIONS:
        # importlib.metadata caches on mtime (one-second granularity on some filesystems).
        importlib.invalidate_caches()
        try:
            index = install_manifest.installed_dependency_index()
        except Exception:  # noqa: BLE001 - None is "cannot audit", which installs
            index = None
        _CLOSURE_INDEX_CACHE = (_INSTALL_ACTIONS, index)
    return _CLOSURE_INDEX_CACHE[1]


def _inputs_unchanged(keys: "list[str]") -> bool:
    recorded = (_PASS_EVIDENCE or {}).get("pass_inputs") or {}
    for key in keys:
        # Membership first: a missing key compared against None would read as unchanged.
        if key not in recorded:
            return False
        current = install_manifest.digest_file(REQ_ROOT / key)
        if current is None or recorded[key] != current:
            return False
    return True


def _local_plugin_payload_is_damaged(dist_name: str) -> bool:
    """Whether the installed seed plugin has files RECORD names and disk does not.

    Errors read as damaged, unlike install_manifest's own convention: this decides
    whether to skip a pip install that takes well under a second from a local path, so
    the conservative answer is cheap here and is not on the fast path for anything else.
    """
    try:
        return bool(install_manifest.damaged_payload_files(dist_name, limit = 1))
    except Exception:  # noqa: BLE001
        return True


def _refuse_step(key: str, reason: str) -> bool:
    """Say why one requirements step cannot be skipped, then answer False."""
    if VERBOSE:
        _note(f"{key}: {reason} -- running the step")
    return False


_INCLUDE_FLAGS = ("-r", "--requirement", "-c", "--constraint")


def _includes_another_requirements_file(req: Path) -> bool:
    """Whether *req* pulls in a second file the pass_inputs digests do not cover.

    Unreadable counts as "yes": a file this cannot read is one whose contents the gate cannot
    stand behind either, and the step running is the safe answer.
    """
    try:
        text = req.read_text(encoding = "utf-8-sig")
    except (OSError, ValueError, UnicodeDecodeError):
        return True
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split()[0].split("=", 1)[0]
        if head in _INCLUDE_FLAGS or any(
            line.startswith(flag) for flag in ("-r", "-c", "--requirement", "--constraint")
        ):
            return True
    return False


def _requirements_satisfied(
    req: Path,
    *,
    no_deps: bool,
    label: str = "",
    constrain: bool = True,
) -> bool:
    """Whether one requirements step can be skipped, on this run's evidence.

    *label* is the caller's bookkeeping only; the manifest key is the file's path under
    REQ_ROOT, so two steps cannot collide and a renamed file cannot inherit another's record.

    *no_deps* decides whether the dependency closure is audited, and is keyword-only and
    required on purpose: a new step that forgot to answer would get the weaker check.
    """
    if _PASS_EVIDENCE is None:
        return False
    key = _pass_input_key(req)
    if key is None:
        return _refuse_step(str(req), "not under the requirements root")
    # (a) the previous run recorded that it did this exact work
    if (_PASS_EVIDENCE.get("step_results") or {}).get(key) not in ("ran", "skipped"):
        return _refuse_step(key, "last run did not record this step")
    # (b) the inputs are byte-identical, constraints and macOS arm64 overrides included.
    keys = [key]
    if constrain:
        keys.append("single-env/constraints.txt")
    if IS_MAC_ARM:
        keys.append("single-env/overrides-darwin-arm64.txt")
    if not _inputs_unchanged(keys):
        return _refuse_step(key, "an input file changed")
    # (c) the output is still on disk.
    importlib.invalidate_caches()
    # Inside the try: a tree replaced mid-pass means run the step, not end the update.
    temps: list[Path] = []
    try:
        effective, temps = _effective_requirements(req)
        if _includes_another_requirements_file(effective):
            # The digest does not cover -r includes, so refuse; no shipped file uses one.
            return _refuse_step(key, "the file includes another requirements file")
        missing = install_manifest.missing_requirements(effective)
        if missing:
            return _refuse_step(key, f"not installed or outside the pin: {missing[:5]}")
        # ...and the output's own dependencies (mammoth stays satisfied with cobble gone). Not for --no-deps.
        if not no_deps:
            unmet = install_manifest.closure_unmet_requirements(effective, _installed_index())
            # Unmet entries the last pass left are not missing work, but only on the same installed set.
            known: set = set()
            if _PASS_EVIDENCE.get("known_unmet_index") == _installed_index_digest():
                known = set((_PASS_EVIDENCE.get("known_unmet") or {}).get(key) or [])
            unmet = [entry for entry in unmet if entry not in known]
            if unmet:
                # Named under UNSLOTH_VERBOSE: an audit that can never pass makes every update a full pass.
                if VERBOSE:
                    _note(f"{key}: {unmet[0]} is not satisfied -- running the step")
                return False
    except Exception as exc:  # noqa: BLE001
        return _refuse_step(key, f"audit raised {exc!r}")
    finally:
        for temp in temps:
            try:
                temp.unlink(missing_ok = True)
            except OSError:
                pass
    if constrain and _violated_constraints():
        return _refuse_step(key, f"constraints violated: {_violated_constraints()[:5]}")
    return True


def _skip_step(
    req: Path,
    progress_label: str,
    *,
    no_deps: bool,
    constrain: bool = True,
    extra_check = None,
    superseded: bool = False,
) -> bool:
    """Announce one requirements step and say whether it is already satisfied.

    The progress slot is spent either way, so the denominator does not depend on how much of
    the install was already there.

    *extra_check* is an on-disk predicate for a file importlib.metadata cannot fully answer:
    today only triton-kernels.txt, whose git ref no version reflects.

    *superseded* says a LATER step has already put a different build of this distribution in
    place, one this file's version pin cannot describe. The requirement really is unsatisfied and
    must still not run: running it would undo the step that supersedes it, which would then redo
    itself, so every pass reinstalls twice and no update is ever a no-op. Deliberately separate
    from *extra_check*, which can only make an answer stricter.
    """
    key = _pass_input_key(req) or str(req)
    if not no_deps and _pass_input_key(req) is not None:
        # Registered even when skipped, so the first pass records what its closure cannot satisfy.
        _AUDITED_STEPS[key] = req
    if superseded:
        _progress(f"{progress_label} (superseded, skipped)")
        _record_step(key, "skipped")
        return True
    satisfied = _requirements_satisfied(req, no_deps = no_deps, constrain = constrain)
    if satisfied and extra_check is not None:
        satisfied = bool(extra_check())
    _progress(f"{progress_label} (satisfied, skipped)" if satisfied else progress_label)
    _record_step(key, "skipped" if satisfied else "ran")
    return satisfied


def _direct_reference_in_requirements(req: Path) -> "tuple[str, str, str] | None":
    """``(url, requested revision, subdirectory)`` for a git requirement, or None.

    The fragment is NOT an inline comment here: ``#subdirectory=`` is part of the URL,
    and a different subdirectory is a different package.
    """
    # ValueError too: a requirements file that is not UTF-8 raises UnicodeDecodeError here.
    try:
        lines = req.read_text(encoding = "utf-8-sig").splitlines()
    except (OSError, ValueError):
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "git+" not in stripped:
            continue
        rest = stripped.partition("git+")[2]
        base, _, fragment = rest.partition("#")
        # The last "@" in the PATH: git+ssh userinfo has an "@" and branches can contain "/".
        scheme, separator, remainder = base.partition("://")
        netloc, slash, path = remainder.partition("/") if separator else ("", "", base)
        cut = path.rfind("@")
        if cut == -1:
            revision = ""
        else:
            revision = path[cut + 1 :]
            path = path[:cut]
        url = f"{scheme}://{netloc}{slash}{path}" if separator else path
        subdirectory = ""
        for part in fragment.split("&"):
            key, sep, value = part.partition("=")
            if sep and key.strip() == "subdirectory":
                subdirectory = value.strip()
        return url, revision, subdirectory
    return None


_COMMIT_REVISION_RE = re.compile(r"[0-9a-fA-F]{7,40}")

_GITHUB_ARCHIVE_HOSTS = ("github.com", "www.github.com")


def _github_archive_url(
    url: str,
    revision: str,
    subdirectory: str = "",
) -> "str | None":
    """The zip GitHub serves for *revision* of *url*, or None when there is no such URL.

    This is the route a host with no working git takes. pip fetches a zip over plain https and
    needs no git binary anywhere, which is the entire point: git is absent from the desktop bundle,
    and on macOS `git` is frequently a bare xcrun shim that exits non-zero, so the git requirement
    is skipped on hosts whose owners have no idea they are missing anything.

    Deliberately narrow, because an archive is weaker evidence than a clone:

    * a FULL 40-character commit only. An archive carries no history and records no ref, so a
      branch or tag would become "whatever that name pointed at when it was fetched" with nothing
      left on disk to tell two fetches apart. A commit has no such ambiguity, and the SHA is in the
      URL, so the recorded url IS the provenance.
    * github.com over http(s) only. Every other forge spells its archive differently, and guessing
      wrong installs nothing rather than something wrong, but there is no reason to guess.
    * no subdirectory. ``#subdirectory=`` is part of the package identity and this URL cannot
      carry it, so a requirement that uses one keeps the git route and skips as before.
    """
    if subdirectory:
        return None
    if len(revision) != 40 or not _COMMIT_REVISION_RE.fullmatch(revision):
        return None
    scheme, separator, remainder = url.partition("://")
    if not separator or scheme.lower() not in ("http", "https"):
        return None
    netloc, _, path = remainder.partition("/")
    if netloc.lower() not in _GITHUB_ARCHIVE_HOSTS:
        return None
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    owner, slash, repo = path.partition("/")
    if not (owner and slash and repo) or "/" in repo:
        return None
    return f"https://github.com/{owner}/{repo}/archive/{revision.lower()}.zip"


def _vcs_url_key(url: str) -> str:
    """A git URL without the trailing slash or ``.git``, which uv drops from direct_url.json.

    The requirements files name ``https://github.com/<org>/<repo>.git`` and uv records the URL
    without the suffix, so an exact compare never matched and every pass rebuilt the checkout.
    """
    url = url.rstrip("/")
    return url[: -len(".git")] if url.endswith(".git") else url


def _direct_reference_is_installed(req: Path, dist_name: str) -> bool:
    """Whether the resident *dist_name* came from the ref *req* names.

    A version says nothing about a git requirement: the same one is published from every
    branch, so a release/3.6.x pin is satisfied on paper by a build from main. direct_url.json
    is the only place the ref pip and uv actually installed survives.
    """
    wanted = _direct_reference_in_requirements(req)
    if wanted is None:
        return False
    url, revision, subdirectory = wanted
    payload = _recorded_direct_url(dist_name)
    if payload is None:
        return False
    vcs = payload.get("vcs_info")
    if not isinstance(vcs, dict):
        # The no-git zip install is recorded as archive_info without a ref, so the URL (which holds the SHA)
        # is the evidence; otherwise the release pin reinstalls over the main build next pass.
        archive = _github_archive_url(url, revision, subdirectory)
        return (
            archive is not None
            and isinstance(payload.get("archive_info"), dict)
            and str(payload.get("url") or "") == archive
        )
    if not (
        _vcs_url_key(str(payload.get("url") or "")) == _vcs_url_key(url)
        and str(vcs.get("requested_revision") or "") == revision
        and str(payload.get("subdirectory") or "") == subdirectory
    ):
        return False
    if _COMMIT_REVISION_RE.fullmatch(revision):
        return True
    # A branch ref can move with unchanged requirements, so ask `git ls-remote`; unreachable keeps it.
    commit_id = str(vcs.get("commit_id") or "").strip().lower()
    if not _COMMIT_REVISION_RE.fullmatch(commit_id or "x"):
        return False
    remote = _git_remote_commit(url, revision)
    if remote is None:
        _note(
            f"{dist_name}: {url} ({revision}) is unreachable -- keeping the installed build "
            f"{commit_id[:12]}",
            _dim,
        )
        return True
    return remote == commit_id or remote.startswith(commit_id) or commit_id.startswith(remote)


def _payload_recorded_intact(dist_name: str) -> bool:
    """True only when the distribution has a RECORD and every file it names is there
    at its recorded size; a missing distribution or RECORD is not intact."""
    try:
        return _recorded_payload_damaged(dist_name) is False
    except Exception:  # noqa: BLE001 - not installed, or unreadable, is not intact
        return False


def _triton_kernels_step() -> None:
    """Install triton kernels, or keep the build that is there.

    The requirement is a git branch, so the evidence is asked of the remote now (one
    ls-remote in `_direct_reference_is_installed`) rather than read from a record, and holds
    even on a pass with no recorded evidence: a full pass has no reason to rebuild a checkout
    the branch still points at, and an unreachable git host must not fail an update over a
    speedup that is already installed. It did, since the step ran every pass.
    """
    req = REQ_ROOT / "triton-kernels.txt"
    if not _has_working_git():
        _progress("triton kernels (skipped, no git)")
        _note("no working git -- skipping triton kernels (training speedup only)")
        return
    asked: dict = {}

    def _ref_current() -> bool:
        asked["current"] = _direct_reference_is_installed(req, "triton_kernels")
        return asked["current"]

    if _skip_step(req, "triton kernels", no_deps = True, constrain = False, extra_check = _ref_current):
        return
    if "current" not in asked:
        _ref_current()
    # Provenance alone is not a build: triton_kernels/ can be gone under its dist-info.
    if (
        asked["current"]
        and not _full_deps_requested()
        and _payload_recorded_intact("triton_kernels")
    ):
        _note("triton kernels: the installed build is what the requirement's ref points at -- kept")
        _record_step(_pass_input_key(req) or str(req), "skipped")
        return
    pip_install(
        "Installing triton kernels",
        "--no-deps",
        "--no-cache-dir",
        req = req,
        constrain = False,
    )


DIFFUSERS_MAIN_ENV = "UNSLOTH_DIFFUSERS_MAIN"

# Diffusers main needs Python >= 3.10, so 3.9 can never satisfy this step.
DIFFUSERS_MAIN_MIN_PYTHON = (3, 10)


def _diffusers_main_requested() -> bool:
    """Whether this install wants the pinned Diffusers main build. Default: yes.

    Opt OUT with ``UNSLOTH_DIFFUSERS_MAIN=0`` (also false/no/off). Anything else, including the
    variable being unset, means yes, so the models that need an unreleased Diffusers work for
    everyone without a flag. The opt-out exists for an install that must stay on the exact release
    everything else is built against, or that cannot reach github.com but does have git.
    """
    value = (os.environ.get(DIFFUSERS_MAIN_ENV) or "").strip().lower()
    return value not in ("0", "false", "no", "off")


def _diffusers_main_resident(req: "Path | None" = None) -> bool:
    """Whether the pinned main build is BOTH what the file names and actually on disk.

    Provenance alone is not a build, the same reason ``_triton_kernels_step`` pairs its ref check
    with this one: ``direct_url.json`` survives inside dist-info while the package tree under it is
    deleted or truncated, and a provenance-only answer would skip the reinstall AND, since the
    release pin reads the same predicate to decide it has been superseded, skip the repair too.
    Diffusers is mandatory, so that combination leaves Studio broken on every later pass rather
    than for one. Either half failing means "install it", which is the recoverable direction.
    """
    if req is None:
        req = REQ_ROOT / "diffusers-main.txt"
    return _direct_reference_is_installed(req, "diffusers") and _payload_recorded_intact(
        "diffusers"
    )


def _diffusers_main_supersedes_release() -> bool:
    """Whether 11c's build already stands in for the release pin, so 11b must not reinstall it.

    Only when the main build is BOTH wanted and resident, AND this pass is allowed to skip work at
    all: under UNSLOTH_STUDIO_FULL_DEPS both steps run, the release first and the commit back on
    top, which is the same order a first install takes and the only order that ends with the tree
    the family gate expects.
    """
    if _full_deps_requested():
        return False
    return _diffusers_main_requested() and _diffusers_main_resident()


_ARCHIVE_SHA256_RE = re.compile(r"#\s*archive-sha256:\s*([0-9a-fA-F]{64})")


def _archive_sha256_in_requirements(req: Path) -> "str | None":
    """The ``# archive-sha256:`` digest *req* pins for its zip, or None unless exactly one."""
    try:
        text = req.read_text(encoding = "utf-8-sig")
    except (OSError, ValueError):
        return None
    found = [
        m.group(1).lower()
        for m in (_ARCHIVE_SHA256_RE.fullmatch(line.strip()) for line in text.splitlines())
        if m
    ]
    return found[0] if len(found) == 1 else None


def _diffusers_main_archive(req: Path) -> "str | None":
    """The hash-pinned zip route 11c takes with no working git, or None when it has none.

    pip and uv record the URL without the ``#sha256=`` fragment, so residency still matches it.
    """
    wanted = _direct_reference_in_requirements(req)
    archive = _github_archive_url(*wanted) if wanted is not None else None
    digest = _archive_sha256_in_requirements(req)
    if archive is None or digest is None:
        return None
    return f"{archive}#sha256={digest}"


def _diffusers_main_needs_dependency_pass() -> bool:
    """For the setup fast path: is the resident Diffusers the wrong one for the requested mode?

    The fast path skips this script whenever the core package is current, so without this an
    install that never ran 11c (updated by an installer that predates it, or opted back in) stays
    on the release until the next version bump. The same gates as 11c decide whether the pass
    could install the build at all, so a host with neither git nor the zip route keeps its fast
    path. A pass that tried
    and failed records "failed" and also keeps it: without that, a host that cannot reach
    github.com would repeat the whole dependency pass on every update.
    """
    req = REQ_ROOT / "diffusers-main.txt"
    if not req.is_file():
        return False
    if not _diffusers_main_requested():
        # Opted out while the build is still resident: 11b puts the release back.
        return _diffusers_main_resident(req)
    if sys.version_info < DIFFUSERS_MAIN_MIN_PYTHON:
        return False
    if not _has_working_git() and _diffusers_main_archive(req) is None:
        return False
    if _diffusers_main_resident(req):
        return False
    try:
        manifest = install_manifest.read_manifest() or {}
        last = (manifest.get("step_results") or {}).get("diffusers-main.txt")
    except Exception:  # noqa: BLE001 - an unreadable manifest is no record of a failed try
        last = None
    return last != "failed"


# Startup repair records its failure here; only the repair reads it, so an explicit update retries.
_DIFFUSERS_MAIN_REPAIR_KEY = "diffusers_main_repair"


def _startup_repair_failed() -> bool:
    try:
        return (install_manifest.read_manifest() or {}).get(_DIFFUSERS_MAIN_REPAIR_KEY) == "failed"
    except Exception:  # noqa: BLE001 - an unreadable manifest is no record of a failed try
        return False


_REPAIR_LOCK_POLL_S = 5


def _repair_diffusers_main() -> int:
    """11c on its own, for the backend's startup self-heal: 0 installed, 1 nothing to do, 2 failed.

    An update from a release that predates 11c runs that release's installer, which never installs
    the build; the backend that starts afterwards is the first new code such a host runs.
    """
    import time

    global USE_UV, _STEP, _TOTAL
    while True:
        with install_manifest.pass_lock() as uncontended:
            # Even "nothing to do" waits for the lock: a sibling repair or update may be rewriting diffusers.
            if uncontended:
                if (
                    not _diffusers_main_requested()
                    or not _diffusers_main_needs_dependency_pass()
                    or _startup_repair_failed()
                ):
                    return 1
                USE_UV = _bootstrap_uv()
                _STEP, _TOTAL = 0, 1
                _diffusers_main_step()
                if _diffusers_main_resident():
                    return 0
                # Or every start retries a fetch this host cannot make.
                install_manifest.update_manifest(**{_DIFFUSERS_MAIN_REPAIR_KEY: "failed"})
                return 2
        time.sleep(_REPAIR_LOCK_POLL_S)


_PREFETCH_SCRATCH_PREFIX = "unsloth-diffusers-prefetch-"


def _prefetch_diffusers_main() -> int:
    """For the startup repair: fetch and build the pinned build into uv's cache, installing nothing.

    The backend stops this at its deadline, which is safe only because ``--target`` points uv at a
    scratch directory, so site-packages is never touched; the ``--repair-diffusers-main`` that
    follows then installs from the cache in about a second. 0 fetched, 1 nothing to fetch (pip keeps
    no cache here, so it fetches during the install), 2 failed.
    """
    import time

    global USE_UV
    req = REQ_ROOT / "diffusers-main.txt"
    if (
        not req.is_file()
        or not _diffusers_main_requested()
        or not _diffusers_main_needs_dependency_pass()
        or _startup_repair_failed()
    ):
        return 1
    USE_UV = _bootstrap_uv()
    if not USE_UV:
        return 1
    # A prefetch stopped at the deadline cannot clean up after itself.
    for stale in Path(tempfile.gettempdir()).glob(f"{_PREFETCH_SCRATCH_PREFIX}*"):
        try:
            if time.time() - stale.stat().st_mtime > 3600:
                shutil.rmtree(stale, ignore_errors = True)
        except OSError:
            pass
    # Same source _diffusers_main_step installs from, so the install hits this cache.
    archive = None if _has_working_git() else _diffusers_main_archive(req)
    scratch = Path(tempfile.mkdtemp(prefix = _PREFETCH_SCRATCH_PREFIX))
    temp_reqs: list[Path] = []
    try:
        args = ("--no-deps", "--target", str(scratch))
        if archive is not None:
            cmd = _build_uv_cmd((*args, f"diffusers @ {archive}"))
        else:
            actual_req, temp_reqs = _effective_requirements(req)
            cmd = _build_uv_cmd(args) + ["-r", _uv_safe_path(actual_req)]
        cmd, env = _pinned_cmd_and_env(cmd)
        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    finally:
        for temp_req in temp_reqs:
            temp_req.unlink(missing_ok = True)
        shutil.rmtree(scratch, ignore_errors = True)
    if result.returncode != 0:
        if result.stdout:
            _safe_print(_redact_install_output(result.stdout))
        return 2
    return 0


def _diffusers_main_step() -> None:
    """Install the pinned Diffusers commit, or leave the release pin alone.

    Three things this has to get right, none of which the ordinary ``_skip_step`` path covers:

    * A VERSION says nothing here. Every build of main reports 0.41.0.dev0, so the usual "is the
      requirement satisfied" check passes against a build from any other commit, or against the
      release the previous step just installed. ``_direct_reference_is_installed`` reads the ref out
      of direct_url.json, which is the only place it survives, so bumping the commit in the file
      actually reinstalls instead of silently keeping the old tree.
    * Diffusers is MANDATORY, unlike triton_kernels, so a host that cannot do a source build must
      not be left with a broken install. It is not: the release pin ran first and is already in
      place, so skipping here leaves a working Studio that simply cannot load the newest model.
      That is the whole reason this is a second step on top of the release pin rather than an
      edit to it, now that it runs by default and therefore meets every host there is.
    * Opting out must put the release back. Setting the variable to 0 and re-running reinstalls the
      pin on the next pass, because the pin step's own inputs are unchanged but the resident
      diffusers is no longer what the release pin names.

    It spends exactly ONE progress slot on every path, including opting out, because the total is
    fixed before any of this is known. An early return without a _progress leaves the bar short of
    its own total for precisely the users who opted out.
    """
    if not _diffusers_main_requested():
        _progress("diffusers main (opted out, skipped)")
        return
    if sys.version_info < DIFFUSERS_MAIN_MIN_PYTHON:
        # Unmarked, pip would clone the repo and only then reject requires-python, on every update.
        _progress("diffusers main (skipped, needs python 3.10)")
        return
    req = REQ_ROOT / "diffusers-main.txt"
    if not req.is_file():
        _progress("diffusers main (skipped, no pin file)")
        return
    # Zip route only when git cannot work (desktop bundle has no git; macOS xcrun shim fails): a clone
    # records the ref, an archive only a URL. None unless the pin is a full GitHub commit.
    archive = None
    if not _has_working_git():
        archive = _diffusers_main_archive(req)
        if archive is None:
            _progress("diffusers main (skipped, no git)")
            _note(
                "No working git, so this install keeps the pinned Diffusers release instead of "
                "the pinned main build. Everything else works; models that need an unreleased "
                "Diffusers will refuse with a message naming the version they want.",
            )
            return
    # UNSLOTH_STUDIO_FULL_DEPS reaches this skip too: residency compares recorded sizes, so same-size
    # corruption reads as intact.
    if not _full_deps_requested() and _diffusers_main_resident(req):
        _progress("diffusers main (satisfied, skipped)")
        _record_step("diffusers-main.txt", "skipped")
        return
    _progress("diffusers main (no git, from archive)" if archive else "diffusers main")
    _record_step("diffusers-main.txt", "ran")
    # pip_install_try, not pip_install: github.com must not be a hard requirement of every install, and
    # the release pin is still resident, so failing costs one model.
    if archive is not None:
        installed = pip_install_try(
            "Installing the pinned Diffusers main build (zip archive, no git)",
            "--no-cache-dir",
            f"diffusers @ {archive}",
            constrain = False,
        )
    else:
        installed = pip_install_try(
            "Installing the pinned Diffusers main build",
            "--no-cache-dir",
            req = req,
            constrain = False,
        )
    if not installed:
        # "failed", not "skipped": the fast path reads it to stop forcing a pass that cannot succeed.
        _record_step("diffusers-main.txt", "failed")
        _note(
            "Could not install the pinned Diffusers main build, so this install keeps the pinned "
            "Diffusers release. Everything else works; models that need an unreleased Diffusers "
            f"will refuse with a message naming the version they want. Set {DIFFUSERS_MAIN_ENV}=0 "
            "to stop trying.",
        )


def _recorded_direct_url(dist_name: str) -> "dict | None":
    """The direct_url.json pip and uv wrote for *dist_name*, or None when there is none
    that parses. This is the only place a git install's ref and commit survive."""
    try:
        from importlib.metadata import distribution
        recorded = distribution(dist_name).read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absent metadata is a reason to install
        return None
    if not recorded:
        return None
    try:
        payload = json.loads(recorded)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _git_remote_commit(
    url: str,
    revision: str,
    timeout: int = 45,
) -> "str | None":
    """The commit *revision* names on the remote right now (lower-case hex), or None
    when the remote cannot be asked: no git, no network, no such ref, a timeout.

    A peeled tag (`refs/tags/x^{}`) wins over the tag object and a branch over a tag of the
    same name: pip and uv record the commit the checkout landed on, peeled for annotated tags.
    """
    exe = shutil.which("git")
    if exe is None or not revision:
        return None
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_ASKPASS", "echo")
    try:
        probe = subprocess.run(
            [exe, "ls-remote", "--", url, revision],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    rows: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        sha, _, ref = line.strip().partition("\t")
        if ref and _COMMIT_REVISION_RE.fullmatch(sha):
            rows[ref] = sha.lower()
    for ref in (
        f"refs/tags/{revision}^{{}}",
        f"refs/heads/{revision}",
        f"refs/tags/{revision}",
        revision,
    ):
        if ref in rows:
            return rows[ref]
    return next(iter(rows.values()), None) if len(rows) == 1 else None


def _uv_version() -> "str | None":
    """The uv this pass used, for the record. None when pip did the work."""
    if not USE_UV:
        return None
    try:
        probe = subprocess.run(
            ["uv", "--version"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 30,
            **_windows_hidden_subprocess_kwargs(),
        )
    except Exception:  # noqa: BLE001 - advisory
        return None
    return probe.stdout.strip() or None if probe.returncode == 0 else None


def _installer_python_tag() -> str:
    """The interpreter's ABI tag, free-threaded builds included.

    Recorded beside `python`, which carries only the version: a GIL and a free-threaded
    3.14 install the same version string and cannot load each other's extensions.
    """
    return "{}{}{}".format(
        sys.version_info.major,
        sys.version_info.minor,
        "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "",
    )


def _patch_metadata_is_pending() -> bool:
    """Whether any METADATA the single-env patch owns still matches a pattern it rewrites.

    The patch is idempotent, so re-running it on a settled install rewrites nothing and costs
    a subprocess plus a metadata walk. Asking first is the same walk without the interpreter
    start; on a fresh venv, where the data-designer steps just ran, the caller does not ask.
    """
    try:
        sys.path.insert(0, str(SINGLE_ENV))
        try:
            import patch_metadata  # noqa: PLC0415
        finally:
            if sys.path and sys.path[0] == str(SINGLE_ENV):
                sys.path.pop(0)
        for name in patch_metadata.TARGETS:
            path = patch_metadata.metadata_path(name)
            if path is None:
                continue
            text = path.read_text(encoding = "utf-8")
            if any(pattern.search(text) for pattern, _replacement in patch_metadata.PATCHES):
                return True
    except Exception:  # noqa: BLE001 - unknown means run it, which is what it did before
        return True
    return False


def _run_patch_metadata() -> None:
    """Apply the single-env metadata patch, in process where that works.

    A subprocess is a whole interpreter start for a stdlib script that edits at most three
    files. It stays the fallback: the script is also a standalone entry point, and an import
    failure must not fail the install. Its tally, which run() captured, is captured here too
    and kept for UNSLOTH_VERBOSE, or it lands in the middle of the progress line.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    try:
        sys.path.insert(0, str(SINGLE_ENV))
        try:
            import patch_metadata  # noqa: PLC0415
        finally:
            if sys.path and sys.path[0] == str(SINGLE_ENV):
                sys.path.pop(0)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            patch_metadata.main()
        if VERBOSE:
            for line in captured.getvalue().splitlines():
                if line.strip():
                    _note(line.strip())
        return
    except Exception:  # noqa: BLE001
        pass
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )


# Build outputs written into the plugin dir; digesting them forced an offline rebuild after install.
_PLUGIN_BUILD_ARTIFACT_DIRS = frozenset({"__pycache__", "build", "dist", ".eggs"})


def _is_plugin_build_artifact(relative: Path) -> bool:
    parts = relative.parts
    if any(part in _PLUGIN_BUILD_ARTIFACT_DIRS for part in parts[:-1]):
        return True
    if any(part.endswith((".egg-info", ".dist-info")) for part in parts[:-1]):
        return True
    return parts[-1].endswith((".pyc", ".pyo"))


def _local_plugin_digest(plugin_dir: Path) -> "str | None":
    """One sha256 over a local plugin tree, so an edited seed plugin reinstalls.

    Sorted (relpath, bytes): a directory listing has no order, and a digest that
    depended on one would differ between two byte-identical trees.
    """
    import hashlib

    if not plugin_dir.is_dir():
        # None never equals a recorded digest: an unreadable plugin reinstalls.
        return None
    digest = hashlib.sha256()
    try:
        paths = sorted(
            path
            for path in plugin_dir.rglob("*")
            if path.is_file() and not _is_plugin_build_artifact(path.relative_to(plugin_dir))
        )
        for path in paths:
            # surrogateescape: POSIX filenames need not be UTF-8.
            digest.update(
                path.relative_to(plugin_dir).as_posix().encode("utf-8", "surrogateescape")
            )
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
    except OSError:
        return None
    return digest.hexdigest()


def _read_own_source() -> "bytes | None":
    try:
        return Path(__file__).read_bytes()
    except OSError:
        return None


# The core-packages step can upgrade this file on disk while the old code keeps running.
_INSTALLER_SOURCE_AT_START = _read_own_source()
# Set on the rerun, so a second replacement cannot loop.
_INSTALLER_RERUN_ENV = "UNSLOTH_INSTALLER_RERUN"


class _InstallerReplaced(Exception):
    """Raised once the core-packages step has replaced this file with another release's copy."""


def _installer_replaced() -> bool:
    if os.environ.get(_INSTALLER_RERUN_ENV) == "1" or _INSTALLER_SOURCE_AT_START is None:
        return False
    current = _read_own_source()
    return current is not None and current != _INSTALLER_SOURCE_AT_START


def _rerun_replaced_installer() -> int:
    """Finish the pass with the installer the update just installed.

    Without this every step a release adds is skipped by the update that installs it: the old
    process upgrades the package in step 3 and completes its own step list. A subprocess rather
    than an exec, because on Windows os.exec* returns to the caller while the child runs. It starts
    after the pass lock is released, so the rerun does not read its parent as a contending peer.
    """
    _note("the update replaced this installer; finishing with the new version")
    try:
        # The child shares the descriptors, so flush first or our buffered output lands after its output.
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    env = dict(os.environ)
    env[_INSTALLER_RERUN_ENV] = "1"
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        env = env,
    ).returncode


def _under_pass_lock(func):
    """Run the pass holding the pass lock, and record whether a peer already had it."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _PASS_UNCONTENDED
        with install_manifest.pass_lock() as uncontended:
            _PASS_UNCONTENDED = uncontended
            return func(*args, **kwargs)

    return wrapper


@_under_pass_lock
def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL, _PROGRESS_LINE_ACTIVE
    global _INSTALL_ACTIONS, _PASS_EVIDENCE, _CONSTRAINTS_CACHE, _CLOSURE_INDEX_CACHE
    global _BNB_ROCM_PASS_PROVENANCE, _BNB_ROCM_PASS_ASSET
    _STEP = 0
    # Module state reset, so a second call in one process (the tests) starts fresh.
    _INSTALL_ACTIONS = 0
    _PASS_EVIDENCE = None
    _CONSTRAINTS_CACHE = None
    _CLOSURE_INDEX_CACHE = None
    _BNB_ROCM_PASS_PROVENANCE = None
    _BNB_ROCM_PASS_ASSET = None
    _STEP_RESULTS.clear()
    _AUDITED_STEPS.clear()
    # An aborted earlier run can leave it set, giving the first message a stray newline.
    _PROGRESS_LINE_ACTIVE = False

    # install.sh sets SKIP_STUDIO_BASE=1; `studio update` does not, so core packages are reinstalled.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # --package installs a different package name (for testing).
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    # Clean-machine CI overlays only unsloth, not the full local source pair.
    ci_source_overlay = os.environ.get("UNSLOTH_CI_SOURCE_OVERLAY", "")
    # Includes lettered steps 8b, 8c (Windows), 11b, 11c, 13b; 11c always spends its slot.
    base_total = 14 if IS_WINDOWS else 15
    if IS_WINDOWS:
        base_total += 1  # 8c, gated exactly as the step is
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
        # Re-resolve after the core phase may move unsloth-zoo; same gate, so the slot is always spent.
        base_total += 1  # MLX stack re-resolve
    if IS_MAC_ARM:
        base_total += 1  # MLX grammar engine (step 11d), same gate as the step itself
    if NO_TORCH and not skip_base:
        # no-torch runtime deps get their own slot inside the core step.
        base_total += 1
    base_requirements = _shared_base_requirements() if skip_base else None
    # A shell-installer handoff skips the core slot only while base.txt has no work.
    _TOTAL = base_total - int(skip_base and base_requirements is None)

    # Before the manifest goes: it is the only record of the last run.
    _PASS_EVIDENCE = _plan_pass(package_name, local_repo, ci_source_overlay)

    # Drop the manifest up front: its absence tells the CLI, setup.sh and preflight a run was interrupted.
    # A stale parked copy goes first; one that cannot be cleared must refuse here.
    _parked = install_manifest.previous_manifest_path()
    install_manifest.consume_previous_manifest()
    if install_manifest.manifest_is_present(_parked):
        _safe_print(
            f"error: could not remove the parked {install_manifest.PREVIOUS_MANIFEST_NAME} "
            f"in {install_manifest.venv_root()}; refusing to install behind evidence the "
            "next run would read as a completed pass",
            file = sys.stderr,
        )
        return 1
    if install_manifest.remove_manifest():
        install_manifest.consume_previous_manifest()
        if install_manifest.manifest_is_present(_parked):
            _safe_print(
                f"error: could not remove the parked {install_manifest.PREVIOUS_MANIFEST_NAME} "
                f"in {install_manifest.venv_root()}; refusing to install behind evidence the "
                "next run would read as a completed pass",
                file = sys.stderr,
            )
            return 1
    else:
        _safe_print(
            f"error: could not remove the stale {install_manifest.MANIFEST_NAME} in "
            f"{install_manifest.venv_root()}; refusing to install behind a marker "
            "that would still report this venv as complete",
            file = sys.stderr,
        )
        return 1

    # Record the mode in a marker that survives a killed pass, or the next update may delete the venv.
    install_manifest.set_no_torch_marker(NO_TORCH)

    USE_UV = _bootstrap_uv()

    # 2. Ensure pip is available (uv venvs omit it). UNSLOTH_STUDIO_FULL_DEPS reaches this skip too.
    if not _full_deps_requested() and _venv_pip_is_usable():
        _progress("pip bootstrap (satisfied, skipped)")
        _record_step("pip", "skipped")
    elif USE_UV:
        _progress("pip bootstrap")
        _record_step("pip", "ran")
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
        _progress("pip bootstrap")
        _record_step("pip", "ran")
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

    # A superseded dist-info makes version() pick an arbitrary version; repair before any fast path.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

    # Intact metadata over a missing payload would audit as satisfied.
    if not _repair_damaged_core_payload(_core_package_names(package_name), local_repo = local_repo):
        return 1

    # macOS arm64: not keyed off skip_base (fresh installs still need MLX), not on --no-torch. Pins align
    # with utils/mlx_repair.py and unsloth-zoo. None when the MLX step did not run.
    _mlx_vlm_spec_used: Optional[str] = None
    if IS_MAC_ARM and not NO_TORCH:
        # Both branches spend the slot, so the denominator does not depend on the host.
        if _mlx_pins_are_installable():
            # _may_skip_on_evidence too, and the arm64 overrides digest: UV_OVERRIDE shapes MLX's graph.
            if (
                _may_skip_on_evidence()
                and _mlx_stack_is_current()
                and _inputs_unchanged(["single-env/overrides-darwin-arm64.txt"])
            ):
                _progress("MLX stack (satisfied, skipped)")
                _record_step("mlx", "skipped")
            else:
                _progress("MLX stack (Apple Silicon)")
                _record_step("mlx", "ran")
                pip_install(
                    "Installing MLX stack (mlx + mlx-lm + mlx-vlm)",
                    "--no-cache-dir",
                    # Never a bare --upgrade, which re-resolves every transitive dependency.
                    *[arg for name in _MLX_NAMES for arg in ("--upgrade-package", name)],
                    *_MLX_PINS,
                    _mlx_vlm_spec_for_installed_zoo(),
                )
            _mlx_vlm_spec_used = _mlx_vlm_spec_for_installed_zoo()
        else:
            _progress("MLX stack (skipped, no wheel for this macOS or Python)")
            _note(
                f"macOS {_macos_release_major() or 'unknown'} on Python "
                f"{sys.version_info.major}.{sys.version_info.minor} publishes no wheel for the "
                f"supported MLX versions (needs macOS {_MLX_MIN_MACOS_MAJOR}+ and Python "
                f"{_MLX_MIN_PYTHON[0]}.{_MLX_MIN_PYTHON[1]}+) -- leaving Train/Export disabled "
                "rather than failing the install"
            )

    # gfx906: record bnb presence before the base install pulls a generic wheel.
    global _GFX906_BNB_ABSENT_BEFORE_BASE
    if not skip_base:
        _GFX906_BNB_ABSENT_BEFORE_BASE = not _bitsandbytes_installed()

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        pass
    elif NO_TORCH:
        # No-torch path: --no-deps throughout (PyPI metadata makes torch a hard dep).
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
        # With deps so pip pins a matching pydantic-core.
        pip_install(
            "Installing pydantic (with deps for compatible core)",
            "--no-cache-dir",
            "pydantic",
        )
        if not _skip_step(REQ_ROOT / "no-torch-runtime.txt", "no-torch runtime deps", no_deps = True):
            pip_install(
                "Installing no-torch runtime deps",
                "--no-cache-dir",
                "--no-deps",
                req = REQ_ROOT / "no-torch-runtime.txt",
            )
        if local_repo:
            _overlay_local_core_packages(local_repo)
    elif local_repo:
        # Overlay the checkout editable with --no-deps so torch is not re-resolved.
        _progress("base packages")
        with _FreezeNewTorchForCoreUpdate():
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
        with _FreezeNewTorchForCoreUpdate():
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

    # The new package may ship a newer copy of this file; raise so the pass lock is released first.
    if _installer_replaced():
        raise _InstallerReplaced

    # The MLX step honoured the OLD zoo's mlx-vlm range; re-resolve when the declared range moved,
    # since the startup self-heal will not correct it.
    if IS_MAC_ARM and not NO_TORCH:
        _mlx_vlm_spec_now = _mlx_vlm_spec_for_installed_zoo()
        if _mlx_vlm_spec_used is None or _mlx_vlm_spec_now == _mlx_vlm_spec_used:
            _progress("MLX stack (zoo unchanged, skipped)")
        else:
            _progress("MLX stack (re-resolved for the new zoo)")
            _record_step("mlx", "ran")
            pip_install(
                "Installing MLX stack for the upgraded unsloth-zoo",
                "--no-cache-dir",
                *[arg for name in _MLX_NAMES for arg in ("--upgrade-package", name)],
                *_MLX_PINS,
                _mlx_vlm_spec_now,
            )

    if not skip_base:
        base_requirements = _shared_base_requirements()

    # Shell installers skip the core phase but still apply this file.
    if base_requirements is not None:
        satisfied = _requirements_satisfied(base_requirements, no_deps = False)
        _record_step("base.txt", "skipped" if satisfied else "ran")
        if skip_base:
            _progress(
                "base requirements (satisfied, skipped)" if satisfied else "base requirements"
            )
        elif satisfied:
            _step(_LABEL, "shared base requirements are current")
        else:
            _step(_LABEL, "applying shared base requirements")
        if not satisfied:
            pip_install(
                "Applying shared base requirements",
                "--no-cache-dir",
                req = base_requirements,
            )

    # 2b. Torch repair (wrong-family / CPU-only); must follow base packages so torch is present.
    if not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("check"))
        # False = required repair the mirror cannot express: abort, never leave the mirror.
        if _ensure_cuda_torch() is False:
            return 1
        if _ensure_rocm_torch() is False:
            return 1
        if _ensure_xpu_torch() is False:
            return 1
        if _ensure_cpu_torch() is False:
            return 1
        # Last, after every torch migration: the swap keys off the installed +xpu label.
        _ensure_xpu_triton()

    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
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

    # 3. Extra dependencies
    if not _skip_step(REQ_ROOT / "extras.txt", "unsloth extras", no_deps = False):
        pip_install(
            "Installing additional unsloth dependencies",
            "--no-cache-dir",
            # extras.txt holds the wheel-less requirements, so binary-only policies fail here first.
            *_sdist_only_build_args(*_extras_sdist_only_packages()),
            req = REQ_ROOT / "extras.txt",
        )

    # 3b. Extra dependencies (no-deps) -- audio model support etc.
    if not _skip_step(REQ_ROOT / "extras-no-deps.txt", "extra codecs", no_deps = True):
        pip_install(
            "Installing extras (no-deps)",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "extras-no-deps.txt",
        )

    # 4. torch-matched torchao override; reinstall only on pin change (Windows can lose shared files).
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    elif _rocm_windows_torch_installed or _installed_torch_is_windows_rocm():
        # Stock torchao dies on import here; only the export worker loads it (unsloth/_torchao_nodist.py).
        _progress("dependency overrides (Windows ROCm)")
        _install_torchao_for_torch(_probe_installed_torch_version(), default_index = True)
    else:
        _progress("dependency overrides")
        _install_torchao_for_torch(_probe_installed_torch_version())

    # 5. Triton kernels (no-deps, from source); not on Windows/macOS or without git. Warn only.
    if not IS_WINDOWS and not IS_MACOS:
        _triton_kernels_step()

    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress("flash-attn")
        _ensure_flash_attn()

    # 8. Unsloth dependencies
    if not _skip_step(REQ_ROOT / "studio.txt", "Unsloth Studio deps", no_deps = False):
        pip_install(
            "Installing Unsloth Studio dependencies",
            "--no-cache-dir",
            req = REQ_ROOT / "studio.txt",
        )

    # 8b. anyio repair (#6483)
    _progress("anyio check")
    _repair_bad_anyio()

    # 8c. Outside skip_base on purpose: install.ps1 sets it and is the path that lands 1.15.
    if IS_WINDOWS:
        _progress("accelerate check")
        _repair_bad_accelerate()

    # 9. Data-designer dependencies
    _dd_deps_ran = not _skip_step(
        SINGLE_ENV / "data-designer-deps.txt", "data designer deps", no_deps = False
    )
    if _dd_deps_ran:
        pip_install(
            "Installing data-designer base dependencies",
            "--no-cache-dir",
            req = SINGLE_ENV / "data-designer-deps.txt",
        )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    _dd_ran = not _skip_step(SINGLE_ENV / "data-designer.txt", "data designer", no_deps = True)
    if _dd_ran:
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
    # Digested, not version-compared: a path-installed plugin's version never moves.
    _plugin_digests: dict[str, str] = {}
    _plugin_work: list[tuple[str, Path]] = []
    _recorded_plugins = (_PASS_EVIDENCE or {}).get("pass_inputs") or {}
    for _plugin_name, plugin_dir in local_dd_plugins:
        _plugin_key = f"plugins/{plugin_dir.name}"
        _plugin_digest = _local_plugin_digest(plugin_dir)
        if _plugin_digest is not None:
            _plugin_digests[_plugin_key] = _plugin_digest
        _plugin_current = (
            _PASS_EVIDENCE is not None
            and _plugin_digest is not None
            and _recorded_plugins.get(_plugin_key) == _plugin_digest
            # The directory name IS the distribution name (each plugin's pyproject.toml).
            and bool(_installed_distribution_version(plugin_dir.name))
            # ...and the payload: a deleted module leaves version and source digest unchanged.
            and not _local_plugin_payload_is_damaged(plugin_dir.name)
        )
        if not _plugin_current:
            _plugin_work.append((_plugin_name, plugin_dir))
    _progress("local plugin" if _plugin_work else "local plugin (satisfied, skipped)")
    _record_step("plugins", "ran" if _plugin_work else "skipped")
    for plugin_name, plugin_dir in _plugin_work:
        pip_install(
            f"Installing local data-designer {plugin_name} plugin",
            "--no-cache-dir",
            "--no-deps",
            str(plugin_dir),
            constrain = False,
        )

    # 11b. The pinned Diffusers release, after every other requirements file and on every path.
    # Stands down once 11c's build is resident (main reports 0.41.0.dev0, never ==0.40.0), else every
    # update would install the release then main again.
    if not _skip_step(
        REQ_ROOT / "diffusers-pin.txt",
        "diffusers pin",
        no_deps = False,
        superseded = _diffusers_main_supersedes_release(),
    ):
        pip_install(
            "Installing the pinned Diffusers release",
            "--no-cache-dir",
            req = REQ_ROOT / "diffusers-pin.txt",
        )

    # 11c. Pinned Diffusers main commit (UNSLOTH_DIFFUSERS_MAIN=0 opts out), on top of the release so a
    # failure degrades to a working install.
    _diffusers_main_step()

    # 11d. Apple Silicon grammar engine, outside skip_base (install.sh always skips base); failure only loses MLX response_format.
    if IS_MAC_ARM:
        if not _full_deps_requested() and _exact_distribution_spec_is_installed(_LLGUIDANCE_PIN):
            _progress("MLX grammar engine (satisfied, skipped)")
        else:
            _progress("MLX grammar engine")
            try:
                pip_install(
                    "Installing the MLX grammar engine (llguidance)",
                    "--no-cache-dir",
                    _LLGUIDANCE_PIN,
                )
            except SystemExit:
                _note(f"{_LLGUIDANCE_PIN} failed to install; MLX response_format stays unavailable")

    # 12. Patch metadata for single-env compatibility
    _finalize_ran = _dd_deps_ran or _dd_ran or _patch_metadata_is_pending()
    _progress("finalizing" if _finalize_ran else "finalizing (satisfied, skipped)")
    _record_step("patch-metadata", "ran" if _finalize_ran else "skipped")
    if _finalize_ran:
        _run_patch_metadata()

    # 13. Final torch repair. Steps above can pull CUDA torch from PyPI, so repair last.
    torch_flavor_tag = ""
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress(_torch_step_label("final"))
        _torch_before_repair = str(_probe_installed_torch_version() or "")
        # False = required repair the mirror cannot express: abort, never leave the mirror.
        if _ensure_cuda_torch() is False:
            return 1
        if _ensure_rocm_torch() is False:
            return 1
        if _ensure_xpu_torch() is False:
            return 1
        if _ensure_cpu_torch() is False:
            return 1
        # Last, after every torch migration: the swap keys off the installed +xpu label.
        _ensure_xpu_triton()
        # Step 4 chose torchao from the torch these repairs then moved (e.g. XPU <2.11 below torchao 0.18).
        _torch_after_repair = str(_probe_installed_torch_version() or "")
        if _torch_after_repair and _torch_after_repair != _torch_before_repair:
            _note(
                f"torch moved from {_torch_before_repair or 'unknown'} to "
                f"{_torch_after_repair} during the repair -- re-selecting torchao"
            )
            _install_torchao_for_torch(_torch_after_repair)
        # Unguarded: torch==2.10.0 accepts 2.10.0+rocm7.1, and an earlier run may have moved it.
        _evict_xformers_built_for_another_torch(scope = "linux torch repair", family_only = True)
        _evict_xformers_requiring_another_torch()

    # 13w. Windows torch flavor invariant: last, after with-deps steps re-resolved torch.
    if IS_WINDOWS and not NO_TORCH:
        _progress(_torch_step_label("flavor"))
        torch_flavor_tag = _expected_torch_flavor_tag()
        if not _ensure_expected_torch_flavor(torch_flavor_tag):
            return 1
        # A direct run has no setup.ps1 postlude to swap triton back; after the invariant.
        _ensure_xpu_triton()
    elif not NO_TORCH:
        # Resolved elsewhere too, for the record only.
        torch_flavor_tag = _expected_torch_flavor_tag()

    # 13x. Optional win_arm64 features excluded by metadata; after the invariant so xformers matches torch.
    _install_wheelhouse_optionals()

    # 13b. torchcodec pinned to the torch minor (markers cannot see torch); after the repair. On a probe
    # timeout read installed metadata rather than guess.
    _codec_torch_ver = None
    _codec_hosted = None
    if not NO_TORCH and (not PLATFORM_LACKS_TORCHCODEC_WHEEL or _wheelhouse_hosts("torchcodec")):
        _codec_torch_ver = _probe_installed_torch_version() or _installed_distribution_version(
            "torch"
        )
    if not NO_TORCH and PLATFORM_LACKS_TORCHCODEC_WHEEL:
        _codec_hosted = _wheelhouse_torchcodec_version(_codec_torch_ver)
    if NO_TORCH:
        _progress("torchcodec (skipped, no torch)")
    elif _codec_hosted:
        # The wheelhouse stands in for the missing wheel: --no-deps, pinned inside the torch window.
        _progress("torchcodec")
        if pip_install_try(
            f"Installing torchcodec=={_codec_hosted} from the Windows on ARM wheelhouse",
            "--no-deps",
            "--no-cache-dir",
            f"torchcodec=={_codec_hosted}",
            constrain = False,
        ):
            _note(f"windows on arm: installed torchcodec=={_codec_hosted} from the wheelhouse")
        else:
            _note(
                "windows on arm: could not install the wheelhouse torchcodec -- audio decoding stays disabled"
            )
    elif PLATFORM_LACKS_TORCHCODEC_WHEEL:
        _progress("torchcodec (skipped, no wheel for this platform)")
        if _wheelhouse_hosts("torchcodec"):
            _note(
                f"windows on arm: the wheelhouse torchcodec is outside the window torch "
                f"{_codec_torch_ver or 'unknown'} selects -- leaving audio decoding disabled"
            )
    elif not _codec_torch_ver:
        _progress("torchcodec (skipped, torch version unknown)")
        _note("could not read the installed torch version -- leaving torchcodec alone")
    elif _select_torchcodec_spec(_codec_torch_ver) is None:
        _progress("torchcodec (skipped, unsupported torch version)")
        _note(
            f"torch {_codec_torch_ver} is below the oldest supported torchcodec pairing "
            f"(torch 2.{_TORCHCODEC_MIN_KNOWN_MINOR}) -- leaving torchcodec alone"
        )
    elif not _torchcodec_spec_is_installable(_select_torchcodec_spec(_codec_torch_ver)):
        # No wheel for this window: skip, since attempting would end the install.
        _progress("torchcodec (skipped, no wheel for this torch on this platform)")
        _note(
            f"torch {_codec_torch_ver} wants {_select_torchcodec_spec(_codec_torch_ver)}, "
            "which publishes no wheel here -- leaving audio decoding disabled"
        )
    else:
        _progress("torchcodec")
        _codec_spec = _select_torchcodec_spec(_codec_torch_ver)
        # Pin the index too: torchcodec ships per accelerator, and the wrong index loads nothing.
        _codec_index = _torchcodec_index_url(_codec_torch_ver, _codec_spec)
        _codec_args = ("--no-deps", "--no-cache-dir")
        _codec_rebuild = False
        _codec_have = _installed_distribution_version("torchcodec") or ""
        _codec_want = None
        if _codec_index:
            _codec_args += ("--index-url", _codec_index)
            # An in-window codec satisfies pip, so check its local tag (+cuNNN / +cpu; PyPI has none) against
            # the tag the PIN fetches (xpu is served cpu).
            _codec_want = _torchcodec_index_tag(_codec_torch_ver)
            if _codec_have and (
                _codec_want is None or _codec_have.partition("+")[2].strip().lower() != _codec_want
            ):
                _codec_args += ("--force-reinstall",)
                _codec_rebuild = True
        _codec_skip = (
            _may_skip_on_evidence()
            and not _codec_rebuild
            and _codec_spec_is_satisfied(_codec_spec, _codec_have)
        )
        _safe_print(
            f"   torch {_codec_torch_ver} detected -- "
            + (
                f"{_codec_spec} is already installed"
                if _codec_skip
                else f"installing {_codec_spec}"
            )
            # Redacted for display only: printed straight to the terminal, not via _redact_install_output.
            + (f" from {_strip_index_url_credentials(_codec_index)}" if _codec_index else "")
            + (" (replacing a build from another index)" if _codec_rebuild else "")
        )
        # pip_install_try: audio is optional and pip_install exits on failure.
        if _codec_skip:
            _record_step("torchcodec", "skipped")
            _codec_ok = True
            _codec_fellback = False
        else:
            _record_step("torchcodec", "ran")
            _codec_ok = pip_install_try("Installing torchcodec", *_codec_args, _codec_spec)
            _codec_fellback = False
        if not _codec_ok and _codec_index and _STEP_RESULTS.get("torchcodec") == "ran":
            # The leaf may not carry this window (cu129 lacks 0.8 / 0.9), so retry unpinned.
            _note(
                f"{_strip_index_url_credentials(_codec_index)} did not serve {_codec_spec} "
                "-- retrying from the default index"
            )
            # Keep --force-reinstall, or pip keeps the wrong-accelerator wheel.
            _codec_retry_args = [
                a
                for i, a in enumerate(_codec_args)
                if a != "--index-url" and _codec_args[i - 1] != "--index-url"
            ]
            # PyPI's wheel is still CUDA on Linux, so the NPP step below still applies.
            _codec_ok = pip_install_try("Installing torchcodec", *_codec_retry_args, _codec_spec)
            _codec_fellback = _codec_ok
        if not _codec_ok:
            _note(
                f"could not install {_codec_spec} -- audio decoding stays disabled, "
                "the rest of the install is unaffected"
            )
        elif _codec_index:
            # Not gated on the codec step. torchcodec <0.12 CUDA builds dlopen NPP, which is not a torch dep, so
            # --no-deps installs succeed and then fail to import.
            _npp_major = _cuda_major_for_npp(_codec_torch_ver, _codec_index)
            if _codec_fellback:
                # Pin dropped: probe every fallback, since an xpu host can still land PyPI's CUDA build.
                _npp_probed = _installed_torchcodec_cuda_major()
                if _npp_probed is not None and _npp_probed != _npp_major:
                    _note(
                        f"the unpinned torchcodec links CUDA {_npp_probed or 'nothing'} "
                        f"rather than {'CUDA ' + _npp_major if _npp_major else 'nothing'}, "
                        "which its torch tag implies -- matching NPP to the wheel"
                    )
                    _npp_major = _npp_probed
            _npp_spec = _npp_requirement(_npp_major) if _npp_major else ""
            _npp_name = re.split(r"[<>=!~]", _npp_spec, maxsplit = 1)[0].strip() if _npp_spec else ""
            _npp_have = _installed_distribution_version(_npp_name) if _npp_name else None
            if _npp_have and _spec_is_satisfied(_npp_spec, _npp_have):
                pass
            elif _npp_spec and not pip_install_try(
                "Installing torchcodec CUDA runtime (NPP)",
                "--no-cache-dir",
                _npp_spec,
            ):
                _note(
                    f"could not install {_npp_spec} -- torchcodec may fail to "
                    "import on a host without the CUDA toolkit, leaving audio disabled"
                )

    # 14. Final check (silent; third-party conflicts are expected). Only when this pass installed
    # something or the last run had no clean answer.
    _pip_check_ok = (_PASS_EVIDENCE or {}).get("pip_check_ok")
    if _INSTALL_ACTIONS > 0 or _pip_check_ok is not True:
        _pip_check_ok = (
            subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                **_windows_hidden_subprocess_kwargs(),
            ).returncode
            == 0
        )

    # 14b. Repair again: a core upgrade can itself leave a superseded record before the manifest write.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

    # 14c. write_manifest reads the installed version; skip_base never reaches the core phase.
    if not _repair_damaged_core_payload(
        _core_package_names(package_name),
        local_repo = local_repo,
        require_present = True,
    ):
        return 1

    # The manifest records the resident flavor under an unusable pin; unreadable is uncertifiable.
    if not NO_TORCH and _explicit_torch_index_is_unusable() and not _resident_torch_flavor_tag():
        _safe_print(
            "   [WARN] could not verify the resident PyTorch flavor after declining the "
            "configured mirror; refusing to mark this update complete."
        )
        return 1

    # 15. Record success, written last so an earlier kill leaves none.
    if (
        install_manifest.write_manifest(
            req_root = REQ_ROOT,
            steps_total = _TOTAL,
            package_name = package_name,
            no_torch = NO_TORCH,
            # Carry the old record forward, except for an unknown-family pin, whose record no longer applies.
            expected_torch_tag = _recordable_torch_flavor_tag(torch_flavor_tag),
            expected_torch_tag_pinned = bool(_recordable_torch_flavor_tag(torch_flavor_tag))
            and _expected_torch_flavor_was_pinned(_recordable_torch_flavor_tag(torch_flavor_tag)),
            woa_torch_index = os.environ.get("UNSLOTH_WOA_SELECTED_TORCH_INDEX"),
            # Digest the CURRENT REQ_ROOT: the core step may have replaced it.
            extra = {
                "pass_inputs": {
                    **install_manifest.pass_input_digests(REQ_ROOT),
                    **_plugin_digests,
                },
                "step_results": dict(_STEP_RESULTS),
                "known_unmet": _closure_record(),
                # The installed set that record describes; honoured only against the same set.
                "known_unmet_index": _installed_index_digest(),
                "bnb_rocm": _BNB_ROCM_PASS_PROVENANCE,
                "bnb_rocm_asset": _BNB_ROCM_PASS_ASSET,
                "pip_check_ok": _pip_check_ok,
                "uv_version": _uv_version(),
                "installer_python_tag": _installer_python_tag(),
            },
        )
        is None
    ):
        _safe_print(
            f"error: could not write {install_manifest.MANIFEST_NAME} to "
            f"{install_manifest.venv_root()}",
            file = sys.stderr,
        )
        return 1

    # 16. Apple Silicon: warn when the installed MLX stack is unusable, or the app silently comes up
    # chat-only. After the manifest: the probe can hang its full timeout.
    if IS_MAC_ARM and not NO_TORCH:
        _report_mlx_stack_health(skipped = _STEP_RESULTS.get("mlx") == "skipped")

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--amd-torch-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        _needs_pass = _amd_torch_needs_dependency_pass()
        # Exit 1 covers five states, so name the deciding one for CI; setup.sh discards both streams.
        _safe_print(
            f"{_AMD_FASTPATH_DECISION_MARKER}needs_pass={_needs_pass} no_torch={NO_TORCH} "
            f"is_linux={IS_LINUX} machine={platform.machine()!r} backend={_TORCH_BACKEND!r} "
            f"probe={_TORCH_RUNTIME_PROBE!r}"
        )
        sys.exit(0 if _needs_pass else 1)
    if sys.argv[1:] == ["--cuda-torch-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        sys.exit(0 if _cuda_torch_needs_dependency_pass() else 1)
    if sys.argv[1:] == ["--missing-torch-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        sys.exit(0 if _missing_torch_needs_dependency_pass() else 1)
    if sys.argv[1:] == ["--repair-diffusers-main"]:
        sys.exit(_repair_diffusers_main())
    if sys.argv[1:] == ["--prefetch-diffusers-main"]:
        sys.exit(_prefetch_diffusers_main())
    if sys.argv[1:] == ["--diffusers-main-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        sys.exit(0 if _diffusers_main_needs_dependency_pass() else 1)
    if any(_arg.startswith("-") for _arg in sys.argv[1:]):
        # Never let a malformed probe call fall through into a multi-gigabyte install.
        _safe_print(f"Unknown argument: {' '.join(sys.argv[1:])}")
        sys.exit(2)
    try:
        sys.exit(install_python_stack())
    except _InstallerReplaced:
        sys.exit(_rerun_replaced_installer())
