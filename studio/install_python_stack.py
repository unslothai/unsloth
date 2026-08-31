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


def _is_windows_arm64() -> bool:
    """Windows on ARM, machine arch rather than process arch: platform.machine() reports
    AMD64 under an emulated x64 Python, and PROCESSOR_ARCHITEW6432 is ARM64 in exactly
    that case. Mirrors Get-HostMachineArch in install.ps1 / setup.ps1."""
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
    future 7.3+), never one at/above the floor.

    A version below every tag (what an unreadable one reads as, on a bundled-runtime host)
    resolves no generic index at all, so the per-arch index is the only route left."""
    key = next((k for k in sorted(_ROCM_TORCH_INDEX, reverse = True) if ver >= k), None)
    return key is None or key < _ROCM_ARCH_INDEX_FLOOR


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
    # A mask hiding every GPU must not buy a force-reinstall onto an OLDER tag, minus
    # bitsandbytes, for the card the user hid. No probe below is filtered by a HIP mask, so
    # without this a container over a populated /sys/class/kfd is downgraded anyway. Above
    # the override, as in _runtime_gfx_target: hiding every GPU is a statement about this run.
    if _visible_masks_select_no_gpu():
        return False
    # Normalize a copied HIP gcnArchName (gfx906:sramecc-:xnack- -> gfx906) so the
    # feature-flag suffix does not defeat the exact comparison (mirrors device_type.py).
    override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":")[0]
    if override:
        return override == "gfx906"
    # Unmasked: "the SOLE arch" is a question about the machine. ROCR_VISIBLE_DEVICES filters
    # rocminfo before it answers, so a mask naming the MI50 on a mixed host would present it
    # as the only card and unlock the downgrade the sole-arch rule exists to withhold there.
    return set(_detect_amd_gfx_codes(ignore_visible_masks = True)) == {"gfx906"}


def _torch_below_211(installed_ver: str) -> bool:
    """True when an installed torch version string is readable and below 2.11.

    Unreadable reads as NOT below: this gates a --force-reinstall of a multi-GB stack, and
    a version string no regex can parse is not evidence the build is the broken one.
    """
    _m = re.match(r"\s*(\d+)\.(\d+)", installed_ver or "")
    return bool(_m) and (int(_m.group(1)), int(_m.group(2))) < (2, 11)


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


# Memoized `import torch` classification of the target venv, reset by pip_install() and
# pip_install_try(), the only things here that change what is installed. None means
# "not probed yet".
_TORCH_RUNTIME_PROBE: "tuple[bool, bool, str | None, str, str] | None" = None

# Prefix on the probe's own stdout line. Import chatter can arrive before the answer, and
# an atexit handler or a CUDA teardown notice can arrive after it, so "the last non-empty
# line" is not reliably ours; "the last line starting with this" is.
_TORCH_PROBE_MARKER = "UNSLOTH_TORCH_PROBE|"

# Prefix on the --amd-torch-needs-dependency-pass decision line: five states share exit 1,
# so a caller (CI above all) needs to read WHICH input decided. setup.sh discards the stream.
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
                    # torch.version is not guaranteed to exist. Reaching through it
                    # unguarded raises, which reads as "torch cannot import" and
                    # force-reinstalls a working venv.
                    "_v = getattr(torch, 'version', None); "
                    "h = getattr(_v, 'hip', '') or ''; "
                    "c = getattr(_v, 'cuda', '') or ''; "
                    f"print('{_TORCH_PROBE_MARKER}' + '|'.join((v, h, c)))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            # The probes this replaced each decoded with errors="replace". text=True alone
            # decodes strictly, and a UnicodeDecodeError is a ValueError, so an undecodable
            # byte in torch's import chatter would escape the except below and take the
            # installer down instead of falling back to the on-disk classifier. Reachable
            # wherever the console code page and the child's output disagree.
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
    if _marked:
        _fields = _marked[-1][len(_TORCH_PROBE_MARKER) :].split("|")
        version, hip, cuda = (_fields + ["", ""])[:3]
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


def _installed_torch_is_windows_rocm() -> bool:
    """Return True when the target venv currently has a Windows ROCm torch build.

    This is a belt-and-suspenders guard for the torchao override step: if the
    earlier ROCm install path failed to set _rocm_windows_torch_installed but the
    venv already contains a ROCm torch wheel, still skip torchao because it
    crashes on import on Windows ROCm.
    """
    if not IS_WINDOWS:
        return False
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran or not _importable:
        return False
    _ver = (_version or "").lower()
    return bool(_hip) or "rocm" in _ver or "rocmsdk" in _ver


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
# Kept verbatim: _index_url_join trims the path head, and only the path head. Trimming the
# whole URL eats a trailing "/" belonging to a query token, and "/" is in the base64
# alphabet. The Linux mirror passes its raw env value for the same reason.
_ROCM_WINDOWS_INDEX_BASE = (
    os.environ.get("UNSLOTH_ROCM_WINDOWS_MIRROR") or "https://repo.amd.com/rocm/whl"
)

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
                # RDNA 1 / Polaris is not an unknown card: naming an override there
                # sends the user after a fix that does not exist (#8529, #8458).
                _unsupported = _unsupported_gfx_arch_from_gpu_name(_names[_sel])
                if _unsupported:
                    # The CPU-only half is false under an explicit index pin, which is
                    # honoured for any arch, so a pinned run says what it is doing instead.
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
                    # Torch ends here, llama.cpp does not: Vulkan drives these cards
                    # (#8458 ran an RX 580 through it). PowerShell syntax because this
                    # branch is Windows-only: a pasted VAR=value parses there as a
                    # command name and sets nothing. Not on ARM64: setup.ps1 THROWS on
                    # that variable there, so this would abort the next update.
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


# GPU marketing-name → gfx arch table, mirroring setup.ps1's $nameArchTable.
# Most-specific first; first match wins. Covers only arches the ROCm
# prebuilts / AMD Windows torch indexes support; unknown names return None
# (callers then fall back cleanly to CPU).
_WIN_GPU_NAME_ARCH_TABLE: "list[tuple[str, str]]" = [
    # RDNA 4 (Navi 48: Radeon RX 9070 XT / 9070 GRE / 9070 / 9080, Radeon AI PRO R9700).
    # R9700 is listed separately: its name holds neither 9070 nor 9080, so it matched
    # nothing and fell through to CPU torch (#7624, #7307).
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


# GPU name -> gfx arch for AMD generations Unsloth's ROCm wheels do NOT cover: RDNA 1
# and Polaris 10/20/30 (unslothai#8529, #8458). Deliberately SEPARATE from
# _WIN_GPU_NAME_ARCH_TABLE: nothing here may ever route to a wheel index. AMD's TheRock
# ships RDNA 1 wheels, but not on the repo.amd.com indexes routed here, and never gfx803.
# Every (?!0) guard stops "RX 570" swallowing "RX 5700", so each row is correct on its
# own regardless of order. Names from LLVM's AMDGPU tables plus libdrm amdgpu.ids/pci.ids
# for the Navi 10/14 professional parts LLVM omits; nothing is guessed, so Polaris 11/12
# (RX 460/550/560, a different die) is left out.
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
        "   The ROCm mirror carries its credential in the URL query or fragment. pip\n"
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
    if IS_WINDOWS:
        return _windows_rocm_index_url(gfx_arch)
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    base = os.environ.get("UNSLOTH_AMD_ROCM_MIRROR") or "https://repo.amd.com/rocm/whl"
    return _index_url_join(base, arch_family)


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
            # The distribution named exactly `rocm` ("rocm[libraries]==7.13.0"), anchored so
            # rocm-sdk-core and triton-rocm do not match. Case-insensitive: Requires-Dist
            # keeps the author's spelling while pip compares normalized, and reading
            # "ROCm[libraries]" as absent would call a per-arch build generic.
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
        # One entry per agent / GPU section; fall back to dedup for flat output. amd-smi
        # names no agent and heads each device with a line-leading "GPU: N" (or "GPU[N]");
        # without that header two cards of one arch collapse and later ordinals read wrong.
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


# Arches the product name can establish on Linux whose wheels differ from the arch
# people spoof them to: exactly what _linux_amd_gfx_from_cpuinfo() returns, RDNA 3.5
# APUs ROCm did not support natively, so HSA_OVERRIDE_GFX_VERSION=11.0.0 became the
# circulated workaround. The correction below only ever fires for these.
_HSA_SPOOFABLE_PHYSICAL_GFX: frozenset[str] = frozenset({"gfx1151", "gfx1150", "gfx1152"})


# Arches the generic pytorch.org ROCm wheels carry, from torch.cuda.get_arch_list() on
# 2026-08-27. Identical across every generic wheel the pins below resolve (2.10.0+rocm7.0,
# 2.10.0+rocm7.1, 2.11.0+rocm7.1, 2.11.0+rocm7.2). The rocm6.3 wheel carries a subset,
# without gfx1150/gfx1151, which the Strix reroute covers. torch 2.13.0+rocm7.1 DOES add
# gfx1103, so raising the <2.11 cap in _ROCM_TORCH_PKG_SPECS["_default"] means rechecking.
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


# Arches whose only wheel route is unreachable once a second GPU is present, so they must
# never depose an integrated card. gfx906's rocm6.3 route is gated on it being the sole
# detected arch (_runtime_target_is_gfx906), which no mixed host satisfies.
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


def _gfx_has_a_wheel_route(gfx: "str | None") -> bool:
    """Whether ANY index this installer can pick carries kernels for ``gfx``.

    Either the generic pytorch.org wheel serves it or AMD publishes a per-arch index. An arch
    in neither (gfx1010 / RDNA 1) cannot be fixed by picking a different index, so it must
    never depose a card that can.
    """
    return bool(gfx) and (gfx in _GENERIC_ROCM_WHEEL_GFX or gfx in _GFX_TO_AMD_INDEX_ARCH)


# _GENERIC_ROCM_WHEEL_GFX was measured on the tags the pins resolve (rocm7.0 and up), but
# _ROCM_TORCH_INDEX also maps ROCm 6.0-6.4, whose wheels predate some of these arches. A host
# can resolve an old tag while the KERNEL names a new card (a current amdgpu beside a stale
# /opt/rocm), leaving that wheel with no kernels for it. AMD's matrices put production
# gfx1200/gfx1201 at ROCm 6.4. gfx1150/gfx1151 need an entry despite the Strix reroute, which
# is gated on a ROCm VERSION and so inactive on a bundled-runtime host reading 0.0; rocm7.0
# is the oldest tag measured to carry them.
_GENERIC_WHEEL_GFX_MIN_ROCM: "dict[str, tuple[int, int]]" = {
    # gfx950 (MI350X / MI355X) dates to ROCm 7.0 and its ISA is genuinely new. It has no
    # _GFX_TO_AMD_INDEX_ARCH entry, so the reroute question still answers no for it: this
    # entry is read by the tag choice, which can pick a generic tag that does carry it.
    "gfx950": (7, 0),
    "gfx1150": (7, 0),
    "gfx1151": (7, 0),
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
    # The tag the version selects, not the version: a 6.3.9 host takes the rocm6.3 wheel.
    # Below the oldest known index nothing resolves, and a wheel older than every tag this
    # installer knows predates the arches those tags were measured to add; reading that as
    # "support unknown" preserves exactly the build that cannot run.
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
    # An empty (or "-1") mask selects NO GPU, deliberately, per _visible_devices_pinned.
    # Decided before any probe runs, because no probe is filtered the way the reroutes need:
    # only ROCR_VISIBLE_DEVICES reaches rocminfo, and amd-smi and KFD sysfs are filtered by
    # nothing. _pick_visible_index would then map the no-GPU mask onto index 0 and reinstall
    # a multi-GB stack for the card the user hid. A mask naming a device is unaffected.
    if _visible_masks_select_no_gpu():
        return None, [], None, []
    # An explicit arch outranks the probes, not merely the cases where they say nothing.
    # install.sh, _runtime_target_is_gfx906 and _detect_windows_gfx_arch all read it first;
    # resolving it last here made this the one place the escape hatch could be overruled by
    # the hardware it exists to override. The masks stay above it: hiding every GPU is a
    # statement about this run, while the arch names what to build for. Split on ":" as the
    # probe normalization does, or a copied gcnArchName ("gfx1151:sramecc-:xnack-") keys
    # no routing table.
    _explicit_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(":", 1)[0]
    if _explicit_gfx:
        # Returning early skips _hsa_spoofed_physical_gfx, so answer the spoof here:
        # HSA_OVERRIDE_GFX_VERSION=11.0.0 is the workaround half these hosts carry, and
        # leaving it set while installing per-arch wheels is #7331 exactly (ROCr keeps
        # handing torch a gfx1100 agent the gfx1151 wheels have no code for). Only when the
        # override names a DIFFERENT arch: naming the arch you spoofed TO is deliberate.
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
        # With no userland reading to distrust the spoof check above declined, but the
        # runtime is still spoofed and the reroutes install code for the physical arch alone
        # (#7331). amdkfd writes gfx_target_version and ROCr never touches it, so a
        # single-arch kernel reading the override contradicts IS the corroborated spoof.
        if physical_gfx is None and len(set(gfx_devices)) == 1:
            _override_arch = _hsa_override_gfx_arch(os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
            if _override_arch is not None and _override_arch != gfx_devices[0]:
                physical_gfx = gfx_devices[0]
    if not gfx_devices and inferred_linux_gfx:
        # Nothing enumerated a device, so #7305's precedence (a runtime-visible arch outranks
        # the product name) has nothing left to protect. One guess is not a device list, so
        # an ordinal past its only entry indexes nothing, while _pick_visible_index's
        # out-of-range rule would answer with the guess and commit a per-arch reinstall to
        # it. Decline, unless the arch was named outright. Probed against a length nothing
        # can exceed, so a real ordinal comes back as itself rather than folding onto 0.
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
    # The machine as the probes saw it, before the ROCr layer reduces it to a lone survivor.
    # _gfx_route_on_host needs that shape, and "before the ROCr layer" holds only for a probe
    # ROCr does not filter: rocminfo runs on the ROCm user-mode stack, so a mask-selected MI50
    # beside a dGPU would read as single-architecture, the one premise _MIXED_HOST_UNROUTABLE
    # denies, and be granted the rocm6.3 tag -- a persistent downgrade of a shared install on
    # the strength of one session's mask. Ask the whole machine instead. Provenance is read
    # once, up here: the re-probe rewrites it, and the ROCr composition below asks about the
    # probe that produced ``gfx_devices``, not the extra one asked on its behalf.
    _probe_source = _LAST_AMD_GFX_PROBE
    if _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ:
        try:
            _unmasked = _detect_amd_gfx_codes(dedup = False, ignore_visible_masks = True)
        except Exception:
            _unmasked = []
        host_codes = list(dict.fromkeys(_unmasked or gfx_devices))
    else:
        host_codes = list(dict.fromkeys(gfx_devices))
    # The two mask layers COMPOSE: ROCr filters first and HIP indexes the survivors (AMD's
    # GPU isolation guide: "the ROCR env var is processed first, which then reduces the
    # number of GPUs that HIP can select from"). Only rocminfo is renumbered by ROCr, whenever
    # ROCR_VISIBLE_DEVICES is set and not only when it is the first mask. amd-smi and KFD
    # sysfs are filtered by nothing, so for those the ROCr layer is applied here first --
    # indexing the unfiltered list picks a GPU the runtime never exposes.
    rocr_applied = _probe_source == "rocminfo" and "ROCR_VISIBLE_DEVICES" in os.environ
    _unlike_adapters = len(set(gfx_devices)) > 1
    if not rocr_applied:
        # amd-smi enumerates in DISCOVERY order over its KFD view, while the masks index
        # HIP/ROCr order from the KFD node id. The two disagree on real hardware (MI350X
        # SPX/NPS1), which is why setup.sh translates through `amd-smi list -e`'s HIP_ID map
        # and declines when it is unavailable or not 1:1. No such map is read here, so on
        # unlike adapters an untranslated ordinal names another card's arch. An UNSET mask is
        # no safer: it still selects HIP ordinal 0, and discovery-order 0 can be the other
        # card. Like adapters are unaffected either way, every ordinal giving the same arch.
        _discovery_ordered = _probe_source == "amd-smi"
        if _discovery_ordered and _unlike_adapters:
            # Discovery order is unusable, the kernel's topology is not: KFD nodes ARE the
            # order HIP and ROCr index. Reading them swaps an ordering no ordinal fits for one
            # that does, rather than declining the repair the reported shape needs (#9396: a
            # Ryzen APU beside a dGPU, with amd-smi installed). Only on an equal device count
            # -- a KFD list of another length is a different view of the machine.
            _kfd_ordered = _kfd_gfx_targets()
            if len(_kfd_ordered) == len(gfx_devices):
                gfx_devices = _kfd_ordered
                _unlike_adapters = len(set(gfx_devices)) > 1
                _discovery_ordered = False
        gfx_devices, _rocr_unresolved = _rocr_visible_subset(gfx_devices)
        # A UUID names a device this cannot place. Judged against the list BEFORE the mask
        # was applied: dropping the tokens that did resolve can leave one arch standing and
        # hide the very ambiguity the UUID created, and "0,GPU-..." is a form AMD documents.
        if (_rocr_unresolved or _discovery_ordered) and _unlike_adapters:
            # The message below offers UNSLOTH_ROCM_GFX_ARCH as the way through, so honour it:
            # naming the arch outright is the one reading no ordinal or UUID contradicts. Read
            # from the environment, not inferred_linux_gfx, which also carries the product
            # guess -- weaker than the probe, and not for a host the probe left open.
            _named_gfx = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower()
            if _named_gfx:
                # The named arch leads the returned set. Stacked masks can leave a list it is
                # not in (ROCr keeps the gfx1200, the user names the gfx1103), and the caller
                # reads that set to decide which arch lacks kernels, so a target missing from
                # it is selected and then never repaired.
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
        # Unpinned mixed host: only enumeration order put the integrated GPU first, and the
        # wheel family is picked for ONE arch, so letting the APU decide strands the discrete
        # card. Same #7776 preference _detect_windows_gfx_arch applies, stopping at a set mask
        # for the same reason -- a pin is the user naming a device. gfx906 is excluded as a
        # candidate because its only route runs through _runtime_target_is_gfx906, which no
        # mixed host satisfies: naming it installs a rocm7.x wheel whose BLAS has no gfx906
        # kernels and strands BOTH cards. It keeps its UNSLOTH_ROCM_GFX_ARCH opt-in.
        _others = [
            g
            for g in gfx_devices
            if g not in _SHADOWING_INTEGRATED_GFX and g not in _MIXED_HOST_UNROUTABLE
        ]
        # Prefer a discrete card the installer can serve: deposing a routable APU for one no
        # index carries (an RDNA 1 dGPU beside a gfx1103) trades a repairable GPU for nothing.
        # Same shape as _detect_windows_gfx_arch's _withWheels guard.
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
    # [0-9] rather than str.isdigit()/\d, both of which accept non-ASCII digits
    # ("١١.0.0" would read as 11.0.0 here and be rejected by install.sh's awk).
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
                f"   {source} reports {inferred_gfx} -- {probed} is a spoof of the "
                f"physical arch.\n"
            )
            return inferred_gfx
        # Say so rather than leaving "Checking whether..." hanging: on a real gfx1100
        # card in a Ryzen AI Max chassis this is the CORRECT outcome.
        _safe_print(
            f"   {source} does not corroborate a spoof "
            f"({physical or 'no answer'}); keeping {probed}.\n"
        )
        return None

    # 1. The kernel, which the override cannot reach. Decisive either way: if it
    # answers at all, no weaker source overrules it.
    kfd = _kfd_gfx_targets()
    if kfd:
        return _confirm(kfd, "KFD topology sysfs")

    # 2. The runtime, asked again without the override and without the visible
    # masks, so a mask cannot hide the second GPU that would veto the correction.
    _saved_probe = _LAST_AMD_GFX_PROBE
    try:
        reprobed = _detect_amd_gfx_codes(
            dedup = False, ignore_hsa_override = True, ignore_visible_masks = True
        )
    except Exception:
        reprobed = []
    finally:
        _LAST_AMD_GFX_PROBE = _saved_probe
    # A re-probe that still answers `probed` is evidence FOR the probe: the override
    # went away and the name did not move, so it is real silicon. Declining here is
    # why a genuine gfx1100 dGPU in a Ryzen AI Max chassis keeps its own wheels.
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

# The HIP layer alone. CUDA_VISIBLE_DEVICES is HIP's alias on AMD, so both name the same
# layer and stay first-set-wins between themselves; ROCR_VISIBLE_DEVICES is the layer
# BENEATH and is applied separately by _rocr_visible_subset.
_HIP_LAYER_MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# A device count no host reaches, for asking what index a mask NAMES rather than which entry
# of a real list it selects: any ordinal below it resolves to itself, not to the 0 fallback.
_INDEX_PROBE_LEN = 1 << 20


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
    # An out-of-range index keeps the whole list, deliberately: _pick_visible_index warns and
    # falls back to GPU 0 for that value (matching setup.ps1's Resolve-VisibleGpuIndex), and a
    # stricter rule here would split the two. ROCR_VISIBLE_DEVICES=1 on a one-GPU box is a
    # typo, and reading it as "no GPU" withdraws the repair from the hosts this exists for.
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
        # A repin BETWEEN per-arch leaves (gfx1151 -> gfx120X-all, the swapped-card case)
        # keeps every shape the version string carries: both are torch 2.11 with a three-part
        # +rocmA.B.C tag. Judged on that alone the pin the user just edited reads as satisfied
        # and is dropped in silence. The installed `rocm` meta-package names the family
        # outright; None means unknowable, so this can only ADD a mismatch, never talk one away.
        _family = _installed_rocm_wheel_family()
        if _family is not None and _family != leaf:
            return True
        # 2.11-allowlist arches expect the AMD per-arch wheel (three-part +rocmA.B.C,
        # torch 2.11+); a generic or pre-2.11 build is a mismatch. Asked before the family
        # can settle it, because these leaves have a FLOOR: a matching family on a 2.10 build
        # is the _grouped_mm bug, which is the one thing the pin has to keep repairing.
        if leaf in _ROCM_GFX_TORCH211_LEAVES:
            return not (_inst_is_211 and _inst_is_perarch)
        # Decisive the other way too, on leaves with no floor. The heuristic below reads any
        # 2.11 build as a mismatch, since that is what a build from some OTHER index looks
        # like -- but these leaves serve 2.11 as well, and the family says this one came from
        # the pinned index. Without it a correctly pinned gfx110X host force-reinstalls under
        # the legacy torch<2.11 cap on every update.
        if _family is not None and _inst_is_perarch:
            return False
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
    # or "cpu". Un-importable means missing or broken: without a pin the base install
    # owns it, but a pinned CUDA index reinstalls it below.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        return
    if not _importable:
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
    if _version is None:
        # Nothing readable came back, so classify nothing: this is where the per-path
        # probe returned when its stdout held no line.
        return
    # marker | +cuXXX local tag | release | family from torch.version.cuda. The last is the
    # only CUDA clue an untagged wheel gives: PyPI forbids the local +cuXXX version.
    _ver = _version.lower()
    _cu_match = re.search(r"\+(cu\d+)", _ver)
    _marker = "hip" if (_hip or "rocm" in _ver) else ("cuda" if _cuda else "cpu")
    _installed_cu = _cu_match.group(1) if _cu_match else ""
    _installed_release = _ver.split("+", 1)[0]
    _runtime_cu = ("cu" + _cuda.replace(".", "")) if _cuda else ""
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

    # Un-importable either way installs from the pin below. One shared probe bounds it.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
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
        _why = "torch could not be probed"
    elif _importable:
        if _version is None:
            return  # unreadable -- the base install step handles a missing torch
        # Flavour AND range: a migrated 2.5+xpu venv is broken, not correct, so the tag
        # alone is not enough. Range matches _XPU_TORCH_PKG_SPEC.
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

    # Classify the torch family. Un-importable means missing or broken, and the
    # explicit CPU pin reinstalls it below.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    if not _ran:
        # A hung import is the wedged-driver case this pin exists to rescue, so returning here
        # made the pin a no-op on exactly that host. Classify off disk instead, and only go on
        # for a GPU label: a merely slow CPU-only box must not reinstall torch every update.
        if not _is_gpu_torch_label(_installed_torch_label_on_disk()):
            return
    if not _ran or not _importable:
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
    if _version is None:
        return  # unreadable -- the base install step handles a missing torch
    # '+xpu' too: an XPU wheel sets neither torch.version.cuda nor .hip, so without it a
    # working Intel build reads as "cpu" and the CPU pin over it does nothing.
    _ver = _version.lower()
    _is_gpu_build = (
        bool(_hip)
        or "rocm" in _ver
        or bool(_cuda)
        or bool(re.search(r"\+cu\d+", _ver))
        or "+xpu" in _ver
    )
    if not _is_gpu_build:
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


def _amd_torch_needs_dependency_pass() -> bool:
    """Return True when setup must run the dependency pass to repair non-ROCm torch.

    Scope is the wheel family, not the ROCm family: any ROCm marker keeps the fast path
    even when the repair would reroute it. Fails closed on an uncertain host or torch,
    and never installs.
    """
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
        # A hidden layer either side leaves no target to classify. Same reading the routing
        # guard uses, so the two can never drift.
        if _visible_masks_select_no_gpu():
            return False
        # Match the repair's two host signals: a visible GPU or an inferred/named arch.
        _inferred_gfx = _infer_linux_amd_gfx_arch()
        _rocm_visible = _has_rocm_gpu()
        if not _rocm_visible and not _inferred_gfx:
            return False
        # A mixed-arch host used to end it here, because probe and mask order could disagree
        # with nothing to resolve it. _runtime_gfx_target now composes both mask layers and
        # returns no target ONLY when the selection is genuinely unreadable, so gate on its
        # answer, not the shape of the host: a mixed box with HIP_VISIBLE_DEVICES naming a
        # card is not ambiguous, and its generic wheel can still lack kernels for it.
        _selected_gfx, _, _selected_spoof, _selected_host = _runtime_gfx_target(_inferred_gfx)
        if _selected_gfx is None:
            return False
        # Every arm below is about the wheel. A corroborated spoof is not: the family can be
        # perfect while ROCr presents an ISA those wheels have no code for, and only
        # _ensure_rocm_torch clears the variable (#7331).
        if _selected_spoof is not None:
            return True
        _pre_ran, _pre_imp, _pre_ver, _pre_hip, _pre_cuda = _probe_torch_runtime()
        _pre_torch = (_pre_ver or "").lower() if (_pre_ran and _pre_imp) else ""
        # The compatibility reroutes have no version floor either, so a host whose ROCm
        # version will not read must not be turned away before they are asked. Only with a
        # torch that reads back: an unreadable one is not evidence its wheels are wrong.
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
        # Both per-arch repairs are version-independent, and the host with no readable
        # version is the one they exist for: a bundled-runtime install has no system ROCm
        # to read. Requiring a version here refuses them at the door.
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
    # Torch IS a ROCm build, which used to end it. The per-arch repairs act on exactly those
    # hosts, so keeping the fast path makes them unreachable from `studio update`: a swapped
    # card, or the reported gfx1103 on a generic wheel, would be repaired on a fresh install
    # and never again. The pass costs a dependency resolution, not a reinstall, since
    # _ensure_rocm_torch still keeps a family that matches. So ask only what it asks, only
    # where it would act, and not under a pin, which commits to an index regardless.
    _pin = _explicit_rocm_torch_index_url()
    if _pin is not None:
        # A pin skips the hardware questions, but not its own: _ensure_rocm_torch reinstalls
        # when the installed local tag names a different family than the pin, and returning
        # here unconditionally meant a changed UNSLOTH_TORCH_INDEX_URL never reached it.
        return _rocm_pin_family_mismatch(_pin, _version.lower())
    _tail_gfx, _, _tail_spoof, _tail_host = _runtime_gfx_target(_infer_linux_amd_gfx_arch())
    _tail_ver = _detect_rocm_version() or (0, 0)
    # A corroborated HSA spoof is cleared by _ensure_rocm_torch and only by it: the family can
    # match perfectly while ROCr presents an ISA the wheels have no code for (#7331), so the
    # wheel question alone keeps the fast path and the variable stays set.
    if _tail_spoof is not None:
        return True
    # The two compatibility reroutes are repairs like any other, and asking only the
    # missing-kernel question skipped both on update: Strix on a generic wheel below the AMD
    # floor still needs the 7.13 fixes, and a sole gfx906 above rocm6.3 has no BLAS kernels.
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
        # A family that reads back is the answer, whichever way it points: an AMD build for
        # ANOTHER arch sits at the same floor tag and carries none of this one's kernels.
        return _family == (leaf or "").lower()
    _tag = re.search(r"\+rocm(\d+)\.(\d+)", installed_ver or "")
    return bool(_tag) and (int(_tag.group(1)), int(_tag.group(2))) >= _ROCM_ARCH_INDEX_FLOOR


def _rocm_compat_reroute_pending(
    runtime_gfx: "str | None", ver: "tuple[int, int]", installed_ver: str
) -> bool:
    """Whether a compatibility reroute _ensure_rocm_torch performs has not been applied yet.

    Neither reroute is about missing kernels, so neither is visible to the wheel-family
    question: Strix wants AMD's 7.13 build over any generic one below the floor, and gfx906
    wants the last tag whose BLAS still carries it. Both compare against what is installed,
    so a host already on the right wheels keeps the fast path.
    """
    if not runtime_gfx:
        return False
    if runtime_gfx in _HSA_SPOOFABLE_PHYSICAL_GFX and _strix_needs_amd_arch_index(ver):
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
            # Only when some index can serve the target ON THIS HOST: a gfx1010, or a
            # mask-selected gfx906 beside another card, has a real mismatch and nowhere to go.
            # _ensure_rocm_torch declines on the same question, and promising a repair it
            # refuses buys a dependency pass on EVERY update and never a working torch.
            return _gfx_route_on_host(runtime_gfx, host_codes)
        # The right SHAPE, not necessarily a working build: below the 2.11 floor these leaves
        # carry the _grouped_mm bug, and answering False on a 2.10 build would keep the fast
        # path and leave it in place on every update.
        _ran, _importable, _ver, _hip, _cuda = _probe_torch_runtime()
        return _leaf in _ROCM_GFX_TORCH211_LEAVES and _torch_below_211(
            (_ver or "").lower() if (_ran and _importable) else ""
        )
    if _owns_sdk:
        # A per-arch install whose family will not read back. _ensure_rocm_torch cannot skip
        # on a family it never read, so it would reinstall the stack on EVERY update once the
        # fast path stops hiding it. Leave it alone.
        return False
    # Which arches a generic wheel carries belongs to THAT wheel, so read the tag off the
    # installed torch when it states one; ``ver`` is the HOST's, and the two part company (pin
    # rocm6.3 once, or upgrade /opt/rocm, and a gfx1200 box on 2.9.1+rocm6.3 looks healthy
    # forever). Both readings of "no code for this card" are needed: the reroute question, and
    # the tag floor for a target with no index to be rerouted TO. Asking only the first kept
    # the fast path on a gfx950 on rocm6.3, so the install path's floor was never reached from
    # `studio update` -- repaired once on a fresh install and never again.
    _installed_tag = _installed_generic_rocm_tag() or ver
    return _generic_rocm_wheel_lacks_kernels(
        runtime_gfx, _installed_tag
    ) or _generic_only_target_below_floor(runtime_gfx, _installed_tag)


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
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_ok = _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
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
        # Whether torch already links against HIP.
        _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
        _torch_already_rocm = (
            _ran and _importable and (bool(_hip) or "rocm" in (_version or "").lower())
        )
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
        # A host running the wheels' own bundled ROCm has no system version to read, and is
        # the same host the missing-kernel route exists for. That route has no version floor,
        # so returning here on an arch the generic wheel cannot serve refuses the repair at
        # the door. The second target resolution is confined to this branch.
        _unknown_ver_gfx, _, _, _unknown_ver_host = _runtime_gfx_target(None)
        _uv_ran, _uv_imp, _uv_ver, _uv_hip, _uv_cuda = _probe_torch_runtime()
        _unknown_ver_torch = (_uv_ver or "").lower() if (_uv_ran and _uv_imp) else ""
        # A per-arch install can also outlive the GPU it was made for: swap a gfx1200 card
        # into a box on gfx110X-all wheels and the generic index would serve it, so the clause
        # above lets the exit stand while those wheels carry no gfx1200 kernels. A matching
        # family below its 2.11 floor is the same story. This is the reading the setup
        # preflight uses, so anything narrower lets the pass run and then declines it.
        if (
            _rocm_pin is None
            and not _inferred_linux_gfx
            and not _generic_rocm_wheel_lacks_kernels(_unknown_ver_gfx)
            and not _rocm_torch_family_needs_repair(_unknown_ver_gfx, None, _unknown_ver_host)
            # The Strix reroute has no version floor to fail: with no tag resolving, the
            # per-arch index is the only route these arches have, and exiting here left a
            # visible Strix host on whatever non-ROCm torch it already had.
            and not _rocm_compat_reroute_pending(_unknown_ver_gfx, (0, 0), _unknown_ver_torch)
        ):
            _safe_print("   ROCm detected but version unreadable -- skipping torch reinstall")
            return
        # Explicit pin or inferred gfx: the index drives the install.
        ver = (0, 0)

    # Whether torch links against HIP, capturing the installed ROCm tag for pin-mismatch
    # detection. Marker is the HIP version, else a "rocm" sentinel when only the version
    # string flags ROCm; empty = CPU/CUDA torch, or un-probeable, which reinstalls.
    _ran, _importable, _version, _hip, _cuda = _probe_torch_runtime()
    _installed_torch_ver = (_version or "").lower() if (_ran and _importable) else ""
    _hip_marker = ""
    if _ran and _importable:
        _hip_marker = _hip if _hip else ("rocm" if "rocm" in _installed_torch_ver else "")
    has_hip_torch = _hip_marker != ""

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
    _inferred_arch_installed = False
    if (
        _inferred_linux_gfx
        and not has_hip_torch
        and _rocm_pin is None
        and (_gfx_override_env or not _has_rocm_gpu())
        # This branch installs for the inferred arch without asking the mask layers, so an
        # ordinal the guess cannot account for reaches the wheel the preflight already
        # declines. An explicit arch is exempt above, and no ordinal contradicts it.
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
            # The same reconciliation the reroutes below make: these wheels carry
            # _inferred_linux_gfx code objects alone, and a spoof naming another arch has ROCr
            # hand them a device none of it matches (#7331). The install is already committed
            # to that arch, so declining here only guarantees it is unusable.
            if _hsa_spoof_contradicts(_inferred_linux_gfx):
                _clear_confirmed_hsa_spoof(_inferred_linux_gfx)

    # An explicit UNSLOTH_ROCM_GFX_ARCH=gfx906 pins the runtime target to the
    # MI50 / Radeon VII path; it must win over the Strix probe-order detection
    # below (a mixed Strix + MI50 host could otherwise route to gfx1151), so the
    # Strix override is skipped when it is set.
    _gfx906_arch_override = (os.environ.get("UNSLOTH_ROCM_GFX_ARCH") or "").strip().lower().split(
        ":"
    )[0] == "gfx906"

    # Where the two AMD per-gfx reroutes below deposit their choice. Strix Halo / Point
    # (gfx1151 / gfx1150) need AMD's per-gfx index (2.11+rocm7.13) because every generic
    # pytorch.org index lacks the fixes (ROCm 7.1 segfaults in _grouped_mm); see
    # _strix_needs_amd_arch_index for that floor. The second reroute has no floor and fires
    # for an arch the generic wheel carries no kernels for at all.
    _arch_index_url: "str | None" = None
    _arch_index_pkgs: "tuple[str, str, str] | None" = None
    # An explicit ROCm pin wins; otherwise both reroutes share one hardware probe. Skipped
    # once the inferred-arch install above has run: it resolves the same index, so re-deriving
    # it here only force-reinstalls what was just downloaded.
    if (
        _explicit_rocm_torch_index_url() is None
        and not _gfx906_arch_override
        and not _inferred_arch_installed
    ):
        _runtime_gfx, gfx_codes, _physical_gfx, _host_codes = _runtime_gfx_target(
            _inferred_linux_gfx
        )
        _strix_gfx = {"gfx1151", "gfx1150", "gfx1152"}
        # Only the Strix reroute has a ROCm-version floor.
        _detected_strix = (
            _strix_gfx.intersection(gfx_codes) if _strix_needs_amd_arch_index(ver) else set()
        )
        if _detected_strix:
            if _runtime_gfx in _strix_gfx and _already_on_amd_arch_leaf(
                _GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx), _installed_torch_ver
            ):
                # Already the build this branch would fetch. It force-reinstalls a multi-GB
                # stack and _ensure_rocm_torch runs twice per install, so acting on the arch
                # alone re-downloads it on every install and every later update.
                _safe_print(
                    f"   torch already runs on the AMD {_runtime_gfx} wheels; keeping it.\n"
                )
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_runtime_gfx)
            elif _runtime_gfx in _strix_gfx:
                _selected_gfx = _runtime_gfx
                # One owner for the mirror env var and the arch-to-leaf map, shared with
                # the inferred-arch install above and the missing-kernel route below.
                _arch_index_url = _amd_arch_index_url(_selected_gfx)
                _arch_index_pkgs = (
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
                    f"   ({_strip_index_url_credentials(_arch_index_url)}) which serves torch\n"
                    f"   2.11.0+rocm7.13.0 with AMD's gfx1150/gfx1151 fixes (more reliable than\n"
                    f"   the generic pytorch.org rocm7.2 index on ROCm 7.3+ hosts).\n"
                )
                # Only on this branch: these wheels carry _selected_gfx kernels, so
                # the runtime must stop reporting the spoofed arch or they have no
                # code for the device (#7331). Never on the paths that keep generic
                # wheels, where the override is the only source of usable kernels.
                if _physical_gfx is not None:
                    _clear_confirmed_hsa_spoof(_selected_gfx)
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                _safe_print(
                    f"   Strix GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects a non-Strix runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

        # If the generic wheel lacks the runtime target, replace it from AMD's per-arch index.
        # No ROCm-version floor here: whichever generic index a host's ROCm version picks,
        # its wheel carries no kernels for these targets (see _GENERIC_ROCM_WHEEL_GFX).
        if _arch_index_url is None:
            # Judged by the installed wheel when it is a generic build naming its own tag,
            # since ``ver`` is the HOST's. ``ver`` still chooses the tag for a reinstall
            # below, which IS a question about the host.
            _kernel_ver = (
                has_hip_torch and not _torch_requires_rocm_sdk() and _installed_generic_rocm_tag()
            ) or ver
            _missing_kernels = {
                g for g in gfx_codes if _generic_rocm_wheel_lacks_kernels(g, _kernel_ver)
            }
            # A sub-2.11 build of a floor leaf is broken wherever it came from, and several of
            # those leaves (gfx120X-all, and gfx1150/gfx1151 when an unreadable ROCm version
            # leaves the Strix arm inactive) serve GPUs the generic wheel DOES list. Gating
            # the floor on missing generic kernels never reaches them, so the preflight forces
            # a pass this block declines. The floor belongs to the family, not to the visit.
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
                # Already running on this family's wheels. The reroute below is a
                # --force-reinstall --no-cache-dir of the multi-GB stack and _ensure_rocm_torch
                # runs twice per install, so a leaf derived from hardware alone re-downloads
                # it twice per install and again on every update. Act only on a family read
                # back positively: an unknowable one is None here and reinstalls, as does a
                # CPU/CUDA torch, and _torch_requires_rocm_sdk rejects a stale `rocm` orphan.
                # The right SHAPE is not a working build -- on these leaves a sub-2.11 wheel
                # carries the _grouped_mm bug, and skipping on family alone preserves it
                # forever. An unreadable ROCm version routes gfx1152 here rather than through
                # the Strix branch, so this is the only floor it meets.
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
                    # Keeping the wheels is not keeping the status quo: they carry the
                    # PHYSICAL arch alone, so a spoof left set has the runtime keep reporting
                    # the one arch they have no code for (#7331). The skip needs the clear as
                    # much as the reinstall arm does.
                    if _physical_gfx is not None:
                        _clear_confirmed_hsa_spoof(_runtime_gfx)
                elif _leaf is not None:
                    _arch_index_url = _amd_arch_index_url(_runtime_gfx)
                    # Keep older per-arch builds valid while bounding companion versions,
                    # except on the leaves whose sub-2.11 builds carry the _grouped_mm bug.
                    # gfx1152 reaches this branch rather than the Strix one when an unreadable
                    # ROCm version reads as 0.0, below the Strix floor.
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
                # Nothing to reroute -- the installed per-arch wheels are the target's own
                # family, which is why neither condition above fires. The spoof is still
                # exported and these wheels carry the PHYSICAL arch alone, so the runtime
                # keeps asking them for code they do not have (#7331). Every arm that KEEPS
                # matching wheels clears it; this one kept them and reached no arm at all.
                # Judged on a family read back from a torch that owns it, never the target.
                _clear_confirmed_hsa_spoof(_runtime_gfx)

        # A per-arch install can outlive the GPU it was made for: add a dGPU, or point
        # HIP_VISIBLE_DEVICES at one, and the target moves to an arch those wheels carry no
        # kernels for. torch.version.hip still reads ROCm, so rocm_torch_ready would stop the
        # generic fallback and the wrong family would survive every update. Same repick
        # _detect_windows_gfx_arch's caller does for #7776, on a family read back positively.
        if _arch_index_url is None and rocm_torch_ready and _runtime_gfx is not None:
            _have = _installed_rocm_wheel_family() if _torch_requires_rocm_sdk() else None
            _want = (_GFX_TO_AMD_INDEX_ARCH.get(_runtime_gfx) or "").lower()
            # A target no index can serve (gfx1010 / RDNA 1) has an empty _want, which reads
            # as "every family is wrong" and spends a multi-GB reinstall on a wheel that
            # cannot carry kernels for it either. Demote only when there is somewhere to go,
            # and ask that of THIS host: gfx906's only route is the rocm6.3 tag, unlocked
            # solely when it is the one detected arch, so demoting a mask-selected MI50 beside
            # another card downloads a second unusable torch (_MIXED_HOST_UNROUTABLE).
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
                # Demoting alone hands the job to the generic fallback, which resolves NO tag
                # when the ROCm version is unreadable, as it is on a bundled-runtime host: the
                # branch would announce the reinstall, install nothing, and keep the family it
                # just called incompatible. The AMD index needs no host version, so use it.
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
                    # The replacement GPU has no AMD per-arch index at all: gfx942, gfx950 and
                    # the other datacentre parts live only on the generic one. With no
                    # readable host version there is no tag either, so the branch would leave
                    # the stale per-arch wheels in place. A tag that RESOLVES can be as
                    # useless, a stale /opt/rocm putting gfx950 on rocm6.3. Both take the
                    # newest generic index this installer knows, which does carry the target.
                    ver = max(_ROCM_TORCH_INDEX)
            elif _have is None and _generic_only_target_below_floor(
                _runtime_gfx, _installed_generic_rocm_tag() or ver
            ):
                # A GENERIC build whose own tag predates the target: no family to compare and
                # no per-arch index to move to, so both arms above decline and
                # torch.version.hip keeps the fallback from running. A gfx950 pinned to
                # rocm6.3 once stays on a wheel with no kernels for it through every update.
                # Demoting is the whole repair (the floor below picks the tag), and the
                # preflight forces the pass for this state, so declining never repairs it.
                _safe_print(
                    f"   installed ROCm torch is a generic build whose own tag predates\n"
                    f"   {_runtime_gfx} -- reinstalling from an index that carries it.\n"
                )
                rocm_torch_ready = False

        # The floor above is reached only by a host already on ROCm wheels that was demoted.
        # A fresh install, or a CPU/CUDA build, walks past it to the generic fallback, where a
        # stale /opt/rocm puts gfx950 on rocm6.3 the same way with no per-arch index to fall
        # back to. Which tag carries an arch is a fact about the arch, not about the install.
        if (
            _arch_index_url is None
            and not rocm_torch_ready
            and _runtime_gfx in _GENERIC_ROCM_WHEEL_GFX
            and _generic_tag_lacks_kernels(_runtime_gfx, ver)
        ):
            ver = max(_ROCM_TORCH_INDEX)

    # gfx906 (MI50 / Radeon VII): is this the runtime GPU target? Used below to skip
    # the generic bitsandbytes wheel (no gfx906 kernels). This must hold even under
    # an explicit torch-index pin: a gfx906 host that pins rocm6.3 (without also
    # setting UNSLOTH_ROCM_GFX_ARCH) would otherwise reinstall the prebuilt bnb wheel
    # over the user's source-built gfx906 bnb. So a pin suppresses only the torch
    # reroute (_gfx906_override below), NOT the gfx906 detection for the bnb skip.
    # _runtime_target_is_gfx906 reads UNSLOTH_ROCM_GFX_ARCH itself with the same
    # normalization, so asking it alone loses nothing, and ORing the override back in would
    # walk the no-GPU mask guard it applies above that read.
    _runtime_is_gfx906 = _runtime_target_is_gfx906()
    # Reroute torch to the last gfx906-capable wheel family (rocm6.3) only when the
    # host ROCm version would otherwise pick a newer, kernel-less index -- and never
    # over an explicit pin or an active Strix reroute (the pin/Strix path installs
    # its own index; only the bnb skip must still apply on those paths).
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

    # Either reroute must fire even when has_hip_torch is True: an existing
    # torch.version.hip == "7.1" is exactly the broken combo they repair. The
    # missing-kernel route declines above when torch already runs on its leaf.
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
            # The caller inspects the failure itself; it is responsible for
            # reporting and exiting if it cannot recover.
            return result
        _report_failed_command(label, result)
    return result


def _report_failed_command(label: str, result: subprocess.CompletedProcess[bytes]) -> None:
    """Print a failed command's redacted output and exit with its code."""
    _step("error", f"{label} failed (exit code {result.returncode})", _red)
    if result.stdout:
        # Redact before printing: the failing pip command may carry a pinned --index-url
        # with userinfo/?token= creds, so raw pip error text would leak them.
        _safe_print(_redact_install_output(result.stdout))
    sys.exit(result.returncode)


# pip will not replace a distribution whose .dist-info carries no RECORD -- it cannot
# know which files to remove -- and an interrupted install (or one uv wrote and pip is
# now asked to take over) leaves exactly that. The venv is REUSED across runs, so the
# stub is not transient: once written, every later install of that package dies on it
# with "The package's contents are unknown", and the dependency step never completes
# again on that machine. Seen on the Windows startup job, where the whole install is a
# uv pass that falls back to pip.
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
    # Both halves are required: the marker alone can appear in unrelated advice, and
    # a "Cannot uninstall" without it is a different problem that deleting won't fix.
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

# Requirements with NO wheel on PyPI at any version, so the installer has always built
# them from source. antlr4-python3-runtime arrives transitively: omegaconf==2.3.1 pins it
# below the 4.13.2 wheel.
#
# A user-level `no-build = true` (uv.toml) or `only-binary = :all:` (pip.conf) makes all
# of them unresolvable and fails the whole extras step (#8530). A PACKAGE-SCOPED
# --no-binary overrides that policy for these names only, so the user's binary-only
# policy still applies everywhere it can be honoured -- a blanket --no-build or
# `:none:` override would discard it entirely.
#
# Keep in sync with the CI `nobuild` allowlists asserting the same contract:
# .github/scripts/clean-machine-assert.sh and .github/scripts/assert-nobuild.ps1.
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
    # extras.txt pins MeCab==0.996.5 on macOS cp314+, the last release carrying an sdist.
    # Conditional because MeCab is a C extension: everywhere else 0.996.13 resolves to a
    # wheel, and exempting it there would force a compiler-dependent build.
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

    Targets the interpreter install_wheel installed into, always ``sys.executable``: its uv
    command passes --python as well as --system, and its pip fallback runs that interpreter.
    --system ALONE would remove from the system Python, leaving the rejected wheel in the
    venv while setup reported it gone.
    """
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
                    # Say so plainly: it is still importable in process, so this is not the
                    # same state as never having installed it.
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
    """Return a temp copy, adjacent when writable, with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line for line in lines if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    # Beside the source so relative -r/-c includes resolve; a read-only tree
    # (root-owned install, non-root user) falls back rather than aborting.
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
        step_label = "overlaying unsloth-zoo from git main"
        install_label = "Overlaying unsloth-zoo from git main"
        args = ("--force-reinstall", "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo")
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
        return "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
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
    # invalid_metadata_paths() hands back Path objects, so normalise before the
    # string work below.
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
        # pip, not uv: the wheel is already built and sitting in staged, and uv would
        # reject the unpinned name under UV_REQUIRE_HASHES with the package records
        # already gone. Routing through pip also earns the PIP_REQUIRE_HASHES relaxation.
        force_pip = True,
    ):
        # The wheel just wrote its own valid metadata at the same path the rewritten
        # record occupied, so the unwinding below must not put the original back over
        # it. See _QuarantinedMetadata.forget_copies.
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
        # The checkout needs no network, but pip's isolated build fetches the build
        # backend, which UV_OFFLINE does not reach: measured, an isolated build with no
        # index fails at "installing build dependencies", and this repo pins those
        # exactly. Build against the interpreter's own backend and forbid the index, so
        # no-network means no network. An unimportable backend fails the build and
        # leaves the installation intact.
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
        # A sole `~` backup counts as one readable version, so metadata_conflict()
        # cannot see it, yet pip and importlib both skip the directory and the real
        # tree is usually renamed away with it. Repair it like any other duplicate.
        if install_manifest.metadata_conflict(
            versions
        ) or install_manifest.pip_backup_metadata_paths(name):
            duplicates.append((name, record_count))

    repaired: list[str] = []
    staging_dirs: list[str] = []
    # One quarantine per package, discarded as soon as that package is back in place.
    # Sharing one would, on a later package's failure, restore the first package's stale
    # record over the install that replaced it: the conflict returns, and its old RECORD
    # then describes a payload that is gone.
    quarantine = _QuarantinedMetadata()
    succeeded = False
    try:
        for name, record_count in duplicates:
            quarantine = _QuarantinedMetadata()
            _step(_LABEL, f"duplicate metadata for {name} detected; reinstalling it", _dim)
            invalid_paths = install_manifest.invalid_metadata_paths(name)
            # Give pip a parseable METADATA beside every intact RECORD so it can
            # uninstall them normally, removing exactly the files they list.
            # Quarantining one instead drops its RECORD, so the uninstall loop removes
            # only what the readable records claim and a module existing solely in the
            # older release stays on disk and importable while the repair reports success.
            unrewritable = [
                path
                for path in invalid_paths
                if not (
                    quarantine.back_up(os.path.join(path, "METADATA"))
                    and _rewrite_minimal_metadata(path, name)
                )
            ]
            # Every record has to end up uninstallable by pip, or nothing is touched.
            # Quarantining one instead only hides it: pip removes just the readable
            # records, the quarantine is discarded once the reinstall succeeds, and
            # whatever the quarantined release owned alone stays on disk and
            # importable while the repair reports success and deletes the directory
            # that was the evidence. Waiting for record_count to reach zero missed
            # every case where another record survives. Refusing leaves the tree as
            # found, so a later run can still see the conflict.
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
            # pip skips its own abandoned backup ("Ignoring invalid distribution
            # ~nsloth") while its METADATA still names the project, so the record
            # counts here but no uninstall by name can consume it. Left in place the
            # loop below cannot converge and the repair fails on every future run,
            # which is the loop this whole path exists to end. Quarantined, not
            # deleted: staging can still fail.
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
            # A backup names a payload pip has already renamed away, so once it is
            # aside there is nothing for a replacement to be laid over: a fresh
            # install is exactly right, and refusing here would abort the installer
            # on the one state it can trivially fix.
            if invalid_paths and not record_count:
                # Nothing is left for pip to uninstall, so the replacement would be
                # laid over a payload no record describes.
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
            # A local or git source installs from a path or URL, so there is
            # nothing to stage; anything else comes off an index, which has to
            # be proven reachable while the current install is still intact.
            overlaid = bool(source_repo) and _is_overlayable_core_package(name)
            # Stage whichever source will be installed, overlay included: an
            # overlay is a git fetch or a build, either of which can fail after
            # the uninstall loop has already removed every record.
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

            # Installer handoffs may already have applied a local or CI source.
            # Restore that provenance now that no ambiguous record remains.
            restored = overlaid and _overlay_local_core_package(name, source_repo, strict = False)
            if not restored:
                # The overlay install is preferred because it keeps the editable
                # or git provenance, but the staged wheel was built from that
                # same source, so falling back to it never substitutes a release.
                restored = pip_install_try(
                    f"Repairing duplicate metadata for {name}",
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    "--no-index",
                    "--find-links",
                    staged,
                    name,
                    # As _restore_from_staged: pip, so a uv hash policy cannot reject the
                    # already-built wheel once every record has been removed.
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
        # Anything short of a completed repair puts the quarantined records back,
        # so a failure leaves the environment as it was found.
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
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            upgrade.append(arg)
            continue
        if arg == "--upgrade-package":
            skip_next = True  # the flag; its value is the package to upgrade
            continue
        cmd.append(arg)
    if upgrade:
        cmd += ["--upgrade", "--upgrade-strategy", "only-if-needed"]
        # Every current caller also names these as positionals or via -r, but a
        # future one might not, and pip would then upgrade nothing.
        cmd += [name for name in upgrade if name not in cmd]
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


# Restrictive policy a pinned install must not inherit from the ENVIRONMENT. The pinned
# branch neutralises the config FILES (UV_NO_CONFIG=1 + PIP_CONFIG_FILE=devnull), but an
# env var outranks a config file, so a hardened shell could still fail a torch repair the
# pin was supposed to make deterministic (#8530).
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
    """Overrides that stop a hardened user pip config failing the installer's own pip.

    Empty for anything that is not a `pip install` / `pip download` / `pip wheel` this
    module drives, every `uv` command included, so the "non-pinned installs inherit the
    caller env unchanged" contract holds on a machine with no hostile pip config.
    `wheel` is in that set because the duplicate-metadata repair stages its replacement
    with `pip wheel`, where require-hashes applies exactly as it does to install: an
    unpinned name is rejected before anything is built, so the repair would abort on a
    hardened machine and leave the conflict it exists to remove.

    `require-hashes = true` makes pip reject any requirement without a --hash, which is
    every requirements file we ship; that is what took the pip FALLBACK down in #8530
    once uv had failed. pip applies env vars AFTER config files, so PIP_REQUIRE_HASHES=0
    overrides it while pip.conf's index-url, trusted-host, cert and proxy stay in force.
    """
    if cmd[:1] == ["uv"] or not any(arg in ("install", "download", "wheel") for arg in cmd):
        return {}
    return {"PIP_REQUIRE_HASHES": "0"}


def _uv_is_offline() -> bool:
    """True when uv has been told not to touch the network."""
    return os.environ.get("UV_OFFLINE", "").strip().lower() not in ("", "0", "false")


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
            # uv's artifact policy, which pip reads none of. Without it the repair can
            # download a wheel under a no-binary rule or build an sdist under an
            # only-binary one, changing the artifact type mid-repair.
            option, _, value = line.partition(" ")
            build_options.extend((option, value.strip()))
        elif line.startswith("# from "):
            # Names the index this package came from, the one to reproduce, but is not
            # usable as written: measured on uv 0.10.7 the annotation drops userinfo
            # while the emitted index lines keep it, so it is matched back to the
            # emitted URL. A private index would answer 401 otherwise.
            origin = line[len("# from ") :].strip() or origin
        elif line and not line.startswith(("#", "-")):
            # uv continues a hashed pin onto the following lines with a backslash.
            pinned = line.split(";", 1)[0].rstrip("\\").strip()
            if _canonical_package_name(_requirement_name(pinned)) == canonical:
                requirement = pinned
    if not requirement:
        return None
    # Replaying uv's answer means replacing pip's candidate sources, not adding to them.
    # An inherited PIP_NO_INDEX would block the index uv picked, and an inherited
    # extra index or find-links directory could satisfy the same version from a source
    # uv never looked at, which is the provenance swap this whole path exists to stop.
    # Empty rather than deleted: measured on pip 26.2, an empty value reads as unset.
    overrides = {
        "PIP_EXTRA_INDEX_URL": "",
        "PIP_NO_INDEX": "",
        "PIP_FIND_LINKS": " ".join(find_links),
    }
    index_url = _credentialed_index(origin, emitted)
    if index_url:
        overrides["PIP_INDEX_URL"] = index_url
    elif find_links:
        # uv resolved this from a flat source with no index in play, which is what a
        # configured no-index looks like on the way out: it emits the find-links entry
        # and no index line at all. Leaving PIP_NO_INDEX cleared would hand pip back
        # the default PyPI and let it stage the same name and version from a source uv
        # was told to exclude.
        overrides["PIP_NO_INDEX"] = "1"
    # Measured on uv 0.10.7: --emit-build-options surfaces the policy from uv.toml but
    # NOT the environment-variable spelling of it, so that half is translated by hand.
    # Only where pip has no setting of its own, which it reads natively.
    # UV_KEYRING_PROVIDER is the same translation: uv reaches an authenticated index
    # through the keyring CLI, and carrying only the URL leaves pip unable to fetch
    # what uv just resolved. uv's two values (disabled, subprocess) are both valid pip
    # values. Only the environment spelling is reachable here; a uv.toml
    # keyring-provider is not emitted, so that half cannot be replayed.
    for uv_name, pip_name in (
        ("UV_NO_BINARY", "PIP_NO_BINARY"),
        ("UV_ONLY_BINARY", "PIP_ONLY_BINARY"),
        ("UV_KEYRING_PROVIDER", "PIP_KEYRING_PROVIDER"),
    ):
        value = os.environ.get(uv_name, "").strip()
        if value and not os.environ.get(pip_name):
            overrides[pip_name] = value
    if hashes:
        # The hashes are what make this safe rather than merely careful. Measured: pip
        # verifies them even with PIP_REQUIRE_HASHES=0, and neither PIP_CONFIG_FILE nor
        # --isolated suppresses a site pip.conf, so pip may still consult a source uv
        # never considered. It can no longer accept a different artifact from one.
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
        # One index can be emitted both with and without credentials; the whole point
        # here is the credentialed form, so it wins over a bare match on the same URL.
        for url in matches:
            if _strip_userinfo(url) != url:
                return url
        if matches:
            return matches[0]
    # The origin is not one of the emitted indexes, so it is a find-links source --
    # uv annotates those with a file:// or directory URL. That belongs in
    # PIP_FIND_LINKS, which is already set from the emitted find-links lines, and
    # must not displace the real index: an sdist picked out of a flat directory still
    # needs the index for its build backend.
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
    the user's mirror. For pinned commands, the uv index/backend vars are removed,
    UV_NO_CONFIG=1 set (a discovered uv.toml outranks the CLI pin), and PIP_CONFIG_FILE
    pointed at os.devnull for the pip fallback. Mirrors install.sh's gate (#6898).

    A non-pinned `pip` command also gets hash-required mode switched off, the one
    relaxation with no command-line equivalent; the wheel-less requirements go through
    the package-scoped --no-binary in _sdist_only_build_args() instead.
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
    """Like pip_install but returns False on failure instead of exiting.
    For optional installs that have a follow-up fallback.
    """
    # Same reason as pip_install: this installs torch too (the Windows AMD ROCm trio),
    # so the memoized classification must not survive it.
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
    # Any pip operation can change which torch is installed, so the memoized
    # classification must not outlive it.
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
        pip_label = f"{label} (pip)" if USE_UV else label
        result = run(pip_label, pip_cmd, check = False)
        if result.returncode != 0:
            # Retry once, and only after clearing something pip named as
            # unremovable: a blind retry of a failing install just doubles the wait.
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


_MLX_HEALTH_PROBE = (
    "import json, sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "from utils.mlx_repair import mlx_stack_blockers;"
    "print(json.dumps(mlx_stack_blockers()))"
)


def _report_mlx_stack_health() -> None:
    """Name what would keep Train off on this Apple Silicon host, if anything.

    Advisory only: the install has already succeeded, chat still works, and the
    background self-heal gets another go at startup. It just must not be silent,
    which is the whole of the reported "Train is blacked out after an update".

    Run out of process: the probe imports mlx, mlx_lm and mlx_vlm, and a half
    installed one of those can abort rather than raise.
    """
    backend = str(SCRIPT_DIR / "backend")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _MLX_HEALTH_PROBE, backend],
            capture_output = True,
            text = True,
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
    # An aborted earlier run leaves it set, and every _safe_print() consumes it --
    # the first message would get a stray newline.
    _PROGRESS_LINE_ACTIVE = False

    # install.sh sets SKIP_STUDIO_BASE=1 to avoid reinstalling the core packages;
    # `studio update` does NOT, so unsloth + unsloth-zoo are reinstalled to pick
    # up new versions. Shared base.txt requirements are handled independently.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # --package installs a different package name (for testing).
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # --local overlays a local repo checkout after updating deps.
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    # Clean-machine CI overlays only unsloth, not the full local source pair.
    ci_source_overlay = os.environ.get("UNSLOTH_CI_SOURCE_OVERLAY", "")
    # +1 for the anyio repair check (step 8b), +1 for the diffusers pin (step 11b, every platform)
    base_total = 12 if IS_WINDOWS else 13
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    if not IS_MACOS and not NO_TORCH:
        base_total += 1  # ROCm torch check (step 2b), non-macOS
        if not IS_WINDOWS:
            base_total += 2  # flash-attn + torch final repair (step 13), Linux
    if IS_MAC_ARM and not skip_base:
        base_total += 1  # MLX stack, same gate as the step itself
    base_requirements = _shared_base_requirements() if skip_base else None
    # Core packages and shared base requirements occupy one progress slot. A
    # shell-installer handoff skips that slot only while base.txt has no work.
    _TOTAL = base_total - int(skip_base and base_requirements is None)

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

    # A superseded dist-info makes version() and every RECORD consumer choose
    # an arbitrary package version. Repair it before any fast package operation,
    # including installer handoffs that set skip_base after their own upgrade.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

    # macOS arm64: install MLX stack at latest (UV_OVERRIDE relaxes the
    # mlx-vlm / mlx-lm transformers pin -- set at module load).
    if IS_MAC_ARM and not skip_base:
        _progress("MLX stack (Apple Silicon)")
        pip_install(
            "Installing MLX stack (mlx + mlx-lm + mlx-vlm)",
            "--no-cache-dir",
            "--upgrade",
            "mlx",
            "mlx-metal",
            "mlx-lm",
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
        # install.sh / install.ps1 already installed both core distributions.
        pass
    elif NO_TORCH:
        # No-torch update path: install unsloth + unsloth-zoo, then runtime deps,
        # both with --no-deps (PyPI metadata declares torch a hard dep; avoid it).
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
            _overlay_local_core_packages(local_repo)
    elif local_repo:
        # Local dev install: update the released core packages, then overlay the
        # checkout as an editable install (--no-deps so torch is not re-resolved).
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

    # Independent of the core phase: the shell installers skip that after
    # installing the two distributions inline, but still apply this file.
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
        # extras.txt is where the wheel-less requirements live, so a user-level
        # no-build/only-binary policy fails this step first (#8530).
        *_sdist_only_build_args(*_extras_sdist_only_packages()),
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

    # 4. Install the torch-matched torchao override. Reinstall only when the pin
    #    changes, since Windows can remove shared files during replacement.
    #    Skip when torch is unavailable or Windows ROCm has no working build.
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
        _torchao_args = ["--no-cache-dir"]
        if not _exact_distribution_spec_is_installed(_torchao_spec):
            _torchao_args.insert(0, "--force-reinstall")
        pip_install(
            "Installing dependency overrides",
            *_torchao_args,
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

    # 11b. The pinned Diffusers release. NOT in base.txt, which is applied early: this must
    #      run after every other requirements file so nothing re-resolves Diffusers back to a
    #      release, and outside every skip_base / NO_TORCH branch so it reaches every path.
    #      constrain stays on: constraints.txt says nothing about diffusers today, and a
    #      future entry there should win rather than be silently bypassed here.
    _progress("diffusers pin")
    pip_install(
        "Installing the pinned Diffusers release",
        "--no-cache-dir",
        req = REQ_ROOT / "diffusers-pin.txt",
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

    # 14b. Repair again before the manifest is written. The pass above runs
    # before the core packages are installed, so an upgrade that itself leaves a
    # superseded record behind -- the exact state this repair exists for -- would
    # otherwise survive it. write_manifest would then record a null version and
    # the installer would report success while every later check rejects the
    # environment. A no-op when nothing is ambiguous.
    if not _repair_duplicate_core_metadata(
        (package_name, "unsloth-zoo"),
        local_repo = local_repo,
        ci_source_overlay = ci_source_overlay,
    ):
        return 1

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

    # 16. Apple Silicon: say so when the MLX stack this install just laid down is not
    # one Train can use. The gate is all-or-nothing across mlx, mlx-lm and mlx-vlm, and
    # a resolver backtrack (or an mlx-vlm built against a different transformers) leaves
    # packages present but unusable. Without this the install reports success and the app
    # silently comes up chat-only, telling the user to run the update that has just
    # finished.
    #
    # AFTER the manifest, not before it. The probe is advisory and out of process, and on
    # the host it exists for the imports are the ones that hang, so it can hold its full
    # timeout; run ahead of the manifest, a kill during that wait leaves every dependency
    # step done and no record of it, and verify-install, the desktop preflight and the
    # setup fast path all then call a complete install incomplete.
    if IS_MAC_ARM and not NO_TORCH:
        _report_mlx_stack_health()

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--amd-torch-needs-dependency-pass"]:
        # Exit 0 forces the dependency pass; exit 1 keeps the fast path.
        _needs_pass = _amd_torch_needs_dependency_pass()
        # Exit 1 covers five states (no-torch venv, resolved non-ROCm backend, non-ROCm pin,
        # absent or masked AMD host, unreadable torch), so a CI failure here would otherwise
        # report a bare `assert 1 == 0` with both streams empty. setup.sh discards both
        # streams, so this costs the installer nothing. _TORCH_RUNTIME_PROBE is read, not
        # called, so no subprocess is added: None means an earlier gate answered first.
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
