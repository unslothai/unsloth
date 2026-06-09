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
import tempfile
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

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_MAC_INTEL = IS_MACOS and platform.machine() == "x86_64"
IS_MAC_ARM = IS_MACOS and platform.machine() == "arm64"
IS_LINUX = sys.platform.startswith("linux")
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

# Per-tag pip specs; rocm7.2 ships torch 2.11.0 (older tags cap at 2.10.x).
_ROCM_TORCH_PKG_SPECS: dict[str, tuple[str, str, str]] = {
    "rocm7.2": (
        "torch>=2.11.0,<2.12.0",
        "torchvision>=0.26.0,<0.27.0",
        "torchaudio>=2.11.0,<2.12.0",
    ),
    # Default for rocm7.1 and earlier: torch 2.x below 2.11
    "_default": (
        "torch>=2.4,<2.11.0",
        "torchvision>=0.19,<0.26.0",
        "torchaudio>=2.4,<2.11.0",
    ),
}
_PYTORCH_WHL_BASE = (
    os.environ.get("UNSLOTH_PYTORCH_MIRROR") or "https://download.pytorch.org/whl"
).rstrip("/")

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
            # Explicit length guard so we don't rely on the broad except
            # below to swallow IndexError when the version file has a
            # single component (e.g. "6\n" on a partial install).
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # Try amd-smi version (outputs "... | ROCm version: X.Y.Z")
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                import re
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

    # Distro package-manager fallbacks. Package-managed ROCm installs can
    # expose GPUs via rocminfo/amd-smi but lack /opt/rocm/.info/version and
    # hipconfig, so probe dpkg (Debian/Ubuntu) and rpm (RHEL/Fedora/SUSE)
    # for the rocm-core version. Matches install.sh::get_torch_index_url so
    # `unsloth studio update` behaves like a fresh `curl | sh` install.
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
    if hipinfo:
        try:
            result = subprocess.run(
                [hipinfo],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                timeout = 10,
            )
            if result.returncode == 0:
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
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        for _args in (("static", "--asic"), ("list",)):
            try:
                result = subprocess.run(
                    [amd_smi, *_args],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    timeout = 10,
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
    return None


def _windows_rocm_index
