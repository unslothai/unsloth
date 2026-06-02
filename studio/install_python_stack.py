#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by both setup.sh (Linux / WSL) and setup.ps1 (Windows) after the
virtual environment is already activated.  Expects `pip` and `python` on
PATH to point at the venv.
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

from backend.utils.native_path_leases import child_env_without_native_path_secret
from backend.utils.wheel_utils import (
    FLASH_ATTN_SPEC,
    flash_attn_package_version,
    flash_attn_wheel_url,
    has_blackwell_gpu,
    has_nvidia_gpu,
    install_optional_kernel,
)

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_MAC_INTEL = IS_MACOS and platform.machine() == "x86_64"
IS_MAC_ARM = IS_MACOS and platform.machine() == "arm64"
IS_LINUX = sys.platform.startswith("linux")
# torchcodec ships wheels only for manylinux_2_28_x86_64,
# macosx_12_0_arm64, and win_amd64 (visible in the 0.10.0 PyPI page).
# Trying to install it on any other host fails the whole
# extras-no-deps step. `unsloth studio update` does not have a
# --no-torch flag, so on these hosts the audio extras must be
# filtered out independent of the NO_TORCH env var.
PLATFORM_LACKS_TORCHCODEC_WHEEL = (
    (IS_LINUX and platform.machine() in {"aarch64", "arm64"})
    or (IS_WINDOWS and platform.machine().lower() in {"arm64", "aarch64"})
    or IS_MAC_INTEL
)

# ── ROCm / AMD GPU support ─────────────────────────────────────────────────────
# Mapping from detected ROCm (major, minor) to the best PyTorch wheel tag on
# download.pytorch.org.  Entries are checked newest-first (>=).
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

# AMD Windows ROCm wheels — repo.amd.com (arch-specific pip index)
# Format: https://repo.amd.com/rocm/whl/{arch_family}/
# Override with UNSLOTH_ROCM_WINDOWS_MIRROR for air-gapped / mirror installs.
_ROCM_WINDOWS_INDEX_BASE = (
    os.environ.get("UNSLOTH_ROCM_WINDOWS_MIRROR") or "https://repo.amd.com/rocm/whl"
).rstrip("/")

# Maps gfx arch → AMD index arch-family suffix.
# Each family is a separate pip index on repo.amd.com.
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
    # Windows ROCm wheel — ships libbitsandbytes_rocm{VER}.dll.
    # BNB auto-detects HIP version from torch.version.hip, which does not always
    # match the DLL suffix in this prerelease wheel (e.g. torch 7.13 with a rocm72
    # DLL).  We scan the installed wheel for the actual DLL name and set
    # BNB_ROCM_VERSION accordingly in _install_bnb_windows_rocm() and worker.py.
    "win_amd64": (
        "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/"
        "download/continuous-release_main/"
        "bitsandbytes-1.33.7.preview-py3-none-win_amd64.whl"
    ),
}
_BNB_ROCM_PYPI_FALLBACK = "bitsandbytes>=0.49.1"


def _bnb_rocm_prerelease_url() -> str | None:
    """Return the continuous-release_main bnb wheel URL for the current
    architecture, or None when no pre-release wheel is available.
    """
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return _BNB_ROCM_PRERELEASE_URLS.get(arch)


def _detect_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm stack, or None."""
    # Check /opt/rocm/.info/version or ROCM_PATH equivalent
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            # Explicit length guard avoids relying on the broad except
            # below to swallow IndexError when the version file contains
            # a single component (e.g. "6\n" on a partial install).
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
                if (
                    len(parts) >= 2
                    and parts[0].isdigit()
                    and parts[1].split("-")[0].isdigit()
                ):
                    return int(parts[0]), int(parts[1].split("-")[0])
        except Exception:
            pass

    # Distro package-manager fallbacks. Package-managed ROCm installs can
    # expose GPUs via rocminfo / amd-smi but still lack /opt/rocm/.info/version
    # and hipconfig, so probe dpkg (Debian/Ubuntu) and rpm (RHEL/Fedora/SUSE)
    # for the rocm-core package version. Matches the chain in
    # install.sh::get_torch_index_url so `unsloth studio update` behaves
    # the same as a fresh `curl | sh` install.
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
    """Resolve HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES to an integer
    index into a list of length num_tokens. Returns 0 (first GPU) for
    unset, empty, '-1', UUID-style, or out-of-range values."""
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

    Probe order matches the PowerShell installer: env-var override first,
    then hipinfo (PATH or HIP_PATH / ROCM_PATH bin), then amd-smi. Without
    the amd-smi fallback, runtime-only AMD installs without hipinfo on PATH
    return early and `studio update` cannot repair a CPU-only venv.

    On multi-GPU hosts, all detected gfx tokens are deduplicated (preserving
    enumeration order) and HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES selects
    which one to install for. The first GPU is used when no env var is set.
    """
    # 1. Explicit override (matches PowerShell installer's env-var path).
    _override = os.environ.get("UNSLOTH_ROCM_GFX_ARCH")
    if _override and _override.strip():
        return _override.strip().lower()

    def _dedup_pick(tokens: list[str]) -> "str | None":
        if not tokens:
            return None
        # Index into the full (ordered) list first so HIP_VISIBLE_DEVICES
        # correctly addresses GPU N on mixed-arch hosts, then return that arch.
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
                # findall picks every gcnArchName line so multi-GPU hosts
                # are enumerable and HIP_VISIBLE_DEVICES selects correctly.
                _tokens = [
                    t.strip().lower()
                    for t in re.findall(r"(?im)^\s*gcnArchName\s*:\s*(\S+)", text)
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


def _windows_rocm_index_url(gfx_arch: str | None) -> str | None:
    """Return the AMD pip index URL for the given GPU arch, or None if unsupported."""
    arch_family = _GFX_TO_AMD_INDEX_ARCH.get(gfx_arch or "")
    if arch_family is None:
        return None
    return f"{_ROCM_WINDOWS_INDEX_BASE}/{arch_family}/"


def _detect_bnb_rocm_dll_ver() -> str | None:
    """Scan the installed bitsandbytes package for libbitsandbytes_rocm{VER}.dll.

    Returns the version suffix string (e.g. ``"72"``, ``"713"``) or ``None``
    if bitsandbytes is not installed or no ROCm DLL is found.  Does NOT import
    bitsandbytes — uses importlib.util.find_spec so it is safe to call before
    BNB is imported.
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
    # Pick the highest numeric suffix so that e.g. "713" wins over "72" when
    # both variants are present in the wheel.  Filesystem glob order is not
    # guaranteed, so always sort rather than stopping at the first match.
    return max(all_vers, key = lambda v: int(v)) if all_vers else None


def _has_rocm_gpu() -> bool:
    """Return True only if an actual AMD GPU is visible (not just ROCm tools installed)."""
    for cmd, check_fn in (
        # rocminfo: look for a real gfx GPU id (3-4 chars, nonzero first digit).
        # gfx000 is the CPU agent; ROCm 6.1+ also emits generic ISA lines like
        # "gfx11-generic" or "gfx9-4-generic" which only have 1-2 digits before
        # the dash and must not be treated as a real GPU.
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
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            if check_fn(result.stdout):
                return True
    # sysfs KFD topology fallback (Linux only) -- matches install.sh's
    # runtime-only detection. On minimal package-managed installs (no
    # rocminfo / no amd-smi GUI tools), the kernel exposes AMD GPUs via
    # /sys/class/kfd so `studio update` can still detect the GPU and
    # repair the venv.
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
                    if gpu_id and gpu_id != "0":  # gpu_id 0 = CPU node
                        return True
        except OSError:
            pass
    return False


def _has_usable_nvidia_gpu() -> bool:
    """Return True only when nvidia-smi exists AND reports at least one GPU.

    Thin wrapper around ``wheel_utils.has_nvidia_gpu`` so tests can keep
    mocking ``_has_usable_nvidia_gpu`` at the ``ips`` module level.
    """
    return has_nvidia_gpu()


def _detect_amd_gfx_codes() -> list[str]:
    """Return the list of AMD gfx ISA strings visible to ROCm (e.g. ['gfx1151']).

    Probes rocminfo first, then falls back to ``amd-smi list`` and
    ``amd-smi static --asic`` for runtime-only Radeon hosts that ship
    amd-smi but no rocminfo. Returns an empty list when no probe yields
    a gfx target.
    """

    def _extract(text: str) -> list[str]:
        codes = re.findall(r"gfx([1-9][0-9a-z]{2,3})", text.lower())
        return list(dict.fromkeys(f"gfx{c}" for c in codes))

    probes: list[list[str]] = []
    if shutil.which("rocminfo"):
        probes.append(["rocminfo"])
    if shutil.which("amd-smi"):
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
    encodes version 1.33.7.preview (parsed as 1.33.7rc0 by PEP 440) while the
    wheel metadata reports 0.50.0.dev0.  uv rejects this filename/metadata
    mismatch -- and bypassing it with UV_SKIP_WHEEL_FILENAME_CHECK still leaves
    uv mangling the bitsandbytes install. Per the AMD install guide
    (https://unsloth.ai/docs/get-started/install/amd/amd-hackathon) the wheel
    must be installed with plain pip, not uv, so we force pip here
    (force_pip=True). plain pip performs no wheel filename/metadata check.
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
    # After install: detect the actual ROCm DLL suffix shipped in the wheel and
    # set BNB_ROCM_VERSION so bitsandbytes loads the correct DLL regardless of
    # what torch.version.hip reports.  The wheel may ship an older suffix (e.g.
    # "72") while torch reports a newer HIP version (e.g. 7.13); the env var
    # override ensures bitsandbytes does not fail looking for a non-existent DLL.
    # The worker subprocess inherits this env var automatically.
    # Fall back to "72" if detection fails (e.g. install was a no-op / dry-run).
    if "BNB_ROCM_VERSION" not in os.environ:
        _ver = _detect_bnb_rocm_dll_ver() or "72"
        os.environ["BNB_ROCM_VERSION"] = _ver
    return True


def _ensure_rocm_torch() -> None:
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch.

    On Linux x86_64: uses pytorch.org ROCm wheel index tags.
    On Windows: uses AMD's repo.amd.com arch-specific pip index.
    No-op on macOS, non-x86_64 Linux, NVIDIA-primary hosts, or when torch
    already links against HIP.
    Uses pip_install() to respect uv, constraints, and --python targeting.
    """
    global _rocm_windows_torch_installed
    # setup.ps1 sets this when it already installed AMD wheels; skip the probe
    # only when torch is actually importable as ROCm. If the venv was wiped
    # between runs, the stale env-var would suppress a needed reinstall.
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
            # setup.ps1 already installed ROCm torch, but we still need to install
            # the AMD Windows BNB wheel here -- the PyPI bitsandbytes wheel ships
            # only CUDA DLLs and will fail to load on ROCm.
            _install_bnb_windows_rocm()
            return
        # torch was wiped between runs; fall through to the full install path
    if IS_MACOS:
        return

    if IS_WINDOWS:
        if _has_usable_nvidia_gpu():
            return
        gfx_arch = _detect_windows_gfx_arch()
        if not gfx_arch:
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
            index_url = _windows_rocm_index_url(gfx_arch)
            if index_url is None:
                print(
                    f"   No AMD Windows torch index for GPU arch {gfx_arch} -- skipping"
                )
                return
            print(f"   {gfx_arch} (Windows) -- installing torch from {index_url}")
            pip_install(
                f"ROCm torch (Windows, {gfx_arch})",
                "--force-reinstall",
                "--index-url",
                index_url,
                "torch",
                "torchvision",
                "torchaudio",
                constrain = False,
            )
        # ROCm torch is installed (or already was); flag it so later install
        # phases do not overwrite it with the generic CPU torch wheel. BNB is
        # a separate dependency -- a BNB install failure must NOT roll the
        # torch ROCm install back.
        _rocm_windows_torch_installed = True
        # Always install AMD Windows bitsandbytes -- the PyPI wheel ships only
        # CUDA DLLs and will fail to load on ROCm.  Install even when torch was
        # already a ROCm build so that `studio update` repairs a broken bnb.
        if not _install_bnb_windows_rocm():
            print(
                "   Warning: AMD Windows bitsandbytes install failed; "
                "ROCm torch is installed but bitsandbytes may need manual install"
            )
        return

    # ── Linux x86_64 only: PyTorch ROCm wheels are not published for aarch64 ──
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return
    # NVIDIA takes precedence on mixed hosts -- but only if an actual GPU is usable
    if _has_usable_nvidia_gpu():
        return
    # Rely on _has_rocm_gpu() (rocminfo / amd-smi GPU data rows) as the
    # authoritative "is this actually an AMD ROCm host?" signal. The old
    # gate required /opt/rocm or hipcc to exist, which breaks on
    # runtime-only ROCm installs (package-managed minimal installs,
    # Radeon software) that ship amd-smi/rocminfo without /opt/rocm or
    # hipcc, and leaves `unsloth studio update` unable to repair a
    # CPU-only venv on those systems.
    if not _has_rocm_gpu():
        return  # no AMD GPU visible

    ver = _detect_rocm_version()
    if ver is None:
        print("   ROCm detected but version unreadable -- skipping torch reinstall")
        return

    # Probe whether torch already links against HIP (ROCm is already working).
    # Do NOT skip for CUDA-only builds since they are unusable on AMD-only
    # hosts (the NVIDIA check above already handled mixed AMD+NVIDIA setups).
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "hip=getattr(torch.version,'hip','') or ''; "
                    "ver=getattr(torch,'__version__','').lower(); "
                    # Print the HIP version when present (back-compat), else
                    # "rocm" sentinel when only torch.__version__ flags ROCm
                    # (AMD SDK / Radeon wheels). Empty string = CPU/CUDA.
                    "print(hip if hip else ('rocm' if 'rocm' in ver else ''))"
                ),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 90,
        )
    except (OSError, subprocess.TimeoutExpired):
        probe = None
    has_hip_torch = (
        probe is not None
        and probe.returncode == 0
        and probe.stdout.decode().strip() != ""
    )

    rocm_torch_ready = has_hip_torch

    # Strix Halo / Strix Point (gfx1151 / gfx1150) segfault under ROCm 7.1
    # in torch._grouped_mm. AMD's per-gfx repo ships torch 2.11.0+rocm7.13.0
    # with the real fix, so route those hosts there instead of the generic
    # pytorch.org rocm7.1 wheel. Mirrors install.sh's Strix override.
    # On mixed hosts (Strix iGPU + non-Strix dGPU), only route to the AMD
    # per-gfx index when the GPU HIP will actually run on is the Strix one --
    # otherwise the dGPU would get an incompatible wheel. Use HIP_VISIBLE_DEVICES
    # to determine the runtime target.
    _strix_override_url: "str | None" = None
    _strix_override_pkgs: "tuple[str, str, str] | None" = None
    if ver < (7, 2):
        gfx_codes = _detect_amd_gfx_codes()
        _strix_gfx = {"gfx1151", "gfx1150"}
        _detected_strix = _strix_gfx.intersection(gfx_codes)
        if _detected_strix:
            # Pick the runtime-visible GPU. If HIP_VISIBLE_DEVICES selects a
            # specific index into gfx_codes, use that gfx; else default to the
            # first listed GPU. Skip the override unless the resolved GPU is
            # Strix.
            _runtime_gfx = (
                gfx_codes[_pick_visible_index(len(gfx_codes))] if gfx_codes else None
            )
            if _runtime_gfx in _strix_gfx:
                _selected_gfx = _runtime_gfx
                _amd_mirror = (
                    os.environ.get("UNSLOTH_AMD_ROCM_MIRROR")
                    or "https://repo.amd.com/rocm/whl"
                ).rstrip("/")
                _strix_override_url = f"{_amd_mirror}/{_selected_gfx}/"
                _strix_override_pkgs = (
                    "torch>=2.11.0,<2.12.0",
                    # Pin torchvision/torchaudio to the 2.11.x-compatible range.
                    # The install uses --index-url (exclusive, no PyPI fallback),
                    # so bare unversioned names risk resolving a build from AMD's
                    # index that targets a different torch major (e.g. 0.27 built
                    # against torch 2.12), which would fail at runtime with an
                    # ABI/version mismatch. Matches _ROCM_TORCH_CONSTRAINT["rocm7.2"].
                    "torchvision>=0.26.0,<0.27.0",
                    "torchaudio>=2.11.0,<2.12.0",
                )
                print(
                    f"\n   {_selected_gfx} (AMD Strix) is the runtime target with ROCm "
                    f"{ver[0]}.{ver[1]}.\n"
                    f"   ROCm 7.1 has a known _grouped_mm segfault on this GPU;\n"
                    f"   routing torch install to AMD's arch-specific index\n"
                    f"   ({_strix_override_url}) which serves torch 2.11.0+rocm7.13.0\n"
                    f"   with the upstream fix.\n"
                )
            else:
                _gfx_str = ", ".join(sorted(_detected_strix))
                print(
                    f"\n   Strix GPU ({_gfx_str}) present but HIP_VISIBLE_DEVICES "
                    f"selects a non-Strix runtime target ({_runtime_gfx});\n"
                    f"   skipping AMD per-gfx index override.\n"
                )

    # Strix override on ROCm 7.1 must fire even when has_hip_torch is True --
    # an existing torch with `torch.version.hip == "7.1"` is exactly the broken
    # combo the override is meant to repair, so skipping it leaves users on
    # the known _grouped_mm segfault.
    if _strix_override_url is not None and _strix_override_pkgs is not None:
        index_url = _strix_override_url
        _torch_pkg, _vision_pkg, _audio_pkg = _strix_override_pkgs
        print(f"   Strix ROCm 7.1 override -- installing torch from {index_url}")
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
    elif not has_hip_torch:
        # Select best matching wheel tag (newest ROCm version <= installed)
        tag = next(
            (
                t
                for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                if ver >= (maj, mn)
            ),
            None,
        )
        if tag is None:
            print(
                f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- "
                f"skipping torch reinstall"
            )
        else:
            index_url = f"{_PYTORCH_WHL_BASE}/{tag}"
            print(f"   ROCm {ver[0]}.{ver[1]} -- installing torch from {index_url}")
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
    # continuous-release_main wheel (bnb PR #1887 4-bit GEMV fix) and falls
    # back to PyPI when the pre-release wheel cannot be installed. Use pip for
    # the pre-release wheel because uv rejects the wheel's filename/metadata
    # version mismatch.
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


def _uv_safe_path(path: object) -> str:
    # uv 0.11.x: `-c <path with space>` truncates at the space; use 8.3 short form.
    s = str(path)
    if not IS_WINDOWS or " " not in s:
        return s
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        rc = get_short(s, buf, 32768)
        if 0 < rc < 32768 and " " not in buf.value:
            return buf.value
    except Exception:
        pass
    return s


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

    Checks UNSLOTH_NO_TORCH env var first.  When unset, falls back to
    platform detection so that Intel Macs automatically use GGUF-only
    mode even when invoked from ``unsloth studio update`` (which does
    not inject the env var).
    """
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return IS_MAC_INTEL


NO_TORCH = _infer_no_torch()


# -- Verbosity control ----------------------------------------------------------
# By default the installer shows a minimal progress bar (one line, in-place).
# Set UNSLOTH_VERBOSE=1 in the environment to restore full per-step output:
#   CLI:        unsloth studio setup --verbose
#   Linux/Mac:  UNSLOTH_VERBOSE=1 ./studio/setup.sh
#   Windows:    $env:UNSLOTH_VERBOSE="1" ; .\studio\setup.ps1
VERBOSE: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

# Progress bar state -- updated by _progress() as each install step runs.
# Update _TOTAL here if you add or remove install steps in install_python_stack().
_STEP: int = 0
_TOTAL: int = 0  # set at runtime in install_python_stack() based on platform

# -- Paths --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)
LOCAL_DD_GITHUB_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-github-repo-seed"
)

# Apple Silicon: override mlx-vlm/mlx-lm's transformers pin (see overrides file).
_MLX_OVERRIDES = SINGLE_ENV / "overrides-darwin-arm64.txt"
if IS_MAC_ARM and _MLX_OVERRIDES.is_file():
    os.environ.setdefault("UV_OVERRIDE", str(_MLX_OVERRIDES))

# -- Unicode-safe printing ---------------------------------------------
# On Windows the default console encoding can be a legacy code page
# (e.g. CP1252) that cannot represent Unicode glyphs such as ✅ or ❌.
# _safe_print() gracefully degrades to ASCII equivalents so the
# installer never crashes just because of a status glyph.

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
        # Stringify, then swap emoji for ASCII equivalents
        text = " ".join(str(a) for a in args)
        for uni, ascii_alt in _UNICODE_TO_ASCII.items():
            text = text.replace(uni, ascii_alt)
        # Final fallback: replace any remaining unencodable chars
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


def _step(label: str, value: str, color_fn = None) -> None:
    """Print a single step line in the column format."""
    if color_fn is None:
        color_fn = _green
    padded = label[:_COL]
    _safe_print(f"  {_dim(padded)}{' ' * (_COL - len(padded))}{color_fn(value)}")


def _progress(label: str) -> None:
    """Print an in-place progress bar aligned to the step column layout."""
    global _STEP
    _STEP += 1
    if VERBOSE:
        return
    width = 20
    filled = int(width * _STEP / _TOTAL)
    bar = "=" * filled + "-" * (width - filled)
    pad = " " * (_COL - len(_LABEL))
    end = "\n" if _STEP >= _TOTAL else ""
    try:
        sys.stdout.write(
            f"\r  {_dim(_LABEL)}{pad}[{bar}] {_STEP:2}/{_TOTAL}  {label:<20}{end}"
        )
        sys.stdout.flush()
    except OSError:
        pass


def run(
    label: str, cmd: list[str], *, quiet: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit."""
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
        env = child_env_without_native_path_secret(),
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        _step("error", f"{label} failed (exit code {result.returncode})", _red)
        if result.stdout:
            print(result.stdout.decode(errors = "replace"))
        sys.exit(result.returncode)
    return result


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"open_spiel", "triton_kernels"}

# Packages to skip when torch is unavailable (Intel Mac GGUF-only mode).
# These packages either *are* torch extensions or have unconditional
# ``Requires-Dist: torch`` in their published metadata, so installing
# them would pull torch back into the environment. ``librosa`` also
# lives in this set even though it does not itself require torch:
# upstream ``llvmlite`` dropped its macOS x86_64 wheel between 0.42.0
# and 0.46.0+ (see https://pypi.org/project/llvmlite/0.47.0/#files --
# only macosx_arm64 / manylinux / win_amd64 remain), so on Intel Mac
# the librosa -> numba -> llvmlite chain triggers a from-source build
# that fails inside CI and on the host without LLVM 14/15 headers.
# Tracked separately in unslothai/unsloth#5046.
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "transformers-cfg",
    "librosa",
}


def _select_flash_attn_version(torch_mm: str) -> str | None:
    return flash_attn_package_version(torch_mm)


def _build_flash_attn_wheel_url(env: dict[str, str]) -> str | None:
    return flash_attn_wheel_url(env)


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
    # NVIDIA-only for now; AMD/Intel/CPU have no working flash-attn path.
    if not _has_usable_nvidia_gpu():
        _step(
            "warning",
            "Skipping flash-attn: no NVIDIA GPU detected",
            _cyan,
        )
        return

    def _status(message: str) -> None:
        if message.startswith("Installing prebuilt"):
            if VERBOSE:
                _step(_LABEL, message, _dim)
            return
        _step("warning", message, _cyan)

    installed = install_optional_kernel(
        FLASH_ATTN_SPEC,
        python_executable = sys.executable,
        use_uv = USE_UV,
        uv_needs_system = UV_NEEDS_SYSTEM,
        allow_pypi_fallback = False,
        status = _status,
    )
    if not installed:
        _step("warning", "Continuing without flash-attn", _cyan)


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
        line
        for line in lines
        if not any(line.strip().lower().startswith(pkg) for pkg in skip)
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
    # Always pass --python so uv targets the correct environment.
    # Without this, uv can ignore an activated venv and install into
    # the system Python (observed on Colab and similar environments).
    cmd.extend(["--python", sys.executable])
    cmd.extend(_translate_pip_args_for_uv(args))
    # Torch is pre-installed by install.sh/setup.ps1.  Do not add
    # --torch-backend by default -- it can cause solver dead-ends on
    # CPU-only machines.  Callers that need it can set UV_TORCH_BACKEND.
    _tb = os.environ.get("UV_TORCH_BACKEND", "")
    if _tb:
        cmd.append(f"--torch-backend={_tb}")
    return cmd


def pip_install_try(
    label: str,
    *args: str,
    constrain: bool = True,
    force_pip: bool = False,
) -> bool:
    """Like pip_install but returns False on failure instead of exiting.
    For optional installs with a follow-up fallback.
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
        env = child_env_without_native_path_secret(),
    )
    if result.returncode == 0:
        return True
    if VERBOSE and result.stdout:
        print(result.stdout.decode(errors = "replace"))
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
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                return
            print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                print(result.stdout.decode(errors = "replace"))

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

    # When called from install.sh (which already installed unsloth into the venv),
    # SKIP_STUDIO_BASE=1 is set to avoid redundant reinstallation of base packages.
    # When called from "unsloth studio update", it is NOT set so base packages
    # (unsloth + unsloth-zoo) are always reinstalled to pick up new versions.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # When --package is used, install a different package name (e.g. roland-sloth for testing)
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # When --local is used, overlay a local repo checkout after updating deps
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    base_total = 10 if IS_WINDOWS else 11
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    if not IS_MACOS and not NO_TORCH:
        base_total += 1  # ROCm torch check (line 1526) -- all non-macOS platforms
        if not IS_WINDOWS:
            base_total += (
                2  # flash-attn (line 1620) + ROCm torch final (line 1705) -- Linux only
            )
    _TOTAL = (base_total - 1) if skip_base else base_total

    # 1. Try to use uv for faster installs (must happen before pip upgrade
    #    because uv venvs don't include pip by default)
    USE_UV = _bootstrap_uv()

    # 2. Ensure pip is available (uv venvs created by install.sh don't include pip)
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
        # pip may not exist yet (uv-created venvs omit it). Try ensurepip
        # first, then upgrade. Only fall back to a direct upgrade when pip
        # is already present.
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

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        pass
    elif NO_TORCH:
        # No-torch update path: install unsloth + unsloth-zoo with --no-deps
        # (current PyPI metadata still declares torch as a hard dep), then
        # runtime deps with --no-deps (avoids transitive torch).
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
        # Resolve pydantic WITH deps so pip pins pydantic-core to the
        # exact version pydantic's metadata declares. Under --no-deps
        # alone pip picks the latest of each and trips pydantic's
        # _ensure_pydantic_core_version check. Transitive deps are
        # torch-free.
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
        # Local dev install: update deps from base.txt, then overlay the
        # local checkout as an editable install (--no-deps so torch is
        # never re-resolved).
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
        # Custom package name (e.g. roland-sloth for testing) — install directly
        _progress("base packages")
        pip_install(
            f"Installing {package_name}",
            "--no-cache-dir",
            package_name,
        )
    else:
        # Update path: upgrade only unsloth + unsloth-zoo while preserving
        # existing torch/CUDA installations.  Torch is pre-installed by
        # install.sh / setup.ps1; --upgrade-package targets only base pkgs.
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
    #     venv received CPU-only torch (common when pip resolves torch from PyPI).
    #     Must come immediately after base packages so torch is present for inspection.
    if not IS_MACOS and not NO_TORCH:
        _progress("ROCm torch check")
        _ensure_rocm_torch()

    # Windows + AMD GPU: if ROCm torch was not installed (wrong Python version
    # or unknown ROCm version), warn the user.
    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
        # Validate actual AMD GPU presence (not just tool existence)
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
            try:
                _wr = subprocess.run(
                    [_wexe, *_wcmd[1:]],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    text = True,
                    timeout = 10,
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

    # 4. Overrides (torchao, transformers) -- force-reinstall
    #    Skip entirely when torch is unavailable (e.g. Intel Mac GGUF-only mode)
    #    because overrides.txt contains torchao which requires torch.
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    else:
        _progress("dependency overrides")
        _override_extra_args: tuple[str, ...] = ()
        if _rocm_windows_torch_installed:
            # torchao in overrides.txt declares torch as a dependency; without
            # --no-deps uv would resolve and install CPU torch from PyPI,
            # overwriting the AMD ROCm wheels we just installed.
            _override_extra_args = ("--no-deps",)
        pip_install(
            "Installing dependency overrides",
            "--force-reinstall",
            "--no-cache-dir",
            *_override_extra_args,
            req = REQ_ROOT / "overrides.txt",
        )

    # 5. Triton kernels (no-deps, from source)
    #    Skip on Windows (no support) and macOS (no support).
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

    # 8. Studio dependencies
    _progress("studio deps")
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

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

    # 13. AMD ROCm: final torch repair.  Multiple install steps above can
    #     pull in CUDA torch from PyPI (base packages, extras, overrides,
    #     studio deps, etc.).  Running the repair as the very last step
    #     ensures ROCm torch is in place at runtime, regardless of which
    #     intermediate step clobbered it.
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress("ROCm torch (final)")
        _ensure_rocm_torch()

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
