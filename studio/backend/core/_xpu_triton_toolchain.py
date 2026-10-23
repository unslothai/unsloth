# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give Intel's XPU Triton a C++ compiler it can drive on Windows (#14807).

XPU Triton builds its SYCL helpers with `icpx`, else `cl` from PATH, else `$CXX`. Studio is not
launched from a Developer Command Prompt, so `cl` is not on PATH and the first Triton kernel fails
with "Failed to find C++ compiler". Pointing CXX at cl does not help on its own: Triton picks MSVC
flags only when the compiler path contains the literal `cl.EXE`, so `cl.exe` gets GNU flags
(`/Wno-psabi`, D8021), and a GNU driver like clang++ gets the MSVC `/LIBPATH:` Triton always adds
on Windows. So: load the Visual Studio environment when it is missing, and hand Triton cl spelled
the way it recognises.
"""

from __future__ import annotations

import os
import sys
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

from ._msvc_env import _have_crt_headers

logger = logging.getLogger(__name__)

# The environment cl needs to compile and link. Nothing else from vcvars is taken.
_VCVARS_KEYS = ("PATH", "INCLUDE", "LIB", "LIBPATH")


def _triton_is_xpu() -> bool:
    """Intel's fork is the only Triton with an `intel` backend (triton-windows ships amd + nvidia). The
    gate has imported it already; a directory check, since `packages_distributions()` scans every dist."""
    path = getattr(sys.modules.get("triton"), "__file__", None)
    return bool(path) and os.path.isdir(os.path.join(os.path.dirname(path), "backends", "intel"))


def _basename(path: str) -> str:
    return os.path.basename(str(path)).lower()


def _is_cl(path: str) -> bool:
    return _basename(path) in ("cl", "cl.exe")


def _is_icpx(path: str) -> bool:
    return _basename(path) in ("icpx", "icpx.exe")


def _triton_cl_spelling(cl_path: str) -> str:
    """The same file, spelled so Triton's case-sensitive `"cl.EXE" in cc` test picks MSVC flags.
    Windows paths are case-insensitive, so only the spelling changes."""
    return os.path.join(os.path.dirname(cl_path), "cl.EXE")


def _msvc_environment_loaded() -> bool:
    return bool(shutil.which("cl")) and _have_crt_headers() and bool(os.environ.get("LIB"))


def _level_zero_sdk_found() -> bool:
    """Triton's helpers include `level_zero/ze_api.h` and link `ze_loader.lib`, which the GPU driver and
    the pip wheels do not ship. Triton looks under LEVEL_ZERO_V1_SDK_PATH, else ZE_PATH; cl also reads INCLUDE."""
    root = os.environ.get("LEVEL_ZERO_V1_SDK_PATH") or os.environ.get("ZE_PATH")
    dirs = [os.path.join(root, "include")] if root else []
    dirs += os.environ.get("INCLUDE", "").split(os.pathsep)
    return any(d and os.path.isfile(os.path.join(d, "level_zero", "ze_api.h")) for d in dirs)


# The official Windows SDK (MIT): headers + ze_loader.lib only, 317 KB. Pinned by hash, fetched once.
_L0_VERSION = "1.34.0"
_L0_URL = (
    "https://github.com/oneapi-src/level-zero/releases/download/"
    f"v{_L0_VERSION}/level-zero-win-sdk-{_L0_VERSION}.zip"
)
_L0_SHA256 = "3050280cc94b7a5db7fb173391fcced9b7284ae196bcaf05a566d2911baa7733"


def _sdk_complete(root: Path) -> bool:
    return (root / "include" / "level_zero" / "ze_api.h").is_file() and (
        root / "lib" / "ze_loader.lib"
    ).is_file()


def _fetch_level_zero_sdk(dest: Path) -> None:
    import hashlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    with urllib.request.urlopen(_L0_URL, timeout = 30) as response:
        blob = response.read()
    if hashlib.sha256(blob).hexdigest() != _L0_SHA256:
        raise ValueError("Level Zero SDK archive does not match its pinned hash")
    dest.parent.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = dest.name + ".", dir = dest.parent))
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for member in archive.infolist():
                # The archive names members with backslashes; zipfile only maps those on Windows.
                parts = member.filename.replace("\\", "/").split("/")
                if member.is_dir() or not parts[-1]:
                    continue
                if any(p in ("", ".", "..") or ":" in p for p in parts):
                    raise ValueError(f"unexpected member {member.filename!r}")
                target = staging.joinpath(*parts)
                target.parent.mkdir(parents = True, exist_ok = True)
                target.write_bytes(archive.read(member))
        # Another worker may have won the race; its copy is the same pinned archive.
        try:
            os.replace(staging, dest)
        except OSError:
            if not _sdk_complete(dest):
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors = True)


def _provide_level_zero_sdk() -> str | None:
    """Studio's cached copy of the SDK, fetched on first need. None when it cannot be had (offline)."""
    try:
        from utils.paths.storage_roots import cache_root  # noqa: PLC0415

        dest = cache_root() / f"level-zero-sdk-{_L0_VERSION}"
        if not _sdk_complete(dest):
            _fetch_level_zero_sdk(dest)
        return str(dest) if _sdk_complete(dest) else None
    except Exception:  # noqa: BLE001
        logger.debug("Could not provide the Level Zero SDK", exc_info = True)
        return None


def _vcvars_path() -> str | None:
    """vcvars64.bat of the newest Visual Studio with the x64 C++ tools, found the way setup.ps1 does."""
    root = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        done = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output = True,
            text = True,
            timeout = 60,
        )
    except Exception:  # noqa: BLE001
        logger.debug("vswhere could not be run", exc_info = True)
        return None
    install = (done.stdout or "").strip().splitlines()
    if done.returncode != 0 or not install:
        return None
    vcvars = os.path.join(install[0].strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if os.path.isfile(vcvars) else None


def _vcvars_environment(vcvars: str) -> dict | None:
    """The variables vcvars64.bat sets, read back from the shell that ran it. `/u` makes `set` write
    UTF-16: the default OEM code page would garble a non-ASCII PATH entry (setuptools' _get_vc_env)."""
    try:
        done = subprocess.run(
            f'cmd /d /u /s /c ""{vcvars}" >nul 2>&1 && set"',
            capture_output = True,
            timeout = 120,
        )
    except Exception:  # noqa: BLE001
        logger.debug("vcvars64.bat could not be run", exc_info = True)
        return None
    if done.returncode != 0:
        return None
    env = {}
    for line in (done.stdout or b"").decode("utf-16-le", errors = "replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.upper() in _VCVARS_KEYS:
            env[key.upper()] = value
    return env if "INCLUDE" in env and "PATH" in env else None


def _load_msvc_environment() -> bool:
    vcvars = _vcvars_path()
    if vcvars is None:
        return False
    env = _vcvars_environment(vcvars)
    if not env:
        return False
    os.environ.update(env)
    return True


def prepare_xpu_triton_compiler(log: logging.Logger) -> None:
    """Make XPU Triton's helper builds work on Windows. Never raises; an unfixable state is logged."""
    if sys.platform != "win32" or not _triton_is_xpu():
        return
    try:
        _prepare(log)
    except Exception:  # noqa: BLE001 -- a toolchain fix must not take down the worker
        logger.debug("Preparing XPU Triton's compiler raised", exc_info = True)


def _prepare(log: logging.Logger) -> None:
    cxx = os.environ.get("CXX")
    if cxx and _is_icpx(cxx):
        return
    if cxx and not _is_cl(cxx):
        log.warning(
            "CXX=%s: Intel XPU Triton on Windows adds MSVC /LIBPATH: flags to every build, so only "
            "cl or icpx can compile its helpers. Unset CXX to let Studio use Visual Studio's cl.",
            cxx,
        )
        return
    if not cxx and shutil.which("icpx"):
        return
    if not _msvc_environment_loaded() and _load_msvc_environment():
        log.info("Loaded the Visual Studio C++ environment for Intel XPU Triton")
    cl = shutil.which(cxx or "cl")
    if cl is None:
        log.warning(
            "Intel XPU Triton needs a C++ compiler on Windows and none was found. Install Visual "
            "Studio Build Tools (C++ workload): winget install Microsoft.VisualStudio.2022.BuildTools "
            '--override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended".'
        )
        return
    os.environ["CXX"] = _triton_cl_spelling(cl)
    if _level_zero_sdk_found():
        return
    sdk = _provide_level_zero_sdk()
    if sdk is not None:
        os.environ["LEVEL_ZERO_V1_SDK_PATH"] = sdk
        return
    log.warning(
        "Intel XPU Triton on Windows also needs the Level Zero SDK, which the GPU driver does not "
        "include, and it could not be downloaded: unzip level-zero-win-sdk-%s.zip from "
        "https://github.com/oneapi-src/level-zero/releases and set LEVEL_ZERO_V1_SDK_PATH to that "
        "folder, then restart Unsloth.",
        _L0_VERSION,
    )
