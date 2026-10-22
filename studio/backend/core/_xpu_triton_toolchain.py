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
import subprocess

logger = logging.getLogger(__name__)

# The environment cl needs to compile and link. Nothing else from vcvars is taken.
_VCVARS_KEYS = ("PATH", "INCLUDE", "LIB", "LIBPATH")


def _triton_is_xpu() -> bool:
    """XPU Triton and triton-windows own the same top-level name; only the distribution says which."""
    try:
        import importlib.metadata as md  # noqa: PLC0415
        dists = md.packages_distributions().get("triton") or ()
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve which distribution owns `triton`", exc_info = True)
        return False
    return any("xpu" in d.lower() for d in dists)


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


def _have_crt_headers() -> bool:
    dirs = os.environ.get("INCLUDE", "").split(os.pathsep)

    def found(name: str) -> bool:
        return any(d and os.path.isfile(os.path.join(d, name)) for d in dirs)

    return found("stdlib.h") and found("vcruntime.h")


def _msvc_environment_loaded() -> bool:
    return bool(shutil.which("cl")) and _have_crt_headers() and bool(os.environ.get("LIB"))


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
    """The variables vcvars64.bat sets, read back from the shell that ran it."""
    try:
        done = subprocess.run(
            f'cmd /d /s /c ""{vcvars}" >nul 2>&1 && set"',
            capture_output = True,
            text = True,
            timeout = 120,
        )
    except Exception:  # noqa: BLE001
        logger.debug("vcvars64.bat could not be run", exc_info = True)
        return None
    if done.returncode != 0:
        return None
    env = {}
    for line in (done.stdout or "").splitlines():
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
