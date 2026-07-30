# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make the MSVC C toolchain reachable to Triton on Windows (AMD), see #7595.

On Windows, `import triton` succeeding does NOT mean a Triton compile will
succeed. Triton's AMD/HIP driver JIT-compiles `hip_utils.c` with `clang-cl` on
first GPU touch, and `clang-cl` finds the CRT headers (`stdlib.h`, ...) through
the `INCLUDE` environment variable that `vcvarsall.bat` sets. Studio's worker
subprocesses don't inherit a Developer-Prompt environment, so on an AMD box the
first torch.compile dies with:

    fatal error: 'stdlib.h' file not found

The installer can't fix this: the compile happens later, in a separate process.
So we establish the env in-process, right before enabling torch.compile.

`ensure_msvc_env_for_triton()` is a no-op off win32 and when the headers are
already reachable. Otherwise it locates VS via `vswhere`, imports the x64
`vcvarsall.bat` environment (INCLUDE / LIB / LIBPATH / PATH) into `os.environ`,
and reports whether the CRT headers are now present. Callers disable
torch.compile with a clear message when it returns False, mirroring the existing
"Triton not found" gate instead of letting the first compile hard-crash.
"""

from __future__ import annotations

import os
import sys
import glob
import logging
import subprocess

logger = logging.getLogger(__name__)

_VS_EDITIONS = ("BuildTools", "Community", "Professional", "Enterprise")
_VS_YEAR_DIRS = ("2022", "2019", "2017")
# Env keys vcvarsall establishes that Triton's clang-cl needs.
_CARRY = ("INCLUDE", "LIB", "LIBPATH", "PATH")


def _have_crt_headers() -> bool:
    """True if stdlib.h is reachable through the current INCLUDE."""
    for d in os.environ.get("INCLUDE", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, "stdlib.h")):
            return True
    return False


def _find_vcvarsall() -> str | None:
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")

    # Preferred: ask vswhere for the newest install of any product.
    vswhere = os.path.join(program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-products", "*", "-latest", "-prerelease",
                 "-property", "installationPath"],
                capture_output = True, text = True, timeout = 30,
            ).stdout.strip()
            if out:
                cand = os.path.join(out, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                if os.path.isfile(cand):
                    return cand
        except (OSError, subprocess.SubprocessError):
            pass

    # Fallback: scan the well-known install roots (newest year, any edition).
    for root in (program_files, program_files_x86):
        for year in _VS_YEAR_DIRS:
            for edition in _VS_EDITIONS:
                cand = os.path.join(root, "Microsoft Visual Studio", year, edition,
                                    "VC", "Auxiliary", "Build", "vcvarsall.bat")
                if os.path.isfile(cand):
                    return cand
        # Some layouts nest the year under the edition; glob as a last resort.
        for cand in glob.glob(os.path.join(
                root, "Microsoft Visual Studio", "*", "*",
                "VC", "Auxiliary", "Build", "vcvarsall.bat")):
            if os.path.isfile(cand):
                return cand
    return None


def _import_vcvars_env(vcvarsall: str, arch: str = "x64") -> bool:
    """Run vcvarsall and copy the toolchain env keys into os.environ."""
    try:
        # `set` after the call dumps the fully-populated developer environment.
        proc = subprocess.run(
            ["cmd", "/c", f'call "{vcvarsall}" {arch} >nul 2>&1 && set'],
            capture_output = True, text = True, timeout = 120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("vcvarsall invocation failed: %s", e)
        return False
    if proc.returncode != 0:
        return False

    for line in proc.stdout.splitlines():
        key, sep, val = line.partition("=")
        if sep and key.upper() in _CARRY:
            os.environ[key.upper()] = val
    return True


def ensure_msvc_env_for_triton() -> bool:
    """Ensure Triton's clang-cl can find the CRT headers on Windows.

    Returns True when the MSVC headers are reachable (or not needed, i.e. off
    win32), False when no Visual Studio / Build Tools install could supply them.
    """
    if sys.platform != "win32":
        return True
    if _have_crt_headers():
        return True

    vcvarsall = _find_vcvarsall()
    if not vcvarsall:
        return False

    _import_vcvars_env(vcvarsall)
    return _have_crt_headers()
