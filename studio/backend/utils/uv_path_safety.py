# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand uv a space-free `-c`/`--override`/`-r` file path (issue #6503).

uv splits `-c`/`--override` (and UV_OVERRIDE) on whitespace, so a path with a
space truncates. Windows uses the 8.3 short form; POSIX copies the file into a
space-free temp dir (removed at exit). Falls back to the original path on error.
Shared by install_python_stack and utils.mlx_repair.
"""

from __future__ import annotations

import atexit
import os
import platform
import shutil
import tempfile

IS_WINDOWS = platform.system() == "Windows"

_UV_SAFE_PATH_TMPDIRS: list[str] = []


@atexit.register
def _cleanup_uv_safe_path_tmpdirs() -> None:
    while _UV_SAFE_PATH_TMPDIRS:
        shutil.rmtree(_UV_SAFE_PATH_TMPDIRS.pop(), ignore_errors = True)


def uv_safe_path(path: object) -> str:
    s = str(path)
    if " " not in s:
        return s
    if IS_WINDOWS:
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
    tmp_dir = None
    try:
        if not os.path.isfile(s):
            return s
        tmp_dir = tempfile.mkdtemp(prefix = "unsloth_uv_")
        if " " in tmp_dir:  # e.g. TMPDIR itself has a space
            shutil.rmtree(tmp_dir, ignore_errors = True)
            return s
        dst = os.path.join(
            tmp_dir, (os.path.basename(s) or "uv_args.txt").replace(" ", "_")
        )
        shutil.copyfile(s, dst)
        _UV_SAFE_PATH_TMPDIRS.append(tmp_dir)
        tmp_dir = None
        return dst
    except Exception:
        if tmp_dir is not None:  # don't leak the temp dir if the copy failed
            shutil.rmtree(tmp_dir, ignore_errors = True)
        return s
