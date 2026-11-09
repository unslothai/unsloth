# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gives uv a space-free -c/--override/-r file path (uv splits -c/--override on whitespace): 8.3
short name on Windows, else a temp copy removed at exit; falls back to the original path on
error."""

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


def _windows_short_path(s: str) -> str | None:
    """The 8.3 alias of ``s``, or None when the volume has none (common off the system drive)."""
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        rc = get_short(s, buf, 32768)
        # A space-free alias can still fail to resolve (#11290).
        if 0 < rc < 32768 and " " not in buf.value and os.path.exists(buf.value):
            return buf.value
    except Exception:
        pass
    return None


def uv_safe_path(path: object) -> str:
    s = str(path)
    if " " not in s:
        return s
    if IS_WINDOWS:
        short = _windows_short_path(s)
        if short is not None:
            return short
        # No 8.3 name (often off on non-system volumes): copy, as install.ps1's Get-UvSafeRequirementsPath does.
    tmp_dir = None
    try:
        if not os.path.isfile(s):
            return s
        tmp_dir = tempfile.mkdtemp(prefix = "unsloth_uv_")
        if " " in tmp_dir and IS_WINDOWS:
            tmp_dir = _windows_short_path(tmp_dir) or tmp_dir
        if " " in tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors = True)
            return s
        source_name = os.path.basename(s) or "uv_args.txt"
        if " " in source_name:
            dst = os.path.join(tmp_dir, source_name.replace(" ", "_"))
            shutil.copyfile(s, dst)
        else:
            alias_dir = os.path.join(tmp_dir, "source")
            source_dir = os.path.abspath(os.path.dirname(s) or os.curdir)
            try:
                os.symlink(source_dir, alias_dir, target_is_directory = True)
                dst = os.path.join(alias_dir, source_name)
            except OSError:
                # No symlink permission: copy instead; loses relative -r/-c includes but keeps the file.
                dst = os.path.join(tmp_dir, source_name)
                shutil.copyfile(s, dst)
        _UV_SAFE_PATH_TMPDIRS.append(tmp_dir)
        tmp_dir = None
        return dst
    except Exception:
        if tmp_dir is not None:  # don't leak the temp dir if the copy failed
            shutil.rmtree(tmp_dir, ignore_errors = True)
        return s
