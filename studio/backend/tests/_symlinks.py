# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skip a test that needs a real symlink where the platform cannot create one.

A non-elevated Windows user without Developer Mode gets ``OSError: [WinError 1314]`` from ``os.symlink``. Probed once
per process with a throwaway link, the way huggingface_hub's ``are_symlinks_supported`` does; Linux and macOS always
pass the probe, so the tests keep running there.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest

_SUPPORTED: list = []  # [bool], probed once


def symlinks_supported(directory: Optional[os.PathLike] = None) -> bool:
    if not _SUPPORTED:
        ok = False
        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory(dir = directory) as tmp:
                src = Path(tmp) / "src"
                src.touch()
                try:
                    os.symlink("src", Path(tmp) / "dst")
                    ok = True
                except (OSError, NotImplementedError):
                    ok = False
        _SUPPORTED.append(ok)
    return _SUPPORTED[0]


def skip_without_symlinks(directory: Optional[os.PathLike] = None) -> None:
    if not symlinks_supported(directory):
        pytest.skip(
            "this platform cannot create symlinks (Windows without Developer Mode or elevation)"
        )
