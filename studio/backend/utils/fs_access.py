# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Filesystem accessibility helpers.

Stdlib-only and dependency-free so it is safe to import from anywhere in
the backend (and to unit-test in isolation).
"""

import os
from pathlib import Path
from typing import Union


def is_accessible_dir(path: Union[str, os.PathLike]) -> bool:
    """Return True iff *path* is a directory the current process can list.

    ``Path.is_dir()`` cannot be used bare for this check: on Python >= 3.12
    its ``os.stat`` call only suppresses "not found"-class errors
    (``ENOENT`` / ``ENOTDIR`` / ``ELOOP``) and now *propagates*
    ``PermissionError`` (``EACCES``) instead of returning ``False`` the way
    it did on Python <= 3.11. Probing a root-owned, mode-700 path such as
    ``/usr/share/ollama/.ollama/models`` therefore raises rather than
    quietly reporting "not a usable directory".

    This helper restores the pre-3.12 intent: any path we cannot stat or
    cannot read+traverse is simply "not a candidate", never an exception.
    """
    try:
        if not Path(path).is_dir():
            return False
    except OSError:
        return False
    return os.access(path, os.R_OK | os.X_OK)
