# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repos that move an older file layout into a folder, and cached files that are still published.

When a GGUF repo replaces its root files with a new layout (same names, different bytes) it can keep
the previous files byte-identical under a legacy-layout folder (``Unsloth_Layout/``). Studio then:

* does not list that folder as extra quants: its files are alternates of the root quant with the
  same relative name, used when they are already on disk;
* treats a cached copy whose bytes are still published anywhere in the repo as current, so a user
  who downloaded the previous layout is neither re-downloaded nor told an update is available.

Kept import-light (no ``utils.models`` import) so hub planning and the loader can share it.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

# Top-level folders holding a repo's previous layout, compared case-insensitively.
LEGACY_LAYOUT_DIRS = frozenset({"unsloth_layout"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parts(path: str) -> list[str]:
    return [part for part in str(path).replace("\\", "/").split("/") if part]


def legacy_layout_alias(path: str) -> Optional[str]:
    """The root-relative name a legacy-layout file stands in for, or None for any other path."""
    parts = _parts(path)
    if len(parts) < 2 or parts[0].lower() not in LEGACY_LAYOUT_DIRS:
        return None
    return "/".join(parts[1:])


def is_legacy_layout_path(path: str) -> bool:
    return legacy_layout_alias(path) is not None


def without_legacy_layout(paths: Iterable[str]) -> list[str]:
    """``paths`` minus legacy-layout files, order kept."""
    return [p for p in paths if not is_legacy_layout_path(p)]


def cached_blob_sha256(path: str) -> Optional[str]:
    """The LFS sha256 of a file in a Hugging Face cache snapshot: its blob's file name.

    None when ``path`` is not a symlink into ``blobs/`` named by a sha256 (a git blob is named by
    its sha1, a Windows cache may hold plain copies)."""
    try:
        target = os.path.realpath(path)
    except OSError:
        return None
    name = os.path.basename(target)
    if os.path.basename(os.path.dirname(target)) != "blobs" or not _SHA256_RE.match(name):
        return None
    return name
