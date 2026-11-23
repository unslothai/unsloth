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
from typing import Callable, Iterable, Optional

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


def shadowed_legacy_layout(paths: Iterable[str]) -> frozenset[str]:
    """The legacy-layout paths among ``paths`` whose root-relative name is also listed: those are alternates of a
    current file. A legacy file with no current counterpart is a file in its own right and is not shadowed."""
    paths = [str(p) for p in paths]
    current = {p.replace("\\", "/").casefold() for p in paths if not is_legacy_layout_path(p)}
    return frozenset(
        p
        for p in paths
        if (alias := legacy_layout_alias(p)) is not None and alias.casefold() in current
    )


def canonical_layout_path(path: str) -> str:
    """The root-relative name a cached file is listed under: a legacy-layout copy groups with the quant it stands in
    for, so a cache holding both copies (or only the legacy one) lists one row under the current name."""
    return legacy_layout_alias(path) or path


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


_SPLIT_SUFFIX_RE = re.compile(r"-\d{3,}-of-(\d{3,})(?=\.gguf$)", re.IGNORECASE)


def _family_of(path: str) -> tuple[str, int]:
    """(canonical shard-family name, shard total; 1 for a whole file) of a root-relative or legacy path."""
    canon = canonical_layout_path(str(path).replace("\\", "/"))
    match = _SPLIT_SUFFIX_RE.search(canon)
    return _SPLIT_SUFFIX_RE.sub("", canon).casefold(), int(match.group(1)) if match else 1


def fold_legacy_layout_files(
    rels: Iterable[str], usable: Callable[[str], bool]
) -> tuple[frozenset[str], dict[str, str]]:
    """How an on-disk listing shows a previous layout: ``(dropped, canonical)``.

    Files are grouped by shard family under their canonical (root-relative) name. Where both layouts are on disk the
    current family is kept when every shard is ``usable``, else a complete legacy family, else the current one; the
    other is ``dropped``. A kept legacy file maps to the root-relative name it is listed under in ``canonical``, so
    one row per quant either way (the physical path is still what loads)."""
    families: dict[tuple[str, int], tuple[list[str], list[str]]] = {}
    for rel in rels:
        current, legacy = families.setdefault(_family_of(rel), ([], []))
        (legacy if is_legacy_layout_path(rel) else current).append(rel)

    def complete(members: list[str], total: int) -> bool:
        return len(members) >= total and all(usable(m) for m in members)

    dropped: set[str] = set()
    canonical: dict[str, str] = {}
    for (_name, total), (current, legacy) in families.items():
        if not legacy:
            continue
        if current and (complete(current, total) or not complete(legacy, total)):
            dropped.update(legacy)
            continue
        dropped.update(current)
        for rel in legacy:
            canonical[rel] = canonical_layout_path(rel)
    return frozenset(dropped), canonical
