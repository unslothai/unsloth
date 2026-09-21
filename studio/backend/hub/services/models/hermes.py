# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermes model inventory: the GGUFs Hermes Desktop's one-click download stages.

Hermes keeps its managed runtime's weights as flat ``.gguf`` files directly in
``<hermes root>/models``, with vision projectors and speculative-decode drafters
tucked under a ``models/assets/`` subdirectory so its own router never lists them
as servable. Which files count is decided by
``hermes_cli.local_runtime.bootstrap.staged_models``; :func:`staged_gguf_files`
mirrors that rule, because a file Hermes does not consider servable is one Studio
must not offer to load either.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from loggers import get_logger

from core.inference.scan_incidents import note_scan_incident
from hub.schemas.inventory import LocalModelInfo
from hub.services.models.common import (
    _classify_local_path,
    _is_main_gguf_filename,
    _sum_file_sizes,
)

logger = get_logger(__name__)

# llama.cpp's split naming, e.g. Model-00001-of-00005.gguf. Case-insensitive like the suffix
# filter below: a shard that matched the filter but not this bypassed grouping, so every part,
# continuations and half-finished downloads included, became a row.
_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def staged_model_id(path: Path) -> str:
    """The id Hermes knows a staged GGUF by: its stem, minus any split suffix."""
    return re.sub(r"-\d{5}-of-\d{5}$", "", path.stem)


def staged_gguf_files(hermes_dir: Path) -> List[Path]:
    """Servable GGUFs staged in *hermes_dir*, a split counted once by its first part.

    A flat glob, never a walk: ``assets/`` holds companions rather than models,
    and Hermes excludes it by scanning only the top level. A split counts only
    when EVERY part is on disk -- a download still in flight is not loadable, and
    surfacing it would offer a model that fails at load. Continuation parts are
    never rows of their own; llama.cpp opens the whole set from part one.
    """
    try:
        # iterdir, not glob: glob SWALLOWS the enumeration OSError and yields nothing, so an
        # unreadable or stale-mounted folder reached the caller as an empty one and the pass
        # published as complete over it (verified against CPython: no search permission makes
        # glob return [] while iterdir raises). The suffix is compared case-INSENSITIVELY,
        # which is what glob does on Windows, where a staged MODEL.GGUF was discovered.
        files = sorted(
            p for p in hermes_dir.iterdir() if p.suffix.lower() == ".gguf" and p.is_file()
        )
    except OSError as exc:
        logger.warning("Error scanning Hermes directory %s: %s", hermes_dir, exc)
        # An empty list is indistinguishable from a directory holding nothing, which a
        # caller memoizing a MISS needs told apart.
        note_scan_incident(f"hermes dir unreadable: {hermes_dir}")
        return []

    # Folded: the shard names are compared against it, and the filter above accepts any
    # casing, so the parts of one split can differ in case.
    names = {p.name.lower() for p in files}
    staged: List[Path] = []
    for path in files:
        # mmproj/drafter companions belong under assets/, but a hand-dropped one
        # at the top level is still not something to serve.
        if not _is_main_gguf_filename(path.name):
            continue
        part = _SPLIT_PART.search(path.name)
        if part is None:
            staged.append(path)
            continue
        if part.group(1) != "00001":
            continue
        stem = path.name[: part.start()]
        total = int(part.group(2))
        if all(
            f"{stem}-{index:05d}-of-{part.group(2)}.gguf".lower() in names
            for index in range(2, total + 1)
        ):
            staged.append(path)
    return staged


def scan_hermes_dir(hermes_dir: Path, *, limit: Optional[int] = None) -> List[LocalModelInfo]:
    """Scan a Hermes models directory for downloaded models."""
    if not hermes_dir.exists() or not hermes_dir.is_dir():
        return []

    from utils.models.model_config import colocated_split_shards

    rows: List[LocalModelInfo] = []
    for path in staged_gguf_files(hermes_dir):
        try:
            updated_at = path.stat().st_mtime
        except OSError:
            updated_at = None
        classified = _classify_local_path(
            path,
            "hermes",
            display_name = staged_model_id(path),
            updated_at = updated_at,
        )
        # The row points at part one and the classifier sizes the one file it was handed;
        # the download is the whole set.
        shards, _complete = colocated_split_shards(path)
        if len(shards) > 1:
            size_bytes = _sum_file_sizes(shards)
            classified = [row.model_copy(update = {"size_bytes": size_bytes}) for row in classified]
        rows += classified
        if limit is not None and len(rows) >= limit:
            break
    return rows
