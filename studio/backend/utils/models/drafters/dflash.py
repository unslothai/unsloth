# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DFlash sidecar discovery for a local GGUF model. DFlash is published as a ``dflash-`` prefixed sibling of the weights it drafts for, so discovery is a naming question first and a header question second. That order is deliberate: a caller with a directory lease (a native grant) hands in ``accept``, and that has to answer before anything is opened, because reading the header of a symlink pointing out of the lease is the very thing the lease exists to prevent, and no later rejection takes a read back."""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from utils.models.gguf_metadata import read_gguf_architecture
from utils.models.drafters.common import (
    _drafter_launch_path,
    _drafter_matches_weight,
    _drafter_names_other_weight,
    _drafter_split_is_complete,
    _drafter_stem_rank,
    _drafter_total_size,
)
from utils.models.drafters.preference import (
    dflash_precision_rank,
    dflash_preference_key,
    dflash_repo_preference_key,
)

logger = logging.getLogger(__name__)


def is_dflash_architecture(path: str) -> bool:
    """Decided by the header's general.architecture = dflash; a dflash- filename can name a real weight."""
    return (read_gguf_architecture(str(path)) or "").lower() == "dflash"


def detect_dflash_file(
    path: str,
    search_root: Optional[str] = None,
    accept: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Root level only; the header is checked, as the published dflash-kquant.gguf names no model family."""

    # Imported per call: model_config imports this module.
    from utils.models.model_config import _local_gguf_load_path

    def _rank(candidate: Path) -> tuple[int, int, int, int, str]:
        # Rank: names this weight's family, then unpaired, then precision, then total size, then name.
        paired = _drafter_matches_weight(candidate.name, weight_name, kind = "dflash")
        return (
            0 if paired else 1,
            _drafter_stem_rank(candidate.name, kind = "dflash") if paired else 0,
            dflash_precision_rank(candidate.name),
            _drafter_total_size(candidate),
            candidate.name.lower(),
        )

    p = Path(path)
    weight_name = p.name if p.suffix.lower() == ".gguf" else None
    start_dir = p.parent if p.is_file() else p
    # Not assets/: the sidecar names no family, so it is ambiguous in Hermes' shared pool.
    dirs = [start_dir]
    if search_root is not None:
        dirs.append(Path(search_root))

    candidates: list[Path] = []
    other_weights: list[str] = []
    seen: set[Path] = set()
    # dict.fromkeys: search_root may equal the weight's parent; avoid scanning it twice.
    for root in dict.fromkeys(dirs):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for candidate in entries:
            lower = candidate.name.lower()
            if not lower.endswith(".gguf"):
                continue
            # Prefix form only: shared predicates know DFlash by dflash-, else a file is both drafter and model.
            if not lower.startswith("dflash-"):
                # Recorded so a sidecar naming a neighbour can be told apart from one naming no family.
                other_weights.append(candidate.name)
                continue
            try:
                launch = _local_gguf_load_path(candidate)
                # is_file() follows links: drops dangling symlinks and dirs that would fail --model-draft.
                if not (launch.is_file() and _drafter_split_is_complete(launch)):
                    continue
                resolved = launch.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(launch)

    # A sidecar naming a neighbour weight's family is that neighbour's drafter; headers cannot tell.
    if weight_name is not None and other_weights:
        kept: list[Path] = []
        for candidate in candidates:
            if _drafter_names_other_weight(candidate.name, weight_name, other_weights):
                logger.info(
                    "detect_dflash_file: dropped %s (names another weight in this folder)",
                    candidate.name,
                )
                continue
            kept.append(candidate)
        candidates = kept

    for candidate in sorted(candidates, key = _rank):
        # Resolve and validate before opening: a granted path can symlink outside the lease.
        try:
            launch = _drafter_launch_path(candidate)
        except OSError:
            continue
        if accept is not None and not accept(launch):
            logger.info(
                "detect_dflash_file: dropped %s (outside the granted directory)",
                candidate.name,
            )
            continue
        if not is_dflash_architecture(launch):
            logger.info(
                "detect_dflash_file: dropped %s (architecture %r is not dflash)",
                candidate.name,
                read_gguf_architecture(launch),
            )
            continue
        logger.info("Detected DFlash drafter: %s", launch)
        return launch
    return None
