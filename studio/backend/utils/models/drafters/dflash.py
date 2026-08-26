# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DFlash sidecar discovery for a local GGUF model.

DFlash is published as a ``dflash-`` prefixed sibling of the weights it drafts
for, so discovery is a naming question first and a header question second. The
order of those two matters and is deliberate: a caller with a directory lease
(a native grant) hands in ``accept``, and that has to answer before anything is
opened, because reading the header of a symlink pointing out of the lease is
the very thing the lease exists to prevent, and no later rejection takes a read
back.
"""

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
    """Whether a GGUF really is a DFlash sidecar, decided by its header.

    ``dflash-`` is a filename convention an ordinary weight can satisfy, by
    accident or otherwise, and llama-server only discovers that at startup: it
    refuses the file as ``--model-draft`` and the load falls back to no
    speculation, after the bytes were already fetched. A DFlash sidecar declares
    ``general.architecture = dflash``, which no real weight does, so that is what
    settles it.

    Kept here, beside the naming rules, because the local scan
    (detect_dflash_file) and the download / cache reuse in llama_cpp all have to
    apply it -- a remote path that trusted the prefix alone would download
    gigabytes the launch then cannot use.
    """
    return (read_gguf_architecture(str(path)) or "").lower() == "dflash"


def detect_dflash_file(
    path: str,
    search_root: Optional[str] = None,
    accept: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Find a DFlash sidecar for a local GGUF model.

    Two things differ from detect_dspark_file, both forced by how DFlash is
    published:

    1. Root level only. ``dspark/`` is always a publisher's companion folder, so
       that scan is safe; ``dflash/`` is a family name a user picks for real
       weights (the reason llama_cpp._DRAFTER_DIR_KINDS leaves it out), so
       reaching into it would launch a weight copy as --model-draft.
    2. No filename pairing. The published sidecar is ``dflash-kquant.gguf``,
       which names no model family at all, so _drafter_matches_weight would
       reject the one file this exists to find. The header is checked instead:
       a DFlash sidecar declares ``general.architecture = dflash``, which no
       real weight does, and that is a stronger signal than a filename. It also
       settles the adversarial case on its own, since a model merely CALLED
       DFlash (``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf``) reports its own
       architecture.

    A sidecar that does name a family (``dflash-Qwen3.6-27B-BF16.gguf``, the
    scheme ggml-org uses) still wins over an unnamed one for the weight it
    matches, so a multi-model folder attaches the specific sidecar first.

    ``accept`` filters candidates in preference order, so a caller with extra
    rules (a native lease) keeps scanning instead of treating the first
    rejection as no sidecar at all.
    """

    # Imported per call: model_config imports this module.
    from utils.models.model_config import _local_gguf_load_path

    def _rank(candidate: Path) -> tuple[int, int, int, int, str]:
        # A sidecar naming THIS weight's family first, then any unpaired one,
        # then precision, then total size so a split copy cannot outrank a
        # smaller single file, then name for a stable order.
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
    dirs = [start_dir]
    if search_root is not None:
        dirs.append(Path(search_root))

    candidates: list[Path] = []
    other_weights: list[str] = []
    seen: set[Path] = set()
    # dict.fromkeys: search_root is the weight's own parent for a flat layout,
    # and scanning it twice doubles the directory reads for nothing.
    for root in dict.fromkeys(dirs):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for candidate in entries:
            lower = candidate.name.lower()
            if not lower.endswith(".gguf"):
                continue
            # Prefix form only, deliberately. The shared companion predicates
            # (_drafter_path_kind, is_mtp_drafter_path) know DFlash by the
            # dflash- prefix, so accepting <model>-dflash.gguf here would let one
            # file be a drafter for discovery AND a selectable Q8_0 main model in
            # the quant picker, and choosing that variant would hand llama-server
            # the drafter as the target. Teaching the predicate the suffix
            # instead would hide a real model whose name merely ends in DFlash,
            # which is the case #7811 exists to protect, so detection gives the
            # form up rather than the picker giving up a model. No published
            # DFlash sidecar uses it; the shipped one is dflash-kquant.gguf.
            if not lower.startswith("dflash-"):
                # Every other GGUF in the folder is a weight some sidecar could
                # be naming. Recorded so a sidecar belonging to a NEIGHBOUR can
                # be told apart from one naming no family at all (below).
                other_weights.append(candidate.name)
                continue
            try:
                # Collapse a split copy to shard 1 before ranking.
                launch = _local_gguf_load_path(candidate)
                # is_file() follows the link, so this also drops a dangling
                # snapshot symlink and a directory named like a sidecar. Without
                # it --model-draft gets a path llama-server cannot open, which
                # fails the whole load rather than falling back to no
                # speculation (detect_dspark_file guards the same way).
                if not (launch.is_file() and _drafter_split_is_complete(launch)):
                    continue
                resolved = launch.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(launch)

    # A sidecar naming a family that belongs to a NEIGHBOUR weight is that
    # neighbour's drafter, not a generic one. _drafter_matches_weight is False
    # both for it and for a sidecar naming no family (dflash-kquant.gguf), so
    # ranking alone bucketed the two together and precision could float the
    # foreign one to the top: loading model B beside dflash-model-A-Q8_0.gguf
    # and dflash-kquant.gguf launched model A's drafter for model B. Both carry
    # a real dflash header, so the architecture check behind the ranking cannot
    # catch it. _drafter_names_other_weight decides against the weights actually
    # present, which keeps the published unpaired sidecar eligible (its stem,
    # "kquant", names no file here) without hardcoding which stems are precision
    # tokens. Shared with the remote paths through dflash_repo_preference_key,
    # so a download and a local scan agree on which sidecar belongs here.
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
        # Resolve and validate before opening anything. A dflash-*.gguf in a
        # directory reached through a native grant can be a symlink whose target
        # sits outside the lease, and ``accept`` is what decides that; reading the
        # header first opened the target before the answer arrived, which a later
        # rejection cannot undo. Callers without a grant pass accept = None and
        # see the same candidates in the same order as before.
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
                # Re-read only on the reject path, and header reads are cached by
                # (path, mtime, size), so naming the offending architecture in the
                # log costs nothing.
                read_gguf_architecture(launch),
            )
            continue
        logger.info("Detected DFlash drafter: %s", launch)
        return launch
    return None
