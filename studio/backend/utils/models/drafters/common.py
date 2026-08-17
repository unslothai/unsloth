# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Primitives shared by every speculative drafter kind.

MTP, DSpark and DFlash all pair a sidecar with a main weight by name, resolve a
launch path through a split set, and answer whether that set is complete. Those
rules live here, once, so a change to the pairing rule cannot reach one kind and
miss another. Nothing here is DFlash specific.
"""

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

# model_config imports this module, so the split and quant naming helpers below
# are pulled in per call rather than at module import time. They are constants
# and pure functions shared with non-drafter code (gguf_variants, the auto
# download paths), which is why they stay where they are rather than moving
# here: this package must not become a second home for the GGUF naming rules.


def _drafter_pairing_stem(name: str, *, kind: str) -> str:
    """The model family a drafter filename names, stripped of its own markers.

    Both published schemes are handled: ``<kind>-<model>`` and the older
    ``<model>-<KIND>``. The shard suffix sits outside the quant token, so it
    goes first or the anchored quant strip below cannot match. Full quant
    vocabulary, not a subset: K/IQ/UD/MXFP drafters pair too, and the optional
    bpw modifier goes with it, as _extract_quant_label does.
    """
    stem = Path(name).stem.lower()
    if stem.startswith(f"{kind}-"):
        stem = stem[len(kind) + 1 :]
    stem = re.sub(r"-[0-9]{5}-of-[0-9]{5}$", "", stem)
    if stem.endswith(f"-{kind}"):
        stem = stem[: -(len(kind) + 1)]
    from utils.models.model_config import _GGUF_KNOWN_QUANT_RE

    return re.sub(
        rf"-(?:{_GGUF_KNOWN_QUANT_RE.pattern})(?:-[0-9]+(?:\.[0-9]+)?bpw)?$",
        "",
        stem,
        flags = re.IGNORECASE,
    )


def _drafter_matches_weight(candidate_name: str, weight_name: Optional[str], *, kind: str) -> bool:
    """Whether a drafter pairs with the weight, by name.

    A multi-model folder must not attach a foreign drafter, so the family the
    drafter names has to PREFIX the weight filename at a non-alphanumeric
    boundary. That blocks one direction of a ``DeepSeek-V4-Flash-Lite`` /
    ``DeepSeek-V4-Flash`` pair but not the other: the shorter family name is a
    prefix of the longer weight, so a base-family sidecar still matches a
    longer-named sibling's weights. Exact equality cannot replace the prefix
    rule -- ``mtp-gemma-4-12B-it.gguf`` really does ship beside
    ``gemma-4-12B-it-qat-*.gguf`` -- so the remaining direction is settled by
    ranking: callers prefer the longest matching stem (see _drafter_stem_rank).
    """
    if weight_name is None:
        return True
    stem = _drafter_pairing_stem(candidate_name, kind = kind)
    weight = weight_name.lower()
    return (
        bool(stem)
        and weight.startswith(stem)
        and (len(weight) == len(stem) or not weight[len(stem)].isalnum())
    )


def _drafter_stem_rank(candidate_name: str, *, kind: str) -> int:
    """Sort key placing the most specific family first (longest stem wins).

    Both ``mtp-DeepSeek-V4-Flash-BF16.gguf`` and
    ``mtp-DeepSeek-V4-Flash-0731-BF16.gguf`` prefix-match a 0731 weight, and
    only the second is really its drafter.
    """
    return -len(_drafter_pairing_stem(candidate_name, kind = kind) or "")


def _drafter_launch_path(candidate: Path) -> str:
    """The path llama-server should receive for *candidate*.

    llama-server takes shard 1 as the model path, and a split copy must stay on
    its snapshot path: the blob target has no sibling shard names. Single-file
    drafters still resolve, as callers expect.
    """
    from utils.models.model_config import _GGUF_SPLIT_FILE_RE, _local_gguf_load_path

    loadable = _local_gguf_load_path(candidate)
    if _GGUF_SPLIT_FILE_RE.match(loadable.name):
        return str(loadable)
    return str(loadable.resolve())


def _drafter_split_is_complete(candidate: Path) -> bool:
    """False for a partial split set, which would fail llama-server's draft
    startup and disable speculation entirely; skip it so a complete copy wins."""
    from utils.models.model_config import colocated_split_shards

    try:
        _, complete = colocated_split_shards(candidate)
    except OSError:
        return False
    return complete


def _drafter_total_size(candidate: Path) -> int:
    """Bytes across every shard. Candidates are collapsed to shard 1, so a split
    copy must be summed or it would outrank a smaller single file."""
    from utils.models.model_config import colocated_split_shards
    try:
        shards, _ = colocated_split_shards(candidate)
        return sum(shard.stat().st_size for shard in shards)
    except OSError:
        return sys.maxsize


def _drafter_names_other_weight(
    candidate_name: str,
    weight_name: Optional[str],
    other_weight_names: Iterable[str],
    *,
    kind: str = "dflash",
) -> bool:
    """Whether a sidecar names a DIFFERENT weight sitting beside it.

    A sidecar that names no family at all (the published ``dflash-kquant.gguf``,
    whose stem is a precision token) has to stay eligible, so "does it name a
    family" cannot be answered from the sidecar name alone. It is answered
    against the weights actually present instead: only a stem that pairs with
    some OTHER weight in the same repo/folder is evidence the sidecar belongs to
    that neighbour rather than to the weight being loaded.
    """
    if weight_name is None:
        return False
    if _drafter_matches_weight(candidate_name, weight_name, kind = kind):
        return False
    return any(
        _drafter_matches_weight(candidate_name, other, kind = kind) for other in other_weight_names
    )


_LISTED_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def split_listing_is_complete(names: Iterable[str], name: str) -> bool:
    """Whether ``names`` carries every shard of the set ``name`` belongs to.

    The listing counterpart of _drafter_split_is_complete, which needs files on disk.
    A repo mid-upload lists part of a set and the fetch refuses that, so the plan and
    the budget must agree. True for a single-file name, which encodes no set.

    Counted within the file's own directory. A repo laid out by quant can hold
    Q4/model-00001-of-00002.gguf beside Q8/model-00002-of-00002.gguf, and matching
    on basenames alone would call both halves of two broken sets one whole one.
    """
    match = _LISTED_SHARD_RE.match(Path(name).name)
    if not match:
        return True
    stem, total = match.group(1), int(match.group(3))
    parent = Path(name).parent
    sibling = re.compile(
        r"^" + re.escape(stem) + r"-(\d{5})-of-" + re.escape(match.group(3)) + r"\.gguf$",
        re.IGNORECASE,
    )
    # Distinct indices inside 1..total, not a count: a mid-publication listing can
    # hold 00001-of-00002 beside a stray 00003-of-00002, and counting would call
    # that pair whole while shard 2 is still missing and llama-server cannot open
    # the set. _drafter_split_is_complete answers the on-disk version the same way.
    seen = set()
    for other in names:
        if Path(other).parent != parent:
            continue
        found = sibling.match(Path(other).name)
        if not found:
            continue
        index = int(found.group(1))
        if 1 <= index <= total:
            seen.add(index)
    return len(seen) == total
