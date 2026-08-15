# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which downloaded media model a requested name means, and whether it is the resident one.

Two halves of one question. The index walks the model roots once per few seconds and maps every
name a downloaded image or video model answers to onto the load spec its route takes. The
matching half then decides whether the backend already holds that exact build, which is what
lets a switch be skipped rather than reloading the model that is already serving.

Only downloaded models are indexed. A name that resolves to nothing is refused by the caller
rather than answered by whichever model happens to be resident.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

IMAGE_TASK = "text-to-image"
VIDEO_TASK = "text-to-video"

# the scan walks several roots and reads gguf headers, and this runs per request
_INDEX_TTL_S = 5.0
_index_lock = threading.Lock()
_index: dict[str, tuple[float, dict[str, "MediaModelPick"]]] = {}

# the video family whose partitions are a load-time choice rather than a property of the files
_H3_FAMILY = "minimax-h3"


@dataclass(frozen = True)
class MediaModelPick:
    """A downloaded media model, in the shape its load route takes."""

    model_id: str
    model_path: str
    gguf_filename: Optional[str] = None
    model_kind: Optional[str] = None
    # true when a sibling build publishes the same quant token, so identity cannot be proven
    ambiguous: bool = False


# sentinel for a name two different models answer to; resolution treats it as no match
_AMBIGUOUS = MediaModelPick("", "")


# ── resolving a name to a downloaded model ──────────────────────────


def _resolve_load_dir(p: Path) -> Path:
    """The directory holding the weights, unwrapping an HF cache repo to its snapshot.

    The chat resolver's helper, reused so both surfaces resolve a cached repo to the same
    local directory rather than to the download-capable repo id.
    """
    from core.inference.local_model_resolver import _resolve_load_dir as _chat_resolve
    return Path(_chat_resolve(p))


def _register(index: dict[str, MediaModelPick], keys, pick: MediaModelPick) -> None:
    """Bind every name *pick* answers to, dropping any that two different models share.

    Display labels collide readily: a cached repo advertises its final component, so
    ``org-a/model`` and ``org-b/model`` both offer ``model``. Taking whichever the scan
    reached first would load arbitrary weights for a name the docs say is usable, and the
    full ids stay available either way.
    """
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            continue
        normalized = key.strip().lower()
        existing = index.get(normalized)
        if existing is None:
            index[normalized] = pick
        elif existing is not _AMBIGUOUS and (existing.model_path, existing.gguf_filename) != (
            pick.model_path,
            pick.gguf_filename,
        ):
            index[normalized] = _AMBIGUOUS


def _name_keys(info) -> tuple[str, ...]:
    """Names a request may use for *info*: its repo id, scanner id and label.

    An absolute path is excluded: the ./models and LM Studio scanners report one as ``id``,
    and a host path is not something an API caller should have to send.
    """
    from core.inference.local_model_resolver import _is_abs_path_id
    return tuple(
        value
        for value in (
            getattr(info, "model_id", None),
            getattr(info, "id", None),
            getattr(info, "display_name", None),
        )
        if isinstance(value, str) and value and not _is_abs_path_id(value)
    )


def _gguf_load_path(info, on_disk: Path, load_dir: Path) -> str:
    """What ``/images/load`` takes as ``model_path`` for a GGUF under *info*.

    An HF cache repo is named by its repo id, as the picker names a Hub pick. Its snapshot
    entries are symlinks into ``blobs/``, and the loader's local branch resolves a symlink
    before its containment check, so a snapshot directory refuses its own file. Anything else
    is a real directory and loads by path.

    Keyed on the layout rather than the scanner's ``source``, which is rewritten to ``custom``
    for a cache tree sitting inside a user-added scan folder while the symlinks stay exactly
    as fragile.
    """
    repo_id = getattr(info, "model_id", None)
    if load_dir != on_disk and isinstance(repo_id, str) and repo_id:
        return repo_id
    return str(load_dir)


def _loader_can_open(load_path: str, filename: str) -> bool:
    """Whether the load routes will resolve *filename* under *load_path*, by their own rule.

    A repo id is opened from the cache by id and has nothing to check here. A directory does:
    an HF cache snapshot's entries are symlinks into ``blobs/``, and both validators resolve a
    symlink before their containment check, so such a directory refuses its own file. The
    scanner hands one over already unwrapped for a non-active cache, where naming the repo id
    instead would only send the loader to the active cache and download the model again.

    A split checkpoint counts as openable only when its whole set is beside it: the loader opens
    the siblings implicitly, and the planners read a local checkpoint as already present, so a
    half-copied set would evict the resident model and then fail.

    Advertising a name the loader then refuses costs a 400 on every request for a model the
    lister shows as downloaded, so an unopenable build is left out of the index.
    """
    from utils.models.model_config import colocated_split_shards

    root = Path(load_path)
    if not root.is_dir():
        # a repo id loads from the cache, where the containment rule does not apply but the
        # split set still has to be whole; an uncached child is the download guard's business
        cached = _cached_repo_file(load_path, filename)
        return True if cached is None else bool(colocated_split_shards(cached)[1])
    from core.inference.diffusion_families import resolve_local_gguf_child

    try:
        child = resolve_local_gguf_child(root, filename)
    except Exception:  # noqa: BLE001 -- whatever the loader refuses, the index does not advertise
        return False
    # a split checkpoint opens its siblings implicitly, so an incomplete set fails at load time
    return bool(colocated_split_shards(child)[1])


def _cached_repo_file(repo_id: str, filename: str) -> Optional[Path]:
    """The cached path of *filename* in *repo_id*, or None when it is not downloaded."""
    from huggingface_hub import try_to_load_from_cache

    from core.inference.diffusion import hub_cache_dir

    try:
        hit = try_to_load_from_cache(repo_id, filename, cache_dir = hub_cache_dir())
    except Exception:  # noqa: BLE001 -- an unreadable cache is not an answer about the shards
        return None
    return Path(hit) if isinstance(hit, str) else None


def _add_gguf_picks(
    index: dict[str, MediaModelPick], info, keys: tuple[str, ...], on_disk: Path, load_dir: Path
) -> bool:
    """Index every GGUF quant under *info*, bare and as ``<id>:<QUANT>``; False if it holds none.

    A bare id means the quant a plain load takes, ranked by the ``preferred_quant`` the chat
    resolver and /v1/models already share, so one id cannot mean different weights per surface.
    Root checkpoints are ranked alone when there are any: a plain local load resolves
    non-recursively and always takes the root, so ranking a qualified ``distilled/...`` build
    alongside them would let one id mean different weights here than in the picker.
    """
    from core.inference.openai_auto_download import preferred_quant
    from utils.models.model_config import list_local_gguf_variants

    if load_dir.is_file():
        if load_dir.suffix.lower() != ".gguf":
            return False
        if _loader_can_open(str(load_dir.parent), load_dir.name):
            _register(
                index,
                keys,
                MediaModelPick(
                    keys[0],
                    str(load_dir.parent),
                    load_dir.name,
                    "gguf",
                ),
            )
        return True
    # filenames come back relative to this directory, which is what the loader joins them onto
    variants, _ = list_local_gguf_variants(str(load_dir))
    by_quant = {v.quant: v for v in variants if v.quant}
    if not by_quant:
        return False
    load_path = _gguf_load_path(info, on_disk, load_dir)
    openable = {
        quant: variant
        for quant, variant in by_quant.items()
        if _loader_can_open(load_path, variant.filename)
    }
    if not openable:
        return True
    for quant, variant in openable.items():
        # model_id stays the bare id so a "not found" error lists models, not one row per quant
        _register(
            index,
            [f"{key}:{quant}" for key in keys],
            MediaModelPick(keys[0], load_path, variant.filename, "gguf"),
        )
    unqualified = [quant for quant in openable if "/" not in quant]
    best = preferred_quant(unqualified or list(openable)) or next(iter(unqualified or openable))
    _register(
        index,
        keys,
        MediaModelPick(keys[0], load_path, openable[best].filename, "gguf"),
    )
    return True


def _loadable_directory(load_dir: Path) -> bool:
    """Whether a non-GGUF directory is something the load routes can actually open.

    Either a full diffusers pipeline, or a directory holding exactly one checkpoint, which both
    routes reinterpret as a single_file load. Several checkpoints and no index is ambiguous, and
    the routes reject it rather than choose, so advertising one would only cost a failed switch.

    Both index layouts count: a Modular Diffusers pipeline (a dense MiniMax-H3) carries
    ``modular_model_index.json`` instead, and the video loader opens either.
    """
    from core.inference.diffusion import resolve_local_single_file

    try:
        if any(
            (load_dir / name).is_file() for name in ("model_index.json", "modular_model_index.json")
        ):
            return True
    except OSError:
        return False
    # a sole checkpoint is reinterpreted as a single_file load, which resolves the name through
    # the same containment check a gguf goes through, so a cache snapshot's symlink is refused
    sole = resolve_local_single_file(str(load_dir))
    return sole is not None and _loader_can_open(str(load_dir), sole)


def _build_index(task: str) -> dict[str, MediaModelPick]:
    """Map every name a downloaded *task* model answers to onto its load spec."""
    from routes.models import _local_model_task, collect_local_models

    index: dict[str, MediaModelPick] = {}
    try:
        candidates = collect_local_models(Path("./models").resolve())
    except Exception as exc:  # noqa: BLE001 -- a failed scan must not 500 the generation
        logger.debug("media auto-switch: local model scan failed: %s", exc)
        return index
    for info in candidates:
        try:
            # a cancelled or incomplete pull still lists, and loading it fails predictably
            if getattr(info, "partial", False):
                continue
            if _local_model_task(info) != task:
                continue
            keys = _name_keys(info)
            if not keys:
                continue
            # an hf cache repo keeps its weights, and its model_index.json, under snapshots/<sha>
            on_disk = Path(info.path).expanduser()
            load_dir = _resolve_load_dir(on_disk)
            if _add_gguf_picks(index, info, keys, on_disk, load_dir):
                continue
            if not _loadable_directory(load_dir):
                continue
            _register(index, keys, MediaModelPick(keys[0], str(load_dir)))
        except Exception as exc:  # noqa: BLE001 -- one unreadable model must not hide the rest
            logger.debug("media auto-switch: skipped %s: %s", getattr(info, "id", "?"), exc)
    return index


def _mark_ambiguous_builds(index: dict[str, MediaModelPick]) -> dict[str, MediaModelPick]:
    """Flag every GGUF pick whose published token another build under its path also publishes.

    A token no build shares identifies its pick, including the empty one an unlabelled file
    publishes: the backend publishes the same empty token for it, so the resident model can
    still be recognised. Only a token two builds answer to is ambiguous, and marking a lone
    build would reload a multi-GB pipeline on every request naming it.
    """
    seen: dict[tuple[str, str], set] = {}
    for pick in index.values():
        if pick is _AMBIGUOUS or pick.model_kind != "gguf":
            continue
        key = (identity_key(pick.model_path), published_token(pick))
        seen.setdefault(key, set()).add(pick.gguf_filename)
    collides = {key for key, files in seen.items() if len(files) > 1}
    if not collides:
        return index
    return {
        name: (
            pick
            if pick is _AMBIGUOUS
            or pick.model_kind != "gguf"
            or (identity_key(pick.model_path), published_token(pick)) not in collides
            else replace(pick, ambiguous = True)
        )
        for name, pick in index.items()
    }


def _cached_index(task: str) -> dict[str, MediaModelPick]:
    now = time.monotonic()
    with _index_lock:
        hit = _index.get(task)
        if hit is not None and now - hit[0] < _INDEX_TTL_S:
            return hit[1]
    built = _mark_ambiguous_builds(_build_index(task))
    with _index_lock:
        # stamped after the scan, so one slower than the ttl is not already expired
        _index[task] = (time.monotonic(), built)
    return built


def invalidate_index() -> None:
    """Drop the cached scan. For tests and anything that changes what is downloaded."""
    with _index_lock:
        _index.clear()


def resolve_local_media_model(name: str, *, task: str) -> Optional[MediaModelPick]:
    """The downloaded *task* model *name* refers to, or None."""
    if not isinstance(name, str) or not name.strip():
        return None
    pick = _cached_index(task).get(name.strip().lower())
    return None if pick is _AMBIGUOUS else pick


def available_media_model_ids(task: str) -> list[str]:
    """Sorted ids a request may name for *task*, for a "not found" error to list."""
    return sorted(
        {pick.model_id for pick in _cached_index(task).values() if pick is not _AMBIGUOUS}
    )


# ── recognising the resident model ──────────────────────────────────


def published_token(pick: MediaModelPick) -> str:
    """The ``gguf_variant`` the backend will publish once *pick* is loaded, lowercased."""
    from hub.utils.gguf import extract_quant_token

    if not pick.gguf_filename:
        return ""
    token = extract_quant_token(pick.gguf_filename)
    return (token or "").strip().lower()


def identity_key(value: str) -> str:
    """A model identity normalized for comparison: a repo id folds case, a path does not."""
    text = str(value or "").strip()
    return os.path.normcase(text) if os.path.isabs(text) else text.lower()


def same_identity(requested: str, resident: str) -> bool:
    """Whether two model identities name the same thing.

    A repo id folds case; a filesystem path does not, since /models/Foo and /models/foo are
    different models where the filesystem says so.
    """
    requested, resident = requested.strip(), resident.strip()
    if not requested or not resident:
        return False
    return identity_key(requested) == identity_key(resident)


def resident_is_gguf(status: dict[str, Any]) -> bool:
    """Whether the resident build is a GGUF, however its engine says so.

    The native sd.cpp status publishes ``dtype="gguf"`` and a quant but no ``model_kind``, so a
    model_kind test alone reads every native checkpoint as a plain pipeline.
    """
    return (
        status.get("model_kind") == "gguf"
        or str(status.get("dtype") or "").strip().lower() == "gguf"
        or bool(status.get("gguf_variant"))
    )


def resident_is_pick(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """Whether the resident build is the one *pick* names, on the identity status publishes.

    A modular MiniMax-H3 build is its partition too: an auto-load of this name selects the
    default keyframe denoiser, so a resident ``ref2va`` does not answer for it.
    """
    if not status.get("loaded"):
        return False
    resident = str(status.get("repo_id") or "").strip().lower()
    if not resident:
        return False
    aliases = {name.strip().lower(), pick.model_id.strip().lower()}
    # not case-folded: /models/Foo and /models/foo are different models where the filesystem says so
    same_path = os.path.normcase(str(status.get("repo_id") or "").strip()) == os.path.normcase(
        pick.model_path.strip()
    )
    if resident not in aliases and not same_path:
        return False
    if not partition_matches(status, pick):
        return False
    if pick.model_kind != "gguf" and not resident_is_gguf(status):
        return True
    loaded_quant = str(status.get("gguf_variant") or "").strip().lower()
    return loaded_quant == published_token(pick)


def satisfied_by(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """Whether the resident model already answers this request.

    Matched on the requested name AND the pick's on-disk path: a model loaded from the Images
    page reports its repo id while one loaded here reports the local path it was given, and
    either has to count as already serving or every request reswaps. Never on ``base_repo``,
    which is a companion encoder/VAE repo and would answer a request for that full pipeline
    with whichever GGUF happens to borrow it.

    A GGUF also has to match on quant. Loose ``.gguf`` files in one scan folder share that
    folder as their ``model_path``, so the path alone would report a sibling as already
    serving and generate on the wrong weights.

    The comparison uses the token the backend actually publishes. Where that token cannot tell
    two indexed builds apart (``IQ4_XS-3.53bpw`` and ``-3.97bpw`` both publish ``IQ4_XS``), the
    pick is marked ambiguous at index time and this answers False: reloading costs a load,
    serving the sibling returns the wrong image.
    """
    if not resident_is_pick(status, name, pick):
        return False
    # ambiguity only blocks the skip, never the "did my load land" check: the reload settles it
    return not pick.ambiguous


def expected_partition(pick: MediaModelPick) -> Optional[str]:
    """The MiniMax-H3 partition this pick will come up on, or None when it is not an H3 model.

    Sent with the load so the recorded provenance matches what status publishes: a GGUF takes
    the partition its filename names, and a modular pipeline takes the keyframe default.
    """
    try:
        from core.inference.video_families import detect_video_family
        from core.inference.video_minimax_h3 import H3_TASK_KEYFRAMES, h3_transformer_task
    except Exception:  # noqa: BLE001 -- no h3 support here means no partition to name
        return None
    # the basename, since a qualified variant lives at ref2va/minimax_h3_ref2va-*.gguf
    name = Path(pick.gguf_filename or "").name.lower()
    if name.startswith("minimax_h3_"):
        return h3_transformer_task(name)
    try:
        # keyed on the family: a modular pipeline resolves to a directory, not a bundle repo id
        for needle in (pick.model_id, pick.model_path):
            fam = detect_video_family(needle) if needle else None
            if fam is not None and getattr(fam, "name", "") == _H3_FAMILY:
                return H3_TASK_KEYFRAMES
    except Exception:  # noqa: BLE001 -- a probe failure must not name a partition
        return None
    return None


def partition_matches(status: dict[str, Any], pick: Optional[MediaModelPick] = None) -> bool:
    """Whether the resident MiniMax-H3 partition is the one this pick would bring up.

    Derived from the checkpoint, not assumed: the native backend publishes ``ref2va`` for a
    ``minimax_h3_ref2va`` denoiser, so hardcoding the keyframe default rejected the very
    checkpoint that had just loaded. Absent a filename the switch sends no ``h3_task`` and the
    load takes the family default.
    """
    resident = str(status.get("h3_task") or "").strip().lower()
    if not resident:
        return True
    try:
        from core.inference.video_minimax_h3 import H3_TASK_KEYFRAMES, h3_transformer_task
    except Exception:  # noqa: BLE001 -- no h3 support here means nothing to compare
        return True
    filename = (pick.gguf_filename if pick else None) or ""
    expected = h3_transformer_task(filename) if filename else H3_TASK_KEYFRAMES
    return resident == str(expected or "").strip().lower()


__all__ = [
    "IMAGE_TASK",
    "VIDEO_TASK",
    "MediaModelPick",
    "available_media_model_ids",
    "expected_partition",
    "identity_key",
    "invalidate_index",
    "partition_matches",
    "published_token",
    "resident_is_gguf",
    "resident_is_pick",
    "resolve_local_media_model",
    "same_identity",
    "satisfied_by",
]
