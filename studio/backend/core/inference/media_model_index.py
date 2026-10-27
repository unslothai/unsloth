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
_index: dict[tuple[str, str], tuple[float, dict[str, "MediaModelPick"]]] = {}

# the video family whose partitions are a load-time choice, not a property of the files
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


def _resolve_load_dir(p: Path) -> Path:
    """Reuses the chat resolver so both surfaces map a cached repo to the same local directory."""
    from core.inference.local_model_resolver import _resolve_load_dir as _chat_resolve
    return Path(_chat_resolve(p))


def _register(index: dict[str, MediaModelPick], keys, pick: MediaModelPick) -> None:
    """Drops names that two models share, since a repo's final component collides across orgs."""
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
    """Absolute paths are excluded: a host path is not something an API caller should have to send."""
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
    """An HF cache repo loads by repo id, since snapshot symlinks into blobs/ are refused by the loader."""
    repo_id = getattr(info, "model_id", None)
    if load_dir != on_disk and isinstance(repo_id, str) and repo_id:
        return repo_id
    return str(load_dir)


def _loader_can_open(load_path: str, filename: str) -> bool:
    """A split checkpoint needs its whole shard set beside it, or the load fails after evicting."""
    from utils.models.model_config import colocated_split_shards

    root = Path(load_path)
    if not root.is_dir():
        # Cached repo id skips containment but needs a whole split set; uncached is the download guard's.
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
    Only the checkpoints ``default_gguf_scope`` names are ranked (the root ones when there are
    any, or a curated repo's default subdirectory such as unsloth/LTX-2.5-GGUF's distilled/), so
    one id cannot mean different weights here than in the picker.
    """
    from core.inference.openai_auto_download import preferred_quant
    from hub.utils.gguf import default_gguf_scope, gguf_variant_family, gguf_variant_key
    from hub.utils.gguf import list_local_gguf_variants as list_build_variants
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
    variants, _ = list_build_variants(str(load_dir))
    legacy_variants, _ = list_local_gguf_variants(str(load_dir))
    legacy_quants = {variant.filename: variant.quant for variant in legacy_variants}
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
    primary_keys: dict[str, str] = {}
    for quant, variant in openable.items():
        # model_id stays the bare id so a "not found" error lists models, not one row per quant
        names = [f"{key}:{quant}" for key in keys]
        primary_keys.update((name.strip().lower(), variant.filename) for name in names)
        _register(
            index,
            names,
            MediaModelPick(keys[0], load_path, variant.filename, "gguf"),
        )
    for quant, variant in openable.items():
        legacy_quant = legacy_quants.get(variant.filename)
        if legacy_quant and legacy_quant.lower() != quant.lower():
            names = [
                name
                for name in (f"{key}:{legacy_quant}" for key in keys)
                if primary_keys.get(name.strip().lower(), variant.filename) == variant.filename
            ]
            _register(
                index,
                names,
                MediaModelPick(keys[0], load_path, variant.filename, "gguf"),
            )
    # a build's stem or filename key keeps naming it after a sibling's removal re-keys it
    claims: dict[str, list[str]] = {}
    for quant, variant in openable.items():
        relative = variant.filename.replace("\\", "/")
        for spelling in {gguf_variant_family(relative), relative}:
            if spelling and spelling.lower() != quant.lower():
                claims.setdefault(spelling.lower(), []).append(variant.filename)
    for spelling, files in claims.items():
        if len(set(files)) != 1:
            continue
        names = [
            name
            for name in (f"{key}:{spelling}" for key in keys)
            if primary_keys.get(name.strip().lower(), files[0]) == files[0]
        ]
        _register(index, names, MediaModelPick(keys[0], load_path, files[0], "gguf"))
    scope = default_gguf_scope(list(openable), *keys, getattr(info, "id", None), str(on_disk))
    best = None
    legacy_groups: dict[str, int] = {}
    for variant in openable.values():
        legacy = gguf_variant_key(variant.filename).lower()
        legacy_groups[legacy] = legacy_groups.get(legacy, 0) + 1
    try:
        root = Path(load_path)
        if root.is_dir():
            from utils.models.model_config import detect_gguf_model

            detected = detect_gguf_model(load_path)
            relative = Path(detected).relative_to(root).as_posix() if detected else None
            preferred = preferred_quant(scope) or scope[0]
            preferred_group = gguf_variant_key(openable[preferred].filename).lower()
            # detection only breaks ties between builds of the preferred quant
            selected = next(
                (
                    quant
                    for quant, variant in openable.items()
                    if variant.filename == relative
                    and gguf_variant_key(variant.filename).lower() == preferred_group
                    and legacy_groups.get(preferred_group, 0) > 1
                ),
                None,
            )
            if selected in scope:
                best = selected
    except (OSError, RuntimeError, ValueError, KeyError):
        pass
    best = best or preferred_quant(scope) or scope[0]
    _register(
        index,
        keys,
        MediaModelPick(keys[0], load_path, openable[best].filename, "gguf"),
    )
    return True


def _loadable_directory(load_dir: Path) -> bool:
    """Only a full pipeline or one checkpoint is loadable: several with no index is ambiguous."""
    from core.inference.diffusion import resolve_local_single_file

    try:
        if any(
            (load_dir / name).is_file() for name in ("model_index.json", "modular_model_index.json")
        ):
            return True
    except OSError:
        return False
    # A sole checkpoint loads as single_file, so the gguf containment check refuses a snapshot symlink.
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
            if load_dir.is_file():
                if load_dir.suffix.lower() == ".safetensors" and _loader_can_open(
                    str(load_dir.parent), load_dir.name
                ):
                    _register(
                        index,
                        keys,
                        MediaModelPick(keys[0], str(load_dir.parent), load_dir.name, "single_file"),
                    )
                continue
            if not _loadable_directory(load_dir):
                continue
            _register(index, keys, MediaModelPick(keys[0], str(load_dir)))
        except Exception as exc:  # noqa: BLE001 -- one unreadable model must not hide the rest
            logger.debug("media auto-switch: skipped %s: %s", getattr(info, "id", "?"), exc)
    return index


def _partition_of(pick: MediaModelPick) -> Optional[str]:
    """The MiniMax-H3 partition this pick brings up, or None when it is not an H3 build."""
    return expected_partition(pick)


def _mark_ambiguous_builds(index: dict[str, MediaModelPick]) -> dict[str, MediaModelPick]:
    """Groups by path and quant; only H3 partitions split a group, since status publishes h3_task."""
    groups: dict[tuple[str, str], list[MediaModelPick]] = {}
    for pick in index.values():
        if pick is _AMBIGUOUS or pick.model_kind != "gguf":
            continue
        groups.setdefault((identity_key(pick.model_path), published_token(pick)), []).append(pick)
    collides = set()
    for key, picks in groups.items():
        files = {pick.gguf_filename for pick in picks}
        if len(files) < 2:
            continue
        partitions = [_partition_of(pick) for pick in picks]
        # every member an H3 build with a partition of its own: status tells them apart
        if all(partitions) and len(set(partitions)) == len(files):
            continue
        collides.add(key)
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
    from utils.account_context import current_account_id

    key = (current_account_id(), task)
    now = time.monotonic()
    with _index_lock:
        hit = _index.get(key)
        if hit is not None and now - hit[0] < _INDEX_TTL_S:
            return hit[1]
    built = _mark_ambiguous_builds(_build_index(task))
    with _index_lock:
        # stamped after the scan, so one slower than the ttl is not already expired
        _index[key] = (time.monotonic(), built)
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
    """Repo ids fold case, but filesystem paths do not, since /models/Foo and /models/foo can differ."""
    requested, resident = requested.strip(), resident.strip()
    if not requested or not resident:
        return False
    return identity_key(requested) == identity_key(resident)


def resident_is_gguf(status: dict[str, Any]) -> bool:
    """Native sd.cpp publishes dtype gguf and no model_kind, so checking model_kind alone misses it."""
    return (
        status.get("model_kind") == "gguf"
        or str(status.get("dtype") or "").strip().lower() == "gguf"
        or bool(status.get("gguf_variant"))
    )


def resident_is_pick(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """A modular MiniMax-H3 build is its own partition: a resident ref2va does not answer for it."""
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
    if pick.model_kind == "single_file" and not resident_is_gguf(status):
        # loose checkpoints in one folder share it as model_path: only the file tells them apart
        return os.path.normcase(str(status.get("gguf_filename") or "")) == os.path.normcase(
            pick.gguf_filename or ""
        )
    if pick.model_kind != "gguf" and not resident_is_gguf(status):
        return True
    resident_filename = str(status.get("gguf_filename") or "").strip()
    if resident_filename:
        return os.path.normcase(resident_filename) == os.path.normcase(pick.gguf_filename or "")
    loaded_quant = str(status.get("gguf_variant") or "").strip().lower()
    return loaded_quant == published_token(pick)


def satisfied_by(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """Matches name and on-disk path, plus quant for GGUFs; never on base_repo, a shared companion repo."""
    if not resident_is_pick(status, name, pick):
        return False
    # only a resident filename can tell same-token builds apart; without one ambiguity blocks the skip
    return not pick.ambiguous or bool(str(status.get("gguf_filename") or "").strip())


def expected_partition(pick: MediaModelPick) -> Optional[str]:
    """Sent with the load so recorded provenance matches the partition status will publish."""
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
    """Derived from the checkpoint: a hardcoded keyframe default rejected a ref2va build just loaded."""
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
