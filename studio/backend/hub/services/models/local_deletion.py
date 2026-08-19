# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deleting a model Studio only DISCOVERED: Ollama, LM Studio, a scan folder, ``./models``.

The HF cache has its own delete (:mod:`hub.services.models.deletion`) because a repo there is a
refcounted blob store with revisions and a manifest to walk. These sources have nothing of the
kind. An Ollama tag is a small manifest file pointing at content-addressed blobs that several
other tags may share, plus the ``.gguf`` symlink Studio itself materialized to load it; everything
else is a directory some other tool laid down. So unlinking the row's ``path`` is never the whole
job -- it either strands support files (the tokenizer, the config, Studio's own link) or, for
Ollama, orphans multi-GB blobs no manifest names any more.

Every delete is planned first (:func:`plan_local_delete`) and the same plan backs both the preview
the confirm dialog shows and the removal that runs, so what the user is told is what happens.

Three things bound what a plan may remove, because the target arrives from a client:

* it must sit strictly beneath a root Studio actually scans (never the root itself),
* it must still look like a model on disk, not merely be spelled like a path under that root,
* and nothing may be holding it open for inference.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from fastapi import HTTPException
from loggers import get_logger

from hub.services.models import ollama
from hub.utils.paths import (
    lmstudio_model_dirs,
    normalize_path,
    ollama_model_dirs,
    path_is_same_or_child,
)

logger = get_logger(__name__)

_LOAD_STATE_UNVERIFIABLE_DETAIL = (
    "Couldn't verify whether this model is still loaded for inference. "
    "Unload it if it is active, then try deleting again."
)

# Suffixes that make a loose file a model in its own right. A directory is judged by the scan's own
# model-shape test instead; this list only covers the standalone-file rows (LM Studio publishes
# bare .gguf files, and ./models can hold one).
_MODEL_FILE_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt"})


@dataclass
class LocalDeletePlan:
    """What a local delete would remove, keep, and why it might not run at all.

    ``blocked_by`` is the whole refusal story: a non-empty list means the plan is a preview only.
    The delete path re-derives the plan immediately before unlinking rather than trusting one the
    client is acting on, which can be minutes old.
    """

    load_id: str
    target: Optional[Path] = None
    source: str = "unknown"
    display_name: str = ""
    # Removed outright: the model directory or file, the Ollama manifest, blobs nothing else
    # references, and the .gguf links Studio materialized for it.
    removals: list[Path] = field(default_factory = list)
    # Directories to collapse upward afterwards when the removals leave them empty, each paired
    # with the root the walk must stop at.
    prune: list[tuple[Path, Path]] = field(default_factory = list)
    reclaimed_bytes: int = 0
    # Ollama blobs kept because another tag still names them, and the tags keeping them.
    retained_bytes: int = 0
    retained_for: list[str] = field(default_factory = list)
    blocked_by: list[str] = field(default_factory = list)
    notes: list[str] = field(default_factory = list)

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_by)


def _safe_realpath(path: Path) -> Optional[Path]:
    try:
        return Path(os.path.realpath(str(path)))
    except (OSError, ValueError):
        return None


def _same_path(left: Path, right: Path) -> bool:
    left_real = _safe_realpath(left)
    right_real = _safe_realpath(right)
    if left_real is None or right_real is None:
        return False
    return os.path.normcase(str(left_real)) == os.path.normcase(str(right_real))


def _dir_size_bytes(path: Path) -> int:
    """Bytes *path* occupies, counting each file once and never following a symlink out of it.

    Symlinked entries contribute nothing: the delete unlinks the link and leaves the bytes it
    points at, so counting the destination would promise disk space back that never returns.
    """
    if path.is_symlink():
        return 0
    try:
        if path.is_file():
            return max(0, path.stat().st_size)
    except OSError:
        return 0
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for dirpath, _dirnames, filenames in os.walk(str(path), followlinks = False):
        # Hardlinked blobs (Ollama's link dir falls back to os.link) would otherwise be counted
        # once per name and overstate what the delete gives back.
        for name in filenames:
            entry = Path(dirpath) / name
            try:
                if entry.is_symlink():
                    continue
                stat = entry.stat()
            except OSError:
                continue
            if stat.st_nlink > 1:
                key = (stat.st_dev, stat.st_ino)
                if key in seen_inodes:
                    continue
                seen_inodes.add(key)
            total += max(0, stat.st_size)
    return total


def _file_size_bytes(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        return max(0, path.stat().st_size)
    except OSError:
        return 0


def local_delete_roots() -> list[Path]:
    """Roots a local delete may act inside: exactly the directories the inventory scans.

    Anything outside these is not a model Studio discovered, so a target under none of them is
    refused however well-formed it looks. Failures here are not silently swallowed -- an empty
    root list would refuse every delete, which is the safe direction.
    """
    roots: list[Path] = []
    for candidate in (*lmstudio_model_dirs(), *ollama_model_dirs()):
        roots.append(candidate)
    try:
        roots.append(Path("./models").resolve())
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug("Could not resolve the default models directory: %s", e)
    try:
        from hub.storage.scan_folders import list_scan_folders
        for folder in list_scan_folders():
            raw = str(folder.get("path") or "").strip()
            if raw:
                roots.append(Path(normalize_path(raw)).expanduser())
    except Exception as e:  # noqa: BLE001 -- an unreadable registry narrows the roots, never widens
        logger.warning("Could not read scan folders while planning a delete: %s", e)
    return roots


def _owning_root(target: Path, roots: Sequence[Path]) -> Optional[Path]:
    """The deepest scanned root *target* lives strictly beneath, or None.

    Deepest, not first: a scan folder nested inside another one must bound the prune walk at
    itself, or collapsing empty parents would eat the inner folder's own registration.
    """
    owning: Optional[Path] = None
    owning_depth = -1
    for root in roots:
        if not path_is_same_or_child(target, root):
            continue
        if _same_path(target, root):
            continue
        real = _safe_realpath(root)
        depth = len(real.parts) if real is not None else 0
        if depth > owning_depth:
            owning, owning_depth = root, depth
    return owning


def _looks_like_a_model(path: Path) -> bool:
    """Whether *path* still reads as a model on disk.

    The root check alone would let any path spelled under a scan folder through, so this is the
    second half of the authorization: a delete can only take something the scan itself would have
    offered as a row.
    """
    try:
        if path.is_file():
            return path.suffix.lower() in _MODEL_FILE_SUFFIXES
        if not path.is_dir():
            return False
    except OSError:
        return False
    from hub.services.models.local_inventory import (
        _has_immediate_model_signal,
        _is_diffusers_pipeline_dir,
    )

    if _is_diffusers_pipeline_dir(path):
        return True
    if _has_immediate_model_signal(path):
        return True
    # A GGUF folder carries no config.json, so the signal probe above misses it.
    try:
        return any(entry.is_file() and entry.suffix.lower() == ".gguf" for entry in path.iterdir())
    except OSError:
        return False


def _loaded_identifiers() -> list[str]:
    """Every model identifier a backend currently holds or is fetching.

    Acquiring a backend fails open (an import error means nothing of the sort is loaded); reading
    one is deliberately unguarded so a raise propagates and the caller refuses the delete rather
    than unlinking weights out from under a live process. Mirrors the HF-cache guards in
    :mod:`hub.services.models.deletion`, but collects identifiers instead of matching a repo id:
    these rows are paths, and an Ollama row is loaded under a link path that matches neither its
    manifest nor its blob by name.
    """
    identifiers: list[str] = []

    def _extend(values: Iterable[object]) -> None:
        for value in values:
            if value:
                identifiers.append(str(value))

    try:
        from routes.inference import get_llama_cpp_backend
        backend = get_llama_cpp_backend()
    except Exception as e:  # noqa: BLE001 -- see the fail-open contract above
        logger.debug("llama.cpp backend unavailable during local delete guard: %s", e)
    else:
        if backend.is_loaded or backend.is_active:
            _extend([backend.model_identifier])

    try:
        from core.inference.orchestrator import peek_inference_backend

        # Peek, never construct: building one just to learn nothing is loaded imports torch.
        inference_backend = peek_inference_backend()
    except Exception as e:  # noqa: BLE001
        logger.debug("Inference backend unavailable during local delete guard: %s", e)
    else:
        if inference_backend is not None:
            _extend([inference_backend.active_model_name])

    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        engine = get_active_diffusion_engine()
    except Exception as e:  # noqa: BLE001
        logger.debug("Diffusion engine unavailable during local delete guard: %s", e)
    else:
        status = engine.status()
        if status.get("loaded"):
            _extend([status.get("repo_id")])
        _extend(getattr(engine, "loaded_repo_ids", tuple)())
        _extend(getattr(engine, "loading_repo_ids", tuple)())

    try:
        from core.inference.video import get_video_backend
        video_backend = get_video_backend()
    except Exception as e:  # noqa: BLE001
        logger.debug("Video backend unavailable during local delete guard: %s", e)
    else:
        status = video_backend.status()
        if status.get("loaded"):
            _extend([status.get("repo_id"), status.get("base_repo")])
        _extend(getattr(video_backend, "loaded_repo_ids", tuple)())
        _extend(getattr(video_backend, "loading_repo_ids", tuple)())

    return identifiers


def _load_state_blocks(paths: Sequence[Path]) -> bool:
    """Whether a backend holds anything at or beneath one of *paths*.

    Compared as real paths, which is what makes the Ollama case work: the loaded identifier is the
    ``.studio_links`` symlink, and resolving it lands on the very blob this delete would remove.
    """
    if not paths:
        return False
    for loaded_id in _loaded_identifiers():
        try:
            loaded_path = Path(loaded_id).expanduser()
        except (OSError, RuntimeError, ValueError):
            continue
        for target in paths:
            if path_is_same_or_child(loaded_path, target):
                return True
    return False


def _ollama_tag_label(rel: Path) -> str:
    """``llama3:8b`` from a manifest path relative to ``manifests/`` (``host/ns/name/tag``)."""
    parts = rel.parts
    if len(parts) < 3:
        return rel.as_posix()
    namespace = list(parts[1:-1])
    # registry.ollama.ai/library/<name> is the default namespace nobody types. Kept when
    # dropping it would leave no name at all, so a short path still labels as something.
    if parts[0] == "registry.ollama.ai" and namespace[:1] == ["library"] and len(namespace) > 1:
        namespace = namespace[1:]
    return f"{'/'.join(namespace)}:{parts[-1]}"


def _plan_ollama(load_id: str, tag_file: Path) -> LocalDeletePlan:
    """Plan an Ollama tag delete: the manifest, its unshared blobs, and Studio's own links.

    Blobs are content-addressed and shared, so what actually comes back is decided by counting
    references across every other manifest in the same root. That count fails CLOSED: one manifest
    this cannot parse keeps every blob, because the alternative is deleting weights a tag Studio
    could not read still needs. The tag itself always goes, so the row leaves the inventory either
    way and the kept bytes are reported rather than quietly dropped.
    """
    plan = LocalDeletePlan(load_id = load_id, source = "ollama")

    ollama_dir = ollama.ollama_dir_for_manifest(tag_file)
    if ollama_dir is None:
        plan.blocked_by.append("This model is not inside a known Ollama models directory.")
        return plan

    manifests_root = ollama_dir / "manifests"
    try:
        rel = tag_file.relative_to(manifests_root)
    except ValueError:
        # Reachable when the reference resolves into the root through a symlink: same-or-child
        # says yes on real paths while relative_to works on the spelled one.
        real_tag = _safe_realpath(tag_file)
        real_root = _safe_realpath(manifests_root)
        if real_tag is None or real_root is None:
            plan.blocked_by.append("Couldn't resolve this model's Ollama manifest.")
            return plan
        try:
            rel = real_tag.relative_to(real_root)
        except ValueError:
            plan.blocked_by.append("Couldn't resolve this model's Ollama manifest.")
            return plan
        tag_file = real_tag

    if not tag_file.is_file():
        plan.blocked_by.append("This model's Ollama manifest is already gone.")
        return plan

    plan.target = tag_file
    plan.display_name = _ollama_tag_label(rel)

    own_digests = ollama.ollama_manifest_blob_digests(tag_file)
    if own_digests is None:
        plan.blocked_by.append("Couldn't read this model's Ollama manifest.")
        return plan

    # Which OTHER tag keeps each digest alive, not merely that one does: "kept because llama3:8b
    # still uses it" is the difference between a number the user can act on and one they cannot.
    sharers: dict[str, set[str]] = {}
    unreadable: list[str] = []
    for other in ollama.iter_ollama_manifest_files(manifests_root):
        if _same_path(other, tag_file):
            continue
        try:
            label = _ollama_tag_label(other.relative_to(manifests_root))
        except ValueError:
            label = other.name
        digests = ollama.ollama_manifest_blob_digests(other)
        if digests is None:
            unreadable.append(label)
            continue
        for digest in digests:
            sharers.setdefault(digest, set()).add(label)

    plan.removals.append(tag_file)
    plan.reclaimed_bytes += _file_size_bytes(tag_file)
    plan.prune.append((tag_file.parent, manifests_root))

    retained_for: set[str] = set()
    for digest in sorted(own_digests):
        blob = ollama.ollama_blob_path(ollama_dir, digest)
        if blob is None or not blob.is_file():
            continue
        size = _file_size_bytes(blob)
        if digest in sharers:
            plan.retained_bytes += size
            retained_for |= sharers[digest]
            continue
        if unreadable:
            # Fail closed: an unparsed manifest may be the one that needs this blob.
            plan.retained_bytes += size
            continue
        plan.removals.append(blob)
        plan.reclaimed_bytes += size
    plan.retained_for = sorted(retained_for)

    if unreadable:
        shown = ", ".join(sorted(unreadable)[:3])
        noun = "manifest" if len(unreadable) == 1 else "manifests"
        plan.notes.append(
            f"{len(unreadable)} Ollama {noun} could not be read ({shown}), so the shared blobs "
            "they might need are being kept. The model itself is still removed."
        )

    # The .gguf links Studio materialized to load this tag -- the leftovers a plain
    # `ollama rm` would never know about, and the ones this delete exists to collect.
    stem_hash = ollama.ollama_manifest_stem_hash(rel)
    for links_root in ollama.ollama_links_root_candidates(ollama_dir):
        link_dir = links_root / stem_hash
        if not link_dir.is_dir():
            continue
        plan.removals.append(link_dir)
        # Links point back at blobs already priced above, so they add no reclaimable bytes.
        plan.prune.append((link_dir.parent, links_root.parent))

    return plan


def _plan_path(load_id: str, target: Path, source: str) -> LocalDeletePlan:
    """Plan a directory-or-file delete for a model discovered on disk."""
    plan = LocalDeletePlan(load_id = load_id, source = source)

    # Before the root check, because a shortcut pointing out of the scanned folder fails
    # containment and would otherwise be reported as the wrong problem.
    if target.is_symlink():
        destination = _safe_realpath(target)
        plan.blocked_by.append(
            "This model is a shortcut to "
            f"{destination if destination is not None else 'another location'}, so removing it "
            "here would free nothing. Delete the folder it points at instead."
        )
        return plan

    root = _owning_root(target, local_delete_roots())
    if root is None:
        plan.blocked_by.append(
            "This model is not inside a folder Studio scans, so it can only be removed outside "
            "the app."
        )
        return plan

    if not target.exists():
        plan.blocked_by.append("This model is already gone from disk.")
        return plan

    if not _looks_like_a_model(target):
        plan.blocked_by.append("This path no longer holds a model, so Studio will not delete it.")
        return plan

    plan.target = target
    plan.display_name = target.name
    plan.removals.append(target)
    plan.reclaimed_bytes = _file_size_bytes(target) if target.is_file() else _dir_size_bytes(target)
    # An LM Studio publisher folder left holding nothing, or the `<name>/` above a bare .gguf, is
    # the "filepath left behind" the whole feature is about -- so collapse upward to the root.
    plan.prune.append((target.parent, root))
    return plan


def plan_local_delete(load_id: str, source: Optional[str] = None) -> LocalDeletePlan:
    """Resolve *load_id* to a removal plan without touching anything.

    *source* is the inventory row's own label and is used only to describe the plan; what may be
    removed is derived from the identifier and the scanned roots, never from what a client claims
    the row was.
    """
    identifier = (load_id or "").strip()
    if not identifier:
        raise HTTPException(status_code = 400, detail = "load_id is required")

    if ollama.is_ollama_manifest_ref(identifier):
        tag_file = ollama.ollama_manifest_ref_tag_file(identifier)
        if tag_file is None:
            raise HTTPException(status_code = 400, detail = "Invalid Ollama model reference")
        return _plan_ollama(identifier, tag_file)

    if source == "hf_cache":
        raise HTTPException(
            status_code = 400,
            detail = "Cached Hugging Face models are removed through the cached-model delete.",
        )

    try:
        target = Path(normalize_path(identifier)).expanduser()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code = 400, detail = "Invalid model path")
    if not target.is_absolute():
        raise HTTPException(status_code = 400, detail = "Model path must be absolute")
    return _plan_path(identifier, target, source or "unknown")


def _prune_empty_dirs(start: Path, root: Path, removed: list[str]) -> None:
    """Collapse *start* and its parents while they are empty, stopping below *root*.

    Containment is rechecked on every step rather than once, so a directory that turns out to sit
    outside the root -- through a symlink, or because the walk climbed past it -- ends the walk
    instead of being removed.
    """
    current = start
    while True:
        if _same_path(current, root) or not path_is_same_or_child(current, root):
            return
        try:
            if next(current.iterdir(), None) is not None:
                return
            current.rmdir()
        except OSError:
            return
        removed.append(str(current))
        current = current.parent


def _execute(plan: LocalDeletePlan) -> list[str]:
    """Carry out *plan*, returning what actually left the disk.

    Ordering matters for Ollama: the manifest goes first, so a crash midway leaves orphaned blobs
    (which the next delete of any sibling tag reclaims) rather than a manifest pointing at blobs
    that are no longer there, which would read as a working model that cannot load.
    """
    removed: list[str] = []
    for path in plan.removals:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
        except OSError as e:
            logger.warning("Could not remove %s while deleting %s: %s", path, plan.load_id, e)
            continue
        removed.append(str(path))

    for start, root in plan.prune:
        _prune_empty_dirs(start, root, removed)
    return removed


def _refresh_inventory_after_delete() -> None:
    """Drop the caches that would keep serving the row that was just removed."""
    try:
        from hub.services.models import cache_inventory
        cache_inventory.invalidate_hf_cache_scans()
    except Exception as e:  # noqa: BLE001 -- a stale listing is not worth failing a done delete
        logger.warning("Could not invalidate inventory scans after a local delete: %s", e)
    try:
        from core.inference.local_model_resolver import invalidate_index, warm_index_soon
        invalidate_index()
        warm_index_soon()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not refresh the local model index after a delete: %s", e)


def _guard_paths(plan: LocalDeletePlan) -> list[Path]:
    paths = list(plan.removals)
    if plan.target is not None:
        paths.append(plan.target)
    return paths


def _blocked_by_load_state(plan: LocalDeletePlan) -> bool:
    """Whether something is holding this model open, refusing the delete if that can't be read."""
    try:
        return _load_state_blocks(_guard_paths(plan))
    except Exception as e:
        logger.warning(
            "Load-state verification failed for %s; refusing delete: %s", plan.load_id, e
        )
        raise HTTPException(status_code = 503, detail = _LOAD_STATE_UNVERIFIABLE_DETAIL)


def local_delete_impact_blocking(load_id: str, source: Optional[str] = None) -> dict:
    plan = plan_local_delete(load_id, source)
    if not plan.blocked and _blocked_by_load_state(plan):
        plan.blocked_by.append("Unload the model before deleting")
    return {
        "load_id": plan.load_id,
        "source": plan.source,
        "target_path": str(plan.target) if plan.target is not None else None,
        "display_name": plan.display_name,
        "reclaimed_bytes": plan.reclaimed_bytes,
        "retained_bytes": plan.retained_bytes,
        "retained_for": plan.retained_for,
        "removed_paths": [str(path) for path in plan.removals],
        "blocked_by": plan.blocked_by,
        "notes": plan.notes,
    }


def delete_local_model_blocking(load_id: str, source: Optional[str] = None) -> dict:
    # Re-planned here rather than trusting a preview the client may have been sitting on for
    # minutes: an `ollama pull` in between can point another tag at these blobs, and a folder can
    # stop being a model. The plan the guards run against is the plan that executes.
    plan = plan_local_delete(load_id, source)
    if plan.blocked:
        raise HTTPException(status_code = 400, detail = plan.blocked_by[0])
    if _blocked_by_load_state(plan):
        raise HTTPException(status_code = 400, detail = "Unload the model before deleting")

    logger.info(
        "Deleting local model %s (%s): %d path(s), ~%d bytes reclaimed",
        plan.display_name or plan.load_id,
        plan.source,
        len(plan.removals),
        plan.reclaimed_bytes,
    )
    try:
        removed = _execute(plan)
    finally:
        _refresh_inventory_after_delete()

    if not removed:
        raise HTTPException(status_code = 500, detail = "Nothing could be removed for this model")
    return {
        "status": "deleted",
        "load_id": plan.load_id,
        "source": plan.source,
        "display_name": plan.display_name,
        "freed_bytes": plan.reclaimed_bytes,
        "retained_bytes": plan.retained_bytes,
        "removed_paths": removed,
        "notes": plan.notes,
    }


async def local_delete_impact_response(load_id: str, source: Optional[str] = None) -> dict:
    """Preview a local delete. Filesystem walks run off the event loop."""
    return await asyncio.to_thread(local_delete_impact_blocking, load_id, source)


async def delete_local_model_response(load_id: str, source: Optional[str] = None) -> dict:
    return await asyncio.to_thread(delete_local_model_blocking, load_id, source)
