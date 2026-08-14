# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in model auto-switch for the image and video generation APIs.

The chat twin lives in ``local_model_resolver`` + ``routes.inference``: a ``/v1`` request
naming a downloaded GGUF loads it before serving. Media had no equivalent, so
``POST /v1/images/generations`` answered 503 unless someone had already picked a model on the
Images page, and ``model`` was documented as informational. This resolves that name against
the downloaded image/video models, drains what the backend is doing, and runs the load the
picker would run.

Off by default (``media_api_auto_switch_model``), so existing clients see no change.

Only downloaded models resolve, and an unknown name is refused rather than answered by
whatever is resident. Nothing here starts a download: the media equivalent of the chat
auto-download setting would let one API key spend tens of GB, which is its own decision.

Both waits are bounded, because Studio's secure-mode tunnel caps an origin response near 100
seconds. Exceeding a bound leaves the work running and asks the caller to retry, the contract
``begin_load`` already gives the UI.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.inference.gpu_arbiter import DIFFUSION, VIDEO
from loggers import get_logger

logger = get_logger(__name__)

IMAGE_TASK = "text-to-image"
VIDEO_TASK = "text-to-video"

# The scan walks several roots and reads GGUF headers, and this runs per request.
_INDEX_TTL_S = 5.0
_index_lock = threading.Lock()
_index: dict[str, tuple[float, dict[str, "MediaModelPick"]]] = {}

# A generation the caller cannot see must not be cut short, so the swap yields instead.
_DRAIN_WAIT_S = 30.0
# Under the ~100s tunnel window, so a slow load ends in a retryable 503, not a dropped socket.
_LOAD_WAIT_S = 90.0
_POLL_S = 0.2
_RETRY_AFTER_S = 15

_BUSY_MSG = (
    "The {kind} model is busy with another request, so it could not be switched in time. "
    "Retry once the current generation finishes."
)
_LOADING_MSG = (
    "Loading '{model}'. It was not resident when this request arrived and is still coming up; "
    "retry shortly."
)
# Cap on ids a "not found" error lists, so it stays readable in a terminal.
_MAX_LISTED_MODELS = 8


@dataclass(frozen = True)
class MediaModelPick:
    """A downloaded media model, in the shape its load route takes."""

    model_id: str
    model_path: str
    gguf_filename: Optional[str] = None
    model_kind: Optional[str] = None


# ── resolving a name to a downloaded model ──────────────────────────


def _resolve_load_dir(p: Path) -> Path:
    """The directory holding the weights, unwrapping an HF cache repo to its snapshot.

    The chat resolver's helper, reused so both surfaces resolve a cached repo to the same
    local directory rather than to the download-capable repo id.
    """
    from core.inference.local_model_resolver import _resolve_load_dir as _chat_resolve
    return Path(_chat_resolve(p))


def _register(index: dict[str, MediaModelPick], keys, pick: MediaModelPick) -> None:
    for key in keys:
        if isinstance(key, str) and key.strip():
            index.setdefault(key.strip().lower(), pick)


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


def _gguf_load_path(info, load_dir: Path) -> str:
    """What ``/images/load`` takes as ``model_path`` for a GGUF under *info*.

    An HF cache repo is named by its repo id, as the picker names a Hub pick. Its snapshot
    entries are symlinks into ``blobs/``, and the loader's local branch resolves a symlink
    before its containment check, so a snapshot directory refuses its own file. Anything else
    is a real directory and loads by path.
    """
    repo_id = getattr(info, "model_id", None)
    if getattr(info, "source", None) == "hf_cache" and isinstance(repo_id, str) and repo_id:
        return repo_id
    return str(load_dir)


def _add_gguf_picks(
    index: dict[str, MediaModelPick], info, keys: tuple[str, ...], load_dir: Path
) -> bool:
    """Index every GGUF quant under *info*, bare and as ``<id>:<QUANT>``; False if it holds none.

    A bare id means the quant a plain load takes, ranked by the ``preferred_quant`` the chat
    resolver and /v1/models already share, so one id cannot mean different weights per surface.
    """
    from core.inference.openai_auto_download import preferred_quant
    from utils.models.model_config import list_local_gguf_variants

    if load_dir.is_file():
        if load_dir.suffix.lower() != ".gguf":
            return False
        _register(index, keys, MediaModelPick(keys[0], str(load_dir.parent), load_dir.name, "gguf"))
        return True
    # Filenames come back relative to this directory, which is what the loader joins them onto.
    variants, _ = list_local_gguf_variants(str(load_dir))
    by_quant = {v.quant: v for v in variants if v.quant}
    if not by_quant:
        return False
    load_path = _gguf_load_path(info, load_dir)
    for quant, variant in by_quant.items():
        # model_id stays the bare id so a "not found" error lists models, not one row per quant.
        _register(
            index,
            [f"{key}:{quant}" for key in keys],
            MediaModelPick(keys[0], load_path, variant.filename, "gguf"),
        )
    best = preferred_quant(list(by_quant)) or next(iter(by_quant))
    _register(index, keys, MediaModelPick(keys[0], load_path, by_quant[best].filename, "gguf"))
    return True


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
            if _local_model_task(info) != task:
                continue
            keys = _name_keys(info)
            if not keys:
                continue
            # Unwrapped once for both kinds: an HF cache repo keeps its weights, and its
            # model_index.json, one level down under snapshots/<sha>.
            load_dir = _resolve_load_dir(Path(info.path).expanduser())
            if _add_gguf_picks(index, info, keys, load_dir):
                continue
            # Not a GGUF, so the load route detects the kind: a diffusers directory loads as a
            # pipeline, and a bare single-file directory is reinterpreted by the route itself.
            _register(index, keys, MediaModelPick(keys[0], str(load_dir)))
        except Exception as exc:  # noqa: BLE001 -- one unreadable model must not hide the rest
            logger.debug("media auto-switch: skipped %s: %s", getattr(info, "id", "?"), exc)
    return index


def _cached_index(task: str) -> dict[str, MediaModelPick]:
    now = time.monotonic()
    with _index_lock:
        hit = _index.get(task)
        if hit is not None and now - hit[0] < _INDEX_TTL_S:
            return hit[1]
    built = _build_index(task)
    with _index_lock:
        # Stamped after the scan, so one slower than the TTL is not already expired.
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
    return _cached_index(task).get(name.strip().lower())


def available_media_model_ids(task: str) -> list[str]:
    """Sorted ids a request may name for *task*, for a "not found" error to list."""
    return sorted({pick.model_id for pick in _cached_index(task).values()})


# ── the switch ──────────────────────────────────────────────────────


def _format_available(ids: list[str]) -> str:
    if not ids:
        return ""
    shown = ", ".join(ids[:_MAX_LISTED_MODELS])
    extra = len(ids) - _MAX_LISTED_MODELS
    return f"{shown} and {extra} more" if extra > 0 else shown


def _backend_for(owner: str) -> Any:
    if owner == DIFFUSION:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        return get_active_diffusion_engine()
    from core.inference.video import get_video_backend
    return get_video_backend()


def _satisfied_by(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """Whether the resident model already answers this request.

    Matched on the requested name AND the pick's on-disk path: a model loaded from the Images
    page reports its repo id while one loaded here reports the local path it was given, and
    either has to count as already serving or every request reswaps. Never on ``base_repo``,
    which is a companion encoder/VAE repo and would answer a request for that full pipeline
    with whichever GGUF happens to borrow it.
    """
    if not status.get("loaded"):
        return False
    resident = str(status.get("repo_id") or "").strip().lower()
    if not resident:
        return False
    wanted = {name.strip().lower(), pick.model_id.strip().lower(), pick.model_path.strip().lower()}
    if resident not in wanted:
        return False
    # A quant-qualified request needs that quant; a bare one takes whichever is loaded.
    if pick.model_kind == "gguf" and ":" in name:
        loaded_quant = str(status.get("gguf_variant") or "").strip().lower()
        return bool(loaded_quant) and name.strip().lower().endswith(f":{loaded_quant}")
    return True


def _backend_busy(backend: Any) -> bool:
    """One off-loop read of whether a load or generation is running. Mirrors media_keepwarm."""
    if backend.loading_repo_ids():
        return True
    return bool((backend.generate_progress() or {}).get("active"))


async def _drain(owner: str, backend: Any) -> bool:
    """Wait out other tracked requests and any in-flight load or generation."""
    from core.inference.media_keepwarm import other_request_count

    deadline = time.monotonic() + _DRAIN_WAIT_S
    while True:
        # This request is itself tracked, so it must not count itself as other work.
        others = other_request_count(owner, current_request_counted = True)
        if others <= 0 and not await asyncio.to_thread(_backend_busy, backend):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)


async def _await_loaded(backend: Any, pick: MediaModelPick) -> bool:
    """Poll the background load until the model is resident; False if it is still going."""
    deadline = time.monotonic() + _LOAD_WAIT_S
    while True:
        progress = await asyncio.to_thread(backend.load_progress) or {}
        phase = progress.get("phase")
        if phase == "error":
            raise RuntimeError(progress.get("error") or "The model failed to load.")
        if phase in (None, "ready"):
            if (await asyncio.to_thread(backend.status)).get("loaded"):
                return True
            # Nothing in flight and nothing resident: the worker cleared without reporting.
            raise RuntimeError(f"'{pick.model_id}' did not load.")
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)


def _refuse(
    message: str,
    *,
    status_code: int,
    openai_errors: bool,
    code: str,
    retry_after: int = 0,
):
    """The HTTPException to raise, in the error shape the calling route publishes."""
    from fastapi import HTTPException
    from utils.api_errors import openai_error_body

    detail: Any = message
    if openai_errors:
        detail = openai_error_body(message, status = status_code, code = code, param = "model")
    return HTTPException(
        status_code = status_code,
        detail = detail,
        headers = {"Retry-After": str(retry_after)} if retry_after else None,
    )


async def _start_load(owner: str, pick: MediaModelPick, current_subject: str) -> None:
    """Run the load its own route would run, as an API load rather than a user one."""
    if owner == DIFFUSION:
        from models.inference import DiffusionLoadRequest
        from routes.inference import load_diffusion_model_gated
        await load_diffusion_model_gated(
            DiffusionLoadRequest(
                model_path = pick.model_path,
                gguf_filename = pick.gguf_filename,
                model_kind = pick.model_kind,
            ),
            current_subject,
            user_initiated = False,
        )
    else:
        from models.inference import VideoLoadRequest
        from routes.video import load_video_model_gated
        await load_video_model_gated(
            VideoLoadRequest(
                model_path = pick.model_path,
                gguf_filename = pick.gguf_filename,
                model_kind = pick.model_kind,
            ),
            current_subject,
            user_initiated = False,
        )
    logger.info("Media auto-switch: loading %s on the %s backend", pick.model_id, owner)


async def maybe_auto_switch_media_model(
    requested_model: Optional[str], *, owner: str, current_subject: str, openai_errors: bool
) -> None:
    """Load the image or video model a generation request names, if it is not resident.

    No-op when the setting is off or nothing was named, so ``model`` keeps its old
    informational meaning for every existing client. With the setting on, a name that resolves
    to no downloaded model is refused: answering it would return one model's output under
    another's name.
    """
    from utils.openai_auto_switch_settings import get_media_auto_switch_enabled

    if not isinstance(requested_model, str) or not requested_model.strip():
        return
    if not get_media_auto_switch_enabled():
        return

    name = requested_model.strip()
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    kind = "image" if owner == DIFFUSION else "video"
    # Off the loop: a cold index walks the model roots and reads GGUF headers.
    pick = await asyncio.to_thread(resolve_local_media_model, name, task = task)
    if pick is None:
        available = _format_available(await asyncio.to_thread(available_media_model_ids, task))
        raise _refuse(
            f"No downloaded {kind} model matches '{name}'."
            + (f" Downloaded {kind} models: {available}." if available else ""),
            status_code = 404,
            openai_errors = openai_errors,
            code = "model_not_found",
        )

    if _satisfied_by(await asyncio.to_thread(_backend_for(owner).status), name, pick):
        return

    async with _switch_lock(owner):
        backend = _backend_for(owner)
        # Re-read under the lock: a concurrent request may have just loaded this model.
        if _satisfied_by(await asyncio.to_thread(backend.status), name, pick):
            return
        if not await _drain(owner, backend):
            raise _refuse(
                _BUSY_MSG.format(kind = kind),
                status_code = 409,
                openai_errors = openai_errors,
                code = "model_busy",
                retry_after = _RETRY_AFTER_S,
            )
        await _start_load(owner, pick, current_subject)
        try:
            # Re-resolved: an engine switch (diffusers <-> native sd.cpp) replaces the object.
            ready = await _await_loaded(_backend_for(owner), pick)
        except RuntimeError as exc:
            # The loader already redacts this text; a bare raise would 500 with it instead.
            raise _refuse(
                f"'{pick.model_id}' could not be loaded: {exc}",
                status_code = 503,
                openai_errors = openai_errors,
                code = "model_load_failed",
            )
        if not ready:
            raise _refuse(
                _LOADING_MSG.format(model = pick.model_id),
                status_code = 503,
                openai_errors = openai_errors,
                code = "model_loading",
                retry_after = _RETRY_AFTER_S,
            )


# One switch at a time per backend, so two requests cannot race the single pipeline slot.
# Per running loop, like _auto_switch_lock in routes.inference: a module-level asyncio.Lock
# binds to the loop that first awaited it and hangs a second one.
_switch_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_switch_locks_guard = threading.Lock()


def _switch_lock(owner: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    # WeakKeyDictionary mutation is not thread-safe, so guard the get-or-create.
    with _switch_locks_guard:
        per_owner = _switch_locks.get(loop)
        if per_owner is None:
            per_owner = _switch_locks[loop] = {}
        lock = per_owner.get(owner)
        if lock is None:
            lock = per_owner[owner] = asyncio.Lock()
        return lock


__all__ = [
    "IMAGE_TASK",
    "VIDEO_TASK",
    "MediaModelPick",
    "available_media_model_ids",
    "invalidate_index",
    "maybe_auto_switch_media_model",
    "resolve_local_media_model",
]
