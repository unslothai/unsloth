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
import contextlib
import os
import threading
import time
import weakref
from dataclasses import dataclass, replace
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

# One end-to-end budget for the whole switch, under the ~100s tunnel window: a drain and a load
# with separate budgets add up past it, and the socket dies instead of returning the 503.
_SWITCH_BUDGET_S = 90.0
# A generation the caller cannot see must not be cut short, so the swap yields instead. Capped
# inside the budget, never on top of it.
_DRAIN_WAIT_S = 30.0
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
_SLOW_MSG = (
    "Selecting the {kind} model took too long to answer inside this request. It is still "
    "being prepared; retry shortly."
)
_UNVERIFIED_MSG = (
    "Could not verify that '{model}' is fully downloaded, so it was not switched in. "
    "Auto-switch never downloads; load it once from the {kind} page and retry."
)
_EDIT_ONLY_MSG = (
    "'{model}' is an edit-only model: it requires an input image, which this endpoint cannot "
    "supply. Name a text-to-image model instead."
)
_UNSIZED_MSG = (
    "'{model}' is missing some of its weights. Auto-switch never downloads, so load it once "
    "from the {kind} page and retry."
)
_INCOMPLETE_MSG = (
    "'{model}' is not fully downloaded: about {gb:.1f} GB of its companion weights are missing. "
    "Auto-switch never downloads, so load it once from the {kind} page and retry."
)
# Stands for "entries are missing but their size is unknown", so the refusal reports no figure.
_UNSIZED_MISSING = -1
# Cap on ids a "not found" error lists, so it stays readable in a terminal.
_MAX_LISTED_MODELS = 8


@dataclass(frozen = True)
class MediaModelPick:
    """A downloaded media model, in the shape its load route takes."""

    model_id: str
    model_path: str
    gguf_filename: Optional[str] = None
    model_kind: Optional[str] = None
    # The variant lister's full label, which distinguishes builds the backend's own quant token
    # collapses (IQ4_XS-3.53bpw vs -3.97bpw, and unlabelled files that have no token at all).
    quant: Optional[str] = None
    # True when another indexed build under the same model_path publishes the same quant token,
    # so a resident model cannot be proven to be this one and the switch must reload.
    ambiguous: bool = False


# Sentinel for a name two different models answer to; resolution treats it as no match.
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


def _published_token(pick: MediaModelPick) -> str:
    """The ``gguf_variant`` the backend will publish once *pick* is loaded, lowercased."""
    from hub.utils.gguf import extract_quant_token

    if not pick.gguf_filename:
        return ""
    token = extract_quant_token(pick.gguf_filename)
    return (token or "").strip().lower()


def _variant_label(filename: str) -> Optional[str]:
    """The variant lister's label for a loose checkpoint, so its identity matches an indexed one."""
    from utils.models.model_config import _extract_quant_label

    label = _extract_quant_label(filename)
    return label or None


def _add_gguf_picks(
    index: dict[str, MediaModelPick], info, keys: tuple[str, ...], on_disk: Path, load_dir: Path
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
        _register(
            index,
            keys,
            MediaModelPick(
                keys[0],
                str(load_dir.parent),
                load_dir.name,
                "gguf",
                quant = _variant_label(load_dir.name),
            ),
        )
        return True
    # Filenames come back relative to this directory, which is what the loader joins them onto.
    variants, _ = list_local_gguf_variants(str(load_dir))
    by_quant = {v.quant: v for v in variants if v.quant}
    if not by_quant:
        return False
    load_path = _gguf_load_path(info, on_disk, load_dir)
    for quant, variant in by_quant.items():
        # model_id stays the bare id so a "not found" error lists models, not one row per quant.
        _register(
            index,
            [f"{key}:{quant}" for key in keys],
            MediaModelPick(keys[0], load_path, variant.filename, "gguf", quant = quant),
        )
    best = preferred_quant(list(by_quant)) or next(iter(by_quant))
    _register(
        index,
        keys,
        MediaModelPick(keys[0], load_path, by_quant[best].filename, "gguf", quant = best),
    )
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
            # A cancelled or incomplete pull still lists, and loading it fails predictably.
            if getattr(info, "partial", False):
                continue
            if _local_model_task(info) != task:
                continue
            keys = _name_keys(info)
            if not keys:
                continue
            # Unwrapped once for both kinds: an HF cache repo keeps its weights, and its
            # model_index.json, one level down under snapshots/<sha>.
            on_disk = Path(info.path).expanduser()
            load_dir = _resolve_load_dir(on_disk)
            if _add_gguf_picks(index, info, keys, on_disk, load_dir):
                continue
            # Not a GGUF, so the load route detects the kind: a diffusers directory loads as a
            # pipeline, and a bare single-file directory is reinterpreted by the route itself.
            _register(index, keys, MediaModelPick(keys[0], str(load_dir)))
        except Exception as exc:  # noqa: BLE001 -- one unreadable model must not hide the rest
            logger.debug("media auto-switch: skipped %s: %s", getattr(info, "id", "?"), exc)
    return index


def _mark_ambiguous_builds(index: dict[str, MediaModelPick]) -> dict[str, MediaModelPick]:
    """Flag every GGUF pick whose published token another build under its path also publishes."""
    seen: dict[tuple[str, str], set] = {}
    for pick in index.values():
        if pick is _AMBIGUOUS or pick.model_kind != "gguf":
            continue
        key = (pick.model_path.strip().lower(), _published_token(pick))
        seen.setdefault(key, set()).add(pick.gguf_filename)
    collides = {key for key, files in seen.items() if len(files) > 1 or key[1] == ""}
    if not collides:
        return index
    return {
        name: (
            pick
            if pick is _AMBIGUOUS
            or pick.model_kind != "gguf"
            or (pick.model_path.strip().lower(), _published_token(pick)) not in collides
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
    pick = _cached_index(task).get(name.strip().lower())
    return None if pick is _AMBIGUOUS else pick


def available_media_model_ids(task: str) -> list[str]:
    """Sorted ids a request may name for *task*, for a "not found" error to list."""
    return sorted(
        {pick.model_id for pick in _cached_index(task).values() if pick is not _AMBIGUOUS}
    )


# ── the switch ──────────────────────────────────────────────────────


def _incomplete_message(pick: MediaModelPick, missing: int, kind: str) -> str:
    """The refusal text, which only quotes a size when the plan could size what it is missing."""
    if missing == _UNSIZED_MISSING:
        return _UNSIZED_MSG.format(model = pick.model_id, kind = kind)
    return _INCOMPLETE_MSG.format(model = pick.model_id, gb = missing / 1e9, kind = kind)


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

    A GGUF also has to match on quant. Loose ``.gguf`` files in one scan folder share that
    folder as their ``model_path``, so the path alone would report a sibling as already
    serving and generate on the wrong weights.

    The comparison uses the token the backend actually publishes. Where that token cannot tell
    two indexed builds apart (``IQ4_XS-3.53bpw`` and ``-3.97bpw`` both publish ``IQ4_XS``, and
    an unlabelled file publishes nothing), the pick is marked ambiguous at index time and this
    answers False: reloading costs a load, serving the sibling returns the wrong image.
    """
    if not _resident_is_pick(status, name, pick):
        return False
    # Ambiguity only blocks the skip, never the "did my load land" check: the reload settles it.
    return not pick.ambiguous


def _resident_is_pick(status: dict[str, Any], name: str, pick: MediaModelPick) -> bool:
    """Whether the resident build is the one *pick* names, on the identity status publishes."""
    if not status.get("loaded"):
        return False
    resident = str(status.get("repo_id") or "").strip().lower()
    if not resident:
        return False
    aliases = {name.strip().lower(), pick.model_id.strip().lower()}
    # The path is compared without folding case: /models/Foo and /models/foo are different
    # models on a case-sensitive filesystem, and folding them would serve one for the other.
    same_path = os.path.normcase(str(status.get("repo_id") or "").strip()) == os.path.normcase(
        pick.model_path.strip()
    )
    if resident not in aliases and not same_path:
        return False
    if pick.model_kind != "gguf":
        return True
    loaded_quant = str(status.get("gguf_variant") or "").strip().lower()
    return loaded_quant == _published_token(pick)


def _backend_busy(backend: Any) -> bool:
    """One off-loop read of whether a load or generation is running. Mirrors media_keepwarm."""
    if backend.loading_repo_ids():
        return True
    return bool((backend.generate_progress() or {}).get("active"))


async def _drain(owner: str, backend: Any, deadline: float) -> bool:
    """Wait out other tracked requests and any in-flight load or generation.

    A request queued on this backend's switch lock is counted by the middleware but is not
    doing any work, so it is discounted here: two concurrent requests for the same absent
    model would otherwise each wait the other out and both return 409. Mirrors the chat
    switch, which excludes its own waiters from ``_wait_for_model_switch_idle``.
    """
    from core.inference.media_keepwarm import other_request_count
    while True:
        # This request is itself tracked and itself a waiter, so it counts as neither.
        others = other_request_count(owner, current_request_counted = True)
        others -= max(0, _waiter_count(owner) - 1)
        if others <= 0 and not await asyncio.to_thread(_backend_busy, backend):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)


async def _await_loaded(backend: Any, name: str, pick: MediaModelPick, deadline: float) -> bool:
    """Poll the background load until the REQUESTED model is resident; False if still going.

    Checked against the pick, not merely "something is loaded": a user load accepted between
    two polls supersedes this one, and returning success there would generate on the
    replacement while reporting the requested model.
    """
    while True:
        progress = await asyncio.to_thread(backend.load_progress) or {}
        phase = progress.get("phase")
        if phase == "error":
            raise RuntimeError(progress.get("error") or "The model failed to load.")
        if phase in (None, "ready"):
            status = await asyncio.to_thread(backend.status)
            # The landed check, not the skip check: this load is ours, so ambiguity is settled.
            if _resident_is_pick(status, name, pick):
                return True
            # Loaded, but not this pick: a load that landed after ours replaced it.
            raise RuntimeError(f"'{pick.model_id}' was replaced by another load before it served.")
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)


async def _bounded(coro, deadline: float, *, kind: str, openai_errors: bool):
    """Await *coro* within the switch budget, refusing rather than outliving the response window.

    The worker thread behind a ``to_thread`` keeps running after this returns; what matters is
    that the request stops waiting on it, since the caller's connection is the thing on a clock.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        coro.close()
        raise _refuse(
            _SLOW_MSG.format(kind = kind),
            status_code = 503,
            openai_errors = openai_errors,
            code = "model_loading",
            retry_after = _RETRY_AFTER_S,
        )
    try:
        return await asyncio.wait_for(coro, timeout = remaining)
    except asyncio.TimeoutError:
        raise _refuse(
            _SLOW_MSG.format(kind = kind),
            status_code = 503,
            openai_errors = openai_errors,
            code = "model_loading",
            retry_after = _RETRY_AFTER_S,
        )


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


def _normalized_pick(pick: MediaModelPick) -> MediaModelPick:
    """The pick as the LOAD route will read it, with a bare single-file directory reinterpreted.

    Both load routes turn a kindless directory holding exactly one checkpoint into a
    ``single_file`` load and then resolve that family's companions. Planning the un-normalized
    pick describes a local pipeline with nothing to fetch, and misses those companions.
    """
    from core.inference.diffusion import resolve_local_single_file

    if pick.model_kind or pick.gguf_filename:
        return pick
    sole = resolve_local_single_file(pick.model_path)
    if sole is None:
        return pick
    return replace(pick, gguf_filename = sole, model_kind = "single_file")


def _is_edit_only(pick: MediaModelPick) -> bool:
    """Whether *pick* is an instruction-editing family, which has no text-to-image mode.

    The local catalog tags these text-to-image, so without this the switch would evict a working
    model for a multi-GB pipeline that /v1/images/generations then refuses for lacking txt2img.
    """
    from core.inference.diffusion import _family_workflows
    from core.inference.diffusion_families import detect_family_for_pick

    try:
        fam = detect_family_for_pick(pick.model_path, pick.gguf_filename, None)
    except Exception:  # noqa: BLE001 -- a detection miss must not refuse a loadable pick
        return False
    if fam is None:
        return False
    return "txt2img" not in _family_workflows(fam)


def _planners_for(owner: str, pick: MediaModelPick) -> list:
    """Every engine whose plan this pick could end up loading through.

    Usually one. ``predict_engine`` treats an absent sd.cpp binary as available whenever its
    installation is allowed, while ``select_and_activate_engine`` falls back to diffusers when
    that install produces nothing runnable, and the two engines read different companion sets.
    Both are verified only in that case: with a runnable binary already on disk the load stays
    native, and demanding the diffusers shards too would refuse a model sd.cpp can serve.
    """
    if owner != DIFFUSION:
        return [_backend_for(owner)]
    from core.inference.diffusion import resolve_model_kind
    from core.inference.diffusion_engine_router import (
        engine_for,
        native_binary_installed,
        predict_engine,
    )
    from core.inference.diffusion_families import detect_family_for_pick
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    fam = detect_family_for_pick(pick.model_path, pick.gguf_filename, None)
    if fam is None:
        return [_backend_for(owner)]
    kind = resolve_model_kind(pick.gguf_filename, pick.model_kind)
    predicted = predict_engine(fam, model_kind = kind)
    names = [predicted]
    if predicted == ENGINE_SD_CPP and not native_binary_installed():
        names.append(ENGINE_DIFFUSERS)
    return [engine_for(name) for name in names]


def _plan_gpu_ordinal() -> Optional[int]:
    """The card the load route will rank for itself, so the plan sizes the same file set.

    Automatic precision is chosen per card, and a different card can select a different hosted
    pre-quantized artifact, which a plan plotted against the default device would omit.
    """
    from core.inference.diffusion_device import (
        resolve_diffusion_device_target,
        resolve_selected_cuda_ordinal,
    )

    if resolve_diffusion_device_target().device != "cuda":
        return None
    return resolve_selected_cuda_ordinal(None)


def _missing_download_bytes(owner: str, pick: MediaModelPick) -> Optional[int]:
    """Bytes this pick would still have to fetch, or 0 when nothing is missing.

    The resolver only indexes downloaded CHECKPOINTS, but a GGUF or single-file pick loads its
    text encoders and VAE from a companion base repo, and the loader prefetches whatever of that
    is absent. Without this an API request could pull tens of gigabytes, which is exactly what
    the setting promises it cannot do. Same planner ``/images/download-plan`` serves, so the
    answer matches what the UI would have staged.

    Planned against the engine that will LOAD this pick, the way /images/download-plan does:
    the resident engine can be native sd.cpp while the target loads through diffusers, and its
    planner refuses the pick, which the catch below would read as nothing missing.

    Returns None when locality could not be established: the image planner raises, and the
    video one returns zero bytes with ``plan_failed`` because its own caller falls back to an
    inline pull. Either way zero is not evidence of a complete cache, and treating it as such
    would allow exactly the download this exists to prevent, so the switch refuses instead.
    """
    target = _normalized_pick(pick)
    # A local full pipeline is complete by definition: from_pretrained reads it off disk, and
    # the planner would ask the Hub about an absolute path and fail, which now reads as
    # unverifiable and would refuse every on-device model.
    if not target.gguf_filename and Path(target.model_path).expanduser().is_dir():
        return 0
    try:
        ordinal = _plan_gpu_ordinal()
        plans = [
            planner.download_plan(
                target.model_path,
                gguf_filename = target.gguf_filename,
                model_kind = target.model_kind,
                gpu_ordinal = ordinal,
            )
            or {}
            for planner in _planners_for(owner, target)
        ]
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.debug("media auto-switch: download plan for %s failed: %s", pick.model_id, exc)
        return None
    if any(plan.get("plan_failed") for plan in plans):
        return None
    missing = max((max(0, int(plan.get("total_bytes") or 0)) for plan in plans), default = 0)
    # An entry whose sibling sizes could not be read still names a file the load will fetch, and
    # both planners coerce an unknown size to zero, so entries decide and bytes only describe.
    if not missing and any(plan.get("entries") for plan in plans):
        return _UNSIZED_MISSING
    return missing


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

    # Started before resolution: the cold scan and the download plan are part of the wait the
    # caller experiences, so a budget that began after them would not bound the response.
    deadline = time.monotonic() + _SWITCH_BUDGET_S
    name = requested_model.strip()
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    kind = "image" if owner == DIFFUSION else "video"

    # An exact match on the resident model needs no discovery: a scan that failed or skipped an
    # entry would otherwise 404 the very model that is loaded for as long as the empty index is
    # cached. Never for a resident GGUF: a bare repo id means the preferred quant, which this
    # comparison cannot see, so it would serve whichever quant happens to be up.
    resident = await asyncio.to_thread(_backend_for(owner).status)
    if (
        resident.get("loaded")
        and resident.get("model_kind") != "gguf"
        and name.lower() == str(resident.get("repo_id") or "").lower()
    ):
        return

    # Off the loop: a cold index walks the model roots and reads GGUF headers.
    pick = await _bounded(
        asyncio.to_thread(resolve_local_media_model, name, task = task),
        deadline,
        kind = kind,
        openai_errors = openai_errors,
    )
    if pick is None:
        available = _format_available(await asyncio.to_thread(available_media_model_ids, task))
        raise _refuse(
            f"No downloaded {kind} model matches '{name}'."
            + (f" Downloaded {kind} models: {available}." if available else ""),
            status_code = 404,
            openai_errors = openai_errors,
            code = "model_not_found",
        )

    # Before anything is evicted: this endpoint only generates from text, and the load would
    # otherwise finish and then be refused for lacking txt2img, with the useful model gone.
    if owner == DIFFUSION and await asyncio.to_thread(_is_edit_only, pick):
        raise _refuse(
            _EDIT_ONLY_MSG.format(model = pick.model_id),
            status_code = 400,
            openai_errors = openai_errors,
            code = "invalid_value",
        )

    if _satisfied_by(resident, name, pick):
        return

    from core.inference.media_keepwarm import admission_gate

    with _note_waiter(owner):
        async with _switch_lock(owner):
            backend = _backend_for(owner)
            # Re-read under the lock: a concurrent request may have just loaded this model.
            if _satisfied_by(await asyncio.to_thread(backend.status), name, pick):
                return
            missing = await _bounded(
                asyncio.to_thread(_missing_download_bytes, owner, pick),
                deadline,
                kind = kind,
                openai_errors = openai_errors,
            )
            if missing is None:
                raise _refuse(
                    _UNVERIFIED_MSG.format(model = pick.model_id, kind = kind),
                    status_code = 409,
                    openai_errors = openai_errors,
                    code = "model_not_downloaded",
                )
            if missing:
                raise _refuse(
                    _incomplete_message(pick, missing, kind),
                    status_code = 409,
                    openai_errors = openai_errors,
                    code = "model_not_downloaded",
                )
            if not await _drain(owner, backend, min(deadline, time.monotonic() + _DRAIN_WAIT_S)):
                raise _refuse(
                    _BUSY_MSG.format(kind = kind),
                    status_code = 409,
                    openai_errors = openai_errors,
                    code = "model_busy",
                    retry_after = _RETRY_AFTER_S,
                )
            # Held ACROSS the last drain observation, not after it: a request admitted between
            # a passing drain and the gate is tracked but has not marked the backend active
            # yet, so it would read as idle and be cancelled by the load's teardown.
            async with admission_gate(owner):
                # Re-resolved under the gate: a concurrent load can activate the other image
                # engine while this request drains, leaving `backend` pointing at the idle one
                # and both checks passing against a backend nothing is using.
                backend = _backend_for(owner)
                # What the drain waited out may have been the very load this request wanted.
                if _satisfied_by(await asyncio.to_thread(backend.status), name, pick):
                    return
                if not await _drain(owner, backend, time.monotonic()):
                    raise _refuse(
                        _BUSY_MSG.format(kind = kind),
                        status_code = 409,
                        openai_errors = openai_errors,
                        code = "model_busy",
                        retry_after = _RETRY_AFTER_S,
                    )
                # Re-planned here because the drain can last 30 seconds, and a cache deletion
                # during it sees a target that is neither loaded nor loading yet, so its guard
                # allows the removal of files this already verified.
                missing = await _bounded(
                    asyncio.to_thread(_missing_download_bytes, owner, pick),
                    deadline,
                    kind = kind,
                    openai_errors = openai_errors,
                )
                if missing is None or missing:
                    raise _refuse(
                        _UNVERIFIED_MSG.format(model = pick.model_id, kind = kind)
                        if missing is None
                        else _incomplete_message(pick, missing, kind),
                        status_code = 409,
                        openai_errors = openai_errors,
                        code = "model_not_downloaded",
                    )
                await _start_load(owner, pick, current_subject)
            try:
                # Re-resolved: an engine switch (diffusers <-> sd.cpp) replaces the object.
                ready = await _await_loaded(_backend_for(owner), name, pick, deadline)
            except RuntimeError as exc:
                # The loader already redacts this text; a bare raise would 500 with it.
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


# Requests parked on a backend's switch lock. They hold no work, so the drain discounts them.
_waiters: dict[str, int] = {}
_waiters_guard = threading.Lock()


@contextlib.contextmanager
def _note_waiter(owner: str):
    with _waiters_guard:
        _waiters[owner] = _waiters.get(owner, 0) + 1
    try:
        yield
    finally:
        with _waiters_guard:
            remaining = _waiters.get(owner, 0) - 1
            if remaining > 0:
                _waiters[owner] = remaining
            else:
                _waiters.pop(owner, None)


def _waiter_count(owner: str) -> int:
    with _waiters_guard:
        return _waiters.get(owner, 0)


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
