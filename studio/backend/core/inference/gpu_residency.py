# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether an image or video model can stay on a GPU next to chat, and what to unload when it can't.

Keep multiple models loaded set to When they fit or Always lets the GPU arbiter keep several residents. This module
prices them: a media resident holds the weights plus generation headroom it was admitted with, a new media load is
priced from the files already on disk, and live free VRAM decides whether it fits. Anything it cannot price is
treated as not fitting, so the caller frees memory first (fit) or refuses (always). Modelled on training_vram.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Optional

from core.inference import gpu_arbiter as arb
from loggers import get_logger

logger = get_logger(__name__)

# fits iff free_mib >= need_mib + FIT_MARGIN_MIB on every GPU the load uses. Live free VRAM already excludes resident
# weights, so the margin only covers estimate error and allocator slack. An image load must also keep its
# transformer on the GPU (the diffusion planner adds its own reserve on top, so this is not doubled there).
FIT_MARGIN_MIB = 1024
# A chat load in flight finishes on its own; the media auto-switch uses the same wait.
K3_RETRY_AFTER_S = 15

MEDIA_OWNERS = (arb.DIFFUSION, arb.VIDEO)
_MEDIA_KIND = {arb.DIFFUSION: "image", arb.VIDEO: "video"}


class MediaNeed(dict):
    """{physical GPU id: weights + generation headroom MiB}, plus the split the planner rule needs."""

    def __init__(self, per_gpu: dict, *, weights_mib: int, headroom_mib: int, kind: str):
        super().__init__({int(gpu): int(mib) for gpu, mib in per_gpu.items()})
        self.weights_mib = int(weights_mib)
        self.headroom_mib = int(headroom_mib)
        self.kind = kind


def current_policy() -> str:
    """Keep multiple models loaded, read as the owner. Never call this under the arbiter lock (it reads a database)."""
    try:
        from utils.multi_model_settings import get_multi_model_mode
        return get_multi_model_mode()
    except Exception as exc:  # noqa: BLE001 - an unreadable setting keeps today's behaviour
        logger.warning("gpu_residency: could not read Keep multiple models loaded: %s", exc)
        return "off"


def _media_records() -> dict[str, dict]:
    """Admitted needs of the media residents still on the GPU. Lock free: acquire_for calls this holding the lock."""
    records = dict(getattr(arb, "_media_needs", {}) or {})
    if not records:
        return {}
    resident = set(arb._view()) if hasattr(arb, "_view") else {arb._owner}
    return {owner: need for owner, need in records.items() if owner in resident}


def media_held_mib() -> dict[int, int]:
    """{physical GPU id: MiB} held by resident image and video models (weights + headroom). Never raises."""
    try:
        held: dict[int, int] = {}
        for need in _media_records().values():
            for gpu, mib in need.items():
                held[gpu] = held.get(gpu, 0) + int(mib)
        return held
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpu_residency: could not price the resident media models: %s", exc)
        return {}


def _media_engine(owner: str) -> Any:
    from core.inference.media_keepwarm import engine_if_imported
    return engine_if_imported(owner)


def _media_busy(owner: str) -> tuple[bool, bool]:
    """(loading, generating) for a media owner; unknown reads as busy, so it is never freed as idle."""
    try:
        engine = _media_engine(owner)
        if engine is None:
            return False, False
        loading = bool(engine.loading_repo_ids())
        progress = engine.generate_progress() or {}
        from core.inference.media_keepwarm import _TRACKERS

        tracker = _TRACKERS.get(owner)
        outstanding = bool(tracker is not None and tracker.outstanding())
        return loading, bool(progress.get("active")) or outstanding
    except Exception:  # noqa: BLE001
        return True, True


def summarize_resident_media() -> dict[str, Any]:
    """Report the resident image and video models. Never raises."""
    summary: dict[str, Any] = {}
    records = {}
    try:
        records = _media_records()
        resident = arb.residents()
    except Exception:  # noqa: BLE001
        resident = {}
    for owner in MEDIA_OWNERS:
        loading, generating = _media_busy(owner) if owner in resident else (False, False)
        summary[owner] = {
            "resident": owner in resident,
            "gpus": sorted(resident.get(owner) or ()),
            "loading": loading,
            "generating": generating,
            "priced": owner in records,
        }
    summary["loading"] = any(summary[owner]["loading"] for owner in MEDIA_OWNERS)
    summary["held_mib"] = media_held_mib()
    return summary


def media_load_need_mib(kind: str, request: Any, gpu_ordinal: Optional[int]) -> Optional[MediaNeed]:
    """What a new image or video load will hold, priced from files already on disk. None when unknown.

    ``gpu_ordinal`` is the visible torch ordinal the route resolved; None (automatic) is only priceable on a single
    visible GPU. Never touches the network and never raises."""
    try:
        from utils.hardware.hardware import get_parent_visible_gpu_ids

        parent = list(get_parent_visible_gpu_ids() or ())
        if gpu_ordinal is None:
            physical = parent[0] if len(parent) == 1 else None
        else:
            physical = parent[gpu_ordinal] if 0 <= gpu_ordinal < len(parent) else None
        if physical is None:  # several GPUs and no pick, or a UUID/MIG mask with no numeric ids
            return None
        priced = {arb.DIFFUSION: _image_need, arb.VIDEO: _video_need}.get(kind, lambda *_: None)(
            request, gpu_ordinal
        )
        if priced is None or not priced[0] or priced[0] <= 0:
            return None
        weights, headroom = int(priced[0]), int(priced[1])
        return MediaNeed(
            {physical: weights + headroom},
            weights_mib = weights,
            headroom_mib = headroom,
            kind = _MEDIA_KIND[kind],
        )
    except Exception as exc:  # noqa: BLE001 - an unpriced load is treated as not fitting
        logger.warning("gpu_residency: could not price the %s load: %s", kind, exc)
        return None


def _cached_pick(repo: str, filename: Optional[str]) -> Optional[str]:
    """The picked checkpoint file when it is already on disk (local path or either Hub cache root), else None."""
    from pathlib import Path

    local = Path(repo).expanduser()
    if local.is_file():
        return str(local)
    if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
        return None
    if local.is_dir():
        return str(local / filename) if (local / filename).is_file() else None
    from huggingface_hub import try_to_load_from_cache
    from core.inference.diffusion import hub_cache_dir

    for root in (hub_cache_dir(), None):
        hit = try_to_load_from_cache(repo, filename, cache_dir = root)
        if isinstance(hit, str) and Path(hit).is_file():
            return hit
    return None


def _image_need(request: Any, gpu_ordinal: Optional[int]) -> Optional[tuple[int, int]]:
    """(weights MiB, headroom MiB) for a diffusers image load, the way DiffusionBackend._plan_memory prices it."""
    from core.inference.diffusion import (
        DiffusionBackend as DB,
        _image_variant_hint,
        resolve_model_kind,
    )
    from core.inference.diffusion_auto_policy import resident_bytes_from_declared
    from core.inference.diffusion_engine_router import predict_engine
    from core.inference.diffusion_families import (
        detect_family_for_pick,
        named_variant_base,
        prefer_ungated_mirror,
        resolve_base_repo,
    )
    from core.inference.diffusion_gguf_pipeline import whole_pipeline_gguf_resident_mib
    from core.inference.diffusion_memory import (
        estimate_gguf_resident_mib,
        estimate_image_runtime_mib,
        estimate_safetensors_dense_mib,
        file_size_mib,
    )
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    repo, name = request.model_path, request.gguf_filename
    if getattr(request, "text_encoder_file", None) or getattr(request, "vae_file", None):
        return None  # supplied companions are priced from their headers at load time only
    kind = resolve_model_kind(name, request.model_kind)
    fam = detect_family_for_pick(repo, name, request.family_override)
    if (
        fam is None
        or predict_engine(fam, model_kind = kind, gpu_ordinal = gpu_ordinal, gguf_filename = name)
        != ENGINE_DIFFUSERS
    ):
        return None
    mib = 1024 * 1024
    if kind == "pipeline":
        base = repo
        files: dict[str, int] = {}

        def grab(d):
            sizes = DB._local_dir_weight_sizes(d, exclude_transformer = False)
            for rel, size in sizes.items():
                files[rel] = max(files.get(rel, 0), size)
            return sizes

        DB._union_over_cached_revs(prefer_ungated_mirror(repo), grab)
        weights = (resident_bytes_from_declared(repo, files.items()) or 0) // mib
    else:
        path = _cached_pick(repo, name)
        if path is None:
            return None
        whole = getattr(fam, "single_file_is_pipeline", False)
        if kind == "single_file":
            fp8 = not whole and "fp8" in str(name).lower()
            weights = estimate_safetensors_dense_mib(file_size_mib(path), fp8_upcast = fp8)
        elif whole:
            weights = whole_pipeline_gguf_resident_mib(path) or estimate_gguf_resident_mib(
                file_size_mib(path)
            )
        else:
            weights = estimate_gguf_resident_mib(file_size_mib(path))
        base = (request.base_repo or "").strip()
        if (
            not base
        ):  # offline only: the link a past load recorded, else the curated variant the name spells
            from hub.utils.companion_assets import read_companion_links
            linked = read_companion_links().get(str(repo).strip().lower()) or []
            base = linked[0] if linked else (named_variant_base(fam, repo, name) or "")
        base = resolve_base_repo(fam, base)
        if not whole:
            companions = max(
                DB._companion_cache_bytes(prefer_ungated_mirror(base)),
                DB._companion_cache_bytes(base),
            )
            if not companions:
                return None  # the text encoder and VAE are not downloaded yet
            weights = (weights or 0) + companions // mib
    hint = _image_variant_hint(fam.name, name, repo, base)
    return weights, estimate_image_runtime_mib(width = None, height = None, family = hint)


def _video_need(request: Any, gpu_ordinal: Optional[int]) -> Optional[tuple[int, int]]:
    """(weights MiB, headroom MiB) for a video load: checkpoint file sizes plus the family's companion table."""
    from core.inference.diffusion import DiffusionBackend as DB
    from core.inference.diffusion_component_sizes import video_components_gb
    from core.inference.diffusion_memory import (
        estimate_gguf_resident_mib,
        estimate_safetensors_dense_mib,
        estimate_video_runtime_mib,
        file_size_mib,
    )
    from core.inference.video import _detect_load_family, resolve_video_model_kind
    from core.inference.video_families import resolve_video_base_repo
    from core.inference.video_minimax_h3 import is_h3_native
    from core.inference.video_moe_pair import moe_partner_filename

    repo, name = request.model_path, request.gguf_filename
    kind = resolve_video_model_kind(name, request.model_kind)
    fam = _detect_load_family(
        repo, name, request.family_override, request.display_repo_id, request.model_kind
    )
    if fam is None or is_h3_native(fam, kind):
        return None  # H3 native runs its own server processes, priced elsewhere
    gb = 1000.0**3 / (1024 * 1024)  # MiB per GB of the family tables
    if kind == "pipeline":
        components = video_components_gb(fam, repo, allow_hub = False)
        if components is None or not DB._cache_bytes(repo):
            return None
        weights = int(sum(components) * gb)
    else:
        names = [name] + ([moe_partner_filename(name)] if getattr(fam, "is_moe", False) else [])
        paths = [_cached_pick(repo, n) if n else None for n in names]
        components = video_components_gb(
            fam, resolve_video_base_repo(fam, request.base_repo, repo), allow_hub = False
        )
        if None in paths or components is None:
            return None
        sizes = [file_size_mib(p) for p in paths]
        price = estimate_gguf_resident_mib if kind == "gguf" else estimate_safetensors_dense_mib
        if None in sizes:
            return None
        weights = sum(price(s) for s in sizes) + int((components[1] + components[2]) * gb)
    preset = fam.resolution_presets[0]
    return weights, estimate_video_runtime_mib(
        width = preset[0], height = preset[1], num_frames = fam.default_num_frames
    )


def _free_mib_by_gpu() -> tuple[Optional[dict[int, dict]], Optional[str]]:
    """{physical id: {free_mib, total_mib}} or (None, reason) when the host can't be priced."""
    try:
        from utils.hardware import DeviceType, get_device, get_visible_gpu_utilization
        if get_device() != DeviceType.CUDA:
            # MLX and other unified-memory hosts share the pool with the CPU: co-residency there is deferred.
            return None, "non_accelerator"
        util = get_visible_gpu_utilization() or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpu_residency: GPU probe failed: %s", exc)
        return None, "probe_error"
    if util.get("index_kind") != "physical":
        return None, "unmapped_gpus"
    rows: dict[int, dict] = {}
    for device in util.get("devices") or ():
        total_gb, used_gb = device.get("vram_total_gb"), device.get("vram_used_gb")
        if device.get("unified_memory") or device.get("shared_memory") or not total_gb:
            rows[device["index"]] = {"unified": True}
            continue
        if used_gb is None:
            continue
        total = int(total_gb * 1024)
        rows[device["index"]] = {
            "free_mib": max(int(total - used_gb * 1024), 0),
            "total_mib": total,
        }
    return rows, None


def _keeps_transformer_resident(need: MediaNeed, gpu: int, free_mib: int, total_mib: int) -> bool:
    """The diffusion planner's verdict for this load at the free memory left: "fits" never means streaming weights."""
    from core.inference.diffusion_memory import (
        DeviceMemory,
        plan_diffusion_memory,
        plan_keeps_transformer_resident,
    )

    ordinal = _ordinal_for_physical(gpu)
    plan = plan_diffusion_memory(
        target = SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = DeviceMemory("cuda", f"cuda:{ordinal}", "discrete_vram", free_mib, total_mib),
        model_dense_mib = need.weights_mib,
        runtime_headroom_mib = need.headroom_mib,
        # Planned as auto whatever the load asks: Balanced and Low VRAM hold less than an auto plan that stays resident.
        requested_mode = None,
    )
    return plan_keeps_transformer_resident(plan)


def _ordinal_for_physical(gpu: int) -> int:
    try:
        from utils.hardware.hardware import get_parent_visible_gpu_ids
        parent = list(get_parent_visible_gpu_ids() or ())
        return parent.index(gpu) if gpu in parent else gpu
    except Exception:  # noqa: BLE001
        return gpu


def can_load_alongside(owner: str, need_mib: Optional[dict]) -> tuple[bool, dict]:
    """Whether ``owner`` fits next to the residents already loaded. Side-effect free; default-deny with a reason."""
    if not need_mib:
        return False, {"reason": "estimate_unavailable"}
    rows, reason = _free_mib_by_gpu()
    if rows is None:
        return False, {"reason": reason}
    records = _media_records()
    own = records.get(owner)
    # A resident still loading has not allocated its weights yet either, so live free VRAM misses all of it.
    loading = {name for name in records if name != owner and _media_busy(name)[0]}
    info: dict[str, Any] = {"need_mib": {}, "free_mib": {}, "short_mib": {}, "reason": "fits"}
    for gpu, mib in need_mib.items():
        row = rows.get(gpu)
        if row is None:
            return False, {"reason": "gpu_unknown", "gpu": gpu}
        if row.get("unified"):
            return False, {"reason": "unified_memory", "gpu": gpu}
        free = row["free_mib"]
        # An idle resident's generation headroom is not allocated yet; it will be when it generates.
        free -= sum(
            int(need[gpu]) if name in loading else getattr(need, "headroom_mib", 0)
            for name, need in records.items()
            if name != owner and gpu in need
        )
        if own is not None and gpu in own:
            # Reloading the same owner frees what it holds now.
            free += max(own[gpu] - getattr(own, "headroom_mib", 0), 0)
        info["need_mib"][gpu], info["free_mib"][gpu] = int(mib), max(free, 0)
        if free < mib + FIT_MARGIN_MIB:
            info["short_mib"][gpu] = int(mib + FIT_MARGIN_MIB - free)
            info["reason"] = "short"
    if info["short_mib"]:
        return False, info
    if getattr(need_mib, "kind", None) == "image":
        for gpu in need_mib:
            if not _keeps_transformer_resident(
                need_mib, gpu, info["free_mib"][gpu], rows[gpu]["total_mib"]
            ):
                info["reason"] = "would_stream"
                return False, info
    return True, info


def chat_load_in_flight() -> bool:
    """Whether a chat load (or swap) is running. Its GPUs aren't known before it launches, so callers treat it as
    overlapping every GPU."""
    try:
        from core.inference import llama_keepwarm, model_slots
        from core.inference.llama_cpp import chat_load_active
        return bool(
            llama_keepwarm._load_lock.locked()
            or chat_load_active()
            or model_slots.loading is not None
        )
    except Exception:  # noqa: BLE001
        return False


_CHAT_LOADING_MESSAGE = (
    "Unsloth Studio is loading a chat model on this GPU. Retry once it has finished loading."
)


class ChatLoadInFlightRefusal(Exception):
    """A media load under Always met a chat load whose size isn't known yet. Retryable, unlike the Always refusal."""

    def __init__(self) -> None:
        super().__init__(_CHAT_LOADING_MESSAGE)

    def as_http_exception(
        self,
        path: Optional[str] = None,
        *,
        openai_errors: bool = False,
    ):
        return chat_loading_http_exception(path, openai_errors = openai_errors)


def chat_loading_http_exception(path: Optional[str] = None, *, openai_errors: bool = False):
    """The transient 409, shaped like GpuBusyForAnotherAccountError.as_http_exception."""
    from fastapi import HTTPException
    from utils.api_errors import error_body_for_path

    retry_after = K3_RETRY_AFTER_S
    if openai_errors and not (path and path.startswith("/v1/")):
        path = "/v1/"
    detail = (
        error_body_for_path(path, _CHAT_LOADING_MESSAGE, status = 409, code = "gpu_busy", param = "model")
        if path and path.startswith("/v1/")
        else {"error": "gpu_busy", "message": _CHAT_LOADING_MESSAGE, "retry_after": retry_after}
    )
    return HTTPException(status_code = 409, detail = detail, headers = {"Retry-After": str(retry_after)})


def _gb(mib: int) -> str:
    return f"{mib / 1024:.1f}"


def need_free_refusal(info: dict, slots: list[dict]) -> Exception:
    """The Always refusal for a load that doesn't fit next to the models already loaded."""
    from utils.account_context import is_owner_context

    short = info.get("short_mib") or {}
    gpu = max(short, key = short.get) if short else None
    if gpu is None or not is_owner_context():
        # No numbers (masked host, unified memory, probe error) or a managed account: the plain refusal.
        return arb.refusal_for(slots)
    message = (
        f"This model needs about {_gb(info['need_mib'][gpu])} GB of GPU memory, and {_gb(info['free_mib'][gpu])} GB "
        "is free next to the models already loaded. Keep multiple models loaded is set to Always, so nothing is "
        "unloaded to make room. Unload one of them, or choose When they fit."
    )
    first = slots[0] if slots else {}
    return arb.MultiModelAlwaysRefusal(
        message, kind = first.get("kind"), model = first.get("model") or None
    )


def foreign_on(owner: str, account_id: Optional[str]) -> int:
    """Generations another account runs on ``owner``'s model; only these block evicting it."""
    try:
        if owner == arb.CHAT:
            from state import active_generations
            return active_generations.foreign_count(account_id)
        if owner == arb.DIFFUSION:
            from hub.services.models import account_access
            if account_access.generation_is_foreign("diffusion"):
                return 1
            return 0
        if owner == arb.VIDEO:
            import sys

            video = sys.modules.get("core.inference.video")
            running = video.generation_account_in_flight() if video is not None else None
            return int(running is not None and running != account_id)
    except Exception as exc:  # noqa: BLE001 - unknown reads as busy, so nothing is torn down under a stream
        logger.warning("gpu_residency: could not count generations on %s: %s", owner, exc)
        return 1
    return 0


def _overlaps(gpus: Optional[Iterable[int]], target: Optional[Iterable[int]]) -> bool:
    return gpus is None or target is None or bool(set(gpus) & set(target))


def _record_evicted(slots: list[dict], evicted: Optional[list], freed: list[str]) -> None:
    """Report evictions the way acquire_for does (to ``evicted`` or the route's sink) and log them."""
    try:
        from core.inference.api_monitor import api_monitor
    except Exception:  # noqa: BLE001
        api_monitor = None
    arb.note_evictions(slots, evicted)
    for slot in slots:
        model = slot.get("model") or slot.get("kind") or ""
        freed.append(model)
        if api_monitor is not None:
            try:
                api_monitor.record_lifecycle(event = "unload", model = model, reason = "evicted")
            except Exception:  # noqa: BLE001
                pass


def _evict_resident(
    name: str, acting: Optional[str], evicted: Optional[list], freed: list[str]
) -> None:
    """Evict one whole resident. Caller holds the arbiter lock."""
    busy = foreign_on(name, acting)
    if busy:
        raise arb.GpuBusyForAnotherAccountError(name, busy)
    slots = arb.resident_slots(name) or [{"kind": _MEDIA_KIND.get(name, "chat"), "model": ""}]
    logger.info("gpu_arbiter: evicting %s to make room", name)
    arb._EVICTORS[name]()
    arb._drop_resident(name)
    _record_evicted(slots, evicted, freed)


def _free_steps(
    owner: str,
    gpu_ids: Optional[Iterable[int]],
    reason: str,
    evicted: Optional[list],
    *,
    need_mib: Optional[dict] = None,
    short_mib: int = 0,
    acting: Optional[str] = None,
) -> list[str]:
    """Free the cheapest residents on ``gpu_ids`` first, re-checking after each step. Caller holds the arbiter lock.

    Order: anything still loading, idle video servers, the idle image or video model, idle chat models kept
    alongside, then the active chat model. Residents on other GPUs are never touched. With ``need_mib`` it stops once
    the load fits; without one (a chat launch prices itself) it stops after the first step that freed something.
    "Keep model in GPU memory" does not protect a model here, as with Off.
    """
    target = set(gpu_ids) if gpu_ids is not None else None
    freed: list[str] = []

    def done() -> bool:
        if need_mib:
            return can_load_alongside(owner, need_mib)[0]
        return bool(freed)

    def overlapping() -> list[str]:
        return [n for n, gpus in arb._view().items() if n != owner and _overlaps(gpus, target)]

    # 1. Residents still loading can't be sized.
    for name in overlapping():
        loading = (
            chat_load_in_flight()
            if name == arb.CHAT
            else name in MEDIA_OWNERS and _media_busy(name)[0]
        )
        if loading:
            _evict_resident(name, acting, evicted, freed)
    if freed and done():
        return freed
    # 2. Idle H3 video servers sit outside every owner's teardown.
    if owner != arb.VIDEO:
        try:
            from core.inference.video_minimax_h3 import release_h3_native_servers
            if release_h3_native_servers(reason):
                freed.append("video-server")
        except Exception as exc:  # noqa: BLE001
            logger.warning("gpu_residency: could not release the idle video server: %s", exc)
        if freed and done():
            return freed
    # 3. The idle image or video model on these GPUs.
    for name in overlapping():
        if name in MEDIA_OWNERS and not any(_media_busy(name)) and not foreign_on(name, acting):
            _evict_resident(name, acting, evicted, freed)
            if done():
                return freed
    # 4. Idle chat models kept alongside, least recently used first.
    if arb.CHAT in overlapping() or owner == arb.CHAT:
        from core.inference import model_slots

        short = short_mib or (
            max((can_load_alongside(owner, need_mib)[1].get("short_mib") or {0: 0}).values())
            if need_mib
            else 0
        )
        victims = model_slots.eviction_victims(None, short, gpu_ids)
        if victims:
            slots = [_kept_slot_report(victim) for victim in victims]
            model_slots.evict_these(victims)
            _record_evicted(slots, evicted, freed)
            if done():
                return freed
    # 5. The active chat model.
    if owner != arb.CHAT and arb.CHAT in overlapping():
        _evict_resident(arb.CHAT, acting, evicted, freed)
    return freed


def _kept_slot_report(slot: Any) -> dict:
    try:
        from core.inference import model_slots
        report = getattr(model_slots, "slot_report", None)
        if report is not None:
            return report(slot)
    except Exception:  # noqa: BLE001
        pass
    request = getattr(slot, "request", None)
    return {"kind": "chat", "model": str(getattr(request, "model_path", "") or "")}


def media_fit_kwargs(
    owner: str,
    request: Any,
    gpu_ordinal: Optional[int],
    policy: Optional[str] = None,
) -> dict:
    """The acquire_for keywords for an image or video load: the policy and, when known, what it will hold."""
    policy = policy or current_policy()
    if policy == "off":
        return {}
    need = media_load_need_mib(owner, request, gpu_ordinal)
    return {"policy": policy, **({"need_mib": need} if need else {})}


def free_for(
    owner: str,
    gpu_ids: Optional[Iterable[int]],
    reason: str,
    evicted: Optional[list] = None,
    *,
    need_mib: Optional[dict] = None,
    short_mib: int = 0,
) -> list[str]:
    """Free residents on ``gpu_ids`` so ``owner`` fits, cheapest first. Appends {kind, model} to ``evicted``."""
    from utils.account_context import current_account_id

    acting = current_account_id()
    with arb._lock:
        arb._sync_from_view()
        return _free_steps(
            owner, gpu_ids, reason, evicted, need_mib = need_mib, short_mib = short_mib, acting = acting
        )


def coreside_locked(
    owner: str,
    *,
    policy: str,
    need_mib: Optional[dict],
    gpu_ids: Optional[Iterable[int]],
    allow_evict: bool,
    evict_first: bool,
    acting: Optional[str],
    evicted: Optional[list],
) -> None:
    """Make room for ``owner`` under When they fit / Always, evicting only what has to go. Caller holds the lock.

    Residents on other GPUs stay. A chat load joins priced image and video models and lets its launch price them. An
    image or video load joins when it fits, else When they fit frees the cheapest residents and Always refuses.
    Anything that can't be priced is evicted first under When they fit, as with Off.
    """
    target = (
        arb.physical_gpu_ids(gpu_ids)
        if gpu_ids is not None
        else (frozenset(need_mib) if need_mib else None)
    )
    overlap = [
        name for name, gpus in arb._view().items() if name != owner and _overlaps(gpus, target)
    ]
    for name in [n for n in overlap if n not in arb._KEPT_UNDER_ALWAYS]:
        # A Decision API server is evicted as before.
        if not allow_evict:
            raise arb.GpuOwnerBusyError(name)
        _evict_resident(name, acting, evicted, [])
    kept = [n for n in overlap if n in arb._KEPT_UNDER_ALWAYS]

    def evict_or_refuse(names: list[str]) -> None:
        if not names:
            return
        if policy == "always":
            raise arb.refusal_for(
                [slot for n in names for slot in arb.resident_slots(n)], arb.KIND_LABELS[names[0]]
            )
        if not allow_evict:
            raise arb.GpuOwnerBusyError(names[-1])
        for name in names:
            _evict_resident(name, acting, evicted, [])
            kept.remove(name)

    if not kept:
        return
    if owner in MEDIA_OWNERS and arb.CHAT in kept and chat_load_in_flight():
        # Its size isn't known until it launches.
        if policy == "always":
            raise ChatLoadInFlightRefusal()
        evict_or_refuse([arb.CHAT])
    if owner == arb.CHAT and not need_mib:
        if evict_first:
            # No estimate before launch (a non-GGUF load, or a load with no retry to make room).
            return evict_or_refuse(list(kept))
        records = _media_records()
        evict_or_refuse([n for n in kept if n not in records or _media_busy(n)[0]])
        if kept and _free_mib_by_gpu()[0] is None:
            evict_or_refuse(list(kept))
        return
    if need_mib is None or evict_first:
        return evict_or_refuse(list(kept))
    fits, info = can_load_alongside(owner, need_mib)
    if fits:
        return
    if policy == "always":
        raise need_free_refusal(info, [slot for n in kept for slot in arb.resident_slots(n)])
    if not allow_evict:
        raise arb.GpuOwnerBusyError(kept[-1])
    _free_steps(owner, target, f"GPU memory for {owner}", evicted, need_mib = need_mib, acting = acting)
    if not can_load_alongside(owner, need_mib)[0]:
        # Still short: unload the rest on these GPUs, as Off does, and let the planner offload.
        for name in [
            n for n, gpus in arb._view().items() if n != owner and _overlaps(gpus, target)
        ]:
            _evict_resident(name, acting, evicted, [])
