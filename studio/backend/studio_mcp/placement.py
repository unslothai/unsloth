# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How and where ``load_model`` puts a model: the media memory knobs, the chat and audio device fields, the placement Unsloth Studio reports back, and ``plan_load``'s read-only preview of what a load would download and unload."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Optional

from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.outputs import Placement, PlanLoadResult, PlanVariant, SlotModel
from studio_mcp.tools import as_dict, integer, opt_text, strings, try_json

CHAT_KINDS = ("llm", "tts", "audio")
# Each media knob and the kinds whose load request takes it.
MEDIA_KNOBS = {
    "memory_mode": ("image", "video"),
    "speed_mode": ("image", "video"),
    "transformer_quant": ("image",),
    "text_encoder_quant": ("image",),
    "h3_task": ("video",),
}
MULTI_MODEL_MODES = ("off", "fit", "always")
STATUS = "/api/inference/status"
MEDIA_STATUS = {"image": "/api/inference/images/status", "video": "/api/inference/video/status"}
MEDIA_PLAN = {
    "image": "/api/inference/images/download-plan",
    "video": "/api/inference/video/download-plan",
}
STT_STATUS = "/api/inference/audio/stt/status"
# audio.cpp folder ids name a GGUF repo but run on audio.cpp, which honours audio_device itself.
AUDIO_CPP_PREFIX = "audio-cpp/audio.cpp-gguf/"
IN_FLIGHT = ("running", "cancelling")
PARTIAL_MESSAGE = "this variant is still downloading or incomplete; wait for it or redownload it"


def media_knobs(kind: str, **knobs: Optional[str]) -> dict[str, str]:
    """The knobs that were set, refused for a kind whose load does not take them."""
    chosen = {name: value for name, value in knobs.items() if value is not None}
    for name in chosen:
        if kind not in MEDIA_KNOBS[name]:
            only = " and ".join(MEDIA_KNOBS[name])
            raise ToolError(f"{name} is only for {only} loads, not kind={kind!r}.")
    return chosen


def is_local(model: str) -> bool:
    return model.startswith(("/", "\\", ".")) or ":" in model[:3]


def is_gguf(model: str, variant: Optional[str]) -> bool:
    return bool(variant) or "gguf" in model.lower()


def check_placement(
    kind: str,
    model: str,
    variant: Optional[str],
    *,
    device: Optional[str],
    alongside: Optional[bool],
    kv_cache_type: Optional[str],
    gpu_layers: Optional[int],
) -> None:
    """Refuse a placement field the load of this kind would ignore, before anything is sent."""
    if kind not in CHAT_KINDS:
        given = {
            "device": device,
            "alongside": alongside,
            "kv_cache_type": kv_cache_type,
            "gpu_layers": gpu_layers,
        }
        for name, value in given.items():
            if value is not None:
                raise ToolError(f"{name} is only for llm, tts and audio loads, not kind={kind!r}.")
        return
    if kind == "llm" and device == "cpu" and not is_gguf(model, variant):
        raise ToolError(
            "device='cpu' needs a GGUF model for kind='llm': Unsloth Studio places only GGUF "
            "chat models on the CPU. Pick a GGUF repo or a variant."
        )
    if gpu_layers is not None and (
        (device == "cpu" and gpu_layers > 0) or (device == "gpu" and gpu_layers == 0)
    ):
        raise ToolError(f"gpu_layers={gpu_layers} contradicts device={device!r}.")


def llm_fields(
    kind: str,
    model: str,
    variant: Optional[str],
    *,
    device: Optional[str],
    alongside: Optional[bool],
    kv_cache_type: Optional[str],
    gpu_layers: Optional[int],
) -> dict[str, Any]:
    """The /api/inference/load fields for the placement asked for."""
    fields: dict[str, Any] = {}
    # audio_device is read by native and audio.cpp audio only; a chat GGUF sent it is refused.
    if kind != "llm" and device is not None:
        fields["audio_device"] = device
    llama_server = is_gguf(model, variant) and not model.lower().startswith(AUDIO_CPP_PREFIX)
    if device == "cpu" and llama_server:
        # No layers, and no speculative drafter, which would put the GPU back to use.
        fields.update(gpu_memory_mode = "manual", gpu_layers = 0, speculative_type = "off")
    elif gpu_layers is not None:
        fields.update(gpu_memory_mode = "manual", gpu_layers = gpu_layers)
    if alongside is not None:
        fields["alongside"] = alongside
    if kv_cache_type is not None:
        fields["cache_type_kv"] = kv_cache_type
    return fields


def multi_model_mode(status: Any) -> Optional[str]:
    mode = as_dict(status).get("multi_model_mode")
    return mode if mode in MULTI_MODEL_MODES else None


async def auto_alongside(
    caller: Caller, device: Optional[str], alongside: Optional[bool]
) -> Optional[bool]:
    """Under Always, a CPU load joins the serving chat model rather than being refused for replacing it."""
    if device != "cpu" or alongside is not None:
        return alongside
    status = await try_json(caller, STATUS)
    if multi_model_mode(status) == "always" and strings(as_dict(status).get("serving")):
        return True
    return None


def llm_placement(answer: Any, device: Optional[str]) -> Optional[Placement]:
    """The device a chat or audio load ended on, from the load answer."""
    answer = as_dict(answer)
    offloaded = integer(answer.get("offloaded_layers"))
    if answer.get("audio_device_fallback") == "cpu" or device == "cpu":
        placed = "cpu"
    elif offloaded is not None:
        placed = "gpu" if offloaded > 0 else "cpu"
    else:
        placed = device if device == "gpu" else None
    return Placement(device = placed) if placed else None


def media_placement(status: Any) -> Optional[Placement]:
    """The placement an image or video status reports, with the device as cpu or gpu."""
    status = as_dict(status)
    if status.get("loaded") is not True:
        return None
    device = opt_text(status.get("device"))
    if device is not None:
        device = "cpu" if device.lower() == "cpu" else "gpu"
    placed = Placement.from_route(status, device = device)
    return placed if any(placed.model_dump().values()) else None


def alongside_applied(before: list[str], after: list[str], loaded: str) -> Optional[bool]:
    """Whether the chat model that served before the load still serves; None when none did."""
    previous = before[0] if before else None
    if previous is None or previous.lower() == loaded.lower():
        return None
    return previous in after


async def post_plan(caller: Caller, path: str, body: dict, knobs: dict) -> Any:
    """The download plan, priced with the knobs; an Unsloth Studio whose plan refuses them is asked without."""
    response = await forward(caller, "POST", path, json_body = {**body, **knobs})
    if knobs and response.status_code == 422:
        response = await forward(caller, "POST", path, json_body = body)
    return raise_for_route(response)


def pick_variant(listing: Any, variant: Optional[str]) -> Optional[dict]:
    """The named variant's row in a gguf-variants listing, else the repo's default."""
    listing = as_dict(listing)
    rows = [row for row in listing.get("variants") or [] if isinstance(row, dict)]
    wanted = (variant or opt_text(listing.get("default_variant")) or "").lower()
    for row in rows:
        if str(row.get("quant", "")).lower() == wanted:
            return row
    return None if variant else (rows[0] if rows else None)


def _split(model: str, variant: Optional[str]) -> tuple[str, Optional[str]]:
    # A GGUF may be named "repo:variant".
    if variant is None and ":" in model and not is_local(model):
        model, _, variant = model.partition(":")
    return model, variant


async def _variants(caller: Caller, model: str) -> Any:
    return await try_json(
        caller, "/api/hub/gguf-variants", params = {"repo_id": model}, hub_header = True
    )


async def refuse_partial_variant(
    caller: Caller,
    model: str,
    kind: str,
    variant: Optional[str],
    hf_token: Optional[str] = None,
) -> None:
    """Refuse a GGUF variant with only part of its files on disk that Unsloth Studio is not downloading itself; loading it fails with a misleading memory error."""
    if kind == "stt" or is_local(model) or model.lower().startswith(AUDIO_CPP_PREFIX):
        return
    repo, variant = _split(model, variant)
    if not is_gguf(repo, variant):
        return
    if hf_token:
        # A gated repo's listing needs the token in the header.
        caller = dataclasses.replace(caller, hf_token = hf_token)
    row = pick_variant(await _variants(caller, repo), variant)
    if row is None or row.get("partial") is not True or row.get("downloaded") is True:
        return
    quant = opt_text(row.get("quant"))
    # A download Unsloth Studio manages is waited on before this, never refused.
    state = await try_json(
        caller, "/api/hub/download-status", params = {"repo_id": repo, "gguf_variant": quant}
    )
    if as_dict(state).get("state") in IN_FLIGHT:
        return
    raise ToolError(
        f"{repo} {quant}: {PARTIAL_MESSAGE}." if quant else f"{repo}: {PARTIAL_MESSAGE}."
    )


def _named(status: Any) -> Optional[str]:
    status = as_dict(status)
    if status.get("loaded") is not True:
        return None
    return opt_text(status.get("display_repo_id")) or opt_text(status.get("repo_id"))


def residents(status: Any, image: Any, video: Any, stt: Any) -> list[SlotModel]:
    slots = [SlotModel(kind = "llm", model = name) for name in strings(as_dict(status).get("serving"))]
    for kind, payload in (("image", image), ("video", video)):
        if _named(payload):
            slots.append(SlotModel(kind = kind, model = _named(payload)))
    for state in as_dict(stt).values():
        if isinstance(state, dict) and opt_text(state.get("loaded_model")):
            slots.append(SlotModel(kind = "stt", model = state["loaded_model"]))
    return slots


def _cpu_primary(status: Any) -> bool:
    # A primary that holds no VRAM gives up the GPU claim, so a media load leaves it alone.
    status = as_dict(status)
    manual = status.get("gpu_memory_mode") == "manual" and status.get("gpu_layers") == 0
    return manual or status.get("audio_device_fallback") == "cpu"


def predict(
    *,
    kind: str,
    model: str,
    device: Optional[str],
    alongside: Optional[bool],
    mode: Optional[str],
    status: Any,
    slots: list[SlotModel],
    has_vision: bool = False,
) -> tuple[list[SlotModel], bool, bool]:
    """(would_evict, would_evict_exact, would_refuse) for one load. Exact under Off; under When they fit and Always, or when the mode is unknown, an upper bound."""
    slot_kind = "llm" if kind in CHAT_KINDS else kind
    wanted = model.split(":", 1)[0].lower() if not is_local(model) else model.lower()
    if kind == "stt" or any(
        s.kind == slot_kind and s.model.split(":", 1)[0].lower() == wanted for s in slots
    ):
        return [], True, False
    chat = [s for s in slots if s.kind == "llm"]
    media = [s for s in slots if s.kind in ("image", "video")]
    exact = True
    if kind in CHAT_KINDS:
        gpu = device != "cpu" or has_vision
        joins = (
            alongside
            if alongside is not None
            else (mode == "always" and device == "cpu" and bool(chat))
        )
        # Off replaces the primary and unloads the idle kept models whatever alongside says.
        off_set = chat + (media if gpu else [])
        upper = [] if joins and not gpu else off_set
        same = []
    else:
        same = [s for s in media if s.kind == kind]
        held = chat
        if chat and _cpu_primary(status):
            held, exact = chat[1:], False
        off_set = held + media
        upper = chat + media
    if mode == "off":
        return off_set, exact, False
    if mode == "fit":
        return upper, False, False
    if mode == "always":
        return [], False, any(s not in same for s in upper)
    # Unknown mode (an Unsloth Studio that does not report it): whatever either rule could unload.
    return upper + [s for s in off_set if s not in upper], False, False


async def _post_json(caller: Caller, path: str, body: dict) -> Any:
    try:
        response = await forward(caller, "POST", path, json_body = body, hub_header = True)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


async def plan_load(
    caller: Caller,
    *,
    model: str,
    kind: str,
    variant: Optional[str],
    device: Optional[str],
    alongside: Optional[bool],
) -> PlanLoadResult:
    status, image, video, stt = await asyncio.gather(
        try_json(caller, STATUS),
        try_json(caller, MEDIA_STATUS["image"]),
        try_json(caller, MEDIA_STATUS["video"]),
        try_json(caller, STT_STATUS),
    )
    repo, variant = _split(model, variant)
    rows: list[dict] = []
    row = listing = None
    if kind != "stt" and is_gguf(repo, variant) and not is_local(repo):
        listing = as_dict(await _variants(caller, repo))
        rows = [r for r in listing.get("variants") or [] if isinstance(r, dict)]
        row = pick_variant(listing, variant)
        if variant and row is None:
            raise ToolError(f"{repo} has no GGUF variant {variant}")
    download = est = None
    if row is not None:
        download = (
            0
            if row.get("downloaded") is True
            else (
                integer(row.get("download_remaining_bytes"))
                or integer(row.get("download_size_bytes"))
                or integer(row.get("size_bytes"))
            )
        )
    if kind in MEDIA_PLAN:
        body: dict[str, Any] = {"model_path": repo}
        if row is not None and opt_text(row.get("filename")):
            body.update(gguf_filename = row["filename"], model_kind = "gguf")
        plan = as_dict(await _post_json(caller, MEDIA_PLAN[kind], body))
        download = integer(plan.get("total_bytes"))
    elif row is not None and row.get("downloaded") is True:
        estimate = as_dict(
            await _post_json(
                caller,
                "/api/inference/estimate-memory",
                {"model_path": repo, "gguf_variant": opt_text(row.get("quant"))},
            )
        )
        est = integer(estimate.get("gpu_bytes")) if estimate.get("available") is True else None
    slots = residents(status, image, video, stt)
    mode = multi_model_mode(status)
    would_evict, exact, refuse = predict(
        kind = kind,
        model = repo,
        device = device,
        alongside = alongside,
        mode = mode,
        status = status,
        slots = slots,
        has_vision = as_dict(listing).get("has_vision") is True,
    )
    return PlanLoadResult(
        model = repo,
        kind = kind,
        variants = [
            PlanVariant.from_route(r, variant = r["quant"]) for r in rows if opt_text(r.get("quant"))
        ],
        download_bytes = download,
        est_gpu_bytes = est,
        residents = slots,
        multi_model_mode = mode,
        would_evict = would_evict,
        would_evict_exact = exact,
        would_refuse = refuse,
    )
