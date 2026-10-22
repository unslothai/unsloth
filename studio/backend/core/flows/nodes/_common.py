# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers handlers share: model residency, LLM text and JSON, image generation, background removal, speech and media
bytes. Every Studio capability goes through ``ctx.call`` (in-process routes as the run's key); model loads reuse the
MCP server's loaders so a flow loads models exactly as an agent would."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import uuid
from typing import Any, Optional
from urllib.parse import quote

from core.flows import assets as asset_store
from core.flows.registry import FlowNodeError

IMAGE_PROGRESS_POLL_S = 1.0
LOAD_DEADLINE_S = 4 * 3600
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")

GALLERY_FILE = {
    "image": "/api/inference/images/gallery/{id}/file",
    "audio": "/v1/audio/gallery/{id}/file",
    "video": "/api/inference/video/gallery/{id}/file",
}


# ---------------------------------------------------------------- values


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    if isinstance(value, dict) and value.get("kind") in ("image", "audio", "video"):
        return value.get("prompt") or f"[{value['kind']}]"
    return json.dumps(value, ensure_ascii = False)


def lookup(value: Any, path: str) -> Any:
    """``a.b.0.c`` into dicts and lists; None when a step is missing."""
    current = value
    for part in [p for p in path.split(".") if p != ""]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def template_vars(ctx: Any) -> dict[str, Any]:
    """What ``{{name}}`` can name: flow params, then every input port, and fields of JSON inputs by path."""
    found: dict[str, Any] = dict(ctx.flow_params or {})
    for key, value in ctx.inputs.items():
        found[key] = value
        if isinstance(value, list) and len(value) == 1:
            found[key] = value[0]
    # Multi inputs fed by labelled nodes are also reachable by the source node's label or id.
    return found


def render(template: str, variables: dict[str, Any]) -> str:
    def sub(match: re.Match) -> str:
        name = match.group(1)
        head, _, rest = name.partition(".")
        if head not in variables:
            return match.group(0)
        value = variables[head]
        if rest:
            value = lookup(value, rest)
        return as_text(value)

    return _PLACEHOLDER.sub(sub, template or "")


def parse_size(size: Any, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    if isinstance(size, str):
        match = re.match(r"^\s*(\d{2,5})\s*x\s*(\d{2,5})\s*$", size)
        if match:
            return int(match.group(1)), int(match.group(2))
    return default


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def first_json(text: str) -> Any:
    """The first JSON object or array in ``text`` (models often wrap JSON in prose or fences)."""
    text = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except ValueError:
                            break
            start = text.find(opener, start + 1)
    raise FlowNodeError("The model did not answer with JSON")


def gallery_media(kind: str, row: dict, **extra: Any) -> dict:
    media_id = str(row["id"])
    value = {
        "kind": kind,
        "ref": f"{kind}:{media_id}",
        "url": row.get("url") or GALLERY_FILE[kind].format(id = quote(media_id, safe = "")),
    }
    for key in ("width", "height", "seed", "mime", "duration_s", "prompt"):
        if row.get(key) is not None:
            value[key] = row[key]
    value.update({k: v for k, v in extra.items() if v is not None})
    return value


# ---------------------------------------------------------------- media bytes


async def media_bytes(ctx: Any, media: Any) -> tuple[bytes, str]:
    if not isinstance(media, dict) or not isinstance(media.get("ref"), str):
        raise FlowNodeError("Expected an image, audio clip or video")
    ref = media["ref"]
    digest = asset_store.parse_ref(ref)
    if digest is not None:
        got = await asyncio.to_thread(asset_store.read, digest)
        if got is None:
            raise FlowNodeError("This flow's embedded file is missing; re-import the flow")
        return got
    kind, _, media_id = ref.partition(":")
    if kind in GALLERY_FILE and media_id:
        path = GALLERY_FILE[kind].format(id = quote(media_id, safe = ""))
    elif kind == "library" and isinstance(media.get("url"), str) and media["url"].startswith("/"):
        path = media["url"]
    else:
        raise FlowNodeError(f"Cannot read {ref!r}")
    data = await ctx.call("GET", path)
    if not isinstance(data, (bytes, bytearray)):
        raise FlowNodeError(f"Could not read {ref!r}")
    mime = asset_store.sniff_mime(bytes(data)) or media.get("mime") or "application/octet-stream"
    return bytes(data), mime


async def data_url(ctx: Any, media: dict) -> str:
    data, mime = await media_bytes(ctx, media)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ---------------------------------------------------------------- models


def _caller(ctx: Any) -> Any:
    return ctx.caller


async def resident_name(ctx: Any, model_kind: str) -> Optional[str]:
    from core.flows.engine import MODEL_SLOT
    from studio_mcp import loading

    slot = MODEL_SLOT.get(model_kind, model_kind)
    if slot in ctx.resident:
        return ctx.resident[slot]
    try:
        slots = await loading.resident_slots(_caller(ctx))
    except Exception:
        return None
    names = slots.get(slot) or []
    name = names[0] if names else None
    ctx.resident[slot] = name
    return name


async def ensure_model(ctx: Any, model_kind: str, choice: Optional[dict]) -> Optional[dict]:
    """Load ``choice`` into its slot unless it is already resident; None keeps what is loaded."""
    from core.flows.engine import LOAD_KIND, MODEL_SLOT
    from studio_mcp import loading

    slot = MODEL_SLOT.get(model_kind, model_kind)
    repo = choice.get("repo") if isinstance(choice, dict) else None
    variant = choice.get("variant") if isinstance(choice, dict) else None
    if not repo:
        if model_kind in ("chat", "vision", "tts", "audio", "image", "video"):
            name = await resident_name(ctx, model_kind)
            if not name:
                what = {"image": "image", "video": "video"}.get(model_kind, "chat")
                raise FlowNodeError(
                    f"No {what} model is loaded. Pick a model on this node, or load one first."
                )
        return None
    if slot in ("embedding",):
        return {"repo": repo, "variant": variant}
    resident = await resident_name(ctx, model_kind)
    if resident and resident.lower() == repo.lower() and not variant:
        return {"repo": repo, "variant": variant}
    ctx.set_status("loading_model", f"Loading {repo}{':' + variant if variant else ''}")
    deadline = time.monotonic() + LOAD_DEADLINE_S
    try:
        if slot == "llm":
            await loading.load_llm(
                _caller(ctx),
                None,
                model = repo,
                variant = variant,
                max_seq_length = None,
                load_in_4bit = True,
                hf_token = None,
                kind = LOAD_KIND.get(model_kind, "llm"),
                deadline = deadline,
            )
        elif slot in ("image", "video"):
            await loading.load_media(
                _caller(ctx),
                None,
                kind = slot,
                model = repo,
                variant = variant,
                hf_token = None,
                deadline = deadline,
            )
        elif slot == "stt":
            await loading.load_stt(_caller(ctx), None, model = repo, variant = variant, hf_token = None)
    except FlowNodeError:
        raise
    except Exception as exc:
        raise FlowNodeError(f"Could not load {repo}: {exc}") from None
    # Another slot's model may have been evicted; read them again next time.
    ctx.resident.clear()
    ctx.resident[slot] = repo
    ctx.set_status("running")
    return {"repo": repo, "variant": variant}


# ---------------------------------------------------------------- LLM


async def _images_content(ctx: Any, prompt: str, images: Optional[list[dict]]) -> Any:
    if not images:
        return prompt
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": await data_url(ctx, image)}})
    return parts


def _messages(system: str, content: Any) -> list[dict]:
    turns: list[dict] = []
    if system:
        turns.append({"role": "system", "content": system})
    turns.append({"role": "user", "content": content})
    return turns


async def _chat_body(
    ctx: Any,
    *,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    images: Optional[list[dict]],
    thinking: bool,
    seed: Optional[int],
    stream: bool,
) -> dict:
    cancel_id = f"flow-{uuid.uuid4().hex[:24]}"
    body: dict[str, Any] = {
        "model": "default",
        "messages": _messages(system, await _images_content(ctx, prompt, images)),
        "stream": stream,
        "enable_tools": False,
        "cancel_id": cancel_id,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "enable_thinking": bool(thinking),
    }
    if seed is not None:
        body["seed"] = int(seed)

    async def cancel() -> None:
        await ctx.call("POST", "/api/inference/cancel", json = {"cancel_id": cancel_id})

    ctx.on_cancel(cancel)
    return body


async def llm_text(
    ctx: Any,
    *,
    model: Optional[dict],
    system: str,
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    images: Optional[list[dict]] = None,
    thinking: bool = False,
    seed: Optional[int] = None,
    stream_port: Optional[str] = "text",
) -> str:
    from core.flows import caller as flow_caller
    from state.tool_policy import tools_force_disabled

    kind = "vision" if images else "chat"
    await ensure_model(ctx, kind, model)
    body = await _chat_body(
        ctx,
        system = system,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        images = images,
        thinking = thinking,
        seed = seed,
        stream = bool(stream_port),
    )
    ctx.check_cancelled()
    if not stream_port:
        with tools_force_disabled():
            payload = await ctx.call("POST", "/v1/chat/completions", json = body)
        return strip_think(_message_text(payload))
    parts: list[str] = []
    in_think = False
    try:
        with tools_force_disabled():
            async for chunk in flow_caller.stream_sse(ctx.caller, "/v1/chat/completions", body):
                ctx.check_cancelled()
                if isinstance(chunk.get("error"), dict):
                    raise FlowNodeError(str(chunk["error"].get("message") or "The model failed"))
                for choice in chunk.get("choices") or []:
                    delta = (choice or {}).get("delta") or {}
                    piece = delta.get("content")
                    if not isinstance(piece, str) or not piece:
                        continue
                    parts.append(piece)
                    # Hide inline thinking from the live preview.
                    if "<think>" in piece:
                        in_think = True
                    if not in_think:
                        ctx.text_delta(stream_port, piece)
                    if "</think>" in piece:
                        in_think = False
    except flow_caller.RouteError as exc:
        raise FlowNodeError(exc.detail, status = exc.status) from None
    return strip_think("".join(parts))


def _message_text(payload: Any) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise FlowNodeError("The model returned no reply")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


async def llm_json(
    ctx: Any,
    *,
    model: Optional[dict],
    system: str,
    prompt: str,
    schema: dict,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    images: Optional[list[dict]] = None,
) -> Any:
    from state.tool_policy import tools_force_disabled

    kind = "vision" if images else "chat"
    await ensure_model(ctx, kind, model)
    body = await _chat_body(
        ctx,
        system = system,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        images = images,
        thinking = False,
        seed = seed,
        stream = False,
    )
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "flow_output", "schema": schema, "strict": True},
    }
    try:
        with tools_force_disabled():
            payload = await ctx.call("POST", "/v1/chat/completions", json = body)
        return first_json(_message_text(payload))
    except FlowNodeError as exc:
        if exc.status not in (400, 422, None) and "JSON" not in str(exc):
            raise
        ctx.log(f"Structured output unavailable ({exc}); asking for plain JSON instead", "warning")
    body.pop("response_format", None)
    body["messages"] = _messages(
        (system + "\n\n" if system else "")
        + "Answer with one JSON value only, no prose, matching this JSON schema:\n"
        + json.dumps(schema),
        body["messages"][-1]["content"],
    )
    body["cancel_id"] = f"flow-{uuid.uuid4().hex[:24]}"
    with tools_force_disabled():
        payload = await ctx.call("POST", "/v1/chat/completions", json = body)
    return first_json(_message_text(payload))


# ---------------------------------------------------------------- images


async def generate_image(
    ctx: Any,
    *,
    model: Optional[dict],
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    init_image: Optional[dict] = None,
    strength: Optional[float] = None,
    references: Optional[list[dict]] = None,
    workflow: Optional[str] = None,
    count: int = 1,
    upscale: Optional[float] = None,
) -> list[dict]:
    await ensure_model(ctx, "image", model)
    if not (prompt or "").strip():
        raise FlowNodeError("The image prompt is empty")
    body: dict[str, Any] = {"prompt": prompt, "width": int(width), "height": int(height)}
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if steps:
        body["steps"] = int(steps)
    if guidance is not None:
        body["guidance"] = float(guidance)
    count = max(1, min(int(count or 1), 8))
    base_seed = int(seed) if seed is not None else None
    if count > 1:
        start = base_seed if base_seed is not None else 0
        body["seeds"] = [(start + i) % (2**32) for i in range(count)]
    elif base_seed is not None:
        body["seed"] = base_seed
    if init_image is not None:
        body["init_image"] = await data_url(ctx, init_image)
        if strength is not None and workflow is None:
            body["strength"] = float(strength)
        body.pop("width", None)
        body.pop("height", None)
    if references:
        body["reference_images"] = [await data_url(ctx, r) for r in references]
    if workflow:
        body["workflow"] = workflow
    if upscale:
        body["upscale"] = float(upscale)

    async def cancel() -> None:
        await ctx.call("POST", "/api/inference/images/generate/cancel")

    ctx.on_cancel(cancel)

    async def poll() -> None:
        seen_preview = None
        while True:
            await asyncio.sleep(IMAGE_PROGRESS_POLL_S)
            try:
                progress = await ctx.call("GET", "/api/inference/images/generate-progress")
            except Exception:
                continue
            if not isinstance(progress, dict) or not progress.get("active", True):
                continue
            preview = progress.get("preview")
            if isinstance(preview, str) and preview and not preview.startswith("data:"):
                preview = f"data:image/jpeg;base64,{preview}"
            if preview == seen_preview:
                preview = None
            else:
                seen_preview = preview
            fraction = progress.get("fraction")
            ctx.progress(
                float(fraction) if isinstance(fraction, (int, float)) else None,
                phase = progress.get("phase") if isinstance(progress.get("phase"), str) else None,
                step = progress.get("step") if isinstance(progress.get("step"), int) else None,
                total = progress.get("total_steps")
                if isinstance(progress.get("total_steps"), int)
                else None,
                preview = preview if isinstance(preview, str) and len(preview) < 400_000 else None,
            )

    poller = asyncio.create_task(poll())
    try:
        payload = await ctx.call("POST", "/api/inference/images/generate", json = body)
    finally:
        poller.cancel()
    rows = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FlowNodeError("Unsloth Studio returned no images")
    return [
        gallery_media("image", row, prompt = prompt)
        for row in rows
        if isinstance(row, dict) and row.get("id")
    ]


def _decode_data_url(value: str) -> bytes:
    _, _, payload = value.partition(",")
    return base64.b64decode(payload or value)


def _trim(im: Any) -> Any:
    bbox = im.getchannel("A").getbbox()
    if bbox:
        pad = 4
        left, top, right, bottom = bbox
        return im.crop(
            (
                max(0, left - pad),
                max(0, top - pad),
                min(im.width, right + pad),
                min(im.height, bottom + pad),
            )
        )
    return im


def chroma_key(data: bytes, tolerance: int = 40) -> tuple[bytes, bytes]:
    """Cut out the subject by removing the border colour (a fallback when SAM 3 is not available)."""
    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = im.size
    work = im.convert("RGB")
    mask = Image.new("L", (w, h), 255)
    seeds = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, 0),
        (w // 2, h - 1),
        (0, h // 2),
        (w - 1, h // 2),
    ]
    marker = (255, 0, 255)
    for seed in seeds:
        if work.getpixel(seed) == marker:
            continue
        ImageDraw.floodfill(work, seed, marker, thresh = tolerance)
    pixels = work.load()
    mpx = mask.load()
    for y in range(h):
        for x in range(w):
            if pixels[x, y] == marker:
                mpx[x, y] = 0
    im.putalpha(mask)
    out, mask_out = io.BytesIO(), io.BytesIO()
    im.save(out, "PNG")
    mask.save(mask_out, "PNG")
    return out.getvalue(), mask_out.getvalue()


async def remove_background(
    ctx: Any,
    image: dict,
    subject: str,
    trim: bool = True,
) -> tuple[dict, dict]:
    from PIL import Image

    data, mime = await media_bytes(ctx, image)
    try:
        answer = await ctx.call(
            "POST",
            "/api/inference/images/segment",
            json = {
                "image": f"data:{mime};base64,{base64.b64encode(data).decode()}",
                "prompt": subject or "subject",
            },
        )
        mask_value = answer.get("mask") if isinstance(answer, dict) else None
        if not isinstance(mask_value, str):
            raise FlowNodeError("The segmenter returned no mask")

        def cut() -> tuple[bytes, bytes]:
            im = Image.open(io.BytesIO(data)).convert("RGBA")
            mask = Image.open(io.BytesIO(_decode_data_url(mask_value))).convert("L").resize(im.size)
            im.putalpha(mask)
            out, mask_out = io.BytesIO(), io.BytesIO()
            (_trim(im) if trim else im).save(out, "PNG")
            mask.save(mask_out, "PNG")
            return out.getvalue(), mask_out.getvalue()

        cutout, mask_png = await asyncio.to_thread(cut)
    except FlowNodeError as exc:
        if exc.status not in (409, 501, 404, None):
            raise
        ctx.log(f"Segmentation unavailable ({exc}); removing the border colour instead", "warning")

        def keyed() -> tuple[bytes, bytes]:
            cut_png, mask_png = chroma_key(data)
            if trim:
                im = Image.open(io.BytesIO(cut_png))
                out = io.BytesIO()
                _trim(im).save(out, "PNG")
                cut_png = out.getvalue()
            return cut_png, mask_png

        cutout, mask_png = await asyncio.to_thread(keyed)
    return (
        await ctx.save_asset(
            cutout, "image/png", kind = "image", extra = {"prompt": image.get("prompt")}
        ),
        await ctx.save_asset(mask_png, "image/png", kind = "image"),
    )


# ---------------------------------------------------------------- audio


async def audio_ref(ctx: Any, media: dict) -> dict:
    """An AudioSourceRef for ``media``: a history clip by id, else the bytes uploaded as an input."""
    ref = media.get("ref", "") if isinstance(media, dict) else ""
    kind, _, media_id = ref.partition(":")
    if kind == "audio" and media_id:
        return {"clip_id": media_id}
    if kind == "voice" and media_id:
        return {"voice_id": media_id}
    data, _ = await media_bytes(ctx, media)
    answer = await ctx.call(
        "POST",
        "/v1/audio/inputs",
        params = {"name": "flow-input.wav"},
        content = data,
        content_type = "application/octet-stream",
    )
    input_id = answer.get("id") if isinstance(answer, dict) else None
    if not input_id:
        raise FlowNodeError("Could not upload the reference audio")
    return {"input_id": input_id}


async def audio_run(ctx: Any, body: dict) -> list[dict]:
    cancel_id = f"flow-{uuid.uuid4().hex[:24]}"
    body = {**body, "cancel_id": cancel_id}

    async def cancel() -> None:
        await ctx.call("POST", "/api/inference/audio/cancel", json = {"cancel_id": cancel_id})

    ctx.on_cancel(cancel)
    payload = await ctx.call("POST", "/api/inference/audio/run", json = body)
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list) or not clips:
        raise FlowNodeError("Unsloth Studio returned no audio")
    return [
        gallery_media("audio", clip, mime = "audio/wav", prompt = body.get("text"))
        for clip in clips
        if isinstance(clip, dict) and clip.get("id")
    ]


async def speak(
    ctx: Any,
    *,
    model: Optional[dict],
    text: str,
    voice: str = "",
    instructions: str = "",
    reference: Optional[dict] = None,
    seed: Optional[int] = None,
) -> dict:
    await ensure_model(ctx, "tts", model)
    if not (text or "").strip():
        raise FlowNodeError("Nothing to say: the text is empty")
    body: dict[str, Any] = {"workflow": "clone" if reference else "speak", "text": text}
    if reference:
        body["inputs"] = {"reference": await audio_ref(ctx, reference)}
    if voice and not reference:
        body["options"] = {"voice": voice}
    if instructions:
        body["instructions"] = instructions
    if seed is not None:
        body["seed"] = int(seed)
    clips = await audio_run(ctx, body)
    return clips[0]
