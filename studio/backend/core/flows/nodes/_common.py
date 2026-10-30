# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers handlers share: model residency, LLM text and JSON, image generation, background removal, speech and media
bytes. Every Studio capability goes through ``ctx.call`` (in-process routes as the run's key); model loads reuse the
MCP server's loaders so a flow loads models exactly as an agent would."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Optional
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


def visible_so_far(text: str) -> str:
    """What a streamed reply can show yet: closed think blocks dropped, an open one cut, a half-sent tag held back."""
    shown = _THINK.sub("", text)
    lowered = shown.lower()
    cut = lowered.find("<think>")
    if cut >= 0:
        return shown[:cut]
    for n in range(min(len("<think>") - 1, len(shown)), 0, -1):
        if "<think>".startswith(lowered[-n:]):
            return shown[:-n]
    return shown


def first_json(text: str) -> Any:
    """The first JSON object or array in ``text`` (models often wrap JSON in prose or fences)."""
    text = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    pairs = sorted((("{", "}"), ("[", "]")), key = lambda p: (text.find(p[0]) < 0, text.find(p[0])))
    for opener, closer in pairs:
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


def gallery_media_present(value: Any) -> bool:
    """False when a value holds an image, clip or video whose gallery entry is gone (the user deleted it)."""
    from core.inference import audio_gallery, image_gallery, video_gallery

    owned = {
        "image": image_gallery.owned_image_path,
        "audio": audio_gallery.owned_audio_path,
        "video": video_gallery.owned_video_path,
    }
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            ref = item.get("ref")
            kind, _, media_id = ref.partition(":") if isinstance(ref, str) else ("", "", "")
            if (
                kind in owned
                and media_id
                and item.get("kind") == kind
                and owned[kind](media_id) is None
            ):
                return False
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return True


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


def _caller(ctx: Any) -> Any:
    return ctx.caller


# A flow's prompts and JSON answers fit easily; the model's full context (262k for Qwen3) would reserve a ~36 GB KV cache.
FLOW_CHAT_CONTEXT = 16384


def chat_model_name(ctx: Any) -> str:
    """The model a chat request names: the model this step bound to when it is kept alongside the primary one
    (Keep multiple models loaded), else "default", the primary."""
    return getattr(ctx, "chat_model", None) or "default"


MULTI_MODEL_MODES = ("fit", "always")


async def _alongside(
    ctx: Any, repo: str, variant: Optional[str], options: Optional[dict] = None
) -> Optional[dict]:
    """Under Keep multiple models loaded (fit or always), serve ``repo`` from its own slot beside the primary chat
    model instead of replacing it: bind to it when it is already kept alongside, else load it alongside. None means
    the setting is off (or unreadable) and the step loads into the primary slot as before."""
    from studio_mcp import loading

    try:
        _slots, status = await loading._resident(_caller(ctx))
    except Exception:
        return None
    if status.get("multi_model_mode") not in MULTI_MODEL_MODES:
        return None
    serving = [s for s in status.get("serving") or [] if isinstance(s, str)]
    if not serving or loading._same_id(serving[0], repo):
        return None
    name = f"{repo}:{variant}" if variant else repo
    if any(loading._same_id(s, repo) for s in serving[1:]):
        kept = await loading._read_json(
            _caller(ctx), f"/api/inference/status?model={quote(repo, safe='')}"
        )
        held = kept.get("gguf_variant") if isinstance(kept, dict) else None
        if variant and isinstance(held, str) and held.lower() != variant.lower():
            raise FlowNodeError(
                f"{repo} is already loaded alongside as {held}. Two quantizations of one model cannot be kept "
                f"loaded at once: pick {held} on this node, or unload it first."
            )
        ctx.chat_model = name
        return {"repo": repo, "variant": variant}
    if getattr(ctx, "gpu_shared", False):
        from core.flows.engine import NeedsExclusiveGpu
        raise NeedsExclusiveGpu()
    ctx.set_status("loading_model", f"Loading {name} alongside {serving[0]}")
    try:
        await loading.load_llm(
            _caller(ctx),
            None,
            model = repo,
            variant = variant,
            max_seq_length = _load_context(options),
            load_in_4bit = True,
            hf_token = None,
            deadline = time.monotonic() + LOAD_DEADLINE_S,
            alongside = True,
            extra_body = {k: v for k, v in (options or {}).items() if k != "max_seq_length"} or None,
        )
    except FlowNodeError:
        raise
    except Exception as exc:
        raise FlowNodeError(f"Could not load {repo} alongside {serving[0]}: {exc}") from None
    ctx.resident.clear()
    ctx.chat_model = name
    ctx.set_status("running")
    return {"repo": repo, "variant": variant}


async def resident_name(ctx: Any, model_kind: str) -> Optional[str]:
    from core.flows.engine import MODEL_SLOT
    from studio_mcp import loading

    slot = MODEL_SLOT.get(model_kind, model_kind)
    try:
        slots = await loading.resident_slots(_caller(ctx))
    except Exception:
        return None
    names = slots.get(slot) or []
    name = names[0] if names else None
    ctx.resident[slot] = name
    return name


_LOADED_VARIANT: dict[str, tuple[str, str]] = {}
_LOADED_OPTIONS: dict[str, tuple[str, str, int]] = {}


def options_hash(options: Optional[dict]) -> Optional[str]:
    if not options:
        return None
    return hashlib.sha256(json.dumps(options, sort_keys = True, default = str).encode()).hexdigest()[
        :16
    ]


async def _resident_variant(ctx: Any, slot: str, repo: str) -> Optional[str]:
    if slot != "llm":
        loaded = _LOADED_VARIANT.get(slot)
        return loaded[1] if loaded and loaded[0] == repo.lower() else None
    from studio_mcp import loading

    try:
        _slots, status = await loading._resident(_caller(ctx))
    except Exception:
        return None
    variant = status.get("gguf_variant")
    return variant.lower() if isinstance(variant, str) else None


async def resident_gguf_variant(ctx: Any) -> Optional[str]:
    """The quant the loaded chat model runs, as its status spells it, or None."""
    from studio_mcp import loading

    try:
        _slots, status = await loading._resident(_caller(ctx))
    except Exception:
        return None
    variant = status.get("gguf_variant") if isinstance(status, dict) else None
    return variant if isinstance(variant, str) and variant else None


async def ensure_model(
    ctx: Any,
    model_kind: str,
    choice: Optional[dict],
    options: Optional[dict] = None,
) -> Optional[dict]:
    """Load ``choice`` into its slot unless it is already resident; None keeps what is loaded.

    ``options`` are load settings merged into the chat load body (context, KV cache type, offload, speculative
    decoding); the model counts as resident only when it was loaded with the same settings."""
    found = await _ensure_model(ctx, model_kind, choice, options)
    requires = (getattr(ctx, "spec", None) or {}).get("requires")
    if requires:
        await _check_backend(ctx, model_kind, requires)
    return found


async def _check_backend(ctx: Any, model_kind: str, requires: list[str]) -> None:
    from core.flows.engine import MODEL_SLOT
    from studio_mcp import loading

    if MODEL_SLOT.get(model_kind, model_kind) != "llm":
        return
    title = (getattr(ctx, "spec", None) or {}).get("title") or "This step"
    try:
        _slots, status = await loading._resident(_caller(ctx))
    except Exception:
        status = None
    if not isinstance(status, dict) or not isinstance(status.get("is_gguf"), bool):
        raise FlowNodeError(
            f"{title} could not check which backend the model loaded on; try again", retryable = True
        )
    backend = "gguf" if status["is_gguf"] else "transformers"
    if backend not in requires:
        want = " or ".join("a GGUF" if r == "gguf" else "a transformers" for r in requires)
        have = "GGUF" if backend == "gguf" else "transformers"
        raise FlowNodeError(f"{title} needs {want} model, not {have}")


async def _ensure_model(
    ctx: Any, model_kind: str, choice: Optional[dict], options: Optional[dict]
) -> Optional[dict]:
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
    if options and slot != "llm":
        raise FlowNodeError("Load settings apply to chat models only")
    if slot == "llm":
        ctx.chat_model = None
    resident = await resident_name(ctx, model_kind)
    if (
        slot == "llm"
        and model_kind in ("chat", "vision")
        and not (resident and resident.lower() == repo.lower())
    ):
        # The card's load settings ride along, so they never cost the resident model its slot.
        kept = await _alongside(ctx, repo, variant, options)
        if kept is not None:
            return kept
    wanted = (repo.lower(), (variant or "").lower())
    settings = options_hash(options)
    from core.inference import gpu_arbiter

    loaded_with = _LOADED_OPTIONS.get(slot)
    if loaded_with is not None and loaded_with[2] != gpu_arbiter.load_count(gpu_arbiter.CHAT):
        _LOADED_OPTIONS.pop(slot, None)
        loaded_with = None
    same_settings = (
        loaded_with is None or loaded_with[0] != repo.lower()
        if settings is None
        else loaded_with is not None and loaded_with[:2] == (repo.lower(), settings)
    )
    if (
        resident
        and resident.lower() == repo.lower()
        and (not variant or await _resident_variant(ctx, slot, repo) == variant.lower())
        and same_settings
    ):
        return {"repo": repo, "variant": variant}
    if getattr(ctx, "gpu_shared", False):
        from core.flows.engine import NeedsExclusiveGpu
        raise NeedsExclusiveGpu()
    ctx.set_status("loading_model", f"Loading {repo}{':' + variant if variant else ''}")
    deadline = time.monotonic() + LOAD_DEADLINE_S
    try:
        if slot == "llm":
            await loading.load_llm(
                _caller(ctx),
                None,
                model = repo,
                variant = variant,
                max_seq_length = _load_context(options),
                load_in_4bit = True,
                hf_token = None,
                kind = LOAD_KIND.get(model_kind, "llm"),
                deadline = deadline,
                extra_body = {k: v for k, v in (options or {}).items() if k != "max_seq_length"}
                or None,
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
    ctx.resident.clear()
    ctx.resident[slot] = repo
    note_loaded(ctx, slot, repo)
    _LOADED_VARIANT[slot] = wanted
    if settings is None:
        _LOADED_OPTIONS.pop(slot, None)
    else:
        _LOADED_OPTIONS[slot] = (repo.lower(), settings, gpu_arbiter.load_count(gpu_arbiter.CHAT))
    ctx.set_status("running")
    return {"repo": repo, "variant": variant}


def _load_context(options: Optional[dict]) -> int:
    """An explicit 0 hands sizing to llama.cpp's fit (Manual GPU memory with Auto layers), so only an absent
    context falls back to the Flows default."""
    value = (options or {}).get("max_seq_length")
    return FLOW_CHAT_CONTEXT if value is None else int(value)


def note_loaded(ctx: Any, slot: str, repo: str) -> None:
    note = getattr(ctx, "note_loaded", None)
    if callable(note):
        note(slot, repo)


async def poll_job(
    ctx: Any,
    *,
    status: Callable[[], Awaitable[Any]],
    parse: Callable[[Any], tuple[str, Optional[dict], Optional[str]]],
    cancel: Optional[Callable[[], Awaitable[None]]] = None,
    interval: float = 5.0,
) -> Any:
    """Follow a started Studio job until it ends and return its last status payload.

    ``parse(payload)`` gives (state, progress, error): state is running, completed, failed or cancelled, progress the
    ``ctx.progress`` keywords (or None) and error the failure message. It may raise FlowNodeError itself. ``cancel``
    stops the job when the run is cancelled."""
    if cancel is not None:
        ctx.on_cancel(cancel)
    while True:
        await asyncio.sleep(interval)
        payload = await status()
        state, progress, error = parse(payload)
        if progress:
            ctx.progress(**progress)
        if state == "completed":
            return payload
        if state in ("failed", "cancelled"):
            ctx.check_cancelled()
            raise FlowNodeError(error or f"The job {state}")


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
    sampling: Optional[dict] = None,
) -> dict:
    cancel_id = f"flow-{uuid.uuid4().hex[:24]}"
    body: dict[str, Any] = {
        "model": chat_model_name(ctx),
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
    body.update(sampling or {})
    # Effort is a level of thinking, as in Chat's composer: off sends none.
    if not thinking:
        body.pop("reasoning_effort", None)

    _register_chat_cancel(ctx, cancel_id)
    return body


def _register_chat_cancel(ctx: Any, cancel_id: str) -> None:
    async def cancel() -> None:
        await ctx.call("POST", "/api/inference/cancel", json = {"cancel_id": cancel_id})

    ctx.on_cancel(cancel)


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
    sampling: Optional[dict] = None,
    load: Optional[dict] = None,
) -> str:
    from core.flows import caller as flow_caller
    from state.tool_policy import tools_force_disabled

    kind = "vision" if images else "chat"
    await (ensure_model(ctx, kind, model, load) if load else ensure_model(ctx, kind, model))
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
        sampling = sampling,
    )
    ctx.check_cancelled()
    if not stream_port:
        with tools_force_disabled():
            payload = await ctx.call("POST", "/v1/chat/completions", json = body)
        return strip_think(_message_text(payload))
    parts: list[str] = []
    shown = 0
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
                    visible = visible_so_far("".join(parts))
                    if len(visible) > shown:
                        ctx.text_delta(stream_port, visible[shown:])
                        shown = len(visible)
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


class JsonFieldStream:
    """The decoded value of one top-level string field, read from JSON as it streams in.

    ``feed(text_so_far)`` returns what the field gained since the last call, so a reply's text can be shown while the
    rest of the object (items, quests) is still being written. An escape cut off at the end waits for the next chunk.
    """

    _ESCAPES = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }

    def __init__(self, field: str):
        self._key = re.compile(r'[{,]\s*"' + re.escape(field) + r'"\s*:\s*"')
        self._start: Optional[int] = None
        self._pos = 0
        self._done = False
        self._pending_high: Optional[int] = None

    def feed(self, text: str) -> str:
        if self._done:
            return ""
        if self._start is None:
            found = self._key.search(text)
            if found is None:
                return ""
            self._start = self._pos = found.end()
        out: list[str] = []
        i = self._pos
        while i < len(text):
            ch = text[i]
            if ch == '"':
                self._done = True
                i += 1
                break
            if ch != "\\":
                out.append(ch)
                i += 1
                continue
            if i + 1 >= len(text):
                break
            code = text[i + 1]
            if code == "u":
                if i + 6 > len(text):
                    break
                try:
                    value = int(text[i + 2 : i + 6], 16)
                except ValueError:
                    value = 0xFFFD
                i += 6
                if 0xD800 <= value < 0xDC00:
                    self._pending_high = value
                    continue
                if 0xDC00 <= value < 0xE000 and self._pending_high is not None:
                    value = 0x10000 + ((self._pending_high - 0xD800) << 10) + (value - 0xDC00)
                self._pending_high = None
                out.append(chr(value) if not 0xD800 <= value < 0xE000 else "\ufffd")
                continue
            out.append(self._ESCAPES.get(code, code))
            i += 2
        self._pos = i
        return "".join(out)


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
    on_text: Optional[Callable[[str], None]] = None,
    retry_bad_json: bool = True,
    sampling: Optional[dict] = None,
    load: Optional[dict] = None,
) -> Any:
    """``on_text`` streams the answer: it is called with the raw JSON written so far after every chunk.
    ``retry_bad_json = False`` raises on an unparsable answer instead of asking again (servers without
    structured output are still asked once more for plain JSON)."""
    from state.tool_policy import tools_force_disabled

    kind = "vision" if images else "chat"
    await (ensure_model(ctx, kind, model, load) if load else ensure_model(ctx, kind, model))
    body = await _chat_body(
        ctx,
        system = system,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        images = images,
        thinking = False,
        seed = seed,
        stream = on_text is not None,
        sampling = sampling,
    )
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "flow_output", "schema": schema, "strict": True},
    }
    try:
        if on_text is not None:
            return first_json(await _stream_text(ctx, body, on_text))
        with tools_force_disabled():
            payload = await ctx.call("POST", "/v1/chat/completions", json = body)
        return first_json(_message_text(payload))
    except FlowNodeError as exc:
        if exc.status not in (400, 422, None) and "JSON" not in str(exc):
            raise
        if not retry_bad_json and exc.status is None:
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
    _register_chat_cancel(ctx, body["cancel_id"])
    if on_text is not None:
        return first_json(await _stream_text(ctx, body, on_text))
    with tools_force_disabled():
        payload = await ctx.call("POST", "/v1/chat/completions", json = body)
    return first_json(_message_text(payload))


async def _stream_text(ctx: Any, body: dict, on_text: Callable[[str], None]) -> str:
    from core.flows import caller as flow_caller
    from state.tool_policy import tools_force_disabled

    body["stream"] = True
    parts: list[str] = []
    try:
        with tools_force_disabled():
            async for chunk in flow_caller.stream_sse(ctx.caller, "/v1/chat/completions", body):
                ctx.check_cancelled()
                if isinstance(chunk.get("error"), dict):
                    raise FlowNodeError(str(chunk["error"].get("message") or "The model failed"))
                for choice in chunk.get("choices") or []:
                    piece = ((choice or {}).get("delta") or {}).get("content")
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
                        on_text("".join(parts))
    except flow_caller.RouteError as exc:
        raise FlowNodeError(exc.detail, status = exc.status) from None
    return strip_think("".join(parts))


IMAGE_STATUS = "/api/inference/images/status"


async def _controlnet_ready(ctx: Any, model: Optional[dict]) -> None:
    """ControlNet needs a bf16 (or bnb 4-bit) load, but Studio's default auto precision picks int8/fp8 where it
    can. With a control image connected, load the picked model unquantized unless it already runs ControlNet."""
    from studio_mcp import loading

    repo = model.get("repo") if isinstance(model, dict) else None
    variant = model.get("variant") if isinstance(model, dict) else None
    try:
        status = await ctx.call("GET", IMAGE_STATUS)
    except FlowNodeError:
        status = None
    status = status if isinstance(status, dict) else {}
    loaded = bool(status.get("loaded"))
    names = {str(status.get(k) or "").lower() for k in ("repo_id", "display_repo_id")}
    same = loaded and (not repo or repo.lower() in names)
    if same and status.get("supports_controlnet") is not False:
        return
    if not repo:
        if loaded:
            raise FlowNodeError(
                "The loaded image model cannot use a ControlNet at its current precision. Pick the model "
                "on this node, or load it in bf16 on the Images page"
            )
        return
    ctx.log(f"Loading {repo} unquantized: ControlNet needs a bf16 model")
    await loading.load_media(
        _caller(ctx),
        None,
        kind = "image",
        model = repo,
        variant = variant,
        hf_token = None,
        deadline = time.monotonic() + LOAD_DEADLINE_S,
        knobs = {"transformer_quant": "off"},
    )
    note_loaded(ctx, "image", repo)


_BAKED_LORAS: dict[str, tuple[str, frozenset, int]] = {}
QUANT_BUILDS = ("int8", "fp8", "nvfp4", "mxfp8")


async def _baked_loras_ready(ctx: Any, model: Optional[dict], loras: list[dict]) -> None:
    """An int8/fp8 image build takes LoRAs only at load time (Studio refuses new adapters at generate time).
    When this node asks for adapters on such a build, reload the model with them baked in, once per set."""
    from core.inference import gpu_arbiter
    from studio_mcp import loading

    wanted = frozenset(item["id"] for item in loras if item.get("weight", 1.0) > 0)
    if not wanted:
        return
    try:
        status = await ctx.call("GET", IMAGE_STATUS)
    except FlowNodeError:
        return
    status = status if isinstance(status, dict) else {}
    resolved = (status.get("resolved") or {}).get("transformer_quant")
    engaged = resolved.get("value") if isinstance(resolved, dict) else None
    engaged = str(engaged or status.get("transformer_quant") or "").lower()
    if not status.get("loaded") or engaged not in QUANT_BUILDS:
        return
    repo = (model.get("repo") if isinstance(model, dict) else None) or status.get("repo_id")
    if not repo:
        return
    record = (str(repo).lower(), wanted, gpu_arbiter.load_count(gpu_arbiter.DIFFUSION))
    if _BAKED_LORAS.get("image") == record:
        return
    ctx.log(
        f"Reloading {repo} with its LoRAs baked in: a quantized ({engaged}) build takes them at load time"
    )
    await loading.load_media(
        _caller(ctx),
        None,
        kind = "image",
        model = repo,
        variant = model.get("variant") if isinstance(model, dict) else None,
        hf_token = None,
        deadline = time.monotonic() + LOAD_DEADLINE_S,
        knobs = {"loras": sorted(loras, key = lambda item: item["id"])},
    )
    note_loaded(ctx, "image", str(repo))
    _BAKED_LORAS["image"] = (
        str(repo).lower(),
        wanted,
        gpu_arbiter.load_count(gpu_arbiter.DIFFUSION),
    )


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
    mask: Optional[dict] = None,
    control: Optional[dict] = None,
    options: Optional[dict] = None,
) -> list[dict]:
    if not (prompt or "").strip():
        raise FlowNodeError("The image prompt is empty")
    if mask is not None and init_image is None:
        raise FlowNodeError("Inpainting needs a source image: connect one to Init image")
    if control is not None and not float(control.get("strength") or 0) > 0:
        control = None
    if control is not None and (init_image is not None or mask is not None or references):
        raise FlowNodeError(
            "ControlNet works with plain text-to-image here: disconnect the Init image, mask and references, "
            "or the control image"
        )
    if control is not None:
        await _controlnet_ready(ctx, model)
    await ensure_model(ctx, "image", model)
    lora_body = route_options({"loras": (options or {}).get("loras")}).get("loras")
    if lora_body and control is None:
        await _baked_loras_ready(ctx, model, lora_body)
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
    if mask is not None:
        body["mask_image"] = await _mask_data_url(ctx, mask, init_image)
    if control is not None:
        body["controlnet"] = {
            "id": control["id"],
            "image": await data_url(ctx, control["image"]),
            "control_type": control.get("control_type") or "passthrough",
            "strength": float(control.get("strength", 1.0)),
            "guidance_start": float(control.get("guidance_start", 0.0)),
            "guidance_end": float(control.get("guidance_end", 1.0)),
        }
    body.update(route_options(options))

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


SAMPLERS = ("default", "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde")
SCHEDULES = ("automatic", "uniform", "karras", "exponential")


def route_options(options: Optional[dict]) -> dict:
    """The image route's sampling, LoRA and hires fields from a node's settings; unset ones are left out."""
    o = options or {}
    body: dict[str, Any] = {}
    loras = []
    for item in o.get("loras") or []:
        if isinstance(item, dict) and str(item.get("id") or "").strip():
            weight = item.get("weight")
            weight = 1.0 if weight is None else max(0.0, min(float(weight), 2.0))
            loras.append({"id": str(item["id"]).strip(), "weight": weight})
    if loras:
        body["loras"] = loras[:8]
    if o.get("sampler") in SAMPLERS[1:]:
        body["sampler"] = o["sampler"]
    if o.get("schedule_type") in SCHEDULES[1:]:
        body["schedule_type"] = o["schedule_type"]
    for key, low, high in (("clip_skip", 1, 12), ("cfg_rescale", 0.0, 1.0), ("shift", 0.01, 100.0)):
        if o.get(key) is not None:
            value = max(low, min(o[key], high))
            body[key] = int(value) if key == "clip_skip" else float(value)
    if o.get("mask_blur") is not None:
        body["mask_blur"] = max(0, min(int(o["mask_blur"]), 64))
    if o.get("only_masked"):
        body["inpaint_only_masked"] = True
    if o.get("inpaint_padding") is not None:
        body["inpaint_padding"] = max(0, min(int(o["inpaint_padding"]), 256))
    if o.get("hires_scale") is not None:
        denoise = o.get("hires_denoise")
        hires: dict[str, Any] = {
            "scale": max(1.0, min(float(o["hires_scale"]), 4.0)),
            "denoise": max(0.0, min(float(0.5 if denoise is None else denoise), 1.0)),
        }
        if o.get("hires_steps"):
            hires["steps"] = max(1, min(int(o["hires_steps"]), 100))
        if str(o.get("hires_upscaler") or "").strip():
            hires["upscaler"] = str(o["hires_upscaler"]).strip()
        body["hires"] = hires
    if str(o.get("upscale_model") or "").strip():
        body["upscale_model"] = str(o["upscale_model"]).strip()
    return body


def _fit_mask(mask: bytes, size: tuple[int, int]) -> tuple[bytes, Optional[tuple[int, int]]]:
    from PIL import Image

    im = Image.open(io.BytesIO(mask))
    im.load()
    original = im.size
    im = im.convert("L")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue(), (original if original != size else None)


def upright_size(data: bytes) -> tuple[int, int]:
    """The image's size as Studio's image routes see it: turned per its EXIF orientation, like phone photos."""
    from PIL import Image

    from core.inference.image_orientation import exif_upright

    return exif_upright(Image.open(io.BytesIO(data))).size


async def _mask_data_url(ctx: Any, mask: dict, init_image: dict) -> str:
    mask_bytes, _ = await media_bytes(ctx, mask)
    init_bytes, _ = await media_bytes(ctx, init_image)
    size = await asyncio.to_thread(upright_size, init_bytes)
    fitted, original = await asyncio.to_thread(_fit_mask, mask_bytes, size)
    if original is not None:
        ctx.log(
            f"The mask is {original[0]}x{original[1]} but the image is {size[0]}x{size[1]}; "
            "it was resized to match",
            "warning",
        )
    return f"data:image/png;base64,{base64.b64encode(fitted).decode()}"


SEGMENTER = "/api/inference/images/segment"


async def ensure_segmenter(ctx: Any) -> None:
    """Download Studio's text-to-mask segmenter on first use, as a model step downloads its model."""
    found = await ctx.call("GET", f"{SEGMENTER}/status")
    if not isinstance(found, dict) or found.get("downloaded") is not False:
        return
    if found.get("supported") is False:
        raise FlowNodeError(
            found.get("reason") or "Mask from text is not available on this install"
        )
    ctx.log("Downloading the mask segmenter (first use only)")
    try:
        await ctx.call("POST", f"{SEGMENTER}/download", json = {})
    except FlowNodeError as exc:
        if "installation owner" not in str(exc):
            raise
        raise FlowNodeError(
            "The mask segmenter is not downloaded yet, and only the installation owner can download it. "
            "Ask them to run Find Mask once, or download it from the mask tool in Images."
        ) from exc

    def parse(payload: Any) -> tuple[str, Optional[dict], Optional[str]]:
        payload = payload if isinstance(payload, dict) else {}
        download = payload.get("download") or {}
        if payload.get("downloaded") and not download.get("downloading"):
            return "completed", None, None
        if download.get("error"):
            return "failed", None, f"Downloading the mask segmenter failed: {download['error']}"
        if download.get("cancelled") or not download.get("downloading"):
            return "cancelled", None, "The mask segmenter download stopped"
        done, total = download.get("bytes_done"), download.get("bytes_total")
        progress = (
            {"phase": "Downloading the mask segmenter", "step": int(done), "total": int(total)}
            if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total
            else None
        )
        return "running", progress, None

    async def cancel() -> None:
        await ctx.call("POST", f"{SEGMENTER}/cancel", json = {})

    await poll_job(
        ctx,
        status = lambda: ctx.call("GET", f"{SEGMENTER}/status"),
        parse = parse,
        cancel = cancel,
        interval = 2.0,
    )


async def segment(ctx: Any, image: dict, subject: str) -> bytes:
    """A grayscale mask PNG (white = the subject) at the image's size, from Studio's SAM 3 segmenter."""
    data, mime = await media_bytes(ctx, image)
    await ensure_segmenter(ctx)
    answer = await ctx.call(
        "POST",
        SEGMENTER,
        json = {
            "image": f"data:{mime};base64,{base64.b64encode(data).decode()}",
            "prompt": subject or "subject",
        },
    )
    mask_value = answer.get("mask") if isinstance(answer, dict) else None
    if not isinstance(mask_value, str):
        raise FlowNodeError("The segmenter returned no mask")
    size = await asyncio.to_thread(upright_size, data)
    fitted, _ = await asyncio.to_thread(_fit_mask, _decode_data_url(mask_value), size)
    return fitted


def _decode_data_url(value: str) -> bytes:
    _, _, payload = value.partition(",")
    return base64.b64decode(payload or value)


def _visible(png: bytes, min_fraction: float = 0.01) -> bool:
    from PIL import Image

    alpha = Image.open(io.BytesIO(png)).convert("RGBA").getchannel("A")
    seen = sum(alpha.point(lambda v: 255 if v > 127 else 0).histogram()[255:])
    return seen >= min_fraction * alpha.size[0] * alpha.size[1]


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
    cutout: Optional[bytes] = None
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
        if not await asyncio.to_thread(_visible, cutout):
            ctx.log(
                f"The segmenter found no {subject}; removing the border colour instead", "warning"
            )
            cutout = None
    except FlowNodeError as exc:
        if exc.status not in (409, 501, 404, None):
            raise
        ctx.log(f"Segmentation unavailable ({exc}); removing the border colour instead", "warning")
    if cutout is None:

        def keyed() -> tuple[bytes, bytes]:
            cut_png, mask_png = chroma_key(data)
            if not _visible(cut_png):
                im = Image.open(io.BytesIO(data)).convert("RGBA")
                out, mask_out = io.BytesIO(), io.BytesIO()
                im.save(out, "PNG")
                Image.new("L", im.size, 255).save(mask_out, "PNG")
                return out.getvalue(), mask_out.getvalue()
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
    reference_text: str = "",
    seed: Optional[int] = None,
) -> dict:
    await ensure_model(ctx, "tts", model)
    if not (text or "").strip():
        raise FlowNodeError("Nothing to say: the text is empty")
    body: dict[str, Any] = {"workflow": "clone" if reference else "speak", "text": text}
    if reference:
        body["inputs"] = {"reference": await audio_ref(ctx, reference)}
        if reference_text.strip():
            body["inputs"]["reference_text"] = reference_text.strip()[:4000]
    if voice and not reference:
        body["options"] = {"voice": voice}
    if instructions:
        body["instructions"] = instructions
    if seed is not None:
        body["seed"] = int(seed)
    clips = await audio_run(ctx, body)
    return clips[0]
