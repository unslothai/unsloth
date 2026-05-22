# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Codex SDK chat provider.

This module wraps the ``codex_app_server`` async SDK so a chat request
routed at ``provider_type="codex"`` can dispatch through the user's
local Codex CLI. The contract back to the frontend is the standard
OpenAI Chat Completions SSE shape, exactly like every other entry in
``external_provider.py``.

The two interesting features here:

* ``_stream_codex_single`` -- translate Codex SDK events into
  OpenAI-format chunks. We prefer ``thread.run_streaming`` when the
  installed SDK exposes it; otherwise we fall back to
  ``await thread.run(...)`` and emit one big content chunk plus a
  usage chunk at the end.

* ``_stream_codex_parallel`` -- the ``parallel_calls`` knob. When > 1,
  spawn N async Codex tasks (capped at 20) via ``asyncio.gather`` and
  emit per-tab ``_toolEvent`` markers so the frontend can render each
  result in its own tab. A final ``codex_gather`` synthesis tab runs a
  single Codex call that takes the N outputs and produces a unified
  answer.

The SDK is imported lazily (inside the helpers that actually need it)
because the spec calls out that ``codex_app_server`` may not even be
importable on the build host. The availability probe in
``codex_availability.py`` is what the frontend uses to gate the
provider entirely; this module just refuses to run if the import
fails at request time.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import time
from typing import Any, AsyncGenerator, Optional

import structlog

logger = structlog.get_logger(__name__)


# Hard cap on parallel Codex fan-out. Picked to match the upper bound
# in the request validator -- exceeding this risks the local Codex CLI
# rate-limiting itself or starving the loop.
MAX_PARALLEL_CALLS = 20


class CodexUnavailableError(RuntimeError):
    """Raised when ``codex_app_server`` is not importable at runtime.

    The availability probe is supposed to hide the provider before any
    request lands here, but we still raise a typed error so the route
    layer can translate it into a 503 the user sees instead of an
    opaque traceback.
    """


def _import_codex() -> Any:
    """Return the imported ``codex_app_server`` module or raise.

    Imported lazily so the rest of the backend keeps starting cleanly
    on hosts that don't have the SDK installed. The frontend calls
    ``GET /api/codex/status`` first and hides the provider when the
    spec isn't importable, so this branch is reached only when the
    user (a) explicitly forces the provider via a stale stored config
    or (b) the install state changes between status probe and chat
    submit.
    """
    if importlib.util.find_spec("codex_app_server") is None:
        raise CodexUnavailableError(
            "codex_app_server is not installed on this host. "
            "Install the Codex Python SDK or use a different provider."
        )
    return importlib.import_module("codex_app_server")


def _last_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the most recent user-role message as a plain string.

    Codex's ``thread.run`` accepts a string (per the docs note
    "plain strings are accepted anywhere a turn input is accepted").
    Studio's chat history is a full OpenAI-style messages array, so
    we flatten it: walk from the end, find the last ``role=user``
    message, and stringify any structured content parts into a
    newline-joined block. Multimodal content (images) is described
    rather than embedded; Codex SDK input shape is text-first.

    This is intentionally conservative -- we don't try to replay the
    whole conversation through Codex per turn because the SDK is
    designed around a stateful ``thread`` object. The thread itself
    holds context across runs; we only need to feed the latest user
    turn.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "text":
                    parts.append(str(entry.get("text") or ""))
                elif entry.get("type") == "image_url":
                    url = (entry.get("image_url") or {}).get("url") or ""
                    parts.append(f"[image: {url[:80]}]" if url else "[image]")
                elif entry.get("type") == "input_document":
                    name = entry.get("filename") or "document"
                    parts.append(f"[document: {name}]")
            return "\n".join(p for p in parts if p)
    return ""


def _system_prompt(messages: list[dict[str, Any]]) -> str:
    """Concatenate all ``role=system`` messages.

    Codex's ``thread_start`` accepts a system prompt for the lifetime
    of the thread. We pass any leading system messages so the user's
    Studio-side system prompt (chat presets) reaches the Codex side
    intact.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict) and entry.get("type") == "text":
                    parts.append(str(entry.get("text") or ""))
    return "\n\n".join(p for p in parts if p)


def _chunk_text(completion_id: str, text: str) -> str:
    """OpenAI Chat Completions content chunk."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}"


def _chunk_stop(completion_id: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(payload)}"


def _chunk_tool_event(completion_id: str, event: dict[str, Any]) -> str:
    """Synthetic OpenAI-shaped chunk carrying an ``_toolEvent`` payload.

    Studio's frontend chat-adapter already understands the
    ``_toolEvent`` envelope and renders tool cards on the fly. We piggy-
    back on the same channel to ship Codex-specific tab markers
    (``codex_tab_open`` / ``codex_tab_chunk`` / ``codex_gather``) so the
    UI doesn't need a brand-new transport.
    """
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
        "_toolEvent": event,
    }
    return f"data: {json.dumps(payload)}"


def _chunk_usage(
    completion_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload)}"


def _coerce_text(payload: Any) -> str:
    """Pull text out of a Codex streaming event or result.

    The SDK shape isn't pinned across versions: events expose ``delta``
    or ``text`` or ``content`` depending on whether the model is in
    plan / answer / tool-use mode. Be defensive -- read whichever
    field is present and fall back to ``str()`` so we never crash
    while translating.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("delta", "text", "content", "message", "final_response"):
            if key in payload:
                value = _coerce_text(payload[key])
                if value:
                    return value
        return ""
    if isinstance(payload, list):
        return "".join(_coerce_text(item) for item in payload)
    text_attr = getattr(payload, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    delta_attr = getattr(payload, "delta", None)
    if isinstance(delta_attr, str):
        return delta_attr
    final_attr = getattr(payload, "final_response", None)
    if isinstance(final_attr, str):
        return final_attr
    return ""


async def _stream_thread_run(
    thread: Any,
    prompt: str,
) -> AsyncGenerator[str, None]:
    """Yield raw text chunks from a Codex thread.

    Prefers ``thread.run_streaming(prompt)`` because that's what the
    docs surface for token-by-token delivery. When the installed SDK
    doesn't have that helper, fall back to ``await thread.run(prompt)``
    and yield the full text once -- this still works end-to-end, just
    without streaming feedback in the UI.
    """
    run_streaming = getattr(thread, "run_streaming", None)
    if run_streaming is not None:
        try:
            stream_obj = run_streaming(prompt)
            # The SDK may return either an async iterator directly or a
            # coroutine that resolves to one. Handle both shapes so a
            # future SDK rev doesn't silently fall off the streaming
            # path.
            if asyncio.iscoroutine(stream_obj):
                stream_obj = await stream_obj
            async for event in stream_obj:
                text = _coerce_text(event)
                if text:
                    yield text
            return
        except Exception as exc:
            logger.warning(
                "codex_provider.run_streaming_failed_fallback",
                error = str(exc),
            )
            # Intentional fallthrough to the non-streaming path so a
            # broken streaming helper doesn't take the whole turn down.

    # Non-streaming fallback: await the full TurnResult, emit one chunk.
    result = await thread.run(prompt)
    text = _coerce_text(result) or getattr(result, "final_response", "") or str(result)
    if text:
        yield text


async def _stream_codex_single(
    model: str,
    system: str,
    prompt: str,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """Run one Codex turn and emit OpenAI-shaped SSE lines."""
    sdk = _import_codex()
    async_codex_cls = getattr(sdk, "AsyncCodex", None)
    if async_codex_cls is None:
        raise CodexUnavailableError(
            "codex_app_server is installed but AsyncCodex is missing -- "
            "upgrade the SDK."
        )

    completion_text_chars = 0

    async with async_codex_cls() as codex:
        # ``thread_start`` accepts a model id; system prompts are
        # passed when supported by the SDK rev (older revs ignore the
        # extra kwarg). Be tolerant about kwargs that may not exist.
        thread_kwargs: dict[str, Any] = {"model": model}
        if system:
            # Try the canonical kwargs first; the SDK shapes vary
            # across versions and we'd rather accept the system prompt
            # being dropped than crash on a missing kwarg.
            thread_kwargs["system"] = system
        try:
            thread = await codex.thread_start(**thread_kwargs)
        except TypeError:
            # Older SDK: only the ``model`` kwarg is accepted. Drop
            # extras and retry; the system prompt then lives only in
            # the prompt itself (we prepend it below).
            thread = await codex.thread_start(model = model)
            if system:
                prompt = f"{system}\n\n{prompt}"
        async for text in _stream_thread_run(thread, prompt):
            completion_text_chars += len(text)
            yield _chunk_text(completion_id, text)

    # Estimate tokens crudely from char counts. The Codex SDK does not
    # consistently expose a token breakdown; we surface a usage chunk
    # purely so the frontend cost / context display has a non-zero
    # value to render. ``int(chars / 4)`` is the standard rough-cut.
    yield _chunk_usage(
        completion_id,
        prompt_tokens = max(1, len(prompt) // 4),
        completion_tokens = max(0, completion_text_chars // 4),
    )
    yield _chunk_stop(completion_id)


async def stream_codex(
    messages: list[dict[str, Any]],
    model: str,
    parallel_calls: int = 1,
) -> AsyncGenerator[str, None]:
    """Top-level entry point used by the inference route.

    When ``parallel_calls == 1`` (the default), this delegates to the
    single-turn helper. When > 1 (and <= ``MAX_PARALLEL_CALLS``), it
    spawns N parallel Codex turns and emits one tab per result plus a
    final synthesis tab. The fan-out runs every spawned turn against
    the SAME user prompt -- the parallel knob is for sampling N
    independent attempts at the same task, which is what the UI tab
    strip surfaces.
    """
    prompt = _last_user_prompt(messages)
    system = _system_prompt(messages)
    completion_id = f"chatcmpl-codex-{int(time.time() * 1000)}"

    if not prompt:
        yield _chunk_text(
            completion_id,
            "(no user prompt found -- send a message before invoking codex)",
        )
        yield _chunk_stop(completion_id)
        return

    clamped = max(1, min(int(parallel_calls or 1), MAX_PARALLEL_CALLS))

    if clamped == 1:
        async for line in _stream_codex_single(model, system, prompt, completion_id):
            yield line
        yield "data: [DONE]"
        return

    async for line in _stream_codex_parallel(
        model = model,
        system = system,
        prompt = prompt,
        n = clamped,
        completion_id = completion_id,
    ):
        yield line
    yield "data: [DONE]"


async def _stream_codex_parallel(
    *,
    model: str,
    system: str,
    prompt: str,
    n: int,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """Fan out N Codex tasks, emit per-tab chunks, then synthesise.

    Each tab id is a 1-based integer. We emit ``codex_tab_open`` up
    front for every tab so the UI can paint the tab strip before any
    response lands, then route streamed text per tab via
    ``codex_tab_chunk`` (one event per text delta). The final
    ``codex_gather`` event carries a synthesis from a separate
    standalone Codex call so the user sees both the per-tab raw
    outputs AND a unified merged answer.

    Bounded concurrency: ``asyncio.gather`` is used with a fresh
    ``AsyncCodex`` instance per tab to keep the SDK from sharing a
    single thread object across coroutines (the SDK is not documented
    as concurrency-safe on a single ``Codex`` handle). N is clamped to
    ``MAX_PARALLEL_CALLS`` by the caller.
    """
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    # Pre-emit one tab_open per tab so the UI can paint the tabs
    # immediately. The frontend creates the tabs lazily on first
    # ``codex_tab_open`` event anyway; pre-emitting just gives us the
    # familiar tabs-first / content-second render order.
    for idx in range(1, n + 1):
        yield _chunk_tool_event(
            completion_id,
            {
                "type": "codex_tab_open",
                "tab_id": idx,
                "query": prompt,
                "total_tabs": n,
            },
        )

    async def _worker(tab_id: int) -> str:
        """Run one Codex turn, push every chunk into the queue, and
        return the full accumulated text so the synthesis step can
        consume it. Errors are surfaced as a ``codex_tab_error``
        tool-event so the tab strip shows which lane failed without
        aborting the whole fan-out.
        """
        collected: list[str] = []
        try:
            sdk = _import_codex()
            async_codex_cls = getattr(sdk, "AsyncCodex")
            async with async_codex_cls() as codex:
                thread_kwargs: dict[str, Any] = {"model": model}
                if system:
                    thread_kwargs["system"] = system
                inner_prompt = prompt
                try:
                    thread = await codex.thread_start(**thread_kwargs)
                except TypeError:
                    thread = await codex.thread_start(model = model)
                    if system:
                        inner_prompt = f"{system}\n\n{prompt}"
                async for text in _stream_thread_run(thread, inner_prompt):
                    collected.append(text)
                    await queue.put(
                        _chunk_tool_event(
                            completion_id,
                            {
                                "type": "codex_tab_chunk",
                                "tab_id": tab_id,
                                "text": text,
                            },
                        )
                    )
        except Exception as exc:
            logger.warning(
                "codex_provider.parallel_tab_failed",
                tab_id = tab_id,
                error = str(exc),
            )
            await queue.put(
                _chunk_tool_event(
                    completion_id,
                    {
                        "type": "codex_tab_error",
                        "tab_id": tab_id,
                        "error": str(exc),
                    },
                )
            )
        finally:
            await queue.put(
                _chunk_tool_event(
                    completion_id,
                    {
                        "type": "codex_tab_close",
                        "tab_id": tab_id,
                    },
                )
            )
        return "".join(collected)

    workers = [asyncio.create_task(_worker(i + 1)) for i in range(n)]

    # asyncio.gather returns a _GatheringFuture, not a coroutine, so it
    # cannot be passed to create_task. Wrap the await in a small helper
    # coroutine so we keep both (a) a handle for awaiting and (b) the
    # ability to schedule the drain side-effect that unblocks the
    # consumer queue. Stash the results into a list the drain finally
    # block reads so per-tab outputs survive any single-worker errors.
    per_tab_texts: list[str] = []

    async def _await_workers() -> None:
        results = await asyncio.gather(*workers, return_exceptions = True)
        for r in results:
            if isinstance(r, BaseException):
                per_tab_texts.append("")
            else:
                per_tab_texts.append(r)

    async def _drain_when_done() -> None:
        try:
            await _await_workers()
        finally:
            await queue.put(None)

    drain_task = asyncio.create_task(_drain_when_done())

    while True:
        line = await queue.get()
        if line is None:
            break
        yield line

    # Drain finished; per_tab_texts is now populated by the helper
    # coroutine above. We already shielded individual errors as
    # ``codex_tab_error`` events, so nothing should leak here -- but
    # log defensively in case a worker future itself raised.
    await drain_task

    synthesis_text = await _run_codex_synthesis(
        model = model,
        prompt = prompt,
        tab_outputs = per_tab_texts,
    )

    yield _chunk_tool_event(
        completion_id,
        {
            "type": "codex_gather",
            "summary": synthesis_text,
            "tab_count": n,
        },
    )

    # Also emit the synthesis as a visible content chunk so any client
    # that ignores the tab tool-events (e.g. a curl user) still sees
    # a final unified answer. The tabbed UI re-uses the same payload
    # via ``codex_gather`` and hides it from the main content lane to
    # avoid duplication.
    if synthesis_text:
        yield _chunk_text(completion_id, synthesis_text)

    yield _chunk_usage(
        completion_id,
        prompt_tokens = max(1, len(prompt) // 4),
        completion_tokens = max(0, len(synthesis_text) // 4),
    )
    yield _chunk_stop(completion_id)


async def _run_codex_synthesis(
    *,
    model: str,
    prompt: str,
    tab_outputs: list[str],
) -> str:
    """Run one extra Codex call that consumes the N per-tab outputs and
    returns a unified synthesis. Returns the empty string on failure --
    the caller already surfaced the per-tab outputs so an empty
    synthesis is recoverable.
    """
    if not tab_outputs:
        return ""
    joined = "\n\n".join(
        f"=== Attempt {i + 1} ===\n{text.strip() or '(no output)'}"
        for i, text in enumerate(tab_outputs)
    )
    synthesis_prompt = (
        "You are given multiple independent attempts at the same task.\n"
        "Your job: synthesise a single best response that takes the "
        "strongest parts of each attempt, resolves disagreements, and "
        "presents a clear unified answer.\n\n"
        f"Original task:\n{prompt}\n\n"
        f"Attempts:\n{joined}\n\n"
        "Unified answer:"
    )
    try:
        sdk = _import_codex()
        async_codex_cls = getattr(sdk, "AsyncCodex")
        async with async_codex_cls() as codex:
            try:
                thread = await codex.thread_start(model = model)
            except TypeError:
                thread = await codex.thread_start()
            result = await thread.run(synthesis_prompt)
        return _coerce_text(result) or getattr(result, "final_response", "") or str(result)
    except Exception as exc:
        logger.warning("codex_provider.synthesis_failed", error = str(exc))
        return ""


# ── Device-auth helper ──────────────────────────────────────────────


async def stream_codex_device_login() -> AsyncGenerator[dict[str, Any], None]:
    """Run ``codex auth login --device-auth`` and yield progress events.

    Yields dicts (NOT SSE lines) that the route layer wraps in SSE.
    First event is always ``{type: "device_url", url: "..."}`` once we
    detect a verification URL in the CLI output. Subsequent events
    forward CLI stdout/stderr line-by-line as ``{type: "log", line: ...}``
    so the UI can show progress. A final ``{type: "done", ok: bool}``
    signals completion.

    The URL extraction matches the CLI's actual output shape (the CLI
    prints something like ``Open https://auth.openai.com/device/...``
    on the device-auth path). We scan every line for the first
    https:// URL containing ``device``; that has historically been
    stable across CLI versions.
    """
    import re

    cli_path = "codex"
    args = ["auth", "login", "--device-auth"]

    try:
        proc = await asyncio.create_subprocess_exec(
            cli_path,
            *args,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        yield {"type": "error", "message": "codex CLI not found on PATH"}
        yield {"type": "done", "ok": False}
        return
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        yield {"type": "done", "ok": False}
        return

    url_re = re.compile(r"https?://\S*device\S*", re.IGNORECASE)
    url_emitted = False
    rc: int = -1

    try:
        assert proc.stdout is not None
        while True:
            line_b = await proc.stdout.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors = "replace").rstrip()
            if not url_emitted:
                match = url_re.search(line)
                if match:
                    yield {"type": "device_url", "url": match.group(0)}
                    url_emitted = True
            yield {"type": "log", "line": line}
    finally:
        try:
            rc = await proc.wait()
        except Exception:
            rc = -1
    yield {"type": "done", "ok": rc == 0, "return_code": rc}
