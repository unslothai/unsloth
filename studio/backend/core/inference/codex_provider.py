# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Codex SDK chat provider.

This module wraps the OpenAI Codex async Python SDK (``openai_codex``
canonical, ``codex_app_server`` legacy alias) so a chat request routed
at ``provider_type="codex"`` can dispatch through the user's local
Codex CLI. The contract back to the frontend is the standard OpenAI
Chat Completions SSE shape, exactly like every other entry in
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
  answer. Workers cancel cleanly when the SSE consumer disconnects so
  cancelled fan-outs never leave zombie Codex calls running.

The SDK is imported lazily (inside the helpers that actually need it)
because the SDK may not be importable on the build host. The
availability probe in ``codex_availability.py`` is what the frontend
uses to gate the provider entirely; this module just refuses to run
if the import fails at request time.
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


# Names the upstream Python SDK has shipped under. ``openai_codex`` is
# the canonical package at ``openai/codex/sdk/python``; ``codex_app_server``
# is kept as a forward-compat alias because the Rust crate uses that name.
# Order matters: first hit wins, so the canonical name is tried first.
_SDK_MODULE_NAMES: tuple[str, ...] = ("openai_codex", "codex_app_server")


class CodexUnavailableError(RuntimeError):
    """Raised when the Codex Python SDK is not importable at runtime.

    The availability probe is supposed to hide the provider before any
    request lands here, but we still raise a typed error so the route
    layer can translate it into a 503 the user sees instead of an
    opaque traceback.
    """


def _import_codex() -> Any:
    """Return the imported Codex SDK module or raise CodexUnavailableError.

    Imported lazily so the rest of the backend keeps starting cleanly
    on hosts that don't have the SDK installed. Probes ``openai_codex``
    first (canonical upstream name), then ``codex_app_server`` (Rust-
    crate-style alias) for forward compatibility. The frontend calls
    ``GET /api/codex/status`` first and hides the provider when no
    name resolves, so this branch is reached only when (a) the user
    explicitly forces the provider via a stale stored config or (b)
    the install state changes between status probe and chat submit.
    """
    for name in _SDK_MODULE_NAMES:
        if importlib.util.find_spec(name) is not None:
            return importlib.import_module(name)
    raise CodexUnavailableError(
        "Codex Python SDK is not installed on this host. "
        "Install with `pip install openai-codex-app-server-sdk` "
        "(import name `openai_codex`, legacy alias `codex_app_server`), "
        "or use a different provider."
    )


def _stringify_content(content: Any) -> str:
    """Flatten an OpenAI-style content field into one plain-text block.

    Multimodal entries (images, documents) are described inline rather
    than embedded since the Codex SDK input shape is text-first.
    """
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


def _last_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Render the conversation as a single prompt for Codex.

    Studio's chat history is a full OpenAI-style messages array. Codex
    opens a fresh ``thread`` per chat-completion request (we have no
    way to cache the SDK ``thread`` keyed on Studio's session id from
    here -- the inference route is stateless), so we MUST serialise the
    full transcript into the prompt or the model loses every prior
    turn.

    Layout:
        User: <user_1>
        Assistant: <assistant_1>
        User: <user_2>
        ...
        Assistant:

    The trailing ``Assistant:`` cue tells Codex this is its turn. When
    there is exactly one user message and no assistant history we drop
    the cue and emit the plain text -- matches the historical single-
    shot behaviour.
    """
    # Filter to user / assistant only; system is handled separately by
    # `_system_prompt` and passed to thread_start when supported.
    convo: list[tuple[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _stringify_content(msg.get("content"))
        if text:
            convo.append((role, text))

    if not convo:
        return ""

    # Trivial case: a single user turn — pass it through unchanged so we
    # don't perturb single-shot behaviour or test expectations.
    if len(convo) == 1 and convo[0][0] == "user":
        return convo[0][1]

    # Multi-turn: render User:/Assistant: blocks then prompt the model.
    lines: list[str] = []
    for role, text in convo:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {text}")
    # If the last turn is from the user (the common case), append an
    # empty Assistant cue so Codex picks up from the right side.
    if convo[-1][0] == "user":
        lines.append("Assistant:")
    return "\n\n".join(lines)


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
            "Codex SDK is installed but AsyncCodex is missing -- " "upgrade the SDK."
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

    cancelled = False
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            yield line
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        # Cancel every in-flight worker so the Codex SDK calls release
        # their quota / sockets instead of running to completion against
        # a disconnected client. Workers shield themselves in `_worker`'s
        # own try/finally so we just signal cancellation and bail.
        for w in workers:
            if not w.done():
                w.cancel()
        drain_task.cancel()
        # Best-effort gather so cancellation propagates and we don't
        # leave coroutines hanging on the event loop.
        try:
            await asyncio.gather(*workers, drain_task, return_exceptions = True)
        except Exception:
            pass
        raise
    finally:
        # If we exited normally, drain_task is already done (it put None
        # on the queue right after gather). On the cancellation path we
        # already gathered above, so this await is a fast no-op.
        if not cancelled:
            try:
                await drain_task
            except Exception as exc:
                logger.warning(
                    "codex_provider.parallel_drain_failed",
                    error = str(exc),
                )

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
        return (
            _coerce_text(result) or getattr(result, "final_response", "") or str(result)
        )
    except Exception as exc:
        logger.warning("codex_provider.synthesis_failed", error = str(exc))
        return ""


# ── Device-auth helper ──────────────────────────────────────────────


async def stream_codex_device_login() -> AsyncGenerator[dict[str, Any], None]:
    """Run ``codex login --device-auth`` and yield progress events.

    Yields dicts (NOT SSE lines) that the route layer wraps in SSE.
    First event is always ``{type: "device_url", url: "..."}`` once we
    detect a verification URL in the CLI output. The one-time code is
    emitted as ``{type: "device_code", code: "ABCD-EFGH"}`` so the UI
    can show it next to the URL (upstream CLI prints both on separate
    lines). Subsequent CLI stdout/stderr lines forward as
    ``{type: "log", line: ...}``. A final ``{type: "done", ok: bool}``
    signals completion.

    Subprocess lifecycle: started in its own process group via
    ``start_new_session=True`` (Unix) so cancellation can SIGTERM the
    whole group and reach any child processes the codex CLI spawns.
    On Windows a fallback uses ``CREATE_NEW_PROCESS_GROUP``. When the
    SSE consumer disconnects, the generator's cleanup terminates the
    process group within a 5s budget then SIGKILL's as a last resort,
    so the CLI never lingers consuming a device-auth session.

    URL handling: upstream ``codex login --device-auth`` prints the URL
    wrapped in ANSI escape sequences (``\x1b[34m...\x1b[0m``). We strip
    ANSI before regex matching so the URL emitted to the frontend is
    clean and clickable.
    """
    import os
    import re
    import signal

    cli_path = "codex"
    args = ["login", "--device-auth"]

    # Detach the subprocess into a new process group on Unix so we can
    # SIGTERM the whole group on cancel without sending it to ourselves.
    # On Windows, ``creationflags=CREATE_NEW_PROCESS_GROUP`` (0x200) gives
    # an equivalent isolation for ``proc.send_signal(signal.CTRL_BREAK_EVENT)``.
    spawn_kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if os.name == "posix":
        spawn_kwargs["start_new_session"] = True
    elif os.name == "nt":
        spawn_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP

    try:
        proc = await asyncio.create_subprocess_exec(cli_path, *args, **spawn_kwargs)
    except FileNotFoundError:
        yield {"type": "error", "message": "codex CLI not found on PATH"}
        yield {"type": "done", "ok": False}
        return
    except Exception as exc:
        logger.warning("codex_provider.login_spawn_failed", error = str(exc))
        yield {"type": "error", "message": "Failed to start codex CLI"}
        yield {"type": "done", "ok": False}
        return

    # Strip ANSI control sequences (the upstream login command wraps the
    # URL and code in `\x1b[34m...\x1b[0m`) before pattern matching.
    ansi_re = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
    # Anchor on the upstream URL shape: ``.../codex/device`` (optionally
    # with a query string). The pattern accepts any host because some
    # builds redirect via a staging host.
    url_re = re.compile(
        r"https?://[^\s\x1b]+?/codex/device(?:\?[^\s\x1b]*)?", re.IGNORECASE
    )
    # One-time-code format from upstream device_code_auth.rs: 4 chars,
    # dash, 4 chars. Pattern is tolerant of any uppercase alphanum.
    code_re = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")

    url_emitted = False
    code_emitted = False
    rc: int = -1
    cancelled = False

    try:
        assert proc.stdout is not None
        while True:
            line_b = await proc.stdout.readline()
            if not line_b:
                break
            raw = line_b.decode("utf-8", errors = "replace").rstrip()
            line = ansi_re.sub("", raw)
            if not url_emitted:
                match = url_re.search(line)
                if match:
                    yield {"type": "device_url", "url": match.group(0)}
                    url_emitted = True
            if not code_emitted:
                cm = code_re.search(line)
                if cm:
                    yield {"type": "device_code", "code": cm.group(1)}
                    code_emitted = True
            yield {"type": "log", "line": line}
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    finally:
        # Tear the subprocess down even on cancellation. Unix: kill the
        # whole process group; Windows: ``CTRL_BREAK_EVENT`` followed by
        # ``terminate()``. Bounded wait so cleanup never deadlocks the
        # SSE close path.
        if proc.returncode is None:
            try:
                if os.name == "posix":
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    except Exception:
                        proc.terminate()
            except Exception as exc:
                logger.warning(
                    "codex_provider.login_terminate_failed",
                    error = str(exc),
                )
            try:
                await asyncio.wait_for(proc.wait(), timeout = 5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout = 2.0)
                except Exception:
                    pass
        try:
            rc = proc.returncode if proc.returncode is not None else -1
        except Exception:
            rc = -1
    if cancelled:
        return
    yield {"type": "done", "ok": rc == 0, "return_code": rc}
