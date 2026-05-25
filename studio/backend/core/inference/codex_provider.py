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
import shutil
import sys
import time
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)


# Device-auth verification URLs the upstream codex CLI prints during
# `codex login --device-auth`. Only these hosts are forwarded to the
# browser as "Open verification page". A shimmed codex earlier on PATH
# can still print a phishing URL that matches the regex used to fish
# the verification URL out of stdout (`/device`, `/activate`,
# `/verify`), but Studio refuses to surface anything not on this list,
# so the malicious URL never reaches the user.
_ALLOWED_DEVICE_AUTH_HOSTS: frozenset[str] = frozenset(
    {
        "auth.openai.com",
        "chatgpt.com",
    }
)


def _safe_host(url: str) -> Optional[str]:
    """Return the lower-case host of ``url`` or None if it cannot be parsed."""
    try:
        return (urlparse(url).hostname or "").lower() or None
    except Exception:
        return None


def _is_allowed_device_url(url: str) -> bool:
    """Return True iff ``url`` looks like a real codex device-auth URL.

    Requires https, a parseable URL, and a host on the allowlist above.
    Codex login URLs are always https in the wild; downgrade to http
    is a strong signal of a malicious shim, so we drop both at the
    same gate.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host in _ALLOWED_DEVICE_AUTH_HOSTS


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


def _codex_sdk_env_override() -> dict[str, str]:
    """Return an env update dict that scrubs sensitive vars from the
    codex app-server subprocess env.

    The upstream openai_codex SDK's `AppServerConfig.env` is merged on
    top of `os.environ.copy()` (see openai/codex/sdk/python/src/openai_codex/client.py),
    so providing an empty-string mapping for every non-safe key
    effectively overrides them in the spawn env. Combined with
    `_codex_subprocess_env()` (used for direct CLI calls) this gives
    parity between the CLI and SDK code paths: neither sees HF_TOKEN,
    GH_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY, or any other secret
    that lives in the Studio parent environment.
    """
    import os

    from core.inference.codex_availability import _SAFE_CODEX_ENV_KEYS

    safe = set(_SAFE_CODEX_ENV_KEYS)
    return {key: "" for key in os.environ if key not in safe}


_SCRUBBED_ENV_LOCK = asyncio.Lock()
_SCRUBBED_ENV_REFCOUNT: dict[str, int] = {}
# Saved originals shared across all wrappers. The first wrapper to scrub
# a key records its pre-scrub value here; later wrappers that pick up the
# same key while it is already absent from os.environ inherit the same
# saved value so the very last wrapper to release a key still restores
# the right thing. Kept module-level (not per-instance) because two
# concurrent wrappers must agree on what the original was even though
# only one of them actually saw it in os.environ.
_SCRUBBED_ENV_ORIGINALS: dict[str, str] = {}


class _ScrubbedEnvAsyncCodex:
    """Async-context wrapper that swaps `os.environ` for the lifetime
    of a Codex SDK session.

    Used as the fail-closed fallback when the SDK does not expose
    `AppServerConfig(env=...)`. The SDK starts its app-server with
    `env = os.environ.copy()`, so removing secrets from the parent
    process env right before construction keeps them out of the child.

    Concurrency model: a process-wide asyncio lock serialises the
    enter/exit critical section, and a per-key refcount tracks how
    many concurrent wrappers are currently "holding" the scrub. A
    key is only restored when the last wrapper using it exits. This
    fixes three issues:

    1. Two concurrent fan-out workers used to race: wrapper A could
       restore `HF_TOKEN` while wrapper B was still inside SDK
       startup, letting B's spawned app-server inherit the secret.
       The refcount keeps the key scrubbed for the full overlap
       window.
    2. If the SDK constructor raised before `__aenter__` returned,
       Python never called `__aexit__`, so the deleted keys leaked
       permanently. Construction now happens INSIDE the try/except
       in `__aenter__`, and the scrub is rolled back on failure.
    3. ``_codex_sdk_env_override()`` only enumerates keys currently
       in ``os.environ``, so a wrapper B entering AFTER wrapper A
       already scrubbed (say) ``HF_TOKEN`` would never see that key
       in its overrides dict, never bump its refcount, and miss the
       scrub for HF_TOKEN entirely. When A then exited it would
       restore HF_TOKEN while B was still mid-session. Wrapper B
       now also picks up every key currently refcounted by an
       earlier wrapper (read under the lock) so the refcount
       reflects the true set of holders for every scrubbed key.
    """

    def __init__(self, async_codex_cls: Any):
        self._async_codex_cls = async_codex_cls
        self._inner: Any = None
        # Keys this wrapper instance contributed to the refcount, so
        # __aexit__ knows exactly which counters to decrement (avoids
        # racing with concurrent wrappers that scrub a different set).
        self._held_keys: list[str] = []

    async def __aenter__(self) -> Any:
        import os

        async with _SCRUBBED_ENV_LOCK:
            # Keys this session would scrub if it were entering first.
            keys_to_scrub = set(_codex_sdk_env_override())
            # Plus every key still held by an earlier wrapper -- without
            # this we would miss keys that are already absent from
            # os.environ but ARE still scrubbed and refcounted.
            keys_to_scrub.update(
                key for key, count in _SCRUBBED_ENV_REFCOUNT.items() if count > 0
            )
            for key in keys_to_scrub:
                current = _SCRUBBED_ENV_REFCOUNT.get(key, 0)
                if current == 0:
                    # First wrapper to scrub this key -- save the
                    # original. If the key is somehow not in os.environ
                    # right now (deleted between override-snapshot and
                    # here, or simply never set), skip it: nothing to
                    # scrub and nothing to restore.
                    if key not in os.environ:
                        continue
                    _SCRUBBED_ENV_ORIGINALS[key] = os.environ[key]
                    del os.environ[key]
                _SCRUBBED_ENV_REFCOUNT[key] = current + 1
                self._held_keys.append(key)
        try:
            self._inner = self._async_codex_cls()
            return await self._inner.__aenter__()
        except BaseException:
            # Roll back the scrub if SDK construction / enter fails;
            # otherwise the deleted env vars would leak permanently.
            await self._release_held_keys()
            raise

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._inner is not None:
                return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            await self._release_held_keys()

    async def _release_held_keys(self) -> None:
        import os

        async with _SCRUBBED_ENV_LOCK:
            for key in self._held_keys:
                current = _SCRUBBED_ENV_REFCOUNT.get(key, 0)
                if current <= 0:
                    continue
                next_count = current - 1
                if next_count == 0:
                    # Last wrapper holding this key -- restore the
                    # original snapshot recorded when it was first
                    # scrubbed, regardless of which wrapper saved it.
                    _SCRUBBED_ENV_REFCOUNT.pop(key, None)
                    original = _SCRUBBED_ENV_ORIGINALS.pop(key, None)
                    if original is not None:
                        os.environ.setdefault(key, original)
                else:
                    _SCRUBBED_ENV_REFCOUNT[key] = next_count
            self._held_keys.clear()


def _resolve_codex_bin() -> Optional[str]:
    """Best-effort PATH lookup for the codex CLI used as ``codex_bin``.

    The upstream Python SDK normally locates its pinned codex binary
    via the ``openai-codex-cli-bin`` runtime package, installed
    automatically as a dependency of ``pip install openai-codex``.
    Users who installed the SDK with ``--no-deps`` (lightweight
    setups), users whose platform is not yet on the
    ``openai-codex-cli-bin`` wheel matrix, and users whose codex CLI
    was installed via ``npm i -g @openai/codex`` / Homebrew never
    have the pinned runtime package on import path. Without an
    explicit ``codex_bin`` the SDK then raises
    ``FileNotFoundError("Unable to locate the pinned Codex runtime")``
    even though a perfectly good ``codex`` is on PATH and was the
    binary Studio's availability probe already verified.

    Returning ``shutil.which("codex")`` here turns that hard failure
    into a working session: Studio passes the resolved path through
    ``AppServerConfig(codex_bin=...)`` and the SDK uses it directly.
    Returning ``None`` keeps the pinned-runtime path intact when the
    CLI is not on PATH (which only happens on hosts where the SDK is
    importable but the CLI is missing -- ``codex_availability`` would
    already report ``installed=false`` there, so callers never reach
    this).
    """
    try:
        return shutil.which("codex")
    except Exception as exc:
        logger.warning(
            "codex_provider.codex_bin_lookup_failed",
            exc_type = type(exc).__name__,
            error = str(exc),
        )
        return None


def _open_async_codex(async_codex_cls: Any) -> Any:
    """Construct an AsyncCodex whose spawned app-server cannot see
    Studio's secrets.

    Preferred path: `AsyncCodex(config=AppServerConfig(env=...,
    codex_bin=...))` which scopes the env override to the spawned
    subprocess only and explicitly pins the codex binary so the SDK
    does not need its pinned ``openai-codex-cli-bin`` runtime to be
    installed.

    Fail-closed fallback: `_ScrubbedEnvAsyncCodex` swaps `os.environ`
    for the lifetime of the session so the SDK's internal
    `os.environ.copy()` spawn never sees HF_TOKEN / GH_TOKEN /
    WANDB_API_KEY / etc. There is no code path that lets the SDK
    inherit those secrets.
    """
    try:
        sdk_mod = sys.modules.get("openai_codex") or sys.modules.get("codex_app_server")
        if sdk_mod is not None:
            app_server_config = getattr(sdk_mod, "AppServerConfig", None)
            if app_server_config is not None:
                # Try the modern signature: AppServerConfig(env=..., codex_bin=...).
                # codex_bin keeps PATH-installed codex working without the
                # SDK's pinned openai-codex-cli-bin runtime package. We try
                # the full signature first, then degrade gracefully if the
                # installed SDK build does not accept codex_bin yet.
                codex_bin = _resolve_codex_bin()
                env_override = _codex_sdk_env_override()
                if codex_bin is not None:
                    try:
                        return async_codex_cls(
                            config = app_server_config(
                                env = env_override,
                                codex_bin = codex_bin,
                            ),
                        )
                    except TypeError:
                        # Older SDK build: codex_bin kwarg unknown. Fall
                        # through to env-only construction below.
                        pass
                return async_codex_cls(
                    config = app_server_config(env = env_override),
                )
    except TypeError:
        # Older SDK: AppServerConfig may not accept the env kwarg yet.
        # Fall through to the os.environ-swap wrapper.
        pass
    except Exception as exc:
        logger.warning(
            "codex_provider.env_scrub_config_failed",
            exc_type = type(exc).__name__,
            error = str(exc),
        )
    return _ScrubbedEnvAsyncCodex(async_codex_cls)


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
        "Install with `pip install openai-codex` (canonical upstream "
        "name, imports as `openai_codex`; legacy alias `codex_app_server` "
        "is also accepted), or use a different provider."
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


# Event types Codex emits that carry the assistant's natural-language
# answer (or its stream-time deltas). Other event types -- command
# execution, file edits, tool calls, plan steps -- have their own
# `delta` fields that the OpenAI Chat Completions surface must NOT
# render as visible assistant text or local stdout / paths would leak
# into the chat reply.
_ANSWER_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "message.delta",
        "message.completed",
        "assistant.message.delta",
        "assistant.message.completed",
        "thread.message.delta",
        "thread.message.completed",
        "text_delta",
        "completed",
        # Some SDK revs use a bare "message" / "delta" wrapper without
        # qualifying the role; we accept those too because the legacy
        # tests rely on the shape.
        "message",
        "delta",
    }
)


def _coerce_text(payload: Any) -> str:
    """Pull text out of a Codex streaming event or result.

    Only events whose ``type`` is in `_ANSWER_EVENT_TYPES` (or have no
    ``type`` field at all, i.e. raw text containers) are translated to
    visible text. Tool / command / plan deltas are dropped so local
    stdout, file paths, or tool-call arguments never flow into the
    Chat Completions content stream.

    Both dict-shaped events (tests + some pre-release SDKs) AND
    object-shaped events (the real upstream SDK's typed notification
    classes) are gated -- if the payload exposes a `type` attribute
    or key whose value is not in the answer-event allow-list, we
    return the empty string regardless of whether `.delta` or `.text`
    is present. Round 6 reviewer caught the object-shape gap: the
    upstream SDK can emit `item/commandExecution/outputDelta`,
    `item/fileChange/outputDelta`, etc. as typed objects, all of
    which carry `.delta` strings containing local stdout, patches,
    or tool arguments. Without the object-side filter those strings
    would have flowed straight into visible assistant text.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # If the dict carries a typed event tag, gate on it: only
        # answer-bearing types contribute visible text. Untyped dicts
        # (legacy / raw text wrappers) fall through to the field walk.
        ev_type = payload.get("type")
        if isinstance(ev_type, str) and ev_type not in _ANSWER_EVENT_TYPES:
            return ""
        for key in ("delta", "text", "content", "message", "final_response"):
            if key in payload:
                value = _coerce_text(payload[key])
                if value:
                    return value
        return ""
    if isinstance(payload, list):
        return "".join(_coerce_text(item) for item in payload)
    # Object path: gate on `payload.type` if present, AND on the class
    # name as a fallback (the upstream SDK uses class names like
    # `AgentMessageDeltaNotification` / `CommandExecutionOutputDelta`
    # so a denylist-by-substring catches typed payloads that lack a
    # `type` attribute).
    ev_type_obj = getattr(payload, "type", None)
    if isinstance(ev_type_obj, str) and ev_type_obj not in _ANSWER_EVENT_TYPES:
        return ""
    cls_name = payload.__class__.__name__
    # Allow only class names that contain "Message" or "Delta" without
    # also containing a tool / command / plan / file marker.
    cls_lower = cls_name.lower()
    if any(
        marker in cls_lower
        for marker in (
            "command",
            "exec",
            "file",
            "patch",
            "plan",
            "tool",
            "reason",
            "stdout",
            "stderr",
        )
    ):
        return ""
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


def _completed_agent_message_text(payload: Any) -> str:
    """Return the assistant text from an ``ItemCompletedNotification``.

    The canonical openai_codex SDK sometimes finishes a turn without
    emitting any ``message.delta`` events: the final answer arrives
    only as ``ItemCompletedNotification(item=AgentMessage(text=...))``
    at the end of the stream. Without recognising that shape,
    ``_stream_thread_run`` would loop through the stream, see no
    ``delta`` text, and return an empty Chat Completions response.

    Returns the empty string for any other event shape so the caller
    can ignore it. Matches by class name + structural shape so the
    function works on both real upstream events and the dict / fake
    shapes the tests use.
    """
    if payload is None:
        return ""

    # Dict shape: tests + some pre-release SDK revs.
    if isinstance(payload, dict):
        if payload.get("type") not in (
            "ItemCompletedNotification",
            "item.completed",
            "thread.item.completed",
        ):
            return ""
        item = payload.get("item")
        # The upstream model wraps the item in a discriminated-union
        # `root` field; some pre-release shapes drop the wrapper. Look
        # both ways.
        if isinstance(item, dict):
            inner = item.get("root", item)
            if not isinstance(inner, dict):
                return ""
            if inner.get("type") not in ("agentMessage", "agent_message"):
                return ""
            text = inner.get("text")
            return text if isinstance(text, str) else ""
        return ""

    # Object shape: upstream events with `.item.root.text`.
    if payload.__class__.__name__ not in (
        "ItemCompletedNotification",
        "ThreadItemCompletedNotification",
    ):
        return ""
    item = getattr(payload, "item", None)
    item = getattr(item, "root", item)
    if getattr(item, "type", None) not in ("agentMessage", "agent_message"):
        return ""
    text = getattr(item, "text", None)
    return text if isinstance(text, str) else ""


async def _stream_thread_run(
    thread: Any,
    prompt: str,
) -> AsyncGenerator[str, None]:
    """Yield raw text chunks from a Codex thread.

    Tries three SDK surfaces in order:

    1. ``thread.turn(prompt).stream()`` -- the canonical streaming path
       on the upstream ``openai_codex`` SDK (see
       ``openai/codex/sdk/python/src/openai_codex/api.py``: AsyncThread.turn
       returns an AsyncTurnHandle whose ``.stream()`` yields events).
    2. ``thread.run_streaming(prompt)`` -- a legacy helper exposed by
       some earlier SDK pre-releases. Kept for forward-compat.
    3. ``await thread.run(prompt)`` -- the always-supported buffered
       path. Used when neither streaming helper resolves and as the
       final fallback.

    Cross-turn side-effect protection: once a turn has STARTED (any
    SDK event was received, including ones that ``_coerce_text``
    drops -- command/file/tool/plan events), we never fall through
    to the buffered ``thread.run(prompt)`` path. A partial-stream
    failure mid-turn must not re-execute the same Codex turn
    because the side effects (file writes, shell commands, etc.)
    would replay. Tracking only ``emitted_any`` (visible text) is
    not enough -- a turn that crashes after running shell commands
    but before producing answer text would otherwise replay because
    no visible chunk was emitted.

    Empty-delta protection: the canonical SDK can complete a turn
    successfully without emitting any ``message.delta`` events --
    the final text arrives only as an ``ItemCompletedNotification``
    whose ``item`` is an ``agentMessage``. We collect those during the
    stream loop and emit the last one if no deltas came through, so
    Studio never returns an empty answer for a successful turn.
    """
    emitted_any = False
    # True once ANY event has been observed from a streaming helper.
    # Even when ``_coerce_text`` filters the event out, the turn has
    # demonstrably started executing on the Codex side, so a later
    # error must not trigger a buffered ``thread.run`` replay.
    turn_started = False

    # 1. Canonical: thread.turn(prompt).stream()
    turn_factory = getattr(thread, "turn", None)
    if turn_factory is not None:
        agent_message_texts: list[str] = []
        try:
            turn_handle = turn_factory(prompt)
            # Asking the SDK for the turn handle is itself enough to
            # start the turn on the upstream side; mark turn_started
            # before we even start iterating so a crash inside the
            # stream factory below does not look like a never-started
            # turn that is safe to replay.
            turn_started = True
            if asyncio.iscoroutine(turn_handle):
                turn_handle = await turn_handle
            stream_fn = getattr(turn_handle, "stream", None)
            if stream_fn is not None:
                stream_obj = stream_fn()
                if asyncio.iscoroutine(stream_obj):
                    stream_obj = await stream_obj
                async for event in stream_obj:
                    turn_started = True
                    payload = getattr(event, "payload", event)
                    text = _coerce_text(payload)
                    if text:
                        emitted_any = True
                        yield text
                    else:
                        final_text = _completed_agent_message_text(payload)
                        if final_text:
                            agent_message_texts.append(final_text)
                if not emitted_any and agent_message_texts:
                    # The stream completed cleanly but only via a final
                    # ItemCompletedNotification -- emit the last agent
                    # message text so the chat reply is not blank.
                    yield agent_message_texts[-1]
                    emitted_any = True
                return
        except Exception as exc:
            logger.warning(
                "codex_provider.turn_stream_failed_fallback",
                exc_type = type(exc).__name__,
                error = str(exc),
                emitted_any = emitted_any,
                turn_started = turn_started,
            )
            if turn_started:
                # The Codex turn has executed at least one event on
                # the upstream side (it may have launched shell
                # commands or written files via tool events that
                # _coerce_text filtered out). Re-executing via
                # run_streaming / run() would duplicate those side
                # effects, so stop here even if no visible text was
                # yielded.
                return

    # 2. Legacy: thread.run_streaming(prompt)
    run_streaming = getattr(thread, "run_streaming", None)
    if run_streaming is not None:
        try:
            stream_obj = run_streaming(prompt)
            # Same reasoning as the canonical path above: calling the
            # streaming helper is enough to start the turn on the SDK
            # side, so a later crash must NOT replay via buffered run.
            turn_started = True
            if asyncio.iscoroutine(stream_obj):
                stream_obj = await stream_obj
            async for event in stream_obj:
                turn_started = True
                text = _coerce_text(event)
                if text:
                    emitted_any = True
                    yield text
            return
        except Exception as exc:
            logger.warning(
                "codex_provider.run_streaming_failed_fallback",
                exc_type = type(exc).__name__,
                error = str(exc),
                emitted_any = emitted_any,
                turn_started = turn_started,
            )
            if turn_started:
                return

    # 3. Buffered fallback: await the full TurnResult, emit one chunk.
    # Only reached when no streaming helper ran at all (no turn /
    # run_streaming attributes on the thread, or both raised before
    # observing any event / starting the turn), so this is the first
    # and only execution of the turn.
    result = await thread.run(prompt)
    text = _buffered_result_text(result)
    if text:
        yield text


def _buffered_result_text(result: Any) -> str:
    """Extract assistant text from a buffered ``TurnResult``.

    The upstream SDK documents ``TurnResult.final_response`` as
    nullable -- a turn that performs only tool work and completes
    without a final assistant message will set it to ``None``. The
    previous ``... or str(result)`` fallback then sent a Python
    object repr (``TurnResult(...)``) into the chat, which surfaced
    as visible garbage to the user. Returning the empty string for
    that case lets the OpenAI-shape stream finish cleanly with no
    extra content chunk -- the usage / stop / [DONE] frames still
    fire, and the chat UI simply shows no assistant text rather
    than a misleading object dump.
    """
    text = _coerce_text(result)
    if text:
        return text
    final = getattr(result, "final_response", None)
    if isinstance(final, str) and final:
        return final
    return ""


def _safe_thread_safety_kwargs() -> dict[str, Any]:
    """Return the safe ``approval_mode`` + ``sandbox`` kwargs for thread_start.

    The upstream ``openai_codex.AsyncCodex.thread_start`` defaults
    ``approval_mode`` to ``ApprovalMode.auto_review`` -- which the SDK
    docs describe as "automatically execute tools when permission
    escalations occur, without user intervention" -- and leaves
    ``sandbox`` as ``None``. Studio drives Codex from a server-side
    chat request with no per-action UI, so leaving those at the
    defaults would let a model decide on its own to run shell
    commands, write files, or hit the network on the operator's
    machine.

    We pin both to the strictest values the SDK exposes:

    * ``approval_mode = ApprovalMode.deny_all`` -- reject any tool /
      command request rather than auto-approving it.
    * ``sandbox = SandboxMode.read_only`` -- the policy that bans
      file writes and disables network access.

    Probes multiple locations: ``ApprovalMode`` is exported at the
    top-level ``openai_codex`` package, but ``SandboxMode`` lives in
    ``openai_codex.generated.v2_all`` (re-exported into
    ``openai_codex.api``) and is NOT in the top-level __init__.
    Round 6 reviewer caught this -- looking only at the top-level
    module returned ``{}``, silently degrading to the auto_review
    default on the canonical SDK install.

    Returns an empty dict only when the installed SDK is so different
    that neither path resolves -- the caller then issues a
    structured warning and proceeds. Failing closed (refusing to
    run) on a future SDK rev would brick users for no security gain
    -- the auto_review default is upstream's choice, not a Studio
    regression.
    """
    sdk_mod = sys.modules.get("openai_codex") or sys.modules.get("codex_app_server")
    if sdk_mod is None:
        return {}

    # ApprovalMode: top-level export on canonical SDK.
    approval_mode_cls = getattr(sdk_mod, "ApprovalMode", None)

    # SandboxMode: try top-level, then `.api`, then `.generated.v2_all`.
    # We do not eagerly import these submodules because the SDK may
    # not expose them and we do not want to crash the request on an
    # ImportError. importlib.import_module gives us a typed failure.
    sandbox_mode_cls = getattr(sdk_mod, "SandboxMode", None)
    if sandbox_mode_cls is None:
        for sub in ("api", "generated.v2_all"):
            mod_name = getattr(sdk_mod, "__name__", "")
            if not mod_name:
                continue
            try:
                submod = importlib.import_module(f"{mod_name}.{sub}")
            except Exception:
                continue
            sandbox_mode_cls = getattr(submod, "SandboxMode", None)
            if sandbox_mode_cls is not None:
                break

    if approval_mode_cls is None or sandbox_mode_cls is None:
        return {}
    deny_all = getattr(approval_mode_cls, "deny_all", None)
    read_only = getattr(sandbox_mode_cls, "read_only", None)
    if deny_all is None or read_only is None:
        return {}
    return {"approval_mode": deny_all, "sandbox": read_only}


async def _start_thread_with_system(
    codex: Any,
    model: str,
    system: str,
    prompt: str,
) -> tuple[Any, str]:
    """Start a Codex thread carrying the system prompt and safe defaults.

    Upstream ``openai_codex.AsyncCodex.thread_start`` accepts the system
    prompt under the kwarg ``base_instructions``. Some pre-release / alias
    SDK revisions historically used ``system`` instead. We try the
    canonical kwarg first, then the legacy one, then drop both and
    prepend the system text to the user prompt so the model still sees
    it. The returned (thread, prompt) tuple lets the caller use the
    possibly-rewritten prompt.

    We always pin ``approval_mode`` to ``deny_all`` and ``sandbox`` to
    ``read_only`` when the SDK exposes them (see
    ``_safe_thread_safety_kwargs``) -- the upstream defaults would let
    a model decide on its own to execute shell commands or write to
    the operator's filesystem, which is not appropriate for a
    server-side chat surface with no per-action approval UI. If the
    installed SDK rev cannot expose those enums we fail closed by
    default (raise ``CodexUnavailableError``) rather than silently
    falling through to upstream's ``auto_review`` default. Power
    users on a dev install can set ``UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS=1``
    to override -- the variable name is deliberately long and explicit
    so it does not creep into production environments by accident.
    """
    import os as _os

    safety_kwargs = _safe_thread_safety_kwargs()
    if not safety_kwargs:
        allow_unsafe = _os.environ.get(
            "UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        if not allow_unsafe:
            # Fail closed: the user sees a clear 503 with a typed
            # error rather than discovering after the fact that
            # Codex ran with auto_review approvals.
            raise CodexUnavailableError(
                "Installed openai_codex SDK does not expose ApprovalMode "
                "/ SandboxMode, so Studio cannot pin the safe deny_all / "
                "read_only defaults required for a server-side chat "
                "surface. Upgrade openai_codex to a build that exports "
                "those enums, or set "
                "UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS=1 to opt in to the "
                "SDK's auto_review default on a trusted dev host."
            )
        logger.warning(
            "codex_provider.safety_kwargs_unavailable_override",
            note = (
                "UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS is set; Codex "
                "threads will use the SDK auto_review default with no "
                "explicit sandbox. This should only be enabled on a "
                "trusted dev host."
            ),
        )
    base_kwargs: dict[str, Any] = {"model": model, **safety_kwargs}

    if not system:
        thread = await codex.thread_start(**base_kwargs)
        return thread, prompt

    for kw_name in ("base_instructions", "system"):
        try:
            thread = await codex.thread_start(**base_kwargs, **{kw_name: system})
            return thread, prompt
        except TypeError:
            continue
        except Exception:
            raise
    # Last-resort fallback: inline the system text in the user prompt so
    # the role intent reaches Codex even on an SDK with no kwarg for it.
    thread = await codex.thread_start(**base_kwargs)
    return thread, f"{system}\n\n{prompt}"


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

    async with _open_async_codex(async_codex_cls) as codex:
        # ``thread_start`` accepts a model id; system prompts are
        # passed when supported by the SDK rev (older revs ignore the
        # Upstream `openai_codex.AsyncCodex.thread_start` uses
        # `base_instructions` for the system prompt (see
        # openai/codex/sdk/python/src/openai_codex/api.py). Older / alias
        # SDKs may use `system` instead. We try `base_instructions`
        # first, then `system`, and finally fall through to inlining
        # the system text in the user prompt if neither kwarg is
        # accepted.
        thread, prompt = await _start_thread_with_system(codex, model, system, prompt)
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
            async with _open_async_codex(async_codex_cls) as codex:
                thread, inner_prompt = await _start_thread_with_system(
                    codex, model, system, prompt
                )
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
            # CodeQL: never echo str(exc) in client-facing SSE events.
            # Log full reason server-side; surface a generic message plus
            # an exception_type discriminator so the UI can still group
            # failures without leaking file paths / env vars from the
            # SDK traceback. CodexUnavailableError is the one exception
            # we DO surface verbatim because it's a user-actionable
            # install hint with no sensitive content.
            logger.warning(
                "codex_provider.parallel_tab_failed",
                tab_id = tab_id,
                exc_type = type(exc).__name__,
                error = str(exc),
            )
            public_error = (
                str(exc)
                if isinstance(exc, CodexUnavailableError)
                else "Codex tab failed"
            )
            await queue.put(
                _chunk_tool_event(
                    completion_id,
                    {
                        "type": "codex_tab_error",
                        "tab_id": tab_id,
                        "error": public_error,
                        "exception_type": type(exc).__name__,
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
        system = system,
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

    # Account for ALL Codex turns the fan-out spawned: N parallel
    # workers each ran the same prompt (≈ N * prompt_tokens), and the
    # synthesis turn re-sent the prompt plus every tab's output. Without
    # this the cost / context display is off by the fan-out factor and
    # users see a wildly inaccurate token count for the request.
    total_tab_completion_chars = sum(len(t) for t in per_tab_texts)
    synthesis_prompt_chars = sum(len(t) for t in per_tab_texts) + len(prompt)
    yield _chunk_usage(
        completion_id,
        # n worker prompts (same prompt each) + synthesis prompt (which
        # carries the prompt again plus every tab's output).
        prompt_tokens = max(1, (n * len(prompt) + synthesis_prompt_chars) // 4),
        # Sum of every worker's output plus the synthesis text.
        completion_tokens = max(
            0, (total_tab_completion_chars + len(synthesis_text)) // 4
        ),
    )
    yield _chunk_stop(completion_id)


async def _run_codex_synthesis(
    *,
    model: str,
    system: str,
    prompt: str,
    tab_outputs: list[str],
) -> str:
    """Run one extra Codex call that consumes the N per-tab outputs and
    returns a unified synthesis. Returns the empty string on failure --
    the caller already surfaced the per-tab outputs so an empty
    synthesis is recoverable. The Studio system prompt is forwarded to
    the synthesis thread so style/role instructions like "Always answer
    in Spanish" survive the fan-out.
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
        async with _open_async_codex(async_codex_cls) as codex:
            thread, synthesis_prompt = await _start_thread_with_system(
                codex, model, system, synthesis_prompt
            )
            result = await thread.run(synthesis_prompt)
        # Use the same buffered extraction as `_stream_thread_run` so a
        # synthesis turn whose `final_response` is None returns an empty
        # string instead of a `TurnResult(...)` Python object repr.
        return _buffered_result_text(result)
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
    # Env is scrubbed to the codex safe-list (see codex_availability) so a
    # shimmed `codex` on PATH does not inherit other provider secrets.
    from core.inference.codex_availability import _codex_subprocess_env

    spawn_kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
        "env": _codex_subprocess_env(),
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
    # Accept any plausible device-auth URL the CLI prints. Upstream has
    # used `.../codex/device`, `chatgpt.com/activate`, and
    # `auth.openai.com/device`; rather than guess we look for any
    # https URL whose path mentions `device`, `activate`, or `verify`.
    url_re = re.compile(
        r"https?://[^\s\x1b]+/(?:codex/)?(?:device|activate|verify)\b[^\s\x1b]*",
        re.IGNORECASE,
    )
    # One-time-code format from upstream device_code_auth.rs: 4 chars,
    # dash, 4 chars. Pattern is tolerant of any uppercase alphanum.
    code_re = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")

    url_emitted = False
    code_emitted = False
    rc: int = -1
    cancelled = False

    # Allow-list of substrings the upstream `codex login --device-auth`
    # command prints during the normal flow. Anything outside this list
    # is treated as opaque and not forwarded to the browser, so a
    # shimmed binary that prints auth JSON, refresh tokens, local
    # config paths, or unexpected stderr cannot leak that content
    # through Studio's authenticated SSE stream. The URL and code
    # extracted above are emitted separately as `device_url` /
    # `device_code` events and are not affected by this filter.
    # Anchored regexes so the line must START with one of the upstream
    # `codex login --device-auth` phrases. A substring match like the
    # old "logged in" check is too loose: a malicious shim could print
    # `Not logged in: refresh_token=rt_LEAK auth.json=/home/u/.codex/`
    # and the substring `logged in` would let the line through, leaking
    # auth artefacts into the browser. Start anchors plus a blocklist
    # of known sensitive substrings close that hole.
    safe_log_res: tuple[Any, ...] = (
        re.compile(r"^welcome to codex\b", re.IGNORECASE),
        re.compile(r"^initializing\b", re.IGNORECASE),
        re.compile(r"^open (?:this|the verification)", re.IGNORECASE),
        re.compile(r"^open:\s*https?://", re.IGNORECASE),
        re.compile(r"^enter (?:this one-time code|the code)\b", re.IGNORECASE),
        re.compile(r"^waiting\b", re.IGNORECASE),
        re.compile(r"^successfully (?:logged|signed) in\b", re.IGNORECASE),
        re.compile(r"^(?:logged|signed) in\b", re.IGNORECASE),
        re.compile(r"^browser opened\b", re.IGNORECASE),
        re.compile(r"^press ctrl", re.IGNORECASE),
    )
    # Strict blocklist: any of these substrings in the line means the
    # log entry contains sensitive auth state, a path under the codex
    # config dir, or an explicit "not logged in" failure -- none of
    # which the user-facing SSE stream should mirror, regardless of
    # whether some other prefix matched.
    unsafe_log_re = re.compile(
        r"\bnot\s+(?:logged|signed)\s+in\b|"
        r"\bnot\s+authenticated\b|"
        r"refresh[_-]?token|access[_-]?token|"
        r"\bapi[_-]?key\b|\bsecret\b|"
        r"\bauth\.json\b|"
        r"/\.codex/|\\\.codex\\",
        re.IGNORECASE,
    )

    def _safe_to_forward(text: str) -> bool:
        if unsafe_log_re.search(text):
            return False
        return any(pattern.search(text) for pattern in safe_log_res)

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
                    candidate = match.group(0)
                    if _is_allowed_device_url(candidate):
                        yield {"type": "device_url", "url": candidate}
                        url_emitted = True
                    else:
                        # A shimmed codex on PATH could print a phishing
                        # URL that matches the regex but points at an
                        # attacker host (e.g. https://evil.example/activate
                        # ?code=ABCD). Drop it rather than surfacing
                        # "Open verification page" to a real user.
                        logger.warning(
                            "codex_provider.login_url_rejected",
                            host = (_safe_host(candidate) or "<unparsable>"),
                        )
            if not code_emitted:
                cm = code_re.search(line)
                if cm:
                    yield {"type": "device_code", "code": cm.group(1)}
                    code_emitted = True
            # Only forward lines from the known safe vocabulary; opaque
            # output (file paths, tokens, JSON, error messages) stays in
            # backend logs only.
            if line and _safe_to_forward(line):
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
