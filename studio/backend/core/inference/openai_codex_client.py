# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fixed-host ChatGPT Codex Responses transport and SSE normalization."""

from __future__ import annotations

import asyncio

from contextlib import asynccontextmanager
import hashlib
import json
import secrets
import threading
import time
from typing import Any, AsyncGenerator, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from core.inference import openai_codex_auth as codex_auth
from core.inference.openai_codex_auth import (
    OPENAI_CODEX_CLIENT_VERSION,
    OPENAI_CODEX_COMPATIBILITY_INSTRUCTIONS,
    OPENAI_CODEX_MODELS_URL,
    OPENAI_CODEX_ORIGINATOR,
    OPENAI_CODEX_RESPONSES_URL,
    OPENAI_CODEX_USER_AGENT,
    CodexReauthorizationRequired,
)
from core.inference.openai_responses_shared import (
    normalize_function_schema,
    responses_function_call,
    responses_function_output,
    response_event_type,
    responses_usage_to_chat,
)


class CodexTransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int = 502,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}


class CodexReauthorizationError(CodexTransportError):
    pass


class CodexQuotaError(CodexTransportError):
    pass


def _codex_call_id(value: Any) -> str | None:
    """Return a stable Responses-compatible call ID (the API caps IDs at 64 chars)."""
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= 64:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{value[:31]}_{digest}"


def _responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Translate chat history to the Responses input shape used by Pi/Codex."""
    instructions = [OPENAI_CODEX_COMPATIBILITY_INSTRUCTIONS]
    items: list[dict[str, Any]] = []
    assistant_index = 0
    for message in messages:
        role, content = message.get("role"), message.get("content", "")
        if role == "system":
            if isinstance(content, str) and content:
                instructions.append(content)
            elif isinstance(content, list):
                instructions.extend(
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                    and isinstance(part.get("text"), str)
                    and part["text"]
                )
            continue
        if role == "tool":
            call_id = _codex_call_id(message.get("tool_call_id"))
            if call_id:
                output = content if isinstance(content, str) else json.dumps(content)
                items.append(responses_function_output(call_id, output))
            continue
        if role == "assistant":
            extra = message.get("extra_content")
            for item in extra.get("openai_codex_reasoning", []) if isinstance(extra, dict) else []:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "reasoning"
                    and isinstance(item.get("encrypted_content"), str)
                ):
                    replay = {
                        "type": "reasoning",
                        "encrypted_content": item["encrypted_content"],
                        "summary": item.get("summary", []),
                    }
                    if isinstance(item.get("id"), str):
                        replay["id"] = item["id"]
                    items.append(replay)
            function_calls: list[dict[str, Any]] = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and function.get("name"):
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    call_id = _codex_call_id(call.get("id"))
                    if call_id:
                        function_calls.append(
                            responses_function_call(call_id, function["name"], arguments)
                        )
            output_parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                output_parts.append({"type": "output_text", "text": content, "annotations": []})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        output_parts.append(
                            {"type": "output_text", "text": part.get("text", ""), "annotations": []}
                        )
            if output_parts:
                items.append(
                    {
                        "type": "message",
                        "id": f"msg_unsloth_{assistant_index}",
                        "role": "assistant",
                        "content": output_parts,
                        "status": "completed",
                    }
                )
                assistant_index += 1

            items.extend(function_calls)
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            if content:
                items.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({"type": "input_text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        parts.append({"type": "input_image", "detail": "auto", "image_url": url})
            if parts:
                items.append({"role": "user", "content": parts})
    return "\n\n".join(instructions), items


def _chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: Any = None,
) -> str:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return "data: " + json.dumps(body, separators = (",", ":"))


def _create_http_client() -> httpx.AsyncClient:
    kwargs = {"timeout": httpx.Timeout(120.0, connect = 20.0), "follow_redirects": False}
    try:
        return httpx.AsyncClient(**kwargs)
    except (ImportError, ValueError) as exc:
        if "Unknown scheme for proxy URL" not in str(exc) and "socksio" not in str(exc):
            raise
        return httpx.AsyncClient(**kwargs, trust_env = False)


def _validated_responses_url() -> str:
    parsed = urlparse(OPENAI_CODEX_RESPONSES_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "chatgpt.com"
        or parsed.port is not None
        or parsed.path != "/backend-api/codex/responses"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError("ChatGPT Codex endpoint configuration is invalid.")
    return OPENAI_CODEX_RESPONSES_URL


def _validated_models_url() -> str:
    parsed = urlparse(OPENAI_CODEX_MODELS_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "chatgpt.com"
        or parsed.port is not None
        or parsed.path != "/backend-api/codex/models"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError("ChatGPT Codex model endpoint configuration is invalid.")
    return OPENAI_CODEX_MODELS_URL


_MODELS_CACHE_TTL_SECONDS = 600
_MODELS_CACHE_MAX_ENTRIES = 32
_models_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# Outlives the cache TTL: a slug listed for this plan stays saveable afterwards.
_offered_models: dict[str, dict[str, dict[str, Any]]] = {}
# Which ChatGPT account each cached catalog belongs to; a reauthorization can rebind
# a connection to a different account whose plan lists different slugs.
_catalog_accounts: dict[str, str] = {}


def _normalize_subscription_model(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    slug = item.get("slug")
    if not isinstance(slug, str) or not slug or len(slug) > 128:
        return None
    display_name = item.get("display_name")
    context_window = item.get("context_window")
    modalities = item.get("input_modalities")
    # Every other field here tolerates whatever upstream sends, and this one has to as
    # well: `or []` only covers a falsy value, so a scalar raised TypeError out of the
    # whole call and cost the catalog every other entry too, not just this one.
    levels = item.get("supported_reasoning_levels")
    efforts = [
        level["effort"]
        for level in (levels if isinstance(levels, list) else [])
        if isinstance(level, dict) and isinstance(level.get("effort"), str)
    ]
    return {
        "id": slug,
        "display_name": display_name if isinstance(display_name, str) and display_name else slug,
        # bool is a subclass of int, so a JSON `true` would otherwise be reported to the
        # picker as a context length of its own.
        "context_length": (
            context_window
            if isinstance(context_window, int) and not isinstance(context_window, bool)
            else None
        ),
        "vision": "image" in modalities if isinstance(modalities, list) else None,
        "reasoning_efforts": efforts,
        # "hide" marks a slug no picker should offer (codex-auto-review, and models that
        # age out of the list). It is a presentation flag, not a revocation: the account
        # can still call one it already saved, so the entry is kept and marked instead of
        # dropped, which is what lets callers tell "not offered" from "not on this plan".
        "listed": item.get("visibility") == "list",
    }


def cached_subscription_models(provider_id: str) -> list[dict[str, Any]] | None:
    entry = _models_cache.get(provider_id)
    if entry is None or entry[0] <= time.time():
        return None
    return entry[1]


def offered_subscription_model_ids(provider_id: str) -> set[str]:
    """Slugs the plan offers, and so the only ones a fetch alone may authorize.

    Hidden entries are cached for their metadata and stay usable when they are already
    on a connection, but a slug the picker never offered must not become invocable just
    because a catalog fetch happened.
    """
    return {
        model_id
        for model_id, model in _offered_models.get(provider_id, {}).items()
        if model.get("listed")
    }


# Connections whose catalog was dropped because the account behind them changed. The
# absence is deliberate, so it must not read as "nothing fetched yet" and license the
# previous account's saved slugs.
_stale_catalogs: set[str] = set()


# The ticket the newest catalog read for each connection is holding, so a read that was
# overtaken (by a rebind, a disconnect, or a newer read) cannot commit its result over
# the one that replaced it.
_catalog_requests: dict[str, int] = {}
# Tickets are drawn from one counter shared by every connection rather than counting up
# per connection. Nothing reads the number itself, only whether it still matches, and a
# value that is never reissued is what lets forget_subscription_models drop the entry
# instead of leaving a larger one behind: a read still in flight then finds no ticket at
# all, and the next read draws a number no earlier read can be holding.
_catalog_request_serial = 0


def _begin_catalog_request(provider_id: str) -> int:
    global _catalog_request_serial
    _catalog_request_serial += 1
    _catalog_requests[provider_id] = _catalog_request_serial
    return _catalog_request_serial


def subscription_catalog_matches_account(provider_id: str, account_id: str | None) -> bool:
    """Whether the catalog held for this connection belongs to the account named.

    The OAuth bundle is shared through the installation DB but the catalog is per
    process, so another worker can rebind a connection this one still has a catalog for.
    """
    known = _catalog_accounts.get(provider_id)
    return known is None or account_id is None or known == account_id


def mark_subscription_catalog_stale(provider_id: str) -> None:
    _stale_catalogs.add(provider_id)


def subscription_catalog_stale(provider_id: str) -> bool:
    return provider_id in _stale_catalogs


def saved_models_proven_for(provider_id: str, account_id: str | None) -> bool:
    """Whether the row's saved models are on record as validated against this account.

    Consulted when this process holds no catalog: the in-memory mark is gone after a
    restart and never existed on a cold worker, so the record kept with the credentials
    is the only thing that still knows a rebind happened.
    """
    if account_id is None:
        return True
    bundle = codex_auth.load_oauth_bundle(provider_id)
    return bool(bundle) and bundle.get("catalog_account_id") == account_id


def subscription_catalog_known(provider_id: str) -> bool:
    """Whether this process has read a catalog for the connection at all.

    Without one the saved row is the only evidence there is; with one, a slug the plan
    does not carry has genuinely gone, which is how a reauthorization to another account
    retires the previous account's selections.
    """
    return provider_id in _offered_models


def offered_subscription_model(provider_id: str, model_id: str) -> dict[str, Any] | None:
    """What the plan said about one slug, for models the static registry cannot describe."""
    return _offered_models.get(provider_id, {}).get(model_id)


def forget_subscription_models(provider_id: str) -> None:
    # Retire any read still in flight: its result describes what was just dropped. The
    # ticket is dropped rather than bumped, so this releases the entry instead of
    # replacing it; see the counter above for why that is still safe.
    _catalog_requests.pop(provider_id, None)
    _models_cache.pop(provider_id, None)
    _offered_models.pop(provider_id, None)
    _catalog_accounts.pop(provider_id, None)
    _stale_catalogs.discard(provider_id)


async def list_subscription_models(
    provider_id: str,
    access_token: str,
    account_id: str,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Model slugs this ChatGPT plan can reach; anything else is a 400 upstream.

    ``force`` skips the cache for an explicit user reload: a plan change or a slug
    rolled out since the last fetch is exactly what that click is asking about.
    """
    if _catalog_accounts.get(provider_id) not in (None, account_id):
        # Reauthorized against a different account: the previous plan's catalog says
        # nothing about this one, and nothing else clears it on the reconnect path.
        forget_subscription_models(provider_id)
    if not force:
        cached = cached_subscription_models(provider_id)
        if cached is not None:
            return cached

    ticket = _begin_catalog_request(provider_id)
    url = _validated_models_url()

    def _headers(token: str, account: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account,
            "originator": OPENAI_CODEX_ORIGINATOR,
            "User-Agent": OPENAI_CODEX_USER_AGENT,
            "Accept": "application/json",
        }

    client = _create_http_client()
    try:
        response = await client.get(
            url,
            headers = _headers(access_token, account_id),
            params = {"client_version": OPENAI_CODEX_CLIENT_VERSION},
        )
        if response.status_code == 401:
            # Upstream can reject a token before its recorded expiry while the refresh
            # credential is still good. The responses transport spends one forced refresh
            # on that; without the same here the editor cannot load a catalog for a
            # connection whose chats recover fine.
            try:
                access_token, account_id = await codex_auth.resolve_access(
                    provider_id,
                    force_refresh = True,
                    expected_access_token = access_token,
                )
            except CodexReauthorizationRequired as exc:
                raise CodexReauthorizationError(
                    "ChatGPT authorization expired. Reconnect this connection.",
                    status = 401,
                ) from exc
            except Exception as exc:
                # The refresh did not get an answer, which is retryable. Calling it a
                # reauthorization would send the user to reconnect a connection whose
                # credentials are probably fine, and the responses transport treats the
                # same failure as transient.
                raise CodexTransportError("Could not refresh ChatGPT authorization.") from exc
            response = await client.get(
                url,
                headers = _headers(access_token, account_id),
                params = {"client_version": OPENAI_CODEX_CLIENT_VERSION},
            )
        if response.status_code == 401:
            # A freshly refreshed token was rejected too, so the connection really is
            # done. Record it against that token the way the streaming path does, or
            # auth_status keeps reporting connected, the editor never offers Reconnect
            # and every later catalog load repeats this.
            # Under the same guard the streaming error path uses: the read and the write
            # inside are one step, so a rotation landing between them cannot be undone by
            # writing the rejected bundle back with the marker on it.
            async with codex_auth.provider_oauth_write_guard(provider_id):
                codex_auth.mark_reauthorization_required(
                    provider_id, expected_access_token = access_token
                )
            raise CodexReauthorizationError(
                "ChatGPT authorization expired. Reconnect this connection.",
                status = 401,
            )
        if response.status_code != 200:
            detail = await _upstream_error_detail(response)
            suffix = f" {detail}" if detail else ""
            raise CodexTransportError(
                f"Could not list ChatGPT Codex models ({response.status_code}).{suffix}",
                status = response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CodexTransportError("ChatGPT returned an unreadable model list.") from exc
    except httpx.HTTPError as exc:
        raise CodexTransportError("Could not reach ChatGPT Codex.") from exc
    finally:
        await client.aclose()

    raw = payload.get("models") if isinstance(payload, dict) else None
    models: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for item in raw or []:
        model = _normalize_subscription_model(item)
        if model is None:
            continue
        if model["id"] in seen_slugs:
            # A slug repeated in one payload describes itself twice, and the list and the
            # by-id map built from it below disagree about which description won: the
            # route offers what the first entry said while the chat gate judges by the
            # last, so a duplicate whose second entry is hidden had the picker offering a
            # model every send then refused. First wins, so both read the same entry.
            continue
        seen_slugs.add(model["id"])
        models.append(model)
    if not any(model.get("listed") for model in models):
        # Nothing offerable came back. The route answers with the curated seed for this,
        # so committing it as a known catalog would leave the picker offering models that
        # validation and chat, reading that same empty catalog, would then refuse.
        return models
    if _catalog_requests.get(provider_id) != ticket:
        # Overtaken while this request was out. Committing now would reinstate a catalog
        # for an account this connection may no longer be on, and clear the mark that
        # says so, so hand the caller the models without storing them.
        return models
    # That counter only sees this process. Rebinding travels through the installation DB,
    # so ask it who owns the connection now before recording this as its catalog.
    current_bundle = codex_auth.load_oauth_bundle(provider_id)
    if current_bundle and current_bundle.get("account_id") != account_id:
        return models
    if len(_models_cache) >= _MODELS_CACHE_MAX_ENTRIES:
        # Only the TTL response cache is bounded here. The per-account authorization
        # evidence deliberately outlives it: dropping it would make every other
        # connection look cold and license saved slugs their account no longer carries.
        # It is keyed by connection and released by forget_subscription_models, so it is
        # bounded by the connections that exist rather than by fetches.
        _models_cache.clear()
    _models_cache[provider_id] = (time.time() + _MODELS_CACHE_TTL_SECONDS, models)
    _offered_models[provider_id] = {model["id"]: model for model in models}
    _catalog_accounts[provider_id] = account_id
    _stale_catalogs.discard(provider_id)
    return models


async def ensure_subscription_models(provider_id: str) -> set[str]:
    """The plan's slugs, fetching them once when this process has none yet.

    The catalog lives in memory, so a restart leaves a saved dynamic slug with
    nothing to authorize it. Callers use this before refusing such a model; an
    unreachable or disconnected upstream returns empty so the caller falls back
    to the seed rather than locking the account out.
    """
    listed = offered_subscription_model_ids(provider_id)
    if listed:
        return listed
    try:
        access_token, account_id = await codex_auth.resolve_access(provider_id)
        models = await list_subscription_models(provider_id, access_token, account_id)
    except (codex_auth.CodexAuthError, CodexReauthorizationError):
        # The connection needs reconnecting, which is a different answer from "this plan
        # does not list that model". Callers turn this into the reauthorization error the
        # user can act on instead of a misleading model-choice rejection.
        raise
    except Exception:
        return set()
    listed = offered_subscription_model_ids(provider_id)
    if listed:
        return listed
    # A read overtaken by a newer one returns its models without storing them. That
    # answer still came from upstream for the account resolved above, so use it rather
    # than reporting an empty plan; only a rebind makes it the wrong account's.
    current_bundle = codex_auth.load_oauth_bundle(provider_id)
    if current_bundle and current_bundle.get("account_id") == account_id:
        return {model["id"] for model in models if model.get("listed")}
    return set()


def _retry_after_ms_seconds(response: httpx.Response) -> str | None:
    """``retry-after-ms`` as whole seconds. The client's own backoff already reads this header,
    so dropping it here left only it honoured and the caller's retry guessing."""
    raw = response.headers.get("retry-after-ms")
    if not raw:
        return None
    try:
        return str(max(0.0, float(raw) / 1000.0))
    except ValueError:
        return None


def _quota_metadata(response: httpx.Response, *, terminal: bool = False) -> dict[str, Any]:
    values = {
        "retry_after": response.headers.get("retry-after") or _retry_after_ms_seconds(response),
        "requests_reset": response.headers.get("x-ratelimit-reset-requests"),
        "tokens_reset": response.headers.get("x-ratelimit-reset-tokens"),
    }
    metadata = {key: value for key, value in values.items() if value}
    if terminal:
        # Exhausted, not throttled. Both leave as a 429, so without this a client waits out
        # a delay that no wait can clear.
        metadata["terminal"] = True
    return metadata


async def _upstream_error_detail(response: httpx.Response) -> str | None:
    """Extract only structured, bounded upstream error text; never echo HTML."""
    try:
        await response.aread()
        payload = response.json()
    except (ValueError, json.JSONDecodeError, httpx.HTTPError, AttributeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code") or error.get("type")
        detail = message if isinstance(message, str) and message.strip() else code
    elif isinstance(error, str):
        detail = error
    else:
        # The subscription endpoint reports rejected models as a bare "detail".
        detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, str):
        return None
    return " ".join(detail.split())[:500] or None


async def _upstream_error_code(response: httpx.Response) -> str | None:
    """The structured error's code or type. _upstream_error_detail prefers the display message
    and drops these, which hides a terminal code behind a generic "slow down" sentence."""
    try:
        # Same read-then-parse as the sibling above, so this does not depend on being called
        # after it. A body whose read already failed raises StreamError, not HTTPError.
        await response.aread()
        payload = response.json()
    except (ValueError, json.JSONDecodeError, httpx.HTTPError, httpx.StreamError, AttributeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code") or error.get("type")
    return code if isinstance(code, str) and code.strip() else None


async def _wait_for_cancel(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.05)


@asynccontextmanager
async def _stream_response(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    cancel_event: threading.Event | None,
):
    """Open a streaming response while allowing pre-header cancellation."""
    context = client.stream("POST", url, headers = headers, json = body)
    if cancel_event is None:
        async with context as response:
            yield response
        return
    if cancel_event.is_set():
        yield None
        return

    enter_task = asyncio.create_task(context.__aenter__())
    cancel_task = asyncio.create_task(_wait_for_cancel(cancel_event))
    done, _pending = await asyncio.wait(
        {enter_task, cancel_task}, return_when = asyncio.FIRST_COMPLETED
    )
    if cancel_task in done and cancel_event.is_set():
        enter_task.cancel()
        await asyncio.gather(enter_task, return_exceptions = True)
        yield None
        return

    cancel_task.cancel()
    await asyncio.gather(cancel_task, return_exceptions = True)
    response = await enter_task
    try:
        yield response
    finally:
        await context.__aexit__(None, None, None)


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 2


def _is_terminal_quota(detail: str | None) -> bool:
    if not detail:
        return False
    lowered = detail.lower()
    return any(
        marker in lowered
        for marker in (
            "usage limit",
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "available balance",
            "billing",
        )
    )


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("retry-after-ms")
    if raw:
        try:
            return min(60.0, max(0.0, float(raw) / 1000.0))
        except ValueError:
            pass
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return min(60.0, max(0.0, float(raw)))
        except ValueError:
            pass
    return float(2**attempt)


async def _retry_pause(delay: float, cancel_event: threading.Event | None) -> None:
    deadline = asyncio.get_running_loop().time() + delay
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.1, remaining))


@asynccontextmanager
async def _validated_stream_response(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    cancel_event: threading.Event | None,
    refresh_access: Callable[[], Awaitable[tuple[str, str]]] | None,
):
    refreshed = False

    token = headers.get("Authorization", "").removeprefix("Bearer ")
    for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
        yielded = False
        try:
            async with _stream_response(
                client,
                url = url,
                headers = headers,
                body = body,
                cancel_event = cancel_event,
            ) as response:
                if response is None:
                    yield None
                    return
                if 200 <= response.status_code < 300:
                    yielded = True
                    yield response
                    return
                if 300 <= response.status_code < 400:
                    raise CodexTransportError(
                        "ChatGPT Codex endpoint returned a forbidden redirect."
                    )
                detail = await _upstream_error_detail(response)
                if response.status_code == 401 and refresh_access is not None and not refreshed:
                    try:
                        token, account_id = await refresh_access()
                    except CodexReauthorizationRequired as exc:
                        raise CodexReauthorizationError(
                            "ChatGPT authorization expired. Reconnect this connection.",
                            status = 401,
                            metadata = {"access_token": token},
                        ) from exc
                    except Exception as exc:
                        raise CodexTransportError(
                            "Could not refresh ChatGPT authorization. Please retry.",
                            status = 502,
                        ) from exc
                    headers["Authorization"] = f"Bearer {token}"
                    headers["chatgpt-account-id"] = account_id
                    refreshed = True
                    continue
                if response.status_code == 401:
                    raise CodexReauthorizationError(
                        "ChatGPT authorization expired. Reconnect this connection.",
                        status = 401,
                        metadata = {"access_token": token},
                    )
                terminal_quota = _is_terminal_quota(detail) or _is_terminal_quota(
                    await _upstream_error_code(response)
                )
                retryable = response.status_code in _RETRYABLE_STATUSES and not terminal_quota
                if retryable and attempt < _MAX_TRANSIENT_RETRIES:
                    await _retry_pause(_retry_delay_seconds(response, attempt), cancel_event)
                    if cancel_event is not None and cancel_event.is_set():
                        yield None
                        return
                    continue
                if response.status_code == 429:
                    raise CodexQuotaError(
                        "ChatGPT subscription quota is temporarily unavailable.",
                        status = 429,
                        metadata = _quota_metadata(response, terminal = terminal_quota),
                    )
                suffix = f" {detail}" if detail else ""
                raise CodexTransportError(
                    f"ChatGPT Codex request failed ({response.status_code}).{suffix}",
                    status = response.status_code,
                )
        except httpx.HTTPError as exc:
            if yielded:
                raise
            if attempt >= _MAX_TRANSIENT_RETRIES:
                raise CodexTransportError("Could not reach ChatGPT Codex.") from exc
            await _retry_pause(float(2**attempt), cancel_event)
            if cancel_event is not None and cancel_event.is_set():
                yield None
                return
    raise CodexTransportError("Could not reach ChatGPT Codex.")


class OpenAICodexClient:
    def __init__(
        self,
        access_token: str,
        account_id: str,
        *,
        refresh_access: Callable[[], Awaitable[tuple[str, str]]] | None = None,
    ) -> None:
        self._token, self._account_id = access_token, account_id
        self._refresh_access = refresh_access
        self._client = _create_http_client()

    async def _refresh_credentials(self) -> tuple[str, str]:
        if self._refresh_access is None:
            raise CodexReauthorizationError(
                "ChatGPT authorization expired. Reconnect this connection.",
                status = 401,
            )
        token, account_id = await self._refresh_access()
        self._token, self._account_id = token, account_id
        return token, account_id

    async def close(self) -> None:
        self._token = ""
        await self._client.aclose()

    async def stream(
        self,
        *,
        provider_id: str,
        thread_id: str | None,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int | None,
        reasoning_effort: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        response_format: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        instructions, input_items = _responses_input(messages)
        conversation_id = thread_id or secrets.token_urlsafe(24)
        affinity = hashlib.sha256(
            f"{provider_id}\0{self._account_id}\0{conversation_id}".encode()
        ).hexdigest()[:48]
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "stream": True,
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": affinity,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        response_type = response_format.get("type") if isinstance(response_format, dict) else None
        if response_type == "json_object":
            body["text"]["format"] = {"type": "json_object"}
        elif response_type == "json_schema":
            schema = response_format.get("json_schema")
            if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
                body["text"]["format"] = {
                    "type": "json_schema",
                    "name": str(schema.get("name") or "response"),
                    "schema": schema["schema"],
                    "strict": bool(schema.get("strict", True)),
                }
        # Pi intentionally does not forward a token cap here. ChatGPT's Codex
        # Responses endpoint rejects max_output_tokens even though the public
        # Responses API accepts it; the subscription service applies its own cap.
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if tools:
            converted = []
            for tool in tools:
                function = (
                    tool.get("function")
                    if isinstance(tool, dict) and tool.get("type") == "function"
                    else None
                )
                if isinstance(function, dict) and function.get("name"):
                    converted.append(
                        {
                            "type": "function",
                            "name": function["name"],
                            "description": function.get("description", ""),
                            "parameters": normalize_function_schema(function.get("parameters")),
                        }
                    )
            if converted:
                body["tools"] = converted
        if isinstance(tool_choice, str) and tool_choice in ("auto", "none", "required"):
            body["tool_choice"] = tool_choice
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            if fn.get("name"):
                body["tool_choice"] = {"type": "function", "name": fn["name"]}
        request_id = hashlib.sha256(f"{affinity}\0{time.time_ns()}".encode()).hexdigest()[:32]
        headers = {
            "Authorization": f"Bearer {self._token}",
            "chatgpt-account-id": self._account_id,
            "originator": OPENAI_CODEX_ORIGINATOR,
            "User-Agent": OPENAI_CODEX_USER_AGENT,
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "session-id": affinity,
            "x-client-request-id": request_id,
        }
        completion_id = f"chatcmpl-codex-{request_id}"
        emitted_terminal, saw_tool_call = False, False
        reasoning_items: list[dict[str, Any]] = []
        cancel_task: asyncio.Task | None = None
        try:
            async with _validated_stream_response(
                self._client,
                url = _validated_responses_url(),
                headers = headers,
                body = body,
                cancel_event = cancel_event,
                refresh_access = (
                    self._refresh_credentials if self._refresh_access is not None else None
                ),
            ) as response:
                if response is None:
                    return
                if cancel_event is not None:

                    async def _close_on_cancel() -> None:
                        await _wait_for_cancel(cancel_event)
                        await response.aclose()

                    cancel_task = asyncio.create_task(_close_on_cancel())
                event_name, tool_indexes = "", {}
                try:
                    async for line in response.aiter_lines():
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        if not line:
                            event_name = ""
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                            kind = response_event_type(event, event_name)
                        except (ValueError, json.JSONDecodeError) as exc:
                            raise CodexTransportError(
                                "ChatGPT returned a malformed stream."
                            ) from exc
                        if kind in ("response.created", "response.in_progress"):
                            continue
                        if kind in ("response.output_text.delta", "response.refusal.delta"):
                            delta = event.get("delta")
                            if not isinstance(delta, str):
                                raise CodexTransportError("ChatGPT returned a malformed stream.")
                            yield _chunk(completion_id, model, {"content": delta})
                        elif kind in (
                            "response.reasoning_summary_text.delta",
                            "response.reasoning_text.delta",
                        ):
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                yield _chunk(completion_id, model, {"reasoning_content": delta})
                        elif kind == "response.output_item.added":
                            item = event.get("item")
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                call_id, name = (
                                    item.get("call_id") or item.get("id"),
                                    item.get("name"),
                                )
                                if isinstance(call_id, str) and isinstance(name, str):
                                    saw_tool_call = True
                                    index = len(set(tool_indexes.values()))
                                    tool_indexes[call_id] = index
                                    if isinstance(item.get("id"), str):
                                        tool_indexes[item["id"]] = index
                                    yield _chunk(
                                        completion_id,
                                        model,
                                        {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {"name": name, "arguments": ""},
                                                }
                                            ]
                                        },
                                    )
                        elif kind == "response.function_call_arguments.delta":
                            call_id, delta = (
                                event.get("call_id") or event.get("item_id"),
                                event.get("delta"),
                            )
                            if not isinstance(call_id, str) or not isinstance(delta, str):
                                raise CodexTransportError("ChatGPT returned a malformed stream.")
                            saw_tool_call = True
                            index = tool_indexes.setdefault(
                                call_id, len(set(tool_indexes.values()))
                            )
                            yield _chunk(
                                completion_id,
                                model,
                                {
                                    "tool_calls": [
                                        {"index": index, "function": {"arguments": delta}}
                                    ]
                                },
                            )
                        elif kind == "response.output_item.done":
                            item = event.get("item") or {}
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                call_id = item.get("call_id") or item.get("id")
                                if (
                                    isinstance(call_id, str)
                                    and call_id not in tool_indexes
                                    and isinstance(item.get("name"), str)
                                ):
                                    saw_tool_call = True
                                    index = tool_indexes.setdefault(
                                        call_id, len(set(tool_indexes.values()))
                                    )
                                    yield _chunk(
                                        completion_id,
                                        model,
                                        {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": item["name"],
                                                        "arguments": item.get("arguments", ""),
                                                    },
                                                }
                                            ]
                                        },
                                    )
                            elif (
                                isinstance(item, dict)
                                and item.get("type") == "reasoning"
                                and isinstance(item.get("encrypted_content"), str)
                            ):
                                reasoning_items.append(
                                    {
                                        "type": "reasoning",
                                        "id": item.get("id"),
                                        "encrypted_content": item["encrypted_content"],
                                        "summary": item.get("summary", []),
                                    }
                                )
                        elif kind in ("response.completed", "response.incomplete"):
                            response_body = event.get("response") or {}
                            delta: dict[str, Any] = {}
                            if reasoning_items:
                                delta["extra_content"] = {"openai_codex_reasoning": reasoning_items}
                            finish = (
                                "length"
                                if kind == "response.incomplete"
                                else ("tool_calls" if saw_tool_call else "stop")
                            )
                            yield _chunk(
                                completion_id,
                                model,
                                delta,
                                finish,
                                responses_usage_to_chat(response_body.get("usage")),
                            )
                            emitted_terminal = True
                            break
                        elif kind in ("response.failed", "error"):
                            error = (
                                event.get("error")
                                or (event.get("response") or {}).get("error")
                                or {}
                            )
                            code = error.get("code") if isinstance(error, dict) else None
                            raise CodexTransportError(
                                f"ChatGPT Codex generation failed{f' ({code})' if code else ''}."
                            )
                except httpx.HTTPError:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    raise
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
        if not emitted_terminal and not (cancel_event is not None and cancel_event.is_set()):
            raise CodexTransportError("ChatGPT stream ended before completion.")
