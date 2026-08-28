# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed admission policy for serving Unsloth without an API key.

Off by default. When an admin turns it on, a request that sends no usable
credential authenticates as the local admin, so ``curl`` and the OpenAI SDKs reach
this server the way they reach LM Studio and Ollama.

Two scopes, so opening up chat does not also open up training:

``inference``
    The OpenAI-compatible endpoints only, named one by one in
    ``_INFERENCE_ROUTES``. Everything else keeps needing a key.
``full``
    Every route, but only for callers arriving and connecting over loopback.

Server-side tools (python, terminal, web search) stay off for a keyless caller
whatever the scope, until the admin ticks them on separately: ``/v1/chat/completions``
runs that tool loop on this machine, so it is a bigger grant than chat itself.

Public tunnels and Colab never receive keyless access. Private-LAN inference is
accepted only through a live settings listener or the launch-managed bind that
matches the ASGI accepting address and port. Signing in to Unsloth is unaffected.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from typing import Any, Optional

KEYLESS_API_ACCESS_SETTING_KEY = "keyless_api_access_scope"
KEYLESS_API_TOOLS_SETTING_KEY = "keyless_api_access_tools"
DEFAULT_KEYLESS_API_TOOLS_ENABLED = False
KEYLESS_SCOPE_OFF = "off"
KEYLESS_SCOPE_INFERENCE = "inference"
KEYLESS_SCOPE_FULL = "full"
KEYLESS_SCOPES = (KEYLESS_SCOPE_OFF, KEYLESS_SCOPE_INFERENCE, KEYLESS_SCOPE_FULL)
DEFAULT_KEYLESS_API_ACCESS_SCOPE = KEYLESS_SCOPE_OFF
APPROVED_DUMMY_BEARERS = frozenset({"not-needed", "lm-studio", "ollama"})
KEYLESS_ADMISSION_STATE_KEY = "keyless_api_admitted"

# Named by method and normalized path: /v1 also aliases model loading, media,
# sandbox, validation, and streaming side-effect routes.
_INFERENCE_ROUTES = frozenset(
    {
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/count_tokens"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/models"),
        ("POST", "/v1/responses"),
    }
)


def _coerce_scope(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in KEYLESS_SCOPES:
        return value.strip().lower()
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


# each read opens its own sqlite connection (~0.5ms), so hold the answer for a moment
_SETTINGS_CACHE_TTL_S = 1.0
_cached_settings: Optional[tuple[float, str, bool]] = None
# bumped by every write, so a refresh can tell whether its read still describes the db
_settings_generation = 0
_cache_lock = threading.Lock()
_write_lock = threading.Lock()
_settings_refresh_inflight: Optional[object] = None
_settings_write_inflight: Optional[object] = None
_async_settings_tasks = weakref.WeakKeyDictionary()
_async_settings_pending_tasks: set[asyncio.Task] = set()


def _reset_scope_cache() -> None:
    """Test hook: forget settings cached before the DB was written directly."""
    global _cached_settings
    with _cache_lock:
        _cached_settings = None


def _read_settings_from_db() -> tuple[str, bool]:
    from storage.studio_db import get_app_settings

    values = get_app_settings([KEYLESS_API_ACCESS_SETTING_KEY, KEYLESS_API_TOOLS_SETTING_KEY])
    scope = _coerce_scope(values.get(KEYLESS_API_ACCESS_SETTING_KEY))
    tools = _coerce_bool(values.get(KEYLESS_API_TOOLS_SETTING_KEY))
    return (
        scope or DEFAULT_KEYLESS_API_ACCESS_SCOPE,
        DEFAULT_KEYLESS_API_TOOLS_ENABLED if tools is None else tools,
    )


def _read_settings() -> tuple[str, bool]:
    try:
        return _read_settings_from_db()
    except Exception:
        return KEYLESS_SCOPE_OFF, False


def _settings_once() -> tuple[str, bool, bool, int]:
    """Read the persisted scope and tool grant; anything unreadable counts as off.

    Unlike a normal setting these remove an authentication requirement, so a damaged
    settings DB must never resolve to an open scope, and neither may a refresh that
    read the DB before a write closed it: sqlite reads block, so a request can be
    holding the old answer when the setting is turned off, and publishing it would
    keep the server open for the rest of the TTL. The generation counter dates each
    read against the writes, so only a read that still describes the DB is published.

    One caller refreshes SQLite; async followers retry without worker tokens; sync
    followers fail closed.
    """
    global _cached_settings, _settings_refresh_inflight
    owner_marker: Optional[object] = None
    try:
        now = time.monotonic()
        with _cache_lock:
            if _settings_write_inflight is not None:
                return KEYLESS_SCOPE_OFF, False, False, _settings_generation
            cached = _cached_settings
            if cached is not None and now - cached[0] < _SETTINGS_CACHE_TTL_S:
                return cached[1], cached[2], False, _settings_generation
            if _settings_refresh_inflight is not None:
                return KEYLESS_SCOPE_OFF, False, True, _settings_generation
            owner_marker = _settings_refresh_inflight = object()
            generation = _settings_generation

        scope, tools = _read_settings()
        with _cache_lock:
            if _settings_write_inflight is not None:
                return KEYLESS_SCOPE_OFF, False, False, _settings_generation
            if generation == _settings_generation:
                _cached_settings = (time.monotonic(), scope, tools)
            published = _cached_settings
            published_generation = _settings_generation
        if published is not None:
            return published[1], published[2], False, published_generation
        return scope, tools, False, published_generation
    finally:
        if owner_marker is not None:
            with _cache_lock:
                if _settings_refresh_inflight is owner_marker:
                    _settings_refresh_inflight = None


def _settings() -> tuple[str, bool]:
    scope, tools, _pending, _generation = _settings_once()
    return scope, tools


async def _settings_async() -> tuple[str, bool, int]:
    return await asyncio.shield(_async_settings_task())


async def _refresh_settings_async() -> tuple[str, bool, int]:
    from starlette.concurrency import run_in_threadpool
    while True:
        scope, tools, pending, generation = await run_in_threadpool(_settings_once)
        if not pending:
            return scope, tools, generation
        await asyncio.sleep(0.01)


def _async_settings_task() -> asyncio.Task:
    loop = asyncio.get_running_loop()
    with _cache_lock:
        task_ref = _async_settings_tasks.get(loop)
        task = task_ref() if task_ref is not None else None
        if task is None or task.done():
            task = loop.create_task(_refresh_settings_async())
            _async_settings_tasks[loop] = weakref.ref(task)
            _async_settings_pending_tasks.add(task)
            task.add_done_callback(
                lambda completed, loop_ref = weakref.ref(loop): _release_async_settings_task(
                    completed, loop_ref
                )
            )
    return task


def _release_async_settings_task(task: asyncio.Task, loop_ref: weakref.ReferenceType) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    with _cache_lock:
        _async_settings_pending_tasks.discard(task)
        loop = loop_ref()
        task_ref = _async_settings_tasks.get(loop) if loop is not None else None
        if task_ref is not None and task_ref() is task:
            del _async_settings_tasks[loop]


def get_keyless_api_access_scope() -> str:
    return _settings()[0]


def get_keyless_api_access_settings() -> tuple[str, bool]:
    """Return the canonical scope and tool grant from one settings generation."""
    return _settings()


def get_keyless_api_tools_enabled() -> bool:
    """Whether a keyless caller may drive the server-side tool loop."""
    return _settings()[1]


def set_keyless_api_access(value: Any, *, tools: Any = None) -> tuple[str, bool]:
    """Persist which routes are served without a key, and whether tools come with them."""
    global _cached_settings, _settings_generation, _settings_write_inflight
    scope = _coerce_scope(value)
    if scope is None:
        raise ValueError(f"Keyless API access scope must be one of: {', '.join(KEYLESS_SCOPES)}.")
    with _write_lock:
        write_marker = object()
        upsert_started = False
        with _cache_lock:
            _settings_write_inflight = write_marker
        try:
            if scope == KEYLESS_SCOPE_OFF:
                allow_tools = False
            else:
                allow_tools = _read_settings_from_db()[1] if tools is None else _coerce_bool(tools)
            if allow_tools is None:
                raise ValueError("Keyless tool access must be true or false.")
            # tools are meaningless without a scope, and leaving them ticked would surprise
            # whoever turns keyless back on later
            allow_tools = allow_tools and scope != KEYLESS_SCOPE_OFF

            from storage.studio_db import upsert_app_settings

            upsert_started = True
            upsert_app_settings(
                {
                    KEYLESS_API_ACCESS_SETTING_KEY: scope,
                    KEYLESS_API_TOOLS_SETTING_KEY: allow_tools,
                },
                read_back = False,
            )
            with _cache_lock:
                _settings_generation += 1
                _cached_settings = (time.monotonic(), scope, allow_tools)
            return scope, allow_tools
        except Exception:
            # The write may have committed, so fail closed.
            if upsert_started:
                with _cache_lock:
                    _settings_generation += 1
                    _cached_settings = (time.monotonic(), KEYLESS_SCOPE_OFF, False)
            raise
        finally:
            with _cache_lock:
                if _settings_write_inflight is write_marker:
                    _settings_write_inflight = None


def access_exposure(app_state: Any) -> Optional[str]:
    """How far this server reaches beyond the machine, or None for localhost only.

    Advisory: it decides how bluntly the UI words the warning, never whether the
    setting may be used. An unknown bind host is reported as network-reachable.
    """
    from utils.host_policy import (
        is_external_host,
        lan_connector_active,
        tunnel_connector_active,
    )

    if bool(getattr(app_state, "remote_access_is_colab", False)):
        return "colab"
    if bool(getattr(app_state, "secure", False)):
        return "public_url"
    if getattr(app_state, "cloudflare_url", None) or tunnel_connector_active():
        return "public_url"
    if lan_connector_active():
        from lan_access import lan_listener_status
        from utils.lan_access_settings import _normalized_ip, _private_non_loopback

        try:
            addresses = tuple(_normalized_ip(value) for value in lan_listener_status()["addresses"])
        except Exception:
            return "network"
        return (
            "private_lan"
            if addresses
            and all(address is not None and _private_non_loopback(address) for address in addresses)
            else "network"
        )
    if bool(getattr(app_state, "lan_access_launch_managed", False)):
        from utils.lan_access_settings import _normalized_ip, _private_non_loopback
        addresses = tuple(
            _normalized_ip(value)
            for value in (getattr(app_state, "lan_access_launch_addresses", ()) or ())
        )
        if addresses and all(
            address is not None and _private_non_loopback(address) for address in addresses
        ):
            return "private_lan"
    bind_host = getattr(app_state, "bind_host", None)
    if not isinstance(bind_host, str) or is_external_host(bind_host):
        return "network"
    return None


def normalize_request_path(path: str, root_path: str = "") -> str:
    """Normalize trailing slash and an ASGI/FastAPI mount root."""
    if not isinstance(path, str) or not path.startswith("/"):
        return ""
    root = root_path.rstrip("/") if isinstance(root_path, str) else ""
    if root and root.startswith("/"):
        if path == root:
            path = "/"
        elif path.startswith(f"{root}/"):
            path = path[len(root) :]
    return path.rstrip("/") or "/"


def scope_covers(
    scope: str,
    method: str,
    path: str,
    root_path: str = "",
) -> bool:
    """Whether ``scope`` includes this exact method and normalized route."""
    normalized = normalize_request_path(path, root_path)
    normalized_method = method.upper() if isinstance(method, str) else ""
    if scope == KEYLESS_SCOPE_FULL:
        return bool(normalized and normalized_method)
    if scope != KEYLESS_SCOPE_INFERENCE:
        return False
    if (normalized_method, normalized) in _INFERENCE_ROUTES:
        return True
    # The router intentionally exposes one dynamic retrieval template. Its method
    # is still explicit; an empty id and every non-GET alias remain denied.
    return normalized_method == "GET" and normalized.startswith("/v1/models/")


def _request_app_state(request: Any):
    try:
        return request.app.state
    except Exception:
        return None


def _hosted_mode_forbidden(app_state: Any) -> bool:
    """Whether the whole launch mode forbids keyless admission."""
    if app_state is None:
        return True
    if bool(getattr(app_state, "remote_access_is_colab", False)) or bool(
        getattr(app_state, "lan_access_is_colab", False)
    ):
        return True
    if bool(getattr(app_state, "secure", False)) or bool(
        getattr(app_state, "lan_access_secure_launch", False)
    ):
        return True
    return False


def _public_tunnel_active(app_state: Any) -> bool:
    """Whether loopback transport may actually be carrying a public tunnel request."""
    if getattr(app_state, "cloudflare_url", None):
        return True
    try:
        from utils.host_policy import tunnel_connector_active
        return tunnel_connector_active()
    except Exception:
        return True


def _full_scope_transport_allowed(request: Any, app_state: Any) -> bool:
    from utils.lan_access_settings import _all_addresses_are, request_is_loopback

    if not request_is_loopback(request):
        return False
    bind_host = getattr(app_state, "bind_host", None)
    scope = getattr(request, "scope", {})
    server = scope.get("server")
    if not isinstance(bind_host, str) or bind_host in ("0.0.0.0", "::"):
        return False
    if not isinstance(server, (tuple, list)) or len(server) < 2:
        return False
    port = server[1]
    return isinstance(port, int) and _all_addresses_are(
        bind_host, port, lambda address: address.is_loopback
    )


def _repeated_header(request: Any, name: bytes) -> bool:
    """Whether the raw ASGI headers carry ``name`` more than once.

    ``Headers.get()`` returns the first of a repeated header, so a predicate built on it may
    decide on a different value than an intermediary acted on. An ambiguous request is
    refused rather than resolved, as `asgi_request_is_keyless` already does for a repeated
    `Authorization`. h11 rejects a repeated `Host`, httptools does not, and neither rejects a
    repeated `Sec-Fetch-Site`, so this cannot be left to the parser.
    """
    try:
        headers = request.scope.get("headers") or ()
        return sum(1 for key, _ in headers if key.lower() == name) > 1
    except Exception:
        return True


def _browser_initiated_elsewhere(request: Any) -> bool:
    """Whether a page on another site made this request, as the browser reports it.

    ``Origin`` cannot say: no browser attaches it to a same-origin GET or to a cross-site
    ``no-cors`` GET, and such a fetch at ``http://127.0.0.1:<port>`` does arrive. Only
    Chromium's Local Network Access (141, enforced from 142, replacing Private Network
    Access) holds it back; Firefox and Safari ship no equivalent. ``Sec-Fetch-Site`` is set
    on every request to a URL the browser considers *potentially trustworthy*, and the
    ``Sec-`` prefix makes it unforgeable. Absence stays admitted: curl, the OpenAI SDKs and
    Safari before 16.4 send nothing, and serving them is the point of the setting.

    Two limits, because the header is weaker than it first appears:

    * Absence only *means* "not a browser" where the URL is potentially trustworthy, which
      is what `_host_authority_is_direct` enforces. On the plain-HTTP private-LAN limb no
      such URL exists, so this predicate is inert there and `Origin` is the only signal left.
    * ``none`` is refused. It is computed before the redirect chain is walked, so an
      attacker-controlled 302 from a user-initiated navigation still arrives saying ``none``
      (measured: Firefox 153, WebKit 26.5). Nobody types an API route into an address bar.
    """
    if _repeated_header(request, b"sec-fetch-site"):
        return True
    try:
        site = request.headers.get("sec-fetch-site")
    except Exception:
        return True  # unreadable headers: deny
    if site is None:
        return False
    return site.strip().lower() != "same-origin"


def _port_suffix_is_numeric(suffix: str) -> bool:
    """Whether ``suffix`` is a well formed ``:<port>`` tail, the only tail an authority has."""
    return len(suffix) > 1 and suffix[0] == ":" and suffix[1:].isdigit()


def _host_authority_is_direct(request: Any, scope: str) -> bool:
    """Whether the caller addressed this server directly rather than through a name.

    Guards DNS rebinding, which the socket checks cannot see: a page on ``evil.example``
    re-pointed at ``127.0.0.1`` keeps its own origin, so every signal above reads as a local
    client and the response is readable by the page. ``Host`` still names the site the page
    was served from. A direct client sends the literal address or ``localhost``; anything
    else is a name, whether rebound or a legitimate mDNS / internal-DNS / reverse-proxy
    alias -- keyless declines both, as `lan_access_settings` also never trusts a name.
    Absent stays admitted: HTTP/1.0 callers send none and no browser omits it.

    The literal is matched as written rather than canonicalised, because this predicate and
    the browser must agree on how "loopback" is spelled. Two families are refused for that
    reason, both measured reaching a ``127.0.0.1`` listener while the browser sent no
    ``Sec-Fetch-*`` at all, neither being potentially trustworthy (``127.0.0.0/8``, ``::1/128``):

    * IPv4-mapped IPv6 -- ``[::ffff:127.0.0.1]``, ``[::ffff:7f00:1]`` -- on Chromium 151,
      Firefox 153 and WebKit 26.5.
    * the unspecified ``0.0.0.0`` and ``[::]``, which connect to loopback on Linux.

    Canonicalising them, as a general purpose normaliser would, is what turned
    absence-means-not-a-browser into a bypass, so parsing happens here rather than through
    `lan_access_settings._normalized_ip`, whose leniency suits the socket addresses it was
    written for and not an authority off the wire.

    The literal must also be one ``scope`` could legitimately be reached at. The socket
    checks see only the hop that connected, so an SSH forward or a reverse proxy in front of
    a loopback bind makes both ASGI endpoints loopback while ``Host`` is the public address
    the page came from. ``full`` is loopback-only by construction
    (`_full_scope_transport_allowed` demands a loopback bind and peer), so its authority must
    be loopback too; ``inference`` may also be reached across the private LAN. This cannot be
    spelled with `is_private`, which means "not globally reachable" and counts the
    documentation ranges in as well.
    """
    import ipaddress

    from utils.lan_access_settings import _private_non_loopback

    if _repeated_header(request, b"host"):
        return False
    try:
        host = request.headers.get("host")
    except Exception:
        return False  # unreadable headers: deny
    if not host:
        return True
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return False
        suffix = host[end + 1 :]
        if suffix and not _port_suffix_is_numeric(suffix):
            return False
        try:
            address = ipaddress.IPv6Address(host[1:end])
        except ValueError:
            return False
    else:
        literal, separator, suffix = host.partition(":")
        if separator and not _port_suffix_is_numeric(":" + suffix):
            return False
        literal = literal.lower()
        # Exactly `localhost`, no trailing root-label dot. Secure Contexts lists
        # `localhost.` as trustworthy, but WebKit's check is a plain string compare: measured
        # on WebKit 26.5 a page dialling `http://localhost.:<port>` sends no `Sec-Fetch-*`
        # while Chromium 151 and Firefox 153 send `cross-site`, so admitting the dotted form
        # would reopen the absence gap on Safari alone. No client spells it.
        if literal == "localhost":
            return True
        try:
            # Unbracketed IPv6 is not a legal authority, so IPv4 only here.
            address = ipaddress.IPv4Address(literal)
        except ValueError:
            return False
    return keyless_authority_address_allowed(address, scope)


def keyless_authority_address_allowed(address: Any, scope: str) -> bool:
    """Whether a parsed authority literal is one ``scope`` could be reached at.

    The single place this is answered, so admission and anything that advertises an address
    to the user cannot drift apart -- `lan_access_settings` reported a keyless-eligible LAN
    URL for an IPv4-mapped literal admission refuses, having kept its own copy of the test.

    Takes an already-parsed address, which must NOT have been canonicalised: the mapped form
    is refused precisely because the browser does not treat it as loopback, so un-mapping it
    erases the distinction being tested.
    """
    from utils.lan_access_settings import _private_non_loopback

    if address is None:
        return False
    if address.is_unspecified:
        return False
    if getattr(address, "ipv4_mapped", None) is not None:
        return False
    if address.is_loopback:
        return True
    return scope == KEYLESS_SCOPE_INFERENCE and _private_non_loopback(address)


def keyless_transport_allowed(request: Any, scope: str) -> bool:
    """Enforce the loopback/private-LAN boundary from authoritative ASGI state."""
    try:
        if request.headers.get("origin") is not None:
            return False
    except Exception:
        return False
    if _browser_initiated_elsewhere(request):
        return False
    if not _host_authority_is_direct(request, scope):
        return False
    app_state = _request_app_state(request)
    if _hosted_mode_forbidden(app_state):
        return False
    if scope == KEYLESS_SCOPE_FULL:
        if _public_tunnel_active(app_state):
            return False
        return _full_scope_transport_allowed(request, app_state)
    if scope != KEYLESS_SCOPE_INFERENCE:
        return False
    from utils.lan_access_settings import request_is_loopback, request_on_lan_access

    if request_on_lan_access(request):
        return True
    if _public_tunnel_active(app_state):
        return False
    return request_is_loopback(request)


def keyless_request_allowed(request: Any) -> bool:
    """Whether the route and transport are eligible for keyless authentication."""
    return _keyless_request_allowed_for_scope(request, get_keyless_api_access_scope())


def _keyless_request_allowed_for_scope(request: Any, scope: str) -> bool:
    if scope == KEYLESS_SCOPE_OFF:
        return False
    asgi_scope = getattr(request, "scope", {})
    method = asgi_scope.get("method", "")
    path = asgi_scope.get("path", "")
    root_path = asgi_scope.get("root_path", "")
    if not scope_covers(scope, method, path, root_path):
        return False
    return keyless_transport_allowed(request, scope)


def mark_keyless_admission(request: Any, admitted: bool) -> None:
    """Publish the authoritative admission result for downstream policy decisions."""
    try:
        setattr(request.state, KEYLESS_ADMISSION_STATE_KEY, bool(admitted))
    except Exception:
        pass


def request_was_admitted_keyless(request: Any) -> Optional[bool]:
    """Return a recorded admission decision, or None before auth has classified it."""
    try:
        value = getattr(request.state, KEYLESS_ADMISSION_STATE_KEY)
    except Exception:
        return None
    return value if isinstance(value, bool) else None


class KeylessToolPolicyMiddleware:
    """Hard-disable server-side tools for a keyless caller that was not granted them.

    ``/v1/chat/completions`` runs python and terminal on this machine through the
    tool loop, and ``unsloth studio run`` turns tools on by default, so serving that
    route without a key would otherwise hand the loop to anyone who can reach it.
    Mirrors what routes/preview.py does for the public ``/p`` surface.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, asgi_scope, receive, send):
        if asgi_scope.get("type") != "http":
            await self.app(asgi_scope, receive, send)
            return
        from starlette.concurrency import run_in_threadpool

        scope, tools, generation = await _settings_async()
        settings = (scope, tools)
        admitted = await run_in_threadpool(asgi_request_is_keyless, asgi_scope, settings)
        with _cache_lock:
            if _settings_write_inflight is not None or generation != _settings_generation:
                settings = (KEYLESS_SCOPE_OFF, False)
                admitted = False
            # Publish under the lock to linearize admission with writes.
            asgi_scope.setdefault("state", {})[KEYLESS_ADMISSION_STATE_KEY] = admitted
        if not admitted:
            await self.app(asgi_scope, receive, send)
            return
        if settings[1]:
            await self.app(asgi_scope, receive, send)
            return
        from state.tool_policy import tools_force_disabled

        with tools_force_disabled():
            await self.app(asgi_scope, receive, send)


def asgi_request_is_keyless(asgi_scope, settings: Optional[tuple[str, bool]] = None) -> bool:
    """Whether this ASGI request is admitted by the setting rather than by a credential.

    Middleware-side twin of ``auth.authentication.admitted_without_credential``, reading
    the raw scope because it runs before the request object exists. An Unsloth session and
    a working API key both authenticate as themselves, so neither is keyless: applying
    the tool restriction to an existing API client would take away tools it already had.
    """
    try:
        from starlette.requests import Request
        request = Request(asgi_scope)
    except Exception:
        return False
    allowed = (
        keyless_request_allowed(request)
        if settings is None
        else _keyless_request_allowed_for_scope(request, settings[0])
    )
    if not allowed:
        return False
    authorization = [
        bytes(value).decode("latin-1")
        for name, value in asgi_scope.get("headers") or ()
        if bytes(name).lower() == b"authorization"
    ]
    if not authorization:
        return True
    if len(authorization) != 1:
        return False
    # The same parser the dependency uses, not a second hand-rolled split. They disagreed on
    # `Authorization: bearer  not-needed`: `partition(" ")` left the token as " not-needed"
    # and reported not-keyless, while the dependency collapsed the space and admitted the
    # dummy. That shape was therefore keyless to every route but not-keyless to
    # `KeylessToolPolicyMiddleware`, which decides whether to clamp the tool grant, so it
    # reached keyless admission with an unclamped tool policy.
    from fastapi.security.utils import get_authorization_scheme_param

    scheme, token = get_authorization_scheme_param(authorization[0])
    return bool(scheme.lower() == "bearer" and token in APPROVED_DUMMY_BEARERS)
