# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automation blocks: triggers, HTTP requests and Python. The engine refuses the last two (and a trigger firing on its
own) unless the owner started the run and approved them for the flow; see ``core/flows/capabilities.py``."""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from core.flows.registry import FlowNodeError, NodeContext, handler

TRIGGER_PARAM = "__trigger"
HTTP_MAX_BYTES = 2 * 1024 * 1024
PY_RESULT_MARKER = "<<FLOW_RESULT>>"
PY_MAX_INPUT_BYTES = 4 * 1024 * 1024
_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec = "seconds")


def _fired(ctx: NodeContext) -> Optional[dict]:
    fired = (ctx.flow_params or {}).get(TRIGGER_PARAM)
    if isinstance(fired, dict) and isinstance(fired.get(ctx.node_id), dict):
        return fired[ctx.node_id]
    return None


@handler("trigger.schedule")
async def schedule(ctx: NodeContext) -> dict:
    fired = _fired(ctx) or {}
    return {"fired_at": str(fired.get("fired_at") or _now()), "payload": fired.get("payload")}


@handler("trigger.webhook")
async def webhook(ctx: NodeContext) -> dict:
    fired = _fired(ctx)
    if fired is None:
        return {"payload": ctx.params.get("test_payload"), "received_at": _now()}
    return {"payload": fired.get("payload"), "received_at": str(fired.get("received_at") or _now())}


class _NoRedirect:
    """Built lazily: a redirect could point at a private host the guard never saw, so it is returned, not followed."""

    @staticmethod
    def handler():
        import urllib.request
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        return NoRedirect()


def _raw_socket(response: Any) -> Any:
    """The socket under an http.client response (or the HTTPError wrapping one), or None."""
    fp = response
    for _ in range(4):
        inner = getattr(fp, "fp", None)
        sock = getattr(getattr(inner, "raw", None), "_sock", None)
        if isinstance(sock, socket.socket):
            return sock
        fp = inner
        if fp is None:
            return None
    return None


def guarded_request(
    method: str,
    url: str,
    headers: dict,
    body: Optional[bytes],
    timeout: float,
    stop: Optional[threading.Event] = None,
) -> tuple[int, str, bytes]:
    """One request to a public host: resolved once, refused for non-public addresses, then pinned to the checked
    addresses (no proxy, no redirects), as the chat web fetch does."""
    import urllib.error
    import urllib.request

    from core.inference import tools as web

    deadline = time.monotonic() + timeout
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FlowNodeError("Give a full http:// or https:// URL")
    if parts.username or parts.password:
        raise FlowNodeError("Put credentials in a header, not in the URL")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    ok, reason, addresses = web._validate_and_resolve_host(parts.hostname, port)
    if not ok:
        raise FlowNodeError(reason)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect.handler(),
        web._PinnedHTTPHandler(addresses, origin = parts.hostname),
        web._SNIHTTPSHandler(parts.hostname, addresses, origin = parts.hostname),
    )
    request = urllib.request.Request(url, data = body, method = method, headers = headers)
    try:
        response = opener.open(request, timeout = timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    except (urllib.error.URLError, OSError) as exc:
        raise FlowNodeError(f"The request failed: {getattr(exc, 'reason', exc)}") from None
    # read(n) waits for all n bytes, so a peer trickling bytes outlasts the deadline and Stop. read1 returns per
    # receive, and the watcher shuts the socket down so a blocked receive returns too.
    read = getattr(response, "read1", None) or response.read
    finished = threading.Event()
    sock = _raw_socket(response)

    def _overdue() -> bool:
        return time.monotonic() > deadline or (stop is not None and stop.is_set())

    def _watch() -> None:
        while not finished.wait(0.05):
            if _overdue():
                try:
                    socket.socket.shutdown(sock, socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    if sock is not None:
        threading.Thread(target = _watch, name = "flows-http-watch", daemon = True).start()
    with response:
        try:
            chunks, size = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise FlowNodeError(f"The response took longer than {int(timeout)} seconds")
                if stop is not None and stop.is_set():
                    raise FlowNodeError("Stopped")
                try:
                    chunk = read(64 * 1024)
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    if _overdue():
                        continue
                    raise FlowNodeError(f"The request failed: {exc}") from None
                if not chunk:
                    if _overdue():
                        continue
                    break
                size += len(chunk)
                if size > HTTP_MAX_BYTES:
                    raise FlowNodeError(
                        f"The response is over {HTTP_MAX_BYTES // (1024 * 1024)} MB"
                    )
                chunks.append(chunk)
            return (
                int(response.status),
                response.headers.get("content-type") or "",
                b"".join(chunks),
            )
        finally:
            finished.set()


@handler("http.request")
async def http_request(ctx: NodeContext) -> dict:
    method = str(ctx.params.get("method") or "GET").upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        raise FlowNodeError(f"Unsupported method {method}")
    url = str(ctx.params.get("url") or "").strip()
    host = urlsplit(url).hostname
    refused = ctx.capability_refusal("http", (host or "").lower().rstrip("."))
    if refused:
        raise FlowNodeError(refused)
    headers = (
        {
            str(k): str(v)
            for k, v in (ctx.params.get("headers") or {}).items()
            if str(k).lower() not in _HOP_HEADERS
        }
        if isinstance(ctx.params.get("headers"), dict)
        else {}
    )
    payload = (
        ctx.inputs.get("body") if ctx.inputs.get("body") is not None else ctx.params.get("body")
    )
    body = None
    if payload is not None and method != "GET":
        body = json.dumps(payload).encode()
        headers.setdefault("Content-Type", "application/json")
    timeout = max(1, min(int(ctx.params.get("timeout_s") or 30), 120))
    stop = threading.Event()

    async def cancel() -> None:
        stop.set()

    ctx.on_cancel(cancel)
    status, content_type, data = await asyncio.to_thread(
        guarded_request, method, url, headers, body, timeout, stop
    )
    text = data.decode("utf-8", "replace")
    parsed: Any = None
    if "json" in content_type.lower() or text.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(text, parse_constant = _no_constant)
        except ValueError:
            parsed = None
    return {"data": parsed, "text": text, "status": status}


def python_session(flow_id: str) -> str:
    """The sandbox a flow's Python blocks share; it goes when the flow is deleted."""
    return f"flow-{flow_id}"


def _program(code: str, value: Any) -> str:
    """The user's code with `inputs` set. What it prints is held back, and `result` is printed as JSON on a marker line
    FIRST, then the printed log: the tool cuts long output, and a long log must not cut the result."""
    encoded = json.dumps(value, default = str)
    if len(encoded) > PY_MAX_INPUT_BYTES:
        raise FlowNodeError("The input is too large to pass to Python")
    return (
        "import contextlib as _flow_cl, io as _flow_io, json as _flow_json, sys as _flow_sys\n"
        f"inputs = _flow_json.loads({encoded!r})\n"
        "result = None\n"
        "_flow_log = _flow_io.StringIO()\n"
        "try:\n"
        "    with _flow_cl.redirect_stdout(_flow_log):\n"
        f"        exec(compile({code!r}, '<python block>', 'exec'), globals())\n"
        "except BaseException:\n"
        "    _flow_sys.stdout.write(_flow_log.getvalue()[-4000:])\n"
        "    raise\n"
        f"_flow_sys.stdout.write({PY_RESULT_MARKER!r} + _flow_json.dumps(result, default = str) + '\\n')\n"
        "_flow_sys.stdout.write(_flow_log.getvalue())\n"
    )


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def parse_python_output(output: str) -> tuple[Any, str]:
    head, marker, tail = output.partition(PY_RESULT_MARKER)
    if not marker:
        raise FlowNodeError("The Python code did not finish: " + output.strip()[-600:])
    line, _, log = tail.partition("\n")
    try:
        return json.loads(line or "null", parse_constant = _no_constant), (head + log).strip()
    except ValueError:
        raise FlowNodeError(
            "Python's `result` could not be turned into JSON (NaN and Infinity are not JSON), or is too long to pass on"
        ) from None


@handler("code.python")
async def python(ctx: NodeContext) -> dict:
    from core.inference.tools import execute_tool

    code = str(ctx.params.get("code") or "")
    if not code.strip():
        raise FlowNodeError("Write some Python first")
    program = _program(code, ctx.inputs.get("input"))
    timeout = max(1, min(int(ctx.params.get("timeout_s") or 60), 600))
    stop = threading.Event()

    async def cancel() -> None:
        stop.set()

    ctx.on_cancel(cancel)
    output = await asyncio.to_thread(
        execute_tool,
        "python",
        {"code": program},
        cancel_event = stop,
        timeout = timeout,
        session_id = python_session(ctx.flow_id),
    )
    result, log = parse_python_output(str(output))
    return {"result": result, "log": log[-20000:]}
