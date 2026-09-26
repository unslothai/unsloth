# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve models from linked Unsloth Studio instances through this server's /v1 API.

A linked model is addressed as ``@<instance>/<remote model id>``. HF org names cannot start
with ``@``, so the prefix never shadows a local id. Requests are forwarded verbatim with the
model id unwrapped, and the response (JSON or SSE) is passed back untouched.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import time
from typing import Optional
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from core.inference.api_monitor import api_monitor
from storage import linked_instances_db

MODEL_PREFIX = "@"
# Set on forwarded requests so a remote never forwards again (A links B links A).
HOP_HEADER = "X-Unsloth-Linked-Hop"
_FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")
_CATALOG_TTL_S = 10.0
_PROBE_TIMEOUT = httpx.Timeout(8.0, connect = 5.0)

_catalog_cache: dict[str, tuple[float, list[dict]]] = {}
_http_client: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        from core.inference.external_provider import _create_shared_http_client

        _http_client = _create_shared_http_client()
    return _http_client


def _is_private_host(host: str) -> bool:
    if host.lower() == "localhost" or host.lower().endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    addresses = {ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos}
    return bool(addresses) and all(a.is_private or a.is_loopback for a in addresses)


def normalize_base_url(base_url: str) -> str:
    """Validated origin without a trailing ``/`` or ``/v1``, so either form can be pasted.

    Plain http is accepted only for loopback and private LAN hosts: the remote's key rides on
    every request, so a public URL must be https (a Cloudflare tunnel already is).
    """
    from core.inference.providers import validate_provider_base_url

    url = validate_provider_base_url(base_url)
    if url.lower().endswith("/v1"):
        url = url[:-3].rstrip("/")
    parts = urlsplit(url)
    if parts.scheme == "http" and not _is_private_host(parts.hostname or ""):
        raise ValueError(
            "Use https for a public URL. Plain http is only allowed on this machine or your LAN."
        )
    return url


def split_model(model: object) -> Optional[tuple[str, str]]:
    if not isinstance(model, str) or not model.startswith(MODEL_PREFIX):
        return None
    name, _, remote = model[len(MODEL_PREFIX):].partition("/")
    if not name or not remote:
        return None
    return name.lower(), remote


def _auth_headers(instance: dict) -> dict[str, str]:
    key = linked_instances_db.get_api_key(instance["id"])
    headers = {HOP_HEADER: "1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _may_use_linked(request: Request) -> bool:
    from auth.authentication import request_admitted_without_credential
    from hub.services.models import account_access

    if request.headers.get(HOP_HEADER):
        return False
    if account_access.managed_account():
        return False
    return not request_admitted_without_credential(request)


async def resolve(request: Request, model: object) -> Optional[tuple[dict, str]]:
    """``(instance, remote model id)`` when ``model`` names a linked instance, else ``None``."""
    parts = split_model(model)
    if parts is None:
        return None
    name, remote_model = parts
    if not _may_use_linked(request):
        raise HTTPException(
            status_code = 403,
            detail = "Linked instances need the owner's API key as Authorization: Bearer, or a UI session. Keyless callers can't use them.",
        )
    instance = await asyncio.to_thread(linked_instances_db.get_instance_by_name, name)
    if instance is None:
        raise HTTPException(status_code = 404, detail = f"No linked instance named '{name}'.")
    return instance, remote_model


def _prompt_preview(body: dict) -> str:
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
    for key in ("prompt", "input"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def _record_usage(entry_id: str, usage: object) -> None:
    if isinstance(usage, dict):
        api_monitor.set_usage(
            entry_id,
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens")),
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens")),
        )


def _record_json(entry_id: str, payload: object) -> None:
    """Reply text and usage from an OpenAI or Anthropic response body."""
    if not isinstance(payload, dict):
        return
    text = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        text = (choices[0].get("message") or {}).get("content") or choices[0].get("text") or ""
    elif isinstance(payload.get("content"), list):
        text = "".join(b.get("text", "") for b in payload["content"] if isinstance(b, dict))
    if isinstance(text, str):
        api_monitor.append_reply(entry_id, text, stamp_first_token = False)
    _record_usage(entry_id, payload.get("usage"))


def _record_sse_line(entry_id: str, line: str) -> Optional[dict]:
    """Record one SSE line; returns the parsed event, if any."""
    if not line.startswith("data:"):
        return None
    try:
        event = json.loads(line[5:].strip())
    except ValueError:
        return None
    if not isinstance(event, dict):
        return None
    text = ""
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        text = (choices[0].get("delta") or {}).get("content") or choices[0].get("text") or ""
    elif event.get("type") == "content_block_delta":
        text = (event.get("delta") or {}).get("text") or ""
    if isinstance(text, str):
        api_monitor.append_reply(entry_id, text)
    _record_usage(entry_id, event.get("usage") or (event.get("message") or {}).get("usage"))
    return event


def _is_usage_only(event: Optional[dict]) -> bool:
    return bool(event) and event.get("choices") == [] and isinstance(event.get("usage"), dict)


async def forward(
    request: Request,
    path: str,
    target: tuple[dict, str],
    *,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> Response:
    """POST the caller's JSON body to ``<instance>/v1/<path>`` with the model id unwrapped.

    Logged in the API monitor like local traffic, under the ``@instance/model`` id.
    """
    instance, remote_model = target
    body = await request.json()
    entry_id = api_monitor.start(
        endpoint = request.url.path,
        method = "POST",
        model = body.get("model") or "",
        prompt = _prompt_preview(body),
        subject = subject,
        via_api_key = via_api_key,
    )
    body["model"] = remote_model
    stream = bool(body.get("stream"))
    # OpenAI streams carry usage only on request. Ask for it so the monitor has token
    # counts, and drop that extra chunk again unless the caller asked too.
    strip_usage = False
    if stream and path in ("chat/completions", "completions"):
        options = body.get("stream_options")
        options = options if isinstance(options, dict) else {}
        if not options.get("include_usage"):
            body["stream_options"] = {**options, "include_usage": True}
            strip_usage = True
    headers = await asyncio.to_thread(_auth_headers, instance)
    for name in _FORWARDED_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value

    client = _client()
    upstream_request = client.build_request(
        "POST",
        f"{instance['base_url']}/v1/{path}",
        json = body,
        headers = headers,
        # A generation can sit silent for minutes before its first token.
        timeout = httpx.Timeout(None if stream else 900.0, connect = 10.0),
    )
    try:
        upstream = await client.send(upstream_request, stream = True)
    except httpx.HTTPError as exc:
        message = f"Linked instance '{instance['name']}' is unreachable ({type(exc).__name__})."
        api_monitor.fail(entry_id, message)
        raise HTTPException(status_code = 502, detail = message) from exc

    media_type = upstream.headers.get("content-type", "application/json")
    if not stream or upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        try:
            payload = json.loads(content)
        except ValueError:
            payload = None
        if upstream.status_code >= 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            api_monitor.fail(
                entry_id, message or f"HTTP {upstream.status_code} from '{instance['name']}'"
            )
        else:
            _record_json(entry_id, payload)
            api_monitor.finish(entry_id)
        return Response(content, status_code = upstream.status_code, media_type = media_type)

    async def relay():
        # Split on bytes: "\n" never occurs inside a multi-byte UTF-8 character.
        pending = b""
        try:
            async for chunk in upstream.aiter_bytes():
                *lines, pending = (pending + chunk).split(b"\n")
                out = []
                for line in lines:
                    event = _record_sse_line(entry_id, line.decode("utf-8", errors = "replace").strip())
                    if not (strip_usage and _is_usage_only(event)):
                        out.append(line + b"\n")
                if out:
                    yield b"".join(out)
            if pending:
                _record_sse_line(entry_id, pending.decode("utf-8", errors = "replace").strip())
                yield pending
            api_monitor.finish(entry_id)
        except asyncio.CancelledError:
            api_monitor.finish(entry_id, "cancelled")
            raise
        except Exception as exc:
            api_monitor.fail_open(
                entry_id, f"{type(exc).__name__} while streaming from '{instance['name']}'"
            )
            raise
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code = upstream.status_code,
        media_type = media_type,
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def fetch_models(instance: dict) -> list[dict]:
    headers = await asyncio.to_thread(_auth_headers, instance)
    response = await _client().get(
        f"{instance['base_url']}/v1/models", headers = headers, timeout = _PROBE_TIMEOUT
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list):
        raise ValueError("Unexpected /v1/models response")
    return [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


async def probe(instance: dict) -> dict:
    started = time.monotonic()
    try:
        models = await fetch_models(instance)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        error = "The API key was rejected." if code in (401, 403) else f"The instance answered HTTP {code}."
        return {"online": False, "error": error, "models": [], "latency_ms": None}
    except (httpx.HTTPError, ValueError):
        return {"online": False, "error": "Not reachable.", "models": [], "latency_ms": None}
    return {
        "online": True,
        "error": None,
        # A remote's own linked models are left out: no chains, no loops.
        "models": [m for m in models if not m["id"].startswith(MODEL_PREFIX)],
        "latency_ms": round((time.monotonic() - started) * 1000),
    }


async def _instance_catalog(instance: dict) -> list[dict]:
    cached = _catalog_cache.get(instance["id"])
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL_S:
        return cached[1]
    result = await probe(instance)
    objects = [
        {
            **model,
            "id": f"{MODEL_PREFIX}{instance['name']}/{model['id']}",
            "owned_by": instance["name"],
            "linked_instance": instance["name"],
        }
        for model in result["models"]
    ]
    _catalog_cache[instance["id"]] = (time.monotonic(), objects)
    return objects


async def catalog_objects(request: Optional[Request]) -> list[dict]:
    """Every linked instance's models for /v1/models; an offline instance contributes none."""
    if request is None or not _may_use_linked(request):
        return []
    instances = await asyncio.to_thread(linked_instances_db.list_instances)
    results = await asyncio.gather(*(_instance_catalog(i) for i in instances))
    return [model for models in results for model in models]


def forget(instance_id: str) -> None:
    _catalog_cache.pop(instance_id, None)


# Read-only endpoints every Unsloth Studio serves to an API key, old releases included.
_INFO_PATHS = {
    "system": "/api/system",
    "hardware": "/api/system/hardware?include_details=true",
    "install": "/api/studio/install-source",
}


def _text(value: object, limit: int = 200) -> Optional[str]:
    return value[:limit] if isinstance(value, str) and value else None


def _number(value: object) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


async def _get_json(instance: dict, headers: dict, path: str) -> dict:
    response = await _client().get(f"{instance['base_url']}{path}", headers = headers, timeout = _PROBE_TIMEOUT)
    response.raise_for_status()
    return _dict(response.json())


def _gpus(system: dict, hardware: dict) -> list[dict]:
    devices = _dict(system.get("gpu")).get("devices")
    if isinstance(devices, list) and devices:
        return [
            {
                "name": _text(d.get("name")) or "GPU",
                "vram_total_gb": _number(d.get("memory_total_gb")),
                "vram_used_gb": _number(d.get("vram_used_gb")),
                "utilization_pct": _number(d.get("vram_utilization_pct")),
            }
            for d in devices[:16]
            if isinstance(d, dict)
        ]
    listed = hardware.get("gpus")
    if isinstance(listed, list):
        return [
            {"name": _text(g.get("name")) or "GPU", "vram_total_gb": _number(g.get("vram_total_gb"))}
            for g in listed[:16]
            if isinstance(g, dict)
        ]
    return []


async def fetch_info(instance: dict) -> dict:
    """Version, runtime and hardware of a linked instance. Each source is optional."""
    headers = await asyncio.to_thread(_auth_headers, instance)
    keys = list(_INFO_PATHS)
    results = await asyncio.gather(
        *(_get_json(instance, headers, _INFO_PATHS[k]) for k in keys), return_exceptions = True
    )
    parts = {k: r for k, r in zip(keys, results) if isinstance(r, dict)}
    if not parts:
        first = results[0]
        if isinstance(first, httpx.HTTPStatusError) and first.response.status_code in (401, 403):
            error = "The API key was rejected."
        else:
            error = "Not reachable."
        return {"online": False, "error": error}
    system, hardware, install = parts.get("system", {}), parts.get("hardware", {}), parts.get("install", {})
    versions = _dict(hardware.get("versions"))
    packages = _dict(system.get("ml_packages"))
    memory, disk = _dict(system.get("memory")), _dict(system.get("disk"))
    return {
        "online": True,
        "error": None,
        "version": _text(install.get("current_version")) or _text(versions.get("unsloth")),
        "install_source": _text(install.get("install_source")),
        "update_available": install.get("update_available") is True,
        "latest_version": _text(install.get("latest_version")),
        "platform": _text(system.get("platform")),
        "python_version": _text(system.get("python_version")),
        "device_backend": _text(system.get("device_backend")),
        "torch": _text(versions.get("torch")) or _text(packages.get("torch")),
        "transformers": _text(versions.get("transformers")) or _text(packages.get("transformers")),
        "cuda": _text(versions.get("cuda")),
        "rocm": _text(versions.get("rocm")),
        "llama_cpp": _text(hardware.get("llama_cpp")),
        "gpus": _gpus(system, hardware),
        "cpu_count": _number(system.get("cpu_count")),
        "memory_total_gb": _number(memory.get("total_gb")),
        "memory_available_gb": _number(memory.get("available_gb")),
        "disk_total_gb": _number(disk.get("total_gb")),
        "disk_free_gb": _number(disk.get("free_gb")),
        "uptime_seconds": _number(system.get("uptime_seconds")),
    }
