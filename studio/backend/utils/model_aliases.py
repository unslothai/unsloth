# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-defined, install-wide model aliases (``fast-chat`` -> a real model id) for API clients.

Resolved once at the ASGI edge, so access checks, auto-switch and the resolver all see the real
id: an alias never grants what its target would not. Names exclude ``/`` and ``:`` so they can
never shadow a repo id or a ``model:quant`` pin."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Optional

MODEL_ALIASES_SETTING_KEY = "openai_api_model_aliases"
MAX_MODEL_ALIASES = 100
MAX_ALIAS_TARGET_LENGTH = 512
_ALIAS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ALIAS_PATH_PREFIXES = ("/v1/", "/api/inference/")
_MODEL_VALUE_RE = re.compile(rb'"model"\s*:\s*"([^"\\]*)"')


def normalize_model_aliases(raw: Any) -> list[dict[str, str]]:
    """Validated ``[{"name", "model"}]``; raises ValueError naming the first bad entry."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Model aliases must be a list.")
    if len(raw) > MAX_MODEL_ALIASES:
        raise ValueError(f"At most {MAX_MODEL_ALIASES} model aliases are allowed.")
    aliases: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each model alias needs a name and a model.")
        name = str(entry.get("name") or "").strip()
        model = str(entry.get("model") or "").strip()
        if name.lower() == "default":
            # Clients and the Decision API send "default" for "whatever is loaded".
            raise ValueError("'default' is reserved and cannot be an alias name.")
        if not _ALIAS_NAME_RE.match(name):
            raise ValueError(
                f"Alias name '{name}' must be 1 to 64 letters, digits, '.', '_' or '-' "
                "and start with a letter or digit."
            )
        if not model or len(model) > MAX_ALIAS_TARGET_LENGTH:
            raise ValueError(f"Alias '{name}' needs a model to point at.")
        if name.lower() in seen:
            raise ValueError(f"Alias '{name}' is defined twice.")
        seen.add(name.lower())
        aliases.append({"name": name, "model": model})
    # One hop only: a target naming another alias would be served as a literal model id.
    for alias in aliases:
        if alias["model"].lower() in seen:
            raise ValueError(f"Alias '{alias['name']}' cannot point at another alias.")
    return aliases


def get_model_aliases() -> list[dict[str, str]]:
    from utils.openai_auto_switch_settings import _cached_setting
    try:
        return normalize_model_aliases(_cached_setting(MODEL_ALIASES_SETTING_KEY, None))
    except ValueError:
        # A row the current rules reject (hand-edited db) disables aliases instead of 500ing /v1.
        return []


def set_model_aliases(raw: Any, model_ids: Iterable[str] = ()) -> list[dict[str, str]]:
    """Replace the whole alias list; the same input always stores the same value. ``model_ids``
    are real ids an alias may not reuse: the rewrite would silently serve other weights."""
    from storage.studio_db import upsert_app_settings
    from utils.openai_auto_switch_settings import _invalidate

    aliases = normalize_model_aliases(raw)
    taken = {model_id.lower() for model_id in model_ids}
    for alias in aliases:
        if alias["name"].lower() in taken:
            raise ValueError(f"'{alias['name']}' is already the name of a model on this server.")
    upsert_app_settings({MODEL_ALIASES_SETTING_KEY: aliases})
    _invalidate(MODEL_ALIASES_SETTING_KEY)
    return aliases


def model_alias_map() -> dict[str, str]:
    """Lowercased alias name -> target model id."""
    return {alias["name"].lower(): alias["model"] for alias in get_model_aliases()}


def resolve_model_alias(name: Any) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return None
    return model_alias_map().get(name.strip().lower())


def alias_model_objects(
    objects: list[dict], pinned_object: Optional[Callable[[str, list[dict]], Optional[dict]]] = None
) -> list[dict]:
    """``/v1/models`` rows for aliases whose target is in ``objects``, the caller's visible
    catalog, so the listing never advertises a name that would 404. An alias the rewrite would
    apply shadows a same-named model row (dropped by the caller). ``pinned_object`` resolves a
    ``model:quant`` target the way ``GET /v1/models/{id}`` does (the quant must be on disk)."""
    aliases = get_model_aliases()
    if not aliases:
        return []
    by_id = {o["id"].lower(): o for o in objects if isinstance(o.get("id"), str)}
    rows = []
    for alias in aliases:
        target = alias["model"]
        listed = by_id.get(target.lower())
        if listed is None and ":" in target and pinned_object is not None:
            listed = pinned_object(target, objects)
        # Chat models only: media routes take ``model`` in multipart or upload bodies the rewrite skips.
        if listed is None or listed.get("task", "text-generation") != "text-generation":
            continue
        # The alias already names its quant; a listed one would invite an unresolvable "alias:Q4".
        row = {k: v for k, v in listed.items() if k not in ("display_name", "quant")}
        row.update(id = alias["name"], alias_for = target)
        rows.append(row)
    return rows


def _rewrite_alias_body(body: bytes, aliases: dict[str, str]) -> Optional[bytes]:
    """The body with an aliased top-level ``model`` replaced by its target, or None if unchanged."""
    # Cheap scan first so a large (image, audio) body is only parsed when it names an alias.
    if not any(
        value.decode("utf-8", "ignore").strip().lower() in aliases
        for value in _MODEL_VALUE_RE.findall(body)
    ):
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not isinstance(model, str):
        return None
    target = aliases.get(model.strip().lower())
    if target is None:
        return None
    payload["model"] = target
    return json.dumps(payload, ensure_ascii = False).encode("utf-8")


class ModelAliasMiddleware:
    """Swap an aliased ``model`` for its target before any route runs. Never reads the body when
    no alias is configured; registered inside MaxBodyMiddleware, so what it buffers is capped."""

    def __init__(
        self,
        app,
        skip_paths: Iterable[str] = (),
    ):
        self.app = app
        # MaxBodyMiddleware's upload passthroughs: their cap is the upload limit, not the JSON one.
        self.skip_paths = frozenset(skip_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method", "").upper() != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root_path = scope.get("root_path") or ""
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :] or "/"
        content_type = dict(scope.get("headers", [])).get(b"content-type", b"").lower()
        # JSON only: raw and multipart uploads stream past the body cap and must never be held.
        if (
            not path.startswith(_ALIAS_PATH_PREFIXES)
            or b"json" not in content_type
            or path.rstrip("/") in self.skip_paths
        ):
            await self.app(scope, receive, send)
            return
        aliases = model_alias_map()
        if not aliases:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                return
            chunks.append(message.get("body", b"") or b"")
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        rewritten = _rewrite_alias_body(body, aliases)
        if rewritten is not None:
            body = rewritten
            headers = [(k, v) for k, v in scope.get("headers", []) if k != b"content-length"]
            headers.append((b"content-length", str(len(body)).encode("ascii")))
            scope = {**scope, "headers": headers}

        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
