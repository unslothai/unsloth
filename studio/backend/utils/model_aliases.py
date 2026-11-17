# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install-wide model aliases resolved once at the ASGI edge, before access checks; never grant more
than the target; names exclude / and :"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Any, Callable, Iterable, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from utils.openai_auto_switch_settings import (
    MAX_SEQ_LENGTH_CEILING,
    MLX_ONLY_SPEC_TYPES,
    PARALLEL_SLOTS_MAX,
    PARALLEL_SLOTS_MIN,
    VALID_KV_CACHE_DTYPES,
    VALID_SPECULATIVE_TYPES,
)

MODEL_ALIASES_SETTING_KEY = "openai_api_model_aliases"
MODEL_ALIAS_CONFIG_SCOPE_KEY = "unsloth.model_alias_load_config"
MAX_MODEL_ALIASES = 100
MAX_ALIAS_TARGET_LENGTH = 512
_ALIAS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ALIAS_PATH_PREFIXES = ("/v1", "/api/inference")
# Chat routes only: media, lifecycle and Decision API ids (e.g. STT small) are never rewritten.
_ALIAS_ROUTES = frozenset(
    (
        "/chat/completions",
        "/chat/count_tokens",
        "/completions",
        "/embeddings",
        "/messages",
        "/messages/count_tokens",
        "/responses",
    )
)
_MODEL_VALUE_RE = re.compile(rb'"model"\s*:\s*"([^"\\]*)"')


class ModelAliasLoadConfig(BaseModel):
    """GGUF launch settings a configured alias layers over its model's saved settings; an omitted
    field keeps the saved value. Only these fields, never arbitrary loader arguments."""

    model_config = ConfigDict(extra = "forbid")
    # 0 hands the context to the loader's Auto sizing.
    max_seq_length: Optional[StrictInt] = Field(None, ge = 0, le = MAX_SEQ_LENGTH_CEILING)
    kv_cache_dtype: Optional[str] = None
    speculative_type: Optional[str] = None
    n_parallel: Optional[StrictInt] = Field(None, ge = PARALLEL_SLOTS_MIN, le = PARALLEL_SLOTS_MAX)
    # A layer count pins Manual GPU memory; "auto" returns to automatic placement.
    gpu_layers: Optional[Union[StrictInt, Literal["auto"]]] = None
    n_cpu_moe: Optional[StrictInt] = Field(None, ge = 0, le = 1024)
    disable_vision: Optional[StrictBool] = None

    @field_validator("kv_cache_dtype")
    @classmethod
    def _supported_cache_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in VALID_KV_CACHE_DTYPES:
            raise ValueError(f"unsupported KV cache type '{value}'")
        return normalized

    @field_validator("speculative_type")
    @classmethod
    def _supported_speculation(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in MLX_ONLY_SPEC_TYPES or normalized not in VALID_SPECULATIVE_TYPES:
            raise ValueError(f"unsupported speculative decoding mode '{value}' for a GGUF model")
        from core.inference.llama_cpp import _canonicalize_spec_mode

        # Stored canonical, the form the loader records, so a spelling never reads as another mode.
        return _canonicalize_spec_mode(normalized)

    @field_validator("gpu_layers")
    @classmethod
    def _layer_count(cls, value):
        if isinstance(value, int) and not 0 <= value <= 2**31 - 1:
            raise ValueError('must be 0 to 2147483647, or "auto"')
        return value

    @model_validator(mode = "after")
    def _cpu_moe_needs_layers(self):
        # Only a manual layer count honors --n-cpu-moe; an inherited mode could silently ignore it.
        if self.n_cpu_moe is not None and not isinstance(self.gpu_layers, int):
            raise ValueError(
                "n_cpu_moe needs gpu_layers set to a layer count in the same load_config"
            )
        return self


def normalize_alias_load_config(raw: Any) -> Optional[dict[str, Any]]:
    """Validated canonical settings, or None for an empty object; raises ValueError."""
    try:
        config = ModelAliasLoadConfig.model_validate(raw).model_dump(exclude_none = True)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "invalid value")).removeprefix("Value error, ")
        raise ValueError(f"{field}: {message}" if field else message) from None
    return config or None


def request_alias_entry(scope: Any, model: Optional[str]) -> Optional[dict]:
    """The middleware's record of a configured alias for this exact rewritten target, or None."""
    entry = scope.get(MODEL_ALIAS_CONFIG_SCOPE_KEY) if isinstance(scope, dict) else None
    if not entry or entry["model"] != model:
        return None
    return entry


def normalize_model_aliases(
    raw: Any,
    *,
    tolerant: bool = False,
    preserved: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Validated ``[{"name", "model"[, "load_config"]}]``; raises ValueError naming the first bad
    entry. ``tolerant`` (reads) marks a ``load_config`` this version cannot read
    ``load_config_unsupported`` instead of failing, so one entry from a newer release disables only
    its own alias. ``preserved`` (writes) maps lowercased names to stored unreadable configs a client
    may send back unchanged."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Model aliases must be a list.")
    if len(raw) > MAX_MODEL_ALIASES:
        raise ValueError(f"At most {MAX_MODEL_ALIASES} model aliases are allowed.")
    aliases: list[dict[str, Any]] = []
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
        alias: dict[str, Any] = {"name": name, "model": model}
        raw_config = entry.get("load_config")
        if raw_config is not None:
            try:
                config = normalize_alias_load_config(raw_config)
            except ValueError as exc:
                if preserved is not None and preserved.get(name.lower()) == raw_config:
                    # A newer release's settings, sent back untouched: keep them as stored.
                    alias["load_config"] = copy.deepcopy(raw_config)
                elif tolerant:
                    alias["load_config_unsupported"] = True
                else:
                    raise ValueError(f"Alias '{name}' has an invalid load_config: {exc}") from None
            else:
                if config:
                    alias["load_config"] = config
        aliases.append(alias)
    # One hop only: a target naming another alias would be served as a literal model id.
    for alias in aliases:
        if alias["model"].lower() in seen:
            raise ValueError(f"Alias '{alias['name']}' cannot point at another alias.")
    return aliases


def alias_is_configured(alias: dict[str, Any]) -> bool:
    return bool(alias.get("load_config") or alias.get("load_config_unsupported"))


_normalized: tuple[Any, list[dict[str, Any]]] = (None, [])


def stored_model_aliases() -> list[Any]:
    """Read persisted aliases directly so saves never trust a stale inference cache."""
    from storage.studio_db import get_app_setting

    raw = get_app_setting(MODEL_ALIASES_SETTING_KEY, None)
    return raw if isinstance(raw, list) else []


def get_model_aliases() -> list[dict[str, Any]]:
    global _normalized
    from utils.openai_auto_switch_settings import _cached_setting

    raw = _cached_setting(MODEL_ALIASES_SETTING_KEY, None)
    # The settings cache hands back the same object until it refreshes: validate once per object.
    if raw is not _normalized[0]:
        try:
            aliases = normalize_model_aliases(raw, tolerant = True)
        except ValueError:
            # A row the current rules reject (hand-edited db) disables aliases instead of 500ing /v1.
            aliases = []
        _normalized = (raw, aliases)
    return _normalized[1]


def model_aliases_etag(stored: list[Any]) -> str:
    """Version of the stored list a client read; a save naming an older one is refused."""
    encoded = json.dumps(stored, sort_keys = True, separators = (",", ":"), ensure_ascii = False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


class ModelAliasConflict(ValueError):
    """The save would overwrite alias settings the client never saw; nothing was written."""


_RELOAD_TO_SAVE = "Reload the page and try again."


def _keep_unseen_configurations(raw: Any, stored: list[Any]) -> Any:
    """A save without an etag comes from a client that may not know ``load_config`` (an older Web UI
    tab or a script). It may edit plain aliases freely and send a configured one back unchanged,
    which keeps its settings. An entry carrying a ``load_config`` key, a value or null, comes from a
    client that knows the field and may change it. Renaming, deleting or retargeting a configured
    alias without that key would drop settings this client cannot see, so it is refused."""
    if not isinstance(raw, list):
        return raw
    by_name = {
        entry.get("name"): entry
        for entry in raw
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    kept = list(raw)
    for stored_entry in stored:
        if not isinstance(stored_entry, dict) or stored_entry.get("load_config") is None:
            continue
        name = stored_entry.get("name")
        sent = by_name.get(name)
        if sent is not None and "load_config" in sent:
            continue
        same_model = (
            sent is not None
            and isinstance(sent.get("model"), str)
            and sent["model"].strip() == str(stored_entry.get("model") or "")
        )
        if not same_model:
            raise ModelAliasConflict(
                f"Alias '{name}' has saved launch settings this page does not show, and this save "
                f"would rename, retarget or remove it. {_RELOAD_TO_SAVE}"
            )
        kept[kept.index(sent)] = {**sent, "load_config": copy.deepcopy(stored_entry["load_config"])}
    return kept


_save_lock = threading.Lock()


def set_model_aliases(
    raw: Any,
    model_ids: Iterable[str] = (),
    etag: Optional[str] = None,
) -> list[Any]:
    """Replace the whole alias list and return what was stored; the same input always stores the
    same value. ``model_ids`` are real ids an alias may not reuse: the rewrite would silently serve
    other weights. ``etag`` (from a read) refuses a save over a list that changed since; without one
    a save keeps configurations the client may not know about (_keep_unseen_configurations)."""
    from storage.studio_db import upsert_app_settings
    from utils.openai_auto_switch_settings import _invalidate

    with _save_lock:
        stored = stored_model_aliases()
        if etag is not None:
            if etag != model_aliases_etag(stored):
                raise ModelAliasConflict(
                    f"The model aliases changed since this page loaded. {_RELOAD_TO_SAVE}"
                )
        else:
            raw = _keep_unseen_configurations(raw, stored)
        preserved = {}
        try:
            for alias in normalize_model_aliases(stored, tolerant = True):
                if alias.get("load_config_unsupported"):
                    entry = next(
                        e for e in stored if isinstance(e, dict) and e.get("name") == alias["name"]
                    )
                    preserved[alias["name"].lower()] = entry.get("load_config")
        except ValueError:
            pass
        aliases = normalize_model_aliases(raw, preserved = preserved)
        taken = {model_id.lower() for model_id in model_ids}
        for alias in aliases:
            if alias["name"].lower() in taken:
                raise ValueError(
                    f"'{alias['name']}' is already the name of a model on this server."
                )
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
    objects: list[dict],
    pinned_object: Optional[Callable[[str, list[dict]], Optional[dict]]] = None,
    loaded_config_matches: Optional[Callable[[dict], bool]] = None,
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
        if alias.get("load_config_unsupported"):
            # Settings from a newer release: requests are refused, so the listing must not offer it.
            continue
        config = alias.get("load_config")
        if config and not (
            listed.get("loaded")
            and loaded_config_matches is not None
            and loaded_config_matches(alias)
        ):
            # Like an unloaded quant pin, this configuration has no resident context or lease.
            row = {k: row[k] for k in ("object", "created", "owned_by", "task") if k in row}
            row["loaded"] = False
        row.update(id = alias["name"], alias_for = target)
        rows.append(row)
    return rows


def _rewrite_alias_payload(body: bytes, aliases: dict[str, str]) -> Optional[tuple[bytes, str]]:
    """``(body with the aliased top-level model replaced by its target, lowercased alias name)``, or
    None if unchanged."""
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
    name = model.strip().lower()
    target = aliases.get(name)
    if target is None:
        return None
    payload["model"] = target
    return json.dumps(payload, ensure_ascii = False).encode("utf-8"), name


class ModelAliasMiddleware:
    """Swap an aliased ``model`` for its target before any route runs. Never reads the body when
    no alias is configured; registered inside MaxBodyMiddleware, so what it buffers is capped."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method", "").upper() != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root_path = scope.get("root_path") or ""
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :] or "/"
        prefix = next((p for p in _ALIAS_PATH_PREFIXES if path.startswith(p)), "")
        content_type = dict(scope.get("headers", [])).get(b"content-type", b"").lower()
        # JSON only: raw and multipart uploads stream past the body cap and must never be held.
        if (
            not prefix
            or path[len(prefix) :].rstrip("/") not in _ALIAS_ROUTES
            or b"json" not in content_type
        ):
            await self.app(scope, receive, send)
            return
        entries = get_model_aliases()
        aliases = {entry["name"].lower(): entry["model"] for entry in entries}
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
        rewritten = _rewrite_alias_payload(body, aliases)
        if rewritten is not None:
            body, name = rewritten
            configured = next((entry for entry in entries if entry["name"].lower() == name), None)
            if configured and alias_is_configured(configured):
                # Routes read only this marker, set here from the stored list, never from the body.
                scope[MODEL_ALIAS_CONFIG_SCOPE_KEY] = {
                    "model": configured["model"],
                    "load_config": dict(configured.get("load_config") or {}),
                    "unsupported": bool(configured.get("load_config_unsupported")),
                }
            headers = [(k, v) for k, v in scope.get("headers", []) if k != b"content-length"]
            headers.append((b"content-length", str(len(body)).encode("ascii")))
            # In place: outer middlewares (keep-warm) read markers the route sets on this same dict.
            scope["headers"] = headers

        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
