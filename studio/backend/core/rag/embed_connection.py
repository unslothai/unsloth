# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG embedder served by a saved connection's OpenAI-compatible ``/embeddings`` endpoint.

Selected by an embedding model of the form ``connection:<connection name or id>/<model id>``,
so nothing is loaded locally and the connection's own server decides where the model runs."""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import logging
import math
import re
import threading
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

from . import config

logger = logging.getLogger(__name__)

_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_ATTEMPTS = 3
# Text Embeddings Inference refuses more inputs per request by default (--max-client-batch-size 32).
_MAX_BATCH = 32
_TIMEOUT = httpx.Timeout(120.0, connect = 10.0)


class ConnectionEmbedderError(RuntimeError):
    """A missing or disabled connection, or a bad answer from it."""


def _find_connection(name: str) -> tuple[dict, str, str]:
    from storage import providers_db

    ref = name.strip()[len(config.CONNECTION_MODEL_PREFIX) :].strip()
    # Longest match wins, so a connection name containing "/" still resolves.
    best = None
    for provider in providers_db.list_providers():
        for label in (provider.get("display_name"), provider.get("id")):
            label = (label or "").strip()
            if label and ref.lower().startswith(label.lower() + "/"):
                if best is None or len(label) > len(best[1]):
                    best = (provider, label)
    if best is None:
        raise ConnectionEmbedderError(
            f"No saved connection matches {name!r}. Use connection:<connection name>/<model id>."
        )
    provider, label = best
    model = ref[len(label) + 1 :].strip()
    if not model:
        raise ConnectionEmbedderError(f"{name!r} names no model after the connection name.")
    return provider, label, model


def resolve_connection_model(name: str) -> tuple[dict, str]:
    provider, label, model = _find_connection(name)
    if not provider.get("is_enabled", True):
        raise ConnectionEmbedderError(f"The connection {label!r} is disabled.")
    return provider, model


def _client_for(model_name: str):
    from core.inference.external_provider import ExternalProviderClient
    from storage import credential_secrets

    destination = ("base_url", "provider_type", "api_type", "extra_headers")
    for _ in range(3):
        provider, model = resolve_connection_model(model_name)
        api_key = credential_secrets.get_provider_api_key(provider["id"]) or ""
        # Re-read: an edit landing between the two reads must not send the new key to the old URL.
        again, _ = resolve_connection_model(model_name)
        if all(provider.get(k) == again.get(k) for k in destination):
            break
    else:
        raise ConnectionEmbedderError(f"The connection for {model_name!r} is being edited; retry.")
    headers = provider.get("extra_headers")
    client = ExternalProviderClient(
        provider_type = provider["provider_type"],
        base_url = provider["base_url"],
        api_key = api_key,
        api_type = provider.get("api_type") or "chat_completions",
        **({"extra_headers": headers} if headers else {}),
    )
    return client, provider, model


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _server_of(base_url: str) -> str:
    # Spellings of one server are one server; userinfo and query carry credentials, not a server.
    try:
        parts = urlsplit(base_url.strip())
        host = (parts.hostname or "").rstrip(".")
        port = parts.port
    except ValueError:
        return base_url.strip()
    scheme = parts.scheme.lower()
    with contextlib.suppress(ValueError):
        host = ipaddress.ip_address(host).compressed
    host = "localhost" if host in _LOOPBACK_HOSTS else f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        host = f"{host}:{port}"
    return f"{scheme}://{host}{parts.path.rstrip('/').removesuffix('/v1').rstrip('/')}"


def connection_destination(provider: dict) -> str:
    server = _server_of(provider.get("base_url") or "")
    server = f"{provider.get('provider_type') or ''}\n{server}"
    return hashlib.sha256(server.encode("utf-8")).hexdigest()[:16]


def connection_identity(name: str, destination: str | None = None) -> str:
    # The server is part of the vector space, not only the connection name and model id.
    if destination is None:
        try:
            destination = connection_destination(_find_connection(name)[0])
        except ConnectionEmbedderError:
            destination = "unresolved"
    return f"{config.embedding_identity('connection', name)}:{destination}"


def stamp_legacy_identities(conn) -> None:
    rows = conn.execute(
        "SELECT DISTINCT embedding_model FROM documents WHERE embedding_model LIKE 'connection:%'"
    ).fetchall()
    for row in rows:
        identity = row[0]
        name = config.embedding_identity_model(identity)
        if name is None or identity != config.embedding_identity("connection", name):
            continue
        conn.execute(
            "UPDATE documents SET embedding_model=? WHERE embedding_model=?",
            (connection_identity(name), identity),
        )


# Every digit, symbol, newline and letter run is >= 1 WordPiece token: an upper bound for 512-token embedders.
_TOKEN_UNITS = re.compile(r"\d|[^\W\d_]+|[^\w\s]|_|\n")


def _estimate_tokens(text: str) -> int:
    return max(
        1, len(_TOKEN_UNITS.findall(text)), math.ceil(len(text.encode("utf-8", "ignore")) / 3)
    )


def _remote_http_client() -> httpx.Client:
    # Same proxy policy as chat on a connection (external_provider._create_shared_http_client, #6090).
    try:
        return httpx.Client(timeout = _TIMEOUT)
    except (ImportError, ValueError) as exc:
        if "Unknown scheme for proxy URL" not in str(exc) and "socksio" not in str(exc):
            raise
        return httpx.Client(timeout = _TIMEOUT, trust_env = False)


class ConnectionEmbeddingBackend:
    def __init__(self, *, attempts: int = _ATTEMPTS) -> None:
        self._attempts = max(1, attempts)
        self._http = _remote_http_client()
        self._dims: dict[str, int] = {}
        self._lock = threading.Lock()
        self._served = threading.local()

    def _post(self, model_name: str, texts: list[str]) -> tuple[list[list[float]], str]:
        from core.inference.external_provider import _append_provider_path

        client, provider, model = _client_for(model_name)
        client.set_header_context(None, model)
        url = _append_provider_path(client.base_url, "embeddings")
        body = {"model": model, "input": texts, "encoding_format": "float"}
        label = provider.get("display_name") or provider.get("id")
        for attempt in range(self._attempts):
            try:
                response = self._http.post(url, json = body, headers = client._auth_headers())
            except httpx.TransportError as exc:
                if attempt + 1 == self._attempts:
                    raise ConnectionEmbedderError(
                        f"Could not reach the connection {label!r}: {exc}"
                    ) from exc
            else:
                if response.status_code not in _RETRY_STATUSES or attempt + 1 == self._attempts:
                    break
            time.sleep(2**attempt)
        if response.status_code >= 400:
            detail = response.text.strip()[:300]
            raise ConnectionEmbedderError(
                f"The connection {label!r} refused embeddings for {model!r} "
                f"(HTTP {response.status_code}): {detail}"
            )
        try:
            items = sorted(response.json()["data"], key = lambda d: d.get("index", 0))
            rows = [item["embedding"] for item in items]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ConnectionEmbedderError(
                f"The connection {label!r} did not return OpenAI-style embeddings for {model!r}."
            ) from exc
        if len(rows) != len(texts):
            raise ConnectionEmbedderError(
                f"The connection {label!r} returned {len(rows)} vectors for {len(texts)} inputs."
            )
        return rows, connection_destination(provider)

    def encode(
        self,
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        name = model_name or config.effective_embedding_model()
        texts = list(texts)
        self._served.destination = None
        if not texts:
            return np.zeros((0, self.dim(model_name = name)), dtype = np.float32)
        rows: list[list[float]] = []
        destinations = set()
        batch = max(1, min(config.EMBED_BATCH, _MAX_BATCH))
        for start in range(0, len(texts), batch):
            part, destination = self._post(name, texts[start : start + batch])
            rows.extend(part)
            destinations.add(destination)
        if len(destinations) > 1:
            raise ConnectionEmbedderError(
                f"The connection for {name!r} changed while embedding; retry."
            )
        self._served.destination = destinations.pop()
        arr = np.asarray(rows, dtype = np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ConnectionEmbedderError(f"Embeddings from {name!r} have shape {arr.shape}.")
        with self._lock:
            self._dims[name] = int(arr.shape[1])
        if normalize:
            norms = np.linalg.norm(arr, axis = 1, keepdims = True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def served_embedding_identity(self) -> str | None:
        return getattr(self._served, "destination", None)

    def dim(self, *, model_name = None) -> int:
        name = model_name or config.effective_embedding_model()
        with self._lock:
            known = self._dims.get(name)
        if known is not None:
            return known
        return int(self.encode(["dimension probe"], model_name = name, normalize = False).shape[1])

    def max_tokens(self, *, model_name = None):
        return None

    def token_counter(self, *, model_name = None):
        return _estimate_tokens

    def warm(self, *, model_name = None):
        self.dim(model_name = model_name)

    # Not _shutdown: an unload must not close the pool under an encode still running on it.
    def close(self) -> None:
        self._http.close()
