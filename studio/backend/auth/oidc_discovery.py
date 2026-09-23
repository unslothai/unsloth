# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded, cached access to an OpenID Provider's discovery metadata and JWKS."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit

import httpx

from .oidc_config import OIDCConfig


_MAX_DOCUMENT_BYTES = 1024 * 1024


class OIDCProviderUnavailable(RuntimeError):
    """Discovery or key material could not be obtained safely from the provider."""


@dataclass(frozen = True, slots = True)
class OIDCProviderMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    end_session_endpoint: Optional[str] = None


def _endpoint(value: object, name: str, *, issuer_scheme: str) -> str:
    if not isinstance(value, str):
        raise OIDCProviderUnavailable(f"OIDC discovery document has no valid {name}")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OIDCProviderUnavailable(f"OIDC discovery document has no valid {name}")
    if parsed.scheme != issuer_scheme:
        raise OIDCProviderUnavailable(f"OIDC discovery {name} uses a different URL scheme")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise OIDCProviderUnavailable(f"OIDC discovery document has no valid {name}")
    return value


class OIDCDiscoveryClient:
    """Fetch and cache discovery/JWKS documents with explicit time and size bounds."""

    def __init__(
        self,
        config: OIDCConfig,
        *,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("OIDC discovery needs an enabled configuration")
        self._config = config
        self._timeout = httpx.Timeout(timeout_seconds)
        self._ttl = cache_ttl_seconds
        self._clock = clock
        self._transport = transport
        self._metadata: Optional[OIDCProviderMetadata] = None
        self._metadata_expires = 0.0
        self._jwks: Optional[dict] = None
        self._jwks_expires = 0.0
        self._lock = threading.RLock()

    @property
    def discovery_url(self) -> str:
        return f"{self._config.issuer}/.well-known/openid-configuration"

    def _json_document(self, url: str, what: str) -> dict:
        try:
            with httpx.Client(
                timeout = self._timeout,
                follow_redirects = False,
                transport = self._transport,
            ) as client:
                response = client.get(url, headers = {"Accept": "application/json"})
                response.raise_for_status()
                if len(response.content) > _MAX_DOCUMENT_BYTES:
                    raise OIDCProviderUnavailable(f"OIDC provider returned oversized {what}")
                document = response.json()
        except OIDCProviderUnavailable:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise OIDCProviderUnavailable(f"OIDC provider {what} is unavailable") from exc
        if not isinstance(document, dict):
            raise OIDCProviderUnavailable(f"OIDC provider returned invalid {what}")
        return document

    def metadata(self, *, force_refresh: bool = False) -> OIDCProviderMetadata:
        now = self._clock()
        if not force_refresh and self._metadata is not None and now < self._metadata_expires:
            return self._metadata
        with self._lock:
            now = self._clock()
            if not force_refresh and self._metadata is not None and now < self._metadata_expires:
                return self._metadata
            document = self._json_document(self.discovery_url, "discovery")
            issuer = document.get("issuer")
            if issuer != self._config.issuer:
                raise OIDCProviderUnavailable("OIDC discovery issuer does not match configuration")
            issuer_scheme = urlsplit(self._config.issuer).scheme
            end_session = document.get("end_session_endpoint")
            metadata = OIDCProviderMetadata(
                issuer = issuer,
                authorization_endpoint = _endpoint(
                    document.get("authorization_endpoint"),
                    "authorization_endpoint",
                    issuer_scheme = issuer_scheme,
                ),
                token_endpoint = _endpoint(
                    document.get("token_endpoint"),
                    "token_endpoint",
                    issuer_scheme = issuer_scheme,
                ),
                jwks_uri = _endpoint(
                    document.get("jwks_uri"), "jwks_uri", issuer_scheme = issuer_scheme
                ),
                end_session_endpoint = (
                    _endpoint(end_session, "end_session_endpoint", issuer_scheme = issuer_scheme)
                    if end_session is not None
                    else None
                ),
            )
            self._metadata = metadata
            self._metadata_expires = now + self._ttl
            return metadata

    def jwks(self, *, force_refresh: bool = False) -> dict:
        now = self._clock()
        if not force_refresh and self._jwks is not None and now < self._jwks_expires:
            return self._jwks
        with self._lock:
            now = self._clock()
            if not force_refresh and self._jwks is not None and now < self._jwks_expires:
                return self._jwks
            document = self._json_document(self.metadata().jwks_uri, "signing keys")
            keys = document.get("keys")
            if (
                not isinstance(keys, list)
                or not keys
                or not all(isinstance(key, dict) for key in keys)
            ):
                raise OIDCProviderUnavailable("OIDC provider returned invalid signing keys")
            self._jwks = document
            self._jwks_expires = now + self._ttl
            return document

    def clear_cache(self) -> None:
        with self._lock:
            self._metadata = None
            self._metadata_expires = 0.0
            self._jwks = None
            self._jwks_expires = 0.0
