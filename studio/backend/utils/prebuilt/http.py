# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP primitives for the prebuilt installers: retry/backoff, token-safe
cross-host redirects, and verified streaming downloads."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .errors import PrebuiltFallback
from .logs import log

USER_AGENT = "unsloth-studio-whisper-prebuilt"
GITHUB_AUTH_HOSTS = {"api.github.com", "github.com"}
RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HTTP_FETCH_ATTEMPTS = 4
HTTP_FETCH_BASE_DELAY_SECONDS = 0.75


def parsed_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:  # noqa: BLE001
        return None
    return hostname.lower() if hostname else None


def should_send_github_auth(url: str | None) -> bool:
    return parsed_hostname(url) in GITHUB_AUTH_HOSTS


def auth_headers(url: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and should_send_github_auth(url):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_api_headers(url: str | None = None) -> dict[str, str]:
    return {"Accept": "application/vnd.github+json", **auth_headers(url)}


def is_github_api_url(url: str | None) -> bool:
    return parsed_hostname(url) == "api.github.com"


class _CrossHostAuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host.

    GitHub redirects release-asset downloads to CDN hosts whose signed URLs can
    reject a foreign Authorization header; urllib forwards headers to redirect
    targets by default (requests/huggingface_hub strip them).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None and parsed_hostname(newurl) != parsed_hostname(req.full_url):
            new_request.headers.pop("Authorization", None)
            new_request.unredirected_hdrs.pop("Authorization", None)
        return new_request


_URL_OPENER = urllib.request.build_opener(_CrossHostAuthStrippingRedirectHandler())


def is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        # GitHub returns 403 (not 429) when the API rate limit is hit; treat that
        # against api.github.com as retryable so a CI fleet gets a backoff cycle
        # before the source-build fallback fires. Other-host 403s stay fatal.
        if exc.code == 403:
            return is_github_api_url(getattr(exc, "url", None))
        return exc.code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    return False


def sleep_backoff(attempt: int) -> None:
    delay = HTTP_FETCH_BASE_DELAY_SECONDS * (2 ** max(attempt - 1, 0))
    delay += random.uniform(0.0, 0.2)
    time.sleep(delay)


def download_bytes(
    url: str,
    *,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers = headers or auth_headers(url))
            with _URL_OPENER.open(request, timeout = timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(f"fetch failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying")
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str) -> Any:
    headers = github_api_headers(url) if is_github_api_url(url) else auth_headers(url)
    data = download_bytes(url, timeout = 30, headers = headers)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise PrebuiltFallback(f"unexpected JSON type from {url}: {type(payload).__name__}")
    return payload


def atomic_replace_from_tempfile(tmp_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    os.replace(tmp_path, destination)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        tmp_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers = auth_headers(url))
            with tempfile.NamedTemporaryFile(
                prefix = destination.name + ".tmp-",
                dir = destination.parent,
                delete = False,
            ) as handle:
                tmp_path = Path(handle.name)
                with _URL_OPENER.open(request, timeout = 120) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file from {url}")
            atomic_replace_from_tempfile(tmp_path, destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok = True)
                except Exception:  # noqa: BLE001
                    pass
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(f"download failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying")
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
