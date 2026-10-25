# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-process verdict on whether the Hub endpoint answers. After one transport failure, cached-load Hub calls use
their fallback for ``TTL_SECONDS`` instead of each waiting out a timeout. HTTP answers never set the verdict.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

# Matches the prequant freshness memo.
TTL_SECONDS = 300.0
METADATA_TIMEOUT_SECONDS = 10.0

_UNREACHABLE: dict[str, float] = {}
_lock = threading.Lock()


class HubUnreachableError(ConnectionError):
    """A ``ConnectionError`` so existing transport-failure handling takes it unchanged."""


def connection_failed(exc: BaseException) -> bool:
    """Not proxy errors: huggingface_hub raises those instead of falling back to the cache."""
    if isinstance(exc, HubUnreachableError):
        return True
    try:
        import httpx
        if isinstance(exc, httpx.ProxyError):
            return False
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass
    try:
        import requests
        if isinstance(exc, requests.exceptions.ProxyError):
            return False
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except ImportError:
        pass
    return isinstance(exc, (OSError, TimeoutError)) and not hasattr(exc, "response")


def endpoint() -> str:
    try:
        from huggingface_hub import constants
        return str(constants.ENDPOINT)
    except Exception:  # noqa: BLE001
        return ""


def hub_unreachable(at: Optional[str] = None) -> bool:
    key = endpoint() if at is None else at
    with _lock:
        down_at = _UNREACHABLE.get(key)
    return down_at is not None and time.monotonic() - down_at < TTL_SECONDS


def note_hub_failure(exc: BaseException, at: Optional[str] = None) -> bool:
    if not connection_failed(exc):
        return False
    if isinstance(exc, HubUnreachableError):
        return True  # a skipped call: the verdict that caused it stays dated at the real failure
    key = endpoint() if at is None else at
    with _lock:
        _UNREACHABLE[key] = time.monotonic()
    return True


def note_hub_success(at: Optional[str] = None) -> None:
    key = endpoint() if at is None else at
    with _lock:
        _UNREACHABLE.pop(key, None)


def _answered(exc: BaseException) -> bool:
    return getattr(exc, "response", None) is not None


def model_info(
    api: Any,
    repo_id: str,
    *,
    skip_when_unreachable: bool = True,
    **kwargs: Any,
) -> Any:
    """``skip_when_unreachable=False`` is for an answer that decides WHAT loads: always asked, only bounded."""
    at = endpoint()
    if skip_when_unreachable and hub_unreachable(at):
        raise HubUnreachableError(
            f"{at or 'the Hub'} did not connect within the last {int(TTL_SECONDS)} s"
        )
    if _is_hf_api(api):
        # Duck-typed clients (test doubles) may not take ``timeout``.
        kwargs.setdefault("timeout", METADATA_TIMEOUT_SECONDS)
    try:
        info = api.model_info(repo_id, **kwargs)
    except Exception as exc:
        if not note_hub_failure(exc, at) and _answered(exc):
            note_hub_success(at)
        raise
    note_hub_success(at)
    return info


def _is_hf_api(api: Any) -> bool:
    try:
        # Defining module: tests may swap the package attribute for a double.
        from huggingface_hub.hf_api import HfApi
        return isinstance(api, HfApi)
    except Exception:  # noqa: BLE001
        return False


class GuardedApi:
    def __init__(
        self,
        api: Any,
        skip_when_unreachable: bool = True,
    ):
        self._api = api
        self._skip = skip_when_unreachable

    def model_info(self, repo_id: str, *args: Any, **kwargs: Any) -> Any:
        if args:
            return self._api.model_info(repo_id, *args, **kwargs)
        return model_info(self._api, repo_id, skip_when_unreachable = self._skip, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


def guarded(api: Any, *, skip_when_unreachable: bool = True) -> Any:
    if api is None or isinstance(api, GuardedApi):
        return api
    return GuardedApi(api, skip_when_unreachable)


def always_asked(api: Any) -> Any:
    if api is None:
        return None
    return GuardedApi(api._api if isinstance(api, GuardedApi) else api, skip_when_unreachable = False)
