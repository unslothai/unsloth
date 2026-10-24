"""One per-process verdict on whether the Hub endpoint answers at all.

A cached load asks the Hub several metadata questions (listings, gated flag, ``base_model`` tag, ref freshness),
each with a cache or "unknown" answer for when the Hub does not reply. On a network that drops packets every one of
them would wait out its own timeout, and huggingface_hub's default client has none (``timeout=None``). After one
transport failure (connect refused / reset, connect or read timeout) the rest are answered from their fallback for
``TTL_SECONDS`` without touching the network. HTTP answers (gated, 404, 5xx, rate limit) say the Hub is up and never
set the verdict.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

# How long one transport failure stands for the whole endpoint. Matches the prequant freshness memo.
TTL_SECONDS = 300.0
# Bound for metadata calls (model_info) that would otherwise use huggingface_hub's unbounded default.
METADATA_TIMEOUT_SECONDS = 10.0

# Endpoint -> monotonic time its connection last failed.
_UNREACHABLE: dict[str, float] = {}
_lock = threading.Lock()


class HubUnreachableError(ConnectionError):
    """Raised in place of a Hub call skipped because the endpoint failed to connect within the TTL. A
    ``ConnectionError`` so a caller's existing transport-failure handling takes it unchanged."""


def connection_failed(exc: BaseException) -> bool:
    """A transport failure that says nothing about the repo: connect refused / reset, connect or read timeout. Not an
    HTTP answer (gated, 404, 5xx, rate limit) and not a proxy error, which huggingface_hub raises instead of falling
    back to the cache."""
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
    """True when ``at`` (default: the live endpoint) failed to connect less than ``TTL_SECONDS`` ago."""
    key = endpoint() if at is None else at
    with _lock:
        down_at = _UNREACHABLE.get(key)
    return down_at is not None and time.monotonic() - down_at < TTL_SECONDS


def note_hub_failure(exc: BaseException, at: Optional[str] = None) -> bool:
    """Record ``exc`` against the endpoint when it is a transport failure. Returns whether it was one."""
    if not connection_failed(exc):
        return False
    if isinstance(exc, HubUnreachableError):
        return True  # a skipped call: the verdict that caused it stays dated at the real failure
    key = endpoint() if at is None else at
    with _lock:
        _UNREACHABLE[key] = time.monotonic()
    return True


def note_hub_success(at: Optional[str] = None) -> None:
    """The endpoint answered (any HTTP status): forget a standing failure."""
    key = endpoint() if at is None else at
    with _lock:
        _UNREACHABLE.pop(key, None)


def _answered(exc: BaseException) -> bool:
    """An HTTP error carries a response: the Hub answered, so it is reachable."""
    return getattr(exc, "response", None) is not None


def model_info(
    api: Any,
    repo_id: str,
    *,
    skip_when_unreachable: bool = True,
    **kwargs: Any,
) -> Any:
    """``api.model_info(repo_id, **kwargs)`` under the verdict: raises ``HubUnreachableError`` without a request
    while the endpoint is down, bounds the call with ``METADATA_TIMEOUT_SECONDS`` unless the caller set one, and
    records the outcome. Every exception the real call raises propagates unchanged.

    ``skip_when_unreachable=False`` is for an answer that decides WHAT loads (a base_model tag, whether a hosted
    pre-quantised checkpoint exists): it is always asked, only bounded and recorded, so a network that came back
    inside the TTL gets the same pick a healthy one does."""
    at = endpoint()
    if skip_when_unreachable and hub_unreachable(at):
        raise HubUnreachableError(
            f"{at or 'the Hub'} did not connect within the last {int(TTL_SECONDS)} s"
        )
    if _is_hf_api(api):
        # HfApi's own keyword; a duck-typed client (a test double, a wrapper) gets the call exactly as written.
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
        # The defining module, not the package attribute, which a test may have swapped for a double.
        from huggingface_hub.hf_api import HfApi
        return isinstance(api, HfApi)
    except Exception:  # noqa: BLE001
        return False


class GuardedApi:
    """An ``HfApi`` whose ``model_info`` goes through :func:`model_info`; everything else is the wrapped client's."""

    def __init__(
        self,
        api: Any,
        skip_when_unreachable: bool = True,
    ):
        self._api = api
        self._skip = skip_when_unreachable

    def model_info(self, repo_id: str, *args: Any, **kwargs: Any) -> Any:
        # Positional extras are the wrapped client's business; only keyword calls are guarded.
        if args:
            return self._api.model_info(repo_id, *args, **kwargs)
        return model_info(self._api, repo_id, skip_when_unreachable = self._skip, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


def guarded(api: Any, *, skip_when_unreachable: bool = True) -> Any:
    """``api`` with its ``model_info`` under the reachability verdict (see :class:`GuardedApi` and
    :func:`model_info`). None stays None."""
    if api is None or isinstance(api, GuardedApi):
        return api
    return GuardedApi(api, skip_when_unreachable)


def always_asked(api: Any) -> Any:
    """``api`` (raw or guarded) with ``model_info`` bounded and recorded but never skipped: for a lookup whose answer
    picks what loads. None stays None."""
    if api is None:
        return None
    return GuardedApi(api._api if isinstance(api, GuardedApi) else api, skip_when_unreachable = False)
