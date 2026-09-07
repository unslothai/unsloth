# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-scoped Hugging Face token helpers."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Literal, MutableMapping, Optional, Union

HfTokenArg = Optional[Union[str, Literal[False]]]

# Anonymous-sentinel cache identity, kept apart from ``None``'s: a slot filled under the
# ambient token must not be served to an API key denied it. Not hex, so no digest collides.
ANONYMOUS_CACHE_IDENTITY = "anon"


def hf_token_arg(hf_token: Optional[str], *, allow_ambient_token: bool) -> HfTokenArg:
    """Return the explicit token, or choose ambient versus anonymous access."""
    token = (hf_token or "").strip()
    if token:
        return token
    return None if allow_ambient_token else False


# Mirrors the list hub/services/download_lifecycle.py scrubs for download workers.
_HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)

# ``auth_check`` has no timeout kwarg in the pinned Hub client, and neither does the
# session under it: 0.x hands requests no timeout at all and 1.30's httpx client carries
# ``Timeout(timeout=None)``. Call ``/auth-check`` through ``get_session().get`` with an
# explicit ``timeout`` instead (same pattern as ``hf_token_validation._check_remote``), so
# a stalled connection cannot leave orphan probe workers behind after the caller returns.
_REPO_ACCESS_PROBE_TIMEOUT_S = 10.0


def apply_token_to_child_env(env: MutableMapping[str, str], hf_token: HfTokenArg) -> None:
    """Grant a spawned probe exactly its caller's credential.

    A child env is seeded from the parent's, so not *setting* a token is not denying one.
    Only the sentinel scrubs; ``None`` keeps the inherited env on purpose.
    """
    if isinstance(hf_token, str) and hf_token:
        # Scrub before granting: setting HF_TOKEN alone leaves an operator credential
        # sitting in HF_HUB_TOKEN or a legacy alias, so the child holds two.
        for key in _HF_TOKEN_ENV_KEYS:
            env.pop(key, None)
        env["HF_TOKEN"] = hf_token
        # An inherited HF_HUB_DISABLE_IMPLICIT_TOKEN=1 would otherwise 401 a gated repo.
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0"
        return
    if is_anonymous(hf_token):
        for key in _HF_TOKEN_ENV_KEYS:
            env.pop(key, None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def normalize_token(hf_token: HfTokenArg) -> HfTokenArg:
    """Trim an explicit token without laundering ``False`` into ``None`` (= ambient)."""
    if is_anonymous(hf_token):
        return False
    return (hf_token or "").strip() or None


def is_anonymous(hf_token: HfTokenArg) -> bool:
    """Named because a bare ``is False`` invites a ``not hf_token`` "simplification"."""
    return hf_token is False


# Positive and negative answers share this TTL: a revoked token must not keep
# reading the host cache, and a flapping Hub must not be hit on every request.
_REPO_ACCESS_TTL_S = 60.0
# A probe that timed out never got an answer *about the credential*, so it is not worth a
# minute. Short enough that one stalled connection does not deny a valid token for the
# whole TTL, long enough that a Hub which is hanging for everyone is not re-dialled per
# request. The request in hand is still denied: fail closed, just not for long.
_REPO_ACCESS_UNREACHABLE_TTL_S = 5.0
_REPO_ACCESS_CACHE_MAX = 1024
_repo_access_cache: dict[tuple[str, str, str], tuple[float, bool]] = {}
_repo_access_lock = threading.Lock()
# One probe per key, not one per caller. The probe runs outside ``_repo_access_lock`` (it
# is a network call and must not hold a lock every reader needs), so without this a cold
# key with N concurrent callers opens N connections -- measured at 32/32, and 16/16 TCP
# connections against a stalled Hub. Callers that arrive during a probe wait for its
# answer.
_repo_access_inflight: dict[tuple[str, str, str], threading.Lock] = {}


class _ProbeTimedOut(Exception):
    """Raised when the /auth-check HTTP call hits its timeout budget."""


def _is_probe_timeout(exc: BaseException) -> bool:
    for cls in type(exc).__mro__:
        name = cls.__name__
        if name in {"Timeout", "ReadTimeout", "ConnectTimeout"}:
            return True
        module = getattr(cls, "__module__", "") or ""
        if module.startswith(("requests.", "httpx.", "urllib3.")) and "Timeout" in name:
            return True
    return False


def reset_repo_access_cache() -> None:
    """Drop memoized Hub access answers. Tests only."""
    with _repo_access_lock:
        _repo_access_cache.clear()
        _repo_access_inflight.clear()


def cache_reads_authorized(
    hf_token: HfTokenArg,
    *,
    repo_id: str,
    repo_type: str = "model",
) -> bool:
    """Whether this caller may read the host Hub disk cache for *repo_id*.

    ``is_anonymous`` authenticates the caller class, not the credential: any
    token-shaped string leaves the sentinel and would otherwise take the disk
    fast paths. Ambient ``None`` is the operator and may use the cache. An
    explicit token is authorized only after ``auth_check`` confirms it can
    read the named repository. ``repo_info`` is not enough: gated public
    metadata still returns for an invalid token, which would serve the host
    cache to a credential that cannot fetch the files.

    Offline: the Hub probe cannot run, so an explicit token is denied unless a
    recent online probe is still memoized (see ``_REPO_ACCESS_TTL_S``). That is
    intentional fail-closed: without wire proof, the cache must not answer for a
    credential that might be a token-shaped string from an API key. UI sessions
    keep ``None`` and still read the cache offline.

    A local filesystem path is not a repo id and is denied without a probe. Only
    ``None`` and the empty string mean ambient; every other non-``str`` is denied,
    so a value that reaches here untyped cannot fall through to the ambient answer.
    """
    if is_anonymous(hf_token):
        return False
    # Deny-by-default on the type, not allow-by-default: ``not isinstance(str)`` would
    # hand the ambient answer to True, 1, b"...", or any object a future untyped
    # ``payload.get("hf_token")`` puts here. Nothing reaches this today -- every HTTP
    # boundary is a pydantic ``Optional[str]``, which rejects rather than coerces -- and
    # this keeps it that way.
    if hf_token is None:
        return True
    if not isinstance(hf_token, str):
        return False
    if not hf_token:
        return True
    repo = (repo_id or "").strip()
    if not repo:
        return False
    if _is_local_path(repo):
        # ``auth_check`` does not validate its argument, it interpolates it into
        # ``{endpoint}/api/{repo_type}s/{repo_id}/auth-check``. Probing a local path would
        # put that path on the wire to the Hub with the caller's bearer token attached,
        # for a round trip whose only possible answer is "no". The host Hub cache is not
        # what a local path names, so there is nothing here to authorize.
        return False
    return _explicit_token_reaches_repo(repo, hf_token, repo_type)


def _is_local_path(repo_id: str) -> bool:
    """Lazy: hub.utils.paths pulls in the path stack, this module is imported beneath it."""
    try:
        from hub.utils.paths import is_local_path
        return is_local_path(repo_id)
    except Exception:
        return False


def _hub_offline() -> bool:
    try:
        from utils.utils import hf_env_offline
        return hf_env_offline()
    except Exception:
        # Fail-open on the *offline question* only: authorization still needs a probe to
        # succeed. Logged because deciding "online" inside an air-gapped install means
        # every explicit-token request pays the full probe timeout.
        import logging
        logging.getLogger(__name__).debug(
            "Could not determine Hub offline state; assuming online", exc_info = True
        )
        return False


def _cached_repo_access(key: tuple[str, str, str], now: float) -> Optional[bool]:
    cached = _repo_access_cache.get(key)
    if cached is not None and cached[0] > now:
        return cached[1]
    return None


def _explicit_token_reaches_repo(repo_id: str, token: str, repo_type: str) -> bool:
    key = (
        repo_id.casefold(),
        repo_type,
        hashlib.sha256(token.encode()).hexdigest()[:16],
    )
    cached = _cached_repo_access(key, time.monotonic())
    if cached is not None:
        return cached
    if _hub_offline():
        # No wire to verify against; only a memo from a recent online probe counts.
        return False

    with _inflight_lock(key):
        # Filled while this caller waited for whoever held the lock: take their answer
        # rather than dialling the Hub a second time for the same question.
        cached = _cached_repo_access(key, time.monotonic())
        if cached is not None:
            return cached
        started = time.monotonic()
        try:
            allowed = _probe_repo_access(repo_id, token, repo_type)
            timed_out = False
        except _ProbeTimedOut:
            allowed = False
            timed_out = True
        # AFTER the probe, not before it. Reading the clock first and storing
        # ``start + TTL`` means a probe slower than the TTL memoizes an entry that is
        # already expired, so every later request re-probes and the memo never takes
        # effect -- exactly the regime a stalled Hub creates.
        finished = time.monotonic()
        # A timed-out probe never heard "no" from the Hub, so it says nothing about the
        # credential and must not deny a valid token for the full TTL.
        if not timed_out:
            timed_out = not allowed and (finished - started) >= _REPO_ACCESS_PROBE_TIMEOUT_S
        expiry = finished + (_REPO_ACCESS_UNREACHABLE_TTL_S if timed_out else _REPO_ACCESS_TTL_S)
        with _repo_access_lock:
            if len(_repo_access_cache) >= _REPO_ACCESS_CACHE_MAX:
                _evict_repo_access_locked()
            _repo_access_cache[key] = (expiry, allowed)
    return allowed


def _evict_repo_access_locked() -> None:
    """Caller holds ``_repo_access_lock``. Drop what has expired before anything live."""
    now = time.monotonic()
    for expired in [k for k, (deadline, _) in _repo_access_cache.items() if deadline <= now]:
        _repo_access_cache.pop(expired, None)
    if len(_repo_access_cache) >= _REPO_ACCESS_CACHE_MAX:
        # Still full of live entries: clear rather than grow without bound. Dropping a
        # valid answer costs one probe; keeping every answer costs memory forever.
        _repo_access_cache.clear()


def _inflight_lock(key: tuple[str, str, str]) -> threading.Lock:
    with _repo_access_lock:
        lock = _repo_access_inflight.get(key)
        if lock is None:
            if len(_repo_access_inflight) >= _REPO_ACCESS_CACHE_MAX:
                # Waiters already hold their own reference, so clearing the registry only
                # risks an extra probe, never a lost wakeup.
                _repo_access_inflight.clear()
            lock = threading.Lock()
            _repo_access_inflight[key] = lock
        return lock


def _probe_repo_access(repo_id: str, token: str, repo_type: str) -> bool:
    try:
        from huggingface_hub import HfApi, constants
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        if repo_type not in constants.REPO_TYPES:
            return False
        path = f"{HfApi().endpoint}/api/{repo_type}s/{repo_id}/auth-check"
        response = get_session().get(
            path,
            headers = build_hf_headers(token = token),
            timeout = _REPO_ACCESS_PROBE_TIMEOUT_S,
        )
        hf_raise_for_status(response)
        return True
    except Exception as exc:
        if _is_probe_timeout(exc):
            raise _ProbeTimedOut from exc
        return False
